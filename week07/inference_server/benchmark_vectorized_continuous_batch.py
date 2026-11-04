import io
import statistics
import time
from contextlib import redirect_stdout

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from vectorized_continuous_batch_engine import (
    VectorizedContinuousBatchEngine,
)

MODEL_NAME = "Qwen/Qwen3-0.6B"
MAX_BATCH_SIZE = 4
MAX_SEQ_LEN = 256
BLOCK_SIZE = 16
MAX_NEW_TOKENS = 32
REPEATS = 3

PROMPTS = [
    "What is a GPU?",
    "Explain how transformer attention works.",
    "What is the purpose of a key-value cache?",
    "Explain paged attention in language model inference.",
    "How does continuous batching improve throughput?",
    "Describe the difference between prefill and decode.",
    "What are the advantages of grouped query attention?",
    "Explain autoregressive text generation.",
]


def synchronize():
    torch.cuda.synchronize()


def run_one_batch(model, tokenizer, request_count, repeat):
    # Reset before engine creation so peak memory includes the paged cache.
    torch.cuda.reset_peak_memory_stats()

    engine = VectorizedContinuousBatchEngine(
        model=model,
        tokenizer=tokenizer,
        max_batch_size=MAX_BATCH_SIZE,
        max_seq_len=MAX_SEQ_LEN,
        block_size=BLOCK_SIZE,
        eos_enabled=False,
    )

    for i in range(request_count):
        engine.submit(
            request_id=f"repeat{repeat}_request{i}",
            prompt=PROMPTS[i % len(PROMPTS)],
            max_new_tokens=MAX_NEW_TOKENS,
        )

    synchronize()
    start = time.perf_counter()

    # Suppress per-token logs so benchmark output stays readable.
    with redirect_stdout(io.StringIO()):
        completed = engine.run_until_complete(max_iterations=500)

    synchronize()
    elapsed = time.perf_counter() - start

    total_tokens = sum(
        len(request.generated_tokens) for request in completed
    )

    ttft_ms = [
        (r.prefill_time - r.created_at) * 1000
        for r in completed
        if r.prefill_time is not None
    ]

    latency_ms = [
        (r.completed_at - r.created_at) * 1000
        for r in completed
        if r.completed_at is not None
    ]

    # Request-level average estimate, not per-token instrumentation.
    itl_ms = [
        (r.completed_at - r.prefill_time) * 1000
        / (len(r.generated_tokens) - 1)
        for r in completed
        if r.completed_at is not None
        and r.prefill_time is not None
        and len(r.generated_tokens) > 1
    ]

    metrics = engine.get_metrics()

    result = {
        "requests": request_count,
        "repeat": repeat,
        "elapsed_s": elapsed,
        "tokens": total_tokens,
        "throughput": total_tokens / elapsed,
        "ttft_ms": statistics.mean(ttft_ms),
        "latency_ms": statistics.mean(latency_ms),
        "itl_ms": statistics.mean(itl_ms) if itl_ms else 0.0,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "max_active": metrics["max_active_batch_size"],
        "paged_cache_mib": metrics["paged_cache_memory_mib"],
        "completed": metrics["completed"],
        "waiting": metrics["waiting"],
        "active": metrics["active"],
    }

    assert result["completed"] == request_count
    assert result["waiting"] == 0
    assert result["active"] == 0

    # The engine should never exceed its configured active capacity.
    assert result["max_active"] <= MAX_BATCH_SIZE

    return result


def print_summary(results):
    print("\n" + "=" * 112)
    print("VECTORIZED CONTINUOUS BATCHING BENCHMARK")
    print("=" * 112)

    print(
        f"{'Requests':>8} | {'Throughput':>15} | {'Avg TTFT':>12} | "
        f"{'Avg latency':>13} | {'Avg ITL':>12} | "
        f"{'Peak allocated':>15} | {'Max active':>10}"
    )
    print("-" * 112)

    for count in sorted(set(r["requests"] for r in results)):
        group = [r for r in results if r["requests"] == count]

        def avg(key):
            return statistics.mean(r[key] for r in group)

        max_active = max(r["max_active"] for r in group)

        print(
            f"{count:>8} | "
            f"{avg('throughput'):>10.2f} tok/s | "
            f"{avg('ttft_ms'):>9.2f} ms | "
            f"{avg('latency_ms'):>10.2f} ms | "
            f"{avg('itl_ms'):>9.2f} ms | "
            f"{avg('peak_allocated_mib'):>10.1f} MiB | "
            f"{max_active:>10}"
        )


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required.")

    print("GPU:", torch.cuda.get_device_name(0))
    print("Model:", MODEL_NAME)
    print("Maximum active batch size:", MAX_BATCH_SIZE)
    print("Generated tokens per request:", MAX_NEW_TOKENS)
    print("Repeats:", REPEATS)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to("cuda")

    model.eval()

    # Warm up the model before measuring.
    warmup = VectorizedContinuousBatchEngine(
        model=model,
        tokenizer=tokenizer,
        max_batch_size=MAX_BATCH_SIZE,
        max_seq_len=MAX_SEQ_LEN,
        block_size=BLOCK_SIZE,
        eos_enabled=False,
    )
    warmup.submit("warmup", "Explain GPU inference.", max_new_tokens=4)

    with redirect_stdout(io.StringIO()):
        warmup.run_until_complete(max_iterations=20)

    synchronize()
    del warmup
    torch.cuda.empty_cache()

    results = []

    # Eight requests with capacity four tests waiting-queue admission.
    for request_count in [1, 2, 4, 8]:
        print(f"\nRunning {request_count} requests")

        for repeat in range(1, REPEATS + 1):
            result = run_one_batch(
                model, tokenizer, request_count, repeat
            )
            results.append(result)

            print(
                f"  Repeat {repeat}: "
                f"{result['throughput']:.2f} tok/s, "
                f"TTFT={result['ttft_ms']:.2f} ms, "
                f"max active={result['max_active']}"
            )

    print_summary(results)

    print("\nGPU memory details:")
    for count in [1, 2, 4, 8]:
        group = [r for r in results if r["requests"] == count]
        print(
            f"  {count} requests: "
            f"peak allocated="
            f"{statistics.mean(r['peak_allocated_mib'] for r in group):.1f} MiB, "
            f"peak reserved="
            f"{statistics.mean(r['peak_reserved_mib'] for r in group):.1f} MiB, "
            f"paged cache="
            f"{statistics.mean(r['paged_cache_mib'] for r in group):.1f} MiB"
        )

    print("\n[PASS] All benchmark runs completed.")


if __name__ == "__main__":
    main()
