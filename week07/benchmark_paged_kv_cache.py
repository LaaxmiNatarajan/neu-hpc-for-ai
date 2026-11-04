
import modal

app = modal.App("qwen3-paged-kv-benchmark")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch",
        "numpy",
        "transformers",
        "accelerate",
        "sentencepiece",
    )
    .add_local_file("paged_kv_cache.py", "/root/paged_kv_cache.py")
    .add_local_file(
        "inference_server/batched_paged_cache.py",
        "/root/batched_paged_cache.py",
    )
    .add_local_file("paged_hf_cache.py", "/root/paged_hf_cache.py")
)

@app.function(image=image, gpu="L40S", timeout=3600)
def benchmark():
    import sys
    import time
    import torch
    import transformers

    sys.path.insert(0, "/root")

    from transformers import AutoTokenizer, AutoModelForCausalLM
    from transformers.cache_utils import DynamicCache
    from paged_hf_cache import PagedTransformersCache

    MODEL_ID = "Qwen/Qwen3-0.6B"
    MAX_CACHE_LEN = 2048
    NEW_TOKENS = 128
    WARMUPS = 1
    REPEATS = 3
    BLOCK_SIZE = 16

    print("GPU:", torch.cuda.get_device_name(0))
    print("PyTorch:", torch.__version__)
    print("Transformers:", transformers.__version__)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to("cuda")
    model.eval()

    prompt = (
        "Explain how paged attention and KV caching "
        "improve autoregressive transformer inference."
    )
    input_ids = tokenizer(
        prompt, return_tensors="pt"
    ).input_ids.to("cuda")

    config = model.config

    def sync():
        torch.cuda.synchronize()

    def make_cache(mode):
        if mode == "dynamic":
            return DynamicCache(config=config)

        if mode == "paged":
            return PagedTransformersCache(
                config=config,
                max_cache_len=MAX_CACHE_LEN,
                block_size=BLOCK_SIZE,
                dtype=torch.bfloat16,
                device="cuda",
            )

        return None

    def run(mode):
        cache = make_cache(mode)
        full_ids = input_ids.clone()
        mask = torch.ones_like(full_ids)

        sync()
        start = time.perf_counter()

        with torch.inference_mode():
            output = model(
                input_ids=full_ids,
                attention_mask=mask,
                past_key_values=cache,
                use_cache=(cache is not None),
            )
            next_token = output.logits[:, -1, :].argmax(
                dim=-1, keepdim=True
            )

        sync()
        prefill_ms = (time.perf_counter() - start) * 1000

        sync()
        start = time.perf_counter()

        with torch.inference_mode():
            for step in range(NEW_TOKENS):
                if step == NEW_TOKENS - 1:
                    break

                mask = torch.cat(
                    [mask, torch.ones(
                        (1, 1), device="cuda", dtype=mask.dtype
                    )],
                    dim=1,
                )

                output = model(
                    input_ids=next_token,
                    attention_mask=mask,
                    past_key_values=cache,
                    use_cache=True,
                )
                next_token = output.logits[:, -1, :].argmax(
                    dim=-1, keepdim=True
                )

        sync()
        decode_ms = (time.perf_counter() - start) * 1000

        memory = (
            cache.memory_mib()
            if mode == "paged" and hasattr(cache, "memory_mib")
            else None
        )

        return {
            "prefill_ms": prefill_ms,
            "decode_ms": decode_ms,
            "total_ms": prefill_ms + decode_ms,
            "throughput": NEW_TOKENS / (
                (prefill_ms + decode_ms) / 1000
            ),
            "memory_mib": memory,
        }

    results = {}

    for mode in ["dynamic", "paged"]:
        print(f"\n========== {mode.upper()} ==========")

        for _ in range(WARMUPS):
            run(mode)

        measurements = []

        for repeat in range(REPEATS):
            result = run(mode)
            measurements.append(result)

            print(
                f"Repeat {repeat + 1}: "
                f"prefill={result['prefill_ms']:.2f} ms, "
                f"decode={result['decode_ms']:.2f} ms, "
                f"total={result['total_ms']:.2f} ms, "
                f"throughput={result['throughput']:.2f} tok/s, "
                f"memory={result['memory_mib']}"
            )

        results[mode] = measurements

    print("\n========== PAGED CACHE SUMMARY ==========")

    for mode, measurements in results.items():
        print(
            mode,
            "average throughput:",
            sum(x["throughput"] for x in measurements) / REPEATS,
            "tokens/sec",
        )

    print("\nPAGED CACHE BENCHMARK COMPLETED")


@app.local_entrypoint()
def main():
    benchmark.remote()
