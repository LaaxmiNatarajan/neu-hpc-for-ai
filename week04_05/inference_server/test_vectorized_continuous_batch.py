import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from vectorized_continuous_batch_engine import (
    VectorizedContinuousBatchEngine,
)


MODEL_NAME = "Qwen/Qwen3-0.6B"


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("This test requires a CUDA GPU.")

    device = "cuda"

    print("=" * 65)
    print("VECTORIZED CONTINUOUS BATCHING TEST")
    print("=" * 65)
    print("GPU:", torch.cuda.get_device_name(0))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(device)

    model.eval()

    engine = VectorizedContinuousBatchEngine(
        model=model,
        tokenizer=tokenizer,
        max_batch_size=2,
        max_seq_len=256,
        block_size=16,
        eos_enabled=False,
    )

    # A finishes after prefill.
    engine.submit(
        request_id="request_A",
        prompt="What is a GPU?",
        max_new_tokens=1,
    )

    # B remains active while C waits.
    engine.submit(
        request_id="request_B",
        prompt="Explain how a transformer uses attention and KV caching.",
        max_new_tokens=5,
    )

    engine.submit(
        request_id="request_C",
        prompt="What is paged attention?",
        max_new_tokens=2,
    )

    print("\n[1] Initial scheduler state")
    print(engine.get_metrics())

    print("\n[2] Running continuous batching")
    completed = engine.run_until_complete(max_iterations=50)

    print("\n[3] Completed request results")

    for request in completed:
        generated_text = tokenizer.decode(
            request.generated_tokens,
            skip_special_tokens=True,
        )

        print(
            f"{request.request_id}: "
            f"tokens={len(request.generated_tokens)}, "
            f"text={generated_text!r}"
        )

    print("\n[4] Final metrics")
    metrics = engine.get_metrics()
    print(metrics)

    assert len(completed) == 3
    assert metrics["waiting"] == 0
    assert metrics["active"] == 0
    assert metrics["completed"] == 3
    assert metrics["max_active_batch_size"] == 2

    # Every request's cache must have been released.
    for request_id in (
        "request_A",
        "request_B",
        "request_C",
    ):
        assert not engine.cache.has_request(request_id)

    assert len(engine.cache.paged.active_sequences) == 0

    print("\n[PASS] Vectorized continuous batching checks passed.")


if __name__ == "__main__":
    main()
