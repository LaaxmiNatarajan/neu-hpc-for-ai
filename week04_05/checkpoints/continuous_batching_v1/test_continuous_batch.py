
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from continuous_batch_engine import ContinuousBatchEngine


MODEL_NAME = "Qwen/Qwen3-0.6B"


def main():
    print("=" * 65)
    print("CONTINUOUS BATCHING TEST")
    print("=" * 65)

    if not torch.cuda.is_available():
        raise RuntimeError("This test requires a CUDA GPU")

    print("GPU:", torch.cuda.get_device_name(0))

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="sdpa",
        trust_remote_code=True,
    )

    model.eval()

    engine = ContinuousBatchEngine(
        model=model,
        tokenizer=tokenizer,
        max_batch_size=2,
        max_seq_len=128,
        block_size=16,
        eos_enabled=False,
    )

    # A finishes after its prefill token.
    engine.submit(
        "A",
        "Explain BGP route selection in simple terms.",
        max_new_tokens=1,
    )

    # B remains active while C waits.
    engine.submit(
        "B",
        "Explain OSPF route selection in simple terms.",
        max_new_tokens=4,
    )

    # C should be admitted after A completes.
    engine.submit(
        "C",
        "Explain what a VLAN is in simple terms.",
        max_new_tokens=2,
    )

    print("\n[TEST] Initial scheduler state:")
    print(engine.get_metrics())

    completed = engine.run_until_complete(
        max_iterations=100
    )

    print("\n" + "=" * 65)
    print("RESULTS")
    print("=" * 65)

    for request in completed:
        generated_text = tokenizer.decode(
            request.generated_tokens,
            skip_special_tokens=True,
        )

        print(f"\nRequest: {request.request_id}")
        print("Prompt:", request.prompt)
        print("Generated tokens:", len(request.generated_tokens))
        print("Generated text:", generated_text)
        print("Status:", request.status)

    metrics = engine.get_metrics()

    print("\nFinal metrics:")
    print(metrics)

    assert len(completed) == 3, (
        f"Expected 3 completed requests, got {len(completed)}"
    )

    completed_ids = {
        request.request_id for request in completed
    }

    assert completed_ids == {"A", "B", "C"}

    assert metrics["active"] == 0
    assert metrics["waiting"] == 0
    assert metrics["completed"] == 3

    # A should have completed and C should have been admitted
    # while B was still active.
    assert metrics["max_batch_size"] == 2

    assert not engine.paged_cache.active_sequences

    print("\n[PASS] All continuous batching checks passed.")


if __name__ == "__main__":
    main()
