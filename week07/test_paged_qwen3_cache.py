
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache

from paged_hf_cache import PagedTransformersCache


MODEL_NAME = "Qwen/Qwen3-0.6B"
MAX_CACHE_LEN = 2048
BLOCK_SIZE = 16
DECODE_STEPS = 32

PROMPT = (
    "Explain how a paged key value cache improves "
    "memory management during transformer inference."
)


def compare_logits(reference, candidate, label):
    ref = reference.float()
    cand = candidate.float()

    max_error = (ref - cand).abs().max().item()
    top1_match = torch.equal(
        ref.argmax(dim=-1),
        cand.argmax(dim=-1),
    )

    print(
        f"{label:28s} | "
        f"max_error={max_error:.8f} | "
        f"top1_match={top1_match}"
    )

    return max_error, top1_match


@torch.inference_mode()
def main():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Run this test in your Modal GPU environment."
        )

    device = "cuda"

    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="sdpa",
    )
    model.eval()

    inputs = tokenizer(
        PROMPT,
        return_tensors="pt",
    ).to(device)

    input_ids = inputs["input_ids"]

    print(f"Prompt tokens: {input_ids.shape[1]}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # -------------------------------------------------
    # 1. Initialize both cache implementations
    # -------------------------------------------------

    dynamic_cache = DynamicCache()

    paged_cache = PagedTransformersCache(
        config=model.config,
        max_cache_len=MAX_CACHE_LEN,
        block_size=BLOCK_SIZE,
        dtype=model.dtype,
        device=device,
        sequence_id="qwen3_test",
    )

    # -------------------------------------------------
    # 2. Prefill with identical prompt tokens
    # -------------------------------------------------

    print("\n========== PREFILL ==========")

    dynamic_output = model(
        input_ids=input_ids,
        past_key_values=dynamic_cache,
        use_cache=True,
    )

    paged_output = model(
        input_ids=input_ids,
        past_key_values=paged_cache,
        use_cache=True,
    )

    prefill_error, prefill_match = compare_logits(
        dynamic_output.logits[:, -1, :],
        paged_output.logits[:, -1, :],
        "Prefill",
    )

    print(
        "Dynamic cache length:",
        dynamic_cache.get_seq_length(),
    )
    print(
        "Paged cache length:",
        paged_cache.get_seq_length(),
    )

    # Select the next token from the reference output.
    next_token = dynamic_output.logits[:, -1, :].argmax(
        dim=-1,
        keepdim=True,
    )

    # -------------------------------------------------
    # 3. Incremental teacher-forced decoding
    # -------------------------------------------------

    print("\n========== INCREMENTAL DECODE ==========")

    max_error = prefill_error
    top1_matches = int(prefill_match)

    for step in range(1, DECODE_STEPS + 1):
        dynamic_output = model(
            input_ids=next_token,
            past_key_values=dynamic_cache,
            use_cache=True,
        )

        paged_output = model(
            input_ids=next_token,
            past_key_values=paged_cache,
            use_cache=True,
        )

        error, match = compare_logits(
            dynamic_output.logits[:, -1, :],
            paged_output.logits[:, -1, :],
            f"Decode step {step:02d}",
        )

        max_error = max(max_error, error)
        top1_matches += int(match)

        # Both cache implementations receive the same next token.
        next_token = dynamic_output.logits[:, -1, :].argmax(
            dim=-1,
            keepdim=True,
        )

    # -------------------------------------------------
    # 4. Final validation and memory report
    # -------------------------------------------------

    print("\n========== FINAL SUMMARY ==========")

    dynamic_length = dynamic_cache.get_seq_length()
    paged_length = paged_cache.get_seq_length()

    print(f"Dynamic cache length: {dynamic_length}")
    print(f"Paged cache length:   {paged_length}")

    print(f"Maximum logit error:  {max_error:.8f}")
    print(
        f"Top-1 agreement:      "
        f"{top1_matches}/{DECODE_STEPS + 1}"
    )

    print(
        f"Paged cache memory:   "
        f"{paged_cache.memory_mib():.2f} MiB"
    )

    print(
        f"Paged block table:    "
        f"{paged_cache.get_block_table(0)}"
    )

    assert dynamic_length == paged_length, (
        "Cache lengths do not match"
    )

    assert dynamic_length == input_ids.shape[1] + DECODE_STEPS, (
        "Unexpected final sequence length"
    )

    assert top1_matches == DECODE_STEPS + 1, (
        "Top-1 predictions differ between cache implementations"
    )

    print("\nPaged Qwen3 cache correctness test passed!")


if __name__ == "__main__":
    main()
