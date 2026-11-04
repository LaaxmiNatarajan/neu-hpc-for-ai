
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from batched_paged_hf_cache import BatchedPagedTransformersCache


MODEL_ID = "Qwen/Qwen3-0.6B"
MAX_CACHE_LEN = 64
BLOCK_SIZE = 16


def find_equal_length_prompts(tokenizer):
    candidates = [
        "Explain BGP route selection in simple terms.",
        "Explain OSPF route selection in simple terms.",
        "Describe BGP routing in simple terms.",
        "Describe OSPF routing in simple terms.",
        "Explain how a router selects the best route.",
        "Explain how a switch forwards network packets.",
        "Describe how BGP selects the best network route.",
        "Describe how OSPF selects the best network route.",
        "What happens during transformer inference?",
        "How does a paged cache improve inference?",
    ]

    groups = {}

    for prompt in candidates:
        ids = tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=True,
        )["input_ids"]

        groups.setdefault(ids.shape[1], []).append((prompt, ids))

    for length, items in groups.items():
        if len(items) >= 2:
            return items[0], items[1]

    raise RuntimeError(
        "Could not find two prompts with equal token lengths."
    )


def compare_logits(label, actual, expected, atol=0.03, rtol=0.03):
    """Strict comparison: use for equivalent batch-size execution paths."""

    actual_float = actual.float()
    expected_float = expected.float()

    max_error = (
        actual_float - expected_float
    ).abs().max().item()

    mean_error = (
        actual_float - expected_float
    ).abs().mean().item()

    top1_match = torch.equal(
        actual.argmax(dim=-1),
        expected.argmax(dim=-1),
    )

    print(f"\n{label}")
    print(f"  Maximum absolute error: {max_error:.8f}")
    print(f"  Mean absolute error:    {mean_error:.8f}")
    print(f"  Top-1 predictions match: {top1_match}")

    torch.testing.assert_close(
        actual_float,
        expected_float,
        atol=atol,
        rtol=rtol,
        msg=f"{label} logits do not match",
    )

    assert top1_match, f"{label}: top-1 predictions differ"


def report_logits(label, actual, expected):
    """Diagnostic comparison; does not fail the test."""

    actual_float = actual.float()
    expected_float = expected.float()

    diff = (actual_float - expected_float).abs()

    max_error = diff.max().item()
    mean_error = diff.mean().item()

    top1_match = torch.equal(
        actual.argmax(dim=-1),
        expected.argmax(dim=-1),
    )

    print(f"\n{label}")
    print(f"  Maximum absolute error: {max_error:.8f}")
    print(f"  Mean absolute error:    {mean_error:.8f}")
    print(f"  Top-1 predictions match: {top1_match}")


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("This test must run on a CUDA GPU.")

    torch.manual_seed(42)

    device = torch.device("cuda:0")
    dtype = torch.bfloat16

    print("=" * 65)
    print("BATCHED QWEN3 PAGED KV CACHE TEST")
    print("=" * 65)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Model: {MODEL_ID}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        device_map="cuda",
        attn_implementation="sdpa",
    )
    model.eval()

    (prompt_a, ids_a), (prompt_b, ids_b) = (
        find_equal_length_prompts(tokenizer)
    )

    prompt_len = ids_a.shape[1]

    assert prompt_a != prompt_b
    assert ids_b.shape[1] == prompt_len

    print("\nSelected prompts:")
    print(f"  Request A: {prompt_a}")
    print(f"  Request B: {prompt_b}")
    print(f"  Token length: {prompt_len}")

    if prompt_len + 1 > MAX_CACHE_LEN:
        raise ValueError("Prompt is too long for configured cache.")

    input_ids = torch.cat([ids_a, ids_b], dim=0).to(device)

    attention_mask = torch.ones_like(input_ids)

    request_ids = ["request_A", "request_B"]

    paged_cache = BatchedPagedTransformersCache(
        config=model.config,
        request_ids=request_ids,
        max_cache_len=MAX_CACHE_LEN,
        block_size=BLOCK_SIZE,
        dtype=dtype,
        device=str(device),
    )

    # ---------------------------------------------------------
    # 1. Custom paged cache: batched prefill
    # ---------------------------------------------------------
    print("\n[1] Custom paged cache: batched prefill")

    with torch.inference_mode():
        batched_output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=paged_cache,
            use_cache=True,
        )

    paged_prefill_logits = batched_output.logits[:, -1, :]

    print("  Custom paged prefill completed.")

    # ---------------------------------------------------------
    # 2. Standard DynamicCache: batched prefill
    # ---------------------------------------------------------
    print("\n[2] Standard DynamicCache: batched prefill")

    with torch.inference_mode():
        batch_reference = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
        )

    batch_reference_logits = batch_reference.logits[:, -1, :]

    print("  Batched DynamicCache prefill completed.")

    # This is the primary correctness comparison:
    # same batch size, same model, same inputs and attention path.
    compare_logits(
        "CUSTOM PAGED CACHE vs BATCHED DYNAMICCACHE - PREFILL",
        paged_prefill_logits,
        batch_reference_logits,
    )

    # ---------------------------------------------------------
    # 3. Individual DynamicCache references
    # ---------------------------------------------------------
    print("\n[3] Standard DynamicCache: individual references")

    with torch.inference_mode():
        reference_a = model(
            input_ids=ids_a.to(device),
            attention_mask=torch.ones(
                (1, prompt_len),
                dtype=torch.long,
                device=device,
            ),
            use_cache=True,
        )

        reference_b = model(
            input_ids=ids_b.to(device),
            attention_mask=torch.ones(
                (1, prompt_len),
                dtype=torch.long,
                device=device,
            ),
            use_cache=True,
        )

    reference_logits_a = reference_a.logits[:, -1, :]
    reference_logits_b = reference_b.logits[:, -1, :]

    # Diagnostic only: batch-size-2 versus batch-size-1 can have
    # different floating-point execution behavior.
    report_logits(
        "Request A: batched paged vs individual DynamicCache",
        paged_prefill_logits[0:1],
        reference_logits_a,
    )

    report_logits(
        "Request B: batched paged vs individual DynamicCache",
        paged_prefill_logits[1:2],
        reference_logits_b,
    )

    # ---------------------------------------------------------
    # 4. Verify independent paged storage
    # ---------------------------------------------------------
    print("\n[4] Checking independent KV storage")

    for request_id in request_ids:
        length = paged_cache.paged_cache.get_seq_length(
            request_id,
            0,
        )

        block_table = paged_cache.get_block_table(
            request_id,
            0,
        )

        print(f"  {request_id}:")
        print(f"    Sequence length: {length}")
        print(f"    Block table: {block_table}")

        assert length == prompt_len

    table_a = paged_cache.get_block_table("request_A", 0)
    table_b = paged_cache.get_block_table("request_B", 0)

    assert set(table_a).isdisjoint(set(table_b)), (
        "Requests unexpectedly share physical blocks."
    )

    print("  Independent physical block allocation verified.")

    # ---------------------------------------------------------
    # 5. Batched decode: custom paged cache vs batched DynamicCache
    # ---------------------------------------------------------
    print("\n[5] Testing batched decode")

    # Use the same next tokens for both batched cache implementations.
    batch_next_tokens = batch_reference_logits.argmax(
        dim=-1,
        keepdim=True,
    )

    decode_attention_mask = torch.ones(
        (2, prompt_len + 1),
        dtype=torch.long,
        device=device,
    )

    with torch.inference_mode():
        paged_decode = model(
            input_ids=batch_next_tokens,
            attention_mask=decode_attention_mask,
            past_key_values=paged_cache,
            use_cache=True,
        )

        batch_reference_decode = model(
            input_ids=batch_next_tokens,
            attention_mask=decode_attention_mask,
            past_key_values=batch_reference.past_key_values,
            use_cache=True,
        )

    paged_decode_logits = paged_decode.logits[:, -1, :]
    batch_decode_logits = batch_reference_decode.logits[:, -1, :]

    compare_logits(
        "CUSTOM PAGED CACHE vs BATCHED DYNAMICCACHE - DECODE",
        paged_decode_logits,
        batch_decode_logits,
    )

    # ---------------------------------------------------------
    # 6. Individual decode references
    # ---------------------------------------------------------
    print("\n[6] Comparing decode with individual references")

    next_a = reference_logits_a.argmax(
        dim=-1,
        keepdim=True,
    )

    next_b = reference_logits_b.argmax(
        dim=-1,
        keepdim=True,
    )

    with torch.inference_mode():
        reference_decode_a = model(
            input_ids=next_a,
            attention_mask=torch.ones(
                (1, prompt_len + 1),
                dtype=torch.long,
                device=device,
            ),
            past_key_values=reference_a.past_key_values,
            use_cache=True,
        )

        reference_decode_b = model(
            input_ids=next_b,
            attention_mask=torch.ones(
                (1, prompt_len + 1),
                dtype=torch.long,
                device=device,
            ),
            past_key_values=reference_b.past_key_values,
            use_cache=True,
        )

    report_logits(
        "Request A: batched paged vs individual decode",
        paged_decode_logits[0:1],
        reference_decode_a.logits[:, -1, :],
    )

    report_logits(
        "Request B: batched paged vs individual decode",
        paged_decode_logits[1:2],
        reference_decode_b.logits[:, -1, :],
    )

    # ---------------------------------------------------------
    # 7. Verify cache lengths after decode
    # ---------------------------------------------------------
    print("\n[7] Checking cache lengths after decode")

    for request_id in request_ids:
        length = paged_cache.paged_cache.get_seq_length(
            request_id,
            0,
        )

        print(f"  {request_id}: {length} tokens")

        assert length == prompt_len + 1

    # ---------------------------------------------------------
    # 8. Verify independent request release
    # ---------------------------------------------------------
    print("\n[8] Testing independent request release")

    paged_cache.release_request("request_A")

    assert not paged_cache.paged_cache.has_request("request_A")
    assert paged_cache.paged_cache.has_request("request_B")

    remaining_key, remaining_value = (
        paged_cache.paged_cache.get_kv("request_B", 0)
    )

    assert remaining_key.shape[2] == prompt_len + 1
    assert remaining_value.shape[2] == prompt_len + 1

    print("  Request A released.")
    print("  Request B remains intact.")

    paged_cache.release_request("request_B")

    assert not paged_cache.paged_cache.has_request("request_B")

    print("  Both requests released successfully.")

    # ---------------------------------------------------------
    # 9. Summary
    # ---------------------------------------------------------
    print("\n" + "=" * 65)
    print("BATCHED QWEN3 TEST COMPLETED")
    print("=" * 65)

    print("  Batched prefill comparison: PASS")
    print("  Batched decode comparison: PASS")
    print("  Independent block allocation: PASS")
    print("  Cache length validation: PASS")
    print("  Independent request release: PASS")
    print(f"  Batch size: {len(request_ids)}")
    print(f"  Prompt length: {prompt_len}")
    print("  Decode steps: 1")
    print(f"  Cache memory: {paged_cache.memory_mib():.2f} MiB")
    print("\nIndividual-reference comparisons are diagnostic only.")


if __name__ == "__main__":
    main()

