import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache

from vectorized_paged_cache import VectorizedPagedCache


MODEL_ID = "Qwen/Qwen3-0.6B"
REQUEST_IDS = ["request_A", "request_B"]
PROMPTS = [
    "Explain what a GPU does.",
    "Explain how GPU memory and a KV cache work in transformer inference.",
]

MAX_NEW_TOKENS = 5
MAX_CACHE_LEN = 256
BLOCK_SIZE = 16


def make_mask(lengths, query_length, max_kv_length, dtype, device):
    """
    Additive attention mask:
      0     = attention allowed
      -inf  = attention blocked

    Right-padded prefill and variable-length decode are supported.
    """
    mask = torch.full(
        (len(lengths), 1, query_length, max_kv_length),
        torch.finfo(dtype).min,
        dtype=dtype,
        device=device,
    )

    for batch_idx, length in enumerate(lengths):
        for query_idx in range(query_length):
            if query_idx < length and query_length > 1:
                allowed = min(query_idx + 1, length)
            else:
                # Padded query rows are ignored when selecting logits.
                # Decode rows can attend to their entire context.
                allowed = length

            mask[batch_idx, 0, query_idx, :allowed] = 0

    return mask


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("This test requires a CUDA GPU.")

    device = "cuda"
    dtype = torch.bfloat16

    print("=" * 70)
    print("VECTORIZED PAGED QWEN3 TEST")
    print("=" * 70)
    print("GPU:", torch.cuda.get_device_name(0))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        device_map=device,
        attn_implementation="sdpa",
    )
    model.eval()

    # Tokenize separately so each request keeps its true prompt length.
    prompt_tokens = [
        tokenizer(prompt, return_tensors="pt").input_ids[0]
        for prompt in PROMPTS
    ]

    prompt_lengths = [len(x) for x in prompt_tokens]
    batch_size = len(prompt_tokens)
    max_prompt_length = max(prompt_lengths)

    input_ids = torch.full(
        (batch_size, max_prompt_length),
        tokenizer.pad_token_id or tokenizer.eos_token_id,
        dtype=torch.long,
        device=device,
    )

    position_ids = torch.zeros_like(input_ids)

    for idx, tokens in enumerate(prompt_tokens):
        length = len(tokens)
        input_ids[idx, :length] = tokens.to(device)
        position_ids[idx, :length] = torch.arange(
            length, device=device
        )

    cache = VectorizedPagedCache(
        config=model.config,
        request_ids=REQUEST_IDS,
        max_cache_len=MAX_CACHE_LEN,
        block_size=BLOCK_SIZE,
        dtype=dtype,
        device=device,
    )

    # ---------------------------------------------------------
    # 1. Vectorized prefill
    # ---------------------------------------------------------
    print("\n[1] Running vectorized prefill")

    cache.set_append_lengths(prompt_lengths)

    prefill_mask = make_mask(
        prompt_lengths,
        max_prompt_length,
        max_prompt_length,
        dtype,
        device,
    )

    with torch.inference_mode():
        batched_output = model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=prefill_mask,
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=0,
        )

    # Select the last REAL prompt token for each request.
    # The final batch column may be padding for shorter prompts.
    batched_logits = torch.stack([
        batched_output.logits[idx, prompt_lengths[idx] - 1, :]
        for idx in range(batch_size)
    ])

    # ---------------------------------------------------------
    # 2. Individual DynamicCache references
    # ---------------------------------------------------------
    print("[2] Creating DynamicCache reference outputs")

    reference_caches = []
    reference_logits = []

    with torch.inference_mode():
        for tokens in prompt_tokens:
            ids = tokens.unsqueeze(0).to(device)
            positions = torch.arange(
                ids.shape[1], device=device
            ).unsqueeze(0)

            reference_cache = DynamicCache(config=model.config)

            output = model(
                input_ids=ids,
                position_ids=positions,
                attention_mask=torch.ones_like(ids),
                past_key_values=reference_cache,
                use_cache=True,
                logits_to_keep=1,
            )

            reference_caches.append(output.past_key_values)
            reference_logits.append(output.logits[:, -1, :])

    print("\n[3] Comparing prefill results")

    for idx, reference in enumerate(reference_logits):
        difference = (
            batched_logits[idx].float() - reference[0].float()
        ).abs().max().item()

        same_top1 = (
            batched_logits[idx].argmax().item()
            == reference[0].argmax().item()
        )

        print(
            f"{REQUEST_IDS[idx]}: "
            f"max_abs_error={difference:.6f}, "
            f"top1_match={same_top1}"
        )

        if not same_top1:
            raise AssertionError("Prefill top-1 prediction mismatch.")

    # ---------------------------------------------------------
    # 4. Vectorized decode vs individual DynamicCache
    # ---------------------------------------------------------
    print("\n[4] Running vectorized decode")

    current_logits = batched_logits
    current_lengths = list(prompt_lengths)

    for step in range(MAX_NEW_TOKENS):
        # Use reference predictions as the next input to both paths.
        next_tokens = []

        for reference in reference_logits:
            next_tokens.append(
                reference[0].argmax().view(1)
            )

        decode_ids = torch.cat(next_tokens).view(batch_size, 1)

        new_lengths = [length + 1 for length in current_lengths]

        cache.set_append_lengths([1] * batch_size)

        decode_mask = make_mask(
            new_lengths,
            query_length=1,
            max_kv_length=max(new_lengths),
            dtype=dtype,
            device=device,
        )

        decode_positions = torch.tensor(
            current_lengths,
            dtype=torch.long,
            device=device,
        ).view(batch_size, 1)

        with torch.inference_mode():
            batched_output = model(
                input_ids=decode_ids,
                position_ids=decode_positions,
                attention_mask=decode_mask,
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            )

        current_logits = batched_output.logits[:, -1, :]

        new_reference_logits = []

        with torch.inference_mode():
            for idx in range(batch_size):
                reference_output = model(
                    input_ids=decode_ids[idx:idx + 1],
                    position_ids=decode_positions[idx:idx + 1],
                    attention_mask=torch.ones(
                        (1, new_lengths[idx]),
                        dtype=torch.long,
                        device=device,
                    ),
                    past_key_values=reference_caches[idx],
                    use_cache=True,
                    logits_to_keep=1,
                )

                new_reference_logits.append(
                    reference_output.logits[:, -1, :]
                )

        print(f"\nDecode step {step + 1}")

        for idx, reference in enumerate(new_reference_logits):
            difference = (
                current_logits[idx].float()
                - reference[0].float()
            ).abs().max().item()

            same_top1 = (
                current_logits[idx].argmax().item()
                == reference[0].argmax().item()
            )

            print(
                f"{REQUEST_IDS[idx]}: "
                f"max_abs_error={difference:.6f}, "
                f"top1_match={same_top1}"
            )

            if not same_top1:
                raise AssertionError("Decode top-1 prediction mismatch.")

        reference_logits = new_reference_logits
        current_lengths = new_lengths

    # ---------------------------------------------------------
    # 5. Final checks
    # ---------------------------------------------------------
    print("\n[5] Final cache checks")

    for idx, request_id in enumerate(REQUEST_IDS):
        actual_length = cache.get_lengths()[idx]
        expected_length = prompt_lengths[idx] + MAX_NEW_TOKENS

        print(
            request_id,
            "length:",
            actual_length,
            "expected:",
            expected_length,
            "block table:",
            cache.get_block_table(request_id),
        )

        assert actual_length == expected_length

    print("Paged cache memory:", round(cache.memory_mib(), 2), "MiB")

    for request_id in REQUEST_IDS:
        cache.release_request(request_id)

    print("\n[PASS] Vectorized paged cache correctness test completed.")


if __name__ == "__main__":
    main()
