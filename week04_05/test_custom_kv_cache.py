import modal

app = modal.App("qwen3-custom-kv-cache-test")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch", "numpy", "transformers", "accelerate")
    .add_local_file("kv_cache.py", "/root/kv_cache.py")
)


@app.function(image=image, gpu="L40S", timeout=3600)
def test_cache():
    import torch
    import transformers

    from transformers import (
        AutoConfig,
        AutoTokenizer,
        AutoModelForCausalLM,
    )
    from transformers.cache_utils import DynamicCache

    from kv_cache import PreallocatedKVCache

    MODEL_ID = "Qwen/Qwen3-0.6B"
    MAX_CACHE_LEN = 1024
    DECODE_STEPS = 8

    print("Transformers:", transformers.__version__)
    print("Model:", MODEL_ID)

    torch.manual_seed(42)

    config = AutoConfig.from_pretrained(MODEL_ID)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to("cuda")

    model.eval()

    print("\nModel configuration:")
    print("Layers:", config.num_hidden_layers)
    print("KV heads:", config.num_key_value_heads)
    print("Head dimension:", config.head_dim)

    prompt = (
        "Explain how KV caching improves transformer inference "
        "performance during autoregressive generation."
    )

    input_ids = tokenizer(
        prompt,
        return_tensors="pt",
    ).input_ids.to("cuda")

    attention_mask = torch.ones_like(input_ids)

    # Reference: Hugging Face DynamicCache.
    reference_cache = DynamicCache(config=config)

    # Our custom preallocated cache.
    custom_cache = PreallocatedKVCache(
        num_layers=config.num_hidden_layers,
        max_cache_len=MAX_CACHE_LEN,
    )

    max_errors = []

    with torch.inference_mode():

        # --------------------------------------------------
        # 1. PREFILL
        # --------------------------------------------------

        print("\n========== PREFILL ==========")

        reference_output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=reference_cache,
            use_cache=True,
        )

        custom_output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=custom_cache,
            use_cache=True,
        )

        reference_logits = reference_output.logits[:, -1, :]
        custom_logits = custom_output.logits[:, -1, :]

        prefill_error = (
            reference_logits.float() - custom_logits.float()
        ).abs().max().item()

        max_errors.append(prefill_error)

        print("Prompt tokens:", input_ids.shape[1])
        print("Reference cache length:", reference_cache.get_seq_length())
        print("Custom cache length:", custom_cache.get_seq_length())
        print("Prefill max logit error:", prefill_error)

        # --------------------------------------------------
        # 2. INCREMENTAL DECODING
        # --------------------------------------------------

        print("\n========== DECODE ==========")

        next_token = reference_logits.argmax(
            dim=-1,
            keepdim=True,
        )

        for step in range(DECODE_STEPS):

            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones(
                        (attention_mask.shape[0], 1),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    ),
                ],
                dim=1,
            )

            reference_output = model(
                input_ids=next_token,
                attention_mask=attention_mask,
                past_key_values=reference_cache,
                use_cache=True,
            )

            custom_output = model(
                input_ids=next_token,
                attention_mask=attention_mask,
                past_key_values=custom_cache,
                use_cache=True,
            )

            reference_logits = reference_output.logits[:, -1, :]
            custom_logits = custom_output.logits[:, -1, :]

            error = (
                reference_logits.float() - custom_logits.float()
            ).abs().max().item()

            max_errors.append(error)

            reference_token = reference_logits.argmax(dim=-1)
            custom_token = custom_logits.argmax(dim=-1)

            print(
                f"Step {step + 1}: "
                f"max_error={error:.6f}, "
                f"token_match={torch.equal(reference_token, custom_token)}"
            )

            if not torch.equal(reference_token, custom_token):
                print("WARNING: Generated token IDs differ")

            next_token = reference_token.unsqueeze(-1)

    # --------------------------------------------------
    # 3. CACHE VALIDATION
    # --------------------------------------------------

    print("\n========== CACHE VALIDATION ==========")

    print("Reference sequence length:", reference_cache.get_seq_length())
    print("Custom sequence length:", custom_cache.get_seq_length())

    expected_length = input_ids.shape[1] + DECODE_STEPS

    assert custom_cache.get_seq_length() == expected_length
    assert len(custom_cache.layers) == config.num_hidden_layers

    for i, layer in enumerate(custom_cache.layers):
        assert layer.get_seq_length() == expected_length, (
            f"Layer {i} has incorrect sequence length"
        )

    print("Allocated KV memory (MiB):", custom_cache.memory_mib())

    print("Maximum logit error:", max(max_errors))

    # Reset validation.
    custom_cache.reset()

    assert custom_cache.get_seq_length() == 0

    print("Cache reset: PASS")
    print("\nCUSTOM KV CACHE TEST COMPLETED")


@app.local_entrypoint()
def main():
    test_cache.remote()
