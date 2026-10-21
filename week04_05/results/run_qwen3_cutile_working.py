
import time
from collections import Counter

import modal

app = modal.App("qwen3-4b-cutile-integration")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "torch",
        "numpy",
        "transformers",
        "accelerate",
        "cuda-tile[tileiras]",
    )
    .add_local_file(
        "attention_kernel.py",
        "/root/attention_kernel.py",
    )
)


@app.function(image=image, gpu="L40S", timeout=3600)
def benchmark():
    import torch
    import torch.nn.functional as F

    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        AttentionInterface,
        AttentionMaskInterface,
    )
    from transformers.integrations.sdpa_attention import (
        sdpa_attention_forward,
    )
    from transformers.masking_utils import sdpa_mask
    from attention_kernel import cutile_attention

    model_name = "Qwen/Qwen3-4B"
    device = "cuda"
    max_validated_kv_len = 1024

    def choose_tile_size(kv_len):
        """Smallest single tile that covers all keys (no multi-tile path)."""
        if not 0 < kv_len <= max_validated_kv_len:
            raise ValueError(f"Unsupported KV length: {kv_len}")
        return max(16, 1 << (kv_len - 1).bit_length())

    tile_usage = Counter()

    print("GPU:", torch.cuda.get_device_name(0))
    print("Model:", model_name)
    print("CUDA Tile sizing: adaptive power-of-two, min=16, max=1024")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(device)

    model.eval()

    prompt = (
        "Explain how grouped-query attention reduces "
        "memory usage during transformer inference."
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    ).to(device)

    prompt_length = inputs["input_ids"].shape[1]

    stats = {
        "cutile": 0,
        "sdpa_fallback": 0,
    }

    fallback_reasons = []
    layer_errors = []

    # Enable attention comparisons during correctness testing,
    # but disable them during performance measurements.
    diagnostics = {"enabled": True}

    
    def standard_causal_mask(mask, q_len, kv_len):
        """Accept only unpadded, right-aligned causal attention.

        With no explicit mask, permit full prefill and single-token decode.
        A multi-token decode without a mask is ambiguous, so fall back.
        """
        if q_len <= 0 or kv_len < q_len:
            return False
        if mask is None:
            return q_len == kv_len or q_len == 1
        if mask.ndim != 4 or mask.shape[-2:] != (q_len, kv_len):
            return False
        if mask.dtype == torch.bool:
            allowed = mask
        elif mask.dtype.is_floating_point:
            # Additive attention mask: 0 means allowed, negative means blocked.
            allowed = mask == 0
            if bool((mask > 0).any().item()):
                return False
        else:
            return False
        q_positions = torch.arange(q_len, device=mask.device) + kv_len - q_len
        k_positions = torch.arange(kv_len, device=mask.device)
        expected = (k_positions[None, :] <= q_positions[:, None])
        expected = expected[None, None, :, :].expand_as(allowed)
        return torch.equal(allowed, expected)


    def cutile_backend(
        module,
        query,
        key,
        value,
        attention_mask,
        **kwargs,
    ):
        q_len = query.shape[-2]
        kv_len = key.shape[-2]

        dropout = kwargs.get("dropout", 0.0)
        sliding_window = kwargs.get("sliding_window", None)

        eligible = (
            query.is_cuda
            and key.is_cuda
            and value.is_cuda
            and 0 < q_len <= kv_len <= max_validated_kv_len
            and dropout == 0.0
            and sliding_window is None
            and standard_causal_mask(
                attention_mask, q_len, kv_len
            )
        )

        if eligible:
            layer_id = stats["cutile"]

            # Use one tile spanning all keys; avoid broken multi-tile accumulation.
            selected_tile = choose_tile_size(kv_len)
            # CUDA Tile output layout: [B, Hq, S, D].
            out = cutile_attention(
                query,
                key,
                value,
                tile_n=selected_tile,
            )
            tile_usage[selected_tile] += 1

            if diagnostics["enabled"]:
                # Reference 1: Hugging Face BF16 SDPA.
                reference_out, _ = sdpa_attention_forward(
                    module,
                    query,
                    key,
                    value,
                    attention_mask,
                    **kwargs,
                )

                reference_bhsd = (
                    reference_out.transpose(1, 2).contiguous()
                )

                bf16_diff = (
                    out.float() - reference_bhsd.float()
                ).abs()

                # Reference 2: FP32 SDPA.
                if attention_mask is None:
                    fp32_is_causal = (q_len == kv_len)
                    if q_len == kv_len:
                        fp32_mask = None
                    else:
                        # SDPA's default is_causal alignment is not right-aligned
                        # for Sq < Sk. Construct the intended reference mask.
                        q_pos = torch.arange(q_len, device=query.device) + kv_len - q_len
                        k_pos = torch.arange(kv_len, device=query.device)
                        fp32_mask = (k_pos[None, :] <= q_pos[:, None])
                elif attention_mask.dtype == torch.bool:
                    fp32_mask = attention_mask
                    fp32_is_causal = False
                else:
                    fp32_mask = attention_mask.float()
                    fp32_is_causal = False

                fp32_reference = F.scaled_dot_product_attention(
                    query.float(),
                    key.float(),
                    value.float(),
                    attn_mask=fp32_mask,
                    dropout_p=0.0,
                    is_causal=fp32_is_causal,
                    enable_gqa=True,
                )

                fp32_diff = (
                    out.float() - fp32_reference.float()
                ).abs()

                layer_errors.append({
                    "layer": layer_id,
                    "bf16_max": bf16_diff.max().item(),
                    "bf16_mean": bf16_diff.mean().item(),
                    "fp32_max": fp32_diff.max().item(),
                    "fp32_mean": fp32_diff.mean().item(),
                })

                print(
                    f"Layer {layer_id:02d}: "
                    f"BF16_max={bf16_diff.max().item():.8f}, "
                    f"BF16_mean={bf16_diff.mean().item():.8f}, "
                    f"FP32_max={fp32_diff.max().item():.8f}, "
                    f"FP32_mean={fp32_diff.mean().item():.8f}"
                )

            stats["cutile"] += 1

            return out.transpose(1, 2).contiguous(), None

        if kv_len > max_validated_kv_len:
            reason = "sequence exceeds validated tile size"
        elif q_len > kv_len:
            reason = "query length exceeds key length"
        elif sliding_window is not None:
            reason = "sliding-window attention"
        elif dropout != 0.0:
            reason = "nonzero attention dropout"
        elif not standard_causal_mask(
            attention_mask, q_len, kv_len
        ):
            reason = "nonstandard attention mask"
        else:
            reason = "unsupported input"

        fallback_reasons.append(reason)
        stats["sdpa_fallback"] += 1

        return sdpa_attention_forward(
            module,
            query,
            key,
            value,
            attention_mask,
            **kwargs,
        )

    AttentionInterface.register(
        "cutile_custom",
        cutile_backend,
    )
    AttentionMaskInterface.register(
        "cutile_custom",
        sdpa_mask,
    )

    print("\n=== Model information ===")
    print("Prompt tokens:", prompt_length)
    print("Transformer layers:", model.config.num_hidden_layers)
    print("Query heads:", model.config.num_attention_heads)
    print("KV heads:", model.config.num_key_value_heads)

    # ------------------------------------------------
    # 1. Reference forward pass with hidden states
    # ------------------------------------------------
    print("\n=== PyTorch SDPA reference ===")

    with torch.inference_mode():
        ref_outputs = model(
            **inputs,
            use_cache=False,
            output_hidden_states=True,
        )

        ref = ref_outputs.logits.float()

        ref_hidden_states = [
            h.float().cpu()
            for h in ref_outputs.hidden_states
        ]

    torch.cuda.synchronize()

    print("Reference logits shape:", tuple(ref.shape))
    print("Reference hidden states:", len(ref_hidden_states))

    # ------------------------------------------------
    # 2. CUDA Tile forward pass with hidden states
    # ------------------------------------------------
    model.set_attn_implementation("cutile_custom")

    print("\n=== CUDA Tile forward pass ===")

    with torch.inference_mode():
        custom_outputs = model(
            **inputs,
            use_cache=False,
            output_hidden_states=True,
        )

        custom = custom_outputs.logits.float()

        custom_hidden_states = [
            h.float().cpu()
            for h in custom_outputs.hidden_states
        ]

    torch.cuda.synchronize()

    # ------------------------------------------------
    # 3. Hidden-state comparison
    # ------------------------------------------------
    print("\n=== Hidden-state comparison ===")

    if len(ref_hidden_states) != len(custom_hidden_states):
        raise RuntimeError(
            "Reference and custom hidden-state counts differ."
        )

    hidden_state_errors = []

    for i, (ref_h, custom_h) in enumerate(
        zip(ref_hidden_states, custom_hidden_states)
    ):
        diff = (ref_h - custom_h).abs()

        max_error = diff.max().item()
        mean_error = diff.mean().item()

        hidden_state_errors.append(
            (i, max_error, mean_error)
        )

        print(
            f"Hidden state {i:02d}: "
            f"max_error={max_error:.8f}, "
            f"mean_error={mean_error:.8f}"
        )

    worst_hidden_state = max(
        hidden_state_errors,
        key=lambda item: item[1],
    )

    print("\n=== Hidden-state summary ===")
    print("Worst hidden-state index:", worst_hidden_state[0])
    print("Worst hidden-state max error:", worst_hidden_state[1])

    # ------------------------------------------------
    # 4. Final logit correctness
    # ------------------------------------------------
    logit_diff = (custom - ref).abs()

    max_logit_error = logit_diff.max().item()
    mean_logit_error = logit_diff.mean().item()

    ref_next = ref[:, -1, :].argmax(dim=-1)
    custom_next = custom[:, -1, :].argmax(dim=-1)

    print("\n=== Logit correctness ===")
    print("Maximum absolute logit error:", max_logit_error)
    print("Mean absolute logit error:", mean_logit_error)
    print("Reference next token:", ref_next.tolist())
    print("CUDA Tile next token:", custom_next.tolist())
    print(
        "Next-token prediction matches:",
        torch.equal(ref_next, custom_next),
    )

    print("Custom kernel calls:", stats["cutile"])
    print("SDPA fallback calls:", stats["sdpa_fallback"])

    expected_layers = model.config.num_hidden_layers

    if stats["cutile"] != expected_layers:
        raise RuntimeError(
            f"Expected {expected_layers} CUDA Tile calls; "
            f"got {stats['cutile']}."
        )

    if stats["sdpa_fallback"] != 0:
        raise RuntimeError(
            "Unexpected SDPA fallback during prefill."
        )

    logits_match = torch.allclose(
        custom,
        ref,
        rtol=2e-2,
        atol=2e-2,
    )

    print("Logit tolerance passed:", logits_match)

    # ------------------------------------------------
    # 5. Attention precision summary
    # ------------------------------------------------
    print("\n=== Attention precision summary ===")

    if layer_errors:
        worst_bf16 = max(
            layer_errors,
            key=lambda item: item["bf16_max"],
        )

        worst_fp32 = max(
            layer_errors,
            key=lambda item: item["fp32_max"],
        )

        print(
            "Worst BF16 comparison layer:",
            worst_bf16["layer"],
        )
        print(
            "Worst BF16 attention error:",
            worst_bf16["bf16_max"],
        )
        print(
            "Worst FP32 comparison layer:",
            worst_fp32["layer"],
        )
        print(
            "Worst FP32 attention error:",
            worst_fp32["fp32_max"],
        )

        print(
            "Average BF16 attention mean error:",
            sum(
                item["bf16_mean"] for item in layer_errors
            ) / len(layer_errors),
        )

        print(
            "Average FP32 attention mean error:",
            sum(
                item["fp32_mean"] for item in layer_errors
            ) / len(layer_errors),
        )

    # ------------------------------------------------
    # 6. Prefill benchmark
    # ------------------------------------------------
    print("\n=== CUDA Tile prefill benchmark ===")

    diagnostics["enabled"] = False

    with torch.inference_mode():
        for _ in range(2):
            model(
                **inputs,
                use_cache=False,
            )

        torch.cuda.synchronize()

        latencies = []

        for _ in range(3):
            torch.cuda.synchronize()
            start = time.perf_counter()

            model(
                **inputs,
                use_cache=False,
            )

            torch.cuda.synchronize()

            latencies.append(
                (time.perf_counter() - start) * 1000
            )

    print(
        "Custom-backend prefill latency (ms):",
        [round(x, 2) for x in latencies],
    )

    print(
        "Mean custom-backend prefill latency (ms):",
        round(sum(latencies) / len(latencies), 2),
    )

    # ------------------------------------------------
    # 7. Text generation
    # ------------------------------------------------
    print("\n=== Generation ===")

    # Run an independent SDPA generation with the same deterministic settings.
    model.set_attn_implementation("sdpa")
    with torch.inference_mode():
        reference_generated = model.generate(
            **inputs,
            max_new_tokens=16,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    model.set_attn_implementation("cutile_custom")

    before_cutile = stats["cutile"]
    before_fallback = stats["sdpa_fallback"]

    torch.cuda.synchronize()
    start = time.perf_counter()

    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=16,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    torch.cuda.synchronize()

    generation_time = time.perf_counter() - start
    generated_tokens = generated.shape[1] - prompt_length

    generated_ids_match = torch.equal(generated, reference_generated)
    print("Generated token IDs match SDPA:", generated_ids_match)
    if not generated_ids_match:
        ref_ids = reference_generated[0, prompt_length:].tolist()
        custom_ids = generated[0, prompt_length:].tolist()
        print("SDPA generated token IDs:", ref_ids)
        print("CUDA Tile generated token IDs:", custom_ids)
    print("Generated tokens:", generated_tokens)
    print("Generation time (s):", round(generation_time, 3))

    print(
        "End-to-end throughput (tokens/s):",
        round(generated_tokens / generation_time, 2),
    )

    print(
        "Generation custom-kernel calls:",
        stats["cutile"] - before_cutile,
    )

    print(
        "Generation SDPA fallback calls:",
        stats["sdpa_fallback"] - before_fallback,
    )

    print(
        "Output:",
        tokenizer.decode(
            generated[0],
            skip_special_tokens=True,
        ),
    )

    # ------------------------------------------------
    # 8. Fallback reasons
    # ------------------------------------------------
    print("\n=== Fallback reasons ===")

    if fallback_reasons:
        for reason, count in Counter(fallback_reasons).items():
            print(f"{reason}: {count}")
    else:
        print("None")

    print("\n=== Adaptive tile usage ===")
    for selected_tile, count in sorted(tile_usage.items()):
        print(f"tile_n={selected_tile}: {count} CUDA Tile calls")

    # ------------------------------------------------
    # 9. Final status
    # ------------------------------------------------
    print("\n=== Final status ===")

    if logits_match:
        print(
            "PASS: CUDA Tile prefill passed "
            "the configured logit tolerance."
        )
    else:
        print(
            "NOT YET VALIDATED: CUDA Tile prefill "
            "exceeds the configured logit tolerance."
        )

    print(
        "Cached decoding CUDA Tile calls:",
        stats["cutile"] - before_cutile - expected_layers,
    )
    print("Generated token IDs match SDPA:", generated_ids_match)
    if not logits_match or not generated_ids_match:
        print("NOT YET FULLY VALIDATED: review model-level numerical differences.")
    if stats["sdpa_fallback"] - before_fallback:
        print("NOTE: Some generation attention calls used SDPA fallback.")


@app.local_entrypoint()
def main():
    benchmark.remote()
