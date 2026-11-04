
import modal

app = modal.App("qwen3-kv-cache-benchmark")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch", "numpy", "transformers", "accelerate")
    .add_local_file("kv_cache.py", "/root/kv_cache.py")
)


@app.function(image=image, gpu="L40S", timeout=3600)
def benchmark():
    import time
    import torch
    import transformers

    from transformers import (
        AutoConfig,
        AutoTokenizer,
        AutoModelForCausalLM,
    )
    from transformers.cache_utils import DynamicCache
    from kv_cache import PreallocatedKVCache

    # ---------------- CONFIGURATION ----------------

    MODEL_ID = "Qwen/Qwen3-0.6B"
    MAX_CACHE_LEN = 2048
    NEW_TOKENS = 256
    WARMUPS = 1
    REPEATS = 3
    VALIDATION_STEPS = 32

    print("Transformers:", transformers.__version__)
    print("Model:", MODEL_ID)

    config = AutoConfig.from_pretrained(MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to("cuda")

    model.eval()

    prompt = (
        "Explain how KV caching improves the performance "
        "of autoregressive transformer inference."
    )

    input_ids = tokenizer(
        prompt,
        return_tensors="pt",
    ).input_ids.to("cuda")

    def synchronize():
        torch.cuda.synchronize()

    # ---------------- BENCHMARK FUNCTION ----------------

    def run(mode):
        full_ids = input_ids.clone()
        attention_mask = torch.ones_like(full_ids)

        cache = None

        if mode == "dynamic":
            cache = DynamicCache(config=config)

        elif mode == "custom":
            cache = PreallocatedKVCache(
                num_layers=config.num_hidden_layers,
                max_cache_len=MAX_CACHE_LEN,
            )

        # ---------------- PREFILL ----------------

        synchronize()
        start = time.perf_counter()

        with torch.inference_mode():
            output = model(
                input_ids=full_ids,
                attention_mask=attention_mask,
                past_key_values=cache,
                use_cache=(mode != "none"),
            )

            next_token = output.logits[:, -1, :].argmax(
                dim=-1,
                keepdim=True,
            )

        synchronize()
        prefill_time = time.perf_counter() - start

        generated = []

        # ---------------- DECODE ----------------

        synchronize()
        start = time.perf_counter()

        with torch.inference_mode():
            for step in range(NEW_TOKENS):

                generated.append(next_token.clone())

                if step == NEW_TOKENS - 1:
                    break

                if mode == "none":
                    # Recompute the complete sequence.
                    full_ids = torch.cat(
                        [full_ids, next_token],
                        dim=1,
                    )

                    attention_mask = torch.ones_like(full_ids)

                    output = model(
                        input_ids=full_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                    )

                else:
                    # Process only the newly generated token.
                    attention_mask = torch.cat(
                        [
                            attention_mask,
                            torch.ones(
                                (1, 1),
                                dtype=attention_mask.dtype,
                                device=input_ids.device,
                            ),
                        ],
                        dim=1,
                    )

                    output = model(
                        input_ids=next_token,
                        attention_mask=attention_mask,
                        past_key_values=cache,
                        use_cache=True,
                    )

                next_token = output.logits[:, -1, :].argmax(
                    dim=-1,
                    keepdim=True,
                )

        synchronize()
        decode_time = time.perf_counter() - start

        generated_ids = torch.cat(generated, dim=1)

        cache_memory = (
            cache.memory_mib()
            if mode == "custom"
            else None
        )

        return {
            "prefill_ms": prefill_time * 1000,
            "decode_ms": decode_time * 1000,
            "total_ms": (prefill_time + decode_time) * 1000,
            "tokens_per_second": (
                NEW_TOKENS / (prefill_time + decode_time)
            ),
            "generated_ids": generated_ids,
            "cache_memory_mib": cache_memory,
        }

    # ---------------- TEACHER-FORCED VALIDATION ----------------

    def validate_teacher_forced(validation_steps=32):

        print("\n========== TEACHER-FORCED VALIDATION ==========")

        states = {}

        # Initialize each mode with identical prompt tokens.
        for mode in ["none", "dynamic", "custom"]:

            cache = None

            if mode == "dynamic":
                cache = DynamicCache(config=config)

            elif mode == "custom":
                cache = PreallocatedKVCache(
                    num_layers=config.num_hidden_layers,
                    max_cache_len=MAX_CACHE_LEN,
                )

            full_ids = input_ids.clone()
            mask = torch.ones_like(full_ids)

            with torch.inference_mode():
                output = model(
                    input_ids=full_ids,
                    attention_mask=mask,
                    past_key_values=cache,
                    use_cache=(mode != "none"),
                )

            states[mode] = {
                "cache": cache,
                "full_ids": full_ids,
                "mask": mask,
                "logits": output.logits[:, -1, :],
                "max_error": 0.0,
                "matching_steps": 0,
            }

        # Every implementation receives the same reference token.
        for step in range(validation_steps):

            reference_logits = states["none"]["logits"].float()

            reference_token = reference_logits.argmax(
                dim=-1,
                keepdim=True,
            )

            for mode in ["dynamic", "custom"]:

                candidate_logits = states[mode]["logits"].float()

                max_error = (
                    reference_logits - candidate_logits
                ).abs().max().item()

                reference_top1 = reference_logits.argmax(dim=-1)
                candidate_top1 = candidate_logits.argmax(dim=-1)

                top1_match = torch.equal(
                    reference_top1,
                    candidate_top1,
                )

                states[mode]["max_error"] = max(
                    states[mode]["max_error"],
                    max_error,
                )

                if top1_match:
                    states[mode]["matching_steps"] += 1

                if step < 5 or step == validation_steps - 1:
                    print(
                        f"Step {step + 1:02d} | "
                        f"{mode:7s} | "
                        f"max_logit_error={max_error:.8f} | "
                        f"top1_match={top1_match}"
                    )

            # Direct comparison: DynamicCache vs CustomCache
            dynamic_logits = states["dynamic"]["logits"].float()
            custom_logits = states["custom"]["logits"].float()

            dynamic_custom_error = (
                dynamic_logits - custom_logits
            ).abs().max().item()

            dynamic_custom_top1 = torch.equal(
                dynamic_logits.argmax(dim=-1),
                custom_logits.argmax(dim=-1),
            )

            if step == 0:
                max_dynamic_custom_error = 0.0
                dynamic_custom_matches = 0

            max_dynamic_custom_error = max(
                max_dynamic_custom_error,
                dynamic_custom_error,
            )

            if dynamic_custom_top1:
                dynamic_custom_matches += 1

            if step < 5 or step == validation_steps - 1:
                print(
                    f"Step {step + 1:02d} | "
                    f"dynamic-vs-custom | "
                    f"max_error={dynamic_custom_error:.8f} | "
                    f"top1_match={dynamic_custom_top1}"
                )

            # Append the identical reference token to all modes.
            for mode in ["none", "dynamic", "custom"]:

                state = states[mode]

                state["full_ids"] = torch.cat(
                    [state["full_ids"], reference_token],
                    dim=1,
                )

                state["mask"] = torch.cat(
                    [
                        state["mask"],
                        torch.ones(
                            (1, 1),
                            dtype=state["mask"].dtype,
                            device=input_ids.device,
                        ),
                    ],
                    dim=1,
                )

                with torch.inference_mode():

                    if mode == "none":
                        output = model(
                            input_ids=state["full_ids"],
                            attention_mask=state["mask"],
                            use_cache=False,
                        )

                    else:
                        output = model(
                            input_ids=reference_token,
                            attention_mask=state["mask"],
                            past_key_values=state["cache"],
                            use_cache=True,
                        )

                state["logits"] = output.logits[:, -1, :]

        # ---------------- VALIDATION SUMMARY ----------------

        print("\n========== VALIDATION SUMMARY ==========")

        for mode in ["dynamic", "custom"]:

            state = states[mode]

            print(
                f"{mode:8s} | "
                f"max_error={state['max_error']:.8f} | "
                f"top1 agreement="
                f"{state['matching_steps']}/{validation_steps}"
            )

            print(
                f"         cache length="
                f"{state['cache'].get_seq_length()}"
            )

        print("\nTeacher-forced validation completed.")

        print(
            "\nDynamic vs Custom maximum logit error:",
            max_dynamic_custom_error,
        )

        print(
            "Dynamic vs Custom top-1 agreement:",
            f"{dynamic_custom_matches}/{validation_steps}",
        )

    # ---------------- RUN BENCHMARKS ----------------

    modes = ["none", "dynamic", "custom"]
    results = {}

    for mode in modes:

        print(f"\n========== {mode.upper()} ==========")

        # Warmup
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
                f"throughput="
                f"{result['tokens_per_second']:.2f} tok/s"
            )

        results[mode] = measurements

    # ---------------- PERFORMANCE SUMMARY ----------------

    print("\n========== PERFORMANCE SUMMARY ==========")

    for mode in modes:

        measurements = results[mode]

        avg_prefill = sum(
            x["prefill_ms"] for x in measurements
        ) / REPEATS

        avg_decode = sum(
            x["decode_ms"] for x in measurements
        ) / REPEATS

        avg_total = sum(
            x["total_ms"] for x in measurements
        ) / REPEATS

        avg_throughput = sum(
            x["tokens_per_second"] for x in measurements
        ) / REPEATS

        print(
            f"{mode:10s} | "
            f"prefill={avg_prefill:9.2f} ms | "
            f"decode={avg_decode:9.2f} ms | "
            f"total={avg_total:9.2f} ms | "
            f"throughput={avg_throughput:8.2f} tok/s"
        )

    # ---------------- GREEDY OUTPUT COMPARISON ----------------

    reference = results["none"][0]["generated_ids"]

    print("\n========== GREEDY OUTPUT COMPARISON ==========")

    for mode in ["dynamic", "custom"]:

        candidate = results[mode][0]["generated_ids"]

        match = torch.equal(reference, candidate)

        print(f"{mode} matches no-cache: {match}")

    # ---------------- TEACHER-FORCED VALIDATION ----------------

    validate_teacher_forced(
        validation_steps=VALIDATION_STEPS
    )

    custom_memory = results["custom"][0]["cache_memory_mib"]

    print(
        "\nCustom cache allocated memory:",
        custom_memory,
        "MiB",
    )

    print("\nBENCHMARK COMPLETED")


@app.local_entrypoint()
def main():
    benchmark.remote()
