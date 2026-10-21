import modal

app = modal.App("qwen3-4b-baseline")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "torch",
        "numpy",
        "transformers",
        "accelerate",
    )
)


@app.function(image=image, gpu="L40S", timeout=3600)
def benchmark():
    import time
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    model_name = "Qwen/Qwen3-4B"

    print("GPU:", torch.cuda.get_device_name(0))
    print("Loading tokenizer and model:", model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to("cuda")
    model.eval()

    config = model.config
    print("Query heads:", config.num_attention_heads)
    print("KV heads:", config.num_key_value_heads)
    print(
        "Head dimension:",
        getattr(
            config,
            "head_dim",
            config.hidden_size // config.num_attention_heads,
        ),
    )
    print("Layers:", config.num_hidden_layers)

    prompt = (
        "Explain how grouped-query attention reduces "
        "memory usage during transformer inference."
    )
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    # Measure prompt processing separately from generation.
    with torch.inference_mode():
        model(**inputs, use_cache=False)
        torch.cuda.synchronize()

        timings = []
        for _ in range(3):
            start = time.perf_counter()
            model(**inputs, use_cache=False)
            torch.cuda.synchronize()
            timings.append((time.perf_counter() - start) * 1000)

        print("Prompt tokens:", inputs["input_ids"].shape[1])
        print("Prefill latency (ms):", [round(t, 2) for t in timings])
        print("Mean prefill latency (ms):", round(sum(timings) / len(timings), 2))

        # Measure end-to-end generation with the reference backend.
        start = time.perf_counter()
        output = model.generate(
            **inputs,
            max_new_tokens=16,
            do_sample=False,
            use_cache=True,
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

    generated = output.shape[1] - inputs["input_ids"].shape[1]
    print("Generated tokens:", generated)
    print("Generation time (s):", round(elapsed, 3))
    print("End-to-end generated-token throughput (tokens/s):",
          round(generated / elapsed, 2))
    print("Output:", tokenizer.decode(output[0], skip_special_tokens=True))
    print("PASS: Qwen3-4B reference inference completed.")


@app.local_entrypoint()
def main():
    benchmark.remote()

