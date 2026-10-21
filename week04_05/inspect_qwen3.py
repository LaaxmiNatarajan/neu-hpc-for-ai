import modal

app = modal.App("inspect-qwen3")

MODEL_ID = "Qwen/Qwen3-4B"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers",
        "accelerate",
        "safetensors",
    )
)


@app.function(
    image=image,
    gpu="L40S",
    timeout=1800,
)
def inspect_model():
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is unavailable.")

    print("GPU:", torch.cuda.get_device_name(0))

    config = AutoConfig.from_pretrained(MODEL_ID)

    print("\n=== Model Configuration ===")

    for field in [
        "model_type",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "vocab_size",
        "max_position_embeddings",
        "torch_dtype",
    ]:
        print(f"{field}: {getattr(config, field, 'not specified')}")

    print("\n=== Attention Configuration ===")

    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    head_dim = getattr(config, "head_dim", None)

    if head_dim is None:
        head_dim = config.hidden_size // num_heads

    print("Query heads:", num_heads)
    print("KV heads:", num_kv_heads)
    print("Head dimension:", head_dim)
    print("GQA groups:", num_heads // num_kv_heads)

    print("\n=== Loading Hugging Face Model ===")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
    ).to("cuda").eval()

    print("Model loaded successfully")

    print("\n=== Attention Module ===")

    attention = model.model.layers[0].self_attn

    print(attention)

    print("\n=== Forward Pass Test ===")

    input_ids = torch.tensor(
        [[151644, 872, 198, 9707, 151645]],
        device="cuda",
        dtype=torch.long,
    )

    with torch.inference_mode():
        outputs = model(input_ids=input_ids)

    print("Logits shape:", tuple(outputs.logits.shape))
    print("Logits finite:", torch.isfinite(outputs.logits).all().item())

    assert torch.isfinite(outputs.logits).all(), "Non-finite logits detected"

    print("\nBaseline model forward pass: PASS")


@app.local_entrypoint()
def main():
    inspect_model.remote()