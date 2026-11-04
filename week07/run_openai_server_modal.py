
import modal

APP_NAME = "qwen3-openai-inference-server"

app = modal.App(APP_NAME)

MODEL_NAME = "Qwen/Qwen3-4B"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch",
        "numpy",
        "transformers",
        "accelerate",
        "sentencepiece",
        "fastapi",
        "uvicorn",
    )
    .add_local_file(
        "paged_kv_cache.py",
        remote_path="/root/paged_kv_cache.py",
    )
    .add_local_file(
        "inference_server/batched_paged_cache.py",
        remote_path="/root/batched_paged_cache.py",
    )
    .add_local_file(
        "inference_server/request_state.py",
        remote_path="/root/request_state.py",
    )
    .add_local_file(
        "inference_server/vectorized_continuous_batch_engine.py",
        remote_path="/root/vectorized_continuous_batch_engine.py",
    )
    .add_local_file(
        "inference_server/openai_server.py",
        remote_path="/root/openai_server.py",
    )
)


@app.function(
    image=image,
    gpu="H100",
    timeout=3600,
    scaledown_window=300,
)
@modal.asgi_app()
def serve():
    import sys
    import torch

    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
    )

    sys.path.insert(0, "/root")

    from vectorized_continuous_batch_engine import (
        VectorizedContinuousBatchEngine,
    )

    from openai_server import create_app

    print("=" * 65)
    print("QWEN3-4B OPENAI INFERENCE SERVER")
    print("=" * 65)

    print("GPU:", torch.cuda.get_device_name(0))
    print("Model:", MODEL_NAME)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to("cuda")

    model.eval()

    print("Model loaded successfully.")

    engine = VectorizedContinuousBatchEngine(
        model=model,
        tokenizer=tokenizer,
        max_batch_size=4,
        max_seq_len=2048,
        block_size=16,
        eos_enabled=True,
    )

    print("Vectorized continuous batching engine initialized.")

    return create_app(engine, tokenizer)


@app.local_entrypoint()
def main():
    print("Deploying Qwen3-4B inference server...")
