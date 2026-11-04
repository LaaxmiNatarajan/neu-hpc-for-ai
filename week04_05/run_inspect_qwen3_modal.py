
import modal

app = modal.App("qwen3-vectorized-api-inspection")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch",
        "numpy",
        "transformers",
        "accelerate",
        "sentencepiece",
    )
    .add_local_file(
        "inference_server/inspect_qwen3_vectorized.py",
        remote_path="/root/inspect_qwen3_vectorized.py",
    )
)


@app.function(
    image=image,
    gpu="L40S",
    timeout=1200,
)
def inspect_model():
    import sys

    sys.path.insert(0, "/root")

    from inspect_qwen3_vectorized import main

    main()


@app.local_entrypoint()
def main():
    inspect_model.remote()
