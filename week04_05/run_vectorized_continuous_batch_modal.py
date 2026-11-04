import modal


APP_NAME = "qwen3-vectorized-continuous-batching-test"

app = modal.App(APP_NAME)

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
        "inference_server/test_vectorized_continuous_batch.py",
        remote_path="/root/test_vectorized_continuous_batch.py",
    )
)


@app.function(
    image=image,
    gpu="L40S",
    timeout=1800,
)
def run_test():
    import sys

    sys.path.insert(0, "/root")

    from test_vectorized_continuous_batch import main

    main()


@app.local_entrypoint()
def main():
    run_test.remote()
