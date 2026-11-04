import modal

app = modal.App("qwen3-batched-prefill-test")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "torch",
        "numpy",
        "transformers",
        "accelerate",
    )
    .add_local_file(
        "paged_kv_cache.py",
        "/root/paged_kv_cache.py",
    )
    .add_local_file(
        "inference_server/batched_paged_cache.py",
        "/root/batched_paged_cache.py",
    )
    .add_local_file(
        "inference_server/batched_paged_hf_cache.py",
        "/root/batched_paged_hf_cache.py",
    )
    .add_local_file(
        "inference_server/test_batched_qwen3.py",
        "/root/test_batched_qwen3.py",
    )
)


@app.function(
    image=image,
    gpu="L40S",
    timeout=3600,
)
def run_test():
    import sys

    sys.path.insert(0, "/root")

    from test_batched_qwen3 import main

    main()


@app.local_entrypoint()
def main():
    run_test.remote()
