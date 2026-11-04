
import modal

app = modal.App("qwen3-paged-kv-cache-test")

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
        "paged_hf_cache.py",
        "/root/paged_hf_cache.py",
    )
    .add_local_file(
        "test_paged_qwen3_cache.py",
        "/root/test_paged_qwen3_cache.py",
    )
)


@app.function(
    image=image,
    gpu="L40S",
    timeout=3600,
)
def run_test():
    import torch
    import transformers

    print("======================================")
    print("QWEN3 PAGED KV CACHE INTEGRATION TEST")
    print("======================================")

    print("PyTorch:", torch.__version__)
    print("Transformers:", transformers.__version__)
    print("GPU:", torch.cuda.get_device_name(0))

    from test_paged_qwen3_cache import main

    main()

    print("\nREMOTE TEST FINISHED")


@app.local_entrypoint()
def main():
    run_test.remote()
