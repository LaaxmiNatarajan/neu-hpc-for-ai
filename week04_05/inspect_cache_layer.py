import modal

app = modal.App("qwen3-cache-layer-inspection")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch", "numpy", "transformers", "accelerate")
)

@app.function(image=image, timeout=600)
def inspect():
    import inspect
    import transformers
    import transformers.cache_utils as cache_utils

    print("\n========== PACKAGE VERSIONS ==========")
    print("Transformers:", transformers.__version__)

    print("\n========== CACHE LAYER SOURCE ==========")
    classes = [
        cache_utils.CacheLayerMixin,
        cache_utils.DynamicLayer,
        cache_utils.StaticLayer,
    ]

    for cls in classes:
        print(f"\n\n{'=' * 20} {cls.__name__} {'=' * 20}")
        try:
            print(inspect.getsource(cls))
        except Exception as e:
            print("Could not inspect:", e)

    print("\n========== QWEN3 ATTENTION SOURCE ==========")
    try:
        from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention

        print("Forward signature:", inspect.signature(Qwen3Attention.forward))
        print(inspect.getsource(Qwen3Attention.forward))
    except Exception as e:
        print("Could not inspect Qwen3Attention:", repr(e))

@app.local_entrypoint()
def main():
    inspect.remote()
