import modal

app = modal.App("qwen3-cache-inspection")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch", "numpy", "transformers", "accelerate")
)

@app.function(image=image, timeout=600)
def inspect():
    import torch
    import transformers
    import inspect as pyinspect
    from transformers import AutoConfig
    from transformers.cache_utils import (
        Cache,
        DynamicCache,
        StaticCache,
    )

    print("\n========== ENVIRONMENT ==========")
    print("PyTorch:", torch.__version__)
    print("Transformers:", transformers.__version__)

    print("\n========== QWEN3 CONFIG ==========")
    config = AutoConfig.from_pretrained("Qwen/Qwen3-4B")
    for key in [
        "model_type",
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "max_position_embeddings",
    ]:
        print(f"{key}: {getattr(config, key, 'Not available')}")

    print("\n========== CACHE CLASSES ==========")
    for cls in [Cache, DynamicCache, StaticCache]:
        print(f"\n{cls.__name__}")
        try:
            print("Constructor:", pyinspect.signature(cls))
        except Exception as e:
            print("Constructor signature unavailable:", e)

        for method_name in ["update", "get_seq_length", "get_max_cache_shape"]:
            method = getattr(cls, method_name, None)
            if method:
                try:
                    print(f"{method_name}: {pyinspect.signature(method)}")
                except Exception:
                    print(f"{method_name}: signature unavailable")

    print("\n========== CACHE LAYER SUPPORT ==========")
    try:
        from transformers.cache_utils import CacheLayerMixin
        print("CacheLayerMixin available")
        print("Methods:", [
            name for name in dir(CacheLayerMixin)
            if not name.startswith("_")
        ])
    except ImportError:
        print("CacheLayerMixin unavailable in this version")

@app.local_entrypoint()
def main():
    inspect.remote()
