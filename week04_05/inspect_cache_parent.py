import modal

app = modal.App("qwen3-cache-parent-inspection")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch", "numpy", "transformers", "accelerate")
)

@app.function(image=image, timeout=600)
def inspect():
    import inspect
    from transformers.cache_utils import Cache
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Model

    print("\n========== CACHE INITIALIZATION ==========")
    print(inspect.getsource(Cache.__init__))

    print("\n========== CACHE MASK SIZES ==========")
    print(inspect.getsource(Cache.get_mask_sizes))

    print("\n========== CACHE SEQUENCE LENGTH ==========")
    print(inspect.getsource(Cache.get_seq_length))

    print("\n========== QWEN3 MODEL FORWARD ==========")
    print(inspect.signature(Qwen3Model.forward))
    print(inspect.getsource(Qwen3Model.forward))

@app.local_entrypoint()
def main():
    inspect.remote()
