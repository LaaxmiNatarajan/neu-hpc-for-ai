import inspect
import torch
import transformers

import transformers.models.qwen3.modeling_qwen3 as qwen3
from transformers.cache_utils import Cache, CacheLayerMixin

def show_source(name, obj):
    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)
    try:
        print("SIGNATURE:", inspect.signature(obj))
        print(inspect.getsource(obj))
    except Exception as exc:
        print("Could not inspect:", repr(exc))

def main():
    print("Transformers:", transformers.__version__)
    print("PyTorch:", torch.__version__)

    # How the model converts attention_mask into its internal causal mask.
    show_source(
        "create_causal_mask",
        qwen3.create_causal_mask,
    )

    # How the model applies attention.
    show_source(
        "eager_attention_forward",
        qwen3.eager_attention_forward,
    )

    # Inspect the decoder layer's handling of masks and cache.
    show_source(
        "Qwen3DecoderLayer.forward",
        qwen3.Qwen3DecoderLayer.forward,
    )

    # Inspect cache methods used by the model.
    show_source("Cache.get_seq_length", Cache.get_seq_length)
    show_source("Cache.get_mask_sizes", Cache.get_mask_sizes)
    show_source("CacheLayerMixin.update", CacheLayerMixin.update)
    show_source("CacheLayerMixin.get_seq_length", CacheLayerMixin.get_seq_length)

    # Inspect available attention implementations.
    print("\n" + "=" * 80)
    print("ATTENTION IMPLEMENTATIONS")
    print("=" * 80)

    registry = qwen3.ALL_ATTENTION_FUNCTIONS
    print(registry)

    for name in ["sdpa", "eager", "flash_attention_2"]:
        try:
            fn = registry.get_interface(name, qwen3.eager_attention_forward)
            print(f"\n{name}: {fn}")
            print("Signature:", inspect.signature(fn))
            print(inspect.getsource(fn))
        except Exception as exc:
            print(name, "inspection failed:", repr(exc))

    print("\nINSPECTION COMPLETED")

if __name__ == "__main__":
    main()
