import torch
from transformers.cache_utils import Cache, CacheLayerMixin


class PreallocatedKVCacheLayer(CacheLayerMixin):
    """
    Preallocated KV storage for one transformer attention layer.

    Tensor layout:
        keys:   [batch, num_kv_heads, max_cache_len, head_dim]
        values: [batch, num_kv_heads, max_cache_len, head_dim]
    """

    is_sliding = False
    is_compileable = False
    is_croppable = False

    def __init__(self, max_cache_len):
        super().__init__()

        if max_cache_len <= 0:
            raise ValueError("max_cache_len must be positive")

        self.max_cache_len = max_cache_len
        self.seq_length = 0

    def lazy_initialization(self, key_states, value_states):
        self.dtype = key_states.dtype
        self.device = key_states.device

        batch_size, num_heads, _, key_dim = key_states.shape
        value_dim = value_states.shape[-1]

        self.keys = torch.empty(
            (batch_size, num_heads, self.max_cache_len, key_dim),
            dtype=self.dtype,
            device=self.device,
        )

        self.values = torch.empty(
            (batch_size, num_heads, self.max_cache_len, value_dim),
            dtype=self.dtype,
            device=self.device,
        )

        self.is_initialized = True

    def update(self, key_states, value_states, *args, **kwargs):
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)

        if key_states.shape[:2] != self.keys.shape[:2]:
            raise ValueError("Batch size or KV-head count changed")

        if key_states.shape[-1] != self.keys.shape[-1]:
            raise ValueError("Key head dimension changed")

        if value_states.shape[-1] != self.values.shape[-1]:
            raise ValueError("Value head dimension changed")

        if key_states.dtype != self.dtype or key_states.device != self.device:
            raise ValueError("Key tensor dtype/device changed")

        if value_states.dtype != self.dtype or value_states.device != self.device:
            raise ValueError("Value tensor dtype/device changed")

        new_tokens = key_states.shape[-2]
        start = self.seq_length
        end = start + new_tokens

        if end > self.max_cache_len:
            raise RuntimeError(
                f"KV cache capacity exceeded: requested {end}, "
                f"capacity {self.max_cache_len}"
            )

        # Append new K/V states into preallocated storage.
        self.keys[:, :, start:end, :].copy_(key_states)
        self.values[:, :, start:end, :].copy_(value_states)

        self.seq_length = end

        # Return only valid cached positions, not unused capacity.
        return (
            self.keys[:, :, :end, :],
            self.values[:, :, :end, :],
        )

    def get_seq_length(self):
        return self.seq_length

    def get_max_length(self):
        return self.max_cache_len

    def get_mask_sizes(self, query_length):
        # Called before the current query is inserted into the cache.
        return self.seq_length + query_length, 0

    def reset(self):
        # Old values remain in allocated storage, but are inaccessible
        # because seq_length is reset to zero.
        self.seq_length = 0


class PreallocatedKVCache(Cache):
    """
    Full-model KV cache containing one PreallocatedKVCacheLayer
    for every transformer layer.
    """

    def __init__(self, num_layers, max_cache_len):
        layers = [
            PreallocatedKVCacheLayer(max_cache_len)
            for _ in range(num_layers)
        ]

        super().__init__(layers=layers)

        self.max_cache_len = max_cache_len

    def reset(self):
        for layer in self.layers:
            layer.reset()

    def memory_bytes(self):
        total = 0

        for layer in self.layers:
            if layer.is_initialized:
                total += layer.keys.numel() * layer.keys.element_size()
                total += layer.values.numel() * layer.values.element_size()

        return total

    def memory_mib(self):
        return self.memory_bytes() / (1024 ** 2)
