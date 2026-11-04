
import torch

from transformers.cache_utils import Cache, CacheLayerMixin

from paged_kv_cache import PagedKVCache
from batched_paged_cache import BatchedPagedCache


class BatchedPagedCacheLayer(CacheLayerMixin):
    """
    Hugging Face cache layer for a batch of independent requests.

    This initial implementation requires all active requests to have
    equal sequence lengths. Each request still owns separate physical
    KV blocks in the PagedKVCache manager.
    """

    is_sliding = False
    is_compileable = False
    is_croppable = False

    def __init__(self, paged_cache, request_ids, layer_idx):
        super().__init__()

        self.paged_cache = paged_cache
        self.request_ids = list(request_ids)
        self.layer_idx = layer_idx
        self.is_initialized = True

    def lazy_initialization(self, key_states, value_states):
        self.is_initialized = True

    def _check_equal_lengths(self):
        lengths = [
            self.paged_cache.get_seq_length(rid, self.layer_idx)
            for rid in self.request_ids
        ]

        if len(set(lengths)) > 1:
            raise NotImplementedError(
                "This initial batched adapter requires equal sequence "
                f"lengths. Current lengths: {lengths}"
            )

        return lengths[0] if lengths else 0

    def update(self, key_states, value_states, *args, **kwargs):
        if key_states.shape[0] != len(self.request_ids):
            raise ValueError(
                f"Expected batch size {len(self.request_ids)}, "
                f"received {key_states.shape[0]}"
            )

        # Split the incoming batch into independent request sequences.
        for batch_idx, request_id in enumerate(self.request_ids):
            self.paged_cache.append(
                request_id,
                self.layer_idx,
                key_states[batch_idx:batch_idx + 1],
                value_states[batch_idx:batch_idx + 1],
            )

        self._check_equal_lengths()

        # Reconstruct a batch-shaped KV tensor for Hugging Face attention.
        keys = []
        values = []

        for request_id in self.request_ids:
            key, value = self.paged_cache.get_kv(
                request_id,
                self.layer_idx,
            )
            keys.append(key)
            values.append(value)

        return torch.cat(keys, dim=0), torch.cat(values, dim=0)

    def get_seq_length(self):
        return self._check_equal_lengths()

    def get_max_length(self):
        return (
            self.paged_cache.manager.num_blocks
            * self.paged_cache.manager.block_size
        )

    def get_mask_sizes(self, query_length):
        return self.get_seq_length() + query_length, 0

    def reset(self):
        # Request-level cleanup is handled by the parent cache.
        pass


class BatchedPagedTransformersCache(Cache):
    """
    Hugging Face Cache wrapper for multiple independent requests.

    Equal-length requests only in this initial version.
    """

    def __init__(
        self,
        config,
        request_ids,
        max_cache_len=64,
        block_size=16,
        num_blocks=None,
        dtype=torch.bfloat16,
        device="cuda",
    ):
        request_ids = [str(rid) for rid in request_ids]

        if not request_ids:
            raise ValueError("At least one request ID is required")

        if len(set(request_ids)) != len(request_ids):
            raise ValueError("Request IDs must be unique")

        self.request_ids = request_ids
        self.max_cache_len = max_cache_len
        self.block_size = block_size

        num_layers = config.num_hidden_layers
        num_kv_heads = config.num_key_value_heads
        head_dim = getattr(
            config,
            "head_dim",
            config.hidden_size // config.num_attention_heads,
        )

        blocks_per_request = (
            max_cache_len + block_size - 1
        ) // block_size

        required_blocks = len(request_ids) * blocks_per_request

        if num_blocks is None:
            num_blocks = required_blocks

        if num_blocks < required_blocks:
            raise ValueError(
                f"Need at least {required_blocks} physical blocks "
                f"for {len(request_ids)} requests; got {num_blocks}"
            )

        self.manager = PagedKVCache(
            num_layers=num_layers,
            num_blocks=num_blocks,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            block_size=block_size,
            dtype=dtype,
            device=device,
        )

        self.paged_cache = BatchedPagedCache(self.manager)

        for request_id in self.request_ids:
            self.paged_cache.create_request(request_id)

        layers = [
            BatchedPagedCacheLayer(
                paged_cache=self.paged_cache,
                request_ids=self.request_ids,
                layer_idx=layer_idx,
            )
            for layer_idx in range(num_layers)
        ]

        super().__init__(layers=layers)

    def get_seq_length(self, layer_idx=0):
        lengths = [
            self.paged_cache.get_seq_length(rid, layer_idx)
            for rid in self.request_ids
        ]

        if len(set(lengths)) > 1:
            raise NotImplementedError(
                f"Unequal sequence lengths are not supported: {lengths}"
            )

        return lengths[0] if lengths else 0

    def get_max_cache_shape(self, layer_idx=0):
        return self.max_cache_len

    def get_max_length(self, layer_idx=0):
        return self.max_cache_len

    def get_block_table(self, request_id, layer_idx=0):
        return self.paged_cache.get_block_table(
            request_id,
            layer_idx,
        )

    def release_request(self, request_id):
        self.paged_cache.release_request(request_id)

    def reset(self):
        for request_id in self.request_ids:
            self.paged_cache.release_request(request_id)

        for request_id in self.request_ids:
            self.paged_cache.create_request(request_id)

    def memory_bytes(self):
        return self.manager.memory_bytes()

    def memory_mib(self):
        return self.manager.memory_mib()

