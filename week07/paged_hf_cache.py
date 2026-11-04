
import torch

from transformers.cache_utils import Cache, CacheLayerMixin

from paged_kv_cache import PagedKVCache


class PagedCacheLayer(CacheLayerMixin):
    """
    Hugging Face CacheLayerMixin adapter for PagedKVCache.
    """

    is_sliding = False
    is_compileable = False
    is_croppable = False

    def __init__(self, manager, layer_idx, sequence_id):
        super().__init__()

        self.manager = manager
        self.layer_idx = layer_idx
        self.sequence_id = sequence_id

        # The manager allocates the physical pools during construction.
        self.is_initialized = True

    def lazy_initialization(self, key_states, value_states):
        # Physical storage is allocated by PagedKVCache.
        self.is_initialized = True

    def update(self, key_states, value_states, *args, **kwargs):
        self.manager.append(
            self.layer_idx,
            self.sequence_id,
            key_states,
            value_states,
        )

        return self.manager.get_kv(
            self.layer_idx,
            self.sequence_id,
        )

    def get_seq_length(self):
        return self.manager.get_seq_length(
            self.layer_idx,
            self.sequence_id,
        )

    def get_max_length(self):
        return self.manager.num_blocks * self.manager.block_size

    def get_mask_sizes(self, query_length):
        return self.get_seq_length() + query_length, 0

    def reset(self):
        # Sequence-level release is handled by PagedTransformersCache.
        pass


class PagedTransformersCache(Cache):
    """
    Hugging Face Cache wrapper around the PagedKVCache manager.

    This adapter supports one active sequence. The underlying manager
    supports multiple sequence IDs, but this wrapper intentionally uses
    one sequence while validating Qwen3 integration.
    """

    def __init__(
        self,
        config,
        max_cache_len=2048,
        block_size=16,
        num_blocks=None,
        dtype=torch.bfloat16,
        device="cuda",
        sequence_id="qwen3_request",
    ):
        num_layers = config.num_hidden_layers
        num_kv_heads = config.num_key_value_heads
        head_dim = getattr(
            config,
            "head_dim",
            config.hidden_size // config.num_attention_heads,
        )

        required_blocks = (
            max_cache_len + block_size - 1
        ) // block_size

        if num_blocks is None:
            num_blocks = required_blocks

        if num_blocks < required_blocks:
            raise ValueError(
                f"num_blocks={num_blocks} is insufficient for "
                f"max_cache_len={max_cache_len}; "
                f"at least {required_blocks} blocks are required"
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

        self.sequence_id = sequence_id
        self.max_cache_len = max_cache_len

        self.manager.create_sequence(sequence_id)

        layers = [
            PagedCacheLayer(
                manager=self.manager,
                layer_idx=layer_idx,
                sequence_id=sequence_id,
            )
            for layer_idx in range(num_layers)
        ]

        super().__init__(layers=layers)

    def get_seq_length(self, layer_idx=0):
        return self.manager.get_seq_length(
            layer_idx,
            self.sequence_id,
        )

    def get_max_cache_shape(self, layer_idx=0):
        return self.max_cache_len

    def get_max_length(self, layer_idx=0):
        return self.max_cache_len

    def reset(self):
        self.manager.release_sequence(self.sequence_id)
        self.manager.create_sequence(self.sequence_id)

    def memory_bytes(self):
        return self.manager.memory_bytes()

    def memory_mib(self):
        return self.manager.memory_mib()

    def get_block_table(self, layer_idx=0):
        return self.manager.get_block_table(
            layer_idx,
            self.sequence_id,
        )
