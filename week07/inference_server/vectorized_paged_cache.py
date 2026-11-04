import torch

from paged_kv_cache import PagedKVCache
from batched_paged_cache import BatchedPagedCache


class VectorizedPagedCache:
    """
    Per-request paged KV storage with batched KV gathering.

    Each request owns independent physical KV blocks.
    The returned KV tensors are padded to the longest active sequence.
    """

    def __init__(
        self,
        config,
        request_ids,
        max_cache_len=256,
        block_size=16,
        dtype=torch.bfloat16,
        device="cuda",
    ):
        self.request_ids = [str(x) for x in request_ids]
        if not self.request_ids:
            raise ValueError("At least one request is required.")
        if len(set(self.request_ids)) != len(self.request_ids):
            raise ValueError("Request IDs must be unique.")

        self.max_cache_len = max_cache_len
        self.device = torch.device(device)
        self.dtype = dtype
        self.num_layers = config.num_hidden_layers

        head_dim = getattr(
            config,
            "head_dim",
            config.hidden_size // config.num_attention_heads,
        )

        blocks_per_request = (
            max_cache_len + block_size - 1
        ) // block_size

        manager = PagedKVCache(
            num_layers=self.num_layers,
            num_blocks=len(self.request_ids) * blocks_per_request,
            num_kv_heads=config.num_key_value_heads,
            head_dim=head_dim,
            block_size=block_size,
            dtype=dtype,
            device=device,
        )

        self.manager = manager
        self.paged = BatchedPagedCache(manager)

        for request_id in self.request_ids:
            self.paged.create_request(request_id)

        # Number of new tokens to append for each request
        # during the next model forward pass.
        self.append_lengths = [0] * len(self.request_ids)

    def set_append_lengths(self, lengths):
        if len(lengths) != len(self.request_ids):
            raise ValueError("One append length is required per request.")
        if any(x < 1 for x in lengths):
            raise ValueError("Append lengths must be positive.")

        self.append_lengths = list(lengths)

    def get_lengths(self, layer_idx=0):
        return [
            self.paged.get_seq_length(request_id, layer_idx)
            for request_id in self.request_ids
        ]

    def get_seq_length(self, layer_idx=0):
        lengths = self.get_lengths(layer_idx)
        return max(lengths, default=0)

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        """
        Input shape:
            [batch_size, num_kv_heads, query_length, head_dim]

        The cache manager receives each request separately.
        Returned K/V tensors are padded to a common sequence length.
        """
        batch_size = key_states.shape[0]

        if batch_size != len(self.request_ids):
            raise ValueError("Batch size does not match request count.")

        if key_states.shape != value_states.shape:
            raise ValueError("Key and value shapes must match.")

        if key_states.shape[2] != max(self.append_lengths):
            raise ValueError(
                f"Input query length {key_states.shape[2]} does not match "
                f"maximum append length {max(self.append_lengths)}."
            )

        for batch_idx, request_id in enumerate(self.request_ids):
            append_length = self.append_lengths[batch_idx]

            key = key_states[
                batch_idx:batch_idx + 1, :, :append_length, :
            ]
            value = value_states[
                batch_idx:batch_idx + 1, :, :append_length, :
            ]

            self.paged.append(
                request_id,
                layer_idx,
                key,
                value,
            )

        gathered_keys = []
        gathered_values = []

        lengths = self.get_lengths(layer_idx)
        max_length = max(lengths)

        for request_id, length in zip(self.request_ids, lengths):
            key, value = self.paged.get_kv(request_id, layer_idx)

            if length < max_length:
                padding = max_length - length
                key = torch.nn.functional.pad(key, (0, 0, 0, padding))
                value = torch.nn.functional.pad(value, (0, 0, 0, padding))

            gathered_keys.append(key)
            gathered_values.append(value)

        return (
            torch.cat(gathered_keys, dim=0),
            torch.cat(gathered_values, dim=0),
        )

    def get_mask_sizes(self, query_length, layer_idx=0):
        return self.get_seq_length(layer_idx) + query_length, 0

    def get_max_length(self):
        return self.max_cache_len

    def get_max_cache_shape(self, layer_idx=0):
        return self.max_cache_len

    def get_block_table(self, request_id, layer_idx=0):
        return self.paged.get_block_table(request_id, layer_idx)

    def release_request(self, request_id):
        self.paged.release_request(request_id)

    def memory_bytes(self):
        return self.manager.memory_bytes()

    def memory_mib(self):
        return self.manager.memory_mib()
