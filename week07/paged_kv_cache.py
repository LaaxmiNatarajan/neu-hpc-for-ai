
import torch


class PagedKVCache:
    """
    Functional paged KV cache prototype.

    Tensor layout:
        Input K/V: [1, num_kv_heads, num_tokens, head_dim]

    Physical storage:
        [num_layers, num_blocks, num_kv_heads, block_size, head_dim]

    Each layer maintains its own block allocator, block tables,
    and sequence lengths.
    """

    def __init__(
        self,
        num_layers,
        num_blocks,
        num_kv_heads,
        head_dim,
        block_size=16,
        dtype=torch.bfloat16,
        device="cuda",
    ):
        if min(num_layers, num_blocks, num_kv_heads, head_dim, block_size) <= 0:
            raise ValueError("All dimensions must be positive")

        self.num_layers = num_layers
        self.num_blocks = num_blocks
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.dtype = dtype
        self.device = torch.device(device)

        # Normalize CUDA device to an explicit index, e.g. cuda:0.
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device(
            "cuda", torch.cuda.current_device()
            )

        self.key_pool = torch.empty(
            (num_layers, num_blocks, num_kv_heads, block_size, head_dim),
            dtype=dtype,
            device=self.device,
        )
        self.value_pool = torch.empty_like(self.key_pool)

        # Each layer has an independent physical block allocator.
        self.free_blocks = [
            list(range(num_blocks - 1, -1, -1))
            for _ in range(num_layers)
        ]

        # block_tables[layer][sequence_id] = [physical block IDs]
        self.block_tables = [{} for _ in range(num_layers)]

        # lengths[layer][sequence_id] = number of valid tokens
        self.lengths = [{} for _ in range(num_layers)]

    def create_sequence(self, sequence_id):
        """Create an empty sequence in every layer."""
        for layer_idx in range(self.num_layers):
            if sequence_id in self.block_tables[layer_idx]:
                raise ValueError(f"Sequence {sequence_id} already exists")

            self.block_tables[layer_idx][sequence_id] = []
            self.lengths[layer_idx][sequence_id] = 0

    def append(self, layer_idx, sequence_id, key_states, value_states):
        """
        Append K/V tokens to one sequence in one layer.

        key_states/value_states shape:
            [1, num_kv_heads, num_tokens, head_dim]
        """
        self._validate_layer(layer_idx)

        if sequence_id not in self.block_tables[layer_idx]:
            raise KeyError(f"Unknown sequence: {sequence_id}")

        if key_states.ndim != 4 or value_states.ndim != 4:
            raise ValueError("K/V tensors must have 4 dimensions")

        if key_states.shape != value_states.shape:
            raise ValueError("Key and value shapes must match")

        batch, heads, token_count, dim = key_states.shape

        if batch != 1:
            raise ValueError("This prototype supports one sequence per append")

        if heads != self.num_kv_heads or dim != self.head_dim:
            raise ValueError("Unexpected KV-head count or head dimension")

        if key_states.device != self.device or value_states.device != self.device:
            raise ValueError("K/V tensors must be on the cache device")

        if key_states.dtype != self.dtype or value_states.dtype != self.dtype:
            raise ValueError("K/V tensors must use the cache dtype")

        if token_count == 0:
            return

        current_length = self.lengths[layer_idx][sequence_id]
        new_length = current_length + token_count

        required_blocks = (
            new_length + self.block_size - 1
        ) // self.block_size

        table = self.block_tables[layer_idx][sequence_id]
        additional_blocks = required_blocks - len(table)

        if additional_blocks > len(self.free_blocks[layer_idx]):
            raise RuntimeError(
                f"Insufficient free blocks in layer {layer_idx}: "
                f"need {additional_blocks}, "
                f"have {len(self.free_blocks[layer_idx])}"
            )

        # Allocate all required blocks before writing.
        for _ in range(additional_blocks):
            table.append(self.free_blocks[layer_idx].pop())

        # Copy tokens across physical blocks.
        source_offset = 0

        while source_offset < token_count:
            absolute_position = current_length + source_offset
            logical_block = absolute_position // self.block_size
            block_offset = absolute_position % self.block_size

            physical_block = table[logical_block]

            count = min(
                token_count - source_offset,
                self.block_size - block_offset,
            )

            self.key_pool[
                layer_idx, physical_block, :, block_offset:block_offset + count, :
            ].copy_(
                key_states[0, :, source_offset:source_offset + count, :]
            )

            self.value_pool[
                layer_idx, physical_block, :, block_offset:block_offset + count, :
            ].copy_(
                value_states[0, :, source_offset:source_offset + count, :]
            )

            source_offset += count

        self.lengths[layer_idx][sequence_id] = new_length

    def get_kv(self, layer_idx, sequence_id):
        """
        Gather a sequence's valid pages into contiguous K/V tensors.

        Returns:
            key:   [1, num_kv_heads, sequence_length, head_dim]
            value: [1, num_kv_heads, sequence_length, head_dim]
        """
        self._validate_layer(layer_idx)

        if sequence_id not in self.block_tables[layer_idx]:
            raise KeyError(f"Unknown sequence: {sequence_id}")

        length = self.lengths[layer_idx][sequence_id]
        table = self.block_tables[layer_idx][sequence_id]

        if length == 0:
            shape = (1, self.num_kv_heads, 0, self.head_dim)
            return (
                torch.empty(shape, dtype=self.dtype, device=self.device),
                torch.empty(shape, dtype=self.dtype, device=self.device),
            )

        key_pages = []
        value_pages = []
        remaining = length

        for physical_block in table:
            valid_tokens = min(remaining, self.block_size)

            key_page = self.key_pool[
                layer_idx, physical_block, :, :valid_tokens, :
            ].unsqueeze(0)

            value_page = self.value_pool[
                layer_idx, physical_block, :, :valid_tokens, :
            ].unsqueeze(0)

            key_pages.append(key_page)
            value_pages.append(value_page)

            remaining -= valid_tokens

            if remaining == 0:
                break

        return (
            torch.cat(key_pages, dim=2),
            torch.cat(value_pages, dim=2),
        )

    def get_block_table(self, layer_idx, sequence_id):
        self._validate_layer(layer_idx)
        return list(self.block_tables[layer_idx][sequence_id])

    def get_seq_length(self, layer_idx, sequence_id):
        self._validate_layer(layer_idx)
        return self.lengths[layer_idx][sequence_id]

    def release_sequence(self, sequence_id):
        """Release all physical blocks belonging to a sequence."""
        for layer_idx in range(self.num_layers):
            if sequence_id not in self.block_tables[layer_idx]:
                continue

            blocks = self.block_tables[layer_idx].pop(sequence_id)
            self.free_blocks[layer_idx].extend(blocks)
            del self.lengths[layer_idx][sequence_id]

    def memory_bytes(self):
        return (
            self.key_pool.numel() * self.key_pool.element_size()
            + self.value_pool.numel() * self.value_pool.element_size()
        )

    def memory_mib(self):
        return self.memory_bytes() / (1024 ** 2)

    def _validate_layer(self, layer_idx):
        if not 0 <= layer_idx < self.num_layers:
            raise IndexError(f"Invalid layer index: {layer_idx}")
