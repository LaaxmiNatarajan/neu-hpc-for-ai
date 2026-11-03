
class BatchedPagedCache:
    """
    Multi-request interface over the existing PagedKVCache manager.

    Each request owns an independent sequence and block table.
    """

    def __init__(self, cache_manager):
        self.manager = cache_manager
        self.active_sequences = set()

    def create_request(self, request_id):
        sequence_id = str(request_id)

        if sequence_id in self.active_sequences:
            raise ValueError(
                f"Sequence {sequence_id} already exists"
            )

        self.manager.create_sequence(sequence_id)
        self.active_sequences.add(sequence_id)

        return sequence_id

    def append(self, request_id, layer_idx, key_states, value_states):
        sequence_id = str(request_id)

        if sequence_id not in self.active_sequences:
            raise KeyError(sequence_id)

        self.manager.append(
            layer_idx,
            sequence_id,
            key_states,
            value_states,
        )

    def get_kv(self, request_id, layer_idx):
        sequence_id = str(request_id)

        if sequence_id not in self.active_sequences:
            raise KeyError(sequence_id)

        return self.manager.get_kv(
            layer_idx,
            sequence_id,
        )

    def get_block_table(self, request_id, layer_idx):
        sequence_id = str(request_id)

        if sequence_id not in self.active_sequences:
            raise KeyError(sequence_id)

        return self.manager.get_block_table(
            layer_idx,
            sequence_id,
        )

    def get_seq_length(self, request_id, layer_idx):
        sequence_id = str(request_id)

        if sequence_id not in self.active_sequences:
            raise KeyError(sequence_id)

        return self.manager.get_seq_length(
            layer_idx,
            sequence_id,
        )

    def release_request(self, request_id):
        sequence_id = str(request_id)

        if sequence_id not in self.active_sequences:
            return

        self.manager.release_sequence(sequence_id)
        self.active_sequences.remove(sequence_id)

    def has_request(self, request_id):
        return str(request_id) in self.active_sequences
