def paged_attention(
    query,
    key_pool,
    value_pool,
    block_table,
    sequence_length,
    block_size,
    scale,
):
    """
    Calculate attention using physical K/V pages.

    query:
        [batch, query_heads, query_length, head_dim]

    key_pool / value_pool:
        Physical page storage.

    block_table:
        Maps logical page positions to physical page IDs.

    sequence_length:
        Number of valid cached tokens.
    """
    raise NotImplementedError
