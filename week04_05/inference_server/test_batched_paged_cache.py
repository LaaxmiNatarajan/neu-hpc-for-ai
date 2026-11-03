
import torch

from paged_kv_cache import PagedKVCache
from batched_paged_cache import BatchedPagedCache


def check(condition, message):
    if not condition:
        raise AssertionError(message)
    print(f"PASS: {message}")


def main():
    device = "cpu"
    dtype = torch.float32

    manager = PagedKVCache(
        num_layers=2,
        num_blocks=32,
        num_kv_heads=4,
        head_dim=8,
        block_size=4,
        dtype=dtype,
        device=device,
    )

    cache = BatchedPagedCache(manager)

    cache.create_request("A")
    cache.create_request("B")

    # Request A: 6 tokens.
    key_a = torch.randn(1, 4, 6, 8, dtype=dtype)
    value_a = torch.randn(1, 4, 6, 8, dtype=dtype)

    # Request B: 10 tokens.
    key_b = torch.randn(1, 4, 10, 8, dtype=dtype)
    value_b = torch.randn(1, 4, 10, 8, dtype=dtype)

    for layer in range(2):
        cache.append("A", layer, key_a, value_a)
        cache.append("B", layer, key_b, value_b)

        recovered_key_a, recovered_value_a = cache.get_kv(
            "A", layer
        )
        recovered_key_b, recovered_value_b = cache.get_kv(
            "B", layer
        )

        check(
            torch.equal(recovered_key_a, key_a)
            and torch.equal(recovered_value_a, value_a),
            f"Request A reconstruction, layer {layer}",
        )

        check(
            torch.equal(recovered_key_b, key_b)
            and torch.equal(recovered_value_b, value_b),
            f"Request B reconstruction, layer {layer}",
        )

        check(
            cache.get_seq_length("A", layer) == 6,
            f"Request A sequence length, layer {layer}",
        )

        check(
            cache.get_seq_length("B", layer) == 10,
            f"Request B sequence length, layer {layer}",
        )

    # Release A without affecting B.
    cache.release_request("A")

    check(not cache.has_request("A"), "Request A released")

    check(cache.has_request("B"), "Request B remains active")

    recovered_key_b, recovered_value_b = cache.get_kv("B", 0)

    check(
        torch.equal(recovered_key_b, key_b)
        and torch.equal(recovered_value_b, value_b),
        "Request B data remains intact after releasing A",
    )

    cache.release_request("B")

    check(
        not cache.has_request("A")
        and not cache.has_request("B"),
        "All requests released",
    )

    print("\nAll batched paged cache tests passed!")


if __name__ == "__main__":
    main()
