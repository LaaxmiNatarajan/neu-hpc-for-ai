
import torch

from paged_kv_cache import PagedKVCache


def make_kv(tokens, heads=2, dim=8):
    key = torch.randn(1, heads, tokens, dim)
    value = torch.randn(1, heads, tokens, dim)
    return key, value


def check_close(actual, expected, name):
    torch.testing.assert_close(actual, expected)
    print(f"PASS: {name}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cache = PagedKVCache(
        num_layers=2,
        num_blocks=8,
        num_kv_heads=2,
        head_dim=8,
        block_size=4,
        dtype=torch.float32,
        device=device,
    )

    # Test 1: Create two independent sequences.
    cache.create_sequence("request_A")
    cache.create_sequence("request_B")

    # Test 2: Append 3 tokens to request A.
    k1, v1 = make_kv(3)
    cache.append(0, "request_A", k1.to(device), v1.to(device))

    # Append another 4 tokens, crossing a page boundary.
    k2, v2 = make_kv(4)
    cache.append(0, "request_A", k2.to(device), v2.to(device))

    expected_k = torch.cat([k1, k2], dim=2).to(device)
    expected_v = torch.cat([v1, v2], dim=2).to(device)

    actual_k, actual_v = cache.get_kv(0, "request_A")

    check_close(actual_k, expected_k, "Paged key reconstruction")
    check_close(actual_v, expected_v, "Paged value reconstruction")

    assert cache.get_seq_length(0, "request_A") == 7
    assert len(cache.get_block_table(0, "request_A")) == 2

    print("PASS: Page-boundary allocation")

    # Test 3: Request B remains independent.
    kb, vb = make_kv(5)
    cache.append(0, "request_B", kb.to(device), vb.to(device))

    actual_kb, actual_vb = cache.get_kv(0, "request_B")

    check_close(actual_kb, kb.to(device), "Request B keys")
    check_close(actual_vb, vb.to(device), "Request B values")

    assert cache.get_seq_length(0, "request_B") == 5
    print("PASS: Independent sequence storage")

    # Test 4: Layers maintain independent cache contents.
    layer1_k, layer1_v = make_kv(2)
    cache.append(1, "request_A", layer1_k.to(device), layer1_v.to(device))

    reconstructed_k, reconstructed_v = cache.get_kv(1, "request_A")

    check_close(reconstructed_k, layer1_k.to(device), "Layer 1 keys")
    check_close(reconstructed_v, layer1_v.to(device), "Layer 1 values")

    assert cache.get_seq_length(1, "request_A") == 2
    print("PASS: Independent layer storage")

    # Test 5: Release and reuse blocks.
    cache.release_sequence("request_A")

    assert cache.get_seq_length(0, "request_B") == 5

    cache.create_sequence("request_C")
    kc, vc = make_kv(6)

    cache.append(0, "request_C", kc.to(device), vc.to(device))

    actual_kc, actual_vc = cache.get_kv(0, "request_C")

    check_close(actual_kc, kc.to(device), "Reused blocks: keys")
    check_close(actual_vc, vc.to(device), "Reused blocks: values")

    print("PASS: Block release and reuse")

    print(f"\nCache memory: {cache.memory_mib():.4f} MiB")
    print("\nAll Paged KV Cache tests passed!")


if __name__ == "__main__":
    main()

