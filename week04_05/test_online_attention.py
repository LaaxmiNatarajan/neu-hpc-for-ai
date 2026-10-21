"""Modal correctness checks for fused CUDA Tile online-softmax GQA."""
import modal

app = modal.App("qwen3-online-attention-test")

# The compiler is required in the REMOTE Modal image, not just local WSL.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "cuda-tile[tileiras]")
    .add_local_file("attention_kernel.py", remote_path="/root/attention_kernel.py")
)


@app.function(image=image, gpu="L40S", timeout=600)
def test_attention():
    import sys
    sys.path.insert(0, "/root")
    import torch
    import torch.nn.functional as F
    from attention_kernel import cutile_attention

    torch.manual_seed(1234)
    print("PyTorch:", torch.__version__, flush=True)
    print("CUDA:", torch.version.cuda, flush=True)
    print("GPU:", torch.cuda.get_device_name(0), flush=True)

    # B, Hq, Hkv, Sq, Sk, D, tile_n
    configurations = [
    # B, Hq, Hkv, Sq, Sk, D, tile_n
    (1, 4, 2, 16, 16, 32, 16),  # Single tile
    (1, 4, 2, 32, 32, 32, 16),  # Two tiles
    (1, 4, 2, 48, 48, 32, 16),  # Three tiles
    (1, 4, 2, 64, 64, 32, 16),  # Four tiles
    (1, 4, 2, 64, 64, 32, 32),  # Two tiles
    (1, 4, 2, 64, 64, 32, 64),  # Single tile
    (1, 4, 2, 1, 64, 32, 16),   # Cached decoding
    (1, 4, 2, 16, 64, 32, 16),  # Cached decoding
    ]
    passed = 0
    total = 0

    for dtype in (torch.float32, torch.bfloat16):
        for B, Hq, Hkv, Sq, Sk, D, tile_n in configurations:
            total += 1
            q = torch.randn(B, Hq, Sq, D, device="cuda", dtype=dtype)
            k = torch.randn(B, Hkv, Sk, D, device="cuda", dtype=dtype)
            v = torch.randn(B, Hkv, Sk, D, device="cuda", dtype=dtype)

            actual = cutile_attention(q, k, v, tile_n=tile_n)

            query_positions = torch.arange(Sq, device="cuda") + Sk - Sq
            key_positions = torch.arange(Sk, device="cuda")
            allowed = key_positions[None, :] <= query_positions[:, None]
            mask = torch.zeros((Sq, Sk), device="cuda", dtype=torch.float32)
            mask = mask.masked_fill(~allowed, float("-inf"))

            reference = F.scaled_dot_product_attention(
                q.float(), k.float(), v.float(),
                attn_mask=mask,
                dropout_p=0.0,
                enable_gqa=True,
            ).to(dtype)

            delta = (actual.float() - reference.float()).abs()
            max_error = delta.max().item()
            mean_error = delta.mean().item()
            atol = 1e-3 if dtype == torch.float32 else 3e-2
            rtol = 1e-3 if dtype == torch.float32 else 3e-2
            ok = torch.allclose(actual.float(), reference.float(), atol=atol, rtol=rtol)
            passed += int(ok)
            print(
                f"{'PASS' if ok else 'FAIL'} "
                f"dtype={dtype} B={B} Hq={Hq} Hkv={Hkv} "
                f"Sq={Sq} Sk={Sk} D={D} tile={tile_n} "
                f"max_error={max_error:.8f} mean_error={mean_error:.8f}",
                flush=True,
            )

    print(f"\nRESULT: {passed}/{total} PASS", flush=True)
    if passed != total:
        raise AssertionError(f"{total - passed} attention tests failed")


@app.local_entrypoint()
def main():
    test_attention.remote()
