"""CUDA Tile vs SDPA benchmark with adaptive tiles and correctness diagnostics.

Run: modal run benchmark_attention.py
Outputs locally: attention_benchmark_adaptive.csv
"""
import csv
import io
import math
import statistics

import modal

app = modal.App("qwen3-attention-adaptive-benchmark")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch", "cuda-tile[tileiras]")
    .add_local_file("attention_kernel.py", "/root/attention_kernel.py")
)

# (batch, query length, KV length, query heads, KV heads, head dimension, dtype)
CASES = [
    (2, 16, 16, 4, 2, 16, "float32"),
    (1, 32, 32, 4, 2, 16, "float32"),
    (2, 64, 64, 4, 2, 16, "float32"),
    (1, 128, 128, 8, 2, 32, "float32"),
    (2, 128, 128, 8, 2, 32, "float32"),
    (4, 256, 256, 8, 2, 32, "float32"),
    # Qwen3-like GQA (32 query heads, 8 KV heads, head dim 128)
    (1, 128, 128, 32, 8, 128, "bfloat16"),
    (2, 1, 128, 32, 8, 128, "bfloat16"),
    (2, 1, 512, 32, 8, 128, "bfloat16"),
    (2, 1, 1024, 32, 8, 128, "bfloat16"),
]


def choose_tile_size(kv_len):
    """Single K/V tile only: avoids known incorrect multi-tile accumulation."""
    if not 1 <= kv_len <= 1024:
        raise ValueError("Only 1 <= KV length <= 1024 is currently validated")
    return max(16, 1 << (kv_len - 1).bit_length())


@app.function(image=image, gpu="L40S", timeout=3600)
def benchmark():
    import torch
    import torch.nn.functional as F
    from attention_kernel import cutile_attention

    torch.manual_seed(1234)
    device = "cuda"
    warmup, repetitions = 5, 20
    rows = []

    def gpu_ms(fn):
        # GPU events measure GPU elapsed time, excluding most Python overhead.
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        samples = []
        for _ in range(repetitions):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end))
        return statistics.median(samples)

    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    print(f"Warmups: {warmup}; repetitions: {repetitions}", flush=True)
    print("NOTE: fixed and adaptive implementations use the same kernel, different tile_n.", flush=True)
    print("NOTE: each result must pass numerical checks before its timing is trusted.\n", flush=True)

    for index, (B, Sq, Sk, Hq, Hkv, D, dtype_name) in enumerate(CASES, 1):
        dtype = getattr(torch, dtype_name)
        torch.manual_seed(1234 + index)
        q = torch.randn((B, Hq, Sq, D), device=device, dtype=dtype)
        k = torch.randn((B, Hkv, Sk, D), device=device, dtype=dtype)
        v = torch.randn((B, Hkv, Sk, D), device=device, dtype=dtype)

        # PyTorch's is_causal=True is not the correct reference for right-aligned
        # cached decoding when Sq < Sk. Build the exact mask explicitly.
        q_pos = torch.arange(Sq, device=device) + (Sk - Sq)
        k_pos = torch.arange(Sk, device=device)
        mask = (k_pos[None, :] <= q_pos[:, None])[None, None, :, :]

        def sdpa():
            return F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, dropout_p=0.0, enable_gqa=True
            )

        adaptive_tile = choose_tile_size(Sk)
        atol = rtol = 2e-2 if dtype == torch.bfloat16 else 1e-4
        row = {
            "batch": B, "sq": Sq, "sk": Sk, "hq": Hq, "hkv": Hkv,
            "head_dim": D, "dtype": dtype_name, "adaptive_tile": adaptive_tile,
            "adaptive_status": "NOT_RUN", "adaptive_max_error": "",
            "adaptive_gpu_ms": "", "fixed_1024_status": "NOT_RUN",
            "fixed_1024_max_error": "", "fixed_1024_gpu_ms": "",
            "sdpa_gpu_ms": "", "adaptive_vs_sdpa": "",
            "fixed_vs_sdpa": "", "adaptive_vs_fixed": "",
            "notes": "",
        }
        print(f"[{index}/{len(CASES)}] B={B}, Sq={Sq}, Sk={Sk}, Hq={Hq}, D={D}, dtype={dtype_name}", flush=True)
        try:
            reference = sdpa()
            row["sdpa_gpu_ms"] = round(gpu_ms(sdpa), 6)
        except Exception as exc:
            row["notes"] = f"SDPA error: {type(exc).__name__}: {exc}"
            print("  SDPA ERROR:", row["notes"], flush=True)
            rows.append(row)
            continue

        for label, tile in (("adaptive", adaptive_tile), ("fixed_1024", 1024)):
            def run_tile(tile_n=tile):
                return cutile_attention(q, k, v, tile_n=tile_n)

            try:
                actual = run_tile()
                torch.cuda.synchronize()
                max_error = (actual.float() - reference.float()).abs().max().item()
                row[f"{label}_max_error"] = round(max_error, 8)
                torch.testing.assert_close(actual, reference, atol=atol, rtol=rtol)
                row[f"{label}_status"] = "PASS"
                row[f"{label}_gpu_ms"] = round(gpu_ms(run_tile), 6)
                row[f"{label}_vs_sdpa"] = round(
                    row["sdpa_gpu_ms"] / row[f"{label}_gpu_ms"], 4
                )
                print(
                    f"  {label:10s} tile={tile:4d}: PASS, max_error={max_error:.7g}, "
                    f"GPU={row[f'{label}_gpu_ms']:.4f} ms, "
                    f"vs SDPA={row[f'{label}_vs_sdpa']:.3f}x", flush=True
                )
            except Exception as exc:
                row[f"{label}_status"] = "FAIL"
                detail = f"{label}: {type(exc).__name__}: {str(exc).splitlines()[0]}"
                row["notes"] = (row["notes"] + " | " + detail).strip(" |")
                print("  FAIL:", detail, flush=True)

        if row["adaptive_status"] == "PASS" and row["fixed_1024_status"] == "PASS":
            row["adaptive_vs_fixed"] = round(
                row["fixed_1024_gpu_ms"] / row["adaptive_gpu_ms"], 4
            )
        rows.append(row)

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    print("\n=== Summary ===", flush=True)
    print("Cases:", len(rows), flush=True)
    print("Adaptive PASS:", sum(r["adaptive_status"] == "PASS" for r in rows), flush=True)
    print("Fixed 1024 PASS:", sum(r["fixed_1024_status"] == "PASS" for r in rows), flush=True)
    print("CSV returned to local entrypoint.", flush=True)
    return buf.getvalue()


@app.local_entrypoint()
def main():
    csv_text = benchmark.remote()
    filename = "attention_benchmark_adaptive.csv"
    with open(filename, "w", encoding="utf-8", newline="") as f:
        f.write(csv_text)
    print("Saved:", filename)
