"""Qwen3-4B batched CUDA Tile vs PyTorch SDPA benchmark matrix on Modal L40S.

Runs B=(1,2,4,8), prefill=(64,128,256,512), 16 greedy tokens.
Keeps each backend's model weights identical, starts with fresh KV cache,
and reports numerical differences without hiding tolerance failures.
"""
import csv
import io
import statistics
import time
from collections import Counter

import modal

app = modal.App("qwen3-4b-cutile-benchmark-matrix")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch", "numpy", "transformers", "accelerate", "cuda-tile[tileiras]")
    .add_local_file("attention_kernel.py", "/root/attention_kernel.py")
)

MODEL_ID = "Qwen/Qwen3-4B"
BATCHES = (1, 2, 4, 8)
PREFILL_LENGTHS = (64, 128, 256, 512)
NEW_TOKENS = 16
WARMUPS = 1
REPEATS = 3
MAX_KV = 1024
CSV_NAME = "batched_qwen3_matrix_results.csv"


@app.function(image=image, gpu="L40S", timeout=7200)
def run_benchmark():
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        AttentionInterface,
        AttentionMaskInterface,
    )
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    from transformers.masking_utils import sdpa_mask
    from attention_kernel import cutile_attention

    torch.manual_seed(1234)
    device = "cuda"
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    print("Model:", MODEL_ID, flush=True)
    print("Batches:", BATCHES, "Prefill lengths:", PREFILL_LENGTHS, flush=True)
    print("Generated tokens:", NEW_TOKENS, flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(device).eval()

    phrase = (
        "Explain grouped-query attention and how it reduces the "
        "key-value cache memory footprint in transformer inference. "
    )
    base_tokens = tokenizer.encode(phrase, add_special_tokens=False)
    if not base_tokens:
        raise RuntimeError("Tokenizer returned no prompt tokens")

    counters = Counter()
    fallback_reasons = Counter()
    tile_usage = Counter()

    def standard_causal_mask(mask, q_len, kv_len):
        if not (0 < q_len <= kv_len):
            return False
        if mask is None:
            return q_len == kv_len or q_len == 1
        if mask.ndim != 4 or mask.shape[-2:] != (q_len, kv_len):
            return False
        if mask.dtype == torch.bool:
            allowed = mask
        elif mask.dtype.is_floating_point:
            if bool((mask > 0).any().item()):
                return False
            allowed = mask == 0
        else:
            return False
        qp = torch.arange(q_len, device=mask.device) + kv_len - q_len
        kp = torch.arange(kv_len, device=mask.device)
        expected = (kp[None, :] <= qp[:, None])[None, None, :, :]
        return torch.equal(allowed, expected.expand_as(allowed))

    def custom_attention(module, query, key, value, attention_mask, **kwargs):
        sq, sk = query.shape[-2], key.shape[-2]
        dropout = kwargs.get("dropout", 0.0)
        sliding = kwargs.get("sliding_window", None)
        supported = (
            query.is_cuda and key.is_cuda and value.is_cuda
            and query.dtype in (torch.float32, torch.bfloat16)
            and query.dtype == key.dtype == value.dtype
            and 0 < sq <= sk <= MAX_KV
            and dropout == 0.0 and sliding is None
            and standard_causal_mask(attention_mask, sq, sk)
        )
        if supported:
            # Keep the same adaptive tile selection as the existing benchmark.
            tile = max(16, 1 << (sk - 1).bit_length())
            output = cutile_attention(query, key, value, tile_n=tile)
            counters["cutile"] += 1
            tile_usage[tile] += 1
            return output.transpose(1, 2).contiguous(), None

        counters["fallback"] += 1
        if sk > MAX_KV:
            reason = "kv exceeds MAX_KV"
        elif sliding is not None:
            reason = "sliding-window attention"
        elif dropout != 0.0:
            reason = "nonzero dropout"
        elif not standard_causal_mask(attention_mask, sq, sk):
            reason = "nonstandard mask"
        else:
            reason = "unsupported input"
        fallback_reasons[reason] += 1
        return sdpa_attention_forward(
            module, query, key, value, attention_mask, **kwargs
        )

    AttentionInterface.register("cutile_matrix", custom_attention)
    AttentionMaskInterface.register("cutile_matrix", sdpa_mask)

    @torch.inference_mode()
    def greedy_run(input_ids, backend, capture=False):
        """Run one prefill plus NEW_TOKENS-1 cached decode steps."""
        model.set_attn_implementation(backend)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        outputs = model(input_ids=input_ids, use_cache=True, return_dict=True)
        next_token = outputs.logits[:, -1:, :].argmax(dim=-1)
        prefill_logits = outputs.logits[:, -1, :].float().cpu() if capture else None
        cache = outputs.past_key_values
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        generated = [next_token.cpu()] if capture else None
        for _ in range(NEW_TOKENS - 1):
            outputs = model(
                input_ids=next_token,
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
            cache = outputs.past_key_values
            next_token = outputs.logits[:, -1:, :].argmax(dim=-1)
            if capture:
                generated.append(next_token.cpu())
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        return {
            "prefill_ms": (t1 - t0) * 1000,
            "decode_ms": (t2 - t1) * 1000,
            "total_ms": (t2 - t0) * 1000,
            "tokens": torch.cat(generated, dim=1) if capture else None,
            "logits": prefill_logits,
        }

    keys = [
        "batch", "prefill_tokens", "new_tokens", "status", "token_match",
        "logit_tolerance_passed", "prefill_logit_max_error",
        "prefill_logit_mean_error", "sdpa_prefill_ms", "cutile_prefill_ms",
        "sdpa_decode_ms", "cutile_decode_ms", "sdpa_total_ms",
        "cutile_total_ms", "sdpa_tokens_per_s", "cutile_tokens_per_s",
        "speedup_vs_sdpa", "sdpa_peak_allocated_gib",
        "cutile_peak_allocated_gib", "cutile_calls_all_runs",
        "sdpa_fallbacks_all_runs", "tile_usage", "fallback_reasons", "error",
    ]
    rows = []
    for prefill in PREFILL_LENGTHS:
        token_ids = (base_tokens * ((prefill // len(base_tokens)) + 1))[:prefill]
        one_input = torch.tensor([token_ids], device=device, dtype=torch.long)
        for batch in BATCHES:
            inputs = one_input.repeat(batch, 1)
            print(f"\n=== Prefill={prefill}, Batch={batch} ===", flush=True)
            results = {}
            errors = []
            for label, backend in (("sdpa", "sdpa"), ("cutile", "cutile_matrix")):
                try:
                    counters.clear()
                    fallback_reasons.clear()
                    tile_usage.clear()
                    for _ in range(WARMUPS):
                        greedy_run(inputs, backend)
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats(device)
                    timings = [greedy_run(inputs, backend) for _ in range(REPEATS)]
                    peak_gib = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
                    captured = greedy_run(inputs, backend, capture=True)
                    r = {
                        "prefill_ms": statistics.median(x["prefill_ms"] for x in timings),
                        "decode_ms": statistics.median(x["decode_ms"] for x in timings),
                        "total_ms": statistics.median(x["total_ms"] for x in timings),
                        "peak_gib": peak_gib,
                        "tokens": captured["tokens"],
                        "logits": captured["logits"],
                        "cutile_calls": counters["cutile"],
                        "fallback_calls": counters["fallback"],
                        "tile_usage": dict(tile_usage),
                        "fallback_reasons": dict(fallback_reasons),
                    }
                    results[label] = r
                    print(
                        f"{label}: prefill={r['prefill_ms']:.2f} ms, "
                        f"decode={r['decode_ms']:.2f} ms, "
                        f"total={r['total_ms']:.2f} ms, "
                        f"peak_allocated={peak_gib:.2f} GiB",
                        flush=True,
                    )
                    if label == "cutile":
                        print(
                            f"  Custom calls={r['cutile_calls']}, "
                            f"fallbacks={r['fallback_calls']}, "
                            f"tile usage={r['tile_usage']}",
                            flush=True,
                        )
                except Exception as exc:
                    msg = f"{label}: {type(exc).__name__}: {exc}"
                    errors.append(msg)
                    print("ERROR:", msg, flush=True)
                    if isinstance(exc, torch.cuda.OutOfMemoryError):
                        torch.cuda.empty_cache()
                    break

            row = {"batch": batch, "prefill_tokens": prefill,
                   "new_tokens": NEW_TOKENS}
            if errors or "sdpa" not in results or "cutile" not in results:
                row.update(status="ERROR", error=" | ".join(errors))
                rows.append(row)
                torch.cuda.empty_cache()
                continue

            a, b = results["sdpa"], results["cutile"]
            same = torch.equal(a["tokens"], b["tokens"])
            diff = (a["logits"] - b["logits"]).abs()
            max_err, mean_err = diff.max().item(), diff.mean().item()
            tol_ok = torch.allclose(a["logits"], b["logits"], rtol=0.02, atol=0.02)
            speedup = a["total_ms"] / b["total_ms"]
            status = "TOKEN_MATCH" if same else "TOKEN_MISMATCH"
            if b["fallback_calls"]:
                status += "+FALLBACK"
            row.update({
                "status": status,
                "token_match": same,
                "logit_tolerance_passed": tol_ok,
                "prefill_logit_max_error": round(max_err, 8),
                "prefill_logit_mean_error": round(mean_err, 8),
                "sdpa_prefill_ms": round(a["prefill_ms"], 3),
                "cutile_prefill_ms": round(b["prefill_ms"], 3),
                "sdpa_decode_ms": round(a["decode_ms"], 3),
                "cutile_decode_ms": round(b["decode_ms"], 3),
                "sdpa_total_ms": round(a["total_ms"], 3),
                "cutile_total_ms": round(b["total_ms"], 3),
                "sdpa_tokens_per_s": round(batch * NEW_TOKENS / (a["total_ms"] / 1000), 3),
                "cutile_tokens_per_s": round(batch * NEW_TOKENS / (b["total_ms"] / 1000), 3),
                "speedup_vs_sdpa": round(speedup, 3),
                "sdpa_peak_allocated_gib": round(a["peak_gib"], 3),
                "cutile_peak_allocated_gib": round(b["peak_gib"], 3),
                "cutile_calls_all_runs": b["cutile_calls"],
                "sdpa_fallbacks_all_runs": b["fallback_calls"],
                "tile_usage": str(b["tile_usage"]),
                "fallback_reasons": str(b["fallback_reasons"]),
                "error": "",
            })
            rows.append(row)
            print(
                f"Token IDs match: {same}; logit tolerance passed: {tol_ok}; "
                f"logit max error: {max_err:.6f}; speedup: {speedup:.3f}x",
                flush=True,
            )
            torch.cuda.empty_cache()

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=keys)
    writer.writeheader()
    writer.writerows(rows)
    print("\n=== Final matrix summary ===", flush=True)
    print(f"Completed rows: {len(rows)} / {len(BATCHES) * len(PREFILL_LENGTHS)}", flush=True)
    for row in rows:
        print(
            f"S={row['prefill_tokens']}, B={row['batch']}: {row['status']}; "
            f"speedup={row.get('speedup_vs_sdpa', 'N/A')}",
            flush=True,
        )
    return output.getvalue()


@app.local_entrypoint()
def main():
    csv_data = run_benchmark.remote()
    with open(CSV_NAME, "w", encoding="utf-8", newline="") as f:
        f.write(csv_data)
    print(f"Saved: {CSV_NAME}", flush=True)
