"""Controlled batched Qwen3-4B greedy inference benchmark on Modal L40S.

Compares PyTorch SDPA and the existing adaptive single-tile CUDA Tile kernel.
The custom backend accepts only unpadded causal prefill or one-token decoding.
"""
import csv
import io
import statistics
import time
from collections import Counter

import modal

app = modal.App("qwen3-4b-batched-attention-benchmark")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch", "numpy", "transformers", "accelerate", "cuda-tile[tileiras]")
    .add_local_file("attention_kernel.py", "/root/attention_kernel.py")
)

MODEL_ID = "Qwen/Qwen3-4B"
BATCHES = (1, 2, 4, 8)
PREFILL_TOKENS = 128
NEW_TOKENS = 16
WARMUPS = 1
REPEATS = 3
MAX_KV = 1024


@app.function(image=image, gpu="L40S", timeout=3600)
def run_benchmark():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, AttentionInterface, AttentionMaskInterface
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    from transformers.masking_utils import sdpa_mask
    from attention_kernel import cutile_attention

    torch.manual_seed(1234)
    device = "cuda"
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    print(f"Model: {MODEL_ID}; prefill={PREFILL_TOKENS}; decode={NEW_TOKENS}", flush=True)
    print("No padding; same prompt tokens for every row; greedy decoding.", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(device).eval()

    # Construct exactly 128 tokens without padding; repeat a meaningful phrase.
    phrase = ("Explain grouped-query attention and how it reduces the "
              "key-value cache memory footprint in transformer inference. ")
    base = tokenizer.encode(phrase, add_special_tokens=False)
    if not base:
        raise RuntimeError("Tokenizer returned no prompt tokens")
    token_ids = (base * ((PREFILL_TOKENS // len(base)) + 1))[:PREFILL_TOKENS]
    base_input = torch.tensor([token_ids], device=device, dtype=torch.long)

    counters = Counter()
    fallback_reasons = Counter()
    tile_usage = Counter()
    # Diagnostics are enabled only for the untimed B=1 custom prefill.
    diagnostic = {"enabled": False, "rows": []}

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
            and 0 < sq <= sk <= MAX_KV
            and dropout == 0.0 and sliding is None
            and standard_causal_mask(attention_mask, sq, sk)
        )
        if supported:
            tile = max(16, 1 << (sk - 1).bit_length())
            output = cutile_attention(query, key, value, tile_n=tile)
            if diagnostic["enabled"] and sq == sk == PREFILL_TOKENS:
                # Same Q/K/V for both paths: isolates attention-kernel error
                # from differences already accumulated in prior layers.
                import torch.nn.functional as F
                reference = F.scaled_dot_product_attention(
                    query, key, value, is_causal=True,
                    dropout_p=0.0, enable_gqa=True,
                )
                difference = (output.float() - reference.float()).abs()
                diagnostic["rows"].append((
                    len(diagnostic["rows"]),
                    difference.max().item(),
                    difference.mean().item(),
                ))
            counters["cutile"] += 1
            tile_usage[tile] += 1
            return output.transpose(1, 2).contiguous(), None
        counters["fallback"] += 1
        reason = ("kv>1024" if sk > MAX_KV else
                  "nonstandard mask" if not standard_causal_mask(attention_mask, sq, sk) else
                  "other unsupported attention")
        fallback_reasons[reason] += 1
        return sdpa_attention_forward(module, query, key, value, attention_mask, **kwargs)

    AttentionInterface.register("cutile_batched", custom_attention)
    AttentionMaskInterface.register("cutile_batched", sdpa_mask)

    @torch.inference_mode()
    def greedy_run(input_ids, backend, capture_tokens=False):
        """Fresh KV cache each run; exactly NEW_TOKENS greedy steps."""
        model.set_attn_implementation(backend)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        diagnostic["enabled"] = bool(capture_tokens and backend == "cutile_batched" and input_ids.shape[0] == 1)
        outputs = model(input_ids=input_ids, use_cache=True, return_dict=True)
        diagnostic["enabled"] = False
        next_token = outputs.logits[:, -1:, :].argmax(dim=-1)
        prefill_logits = outputs.logits[:, -1, :].float() if capture_tokens else None
        cache = outputs.past_key_values
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        generated = [next_token] if capture_tokens else None
        # First generated token comes from prefill. Process 15 cached steps.
        for _ in range(NEW_TOKENS - 1):
            outputs = model(
                input_ids=next_token,
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
            cache = outputs.past_key_values
            next_token = outputs.logits[:, -1:, :].argmax(dim=-1)
            if capture_tokens:
                generated.append(next_token)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        ids = torch.cat(generated, dim=1).cpu() if capture_tokens else None
        return {
            "prefill_ms": (t1 - t0) * 1000,
            "decode_ms": (t2 - t1) * 1000,
            "total_ms": (t2 - t0) * 1000,
            "tokens": ids,
            "logits": prefill_logits,
        }

    rows = []
    for batch in BATCHES:
        inputs = base_input.repeat(batch, 1)
        print(f"\n=== Batch {batch} ===", flush=True)
        results = {}
        failed = False
        for label, backend in (("sdpa", "sdpa"), ("cutile", "cutile_batched")):
            try:
                counters.clear()
                fallback_reasons.clear()
                tile_usage.clear()
                # Warm up backend and compile CUDA Tile kernels before measuring.
                for _ in range(WARMUPS):
                    greedy_run(inputs, backend)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats(device)
                timings = []
                for _ in range(REPEATS):
                    timings.append(greedy_run(inputs, backend))
                peak_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
                # Capture one untimed output to compare token IDs and prefill logits.
                diagnostic["rows"].clear()
                captured = greedy_run(inputs, backend, capture_tokens=True)
                if label == "cutile" and batch == 1:
                    print("\n=== Attention diagnostic (B=1, same Q/K/V, untimed) ===", flush=True)
                    for layer, max_error, mean_error in diagnostic["rows"]:
                        print(f"Layer {layer:02d}: max_error={max_error:.8f}, mean_error={mean_error:.8f}", flush=True)
                    if diagnostic["rows"]:
                        worst = max(diagnostic["rows"], key=lambda x: x[1])
                        print(f"Worst attention layer={worst[0]}, max_error={worst[1]:.8f}", flush=True)
                    else:
                        print("No diagnostic rows captured; inspect the custom attention path.", flush=True)
                results[label] = {
                    "prefill_ms": statistics.median(x["prefill_ms"] for x in timings),
                    "decode_ms": statistics.median(x["decode_ms"] for x in timings),
                    "total_ms": statistics.median(x["total_ms"] for x in timings),
                    "peak_gb": peak_gb,
                    "tokens": captured["tokens"],
                    "logits": captured["logits"],
                    "cutile_calls": counters["cutile"],
                    "fallback_calls": counters["fallback"],
                    "fallback_reasons": dict(fallback_reasons),
                    "tile_usage": dict(tile_usage),
                }
                r = results[label]
                print(f"{label}: prefill={r['prefill_ms']:.2f} ms, "
                      f"decode={r['decode_ms']:.2f} ms, total={r['total_ms']:.2f} ms, "
                      f"peak_allocated={peak_gb:.2f} GiB", flush=True)
                if label == "cutile":
                    print(f"  Custom calls={r['cutile_calls']}, "
                          f"fallbacks={r['fallback_calls']}, "
                          f"tile usage={r['tile_usage']}", flush=True)
                    if r["fallback_reasons"]:
                        print("  Fallback reasons:", r["fallback_reasons"], flush=True)
            except Exception as exc:
                failed = True
                print(f"{label} FAILED for batch={batch}: {type(exc).__name__}: {exc}", flush=True)
                if isinstance(exc, torch.cuda.OutOfMemoryError):
                    torch.cuda.empty_cache()
                break
        if failed:
            rows.append({"batch": batch, "status": "ERROR"})
            continue
        a, b = results["sdpa"], results["cutile"]
        same = torch.equal(a["tokens"], b["tokens"])
        diff = (a["logits"] - b["logits"]).abs()
        max_err, mean_err = diff.max().item(), diff.mean().item()
        logit_tolerance_passed = torch.allclose(
            a["logits"], b["logits"], rtol=0.02, atol=0.02
        )
        throughput_a = batch * NEW_TOKENS / (a["total_ms"] / 1000)
        throughput_b = batch * NEW_TOKENS / (b["total_ms"] / 1000)
        speedup = a["total_ms"] / b["total_ms"]
        status = "TOKEN_MATCH" if same else "TOKEN_MISMATCH"
        if b["fallback_calls"]:
            status += "+FALLBACK"
        print(f"Token IDs match: {same}; prefill logits max_err={max_err:.6f}, "
              f"mean_err={mean_err:.6f}; logit tolerance passed={logit_tolerance_passed}; "
              f"speedup={speedup:.3f}x", flush=True)
        rows.append({
            "batch": batch, "prefill_tokens": PREFILL_TOKENS,
            "new_tokens": NEW_TOKENS, "status": status,
            "token_match": same,
            "logit_tolerance_passed": logit_tolerance_passed,
            "prefill_logit_max_error": round(max_err, 8),
            "prefill_logit_mean_error": round(mean_err, 8),
            "sdpa_prefill_ms": round(a["prefill_ms"], 3),
            "cutile_prefill_ms": round(b["prefill_ms"], 3),
            "sdpa_decode_ms": round(a["decode_ms"], 3),
            "cutile_decode_ms": round(b["decode_ms"], 3),
            "sdpa_total_ms": round(a["total_ms"], 3),
            "cutile_total_ms": round(b["total_ms"], 3),
            "sdpa_tokens_per_s": round(throughput_a, 3),
            "cutile_tokens_per_s": round(throughput_b, 3),
            "speedup_vs_sdpa": round(speedup, 3),
            "sdpa_peak_allocated_gib": round(a["peak_gb"], 3),
            "cutile_peak_allocated_gib": round(b["peak_gb"], 3),
            "cutile_calls_all_runs": b["cutile_calls"],
            "sdpa_fallbacks_all_runs": b["fallback_calls"],
        })
        del results
        torch.cuda.empty_cache()

    keys = [
        "batch", "prefill_tokens", "new_tokens", "status", "token_match",
        "logit_tolerance_passed",
        "prefill_logit_max_error", "prefill_logit_mean_error",
        "sdpa_prefill_ms", "cutile_prefill_ms", "sdpa_decode_ms", "cutile_decode_ms",
        "sdpa_total_ms", "cutile_total_ms", "sdpa_tokens_per_s",
        "cutile_tokens_per_s", "speedup_vs_sdpa",
        "sdpa_peak_allocated_gib", "cutile_peak_allocated_gib",
        "cutile_calls_all_runs", "sdpa_fallbacks_all_runs",
    ]
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=keys)
    writer.writeheader()
    writer.writerows(rows)
    print("\n=== Summary ===", flush=True)
    for row in rows:
        print(f"B={row['batch']}: {row['status']}", flush=True)
    return output.getvalue()


@app.local_entrypoint()
def main():
    csv_data = run_benchmark.remote()
    destination = "batched_qwen3_diagnostic_results.csv"
    with open(destination, "w", encoding="utf-8", newline="") as f:
        f.write(csv_data)
    print(f"Saved: {destination}")
