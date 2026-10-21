# Batched Qwen3-4B Inference with CUDA Tile Attention

## Project overview

This project implements a custom attention kernel using **CUDA Tile**, integrates it into **Qwen3-4B** inference, and compares its behavior and performance with **PyTorch scaled dot-product attention (SDPA)** on an **NVIDIA L40S** GPU through Modal.

The custom attention implementation supports grouped-query attention (GQA), causal masking, batched inputs, and cached single-token decoding. It uses a single-pass, online-softmax attention algorithm with FP32 accumulation and output conversion to the input dtype where appropriate.

> **Validation caveat:** Kernel-level numerical comparisons and generated token agreement are encouraging, but full-model logit comparisons **did not pass** the configured `torch.allclose(rtol=0.02, atol=0.02)` threshold. This project should not be described as fully numerically equivalent to SDPA.

## Project structure

| Path | Purpose |
| --- | --- |
| `attention_kernel.py` | Active CUDA Tile attention implementation |
| `attention_tests.sqlite3` | SQLite database of reference attention test cases |
| `create_attention_db.py` | Generate the initial reference database |
| `expand_attention_db.py` | Expand SQLite test coverage |
| `validate_extended_attention.py` | Validate attention against reference test cases |
| `test_online_attention.py` | Additional online-attention tests |
| `run_qwen3_baseline.py` | Baseline Qwen3-4B execution |
| `run_qwen3_cutile.py` | Qwen3-4B integration and numerical diagnostics |
| `benchmark_attention.py` | Attention microbenchmark |
| `benchmark_batched_qwen3_matrix.py` | Full 16-configuration batched inference benchmark |
| `inspect_qwen3.py` | Model inspection utility |
| `batched_qwen3_matrix_results.csv` | Final 16-configuration benchmark measurements |
| `attention_benchmark_adaptive.csv` | Attention microbenchmark measurements |
| `final_results/` | Submission-ready copies of results |
| `logs/` | Saved Modal run logs |
| `archive/` | Historical/experimental scripts, if archived |

The SQLite database is about 71 MB; retain it if your submission requires reproducible reference test cases.

## Environment

- Execution platform: Modal
- GPU: NVIDIA L40S
- Model: `Qwen/Qwen3-4B`
- Main libraries: PyTorch, Transformers, CUDA Tile (`cuda-tile[tileiras]`), NumPy, and Modal
- Model weights are retrieved from Hugging Face at runtime; they are not bundled with the project.

Run commands from the directory containing `attention_kernel.py` and the benchmark scripts. CUDA Tile is executed inside the Modal GPU environment; it need not be importable in the local WSL environment.

## How to run

### 1. Check local Python syntax

```bash
python -m py_compile attention_kernel.py run_qwen3_cutile.py benchmark_batched_qwen3_matrix.py
```

### 2. Validate attention

```bash
modal run validate_extended_attention.py
```

Use the relevant Modal entrypoints in your local scripts if they differ from the command above. The previously recorded extended SQLite validation completed **16/16 cases successfully**.

### 3. Run Qwen3-4B integration diagnostics

```bash
modal run run_qwen3_cutile.py 2>&1 | tee logs/qwen3_validation.log
```

### 4. Run the benchmark matrix

```bash
mkdir -p logs final_results
modal run benchmark_batched_qwen3_matrix.py 2>&1 | tee logs/batched_matrix_final.log
cp batched_qwen3_matrix_results.csv final_results/
```

The benchmark compares SDPA and CUDA Tile using four batch sizes (`1, 2, 4, 8`), four prefill lengths (`64, 128, 256, 512` tokens), and **16 generated tokens** per run. The reported total latency covers prefill and decode as implemented in the benchmark. Speedup is `SDPA total latency / CUDA Tile total latency`; values greater than 1 indicate CUDA Tile is faster.

## Correctness results

| Test | Observed result |
| --- | --- |
| Extended SQLite reference validation | 16/16 passed |
| FP32 attention maximum absolute error (Qwen3 diagnostic) | Approximately `1.335e-05` |
| BF16 attention maximum absolute error (Qwen3 diagnostic) | `0.125` |
| 16-configuration benchmark completion | 16/16 |
| Generated token ID agreement with SDPA | 16/16 configurations |
| SDPA fallbacks in custom benchmark | 0 |
| Full-model logit `allclose(rtol=0.02, atol=0.02)` | Failed in all 16 benchmark configurations |

Matching generated token IDs does **not** imply that all logits or intermediate states match within tolerance. Full-model numerical differences remain a limitation requiring further investigation.

## Performance results

**End-to-end speedup versus SDPA** (greater than `1.0×` favors CUDA Tile):

| Prefill tokens | Batch 1 | Batch 2 | Batch 4 | Batch 8 |
| --- | ---: | ---: | ---: | ---: |
| 64 | 1.033× | 0.977× | 1.036× | 0.964× |
| 128 | 0.963× | 1.060× | 0.969× | 0.922× |
| 256 | 0.941× | 0.831× | 0.721× | 0.607× |
| 512 | 0.499× | 0.369× | 0.238× | 0.147× |

The best measured end-to-end speedup was **1.060×** (prefill 128, batch 2). At prefill 512 and batch 8, SDPA completed in about **725.02 ms**, whereas CUDA Tile took about **4932.69 ms**. CUDA Tile prefill time grows substantially with larger sequence lengths and batches in this implementation.

The benchmark's adaptive tile selection uses a power-of-two tile large enough for the KV sequence, which may contribute to the poor long-sequence performance. Smaller-tile online-softmax processing is a possible optimization to investigate, **not a verified performance improvement**.

## Limitations and future work

1. Investigate BF16 numerical differences and accumulated full-model logit discrepancies; do not loosen tolerance solely to report a pass.
2. Explore multi-tile attention and alternative tile sizes for long sequences.
3. Repeat performance measurements across GPU runs to characterize variability.
4. Add broader correctness cases, including varied prompts and cache lengths.
5. Record dependency versions and exact Modal image configuration for reproducibility.


