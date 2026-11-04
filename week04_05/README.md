# Batched Qwen3-4B Inference with CUDA Tile Attention

## Project Overview

This project implements a custom attention kernel using **CUDA Tile**, integrates it into **Qwen3-4B inference**, and compares its correctness and performance against PyTorch's **Scaled Dot-Product Attention (SDPA)** on an **NVIDIA L40S GPU** through Modal.

The custom attention implementation supports:

- Grouped-Query Attention (GQA)
- Causal masking
- Batched attention inputs
- Cached single-token decoding
- Online softmax with FP32 accumulation
- Output conversion to the input data type where appropriate

The kernel uses a single-pass online-softmax attention algorithm to compute attention without explicitly materializing the full attention score matrix.

> **Important validation note:** Kernel-level numerical comparisons and generated token agreement are encouraging. However, full-model logit comparisons did not pass the configured `torch.allclose(rtol=0.02, atol=0.02)` threshold. Therefore, the implementation should not be considered fully numerically equivalent to PyTorch SDPA.

## Project Structure

| Path | Description |
|---|---|
| `attention_kernel.py` | Active CUDA Tile attention implementation |
| `attention_tests.sqlite3` | SQLite database containing reference attention test cases |
| `create_attention_db.py` | Creates the initial reference test database |
| `expand_attention_db.py` | Expands SQLite test coverage |
| `validate_extended_attention.py` | Validates the attention kernel against reference cases |
| `test_online_attention.py` | Additional online-attention tests |
| `run_qwen3_baseline.py` | Runs the baseline Qwen3-4B implementation |
| `run_qwen3_cutile.py` | Integrates CUDA Tile attention with Qwen3-4B and runs numerical diagnostics |
| `benchmark_attention.py` | Runs attention microbenchmarks |
| `benchmark_batched_qwen3_matrix.py` | Runs the 16-configuration batched inference benchmark |
| `inspect_qwen3.py` | Utility for inspecting Qwen3 model configuration |
| `batched_qwen3_matrix_results.csv` | Results from the final 16-configuration benchmark |
| `attention_benchmark_adaptive.csv` | Attention microbenchmark results |
| `final_results/` | Submission-ready copies of benchmark results |
| `logs/` | Saved Modal execution logs |
| `archive/` | Historical and experimental scripts, if archived |

**Note:** The SQLite database is approximately 71 MB. Retain it in the repository or submission package if reproducibility of the reference test cases is required.

## Environment and Requirements

| Component | Configuration |
|---|---|
| Execution platform | Modal |
| GPU | NVIDIA L40S |
| Model | `Qwen/Qwen3-4B` |
| Deep learning framework | PyTorch |
| Model library | Transformers |
| GPU kernel framework | CUDA Tile (`cuda-tile[tileiras]`) |
| Other dependencies | NumPy, Modal |

Model weights are downloaded from Hugging Face at runtime and are not included in this repository.

Run commands from the project directory containing `attention_kernel.py` and the benchmark scripts.

**Environment note:** CUDA Tile runs inside the Modal GPU environment. It does not need to be importable in the local WSL environment.

## How to Run

### 1. Check Python Syntax

Run a local syntax check before submitting jobs to Modal:

```bash
python -m py_compile \
    attention_kernel.py \
    run_qwen3_cutile.py \
    benchmark_batched_qwen3_matrix.py
```

### 2. Validate the Attention Kernel

Run the extended reference validation:

```bash
modal run validate_extended_attention.py
```

The previously recorded extended SQLite validation completed successfully with **16/16 test cases passing**.

### 3. Run Qwen3-4B Integration Diagnostics

Run the integration and numerical diagnostics:

```bash
mkdir -p logs

modal run run_qwen3_cutile.py \
    2>&1 | tee logs/qwen3_validation.log
```

### 4. Run the Batched Benchmark Matrix

Run the complete benchmark:

```bash
mkdir -p logs final_results

modal run benchmark_batched_qwen3_matrix.py \
    2>&1 | tee logs/batched_matrix_final.log
```

Copy the resulting CSV into the submission