# Qwen3 Inference Optimization: KV Cache, PagedAttention, and Continuous Batching

## Project Overview

This project focuses on improving Qwen3 inference by implementing and evaluating three important techniques:

1. **KV Cache** – Reuses previously computed keys and values during autoregressive generation.
2. **Paged KV Cache** – Manages KV memory using fixed-size physical blocks instead of requiring one contiguous allocation per sequence.
3. **Continuous Batching** – Dynamically schedules incoming requests so that available GPU batch capacity can be used more efficiently.

The project also includes a custom OpenAI-compatible inference server and performance experiments using NVIDIA AIPerf.

The implementations were developed and tested using Qwen3 models on Modal GPU instances.

## Project Structure

```text
week07/
├── README.md
├── qwen3_inference_optimisation.md
│
├── kv_cache.py
├── paged_kv_cache.py
├── paged_hf_cache.py
├── paged_attention.py
│
├── benchmark_kv_cache.py
├── benchmark_paged_kv_cache.py
├── test_custom_kv_cache.py
├── test_paged_kv_cache.py
├── test_paged_qwen3_cache.py
│
├── run_paged_qwen3_modal.py
├── run_continuous_batch_modal.py
├── run_batched_qwen3_modal.py
├── run_vectorized_continuous_batch_modal.py
├── run_vectorized_paged_qwen3_modal.py
├── run_openai_server_modal.py
├── run_vectorized_continuous_batch_benchmark_modal.py
│
├── inference_server/
│   ├── openai_server.py
│   ├── request_state.py
│   ├── vectorized_continuous_batch_engine.py
│   ├── vectorized_paged_cache.py
│   ├── benchmark_vectorized_continuous_batch.py
│   ├── test_vectorized_continuous_batch.py
│   └── test_vectorized_paged_qwen3.py
│
├── results/
├── artifacts/
└── logs/
```

## 1. KV Cache Implementation

During autoregressive generation, a transformer generates one token at a time. Without caching, the model would repeatedly calculate keys and values for all previous tokens.

The KV cache stores these previously computed tensors and reuses them during subsequent decoding steps.

### Implementation

The standard KV cache implementation uses preallocated tensor storage to maintain key and value states across decoding steps.

Main implementation:

- `kv_cache.py`
- `paged_hf_cache.py`

The implementation was compared with Hugging Face's `DynamicCache` using Qwen3-0.6B on an NVIDIA L40S GPU.

### Correctness

The custom cache and Hugging Face DynamicCache produced:

- Maximum logit difference: `0`
- Matching top-1 predictions: `32/32` decoding steps

### Performance

| Metric | DynamicCache | Custom KV Cache |
|---|---:|---:|
| Prefill time | 11.81 ms | 12.31 ms |
| Decode time | 2904.57 ms | 2946.37 ms |
| Total time | 2916.39 ms | 2958.68 ms |
| Throughput | 87.81 tokens/sec | 86.62 tokens/sec |

The custom implementation was slightly slower in this single-request experiment. Its main contribution is demonstrating explicit preallocated cache management rather than providing a measured speedup over DynamicCache.

## 2. Paged KV Cache

A standard KV cache typically allocates memory for each sequence in a contiguous layout. This can lead to memory fragmentation and inefficient memory utilization when requests have different sequence lengths.

Paged KV caching addresses this by dividing the KV memory into fixed-size blocks.

### Implementation

The paged cache implementation includes:

- Fixed-size physical KV blocks
- Per-sequence block tables
- Free-block management
- Cache append and retrieval operations
- Cache release and reuse

Main files:

- `paged_kv_cache.py`
- `paged_hf_cache.py`
- `paged_attention.py`
- `inference_server/vectorized_paged_cache.py`

The vectorized implementation also supports variable-length append operations by gathering and padding the required KV data.

### Correctness

The paged cache was compared against Hugging Face DynamicCache.

Recorded results:

- Maximum logit difference: `0`
- Matching top-1 predictions: `33/33` decoding steps

### Performance

| Metric | DynamicCache | Paged KV Cache |
|---|---:|---:|
| Prefill time | 28.19 ms | 30.92 ms |
| Decode time | 3376.83 ms | 3997.40 ms |
| Total time | 3405.02 ms | 4028.31 ms |
| Throughput | 37.60 tokens/sec | 31.78 tokens/sec |

The paged cache was approximately 15.5% slower in this single-request experiment.

This implementation focuses on memory organization and block management. It gathers the required physical blocks into contiguous K/V tensors before attention computation; it should not be confused with a fused PagedAttention kernel.

## 3. Continuous Batching

Traditional static batching processes a fixed group of requests together. A request that finishes early may leave unused batch capacity while other requests continue decoding.

Continuous batching dynamically admits new requests as active requests finish.

### Implementation

The vectorized continuous batching engine supports:

- Dynamic request admission
- Variable-length prefill batches
- Batched decode operations
- Per-request sequence state
- KV cache allocation and release
- Active batch capacity management

Main files:

- `inference_server/vectorized_continuous_batch_engine.py`
- `inference_server/request_state.py`
- `inference_server/vectorized_paged_cache.py`

### Correctness Testing

The continuous batching test submitted three requests with a maximum active batch capacity of two.

The scheduler:

1. Started two requests.
2. Admitted the third request after one active request finished.
3. Processed the remaining requests together during decoding.
4. Completed all requests.
5. Released all allocated cache blocks.

The test completed successfully.

### Development Benchmark

The following measurements were collected using Qwen3-0.6B on an NVIDIA L40S GPU, with 32 generated tokens per request and a maximum active batch capacity of four.

| Requests | Throughput (tokens/sec) | Average TTFT (ms) | Average Latency (ms) | Maximum Active Batch |
|---:|---:|---:|---:|---:|
| 1 | 63.37 | 16.31 | 504.95 | 1 |
| 2 | 111.02 | 18.16 | 576.48 | 2 |
| 4 | 180.51 | 24.62 | 709.17 | 4 |
| 8 | 177.44 | 382.63 | 1080.65 | 4 |

Throughput increased as the number of concurrent requests grew from one to four.

At eight requests, throughput remained approximately flat because the engine was configured for a maximum of four active requests. Additional requests waited for available capacity, which increased TTFT.

These results demonstrate the benefits of batching while also showing how queueing and active-batch limits affect latency.

## 4. OpenAI-Compatible Inference Server

A custom inference server was developed to expose the Qwen3 model through an OpenAI-compatible API.

Main implementation:

- `inference_server/openai_server.py`
- `inference_server/request_state.py`
- `inference_server/vectorized_continuous_batch_engine.py`

The server uses FastAPI and supports:

| Endpoint | Purpose |
|---|---|
| `/health` | Health check |
| `/metrics` | Server metrics |
| `/v1/chat/completions` | Chat completion requests |

The server supports streaming and non-streaming responses.

A background engine worker manages incoming requests and coordinates prefill, decoding, and request scheduling.

## 5. AIPerf Benchmarking

NVIDIA AIPerf was used to evaluate the custom Qwen3-4B inference server under different concurrency levels.

The benchmark used:

- Model: `Qwen/Qwen3-4B`
- Input length: 128 tokens
- Output length: 128 tokens
- Requests: 50 per run
- Streaming: Enabled
- Concurrency: 1, 2, and 4

### Results

| Metric | Concurrency 1 | Concurrency 2 | Concurrency 4 |
|---|---:|---:|---:|
| Effective concurrency | 1.00 | 1.98 | 3.93 |
| Average TTFT (ms) | 104.84 | 1243.32 | 2402.30 |
| Average request latency (ms) | 4836.80 | 6578.00 | 8601.29 |
| Average ITL (ms) | 37.25 | 42.01 | 48.81 |
| Output throughput (tokens/sec) | 26.41 | 38.57 | 58.40 |
| Output throughput per user (tokens/sec) | 27.21 | 25.51 | 22.33 |
| Request throughput (requests/sec) | 0.21 | 0.30 | 0.46 |
| Completed requests | 50 | 50 | 50 |
| Error rate | 0% | 0% | 0% |

### Observations

- All 50 requests completed successfully at each concurrency level.
- No request errors were recorded.
- Aggregate output throughput increased from 26.41 tokens/sec at concurrency one to 58.40 tokens/sec at concurrency four.
- The highest aggregate output throughput was observed at concurrency four.
- Per-user throughput decreased as concurrency increased.
- Average TTFT and request latency increased under higher concurrency, indicating additional queueing and scheduling overhead.

The concurrency-four run achieved approximately 2.21 times the aggregate output throughput of the concurrency-one run.

**Hardware note:** The AIPerf runs were executed through Modal, but GPU telemetry was not captured for these runs. Therefore, the hardware used for the AIPerf measurements is not asserted here.

## 6. How to Run

Run commands from the `week07` directory.

### Environment

The experiments use Modal for GPU execution. Make sure Modal is installed and authenticated locally.

```bash
pip install modal
modal setup
```

### Run KV Cache Tests

```bash
python test_custom_kv_cache.py
```

### Run Paged KV Cache Tests

```bash
python test_paged_kv_cache.py
```

### Run Paged Qwen3 Validation

```bash
modal run run_paged_qwen3_modal.py
```

### Run Continuous Batching Tests

```bash
modal run run_vectorized_continuous_batch_modal.py
```

### Run Vectorized Paged Cache Validation

```bash
modal run run_vectorized_paged_qwen3_modal.py
```

### Run Continuous Batching Benchmark

```bash
modal run run_vectorized_continuous_batch_benchmark_modal.py
```

### Run the OpenAI-Compatible Server

```bash
modal run run_openai_server_modal.py
```

The exact command-line entrypoints should match the Modal functions defined in each script.

## 7. Results and Artifacts

The `results/`, `artifacts/`, and `logs/` directories contain saved benchmark outputs and execution logs.

The AIPerf benchmark artifacts include:

- `profile_export.jsonl`
- `profile_export_console.txt`
- `profile_export_aiperf.json`
- `profile_export_aiperf.csv`
- `inputs.json`
- `logs/aiperf.log`

These files preserve the measurements used in the performance analysis.

Generated artifacts and large files should be reviewed before committing them to Git. Avoid committing virtual environments, temporary Python files, downloaded model checkpoints, or unrelated debugging outputs.

## 8. Key Takeaways

This project demonstrates how different inference optimizations contribute to transformer serving:

- **KV caching** avoids recalculating historical key and value states during autoregressive decoding.
- **Paged KV caching** introduces block-based memory management and supports more flexible allocation across sequences.
- **Continuous batching** dynamically schedules requests to improve GPU utilization and aggregate throughput.
- **Inference serving** connects the optimization work to an OpenAI-compatible API.
- **AIPerf benchmarking** provides a way to evaluate throughput, latency, and behavior under different concurrency levels.

The experiments also highlight an important point: an optimization that improves memory management or batching flexibility does not automatically improve single-request latency. Performance depends on the workload, batch size, sequence length, GPU, and implementation details.
