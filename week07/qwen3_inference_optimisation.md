# Qwen3 Inference Optimization: KV Cache, Paged KV Cache, and Continuous Batching

**Course:** High Performance Computing for AI  
**Model:** Qwen3-4B for server benchmarking; Qwen3-0.6B for selected development tests  
**Benchmarking tool:** NVIDIA AIPerf

## Abstract

This project explores ways to make large language model inference more efficient by implementing and evaluating three important techniques: KV caching, paged KV caching, and continuous batching. Each technique addresses a different challenge in inference, including repeated computation, memory management, and serving multiple requests efficiently.

I implemented a standard KV cache, a paged KV cache, and a vectorized continuous batching engine for Qwen3. I validated the cache implementations against Hugging Face reference behavior and tested the batching engine with multiple concurrent requests.

For performance evaluation, I used NVIDIA AIPerf to benchmark the custom Qwen3-4B inference server at concurrency levels of 1, 2, and 4. The results showed that aggregate output throughput increased from 26.41 tokens/sec at concurrency 1 to 58.40 tokens/sec at concurrency 4. At the same time, individual request latency increased, highlighting the trade-off between throughput and responsiveness.

## 1. Introduction

Large language models generate text one token at a time. Although this makes generation flexible, it also creates a significant computational challenge. During every generation step, the model needs information from the tokens that came before the current token.

Without optimization, the model may repeatedly calculate attention information for previously processed tokens. This wastes computation and increases inference latency.

I focused on three techniques to address these challenges:

1. **KV Cache:** Stores previously calculated keys and values so they can be reused during generation.
2. **Paged KV Cache:** Organizes cached keys and values into smaller blocks, allowing memory to be allocated and released more flexibly.
3. **Continuous Batching:** Allows new requests to enter an active inference batch as other requests finish, rather than waiting for the entire batch to complete.

The goal was not just to implement these techniques, but also to understand how they affect correctness, memory usage, throughput, and request latency.

## 2. Background

### 2.1 Transformer attention

A transformer uses an attention mechanism to determine which previously processed tokens are important when generating the next token.

The attention calculation uses three main representations:

- **Query (Q):** Represents what the current token is looking for.
- **Key (K):** Represents information associated with each token.
- **Value (V):** Contains the information that attention combines to produce its output.

During autoregressive generation, the model processes one new token at a time. The keys and values of earlier tokens are still needed for subsequent attention calculations.

### 2.2 KV caching

A KV cache stores the keys and values computed during earlier steps.

Instead of recalculating them for every generated token, the model reuses the stored tensors and calculates keys and values only for the new token.

This reduces repeated computation and makes token-by-token generation more practical.

### 2.3 Paged KV caching

A conventional cache generally reserves contiguous memory for a sequence. This can become inefficient when many requests have different sequence lengths.

Paged KV caching divides the cache into fixed-size blocks. Each request maintains a block table that maps its logical token positions to physical memory blocks.

This makes it easier to allocate memory as requests grow and release blocks when requests finish.

### 2.4 Continuous batching

In static batching, a batch is typically formed before execution. If one request finishes earlier than the others, its position may remain unused until the batch completes.

Continuous batching handles this differently. When a request finishes, another waiting request can be admitted into the active batch.

This helps keep the GPU occupied and can improve aggregate throughput when multiple users are sending requests.

## 3. Implementation

### 3.1 Standard KV Cache

I implemented a custom KV cache using preallocated tensor storage. Instead of creating new key and value tensors at every decoding step, the cache stores the tensors and updates them as new tokens are generated.

The implementation maintains the cached keys and values for each layer and tracks the current sequence length.

The main operations are:

- Allocate cache storage.
- Append newly calculated keys and values.
- Retrieve cached tensors for attention.
- Track the current cache position.

I also tested the implementation against Hugging Face's `DynamicCache` to verify that the cached values and resulting model outputs were consistent.

### 3.2 Paged KV Cache

I implemented a paged cache using fixed-size physical blocks.

The implementation maintains:

- A collection of available physical blocks.
- A block table for each sequence.
- The number of tokens stored in each sequence.
- Allocation and release operations.

When a sequence grows, additional blocks are allocated as needed. When a request finishes, its blocks are returned to the free-block pool.

I also developed a Hugging Face-compatible adapter and a batched paged-cache implementation. The vectorized cache supports variable-length append operations by padding gathered key and value tensors before attention.

One important implementation detail is that the current attention path gathers the required physical blocks into contiguous K/V tensors before performing attention. Therefore, this implementation demonstrates paged cache memory management, but it does not use a fused PagedAttention GPU kernel.

### 3.3 Vectorized Continuous Batching

I developed a vectorized continuous batching engine in `inference_server/vectorized_continuous_batch_engine.py`.

The engine manages requests through a waiting queue and an active request collection.

Its main workflow is:

1. Receive an incoming request.
2. Add the request to the waiting queue.
3. Admit requests into the active batch when capacity is available.
4. Run batched prefill for newly admitted requests.
5. Run batched decoding for active requests.
6. Detect requests that have completed generation.
7. Release their cache resources.
8. Admit waiting requests into the available slots.

The implementation uses separate batched prefill and decode passes. This allowed me to keep the scheduling logic manageable while supporting multiple active requests.

I also integrated the engine into an OpenAI-compatible FastAPI server. The server supports streaming and non-streaming chat completion requests, along with health and metrics endpoints.

## 4. Correctness Evaluation

Before evaluating performance, I tested whether the implementations produced correct results and managed memory safely.

### 4.1 Standard KV Cache Validation

I compared the custom cache with Hugging Face's `DynamicCache`.

The teacher-forced comparison showed:

- Maximum logit difference: 0.
- Top-1 prediction agreement: 32 out of 32 decoding steps.

This confirmed that the custom cache produced matching logits in the tested configuration.

### 4.2 Paged KV Cache Validation

I compared the paged cache against Hugging Face's `DynamicCache`.

The test results showed:

- Maximum logit difference: 0.
- Top-1 prediction agreement: 33 out of 33 decoding steps.

This verified that the paged cache returned the correct historical key and value information for the tested sequence.

### 4.3 Continuous Batching Validation

I tested the continuous batching engine with three requests and a maximum active batch capacity of two.

The test demonstrated the following behavior:

- Initially, two requests were admitted while the third waited.
- When the first request completed, the waiting request was admitted.
- The remaining active requests continued decoding together.
- All requests completed successfully.
- All cache blocks were released after completion.

The maximum active batch size reached two, as expected.

These tests gave me confidence that request admission, decoding, completion handling, and cache cleanup were functioning as intended.

## 5. Experimental Setup

I used two model configurations for different parts of the evaluation.

| Evaluation | Model | Hardware used in development test |
|---|---|---|
| Standard and paged cache experiments | Qwen3-0.6B | NVIDIA L40S |
| Continuous batching development test | Qwen3-0.6B | NVIDIA L40S |
| AIPerf custom-server benchmark | Qwen3-4B | GPU telemetry unavailable |

The cache and development tests were conducted separately from the final server benchmark. Their results should therefore be interpreted within their respective test configurations.

For the Qwen3-4B server evaluation, I used NVIDIA AIPerf with streaming enabled.

The main benchmark configuration was:

| Parameter | Value |
|---|---|
| Model | Qwen/Qwen3-4B |
| Input sequence length | 128 tokens |
| Output sequence length | 128 tokens |
| Request count | 50 per run |
| Concurrency | 1, 2, and 4 |
| Endpoint type | OpenAI-compatible chat |
| Benchmark tool | NVIDIA AIPerf |

## 6. Performance Evaluation

### 6.1 Standard KV Cache Performance

I evaluated the custom preallocated cache against Hugging Face's `DynamicCache` using Qwen3-0.6B.

| Metric | DynamicCache | Custom KV Cache |
|---|---:|---:|
| Average prefill time | 11.81 ms | 12.31 ms |
| Average decode time | 2,904.57 ms | 2,946.37 ms |
| Total time | 2,916.39 ms | 2,958.68 ms |
| Throughput | 87.81 tokens/sec | 86.62 tokens/sec |

The custom cache did not provide a measurable speed improvement over Hugging Face's implementation in this particular test. Its throughput was slightly lower.

This was an important result because it showed that implementing a cache manually does not automatically make inference faster. The performance also depends on tensor operations, memory access patterns, and how efficiently the cache integrates with the model.

The main benefit of the custom implementation was gaining control over cache allocation and understanding how the cache works internally.

### 6.2 Paged KV Cache Performance

I then compared the paged cache with Hugging Face's `DynamicCache` using Qwen3-0.6B and a 128-token generation workload.

| Metric | DynamicCache | Paged KV Cache |
|---|---:|---:|
| Average prefill time | 28.19 ms | 30.92 ms |
| Average decode time | 3,376.83 ms | 3,997.40 ms |
| Total time | 3,405.02 ms | 4,028.31 ms |
| Throughput | 37.60 tokens/sec | 31.78 tokens/sec |

The paged cache was approximately 15.5% slower in this single-request experiment.

The additional overhead came from gathering the required key and value blocks into contiguous tensors before performing attention. This adds memory operations that are not present in the same way in a conventional contiguous cache.

Although the paged implementation was slower in this experiment, it provides a more flexible memory organization. Its potential benefits become more relevant when managing many requests with different sequence lengths and memory requirements.

### 6.3 Continuous Batching Development Benchmark

I evaluated the vectorized continuous batching engine with Qwen3-0.6B on an NVIDIA L40S.

The test used a maximum active batch capacity of four and generated 32 tokens per request.

| Number of requests | Throughput | Average TTFT | Average latency | Maximum active batch |
|---:|---:|---:|---:|---:|
| 1 | 63.37 tok/s | 16.31 ms | 504.95 ms | 1 |
| 2 | 111.02 tok/s | 18.16 ms | 576.48 ms | 2 |
| 4 | 180.51 tok/s | 24.62 ms | 709.17 ms | 4 |
| 8 | 177.44 tok/s | 382.63 ms | 1,080.65 ms | 4 |

Throughput increased as the number of active requests increased from one to four. This indicates that batching helped the engine process more tokens collectively.

When the number of submitted requests increased to eight, throughput remained close to the four-request result because the active batch capacity was limited to four. The additional requests had to wait for active slots to become available.

The increase in TTFT at eight requests also reflects this waiting time. This test demonstrated why batch capacity and request scheduling are important when designing an inference server.

### 6.4 Qwen3-4B AIPerf Benchmark

For the final server-level performance evaluation, I used NVIDIA AIPerf to send 50 requests at each concurrency level.

All three runs completed successfully with a 0% request error rate.

#### Overall performance results

| Metric | Concurrency 1 | Concurrency 2 | Concurrency 4 |
|---|---:|---:|---:|
| Effective concurrency | 1.00 | 1.98 | 3.93 |
| TTFT (ms) | 104.84 | 1,243.32 | 2,402.30 |
| Request latency (ms) | 4,836.80 | 6,578.00 | 8,601.29 |
| Inter-token latency (ms) | 37.25 | 42.01 | 48.81 |
| Aggregate output throughput (tokens/sec) | 26.41 | 38.57 | 58.40 |
| Output throughput per user (tokens/sec) | 27.21 | 25.51 | 22.33 |
| Request throughput (requests/sec) | 0.21 | 0.30 | 0.46 |
| Completed requests | 50 | 50 | 50 |
| Error rate | 0% | 0% | 0% |

#### Aggregate throughput

The most noticeable result was the improvement in aggregate output throughput as concurrency increased.

At concurrency one, the server achieved 26.41 tokens/sec. At concurrency two, it increased to 38.57 tokens/sec. At concurrency four, it reached 58.40 tokens/sec.

This represents approximately a 2.21× increase in aggregate output throughput between concurrency one and four.

The effective concurrency measurements also show that the server was processing multiple requests at the same time during the higher-concurrency tests.

#### Latency and responsiveness

The improvement in aggregate throughput came with an increase in latency.

Average request latency increased from approximately 4.84 seconds at concurrency one to 8.60 seconds at concurrency four.

Time to first token also increased significantly:

| Concurrency | Average TTFT |
|---:|---:|
| 1 | 104.84 ms |
| 2 | 1,243.32 ms |
| 4 | 2,402.30 ms |

This suggests that some requests experienced additional waiting or scheduling delays before generation began.

The median and 90th-percentile TTFT values help explain this behavior:

| Metric | Concurrency 1 | Concurrency 2 | Concurrency 4 |
|---|---:|---:|---:|
| TTFT p50 | 98.26 ms | 261.21 ms | 289.13 ms |
| TTFT p90 | 107.56 ms | 4,561.59 ms | 6,763.18 ms |

At higher concurrency, the 90th-percentile TTFT increased much more than the median. This means that while many requests still received their first token relatively quickly, a smaller group experienced much longer delays.

#### Per-user throughput

Another interesting observation was that output throughput per user decreased as concurrency increased.

It was 27.21 tokens/sec/user at concurrency one, 25.51 at concurrency two, and 22.33 at concurrency four.

This is understandable because multiple requests share the available GPU resources. The server generates more tokens overall, but each individual request receives a smaller share of the available processing capacity.

Taken together, the results show that concurrency four provides higher overall throughput, while concurrency one provides better responsiveness for an individual request.

## 7. Discussion

The experiments helped me understand that inference optimization is not about improving just one metric. A change that improves throughput may also increase latency or memory overhead.

The standard KV cache provided a reusable storage mechanism for previously computed attention information. However, the custom implementation was slightly slower than Hugging Face's `DynamicCache` in the measured single-request experiment.

The paged cache introduced more flexible memory allocation through fixed-size blocks. Its performance was slower in the tested configuration because the current attention path gathers blocks into contiguous tensors. This highlighted the importance of optimizing memory access in addition to memory organization.

Continuous batching showed a clearer improvement in aggregate throughput. The development benchmark demonstrated that processing several requests together increased overall token generation. The Qwen3-4B AIPerf results also confirmed that the custom server could handle concurrency levels of one, two, and four successfully.

One of the most useful lessons was understanding the difference between system throughput and individual request performance. A server can generate more tokens overall while individual users experience longer waits.

## 8. Conclusion

In this project, I implemented and evaluated three important techniques for improving large language model inference: standard KV caching, paged KV caching, and continuous batching.

The correctness tests confirmed that the custom cache implementations matched the Hugging Face reference outputs in the tested configurations. The continuous batching tests also demonstrated that requests could be admitted dynamically, processed together, completed, and removed from the cache without leaving allocated blocks behind.

The performance experiments provided several useful observations:

- The custom standard KV cache was functionally correct but did not outperform Hugging Face's `DynamicCache` in the measured test.
- The paged KV cache offered flexible block-based memory management, although gathering blocks introduced overhead in the current implementation.
- Continuous batching improved aggregate throughput as more requests were processed concurrently.
- The Qwen3-4B server achieved 58.40 output tokens/sec at concurrency four, compared with 26.41 tokens/sec at concurrency one.
- Higher concurrency increased request latency and TTFT, demonstrating the trade-off between throughput and responsiveness.

Overall, this project gave me practical experience with the internal workings of transformer inference, KV-cache memory management, request scheduling, and performance benchmarking. It also showed why correctness testing and careful interpretation of benchmark metrics are just as important as implementing an optimization.