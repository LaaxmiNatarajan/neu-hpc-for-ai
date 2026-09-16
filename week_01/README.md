# Matrix Multiplication: Single-Threaded vs. Multi-Threaded (Pthreads)

A simple, optimized C project that compares standard single-threaded matrix multiplication against a multi-threaded implementation using POSIX threads (`pthreads`). It runs tests across different matrix shapes and measures speedup up to 128 threads in the cloud using [Modal](https://modal.com).


## Project Structure

```text
├── include/
│   └── matrix.h          # Function definitions and shared structs
├── src/
│   ├── matrix_single.c   # Single-threaded implementation & verification
│   └── matrix_multi.c    # Multi-threaded pthread implementation
├── tests/
│   └── test_matrix.c     # Test runner for edge and corner cases
├── bench/
│   └── bench_matrix.c    # Benchmark measuring speedup across thread counts
├── Makefile              # Build commands
├── run_modal.py          # Script to run everything on Modal cloud
└── README.md

Test Cases CoveredThe test suite in tests/test_matrix.c runs each test case against 1, 2, 4, 8, and 16 threads to make sure multi-threading produces the exact same results as single-threading:

Scalar (1 × 1 × 1): Single number multiplication; checks boundary loop handling.
Dot Product (1 × 50 × 1): Vector dot product reducing across dimension $K$ to a single number.
Outer Product (50 × 1 × 50): Multiplies a column vector by a row vector to create a 2D matrix.
Row Vector × Matrix (1 × 64 × 64): Single-row matrix multiplied by a square matrix.
Matrix × Column Vector (64 × 64 × 1): Square matrix multiplied by a single-column matrix.
Prime Dimensions (13 × 29 × 37): Odd shapes that do not divide evenly across threads, testing remainder row handling.
Symmetric Square (64 × 64 × 64): Standard power-of-two matrix baseline.
Tall-and-Skinny (7 × 128 × 11): Fewer rows ($M = 7$) than available threads, testing that thread clamping works without crashing.