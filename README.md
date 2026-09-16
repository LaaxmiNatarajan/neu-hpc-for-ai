# neu-hpc-for-ai
# High-Performance Computing for AI (`neu-hpc-for-ai`)

This repository tracks weekly coursework on multi-core CPU parallelism, SIMD vectorisation, cache optimisations, GPU acceleration, and distributed execution.

---

## Course Structure

```text
neu-hpc-for-ai/
├── week_01/               # CPU Parallelism & Cache Optimization
│   ├── include/           # Header files (matrix.h)
│   ├── src/               # Baseline & multi-threaded pthread engines
│   ├── tests/             # Correctness & edge-case test suite
│   ├── bench/             # Scaled benchmarking harness (1 to 128 threads)
│   ├── Makefile           # Local build automation
│   ├── run_modal.py       # Cloud orchestration script via Modal
│   └── README.md          # Week 1 documentation
├── week_02/               # [Upcoming: SIMD, AVX-512, & Memory Hierarchy]
└── ...
