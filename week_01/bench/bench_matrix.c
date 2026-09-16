#define _POSIX_C_SOURCE 199309L
#include "matrix.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double get_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    int dim = (argc > 1) ? atoi(argv[1]) : 1536;
    printf("================ Matrix Bench (%dx%d) ================\n", dim, dim);

    size_t bytes = (size_t)dim * dim * sizeof(double);
    double *A = malloc(bytes);
    double *B = malloc(bytes);
    double *C_single = malloc(bytes);
    double *C_multi = malloc(bytes);

    for (size_t i = 0; i < (size_t)dim * dim; ++i) {
        A[i] = 1.0001;
        B[i] = 0.9999;
    }

    int thread_counts[] = {1, 4, 16, 32, 64, 128};
    int num_tests = sizeof(thread_counts) / sizeof(thread_counts[0]);
    double baseline_time = 0.0;

    for (int i = 0; i < num_tests; ++i) {
        int threads = thread_counts[i];
        double start = get_time_sec();

        if (threads == 1) {
            matmul_single(A, B, C_single, dim, dim, dim);
        } else {
            matmul_pthread(A, B, C_multi, dim, dim, dim, threads);
        }

        double elapsed = get_time_sec() - start;

        if (threads == 1) {
            baseline_time = elapsed;
            printf("| Threads: %3d | Time: %8.4fs | Speedup: %6.2fx | Status: BASELINE |\n",
                   threads, elapsed, 1.0);
        } else {
            bool ok = verify_results(C_single, C_multi, dim * dim, 1e-5);
            printf("| Threads: %3d | Time: %8.4fs | Speedup: %6.2fx | Status: %s   |\n",
                   threads, elapsed, baseline_time / elapsed, ok ? "VERIFIED" : "MISMATCH");
        }
    }

    free(A); free(B); free(C_single); free(C_multi);
    return 0;
}
