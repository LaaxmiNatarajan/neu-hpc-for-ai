#include "matrix.h"
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    srand(42);
    printf("================ Running Functional & Corner Case Tests ================\n");

    struct { int M, K, N; const char *desc; } test_cases[] = {
        {1, 1, 1, "Scalar (1x1)"},
        {1, 50, 1, "Dot Product (1x50 * 50x1)"},
        {50, 1, 50, "Outer Product (50x1 * 1x50)"},
        {1, 64, 64, "Vector x Matrix (1x64 * 64x64)"},
        {64, 64, 1, "Matrix x Vector (64x64 * 64x1)"},
        {13, 29, 37, "Prime Dimension Incompatibilities"},
        {64, 64, 64, "Square Matrix (64x64)"},
        {7, 128, 11, "Tall-and-Skinny Intermediate Matrix"}
    };

    int num_cases = sizeof(test_cases) / sizeof(test_cases[0]);
    int thread_counts[] = {1, 2, 4, 8, 16};

    for (int c = 0; c < num_cases; ++c) {
        int M = test_cases[c].M;
        int K = test_cases[c].K;
        int N = test_cases[c].N;

        double *A = malloc(M * K * sizeof(double));
        double *B = malloc(K * N * sizeof(double));
        double *C_single = malloc(M * N * sizeof(double));
        double *C_multi = malloc(M * N * sizeof(double));

        for (int i = 0; i < M * K; ++i) A[i] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
        for (int i = 0; i < K * N; ++i) B[i] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;

        matmul_single(A, B, C_single, M, K, N);

        for (int t = 0; t < 5; ++t) {
            int tc = thread_counts[t];
            matmul_pthread(A, B, C_multi, M, K, N, tc);
            if (!verify_results(C_single, C_multi, M * N, 1e-5)) {
                printf("[FAILED] Case '%s' failed with %d threads.\n", test_cases[c].desc, tc);
                return 1;
            }
        }
        printf("[PASS] %-40s (Dim: %dx%dx%d)\n", test_cases[c].desc, M, K, N);

        free(A); free(B); free(C_single); free(C_multi);
    }

    printf("All test cases passed validation successfully!\n");
    return 0;
}
