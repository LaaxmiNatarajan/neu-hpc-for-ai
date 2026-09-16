#include "matrix.h"
#include <math.h>

void matmul_single(const double *A, const double *B, double *C, int M, int K, int N) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            double sum = 0.0;
            for (int p = 0; p < K; ++p) {
                sum += A[i * K + p] * B[p * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

bool verify_results(const double *C_ref, const double *C_test, int size, double eps) {
    for (int i = 0; i < size; ++i) {
        if (fabs(C_ref[i] - C_test[i]) > eps) {
            return false;
        }
    }
    return true;
}
