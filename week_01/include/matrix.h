#ifndef MATRIX_H
#define MATRIX_H

#include <stdbool.h>

typedef struct {
    const double *A;
    const double *B;
    double *C;
    int M, K, N;
    int start_row;
    int end_row;
} ThreadTask;

/* Single-threaded matrix multiplication: C = A x B */
void matmul_single(const double *A, const double *B, double *C, int M, int K, int N);

/* Multi-threaded matrix multiplication using pthreads */
void matmul_pthread(const double *A, const double *B, double *C, int M, int K, int N, int num_threads);

/* Verification: validates difference within absolute epsilon tolerance */
bool verify_results(const double *C_ref, const double *C_test, int size, double eps);

#endif
