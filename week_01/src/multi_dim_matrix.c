#include "matrix.h"
#include <pthread.h>
#include <stdlib.h>

static void *matmul_worker(void *arg) {
    ThreadTask *task = (ThreadTask *)arg;
    for (int i = task->start_row; i < task->end_row; ++i) {
        for (int j = 0; j < task->N; ++j) {
            double sum = 0.0;
            for (int p = 0; p < task->K; ++p) {
                sum += task->A[i * task->K + p] * task->B[p * task->N + j];
            }
            task->C[i * task->N + j] = sum;
        }
    }
    return NULL;
}

void matmul_pthread(const double *A, const double *B, double *C, int M, int K, int N, int num_threads) {
    if (num_threads <= 1 || M < 2) {
        matmul_single(A, B, C, M, K, N);
        return;
    }

    if (num_threads > M) {
        num_threads = M;
    }

    pthread_t threads[num_threads];
    ThreadTask tasks[num_threads];

    int rows_per_thread = M / num_threads;
    int remainder = M % num_threads;
    int current_row = 0;

    for (int t = 0; t < num_threads; ++t) {
        int count = rows_per_thread + (t < remainder ? 1 : 0);
        tasks[t] = (ThreadTask){
            .A = A, .B = B, .C = C,
            .M = M, .K = K, .N = N,
            .start_row = current_row,
            .end_row = current_row + count
        };
        current_row += count;
        pthread_create(&threads[t], NULL, matmul_worker, &tasks[t]);
    }

    for (int t = 0; t < num_threads; ++t) {
        pthread_join(threads[t], NULL);
    }
}
