#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#define USER_COUNT 944
#define ITEM_COUNT 2000
#define IDX(u, i, width) ((u) * width + i)

#define BLOCK_SIZE 128
#define UPDIV(n,div) ((n + div - 1)/div)

__global__ void similarity_kernel(double * mean, double *users, double *out, const int user,const int N, const int M) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    //printf("tid %d\n", tid);
    if (tid > M) return;
    double a = 0.0;
    double b = 0.0;
    double c = 0.0;
    int commons = 0;

    double v_mean = mean[user];
    __shared__ user_rate_shared[BLOCK_SIZE];
    for (int i = 0; i < N; i++) {
        if ((users[IDX(user, i, N)] != 0) && (users[IDX(tid, i, N)] != 0)) {
            double u_mean = mean[tid];
            commons++;
            double rui = users[IDX(tid,i,N)];
            double rvi = users[IDX(user,i,N)];
            a += (rui - u_mean)*(rvi - v_mean);
            b += (rui-u_mean)*(rui-u_mean);
            c += (rvi-v_mean)*(rvi-v_mean);

        }
    }

    double answer;
    if (b*c == 0) {
        answer = a;
    } else {
        answer = a / (sqrt(b) * sqrt(c));
    }
    // fix pearson
    if (commons < 5) {
        answer *= 0.2 * commons;
    }

    out[tid] = answer;
}

__global__ void mean_kernel(double *users, double *out, const int N, const int M) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    double sum = 0;
    int count = 0;
    if (tid > M) return;
    for (int i = 0; i < N; i++) {
        double rate = users[tid * N + i];
        sum += rate;
        if (rate) count++;
    }
    out[tid] = sum / count;
}

void CUDA_get_similar_users(double *um, int user, double *similarity_copy, int topn = 5) {

    double *d_users;
    double *d_mean;
    double *d_out;
    cudaMalloc((void **)&d_users, USER_COUNT * ITEM_COUNT * sizeof(double));
    cudaMalloc((void **)&d_mean, USER_COUNT * sizeof(double));
    cudaMalloc((void **)&d_out , USER_COUNT * sizeof(double));

    cudaMemcpy(d_users, um, USER_COUNT * ITEM_COUNT * sizeof(double), cudaMemcpyHostToDevice);
    mean_kernel<<<USER_COUNT / 1024 + 1, 1024 >>>(d_users, d_mean, ITEM_COUNT, USER_COUNT);
    similarity_kernel<<<USER_COUNT / 256 + 1, 256>>>(d_mean, d_users, d_out, user, ITEM_COUNT, USER_COUNT);
    cudaMemcpy(similarity_copy, d_out, USER_COUNT * sizeof(double), cudaMemcpyDeviceToHost);

    similarity_copy[user] = 0;

    cudaError_t errCode = cudaPeekAtLastError();
    if (errCode != cudaSuccess) {
        fprintf(stderr, "WARNING: A CUDA error occured: code=%d, %s\n", errCode, cudaGetErrorString(errCode));
    }

    cudaFree(d_users);
    cudaFree(d_mean);
    cudaFree(d_out);
}
