#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#define USER_COUNT 944
#define ITEM_COUNT 2000
#define IDX(u, i, width) ((u) * (ITEM_COUNT) + i)
#define IDX_T(u,i) ((i) * USER_COUNT + u)

#define BLOCK_SIZE 128
#define TILE_WIDTH 16
#define UPDIV(n,div) ((n + div - 1)/div)

__global__ void recommendation_kernel(double *sim, double *users, double *out, const int user, const int N, const int M) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    __shared__ double sim_shared[BLOCK_SIZE];

    double sum = 0;
    for (int j = 0; j < UPDIV(N, BLOCK_SIZE); j++) {

        sim_shared[threadIdx.x] = ((j * BLOCK_SIZE + threadIdx.x) < N)
                                  ? sim[(threadIdx.x + j * BLOCK_SIZE)]
                                  : 0.0;
        __syncthreads();


        if (tid < M && users[IDX(user, tid, M)] == 0) {
            #pragma unroll
            for (int e = 0; e < BLOCK_SIZE; e++) {
                int i = e + BLOCK_SIZE * j;
                if (i < N) {
                    sum += sim_shared[e] * users[IDX(i, tid, M)];
                }
            }
        }
        __syncthreads();
    }
    /*
    if (users[IDX(user, tid, M)] == 0)  {
        for (int i = 0; i < USER_COUNT; i++) {
            sum += sim[i] * users[IDX(i, tid, M)];
        }
    }
    */


    if (tid < M) out[tid] = sum;

}


__global__ void similarity_kernel(double * mean, double *users, double *out, const int user,const int N, const int M) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    //printf("tid %d\n", tid);
    double a = 0.0;
    double b = 0.0;
    double c = 0.0;
    int commons = 0;

    double v_mean = mean[user];
    __shared__ double user_rate_shared[BLOCK_SIZE];

    for (int j = 0; j < UPDIV(N, BLOCK_SIZE); j++) {

        if ((j * BLOCK_SIZE + threadIdx.x) < N) {
            user_rate_shared[threadIdx.x] = users[IDX(user, (threadIdx.x + j * BLOCK_SIZE), N)];
        } else {
            user_rate_shared[threadIdx.x] = 0.f;
        }
        /*
        user_rate_shared[threadIdx.x] = ((j * BLOCK_SIZE + threadIdx.x) < N)
                                        ? users[IDX(user, (threadIdx.x + j * BLOCK_SIZE) ,N)]
                                        : 0.0;
                                        */
        __syncthreads();

        #pragma unroll

        for (int e = 0; e < BLOCK_SIZE; e++) {
            int i = e + BLOCK_SIZE * j;

            if ((tid < M) && (user_rate_shared[e] != 0) && (users[IDX(tid, (i), N)] != 0)) {
                double u_mean = mean[tid];
                commons++;
                double rui = users[IDX(tid,i,N)];
                double rvi = user_rate_shared[e];
                a += (rui - u_mean)*(rvi - v_mean);
                b += (rui-u_mean)*(rui-u_mean);
                c += (rvi-v_mean)*(rvi-v_mean);

            }
        }
        __syncthreads();
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

    if (tid < M) out[tid] = answer;
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

__global__ void matrixMultiply(double* A, double* B, double* C, double *mean,
  		       int numARows, int numAColumns,
			       int numBRows, int numBColumns,
			       int numCRows, int numCColumns) {
    //@@ Insert code to implement matrix multiplication here
    __shared__ double ds_M[TILE_WIDTH][TILE_WIDTH];
    __shared__ double ds_N[TILE_WIDTH][TILE_WIDTH];
    int bx = blockIdx.x, by = blockIdx.y,
       tx = threadIdx.x, ty = threadIdx.y,
       Row = by * TILE_WIDTH + ty,
       Col = bx * TILE_WIDTH + tx;

    double u_mean = mean[Row];
    double v_mean = mean[Col];
    double answer = 0;
    double a = 0;
    double b = 0;
    double c = 0;
    int commons = 0;

    for (int m = 0; m < UPDIV(numAColumns, TILE_WIDTH); ++m) {
       if (Row < numARows && m*TILE_WIDTH+tx < numAColumns)
          //ds_M[ty][tx] = A[Row*numAColumns + m*TILE_WIDTH+tx];
          ds_M[ty][tx] = A[IDX(Row, m*TILE_WIDTH+tx, ITEM_COUNT)];
       else
          ds_M[ty][tx] = 0;
       if (Col < numBColumns && m*TILE_WIDTH+ty < numBRows)
          //ds_N[ty][tx] = B[(m*TILE_WIDTH+ty)*numBColumns+Col];
          ds_N[ty][tx] = B[IDX_T(Col, m*TILE_WIDTH + ty)];
       else
          ds_N[ty][tx] = 0;

       __syncthreads();
       for (int k = 0; k < TILE_WIDTH; ++k) {
           if ((ds_M[ty][k] != 0) && (ds_N[k][tx] != 0)) {
               commons++;
               double rui = ds_M[ty][k];
               double rvi = ds_N[k][tx];
               a += (rui-u_mean)*(rvi-v_mean);
               b += (rui-u_mean)*(rui-u_mean);
               c += (rvi-v_mean)*(rvi-v_mean);
           }
       }

           // Pvalue += ds_M[ty][k] * ds_N[k][tx];
       __syncthreads();
    }
    if (Row < numCRows && Col < numCColumns) {
        if (Row == Col) {
            C[Row*numCColumns+Col] = 0;
            return;
        }
        if (b*c == 0) {
          answer = a;
        } else {
          answer = a / (sqrt(b) * sqrt(c));
        }
        // fix pearson
        if (commons < 5) {
          answer *= 0.2 * commons;
        }
        C[Row*numCColumns+Col] = answer;
    }
}


void computeCorrelation(double *rate, double *rate_t, double *out) {
    double *d_users;
    double *d_users_t;
    double *d_mean;
    double *d_out;
    cudaMalloc((void **)&d_users, USER_COUNT * ITEM_COUNT * sizeof(double));
    cudaMalloc((void **)&d_users_t, USER_COUNT * ITEM_COUNT * sizeof(double));
    cudaMalloc((void **)&d_mean, USER_COUNT * sizeof(double));
    cudaMalloc((void **)&d_out , USER_COUNT * USER_COUNT * sizeof(double));

    cudaMemcpy(d_users,   rate,   USER_COUNT * ITEM_COUNT * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_users_t, rate_t, USER_COUNT * ITEM_COUNT * sizeof(double), cudaMemcpyHostToDevice);
    mean_kernel<<<USER_COUNT / 1024 + 1, 1024 >>>(d_users, d_mean, ITEM_COUNT, USER_COUNT);

    dim3 dim_grid(UPDIV(USER_COUNT, TILE_WIDTH), UPDIV(USER_COUNT, TILE_WIDTH));
    dim3 dim_block(TILE_WIDTH, TILE_WIDTH, 1);

    matrixMultiply<<<dim_grid, dim_block>>>(d_users, d_users_t, d_out, d_mean, USER_COUNT, ITEM_COUNT, ITEM_COUNT, USER_COUNT, USER_COUNT, USER_COUNT);
    cudaMemcpy(out, d_out, USER_COUNT * USER_COUNT * sizeof(double), cudaMemcpyDeviceToHost);


    cudaError_t errCode = cudaPeekAtLastError();
    if (errCode != cudaSuccess) {
        fprintf(stderr, "WARNING: A CUDA error occured: code=%d, %s\n", errCode, cudaGetErrorString(errCode));
    }

    cudaFree(d_users);
    cudaFree(d_users_t);
    cudaFree(d_mean);
    cudaFree(d_out);

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

    dim3 dim_grid(UPDIV(USER_COUNT, BLOCK_SIZE));
    dim3 dim_block(BLOCK_SIZE);

    similarity_kernel<<<dim_grid, dim_block>>>(d_mean, d_users, d_out, user, ITEM_COUNT, USER_COUNT);
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

void CUDA_get_recommendations(double *um, int user, double *recommendation_copy, int topn = 5) {

    double *d_users;
    double *d_mean;
    double *d_sim;
    double *d_out;
    cudaMalloc((void **)&d_users, USER_COUNT * ITEM_COUNT * sizeof(double));
    cudaMalloc((void **)&d_mean, USER_COUNT * sizeof(double));
    cudaMalloc((void **)&d_sim , USER_COUNT * sizeof(double));
    cudaMalloc((void **)&d_out , ITEM_COUNT * sizeof(double));

    cudaMemcpy(d_users, um, USER_COUNT * ITEM_COUNT * sizeof(double), cudaMemcpyHostToDevice);
    mean_kernel<<<USER_COUNT / 1024 + 1, 1024 >>>(d_users, d_mean, ITEM_COUNT, USER_COUNT);

    dim3 dim_grid(UPDIV(USER_COUNT, BLOCK_SIZE));
    dim3 dim_block(BLOCK_SIZE);

    similarity_kernel<<<dim_grid, dim_block>>>(d_mean, d_users, d_sim, user, ITEM_COUNT, USER_COUNT);

    dim3 rdim_grid(UPDIV(ITEM_COUNT, BLOCK_SIZE));
    recommendation_kernel<<<rdim_grid, dim_block>>>(d_sim, d_users, d_out, user, USER_COUNT, ITEM_COUNT);

    cudaMemcpy(recommendation_copy, d_out, ITEM_COUNT * sizeof(double), cudaMemcpyDeviceToHost);

    cudaError_t errCode = cudaPeekAtLastError();
    if (errCode != cudaSuccess) {
        fprintf(stderr, "WARNING: A CUDA error occured: code=%d, %s\n", errCode, cudaGetErrorString(errCode));
    }

    cudaFree(d_users);
    cudaFree(d_mean);
    cudaFree(d_sim);
    cudaFree(d_out);
}


void CUDA_get_rec_with_sim(double *users, double *sim, int user, double *recommendation_copy, int topn = 5) {

    double *d_sim;
    double *d_users;
    double *d_out;
    cudaMalloc((void **)&d_sim , USER_COUNT * sizeof(double));
    cudaMalloc((void **)&d_users, USER_COUNT * ITEM_COUNT * sizeof(double));
    cudaMalloc((void **)&d_out , ITEM_COUNT * sizeof(double));

    cudaMemcpy(d_users, users, USER_COUNT * ITEM_COUNT * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sim, &(sim[user * USER_COUNT]), USER_COUNT * sizeof(double), cudaMemcpyHostToDevice);

    dim3 dim_grid(UPDIV(ITEM_COUNT, BLOCK_SIZE));
    dim3 dim_block(BLOCK_SIZE);

    recommendation_kernel<<<dim_grid, dim_block>>>(d_sim, d_users, d_out, user, USER_COUNT, ITEM_COUNT);

    cudaMemcpy(recommendation_copy, d_out, ITEM_COUNT * sizeof(double), cudaMemcpyDeviceToHost);

    cudaError_t errCode = cudaPeekAtLastError();
    if (errCode != cudaSuccess) {
        fprintf(stderr, "WARNING: A CUDA error occured: code=%d, %s\n", errCode, cudaGetErrorString(errCode));
    }

    cudaFree(d_users);
    cudaFree(d_sim);
    cudaFree(d_out);
}
