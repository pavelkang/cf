#define USER_COUNT 944
#define ITEM_COUNT 2000
#define IDX(u, i, width) ((u) * (ITEM_COUNT) + i)

#define BLOCK_SIZE 128
#define UPDIV(n,div) ((n + div - 1)/div)

__global__ void matrixMultiply(double* A, double* B, double* C,
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
    double answer = 0;
    double a = 0;
    double b = 0;
    double c = 0;
    int commons = 0;

    for (int m = 0; m < (numAColumns-1)/TILE_WIDTH+1; ++m) {
       if (Row < numARows && m*TILE_WIDTH+tx < numAColumns)
          ds_M[ty][tx] = A[Row*numAColumns + m*TILE_WIDTH+tx];
       else
          ds_M[ty][tx] = 0;
       if (Col < numBColumns && m*TILE_WIDTH+ty < numBRows)
          ds_N[ty][tx] = B[(m*TILE_WIDTH+ty)*numBColumns+Col];
       else
          ds_N[ty][tx] = 0;

       __syncthreads();
       for (int k = 0; k < TILE_WIDTH; ++k)
           if ((ds_M[ty][k] != 0) && (ds_N[k][tx] != 0)) {
               commons++;
               double rui = ds_M[ty][k];
               double rvi = ds_N[k][tx];
               double u_mean = mean[Row];
               double v_mean = mean[Col];
               a += (rui - u_mean)*(rvi - v_mean);
               b += (rui-u_mean)*(rui-u_mean);
               c += (rvi-v_mean)*(rvi-v_mean);
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
    cudaMalloc((void **)&d_mean, USER_COUNT * sizeof(double));
    cudaMalloc((void **)&d_out , USER_COUNT * ITEM_COUNT * sizeof(double));

    cudaMemcpy(d_users,   rate,   USER_COUNT * ITEM_COUNT * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_users_t, rate_t, USER_COUNT * ITEM_COUNT * sizeof(double), cudaMemcpyHostToDevice);
    mean_kernel<<<USER_COUNT / 1024 + 1, 1024 >>>(d_users, d_mean, ITEM_COUNT, USER_COUNT);

    dim3 dim_grid(UPDIV(USER_COUNT, BLOCK_SIZE));
    dim3 dim_block(BLOCK_SIZE);

    matrixMultiply<<<dim_grid, dim_block>>>(d_users, d_users_t, d_out, USER_COUNT, ITEM_COUNT, ITEM_COUNT, USER_COUNT, USER_COUNT, USER_COUNT);
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
