#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include "rec.h"

using namespace std;

#define IDX(u, i, width) ((u) * (ITEM_COUNT) + i)

#define BLOCK_SIZE 128
#define UPDIV(n,div) ((n + div - 1)/div)

__global__ void mean_kernel(int *compact_data, int *compact_index, double *mean) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  // user starts with 1, so tid=0 work on user 1
  if (tid >= USER_SIZE)
    return ;
  int startIdx = compact_index[tid];
  int endIdx;
  if (tid == USER_SIZE-1) {
    endIdx = DATA_SIZE;
  } else {
    endIdx = compact_index[tid+1];
  }
  double sum = 0.0;
  for (int i = startIdx; i < endIdx; i+=2) {
    sum += compact_data[i+1];
  }
  mean[tid] = 2 * sum / (endIdx - startIdx);
}

__global__ void recommendation_kernel(int user, int *compact_data, int *compact_index,
                           double *sim, double *like) {
  user = user - 1;
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= USER_SIZE)
    return ;
  int u_start = compact_index[user];
  int v_start = compact_index[tid];
  int u_end, v_end;
  if (user == USER_SIZE - 1) {
    u_end = DATA_SIZE - 1;
  } else {
    u_end = compact_index[user+1];
  }
  if (tid == DATA_SIZE - 1) {
    v_end = DATA_SIZE - 1;
  } else {
    v_end = compact_index[tid+1];
  }
  int i = u_start;
  int j = v_start;
  int item_i, item_j;
  while (i < u_end && j < v_end) {
    item_i = compact_data[i];
    item_j = compact_data[j];
    if (item_i == item_j) {
      i += 2;
      j += 2;
    } else if (item_i < item_j) {
      // possible item_j appear in u's ratings
      i += 2;
    } else {
      // item_i > item_j, item_j won't be rated by u
      like[item_j] += sim[tid] * compact_data[j+1];
      j += 2;
    }
  }
  if (j != v_end) {
    while (j < v_end) {
      item_j = compact_data[j];
      like[item_j] += sim[tid] * compact_data[j+1];
      j += 2;
    }
  }
}

__global__ void similarity_kernel(int user, int *compact_data,
                                  int *compact_index,
                                  double *sim, double *mean) {
  user = user - 1;
  // tid is other_user
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= USER_SIZE)
    return ;
  double a = 0.0;
  double b = 0.0;
  double c = 0.0;
  int commons = 0;
  double u_mean = mean[user];
  double v_mean = mean[tid];
  int u_start = compact_index[user];
  int v_start = compact_index[tid];
  int u_end, v_end;
  if (user == USER_SIZE - 1) {
    u_end = DATA_SIZE - 1;
  } else {
    u_end = compact_index[user+1];
  }
  if (tid == DATA_SIZE - 1) {
    v_end = DATA_SIZE - 1;
  } else {
    v_end = compact_index[tid+1];
  }

  int i = u_start;
  int j = v_start;
  double rui, rvi;
  int item_i, item_j;
  while (i < u_end && j < v_end) {
    item_i = compact_data[i];
    item_j = compact_data[j];
    if (item_i == item_j) { // common item
      commons++;
      rui = compact_data[i+1];
      rvi = compact_data[j+1];
      a += (rui - u_mean) * (rvi - v_mean);
      b += (rui - u_mean) * (rui - u_mean);
      c += (rvi - v_mean) * (rvi - v_mean);
      i += 2;
      j += 2;
    } else if (item_i < item_j) {
      i += 2;
    } else {
      j += 2;
    }
  }
  double answer;
  if (b * c == 0) {
    answer = a;
  } else {
    answer = a / (sqrt(b) * sqrt(c));
  }
  if (commons < 5) {
    answer *= 0.2 * commons;
  }
  sim[tid] = answer;
}

/*
  populate the sim vector
 */
void CUDA_populate_user_sim_vec(int target_user, int *compact_data,
                                int *compact_index, double *sim, int topn) {
  // calculate the mean for each user
  int *compact_data_cuda;
  int *compact_index_cuda;
  double *mean_cuda;
  double *sim_cuda;
  //double mean[USER_SIZE];
  cudaMalloc((void **)&compact_data_cuda, DATA_SIZE*sizeof(int));
  cudaMalloc((void **)&compact_index_cuda, USER_SIZE*sizeof(int));
  cudaMalloc((void **)&mean_cuda, USER_SIZE*sizeof(double));
  cudaMalloc((void **)&sim_cuda, USER_SIZE*sizeof(double));
  cudaMemcpy(compact_data_cuda, compact_data, DATA_SIZE*sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(compact_index_cuda, compact_index, USER_SIZE*sizeof(int),
             cudaMemcpyHostToDevice);
  int tpb = 1024;
  mean_kernel<<<UPDIV(USER_SIZE, tpb), tpb>>>(compact_data_cuda,
                                               compact_index_cuda,
                                               mean_cuda);
  /* cudaMemcpy(mean, mean_cuda, USER_SIZE*sizeof(double), cudaMemcpyDeviceToHost); */
  /* for (int i=0; i < USER_SIZE; i++) { */
  /*   cout << "mean for user " << i << ": " << mean[i] << endl; */
  /* } */
  similarity_kernel<<<UPDIV(USER_SIZE, tpb), tpb>>>(target_user,
                                                     compact_data_cuda,
                                                     compact_index_cuda,
                                                     sim_cuda, mean_cuda);
  cudaMemcpy(sim, sim_cuda, USER_SIZE*sizeof(double), cudaMemcpyDeviceToHost);
  cout << "finished sim" << endl;
  for (int i=0; i < USER_SIZE; i++) {
    cout << "sim " << i << ": " << sim[i] << endl;
  }
  cudaFree(sim_cuda);
  cudaFree(mean_cuda);
}

void CUDA_populate_item_like_vec(int user, int *compact_data,
                                 int *compact_index, double *sim,
                                 double *like, int topn) {
  // calculate the mean for each user
  int *compact_data_cuda;
  int *compact_index_cuda;
  double *mean_cuda;
  double *sim_cuda;
  double *like_cuda;
  double mean[USER_SIZE];
  cudaMalloc((void **)&compact_data_cuda, DATA_SIZE*sizeof(int));
  cudaMalloc((void **)&compact_index_cuda, USER_SIZE*sizeof(int));
  cudaMalloc((void **)&mean_cuda, USER_SIZE*sizeof(double));
  cudaMalloc((void **)&sim_cuda, USER_SIZE*sizeof(double));
  cudaMalloc((void **)&like_cuda, ITEM_SIZE*sizeof(double));
  cudaMemcpy(compact_data_cuda, compact_data, DATA_SIZE*sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(compact_index_cuda, compact_index, USER_SIZE*sizeof(int),
             cudaMemcpyHostToDevice);
  int tpb = 1024;
  mean_kernel<<<UPDIV(USER_SIZE, tpb), tpb>>>(compact_data_cuda,
                                               compact_index_cuda,
                                               mean_cuda);
  similarity_kernel<<<UPDIV(USER_SIZE, tpb), tpb>>>(user,
                                                    compact_data_cuda,
                                                    compact_index_cuda,
                                                    sim_cuda, mean_cuda);
  recommendation_kernel<<<UPDIV(USER_SIZE, tpb), tpb>>>(user,
                                                        compact_data_cuda,
                                                        compact_index_cuda,
                                                        sim_cuda, like_cuda);
  cudaFree(sim_cuda);
  cudaFree(mean_cuda);
}
