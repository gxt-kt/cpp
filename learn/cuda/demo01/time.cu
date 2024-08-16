#include "cuda_common.cuh"
#include <stdio.h>

#include "/home/gxt_kt/Projects/debugstream/debugstream.hpp"

__global__ void hello_from_gpu() {
  const int bid = blockIdx.x;
  const int tid = threadIdx.x;

  const int id = threadIdx.x + blockIdx.x * blockDim.x;
  printf("Hello World from block %d and thread %d, global id %d\n", bid, tid,
         id);
}

int main(void) {

  CUDAERRORCHECK(cudaSetDevice(1));

  // 使用event计算时间
  float time_elapsed = 0;
  cudaEvent_t start, stop;
  cudaEventCreate(&start); // 创建Event
  cudaEventCreate(&stop);

  cudaEventRecord(start, 0); // 记录当前时间
  //===================
  // ....代码执行处
  hello_from_gpu<<<2, 4>>>();
  cudaDeviceSynchronize();
  //===================
  cudaEventRecord(stop, 0); // 记录当前时间

  cudaEventSynchronize(start); // Waits for an event to complete.
  cudaEventSynchronize(stop); // Waits for an event to complete.Record之前的任务
  cudaEventElapsedTime(&time_elapsed, start, stop); // 计算时间差
  cudaEventDestroy(start);                          // destory the event
  cudaEventDestroy(stop);

  printf("执行时间：%f(ms)\n", time_elapsed);

  return 0;
}