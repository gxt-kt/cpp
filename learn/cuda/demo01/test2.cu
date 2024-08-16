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
  CUDAERRORCHECK(cudaSetDevice(0));
  hello_from_gpu<<<2, 4>>>();
  cudaDeviceSynchronize();

  PrintGPUInfo(0);
  gDebug(GetSPcores(0));

  return 0;
}


