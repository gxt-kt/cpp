#include "common.cuh"
#include <cuda_runtime.h>
#include <iostream>

__global__ void kernel_1(float *d_A, const int N) {
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int n = bid * blockDim.x + tid;
  __shared__ float s_array[32];

  if (n < N) {
    s_array[tid] = d_A[n];
  }
  __syncthreads();

  if (tid == 0) {
    for (int i = 0; i < 32; ++i) {
      printf("s_array[%d]: %f, blockIdx: %d\n", i, s_array[i], bid);
    }
  }
}

int main(int argc, char **argv) {
  int devID = 0;
  cudaDeviceProp deviceProps;
  CUDAERRORCHECK(cudaGetDeviceProperties(&deviceProps, devID));
  std::cout << "运行GPU设备:" << deviceProps.name << std::endl;

  int nElems = 64;
  int nbytes = nElems * sizeof(float);

  float *h_A = nullptr;
  h_A = (float *)malloc(nbytes);
  for (int i = 0; i < nElems; ++i) {
    h_A[i] = float(i);
  }

  float *d_A = nullptr;
  CUDAERRORCHECK(cudaMalloc(&d_A, nbytes));
  CUDAERRORCHECK(cudaMemcpy(d_A, h_A, nbytes, cudaMemcpyHostToDevice));

  dim3 grid(2);
  dim3 block(32);
  kernel_1<<<grid, block>>>(d_A, nElems);

  CUDAERRORCHECK(cudaFree(d_A));
  free(h_A);
  CUDAERRORCHECK(cudaDeviceReset());
}
