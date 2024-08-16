#include "common.cuh"
#include <cuda_runtime.h>
#include <iostream>

__constant__ float c_data;
__constant__ float c_data2 = 6.6f;

__global__ void kernel_1(void) {

  printf("Constant data c_data = %.2f.\n", c_data);
}

__global__ void kernel_2(int N) {
  int idx = threadIdx.x;
  if (idx < N) {
  }
}

int main(int argc, char **argv) {

  int devID = 0;
  cudaDeviceProp deviceProps;
  CUDAERRORCHECK(cudaGetDeviceProperties(&deviceProps, devID));
  std::cout << "运行GPU设备:" << deviceProps.name << std::endl;

  float h_data = 8.8f;
  CUDAERRORCHECK(cudaMemcpyToSymbol(c_data, &h_data, sizeof(float)));

  dim3 block(1);
  dim3 grid(1);
  kernel_1<<<grid, block>>>();
  CUDAERRORCHECK(cudaDeviceSynchronize());
  CUDAERRORCHECK(cudaMemcpyFromSymbol(&h_data, c_data2, sizeof(float)));
  printf("Constant data h_data = %.2f.\n", h_data);

  CUDAERRORCHECK(cudaDeviceReset());

  return 0;
}
