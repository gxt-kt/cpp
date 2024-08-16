#include <stdio.h>

// #include <iostream>

__global__ void hello_from_gpu() {
  // 核函数不支持iostream
  // std::cout << "hello cuda" << std::endl;
  printf("Hello World from the the GPU\n");
}

int main(void) {
  hello_from_gpu<<<4, 4>>>(); // 第一个指线程块数，第二个指每个线程块有多少线程
  {
    cudaError_t cudaerr = cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
    if (cudaerr != cudaSuccess) {
      printf("kernel launch failed with error \"%s\".\n",
             cudaGetErrorString(cudaerr));
    }
  }
  return 0;
}
