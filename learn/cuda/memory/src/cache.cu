#include "common.cuh"
#include <cuda_runtime.h>
#include <iostream>

__global__ void kernel(void) {}

int main(int argc, char **argv) {

  int devID = 0;
  cudaDeviceProp deviceProps;
  CUDAERRORCHECK(cudaGetDeviceProperties(&deviceProps, devID));
  std::cout << "运行GPU设备:" << deviceProps.name << std::endl;

  if (deviceProps.globalL1CacheSupported) {
    std::cout << "支持全局内存L1缓存" << std::endl;
  } else {
    std::cout << "不支持全局内存L1缓存" << std::endl;
  }
  std::cout << "L2缓存大小：" << deviceProps.l2CacheSize / (1024 * 1024) << "M"
            << std::endl;

  dim3 block(1);
  dim3 grid(1);
  kernel<<<grid, block>>>();
  CUDAERRORCHECK(cudaDeviceSynchronize());

  CUDAERRORCHECK(cudaDeviceReset());

  return 0;
}
