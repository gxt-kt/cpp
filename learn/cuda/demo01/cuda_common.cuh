#pragma once

#include "stdio.h"

#include "/home/gxt_kt/Projects/debugstream/debugstream.hpp"

#define cuda_likely(_x) __builtin_expect(!!(_x), 1)
#define cuda_unlikely(_x) __builtin_expect(!!(_x), 0)

inline cudaError_t ErrorCheck(cudaError_t error_code, const char *filename,
                              int lineNumber) {
  if (cuda_unlikely(error_code != cudaSuccess)) {
    printf("[ERROR](%d) %s:%d [%s]: %s\n", error_code, filename, lineNumber,
           cudaGetErrorName(error_code), cudaGetErrorString(error_code));
    // exit(-1);
  }
  return error_code;
}

#define CUDAERRORCHECK(err) ErrorCheck(err, __FILE__, __LINE__);

inline void SetGPU() {
  // 检测计算机GPU数量
  int iDeviceCount = 0;
  cudaError_t error = cudaGetDeviceCount(&iDeviceCount);

  if (error != cudaSuccess || iDeviceCount == 0) {
    printf("No CUDA campatable GPU found!\n");
    exit(-1);
  } else {
    printf("The count of GPUs is %d.\n", iDeviceCount);
  }
  // 设置执行
  int iDev = 0;
  error = cudaSetDevice(iDev);
  if (error != cudaSuccess) {
    printf("fail to set GPU 0 for computing.\n");
    exit(-1);
  } else {
    printf("set GPU 0 for computing.\n");
  }
}

inline void PrintGPUInfo(int device_id) {
  cudaDeviceProp prop;
  CUDAERRORCHECK(cudaGetDeviceProperties(&prop, device_id));

  printf("\n\n\n===================Print GPU Info Begin===================\n");
  printf("Device id:                                 %d\n", device_id);
  printf("Device name:                               %s\n", prop.name);
  printf("Compute capability:                        %d.%d\n", prop.major,
         prop.minor);
  printf("Amount of global memory:                   %g GB\n",
         prop.totalGlobalMem / (1024.0 * 1024 * 1024));
  printf("Amount of constant memory:                 %g KB\n",
         prop.totalConstMem / 1024.0);
  printf("Maximum grid size:                         %d %d %d\n",
         prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
  printf("Maximum block size:                        %d %d %d\n",
         prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
  printf("Number of SMs:                             %d\n",
         prop.multiProcessorCount);
  printf("Maximum amount of shared memory per block: %g KB\n",
         prop.sharedMemPerBlock / 1024.0);
  printf("Maximum amount of shared memory per SM:    %g KB\n",
         prop.sharedMemPerMultiprocessor / 1024.0);
  printf("Maximum number of registers per block:     %d K\n",
         prop.regsPerBlock / 1024);
  printf("Maximum number of registers per SM:        %d K\n",
         prop.regsPerMultiprocessor / 1024);
  printf("Maximum number of threads per block:       %d\n",
         prop.maxThreadsPerBlock);
  printf("Maximum number of threads per SM:          %d\n",
         prop.maxThreadsPerMultiProcessor);
  printf("===================Print GPU Info End===================\n\n\n");
}

inline int GetSPcores(int device_id) {
  cudaDeviceProp devProp;
  CUDAERRORCHECK(cudaGetDeviceProperties(&devProp, device_id));
  int cores = 0;
  int mp = devProp.multiProcessorCount;
  switch (devProp.major) {
  case 2: // Fermi
    if (devProp.minor == 1)
      cores = mp * 48;
    else
      cores = mp * 32;
    break;
  case 3: // Kepler
    cores = mp * 192;
    break;
  case 5: // Maxwell
    cores = mp * 128;
    break;
  case 6: // Pascal
    if ((devProp.minor == 1) || (devProp.minor == 2))
      cores = mp * 128;
    else if (devProp.minor == 0)
      cores = mp * 64;
    else
      printf("Unknown device type\n");
    break;
  case 7: // Volta and Turing
    if ((devProp.minor == 0) || (devProp.minor == 5))
      cores = mp * 64;
    else
      printf("Unknown device type\n");
    break;
  case 8: // Ampere
    if (devProp.minor == 0)
      cores = mp * 64;
    else if (devProp.minor == 6)
      cores = mp * 128;
    else if (devProp.minor == 9)
      cores = mp * 128; // ada lovelace
    else
      printf("Unknown device type\n");
    break;
  case 9: // Hopper
    if (devProp.minor == 0)
      cores = mp * 128;
    else
      printf("Unknown device type\n");
    break;
  default:
    printf("Unknown device type\n");
    break;
  }
  return cores;
}