#include "/home/gxt_kt/Projects/debugstream/debugstream.hpp"
#include <stdio.h>

__global__ void Add(float *a, float *b, float *c, size_t cnts) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= cnts)
    return;
  *(c + tid) = *(a + tid) + *(b + tid);
}

int main() {
  int gpu_counts = 0;
  cudaError_t err = cudaGetDeviceCount(&gpu_counts);
  if (err != cudaSuccess) {
    gDebug(cudaGetErrorString(err));
  }
  gDebug(gpu_counts);
  cudaSetDevice(gpu_counts);

  const int cnts = 1025;
  const int bytes = cnts * sizeof(float);

  auto host_a = std::shared_ptr<float[]>(new float[cnts]);
  auto host_b = std::shared_ptr<float[]>(new float[cnts]);
  auto host_c = std::shared_ptr<float[]>(new float[cnts]);
  memset(host_a.get(), 0, bytes);
  memset(host_b.get(), 0, bytes);
  memset(host_c.get(), 0, bytes);

  float *cuda_a;
  float *cuda_b;
  float *cuda_c;
  cudaMalloc(&cuda_a, bytes);
  cudaMalloc(&cuda_b, bytes);
  cudaMalloc(&cuda_c, bytes);
  cudaMemset(cuda_a, 0, bytes);
  cudaMemset(cuda_b, 0, bytes);
  cudaMemset(cuda_c, 0, bytes);

  for (int i = 0; i < cnts; i++) {
    host_a[i] = i * 0.1;
    host_b[i] = i * 0.2;
  }

  cudaMemcpy(cuda_a, host_a.get(), bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(cuda_b, host_b.get(), bytes, cudaMemcpyHostToDevice);

  dim3 grid(32, 1, 1);
  dim3 block((cnts + 32 - 1) / 32, 1, 1);
  Add<<<grid, block>>>(cuda_a, cuda_b, cuda_c, cnts);

  cudaDeviceSynchronize();
  cudaMemcpy(host_c.get(), cuda_c, bytes, cudaMemcpyDeviceToHost);
  for (int i = 0; i < cnts; i++) {
    gDebug() << VAR(i, host_c.get()[i]);
  }

  cudaFree(cuda_a);
  cudaFree(cuda_b);
  cudaFree(cuda_c);

  cudaDeviceReset();

  return 0;
}
