#include "common.cuh"

__global__ void Reduce(int *data, int n) {
  // int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int tid = GetTid();

#define REDUCE_TYPE 2

#if REDUCE_TYPE == 0
  if (tid == 0) {
    for (int i = 1; i < n; i++) {
      data[0] += data[i];
    }
  }
#elif REDUCE_TYPE == 1
  for (int stride = 1; stride < n; stride *= 2) {
    if (tid % (2 * stride) == 0) {
      data[tid] += data[tid + stride];
    }
    __syncthreads();
  }
#elif REDUCE_TYPE == 2
  for (int cnt = n / 2; cnt >= 1; cnt /= 2) {
    if (tid >= cnt) {
      return;
    }
    data[tid] += data[tid + cnt];
    __syncthreads();
  }
#endif
}

int main() {
  SetGPU();
  const int n = 8192;
  std::shared_ptr<int[]> nums = std::shared_ptr<int[]>(new int[n]);
  for (int i = 0; i < n; i++) {
    nums[i] = 1 + i;
  }
  int *cuda_nums;
  CUDAERRORCHECK(cudaMalloc(&cuda_nums, n * sizeof(int)));
  CUDAERRORCHECK(cudaMemcpy(cuda_nums, nums.get(), n * sizeof(int),
                            cudaMemcpyHostToDevice));

  dim3 grid(n / 512);
  dim3 blocks(512);
  TIME_BEGIN_US(reduce);
  Reduce<<<grid, blocks>>>(cuda_nums, n);
  cudaDeviceSynchronize();
  TIME_END(reduce);

  int result;
  CUDAERRORCHECK(
      cudaMemcpy(&result, cuda_nums, 1 * sizeof(int), cudaMemcpyDeviceToHost));
  gDebug(result);

  cudaFree(cuda_nums);
  return 0;
}
