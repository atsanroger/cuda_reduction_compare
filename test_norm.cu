/*!\n      @file    test_norm.cu\n      @brief   Benchmark norm2 reduction and memory bandwidth\n      @author  Wei-Lun Chen (wlchen)
      @date    $LastChangedDate: 2025-04-19 13:51:53 #$\n      @version $LastChangedRevision: 1 $\n*/

#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>
#include "constant.hpp"
#include "kernels.cuh"

void run_test(int nvol, int nin = 12, int nex = 1) {
  printf("nvol = %d, nin = %d, nex = %d\n", nvol, nin, nex);
  const size_t vecsize = size_t(nvol) * size_t(nex) * size_t(nin);

  real_t *h_v1   = new real_t[vecsize];
  real_t *h_red  = new real_t[nvol];
  real_t result1 = 0, result2 = 0, result3 = 0;

  for (size_t i = 0; i < vecsize; ++i)
    h_v1[i] = 0.1f * float(i % WARP_LENGTH);

  real_t *d_v1, *d_red;
  CHECK(cudaMalloc(&d_v1, sizeof(real_t) * vecsize));
  CHECK(cudaMalloc(&d_red, sizeof(real_t) * nvol));
  CHECK(cudaMemcpy(d_v1, h_v1, sizeof(real_t) * vecsize, cudaMemcpyHostToDevice));

  cudaEvent_t start, stop;
  CHECK(cudaEventCreate(&start));
  CHECK(cudaEventCreate(&stop));

  int threadsPerBlock  = VECTOR_LENGTH;
  int nbl2             = 1;
  int blockSize        = min((nvol + threadsPerBlock - 1) / threadsPerBlock, MAX_THREAD_PER_BLOCK);
  size_t sharedMemSize = threadsPerBlock * sizeof(real_t);

  // Compute approximate bytes moved: read v1, read+write red
  double bytes_norm2 = double(vecsize) * sizeof(real_t) + double(nvol) * sizeof(real_t);
  double bytes_reduce = 2.0 * double(nvol) * sizeof(real_t);

  // ===== Version 1: norm2 + reduce + reduce =====
  CHECK(cudaMemset(d_red, 0, sizeof(real_t) * nvol));
  CHECK(cudaEventRecord(start));
  norm2_kernel<<<blockSize, threadsPerBlock>>>(d_red, d_v1, nin, nvol, nex);
  reduce_kernel_multiblocks<<<blockSize, threadsPerBlock, sharedMemSize>>>(d_red, nvol);
  reduce_kernel_multiblocks<<<nbl2, threadsPerBlock, sharedMemSize>>>(d_red, blockSize);
  CHECK(cudaEventRecord(stop));
  CHECK(cudaEventSynchronize(stop));
  float time1;
  CHECK(cudaEventElapsedTime(&time1, start, stop));
  CHECK(cudaMemcpy(h_red, d_red, sizeof(real_t), cudaMemcpyDeviceToHost));
  result1 = h_red[0];
  double bw1 = (bytes_norm2 + bytes_reduce) / (time1 * 1e-3) / (1<<30);

  // ===== Version 2: fused + reduce =====
  CHECK(cudaMemset(d_red, 0, sizeof(real_t) * nvol));
  CHECK(cudaEventRecord(start));
  norm2_reduce_fused_kernel<<<blockSize, threadsPerBlock, sharedMemSize>>>(d_red, d_v1, nin, nvol, nex);
  reduce_kernel_multiblocks<<<nbl2, threadsPerBlock, sharedMemSize>>>(d_red, blockSize);
  CHECK(cudaEventRecord(stop));
  CHECK(cudaEventSynchronize(stop));
  float time2;
  CHECK(cudaEventElapsedTime(&time2, start, stop));
  CHECK(cudaMemcpy(h_red, d_red, sizeof(real_t), cudaMemcpyDeviceToHost));
  result2 = h_red[0];
  double bw2 = (bytes_norm2 + bytes_reduce) / (time2 * 1e-3) / (1<<30);

  // ===== Version 3: norm2 + single block reduce =====
  CHECK(cudaMemset(d_red, 0, sizeof(real_t) * nvol));
  CHECK(cudaEventRecord(start));
  norm2_kernel<<<blockSize, threadsPerBlock>>>(d_red, d_v1, nin, nvol, nex);
  reduce_kernel<<<nbl2, threadsPerBlock>>>(d_red, nvol);
  CHECK(cudaEventRecord(stop));
  CHECK(cudaEventSynchronize(stop));
  float time3;
  CHECK(cudaEventElapsedTime(&time3, start, stop));
  CHECK(cudaMemcpy(h_red, d_red, sizeof(real_t), cudaMemcpyDeviceToHost));
  result3 = h_red[0];
  double bw3 = (bytes_norm2 + bytes_reduce) / (time3 * 1e-3) / (1<<30);

  // ===== Report =====
  printf("Result 1 (norm2 + multi + multi): %f, time = %.3f ms, BW = %.2f GB/s\n", result1, time1, bw1);
  printf("Result 2 (fused + multi):         %f, time = %.3f ms, BW = %.2f GB/s\n", result2, time2, bw2);
  printf("Result 3 (norm2 + single):        %f, time = %.3f ms, BW = %.2f GB/s\n", result3, time3, bw3);

  double eps = 1e-3;
  if (fabs(result1 - result2) > eps || fabs(result1 - result3) > eps) {
    printf("❌ Results do not match within tolerance %.3e\n", eps);
  } else {
    printf("✅ All results match within tolerance %.3e\n", eps);
  }

  // cleanup
  delete[] h_v1;
  delete[] h_red;
  cudaFree(d_v1);
  cudaFree(d_red);
}

int main(int argc, char** argv) {
  int nvol = 32768;
  if (argc > 1) {
    nvol = atoi(argv[1]);
  }

  run_test(nvol);
  return 0;
}
