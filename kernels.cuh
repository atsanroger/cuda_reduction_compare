/*!
      @file    kernels.cuh
      @brief
      @author  Wei-Lun Chen (wlchen)
      @date    $LastChangedDate: 2025-04-19 13:51:53 #$
      @version $LastChangedRevision: 0 $
*/

#pragma once

#include "constant.hpp"      
#include <cuda_runtime.h>    


//================================================================
__global__ void reduce_kernel_multiblocks(real_t* red, int nvol) {
  extern __shared__ real_t sdata[];

  const int tid      = threadIdx.x;
  const int idx      = blockIdx.x * blockDim.x + tid;
  const int gridSize = blockDim.x * gridDim.x;

  real_t sum = 0;
  for (int i = idx; i < nvol; i += gridSize) {
    sum += red[i];
  }
  sdata[tid] = sum;
  __syncthreads();

  for (int s = blockDim.x / 2; s >=  WARP_LENGTH; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  if (tid < WARP_LENGTH){

    sum = sdata[tid];

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (tid == 0){
      red[blockIdx.x] = sum;
    }
  }
}

//====================================================================
__global__ void norm2_kernel(real_t* red, const real_t* __restrict__ v1,
														 int nin, int nvol, int nex)
{
	const int idx      = blockIdx.x * blockDim.x + threadIdx.x;
	const int gridSize = blockDim.x * gridDim.x;    

	for (int ist = idx; ist < nvol; ist += gridSize) {
		real_t at = 0.0;
		for (int ex = 0; ex < nex; ++ex) {
			int ist2 = ist + nvol * ex;
			#pragma unroll
				for (int in = 0; in < nin; ++in) {
					real_t vt = v1[IDX2(nin, in, ist2)];
					at += vt * vt;
				}
			}
			red[ist] = at;  
	}
}

// from Matsufuru
// __global__ void norm2_kernel(real_t* red, real_t* __restrict__ v1,
// 														 int nin, int nvol, int nex)
// {
// 	const int idx = blockIdx.x * blockDim.x + threadIdx.x;

// 	if( idx < nvol ){

// 		real_t at = 0.0;
// 		for(int ex = 0; ex < nex; ++ex){
// 			int ist2 = idx + nvol * ex;
// 			for(int in = 0; in < nin; ++in){
// 				real_t vt1 = v1[IDX2(nin, in, ist2)];
// 				at += vt1 * vt1;
// 			}
// 		}

// 		red[idx] = at;
// 	}
// }

//====================================================
__global__ void norm2_reduce_fused_kernel(real_t* red, 
                                          const real_t* __restrict__ v1, 
                                          int nin, int nvol, int nex)
{
  extern __shared__ real_t sdata[];

  const int tid      = threadIdx.x;
  const int idx      = blockIdx.x * blockDim.x + threadIdx.x;
  const int gridSize = blockDim.x * gridDim.x;

  real_t sum = 0.0;

  for (int ist = idx; ist < nvol; ist += gridSize) {
    real_t at  = 0.0;
    for (int ex = 0; ex < nex; ++ex) {
      int ist2 = ist + nvol * ex;
      for (int in = 0; in < nin; ++in) {
        real_t vt = v1[IDX2(nin, in, ist2)];
        at += vt * vt;
      }
    }
    sum += at;
  }

  sdata[tid] = sum;
  __syncthreads();

  for (int s = blockDim.x/2; s >= WARP_LENGTH; s >>= 1) {
    if (tid < s){
    	sdata[tid] += sdata[tid + s];
		}
    __syncthreads();
  }

  if (tid < WARP_LENGTH){

    sum = sdata[tid];

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (tid == 0){
      red[blockIdx.x] = sum;
    }
  }
}
//--------------------------------------------------------
__global__ void reduce_kernel(real_t* red, int nvol)
{
  __shared__ real_t red2[VECTOR_LENGTH];

  const int ith   = threadIdx.x;
  const int nth   = VECTOR_LENGTH;
  const int nvol2 = nvol/nth;

  real_t at = 0.0;
  for(int ist2 = 0; ist2 < nvol2; ++ist2){
    int ist = ith + nth * ist2;
    at += red[ist];
  }
  red2[ith] = at;

  __syncthreads();

  if(ith == 0){
    real_t at2 = red2[0];
    #pragma unroll
    for(int i = 1; i < VECTOR_LENGTH; ++i){
      at2 += red2[i];
    }
    red[0] = at2;
  }

}
