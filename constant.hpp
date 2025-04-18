/*!
      @file    constant.hpp
      @brief
      @author  Wei-Lun Chen (wlchen)
      @date    $LastChangedDate: 2025-04-19 13:51:53 #$
      @version $LastChangedRevision: 0 $
*/

#pragma once
using real_t = double;

#define NWP     32

#define NUM_WORKERS 1
#define VECTOR_LENGTH 64

#define WARP_LENGTH 32
#define MAX_THREAD_PER_BLOCK 1024

// general index
#define IDX2(nin, in, ist)      (((ist)%NWP) + NWP*((in) + (nin)*((ist)/NWP)))


#define CHECK(call) \
  { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
      fprintf(stderr, "CUDA Error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
      exit(err); \
    } \
  }

