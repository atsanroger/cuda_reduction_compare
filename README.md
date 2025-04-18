# CUDA Reduction Comparison

This project compares three types of reduction strategies plus the `norm2` function used in the [Bridge++ Lattice QCD framework](https://bridge.kek.jp/Lattice-code/).

## Implemented Methods

### 1. Harris-Style Reduction + Norm2
The first method is based on the well-known CUDA reduction techniques introduced by Mark Harris ([PDF link](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf)).  
All optimization strategies described in the paper are implemented, except for the "Idle Threads" mitigation.  
Execution flow: `norm2` → multi-block reduction → final single-block reduction.

### 2. Fused Kernel (Norm2 + Partial Reduction)
The second method fuses the `norm2` computation and per-block reduction into a single kernel, followed by a final single-block reduction kernel.  
This reduces kernel launch overhead and improves memory locality.

### 3. Separate Norm2 + Single Reduction
The third method performs `norm2` followed by a single reduction kernel over the entire result buffer, which is the orign style in Bridge++

## How to Run

```bash
make -j 8
```

This project supports GPUs from Pascal to Hopper architectures.
