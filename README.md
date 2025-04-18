# CUDA Reduction-Norm2 Comparison

 `norm2` and  `reduction` are important building block of linear algebra, which is often used to obtain the length of a vector.

This project compares three types of reduction strategies plus the `norm2` function used in the [Bridge++ Lattice QCD framework](https://bridge.kek.jp/Lattice-code/).

## Implemented Methods

### 1. Harris-Style Reduction + Norm2
The first method is based on the well-known CUDA reduction techniques introduced by Mark Harris ([PDF link](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf)).  
All optimization strategies described in the paper are implemented, except for the "Idle Threads" mitigation.  
### 🔁 **Execution Flow**

`norm2` ➡️ `multi-block reduction` ➡️ `final single-block reduction`

### 2. Fused Kernel (Norm2 + Partial Reduction)
The second method fuses the `norm2` computation and per-block reduction into a single kernel, followed by a final single-block reduction kernel.  
This reduces kernel launch overhead and improves memory locality.

### 🔁 **Execution Flow**

`norm2+multi-block reduction fused kernel` ➡️ `final single-block reduction`


### 3. Separate Norm2 + Single Reduction
The third method performs `norm2` followed by a single reduction kernel over the entire result buffer, which is the orign style in Bridge++

### 🔁 **Execution Flow**

`norm2` ➡️ `Single-block reduction`

## How to Compile

```bash
make -j 8
```

This project supports GPUs from Pascal to Hopper architectures.

## How to Use

```bash
./test_norm nvol
```

Set the argument with your lucky number!

## Pic

Bandwidth and execution time are print in the main function, a simple pic generator against array length is written in the jupyter note
