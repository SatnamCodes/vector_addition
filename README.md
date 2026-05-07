# CUDA Vector Addition - Learning Summary

## Overview
This project implements vector addition on NVIDIA GPUs using CUDA. It adds two vectors of 100 million elements (N = 100,000,000) and measures kernel execution time.

## Key Concepts Learned

### 1. **Global Index Calculation**
```c
int i = blockIdx.x * blockDim.x + threadIdx.x;
```
Each thread computes a unique global index using:
- `blockIdx.x`: which block in the grid
- `blockDim.x`: number of threads per block
- `threadIdx.x`: thread index within the block

### 2. **Thread Configuration**
- **Threads per Block**: 256 (chosen as a multiple of 32 warp size for efficiency)
- **Number of Blocks**: `(N + 255) / 256` = 390,625 blocks
- **Total Threads**: ~100 million (one per element)

### 3. **Memory Management**
- Allocate host (CPU) memory with `malloc()`
- Allocate device (GPU) memory with `cudaMalloc()`
- Transfer data: `cudaMemcpy()` with `cudaMemcpyHostToDevice` and `cudaMemcpyDeviceToHost`

### 4. **Performance Measurement**
- Used CUDA Events (`cudaEventCreate()`, `cudaEventRecord()`, `cudaEventElapsedTime()`)
- Ensures accurate GPU kernel timing without CPU overhead

## Execution Results
Ran kernel 5 times with timing measurements:

| Run | Execution Time (ms) |
|-----|-------------------|
| 1   | 5.577              |
| 2   | 5.325              |
| 3   | 5.315              |
| 4   | 5.530              |
| 5   | 6.794              |
| **Avg** | **5.708**      |

### Observations
- Consistent performance across runs (5-6.8 ms)
- Slight variance likely due to system scheduling
- All outputs correct: 1.0 + 2.0 = 3.0 ✓

## Why This Matters
- **Speedup over CPU**: A CPU version would take ~50-100ms for 100M operations
- **GPU Parallelism**: 100+ million threads execute in parallel, completing in ~5.7ms
- **Efficiency**: 256 threads/block = 8 warps (1 warp = 32 threads), perfect alignment with GPU architecture

## Compilation & Execution
```bash
nvcc -o vector_add vector_add.cu
./vector_add
```

## Key Takeaway
CUDA enables massive parallelism by distributing work across thousands of threads. The kernel demonstrates the fundamental GPU programming pattern: each thread handles one data element independently, enabling 10-20x speedups over CPU implementations for data-parallel workloads.
