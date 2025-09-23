# MatMul Implementation with Spatz 


This program gives us the dequantization-based double-buffered  GEMM and GEMV kernels with Softhiers existing architecture.

## Overview
The implementation provides computation kernels that use
- AQLM Quantization : Additive Quantization with 2 Codebooks, each having 256 centroids containing 8 values
- Double Buffering: Overlapping DMA Transfers with computation and dequantization
- Tiled Execution : Process Matrix in tiles to compute large matrices while respecting L1 memory constraints
- Vector Processing : Using Spatz VP for dequantization

## Prerequisites
This program assumes you already have Quantized weight matrices (codebooks, indices and scales) in the base repository's `quantfiles`  folder

## Setup and Execution
1) Generate Data Headers and select the kernel (gemv or gemm)
```bash
python {SOFTHIER_ROOT}/examples/SoftHier/assembled/DQ/software/util/store_cb_and_idx.py -kernel <gemv|gemm>
```
2) Buiild hardware and software   via the  Makefile `in this folder`



```bash
examples/SoftHier/assembled/DQ$  make dq
```
  L1 Memory Allocation:

  - Codebooks: [256 × 8 × 2] × sizeof(fp16)
  - Scales: M × sizeof(fp16)
  - Index buffers: 2 × M × max_groups × sizeof(uint16) [2 uint8_t indices packed as uint16_t]
  - Activation buffers: 2 × max_rows × N × sizeof(fp16)
  - Weight tile: M × max_P × sizeof(fp16)
  - Result tile: max_rows × max_P × sizeof(fp16)
## Data Layout
  - Activation Matrix A: Horizontally tiled [A0;  A1; A2; ...]
  - Weight Matrix B: Vertically tiled [B0 | B1 | B2 | ...]
  - Result Matrix/Vector : Stored in HBM
# Implementation of the kernels
## Dimensions
### GEMM
 The GEMM kernel implements $C = A \times B$ where:
  - A: [M × N] activation matrix
  - B: [N × K] quantized weight matrix
  - C: [M × K] result matrix
### GEMV
  The GEMV kernel implements y = $x^T \times W$ where:
  - x: [M × 1] input vector ( $x^T$: [1 × M] )
  - W: [M × K] quantized weight matrix
  - y: [K × 1] result vector
## Implementation
### GEMM
  1. Initialization:
    - Allocate double buffers for indices and activation tiles
    - Load codebooks and scales to L1
    - Allocate HBM result
  2. Outer Loop (B tiles):
    - Load indices for current B_{tile}$
    - Dequantize $B_{tile}$ using Spatz vector unit
    - Process all A tiles with current B_{tile}$
  3. Inner Loop (A tiles):
    - Prefetch next $A_{tile}$ (overlapped with compute)
    - Compute matrix multiplication using RedMule or Spatz
    - Store result tile to HBM with 2D strided DMA
  4. Verification:
    - Compare results against golden reference (preloaded with `store_cb_and_idx.py`)
### GEMV
  1. Initialization:
    - Load input vector x,codebooks and scales to L1 
    - Allocate souble buffers for indices
    - Allocate Hbm result
  3. Loop (B tiles):
    - Dequantize $B_{tile}$ using current indices
    - Prefetch next tile's indices (overlapped)
    - Compute  $x^T \times W$  for current $B_{tile}$
    - Store partial result to HBM
  4. Verification:
    - Compare results against golden reference (preloaded with `store_cb_and_idx.py`)
## Debugging
Enable Debug output with setting the following in `main.c`

#define DEBUG 1


# Todo