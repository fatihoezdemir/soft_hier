#ifndef _GEMM_PRELOAD_H_
#define _GEMM_PRELOAD_H_

#define X_ADDR ((uint64_t) 0xc0000000)
#define Z_EADDR ((uint64_t) 0xc0080000)
#define W_ADDR ((uint64_t) 0x9c0000000)
#define Z_GADDR ((uint64_t) 0x9c0080000)

// VQ Data Addresses
#define VQ_CODEBOOKS_ADDR ((uint64_t) 0x9c0100000)
#define VQ_INDICES_ADDR ((uint64_t) 0x9c0102000)
#define VQ_SCALES_ADDR ((uint64_t) 0x9c0112000)

#endif // _GEMM_PRELOAD_H_
