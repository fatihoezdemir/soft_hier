#ifndef _VQ_KERNELS_VPTQ_H_
#define _VQ_KERNELS_VPTQ_H_
#include "flex_printf.h"
#include "gemm_setup.h"
#include "spatz_rvv_extensions.h"

#include <stdbool.h>

#if VQ_ENABLED == 1


// vptq baseline_kernel
static inline void summa_vq_dequantize_tile_vptq_baseline(const SummaGEMMInfo* info, uint32_t dst_L1_W, int buffer_idx,
                                                          uint32_t src_L1_Scales, int k_tile) {
    (void)src_L1_Scales;
    (void)k_tile;
    const uint16_t* idx_cb0_base =
        (const uint16_t*)(uintptr_t)((buffer_idx == 0) ? info->vq.L1_IDX1[0] : info->vq.L1_IDX2[0]);
    const uint16_t* cb0_base = (const uint16_t*)(uintptr_t)info->vq.L1_CB[0];
    uint16_t* W_out          = (uint16_t*)(uintptr_t)dst_L1_W;

    const uint32_t groups_per_col = info->vq.K_tile_compressed; // compressed K groups per column (pre-transposed)

    for (uint32_t n = 0; n < info->N_tile; ++n) {
        const uint16_t* idx_ptr = idx_cb0_base + n * groups_per_col;
        uint16_t* out_ptr       = W_out + n * info->K_tile; // output is N_tile x K_tile (col-major scratch)

        uint32_t remaining_groups = groups_per_col;
        while (remaining_groups > 0) {
            uint32_t vl_groups = 0;
            asm volatile("vsetvli %0, %1, e16, m1, ta, ma" : "=r"(vl_groups) : "r"(remaining_groups));
            uint32_t groups_this_iter = vl_groups;
            uint32_t payload_elems    = groups_this_iter * VQ_GROUP_SIZE;

            asm volatile("vsetvli zero, %3, e16, m1, ta, ma\n" // set vl for index vector (groups)
                         "vle16.v    v0, (%0)\n"               // load indices
                         "vsetvli zero, %4, e16, m8, ta, ma\n" // set vl for payload (groups * 8)
                         "mv        t0, %1\n"
                         ".word     %2\n"                      // block gather centroids -> v8
                         "vse16.v   v8, (%5)\n"                // store dequantized weights
                         :
                         : "r"(idx_ptr),                             // %0
                           "r"(cb0_base),                            // %1
                           "i"(VLBLK1EI16_V(RVV_V8, RVV_V0, RVX_T0, 1)), // %2
                           "r"(groups_this_iter),                    // %3
                           "r"(payload_elems),                       // %4
                           "r"(out_ptr)                              // %5
                         : "t0", "v0", "v8", "memory");

            idx_ptr += groups_this_iter;
            out_ptr += payload_elems;
            remaining_groups -= groups_this_iter;
        }
    }
}

// Dequantize without pre-transposed indices using block gather + strided store (avoid transpose buffer)
static inline void summa_vq_dequantize_tile_vptq_ssseg(const SummaGEMMInfo* info, uint32_t dst_L1_W, int buffer_idx,
                                                       uint32_t src_L1_Scales, int k_tile) {
    (void)src_L1_Scales;
    (void)k_tile;
    const uint16_t* idx_base =
        (const uint16_t*)(uintptr_t)((buffer_idx == 0) ? info->vq.L1_IDX1[0] : info->vq.L1_IDX2[0]); // Kc x N layout
    const uint16_t* cb0_base = (const uint16_t*)(uintptr_t)info->vq.L1_CB[0];
    uint16_t* W_out          = (uint16_t*)(uintptr_t)dst_L1_W;

    const uint32_t stride_bytes = info->N_tile * DATA_TYPE_BYTE; // row-major stride between rows
    const uint32_t groups_kc    = info->vq.K_tile_compressed;    // Kc rows

    for (uint32_t kc = 0; kc < groups_kc; ++kc) {
        const uint16_t* idx_row = idx_base + kc * info->N_tile; // one compressed row across N
        for (uint32_t n = 0; n < info->N_tile; ++n) {
            const uint16_t* idx_ptr = idx_row + n;
            uint16_t* out_ptr =
                W_out + (kc * VQ_GROUP_SIZE) * info->N_tile + n; // row-major: base of row (kc*8) at column n

            // VL=1 for index, payload=8 fp16 values → block gather then custom strided segment store
            asm volatile("vsetvli zero, %6, e16, m1, ta, ma\n"             
                         "vle16.v    v0, (%1)\n"                           // load index into v0
                         "mv         t0, %2\n"                             // codebook base
                         ".word      %3\n"                                // block gather -> v8 (centroid length 8)
                         "vsetvli    zero, %6, e16, m1, ta, ma\n"          // VL=1 segment (nf=8 → 8 elems)
                         "mv         t0, %5\n"                             // base address -> t0
                         "mv         t1, %4\n"                             // stride bytes -> t1
                         ".word      %7\n"                                // vssseg8e16.v v8, (t0), t1
                         :
                         : "r"(cb0_base),                                  // %0 (unused)
                           "r"(idx_ptr),                                   // %1
                           "r"(cb0_base),                                  // %2
                           "i"(VLBLK1EI16_V(RVV_V8, RVV_V0, RVX_T0, 1)),   // %3
                           "r"(stride_bytes),                              // %4
                           "r"(out_ptr),                                   // %5
                           "r"(1u),                                        // %6 (VL segments)
                           "i"(VSSSEG8E16_V(RVV_V8, RVX_T0, RVX_T1, 1))    // %7
                         : "t0", "t1", "v0", "v8", "memory");
        }
    }
}

#endif
#endif
