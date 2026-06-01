#ifndef _VQ_KERNELS_GPTVQ_H_
#define _VQ_KERNELS_GPTVQ_H_

#include "flex_printf.h"
#include "gemm_setup.h"
#include "spatz_rvv_extensions.h"

#if VQ_ENABLED == 1

static inline uint32_t summa_vq_gptvq_payload_vlmax(void) {
    uint32_t vl;
    asm volatile("vsetvli %0, zero, e16, m8, ta, ma" : "=r"(vl));
    return vl;
}

static inline void summa_vq_dequantize_tile_gptvq(const SummaGEMMInfo* info, uint32_t dst_L1_W, int buffer_idx,
                                                  uint32_t src_L1_Scales, int k_tile, uint32_t k_start_row,
                                                  uint32_t k_rows) {
    (void)src_L1_Scales;
    (void)k_tile;
    (void)k_start_row;
    (void)k_rows;

    const uint8_t* idx_base =
        (const uint8_t*)(uintptr_t)((buffer_idx == 0) ? info->vq.L1_IDX1[0] : info->vq.L1_IDX2[0]);
    const uint16_t* cb_base =
        (const uint16_t*)(uintptr_t)((buffer_idx == 0) ? info->vq.L1_CB1[0] : info->vq.L1_CB2[0]);
    uint16_t* W_out = (uint16_t*)(uintptr_t)dst_L1_W;

    const uint32_t groups_per_column = info->vq.K_tile_compressed;
    const uint32_t max_groups_per_iter = summa_vq_gptvq_payload_vlmax() / VQ_GROUP_SIZE;

    if (max_groups_per_iter == 0) {
        return;
    }

    for (uint32_t n = 0; n < info->N_tile; ++n) {
        const uint8_t* idx_ptr = idx_base + n * groups_per_column;
        uint16_t* out_ptr      = W_out + n * info->K_tile;

        uint32_t remaining_groups = groups_per_column;
        while (remaining_groups > 0) {
            const uint32_t request_groups =
                (remaining_groups < max_groups_per_iter) ? remaining_groups : max_groups_per_iter;
            uint32_t groups_this_iter = 0;
            asm volatile("vsetvli %0, %1, e8, m1, ta, ma" : "=r"(groups_this_iter) : "r"(request_groups));
            const uint32_t payload_elems = groups_this_iter * VQ_GROUP_SIZE;

            asm volatile("vle8.v    v0, (%[idx])\n"
                         "vsetvli zero, %[payload], e16, m8, ta, ma\n"
                         "mv        t0, %[cb]\n"
                         ".word     %[gather]\n"
                         "vse16.v   v8, (%[out])\n"
                         :
                         : [idx] "r"(idx_ptr),
                           [cb] "r"(cb_base),
                           [gather] "i"(VLBLK4EI8_V(RVV_V8, RVV_V0, RVX_T0, 1)),
                           [payload] "r"(payload_elems),
                           [out] "r"(out_ptr)
                         : "t0", "v0", "v8", "memory");

            idx_ptr += groups_this_iter;
            out_ptr += payload_elems;
            remaining_groups -= groups_this_iter;
        }
    }
}

static inline void summa_vq_dequantize_tile_gptvq_baseline(const SummaGEMMInfo* info, uint32_t dst_L1_W,
                                                           int buffer_idx, uint32_t src_L1_Scales, int k_tile,
                                                           uint32_t k_start_row, uint32_t k_rows) {
    (void)src_L1_Scales;
    (void)k_tile;
    (void)k_start_row;
    (void)k_rows;

    const uint8_t* idx_base =
        (const uint8_t*)(uintptr_t)((buffer_idx == 0) ? info->vq.L1_IDX1[0] : info->vq.L1_IDX2[0]);
    const uint16_t* cb_base =
        (const uint16_t*)(uintptr_t)((buffer_idx == 0) ? info->vq.L1_CB1[0] : info->vq.L1_CB2[0]);
    uint16_t* W_out = (uint16_t*)(uintptr_t)dst_L1_W;

    const uint32_t groups_per_column = info->vq.K_tile_compressed;
    const uint32_t group_elems = VQ_GROUP_SIZE;

    asm volatile("vsetvli zero, %0, e16, m1, ta, ma" ::"r"(group_elems));

    for (uint32_t n = 0; n < info->N_tile; ++n) {
        const uint8_t* idx_ptr = idx_base + n * groups_per_column;
        uint16_t* out_ptr      = W_out + n * info->K_tile;

        for (uint32_t kc = 0; kc < groups_per_column; ++kc) {
            const uint16_t* centroid = cb_base + (uint32_t)idx_ptr[kc] * VQ_GROUP_SIZE;
            asm volatile("vle16.v  v8, (%[cb])\n"
                         "vse16.v  v8, (%[out])\n"
                         :
                         : [cb] "r"(centroid), [out] "r"(out_ptr)
                         : "v8", "memory");
            out_ptr += VQ_GROUP_SIZE;
        }
    }
}

#endif
#endif
