#ifndef _VQ_KERNELS_H_
#define _VQ_KERNELS_H_
#include "gemm_setup.h"
#include "spatz_rvv_extensions.h"

#if VQ_ENABLED == 1
static inline uint32_t vlmax_e16m8(void) {
    uint32_t vl;
    asm volatile("vsetvli %0, zero, e16, m8, ta, ma" : "=r"(vl));
    return vl;
}
// need to do additional check row_length * VQ_GROUP_SIZE <= VLMAX(e16, m8)
static inline void summa_vq_dequantize_tile(const SummaGEMMInfo* info, uint32_t dst_L1_W, int buffer_idx,
                                            uint32_t src_L1_Scales, int k_tile) {

    // Get base addresses for indices and codebooks
    const uint8_t* idx_cb0_base =
        (const uint8_t*)(uintptr_t)((buffer_idx == 0) ? info->vq.L1_IDX1[0] : info->vq.L1_IDX2[0]);
    const uint8_t* idx_cb1_base =
        (const uint8_t*)(uintptr_t)((buffer_idx == 0) ? info->vq.L1_IDX1[1] : info->vq.L1_IDX2[1]);
    const uint16_t* cb0_base = (const uint16_t*)(uintptr_t)info->vq.L1_CB[0];
    const uint16_t* cb1_base = (const uint16_t*)(uintptr_t)info->vq.L1_CB[1];
    const uint16_t* scales   = (const uint16_t*)(uintptr_t)src_L1_Scales;

    // Keep codebook base addresses pinned to the registers the custom VLBLK instruction consumes (t0/t1)
    uint16_t* W_out           = (uint16_t*)(uintptr_t)dst_L1_W;
    const uint32_t row_length = info->vq.N_tile_compressed; // Groups per row
                const uint32_t avl_elems = row_length * VQ_GROUP_SIZE;
            uint32_t vl_elems        = 0;

            // fp16
            asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl_elems) : "r"(avl_elems));
            uint32_t groups_this_iter = vl_elems / VQ_GROUP_SIZE;
            const uint32_t payload_elems = groups_this_iter * VQ_GROUP_SIZE;
    for (uint16_t r = 0; r < info->K_tile; ++r) {
        // Get row-specific pointers
        const uint8_t* idx_cb0_row = idx_cb0_base + r * row_length;
        const uint8_t* idx_cb1_row = idx_cb1_base + r * row_length;
        uint16_t* W_out_row        = W_out + r * info->vq.N_tile_compressed * VQ_GROUP_SIZE;
        // Load scale for this row (1 scale per K-dimension row)
        const uint16_t* scale_ptr = &scales[r];
            asm volatile("flh fa0, (%9)\n"
                        "vsetvli   zero, %4, e8,  m1, ta, ma\n"
                         "vle8.v    v0, (%0)\n"
                         "vle8.v    v1, (%1)\n"
                         "vsetvli   zero, %5, e16, m8, ta, ma\n"
                         "mv        t0, %2\n"
                         ".word     %6\n"
                         "vfmul.vf  v24, v8, fa0\n"
                         "mv        t1, %3\n"
                         ".word     %7\n"
                         "vfmacc.vf v24, fa0, v16\n"
                         "vse16.v   v24, (%8)\n"
                         :
                         : "r"(idx_cb0_row ),              // %0
                           "r"(idx_cb1_row ),              // %1
                           "r"(cb0_base),                                // %2
                           "r"(cb1_base),                                // %3
                           "r"(groups_this_iter),                        // %4
                           "r"(payload_elems),                           // %5
                           "i"(VLBLK1EI8_V(RVV_V8, RVV_V0, RVX_T0, 1)),  // %6
                           "i"(VLBLK1EI8_V(RVV_V16, RVV_V1, RVX_T1, 1)), // %7
                           "r"(W_out_row),                                  // %8
                           "r"(scale_ptr)                                //%9
                         : "t0", "t1", "v0", "v1", "v8", "v16", "v24","fa0", "memory");

    }
}

static inline void summa_vq_dequantize_tile_overhead(const SummaGEMMInfo* info, uint32_t dst_L1_W, int buffer_idx,
                                            uint32_t src_L1_Scales, int k_tile) {

    // Get base addresses for indices and codebooks
    const uint8_t* idx_cb0_base =
        (const uint8_t*)(uintptr_t)((buffer_idx == 0) ? info->vq.L1_IDX1[0] : info->vq.L1_IDX2[0]);
    const uint8_t* idx_cb1_base =
        (const uint8_t*)(uintptr_t)((buffer_idx == 0) ? info->vq.L1_IDX1[1] : info->vq.L1_IDX2[1]);
    const uint16_t* cb0_base = (const uint16_t*)(uintptr_t)info->vq.L1_CB[0];
    const uint16_t* cb1_base = (const uint16_t*)(uintptr_t)info->vq.L1_CB[1];
    const uint16_t* scales   = (const uint16_t*)(uintptr_t)src_L1_Scales;

    // Keep codebook base addresses pinned to the registers the custom VLBLK instruction consumes (t0/t1)

    uint16_t* W_out           = (uint16_t*)(uintptr_t)dst_L1_W;
    const uint32_t row_length = info->vq.N_tile_compressed; // Groups per row

    for (uint16_t r = 0; r < info->K_tile; ++r) {
        // Get row-specific pointers
        const uint8_t* idx_cb0_row = idx_cb0_base + r * row_length;
        const uint8_t* idx_cb1_row = idx_cb1_base + r * row_length;
        uint16_t* W_out_row        = W_out + r * info->vq.N_tile_compressed * VQ_GROUP_SIZE;

        // Load scale for this row (1 scale per K-dimension row)
        const uint16_t* scale_ptr = &scales[r];
        asm volatile("flh fa0, (%0)" ::"r"(scale_ptr) : "fa0", "memory");

        uint32_t remaining_groups = row_length;
        uint32_t group_offset     = 0;

        while (remaining_groups > 0) {
            const uint32_t avl_elems = remaining_groups * VQ_GROUP_SIZE;
            uint32_t vl_elems        = 0;
            // fp16
            asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl_elems) : "r"(avl_elems));
            uint32_t groups_this_iter = vl_elems / VQ_GROUP_SIZE;

 

            const uint32_t payload_elems = groups_this_iter * VQ_GROUP_SIZE;
            uint16_t* out_ptr            = W_out_row + group_offset * VQ_GROUP_SIZE;

            asm volatile("vsetvli   zero, %4, e8,  m1, ta, ma\n"
                         "vle8.v    v0, (%0)\n"
                         "vle8.v    v1, (%1)\n"
                         "vsetvli   zero, %5, e16, m8, ta, ma\n"
                         "mv        t0, %2\n"
                         ".word     %6\n"
                         "vfmul.vf  v24, v8, fa0\n"
                         "mv        t1, %3\n"
                         ".word     %7\n"
                         "vfmacc.vf v24, fa0, v16\n"
                         "vse16.v   v24, (%8)\n"
                         :
                         : "r"(idx_cb0_row + group_offset),              // %0
                           "r"(idx_cb1_row + group_offset),              // %1
                           "r"(cb0_base),                                // %2
                           "r"(cb1_base),                                // %3
                           "r"(groups_this_iter),                        // %4
                           "r"(payload_elems),                           // %5
                           "i"(VLBLK1EI8_V(RVV_V8, RVV_V0, RVX_T0, 1)),  // %6
                           "i"(VLBLK1EI8_V(RVV_V16, RVV_V1, RVX_T1, 1)), // %7
                           "r"(out_ptr)                                  // %8
                         : "t0", "t1", "v0", "v1", "v8", "v16", "v24", "memory");

            group_offset += groups_this_iter;
            remaining_groups -= groups_this_iter;
            // if(flex_get_cluster_id()==0 && r==3){
            //     printf("\ngroups this iter %d avl %d vl %d",groups_this_iter,vl_elems,remaining_groups);
            // }
        }
    }
}

#endif
#endif