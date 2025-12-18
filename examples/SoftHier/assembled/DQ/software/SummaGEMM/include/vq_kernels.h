#ifndef _VQ_KERNELS_H_
#define _VQ_KERNELS_H_
#include "flex_printf.h"
#include "gemm_setup.h"
#include "spatz_rvv_extensions.h"

#include <stdbool.h>

#if VQ_ENABLED == 1
// initally  we support horizontally sliced tiles
static inline void summa_partition_k_rows(uint32_t total_rows, uint32_t worker_rank, uint32_t worker_count,
                                          uint32_t* k_start, uint32_t* k_rows) {
    if (worker_count == 0 || worker_rank >= worker_count) {
        *k_start = 0;
        *k_rows  = 0;
        return;
    }
    uint32_t base = total_rows / worker_count;
    uint32_t rem  = total_rows % worker_count;
    *k_rows       = base + ((worker_rank < rem) ? 1u : 0u);
    *k_start      = worker_rank * base + ((worker_rank < rem) ? worker_rank : rem);
}

static inline uint32_t vlmax_e16m8(void) {
    uint32_t vl;
    asm volatile("vsetvli %0, zero, e16, m8, ta, ma" : "=r"(vl));
    return vl;
}

static inline void summa_fp16_zero(uint32_t dst_addr, uint32_t elems) {
    uint32_t remaining = elems;
    uint32_t dst       = dst_addr;
    while (remaining > 0) {
        uint32_t avl;
        asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(avl) : "r"(remaining));
        asm volatile("vmv.v.i v8, 0" ::: "v8");
        asm volatile("vse16.v v8, (%0)" ::"r"(dst) : "v8", "memory");
        remaining -= avl;
        dst += avl * sizeof(uint16_t);
    }
}

static inline void summa_fp16_copy(uint32_t dst_addr, uint32_t src_addr, uint32_t elems) {
    uint32_t remaining = elems;
    uint32_t dst       = dst_addr;
    uint32_t src       = src_addr;
    while (remaining > 0) {
        uint32_t avl;
        asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(avl) : "r"(remaining));
        asm volatile("vle16.v v8, (%0)" ::"r"(src) : "v8");
        asm volatile("vse16.v v8, (%0)" ::"r"(dst) : "v8", "memory");
        remaining -= avl;
        dst += avl * sizeof(uint16_t);
        src += avl * sizeof(uint16_t);
    }
}

static inline void summa_fp16_accumulate(uint32_t dst_addr, uint32_t src_addr, uint32_t elems) {
    uint32_t remaining = elems;
    uint32_t dst       = dst_addr;
    uint32_t src       = src_addr;
    while (remaining > 0) {
        uint32_t avl;
        asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(avl) : "r"(remaining));
        asm volatile("vle16.v v8, (%0)" ::"r"(dst) : "v8");
        asm volatile("vle16.v v16, (%0)" ::"r"(src) : "v16");
        asm volatile("vfadd.vv v8, v8, v16" ::: "v8");
        asm volatile("vse16.v v8, (%0)" ::"r"(dst) : "v8", "memory");
        remaining -= avl;
        dst += avl * sizeof(uint16_t);
        src += avl * sizeof(uint16_t);
    }
}

static inline void summa_vq_reduce_partials_fp16(const SummaGEMMInfo* info, uint32_t dst_L1_Z) {
#if defined(KERNEL_VARIANT_FUSED) && (KERNEL_VARIANT_FUSED == 1)
    const uint32_t partials = info->spatz_num;
    const uint32_t elems    = info->L1_Z_size / sizeof(uint16_t);
    uint32_t first_idx      = ARCH_SPATZ_ATTACED_CORES;
    for (uint32_t sid = 0; sid < partials; ++sid) {
        uint32_t k_start = 0, k_rows = 0;
        summa_partition_k_rows(info->K_tile, sid, partials, &k_start, &k_rows);
        if (k_rows > 0) {
            first_idx = sid;
            break;
        }
    }
    if (first_idx == ARCH_SPATZ_ATTACED_CORES) {
        summa_fp16_zero(dst_L1_Z, elems);
        return;
    }

    uint32_t base_src = info->L1_Z_partial[first_idx];
    if (base_src != dst_L1_Z) {
        summa_fp16_copy(dst_L1_Z, base_src, elems);
    }
    for (uint32_t sid = first_idx + 1; sid < partials; ++sid) {
        uint32_t k_start = 0, k_rows = 0;
        summa_partition_k_rows(info->K_tile, sid, partials, &k_start, &k_rows);
        if (k_rows == 0) {
            continue;
        }
        summa_fp16_accumulate(dst_L1_Z, info->L1_Z_partial[sid], elems);
    }
#else
    (void)info;
    (void)dst_L1_Z;
#endif
}

// need to do additional check row_length * VQ_GROUP_SIZE <= VLMAX(e16, m8)
/* AQLM Vector  Dequantization Algorithm
Equation : decoding 1 centroid group i  W_hat[i]= scale[row]* (cb1[idx1[i]] +cb2[idx2[i]])
W_hat[row]= concat( W_hat[i])
we have one scale per W_hat row.
for higher fpu utiilization we do the dataflow as follows
instead of ld a,b add c=a+b mul  d=s*c st d
we do ld a mul c=a*s ld b macc c+= b*s st c so we overlap load stroe units with  fpu units

*/
static inline void summa_vq_dequantize_tile(const SummaGEMMInfo* info, uint32_t dst_L1_W, int buffer_idx,
                                            uint32_t src_L1_Scales, int k_tile, uint32_t k_start_row,
                                            uint32_t k_rows) {

    if (k_rows == 0) {
        return;
    }
    (void)k_tile;
    // Get base addresses for indices and codebooks
    const uint8_t* idx_cb0_base = (const uint8_t*)(uintptr_t)((buffer_idx == 0) ? info->vq.L1_IDX1[0]
                                                                                : info->vq.L1_IDX2[0]);
    const uint8_t* idx_cb1_base = (const uint8_t*)(uintptr_t)((buffer_idx == 0) ? info->vq.L1_IDX1[1]
                                                                                : info->vq.L1_IDX2[1]);
    const uint16_t* cb0_base = (const uint16_t*)(uintptr_t)info->vq.L1_CB[0];
    const uint16_t* cb1_base = (const uint16_t*)(uintptr_t)info->vq.L1_CB[1];
    const uint16_t* scales   = (const uint16_t*)(uintptr_t)src_L1_Scales + k_start_row;

    // Keep codebook base addresses pinned to the registers the custom VLBLK instruction consumes (t0/t1)
    uint16_t* W_out           = (uint16_t*)(uintptr_t)dst_L1_W +
                      k_start_row * info->vq.N_tile_compressed * VQ_GROUP_SIZE;
    const uint32_t row_length = info->vq.N_tile_compressed; // Groups per row
    const uint32_t avl_elems  = row_length * VQ_GROUP_SIZE;
    uint32_t vl_elems         = 0;

    // fp16
    asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl_elems) : "r"(avl_elems));
    uint32_t groups_this_iter    = vl_elems / VQ_GROUP_SIZE;
    const uint32_t payload_elems = groups_this_iter * VQ_GROUP_SIZE;
    for (uint32_t r = 0; r < k_rows; ++r) {
        // Get row-specific pointers
        const uint8_t* idx_cb0_row = idx_cb0_base + (k_start_row + r) * row_length;
        const uint8_t* idx_cb1_row = idx_cb1_base + (k_start_row + r) * row_length;
        uint16_t* W_out_row        = W_out + r * info->vq.N_tile_compressed * VQ_GROUP_SIZE;
        // Load scale for this row (1 scale per K-dimension row)
        const uint16_t* scale_ptr = &scales[r];
        asm volatile(
            "flh fa0, (%9)\n"                       // load scale of row
            "vsetvli   zero, %4, e8,  m1, ta, ma\n" // set vl to to load  8-bit indices
            "vle8.v    v0, (%0)\n"                  // load idx matrix of cb1
            "vle8.v    v1, (%1)\n"                  // load idx matrix of cb2
            "vsetvli   zero, %5, e16, m8, ta, ma\n" // re-set vl to vector index block load contiguous centroid grouops
            "mv        t0, %2\n"                    // cb0 address
            ".word     %6\n"                        // vector index block load idx0 centroids
            "vfmul.vf  v24, v8, fa0\n"              // c= a*scale
            "mv        t1, %3\n"                    // cb1 base address
            ".word     %7\n"                        // vector index block load idx1 centroids
            "vfmacc.vf v24, fa0, v16\n"             // c+= b* scalse
            "vse16.v   v24, (%8)\n"                 // store b
            :
            : "r"(idx_cb0_row),                             // %0
              "r"(idx_cb1_row),                             // %1
              "r"(cb0_base),                                // %2
              "r"(cb1_base),                                // %3
              "r"(groups_this_iter),                        // %4
              "r"(payload_elems),                           // %5
              "i"(VLBLK1EI8_V(RVV_V8, RVV_V0, RVX_T0, 1)),  // %6
              "i"(VLBLK1EI8_V(RVV_V16, RVV_V1, RVX_T1, 1)), // %7
              "r"(W_out_row),                               // %8
              "r"(scale_ptr)                                //%9
            : "t0", "t1", "v0", "v1", "v8", "v16", "v24", "fa0", "memory");
    }
}

// summa_vq_dequantize_tile but here we fuse the dequantization with computation to avoid multiple load/stores as well
// as sync barriers (compute is already gustavson outer product algorithm) compute from: spatz_gemv_fp16_full_legacy
// TODO , add another fused dequantizer with support in the case that the row cannot be dequantized withinan iteration
// idea: adhjust accumulator Z store
static inline void summa_vq_dequantize_tile_fused(const SummaGEMMInfo* info, uint32_t dst_L1_Z, int buffer_idx,
                                                  uint32_t src_L1_Scales, uint32_t src_L1_x, int k_tile,
                                                  uint32_t k_start_row, uint32_t k_rows) {
    if (k_rows == 0) {
        return;
    }
    // Fused path: dequantize and accumulate directly into dst_L1_Z with a single store per row.
    const uint8_t* idx_cb0_base =
        (const uint8_t*)(uintptr_t)((buffer_idx == 0) ? info->vq.L1_IDX1[0] : info->vq.L1_IDX2[0]);
    const uint8_t* idx_cb1_base =
        (const uint8_t*)(uintptr_t)((buffer_idx == 0) ? info->vq.L1_IDX1[1] : info->vq.L1_IDX2[1]);
    const uint16_t* cb0_base = (const uint16_t*)(uintptr_t)info->vq.L1_CB[0];
    const uint16_t* cb1_base = (const uint16_t*)(uintptr_t)info->vq.L1_CB[1];

    const uint16_t* scales = (const uint16_t*)(uintptr_t)src_L1_Scales + k_start_row;
    const uint16_t* x_ptr  = (const uint16_t*)(uintptr_t)src_L1_x + k_start_row;
    uint16_t* C_base       = (uint16_t*)(uintptr_t)dst_L1_Z;

    const uint32_t row_length    = info->vq.N_tile_compressed; // groups per K row
    const uint32_t payload_elems = row_length * VQ_GROUP_SIZE; // total elements per row

    for (uint32_t k = 0; k < k_rows; ++k) {
        const uint32_t k_global    = k_start_row + k;
        const uint8_t* idx_cb0_row = idx_cb0_base + k_global * row_length;
        const uint8_t* idx_cb1_row = idx_cb1_base + k_global * row_length;
        const uint16_t* scale_ptr  = &scales[k];
        const uint16_t* a_ptr      = &x_ptr[k]; // x is a vector (length K_tile)

        // Determine whether to keep previous output (previous K or previous tile) or start fresh
        const bool use_acc = (k_tile > 0) || (k > 0);

        // Set VL for the full row payload
        uint32_t vl_elems = 0;
        asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl_elems) : "r"(payload_elems));
        uint32_t groups_this_iter = vl_elems / VQ_GROUP_SIZE;

        // Initialize accumulator
        if (use_acc) {
            asm volatile("vsetvli zero, %0, e16, m8, ta, ma\n"
                         "vle16.v v24, (%1)\n"
                         :
                         : "r"(payload_elems), "r"(C_base)
                         : "v24", "memory");
        } else {
            asm volatile("vmv.v.i v24, 0" ::: "v24");
        }

        // Dequantize and accumulate
        asm volatile("flh       fa0, (%6)\n"                 // scale[k]
                     "flh       fa1, (%7)\n"                 // x[k]
                     "fmul.h    fa0, fa0, fa1\n"             // a_scaled = scale * x (half precision)
                     "vsetvli   zero, %2, e8,  m1, ta, ma\n" // set vl for idx load (groups)
                     "vle8.v    v0, (%0)\n"                  // idx cb0
                     "vle8.v    v1, (%1)\n"                  // idx cb1
                     "vsetvli   zero, %3, e16, m8, ta, ma\n" // set vl for payload elems
                     "mv        t0, %4\n"
                     ".word     %5\n" // load cb0 centroids -> v8
                     "mv        t1, %8\n"
                     ".word     %9\n" // load cb1 centroids -> v16
                     "vfmacc.vf v24, fa0, v8\n"
                     "vfmacc.vf v24, fa0, v16\n"
                     :
                     : "r"(idx_cb0_row),                            // %0
                       "r"(idx_cb1_row),                            // %1
                       "r"(groups_this_iter),                       // %2
                       "r"(payload_elems),                          // %3
                       "r"(cb0_base),                               // %4
                       "i"(VLBLK1EI8_V(RVV_V8, RVV_V0, RVX_T0, 1)), // %5
                       "r"(scale_ptr),                              // %6
                       "r"(a_ptr),                                  // %7
                       "r"(cb1_base),                               // %8
                       "i"(VLBLK1EI8_V(RVV_V16, RVV_V1, RVX_T1, 1)) // %9
                     : "t0", "t1", "v0", "v1", "v8", "v16", "v24", "fa0", "fa1", "memory");

        // Store once per row after full accumulation
        asm volatile("vsetvli zero, %0, e16, m8, ta, ma\n"
                     "vse16.v v24, (%1)\n"
                     :
                     : "r"(payload_elems), "r"(C_base)
                     : "memory");
    }
}

static inline void summa_vq_dequantize_tile_overhead(const SummaGEMMInfo* info, uint32_t dst_L1_W, int buffer_idx,
                                                     uint32_t src_L1_Scales, int k_tile, uint32_t k_start_row,
                                                     uint32_t k_rows) {

    if (k_rows == 0) {
        return;
    }
    (void)k_tile;
    // Get base addresses for indices and codebooks
    const uint8_t* idx_cb0_base = (const uint8_t*)(uintptr_t)((buffer_idx == 0) ? info->vq.L1_IDX1[0]
                                                                                : info->vq.L1_IDX2[0]);
    const uint8_t* idx_cb1_base = (const uint8_t*)(uintptr_t)((buffer_idx == 0) ? info->vq.L1_IDX1[1]
                                                                                : info->vq.L1_IDX2[1]);
    const uint16_t* cb0_base = (const uint16_t*)(uintptr_t)info->vq.L1_CB[0];
    const uint16_t* cb1_base = (const uint16_t*)(uintptr_t)info->vq.L1_CB[1];
    const uint16_t* scales   = (const uint16_t*)(uintptr_t)src_L1_Scales + k_start_row;

    // Keep codebook base addresses pinned to the registers the custom VLBLK instruction consumes (t0/t1)

    uint16_t* W_out           = (uint16_t*)(uintptr_t)dst_L1_W +
                      k_start_row * info->vq.N_tile_compressed * VQ_GROUP_SIZE;
    const uint32_t row_length = info->vq.N_tile_compressed; // Groups per row

    for (uint32_t r = 0; r < k_rows; ++r) {
        // Get row-specific pointers
        const uint8_t* idx_cb0_row = idx_cb0_base + (k_start_row + r) * row_length;
        const uint8_t* idx_cb1_row = idx_cb1_base + (k_start_row + r) * row_length;
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
