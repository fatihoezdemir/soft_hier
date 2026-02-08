#ifndef DQ_HELPERS_H
#define DQ_HELPERS_H

#include "debug.h"
#include "dq_data_hbm.h"
#include "flex_runtime.h"
#include "spatz_rvv_extensions.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
extern const int SPATZ_CORE;
// TODO not valid for multicluster scenario, maybe create array of clusters?
typedef struct {
    volatile uint16_t* cb;         // codebook base in L1
    volatile uint16_t* scales;     // scales in L1
    volatile uint16_t* indices;    // indices in L1
    volatile uint8_t* indices0_u8; // indices in L1
    volatile uint8_t* indices_u8;  // indices in L1

    volatile uint8_t* indices1_u8;  // indices in L1
    volatile uint16_t* W_dq_buf[2]; // optional double buffer for dequantized weights
    volatile uint16_t* W_dq;        // dequantized weights tile in L1
} L1_DQ_Handles;
extern L1_DQ_Handles g_l1_dq;

void dequant_group_debug(const uint16_t* a /*cb0[idx0[i]]*/, const uint16_t* b /*cb1[idx1[i]]*/, const uint16_t* scale,
                         uint16_t* out) {
    asm volatile("vsetvli zero, %0, e16, m1, ta, ma" ::"r"(8u));
    asm volatile("vle16.v v8, (%0)" ::"r"(a) : "v8", "memory"); // cb0[idx0]
    asm volatile("vle16.v v9, (%0)" ::"r"(b) : "v9", "memory"); // cb1[idx1]
    asm volatile("lhu t0, (%0)" ::"r"(scale) : "t0", "memory"); // Load scale and broadcast to vector
    asm volatile("vmv.v.x v12, t0" ::: "t0", "v12");
    asm volatile("vfadd.vv v10, v8, v9" ::: "v8", "v9", "v10");     // Add codebook entries
    asm volatile("vfmul.vv v11, v10, v12" ::: "v10", "v12", "v11"); // Multiply by scale
    asm volatile("vse16.v v11, (%0)" ::"r"(out) : "v11", "memory"); // Store result
}

//        "lhu      t0, (%[scale])\n\t"          "vmv.v.x  v2, t0\n\t"
//                  "vfmul.vv v0, v0, v2\n\t"  // v0 = a * scale   "vfmacc.vv v0, v2, v1\n\t" // v0 += scale * b
//                  "vse16.v  v0, (%[out])\n\t"

void dequant_groupmacc(const uint16_t* a /*cb0[idx0[i]]*/, const uint16_t* b /*cb1[idx1[i]]*/, const uint16_t* scale,
                       uint16_t* out) {
    asm volatile("vsetvli  zero, %[g], e16, m1, ta, ma\n\t"
                 "vle16.v  v0, (%[a])\n\t"
                 "vle16.v  v1, (%[b])\n\t"
                 "flh      fa0, (%[s])\n\t"  // Load fp16 scale with NaN-boxing
                 "vfmul.vf v2, v0, fa0\n\t"  // v2 = a * s
                 "vfmacc.vf v2, fa0, v1\n\t" // v2 += b * s  => (a+b)*s
                 "vse16.v  v2, (%[out])\n\t" // store
                 :
                 : [a] "r"(a), [b] "r"(b), [s] "r"(scale), [out] "r"(out), [g] "r"(8u)
                 : "v0", "v1", "v2", "fa0");
}
void dequant_group(const uint16_t* a /** cb0[idx0[i]] */, const uint16_t* b, /** cb1[idx1[i]] */ const uint16_t* scale,
                   uint16_t* out) {
    asm volatile("vsetvli zero, %[g], e16, m1, ta, ma\n\t"
                 // "vlseg8e8.v vd, (rs1), vm"
                 "vmv.v.i v2, 0\n\t"
                 "vle16.v v0, (%[a])\n\t"
                 "vle16.v v1, (%[b])\n\t"
                 "flw fa0, (%[s])\n\t"      // f0 = scale (fp16)
                 "vfadd.vv v2, v0, v1\n\t"  // v2 = a * s
                 "vfmul.vf v3, v2, fa0\n\t" // v2 += b * s => (a+b)s
                 "vse16.v v3, (%[out])\n\t" // store

                 :
                 : [a] "r"(a), [b] "r"(b), [s] "r"(scale), [out] "r"(out), [g] "r"(8u)
                 : "v0", "v1", "v2", "fa0");
}
void dequant_group_legacy(const uint16_t* a /*cb0[idx0[i]]*/, const uint16_t* b /*cb1[idx1[i]]*/, const uint16_t* scale,
                          uint16_t* out) {
    const uint32_t group_elems = VQ_GROUP_SIZE;
    // FINALUse vector-vector multiply with scale broadcast via integer register
    asm volatile("vsetvli zero, %[vl], e16, m1, ta, ma\n\t"
                 "vle16.v  v0, (%[a])\n\t" // cb0[idx0]
                 "vle16.v  v1, (%[b])\n\t" // cb1[idx1]
                 "lhu      t0, (%[scale])\n\t"
                 "vmv.v.x  v2, t0\n\t"
                 "vfadd.vv v3, v0, v1\n\t" // sum
                 "vfmul.vv v3, v3, v2\n\t" // multiply by scale
                 "vse16.v  v3, (%[out])\n\t"
                 :
                 : [a] "r"(a), [b] "r"(b), [out] "r"(out), [scale] "r"(scale), [vl] "r"(group_elems)
                 : "t0", "v0", "v1", "v2", "v3", "memory");
}

// Load scale once per row and reuse across all groups in the tile.
void dequantize_block_tile_compact(uint16_t row_start, uint16_t rows,
                                   uint16_t group_count,       // groups in this tile
                                   uint16_t idx_groups_stride) // should equal group_count for compactstorage
{

    uint16_t* W_tile       = (uint16_t*)(uintptr_t)g_l1_dq.W_dq;          // COMPACT tile buffer
    const uint16_t* idx    = (const uint16_t*)(uintptr_t)g_l1_dq.indices; // COMPACT indices
    const uint16_t* scales = (const uint16_t*)(uintptr_t)g_l1_dq.scales;
    const uint16_t* cb0    = (const uint16_t*)(uintptr_t)g_l1_dq.cb;
    const uint16_t* cb1    = cb0 + VQ_CB_NUM_CENTROIDS * VQ_GROUP_SIZE;

    const uint16_t row_end = row_start + rows;
    const uint32_t tile_P  = group_count * VQ_GROUP_SIZE;

    // printf("[DEBUG][DQ] compact tile dequant: rows [%u..%u), groups=%u, tile_P=%u\n\t", row_start, row_end,
    //        group_count, tile_P);

    // flex_timer_start(); // puttingtimer here doesntchange theruntime

    asm volatile("vsetvli zero, %0, e16, m1, ta, ma" ::"r"((uint32_t)VQ_GROUP_SIZE));
    for (uint16_t r = row_start; r < row_end; ++r) {

        // Indices for row r in the compact tile
        const uint8_t* p_idx = (const uint8_t*)idx + 2u * r * idx_groups_stride;

        // Output row r in the COMPACT W_dq tile
        uint16_t* p_out = W_tile + r * tile_P;

        const uint16_t* scale_ptr = &scales[r];
        for (uint16_t g = 0; g < group_count; ++g) {
            const uint8_t idx0 = p_idx[0];
            const uint8_t idx1 = p_idx[1];
            p_idx += 2;

            const uint16_t* a = cb0 + (unsigned)idx0 * VQ_GROUP_SIZE;
            const uint16_t* b = cb1 + (unsigned)idx1 * VQ_GROUP_SIZE;
            // dequant_group_legacy(a, b, scale_ptr, p_out);
            dequant_group(a, b, scale_ptr, p_out);
            // dequant_groupmacc(a, b, scale_ptr, p_out);
            p_out += VQ_GROUP_SIZE;
        }
    }
}
void dequantize_block_tile_compactu8(uint16_t row_start, uint16_t rows,
                                     uint16_t group_count,       // groups in this tile
                                     uint16_t idx_groups_stride) // should equal group_count for compactstorage
{
    uint16_t* W_tile = (uint16_t*)(uintptr_t)g_l1_dq.W_dq; // COMPACT tile buffer
    // const uint16_t* idx    = (const uint16_t*)(uintptr_t)g_l1_dq.indices; // COMPACT indices
    const uint8_t* idx0 = (const uint8_t*)(uintptr_t)g_l1_dq.indices0_u8; // COMPACT indices
    const uint8_t* idx1 = (const uint8_t*)(uintptr_t)g_l1_dq.indices1_u8; // COMPACT indices

    const uint16_t* scales = (const uint16_t*)(uintptr_t)g_l1_dq.scales;
    const uint16_t* cb0    = (const uint16_t*)(uintptr_t)g_l1_dq.cb;
    const uint16_t* cb1    = cb0 + VQ_CB_NUM_CENTROIDS * VQ_GROUP_SIZE;

    const uint16_t row_end = row_start + rows;
    const uint32_t tile_P  = group_count * VQ_GROUP_SIZE;

    // printf("[DEBUG][DQ] compact tile dequant: rows [%u..%u), groups=%u, tile_P=%u\n\t", row_start, row_end,
    //        group_count, tile_P);

    // flex_timer_start(); // puttingtimer here doesntchange theruntime

    asm volatile("vsetvli zero, %0, e16, m1, ta, ma" ::"r"((uint32_t)VQ_GROUP_SIZE));
    for (uint16_t r = row_start; r < row_end; ++r) {

        // Indices for row r in the compact tile
        // const uint8_t* p_idx = (const uint8_t*)idx + 2u * r * idx_groups_stride;
        const uint8_t* p_idx0 = (const uint8_t*)idx0 + 1u * r * idx_groups_stride;
        const uint8_t* p_idx1 = (const uint8_t*)idx1 + 1u * r * idx_groups_stride;

        // Output row r in the COMPACT W_dq tile
        uint16_t* p_out = W_tile + r * tile_P;

        const uint16_t* scale_ptr = &scales[r];
        for (uint16_t g = 0; g < group_count; ++g) {

            const uint16_t* a = cb0 + (unsigned)idx0 * VQ_GROUP_SIZE;
            const uint16_t* b = cb1 + (unsigned)idx1 * VQ_GROUP_SIZE;
            // dequant_group_legacy(a, b, scale_ptr, p_out);
            dequant_group(a, b, scale_ptr, p_out);
            // dequant_groupmacc(a, b, scale_ptr, p_out);
            p_out += VQ_GROUP_SIZE;
        }
    }
    // flex_timer_end();
}
void dequant_groupmacc_improved(const uint8_t* idx0, const uint8_t* idx1, const uint16_t* cb0_ptr,
                                const uint16_t* cb1_ptr, const uint16_t* scale, uint16_t* out, uint16_t groups_total) {
    if (groups_total == 0) {
        return;
    }

    uint32_t remaining_groups = groups_total;
    uint32_t group_offset     = 0;
    asm volatile("mv t0, %0" : : "r"(cb0_ptr) : "t0");
    asm volatile("mv t1, %0" : : "r"(cb1_ptr) : "t1");
    asm volatile("flh fa0, (%0)" ::"r"(scale) : "fa0", "memory");
    while (remaining_groups > 0) {
        const uint32_t avl_elems = (uint32_t)remaining_groups * VQ_GROUP_SIZE;
        uint32_t vl_elems        = 0;
        asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl_elems) : "r"(avl_elems));

        uint32_t groups_this_iter = vl_elems / VQ_GROUP_SIZE;
        if (groups_this_iter == 0u) {
            groups_this_iter = 1u;
            asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl_elems) : "r"((uint32_t)VQ_GROUP_SIZE));
        }

        const uint8_t* idx0_ptr = idx0 + group_offset;
        const uint8_t* idx1_ptr = idx1 + group_offset;
        uint16_t* out_ptr       = out + (size_t)group_offset * VQ_GROUP_SIZE;

        uint32_t vl_idx = 0;
        (void)vl_idx;
        const uint32_t payload_elems = groups_this_iter * VQ_GROUP_SIZE;
        asm volatile("vsetvli %0, %1, e8, m1, ta, ma" : "=r"(vl_idx) : "r"(groups_this_iter));
        asm volatile("vle8.v v0, (%0)" ::"r"(idx0_ptr) :);
        asm volatile("vle8.v v1, (%0)" ::"r"(idx1_ptr) :);

        asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl_elems) : "r"(payload_elems));

        EMIT_WORD_IMM(VLBLK1EI8_V(RVV_V8, RVV_V0, RVX_T0, 1));
        asm volatile("vfmul.vf v24, v8, fa0");
        EMIT_WORD_IMM(VLBLK1EI8_V(RVV_V16, RVV_V1, RVX_T1, 1));

        asm volatile("vfmacc.vf v24, fa0, v16\n\t"
                     "vse16.v v24, (%0)"
                     :
                     : "r"(out_ptr)
                     : "memory", "v24");

        remaining_groups -= groups_this_iter;
        group_offset += groups_this_iter;
    }
}

void dequantize_block_tile_compact_fused_gemvs(uint16_t row_start, uint16_t rows, uint16_t group_count,
                                               uint16_t idx_groups_stride, const uint16_t* x_vec, uint16_t* y_out) {
    if (flex_get_core_id() == 0 && flex_get_cluster_id() == 0) {
        const uint8_t* idx0 = (const uint8_t*)(uintptr_t)g_l1_dq.indices0_u8;
        const uint8_t* idx1 = (const uint8_t*)(uintptr_t)g_l1_dq.indices1_u8;

        const uint16_t* scales = (const uint16_t*)(uintptr_t)g_l1_dq.scales;
        const uint16_t* cb0    = (const uint16_t*)(uintptr_t)g_l1_dq.cb;
        const uint16_t* cb1    = cb0 + VQ_CB_NUM_CENTROIDS * VQ_GROUP_SIZE;

        const uint16_t row_end          = row_start + rows;
        const uint16_t row_length_elems = VQ_NUM_GROUPS_PER_ROW * VQ_GROUP_SIZE;
        const uint32_t tile_P           = group_count * VQ_GROUP_SIZE;
        uint8_t accum_started           = 0;
        for (uint16_t r = row_start; r < row_end; ++r) {

            const uint8_t* row_idx0 = (const uint8_t*)idx0 + 1u * r * idx_groups_stride;
            const uint8_t* row_idx1 = (const uint8_t*)idx1 + 1u * r * idx_groups_stride;

            const uint16_t* scale_ptr     = &scales[r];
            const uint16_t* x_ptr         = &x_vec[r];
            const uint16_t scale_res      = (uint16_t)(x_vec[r] * scales[r]);
            const uint16_t* scale_resaddr = &scale_res;
            debug("\nscale 0x%04x", scale_res);

            // Load scale and x into FP registers

            asm volatile("flh fa0, (%0)" ::"r"(scale_resaddr) : "fa0", "memory");
            // asm volatile("flw fa1, (%0)" ::"r"(x_ptr) : "fa1", "memory");
            asm volatile("vmv.v.x v24, t0\n\t" ::);

            // uint32_t remaining_groups = group_count;
            // uint32_t group_offset     = 0;

            const uint32_t avl_elems = (uint32_t)VQ_NUM_GROUPS_PER_ROW * VQ_GROUP_SIZE;
            uint32_t vl_elems        = 0;
            asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl_elems) : "r"(avl_elems));

            uint32_t groups_this_iter = vl_elems / VQ_GROUP_SIZE;
            if (groups_this_iter == 0u) {
                groups_this_iter = 1u;
                asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl_elems) : "r"((uint32_t)VQ_GROUP_SIZE));
            }

            const uint8_t* idx0_ptr = row_idx0;
            const uint8_t* idx1_ptr = row_idx1;
            uint16_t* y_ptr         = y_out;

            uint32_t vl_idx = 0;
            asm volatile("vsetvli %0, %1, e8, m1, ta, ma" : "=r"(vl_idx) : "r"(groups_this_iter));
            (void)vl_idx;
            asm volatile("vle8.v v0, (%0)" ::"r"(idx0_ptr) : "memory");
            asm volatile("vle8.v v1, (%0)" ::"r"(idx1_ptr) : "memory");

            const uint32_t payload_elems = groups_this_iter * VQ_GROUP_SIZE;
            asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl_elems) : "r"(payload_elems));
            if (accum_started != 0) {
                asm volatile("vfmacc.vf v24, fa0, v8");
            }
            if (accum_started == 0) { // if this is the start, we just do vfmul then we
                asm volatile("vmv.v.i v24, 0");

                EMIT_WORD_IMM(VLBLK1EI8_V(RVV_V8, RVV_V0, RVX_T0, 1));
                asm volatile("vfmul.vf v24, v8, fa0");
                EMIT_WORD_IMM(VLBLK1EI8_V(RVV_V16, RVV_V1, RVX_T1, 1));
                asm volatile("vfmacc.vf v24, fa0, v16");
                ++accum_started;
            } else {
                EMIT_WORD_IMM(VLBLK1EI8_V(RVV_V8, RVV_V0, RVX_T0, 1));
                asm volatile("vfmacc.vf v24, fa0, v8");
                EMIT_WORD_IMM(VLBLK1EI8_V(RVV_V16, RVV_V1, RVX_T1, 1));
                asm volatile("vfmacc.vf v24, fa0, v16");
                ++accum_started;
            }

            // asm volatile("vle16.v v20, (%0)" ::"r"(y_ptr) : "memory");

            //     remaining_groups -= groups_this_iter;
            //     group_offset += groups_this_iter;
            // }
        }
        asm volatile("vse16.v v24, (%0)" ::"r"(y_out) : "memory");
    }
}

void dequant_groupmacc_improvedfused(const uint8_t* idx0, const uint8_t* idx1, const uint16_t* cb0_ptr,
                                     const uint16_t* cb1_ptr, const uint16_t* x_vec, const uint16_t* scale,
                                     uint16_t* out, uint16_t groups_total, uint16_t r) {
    if (groups_total == 0) {
        return;
    }

    uint32_t remaining_groups = groups_total;
    uint32_t group_offset     = 0;
    asm volatile("mv t0, %0" : : "r"(cb0_ptr) : "t0");
    asm volatile("mv t1, %0" : : "r"(cb1_ptr) : "t1");
    // asm volatile("flw fa0, (%0)" ::"r"(scale) : "fa0", "memory");
    asm volatile("flh fa0, (%0)\n"        // fa0 = fp16 scale[r]
                 "flh fa1, (%1)\n"        // fa1 = fp16 x[r]
                 "fmul.h fa2, fa0, fa1\n" // fa2 = scale * x (fp16 multiply)    // store result as raw fp16
                 :
                 : "r"(scale), "r"(x_vec)
                 : "fa0", "fa1", "fa2", "memory");

    while (remaining_groups > 0) {
        const uint32_t avl_elems = (uint32_t)remaining_groups * VQ_GROUP_SIZE;
        uint32_t vl_elems        = 0;
        asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl_elems) : "r"(avl_elems));

        uint32_t groups_this_iter = vl_elems / VQ_GROUP_SIZE;
        if (groups_this_iter == 0u) {
            groups_this_iter = 1u;
            asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl_elems) : "r"((uint32_t)VQ_GROUP_SIZE));
        }

        const uint8_t* idx0_ptr = idx0 + group_offset;
        const uint8_t* idx1_ptr = idx1 + group_offset;
        uint16_t* out_ptr       = out + (size_t)group_offset * VQ_GROUP_SIZE;

        uint32_t vl_idx = 0;
        asm volatile("vsetvli %0, %1, e8, m1, ta, ma" : "=r"(vl_idx) : "r"(groups_this_iter));
        (void)vl_idx;
        asm volatile("vle8.v v0, (%0)" ::"r"(idx0_ptr) : "memory");

        asm volatile("vle8.v v1, (%0)" ::"r"(idx1_ptr) : "memory");

        const uint32_t payload_elems = groups_this_iter * VQ_GROUP_SIZE;

        asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl_elems) : "r"(payload_elems));
        if (r == 0) {
            EMIT_WORD_IMM(VLBLK1EI8_V(RVV_V8, RVV_V0, RVX_T0, 1));
            asm volatile("vfmul.vf v24, v8, fa2");
            EMIT_WORD_IMM(VLBLK1EI8_V(RVV_V16, RVV_V1, RVX_T1, 1));

            asm volatile("vfmacc.vf v24, fa2, v16");
            // asm volatile("vfadd.vv v24, v8, v16");
            // asm volatile("vfmul.vf v24, v24, fa0");
        } else {
            EMIT_WORD_IMM(VLBLK1EI8_V(RVV_V8, RVV_V0, RVX_T0, 1));
            asm volatile("vfmacc.vf v24, fa2, v8");
            EMIT_WORD_IMM(VLBLK1EI8_V(RVV_V16, RVV_V1, RVX_T1, 1));

            asm volatile("vfmacc.vf v24, fa2, v16");
        }
        asm volatile("vse16.v v24, (%0)" ::"r"(out_ptr) : "memory");

        remaining_groups -= groups_this_iter;
        group_offset += groups_this_iter;
    }
}

void dequantize_block_tile_compact_improvedfused(uint16_t row_start, uint16_t rows,
                                                 uint16_t group_count, // groups in this tile
                                                 uint16_t idx_groups_stride, uint16_t* x_vec,
                                                 uint16_t* y_out) // should equal group_count for compactstorage
{
    if (flex_get_core_id() == 0 && flex_get_cluster_id() == 0) {
        uint16_t* W_tile    = (uint16_t*)(uintptr_t)g_l1_dq.W_dq;             // COMPACT tile buffer
        const uint8_t* idx0 = (const uint8_t*)(uintptr_t)g_l1_dq.indices0_u8; // COMPACT indices
        const uint8_t* idx1 = (const uint8_t*)(uintptr_t)g_l1_dq.indices1_u8; // COMPACT indices

        const uint16_t* scales = (const uint16_t*)(uintptr_t)g_l1_dq.scales;
        const uint16_t* cb0    = (const uint16_t*)(uintptr_t)g_l1_dq.cb;
        const uint16_t* cb1    = cb0 + VQ_CB_NUM_CENTROIDS * VQ_GROUP_SIZE;

        const uint16_t row_end = row_start + rows;
        const uint32_t tile_P  = group_count * VQ_GROUP_SIZE;

        // printf("[DEBUG][DQ] compact tile dequant: rows [%u..%u), groups=%u, tile_P=%u\n\t", row_start, row_end,
        //        group_count, tile_P);

        flex_timer_start(); // puttingtimer here doesntchange theruntime

        // asm volatile("vsetvli zero, %0, e16, m1, ta, ma" ::"r"((uint32_t)VQ_GROUP_SIZE));no group size , the best is
        // if we load as much ass possible
        for (uint16_t r = row_start; r < row_end; ++r) {

            // Indices for row r in the compact tile
            const uint8_t* p_idx0 = (const uint8_t*)idx0 + 1u * r * idx_groups_stride;
            const uint8_t* p_idx1 = (const uint8_t*)idx1 + 1u * r * idx_groups_stride;

            // Output row r in the COMPACT W_dq tile
            uint16_t* p_out = y_out;

            const uint16_t* scale_ptr = &scales[r];
            const uint16_t* x_ptr     = &x_vec[r];

            dequant_groupmacc_improvedfused(p_idx0, p_idx1, cb0, cb1, x_ptr, scale_ptr, p_out, group_count, r);
        }
        flex_timer_end();
    }
}
// }
void dequantize_block_tile_compact_improved(uint16_t row_start, uint16_t rows,
                                            uint16_t group_count,       // groups in this tile
                                            uint16_t idx_groups_stride) // should equal group_count for compactstorage
{
    if (flex_get_core_id() == SPATZ_CORE && flex_get_cluster_id() == 0) {
        uint16_t* W_tile    = (uint16_t*)(uintptr_t)g_l1_dq.W_dq;             // COMPACT tile buffer
        const uint8_t* idx0 = (const uint8_t*)(uintptr_t)g_l1_dq.indices0_u8; // COMPACT indices
        const uint8_t* idx1 = (const uint8_t*)(uintptr_t)g_l1_dq.indices1_u8; // COMPACT indices

        const uint16_t* scales = (const uint16_t*)(uintptr_t)g_l1_dq.scales;
        const uint16_t* cb0    = (const uint16_t*)(uintptr_t)g_l1_dq.cb;
        const uint16_t* cb1    = cb0 + VQ_CB_NUM_CENTROIDS * VQ_GROUP_SIZE;

        const uint16_t row_end = row_start + rows;
        const uint32_t tile_P  = group_count * VQ_GROUP_SIZE;

        // printf("[DEBUG][DQ] compact tile dequant: rows [%u..%u), groups=%u, tile_P=%u\n\t", row_start, row_end,
        //        group_count, tile_P);

        flex_timer_start(); // puttingtimer here doesntchange theruntime

        // asm volatile("vsetvli zero, %0, e16, m1, ta, ma" ::"r"((uint32_t)VQ_GROUP_SIZE));no group size , the best is
        // if we load as much ass possible
        for (uint16_t r = row_start; r < row_end; ++r) {

            // Indices for row r in the compact tile
            const uint8_t* p_idx0 = (const uint8_t*)idx0 + 1u * r * idx_groups_stride;
            const uint8_t* p_idx1 = (const uint8_t*)idx1 + 1u * r * idx_groups_stride;

            // Output row r in the COMPACT W_dq tile
            uint16_t* p_out = W_tile + r * tile_P;

            const uint16_t* scale_ptr = &scales[r];

            dequant_groupmacc_improved(p_idx0, p_idx1, cb0, cb1, scale_ptr, p_out, group_count);
        }
        flex_timer_end();
    }
}

// static inline void dequantize_full_matrix(void) { dequantize_block(0, FP16_M, 0, VQ_NUM_GROUPS_PER_ROW); }

#endif

// group_offset     = 0;

//                 const uint32_t avl_elems = (uint32_t)remaining_groups * VQ_GROUP_SIZE;
//                 uint32_t       vl_elems  = 0;
//                 asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl_elems) : "r"(avl_elems));

//                 uint32_t groups_this_iter = vl_elems / VQ_GROUP_SIZE;
//                 if (groups_this_iter == 0u) {
//                     groups_this_iter = 1u;
//                     asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl_elems) : "r"((uint32_t)VQ_GROUP_SIZE));
//                 }

//                 const uint8_t* idx0_ptr = row_idx0 + group_offset;
//                 const uint8_t* idx1_ptr = row_idx1 + group_offset;
//                 uint16_t*      y_ptr    = y_out + (size_t)group_offset * VQ_GROUP_SIZE;

//                 uint32_t vl_idx = 0;
//                 asm volatile("vsetvli %0, %1, e8, m1, ta, ma" : "=r"(vl_idx) : "r"(groups_this_iter));
//                 (void)vl_idx;
//                 asm volatile("vle8.v v0, (%0)" ::"r"(idx0_ptr) : "memory");
//                 asm volatile("vle8.v v1, (%0)" ::"r"(idx1_ptr) : "memory");

//                 const uint32_t payload_elems = groups_this_iter * VQ_GROUP_SIZE;
//                 asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl_elems) : "r"(payload_elems));

//                 EMIT_WORD_IMM(VLBLK1EI8_V(RVV_V8, RVV_V0, RVX_T0, 1));
//                 EMIT_WORD_IMM(VLBLK1EI8_V(RVV_V16, RVV_V1, RVX_T1, 1));

//                 asm volatile("vfmul.vf v24, v8, fa0");
//                 asm volatile("vfmacc.vf v24, fa0, v16");

//                 asm volatile("vle16.v v20, (%0)" ::"r"(y_ptr) : "memory");
//                 asm volatile("vfmacc.vf v20, fa0, v24");
//                 asm volatile("vse16.v v20, (%0)" ::"r"(y_ptr) : "memory");

//             //     remaining_groups -= groups_this_iter;
//             //     group_offset += groups_this_iter;
//             // }
//         }
