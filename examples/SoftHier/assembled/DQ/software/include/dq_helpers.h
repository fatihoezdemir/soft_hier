#ifndef DQ_HELPERS_H
#define DQ_HELPERS_H

#include "dq_data_hbm.h"
#include "flex_runtime.h"
#include "spatz_rvv_extensions.h"

#include <stdint.h>
#include <stdio.h>

// Debug print for uint16_t
#define DEBUG_PRINT_U16(ptr, count, desc)                                                                              \
    do {                                                                                                               \
        printf("[DEBUG] First %d %s:\n", (count), (desc));                                                             \
        for (int _i = 0; _i < (count); _i++) {                                                                         \
            printf("\t  [%d] = 0x%04x\n", _i, (ptr)[_i]);                                                              \
        }                                                                                                              \
    } while (0)
#if DEBUG
#define CDEBUG_PRINT_U16(ptr, count, desc) DEBUG_PRINT_U16(ptr, count, desc)
#else
#define CDEBUG_PRINT_U16(ptr, count, desc)                                                                             \
    do {                                                                                                               \
    } while (0)

#endif

// TODO not valid for multicluster scenario, maybe create array of clusters?
typedef struct {
    volatile uint32_t* cb;      // codebook base in L1
    volatile uint32_t* scales;  // scales in L1
    volatile uint32_t* indices; // indices in L1
    volatile uint32_t* W_dq;    // dequantized weights tile in L1
} L1_DQ_Handles;
extern L1_DQ_Handles g_l1_dq;

// TODO Try out to dequant full row (or multiple groups) with register grouping,
// 8*16*sizeof(float16) bits

void dequant_group(const uint16_t* a /*cb0[idx0[i]]*/, const uint16_t* b /*cb1[idx1[i]]*/, const uint16_t* scale,
                   uint16_t* out) {
    uint32_t vl;
    uint32_t gsize = 8;
    // FINALUse vector-vector multiply with scale broadcasting via vmv.v.x
    asm volatile("vsetvli zero, %[gsize], e16, m1, ta, ma\n\t"
                 "vle16.v  v8, (%[a])\n\t"    // cb0[idx0]
                 "vle16.v  v9, (%[b])\n\t"    // cb1[idx1]
                 "lhu t0, (%[sp])\n\t"        // load scale as integer
                 "vmv.v.x v12, t0\n\t"        // broadcast scale to vector
                 "vfadd.vv v10, v8, v9\n\t"   // sum
                 "vfmul.vv v11, v10, v12\n\t" // vector-vector multiply
                 "vse16.v  v11, (%[out])\n\t" // store
                 :
                 : [a] "r"(a), [b] "r"(b), [out] "r"(out), [sp] "r"(scale), [gsize] "r"(8u)
                 : "t0", "v8", "v9", "v10", "v11", "v12", "memory");

    // printf("  Out[0]=0x%04x Out[1]=0x%04x Out[2]=0x%04x (after store)\n", out[0],out[1],out[2]);
}

// Dequant a vertical tile into a buffer W_dq_tile
// Layout: row-major [rows x tile_P], tile_P = group_count*8
void dequantize_block_tile_compact(uint16_t row_start, uint16_t rows,
                                   uint16_t group_count,       // groups in this tile
                                   uint16_t idx_groups_stride) // should equal group_count for compactstorage
{
    if (flex_get_core_id() == 0 && flex_get_cluster_id() == 0) {
        uint16_t* W_tile       = (uint16_t*)(uintptr_t)g_l1_dq.W_dq;          // COMPACT tile buffer
        const uint16_t* idx    = (const uint16_t*)(uintptr_t)g_l1_dq.indices; // COMPACT indices
        const uint16_t* scales = (const uint16_t*)(uintptr_t)g_l1_dq.scales;
        const uint16_t* cb0    = (const uint16_t*)(uintptr_t)g_l1_dq.cb;
        const uint16_t* cb1    = cb0 + VQ_CB_NUM_CENTROIDS * VQ_GROUP_SIZE;

        const uint16_t row_end = row_start + rows;
        const uint32_t tile_P  = group_count * VQ_GROUP_SIZE;

        // printf("[DEBUG][DQ] compact tile dequant: rows [%u..%u), groups=%u, tile_P=%u\n\t", row_start, row_end,
        //        group_count, tile_P);

        flex_timer_start();
        for (uint16_t r = row_start; r < row_end; ++r) {
            const uint16_t s = scales[r];

            // Indices for row r in the compact tile
            const uint8_t* p_idx = (const uint8_t*)idx + 2u * r * idx_groups_stride;

            // Output row r in the COMPACT W_dq tile
            uint16_t* p_out = W_tile + r * tile_P;

            for (uint16_t g = 0; g < group_count; ++g) {
                const uint8_t idx0 = p_idx[0];
                const uint8_t idx1 = p_idx[1];
                p_idx += 2;

                const uint16_t* a = cb0 + (unsigned)idx0 * VQ_GROUP_SIZE;
                const uint16_t* b = cb1 + (unsigned)idx1 * VQ_GROUP_SIZE;

                dequant_group(a, b, &s, p_out);
                p_out += VQ_GROUP_SIZE;
            }
        }
        flex_timer_end();
    }
}

// static inline void dequantize_full_matrix(void) { dequantize_block(0, FP16_M, 0, VQ_NUM_GROUPS_PER_ROW); }

#endif