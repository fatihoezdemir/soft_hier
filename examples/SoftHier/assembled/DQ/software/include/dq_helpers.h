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
    volatile uint16_t* cb;      // codebook base in L1
    volatile uint16_t* scales;  // scales in L1
    volatile uint16_t* indices; // indices in L1
    volatile uint16_t* W_dq;    // dequantized weights tile in L1
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
void dequant_group(const uint16_t* a /*cb0[idx0[i]]*/, const uint16_t* b /*cb1[idx1[i]]*/, const uint32_t* scale,
                   uint16_t* out) {
    asm volatile("vsetvli  zero, %[g], e16, m1, ta, ma\n\t"

                 // "vlseg2e8.v v0, (%[a])\n\t"

                 // "vlseg8e8.v vd, (rs1), vm"
                 "vmv.v.i  v2, 0\n\t"
                 "vle16.v  v0, (%[a])\n\t"
                 "vle16.v  v1, (%[b])\n\t"
                 "flw      fa0, (%[s])\n\t"  // f0 = scale (fp32) scale swere uinsigned
                 "vfadd.vv v2, v0, v1\n\t"   // v2 = a * s
                 "vfmul.vf v3, v2, fa0\n\t"  // v2 += b * s  => (a+b)*s
                 "vse16.v  v3, (%[out])\n\t" // store
                 :
                 : [a] "r"(a), [b] "r"(b), [s] "r"(scale), [out] "r"(out), [g] "r"(8u)
                 : "v0", "v1", "v2", "fa0");
}
void dequant_group_legacy(const uint16_t* a /*cb0[idx0[i]]*/, const uint16_t* b /*cb1[idx1[i]]*/, const uint32_t* scale,
                          uint16_t* out) {
    uint32_t vl;
    uint32_t gsize = 8;
    // FINALUse vector-vector multiply with scale broadcasting via vmv.v.x
    asm volatile("vsetvli zero, %[gsize], e16, m1, ta, ma\n\t"
                 // "vlseg2e8.v v5, (%[a])\n\t" illegal instruciotn
                 "vle16.v  v0, (%[a])\n\t"   // cb0[idx0]
                 "vle16.v  v1, (%[b])\n\t"   // cb1[idx1]
                 "lhu t0, (%[sp])\n\t"       // load scale as integer scale swere uinsigned
                 "vmv.v.x v2, t0\n\t"        // broadcast scale to vector
                 "vfadd.vv v3, v0, v1\n\t"   // sum
                 "vfmul.vv v4, v3, v2\n\t"   // vector-vector multiply
                 "vse16.v  v4, (%[out])\n\t" // store
                 :
                 : [a] "r"(a), [b] "r"(b), [out] "r"(out), [sp] "r"(scale), [gsize] "r"(8u)
                 : "t0", "v0", "v1", "v2", "v3", "v4");
}

// TODO load scale once per row
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

        flex_timer_start(); // puttingtimer here doesntchange theruntime

        asm volatile("vsetvli zero, %0, e16, m1, ta, ma" ::"r"(8u));
        for (uint16_t r = row_start; r < row_end; ++r) {

            // Indices for row r in the compact tile
            const uint8_t* p_idx = (const uint8_t*)idx + 2u * r * idx_groups_stride;

            // Output row r in the COMPACT W_dq tile
            uint16_t* p_out = W_tile + r * tile_P;

            const uint32_t* scp = (const uint32_t*)&scales[r];
            // const uint16_t s = scales[r];
            asm volatile("lhu t0, (%0)" ::"r"(scp) : "t0", "memory"); // Load scale and broadcast to vector

            const uint16_t scale_val = scales[r];
            for (uint16_t g = 0; g < group_count; ++g) {
                const uint8_t idx0 = p_idx[0];
                const uint8_t idx1 = p_idx[1];
                p_idx += 2;

                const uint16_t* a = cb0 + (unsigned)idx0 * VQ_GROUP_SIZE;
                const uint16_t* b = cb1 + (unsigned)idx1 * VQ_GROUP_SIZE;
                // dequant_group(a, b, scp, p_out);
                dequant_group(a, b, scp, p_out);

                // dequant_group_s_by_val(a, b, s, p_out);

                p_out += VQ_GROUP_SIZE;
            }
        }
        flex_timer_end();
    }
}

// static inline void dequantize_full_matrix(void) { dequantize_block(0, FP16_M, 0, VQ_NUM_GROUPS_PER_ROW); }

#endif