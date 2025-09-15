
#include "dq_matmul_fp16.h"
#include "flex_runtime.h"
#include "spatz_rvv_extensions.h"

#include <stdint.h>
#include <stdio.h>

// Debug print for uint16_t arrays
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
// data
typedef struct { // 32 bit architecture
    volatile uint32_t cb;
    volatile uint32_t indices;
    volatile uint32_t scales;
    volatile uint32_t activation;
    volatile uint32_t W_dq;
    volatile uint32_t result;
    // volatile bool valid;   // TODO LATER
} shared_ptrs_t; // saved in sync
#define L1_PTRS (*(shared_ptrs_t*)(ARCH_SYNC_BASE))

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
/**
 *
 */
void dequantize(uint16_t start_group, uint16_t end_group) {
    if (flex_get_core_id() == 0 /*spatz*/ && flex_get_cluster_id() == 0) {
        // init counters
        printf("\n[DEBUG][DQ] dequantization from group %d  to group %d\n", (int)start_group, (int)end_group);

        uint16_t* W_hat            = (uint16_t*)(uintptr_t)L1_PTRS.W_dq; // (128*16)*8
        const uint16_t* idx_packed = (uint16_t*)(uintptr_t)L1_PTRS.indices;
        const uint16_t* scales     = (uint16_t*)(uintptr_t)L1_PTRS.scales; // (128*16)*8
        const uint16_t* cb0        = (uint16_t*)(uintptr_t)L1_PTRS.cb;
        const uint16_t* cb1        = cb0 + VQ_CB_NUM_CENTROIDS * VQ_GROUP_SIZE; // cb0+256*8

        flex_timer_start();
        for (size_t i = start_group; i < end_group; ++i) {

            //    indices_flattened =  (indices[1] << 8) | indices[0]  idx1 idx0 eg 0xDE33
            uint8_t* ptr       = (uint8_t*)idx_packed;
            const uint8_t idx0 = ptr[2 * i];     // idx0[i] - lower byte
            const uint8_t idx1 = ptr[2 * i + 1]; // idx1[i] - upper byte

            const uint16_t* a  = cb0 + (uint16_t)idx0 * VQ_GROUP_SIZE; // For NOW: Compiler helps with the indexing
            const uint16_t* b  = cb1 + (uint16_t)idx1 * VQ_GROUP_SIZE; // to get cb1[idx1]
            uint16_t scale_idx = i / VQ_NUM_GROUPS_PER_ROW /*16*/;     // TODO this dequantization is only valid for
                                                                       // vertically split matrices
            const uint16_t* scale = &scales[scale_idx];                // Or
            // printf("scale %f\n",&scale);

            dequant_group(a, b, scale, W_hat + i * VQ_GROUP_SIZE); // LSA, cb0[idx0]+cb1[idx]
        }
        flex_timer_end();
    }
}

void dequantize_block(uint16_t row_start, uint16_t rows, uint16_t group_start, uint16_t group_counts) {
    if (flex_get_core_id() == 0 /*spatz*/ && flex_get_cluster_id() == 0) {
        // init counters
        printf("\n[DEBUG][DQ] dequantization from group %d  to group %d\n", (int)row_start, (int)rows);

        uint16_t* W_hat            = (uint16_t*)(uintptr_t)L1_PTRS.W_dq; // (128*16)*8
        const uint16_t* idx_packed = (uint16_t*)(uintptr_t)L1_PTRS.indices;
        const uint16_t* scales     = (uint16_t*)(uintptr_t)L1_PTRS.scales; // (128*16)*8
        const uint16_t* cb0        = (uint16_t*)(uintptr_t)L1_PTRS.cb;
        const uint16_t* cb1        = cb0 + VQ_CB_NUM_CENTROIDS * VQ_GROUP_SIZE; // cb0+256*8
        uint16_t row_end           = row_start + rows;
        uint16_t group_end         = group_start + group_counts;
        flex_timer_start();
        for (size_t r = row_start; r < row_end; ++r) {
            const uint16_t s = scales[r];
            //    indices_flattened =  (indices[1] << 8) | indices[0]  idx1 idx0 eg 0xDE33
            uint8_t* p_idx  = (uint8_t*)idx_packed + 2 * (r * VQ_NUM_GROUPS_PER_ROW + group_start);
            uint16_t* p_out = W_hat + (r * VQ_NUM_GROUPS_PER_ROW + group_start) * VQ_GROUP_SIZE;
            for (int g = group_start; g < group_end; ++g) {
                uint8_t idx0 = p_idx[0];
                uint8_t idx1 = p_idx[1];
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

// void dequantize_full_matrix() { dequantize(0, VQ_TOTAL_GROUPS); }
static inline void dequantize_full_matrix(void) { dequantize_block(0, FP16_M, 0, VQ_NUM_GROUPS_PER_ROW); }
