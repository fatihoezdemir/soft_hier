/*********
 * Benchmarking GEMV and GEMM kernels
 * TODO deprecated,change
 * Each benchmark in sequential steps (but the processes eg dma and compute or dma and dequantize happen at same time)
 * db:  1)load scales,codebooks and left half of indices
 *      2)dequantize partially wuth indices from step 1) and also load upper half of activation matrix
 *      3)compute partial Matrix resultand load other half of indices
 *          (upper half of activation x left half of indices would give us upper left quarter of the full resutl)
 *      4)load other half of activation matrix and dequantize other half of weight matrix
 *      5) compute  the other 3 tiles of C= A*B
 * Reminder: dequantization in AQLM is like this :
 *  in case of dequantizing a row :W_hat[row]= scale[row]*( concat(cb1[idx1[i]]+cb2[idx2[i]] for all groups i in a
 * certain row
 * ))
 *
 */
/* activation matrix A= [ A0
                          A1
                          A2
                          ... ]   horizontal splits
Weight Matrix B = [B0 B1 B2 ...] vertical splits
*/
#define DEBUG 0
#define REDMULE_ON 1

#include "flex_alloc.h"
#include "flex_dma_pattern.h"
#include "flex_dump.h"
#include "flex_libfp16.h"
#include "flex_libfp8.h"
#include "flex_printf.h"
#include "flex_redmule.h"
#include "flex_runtime.h"
#include "include/debug.h"
#include "include/dq_compute_spatz.h"
#include "include/dq_helpers.h"

#include <inttypes.h>
#include <stdio.h>
#define SPATZ_CORE 0
#define DOUBLEBUFFER 1

// Safe printing and timing macros - only DMA core on cluster 0 prints
#define PDEBUG(...)                                                                                                    \
    do {                                                                                                               \
        if (flex_is_dm_core() && flex_get_cluster_id() == 0)                                                           \
            printf(__VA_ARGS__);                                                                                       \
    } while (0)

#define PERF_START_DM()                                                                                                \
    do {                                                                                                               \
        if (flex_is_first_core() && flex_get_cluster_id() == 0) {                                                      \
            /* printf("[SYNC]");  */                                                                                   \     
            flex_timer_start();                                                                                        \
        }                                                                                                              \
    } while (0)
#define PERF_END_DM()                                                                                                  \
    do {                                                                                                               \
        if (flex_is_first_core() && flex_get_cluster_id() == 0)                                                        \
            flex_timer_end();                                                                                          \
    } while (0)

// Tiling configuration TODO make a tilinginfo struct
#define NUM_TILES 1
#define GROUPS_PER_TILE (VQ_NUM_GROUPS_PER_ROW / NUM_TILES)
#define REMAINDER_GROUPS (VQ_NUM_GROUPS_PER_ROW % NUM_TILES)

// For activation matrix tiling (must match weight matrix number of rows for GEMM)
#define COLS_PER_TILE (FP16_K / NUM_TILES)
#define ROWS_PER_TILE (FP16_M / NUM_TILES)



#if MHSA != 1 // only for gemv  and gemm kernels
              /**
               load  codebook and scales to l1 and wait until transfer done
               */

volatile uint64_t C_hbm_base; // HBM base address for result matrix
typedef struct {
    volatile uint32_t idx_buf[2]; // buffers for indices
    volatile uint32_t A_buf[2];   //  buffers for activation tiles
} shared_ptrs_t;                  // saved in sync
// #define L1_PTRS (*(shared_ptrs_t*)(ARCH_SYNC_BASE))
#if GEMM == 1
typedef struct {
    volatile uint32_t idx_buf[2];
    volatile uint32_t A_buf[2];    //  buffers for activation tiles
    volatile uint32_t* activation; //__attribute__((section(".l1_prio")));
    volatile uint32_t* C_tile;     // Result tile buffer
} L1_buffers;
#elif GEMV == 1
typedef struct {
    volatile uint32_t idx_buf[2];
    volatile uint32_t* x_vec; //__attribute__((section(".l1_prio")));
    volatile uint32_t* y_vec; // Result tile buffer
} L1_buffers;
#elif MHSA == 1
#endif

L1_buffers l1_buffers;
L1_DQ_Handles g_l1_dq = {0};

void load_codebook_to_l1() {
    // 1) first allocate codebook and indices
    g_l1_dq.cb = (uint32_t*)(uintptr_t)flex_l1_malloc(
        VQ_CB_NUM_CENTROIDS * VQ_GROUP_SIZE * VQ_NUM_CBS *
        sizeof(uint16_t)); // codebook size, [256,8,2],2 codebooks,256 centroids of size 8(8 is group size, 1
    // decoding decodes entry to 8 values)
    g_l1_dq.scales = (uint32_t*)(uintptr_t)flex_l1_malloc(
        FP16_M * sizeof(uint16_t)); // scales size (128) is the numnerb of rows of W
    // flex_timer_start();
    flex_dma_async_1d((uint64_t)(uintptr_t)g_l1_dq.cb, (uint64_t)(uintptr_t)&matrix_cb_fp16[0],
                      VQ_CB_NUM_CENTROIDS * VQ_NUM_CBS * VQ_GROUP_SIZE * sizeof(uint16_t));
    flex_dma_async_1d((uint64_t)(uintptr_t)g_l1_dq.scales, (uint64_t)(uintptr_t)&matrix_scales_fp16[0],
                      FP16_M * sizeof(uint16_t));
    flex_dma_async_wait_all();
    // flex_timer_end();
}
#endif
// calculate actual size for a specific tile
static inline uint32_t get_groups_for_tile(uint32_t tile_index) {
    return GROUPS_PER_TILE + (tile_index < REMAINDER_GROUPS ? 1 : 0);
}

static inline uint32_t get_rows_for_tile(uint32_t tile_index) {
    return ROWS_PER_TILE + (tile_index < (FP16_M % NUM_TILES) ? 1 : 0);
}

// Account for extra groups in previous tiles, All remainder groups are in first tiles
static inline uint32_t get_start_group_for_tile(uint32_t tile_index) {
    uint32_t start_group = tile_index * GROUPS_PER_TILE;
    start_group += (tile_index < REMAINDER_GROUPS) ? tile_index : REMAINDER_GROUPS;
    return start_group;
}
static inline uint32_t get_start_row_for_tile(uint32_t tile_index) {
    uint32_t start_row      = tile_index * ROWS_PER_TILE;
    uint32_t remainder_rows = FP16_M % NUM_TILES;
    start_row += (tile_index < remainder_rows) ? tile_index : remainder_rows;
    return start_row;
}
// Load a specific tile of the weight matrix indices into a compact buffer
void load_indices_tile(void* dest, const void* src, uint32_t tile_index) {
    // Calculate groups for this tile
    uint32_t groups_this_tile = get_groups_for_tile(tile_index);
    // Calculate starting group position
    uint32_t start_group = get_start_group_for_tile(tile_index);

    flex_timer_start();
    flex_dma_async_2d((uint64_t)(uintptr_t)dest,                                 // compact dest buffer, no offset
                      (uint64_t)(uintptr_t)src + start_group * sizeof(uint16_t), // source with offset
                      groups_this_tile * sizeof(uint16_t),                       // transfer size per row
                      groups_this_tile * sizeof(uint16_t),                       // dest stride = tile width (compact)
                      VQ_NUM_GROUPS_PER_ROW * sizeof(uint16_t),                  // source stride (full row)
                      FP16_M                                                     // all rows
    );
    flex_dma_async_wait_all();
    flex_timer_end();
}

// Load a horizontal stripe of the activation matrix (split by rows only)
// Since rows are contiguous in memory, we can use 1D DMA
void load_activation_horizontal_tile(void* dest, const void* src, uint32_t row_tile_index) {
    uint32_t rows_this_tile = get_rows_for_tile(row_tile_index);
    uint32_t start_row      = get_start_row_for_tile(row_tile_index);
    uint32_t start_element  = start_row * FP16_N;
    uint32_t num_elements   = rows_this_tile * FP16_N;

    flex_timer_start();
    // Load horizontal stripe using 1D DMA (more efficient for contiguous data) Dest has no offset since it's a compact
    flex_dma_async_1d((uint64_t)(uintptr_t)dest, (uint64_t)(uintptr_t)src + start_element * sizeof(uint16_t),
                      num_elements * sizeof(uint16_t));
    flex_dma_async_wait_all();
    flex_timer_end();
}

#if GEMM == 1

void double_buffer_gemm() {                   // TODO currently only load is overlapped,also overlap store
    uint32_t CID     = flex_get_cluster_id(); // Get cluster ID
    uint32_t core_id = flex_get_core_id();
    uint32_t max_g   = get_groups_for_tile(0);
    uint32_t max_r   = get_rows_for_tile(0);

    const uint32_t max_P = max_g * VQ_GROUP_SIZE;

    // allocate necessary data in L1
    if (flex_is_dm_core() && CID == 0) {
        debug("[DEBUG][DMA] Allocating double buffers for pipelined execution\n");

        // Allocate HBM for result matrix C
        const uint32_t C_bytes = FP16_M * FP16_K * sizeof(uint16_t);
        C_hbm_base             = (uint64_t)(uintptr_t)flex_hbm_malloc(C_bytes);

        // Load codebooks and scales (one-time load)
        load_codebook_to_l1();

        uint32_t total_allocated = 0;
        uint32_t idx_buf_size    = FP16_M * max_g * sizeof(uint16_t);
        uint32_t A_buf_size      = max_r * FP16_N * sizeof(uint16_t);
        uint32_t W_buf_size      = FP16_M * max_P * sizeof(uint16_t);
        uint32_t C_buf_elems     = max_r * max_P;
        uint32_t C_buf_size      = C_buf_elems * sizeof(uint16_t);

        // Allocate  buffers for indices and activation
        l1_buffers.idx_buf[0] = (uint32_t)(uintptr_t)flex_l1_malloc(idx_buf_size);
        l1_buffers.idx_buf[1] = (uint32_t)(uintptr_t)flex_l1_malloc(idx_buf_size);
        l1_buffers.A_buf[0]   = (uint32_t)(uintptr_t)flex_l1_malloc(A_buf_size);
        l1_buffers.A_buf[1]   = (uint32_t)(uintptr_t)flex_l1_malloc(A_buf_size);
        // Allocate single buffers for dequantized weights and result tile
        g_l1_dq.W_dq      = (uint32_t*)(uintptr_t)flex_l1_malloc(W_buf_size);
        l1_buffers.C_tile = (uint32_t*)(uintptr_t)flex_l1_malloc(C_buf_size);

        //     uint16_t* c_ptr = (uint16_t*)(uintptr_t)l1_buffers.C_tile;
        //     for(int i=0;i<C_buf_size;++i){
        //         c_ptr[i]=0;
        //         debug("0x%04x ", c_ptr[i]);
        // }
    }
    PERF_START_DM();
    flex_intra_cluster_sync();
    PERF_END_DM();

    // ---------- Outer loop over tiles of B_hat ----------
    // PDEBUG("\n=== STARTING DOUBLE-BUFFERED PIPELINE (NUM_TILES=%u) ===\n", NUM_TILES);
    int curB = 0;
    for (uint32_t bt = 0; bt < NUM_TILES; ++bt) {
        const uint32_t g   = get_groups_for_tile(bt); // groups in this B tile
        const uint32_t P   = g * VQ_GROUP_SIZE;       // columns in compact W/C
        const uint64_t c0B = (uint64_t)(get_start_group_for_tile(bt) * VQ_GROUP_SIZE) * sizeof(uint16_t);

        debug("\n\t>>> B-TILE %u: groups=%u, P=%u, curB_buf=%d\n", bt, g, P, curB);

        // (Step 0) DMA: Load indices for B_t -> idx_buf[curB]
        if (flex_is_dm_core() && CID == 0) {
            debug("\t[DMA] Load B%u indices to idx_buf[%d]=0x%08x\n\t", bt, curB, l1_buffers.idx_buf[curB]);
            g_l1_dq.indices = (uint32_t*)l1_buffers.idx_buf[curB];
            load_indices_tile((void*)l1_buffers.idx_buf[curB], &matrix_idx_packed_uint16[0], bt);
        }

        PERF_START_DM();
        flex_intra_cluster_sync();
        PERF_END_DM();

        // (Step 1) Spatz: Dequantize B_t -> W tile (compact)
        if (core_id == SPATZ_CORE && CID == 0) {
            // debug("\t[SPATZ] Dequantizing B%u\n\t", bt);
            g_l1_dq.indices = (uint32_t*)l1_buffers.idx_buf[curB];
            dequantize_block_tile_compact(/*row_start=*/0, /*rows=*/FP16_M,
                                          /*group_count=*/g, /*idx_groups_stride=*/g);
        }
        // ---------- Inner loop over all tiles of A (double-buffer) ----------
        int curA = 0;
        // Prologue: DMA load A0 -> A_buf[curA]
        if (flex_is_dm_core() && CID == 0) {
            debug("\t[DMA] Load A0 to A_buf[%d]=0x%08x for B%u\n\t", curA, l1_buffers.A_buf[curA], bt);
            load_activation_horizontal_tile((void*)l1_buffers.A_buf[curA], &matrix_activation_fp16[0],
                                            /*row_tile_index=*/0);
            debug("\t[SPATZ] Dequantizing\n\t");
        }
        PERF_START_DM();
        flex_intra_cluster_sync();
        PERF_END_DM();
        for (uint32_t at = 0; at < NUM_TILES; ++at) {
            const uint32_t r  = get_rows_for_tile(at);
            const uint32_t sr = get_start_row_for_tile(at); // row start
            if (core_id == SPATZ_CORE && CID == 0) {
                flex_redmule_config(r, FP16_N, P); // hardcode r N P
            }
            debug("\n\t\t>> A-0 %u: rows=%u, start_row=%u, curA_buf=%d\n", at, r, sr, curA);

            // Calculate destination address in allocated HBM buffer
            const uint32_t col_start_elems = get_start_group_for_tile(bt) * VQ_GROUP_SIZE;
            const uint32_t row_start_elems = sr * FP16_K;
            const uint64_t dst = C_hbm_base + (uint64_t)(row_start_elems + col_start_elems) * sizeof(uint16_t);

            // DMA: Prefetch next A tile (overlaps with Spatz compute)
            if (at + 1 < NUM_TILES && flex_is_dm_core() && CID == 0) {
                int nxtA = curA ^ 1; // Toggle buffer
                debug("\t\t[DMA] Load A%u to A_buf[%d]=0x%08x (while computing)\n\t\t", at + 1, nxtA,
                      l1_buffers.A_buf[nxtA]);
                load_activation_horizontal_tile((void*)l1_buffers.A_buf[nxtA], &matrix_activation_fp16[0], at + 1);
            }

            // While processing first A tile, prefetch next B indices
            if (at == 0 && bt + 1 < NUM_TILES && flex_is_dm_core() && CID == 0) {
                int nxtB = curB ^ 1; // Toggle buffer
                debug("\t\t[DMA] Prefetching B%u indices to idx_buf[%d]=0x%08x\n\t\t", bt + 1, nxtB,
                      l1_buffers.idx_buf[nxtB]);
                load_indices_tile((void*)l1_buffers.idx_buf[nxtB], &matrix_idx_packed_uint16[0], bt + 1);
                // flex_dma_async_1d((uint64_t) (uintptr_t) l1_buffers.C_tile, zomem(0), 8192);
                bare_dma_wait_all();
            }

            // Spatz: Compute A_at × B_bt (compact P) -> C_tile
            if (core_id == SPATZ_CORE && CID == 0) {
                l1_buffers.activation = (uint32_t*)l1_buffers.A_buf[curA];

                // debug("    [REDMULE] Computing A[%u×%u] × B[%u×%u] -> C[%u×%u]\n",
                //     r, FP16_N, FP16_N, P, r, P);
                uint32_t C_buf_elems = max_r * max_P;
                uint32_t C_buf_size  = C_buf_elems * sizeof(uint16_t);
                uint16_t* c_ptr      = (uint16_t*)(uintptr_t)l1_buffers.C_tile;
                for (int i = 0; i < C_buf_size; ++i) { // TODO temporary solution,fix later
                    c_ptr[i] = 0;
                    //         // debug("0x%04x ", c_ptr[i]);
                }
                debug("\t\t[REDMULE] Compute\n\t\t");
                // flex_timer_start();
                // flex_redmule_trigger((uint32_t)l1_buffers.A_buf[curA], (uint32_t)g_l1_dq.W_dq,
                //                      (uint32_t)l1_buffers.C_tile, REDMULE_FP_16);
                // flex_redmule_wait();
                // flex_timer_end();

                debug("\t\t[spatz] compute\n\t\t");
                // flex_timer_start();

                spatz_matmul_fp16_full_legacy((uint16_t*)l1_buffers.A_buf[curA], // [r x N]
                                              (uint16_t*)g_l1_dq.W_dq,           // [N x P] (compact)
                                              (uint16_t*)l1_buffers.C_tile,      // [r x P] (compact)
                                              r, FP16_N, P);
                // flex_timer_end();
            }
            PERF_START_DM();
            flex_intra_cluster_sync();
            PERF_END_DM();

            // DMA: Store compact C_tile (r×P) back to HBM with proper striding
            if (flex_is_dm_core() && CID == 0) {
                debug("\t\t[DEBUG][DMA] Storing A%u x B%u to dst=0x%08x (r=%u, P=%u)\n\t\t", at, bt, (uint32_t)dst, r,
                      P);
                flex_timer_start();
                flex_dma_async_2d(dst, (uint64_t)(uintptr_t)l1_buffers.C_tile, P * sizeof(uint16_t),
                                  FP16_K * sizeof(uint16_t), // Full C matrix width
                                  P * sizeof(uint16_t),      //  tile width
                                  /*rows*/ r);
                flex_dma_async_wait_all();
                flex_timer_end();
            }
            PERF_START_DM();
            flex_intra_cluster_sync();
            PERF_END_DM();
            // Swap A_buf for next iteration
            curA ^= 1;
        }
        // Swap idx_buf B for next outer iteration
        curB ^= 1;
    }

    // Verify computation results against golden reference
    if (flex_is_dm_core() && CID == 0) {
        debug("\n[VERIFICATION] Double-buffered GEMM complete! Verifying results...\n");

        //  L1_PTRS.result points to C_tile buffer
        uint16_t* hbm_result_ptr = (uint16_t*)(uintptr_t)C_hbm_base;
        // debug("[DEBUG] Result matrix is in HBM at 0x%08x\n", (uint32_t)C_hbm_base);
        // debug("[DEBUG] First 8 computed results from HBM: ");
        // for (int i = 0; i < 8; i++) {
        //     debug("0x%04x ", hbm_result_ptr[i]);
        // }
        // debug("\n");

// Verify matrix multiplication results against golden reference
        debug("[DEBUG] Verifying GEMM results against golden reference...\n");
        // spatz_verify_16(FP16_M * FP16_N, hbm_result_ptr, (uint16_t*)matrix_golden_fp16, 0.5f);
        debug("[VERIFICATION] Verification complete!\n");
    }
}
#endif

#if GEMV == 1
void double_buffer_gemv() {
    uint32_t CID     = flex_get_cluster_id();
    uint32_t core_id = flex_get_core_id();
    // Calculate max sizes for buffer allocation
    uint32_t max_g = get_groups_for_tile(0);
    const uint32_t max_P = max_g * VQ_GROUP_SIZE;

    if (flex_is_dm_core() && CID == 0) {
        // Allocate HBM for result vector y
        const uint32_t y_bytes = FP16_K * sizeof(uint16_t);
        C_hbm_base             = (uint64_t)(uintptr_t)flex_hbm_malloc(y_bytes);
        // debug("[DEBUG][DMA] y_hbm_base  alloc= 0x%08x (size=%u bytes)\n", (uint32_t)C_hbm_base, y_bytes);

        flex_timer_start();
        load_codebook_to_l1(); // load codebook
        flex_timer_end();

        // Buffer sizes
        uint32_t idx_buf_size = FP16_M * max_g * sizeof(uint16_t);
        uint32_t x_size       = FP16_M * sizeof(uint16_t);
        uint32_t W_buf_size   = FP16_M * max_P * sizeof(uint16_t);
        uint32_t y_tile_size  = max_P * sizeof(uint16_t);

        // Allocate  buffers for indices
        l1_buffers.idx_buf[0] = (uint32_t)(uintptr_t)flex_l1_malloc(idx_buf_size);
        l1_buffers.idx_buf[1] = (uint32_t)(uintptr_t)flex_l1_malloc(idx_buf_size);
        l1_buffers.x_vec      = (uint32_t*)(uintptr_t)flex_l1_malloc(x_size);
        g_l1_dq.W_dq          = (uint32_t*)(uintptr_t)flex_l1_malloc(W_buf_size);
        l1_buffers.y_vec      = (uint32_t*)(uintptr_t)flex_l1_malloc(y_tile_size);

        // Load the input vector x (only needed once)
        debug("[DEBUG][DMA] Load embeddign vector x to l1\n");
        flex_timer_start();
        flex_dma_async_1d((uint64_t)(uintptr_t)l1_buffers.x_vec, (uint64_t)(uintptr_t)&matrix_activation_fp16[0],
                          FP16_M * sizeof(uint16_t));
        flex_dma_async_wait_all();
        flex_timer_end();
    }
    flex_intra_cluster_sync();

    PDEBUG("\n=== STARTING DOUBLE-BUFFERED GEMV (NUM_TILES=%u) ===\n", NUM_TILES);

    int curIdx = 0;

    // Preload first tile's indices
    if (NUM_TILES > 0 && flex_is_dm_core() && CID == 0) {
        debug("[DMA] Preloading indices for tile 0 to idx_buf[0]\n");
        flex_timer_start();
        load_indices_tile((void*)l1_buffers.idx_buf[0], &matrix_idx_packed_uint16[0], 0);
        flex_timer_end();
    }
    flex_intra_cluster_sync();

    // loop over B_tile
    for (uint32_t bt = 0; bt < NUM_TILES; ++bt) {
        const uint32_t g         = get_groups_for_tile(bt);
        const uint32_t P         = g * VQ_GROUP_SIZE;
        const uint32_t col_start = get_start_group_for_tile(bt) * VQ_GROUP_SIZE;

        // Step 1: Dequantize current tile (indices already loaded)
        if (core_id == SPATZ_CORE && CID == 0) {
            // debug(" \t[SPATZ] Dequantizing tile %u using idx_buf[%d]\n\t", bt, curIdx);
            g_l1_dq.indices = (uint32_t*)l1_buffers.idx_buf[curIdx];
            // g_l1_dq.W_dq    = W_tile;
            dequantize_block_tile_compact(0, FP16_M, g, g);
            // flex_redmule_config( 1, FP16_M, P);// hardcode r N P
        }

        // Overlap: Prefetch next tile's indices while dequantizing
        if (bt + 1 < NUM_TILES && flex_is_dm_core() && CID == 0) {
            int nextIdx = curIdx ^ 1;
            debug(" \t[DMA] Load indices for tile %u to B[%d] (overlapped with dequant)\n\t", bt + 1,
                  nextIdx);
            // flex_timer_start();
            load_indices_tile((void*)l1_buffers.idx_buf[nextIdx], &matrix_idx_packed_uint16[0], bt + 1);
            // flex_timer_end();
            debug(" \t[SPATZ] Dequantize tile %u via idx_buf[%d]\n\t", bt, curIdx);
        }

        flex_intra_cluster_sync();

        // Step 2: Compute x^T * W_t -> y_t(vertical)
        if (core_id == SPATZ_CORE && CID == 0) {
            debug("\t[REDMULE] Computing x^T * W_tile%u: (1x%u) * (%ux%u) -> (1x%u) \n\t", bt,
                  FP16_M, FP16_M, P, P);

            uint32_t y_tile_size = max_P * sizeof(uint16_t);
            uint16_t* c_ptr      = (uint16_t*)(uintptr_t)l1_buffers.y_vec;
            for (int i = 0; i < y_tile_size; ++i) { // TODO temporary solution,fix later
                c_ptr[i] = 0;
                //         // debug("0x%04x ", c_ptr[i]);
            }
            flex_timer_start();

            flex_redmule_config(1, FP16_M, P); // hardcode r N P

            flex_redmule_trigger((uint32_t)l1_buffers.x_vec, (uint32_t)g_l1_dq.W_dq, (uint32_t)l1_buffers.y_vec,
                                 REDMULE_FP_16);
            flex_redmule_wait();
            flex_timer_end();
            debug("\t[SPATZ] Computing \n\t");
            flex_timer_start();
            spatz_gemv_fp16_full_legacy((uint16_t*)l1_buffers.x_vec, // x vector [1 x M]
                                          (uint16_t*)g_l1_dq.W_dq,     // W_tile [M x P]
                                          (uint16_t*)l1_buffers.y_vec, // y_tile [1 x P]
                                          1, FP16_M, P);
            flex_timer_end();
            debug("\t[SPATZ] Computing  8 elements\n\t");
            flex_timer_start();

            spatz_matmul_fp16_8wide((uint16_t*)l1_buffers.x_vec, // x vector [1 x M]
                                    (uint16_t*)g_l1_dq.W_dq,     // W_tile [M x P]
                                    (uint16_t*)l1_buffers.y_vec, // y_tile [1 x P]
                                    1, FP16_M, P);
            flex_timer_end();
        }

        // crct: Wait for compute to finish before DMA can read C_tile
        flex_intra_cluster_sync();
        // Step 3: Store result to HBM
        if (flex_is_dm_core() && CID == 0) {
            const uint64_t dst = C_hbm_base + col_start * sizeof(uint16_t);
            debug("\t[DMA] Storing y_tile%u to HBM at offset %u\n\t", bt, col_start);
            flex_timer_start();
            flex_dma_async_1d(dst, (uint64_t)(uintptr_t)l1_buffers.y_vec, P * sizeof(uint16_t));
            flex_dma_async_wait_all();
            flex_timer_end();
        }

        // Wait for DMA to finish before reusing C_tile in next iteration
        flex_intra_cluster_sync();

        // Swap index buffer for next iteration
        curIdx ^= 1;
    }

    // Verification
    if (flex_is_dm_core() && CID == 0) {
        // debug("\n[VERIFICATION] Double-buffered GEMV complete!\n");
        uint16_t* hbm_result_ptr = (uint16_t*)(uintptr_t)C_hbm_base;
        // debug("[DEBUG] Result vector is in HBM at 0x%08x\n", (uint32_t)L1_PTRS.C_hbm_base);
        // debug("[DEBUG] First 8 computed results: ");
        // for (int i = 0; i < 8 && i < FP16_K; i++) {
        //     debug("0x%04x ", hbm_result_ptr[i]);
        // }
        spatz_verify_16(FP16_K, hbm_result_ptr, (uint16_t*)matrix_golden_fp16, 0.25f);

        debug("\n");
        debug("[VERIFICATION] GEMV verification complete!\n");
    }
}
#endif

int main() {
    uint32_t eoc_val = 0;
    flex_barrier_xy_init();
    flex_global_barrier_xy();
    flex_alloc_init();
    flex_intra_cluster_sync(); // Cluster barrier
    flex_global_barrier_xy();
    // if(FP16_M*FP16_N*3 >ARCH_CLUSTER_TCDM_SIZE)
    // flex_eoc(eoc_val);
    /**************************************/
    /*  Program Execution Region -- Start */
    /**************************************/

    // Choose which version to run based on GEMM macro
#if GEMM == 1
    // printf("[INFO] Running double-buffered GEMM with pipelined execution\n");
    double_buffer_gemm();
#elif GEMV == 1
    // printf("[INFO] Running double-buffered GEMV\n");
    double_buffer_gemv();

#endif
    /**************************************/
    /*  Program Execution Region -- Stop  */
    /**************************************/
    flex_global_barrier_xy();
    if (flex_get_core_id() == 0 && flex_get_cluster_id() == 0)
        printf("\nfinished!");
    flex_global_barrier_xy();
    flex_eoc(eoc_val);
    return 0;
}