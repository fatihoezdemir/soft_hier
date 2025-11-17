#include "dq_compute_spatz.h" // spatz computation kernels
#include "dq_data_hbm.h"
#include "dq_load_data_l1.h"
#include "flex_alloc.h"
#include "flex_dma_pattern.h"
#include "flex_libfp16.h"
#include "flex_libfp8.h"
#include "flex_redmule.h"

#include <stdbool.h>
extern const int SPATZ_CORE;
extern const int REDMULE_ATTACHED_CORE;

// Tiling configuration TODO make a tilinginfo struct
extern const int NUM_TILES;
#if GEMM == 1
typedef struct {

    volatile uint16_t* idx0_buf_packed[2];
    volatile uint8_t* idx0_buf[2];
    volatile uint8_t* idx_buf[2];
    volatile uint8_t* idx1_buf[2];
    volatile uint16_t* W_dq_buf[2];
    volatile uint16_t* A_buf[2];      //  buffers for activation tiles
    volatile uint16_t* activation;    //__attribute__((section(".l1_prio")));
    volatile uint16_t* C_tile;        // Result tile buffer //TODO delete this
    volatile uint16_t* C_tile_buf[2]; // Result tile buffer
    volatile uint64_t C_hbm_base;
} L1_buffers;
L1_buffers l1_buffers;
L1_DQ_Handles g_l1_dq = {0};
// Helper function to allocate double-buffered L1 memory for uint16_t indices
static inline void allocate_l1_buffers_u16(L1_buffers* buffers, uint32_t idx_buf_size, uint32_t A_buf_size,
                                           uint32_t W_buf_size, uint32_t C_buf_size) {
    for (int i = 0; i < 2; ++i) {
        buffers->idx0_buf_packed[i] = (uint16_t*)(uintptr_t)flex_l1_malloc(idx_buf_size);
        buffers->W_dq_buf[i]        = (uint16_t*)(uintptr_t)flex_l1_malloc(W_buf_size);
        buffers->C_tile_buf[i]      = (uint16_t*)(uintptr_t)flex_l1_malloc(C_buf_size);
        buffers->A_buf[i]           = (uint16_t*)(uintptr_t)flex_l1_malloc(A_buf_size);
    }
}

// Helper function to allocate double-buffered L1 memory for uint8_t indices
static inline void allocate_l1_buffers_u8(L1_buffers* buffers, uint32_t idx_buf_size, uint32_t A_buf_size,
                                          uint32_t W_buf_size, uint32_t C_buf_size) {
    for (int i = 0; i < 2; ++i) {
        buffers->idx0_buf[i]   = (uint8_t*)(uintptr_t)flex_l1_malloc(idx_buf_size);
        buffers->idx1_buf[i]   = (uint8_t*)(uintptr_t)flex_l1_malloc(idx_buf_size);
        buffers->W_dq_buf[i]   = (uint16_t*)(uintptr_t)flex_l1_malloc(W_buf_size);
        buffers->C_tile_buf[i] = (uint16_t*)(uintptr_t)flex_l1_malloc(C_buf_size);
        buffers->A_buf[i]      = (uint16_t*)(uintptr_t)flex_l1_malloc(A_buf_size);
    }
}

void dq_gemm_triple_buffer_baseline() {
    uint32_t CID     = flex_get_cluster_id(); // Get cluster ID
    uint32_t core_id = flex_get_core_id();
    uint32_t max_g   = get_groups_for_tile(0);
    uint32_t max_r   = get_rows_for_tile(0);

    const uint32_t max_P = max_g * VQ_GROUP_SIZE;

    // allocate necessary data in L1 and hbm
    if (flex_is_dm_core() && CID == 0) {
        const uint32_t C_bytes = FP16_M * FP16_K * sizeof(uint16_t); // Allocate HBM for Big matrix C
        l1_buffers.C_hbm_base  = (uint64_t)(uintptr_t)flex_hbm_malloc(C_bytes);

        dq_load_codebook_l1(); // Load codebooks and scales (one-time load)

        const uint32_t idx_buf_size = FP16_M * max_g * sizeof(uint16_t);
        const uint32_t A_buf_size   = max_r * FP16_N * sizeof(uint16_t);
        const uint32_t W_buf_size   = FP16_M * max_P * sizeof(uint16_t);
        const uint32_t C_buf_elems  = max_r * max_P;
        const uint32_t C_buf_size   = C_buf_elems * sizeof(uint16_t);

        // Allocate double buffers for indices and activation
        allocate_l1_buffers_u16(&l1_buffers, idx_buf_size, A_buf_size, W_buf_size, C_buf_size);
        g_l1_dq.W_dq = l1_buffers.W_dq_buf[0];
        // l1_buffers.C_tile = (uint16_t*)(uintptr_t)flex_l1_malloc(C_buf_size);

        // Initialize C buffers to zero once
        for (int i = 0; i < 2; ++i) {
            flex_dma_async_1d((uint64_t)(uintptr_t)l1_buffers.C_tile_buf[i], zomem(0), C_buf_size);
        }
        /********************************************************************************
         * PIPELINE PROLOGUE    |   iDMA                | Spatz              |
         *             ------------------------------------------------------|
         *                      | Load B0_idx           |
         *                      ---------------------------------------------|
         *                      | load A_tile0 ->A0     |  Dequantize B0->W0 |
         ********************************************************************************/

        debug("[DMA] Load B%u indices to idx_buf[%d]=0x%08x\n", 0, 0, l1_buffers.idx0_buf_packed[0]);
        g_l1_dq.indices = (uint16_t*)l1_buffers.idx0_buf_packed[0];
        dq_load_indices_tile((void*)l1_buffers.idx0_buf_packed[0], &matrix_idx_packed_uint16[0], 0);
        flex_dma_async_wait_all();
    }

    flex_intra_cluster_sync();

    int readyW = 0;

    if (core_id == SPATZ_CORE && CID == 0 && NUM_TILES > 0) {
        const uint32_t g0 = GROUPS_PER_TILE;
        g_l1_dq.indices   = l1_buffers.idx0_buf_packed[0];
        g_l1_dq.W_dq      = l1_buffers.W_dq_buf[readyW];
        dequantize_block_tile_compact(/*row_start=*/0, /*rows=*/FP16_M, /*group_count=*/g0,
                                      /*idx_groups_stride=*/g0);
    }
    if (flex_is_dm_core() && CID == 0) {
        // Prologue: DMA load A0 -> A_buf[curA] only for first B tile; subsequent tiles are prefetched
        const uint32_t g = GROUPS_PER_TILE;   // groups in this B tile
        const uint32_t P = g * VQ_GROUP_SIZE; // columns in compact W/C
        // debug("\t[DMA] Load A0 to A_buf[%d]=0x%08x for B%u\n\t", 0, l1_buffers.A_buf[0], 0);
        dq_load_activation_tile((void*)l1_buffers.A_buf[0], &matrix_activation_fp16[0],
                                /*row_tile_index=*/0);
    }
    flex_intra_cluster_sync();

    // ---------- Outer loop over tiles of B_hat ----------
    // PDEBUG("\n=== STARTING DOUBLE-BUFFERED PIPELINE (NUM_TILES=%u) ===\n", NUM_TILES);
    int curB = 0;

    // Variables to save previous iteration's store info for pipelining
    uint32_t prev_r = 0, prev_P = 0;
    uint64_t prev_dst = 0;
    int curC          = 0; // Double buffer for C_tile (track across all iterations)

    for (uint32_t bt = 0; bt < NUM_TILES; ++bt) {
        const uint32_t g = get_groups_for_tile(bt); // groups in this B tile
        const uint32_t P = g * VQ_GROUP_SIZE;       // columns in compact W/C
        int curW         = readyW;
        if (core_id == SPATZ_CORE && CID == 0) {
            g_l1_dq.W_dq    = l1_buffers.W_dq_buf[curW];
            g_l1_dq.indices = l1_buffers.idx0_buf_packed[curB];
        }
        // ---------- Inner loop over all tiles of A (double-buffer) ----------
        int curA = 0;

        for (uint32_t at = 0; at < NUM_TILES; ++at) {
            const uint32_t r          = get_rows_for_tile(at);
            const uint32_t sr         = get_start_row_for_tile(at); // row start
            const bool is_last_a_tile = (at + 1u == NUM_TILES);
            const bool has_next_bt    = (bt + 1u < NUM_TILES);

            // Calculate destination address in allocated HBM buffer
            const uint32_t col_start_elems = get_start_group_for_tile(bt) * VQ_GROUP_SIZE;
            const uint32_t row_start_elems = sr * FP16_K;
            const uint64_t dst =
                l1_buffers.C_hbm_base + (uint64_t)(row_start_elems + col_start_elems) * sizeof(uint16_t);

            if (flex_is_dm_core() && CID == 0) {
                printf("\t\t>> Tile B%u A%u: rows=%u, start_row=%u, curA_buf=%d\n", bt, at, r, sr, curA);
                int prevC = curC ^ 1; // Previous C buffer
                if (at > 0) {
                    // printf ("\n dm core %u current core %u ",flex_is_dm_core(), flex_get_core_id());
                    // Store previous iteration: at=1 stores C0, at=2 stores C1, at=3 stores C2

                    debug("\t\t[DMA][ST] Storing prev A%u x B%u from C_buf[%d] to dst=0x%08x (r=%u, P=%u)\n\t\t",
                          at - 1, bt, // ERROR HER
                          prevC, (uint32_t)prev_dst, prev_r, prev_P);

                    flex_dma_async_2d(prev_dst, (uint64_t)(uintptr_t)l1_buffers.C_tile_buf[prevC],
                                      prev_P * sizeof(uint16_t), FP16_K * sizeof(uint16_t), prev_P * sizeof(uint16_t),
                                      /*rows*/ prev_r);

                    // Zero the buffer we just stored for reuse
                    flex_dma_async_wait_all();

                    uint32_t C_buf_size = prev_r * prev_P * sizeof(uint16_t);
                    flex_dma_async_1d((uint64_t)(uintptr_t)l1_buffers.C_tile_buf[prevC], zomem(0), C_buf_size);

                } else if (at == 0 && bt > 0) {
                    // at=0: store C3 from previous bt
                    const uint32_t prev_at = NUM_TILES - 1;
                    debug("\t\t[DMA] Storing prev A%u x B%u from C_buf[%d] to dst=0x%08x (r=%u, P=%u)\n\t\t", prev_at,
                          bt - 1, prevC, (uint32_t)prev_dst, prev_r, prev_P);

                    flex_dma_async_2d(prev_dst, (uint64_t)(uintptr_t)l1_buffers.C_tile_buf[prevC],
                                      prev_P * sizeof(uint16_t), FP16_K * sizeof(uint16_t), prev_P * sizeof(uint16_t),
                                      /*rows*/ prev_r);

                    flex_dma_async_wait_all();
                    // Zero the buffer we just stored for reuse
                    uint32_t C_buf_size = prev_r * prev_P * sizeof(uint16_t);
                    flex_dma_async_1d((uint64_t)(uintptr_t)l1_buffers.C_tile_buf[prevC], zomem(0), C_buf_size);
                }
                // Ensure the store has drained before zeroing the source buffer.

                // load a tile
                if (at + 1 < NUM_TILES) { // load next a tile
                    int nxtA = curA ^ 1;  // Toggle buffer
                    dq_load_activation_tile((void*)l1_buffers.A_buf[nxtA], &matrix_activation_fp16[0], at + 1);
                } else if (is_last_a_tile &&
                           has_next_bt) { // for the last a tile iteration, load tile a0 again so the next b tile can
                                          // compute it , here we begin at row 0 again.
                    int nxtA_for_next_bt = curA ^ 1;
                    dq_load_activation_tile((void*)l1_buffers.A_buf[nxtA_for_next_bt], &matrix_activation_fp16[0],
                                            /*row_tile_index=*/0);
                }

                if (at == 0 &&
                    has_next_bt) { // at a_t=0 , load indices for next B tile already into the buffer to hide latencies
                    int nxtB = curB ^ 1; // Toggle buffer
                    dq_load_indices_tile((void*)l1_buffers.idx0_buf_packed[nxtB], &matrix_idx_packed_uint16[0], bt + 1);
                }
            }

            // REDMULE: Compute A_at × B_bt (compact P) -> C_tile_buf[curC]
            if (core_id == REDMULE_ATTACHED_CORE && CID == 0) {
                l1_buffers.activation = (uint16_t*)l1_buffers.A_buf[curA];

                // flex_timer_start();
                flex_redmule_config(r, P, FP16_N);
                flex_redmule_trigger((uint32_t)l1_buffers.A_buf[curA], (uint32_t)g_l1_dq.W_dq,
                                     (uint32_t)l1_buffers.C_tile_buf[curC], REDMULE_FP_16);
                flex_redmule_wait();

                // flex_timer_end();
            }

            // SPATZ: Dequantize next B while DMA stores (overlap!)
            if (core_id == SPATZ_CORE && CID == 0 && is_last_a_tile && has_next_bt) {
                const int nxtB        = curB ^ 1;
                const int targetW     = curW ^ 1;
                const uint32_t g_next = get_groups_for_tile(bt + 1);
                g_l1_dq.indices       = l1_buffers.idx0_buf_packed[nxtB];
                g_l1_dq.W_dq          = l1_buffers.W_dq_buf[targetW];
                dequantize_block_tile_compact(/*row_start=*/0, /*rows=*/FP16_M,
                                              /*group_count=*/g_next, /*idx_groups_stride=*/g_next);

                readyW ^= 1;
            }

            // Save current iteration's parameters for next iteration's store
            if (flex_is_dm_core() && CID == 0) {
                // debug(" \n\tfinished B%d A%d iteration of a \n", bt,at);
                prev_r   = r;
                prev_P   = P;
                prev_dst = dst;
            }
            if (flex_is_dm_core() && CID == 0) {
                debug("finished iteration\n");
                flex_dma_async_wait_all();
            }
            flex_intra_cluster_sync(); // Final sync before swapping buffers
            // Swap A_buf and C_buf for next iteration
            curA ^= 1;
            curC ^= 1;
        }
        // Swap idx_buf B for next outer iteration
        curB ^= 1;
    }
    // PIPELINE EPILOGUE - Store the very last C_tile (A[NUM_TILES-1] × B[NUM_TILES-1])
    if (flex_is_dm_core() && CID == 0) {
        // After all loops, curC has been toggled, so the last result is in curC^1
        int final_C_buf = curC ^ 1;
        flex_dma_async_2d(prev_dst, (uint64_t)(uintptr_t)l1_buffers.C_tile_buf[final_C_buf], prev_P * sizeof(uint16_t),
                          FP16_K * sizeof(uint16_t), // Full C matrix width
                          prev_P * sizeof(uint16_t), //  tile width
                          /*rows*/ prev_r);
        flex_dma_async_wait_all();
    }
    flex_intra_cluster_sync(); // Final sync before swapping buffers
    // Verify computation results against golden reference
    if (flex_is_dm_core() && CID == 0) {
        debug("\n[VERIFICATION] Double-buffered GEMM complete! Verifying results...\n");

        //  L1_PTRS.result points to C_tile buffer
        uint16_t* hbm_result_ptr = (uint16_t*)(uintptr_t)l1_buffers.C_hbm_base;
        // Verify matrix multiplication results against golden reference
        debug("[DEBUG] Verifying GEMM results against golden reference...\n");
#ifdef VERIFY_VALUES
        spatz_verify_16(FP16_M * FP16_N, hbm_result_ptr, (uint16_t*)matrix_golden_fp16, 1.0F);
#endif
        debug("[VERIFICATION] Verification complete!\n");

        for (int i = 0; i < 2; ++i) {
            flex_l1_free((void*)l1_buffers.W_dq_buf[i]);
            flex_l1_free((void*)l1_buffers.idx0_buf_packed[i]);
            flex_l1_free((void*)l1_buffers.A_buf[i]);
            flex_l1_free((void*)l1_buffers.C_tile_buf[i]);

            l1_buffers.W_dq_buf[i] = NULL;
        }
        g_l1_dq.W_dq = NULL;
        flex_l1_free((void*)g_l1_dq.cb);
        flex_l1_free((void*)g_l1_dq.scales);
    }
}

void dq_gemm_triple_buffer_baselineu8() {
    uint32_t CID         = flex_get_cluster_id(); // Get cluster ID
    uint32_t core_id     = flex_get_core_id();
    uint32_t max_g       = get_groups_for_tile(0);
    uint32_t max_r       = get_rows_for_tile(0);
    const uint32_t max_P = max_g * VQ_GROUP_SIZE;
    /********************************************************************************
     * PIPELINE PROLOGUE    |   iDMA                | Spatz              |
     *             ------------------------------------------------------|
     *                      | Load B0_idx           |
     *                      ---------------------------------------------|
     *                      | load A_tile0 ->A0     |  Dequantize B0->W0 |
     ********************************************************************************/
    // allocate necessary data in L1
    if (flex_is_dm_core() && CID == 0) {
        debug("[DEBUG][DMA] Allocating double buffers for pipelined execution\n");

        // Allocate HBM for result matrix C
        const uint32_t C_bytes = FP16_M * FP16_K * sizeof(uint16_t);
        l1_buffers.C_hbm_base  = (uint64_t)(uintptr_t)flex_hbm_malloc(C_bytes);

        dq_load_codebook_l1(); // Load codebooks and scales (one-time load)

        const uint32_t idx_buf_size = FP16_M * max_g * sizeof(uint8_t);
        const uint32_t A_buf_size   = max_r * FP16_N * sizeof(uint16_t);
        const uint32_t W_buf_size   = FP16_M * max_P * sizeof(uint16_t);
        const uint32_t C_buf_elems  = max_r * max_P;
        const uint32_t C_buf_size   = C_buf_elems * sizeof(uint16_t);

        // Allocate double buffers for indices and activation (uint8_t indices)
        allocate_l1_buffers_u8(&l1_buffers, idx_buf_size, A_buf_size, W_buf_size, C_buf_size);
        g_l1_dq.W_dq = l1_buffers.W_dq_buf[0];

        // Initialize C buffers to zero once
        for (int i = 0; i < 2; ++i) {
            flex_dma_async_1d((uint64_t)(uintptr_t)l1_buffers.C_tile_buf[i], zomem(0), C_buf_size);
        }

        debug("[DMA] Load B%u indices to idx_buf[%d]=0x%08x\n", 0, 0, l1_buffers.idx0_buf_packed[0]);
        // g_l1_dq.indices = (uint16_t*)l1_buffers.idx0_buf_packed[0];
        g_l1_dq.indices0_u8 = l1_buffers.idx0_buf[0];
        g_l1_dq.indices1_u8 = l1_buffers.idx1_buf[0];
        // dq_load_indices_tile((void*)l1_buffers.idx0_buf_packed[0], &matrix_idx_packed_uint16[0], 0);
        dq_load_indices_tile_u8((void*)l1_buffers.idx0_buf[0], &matrix_idx0_uint8[0], 0);
        dq_load_indices_tile_u8((void*)l1_buffers.idx1_buf[0], &matrix_idx1_uint8[0], 0);

        flex_dma_async_wait_all();
    }

    flex_intra_cluster_sync();

    int readyW = 0;

    if (core_id == SPATZ_CORE && CID == 0 && NUM_TILES > 0) {
        const uint32_t g0   = GROUPS_PER_TILE;
        g_l1_dq.indices0_u8 = l1_buffers.idx0_buf[0];
        g_l1_dq.indices1_u8 = l1_buffers.idx1_buf[0];
        g_l1_dq.W_dq        = l1_buffers.W_dq_buf[readyW];
        // dequantize_block_tile_compactu8(/*row_start=*/0, /*rows=*/FP16_M, /*group_count=*/g0,
        //                               /*idx_groups_stride=*/g0);
        dequantize_block_tile_compact_improved(0, FP16_M, g0, g0);
    }
    if (flex_is_dm_core() && CID == 0) {
        // Prologue: DMA load A0 -> A_buf[curA] only for first B tile; subsequent tiles are prefetched
        const uint32_t g = GROUPS_PER_TILE;   // groups in this B tile
        const uint32_t P = g * VQ_GROUP_SIZE; // columns in compact W/C
        // debug("\t[DMA] Load A0 to A_buf[%d]=0x%08x for B%u\n\t", 0, l1_buffers.A_buf[0], 0);
        dq_load_activation_tile((void*)l1_buffers.A_buf[0], &matrix_activation_fp16[0],
                                /*row_tile_index=*/0);
    }
    flex_intra_cluster_sync();

    // ---------- Outer loop over tiles of B_hat ----------
    // PDEBUG("\n=== STARTING DOUBLE-BUFFERED PIPELINE (NUM_TILES=%u) ===\n", NUM_TILES);
    int curB = 0;

    // Variables to save previous iteration's store info for pipelining
    uint32_t prev_r = 0, prev_P = 0;
    uint64_t prev_dst = 0;
    int curC          = 0; // Double buffer for C_tile (track across all iterations)

    for (uint32_t bt = 0; bt < NUM_TILES; ++bt) {
        const uint32_t g = get_groups_for_tile(bt); // groups in this B tile
        const uint32_t P = g * VQ_GROUP_SIZE;       // columns in compact W/C
        int curW         = readyW;
        if (core_id == SPATZ_CORE && CID == 0) {
            g_l1_dq.W_dq        = l1_buffers.W_dq_buf[curW];
            g_l1_dq.indices0_u8 = l1_buffers.idx0_buf[curB];
            g_l1_dq.indices1_u8 = l1_buffers.idx1_buf[curB];
        }
        // ---------- Inner loop over all tiles of A (double-buffer) ----------
        int curA = 0;

        for (uint32_t at = 0; at < NUM_TILES; ++at) {
            const uint32_t r          = get_rows_for_tile(at);
            const uint32_t sr         = get_start_row_for_tile(at); // row start
            const bool is_last_a_tile = (at + 1u == NUM_TILES);
            const bool has_next_bt    = (bt + 1u < NUM_TILES);

            // Calculate destination address in allocated HBM buffer
            const uint32_t col_start_elems = get_start_group_for_tile(bt) * VQ_GROUP_SIZE;
            const uint32_t row_start_elems = sr * FP16_K;
            const uint64_t dst =
                l1_buffers.C_hbm_base + (uint64_t)(row_start_elems + col_start_elems) * sizeof(uint16_t);

            if (flex_is_dm_core() && CID == 0) {
                debug("\t\t>> Tile B%u A%u: rows=%u, start_row=%u, curA_buf=%d\n", bt, at, r, sr, curA);
                int prevC = curC ^ 1; // Previous C buffer
                if (at > 0) {
                    // printf ("\n dm core %u current core %u ",flex_is_dm_core(), flex_get_core_id());
                    // Store previous iteration: at=1 stores C0, at=2 stores C1, at=3 stores C2

                    debug("\t\t[DMA][ST] Storing prev A%u x B%u from C_buf[%d] to dst=0x%08x (r=%u, P=%u)\n\t\t",
                          at - 1, bt, // ERROR HER
                          prevC, (uint32_t)prev_dst, prev_r, prev_P);

                    TIMER_START();
                    flex_dma_async_2d(prev_dst, (uint64_t)(uintptr_t)l1_buffers.C_tile_buf[prevC],
                                      prev_P * sizeof(uint16_t), FP16_K * sizeof(uint16_t), prev_P * sizeof(uint16_t),
                                      /*rows*/ prev_r);
                    TIMER_END();
                    // Ensure the store has drained before zeroing the source buffer.
                    flex_dma_async_wait_all();
                    // Zero the buffer we just stored for reuse
                    TIMER_START();
                    uint32_t C_buf_size = prev_r * prev_P * sizeof(uint16_t);
                    flex_dma_async_1d((uint64_t)(uintptr_t)l1_buffers.C_tile_buf[prevC], zomem(0), C_buf_size);
                    TIMER_END();
                } else if (at == 0 && bt > 0) {
                    // at=0: store C3 from previous bt
                    const uint32_t prev_at = NUM_TILES - 1;
                    debug("\t\t[DMA] Storing prev A%u x B%u from C_buf[%d] to dst=0x%08x (r=%u, P=%u)\n\t\t", prev_at,
                          bt - 1, prevC, (uint32_t)prev_dst, prev_r, prev_P);
                    TIMER_START();
                    flex_dma_async_2d(prev_dst, (uint64_t)(uintptr_t)l1_buffers.C_tile_buf[prevC],
                                      prev_P * sizeof(uint16_t), FP16_K * sizeof(uint16_t), prev_P * sizeof(uint16_t),
                                      /*rows*/ prev_r);
                    TIMER_END();
                    // Ensure the store has drained before zeroing the source buffer.
                    flex_dma_async_wait_all();
                    // Zero the buffer we just stored for reuse
                    uint32_t C_buf_size = prev_r * prev_P * sizeof(uint16_t);
                    flex_dma_async_1d((uint64_t)(uintptr_t)l1_buffers.C_tile_buf[prevC], zomem(0), C_buf_size);
                }

                // load a tile
                if (at + 1 < NUM_TILES) { // load next a tile
                    int nxtA = curA ^ 1;  // Toggle buffer
                    dq_load_activation_tile((void*)l1_buffers.A_buf[nxtA], &matrix_activation_fp16[0], at + 1);
                } else if (is_last_a_tile &&
                           has_next_bt) { // for the last a tile iteration, load tile a0 again so the next b tile can
                                          // compute it , here we begin at row 0 again.
                    int nxtA_for_next_bt = curA ^ 1;
                    dq_load_activation_tile((void*)l1_buffers.A_buf[nxtA_for_next_bt], &matrix_activation_fp16[0],
                                            /*row_tile_index=*/0);
                }

                if (at == 0 &&
                    has_next_bt) { // at a_t=0 , load indices for next B tile already into the buffer to hide latencies
                    int nxtB = curB ^ 1; // Toggle buffer
                    dq_load_indices_tile_u8((void*)l1_buffers.idx0_buf[nxtB], &matrix_idx0_uint8[0], bt + 1);
                    dq_load_indices_tile_u8((void*)l1_buffers.idx1_buf[nxtB], &matrix_idx1_uint8[0], bt + 1);
                }
            }

            // REDMULE: Compute A_at × B_bt (compact P) -> C_tile_buf[curC]
            if (core_id == REDMULE_ATTACHED_CORE && CID == 0) {
                l1_buffers.activation = (uint16_t*)l1_buffers.A_buf[curA];

                // flex_timer_start();
                flex_redmule_config(r, FP16_N, P);
                flex_redmule_trigger((uint32_t)l1_buffers.A_buf[curA], (uint32_t)g_l1_dq.W_dq,
                                     (uint32_t)l1_buffers.C_tile_buf[curC], REDMULE_FP_16);
                flex_redmule_wait();

                // flex_timer_end();
            }

            // SPATZ: Dequantize next B while DMA stores (overlap!)
            if (core_id == SPATZ_CORE && CID == 0 && is_last_a_tile && has_next_bt) {
                const int nxtB        = curB ^ 1;
                const int targetW     = curW ^ 1;
                const uint32_t g_next = get_groups_for_tile(bt + 1);
                g_l1_dq.indices0_u8   = l1_buffers.idx0_buf[nxtB];
                g_l1_dq.indices1_u8   = l1_buffers.idx1_buf[nxtB];
                g_l1_dq.W_dq          = l1_buffers.W_dq_buf[targetW];
                // dequantize_block_tile_compactu8(/*row_start=*/0, /*rows=*/FP16_M,
                //                               /*group_count=*/g_next, /*idx_groups_stride=*/g_next);
                dequantize_block_tile_compact_improved(0, FP16_M, g_next, g_next);
                readyW ^= 1;
            }

            // Save current iteration's parameters for next iteration's store
            if (flex_is_dm_core() && CID == 0) {
                // debug(" \n\tfinished B%d A%d iteration of a \n", bt,at);
                prev_r   = r;
                prev_P   = P;
                prev_dst = dst;
            }
            if (flex_is_dm_core() && CID == 0) {
                flex_dma_async_wait_all();
            }
            flex_intra_cluster_sync(); // Final sync before swapping buffers
            // Swap A_buf and C_buf for next iteration
            curA ^= 1;
            curC ^= 1;
        }
        // Swap idx_buf B for next outer iteration
        curB ^= 1;
    }

    // flex_intra_cluster_sync();

    // PIPELINE EPILOGUE - Store the very last C_tile (A[NUM_TILES-1] × B[NUM_TILES-1])
    if (flex_is_dm_core() && CID == 0) {
        // After all loops, curC has been toggled, so the last result is in curC^1
        int final_C_buf = curC ^ 1;
        // debug("\t\t[DEBUG][DMA][EPILOGUE] curC=%d, final_C_buf=%d\n\t\t", curC, final_C_buf);
        // debug("\t\t[DEBUG][DMA][EPILOGUE] Storing final A%u x B%u from C_buf[%d] to dst=0x%08x (r=%u, P=%u)\n\t\t",
        // flex_timer_start();
        flex_dma_async_2d(prev_dst, (uint64_t)(uintptr_t)l1_buffers.C_tile_buf[final_C_buf], prev_P * sizeof(uint16_t),
                          FP16_K * sizeof(uint16_t), // Full C matrix width
                          prev_P * sizeof(uint16_t), //  tile width
                          /*rows*/ prev_r);
        flex_dma_async_wait_all();
        // flex_timer_end();
    }
    flex_intra_cluster_sync(); // Final sync before swapping buffers
    // Verify computation results against golden reference
    if (flex_is_dm_core() && CID == 0) {
        debug("\n[VERIFICATION] Double-buffered GEMM complete! Verifying results...\n");

        //  L1_PTRS.result points to C_tile buffer
        uint16_t* hbm_result_ptr = (uint16_t*)(uintptr_t)l1_buffers.C_hbm_base;
        // debug("[DEBUG] Result matrix is in HBM at 0x%08x\n", (uint32_t)C_hbm_base);
        // debug("[DEBUG] First 8 computed results from HBM: ");
        // for (int i = 0; i < 8; i++) {
        //     debug("0x%04x ", hbm_result_ptr[i]);
        // }
        // debug("\n");

        // Verify matrix multiplication results against golden reference
        debug("[DEBUG] Verifying GEMM results against golden reference...\n");
#if VERIFY_VALUES == 1
        spatz_verify_16(FP16_M * FP16_N, hbm_result_ptr, (uint16_t*)matrix_golden_fp16, 1.0F);
#endif

        debug("[VERIFICATION] Verification complete!\n");
        debug("[VERIFICATION] Freeing allocated buffers!\n");

        for (int i = 0; i < 2; ++i) {
            flex_l1_free((void*)l1_buffers.idx0_buf[i]);
            flex_l1_free((void*)l1_buffers.idx1_buf[i]);
            flex_l1_free((void*)l1_buffers.A_buf[i]);
            flex_l1_free((void*)l1_buffers.C_tile_buf[i]);
            flex_l1_free((void*)l1_buffers.W_dq_buf[i]);
            l1_buffers.W_dq_buf[i] = NULL;
        }
        g_l1_dq.W_dq = NULL;
        flex_l1_free((void*)g_l1_dq.cb);
        flex_l1_free((void*)g_l1_dq.scales);
        debug("[VERIFICATION] Freeing finished!\n");
    }
    flex_intra_cluster_sync(); // Sync all cores before exit
}

#endif
