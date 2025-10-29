#include "dq_compute_spatz.h" // spatz computation kernels
#include "dq_data_hbm.h"
#include "dq_load_data_l1.h"
#include "flex_alloc.h"
#include "flex_dma_pattern.h"
#include "flex_libfp16.h"
#include "flex_libfp8.h"
#include "flex_redmule.h"
extern const int SPATZ_CORE;
extern const int DOUBLEBUFFER;
// Tiling configuration TODO make a tilinginfo struct
extern const int NUM_TILES;
#if GEMM == 1
typedef struct {

    volatile uint16_t* idx0_buf_packed[2];
    volatile uint8_t* idx0_buf[2];
    volatile uint8_t* idx1_buf[2];
    volatile uint16_t* A_buf[2];      //  buffers for activation tiles
    volatile uint16_t* activation;    //__attribute__((section(".l1_prio")));
    volatile uint16_t* C_tile;        // Result tile buffer
    volatile uint16_t* C_tile_buf[2]; // Result tile buffer
    volatile uint64_t C_hbm_base;
} L1_buffers;
L1_buffers l1_buffers;
L1_DQ_Handles g_l1_dq = {0};

void dq_gemm_double_buffer_baseline() {       // TODO currently only load is overlapped,also overlap store
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
        l1_buffers.C_hbm_base  = (uint64_t)(uintptr_t)flex_hbm_malloc(C_bytes);

        // Load codebooks and scales (one-time load)
        dq_load_codebook_l1();

        const uint32_t idx_buf_size = FP16_M * max_g * sizeof(uint16_t);
        const uint32_t A_buf_size   = max_r * FP16_N * sizeof(uint16_t);
        const uint32_t W_buf_size   = FP16_M * max_P * sizeof(uint16_t);
        const uint32_t C_buf_elems  = max_r * max_P;
        const uint32_t C_buf_size   = C_buf_elems * sizeof(uint16_t);

        // Allocate  buffers for indices and activation
        l1_buffers.idx0_buf_packed[0]    = (uint16_t*)(uintptr_t)flex_l1_malloc(idx_buf_size);
        l1_buffers.idx0_buf_packed[1]    = (uint16_t*)(uintptr_t)flex_l1_malloc(idx_buf_size);
        l1_buffers.A_buf[0]      = (uint16_t*)(uintptr_t)flex_l1_malloc(A_buf_size);
        l1_buffers.A_buf[1]      = (uint16_t*)(uintptr_t)flex_l1_malloc(A_buf_size);
        l1_buffers.C_tile_buf[0] = (uint16_t*)(uintptr_t)flex_l1_malloc(C_buf_size);
        l1_buffers.C_tile_buf[1] = (uint16_t*)(uintptr_t)flex_l1_malloc(C_buf_size);
        // Allocate double buffers for dequantized weights and single result tile
        g_l1_dq.W_dq_buf[0] = (uint16_t*)(uintptr_t)flex_l1_malloc(W_buf_size);
        g_l1_dq.W_dq_buf[1] = (uint16_t*)(uintptr_t)flex_l1_malloc(W_buf_size);
        g_l1_dq.W_dq        = g_l1_dq.W_dq_buf[0];
        l1_buffers.C_tile = (uint16_t*)(uintptr_t)flex_l1_malloc(C_buf_size);
    }
    int curB = 0;

        // (Step 0) DMA: Load indices for B_t -> idx_buf[curB]
    flex_intra_cluster_sync();
    if (flex_is_dm_core() && CID == 0) {
        const uint32_t g = get_groups_for_tile(0); // groups in this B tile
        const uint32_t P = g * VQ_GROUP_SIZE;       // columns in compact W/C
        debug("\t>>> B-TILE %u: groups=%u, P=%u, curB_buf=%d\n", 0, g, P, 0);
        debug("\t[DMA] Load B%u indices to idx_buf[%d]=0x%08x\n\t", 0, 0, l1_buffers.idx0_buf_packed[0]);
        g_l1_dq.indices = (uint16_t*)l1_buffers.idx0_buf_packed[0];
        dq_load_indices_tile((void*)l1_buffers.idx0_buf_packed[0], &matrix_idx_packed_uint16[0], 0);
    }
    flex_intra_cluster_sync();
    if (core_id == SPATZ_CORE && CID == 0) {
        const uint32_t g = get_groups_for_tile(0); // groups in this B tile
        // debug("\t[SPATZ] Dequantizing B%u\n\t", bt);
        g_l1_dq.W_dq    = g_l1_dq.W_dq_buf[curB];
        g_l1_dq.indices = l1_buffers.idx0_buf_packed[curB];
        dequantize_block_tile_compact(/*row_start=*/0, /*rows=*/FP16_M,
                                      /*group_count=*/g, /*idx_groups_stride=*/g);
    }
    // ---------- Outer loop over tiles of B_hat ----------
    // PDEBUG("\n=== STARTING DOUBLE-BUFFERED PIPELINE (NUM_TILES=%u) ===\n", NUM_TILES);
    for (uint32_t bt = 0; bt < NUM_TILES; ++bt) {
        const uint32_t g = get_groups_for_tile(bt); // groups in this B tile
        const uint32_t P = g * VQ_GROUP_SIZE;       // columns in compact W/C

        if (core_id == SPATZ_CORE && CID == 0) {
            g_l1_dq.W_dq    = g_l1_dq.W_dq_buf[curB];
            g_l1_dq.indices = l1_buffers.idx0_buf_packed[curB];
        }

        // ---------- Inner loop over all tiles of A (double-buffer) ----------
        int curA = 0;
        int curC = 0;
        // Prologue: DMA load A0 -> A_buf[curA] A0
        if (flex_is_dm_core() && CID == 0) {
            debug("\t[DMA] Load A0 to A_buf[%d]=0x%08x for B%u\n\t", curA, l1_buffers.A_buf[curA], bt);
            dq_load_activation_tile((void*)l1_buffers.A_buf[0], &matrix_activation_fp16[0],
                                    /*row_tile_index=*/0);
            debug("\t[SPATZ] Dequantizing\n\t");
        }
        // TODO C tile

        flex_intra_cluster_sync();
        for (uint32_t at = 0; at < NUM_TILES; ++at) {
            const uint32_t r  = get_rows_for_tile(at);
            const uint32_t sr = get_start_row_for_tile(at); // row start


            
            // Calculate destination address in allocated HBM buffer
            const uint32_t col_start_elems = get_start_group_for_tile(bt) * VQ_GROUP_SIZE;
            const uint32_t row_start_elems = sr * FP16_K;
            const uint64_t dst =
            l1_buffers.C_hbm_base + (uint64_t)(row_start_elems + col_start_elems) * sizeof(uint16_t);
            
            // DMA: Prefetch next A tile (overlaps with Spatz compute)
            if (at + 1 < NUM_TILES && flex_is_dm_core() && CID == 0) {
                int nxtA = curA ^ 1; // Toggle buffer
                debug("\t\t>> A-0 %u: rows=%u, start_row=%u, curA_buf=%d\n", at, r, sr, curA);
                debug("\t\t[DMA] Load A%u to A_buf[%d]=0x%08x (while computing)\t\t", at + 1, nxtA,
                      l1_buffers.A_buf[nxtA]);
                dq_load_activation_tile((void*)l1_buffers.A_buf[nxtA], &matrix_activation_fp16[0], at + 1);
            }

            // While processing first A tile, prefetch next B indices
            if (at == 0 && bt + 1 < NUM_TILES && flex_is_dm_core() && CID == 0) {
                int nxtB = curB ^ 1; // Toggle buffer
                debug("\t\t[DMA] Prefetching B%u indices to idx_buf[%d]=0x%08x\n\t\t", bt + 1, nxtB,
                      l1_buffers.idx0_buf_packed[nxtB]);
                dq_load_indices_tile((void*)l1_buffers.idx0_buf_packed[nxtB], &matrix_idx_packed_uint16[0], bt + 1);

                bare_dma_wait_all();
            }


            // REDMULE: Compute A_at × B_bt (compact P) -> C_tile
            if (core_id == 2 && CID == 0) {
                l1_buffers.activation = (uint16_t*)l1_buffers.A_buf[curA];
                g_l1_dq.W_dq          = g_l1_dq.W_dq_buf[curB];

                // debug("    [REDMULE] Computing A[%u×%u] × B[%u×%u] -> C[%u×%u]\n",
                //     r, FP16_N, FP16_N, P, r, P);
                uint32_t C_buf_elems = max_r * max_P;
                uint32_t C_buf_size  = C_buf_elems * sizeof(uint16_t);

                flex_redmule_config(r, FP16_N, P);
                flex_dma_async_1d((uint64_t) (uintptr_t) l1_buffers.C_tile, zomem(0), C_buf_size);
                flex_dma_async_wait_all();
                debug("\t\t[REDMULE] Compute\n\t\t");
                flex_timer_start();
                flex_redmule_trigger((uint32_t)l1_buffers.A_buf[curA], (uint32_t)g_l1_dq.W_dq,
                                     (uint32_t)l1_buffers.C_tile, REDMULE_FP_16);
                flex_redmule_wait();
                flex_timer_end();
            }
            if (core_id == SPATZ_CORE && CID == 0 && at == NUM_TILES - 1 && bt + 1 < NUM_TILES) {//for each last a tile iteration, dequantize(overlaps with redumuel)
                int nxtB             = curB ^ 1; // Toggle buffer
                const uint32_t next_g = get_groups_for_tile(bt + 1);
                g_l1_dq.indices       = l1_buffers.idx0_buf_packed[nxtB];
                g_l1_dq.W_dq          = g_l1_dq.W_dq_buf[nxtB];
                dequantize_block_tile_compact(/*row_start=*/0, /*rows=*/FP16_M,
                                              /*group_count=*/next_g, /*idx_groups_stride=*/next_g);
                g_l1_dq.W_dq    = g_l1_dq.W_dq_buf[curB];
                g_l1_dq.indices = l1_buffers.idx0_buf_packed[curB];
            }


            flex_intra_cluster_sync();

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

            flex_intra_cluster_sync();
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
        uint16_t* hbm_result_ptr = (uint16_t*)(uintptr_t)l1_buffers.C_hbm_base;
        // debug("[DEBUG] Result matrix is in HBM at 0x%08x\n", (uint32_t)C_hbm_base);
        // debug("[DEBUG] First 8 computed results from HBM: ");
        // for (int i = 0; i < 8; i++) {
        //     debug("0x%04x ", hbm_result_ptr[i]);
        // }
        // debug("\n");

        // Verify matrix multiplication results against golden reference
        debug("[DEBUG] Verifying GEMM results against golden reference...\n");
        spatz_verify_16(FP16_M * FP16_N, hbm_result_ptr, (uint16_t*)matrix_golden_fp16, 1.0F);
        debug("[VERIFICATION] Verification complete!\n");
        flex_l1_free((void *)l1_buffers.idx0_buf_packed[0]);
        flex_l1_free((void *)    l1_buffers.idx0_buf_packed[1]     );
        flex_l1_free((void *)    l1_buffers.A_buf[0]      );
        flex_l1_free((void *)    l1_buffers.A_buf[1]      );
        flex_l1_free((void *)   l1_buffers.C_tile_buf[0]    );
        flex_l1_free((void *)   l1_buffers.C_tile_buf[1]     );
        flex_l1_free((void *)   l1_buffers.C_tile    );


        flex_l1_free((void *)   g_l1_dq.W_dq_buf[0]);
        flex_l1_free((void *)   g_l1_dq.W_dq_buf[1]);
        flex_l1_free((void *)   g_l1_dq.cb    );
        flex_l1_free((void *)   g_l1_dq.scales    );


    }


}

void dq_gemm_double_buffer_baseline_extended() { // TODO currently only load is overlapped,also overlap store
    uint32_t CID     = flex_get_cluster_id();    // Get cluster ID
    uint32_t core_id = flex_get_core_id();
    uint32_t max_g   = get_groups_for_tile(0);
    uint32_t max_r   = get_rows_for_tile(0);

    const uint32_t max_P = max_g * VQ_GROUP_SIZE;

    // allocate necessary data in L1
    if (flex_is_dm_core() && CID == 0) {
        debug("[DEBUG][DMA] Allocating double buffers for pipelined execution\n");

        // Allocate HBM for result matrix C
        const uint32_t C_bytes = FP16_M * FP16_K * sizeof(uint16_t);
        l1_buffers.C_hbm_base  = (uint64_t)(uintptr_t)flex_hbm_malloc(C_bytes);

        // Load codebooks and scales (one-time load)
        dq_load_codebook_l1();

        const uint32_t idx_buf_size = FP16_M * max_g * sizeof(uint8_t);
        const uint32_t A_buf_size   = max_r * FP16_N * sizeof(uint16_t);
        const uint32_t W_buf_size   = FP16_M * max_P * sizeof(uint16_t);
        const uint32_t C_buf_elems  = max_r * max_P;
        const uint32_t C_buf_size   = C_buf_elems * sizeof(uint16_t);

        // Allocate  buffers for indices and activation
        l1_buffers.idx0_buf[0] = (uint8_t*)(uintptr_t)flex_l1_malloc(idx_buf_size);
        l1_buffers.idx0_buf[1] = (uint8_t*)(uintptr_t)flex_l1_malloc(idx_buf_size);
        l1_buffers.idx1_buf[0] = (uint8_t*)(uintptr_t)flex_l1_malloc(idx_buf_size);
        l1_buffers.idx1_buf[1] = (uint8_t*)(uintptr_t)flex_l1_malloc(idx_buf_size);
        l1_buffers.A_buf[0]      = (uint16_t*)(uintptr_t)flex_l1_malloc(A_buf_size);
        l1_buffers.A_buf[1]      = (uint16_t*)(uintptr_t)flex_l1_malloc(A_buf_size);
        l1_buffers.C_tile_buf[0] = (uint16_t*)(uintptr_t)flex_l1_malloc(C_buf_size);
        l1_buffers.C_tile_buf[1] = (uint16_t*)(uintptr_t)flex_l1_malloc(C_buf_size);
        // Allocate single buffers for dequantized weights and result tile
        g_l1_dq.W_dq      = (uint16_t*)(uintptr_t)flex_l1_malloc(W_buf_size);
        l1_buffers.C_tile = (uint16_t*)(uintptr_t)flex_l1_malloc(C_buf_size);
    }

    flex_intra_cluster_sync();

    // ---------- Outer loop over tiles of B_hat ----------
    // PDEBUG("\n=== STARTING DOUBLE-BUFFERED PIPELINE (NUM_TILES=%u) ===\n", NUM_TILES);
    int curB = 0;
    for (uint32_t bt = 0; bt < NUM_TILES; ++bt) {
        const uint32_t g = get_groups_for_tile(bt); // groups in this B tile
        const uint32_t P = g * VQ_GROUP_SIZE;       // columns in compact W/C

        PDEBUG("\t>>> B-TILE %u: groups=%u, P=%u, curB_buf=%d\n", bt, g, P, curB);

        // (Step 0) DMA: Load indices for B_t -> idx_buf[curB]
        if (flex_is_dm_core() && CID == 0) {
            debug("\t[DMA] Load B%u indices to idx_buf[%d]=0x%08x\n\t", bt, curB, l1_buffers.idx0_buf_packed[curB]);

            g_l1_dq.indices0_u8 = l1_buffers.idx0_buf[curB];
            g_l1_dq.indices1_u8 = l1_buffers.idx1_buf[curB];
            // dq_load_indices_tile((void*)l1_buffers.idx0_buf_packed[curB], &matrix_idx_packed_uint16[0], bt);
            dq_load_indices_tile_u8((void*)l1_buffers.idx0_buf[0], &matrix_idx0_uint8[0], 0);
            dq_load_indices_tile_u8((void*)l1_buffers.idx1_buf[0], &matrix_idx1_uint8[0], 0);
        }

        flex_intra_cluster_sync();

        // (Step 0) Spatz: Dequantize B_t -> W tile (compact)
        if (core_id == SPATZ_CORE && CID == 0) {
            // debug("\t[SPATZ] Dequantizing B%u\n\t", bt);

            g_l1_dq.indices0_u8 = l1_buffers.idx0_buf[curB];
            g_l1_dq.indices1_u8 = l1_buffers.idx1_buf[curB];
            dequantize_block_tile_compact_improved(0, FP16_M, g, g);
        }
        // ---------- Inner loop over all tiles of A (double-buffer) ----------
        int curA = 0;
        int curC = 0;
        // Prologue: DMA load A0 -> A_buf[curA] A0
        if (flex_is_dm_core() && CID == 0) {
            debug("\t[DMA] Load A0 to A_buf[%d]=0x%08x for B%u\n\t", curA, l1_buffers.A_buf[curA], bt);
            dq_load_activation_tile((void*)l1_buffers.A_buf[0], &matrix_activation_fp16[0],
                                    /*row_tile_index=*/0);
            debug("\t[SPATZ] Dequantizing\n\t");
        }
        // TODO C tile
        if (flex_is_dm_core() && CID == 0 && bt > 0) {
            // debug("\t[DMA] store  C[%d]=0x%08x for B%u\n\t", curA, l1_buffers.A_buf[curA], bt);
            int nxtC = curC ^ 1; // Toggle buffer

            // dq_load_activation_tile((void*)l1_buffers.A_buf[0], &matrix_activation_fp16[0],
            //                         /*row_tile_index=*/0);
        }

        flex_intra_cluster_sync();
        for (uint32_t at = 0; at < NUM_TILES; ++at) {
            const uint32_t r  = get_rows_for_tile(at);
            const uint32_t sr = get_start_row_for_tile(at); // row start
            if (core_id == SPATZ_CORE && CID == 0) {
                flex_redmule_config(r, FP16_N, P);
            }

            PDEBUG("\t\t>> A-0 %u: rows=%u, start_row=%u, curA_buf=%d\n", at, r, sr, curA);

            // Calculate destination address in allocated HBM buffer
            const uint32_t col_start_elems = get_start_group_for_tile(bt) * VQ_GROUP_SIZE;
            const uint32_t row_start_elems = sr * FP16_K;
            const uint64_t dst =
                l1_buffers.C_hbm_base + (uint64_t)(row_start_elems + col_start_elems) * sizeof(uint16_t);

            // DMA: Prefetch next A tile (overlaps with Spatz compute)
            if (at + 1 < NUM_TILES && flex_is_dm_core() && CID == 0) {
                int nxtA = curA ^ 1; // Toggle buffer
                debug("\t\t[DMA] Load A%u to A_buf[%d]=0x%08x (while computing)\t\t", at + 1, nxtA,
                      l1_buffers.A_buf[nxtA]);
                dq_load_activation_tile((void*)l1_buffers.A_buf[nxtA], &matrix_activation_fp16[0], at + 1);
            }

            // While processing first A tile, prefetch next B indices
            if (at == 0 && bt + 1 < NUM_TILES && flex_is_dm_core() && CID == 0) {
                int nxtB = curB ^ 1; // Toggle buffer
                debug("\t\t[DMA] Prefetching B%u indices to idx_buf[%d]=0x%08x\n\t\t", bt + 1, nxtB,
                      l1_buffers.idx0_buf_packed[nxtB]);
                dq_load_indices_tile((void*)l1_buffers.idx0_buf_packed[nxtB], &matrix_idx_packed_uint16[0], bt + 1);
                // flex_dma_async_1d((uint64_t) (uintptr_t) l1_buffers.C_tile, zomem(0), 8192);
                bare_dma_wait_all();
            }

            // Spatz: Compute A_at × B_bt (compact P) -> C_tile
            if (core_id == SPATZ_CORE && CID == 0) {
                l1_buffers.activation = (uint16_t*)l1_buffers.A_buf[curA];

                // debug("    [REDMULE] Computing A[%u×%u] × B[%u×%u] -> C[%u×%u]\n",
                //     r, FP16_N, FP16_N, P, r, P);
                uint32_t C_buf_elems = max_r * max_P;
                uint32_t C_buf_size  = C_buf_elems * sizeof(uint16_t);
                uint16_t* c_ptr      = (uint16_t*)(uintptr_t)l1_buffers.C_tile;
                uint16_t* c2_ptr     = (uint16_t*)(uintptr_t)l1_buffers.C_tile_buf[0];
                uint16_t* c3_ptr     = (uint16_t*)(uintptr_t)l1_buffers.C_tile_buf[1];

                for (int i = 0; i < C_buf_size; ++i) { // TODO temporary solution,fix later
                    c_ptr[i] = 0;
                    //         // debug("0x%04x ", c_ptr[i]);
                }
                debug("\t\t[REDMULE] Compute\n\t\t");
                flex_timer_start();
                flex_redmule_trigger((uint32_t)l1_buffers.A_buf[curA], (uint32_t)g_l1_dq.W_dq,
                                     (uint32_t)l1_buffers.C_tile, REDMULE_FP_16);

                flex_redmule_wait();
                flex_timer_end();

                // debug("\t\t[spatz] compute\n\t\t");
                // flex_timer_start();

                // spatz_matmul_fp16_full_legacy(l1_buffers.A_buf[curA], // [r x N]
                //                               g_l1_dq.W_dq,           // [N x P] (compact)
                //                               l1_buffers.C_tile,      // [r x P] (compact)
                //                               r, FP16_N, P);
                // flex_timer_end();
            }

            flex_intra_cluster_sync();

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

            flex_intra_cluster_sync();
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
        uint16_t* hbm_result_ptr = (uint16_t*)(uintptr_t)l1_buffers.C_hbm_base;
        // debug("[DEBUG] Result matrix is in HBM at 0x%08x\n", (uint32_t)C_hbm_base);
        // debug("[DEBUG] First 8 computed results from HBM: ");
        // for (int i = 0; i < 8; i++) {
        //     debug("0x%04x ", hbm_result_ptr[i]);
        // }
        // debug("\n");

        // Verify matrix multiplication results against golden reference
        debug("[DEBUG] Verifying GEMM results against golden reference...\n");
        spatz_verify_16(FP16_M * FP16_N, hbm_result_ptr, (uint16_t*)matrix_golden_fp16, 1.0F);
        debug("[VERIFICATION] Verification complete!\n");
    }
}

#endif
