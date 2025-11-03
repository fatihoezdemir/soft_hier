#include "dq_compute_spatz.h" // spatz computation kernels
#include "dq_data_hbm.h"
#include "flex_alloc.h"
#include "flex_dma_pattern.h"
#include "flex_libfp16.h"
#include "flex_libfp8.h"
#include "flex_redmule.h"
#if GEMV == 1
typedef struct {
    volatile uint16_t* idx0_buf_packed[2];
    volatile uint8_t* idx0_buf[2];
    volatile uint8_t* idx1_buf[2];

    volatile uint16_t* x_vec; //__attribute__((section(".l1_prio")));
    volatile uint16_t* y_vec; // Result tile buffer
    volatile uint64_t C_hbm_base;

} L1_buffers;

L1_buffers l1_buffers;
// extern
L1_DQ_Handles g_l1_dq = {0};

void dq_gemv_double_buffer_baseline() {
    uint32_t CID     = flex_get_cluster_id();
    uint32_t core_id = flex_get_core_id();
    // Calculate max sizes for buffer allocation
    uint32_t max_g       = get_groups_for_tile(0);
    const uint32_t max_P = max_g * VQ_GROUP_SIZE;

    if (flex_is_dm_core() && CID == 0) {
        // Allocate HBM for result vector y
        const uint32_t y_bytes = FP16_K * sizeof(uint16_t);
        l1_buffers.C_hbm_base  = (uint64_t)(uintptr_t)flex_hbm_malloc(y_bytes);
        // debug("[DEBUG][DMA] y_hbm_base  alloc= 0x%08x (size=%u bytes)\n", (uint32_t)C_hbm_base, y_bytes);

        flex_timer_start();
        dq_load_codebook_l1(); // load codebook
        flex_timer_end();

        // Buffer sizes
        uint32_t idx_buf_size = FP16_M * max_g * sizeof(uint16_t);
        uint32_t x_size       = FP16_M * sizeof(uint16_t);
        uint32_t W_buf_size   = FP16_M * max_P * sizeof(uint16_t);
        uint32_t y_tile_size  = max_P * sizeof(uint16_t);

        // Allocate  buffers for indices
        l1_buffers.idx0_buf_packed[0] = (uint16_t*)(uintptr_t)flex_l1_malloc(idx_buf_size);
        l1_buffers.idx0_buf_packed[1] = (uint16_t*)(uintptr_t)flex_l1_malloc(idx_buf_size);
        l1_buffers.x_vec              = (uint16_t*)(uintptr_t)flex_l1_malloc(x_size);
        g_l1_dq.W_dq                  = (uint16_t*)(uintptr_t)flex_l1_malloc(W_buf_size);
        l1_buffers.y_vec              = (uint16_t*)(uintptr_t)flex_l1_malloc(y_tile_size);

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
        dq_load_indices_tile((void*)l1_buffers.idx0_buf_packed[0], &matrix_idx_packed_uint16[0], 0);
        flex_timer_end();
        debug("\t[SPATZ] Dequantizing\n\t");
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
            g_l1_dq.indices = l1_buffers.idx0_buf_packed[curIdx];
            // g_l1_dq.W_dq    = W_tile;
            dequantize_block_tile_compact(0, FP16_M, g, g);
        }

        // Overlap: Prefetch next tile's indices while dequantizing
        if (bt + 1 < NUM_TILES && flex_is_dm_core() && CID == 0) {
            int nextIdx = curIdx ^ 1;
            debug(" \t[DMA] Load indices for tile %u to B[%d] (overlapped with dequant)\n\t", bt + 1, nextIdx);
            // flex_timer_start();
            dq_load_indices_tile((void*)l1_buffers.idx0_buf[nextIdx], &matrix_idx_packed_uint16[0], bt + 1);
            // flex_timer_end();
            debug(" \t[SPATZ] Dequantize tile %u via idx_buf[%d]\n\t", bt, curIdx);
        }

        flex_intra_cluster_sync();

        // Step 2: Compute x^T * W_t -> y_t(vertical)
        if (core_id == SPATZ_CORE && CID == 0) {
            debug("\t[REDMULE] Computing x^T * W_tile%u: (1x%u) * (%ux%u) -> (1x%u) \n\t", bt, FP16_M, FP16_M, P, P);

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

        }

        // crct: Wait for compute to finish before DMA can read C_tile
        flex_intra_cluster_sync();
        // Step 3: Store result to HBM
        if (flex_is_dm_core() && CID == 0) {
            const uint64_t dst = l1_buffers.C_hbm_base + col_start * sizeof(uint16_t);
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
        uint16_t* hbm_result_ptr = (uint16_t*)(uintptr_t)l1_buffers.C_hbm_base;
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

void dq_gemv_double_buffer_extended() {
    uint32_t CID     = flex_get_cluster_id();
    uint32_t core_id = flex_get_core_id();
    // Calculate max sizes for buffer allocation
    uint32_t max_g       = get_groups_for_tile(0);
    const uint32_t max_P = max_g * VQ_GROUP_SIZE;

    if (flex_is_dm_core() && CID == 0) {
        // Allocate HBM for result vector y
        const uint32_t y_bytes = FP16_K * sizeof(uint16_t);
        l1_buffers.C_hbm_base  = (uint64_t)(uintptr_t)flex_hbm_malloc(y_bytes);
        // debug("[DEBUG][DMA] y_hbm_base  alloc= 0x%08x (size=%u bytes)\n", (uint32_t)C_hbm_base, y_bytes);

        flex_timer_start();
        dq_load_codebook_l1(); // load codebook
        flex_timer_end();

        // Buffer sizes
        uint32_t idx_buf_size = FP16_M * max_g * sizeof(uint8_t);
        uint32_t x_size       = FP16_M * sizeof(uint16_t);
        uint32_t W_buf_size   = FP16_M * max_P * sizeof(uint16_t);
        uint32_t y_tile_size  = max_P * sizeof(uint16_t);

        // Allocate  buffers for indices
        for (int i=0;i<2;++i){
        l1_buffers.idx0_buf[i] = (uint8_t*)(uintptr_t)flex_l1_malloc(idx_buf_size);
        l1_buffers.idx1_buf[i] = (uint8_t*)(uintptr_t)flex_l1_malloc(idx_buf_size);
        }
        l1_buffers.x_vec       = (uint16_t*)(uintptr_t)flex_l1_malloc(x_size);
        g_l1_dq.W_dq           = (uint16_t*)(uintptr_t)flex_l1_malloc(W_buf_size);
        l1_buffers.y_vec       = (uint16_t*)(uintptr_t)flex_l1_malloc(y_tile_size);

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
        // dq_load_indices_tile((void*)l1_buffers.idx0_buf[0], &matrix_idx1_uint8[0], 0);
        dq_load_indices_tile_u8((void*)l1_buffers.idx0_buf[0], &matrix_idx0_uint8[0], 0);
        dq_load_indices_tile_u8((void*)l1_buffers.idx1_buf[0], &matrix_idx1_uint8[0], 0);
        flex_timer_end();
        // DEBUG_PRINT_U8(l1_buffers.idx0_buf[0], 10, "idx1");
        // DEBUG_PRINT_U8(l1_buffers.idx1_buf[0], 10, "idx1");

        debug("\t[SPATZ] Dequantizing\n\t");
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
            g_l1_dq.indices0_u8 = l1_buffers.idx0_buf[curIdx];
            g_l1_dq.indices1_u8 = l1_buffers.idx1_buf[curIdx];

            // g_l1_dq.W_dq    = W_tile;
            dequantize_block_tile_compact_improved(0, FP16_M, g, g);
            // dequantize_block_tile_compact(0, FP16_M, g, g);
        }

        // Overlap: Prefetch next tile's indices while dequantizing
        if (bt + 1 < NUM_TILES && flex_is_dm_core() && CID == 0) {
            int nextIdx = curIdx ^ 1;
            debug(" \t[DMA] Load indices for tile %u to B[%d] (overlapped with dequant)\n\t", bt + 1, nextIdx);
            // flex_timer_start();
            // dq_load_indices_tile((void*)l1_buffers.idx0_buf[nextIdx], &matrix_idx_packed_uint16[0], bt + 1);
            dq_load_indices_tile_u8((void*)l1_buffers.idx0_buf[nextIdx], &matrix_idx0_uint8[0], bt + 1);
            dq_load_indices_tile_u8((void*)l1_buffers.idx1_buf[nextIdx], &matrix_idx1_uint8[0], bt + 1);
            // flex_timer_end();
            debug(" \t[SPATZ] Dequantize tile %u via idx_buf[%d]\n\t", bt, curIdx);
        }

        flex_intra_cluster_sync();

        // Step 2: Compute x^T * W_t -> y_t(vertical)
        if (core_id == SPATZ_CORE && CID == 0) {
            debug("\t[REDMULE] Computing x^T * W_tile%u: (1x%u) * (%ux%u) -> (1x%u) \n\t", bt, FP16_M, FP16_M, P, P);
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

        }

        // crct: Wait for compute to finish before DMA can read C_tile
        flex_intra_cluster_sync();
        // Step 3: Store result to HBM
        if (flex_is_dm_core() && CID == 0) {
            const uint64_t dst = l1_buffers.C_hbm_base + col_start * sizeof(uint16_t);
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
        uint16_t* hbm_result_ptr = (uint16_t*)(uintptr_t)l1_buffers.C_hbm_base;
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

void dq_gemv_double_buffer_fused() {
    uint32_t CID     = flex_get_cluster_id();
    uint32_t core_id = flex_get_core_id();
    // Calculate max sizes for buffer allocation
    uint32_t max_g       = get_groups_for_tile(0);
    const uint32_t max_P = max_g * VQ_GROUP_SIZE;

    if (flex_is_dm_core() && CID == 0) {
        // Allocate HBM for result vector y
        const uint32_t y_bytes = FP16_K * sizeof(uint16_t);
        l1_buffers.C_hbm_base  = (uint64_t)(uintptr_t)flex_hbm_malloc(y_bytes);
        // debug("[DEBUG][DMA] y_hbm_base  alloc= 0x%08x (size=%u bytes)\n", (uint32_t)C_hbm_base, y_bytes);

        flex_timer_start();
        dq_load_codebook_l1(); // load codebook
        flex_timer_end();

        // Buffer sizes
        uint32_t idx_buf_size = FP16_M * max_g * sizeof(uint8_t);
        uint32_t x_size       = FP16_M * sizeof(uint16_t);
        uint32_t W_buf_size   = FP16_M * max_P * sizeof(uint16_t);
        uint32_t y_tile_size  = max_P * sizeof(uint16_t);

        // Allocate  buffers for indices
        l1_buffers.idx0_buf[0] = (uint8_t*)(uintptr_t)flex_l1_malloc(idx_buf_size);
        l1_buffers.idx0_buf[1] = (uint8_t*)(uintptr_t)flex_l1_malloc(idx_buf_size);
        l1_buffers.idx1_buf[0] = (uint8_t*)(uintptr_t)flex_l1_malloc(idx_buf_size);
        l1_buffers.idx1_buf[1] = (uint8_t*)(uintptr_t)flex_l1_malloc(idx_buf_size);
        l1_buffers.x_vec       = (uint16_t*)(uintptr_t)flex_l1_malloc(x_size);
        g_l1_dq.W_dq           = (uint16_t*)(uintptr_t)flex_l1_malloc(W_buf_size);
        l1_buffers.y_vec       = (uint16_t*)(uintptr_t)flex_l1_malloc(y_tile_size);

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
        // dq_load_indices_tile((void*)l1_buffers.idx0_buf[0], &matrix_idx1_uint8[0], 0);
        dq_load_indices_tile_u8((void*)l1_buffers.idx0_buf[0], &matrix_idx0_uint8[0], 0);
        dq_load_indices_tile_u8((void*)l1_buffers.idx1_buf[0], &matrix_idx1_uint8[0], 0);
        flex_timer_end();
        // DEBUG_PRINT_U8(l1_buffers.idx0_buf[0], 10, "idx1");
        // DEBUG_PRINT_U8(l1_buffers.idx1_buf[0], 10, "idx1");
    }
    flex_intra_cluster_sync();

    // loop over B_tile
    for (uint32_t bt = 0; bt < NUM_TILES; ++bt) {
        const uint32_t g         = get_groups_for_tile(bt);
        const uint32_t P         = g * VQ_GROUP_SIZE;
        const uint32_t col_start = get_start_group_for_tile(bt) * VQ_GROUP_SIZE;

        // Overlap: Prefetch next tile's indices while dequantizing
        if (bt + 1 < NUM_TILES && flex_is_dm_core() && CID == 0) {
            int nextIdx = curIdx ^ 1;
            debug(" \t[DMA] Load indices for tile %u to B[%d] (overlapped with dequant)\n\t", bt + 1, nextIdx);
            // flex_timer_start();
            // dq_load_indices_tile((void*)l1_buffers.idx0_buf[nextIdx], &matrix_idx_packed_uint16[0], bt + 1);
            dq_load_indices_tile_u8((void*)l1_buffers.idx0_buf[nextIdx], &matrix_idx0_uint8[0], bt + 1);
            dq_load_indices_tile_u8((void*)l1_buffers.idx1_buf[nextIdx], &matrix_idx1_uint8[0], bt + 1);
            // flex_timer_end();
            debug(" \t[SPATZ] Prepared tile %u via idx_buf[%d]\n\t", bt, curIdx);
        }

        flex_intra_cluster_sync();

        // Step 2: Fused dequantization + GEMV accumulation
        if (core_id == SPATZ_CORE && CID == 0) {
            debug("\t[FUSED] Accumulating tile %u directly from indices (P=%u)\n\t", bt, P);
            uint16_t* y_tile = (uint16_t*)(uintptr_t)l1_buffers.y_vec;
            for (uint32_t i = 0; i < P; ++i) {
                y_tile[i] = 0;
            }
            g_l1_dq.indices0_u8    = l1_buffers.idx0_buf[curIdx];
            g_l1_dq.indices1_u8    = l1_buffers.idx1_buf[curIdx];
            const uint16_t* x_tile = (const uint16_t*)(uintptr_t)l1_buffers.x_vec;
            flex_timer_start();
            // dequantize_block_tile_compact_fused_gemv(0, FP16_M, g, g, x_tile, y_tile);
            dequantize_block_tile_compact_improvedfused(0, FP16_M, g, g, x_tile, y_tile);

            flex_timer_end();
        }

        // crct: Wait for compute to finish before DMA can read C_tile
        flex_intra_cluster_sync();
        // Step 3: Store result to HBM
        if (flex_is_dm_core() && CID == 0) {
            const uint64_t dst = l1_buffers.C_hbm_base + col_start * sizeof(uint16_t);
            debug("\t[DMA] Storing y_tile%u to HBM at offset %u\n\t", bt, col_start);
            // DEBUG_PRINT_U8((uint16_t*)(uintptr_t)l1_buffers.y_vec, 10, "resulkt");

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
        uint16_t* hbm_result_ptr = (uint16_t*)(uintptr_t)l1_buffers.C_hbm_base;
        // debug("[DEBUG] Result vector is in HBM at 0x%08x\n", (uint32_t)L1_PTRS.C_hbm_base);
        // debug("[DEBUG] First 8 computed results: ");
        // for (int i = 0; i < 8 && i < FP16_K; i++) {
        //     debug("0x%04x ", hbm_result_ptr[i]);
        // }
        spatz_verify_16(FP16_K, hbm_result_ptr, (uint16_t*)matrix_golden_fp16, 1.0f);

        debug("\n");
        debug("[VERIFICATION] GEMV verification complete!\n");
    }
}

#endif
