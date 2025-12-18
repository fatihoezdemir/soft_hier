#ifndef SUMMA_VPTQ_PIPELINES_HH
#define SUMMA_VPTQ_PIPELINES_HH

#include "flex_transpose_engine.h"
#include "gemm.h"
#include "gemm_setup.h"
#include "vq_kernels_vptq.h"
#if VQ_ENABLED == 1

#if VQ_TRANSPOSE_ENABLED == 1
static inline void summa_vptq_transpose_tile(const SummaGEMMInfo* info, uint32_t src_addr, uint32_t dst_addr) {
    flex_transpose_engine_config(info->N_tile, info->K_tile, src_addr, dst_addr, DATA_TYPE_BYTE);
    flex_transpose_engine_trigger();
    flex_transpose_engine_wait();
}
#endif

// VPTQ baseline: dequantize into a column-major scratch buffer, transpose, then feed REDMULE.
static inline void run_gemv_pipelinevptq_baseline(SummaGEMMInfo* info, int m, int n, uint32_t* DMA_L1_Z,
                                                  uint32_t* REDMULE_L1_Z) {
    const int tiles      = info->K_iter;
    const int SPATZ_CORE = 2;
    uint32_t core_id     = flex_get_core_id();

    if (tiles <= 0) {
        flex_global_barrier_xy();
        return;
    }

    uint32_t x_buffers[2] = {info->L1_X1, info->L1_X2};
    uint32_t w_buffers[2] = {info->L1_W1, info->L1_W2};
#if VQ_USE_SCALES == 1
    uint32_t scale_buffers[2] = {info->vq.L1_Scales[0], info->vq.L1_Scales[1]};
#else
    uint32_t scale_buffers[2] = {0u, 0u};
#endif

    // ─────────────────────────────────────────────────────────────────
    // PROLOGUE P0: DMA indices and scales for tile 0
    // ─────────────────────────────────────────────────────────────────
    if (flex_is_dm_core() && info->cluster_for_colwise == 1) {
        summa_vq_load_indices(info, 0, m, n, 0);
#if VQ_USE_SCALES == 1
        summa_vq_load_scales(info, scale_buffers[0], m, n, 0);
#endif
    }
    flex_intra_cluster_sync();

    // ─────────────────────────────────────────────────────────────────
    // PROLOGUE P1: DMA X[0] + idx/scale[1] || SPATZ dequantize W[0] → transpose
    // ─────────────────────────────────────────────────────────────────
    if (flex_is_dm_core()) {
        if (info->cluster_for_rowwise == 1) {
            summa_load_X_tile(info, x_buffers[0], m, n, 0);
        }
        if (info->cluster_for_colwise == 1 && tiles > 1) {
            summa_vq_load_indices(info, 1, m, n, 1);
#if VQ_USE_SCALES == 1
            summa_vq_load_scales(info, scale_buffers[1], m, n, 1);
#endif
        }
    } else if (core_id == SPATZ_CORE) {
        uint32_t deq_dst = w_buffers[0];
#if VQ_TRANSPOSE_ENABLED == 1
        deq_dst = info->L1_DETRANSPOSED;
#endif
        if (flex_get_cluster_id() == 0)
            flex_timer_start();
        summa_vq_dequantize_tile_vptq_baseline(info, deq_dst, 0, scale_buffers[0], 0);
        if (flex_get_cluster_id() == 0)
            flex_timer_end();
    }
    flex_intra_cluster_sync();
#if VQ_TRANSPOSE_ENABLED == 1
    if (flex_is_first_core()) {
        if (flex_get_cluster_id()==0)
        flex_timer_start();
        summa_vptq_transpose_tile(info, info->L1_DETRANSPOSED, w_buffers[0]);
            if (flex_get_cluster_id() == 0)
            flex_timer_end();
    }
    flex_intra_cluster_sync();
#endif

    // ─────────────────────────────────────────────────────────────────
    // PROLOGUE P2: Trigger REDMULE for tile 0
    // ─────────────────────────────────────────────────────────────────
    grid_sync_group_barrier_xy(&(info->group));
    if (flex_is_first_core()) {
        flex_redmule_config(info->M_tile, info->K_tile, info->N_tile);
        flex_redmule_trigger(x_buffers[0], w_buffers[0], *REDMULE_L1_Z, REDMULE_COMPUTE_TYPE);
    }
    flex_intra_cluster_sync();

    // Pipeline indices for steady state
    int next_x_tile   = 1;
    int next_idx_tile = (tiles > 1) ? 2 : tiles;
    int next_deq_tile = 1;

    // ─────────────────────────────────────────────────────────────────
    // STEADY STATE
    // ─────────────────────────────────────────────────────────────────
    for (int tile = 1; tile < tiles; ++tile) {
        uint32_t redmule_x = x_buffers[tile & 0x1];
        uint32_t redmule_w = w_buffers[tile & 0x1];

        grid_sync_group_barrier_xy(&(info->group));

        // Prefetch / dequantize / transpose while REDMULE works on (tile-1)
        if (flex_is_dm_core()) {
            if (info->cluster_for_rowwise == 1 && next_x_tile < tiles) {
                uint32_t dst_x = x_buffers[next_x_tile & 0x1];
                summa_load_X_tile(info, dst_x, m, n, next_x_tile);
                ++next_x_tile;
            }
            if (info->cluster_for_colwise == 1 && next_idx_tile < tiles) {
                int buffer_idx = next_idx_tile & 0x1;
                summa_vq_load_indices(info, buffer_idx, m, n, next_idx_tile);
#if VQ_USE_SCALES == 1
                uint32_t dst_scale = scale_buffers[buffer_idx];
                summa_vq_load_scales(info, dst_scale, m, n, next_idx_tile);
#endif
                ++next_idx_tile;
            }
        } else if (core_id == SPATZ_CORE) {
            if (next_deq_tile < tiles) {
                uint32_t deq_dst = redmule_w;
#if VQ_TRANSPOSE_ENABLED == 1
                deq_dst = info->L1_DETRANSPOSED;
#endif
                int buffer_idx     = next_deq_tile & 0x1;
                uint32_t src_scale = scale_buffers[buffer_idx];
                if (flex_get_cluster_id() == 0)
                    flex_timer_start();
                summa_vq_dequantize_tile_vptq_baseline(info, deq_dst, buffer_idx, src_scale, next_deq_tile);
                if (flex_get_cluster_id() == 0)
                    flex_timer_end();
                ++next_deq_tile;
            }
        }

        flex_intra_cluster_sync();
#if VQ_TRANSPOSE_ENABLED == 1
        if (flex_is_first_core()) {
            summa_vptq_transpose_tile(info, info->L1_DETRANSPOSED, redmule_w);
        }
        flex_intra_cluster_sync();
#endif

        if (flex_is_first_core()) {
            flex_redmule_wait();
        }
        flex_intra_cluster_sync();

        if (info->group_reduction == 1) {
            grid_sync_group_barrier_xy(&(info->group));
        }

        // Store tile (tile-1)
        if (flex_is_dm_core() && info->store_recorded == 1 && info->store_active == 1) {
            if (info->group_reduction == 0) {
                summa_reduce_and_store_Z(info, *DMA_L1_Z, true);
                info->store_id       = info->summa_group_x;
                info->store_recorded = 0;
            } else {
                uint32_t start_id = (info->store_id < info->store_step) ? 0 : info->store_id - info->store_step;
                uint32_t bid      = (start_id + info->store_id_offset) % info->summa_group_x;
                uint32_t eid      = (info->store_id + info->store_id_offset) % info->summa_group_x;
                if (((info->cluster_in_group_id_x >= bid && info->cluster_in_group_id_x < eid) && eid > bid) ||
                    ((info->cluster_in_group_id_x >= bid || info->cluster_in_group_id_x < eid) && eid <= bid)) {
                    summa_reduce_and_store_Z(info, *DMA_L1_Z, true);
                }

                info->store_id = start_id;
            }
        }

        // Trigger REDMULE for tile (tile)
        if (flex_is_first_core()) {
            flex_redmule_config(info->M_tile, info->K_tile, info->N_tile);
            flex_redmule_trigger(redmule_x, redmule_w, *REDMULE_L1_Z, REDMULE_COMPUTE_TYPE);
        }
        flex_intra_cluster_sync();
    }

    // ─────────────────────────────────────────────────────────────────
    // EPILOGUE
    // ─────────────────────────────────────────────────────────────────
    if (info->group_reduction == 1) {
        flex_global_barrier_xy();
    } else {
        grid_sync_group_barrier_xy(&(info->group));
    }

    info->store_recorded = 1;
    info->store_m        = m;
    info->store_n        = n;
    info->store_id       = info->summa_group_x;

    uint32_t tmp_z = *DMA_L1_Z;
    *DMA_L1_Z      = *REDMULE_L1_Z;
    *REDMULE_L1_Z  = tmp_z;

    if (flex_is_first_core()) {
        flex_redmule_wait(); // Wait for last tile's compute to finish
    }
    flex_intra_cluster_sync();
}

static inline void run_gemm_pipelinevptq_baseline(SummaGEMMInfo* info, int m, int n, uint32_t* DMA_L1_Z,
                                                  uint32_t* REDMULE_L1_Z) {
    const int tiles      = info->K_iter;
    const int SPATZ_CORE = 2;
    uint32_t core_id     = flex_get_core_id();

    if (tiles <= 0) {
        flex_global_barrier_xy();
        return;
    }

    uint32_t x_buffers[2] = {info->L1_X1, info->L1_X2};
    uint32_t w_buffers[2] = {info->L1_W1, info->L1_W2};
#if VQ_USE_SCALES == 1
    uint32_t scale_buffers[2] = {info->vq.L1_Scales[0], info->vq.L1_Scales[1]};
#else
    uint32_t scale_buffers[2] = {0u, 0u};
#endif

    /*
    PROLOGUE:
      DMA idx/scale for tile0 → dequant on SPATZ → transpose → trigger REDMULE
    */
    if (flex_is_dm_core() && info->cluster_for_colwise == 1) {
        summa_vq_load_indices(info, 0, m, n, 0);
#if VQ_USE_SCALES == 1
        summa_vq_load_scales(info, scale_buffers[0], m, n, 0);
#endif
    }
    flex_intra_cluster_sync();

    if (flex_is_dm_core()) {
        if (info->cluster_for_rowwise == 1) {
            summa_load_X_tile(info, x_buffers[0], m, n, 0);
        }
        if (info->cluster_for_colwise == 1 && tiles > 1) {
            summa_vq_load_indices(info, 1, m, n, 1);
#if VQ_USE_SCALES == 1
            summa_vq_load_scales(info, scale_buffers[1], m, n, 1);
#endif
        }
    } else if (core_id == SPATZ_CORE) {
        uint32_t deq_dst = w_buffers[0];
#if VQ_TRANSPOSE_ENABLED == 1
        deq_dst = info->L1_DETRANSPOSED;
#endif
        if (flex_get_cluster_id() == 0)
            flex_timer_start();
        summa_vq_dequantize_tile_vptq_baseline(info, deq_dst, 0, scale_buffers[0], 0);
        if (flex_get_cluster_id() == 0)
            flex_timer_end();
    }
    #if VQ_TRANSPOSE_ENABLED == 1
    if (flex_is_first_core()) {
        if (flex_get_cluster_id()==0)
        flex_timer_start();
        flex_intra_cluster_sync();
        summa_vptq_transpose_tile(info, info->L1_DETRANSPOSED, w_buffers[0]);
        if (flex_get_cluster_id() == 0)
        flex_timer_end();
        
    }
    flex_intra_cluster_sync();
#endif

    grid_sync_group_barrier_xy(&(info->group));
    if (flex_is_first_core()) {
        flex_redmule_config(info->M_tile, info->K_tile, info->N_tile);
        flex_redmule_trigger(x_buffers[0], w_buffers[0], *REDMULE_L1_Z, REDMULE_COMPUTE_TYPE);
    }
    flex_intra_cluster_sync();

    int next_x_tile   = 1;
    int next_idx_tile = (tiles > 1) ? 2 : tiles;
    int next_deq_tile = 1;

    /*
    STEADY STATE:
      Prefetch X/idx/scale for next tiles, dequantize current tile, transpose, drain REDMULE, store, retrigger.
    */
    for (int tile = 1; tile < tiles; ++tile) {
        uint32_t redmule_x = x_buffers[tile & 0x1];
        uint32_t redmule_w = w_buffers[tile & 0x1];

        grid_sync_group_barrier_xy(&(info->group));

        if (flex_is_dm_core()) {
            if (info->cluster_for_rowwise == 1 && next_x_tile < tiles) {
                uint32_t dst_x = x_buffers[next_x_tile & 0x1];
                summa_load_X_tile(info, dst_x, m, n, next_x_tile);
                ++next_x_tile;
            }

            if (info->cluster_for_colwise == 1 && next_idx_tile < tiles) {
                int buffer_idx = next_idx_tile & 0x1;
                summa_vq_load_indices(info, buffer_idx, m, n, next_idx_tile);
#if VQ_USE_SCALES == 1
                uint32_t dst_scale = scale_buffers[buffer_idx];
                summa_vq_load_scales(info, dst_scale, m, n, next_idx_tile);
#endif
                ++next_idx_tile;
            }
        } else if (core_id == SPATZ_CORE) {
            if (next_deq_tile < tiles) {
                uint32_t deq_dst = redmule_w;
#if VQ_TRANSPOSE_ENABLED == 1
                deq_dst = info->L1_DETRANSPOSED;
#endif
                int buffer_idx     = next_deq_tile & 0x1;
                uint32_t src_scale = scale_buffers[buffer_idx];
                if (flex_get_cluster_id() == 0)
                    flex_timer_start();
                summa_vq_dequantize_tile_vptq_baseline(info, deq_dst, buffer_idx, src_scale, next_deq_tile);
                if (flex_get_cluster_id() == 0)
                    flex_timer_end();
                ++next_deq_tile;
            }
        }

        #if VQ_TRANSPOSE_ENABLED == 1
        if (flex_is_first_core()) {
            if (flex_get_cluster_id()==0)
            flex_timer_start();
            flex_intra_cluster_sync();
        summa_vptq_transpose_tile(info, info->L1_DETRANSPOSED, redmule_w);
        
        if (flex_get_cluster_id() == 0)
        flex_timer_end();
    }
    flex_intra_cluster_sync();
#endif

        if (flex_is_first_core()) {
            flex_redmule_wait();
        }
        flex_intra_cluster_sync();

        if (flex_is_dm_core() && info->store_recorded == 1 && info->store_active == 1) {
            uint32_t start_id = (info->store_id < info->store_step) ? 0 : info->store_id - info->store_step;
            uint32_t bid      = (start_id + info->store_id_offset) % info->summa_group_x;
            uint32_t eid      = (info->store_id + info->store_id_offset) % info->summa_group_x;

            if (((info->cluster_in_group_id_x >= bid && info->cluster_in_group_id_x < eid) && eid > bid) ||
                ((info->cluster_in_group_id_x >= bid || info->cluster_in_group_id_x < eid) && eid <= bid)) {
                summa_reduce_and_store_Z(info, *DMA_L1_Z, true);
            }

            info->store_id = start_id;
        }

        if (flex_is_first_core()) {
            flex_redmule_config(info->M_tile, info->K_tile, info->N_tile);
            flex_redmule_trigger(redmule_x, redmule_w, *REDMULE_L1_Z, REDMULE_COMPUTE_TYPE);
        }
        flex_intra_cluster_sync();
    }

    // EPILOGUE: drain the last tile and finalize
    if (info->group_reduction == 1) {
        flex_global_barrier_xy();
    } else {
        grid_sync_group_barrier_xy(&(info->group));
    }

    info->store_recorded = 1;
    info->store_m        = m;
    info->store_n        = n;
    info->store_id       = info->summa_group_x;

    uint32_t tmp_z = *DMA_L1_Z;
    *DMA_L1_Z      = *REDMULE_L1_Z;
    *REDMULE_L1_Z  = tmp_z;

    if (flex_is_first_core()) {
        flex_redmule_wait();
    }
    flex_intra_cluster_sync();
}

#endif
#endif
