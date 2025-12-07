#ifndef _SUMMA_GEMV_H_
#define _SUMMA_GEMV_H_

#include "flex_dma_pattern.h"
#include "flex_group_barrier.h"
#include "flex_printf.h"
#include "flex_redmule.h"
#include "flex_runtime.h"
#include "gemm_setup.h"
#include "spatz_compute.h"
#include "summa_dma.h"
#include "summa_index.h"
#include "vq_kernels.h"
// run_gemv_pipeline is basically run_gemm_pipeline vice versa for run_gemv_pipeline_vq , TODO add spatz multiple core
// support
static inline void run_gemv_pipeline(SummaGEMMInfo* info, int m, int n, uint32_t* DMA_L1_Z, uint32_t* REDMULE_L1_Z) {
    const int tiles       = info->K_iter;
    uint32_t x_buffers[2] = {info->L1_X1, info->L1_X2};
    uint32_t w_buffers[2] = {info->L1_W1, info->L1_W2};

    // ─────────────────────────────────────────────────────────────
    // PROLOGUE: DMA X[0], W[0]
    // ─────────────────────────────────────────────────────────────
    if (flex_is_dm_core()) {
        if (info->cluster_for_rowwise == 1) {
            summa_load_X_tile(info, x_buffers[0], m, n, 0);
        }
        if (info->cluster_for_colwise == 1) {
            summa_load_W_tile(info, w_buffers[0], m, n, 0);
        }
    }
    flex_intra_cluster_sync();

    // ─────────────────────────────────────────────────────────────
    // PIPELINE: GEMM-style double buffering
    // ─────────────────────────────────────────────────────────────
    for (int k = 1; k <= tiles; ++k) {
        uint32_t dma_x     = x_buffers[k & 0x1];
        uint32_t dma_w     = w_buffers[k & 0x1];
        uint32_t redmule_x = x_buffers[(k + 1) & 0x1];
        uint32_t redmule_w = w_buffers[(k + 1) & 0x1];

        // Fence previous compute before store/advance
        grid_sync_group_barrier_xy(&(info->group));
        if (flex_is_first_core()) {
            flex_redmule_wait();
        }
        flex_intra_cluster_sync();

        // Prefetch next tile while we have the previous result ready to store
        if (flex_is_dm_core()) {
            if (k < tiles) {
                if (info->cluster_for_rowwise == 1) {
                    summa_load_X_tile(info, dma_x, m, n, k);
                }
                if (info->cluster_for_colwise == 1) {
                    summa_load_W_tile(info, dma_w, m, n, k);
                }
            }

            if (info->store_recorded == 1 && info->store_active == 1) {
                if (info->group_reduction == 0) {
                    // No inter-group reduction: store tile immediately and clear buffer
                    summa_reduce_and_store_Z(info, *DMA_L1_Z, true);
                    info->store_id       = info->summa_group_x;
                    info->store_recorded = 0;
                } else {
                    uint32_t start_id = (info->store_id < info->store_step) ? 0 : info->store_id - info->store_step;
                    uint32_t bid      = (start_id + info->store_id_offset) % info->summa_group_x;
                    uint32_t eid      = (info->store_id + info->store_id_offset) % info->summa_group_x;
                    if (((info->cluster_in_group_id_x >= bid && info->cluster_in_group_id_x < eid) && eid > bid) ||
                        ((info->cluster_in_group_id_x >= bid || info->cluster_in_group_id_x < eid) && eid <= bid)) {
                        if (flex_is_dm_core() && flex_get_cluster_id() == 0)
                            flex_timer_start();
                        summa_reduce_and_store_Z(info, *DMA_L1_Z, true);
                        if (flex_is_dm_core() && flex_get_cluster_id() == 0)
                            flex_timer_end();
                    }

                    info->store_id = start_id;
                }
            }
        }

        // Trigger compute for tile (k-1)
        if (flex_is_first_core()) {
            flex_redmule_config(info->M_tile, info->K_tile, info->N_tile);
            flex_redmule_trigger(redmule_x, redmule_w, *REDMULE_L1_Z, REDMULE_COMPUTE_TYPE);
        }
    }

    // Drain final tile and swap Z buffers for next iteration
    if (info->group_reduction == 1) {
        flex_global_barrier_xy();
    } else {
        grid_sync_group_barrier_xy(&(info->group));
    }
    info->store_recorded = 1; // flag: have data ready to store in next iteration
    info->store_m        = m;
    info->store_n        = n;
    info->store_id       = info->summa_group_x;
    uint32_t tmp_z       = *DMA_L1_Z;
    *DMA_L1_Z            = *REDMULE_L1_Z;
    *REDMULE_L1_Z        = tmp_z;
    if (flex_is_first_core()) {
        flex_redmule_wait();
    }
    flex_intra_cluster_sync();
}

#if VQ_ENABLED == 1
static inline void run_gemv_pipelinevq(SummaGEMMInfo* info, int m, int n, uint32_t* DMA_L1_Z, uint32_t* REDMULE_L1_Z) {
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
    // PROLOGUE P1: DMA X[0] + idx/scale[1] || SPATZ dequantize W[0]
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
        if (flex_get_cluster_id() == 0)
            flex_timer_start();
        summa_vq_dequantize_tile(info, w_buffers[0], 0, scale_buffers[0], 0);
        if (flex_get_cluster_id() == 0)
            flex_timer_end();
    }
    flex_intra_cluster_sync();

    // ─────────────────────────────────────────────────────────────────
    // PROLOGUE P2: Trigger REDMULE for tile 0
    // ─────────────────────────────────────────────────────────────────
    grid_sync_group_barrier_xy(&(info->group));
    if (flex_is_first_core()) {
        flex_redmule_config(info->M_tile, info->K_tile, info->N_tile);
        flex_redmule_trigger(x_buffers[0], w_buffers[0], *REDMULE_L1_Z, REDMULE_COMPUTE_TYPE);
        // NO WAIT HERE
    }
    flex_intra_cluster_sync();

    // Pipeline indices for steady state
    int next_x_tile   = 1;
    int next_idx_tile = (tiles > 1) ? 2 : tiles;
    int next_deq_tile = 1;

    // ─────────────────────────────────────────────────────────────────
    // STEADY STATE: overlap DMA (tile k+1), SPATZ dequant (tile k), REDMULE compute (tile k-1)
    // ─────────────────────────────────────────────────────────────────
    for (int tile = 1; tile < tiles; ++tile) {
        uint32_t redmule_x = x_buffers[tile & 0x1];
        uint32_t redmule_w = w_buffers[tile & 0x1];

        // ─────────────────────────────────────────────────────────────
        // STAGE 1: While REDMULE computes tile (tile-1), prefetch tiles
        // ─────────────────────────────────────────────────────────────
        if (flex_is_dm_core()) {
            // DMA: Load X for tile (tile)
            if (info->cluster_for_rowwise == 1 && next_x_tile < tiles) {
                uint32_t dst_x = x_buffers[next_x_tile & 0x1];
                summa_load_X_tile(info, dst_x, m, n, next_x_tile);
                ++next_x_tile;
            }

            // DMA: Load indices/scales for tile (tile+1)
            if (info->cluster_for_colwise == 1 && next_idx_tile < tiles) {
                int buffer_idx = next_idx_tile & 0x1;
                summa_vq_load_indices(info, buffer_idx, m, n, next_idx_tile);
#if VQ_USE_SCALES == 1
                uint32_t dst_scale = scale_buffers[buffer_idx];
                summa_vq_load_scales(info, dst_scale, m, n, next_idx_tile); //
#endif
                ++next_idx_tile;
            }
        } else if (core_id == SPATZ_CORE) {
            // SPATZ: Dequantize W for tile (tile)
            if (next_deq_tile < tiles) {
                uint32_t dst_w     = w_buffers[next_deq_tile & 0x1];
                int buffer_idx     = next_deq_tile & 0x1;
                uint32_t src_scale = scale_buffers[buffer_idx];
                if (flex_get_cluster_id() == 0)
                    flex_timer_start();
                summa_vq_dequantize_tile(info, dst_w, buffer_idx, src_scale, next_deq_tile);
                if (flex_get_cluster_id() == 0)
                    flex_timer_end();
                ++next_deq_tile;
            }
        }

        // Ensure DMA and SPATZ finish their preparation work
        flex_intra_cluster_sync();

        // wait for RedMULE before finishing tile (tile-1)
        if (flex_is_first_core()) {
            flex_redmule_wait(); // ← CRITICAL: Fence before store!
        }
        flex_intra_cluster_sync();

        // For collective store paths, align clusters; otherwise let DMA overlap freely
        if (info->group_reduction == 1) {
            grid_sync_group_barrier_xy(&(info->group));
        }

        // STAGE 3: STORE tile (tile-1) result (now safe - compute finished!)
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
                    if (flex_is_dm_core() && flex_get_cluster_id() == 0)
                        flex_timer_start();
                    summa_reduce_and_store_Z(info, *DMA_L1_Z, true);
                    if (flex_is_dm_core() && flex_get_cluster_id() == 0)
                        flex_timer_end();
                }

                info->store_id = start_id;
            }
        }

        // ─────────────────────────────────────────────────────────────
        // STAGE 4: TRIGGER REDMULE for tile (tile)
        // ─────────────────────────────────────────────────────────────
        if (flex_is_first_core()) {
            flex_redmule_config(info->M_tile, info->K_tile, info->N_tile);
            flex_redmule_trigger(redmule_x, redmule_w, *REDMULE_L1_Z, REDMULE_COMPUTE_TYPE);
            // NO WAIT! Let next iteration's DMA/SPATZ overlap with this compute
        }
        flex_intra_cluster_sync();
    }

    // ─────────────────────────────────────────────────────────────────
    // EPILOGUE: drain last tile and finalize
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

    // Swap Z buffers for final store
    uint32_t tmp_z = *DMA_L1_Z;
    *DMA_L1_Z      = *REDMULE_L1_Z;
    *REDMULE_L1_Z  = tmp_z;

    if (flex_is_first_core()) {
        flex_redmule_wait(); // Wait for last tile's compute to finish
    }
    flex_intra_cluster_sync();
}

// Triple staged, dequantization-based pipeline with SPATZ compute instead of REDMULE
static inline void run_gemv_pipelinevq_spatz(SummaGEMMInfo* info, int m, int n, uint32_t* DMA_L1_Z,
                                             uint32_t* REDMULE_L1_Z) {
    const int tiles        = info->K_iter;
    const int SPATZ_CORE   = 2; // dedicated to dequant
    const int COMPUTE_CORE = 0; // dedicated to GEMV compute
    uint32_t core_id       = flex_get_core_id();

    if (tiles <= 0) {
        flex_global_barrier_xy();
        return;
    }

    uint32_t x_buffers[2] = {info->L1_X1, info->L1_X2};
    uint32_t w_buffers[2] = {info->L1_W1, info->L1_W2};
    uint32_t compute_L1_Z = *REDMULE_L1_Z; // accumulator buffer for this output tile
#if VQ_USE_SCALES == 1
    uint32_t scale_buffers[2] = {info->vq.L1_Scales[0], info->vq.L1_Scales[1]};
#else
    uint32_t scale_buffers[2] = {0u, 0u};
#endif

    // ─────────────────────────────────────────────────────────────
    // PROLOGUE P0: DMA idx/scale for tile 0
    // ─────────────────────────────────────────────────────────────
    if (flex_is_dm_core() && info->cluster_for_colwise == 1) {
        summa_vq_load_indices(info, 0, m, n, 0);
#if VQ_USE_SCALES == 1
        summa_vq_load_scales(info, scale_buffers[0], m, n, 0);
#endif
    }
    flex_intra_cluster_sync();

    // ─────────────────────────────────────────────────────────────
    // PROLOGUE P1: DMA X[0] + idx/scale[1] || SPATZ dequantize W[0]
    // ─────────────────────────────────────────────────────────────
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
        summa_vq_dequantize_tile(info, w_buffers[0], 0, scale_buffers[0], 0);
    }
    flex_intra_cluster_sync();

    grid_sync_group_barrier_xy(&(info->group));

    // ─────────────────────────────────────────────────────────────
    // Steady state: dma tile k+1 indices and x_vec while dq tile k+1 to W and compute tile k
    // Accumulates into compute_L1_Z across all K tiles.
    // ─────────────────────────────────────────────────────────────
    int next_x_tile   = 1;
    int next_idx_tile = (tiles > 1) ? 2 : tiles;
    int next_deq_tile = 1;

    for (int tile = 0; tile < tiles; ++tile) {
        // Prefetch for next tile on DMA core
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
                uint32_t dst_w     = w_buffers[next_deq_tile & 0x1];
                int buffer_idx     = next_deq_tile & 0x1;
                uint32_t src_scale = scale_buffers[buffer_idx];
                summa_vq_dequantize_tile(info, dst_w, buffer_idx, src_scale, next_deq_tile);
                ++next_deq_tile;
            }
        } else if (core_id == COMPUTE_CORE) {
            bool accumulate = (tile > 0);
            spatz_gemv_fp16_full_legacy((uint16_t*)x_buffers[tile & 0x1], (uint16_t*)w_buffers[tile & 0x1],
                                        (uint16_t*)compute_L1_Z, info->M_tile, info->K_tile, info->N_tile, accumulate);
        }

        // Ensure DMA/dequant/compute for this tile finished before moving on
        flex_intra_cluster_sync();

        // For collective store , align clusters; otherwise let DMA overlap
        if (info->group_reduction == 1) {
            grid_sync_group_barrier_xy(&(info->group));
        }

        // Store previous output tile (if any) while compute buffers are busy on current tile
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

        flex_intra_cluster_sync();
    }

    // PIPELINE EPILOGUE: record current output tile and swap Z buffers once
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

    flex_intra_cluster_sync();
}

// Triple staged, dequantization-based pipeline with fused dequantization and computation, reduces load and stores as
// well as some synchronization barriers, synchronization-wise it is similar to the non-dq based run_gemv_pipeline
// kernel,we just do dequant and compute at once
static inline void run_gemv_pipelinevq_fused(SummaGEMMInfo* info, int m, int n, uint32_t* DMA_L1_Z,
                                             uint32_t* REDMULE_L1_Z) {
    const int tiles       = info->K_iter;
    uint32_t x_buffers[2] = {info->L1_X1, info->L1_X2};
    const int SPATZ_CORE  = 2; // dedicate core 2 for fused compute (matches other VQ pipelines)
    uint32_t core_id      = flex_get_core_id();
#if VQ_USE_SCALES == 1
    uint32_t scale_buffers[2] = {info->vq.L1_Scales[0], info->vq.L1_Scales[1]};
#else
    uint32_t scale_buffers[2] = {0u, 0u};
#endif
    // ─────────────────────────────────────────────────────────────
    // PROLOGUE: DMA X[0], idx [0] scales[0]
    // ─────────────────────────────────────────────────────────────
    if (flex_is_dm_core()) {
        if (info->cluster_for_rowwise == 1) {
            summa_load_X_tile(info, x_buffers[0], m, n, 0);
        }
        if (info->cluster_for_colwise == 1) {
            summa_vq_load_indices(info, 0, m, n, 0);
#if VQ_USE_SCALES == 1
            summa_vq_load_scales(info, scale_buffers[0], m, n, 0);
#endif
        }
    }
    flex_intra_cluster_sync();
    // ─────────────────────────────────────────────────────────────
    // PIPELINE: GEMM-style double buffering
    // ─────────────────────────────────────────────────────────────
    for (int k = 1; k <= tiles; ++k) {
        uint32_t dma_x     = x_buffers[k & 0x1]; // buffer for next tile preload
        uint32_t dma_idx   = k & 0x1;
        uint32_t dma_scale = scale_buffers[k & 0x1];

        uint32_t spatz_x     = x_buffers[(k - 1) & 0x1]; // tile k-1 is in buffer (k-1)
        uint32_t spatz_idx   = (k - 1) & 0x1;
        uint32_t spatz_scale = scale_buffers[(k - 1) & 0x1];
        // Fence previous compute before store/advance
        grid_sync_group_barrier_xy(&(info->group));

        // Prefetch next tile while we have the previous result ready to store
        if (flex_is_dm_core()) {
            if (k < tiles) {
                if (info->cluster_for_rowwise == 1) {
                    summa_load_X_tile(info, dma_x, m, n, k);
                }
                if (info->cluster_for_colwise == 1) {
                    summa_vq_load_indices(info, dma_idx, m, n, k);
#if VQ_USE_SCALES == 1
                    summa_vq_load_scales(info, dma_scale, m, n, k);
#endif
                }
            }
            if (info->store_recorded == 1 && info->store_active == 1) {
                if (info->group_reduction == 0) {
                    // No inter-group reduction: store tile immediately and clear buffer
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
        }
        // Trigger compute for tile (k-1)
        if (core_id == SPATZ_CORE) {
            summa_vq_dequantize_tile_fused(info, *REDMULE_L1_Z, spatz_idx, spatz_scale, spatz_x, k - 1);
        }
        flex_intra_cluster_sync();
    }

    // Drain final tile and swap Z buffers for next iteration
    if (info->group_reduction == 1) {
        flex_global_barrier_xy();
    } else {
        grid_sync_group_barrier_xy(&(info->group));
    }
    info->store_recorded = 1; // flag: have data ready to store in next iteration
    info->store_m        = m;
    info->store_n        = n;
    info->store_id       = info->summa_group_x;
    uint32_t tmp_z       = *DMA_L1_Z;
    *DMA_L1_Z            = *REDMULE_L1_Z;
    *REDMULE_L1_Z        = tmp_z;
    if (flex_is_first_core()) {
        flex_redmule_wait();
    }
    flex_intra_cluster_sync();
}
#endif

void SummaGEMVRun(SummaGEMMInfo* info) {
    flex_global_barrier_xy();

    // Load VQ codebooks - all cores must call this for proper synchronization
#if VQ_ENABLED == 1
    if (info->cluster_for_colwise == 1) {
        vq_load_cb(info);
    }
    flex_global_barrier_xy();
#endif

    if (info->cluster_active) {
        uint32_t DMA_L1_Z     = info->L1_Z2;
        uint32_t REDMULE_L1_Z = info->L1_Z1;

        initZBuffer(info);

        for (int n = 0; n < info->N_iter; ++n) {
#if VQ_ENABLED == 1

            run_gemv_pipelinevq_fused(info, 0 /*m*/, n, &DMA_L1_Z, &REDMULE_L1_Z);
            // run_gemv_pipelinevq_spatz(info, 0 /*m*/, n, &DMA_L1_Z, &REDMULE_L1_Z);
            // run_gemv_pipelinevq(info, 0 /*m*/, n, &DMA_L1_Z, &REDMULE_L1_Z);

#else
            run_gemv_pipeline(info, 0 /*m*/, n, &DMA_L1_Z, &REDMULE_L1_Z);
#endif
        }
        // Final store (all pipelines set store_recorded/store_id)
        if (flex_is_dm_core() && info->store_active == 1) {
            summa_reduce_and_store_Z(info, DMA_L1_Z, false);
        }
        flex_global_barrier_xy();
    }
}
#endif //_SUMMA_GEMV_H_
