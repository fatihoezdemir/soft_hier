#ifndef _SUMMA_GEMV_H_
#define _SUMMA_GEMV_H_

#include "flex_dma_pattern.h"
#include "flex_group_barrier.h"
#include "flex_printf.h"
#include "flex_redmule.h"
#include "flex_runtime.h"
#include "gemm_setup.h"
#include "summa_dma.h"
#include "summa_index.h"

// run_gemv_pipeline is basically run_gemm_pipeline vice versa for run_gemv_pipeline_vq , TODO change this , adapt
static inline void run_gemv_pipeline(SummaGEMMInfo* info, int m, int n, uint32_t* DMA_L1_Z, uint32_t* REDMULE_L1_Z) {
    const int tiles = info->K_iter;
    uint32_t x_buffers[2] = {info->L1_X1, info->L1_X2};
    uint32_t w_buffers[2] = {info->L1_W1, info->L1_W2};

    // ─────────────────────────────────────────────────────────────────
    // PROLOGUE: DMA X[0], W[0]
    // ─────────────────────────────────────────────────────────────────
    if (flex_is_dm_core()) {
        if (info->cluster_for_rowwise == 1) {
            summa_load_X_tile(info, x_buffers[0], m, n, 0);
        }
        if (info->cluster_for_colwise == 1) {
            summa_load_W_tile(info, w_buffers[0], m, n, 0);
        }
    }
    flex_intra_cluster_sync();
    // Trigger REDMULE for tile 0 (non-blocking)
    grid_sync_group_barrier_xy(&(info->group));
    if (flex_is_first_core()) {
        flex_redmule_config(info->M_tile, info->K_tile, info->N_tile);
        flex_redmule_trigger(x_buffers[0], w_buffers[0], *REDMULE_L1_Z, REDMULE_COMPUTE_TYPE);
    }
    flex_intra_cluster_sync();

    int next_x_tile = 1;
    int next_w_tile = 1;

    // ─────────────────────────────────────────────────────────────────
    // Pipeline STEADY STATE: overlap DMA (tile k) with compute (tile k-1)
    // ─────────────────────────────────────────────────────────────────
    for (int tile = 1; tile < tiles; ++tile) {
        uint32_t redmule_x = x_buffers[tile & 0x1];
        uint32_t redmule_w = w_buffers[tile & 0x1];

        // Keep SUMMA group clusters in lockstep
        grid_sync_group_barrier_xy(&(info->group));

        // DMA prefetch next tile while REDMULE works on previous
        if (flex_is_dm_core()) {
            if (info->cluster_for_rowwise == 1 && next_x_tile < tiles) {
                summa_load_X_tile(info, x_buffers[next_x_tile & 0x1], m, n, next_x_tile);
                ++next_x_tile;
            }
            if (info->cluster_for_colwise == 1 && next_w_tile < tiles) {
                summa_load_W_tile(info, w_buffers[next_w_tile & 0x1], m, n, next_w_tile);
                ++next_w_tile;
            }
        }

        // Ensure DMA is done before we fence compute and store
        flex_intra_cluster_sync();

        // Finish REDMULE for tile (tile-1) before storing its result
        if (flex_is_first_core()) {
            flex_redmule_wait();
        }
        flex_intra_cluster_sync();

        // Store the completed tile (tile-1) if requested
        if (flex_is_dm_core() && info->store_active == 1 && info->store_recorded == 1) {
            if (info->group_reduction == 1) {
                uint32_t start_id = (info->store_id < info->store_step) ? 0 : info->store_id - info->store_step;
                uint32_t bid      = (start_id + info->store_id_offset) % info->summa_group_x;
                uint32_t eid      = (info->store_id + info->store_id_offset) % info->summa_group_x;
                if (((info->cluster_in_group_id_x >= bid && info->cluster_in_group_id_x < eid) && eid > bid) ||
                    ((info->cluster_in_group_id_x >= bid || info->cluster_in_group_id_x < eid) && eid <= bid)) {
                    // if (flex_is_dm_core() && flex_get_cluster_id() == 0)
                        // flex_timer_start();
                    summa_reduce_and_store_Z(info, *DMA_L1_Z, true);
                    // if (flex_is_dm_core() && flex_get_cluster_id() == 0)
                        // flex_timer_end();
                }

                info->store_id = start_id;
            } else if (info->group_reduction == 0) {
                // No inter-group reduction: store tile immediately and clear buffer
                summa_reduce_and_store_Z(info, *DMA_L1_Z, true);
                info->store_id       = info->summa_group_x;
                info->store_recorded = 0;
            }
        }

        // Trigger compute for tile (tile)
        if (flex_is_first_core()) {
            flex_redmule_config(info->M_tile, info->K_tile, info->N_tile);
            flex_redmule_trigger(redmule_x, redmule_w, *REDMULE_L1_Z, REDMULE_COMPUTE_TYPE);
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

    /*

    │ P2: Trigger REDMULE Z[0] = X[0] * W[0] (non-blocking)           │

        Pipeline Prologue:
        ┌──────────────────────────────────────────────────────┐
        │             P0             │ P1                      │
        ├──────────────────────────────────────────────────────┤
        │DMA      │idx[0],scale s[0]  │ idx[1] ,s[1],X[0]      │
        │SPATZ    │ ---               │ W[0]=DQ(idx[0],s[0])   │
        │REDMULE  │ ---               │  ---                   │
        └──────────────────────────────────────────────────────┘
        */
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

//     // ─────────────────────────────────────────────────────────────────
//     // PROLOGUE P1: DMA X[0] + idx/scale[1] || SPATZ dequantize W[0]
//     // ─────────────────────────────────────────────────────────────────
//     if (flex_is_dm_core()) {
//         if (info->cluster_for_rowwise == 1) {
//             summa_load_X_tile(info, x_buffers[0], m, n, 0);
//         }
//         if (info->cluster_for_colwise == 1 && tiles > 1) {
//             summa_vq_load_indices(info, 1, m, n, 1);
// #if VQ_USE_SCALES == 1
//             summa_vq_load_scales(info, scale_buffers[1], m, n, 1);
// #endif
//         }
//     } else if (core_id == SPATZ_CORE) {

//         summa_vq_dequantize_tile(info, w_buffers[0], 0, scale_buffers[0], 0);

//     }
//     flex_intra_cluster_sync();

//     // ─────────────────────────────────────────────────────────────────
//     // PROLOGUE P2: Trigger REDMULE for tile 0
//     // ─────────────────────────────────────────────────────────────────
//     grid_sync_group_barrier_xy(&(info->group));
//     if (flex_is_first_core()) {
//         flex_redmule_config(info->M_tile, info->K_tile, info->N_tile);
//         flex_redmule_trigger(x_buffers[0], w_buffers[0], *REDMULE_L1_Z, REDMULE_COMPUTE_TYPE);
//         // NO WAIT HERE
//     }
//     flex_intra_cluster_sync();

//     // Pipeline indices for steady state
//     int next_x_tile   = 1;
//     int next_idx_tile = (tiles > 1) ? 2 : tiles;
//     int next_deq_tile = 1;

//     /*

//           Pipeline STEADY STATE: For tile k (k = 1..tiles-1)
//     While REDMULE computes tile k-1 (in background):
//       ┌─────────────────────────────────────────────────────────────┐
//       │ DMA:    Load X[k+1], idx[k+1], scale[k+1]                   │
//       │ SPATZ:  Dequantize W[k] ← idx[k], scale[k]                  │
//       │ REDMULE: Computing Z[k-1] = X[k-1] * W[k-1] (background)    │
//       └─────────────────────────────────────────────────────────────┘

//     Then:
//       1. WAIT for REDMULE to finish tile k-1
//       2. STORE tile k-1 result (after wait)
//       3. TRIGGER REDMULE for tile k (non-blocking)
//     */

//     for (int tile = 1; tile < tiles; ++tile) {
//         uint32_t redmule_x = x_buffers[tile & 0x1];
//         uint32_t redmule_w = w_buffers[tile & 0x1];

//         // Synchronize clusters in SUMMA group before  new iteration
//         grid_sync_group_barrier_xy(&(info->group));

//         // ─────────────────────────────────────────────────────────────
//         // STAGE 1: While REDMULE computes tile (tile-1), prefetch tiles
//         // ─────────────────────────────────────────────────────────────
//         if (flex_is_dm_core()) {
//             // DMA: Load X for tile (tile)
//             if (info->cluster_for_rowwise == 1 && next_x_tile < tiles) {
//                 uint32_t dst_x = x_buffers[next_x_tile & 0x1];
//                 summa_load_X_tile(info, dst_x, m, n, next_x_tile);
//                 ++next_x_tile;
//             }

//             // DMA: Load indices/scales for tile (tile+1)
//             if (info->cluster_for_colwise == 1 && next_idx_tile < tiles) {
//                 int buffer_idx = next_idx_tile & 0x1;
//                 summa_vq_load_indices(info, buffer_idx, m, n, next_idx_tile);
// #if VQ_USE_SCALES == 1
//                 uint32_t dst_scale = scale_buffers[buffer_idx];
//                 summa_vq_load_scales(info, dst_scale, m, n, next_idx_tile); //
// #endif
//                 ++next_idx_tile;
//             }
//         } else if (core_id == SPATZ_CORE) {
//             // SPATZ: Dequantize W for tile (tile)
//             if (next_deq_tile < tiles) {
//                 uint32_t dst_w     = w_buffers[next_deq_tile & 0x1];
//                 int buffer_idx     = next_deq_tile & 0x1;
//                 uint32_t src_scale = scale_buffers[buffer_idx];
//                 if (flex_get_cluster_id() == 0)
//                     flex_timer_start();
//                 summa_vq_dequantize_tile(info, dst_w, buffer_idx, src_scale, next_deq_tile);
//                 if (flex_get_cluster_id() == 0)
//                     flex_timer_end();
//                 ++next_deq_tile;
//             }
//         }

//         // Ensure DMA and SPATZ finish their preparation work
//         flex_intra_cluster_sync();

//         // wait for RedMulE beofore finishing tile (tile-1)
//         if (flex_is_first_core()) {
//             flex_redmule_wait(); // ← CRITICAL: Fence before store!
//         }
//         flex_intra_cluster_sync();

//         // STAGE 3: STORE tile (tile-1) result (now safe - compute finished!)
//         if (flex_is_dm_core() && info->store_recorded == 1 && info->store_active == 1) {
//             if (info->group_reduction == 0) {
//                 summa_reduce_and_store_Z(info, *DMA_L1_Z, true);
//                 info->store_id = info->summa_group_x;
//                 info->store_recorded = 0;
//             } else {
//                 uint32_t start_id = (info->store_id < info->store_step) ? 0 : info->store_id - info->store_step;
//                 uint32_t bid      = (start_id + info->store_id_offset) % info->summa_group_x;
//                 uint32_t eid      = (info->store_id + info->store_id_offset) % info->summa_group_x;

//                 if (((info->cluster_in_group_id_x >= bid && info->cluster_in_group_id_x < eid) && eid > bid) ||
//                     ((info->cluster_in_group_id_x >= bid || info->cluster_in_group_id_x < eid) && eid <= bid)) {
//                     if (flex_is_dm_core() && flex_get_cluster_id() == 0)
//                         flex_timer_start();
//                     summa_reduce_and_store_Z(info, *DMA_L1_Z, true);
//                     if (flex_is_dm_core() && flex_get_cluster_id() == 0)
//                         flex_timer_end();
//                 }

//                 info->store_id = start_id;
//             }
//         }

//         // ─────────────────────────────────────────────────────────────
//         // STAGE 4: TRIGGER REDMULE for tile (tile)
//         // ─────────────────────────────────────────────────────────────
//         if (flex_is_first_core()) {
//             flex_redmule_config(info->M_tile, info->K_tile, info->N_tile);
//             flex_redmule_trigger(redmule_x, redmule_w, *REDMULE_L1_Z, REDMULE_COMPUTE_TYPE);
//             // NO WAIT! Let next iteration's DMA/SPATZ overlap with this compute
//         }
//         flex_intra_cluster_sync();
//     }

//     /*
//     ═══════════════════════════════════════════════════════════════════
//     Pipeline Epilogue: Drain the last tile and finalize
//     ═══════════════════════════════════════════════════════════════════
//     */
//     if (info->group_reduction == 1) {
//         flex_global_barrier_xy();
//     } else {
//         grid_sync_group_barrier_xy(&(info->group));
//     }

//     info->store_recorded = 1;
//     info->store_m        = m;
//     info->store_n        = n;
//     info->store_id       = info->summa_group_x;

//     // Swap Z buffers for final store
//     uint32_t tmp_z = *DMA_L1_Z;
//     *DMA_L1_Z      = *REDMULE_L1_Z;
//     *REDMULE_L1_Z  = tmp_z;

//     if (flex_is_first_core()) {
//         flex_redmule_wait(); // Wait for last tile's compute to finish
//     }
//     flex_intra_cluster_sync();

    
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
    // ═══════════════════════════════════════════════════════════════════════

    if (info->cluster_active) {
        uint32_t DMA_L1_Z     = info->L1_Z2;
        uint32_t REDMULE_L1_Z = info->L1_Z1;

        initZBuffer(info);

        for (int n = 0; n < info->N_iter; ++n) {
#if VQ_ENABLED == 1
            run_gemv_pipelinevq(info, 0 /*m*/, n, &DMA_L1_Z, &REDMULE_L1_Z);
#else
            run_gemv_pipeline(info, 0 /*m*/, n, &DMA_L1_Z, &REDMULE_L1_Z);
#endif
        }
        // store
        if (flex_is_dm_core() && info->store_active == 1) {
            summa_reduce_and_store_Z(info, DMA_L1_Z, false);
        }
        flex_global_barrier_xy();
    }
}
#endif //_SUMMA_GEMV_H_
