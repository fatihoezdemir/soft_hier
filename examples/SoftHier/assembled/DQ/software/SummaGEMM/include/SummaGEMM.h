#ifndef _SUMMA_GEMM_H_
#define _SUMMA_GEMM_H_

#include "flex_dma_pattern.h"
#include "flex_group_barrier.h"
#include "flex_printf.h"
#include "flex_redmule.h"
#include "flex_runtime.h"
#include "gemm_setup.h"
#include "summa_dma.h"
#include "summa_index.h"
#include "vq_kernels.h"
/*
     ┌─────┬─────┬─────┐
     │X00→→│→→→→→│→→→→→│  Row 0 broadcasts X00
     ├─────┼─────┼─────┤
     │X10→→│→→→→→│→→→→→│  Row 1 broadcasts X10
     ├─────┼─────┼─────┤
     │X20→→│→→→→→│→→→→→│  Row 2 broadcasts X20
     └─────┴─────┴─────┘
       ↑     ↑     ↑
       W00   W01   W02    (Column broadcasts:indices+scales or weights)
       ↑     ↑     ↑
*/
static inline void run_gemm_pipeline(SummaGEMMInfo* info, int m, int n, uint32_t* DMA_L1_Z, uint32_t* REDMULE_L1_Z) {
    // Prefetching PIPELINE PROLOGUE_|""
    if (flex_is_dm_core()) {
        if (info->cluster_for_rowwise == 1) {
            // load X from west edge
            summa_load_X_tile(info, info->L1_X1, m, n, 0);
        }

        if (info->cluster_for_colwise == 1) {
            // load W from south edge
            summa_load_W_tile(info, info->L1_W1, m, n, 0);
        }
    }

    // PIPELINE START
    // Double buffering, update pointers
    for (int k = 1; k <= info->K_iter; ++k) {
        uint32_t DMA_L1_X;
        uint32_t DMA_L1_W;
        uint32_t REDMULE_L1_X;
        uint32_t REDMULE_L1_W;

        DMA_L1_X     = (k % 2 == 1) ? info->L1_X2 : info->L1_X1;
        DMA_L1_W     = (k % 2 == 1) ? info->L1_W2 : info->L1_W1;
        REDMULE_L1_X = (k % 2 == 1) ? info->L1_X1 : info->L1_X2;
        REDMULE_L1_W = (k % 2 == 1) ? info->L1_W1 : info->L1_W2;
        // SYNC
        grid_sync_group_barrier_xy(&(info->group));
        if (flex_is_first_core())
            flex_redmule_wait();
        flex_intra_cluster_sync();
        if (flex_is_dm_core()) {
            if (k < info->K_iter) {
                if (info->cluster_for_rowwise == 1) {
                    // load X from west edge
                    summa_load_X_tile(info, DMA_L1_X, m, n, k);
                }
                if (info->cluster_for_colwise == 1) {
                    // load W from south edge
                    summa_load_W_tile(info, DMA_L1_W, m, n, k);
                }
            }

            if (info->store_recorded == 1 && info->store_active == 1) {
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

        if (flex_is_first_core()) {
            flex_redmule_config(info->M_tile, info->K_tile, info->N_tile);
            flex_redmule_trigger(REDMULE_L1_X, REDMULE_L1_W, *REDMULE_L1_Z, REDMULE_COMPUTE_TYPE);
        }
    }

    // Storing
    if (info->group_reduction == 1) {
        flex_global_barrier_xy();
    } else {
        grid_sync_group_barrier_xy(&(info->group));
    }
    info->store_recorded = 1;
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
static inline void run_gemm_pipelinevq(SummaGEMMInfo* info, int m, int n, uint32_t* DMA_L1_Z, uint32_t* REDMULE_L1_Z) {
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

    /*

          Pipeline STEADY STATE: For tile k (k = 1..tiles-1)
    While REDMULE computes tile k-1 (in background):
      ┌─────────────────────────────────────────────────────────────┐
      │ DMA:    Load X[k+1], idx[k+1], scale[k+1]                   │
      │ SPATZ:  Dequantize W[k] ← idx[k], scale[k]                  │
      │ REDMULE: Computing Z[k-1] = X[k-1] * W[k-1] (background)    │
      └─────────────────────────────────────────────────────────────┘

    Then:
      1. WAIT for REDMULE to finish tile k-1
      2. STORE tile k-1 result (after wait)
      3. TRIGGER REDMULE for tile k (non-blocking)
    */

    for (int tile = 1; tile < tiles; ++tile) {
        uint32_t redmule_x = x_buffers[tile & 0x1];
        uint32_t redmule_w = w_buffers[tile & 0x1];

        // Synchronize clusters in SUMMA group before  new iteration
        grid_sync_group_barrier_xy(&(info->group));

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
                summa_vq_load_scales(info, dst_scale, m, n, next_idx_tile);
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

        // wait for RedMulE beofore finishing tile (tile-1)
        if (flex_is_first_core()) {
            flex_redmule_wait(); // ← CRITICAL: Fence before store!
        }
        flex_intra_cluster_sync();

        // STAGE 3: STORE tile (tile-1) result (now safe - compute finished!)
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

    /*
    ═══════════════════════════════════════════════════════════════════
    Pipeline Epilogue: Drain the last tile and finalize
    ═══════════════════════════════════════════════════════════════════
    */
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

#endif

void SummaGEMMRun(SummaGEMMInfo* info) {
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

        for (int m = 0; m < info->M_iter; ++m) {
            for (int n = 0; n < info->N_iter; ++n) {
                run_gemm_pipelinevq(info, m, n, &DMA_L1_Z, &REDMULE_L1_Z);
            }
        }
        // store
        if (flex_is_dm_core() && info->store_active == 1) {
            summa_reduce_and_store_Z(info, DMA_L1_Z, false);
        }
        flex_global_barrier_xy();
    }
}
#endif //_SUMMA_GEMM_H_
