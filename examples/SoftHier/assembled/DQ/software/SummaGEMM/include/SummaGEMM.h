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
/*
     ┌─────┬─────┬─────┐
     │X00→→│→→→→→│→→→→→│  Row 0 broadcasts X00
     ├─────┼─────┼─────┤
     │X10→→│→→→→→│→→→→→│  Row 1 broadcasts X10
     ├─────┼─────┼─────┤
     │X20→→│→→→→→│→→→→→│  Row 2 broadcasts X20
     └─────┴─────┴─────┘
       ↑     ↑     ↑
       W00   W01   W02    (Column broadcasts:indices or weights)
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

static inline void run_gemm_pipelinevq(SummaGEMMInfo* info, int m, int n, uint32_t* DMA_L1_Z, uint32_t* REDMULE_L1_Z) {
    const int tiles      = info->K_iter;
    const int SPATZ_CORE = 1;
    uint32_t core_id     = flex_get_core_id();
    if (tiles > 0) {
        uint32_t x_buffers[2]   = {info->L1_X1, info->L1_X2};
        uint32_t w_buffers[2]   = {info->L1_W1, info->L1_W2};
        uint32_t idx_buffers[2] = {info->vq.L1_IDX1[0], info->vq.L1_IDX2[0]};
#if VQ_USE_SCALES == 1
        uint32_t scale_buffers[2] = {info->vq.L1_Scales[0], info->vq.L1_Scales[1]};
#else
        uint32_t scale_buffers[2] = {0u, 0u};
#endif
        /*
        Pipeline Prologue:
        ┌───────────────────────────────────────────────────────────────┐
        │             k-2                      │ k-1                    │
        ├───────────────────────────────────────────────────────────────┤
        │DMA      │ codebook,idx[0],scale s[0] │ idx[1] ,s[1],X[0]      │
        │SPATZ    │ ---                        │ W[0]=DQ(idx[0],s[0])   │
        │REDMULE  │ ---                        │  ---                   │
        └───────────────────────────────────────────────────────────────┘
        */
        // Prologue P0: DMA only (indices + scales)
        if (flex_is_dm_core() && info->cluster_for_colwise == 1) {
            summa_vq_load_indices(info, idx_buffers[0], m, n, 0);
#if VQ_USE_SCALES == 1
            summa_vq_load_scales(info, scale_buffers[0], m, n, 0);
#endif
        }
        flex_intra_cluster_sync();

        // Prologue P1: overlap DMA for X/next indices with RVV dequantization
        if (flex_is_dm_core()) {
            if (info->cluster_for_rowwise == 1) {
                summa_load_X_tile(info, x_buffers[0], m, n, 0);
            }
            if (info->cluster_for_colwise == 1 && tiles > 1) {
                summa_vq_load_indices(info, idx_buffers[1], m, n, 1);
#if VQ_USE_SCALES == 1
                summa_vq_load_scales(info, scale_buffers[1], m, n, 1);
#endif
            }
        } else if (core_id == SPATZ_CORE) {
            summa_vq_dequantize_tile(info, w_buffers[0], idx_buffers[0], scale_buffers[0], 0);
        }
        flex_intra_cluster_sync();
        flex_global_barrier_xy();

        int next_x_tile   = 1;
        int next_idx_tile = (tiles > 1) ? 2 : tiles;
        int next_deq_tile = 1;
        /*
        Pipeline Steady State:
        ┌───────────────────────────────────────────────────────────────┐
        │             k                        │ k+1                    │
        ├───────────────────────────────────────────────────────────────┤
        │DMA      │ idx[0],s[0],X[1]           │ idx[1],s[1],X[0]       │
        │SPATZ    │ W[1]=DQ(idx[1],s[1])       │ W[0]=DQ(idx[0],s[0])   │
        │REDMULE  │ Z[0]=X[0] * W[0]           │  Z[1]=X[1] * W[1]      │
        └───────────────────────────────────────────────────────────────┘
        */
        for (int tile = 0; tile < tiles; ++tile) {
            uint32_t redmule_x = x_buffers[tile & 0x1];
            uint32_t redmule_w = w_buffers[tile & 0x1];

            grid_sync_group_barrier_xy(&(info->group));
            if (flex_is_first_core())
                flex_redmule_wait();
            flex_intra_cluster_sync();

            if (flex_is_dm_core()) {
                if (info->cluster_for_rowwise == 1 && next_x_tile < tiles) {
                    uint32_t dst_x = x_buffers[next_x_tile & 0x1];
                    summa_load_X_tile(info, dst_x, m, n, next_x_tile);
                    ++next_x_tile;
                }
                if (info->cluster_for_colwise == 1 && next_idx_tile < tiles) {
                    uint32_t dst_idx = idx_buffers[next_idx_tile & 0x1];
                    summa_vq_load_indices(info, dst_idx, m, n, next_idx_tile);
#if VQ_USE_SCALES == 1
                    uint32_t dst_scale = scale_buffers[next_idx_tile & 0x1];
                    summa_vq_load_scales(info, dst_scale, m, n, next_idx_tile);
#endif
                    ++next_idx_tile;
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
            } else {
                if (next_deq_tile < tiles && core_id == SPATZ_CORE) {
                    uint32_t dst_w     = w_buffers[next_deq_tile & 0x1];
                    uint32_t src_idx   = idx_buffers[next_deq_tile & 0x1];
                    uint32_t src_scale = scale_buffers[next_deq_tile & 0x1];
                    summa_vq_dequantize_tile(info, dst_w, src_idx, src_scale, next_deq_tile);
                    ++next_deq_tile;
                }
            }

            flex_intra_cluster_sync();

            if (flex_is_first_core()) {
                flex_redmule_config(info->M_tile, info->K_tile, info->N_tile);
                flex_redmule_trigger(redmule_x, redmule_w, *REDMULE_L1_Z, REDMULE_COMPUTE_TYPE);
            }
        }
    } else {
        flex_global_barrier_xy();
    }

    // Pipeline Epilogue : Storing
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

void SummaGEMMRun(SummaGEMMInfo* info) {
    flex_global_barrier_xy();

    // Load VQ codebooks - all cores must call this for proper synchronization
  if (info->cluster_for_colwise == 1){
    vq_load_cb(info);
  }
    flex_global_barrier_xy();

    if (flex_get_cluster_id() == 15 && flex_is_dm_core()) {
        printf("\n=== VQ Codebook Load Complete ===\n Cluster 15 VQ Codebooks:\n");
        for (int i = 0; i < VQ_NUM_CBS; ++i) {
            const uint16_t* cb_ptr = (const uint16_t*)(uintptr_t)info->vq.L1_CB[i];
            VQ_PRINT_CB(cb_ptr, VQ_CB_NUM_CENTROIDS, VQ_GROUP_SIZE, i, 4);
        }
    }

    if (info->cluster_active) {
        if (flex_is_dm_core())
            printf("%d\t", flex_get_cluster_id());
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
