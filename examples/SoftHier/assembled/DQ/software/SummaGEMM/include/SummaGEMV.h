#ifndef _SUMMA_GEMV_H_
#define _SUMMA_GEMV_H_

#include "flex_dma_pattern.h"
#include "flex_group_barrier.h"
#include "flex_printf.h"
#include "flex_redmule.h"
#include "flex_runtime.h"
#include "gemm_setup.h"
#include "spatz_compute.h"
#include "summa_aqlm_pipelines.h"
#include "summa_dma.h"
#include "summa_index.h"
#include "summa_vptq_pipelines.h"
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

            // run_gemv_pipelinevq_fused(info, 0 /*m*/, n, &DMA_L1_Z, &REDMULE_L1_Z);
            // run_gemv_pipelinevq_spatz(info, 0 /*m*/, n, &DMA_L1_Z, &REDMULE_L1_Z);
            run_gemv_pipelinevq(info, 0 /*m*/, n, &DMA_L1_Z, &REDMULE_L1_Z);
            // run_gemv_pipelinevptq_baseline(info, 0 /*m*/, n, &DMA_L1_Z, &REDMULE_L1_Z);

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
       //