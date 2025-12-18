#ifndef _SUMMA_GEMM_H_
#define _SUMMA_GEMM_H_

#include "flex_dma_pattern.h"
#include "flex_group_barrier.h"
#include "flex_printf.h"
#include "flex_redmule.h"
#include "flex_runtime.h"
#include "gemm_setup.h"
#include "summa_aqlm_pipelines.h"
#include "summa_dma.h"
#include "summa_index.h"
#include "summa_vptq_pipelines.h"
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
#if VQ_ENABLED == 1
                run_gemm_pipelinevq(info, m, n, &DMA_L1_Z, &REDMULE_L1_Z);
                // run_gemm_pipelinevptq_baseline(info, m, n, &DMA_L1_Z, &REDMULE_L1_Z);

#else
                run_gemm_pipeline(info, m, n, &DMA_L1_Z, &REDMULE_L1_Z);
#endif
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
