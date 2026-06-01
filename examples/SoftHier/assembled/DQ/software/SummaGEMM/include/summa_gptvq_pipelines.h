#ifndef SUMMA_GPTVQ_PIPELINES_HH
#define SUMMA_GPTVQ_PIPELINES_HH

#include "flex_transpose_engine.h"
#include "gemm.h"
#include "gemm_setup.h"
#include "summa_dma.h"
#include "vq_debug.h"
#include "vq_kernels.h"
#include "vq_kernels_gptvq.h"

#if VQ_ENABLED == 1

#if VQ_TRANSPOSE_ENABLED == 1
static inline void summa_gptvq_transpose_tile(const SummaGEMMInfo* info, uint32_t src_addr, uint32_t dst_addr) {
    flex_transpose_engine_config(info->N_tile, info->K_tile, src_addr, dst_addr, DATA_TYPE_BYTE);
    flex_transpose_engine_trigger();
    flex_transpose_engine_wait();
}
#endif

#ifndef GPTVQ_DEBUG_TILE_COMPARE
#define GPTVQ_DEBUG_TILE_COMPARE 0
#endif

#ifndef GPTVQ_DEBUG_COMPARE_GOLDEN_TILE
#define GPTVQ_DEBUG_COMPARE_GOLDEN_TILE 0
#endif

#if GPTVQ_DEBUG_TILE_COMPARE == 1 && defined(VQ_DEBUG_GPTVQ_TILE_W_T_ADDR) && defined(VQ_DEBUG_GPTVQ_TILE_W_ADDR)
static inline void summa_gptvq_debug_load_ref_tile(uint32_t dst_addr, uint64_t src_addr, uint32_t size_bytes) {
    flex_dma_async_1d(dst_addr, src_addr, size_bytes);
    flex_dma_async_wait_all();
}
#endif

// Global default GPTVQ dequant kernel.
// Override once before including this header:
//   #define SUMMA_GPTVQ_DEQ_KERNEL summa_vq_dequantize_tile_gptvq_baseline
// Valid options:
//   summa_vq_dequantize_tile_gptvq
//   summa_vq_dequantize_tile_gptvq_baseline
#ifndef SUMMA_GPTVQ_DEQ_KERNEL
#define SUMMA_GPTVQ_DEQ_KERNEL summa_vq_dequantize_tile_gptvq_baseline
#endif

#ifndef SUMMA_GEMM_GPTVQ_DEQ_KERNEL
#define SUMMA_GEMM_GPTVQ_DEQ_KERNEL SUMMA_GPTVQ_DEQ_KERNEL
#endif

#define SUMMA_GEMM_GPTVQ_DEQ_CALL(info, dst_w, buffer_idx, src_scale, tile_idx, deq_k_start, deq_k_rows) \
    SUMMA_GEMM_GPTVQ_DEQ_KERNEL((info), (dst_w), (buffer_idx), (src_scale), (tile_idx), (deq_k_start), (deq_k_rows))

static inline void run_gemm_pipelinegptvq(SummaGEMMInfo* info, int m, int n, uint32_t* DMA_L1_Z,
                                          uint32_t* REDMULE_L1_Z) {
    const int tiles = info->K_iter;
    const uint32_t spatz_core = info->spatz_compute_core;
    const uint32_t core_id = flex_get_core_id();

    if (tiles <= 0) {
        flex_global_barrier_xy();
        return;
    }

    uint32_t x_buffers[2] = {info->L1_X1, info->L1_X2};
    uint32_t w_buffers[2] = {info->L1_W1, info->L1_W2};

    for (int tile = 0; tile < tiles; ++tile) {
        const uint32_t cur_x = x_buffers[tile & 0x1];
        const uint32_t cur_w = w_buffers[tile & 0x1];
        const int buffer_idx = tile & 0x1;

        grid_sync_group_barrier_xy(&(info->group));

        if (flex_is_dm_core()) {
            if (info->cluster_for_rowwise == 1) {
                summa_load_X_tile(info, cur_x, m, n, tile);
            }
            if (info->cluster_for_colwise == 1) {
                summa_vq_load_codebooks(info, buffer_idx, m, n, tile);
                summa_vq_load_indices(info, buffer_idx, m, n, tile);
            }
        }
        flex_intra_cluster_sync();
        grid_sync_group_barrier_xy(&(info->group));

        if (!flex_is_dm_core() && core_id == spatz_core) {
            uint32_t deq_dst = cur_w;
#if VQ_TRANSPOSE_ENABLED == 1
            deq_dst = info->L1_DETRANSPOSED;
#endif
            SUMMA_GEMM_GPTVQ_DEQ_CALL(info, deq_dst, buffer_idx, 0u, tile, 0u, info->vq.K_tile_compressed);
        }
        flex_intra_cluster_sync();

#if VQ_TRANSPOSE_ENABLED == 1
        if (flex_is_first_core()) {
            summa_gptvq_transpose_tile(info, info->L1_DETRANSPOSED, cur_w);
        }
        flex_intra_cluster_sync();
#endif

        if (tile > 0) {
            if (flex_is_first_core()) {
                flex_redmule_wait();
            }
            flex_intra_cluster_sync();

            if (flex_is_dm_core() && info->store_recorded == 1 && info->store_active == 1) {
                uint32_t start_id = (info->store_id < info->store_step) ? 0 : info->store_id - info->store_step;
                uint32_t bid = (start_id + info->store_id_offset) % info->summa_group_x;
                uint32_t eid = (info->store_id + info->store_id_offset) % info->summa_group_x;

                if (((info->cluster_in_group_id_x >= bid && info->cluster_in_group_id_x < eid) && eid > bid) ||
                    ((info->cluster_in_group_id_x >= bid || info->cluster_in_group_id_x < eid) && eid <= bid)) {
                    summa_reduce_and_store_Z(info, *DMA_L1_Z, true);
                }

                info->store_id = start_id;
            }
        }

        if (flex_is_first_core()) {
            flex_redmule_config(info->M_tile, info->K_tile, info->N_tile);
            flex_redmule_trigger(cur_x, cur_w, *REDMULE_L1_Z, REDMULE_COMPUTE_TYPE);
        }
        flex_intra_cluster_sync();
    }

    if (info->group_reduction == 1) {
        flex_global_barrier_xy();
    } else {
        grid_sync_group_barrier_xy(&(info->group));
    }

    info->store_recorded = 1;
    info->store_m = m;
    info->store_n = n;
    info->store_id = info->summa_group_x;

    uint32_t tmp_z = *DMA_L1_Z;
    *DMA_L1_Z = *REDMULE_L1_Z;
    *REDMULE_L1_Z = tmp_z;

    if (flex_is_first_core()) {
        flex_redmule_wait();
    }
    flex_intra_cluster_sync();
}

#endif
#endif
