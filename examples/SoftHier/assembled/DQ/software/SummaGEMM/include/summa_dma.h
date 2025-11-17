#ifndef _SUMMA_DMA_H_
#define _SUMMA_DMA_H_

#include "flex_dma_pattern.h"
#include "gemm_setup.h"
#include "preload.h"

static inline void initZBuffer(const SummaGEMMInfo* info) {
    // Initialize Z buffer
    if (flex_is_dm_core()) {
        flex_dma_async_1d(info->L1_Z1, zomem(0), info->L1_Z_size);
        flex_dma_async_1d(info->L1_Z2, zomem(0), info->L1_Z_size);
        flex_dma_async_wait_all();
    }
    flex_intra_cluster_sync();
}

static inline void summa_load_W_tile(const SummaGEMMInfo* info, uint32_t dst_L1_W, int m, int n, int k) {
    flex_dma_async_2d(dst_L1_W, /*destination*/
                      info->W_tile_base + m * info->W_tile_M_iter_offset + n * info->W_tile_N_iter_offset +
                          k * info->W_tile_K_iter_offset, /*source*/
                      info->N_tile * DATA_TYPE_BYTE,      /*transfer size*/
                      info->N_tile * DATA_TYPE_BYTE,      /*destination stride*/
                      info->N_size * DATA_TYPE_BYTE,      /*source stride*/
                      info->K_tile /*repeat*/);           // Start 2D iDMA
    flex_dma_async_wait_all();                            // Wait for iDMA Finishing
    // col-wise multicast
    if (info->summa_group_y > 1) {
        flex_dma_async_broadcast(dst_L1_W /*dst_offset*/, dst_L1_W /*src_offset*/, info->L1_W_size /*transfer_size*/,
                                 (ARCH_NUM_CLUSTER_X - 1) /*row_mask*/, info->group.wakeup_col_mask /*col_mask*/);
        flex_dma_async_wait_all(); // Wait for iDMA Finishing
    }
}

static inline void summa_load_X_tile(const SummaGEMMInfo* info, uint32_t dst_L1_X, int m, int n, int k) {
#if GEMM_RESHA_X_FROM_ENABLE == 1
    uint64_t mapped_offset = summa_compute_reshaped_X_offset(info, m, n, k);
    flex_dma_async_2d(dst_L1_X, info->X_tile_base + mapped_offset, info->K_tile * DATA_TYPE_BYTE,
                      info->K_tile * DATA_TYPE_BYTE, GEMM_RESHAPE_X_FROM_K * DATA_TYPE_BYTE, info->M_tile);
#else
    uint64_t base = (uint64_t)m * info->X_tile_M_iter_offset + (uint64_t)n * info->X_tile_N_iter_offset +
                    (uint64_t)k * info->X_tile_K_iter_offset;
    flex_dma_async_2d(dst_L1_X, info->X_tile_base + base, info->K_tile * DATA_TYPE_BYTE, info->K_tile * DATA_TYPE_BYTE,
                      info->K_size * DATA_TYPE_BYTE, info->M_tile);
#endif
    flex_dma_async_wait_all();

    if (info->summa_group_x > 1) {
        flex_dma_async_broadcast(dst_L1_X, dst_L1_X, info->L1_X_size, info->group.wakeup_row_mask,
                                 (ARCH_NUM_CLUSTER_Y - 1));
        flex_dma_async_wait_all();
    }
}

static inline void summa_reduce_and_store_Z(SummaGEMMInfo* info, uint32_t DMA_L1_Z, bool clear) {
    if (info->group_reduction == 1) {
        flex_dma_async_reduction(DMA_L1_Z, DMA_L1_Z, info->L1_Z_size, COLLECTIVE_REDSUM_TYPE,
                                 ~info->group.wakeup_row_mask, ~info->group.wakeup_col_mask);
        flex_dma_async_wait_all();
    }

#if GEMM_RESHA_Z_TO_ENABLE == 1
    uint64_t mapped_offset = summa_compute_reshaped_Z_offset(info, info->store_m, info->store_n);

    flex_dma_async_2d(info->Z_tile_base + mapped_offset, DMA_L1_Z, info->N_tile * DATA_TYPE_BYTE,
                      GEMM_RESHAPE_Z_TO_N * DATA_TYPE_BYTE, info->N_tile * DATA_TYPE_BYTE, info->M_tile);
#else
    uint64_t base =
        (uint64_t)info->store_m * info->Z_tile_M_iter_offset + (uint64_t)info->store_n * info->Z_tile_N_iter_offset;

    flex_dma_async_2d(info->Z_tile_base + base, DMA_L1_Z, info->N_tile * DATA_TYPE_BYTE, info->N_size * DATA_TYPE_BYTE,
                      info->N_tile * DATA_TYPE_BYTE, info->M_tile);
#endif
    flex_dma_async_wait_all();
    if (clear) {
        // Clear Z buffer for reuse
        flex_dma_async_1d(DMA_L1_Z, zomem(0), info->L1_Z_size);
        flex_dma_async_wait_all();
        /* code */
    }
}
#if VQ_ENABLED == 1
static inline void vq_load_cb(SummaGEMMInfo* info) {

    flex_dma_async_1d((uint64_t)(uintptr_t) info->L1_CB[0], (uint64_t)(uintptr_t)VQ_CODEBOOKS_ADDR,
                      info->L1_CB_size);

    flex_dma_async_1d((uint64_t)(uintptr_t) info->L1_CB[1], (uint64_t)(uintptr_t)VQ_CODEBOOKS_ADDR+info->L1_CB_size,
                      info->L1_CB_size);         
    flex_dma_async_wait_all();

}






#endif


#endif //_SUMMA_DMA_H_