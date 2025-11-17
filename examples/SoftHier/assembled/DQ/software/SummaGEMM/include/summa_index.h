#include "gemm_setup.h"
#if GEMM_RESHA_X_FROM_ENABLE == 1
static inline uint64_t summa_compute_reshaped_X_offset(const SummaGEMMInfo* info, int m, int n, int k) {
    uint64_t origin_elem_offest = (info->X_tile_base_offset + m * info->X_tile_M_iter_offset +
                                   n * info->X_tile_N_iter_offset + k * info->X_tile_K_iter_offset) /
                                  DATA_TYPE_BYTE;
    uint64_t origin_k = origin_elem_offest % GEMM_K_SIZE;
    uint64_t origin_m = origin_elem_offest / GEMM_K_SIZE;
#if defined(GEMM_RESHA_X_FROM_TALL)
    uint64_t num_bulk = origin_k / GEMM_RESHAPE_X_FROM_K;
    uint64_t mapped_k = origin_k % GEMM_RESHAPE_X_FROM_K;
    uint64_t mapped_m = num_bulk * info->M_size + origin_m;
#endif
#if defined(GEMM_RESHA_X_FROM_THIN)
    uint64_t num_bulk = origin_m / GEMM_RESHA_X_FROM_M;
    uint64_t mapped_m = origin_m % GEMM_RESHA_X_FROM_M;
    uint64_t mapped_k = num_bulk * info->K_size + origin_k;
#endif
    uint64_t mapped_offset = (mapped_m * GEMM_RESHAPE_X_FROM_K + mapped_k) * DATA_TYPE_BYTE;

    return (mapped_m * GEMM_RESHAPE_X_FROM_K + mapped_k) * DATA_TYPE_BYTE;
}
#endif

#if GEMM_RESHA_Z_TO_ENABLE == 1
static inline uint64_t summa_compute_reshaped_Z_offset_store(const SummaGEMMInfo* info) {
    uint64_t origin_elem_offest = (info->Z_tile_base_offset + (uint64_t)info->store_m * info->Z_tile_M_iter_offset +
                                   (uint64_t)info->store_n * info->Z_tile_N_iter_offset) /
                                  DATA_TYPE_BYTE;

    uint64_t origin_n = origin_elem_offest % GEMM_N_SIZE;
    uint64_t origin_m = origin_elem_offest / GEMM_N_SIZE;

#if defined(GEMM_RESHA_Z_TO_TALL)
    uint64_t num_buln = origin_n / GEMM_RESHAPE_Z_TO_N;
    uint64_t mapped_n = origin_n % GEMM_RESHAPE_Z_TO_N;
    uint64_t mapped_m = num_buln * info->M_size + origin_m;
#elif defined(GEMM_RESHA_Z_TO_THIN)
    uint64_t num_buln = origin_m / GEMM_RESHA_Z_TO_M;
    uint64_t mapped_m = origin_m % GEMM_RESHA_Z_TO_M;
    uint64_t mapped_n = num_buln * info->N_size + origin_n;
#endif

    return (mapped_m * GEMM_RESHAPE_Z_TO_N + mapped_n) * DATA_TYPE_BYTE;
}
#endif
