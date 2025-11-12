#ifndef _GEMM_CONFIG_H_
#define _GEMM_CONFIG_H_
#include <stdint.h>

#define GEMM_M_SIZE ((uint64_t)512)
#define GEMM_N_SIZE ((uint64_t)512)
#define GEMM_K_SIZE ((uint64_t)512)
#define GEMM_M_TILE ((uint64_t)128)
#define GEMM_N_TILE ((uint64_t)128)
#define GEMM_K_TILE ((uint64_t)128)
#define GEMM_SUMMA_SCALE_X ((uint64_t)4)
#define GEMM_SUMMA_SCALE_Y ((uint64_t)4)
#define GEMM_SUMMA_GROUP_NUMBER ((uint64_t)1)
#define GEMM_SUMMA_GROUP_REDUCE ((uint64_t)0)
#define GEMM_SUMMA_GROUP_SPLITK ((uint64_t)0)
#define GEMM_SUMMA_GROUP_SPLITN ((uint64_t)0)
#define GEMM_SUMMA_GROUP_GAP_X ((uint64_t)0)
#define GEMM_SUMMA_GROUP_GAP_W ((uint64_t)0)
#define GEMM_SUMMA_GROUP_GAP_Z ((uint64_t)0)
#define GEMM_RESHA_X_FROM_ENABLE 0
#define GEMM_RESHA_Z_TO_ENABLE 0
#define GEMM_RESHA_X_FROM_M ((uint64_t)128)
#define GEMM_RESHA_Z_TO_M ((uint64_t)2048)
#define GEMM_SUMMA_NUMER ((uint64_t)1)
#define GEMM_SUMMA_NUMER_CHUNK ((uint64_t)8192)
#define GEMM_FP16
#define DATA_TYPE_WIDTH             16
#define DATA_TYPE_BYTE              2
typedef uint16_t                    gemm_data_t;
#define REDMULE_COMPUTE_TYPE        REDMULE_FP_16
#define COLLECTIVE_REDSUM_TYPE      COLLECTIVE_REDADD_FP_16
#define COLLECTIVE_REDMAX_TYPE      COLLECTIVE_REDMAX_FP_16
#define VQ_ENABLED 1
#define VQ_ALGORITHM_AQLM
#define VQ_NUM_CBS ((uint64_t)2)
#define VQ_NBITS_PER_CB ((uint64_t)8)
#define VQ_GROUP_SIZE ((uint64_t)8)
#define VQ_CB_SIZE ((uint64_t)256)
#define VQ_NUM_GROUPS_PER_ROW ((uint64_t)64)
#define VQ_TOTAL_GROUPS ((uint64_t)32768)
#define VQ_NUM_GROUPS_PER_ROW_TILE ((uint64_t)16)
#define VQ_NUM_SCALES ((uint64_t)0)
#define VQ_CODEBOOK_FORMAT_FP16
#define VQ_INDEX_FORMAT_PACKED
#define STR(x) #x
#define XSTR(x) STR(x)

#endif // _GEMM_CONFIG_H_
