#ifndef GEMM_SETUP_H_
#define GEMM_SETUP_H_

#include "flex_group_barrier.h"
#include "flex_runtime.h"
#include "gemm.h"
#if VQ_ENABLED == 1
typedef struct VQ {
    uint32_t idx_size;
    uint32_t cb_size;
    uint32_t scale_size;
    uint64_t VQ_CB_address[VQ_NUM_CBS];    // Array of codebook addresses
    uint64_t VQ_Index_address[VQ_NUM_CBS]; // Array of indices addresses (one per codebook)
    uint64_t VQ_Scale_address;
    // Addressing Information

    // L1 location information
    uint32_t L1_CB[VQ_NUM_CBS];   //
    uint32_t L1_IDX1[VQ_NUM_CBS]; // 2 indices buffers /codebook
    uint32_t L1_IDX2[VQ_NUM_CBS];

    uint32_t L1_IDX_size;
    uint32_t L1_CB_size;
#if VQ_USE_SCALES == 1
    uint32_t L1_Scales[2];
#endif
    // Pre-computed constants to avoid recomputation during load
    uint32_t N_compressed;              // N_size / VQ_GROUP_SIZE (full stride)
    uint32_t N_tile_compressed;         // N_tile / VQ_GROUP_SIZE
    uint32_t N_compressed_per_group;    // (N_size_per_group) / VQ_GROUP_SIZE
    uint32_t N_group_offset_compressed; // group_n_offset / VQ_GROUP_SIZE
} vq;

#endif

static inline void summa_config_assert(uint32_t condition, uint32_t eoc_code) {
    if (!condition) {
        printf("exitin");
        flex_eoc(eoc_code);
        while (1) {
        }
    }
}

typedef struct SummaGEMMInfo {
    // General information
    uint64_t X_address;
    uint64_t W_address;
    uint64_t Z_address;
    uint32_t M_size;
    uint32_t N_size;
    uint32_t N_size_per_group;
    uint32_t K_size; /*shared dimension*/
    uint32_t M_tile;
    uint32_t N_tile;
    uint32_t K_tile;
    uint32_t group_reduction;
    uint32_t group_splitK;
    uint32_t group_splitN;
    uint32_t group_n_offset;
    uint32_t summa_group_x;
    uint32_t summa_group_y;
    uint32_t summa_groups;

    // Cluster information
    FlexPosition cluster_pos;
    uint32_t cluster_global_id;
    uint32_t cluster_active;

    // Group Info
    GridSyncGroupInfo group;
    uint32_t cluster_in_group_id;
    uint32_t cluster_in_group_id_x;
    uint32_t cluster_in_group_id_y;
    uint32_t cluster_for_rowwise;
    uint32_t cluster_for_colwise;

    // Tiling information
    uint32_t M_iter;
    uint32_t N_iter;
    uint32_t K_iter;

    // Store Control
    uint32_t store_recorded;
    uint32_t store_m;
    uint32_t store_n;
    uint32_t store_step;
    uint32_t store_id;
    uint32_t store_id_offset;
    uint32_t store_active;

    // Addressing Information
    uint64_t X_tile_base;
    uint64_t W_tile_base;
    uint64_t Z_tile_base;
#if GEMM_RESHA_X_FROM_ENABLE == 1
    uint64_t X_tile_base_offset;
#endif
#if GEMM_RESHA_Z_TO_ENABLE == 1
    uint64_t Z_tile_base_offset;
#endif
    uint64_t X_tile_M_iter_offset;
    uint64_t X_tile_N_iter_offset;
    uint64_t X_tile_K_iter_offset;
    uint64_t W_tile_M_iter_offset;
    uint64_t W_tile_N_iter_offset;
    uint64_t W_tile_K_iter_offset;
    uint64_t Z_tile_M_iter_offset;
    uint64_t Z_tile_N_iter_offset;

    // L1 location information
    uint32_t L1_X1;
    uint32_t L1_W1;
    uint32_t L1_Z1;
    uint32_t L1_X2;
    uint32_t L1_W2;
    uint32_t L1_Z2;

    // Usefull parameters for address calculation
    uint32_t L1_X_size;
    uint32_t L1_W_size;
    uint32_t L1_Z_size;
#if VQ_ENABLED == 1
    // General information
    vq vq;

#endif

    uint32_t L1_AREA;

} SummaGEMMInfo;

SummaGEMMInfo SummaGEMMAnaylze(uint64_t X_address, uint64_t W_address, uint64_t Z_address, uint32_t M_size,
                               uint32_t N_size, uint32_t K_size, /*shared dimension*/
                               uint32_t M_tile, uint32_t N_tile, uint32_t K_tile, uint32_t group_x, uint32_t group_y,
                               uint32_t num_group, uint32_t group_reduction, uint32_t group_splitK,
                               uint32_t group_splitN, uint32_t X_address_group_gap, uint32_t W_address_group_gap,
                               uint32_t Z_address_group_gap
#if VQ_ENABLED == 1
                               ,
                               const uint64_t CB_addresses[VQ_NUM_CBS], const uint64_t idx_addresses[VQ_NUM_CBS],
                               uint64_t scale_address, uint32_t cb_size, uint32_t idx_size, uint32_t scale_size
#endif
) {
    SummaGEMMInfo info;

    info.X_address       = X_address;
    info.W_address       = W_address;
    info.Z_address       = Z_address;
    info.M_size          = M_size;
    info.N_size          = N_size;
    info.K_size          = K_size;
    info.M_tile          = M_tile;
    info.N_tile          = N_tile;
    info.K_tile          = K_tile;
    info.group_reduction = group_reduction;
    info.group_splitK    = group_splitK;
    info.group_splitN    = group_splitN;
    info.summa_group_x   = group_x;
    info.summa_group_y   = group_y;
    info.summa_groups    = num_group;
    summa_config_assert(group_x > 0 && group_y > 0, 0xE001);
    summa_config_assert(num_group > 0, 0xE002);

    const uint32_t m_block = group_y * M_tile;
    const uint32_t n_block = group_x * N_tile;

    summa_config_assert((M_size % m_block) == 0, 0xE003);
    if (group_splitN) {
        summa_config_assert((N_size % num_group) == 0, 0xE004);
        summa_config_assert(((N_size / num_group) % n_block) == 0, 0xE005);
    } else {
        summa_config_assert((N_size % n_block) == 0, 0xE006);
    }

    if (group_splitK) {
        summa_config_assert((group_splitK * num_group) != 0, 0xE007);
        summa_config_assert((K_size % (num_group * group_splitK)) == 0, 0xE008);
        const uint32_t k_chunk = K_size / (num_group * group_splitK);
        summa_config_assert((k_chunk % K_tile) == 0, 0xE009);
    } else {
        summa_config_assert((K_size % K_tile) == 0, 0xE00A);
    }
#if VQ_ENABLED == 1
    summa_config_assert((N_tile % VQ_GROUP_SIZE) == 0, 0xE00B);
#endif
    const uint32_t n_size_per_group = group_splitN ? (N_size / num_group) : N_size;
#if VQ_ENABLED == 1
    summa_config_assert((n_size_per_group % VQ_GROUP_SIZE) == 0, 0xE00C);
#endif

    // Group infomation
    FlexPosition pos           = get_pos(flex_get_cluster_id());
    info.cluster_pos           = pos;
    info.cluster_global_id     = flex_get_cluster_id();
    info.group                 = grid_sync_group_init(info.summa_group_x, info.summa_group_y);
    info.cluster_in_group_id_x = pos.x % info.group.grid_x_dim;
    info.cluster_in_group_id_y = pos.y % info.group.grid_y_dim;
    info.cluster_in_group_id =
        info.cluster_in_group_id_x + info.group.this_grid_cluster_num_x * info.cluster_in_group_id_y;
    info.cluster_active = info.group.this_grid_id < info.summa_groups ? 1 : 0;
    info.cluster_for_rowwise =
        ((info.cluster_in_group_id_x % info.group.grid_y_dim) == (info.cluster_in_group_id_y % info.group.grid_x_dim) &&
         (info.cluster_in_group_id_x == (pos.y % info.group.grid_x_dim)))
            ? 1
            : 0;
    info.cluster_for_colwise =
        ((info.cluster_in_group_id_x % info.group.grid_y_dim) == (info.cluster_in_group_id_y % info.group.grid_x_dim) &&
         (info.cluster_in_group_id_y == (pos.x % info.group.grid_y_dim)))
            ? 1
            : 0;

    info.N_size_per_group = n_size_per_group;
    info.group_n_offset   = info.group_splitN ? (info.group.this_grid_id * n_size_per_group) : 0;

    info.M_iter          = (M_size + info.summa_group_y * M_tile - 1) / (info.summa_group_y * M_tile);
    info.N_iter          = (info.N_size_per_group + info.summa_group_x * N_tile - 1) / (info.summa_group_x * N_tile);
    info.K_iter          = info.group_splitK
                               ? (((K_size / (info.summa_groups * info.group_splitK)) + K_tile - 1) / K_tile)
                               : ((K_size + K_tile - 1) / K_tile); // if splitK, each group handles a portion of K else whole K
    info.store_recorded  = 0;
    info.store_m         = 0;
    info.store_n         = 0;
    info.store_step      = (info.summa_group_x + info.K_iter - 1) / info.K_iter;
    info.store_id        = info.summa_group_x;
    info.store_id_offset = pos.y % info.summa_group_x;
    info.store_active    = (info.group_reduction == 0) ? 1 : (info.group.this_grid_id == 0) ? 1 : 0;
    uint64_t n_group_byte_offset = (uint64_t)info.group_n_offset * DATA_TYPE_BYTE;
#if GEMM_RESHA_X_FROM_ENABLE == 1
    info.X_tile_base_offset = info.cluster_in_group_id_y * M_tile * K_size * DATA_TYPE_BYTE;
    info.X_tile_base        = X_address + X_address_group_gap * info.group.this_grid_id;
#else
    info.X_tile_base = X_address + X_address_group_gap * info.group.this_grid_id +
                       info.cluster_in_group_id_y * M_tile * K_size * DATA_TYPE_BYTE;
#endif
    info.W_tile_base = W_address + W_address_group_gap * info.group.this_grid_id + n_group_byte_offset +
                       info.cluster_in_group_id_x * N_tile * DATA_TYPE_BYTE;
#if GEMM_RESHA_Z_TO_ENABLE == 1
    info.Z_tile_base_offset = info.cluster_in_group_id_y * M_tile * N_size * DATA_TYPE_BYTE +
                              info.cluster_in_group_id_x * N_tile * DATA_TYPE_BYTE + n_group_byte_offset;
    info.Z_tile_base = Z_address + Z_address_group_gap * info.group.this_grid_id;
#else
    info.Z_tile_base = Z_address + Z_address_group_gap * info.group.this_grid_id + n_group_byte_offset +
                       info.cluster_in_group_id_y * M_tile * N_size * DATA_TYPE_BYTE +
                       info.cluster_in_group_id_x * N_tile * DATA_TYPE_BYTE;
#endif
    info.L1_X_size            = M_tile * K_tile * DATA_TYPE_BYTE;
    info.L1_W_size            = N_tile * K_tile * DATA_TYPE_BYTE;
    info.L1_Z_size            = M_tile * N_tile * DATA_TYPE_BYTE;
    info.X_tile_M_iter_offset = info.summa_group_x * M_tile * K_size * DATA_TYPE_BYTE;
    info.X_tile_N_iter_offset = 0;
    info.X_tile_K_iter_offset = K_tile * DATA_TYPE_BYTE;
    info.W_tile_M_iter_offset = 0;
    info.W_tile_N_iter_offset = info.summa_group_x * N_tile * DATA_TYPE_BYTE;
    info.W_tile_K_iter_offset = K_tile * N_size * DATA_TYPE_BYTE;
    info.Z_tile_M_iter_offset = info.summa_group_y * M_tile * N_size * DATA_TYPE_BYTE;
    info.Z_tile_N_iter_offset = info.summa_group_x * N_tile * DATA_TYPE_BYTE;
    uint32_t off              = local(0);
    info.L1_X1                = off;
    off += info.L1_X_size;
    info.L1_W1 = off;
    off += info.L1_W_size; //
    info.L1_Z1 = off;
    off += info.L1_Z_size; //
    info.L1_X2 = off;
    off += info.L1_X_size;
    info.L1_W2 = off;
    off += info.L1_W_size; //
    info.L1_Z2 = off;
    off += info.L1_Z_size; //

#if VQ_ENABLED == 1
    // Copy codebook addresses array
    for (int i = 0; i < VQ_NUM_CBS; i++) {
        info.vq.VQ_CB_address[i] = CB_addresses[i];
    }
    // Copy indices addresses array
    for (int i = 0; i < VQ_NUM_CBS; i++) {
        info.vq.VQ_Index_address[i] = idx_addresses[i];
    }
    info.vq.VQ_Scale_address            = scale_address;
    info.vq.cb_size                     = cb_size;
    info.vq.idx_size                    = idx_size;
    const uint32_t expected_scale_bytes = info.K_tile * VQ_CB_BYTES;
    // summa_config_assert(scale_size == expected_scale_bytes, 0xE00C);
    info.vq.scale_size = info.K_tile * VQ_CB_BYTES;

    // Pre-compute constants to avoid recomputation during loads
    info.vq.N_compressed              = info.N_size / VQ_GROUP_SIZE;
    info.vq.N_compressed_per_group    = info.N_size_per_group / VQ_GROUP_SIZE;
    info.vq.N_group_offset_compressed = info.group_splitN ? (info.group_n_offset / VQ_GROUP_SIZE) : 0;
    info.vq.N_tile_compressed         = info.N_tile / VQ_GROUP_SIZE;

    uint32_t single_cb_tile_bytes = info.vq.N_tile_compressed * info.K_tile * VQ_IDX_BYTES;
    info.vq.L1_IDX_size           = single_cb_tile_bytes; // Size per codebook (not total)
    info.vq.L1_CB_size            = VQ_CB_NUM_CENTROIDS * VQ_GROUP_SIZE * VQ_CB_BYTES;

    off += info.L1_W_size; //

    for (int i = 0; i < VQ_NUM_CBS; i++) {
        info.vq.L1_CB[i] = off;
        off += info.vq.L1_CB_size; //+W
    }

    // Allocate separate double buffers per codebook (2N buffers total)
    for (int i = 0; i < VQ_NUM_CBS; i++) {
        info.vq.L1_IDX1[i] = off;
        off += single_cb_tile_bytes;
    }
    for (int i = 0; i < VQ_NUM_CBS; i++) {
        info.vq.L1_IDX2[i] = off;
        off += single_cb_tile_bytes;
    }
#if VQ_USE_SCALES == 1
    uint32_t L1_scales_size = K_tile * VQ_CB_BYTES;
    info.vq.L1_Scales[0]    = off;
    off += L1_scales_size;
    info.vq.L1_Scales[1] = off;
    off += L1_scales_size;
#endif


#endif
    if (flex_get_cluster_id() == 1 && flex_is_dm_core()) {
        printf("\n M N K iter: %d %d %d\n", info.M_iter, info.N_iter, info.K_iter);
    }
    info.L1_AREA = off;
    return info;
}

#endif // GEMM_SETUP_H_
