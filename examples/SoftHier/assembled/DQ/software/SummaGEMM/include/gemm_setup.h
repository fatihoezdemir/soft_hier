#ifndef GEMM_SETUP_H_
#define GEMM_SETUP_H_

#include "flex_runtime.h"
#include "flex_group_barrier.h"
#include "gemm.h"
typedef struct SummaGEMMInfo
{
    //General information
    uint64_t                    X_address;
    uint64_t                    W_address;
    uint64_t                    Z_address;
    uint32_t                    M_size;
    uint32_t                    N_size;
    uint32_t                    K_size; /*shared dimension*/
    uint32_t                    M_tile;
    uint32_t                    N_tile;
    uint32_t                    K_tile;
    uint32_t                    group_reduction;
    uint32_t                    group_splitK;
    uint32_t                    group_splitN;
    uint32_t                    summa_group_x;
    uint32_t                    summa_group_y;
    uint32_t                    summa_groups;

    //Cluster information
    FlexPosition                cluster_pos;
    uint32_t                    cluster_global_id;
    uint32_t                    cluster_active;

    //Group Info
    GridSyncGroupInfo           group;
    uint32_t                    cluster_in_group_id;
    uint32_t                    cluster_in_group_id_x;
    uint32_t                    cluster_in_group_id_y;
    uint32_t                    cluster_for_rowwise;
    uint32_t                    cluster_for_colwise;

    //Tiling information
    uint32_t                    M_iter;
    uint32_t                    N_iter;
    uint32_t                    K_iter;

    //Store Control
    uint32_t                    store_recorded;
    uint32_t                    store_m;
    uint32_t                    store_n;
    uint32_t                    store_step;
    uint32_t                    store_id;
    uint32_t                    store_id_offset;
    uint32_t                    store_active;

    //Addressing Information
    uint64_t                    X_tile_base;
    uint64_t                    W_tile_base;
    uint64_t                    Z_tile_base;
#if GEMM_RESHA_X_FROM_ENABLE == 1
    uint64_t                    X_tile_base_offset;
#endif
#if GEMM_RESHA_Z_TO_ENABLE == 1
    uint64_t                    Z_tile_base_offset;
#endif
    uint64_t                    X_tile_M_iter_offset;
    uint64_t                    X_tile_N_iter_offset;
    uint64_t                    X_tile_K_iter_offset;
    uint64_t                    W_tile_M_iter_offset;
    uint64_t                    W_tile_N_iter_offset;
    uint64_t                    W_tile_K_iter_offset;
    uint64_t                    Z_tile_M_iter_offset;
    uint64_t                    Z_tile_N_iter_offset;

    //L1 location information
    uint32_t                    L1_X1;
    uint32_t                    L1_W1;
    uint32_t                    L1_Z1;
    uint32_t                    L1_X2;
    uint32_t                    L1_W2;
    uint32_t                    L1_Z2;
    #if VQ_ENABLED == 1
    
    #endif
    uint32_t                    L1_AREA;

    //Usefull parameters for address calculation
    uint32_t                    L1_X_size;
    uint32_t                    L1_W_size;
    uint32_t                    L1_Z_size;

}SummaGEMMInfo;

SummaGEMMInfo SummaGEMMAnaylze(
    uint64_t                    X_address,
    uint64_t                    W_address,
    uint64_t                    Z_address,
    uint32_t                    M_size,
    uint32_t                    N_size,
    uint32_t                    K_size, /*shared dimension*/
    uint32_t                    M_tile,
    uint32_t                    N_tile,
    uint32_t                    K_tile,
    uint32_t                    group_x,
    uint32_t                    group_y,
    uint32_t                    num_group,
    uint32_t                    group_reduction,
    uint32_t                    group_splitK,
    uint32_t                    group_splitN,
    uint32_t                    X_address_group_gap,
    uint32_t                    W_address_group_gap,
    uint32_t                    Z_address_group_gap)
{
    SummaGEMMInfo info;

    info.X_address              = X_address;
    info.W_address              = W_address;
    info.Z_address              = Z_address;
    info.M_size                 = M_size;
    info.N_size                 = N_size;
    info.K_size                 = K_size;
    info.M_tile                 = M_tile;
    info.N_tile                 = N_tile;
    info.K_tile                 = K_tile;
    info.group_reduction        = group_reduction;
    info.group_splitK           = group_splitK;
    info.group_splitN           = group_splitN;
    info.summa_group_x          = group_x;
    info.summa_group_y          = group_y;
    info.summa_groups           = num_group;


    //Group infomation
    FlexPosition pos            = get_pos(flex_get_cluster_id());
    info.cluster_pos            = pos;
    info.cluster_global_id      = flex_get_cluster_id();
    info.group                  = grid_sync_group_init(info.summa_group_x,info.summa_group_y);
    info.cluster_in_group_id_x  = pos.x % info.group.grid_x_dim;
    info.cluster_in_group_id_y  = pos.y % info.group.grid_y_dim;
    info.cluster_in_group_id    = info.cluster_in_group_id_x + info.group.this_grid_cluster_num_x * info.cluster_in_group_id_y;
    info.cluster_active         = info.group.this_grid_id < info.summa_groups ? 1 : 0;
    info.cluster_for_rowwise    = ((info.cluster_in_group_id_x % info.group.grid_y_dim) == (info.cluster_in_group_id_y % info.group.grid_x_dim) && (info.cluster_in_group_id_x == (pos.y % info.group.grid_x_dim)))? 1 : 0;
    info.cluster_for_colwise    = ((info.cluster_in_group_id_x % info.group.grid_y_dim) == (info.cluster_in_group_id_y % info.group.grid_x_dim) && (info.cluster_in_group_id_y == (pos.x % info.group.grid_y_dim)))? 1 : 0;

    info.M_iter                 = (M_size + info.summa_group_y * M_tile - 1) / (info.summa_group_y * M_tile);
    info.N_iter                 = info.group_splitN? (((N_size / info.summa_groups) + info.summa_group_x * N_tile - 1) / (info.summa_group_x * N_tile)) : (N_size + info.summa_group_x * N_tile - 1) / (info.summa_group_x * N_tile);
    info.K_iter                 = info.group_splitK? (((K_size / (info.summa_groups * info.group_splitK)) + K_tile - 1) / K_tile) : ((K_size + K_tile - 1) / K_tile);// if splitK, each group handles a portion of K else whole K
    info.store_recorded         = 0;
    info.store_m                = 0;
    info.store_n                = 0;
    info.store_step             = (info.summa_group_x + info.K_iter - 1) / info.K_iter;
    info.store_id               = info.summa_group_x;
    info.store_id_offset        = pos.y % info.summa_group_x;
    info.store_active           = (info.group_reduction == 0) ? 1 : (info.group.this_grid_id == 0)? 1 : 0;
#if GEMM_RESHA_X_FROM_ENABLE == 1
    info.X_tile_base_offset     = info.cluster_in_group_id_y * M_tile * K_size * DATA_TYPE_BYTE;
    info.X_tile_base            = X_address + X_address_group_gap * info.group.this_grid_id;
#else
    info.X_tile_base            = X_address + X_address_group_gap * info.group.this_grid_id + info.cluster_in_group_id_y * M_tile * K_size * DATA_TYPE_BYTE; 
#endif
    info.W_tile_base            = W_address + W_address_group_gap * info.group.this_grid_id + info.cluster_in_group_id_x * N_tile          * DATA_TYPE_BYTE;
#if GEMM_RESHA_Z_TO_ENABLE == 1
    info.Z_tile_base_offset     = info.cluster_in_group_id_y * M_tile * N_size * DATA_TYPE_BYTE + info.cluster_in_group_id_x * N_tile * DATA_TYPE_BYTE;
    info.Z_tile_base            = Z_address + Z_address_group_gap * info.group.this_grid_id;
#else
    info.Z_tile_base            = Z_address + Z_address_group_gap * info.group.this_grid_id + info.cluster_in_group_id_y * M_tile * N_size * DATA_TYPE_BYTE + info.cluster_in_group_id_x * N_tile * DATA_TYPE_BYTE;
#endif
    info.X_tile_M_iter_offset   = info.summa_group_x * M_tile * K_size * DATA_TYPE_BYTE;
    info.X_tile_N_iter_offset   = 0;
    info.X_tile_K_iter_offset   = K_tile * DATA_TYPE_BYTE;
    info.W_tile_M_iter_offset   = 0;
    info.W_tile_N_iter_offset   = info.summa_group_x * N_tile * DATA_TYPE_BYTE;
    info.W_tile_K_iter_offset   = K_tile * N_size * DATA_TYPE_BYTE;
    info.Z_tile_M_iter_offset   = info.summa_group_y * M_tile * N_size * DATA_TYPE_BYTE;
    info.Z_tile_N_iter_offset   = info.summa_group_x * N_tile * DATA_TYPE_BYTE;
    info.L1_X1                  = local(0);
    info.L1_W1                  = info.L1_X1 + M_tile * K_tile * DATA_TYPE_BYTE;
    info.L1_Z1                  = info.L1_W1 + N_tile * K_tile * DATA_TYPE_BYTE;
    info.L1_X2                  = info.L1_Z1 + M_tile * N_tile * DATA_TYPE_BYTE;
    info.L1_W2                  = info.L1_X2 + M_tile * K_tile * DATA_TYPE_BYTE;
    info.L1_Z2                  = info.L1_W2 + N_tile * K_tile * DATA_TYPE_BYTE;
    info.L1_AREA                = info.L1_Z2 + M_tile * N_tile * DATA_TYPE_BYTE;
    info.L1_X_size              = M_tile * K_tile * DATA_TYPE_BYTE;
    info.L1_W_size              = N_tile * K_tile * DATA_TYPE_BYTE;
    info.L1_Z_size              = M_tile * N_tile * DATA_TYPE_BYTE;

    return info;
}

#endif // GEMM_SETUP_H_