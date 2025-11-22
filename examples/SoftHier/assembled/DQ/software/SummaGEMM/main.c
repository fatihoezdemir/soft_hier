// Copyright 2025 ETH Zurich and
// University of Bologna

// Solderpad Hardware License
// Version 0.51, see LICENSE for details.

// SPDX-License-Identifier: SHL-0.51

// Author: Chi Zhang <chizhang@iis.ee.ethz.ch>, ETH Zurich
// Date: 1.Oct.2025

#include "SummaGEMMVQ.h"
#include "flex_dump.h"
#include "flex_runtime.h"
#include "gemm.h"
#include "gemm_addresses.h"

// Include data arrays for linker script workflow (if generated)
#if __has_include("gemm_data.h")
#include "gemm_data.h"
#endif

#include "include/SummaGEMM.h"
int main() {
    uint32_t eoc_val = 0;
    flex_barrier_xy_init();
    // flex_alloc_init();
    flex_global_barrier_xy();
    /**************************************/
    /*  Program Execution Region -- Start */
    /**************************************/

    SummaGEMMInfo info = SummaGEMMAnaylze(
        X_ADDR /*X_address*/, W_ADDR /*W_address*/, Z_EADDR /*Z_address*/, GEMM_M_SIZE /*M_size*/,
        GEMM_N_SIZE /*N_size*/, GEMM_K_SIZE /*K_size*/, /*shared dimension*/
        GEMM_M_TILE /*M_tile*/, GEMM_N_TILE /*N_tile*/, GEMM_K_TILE /*K_tile*/, GEMM_SUMMA_SCALE_X /*group_x*/,
        GEMM_SUMMA_SCALE_Y /*group_y*/, GEMM_SUMMA_GROUP_NUMBER /*num_group*/,
        GEMM_SUMMA_GROUP_REDUCE /*group_reduction*/, GEMM_SUMMA_GROUP_SPLITK /*group_splitK*/,
        GEMM_SUMMA_GROUP_SPLITN /*group_splitN*/, GEMM_SUMMA_GROUP_GAP_X /*X_address_group_gap*/,
        GEMM_SUMMA_GROUP_GAP_W /*W_address_group_gap*/, GEMM_SUMMA_GROUP_GAP_Z /*Z_address_group_gap*/
#if VQ_ENABLED == 1
        ,
        (uint64_t[VQ_NUM_CBS])VQ_CODEBOOKS_ADDRS, (uint64_t[VQ_NUM_CBS])VQ_INDICES_ADDRS, VQ_SCALES_ADDR,
        VQ_CB_NUM_CENTROIDS,
        VQ_NUM_SCALES, // correct it , idx size
        VQ_NUM_SCALES  // correct it , scale size
#endif
    );
    flex_global_barrier_xy();

    // execute SUMMA GEMM
    if (flex_get_core_id() == 0 && flex_get_cluster_id() == 0)
        flex_timer_start();
    flex_global_barrier_xy();
    if (flex_get_cluster_id() == 0 && flex_is_dm_core()) {
        printf(" hbm west: %lx north: %lx east: %lx south: %lx\n", hbm_west(0, 0), hbm_north(0, 0), hbm_east(0, 0),
               hbm_south(0, 0));
        printf(" hbm west: %lx north: %lx east: %lx south: %lx\n", hbm_west(0, 0), hbm_north(0, 0), hbm_east(0, 0),
               hbm_south(0, 0));
        printf(" hbm west: %lx north: %lx east: %lx south: %lx\n", hbm_west(0, 0), hbm_north(0, 0), hbm_east(0, 0),
               hbm_south(0, 0));
        printf(" hbm west: %lx north: %lx east: %lx south: %lx\n", hbm_west(0, 0), hbm_north(0, 0), hbm_east(0, 0),
               hbm_south(0, 0));
        printf(" hbm west: %lx north: %lx east: %lx south: %lx\n", hbm_west(0, 0), hbm_north(0, 0), hbm_east(0, 0),
               hbm_south(0, 0));
    }
#if VQ_ENABLED == 1
    SummaGEMMRun(&info);
#else
    SummaGEMMRunVQ(&info);
#endif
    if (flex_get_core_id() == 0 && flex_get_cluster_id() == 0)
        flex_timer_end();
    flex_global_barrier_xy();

    // dump results
    if (GEMM_SUMMA_NUMER) {
        // postload part of O
        if (flex_get_cluster_id() == 0 && flex_is_dm_core()) {
            // 1. Calculate the whole size of the output
            uint64_t output_size      = GEMM_M_SIZE * GEMM_N_SIZE * DATA_TYPE_BYTE;
            uint64_t num_output_chunk = (output_size + GEMM_SUMMA_NUMER_CHUNK - 1) / GEMM_SUMMA_NUMER_CHUNK;

            // 2. iterate over chunks
            flex_dump_open();
            for (uint64_t i = 0; i < num_output_chunk; ++i) {
                flex_dump_hbm(i * GEMM_SUMMA_NUMER_CHUNK + Z_EADDR - ARCH_HBM_START_BASE, GEMM_SUMMA_NUMER_CHUNK);
                flex_dump_hbm(i * GEMM_SUMMA_NUMER_CHUNK + Z_GADDR - ARCH_HBM_START_BASE, GEMM_SUMMA_NUMER_CHUNK);
            }
            flex_dump_close();
        }
    }
    flex_global_barrier_xy();
    if (flex_get_cluster_id() == 0 && flex_is_dm_core()) {
        printf(" \n\n FINISHED\n\n");
    }
    /**************************************/
    /*  Program Execution Region -- Stop  */
    /**************************************/
    flex_global_barrier_xy();
    flex_eoc(eoc_val);
    return 0;
}