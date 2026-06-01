// Copyright 2025 ETH Zurich and
// University of Bologna

// Solderpad Hardware License
// Version 0.51, see LICENSE for details.

// SPDX-License-Identifier: SHL-0.51

// Author: Chi Zhang <chizhang@iis.ee.ethz.ch>, ETH Zurich
// Date: 1.Oct.2025

#include "flex_dump.h"
#include "flex_runtime.h"
#include "gemm.h"
#include "gemm_addresses.h"

// Include data arrays for linker script workflow (if generated)
#if __has_include("gemm_data.h")
#include "gemm_data.h"
#endif

// Single dequant-kernel selector for GEMM + GEMV paths.
// Change this one macro to switch both:
//   summa_vq_dequantize_tile
//   summa_vq_dequantize_tile_baseline
//   summa_vq_dequantize_tile_arith
#ifndef SUMMA_VQ_DEQ_KERNEL
#define SUMMA_VQ_DEQ_KERNEL summa_vq_dequantize_tile_arith
#endif

#include "include/SummaGEMM.h"
#include "include/SummaGEMV.h"

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
        /* idx_size */ (GEMM_N_TILE / VQ_GROUP_SIZE) * GEMM_K_TILE * VQ_IDX_BYTES,
        /* scale_size */
        (
#if defined(VQ_USE_SCALES) && (VQ_USE_SCALES == 1)
            GEMM_K_TILE * VQ_CB_BYTES
#else
            0
#endif
            )
#endif
    );
    flex_global_barrier_xy();

    // execute SUMMA GEMM
    flex_global_barrier_xy();
    if (flex_get_cluster_id() == 0 && flex_is_dm_core()) {
        printf("------------------------------------------------------------\n");
        printf("                    L1 MEMORY LAYOUT INFO                   \n");
        printf("------------------------------------------------------------\n");
        printf("  X1:  0x%05lx   Size: %-5lu bytes\n", info.L1_X1, info.L1_X_size);
        printf("  W1:  0x%05lx   Size: %-5lu bytes\n", info.L1_W1, info.L1_W_size);
        printf("  Z1:  0x%05lx   Size: %-5lu bytes\n", info.L1_Z1, info.L1_Z_size);
        printf("------------------------------------------------------------\n");
        printf("  X2:  0x%05lx   Size: %-5lu bytes\n", info.L1_X2, info.L1_X_size);
        printf("  W2:  0x%05lx   Size: %-5lu bytes\n", info.L1_W2, info.L1_W_size);
        printf("  Z2:  0x%05lx   Size: %-5lu bytes\n", info.L1_Z2, info.L1_Z_size);
        printf("------------------------------------------------------------\n");
#if VQ_ENABLED == 1
        printf("  VQ BUFFERS (NUM_CBS=%d):\n", VQ_NUM_CBS);
        printf("------------------------------------------------------------\n");
        for (int i = 0; i < VQ_NUM_CBS; i++) {
            uint32_t cb_hbm_hi = (uint32_t)(info.vq.VQ_CB_address[i] >> 32);
            uint32_t cb_hbm_lo = (uint32_t)(info.vq.VQ_CB_address[i] & 0xffffffffu);
            printf("  Codebook[%d]:   L1: 0x%05lx   Size: %-5lu bytes\n", i, info.vq.L1_CB[i], info.vq.L1_CB_size);
            printf("                 HBM: 0x%08lx%08lx\n", (unsigned long)cb_hbm_hi, (unsigned long)cb_hbm_lo);
        }
        printf("------------------------------------------------------------\n");
        for (int i = 0; i < VQ_NUM_CBS; i++) {
            uint32_t idx_hbm_hi = (uint32_t)(info.vq.VQ_Index_address[i] >> 32);
            uint32_t idx_hbm_lo = (uint32_t)(info.vq.VQ_Index_address[i] & 0xffffffffu);
            printf("  Indices[%d] (double-buffered):\n", i);
            printf("    IDX1[%d]:     L1: 0x%05lx   Size: %-5lu bytes\n", i, info.vq.L1_IDX1[i], info.vq.L1_IDX_size);
            printf("    IDX2[%d]:     L1: 0x%05lx   Size: %-5lu bytes\n", i, info.vq.L1_IDX2[i], info.vq.L1_IDX_size);
            printf("                 HBM: 0x%08lx%08lx\n", (unsigned long)idx_hbm_hi, (unsigned long)idx_hbm_lo);
        }
        printf("------------------------------------------------------------\n");
#if VQ_USE_SCALES == 1
        uint32_t L1_scales_size = info.K_tile * VQ_CB_BYTES;
        uint32_t scale_hbm_hi = (uint32_t)(info.vq.VQ_Scale_address >> 32);
        uint32_t scale_hbm_lo = (uint32_t)(info.vq.VQ_Scale_address & 0xffffffffu);
        printf("  Scales (double-buffered):\n");
        printf("    Scale[0]:    L1: 0x%05lx   Size: %-5lu bytes\n", info.vq.L1_Scales[0], L1_scales_size);
        printf("    Scale[1]:    L1: 0x%05lx   Size: %-5lu bytes\n", info.vq.L1_Scales[1], L1_scales_size);
        printf("                 HBM: 0x%08lx%08lx\n", (unsigned long)scale_hbm_hi, (unsigned long)scale_hbm_lo);
        printf("    Scale size:   %lu bytes\n", info.vq.scale_size);
        printf("------------------------------------------------------------\n");
#endif
        printf("  VQ Configuration:\n");
        printf("    CB size:      %lu bytes (%d centroids x %d values x %d bytes)\n", info.vq.L1_CB_size,
               VQ_CB_NUM_CENTROIDS, VQ_GROUP_SIZE, VQ_CB_BYTES);
#if VQ_COMPRESS_K == 1
        printf("    IDX size:     %lu bytes (%lu x %lu tile)\n", info.vq.L1_IDX_size, info.N_tile,
               info.vq.K_tile_compressed);
#else
        printf("    IDX size:     %lu bytes (%lu x %lu tile)\n", info.vq.L1_IDX_size, info.K_tile,
               info.vq.N_tile_compressed);
#endif
        printf("    Group size:   %d\n", VQ_GROUP_SIZE);
        printf("    N_compressed: %lu (N_tile: %lu)\n", info.vq.N_compressed, info.vq.N_tile_compressed);
        printf("    K_compressed: %lu (K_tile: %lu)\n", info.vq.K_compressed, info.vq.K_tile_compressed);
        printf("------------------------------------------------------------\n");
#endif
        printf("  Total L1 Area  : 0x%08lx (%lu bytes)\n", info.L1_AREA, info.L1_AREA);
        printf("  L1 Usage Ratio : %.2f %% of ARCH_CLUSTER_TCDM_SIZE\n", 100.0 * info.L1_AREA / ARCH_CLUSTER_TCDM_SIZE);
        printf("------------------------------------------------------------\n");
        printf("  HBM Connections:\n");
        printf("      West : 0x%08lx\n", hbm_west(0, 0));
        printf("      North: 0x%08lx\n", hbm_north(0, 0));
        printf("      East : 0x%08lx\n", hbm_east(0, 0));
        printf("      South: 0x%08lx\n", hbm_south(0, 0));
        printf("------------------------------------------------------------\n");
    }


    if (flex_is_dm_core() && flex_get_cluster_id() == 0)
        flex_timer_start();
#ifdef COMPUTE_KERNEL_GEMM
    SummaGEMMRun(&info);
#endif
#if defined(COMPUTE_KERNEL_GEMV)
    SummaGEMVRun(&info);
#endif
    if (flex_is_dm_core() &&flex_get_cluster_id() == 0)
        flex_timer_end();
    flex_global_barrier_xy();
    // if (flex_get_cluster_id() == 0 && flex_is_dm_core()) {
    //     printf(" \n\n FINISHED,now dumping\n\n");
    // }
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
