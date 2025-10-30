/*********
 * Benchmarking GEMV and GEMM kernels
 * TODO deprecated,change
 * Each benchmark in sequential steps (but the processes eg dma and compute or dma and dequantize happen at same time)
 * db:  1)load scales,codebooks and left half of indices
 *      2)dequantize partially wuth indices from step 1) and also load upper half of activation matrix
 *      3)compute partial Matrix resultand load other half of indices
 *          (upper half of activation x left half of indices would give us upper left quarter of the full resutl)
 *      4)load other half of activation matrix and dequantize other half of weight matrix
 *      5) compute  the other 3 tiles of C= A*B
 * Reminder: dequantization in AQLM is like this :
 *  in case of dequantizing a row :W_hat[row]= scale[row]*( concat(cb1[idx1[i]]+cb2[idx2[i]] for all groups i in a
 * certain row
 * ))
 *
 */
/* activation matrix A= [ A0
                          A1
                          A2
                          ... ]   horizontal splits
Weight Matrix B = [B0 B1 B2 ...] vertical splits
*/
#include "include/dq_data_hbm.h"

#include <stdint.h>
#define DEBUG 1
#define REDMULE_ON 1

#include "flex_alloc.h"
#include "flex_dma_pattern.h"
#include "flex_dump.h"
#include "flex_libfp16.h"
#include "flex_libfp8.h"
#include "flex_printf.h"
#include "flex_redmule.h"
#include "flex_runtime.h"
#include "include/debug.h"
#include "include/dq_compute_spatz.h" // spatz computation kernels
#include "include/dq_gemm_double_buffer_baseline.h"
#include "include/dq_gemv_double_buffer_baseline.h"
#include "include/dq_helpers.h"
#include "include/dq_load_data_l1.h"

#include <stdio.h>
const int SPATZ_CORE   = 0;
const int DOUBLEBUFFER = 1;
// Tiling configuration TODO make a tilinginfo struct
const int NUM_TILES = 2;

int main() {

    uint32_t eoc_val = 0;
    flex_barrier_xy_init();
    flex_global_barrier_xy();
    flex_alloc_init();
    flex_intra_cluster_sync(); // Cluster barrier
    flex_global_barrier_xy();

    uint32_t total_l1_required =
        (VQ_CB_NUM_CENTROIDS * VQ_GROUP_SIZE * VQ_NUM_CBS + FP16_M) * sizeof(uint16_t) + // codebook + scales
        2 * FP16_M * GROUPS_PER_TILE * sizeof(uint16_t) +                                // double idx buffers
        2 * ROWS_PER_TILE * FP16_N * sizeof(uint16_t) +                                  // double A buffers
        FP16_M * GROUPS_PER_TILE * VQ_GROUP_SIZE * sizeof(uint16_t) +                    // W buffer
        ROWS_PER_TILE * GROUPS_PER_TILE * VQ_GROUP_SIZE * sizeof(uint16_t);              // C buffer

    if (total_l1_required > ARCH_CLUSTER_TCDM_SIZE) {
        if (flex_get_core_id() == 0 && flex_get_cluster_id() == 0) {
            printf("ERROR: L1 overflow! Need %uKB, have %uKB\n", total_l1_required >> 10, ARCH_CLUSTER_TCDM_SIZE >> 10);
        }
        flex_eoc(eoc_val);
    }
    /**************************************/
    /*  Program Execution Region -- Start */
    /**************************************/

    // Choose which version to run based on Kernel macro
#if GEMM == 1
    // [INFO] Running double-buffered GEMM with pipelined execution
    // dq_gemm_double_buffer_baseline();
#elif GEMV == 1
    //[INFO] Running double-buffered GEMV
    dq_gemv_double_buffer_baseline();

#endif
    /**************************************/
    /*  Program Execution Region -- Stop  */
    /**************************************/
    flex_global_barrier_xy();
    if (flex_get_core_id() == 0 && flex_get_cluster_id() == 0) {
        printf("\nfinished!");
    }
    flex_global_barrier_xy();
    flex_eoc(eoc_val);
    return 0;
}