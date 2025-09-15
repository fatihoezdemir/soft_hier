/**********************************************************
 * Benchmarking GEMV and GEMM kernels
 * Each kernel has 2 benchmarks:
 *  1) Singlebuffer sb: compute after everything is loaded to L1
 *  2) Doublebuffer db: overlap compute with DMA Transfers, load partial Matrices
 *
 *
 * Each benchmark in sequential steps (but the processes eg dma and compute or dma and dequantize happen at same time)
 * sb : 1) allocate everything , load codebook, indices and scale
 *      2) load synthetic activation matrix and dequantize the weight matrix
 *      3) Compute(once with spatz and once with redmule)
 *
 * db:  1)load scales,codebooks and left half of indices
 *      2)dequantize partially wuth indices from step 1) and also load upper half of activation matrix
 *      3)compute partial Matrix resultand load other half of indices
 *          (upper half of activation x left half of indices would give us upper left quarter of the full resutl)
 *      4)load other half of activation matrix and dequantize other half of weight matrix
 *      5) compute  the other 3 tiles of C= A*B
 * Reminder: dequantization in AQLM is like this :
 *  in case of dequantizing a row :W_hat[row]= scale[row]*( concat(cb1[idx1[i]]+cb2[idx2[i]] for all i in groups per row
 * ))
 *
 */

#define DEBUG 0
#include "dq_helpers.h"
#include "flex_alloc.h"
#include "flex_dma_pattern.h"
#include "flex_dump.h"
#include "flex_libfp16.h"
#include "flex_libfp8.h"
#include "flex_printf.h"
#include "flex_runtime.h"

#define SPATZ_CORE 0
#define DOUBLEBUFFER 0
// inlining gains like 40 cycle performacmce out of 5 mio
static inline void spatz_matmul_fp16_full_legacy(uint16_t* matrix_a, uint16_t* matrix_b, uint16_t* matrix_c,
                                                 const uint32_t M, const uint32_t N, const uint32_t P) {
    // init counters
    uint32_t p   = 0;
    uint32_t avl = P;
    uint32_t vl;
    do {
        asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(avl));
        for (uint32_t m = 0; m < M; m++) {
            uint16_t* p_c = matrix_c + m * P + p;
            for (uint32_t n = 0; n < N; n++) {
                // Load scalar a using integer load + broadcast (same as dequantization fix)
                uint16_t* p_a = &matrix_a[m * N + n];
                uint16_t* p_b = &matrix_b[p + n * P];

                asm volatile("vle16.v v0, (%0)" ::"r"(p_b)); // load b vector

                if (n == 0) {                               // first iteration - initialize accumulator
                    asm volatile("lhu t0, (%0)\n\t"         // load scalar a as integer
                                 "vmv.v.x v8, t0\n\t"       // broadcast to vector (v8 is legal start for m8)
                                 "vfmul.vv v16, v0, v8\n\t" // multiply: v16 = b * a
                                 ::"r"(p_a)
                                 : "t0", "v0", "v8", "v16", "memory");
                } else {                                     // accumulate: v16 = v16 + (b * a)
                    asm volatile("lhu t0, (%0)\n\t"          // load scalar a as integer
                                 "vmv.v.x v8, t0\n\t"        // broadcast to vector (v8 is legal start for m8)
                                 "vfmacc.vv v16, v0, v8\n\t" // fused multiply-accumulate
                                 ::"r"(p_a)
                                 : "t0", "v0", "v8", "v16", "memory");
                }
            }
            asm volatile("vse16.v v16, (%0)" ::"r"(p_c)); // store result
        }
        avl -= vl;
        p += vl;
    } while (avl > 0);
}

void load_data_to_l1() {
    if (flex_is_dm_core() && (flex_get_cluster_id() == 0)) {

        L1_PTRS.cb = (uint32_t)(uintptr_t)flex_l1_malloc(
            VQ_CB_NUM_CENTROIDS * VQ_GROUP_SIZE * VQ_NUM_CBS *
            sizeof(uint16_t)); // codebook size, [256,8,2],2 codebooks,256 centroids of size 8(8 is group size, 1
                               // decoding decodes entry to 8 values)
        L1_PTRS.indices = (uint32_t)(uintptr_t)flex_l1_malloc(
            VQ_TOTAL_GROUPS *
            sizeof(uint16_t)); // indices codebook, actual shape was [128,16,2],2 is number of codebooks, require 2
                               // lookups , 16 groups of size 8, indices are actually uint8 but we squeezed last
                               // dimenstion to make [128,16] and jhav e uint16
        L1_PTRS.scales = (uint32_t)(uintptr_t)flex_l1_malloc(
            FP16_M * sizeof(uint16_t)); // scales size (128) is the numnerb of rows of W

        L1_PTRS.W_dq = (uint32_t)(uintptr_t)flex_l1_malloc(
            FP16_M * FP16_N * sizeof(uint16_t)); // dequantized weight matrix allocatio W_dq
        L1_PTRS.activation = (uint32_t)(uintptr_t)flex_l1_malloc(
            FP16_M * FP16_N * sizeof(uint16_t)); // activation matrix to later do act*W_dq
        L1_PTRS.result = (uint32_t)(uintptr_t)flex_l1_malloc(FP16_M * FP16_N * sizeof(uint16_t)); // to store act*W_dq

        flex_global_barrier_xy(); // not needed
        printf("\[DEBUG][DMA][HBM->L1] 1) load codebook, scales ,codebooks etc\n");
        flex_timer_start();
        flex_dma_async_1d((uint64_t)(uintptr_t)L1_PTRS.cb, (uint64_t)(uintptr_t)&matrix_cb_fp16[0],
                          VQ_CB_NUM_CENTROIDS * VQ_NUM_CBS * VQ_GROUP_SIZE * sizeof(uint16_t));
        flex_dma_async_1d((uint64_t)(uintptr_t)L1_PTRS.scales, (uint64_t)(uintptr_t)&matrix_scales_fp16[0],
                          FP16_M * sizeof(uint16_t));
#if DOUBLEBUFFER == 0
        flex_dma_async_1d((uint64_t)(uintptr_t)L1_PTRS.indices, (uint64_t)(uintptr_t)&matrix_idx_packed_uint16[0],
                          FP16_M * VQ_NUM_GROUPS_PER_ROW * sizeof(uint16_t));

#elif DOUBLEBUFFER == 1
        flex_dma_async_2d((uint64_t)(uintptr_t)L1_PTRS.indices,              // destination
                          (uint64_t)(uintptr_t)&matrix_idx_packed_uint16[0], // source - left half
                          VQ_NUM_GROUPS_PER_ROW / 2 * sizeof(uint16_t),      // transfer 8 groups per row
                          VQ_NUM_GROUPS_PER_ROW * sizeof(uint16_t),          // destination stride (full row)
                          VQ_NUM_GROUPS_PER_ROW * sizeof(uint16_t),          // source stride (full row)
                          FP16_M                                             // 128 rows
        );
#endif
        flex_dma_async_wait_all();
        flex_timer_end();
    }
}

void load_data_to_l1_new() {
    if (flex_is_dm_core() && (flex_get_cluster_id() == 0)) {

        L1_PTRS.cb = (uint32_t)(uintptr_t)flex_l1_malloc(
            VQ_CB_NUM_CENTROIDS * VQ_GROUP_SIZE * VQ_NUM_CBS *
            sizeof(uint16_t)); // codebook size, [256,8,2],2 codebooks,256 centroids of size 8(8 is group size, 1
                               // decoding decodes entry to 8 values)
        L1_PTRS.indices = (uint32_t)(uintptr_t)flex_l1_malloc(
            VQ_TOTAL_GROUPS *
            sizeof(uint16_t)); // indices codebook, actual shape was [128,16,2],2 is number of codebooks, require 2
                               // lookups , 16 groups of size 8, indices are actually uint8 but we squeezed last
                               // dimenstion to make [128,16] and jhav e uint16
        L1_PTRS.scales = (uint32_t)(uintptr_t)flex_l1_malloc(
            FP16_M * sizeof(uint16_t)); // scales size (128) is the numnerb of rows of W

        L1_PTRS.W_dq = (uint32_t)(uintptr_t)flex_l1_malloc(
            FP16_M * FP16_N * sizeof(uint16_t)); // dequantized weight matrix allocatio W_dq
        L1_PTRS.activation = (uint32_t)(uintptr_t)flex_l1_malloc(
            FP16_M * FP16_N * sizeof(uint16_t)); // activation matrix to later do act*W_dq
        L1_PTRS.result = (uint32_t)(uintptr_t)flex_l1_malloc(FP16_M * FP16_N * sizeof(uint16_t)); // to store act*W_dq

        flex_global_barrier_xy(); // not needed
        printf("\[DEBUG][DMA][HBM->L1] 1) load codebook, scales ,codebooks etc\n");
        flex_timer_start();
        flex_dma_async_1d((uint64_t)(uintptr_t)L1_PTRS.cb, (uint64_t)(uintptr_t)&matrix_cb_fp16[0],
                          VQ_CB_NUM_CENTROIDS * VQ_NUM_CBS * VQ_GROUP_SIZE * sizeof(uint16_t));
        flex_dma_async_1d((uint64_t)(uintptr_t)L1_PTRS.scales, (uint64_t)(uintptr_t)&matrix_scales_fp16[0],
                          FP16_M * sizeof(uint16_t));
#if DOUBLEBUFFER == 0
        flex_dma_async_1d((uint64_t)(uintptr_t)L1_PTRS.indices, (uint64_t)(uintptr_t)&matrix_idx_packed_uint16[0],
                          FP16_M * VQ_NUM_GROUPS_PER_ROW * sizeof(uint16_t));

#elif DOUBLEBUFFER == 1
        flex_dma_async_2d((uint64_t)(uintptr_t)L1_PTRS.indices,              // destination
                          (uint64_t)(uintptr_t)&matrix_idx_packed_uint16[0], // source - left half
                          VQ_NUM_GROUPS_PER_ROW / 2 * sizeof(uint16_t),      // transfer 8 groups per row
                          VQ_NUM_GROUPS_PER_ROW * sizeof(uint16_t),          // destination stride (full row)
                          VQ_NUM_GROUPS_PER_ROW * sizeof(uint16_t),          // source stride (full row)
                          FP16_M                                             // 128 rows
        );
#endif
        flex_dma_async_wait_all();
        flex_timer_end();
    }
}
void load_indices_second_half() { // this is for the double buffer case only, we load right halfof indices an activation
                                  // matrix
    if (flex_is_dm_core() && flex_get_cluster_id() == 0) {
        printf("\[DEBUG][DMA][HBM->L1] 3) load right half of the indices\n");
        flex_timer_start();
        flex_dma_async_2d(
            (uint64_t)(uintptr_t)L1_PTRS.indices + (VQ_NUM_GROUPS_PER_ROW / 2) * sizeof(uint16_t), // dest offset
            (uint64_t)(uintptr_t)&matrix_idx_packed_uint16[8], // source offset (right half)
            VQ_NUM_GROUPS_PER_ROW / 2 * sizeof(uint16_t),      // transfer 8 elements per row
            VQ_NUM_GROUPS_PER_ROW * sizeof(uint16_t),          // destination stride
            VQ_NUM_GROUPS_PER_ROW * sizeof(uint16_t),          // source stride (full row)
            FP16_M                                             // 128 rows=-__--_+
        );      
        flex_dma_async_wait_all();
        flex_timer_end();
        printf("[DEBUG] Second half of indices loaded\n");
    }
}
void  single_buffer_gemm (){
    if ( flex_is_dm_core() && flex_get_core_id() ==0){
        load_data_to_l1_new();
    }    
        
        

}
    int main() {
    uint32_t eoc_val = 0;
    flex_barrier_xy_init();
    flex_global_barrier_xy();
    flex_alloc_init();
    flex_intra_cluster_sync(); // Cluster barrier
    flex_global_barrier_xy();

    /**************************************/
    /*  Program Execution Region -- Start */
    /**************************************/
    // step 1 Load codebook,scales and indices (only right half indices in case of Double buffering)
    // load_data_to_l1();
    // single_buffer_gemm();    
    flex_global_barrier_xy(); // why this doe/snt work without

    flex_intra_cluster_sync();

    // Step 2 Load Activation matrix and  dequantize at same time( DB: only upper half load )
    if (flex_is_dm_core() && (flex_get_cluster_id() == 0)) {
        printf("\[DEBUG][DMA][HBM->L1] 2) load activation matrix and dequantize \n");
        flex_timer_start();
#if DOUBLEBUFFER == 1
        flex_dma_async_1d((uint64_t)(uintptr_t)L1_PTRS.activation, (uint64_t)(uintptr_t)&matrix_activation_fp16[0],
                          FP16_M * FP16_N / 2 * sizeof(uint16_t));
#elif DOUBLEBUFFER == 0
        flex_dma_async_1d((uint64_t)(uintptr_t)L1_PTRS.activation, (uint64_t)(uintptr_t)&matrix_activation_fp16[0],
                          FP16_M * FP16_N * sizeof(uint16_t));
#endif

        flex_dma_async_wait_all();
        flex_timer_end();
    }

// #if DOUBLEBUFFER == 1
//     dequantize(0, VQ_TOTAL_GROUPS / 2); // dequantize left half of weight matrix
// #elif DOUBLEBUFFER == 0
//     dequantize_full_matrix();
// #endif

//     flex_intra_cluster_sync();
//     flex_global_barrier_xy(); // why this doe/snt work without

// // step 3 : compute (and for db, load second half of indices at the same time)
// #if DOUBLEBUFFER == 1
//     load_indices_second_half();
// #endif
//     if (flex_get_cluster_id() == 0 && flex_get_core_id() == SPATZ_CORE) {
//         printf("\n[DEBUG][COMPUTE] 3) first result now "); // we compute MATMUL of ACT_UPPERHALF * W_DQ_LEFTHALF

//         // Debug: print first few activation and weight values before matrix multiplication
//         CDEBUG_PRINT_U16(((uint16_t*)L1_PTRS.activation), 10, "activation values");

//         flex_timer_start();
//         spatz_matmul_fp16_full_legacy((uint16_t*)L1_PTRS.activation, (uint16_t*)L1_PTRS.W_dq, (uint16_t*)L1_PTRS.result,
// #if DOUBLEBUFFER == 1
//                                       FP16_M / 2, FP16_N, FP16_K / 2
// #elif DOUBLEBUFFER == 0
//                                       FP16_M, FP16_N, FP16_K
// #endif
//         );
//         flex_timer_end();

//         // Debug: print first few results to verify matrix multiplication worked
//         CDEBUG_PRINT_U16(((uint16_t*)L1_PTRS.result), 10, "result values");

// // Verify matrix multiplication results against golden reference
// #if DOUBLEBUFFER == 0
//         printf("[DEBUG] Verifying GEMM results against golden reference\n");
//         spatz_verify_16(FP16_M * FP16_N, (uint16_t*)L1_PTRS.result, (uint16_t*)matrix_golden_fp16, 0.25f);
// #endif
//     }

// // compute once with spatz and once with redmule to benchmark
// //  //single buffered benchmark finished
// //  //double buffered benchmark needs further go
// //  //step 4  DB: load right half of activation matrix and dequantize at the same time
// #if DOUBLEBUFFER == 1
//     if (flex_is_dm_core() && (flex_get_cluster_id() == 0)) {
//         printf("\n[DEBUG][DMA][HBM->L1] 4) load second half of activation matrix and dequantize");
//         flex_timer_start();
//         flex_dma_async_1d((uint64_t)(uintptr_t)L1_PTRS.activation + (FP16_M * FP16_N / 2) * sizeof(uint16_t),
//                           (uint64_t)(uintptr_t)&matrix_activation_fp16[FP16_M * FP16_N / 2],
//                           FP16_M * FP16_N / 2 * sizeof(uint16_t));
//         flex_dma_async_wait_all();
//         flex_timer_end();
//     }
//     if (flex_get_cluster_id() == 0 && flex_get_core_id() == SPATZ_CORE) {
//         printf("\n[DEBUG][DQ] 4) dequantize right half of indices ");
//     }
//     dequantize(VQ_TOTAL_GROUPS / 2, VQ_TOTAL_GROUPS); // dequantize right half of weight matrix

// #endif

//     flex_intra_cluster_sync();
//     flex_global_barrier_xy();
// // //step 5 compute the second part of the matrix
// #if DOUBLEBUFFER == 1
//     if (flex_get_cluster_id() == 0 && flex_get_core_id() == SPATZ_CORE) {
//         printf("\n[DEBUG][COMPUTE] 5) last partial now "); // we compute MATMUL of ACT_UPPERHALF * W_DQ_LEFTHALF
//         uint16_t* A0 = (uint16_t*)L1_PTRS.activation;      // [0..M0-1, :]
//         uint16_t* A1 = A0 + FP16_M / 2 * FP16_N;           // [M0..M-1, :]

//         uint16_t* B0 = (uint16_t*)L1_PTRS.W_dq;  // [:, 0..K0-1]
//         uint16_t* B1 = B0 + FP16_N * FP16_K / 2; // [:, K0..K-1]   <-- offset by N*K0

//         uint16_t* C00 = (uint16_t*)L1_PTRS.result; // [0..M0-1, 0..K0-1]
//         uint16_t* C01 = C00 + FP16_K / 2;          // [0..M0-1, K0..K-1]
//         uint16_t* C10 = C00 + FP16_M / 2 * FP16_K; // [M0..M-1, 0..K0-1]
//         uint16_t* C11 = C10 + FP16_K / 2;          // [M0..M-1, K0..K-1]
//         // spatz_matmul_fp16_full_legacy((uint16_t*)L1_PTRS.activation + (FP16_M * FP16_N / 2), // lower half of
//         // activation
//         //                               (uint16_t*)L1_PTRS.W_dq +
//         //                                   (VQ_TOTAL_GROUPS / 2 * VQ_GROUP_SIZE),         // right half of W_dq
//         //                               (uint16_t*)L1_PTRS.result + (FP16_M / 2 * FP16_K), // lower half of result
//         //                               FP16_M / 2, FP16_N, FP16_K / 2);
//         flex_timer_start();
//         spatz_matmul_fp16_full_legacy(A0, B1, C01, FP16_M / 2, FP16_N, FP16_K / 2); // top-right
//         spatz_matmul_fp16_full_legacy(A1, B0, C10, FP16_M / 2, FP16_N, FP16_K / 2); // bottom-left
//         spatz_matmul_fp16_full_legacy(A1, B1, C11, FP16_M / 2, FP16_N, FP16_K / 2); // bottom-right
//         flex_timer_end();
//         //   printf("[DEBUG] Verifying GEMM results against golden reference\n");
//         //   spatz_verify_16(FP16_M * FP16_N, (uint16_t*)L1_PTRS.result, (uint16_t*)matrix_golden_fp16, 0.25f);
//     }
// #endif

//     // flex_global_barrier_xy();
//     // //Step 3 STORE BACK TO HBM

// #if DOUBLEBUFFER == 0
//     if (flex_is_dm_core() && (flex_get_cluster_id() == 0)) {
//         printf("[DEBUG][DMA][L1->HBM] Store result back to HBM!\n");

//         // Use a fixed HBM offset like in the tutorial, not malloc
//         uint64_t w_hat_hbm_offset = 0x10000; // Fixed offset in HBM //TODO change this
//         size_t transfer_size      = 16384 * sizeof(uint16_t);

//         flex_timer_start();
//         // Store L1 data to HBM at the fixed offset
//         flex_dma_async_1d(hbm_addr(w_hat_hbm_offset), (uint64_t)(uintptr_t)L1_PTRS.result, transfer_size);
//         flex_dma_async_wait_all(); // Wait for transfer to complete
//         flex_timer_end();

//         // Dump from the HBM offset where we just stored the data
//         printf("[DEBUG] Dumping W_hat from HBM\n");
//         flex_dump_open();
//         flex_dump_hbm(w_hat_hbm_offset, transfer_size);
//         flex_dump_close();
//     }
// #endif
    /**************************************/
    /*  Program Execution Region -- Stop  */
    /**************************************/
    flex_global_barrier_xy();
    if (flex_get_core_id() == 0 && flex_get_cluster_id() == 0)
        printf("\nfinished!");
    flex_global_barrier_xy();
    flex_eoc(eoc_val);
    return 0;
}