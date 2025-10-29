#ifndef DQ_COMPUTE_SPATZ_HH
#define DQ_COMPUTE_SPATZ_HH

#include "spatz_rvv_extensions.h"

#include <inttypes.h>
#include <stdio.h>

static inline void spatz_gemv_fp16_full_legacy(uint16_t* matrix_a, uint16_t* matrix_b, uint16_t* matrix_c,
                                               const uint32_t M, const uint32_t N, const uint32_t P) {

    uint32_t p   = 0;
    uint32_t avl = P;
    uint32_t vl;
    uint32_t m = 0;
    do {
        asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(avl));
        // for (uint32_t m = 0; m < M; m++) {
        uint16_t* p_c = matrix_c + m * P + p;
        for (uint32_t n = 0; n < N; n++) {
            // Load scalar a using integer load + broadcast (same as dequantization fix)
            uint16_t* p_a = &matrix_a[m * N + n]; // single scalar A[m][n]
            uint16_t* p_b = &matrix_b[p + n * P]; // vector B[n][p]..B[n][p+vl]

            asm volatile("vle16.v v0, (%0)" ::"r"(p_b)); // load b vector

            if (n == 0) {                               // first iteration  initialize accumulator
                asm volatile("lhu t0, (%0)\n\t"         // load scalar a as integer
                             "vmv.v.x v8, t0\n\t"       // broadcast to vector (v8 is  start for m8)
                             "vfmul.vv v16, v0, v8\n\t" // multiply: v16 = b * a
                             ::"r"(p_a)
                             : "t0", "v0", "v8", "v16");
            } else {                                     // accumulate: v16 = v16 + (b * a)
                asm volatile("lhu t0, (%0)\n\t"          // load scalar a as integer
                             "vmv.v.x v8, t0\n\t"        // broadcast to vector (v8 is  start for m8)
                             "vfmacc.vv v16, v0, v8\n\t" // fmacc
                             ::"r"(p_a)
                             : "t0", "v0", "v8", "v16");
            }
        }
        asm volatile("vse16.v v16, (%0)" ::"r"(p_c)); // store
        // }
        avl -= vl;
        p += vl;
    } while (avl > 0);
}

// inlining gains ca ~ 40 cycle performacmce out of 5 mio
static inline void spatz_matmul_fp16_full_legacy(volatile uint16_t* matrix_a, volatile uint16_t* matrix_b,
                                                 volatile uint16_t* matrix_c, const uint32_t M, const uint32_t N,
                                                 const uint32_t P) {
    uint32_t p   = 0;
    uint32_t avl = P;
    uint32_t vl;
    asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(avl));
    do {
        for (uint32_t m = 0; m < M; m++) {
            volatile uint16_t* p_c = matrix_c + m * P + p;
            for (uint32_t n = 0; n < N; n++) {
                // Load scalar a using integer load + broadcast (same as dequantization fix)
                volatile uint16_t* p_a = &matrix_a[m * N + n]; // single scalar A[m][n]
                volatile uint16_t* p_b = &matrix_b[p + n * P]; // vector B[n][p]..B[n][p+vl]

                asm volatile("vle16.v v0, (%0)" ::"r"(p_b)); // load b vector

                if (n == 0) {                               // first iteration  initialize accumulator
                    asm volatile("lhu t0, (%0)\n\t"         // load scalar a as integer
                                 "vmv.v.x v8, t0\n\t"       // broadcast to vector (v8 is  start for m8)
                                 "vfmul.vv v16, v0, v8\n\t" // multiply: v16 = b * a
                                 ::"r"(p_a)
                                 : "t0", "v0", "v8", "v16");
                } else {                                     // accumulate: v16 = v16 + (b * a)
                    asm volatile("lhu t0, (%0)\n\t"          // load scalar a as integer
                                 "vmv.v.x v8, t0\n\t"        // broadcast to vector (v8 is  start for m8)
                                 "vfmacc.vv v16, v0, v8\n\t" // fmacc
                                 ::"r"(p_a)
                                 : "t0", "v0", "v8", "v16");
                }
            }
            asm volatile("vse16.v v16, (%0)" ::"r"(p_c)); // store
        }
        avl -= vl;
        p += vl;
    } while (avl > 0);
}

static inline void spatz_matmul_fp16_8wide(uint16_t* matrix_a, uint16_t* matrix_b, uint16_t* matrix_c, const uint32_t M,
                                           const uint32_t N, const uint32_t P) {

    uint32_t p   = 0;
    uint32_t avl = P;
    uint32_t vl;
    do {
        avl = (avl > 8) ? 8 : avl; // limit to 8-wide
        asm volatile("vsetvli %0, %1, e16, m1, ta, ma" : "=r"(vl) : "r"(avl));
        for (uint32_t m = 0; m < M; m++) {
            uint16_t* p_c = matrix_c + m * P + p;
            for (uint32_t n = 0; n < N; n++) {
                // Load scalar a using integer load + broadcast (same as dequantization fix)
                uint16_t* p_a = &matrix_a[m * N + n]; // single scalar A[m][n]
                uint16_t* p_b = &matrix_b[p + n * P]; // vector B[n][p]..B[n][p+vl]

                asm volatile("vle16.v v0, (%0)" ::"r"(p_b)); // load b vector

                if (n == 0) {                               // first iteration  initialize accumulator
                    asm volatile("lhu t0, (%0)\n\t"         // load scalar a as integer
                                 "vmv.v.x v8, t0\n\t"       // broadcast to vector (v8 is  start for m1)
                                 "vfmul.vv v16, v0, v8\n\t" // multiply: v16 = b * a
                                 ::"r"(p_a)
                                 : "t0", "v0", "v8", "v16");
                } else {                                     // accumulate: v16 = v16 + (b * a)
                    asm volatile("lhu t0, (%0)\n\t"          // load scalar a as integer
                                 "vmv.v.x v8, t0\n\t"        // broadcast to vector (v8 is  start for m1)
                                 "vfmacc.vv v16, v0, v8\n\t" // fmacc
                                 ::"r"(p_a)
                                 : "t0", "v0", "v8", "v16");
                }
            }
            asm volatile("vse16.v v16, (%0)" ::"r"(p_c)); // store
        }
        p += vl;
        avl = P - p;
    } while (p < P);
}

#endif