#ifndef DQ_LOAD_DATA_L1_HH
#define DQ_LOAD_DATA_L1_HH
#include "dq_data_hbm.h"
#include "dq_helpers.h"
#include "flex_alloc.h"
#include "flex_dma_pattern.h"

extern const int NUM_TILES;

#define GROUPS_PER_TILE (VQ_NUM_GROUPS_PER_ROW / NUM_TILES)
#define REMAINDER_GROUPS (VQ_NUM_GROUPS_PER_ROW % NUM_TILES)

// For activation matrix tiling (must match weight matrix number of rows for GEMM)
#define COLS_PER_TILE (FP16_K / NUM_TILES)
#define ROWS_PER_TILE (FP16_M / NUM_TILES)
// calculate actual size for a specific tile
static inline uint32_t get_groups_for_tile(uint32_t tile_index) {
    return GROUPS_PER_TILE + (tile_index < REMAINDER_GROUPS ? 1 : 0);
}

static inline uint32_t get_rows_for_tile(uint32_t tile_index) {
    return ROWS_PER_TILE + (tile_index < (FP16_M % NUM_TILES) ? 1 : 0);
}

// Account for extra groups in previous tiles, All remainder groups are in first tiles
static inline uint32_t get_start_group_for_tile(uint32_t tile_index) {
    uint32_t start_group = tile_index * GROUPS_PER_TILE;
    start_group += (tile_index < REMAINDER_GROUPS) ? tile_index : REMAINDER_GROUPS;
    return start_group;
}
static inline uint32_t get_start_row_for_tile(uint32_t tile_index) {
    uint32_t start_row      = tile_index * ROWS_PER_TILE;
    uint32_t remainder_rows = FP16_M % NUM_TILES;
    start_row += (tile_index < remainder_rows) ? tile_index : remainder_rows;
    return start_row;
}

void dq_load_codebook_l1() {
    // 1) first allocate codebook and indices
    g_l1_dq.cb = (uint16_t*)(uintptr_t)flex_l1_malloc(
        VQ_CB_NUM_CENTROIDS * VQ_GROUP_SIZE * VQ_NUM_CBS *
        sizeof(uint16_t)); // codebook size, [256,8,2],2 codebooks,256 centroids of size 8(8 is group size, 1
    // decoding decodes entry to 8 values)
    g_l1_dq.scales = (uint16_t*)(uintptr_t)flex_l1_malloc(
        FP16_M * sizeof(uint16_t)); // scales size (128) is the numnerb of rows of W
    // // flex_timer_start();
    flex_dma_async_1d((uint64_t)(uintptr_t)g_l1_dq.cb, (uint64_t)(uintptr_t)&matrix_cb_fp16[0],
                      VQ_CB_NUM_CENTROIDS * VQ_NUM_CBS * VQ_GROUP_SIZE * sizeof(uint16_t));
    flex_dma_async_1d((uint64_t)(uintptr_t)g_l1_dq.scales, (uint64_t)(uintptr_t)&matrix_scales_fp16[0],
                      FP16_M * sizeof(uint16_t));
    flex_dma_async_wait_all();
    // // flex_timer_end();
}

// Load a specific tile of the weight matrix indices into a compact buffer
void dq_load_indices_tile(void* dest, const void* src, uint32_t tile_index) {
    // Calculate groups for this tile
    uint32_t groups_this_tile = get_groups_for_tile(tile_index);
    // Calculate starting group position
    uint32_t start_group = get_start_group_for_tile(tile_index);

    flex_dma_async_2d((uint64_t)(uintptr_t)dest,                                 // compact dest buffer, no offset
                      (uint64_t)(uintptr_t)src + start_group * sizeof(uint16_t), // source with offset
                      groups_this_tile * sizeof(uint16_t),                       // transfer size per row
                      groups_this_tile * sizeof(uint16_t),                       // dest stride = tile width (compact)
                      VQ_NUM_GROUPS_PER_ROW * sizeof(uint16_t),                  // source stride (full row)
                      FP16_M                                                     // all rows
    );
}

void dq_load_indices_tile_u8(void* dest, const void* src, uint32_t tile_index) {
    // Calculate groups for this tile
    uint32_t groups_this_tile = get_groups_for_tile(tile_index);
    // Calculate starting group position
    uint32_t start_group = get_start_group_for_tile(tile_index);

    flex_dma_async_2d((uint64_t)(uintptr_t)dest,                                // compact dest buffer, no offset
                      (uint64_t)(uintptr_t)src + start_group * sizeof(uint8_t), // source with offset
                      groups_this_tile * sizeof(uint8_t),                       // transfer size per row
                      groups_this_tile * sizeof(uint8_t),                       // dest stride = tile width (compact)
                      VQ_NUM_GROUPS_PER_ROW * sizeof(uint8_t),                  // source stride (full row)
                      FP16_M                                                    // all rows
    );
    flex_dma_async_wait_all();
}

void dq_load_indices_tile_u8_flattened(void* dest, const void* src, uint32_t tile_index) {
    // Calculate groups for this tile
    uint32_t groups_this_tile = get_groups_for_tile(tile_index);
    // Calculate starting group position
    uint32_t start_group = get_start_group_for_tile(tile_index);

    flex_dma_async_2d((uint64_t)(uintptr_t)dest,                                // compact dest buffer, no offset
                      (uint64_t)(uintptr_t)src + start_group * sizeof(uint8_t), // source with offset
                      groups_this_tile * sizeof(uint8_t),                       // transfer size per row
                      groups_this_tile * sizeof(uint8_t),                       // dest stride = tile width (compact)
                      VQ_NUM_GROUPS_PER_ROW * sizeof(uint8_t),                  // source stride (full row)
                      FP16_M                                                    // all rows
    );
    flex_dma_async_wait_all();
}

// Load a horizontal stripe of the activation matrix (split by rows only)
// Since rows are contiguous in memory, we can use 1D DMA
void dq_load_activation_tile(void* dest, const void* src, uint32_t row_tile_index) {
    uint32_t rows_this_tile = get_rows_for_tile(row_tile_index);
    uint32_t start_row      = get_start_row_for_tile(row_tile_index);
    uint32_t start_element  = start_row * FP16_N;
    uint32_t num_elements   = rows_this_tile * FP16_N;

    // flex_timer_start();
    // Load horizontal stripe using 1D DMA (more efficient for contiguous data) Dest has no offset since it's a compact
    flex_dma_async_1d((uint64_t)(uintptr_t)dest, (uint64_t)(uintptr_t)src + start_element * sizeof(uint16_t),
                      num_elements * sizeof(uint16_t));
    // flex_dma_async_wait_all();
    // flex_timer_end();
}
// void dq_store_C_tile(void* dst,void* src){

//     flex_dma_async_2d((uint64_t)(uintptr_t)dst, (uint64_t)(uintptr_t)src/*l1_buffers.C_tile*/, P * sizeof(uint16_t),
//     FP16_K * sizeof(uint16_t), // Full C matrix width
//     P * sizeof(uint16_t),      //  tile width
//     /*rows*/ r);
// flex_dma_async_wait_all();

// }

#endif