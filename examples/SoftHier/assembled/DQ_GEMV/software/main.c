#include "flex_runtime.h"
#include "flex_dma_pattern.h"
#include "flex_printf.h"
#include "flex_alloc.h"

#define M 64
#define N 64
int main()
{
    uint32_t eoc_val = 0;
    flex_barrier_xy_init();
    flex_global_barrier_xy();
    if (flex_get_core_id() == 0 && flex_get_cluster_id() == 0) flex_timer_start();
    flex_global_barrier_xy();
    flex_alloc_init();

    flex_intra_cluster_sync();//Cluster barrier
    flex_global_barrier_xy();

    /**************************************/
    /*  Program Execution Region -- Start */
    /**************************************/
    uint32_t size_W = M * N *  sizeof(uint16_t) /*fp16 has 2 bytes*/;
    uint32_t size_x = N * sizeof(uint16_t) /*fp16 has 2 bytes*/;
    uint32_t size_y = M * sizeof(uint32_t) /*fp32 has 4 bytes*/;

    uint64_t W_HBM_off = 0x400;
    uint32_t W_L1_off = 0x14;// both are nonzero due to allocator taking size
    uint64_t x_HBM_off = W_HBM_off + size_W;
    uint32_t x_L1_off= W_L1_off + size_W;

    uint64_t y_HBM_off = W_HBM_off + size_W + size_x;// allocation only
    uint32_t y_L1_off= W_L1_off  + size_W + size_x;
    
    


    uint16_t addr_W;
    uint16_t addr_x;
    uint32_t addr_y;
    // uint32_t addr_golden;
    //STEP 0 (optional) check if the values are preloaded to hbm correctly
    if (flex_is_dm_core() && (flex_get_cluster_id() == 0))
    {
        volatile uint16_t* local_ptr = (volatile uint16_t *)local(W_L1_off);
        volatile uint16_t* hbmptr_w = (volatile uint16_t *) hbm_addr( W_HBM_off);
        volatile uint16_t* hbmptr_x = (volatile uint16_t *) hbm_addr( x_HBM_off);
        volatile uint16_t* hbmptr_y = (volatile uint16_t *) hbm_addr( y_HBM_off);

        printf("[Before load HBM to L1] the first elements of HBM  are:\n");
        for (int i = 0; i < 16; ++i)
        {
            // printf("  HBM[%d]  0x%04x\n",i, hbmptr_w[i]);//todo float displayer, convert
            printf("  HBM[%d]  0x%04x\n",i, hbmptr_x[i]);//todo float displayer, convert

        }
    }
    flex_intra_cluster_sync();

    //STEP 1 : Allocate W, x and y in cluster 0
    if (flex_is_dm_core() && (flex_get_cluster_id() == 0)){
        addr_W = (uint16_t )  flex_l1_malloc(size_W);
        addr_x = (uint16_t ) flex_l1_malloc(size_x);
        addr_y = (uint32_t ) flex_l1_malloc( size_y);
        volatile uint16_t *W_view = (volatile uint16_t*)(uintptr_t)local((uint64_t)addr_W);
        volatile uint16_t *x_view = (volatile uint16_t*)(uintptr_t)local((uint64_t)addr_x);

        printf("malloc addreses  in L1 \nW=0x%04x \nx=0x%04x \ny=0x%04x \n",addr_W,addr_x,addr_y);


        volatile uint16_t* local_ptr_w = (volatile uint16_t *) local( W_L1_off);
        volatile uint16_t* local_ptr_x = (volatile uint16_t *) local( x_L1_off);
        volatile uint16_t* local_ptr_y = (volatile uint16_t *) local( y_L1_off);
        // DMA  W and x
        printf("l1 local offset pointers(precalculated, not used,replaced by malloc )\nW=0x%04x \nx=0x%04x\n",local_ptr_w,local_ptr_x);
        printf("dma destination addreses  in L1 with malloc \nW=0x%04x \nx=0x%04x\n",addr_W,addr_x);

        flex_dma_async_1d(local(addr_W), hbm_addr(W_HBM_off), size_W);
        flex_dma_async_1d(local(addr_x), hbm_addr(x_HBM_off), size_x);


        flex_dma_async_wait_all();

        printf("[After  load HBM to L1] the first 8 elements of local L1 x malloc  are:\n");
        for (int i = 0; i < 65; ++i){
                printf(" x[%d]   0x%04x\n",i, x_view[i]);//how to change to addr_W
                // printf(" L1[%d]   0x%04x\n",i, local_ptr_w[i]);
        }
        
        

    }

    flex_global_barrier_xy();
    //Step 3 Do a GEMV
    if (flex_is_first_core() && (flex_get_cluster_id() == 0)) {
        for (int i=0;i<M;++i){}

    }



    /**************************************/
    /*  Program Execution Region -- Stop  */
    /**************************************/
    flex_global_barrier_xy();
    if (flex_get_core_id() == 0 && flex_get_cluster_id() == 0) flex_timer_end();
    flex_global_barrier_xy();
    flex_eoc(eoc_val);
    return 0;
}