import os
import sys
import numpy as np
sys.path.append("soft_hier/soft_hier/flex_cluster_utilities/")

import preload as pld

# np.random.seed(42)

if __name__ == '__main__':
    # start_address = 0x00000000
    # K = 64
    # M = 64
    # N = 64
    # rng = np.random.default_rng(seed=42)
    # A_host = rng.random((M, K)).astype(np.float16)
    # B_host = np.ones((K, N)).astype(np.float16)
    # C_host = np.zeros((M, N)).astype(np.float16)
    # golden = np.matmul(A_host, B_host).astype(np.float16)

    # A_address = 64 + start_address
    # B_address = 64 + A_host.nbytes + start_address
    # C_address = 64 + A_host.nbytes + B_host.nbytes + start_address
    # golden_address = 64 + A_host.nbytes + B_host.nbytes + C_host.nbytes + start_address
    # # create a uint32 np array to store the addresses
    # args = np.array([A_address, B_address, C_address, K, M, N, golden_address], dtype=np.uint32)
    # # print args in hex

    # script_dir = os.path.dirname(os.path.abspath(__file__))

    # pld.make_preload_elf(script_dir+"/mypreload.elf",
    #                      [ A_host, B_host, C_host, golden],
    #                      [A_address , B_address , C_address, golden_address]
    #                      )
    # print(f"A {A_host[0:2,:]}\n B {B_host[0:2,:]} \n C  {C_host[0:2,:]}\n golden {golden[0:2,:]}")

    rng = np.random.default_rng(seed=42)
    W = rng.random((64,64)).astype(np.float16)
    x = rng.random(( 64)).astype(np.float16)
    y = np.zeros((64)).astype(np.float32)
    golden = np.matvec(W, x).astype(np.float32)
    print(W)
    matrix_size_in_byte = W.size * 2  # float16 is 2 bytes each
    print(f"matrix size {matrix_size_in_byte}  W {W.nbytes} bytes, x {x.nbytes} bytes, y {y.nbytes} bytes , golden {golden.nbytes} bytes")
    HBM_base_address     = 0xc0000400

    preload_W_into_HBM_addr = HBM_base_address 
    preload_x_into_HBM_addr = HBM_base_address + W.nbytes  
    preload_y_into_HBM_addr = HBM_base_address + W.nbytes + x.nbytes
    preload_golden_into_HBM_addr = HBM_base_address + W.nbytes + x.nbytes + y.nbytes

    pld.make_preload_elf(
        "myy_preload.elf",
        [W, x, y, golden],
        [
        preload_W_into_HBM_addr,
        preload_x_into_HBM_addr,
        preload_y_into_HBM_addr,
        preload_golden_into_HBM_addr
        ]
    )
