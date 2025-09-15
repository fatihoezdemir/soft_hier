#!/usr/bin/env python3
# Copyright 2025 ETH Zurich and University of Bologna.
#
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Author: Bowen Wang, ETH Zurich
# This script generates `data_spatz_matmul_fp8.h`
# 
# Input dense matrix A (fp_8), input sparse matrix B (fp_8)
# Output dense matrix C (fp_16)
from argparse import Namespace
import os
import numpy as np
import argparse
import sys
# Add the correct path relative to the script location
script_dir = os.path.dirname(os.path.abspath(__file__))
flex_utils_path = os.path.join(script_dir, '../../../../../../soft_hier/flex_cluster_utilities/')
sys.path.append(flex_utils_path)
from flex_libfp8 import write_matrix_to_header, generate_sparse_fp8_matrix, generate_fp8_matrix, extract_nm_sparsity, write_index_to_header_multi_format,write_indices_u16

out_features = 128
in_features  = 128
dtype_str= "16" #if dtype==torch.float16 else "32"
args = Namespace(
    # Device settings
    
    # Group sizes (from notebook)
    out_group_size=1,# default is 1
    in_group_size=8, #default is 8 
    
    # Codebook settings (from notebook)
    num_codebooks=2,  # Only 2 codebooks as per notebook
    nbits_per_codebook=8,  # 2^8 = 256 centroids per codebook
    # codebook_size=4096,  # Number of centroids
    
    # Additional quantization parameters
    codebook_value_nbits=16,  # less than 16 means we quantize codebooks as well
    codebook_value_num_groups=1,
    scale_nbits=0,  # 0 means no scales, 16 means no compression
    
    # Training settings (from notebook)
    lr=1e-4,  # From notebook
    max_epochs=10,  # Keeping lower for testing, notebook uses 1000 with beam search every 100
    steps_per_epoch=100,  # Reasonable compromise
    beam_size=8,  # From notebook
    
    # Other settings (from notebook)
    init_max_iter=100,  # From notebook
    init_max_points_per_centroid=None,
    print_frequency=7,  # From notebook
    relative_mse_tolerance=None,
    output_format='fp16',  # Add output format for compatibility
    )

def main():
    parser = argparse.ArgumentParser(description='FP8 Matrix Generator')
    parser.add_argument('-kernel', type=str, required=False, default="gemm", help='Computation Kernel: gemv and gemm')

    # # matrix dimension
    parser.add_argument('-M', type=int, required=False, default=out_features, help='Number of rows in matrix A and C')
    parser.add_argument('-N', type=int, required=False, default=in_features, help='Number of cols in A, rows in B')
    parser.add_argument('-K', type=int, required=False, default=128, help='Number of columns in matrix B and C')

    cli_args = parser.parse_args()
    kernel_type=cli_args.kernel
    M, N, K = cli_args.M, cli_args.N, cli_args.K
    parent_dir ="/home/fo/Desktop/master/soft_hier/quantfiles/"
    config_str ="_cb"+ str(args.num_codebooks)+ '_B'+str(args.nbits_per_codebook)+'_g'+str(args.in_group_size)
    dim_str = "M_" + str(M) + "_N" + str(N)
    filename_W='aqlm_W'+config_str+dim_str +'.npy'
    filename_indices='aqlm_idx'+config_str+dim_str +'.npy'
    filename_codebooks = 'aqlm_cb'+config_str+dim_str +'.npy'
    filename_scales = 'aqlm_scales'+config_str+dim_str +'.npy'
    filename_W_hat = "aqlm_W_hat" +config_str+dim_str +'.npy'
    W=np.load(parent_dir+filename_W)
    indices= np.load(parent_dir+filename_indices)
    codebooks = np.load(parent_dir+filename_codebooks)
    scales = np.load(parent_dir+ filename_scales)
    W_hat = np.load(parent_dir+ filename_W_hat)
    
    print(f"Weight matrix shape {W.shape}")
    print(f"Weight matrix shape {W_hat.shape}")

    print(f"codebook shape {codebooks.shape}")
    print(f"indices shape {indices.shape}")
    print(f"scales shape {scales.shape}")
    # print(f"indices \n {indices}")
    # print("scales",scales)
    # print(f"orig codebook\n  {codebooks}")

    print(codebooks[1].reshape(-1))

    codebook_flattened = np.concatenate((codebooks[0].reshape(-1),codebooks[1].reshape(-1))) if codebooks.shape[0]>=1 else ConnectionError
    indices_flattened =  (indices[:,:,1].astype(np.uint16) << 8) | indices[:,:,0].astype(np.uint16)

    print("codebook flattened ",codebook_flattened.shape)

    print("flattening indices to  ",indices_flattened.shape)
    print("Result is 1 packed uint16 index  that contains both uint8 indices : idx_codebook1|idx_codebook0  ",)

    # print("indices_flattened  ",indices_flattened)


    # # Generate float16 input matrices (fp16 represented fp8 precision)
    # Act = generate_fp8_matrix(128, 128)#TODO Parametrize
    rng = np.random.default_rng(seed=42)
    filename_activation=f'{kernel_type}_M{str(M)}_N{str(N)}_K{str(K)}_activation.npy'
    filename_golden_C=f'{kernel_type}_M{str(M)}_N{str(N)}_K{str(K)}_golden_result.npy'


    if(kernel_type=="gemm"):
        Act = rng.random([out_features,in_features]).astype(np.float16)
        print("ACT and What shape ",Act.shape,W_hat.shape)
        golden = np.matmul(Act.astype(np.float16),W_hat.astype(np.float16))
    elif(kernel_type=="gemv"):
        Act = rng.random([out_features]).astype(np.float16)
        print("ACT and What shape ",Act.shape,W_hat.shape)
        golden = np.vecmat(Act.astype(np.float16),W_hat.astype(np.float16))

    print("ACT and What shape ",Act.shape,W_hat.shape)

    np.save(f'{parent_dir}{filename_activation}',Act)
    np.save(f'{parent_dir}{filename_golden_C}',golden)
    # # Compute output path
    script_dir = os.path.dirname(os.path.realpath(__file__))
    include_dir = os.path.abspath(os.path.join(script_dir, '..', 'include'))
    os.makedirs(include_dir, exist_ok=True)
    header_path = os.path.join(include_dir, 'dq_matmul_fp16.h')

    with open(header_path, 'w') as f:
        f.write('// This file is generated with `spatz_matmul_datagen.py`\n\n')
        f.write('#ifndef DQ_MATMUL_FP16_H\n')
        f.write('#define DQ_MATMUL_FP16_H\n\n')
        f.write('#include <stdint.h> \n\n')

    #     # matrix dimension
        f.write(f'#define GEMV {1}\n') if kernel_type=='gemv' else f.write(f'#define GEMM {1}\n')
        f.write(f'#define FP16_M {out_features} // rows of A and C\n')
        f.write(f'#define FP16_N {in_features} // shared dim of Aand B\n')
        f.write(f'#define FP16_K {in_features} // cols of B and C \n')
        f.write(f'#define VQ_GROUP_SIZE {args.in_group_size} // \n')
        f.write(f'#define VQ_NUM_GROUPS_PER_ROW {int(in_features/args.in_group_size)} // how often we need to dequantize to get a full reconstructed row\n')
        f.write(f'#define VQ_TOTAL_GROUPS {int(out_features*in_features/args.in_group_size)} // number of dequantizations to fully dequantize the matrix\n')

        f.write(f'#define VQ_NUM_CBS {int(args.num_codebooks)} // number of codebooks\n')
        f.write(f'#define VQ_CB_NUM_CENTROIDS {int(2**args.nbits_per_codebook)} // number of codebook entries (2^num_bits) \n\n')

        write_matrix_to_header(f, 'matrix_cb_fp16', codebook_flattened, fmt='fp16', dtype='uint16_t')#args.input_format)
        write_indices_u16(     f, 'matrix_idx_packed_uint16', indices)
        write_matrix_to_header(f, 'matrix_activation_fp16', Act, fmt='fp16', dtype='uint16_t')#args.input_format)
        write_matrix_to_header(f, 'matrix_scales_fp16', scales, fmt='fp16', dtype='uint16_t')#args.input_format)
        write_matrix_to_header(f, 'matrix_golden_fp16', golden, fmt='fp16', dtype='uint16_t')#args.input_format)


        f.write('#endif // DQ_MATMUL_FP16_H\n')

    print('-' * 60)
    print(f'Kernel {kernel_type.upper()} with VQ Matrix dimensions {M} {N} {K}')
    print(f'Codebook shape {codebook_flattened.shape}')

    print(f'[✓] Header written to: {header_path}')

if __name__ == '__main__':
    main()
