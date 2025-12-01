#
# Copyright (C) 2025 ETH Zurich and University of Bologna
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# Author: Chi Zhang <chizhang@ethz.ch>

import re
import os
import io
import sys
import math
import shutil
import torch
import argparse
import numpy as np
import importlib.util
from tqdm import tqdm
import preload as pld
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from quantizers.vq_data_handler import VQDataHandler, VQConfig

def import_module_from_path(module_path):
    """
    Dynamically import a module from an absolute path and mimic `from module import *`.
    """
    module_name = os.path.splitext(os.path.basename(module_path))[0]  # Extract the file name without extension
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None:
        raise ImportError(f"Cannot find a module at path: {module_path}")
    
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    # Mimic `from module import *`
    globals().update(vars(module))
    return module

def gen_vq_preload_data(gemm):
    """Generate VQ-specific preload data (codebooks, indices, scales)"""
    # Create VQ config from GEMM config
    # W matrix has shape (K, N) for X @ W where X is (M, K)
    vq_config = VQConfig(
        k_size=gemm.k_size,
        n_size=gemm.n_size,
        num_codebooks=gemm.vq_num_cb,
        nbits_per_codebook=gemm.vq_nbits_per_cb,
        in_group_size=gemm.vq_group_size,
    )

    # Initialize VQ handler
    vq_handler = VQDataHandler(vq_config)

    # Check if we should load from pretrained or cache
    cache_dir = os.path.join(os.path.dirname(__file__), '../../vq_cache')

    # Determine algorithm and source from gemm config
    vq_algorithm = getattr(gemm, 'vq_algorithm', 'aqlm')  # Default to 'aqlm'
    vq_source = getattr(gemm, 'vq_source', 'gen')  # Default to 'gen' if not specified
    source_prefix = f"{vq_algorithm}_{vq_source}"
    source_name = "generated" if vq_source == "gen" else "downloaded"

    # Match naming convention from quantizer scripts
    # AQLM: {num_codebooks}x{in_group_size}, e.g., 2x8
    # VPTQ: 1x{vlen_normal}, e.g., 1x6 for 4096 centroids
    if vq_algorithm == 'aqlm':
        config_str = f"{vq_config.num_codebooks}x{vq_config.in_group_size}"
    elif vq_algorithm == 'vptq':
        # For VPTQ, use single codebook with vector length
        config_str = f"1x{vq_config.in_group_size}"
    else:
        raise ValueError(f"Unsupported vq_algorithm: {vq_algorithm}. Use 'aqlm' or 'vptq'.")

    dim_str = f"_dim{vq_config.k_size}x{vq_config.n_size}"

    codebooks_path = os.path.join(cache_dir, f"{source_prefix}{config_str}{dim_str}_cb.npy")
    indices_path = os.path.join(cache_dir, f"{source_prefix}{config_str}{dim_str}_idx.npy")
    scales_path = os.path.join(cache_dir, f"{source_prefix}{config_str}{dim_str}_scales.npy")
    w_hat_path = os.path.join(cache_dir, f"{source_prefix}{config_str}{dim_str}_W_hat.npy")

    print(f"VQ algorithm: {vq_algorithm}")
    print(f"VQ source: {source_name} (vq_source='{vq_source}')")
    print(f"Checking for VQ cache at: {indices_path}")
    if os.path.exists(codebooks_path) and os.path.exists(indices_path):
        print(f"Loading VQ data from cache: {cache_dir}")
        vq_handler.load_from_files(codebooks_path, indices_path, scales_path, w_hat_path)
    else:
        print("WARNING: VQ cache not found. Generating test VQ data with random quantization.")
        print(f"Expected files at: {cache_dir}")

        # Generate random codebooks and indices for testing the VQ pipeline
        # NOTE: This creates random quantized data, NOT a quantization of the reference W matrix
        # -> numerical errors will be high 
        np.random.seed(0)  # Use consistent seed
        vq_handler.codebooks = np.random.randn(
            vq_config.num_codebooks,
            vq_config.codebook_size,
            vq_config.in_group_size
        ).astype(np.float16)

        # Random indices (in real scenario, these would come from k-means clustering)
        vq_handler.indices = np.random.randint(
            0, vq_config.codebook_size,
            size=(vq_config.k_size, vq_config.num_groups_per_row, vq_config.num_codebooks),
            dtype=np.uint8
        )
        vq_handler.scales = np.ones(vq_config.k_size, dtype=np.float16)

        # Dequantize to get reconstructed W
        vq_handler.dequantize()

        print("NOTE: Random VQ data will produce high MSE vs reference. This tests shape correctness only.")
        print("For accurate numerical testing, use real quantized weights via 'make vq-download'.")

    # Prepare data for preload
    vq_data = vq_handler.prepare_for_preload(dtype=gemm.dtype)

    return vq_data, vq_handler.W_reconstructed


def gen_gemm_preload_numpy_arrays(gemm):

    seed = 0
    torch.manual_seed(seed)

    # pick device; fp8 compute is limited, so we do compute in f16/f32 and store in fp8
    device = torch.device("cpu")

    # make sure FP8 (e5m2) is available in this PyTorch
    if not hasattr(torch, "float8_e5m2"):
        raise RuntimeError("This PyTorch build does not expose torch.float8_e5m2.")

    fptype = torch.float8_e5m2 if gemm.dtype == "fp8" else torch.float16

    # helper: create random fptype tensors by sampling in f16 then casting to fptype for storage
    def rand_fptype(shape, device):
        # sample in a safe range to reduce saturation when casting to fptype
        # normal(0, 0.5) tends to survive e5m2 rounding reasonably well
        x = torch.randn(shape, device=device, dtype=torch.float32) * 0.1 + 0.0625
        return x.to(fptype)

    compute_dtype = torch.float16

    # X W and Z in fptype
    X_fptype = rand_fptype((gemm.m_size,  gemm.k_size), device)

    # Determine whether we should load VQ weights (even if HW support is disabled)
    vq_hw_enabled = bool(getattr(gemm, 'vq_enabled', 0))
    vq_force_weights = bool(getattr(gemm, 'vq_force_weight_load', 0))
    use_vq_weights = vq_hw_enabled or vq_force_weights

    # Handle W matrix based on VQ enablement / force flag
    if use_vq_weights:
        if vq_hw_enabled:
            print("VQ enabled - using quantized weight matrix")
        else:
            print("VQ weight load forced - using quantized weight matrix without HW VQ support")
        vq_data, W_reconstructed_np = gen_vq_preload_data(gemm)
        # Convert reconstructed weights to torch tensor
        W_reconstructed = torch.from_numpy(W_reconstructed_np).to(device)
        if W_reconstructed.dtype != fptype:
            W_reconstructed = W_reconstructed.to(fptype)
        W_fptype = W_reconstructed
        print(f"W_reconstructed shape: {W_reconstructed.shape} (expected: K={gemm.k_size}, N={gemm.n_size})")
        assert W_reconstructed.shape == (gemm.k_size, gemm.n_size), \
            f"W shape mismatch: got {W_reconstructed.shape}, expected ({gemm.k_size}, {gemm.n_size})"
        if not vq_hw_enabled:
            # Skip embedding VQ payload when HW support is disabled
            vq_data = None
    else:
        W_fptype = rand_fptype((gemm.k_size,  gemm.n_size), device)
        vq_data = None

    X = X_fptype.to(compute_dtype)
    W = W_fptype.to(compute_dtype)
    Z = X @ W
    Z_fptype = Z.to(fptype)

    # Reshaping
    if gemm.resha_x_from_enable:
        resha_x_k_size = (gemm.m_size * gemm.k_size) // gemm.resha_x_from_m
        if gemm.resha_x_from_m > gemm.m_size:
            X_fptype = torch.cat(X_fptype.split(resha_x_k_size, dim=1), dim=0)
        elif gemm.resha_x_from_m < gemm.m_size:
            X_fptype = torch.cat(X_fptype.split(gemm.resha_x_from_m, dim=0), dim=1)
        pass
    if gemm.resha_z_to_enable:
        resha_z_n_size = (gemm.m_size * gemm.n_size) // gemm.resha_z_to_m
        if gemm.resha_z_to_m > gemm.m_size:
            Z_fptype = torch.cat(Z_fptype.split(resha_z_n_size, dim=1), dim=0)
        elif gemm.resha_z_to_m < gemm.m_size:
            Z_fptype = torch.cat(Z_fptype.split(gemm.resha_z_to_m, dim=0), dim=1)
        pass

    # Duplicate for groups, foezdemir: no need to duplicate?
    X_list = []
    W_list = []
    Z_list = []
    if gemm.summa_group_gap_x != 0:
        if getattr(gemm, "summa_group_splitk", 0) > 0:
            print("SplitK active: skip duplicating X; address gaps per-group slices.")
        else:
            for g in range(gemm.summa_group_number):
                X_list.append(X_fptype.clone())
            X_fptype = torch.cat(X_list, dim=0)
    if gemm.summa_group_gap_w != 0:
        if getattr(gemm, "summa_group_splitk", 0) > 0:
            print("SplitK active: skip duplicating W; address gaps  per-group slices.")
        else:
            for g in range(gemm.summa_group_number):
                W_list.append(W_fptype.clone())
            W_fptype = torch.cat(W_list, dim=0)
    if gemm.summa_group_gap_z != 0:
        for g in range(gemm.summa_group_number):
            Z_list.append(Z_fptype.clone())
        Z_fptype = torch.cat(Z_list, dim=0)

    # convert to NumPy: cast to float16 (NumPy may not support PyTorch fptype dtype)
    if fptype == torch.float8_e5m2:
        X_np = X_fptype.view(torch.uint8).cpu().numpy()
        W_np = W_fptype.view(torch.uint8).cpu().numpy()
        Z_np = Z_fptype.view(torch.uint8).cpu().numpy()
    else:
        X_np = X_fptype.view(torch.uint16).cpu().numpy()
        W_np = W_fptype.view(torch.uint16).cpu().numpy()
        Z_np = Z_fptype.view(torch.uint16).cpu().numpy()

    # quick sanity prints
    print("X_fptype:", tuple(X_fptype.shape), X_fptype.dtype)
    print("W_fptype:", tuple(W_fptype.shape), W_fptype.dtype)
    print("Z_fptype:", tuple(Z_fptype.shape), Z_fptype.dtype)

    print("X_np:", X_np.shape, X_np.dtype)
    print("W_np:", W_np.shape, W_np.dtype)
    print("Z_np:", Z_np.shape, Z_np.dtype)

    Z_empty = np.zeros(Z_np.shape, dtype=Z_np.dtype)
    Z_golden = Z_np

    # Return VQ data if enabled
    if hasattr(gemm, 'vq_enabled') and gemm.vq_enabled and vq_data is not None:
        return X_np, W_np, Z_empty, Z_golden, vq_data
    else:
        return X_np, W_np, Z_empty, Z_golden, None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate C header files from a GEMM preload file and Dynamically import Python modules from absolute or relative paths.")
    parser.add_argument("elf_path", type=str, help="Path of preload elf file")
    parser.add_argument("header_path", type=str, help="Path to GEMM preload C header file")
    parser.add_argument(
        "module_paths",
        metavar="module_path",
        type=str,
        nargs="+",
        help="Paths to Python modules to import (absolute or relative)."
    )

    args = parser.parse_args()

    # Process each provided module path
    for module_path in args.module_paths:
        # Convert to absolute path
        absolute_path = os.path.abspath(module_path)

        if not os.path.isfile(absolute_path):
            print(f"Error: {absolute_path} is not a valid file.")
            continue

        try:
            # Import the module dynamically
            module = import_module_from_path(absolute_path)
            print(f"Successfully imported: {module.__name__} from {absolute_path}")
        except Exception as e:
            print(f"Failed to import {absolute_path}: {e}")

    # Instanciate arch and gemm configurations
    gemm = SummaGEMM()
    arch = FlexClusterArch()

    # Generate the preload
    X_np, W_np, Z_empty, Z_golden, vq_data = gen_gemm_preload_numpy_arrays(gemm)
    X_addr = arch.hbm_start_base
    W_addr = arch.hbm_start_base + arch.hbm_node_addr_space * 2 * arch.num_cluster_y + arch.hbm_node_addr_space * arch.num_cluster_x
    Z_eaddr = X_addr + X_np.nbytes
    Z_gaddr = W_addr + W_np.nbytes

    # Calculate VQ data addresses if enabled
    VQ_codebooks_addrs = []
    VQ_indices_addrs = []
    if vq_data is not None:
        # Place VQ data after Z_golden
        current_addr = Z_gaddr + Z_golden.nbytes

        # Allocate addresses for each codebook separately
        num_codebooks = len(vq_data['codebooks_split'])
        for i, cb in enumerate(vq_data['codebooks_split']):
            VQ_codebooks_addrs.append(current_addr)
            print(f"VQ_codebook_{i}_addr = {current_addr: #x}")
            current_addr += cb.nbytes

        # Allocate """""""" separately
        for i, idx in enumerate(vq_data['indices_split']):
            VQ_indices_addrs.append(current_addr)
            print(f"VQ_indices_{i}_addr = {current_addr: #x}")
            current_addr += idx.nbytes

        VQ_scales_addr = current_addr
        print(f"VQ_scales_addr = {VQ_scales_addr: #x}")

    print(f"X_addr = {X_addr: #x}")
    print(f"Z_eaddr = {Z_eaddr: #x}")
    print(f"W_addr = {W_addr: #x}")
    print(f"Z_gaddr = {Z_gaddr: #x}")

    #generate preload elf
    if gemm.summa_numer == 1:
        if vq_data is not None:
            # Include VQ data in preload - store each codebook and indices separately
            data_arrays = [X_np, Z_empty, W_np, Z_golden]
            data_addrs = [X_addr, Z_eaddr, W_addr, Z_gaddr]

            # Add each codebook separately
            for cb in vq_data['codebooks_split']:
                data_arrays.append(cb)
            data_addrs.extend(VQ_codebooks_addrs)

            # Add each indices array separately
            for idx in vq_data['indices_split']:
                data_arrays.append(idx)
            data_addrs.extend(VQ_indices_addrs)

            # Add scales
            data_arrays.append(vq_data['scales'])
            data_addrs.append(VQ_scales_addr)

            pld.make_preload_elf(args.elf_path, data_arrays, data_addrs)
            print(f"Generated preload ELF with VQ data ({len(vq_data['codebooks_split'])} codebooks, {len(vq_data['indices_split'])} indices arrays)")
        else:
            pld.make_preload_elf(args.elf_path,
                [X_np,    Z_empty,  W_np,   Z_golden],
                [X_addr,  Z_eaddr,  W_addr, Z_gaddr])
    else:
        dumnp = np.array([1,1,1,1,1])
        pld.make_preload_elf(args.elf_path,
            [dumnp,   dumnp,    dumnp,  dumnp],
            [X_addr,  Z_eaddr,  W_addr, Z_gaddr])
        pass

    # Generate gemm_addresses.h (shared address header)
    addresses_header_path = os.path.join(os.path.dirname(args.header_path), 'gemm_addresses.h')
    with open(addresses_header_path, 'w') as file:
        file.write('// This file is generated by gemm_preload.py\n')
        file.write('// HBM address definitions for GEMM (preload workflow)\n\n')
        file.write('#ifndef _GEMM_ADDRESSES_H_\n')
        file.write('#define _GEMM_ADDRESSES_H_\n\n')
        file.write('#include <stdint.h>\n\n')
        file.write('// HBM Addresses\n')
        file.write(f'#define {"X_addr".upper()} ((uint64_t){X_addr: #x})\n')
        file.write(f'#define {"Z_eaddr".upper()} ((uint64_t){Z_eaddr: #x})\n')
        file.write(f'#define {"W_addr".upper()} ((uint64_t){W_addr: #x})\n')
        file.write(f'#define {"Z_gaddr".upper()} ((uint64_t){Z_gaddr: #x})\n')

        # Add VQ addresses if enabled
        if vq_data is not None:
            file.write(f'\n// VQ Data Addresses - Multiple Codebooks\n')
            for i, addr in enumerate(VQ_codebooks_addrs):
                file.write(f'#define VQ_CODEBOOK_{i}_ADDR ((uint64_t){addr: #x})\n')

            file.write(f'\n// VQ Codebook addresses as array initializer\n')
            addrs_str = ', '.join([f'{addr:#x}' for addr in VQ_codebooks_addrs])
            file.write(f'#define VQ_CODEBOOKS_ADDRS {{{addrs_str}}}\n')

            file.write(f'\n// VQ Indices Addresses - Multiple Arrays (one per codebook)\n')
            for i, addr in enumerate(VQ_indices_addrs):
                file.write(f'#define VQ_INDICES_{i}_ADDR ((uint64_t){addr: #x})\n')

            file.write(f'\n// VQ Indices addresses as array initializer\n')
            indices_addrs_str = ', '.join([f'{addr:#x}' for addr in VQ_indices_addrs])
            file.write(f'#define VQ_INDICES_ADDRS {{{indices_addrs_str}}}\n')

            file.write(f'\n#define VQ_SCALES_ADDR ((uint64_t){VQ_scales_addr: #x})\n')

        file.write('\n#endif // _GEMM_ADDRESSES_H_\n')

    print(f'Address header "{addresses_header_path}" generated successfully.')

    # Generate preload.h (for backward compatibility, just includes gemm_addresses.h)
    with open(args.header_path, 'w') as file:
        file.write('// This file is generated by gemm_preload.py\n')
        file.write('// For backward compatibility - includes gemm_addresses.h\n\n')
        file.write('#ifndef _GEMM_PRELOAD_H_\n')
        file.write('#define _GEMM_PRELOAD_H_\n\n')
        file.write('#include "gemm_addresses.h"\n\n')
        file.write('#endif // _GEMM_PRELOAD_H_\n')

    print(f'Preload header "{args.header_path}" generated successfully.')
