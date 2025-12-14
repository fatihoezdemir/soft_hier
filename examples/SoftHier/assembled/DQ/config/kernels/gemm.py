#
# Copyright (C) 2025 ETH Zurich and University of Bologna
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copyb of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#8192+2*128+6*(128**2+128*256)+3/4*256*128s
#tryout

# Author: Chi Zhang <chizhang@ethz.ch>
#GEMM
#      K         N            N
#   |-----|   |-----|      |-----|
# M |  X  | x |  W  | K => |  Z  | M
#   |-----|   |-----|      |-----|
#GEMV
#      K         N            N
#             |-----|
# 1 | x.T | x |  W  | K => |  Z  | 1
#             |-----|
"""
X Matrix (M×K) - Distributed by ROWS:
     k0    k1    k2   ← K dimension partitions
  ┌─────┬─────┬─────┐
m0│ X00 │ X01 │ X02 │ ← Row 0 clusters
  ├─────┼─────┼─────┤
m1│ X10 │ X11 │ X12 │ ← Row 1 clusters
  ├─────┼─────┼─────┤
m2│ X20 │ X21 │ X22 │ ← Row 2 clusters
  └─────┴─────┴─────┘

W Matrix (K×N) - Distributed by COLUMNS:
     n0    n1    n2   ← N dimension partitions
  ┌─────┬─────┬─────┐
k0│ W00 │ W01 │ W02 │ ← K partition 0
  ├─────┼─────┼─────┤
k1│ W10 │ W11 │ W12 │ ← K partition 1
  ├─────┼─────┼─────┤
k2│ W20 │ W21 │ W22 │ ← K partition 2
  └─────┴─────┴─────┘

Z Matrix (M×N) - Each cluster computes ONE tile:
     n0    n1    n2
  ┌─────┬─────┬─────┐
m0│ Z00 │ Z01 │ Z02 │
  ├─────┼─────┼─────┤
m1│ Z10 │ Z11 │ Z12 │
  ├─────┼─────┼─────┤
m2│ Z20 │ Z21 │ Z22 │
  └─────┴─────┴─────┘
"""

try:
    from .base_kernel import BaseKernel
    from .algorithms import create_algorithm
except ImportError:
    # Fallback for dynamic imports
    from base_kernel import BaseKernel
    from algorithms import create_algorithm


class SummaGEMM(BaseKernel):

    def __init__(self, **kwargs):
        super().__init__()

        # GEMM parameters - can be overridden via kwargs
        self.dtype = kwargs.get('dtype', 'fp16')
        self.compute_kernel_gemm = kwargs.get('compute_kernel_gemm', 1)
        self.m_size = kwargs.get('M', kwargs.get('m_size', 512))
        self.n_size = kwargs.get('N', kwargs.get('n_size', 512))
        self.k_size = kwargs.get('K', kwargs.get('k_size', 512))

        # Tile sizes
        self.m_tile = kwargs.get('m_tile', 128/2)
        self.n_tile = kwargs.get('n_tile', 128/2)
        self.k_tile = kwargs.get('k_tile', 128)

        # SUMMA cluster configuration
        self.summa_scale_x = kwargs.get('summa_scale_x', 4)
        self.summa_scale_y = kwargs.get('summa_scale_y', 4)

        # SUMMA group configuration
        self.summa_group_number = kwargs.get('summa_group_number', 1)
        self.summa_group_reduce = kwargs.get('summa_group_reduce', 0)
        self.summa_group_splitk = kwargs.get('summa_group_splitk', 0)
        self.summa_group_splitn = kwargs.get('summa_group_splitn', 0)
        self.summa_group_gap_x = 0
        self.summa_group_gap_w = 0
        self.summa_group_gap_z = 0

        # Reshape options
        self.resha_x_from_enable = kwargs.get('resha_x_from_enable', 0)
        self.resha_z_to_enable = kwargs.get('resha_z_to_enable', 0)
        self.resha_x_from_m = kwargs.get('resha_x_from_m', 128)
        self.resha_z_to_m = kwargs.get('resha_z_to_m', 2048)

        # Numerical verification
        self.summa_numer = kwargs.get('summa_numer', 1)
        self.summa_numer_chunk = kwargs.get('summa_numer_chunk', 8192)

        # VQ Configuration - auto-enable if using DQ variant
        kernel_variant_requested = kwargs.get('kernel_variant', None)
        if kernel_variant_requested in ['dq', 'splitk','baseline']:
            self.vq_enabled = kwargs.get('vq_enabled', 1)  # Auto-enable VQ for DQ variant
        else:
            self.vq_enabled = kwargs.get('vq_enabled', 1 if kernel_variant_requested is None else 0)

        self.vq_force_weight_load = kwargs.get('vq_force_weight_load', 0)
        self.vq_source = kwargs.get('vq_source', 'gen')  # "gen" or "dl"

        # Create VQ algorithm instance via factory
        vq_algorithm_name = kwargs.get('vq_algorithm', 'aqlm')
        enable_transpose = kwargs.get('enable_transpose', False)

        num_codebooks = kwargs.get('num_codebooks', 2 if vq_algorithm_name == 'aqlm' else 1)
        cb_size = kwargs.get('cb_size', 256 if vq_algorithm_name == 'aqlm' else 4096)
        
        # Kernel variant selection
        # GEMV variants:
        #  'baseline' - no DQ, standard REDMULE
        #  'dq' - DQ-based, separate dequant (SPATZ) + compute (REDMULE)
        #  'splitk' - K-parallel (for future, multi-core)
        self.kernel_variant = kwargs.get('kernel_variant', 'baseline' if self.vq_enabled else 'baseline')
        
        if self.kernel_variant == 'baseline' and vq_algorithm_name == 'vptq':
          enable_transpose = kwargs.get('enable_transpose', True)
        self.vq_alg = create_algorithm(
            vq_algorithm_name,
            enable_transpose=enable_transpose,
            num_codebooks=num_codebooks,
            cb_size=cb_size
        )

        # Validate config against algorithm constraints
        self.vq_alg.validate_config(kwargs)

        # Extract algorithm-specific configs
        self.vq_algorithm = vq_algorithm_name

        # Derived VQ parameters
        self.compressed_dim = kwargs.get('compressed_dim', 'N')
        self.vq_num_groups_per_row_tile = int(self.n_tile / self.vq_alg.group_size)


        # Pretrained model settings
        self.vq_use_pretrained = kwargs.get('vq_use_pretrained', False)
        self.vq_repo_id = kwargs.get('vq_repo_id', 'ISTA-DASLab/Llama-2-7b-AQLM-2Bit-2x8-hf')
        self.vq_weight_file = kwargs.get('vq_weight_file', None)
        self.vq_layer_prefix = kwargs.get('vq_layer_prefix', None)

        # Kernel variant selection
        # GEMM only has: 'baseline' (no DQ) or 'dq' (DQ-based with REDMULE)
        self.kernel_variant = kwargs.get('kernel_variant', 'baseline' if self.vq_enabled else 'baseline')

        # Setup split-K if enabled
        self._setup_splitk()

        # Validate alignment
        self._validate_alignment()

    def get_kernel_function(self):
        """Return C function name based on kernel variant."""
        if self.kernel_variant == 'baseline':
            return 'run_gemm_pipeline'
        elif self.kernel_variant == 'dq':
            return 'run_gemm_pipelinevq'
        else:
            raise ValueError(f"Unknown kernel_variant for GEMM: {self.kernel_variant}. Valid: 'baseline', 'dq'")

    def _validate_alignment(self):
        """Validate tile alignment (extended for VQ)."""
        super()._validate_alignment()

        # VQ-specific validation
        # if self.n_tile % self.vq_group_size != 0:
        #     raise ValueError(
        #         f"n_tile {self.n_tile} must be a multiple of "
        #         f"VQ group size ({self.vq_group_size})"
        #     )
