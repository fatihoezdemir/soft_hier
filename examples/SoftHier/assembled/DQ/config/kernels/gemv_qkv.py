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

try:
    from .base_kernel import BaseKernel
    from .algorithms import create_algorithm
except ImportError:
    # Fallback for dynamic imports
    from base_kernel import BaseKernel
    from algorithms import create_algorithm


class SummaGEMV(BaseKernel):

    def __init__(self, **kwargs):
        super().__init__()

        # GEMV parameters - can be overridden via kwargs
        self.dtype = kwargs.get('dtype', 'fp16')
        self.m_size = kwargs.get('M', kwargs.get('m_size', 1))  # GEMV has M=1
        # Default to 4K columns to fully occupy 4 groups × 4 clusters (128-col tiles)
        self.n_size = kwargs.get('N', kwargs.get('n_size', 4096))
        self.k_size = kwargs.get('K', kwargs.get('k_size', 512))
        self.compute_kernel_gemm = kwargs.get('compute_kernel_gemm', None)
        self.compute_kernel_gemv = kwargs.get('compute_kernel_gemv', 1)

        # Tile sizes (GEMV-specific defaults)
        self.m_tile = kwargs.get('m_tile', 1)   # M=1 for vector
        self.n_tile = kwargs.get('n_tile', 256)
        self.k_tile = kwargs.get('k_tile', 256)

        # SUMMA cluster configuration (GEMV-specific)
        self.summa_scale_x = kwargs.get('summa_scale_x', 4)
        self.summa_scale_y = kwargs.get('summa_scale_y', 1)  # Single row for GEMV

        # SUMMA group configuration (GEMV uses split-N)
        self.summa_group_number = kwargs.get('summa_group_number', 4)
        self.summa_group_reduce = kwargs.get('summa_group_reduce', 0)
        self.summa_group_splitk = kwargs.get('summa_group_splitk', 0)
        self.summa_group_splitn = kwargs.get('summa_group_splitn', 1)  # GEMV uses split-N 
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
        self.summa_numer_chunk = kwargs.get('summa_numer_chunk', 256)

        # VQ Configuration - auto-enable if using DQ/fused variants
        kernel_variant_requested = kwargs.get('kernel_variant', None)
        if kernel_variant_requested in ['dq', 'fused', 'splitk','baseline']:
            self.vq_enabled = kwargs.get('vq_enabled', 1)  # Auto-enable VQ for DQ variants
        else:
            self.vq_enabled = kwargs.get('vq_enabled', 1)  # GEMV has VQ disabled by default

        self.vq_force_weight_load = kwargs.get('vq_force_weight_load', 0)
        self.vq_source = kwargs.get('vq_source', 'gen')  # "gen" or "dl"

        # Create VQ algorithm instance via factory
        # VPTQ Transpose Feature:
        # - enable_transpose=True: Transpose indices from (K, num_groups, 1) to (num_groups, K, 1)
        # - Only useful for VPTQ (baseline) with transpose engine, not needed for AQLM
        #   Example: gemv = SummaGEMV(vq_algorithm='vptq', enable_transpose=True)
        self.vq_algorithm_name = kwargs.get('vq_algorithm', 'aqlm')
        enable_transpose = kwargs.get('enable_transpose', False)

        # Kernel variant selection
        # GEMV variants:
        #  'nodq' - no DQ, standard REDMULE
        #  'baseline' - DQ-based with existing RVV
        #  'acc_load' - DQ-based, with vlxblk instruction (for col-major like vptq, use transpose engine)
        #  'acc_load_store' - DQ-based, with vlxblk and stride segment store instruction (only col-major like vptq, aqlm etc unsupported)
        #  'spatz_comp' DQ-based, with spatz as computing (only aqlm)
        #  'fused' - 'Fused dq and compute
        #  'splitk' - K-parallel (for future, multi-core)(default is splitn)
        self.kernel_variant = kwargs.get('kernel_variant', 'fused' if self.vq_enabled else 'baseline')

        # For baseline + VPTQ, default to enabling transpose to match HW access patterns
        if self.kernel_variant == 'baseline' and self.vq_algorithm_name in ('vptq', 'gptvq'):
            enable_transpose = kwargs.get('enable_transpose', True)

        # Instantiate algorithm before validation
        self.vq_alg = create_algorithm(
            self.vq_algorithm_name,
            enable_transpose=enable_transpose,
            group_size=kwargs.get('vq_group_size', kwargs.get('gptvq_group_size', 4)),
            cb_size=kwargs.get('vq_cb_size', kwargs.get('gptvq_num_centroids', 256)),
        )

        # Validate config against algorithm constraints
        self.vq_alg.validate_config(kwargs)

        # Derived VQ parameters
        self.vq_num_groups_per_row_tile = int(self.n_tile / self.vq_alg.group_size)

        # Pretrained model settings
        self.vq_use_pretrained = kwargs.get('vq_use_pretrained', False)
        self.vq_repo_id = kwargs.get('vq_repo_id', 'ISTA-DASLab/Llama-2-7b-AQLM-2Bit-2x8-hf')
        self.vq_weight_file = kwargs.get('vq_weight_file', None)
        self.vq_layer_prefix = kwargs.get('vq_layer_prefix', None)


        self._setup_splitk() # Setup split-K if enabled
        self._validate_alignment()         # Validate alignment

    def get_kernel_function(self):
        """Return C function name based on kernel variant."""
        if self.kernel_variant == 'baseline':
            return 'run_gemv_pipeline'
        elif self.kernel_variant == 'dq':
            return 'run_gemv_pipelinevq'
        elif self.kernel_variant == 'fused':
            return 'run_gemv_pipelinevq_fused'
        elif self.kernel_variant == 'splitk':
            return 'run_gemv_pipelinevq_splitk'
        else:
            raise ValueError(f"Unknown kernel_variant for GEMV: {self.kernel_variant}. Valid: 'baseline', 'dq', 'fused', 'splitk'")

    def _validate_alignment(self):
        """Validate tile alignment (GEMV-specific)."""
        # Skip M validation for GEMV since M=1
        n_block = self.summa_scale_x * self.n_tile

        if self.n_size % n_block != 0:
            raise ValueError(
                f"N dimension {self.n_size} must be a multiple of "
                f"summa_scale_x*n_tile ({n_block})"
            )
        if self.k_size % self.k_tile != 0:
            raise ValueError(
                f"K dimension {self.k_size} must be a multiple of "
                f"k_tile ({self.k_tile})"
            )

        # VQ-specific validation
        if self.vq_alg.get_algorithm_name()!='vptq' and self.n_tile % self.vq_alg.group_size != 0:
            raise ValueError(
                f"n_tile {self.n_tile} must be a multiple of "
                f"VQ group size ({self.vq_alg.group_size})"
            )

        # Split-N validation
        if self.summa_group_splitn:
            if self.n_size % self.summa_group_number != 0:
                raise ValueError(
                    f"SplitN requires N ({self.n_size}) divisible by "
                    f"summa_group_number ({self.summa_group_number})"
                )
            n_size_per_group = self.n_size // self.summa_group_number
            if n_size_per_group % n_block != 0:
                raise ValueError(
                    f"Per-group N ({n_size_per_group}) must be a multiple of "
                    f"summa_scale_x*n_tile ({n_block}) for split-N. "
                    f"Reduce summa_group_number or summa_scale_x or increase N."
                )


# Backward compat
SummaGEMM = SummaGEMV
