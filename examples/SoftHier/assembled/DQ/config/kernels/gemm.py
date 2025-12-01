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

class SummaGEMM:

    def __init__(self):

        #GEMM parameters
        self.dtype                   = 'fp16'
        self.compute_kernel_gemm                    = 1
        self.m_size                  = 512
        self.n_size                  = 512 * 1
        self.k_size                  = 512 * 1

        #Hyperparamters Settings
        ## [Tile ]: tile size for each cluster
        self.m_tile                  = 128
        self.n_tile                  = 128 // 1
        self.k_tile                  = 128 // 1
        ## [Scale]: How many clusters (x=scale, y=scale) are assigned one GEMM.
        ##          For a set of clusters (x=scale, y=scale) we would call it **Group**
        self.summa_scale_x           = 4
        self.summa_scale_y           = 4
        ## [Group]: How many summa group
        ##          Do we need to reduce all groups
        ##          Adress gaps between groups
        self.summa_group_number      = 1
        self.summa_group_reduce      = 0
        self.summa_group_splitk      = 0
        self.summa_group_splitn      = 0
        self.summa_group_gap_x       = 0
        self.summa_group_gap_w       = 0
        self.summa_group_gap_z       = 0
        ## [Resha]: Reshape options on input and output
        self.resha_x_from_enable     = 0
        self.resha_z_to_enable       = 0
        self.resha_x_from_m          = 128
        self.resha_z_to_m            = 2048
        ## [Numer]: Whether to check numerical correctness
        self.summa_numer             = 1
        self.summa_numer_chunk       = 8192

        # [VQ]: Vector Quantization Settings
        self.vq_enabled                = 1
        
        self.vq_force_weight_load      = 0
        self.vq_source                 = "gen"   # Source: "gen" (generated) or "dl" (downloaded)
        self.vq_algorithm              = "aqlm"  # Algorithm: "aqlm", "vptq", etc.
        self.vq_use_scales            = 1 if  self.vq_algorithm =="aqlm" else 0  # Whether to use per-group scales
        self.vq_num_cb                 = 2 if  self.vq_algorithm =="aqlm" else 1      # Number of codebooks
        self.vq_nbits_per_cb           = 8       # Bits per codebook index (2^8 = 256 centroids)
        self.vq_group_size             = 8       # Centroid vector length (group size)
        self.vq_cb_size                = 256     # Codebook size (number of centroids in each codebook)
        self.compressed_dim            = "N"# Compressed dimension: "N"(columns ) or "K" (rows)
        self.vq_num_groups_per_row_tile = int(self.n_tile / self.vq_group_size)

        # Optional VQ features
        self.vq_codebook_format        = "fp16"  # Codebook storage format
        self.vq_index_format           = "separate" # for multicodebook only: Index format: "packed" or "separate" or "flattened"

        # Pretrained model settings (optional)
        self.vq_use_pretrained         = False   # Load from HuggingFace model
        self.vq_repo_id                = "ISTA-DASLab/Llama-2-7b-AQLM-2Bit-2x8-hf"    # e.g., "ISTA-DASLab/Llama-2-7b-AQLM-PV-2Bit-2x8-hf"
        self.vq_weight_file            = None    # e.g., "model.safetensors"
        self.vq_layer_prefix           = None    # e.g., "model.layers.0.mlp.down_proj"
        dtype_bytes = self._dtype_nbytes()

        if self.summa_group_splitk > 0:
            total_k_partitions = self.summa_group_number * self.summa_group_splitk
            if total_k_partitions <= 1:
                raise ValueError("SplitK requires summa_group_number * summa_group_splitk > 1.")
            if self.k_size % total_k_partitions != 0:
                raise ValueError(
                    f"K dimension {self.k_size} must be divisible by summa_group_number*summa_group_splitk ({total_k_partitions})."
                )
            k_chunk = self.k_size // total_k_partitions
            self.summa_group_gap_x = k_chunk * dtype_bytes
            self.summa_group_gap_w = k_chunk * self.n_size * dtype_bytes
            self.summa_group_gap_z = 0
            self.summa_group_reduce = 1

        self._validate_alignment()

    def _validate_alignment(self):
        """catch invalid tilinglayout choices."""
        m_block = self.summa_scale_y * self.m_tile
        n_block = self.summa_scale_x * self.n_tile

        if self.m_size % m_block != 0:
            raise ValueError(f"M dimension {self.m_size} must be a multiple of summa_scale_y*m_tile ({m_block}).")
        if self.n_size % n_block != 0:
            raise ValueError(f"N dimension {self.n_size} must be a multiple of summa_scale_x*n_tile ({n_block}).")
        if self.k_size % self.k_tile != 0:
            raise ValueError(f"K dimension {self.k_size} must be a multiple of k_tile ({self.k_tile}).")
        if self.n_tile % self.vq_group_size != 0:
            raise ValueError(f"n_tile {self.n_tile} must be a multiple of VQ group size ({self.vq_group_size}).")

    def _dtype_nbytes(self):
        dtype_bytes = {
            'fp16': 2,
            'fp8': 1,
            'uint8': 1,
            'int8': 1,
            'fp32': 4,
        }
        if self.dtype not in dtype_bytes:
            raise ValueError(f"Unsupported dtype '{self.dtype}' for gap calculation.")
        return dtype_bytes[self.dtype]
