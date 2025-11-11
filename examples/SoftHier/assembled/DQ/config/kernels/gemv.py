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
#8192+2*128+6*(128**2+128*256)+3/4*256*128s

# Author: Chi Zhang <chizhang@ethz.ch>

#      K         N            N
#   |-----|   |-----|      |-----|
# M |  X  | x |  W  | K => |  Z  | M
#   |-----|   |-----|      |-----|

class SummaGEMM:

    def __init__(self):

        #GEMM parameters
        self.dtype                   = 'fp16'
        self.m_size                  = 512
        self.n_size                  = 512
        self.k_size                  = 512

        #Hyperparamters Settings
        ## [Tile ]: tile size for each cluster
        self.m_tile                  = 128
        self.n_tile                  = 128
        self.k_tile                  = 128
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
        self.vq_enabled                = 0
        if(self.vq_enabled):
            self.vq_algorithm            = "aqlm"  # Algorithm: "aqlm", "vptq", etc.
            self.vq_num_cb               = 2       # Number of codebooks
            self.vq_nbits_per_cb         = 8       # Bits per codebook index (2^8 = 256 centroids)
            self.vq_group_size           = 8       # Centroid vector length (group size)
            self.vq_cb_size              = 256     # Codebook size (number of centroids in each codebook)
            self.vq_num_groups_per_row_tile = int(self.n_tile / self.vq_group_size)

            # Optional VQ features
            self.vq_codebook_format      = "fp16"  # Codebook storage format
            self.vq_index_format         = "packed" # Index format: "packed" or "separate" or "flattened"

            # Pretrained model settings (optional)
            self.vq_use_pretrained       = False   # Load from HuggingFace model
            self.vq_repo_id              = "ISTA-DASLab/Llama-2-7b-AQLM-PV-2Bit-2x8-hf"    # e.g., "ISTA-DASLab/Llama-2-7b-AQLM-PV-2Bit-2x8-hf"
            self.vq_weight_file          = None    # e.g., "model.safetensors"
            self.vq_layer_prefix         = None    # e.g., "model.layers.0.mlp.down_proj"
            

