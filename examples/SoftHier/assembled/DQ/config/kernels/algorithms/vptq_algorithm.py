"""
VPTQ  channel-wise algorithm config.

VPTQ uses(the best config that also fits in L1) single-codebook quantization:
  W =CB[idx]

Can optionally support transpose indexing for better memory access patterns.
"""

from .algorithm_base import VQAlgorithm


class VPTQAlgorithm(VQAlgorithm):
    """VPTQ single-codebook configuration."""

    def __init__(self, enable_transpose=False, cb_size=4096):
        super().__init__()

        # VPTQ-specific defaults
        self.num_codebooks = 1                  # Single codebook
        self.codebook_format = 'fp16'
        self.group_size = 6                     # Elements per centroid (can be larger)
        self.nbits_per_cb = 12                  # 2^12 = 4096 centroids (can be larger than AQLM)
        self.cb_size = cb_size                  # Number of centroids
        self.use_scales = False 
        self.enable_transpose = enable_transpose
        self.compress_dim = 'k'                 # Compress rows (K)
        self.idx_bytes = 2
        self.index_format ='separate'

    def get_num_codebooks(self):
        return 1  # VPTQ always uses single codebook

    def get_use_scales(self):
        return self.use_scales 

    def get_buffer_requirements(self):
        """
        VPTQ needs:
        - One L1 codebook buffer
        - Double-buffered indices
        - Optional transpose intermediate buffers
        """
        buffers = {
            'L1_CB': 1,             # Single codebook
            'L1_IDX1': 1,           # Indices buffer 1
            'L1_IDX2': 1,           # Indices buffer 2
        }

        # Add transpose buffers if needed
        if self.enable_transpose:
            buffers['L1_transpose_interm1'] = 1
            buffers['L1_transpose_interm2'] = 1

        return buffers

    def get_dequant_kernel(self):
        """Select kernel based on transpose setting."""
        if self.enable_transpose:
            return "single_cb_transpose"
        return "single_cb"

    def validate_config(self, config):
        """Check VPTQ constraints."""
        # VPTQ supports both row-wise and transpose indexing (for row-wise we use transpose engine)
        # No special constraints for transpose

        # Warn if using very large codebook (memory intensive)
        if self.cb_size > 4096:
            print(f"Warning: Large codebook size {self.cb_size} may exceed L1 capacity")

        # Check that group size is reasonable
        if self.group_size < 1:
            raise ValueError(f"Group size must be >= 1, got {self.group_size}")
