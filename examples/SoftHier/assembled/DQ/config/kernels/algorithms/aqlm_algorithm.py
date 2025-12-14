"""
AQLM multi-codebook additive algorithm configuration.

AQLM uses multi-codebook quantization:
  W = scale * (CB0[idx0] + CB1[idx1] + ... + CB_{n-1}[idx_{n-1}])

Typical config: 2 codebooks x 256 centroids x 8 elements
"""

from .algorithm_base import VQAlgorithm


class AQLMAlgorithm(VQAlgorithm):
    """AQLM multi-codebook configuration."""

    def __init__(self, num_codebooks=2):
        super().__init__()

        # AQLM-specific defaults
        self.num_codebooks = num_codebooks  # Typically 2
        self.codebook_format = 'fp16'
        self.group_size = 8                 # Elements per centroid
        self.nbits_per_cb = 8               # 2^8 = 256 centroids
        self.cb_size = 256                  # Number of centroids
        self.use_scales = True              # AQLM always uses per-row scales
        self.enable_transpose = False       # AQLM is row-wise only
        self.compress_dim = 'n'
        self.idx_bytes = 1
        self.index_format ='separate'
    def get_num_codebooks(self):
        return self.num_codebooks

    def get_use_scales(self):
        return self.use_scales  # AQLM always uses scales

    def get_buffer_requirements(self):
        """
        AQLM needs:
        - One L1 buffer per codebook (stored once)
        - Double-buffered indices per codebook
        - Double-buffered scales
        """
        return {
            'L1_CB': self.num_codebooks,        # Codebooks (loaded once)
            'L1_IDX1': self.num_codebooks,      # Indices buffer 1
            'L1_IDX2': self.num_codebooks,      # Indices buffer 2
            'L1_Scales': 2,                     # Double-buffered scales
            'L1_transpose': 0                   # No transpose support
        }

    def get_dequant_kernel(self):
        """AQLM uses multi-codebook addition kernel."""
        return "multi_cb_add"

    def validate_config(self, config):
        """Check AQLM constraints."""

        # AQLM doesn't support transpose indexing (row-wise only)
        if config.get('enable_transpose', False):
            raise ValueError(
                "AQLM does not support transpose indexing. "
                "AQLM is row-wise quantization only."
            )

        # Check that we have at least 1 codebook
        if self.num_codebooks < 1:
            raise ValueError(f"AQLM requires at least 1 codebook, got {self.num_codebooks}")

        # Typically AQLM uses 2 codebooks, warn if different
        if self.num_codebooks != 2:
            print(f"Warning: AQLM typically uses 2 codebooks, you're using {self.num_codebooks}")
