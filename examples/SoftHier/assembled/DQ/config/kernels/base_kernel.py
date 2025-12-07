"""
Base class for SUMMA GEMM/GEMV kernels.
Common functions shared between GEMM and GEMV config.
"""


class BaseKernel:
    """Base config for SUMMA kernels."""

    def __init__(self):
        # Matrix dimensions (override in subclass)
        self.dtype = 'fp16'
        self.m_size = 512
        self.n_size = 512
        self.k_size = 512

        # Tile dimensions (override in subclass)
        self.m_tile = 128
        self.n_tile = 128
        self.k_tile = 128

        # SUMMA cluster configuration
        self.summa_scale_x = 4
        self.summa_scale_y = 4

        # SUMMA group configuration
        self.summa_group_number = 1
        self.summa_group_reduce = 0
        self.summa_group_splitk = 0
        self.summa_group_splitn = 0
        self.summa_group_gap_x = 0
        self.summa_group_gap_w = 0
        self.summa_group_gap_z = 0

        # Reshape options
        self.resha_x_from_enable = 0
        self.resha_z_to_enable = 0
        self.resha_x_from_m = 128
        self.resha_z_to_m = 2048

        # Numerical verification
        self.summa_numer = 1
        self.summa_numer_chunk = 8192

    def _dtype_nbytes(self):
        """Get number of bytes for dtype."""
        dtype_bytes = {
            'fp16': 2,
            'fp8': 1,
            'uint8': 1,
            'int8': 1,
            'fp32': 4,
        }
        if self.dtype not in dtype_bytes:
            raise ValueError(f"Unsupported dtype '{self.dtype}'")
        return dtype_bytes[self.dtype]

    def _validate_alignment(self):
        """Validate tile alignment with matrix dimensions."""
        m_block = self.summa_scale_y * self.m_tile
        n_block = self.summa_scale_x * self.n_tile

        if self.m_size % m_block != 0:
            raise ValueError(
                f"M dimension {self.m_size} must be a multiple of "
                f"summa_scale_y*m_tile ({m_block})"
            )
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

    def _setup_splitk(self):
        """Configure split-K if enabled."""
        if self.summa_group_splitk > 0:
            dtype_bytes = self._dtype_nbytes()
            total_k_partitions = self.summa_group_number * self.summa_group_splitk

            if total_k_partitions <= 1:
                raise ValueError("SplitK requires summa_group_number * summa_group_splitk > 1")

            if self.k_size % total_k_partitions != 0:
                raise ValueError(
                    f"K dimension {self.k_size} must be divisible by "
                    f"summa_group_number*summa_group_splitk ({total_k_partitions})"
                )

            k_chunk = self.k_size // total_k_partitions
            self.summa_group_gap_x = k_chunk * dtype_bytes
            self.summa_group_gap_w = k_chunk * self.n_size * dtype_bytes
            self.summa_group_gap_z = 0
            self.summa_group_reduce = 1
