"""
Base class for VQ algorithms.
Each algorithm (AQLM, VPTQ, etc.) implements this interface.
"""

from abc import ABC, abstractmethod


class VQAlgorithm(ABC):
    """Base interface for VQ algorithms."""

    def __init__(self):
        # Default params - override in subclass
        self.num_codebooks = 1
        self.group_size = 8
        self.nbits_per_cb = 8
        self.cb_size = 256
        self.use_scales = True
        self.enable_transpose = False

    @abstractmethod
    def get_num_codebooks(self):
        """How many codebooks? (2 for AQLM, 1 for VPTQ)"""
        pass

    @abstractmethod
    def get_use_scales(self):
        """Does this algorithm use per-row scales?"""
        pass

    @abstractmethod
    def get_buffer_requirements(self):
        """
        Returns dict of L1 buffer needs.
        Example: {'L1_CB': 2, 'L1_IDX1': 2, 'L1_IDX2': 2, 'L1_Scales': 2}
        """
        pass

    @abstractmethod
    def get_dequant_kernel(self):
        """Name of the C dequant kernel to use."""
        pass

    @abstractmethod
    def validate_config(self, config):
        """Check if config is valid for this algorithm. Raise ValueError if not."""
        pass

    def get_algorithm_name(self):
        """Returns 'aqlm' or 'vptq' etc."""
        return self.__class__.__name__.lower().replace('algorithm', '')

    def get_config_dict(self):
        """Export config as dict."""
        return {
            'num_codebooks': self.num_codebooks,
            'group_size': self.group_size,
            'nbits_per_cb': self.nbits_per_cb,
            'cb_size': self.cb_size,
            'use_scales': self.use_scales,
            'enable_transpose': self.enable_transpose
        }
