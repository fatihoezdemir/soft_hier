"""
Creating VQ algorithm instances.
"""

from .aqlm_algorithm import AQLMAlgorithm
from .gptvq_algorithm import GPTVQAlgorithm
from .vptq_algorithm import VPTQAlgorithm


def create_algorithm(algorithm_name, **kwargs):
    """
    Create a VQ algorithm.
    Args:
        algorithm_name: Name of algorithm ('aqlm', 'vptq', etc.)
        **kwargs: algorithm-specific parameters
    Returns:
        VQAlgorithm instance
    Example:
        alg = create_algorithm('aqlm')
        alg = create_algorithm('vptq', enable_transpose=True, cb_size=4096)
        alg = create_algorithm('gptvq', cb_size=256, group_size=4)
    """

    algorithm_name = algorithm_name.lower()

    if algorithm_name == 'aqlm':
        enable_transpose = kwargs.get('enable_transpose', False)
        num_codebooks = kwargs.get('num_codebooks', 2)
        cb_size = kwargs.get('cb_size', 256)
        return AQLMAlgorithm(num_codebooks=num_codebooks)

    elif algorithm_name == 'vptq':
        enable_transpose = kwargs.get('enable_transpose', False)
        cb_size = kwargs.get('cb_size', 4096)
        return VPTQAlgorithm(enable_transpose=enable_transpose, cb_size=cb_size)

    elif algorithm_name == 'gptvq':
        enable_transpose = kwargs.get('enable_transpose', False)
        cb_size = kwargs.get('cb_size', 256)
        group_size = kwargs.get('group_size', 4)
        return GPTVQAlgorithm(
            enable_transpose=enable_transpose,
            cb_size=cb_size,
            group_size=group_size,
        )

    else:#
        raise ValueError(
            f"Unknown VQ algorithm: {algorithm_name}. "
            f"Supported algorithms: 'aqlm', 'vptq', 'gptvq'"
        )
