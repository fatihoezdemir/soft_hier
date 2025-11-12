

"""
Vector Quantization Configuration Utilities
Helpers for generating VQ-related C header definitions
"""
# script_dir = os.path.dirname(os.path.abspath(__file__))
# flex_utils_path = os.path.join(script_dir, '../../../../../../soft_hier/flex_cluster_utilities/')
# sys.path.append(flex_utils_path)
# from flex_libfp8 import write_matrix_to_header, generate_sparse_fp8_matrix, generate_fp8_matrix, extract_nm_sparsity, write_index_to_header_multi_format,write_indices_u16

from typing import List
import numpy as np

def generate_vq_defines(gemm, header_prefix: str = "GEMM") -> List[str]:
    """
    Generate C preprocessor defines for VQ configuration

    Args:
        gemm: SummaGEMM configuration object
        header_prefix: Prefix for C defines (default: "GEMM")

    Returns:
        List of C #define strings
    """
    if not hasattr(gemm, 'vq_enabled') or not gemm.vq_enabled:
        return []

    defines = []

    # Enable flag
    defines.append(f"#define {header_prefix}_VQ_ENABLED 1")

    # Algorithm type
    if hasattr(gemm, 'vq_algorithm'):
        defines.append(f"#define {header_prefix}_VQ_ALGORITHM_{gemm.vq_algorithm.upper()}")

    # Core VQ parameters
    defines.append(f"#define {header_prefix}_VQ_NUM_CBS ((uint64_t){gemm.vq_num_cb})")
    defines.append(f"#define {header_prefix}_VQ_NBITS_PER_CB ((uint64_t){gemm.vq_nbits_per_cb})")
    defines.append(f"#define {header_prefix}_VQ_GROUP_SIZE ((uint64_t){gemm.vq_group_size})")
    defines.append(f"#define {header_prefix}_VQ_CB_SIZE ((uint64_t){gemm.vq_cb_size})")

    # Derived parameters
    num_groups_per_row = gemm.n_size // gemm.vq_group_size
    total_groups = gemm.k_size * num_groups_per_row

    defines.append(f"#define {header_prefix}_VQ_NUM_GROUPS_PER_ROW ((uint64_t){num_groups_per_row})")
    defines.append(f"#define {header_prefix}_VQ_TOTAL_GROUPS ((uint64_t){total_groups})")

    # Tile-level parameters (for cluster-level computation)
    if hasattr(gemm, 'vq_num_groups_per_row_tile'):
        defines.append(f"#define {header_prefix}_VQ_NUM_GROUPS_PER_ROW_TILE ((uint64_t){gemm.vq_num_groups_per_row_tile})")
    else:
        num_groups_per_row_tile = gemm.n_tile // gemm.vq_group_size
        defines.append(f"#define {header_prefix}_VQ_NUM_GROUPS_PER_ROW_TILE ((uint64_t){num_groups_per_row_tile})")

    # Scales (if used)
    if hasattr(gemm, 'vq_use_scales') and gemm.vq_use_scales:
        defines.append(f"#define {header_prefix}_VQ_USE_SCALES 1")
        defines.append(f"#define {header_prefix}_VQ_NUM_SCALES ((uint64_t){gemm.k_size})")
    else:
        defines.append(f"#define {header_prefix}_VQ_NUM_SCALES ((uint64_t)0)")

    # Codebook storage format
    if hasattr(gemm, 'vq_codebook_format'):
        defines.append(f"#define {header_prefix}_VQ_CODEBOOK_FORMAT_{gemm.vq_codebook_format.upper()}")

    # Index storage format
    if hasattr(gemm, 'vq_index_format'):
        defines.append(f"#define {header_prefix}_VQ_INDEX_FORMAT_{gemm.vq_index_format.upper()}")

    return defines


def validate_vq_config(gemm) -> List[str]:
    """
    Validate VQ configuration consistency

    Args:
        gemm: SummaGEMM configuration object

    Returns:
        List of error messages (empty if valid)
    """
    errors = []

    if not hasattr(gemm, 'vq_enabled') or not gemm.vq_enabled:
        return errors  # Not using VQ, skip validation

    # Check matrix dimensions are divisible by group size
    if gemm.n_size % gemm.vq_group_size != 0:
        errors.append(f"N size ({gemm.n_size}) must be divisible by VQ group size ({gemm.vq_group_size})")

    if gemm.n_tile % gemm.vq_group_size != 0:
        errors.append(f"N tile ({gemm.n_tile}) must be divisible by VQ group size ({gemm.vq_group_size})")

    # Check codebook size matches bits per codebook
    expected_cb_size = 2 ** gemm.vq_nbits_per_cb
    if gemm.vq_cb_size != expected_cb_size:
        errors.append(f"VQ codebook size ({gemm.vq_cb_size}) doesn't match 2^nbits ({expected_cb_size})")

    # Check K dimension matches N dimension (for quantized weight matrix)
    if hasattr(gemm, 'k_size') and gemm.k_size != gemm.n_size:
        # This is actually OK for X @ W where W is (K, N) and quantized
        # But worth noting
        pass

    # Validate tile alignment
    if hasattr(gemm, 'vq_num_groups_per_row_tile'):
        expected = gemm.n_tile // gemm.vq_group_size
        if gemm.vq_num_groups_per_row_tile != expected:
            errors.append(f"VQ groups per row tile ({gemm.vq_num_groups_per_row_tile}) doesn't match n_tile/group_size ({expected})")

    return errors


def generate_vq_data_header(output_path: str,
                            codebooks: 'np.ndarray',
                            indices_packed: 'np.ndarray',
                            indices_cb0: 'np.ndarray',
                            indices_cb1: 'np.ndarray',
                            scales: 'np.ndarray',
                            dtype: str = 'fp16',
                            header_guard: str = "VQ_DATA_H") -> None:
    """
    Generate C header file with VQ data (for linker-script based data loading)
    """
    import numpy as np

    with open(output_path, 'w') as f:
        f.write('// This file is generated by VQ data handler\n\n')
        f.write(f'#ifndef {header_guard}\n')
        f.write(f'#define {header_guard}\n\n')
        f.write('#include <stdint.h>\n\n')

        # Write codebooks
        f.write(f'// Codebooks ({codebooks.shape[0]} elements)\n')
        if dtype == 'fp16':
            write_array_uint16(f, 'vq_codebooks_fp16', codebooks)
        else:
            write_array_uint8(f, 'vq_codebooks_fp8', codebooks)

        # Write packed indices
        f.write(f'\n// Packed indices ({indices_packed.shape[0]}x{indices_packed.shape[1]})\n')
        write_array_uint16(f, 'vq_indices_packed', indices_packed.flatten())

        # Write separate codebook indices
        f.write(f'\n// Codebook 0 indices ({indices_cb0.shape[0]}x{indices_cb0.shape[1]})\n')
        write_array_uint8(f, 'vq_indices_cb0', indices_cb0.flatten())

        if indices_cb1 is not None and indices_cb1.size > 0:
            f.write(f'\n// Codebook 1 indices ({indices_cb1.shape[0]}x{indices_cb1.shape[1]})\n')
            write_array_uint8(f, 'vq_indices_cb1', indices_cb1.flatten())

        # Write scales if present
        if scales is not None and scales.size > 0:
            f.write(f'\n// Scales ({scales.shape[0]} elements)\n')
            if dtype == 'fp16':
                write_array_uint16(f, 'vq_scales_fp16', scales.flatten())
            else:
                write_array_uint8(f, 'vq_scales_fp8', scales.flatten())

        f.write(f'\n#endif // {header_guard}\n')


def write_array_uint8(f, name: str, arr: 'np.ndarray') -> None:
    """Write uint8 array to C header"""
    f.write(f'const uint8_t {name}[{arr.size}] = {{\n')
    for i in range(0, arr.size, 16):
        chunk = arr[i:i+16]
        f.write('    ' + ', '.join(f'0x{x:02x}' for x in chunk))
        if i + 16 < arr.size:
            f.write(',\n')
        else:
            f.write('\n')
    f.write('};\n')


def write_array_uint16(f, name: str, arr: 'np.ndarray') -> None:
    """Write uint16 array to C header"""
    f.write(f'const uint16_t {name}[{arr.size}] = {{\n')
    for i in range(0, arr.size, 8):
        chunk = arr[i:i+8]
        f.write('    ' + ', '.join(f'0x{x:04x}' for x in chunk))
        if i + 8 < arr.size:
            f.write(',\n')
        else:
            f.write('\n')
    f.write('};\n')