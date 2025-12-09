"""
VPTQ Quantizer for SoftHier VQ Kernel
"""

import torch
import torch.nn as nn
import logging
import argparse
import math
import numpy as np
import pathlib #easier than f-based string
import sys

# Setup paths
ROOT = pathlib.Path(__file__).resolve().parents[2]

# Add VPTQ to path - try multiple locations
VPTQ_PATHS = [
    pathlib.Path('/home/fo/Desktop/master/VPTQ'),  # Main VPTQ location
    ROOT / 'third_party' / 'VPTQ',                 # Local third_party
]

vptq_found = False
for vptq_path in VPTQ_PATHS:
    if vptq_path.exists() and (vptq_path / 'vptq').exists():
        sys.path.insert(0, str(vptq_path))
        vptq_found = True
        print(f"Using VPTQ from: {vptq_path}")
        break


# Try to import VPTQ modules, do not import from pip vptq as the real quantizer is in the algorithm branch and not included in pip
# try:
from vptq.quantizer import NPVectorQuantizer, QuantizationArguments
from vptq.vptq import VPTQ
from vptq.layers.vqlinear import VQuantLinear


torch.manual_seed(0)

# Set up logger
logger = logging.getLogger('vptq-quantizer')
if not logger.handlers:
    logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


def make_calib(in_features, B=512, *, device=None, dtype=None, seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    X = torch.randn(in_features, B, device=device, dtype=dtype, generator=g)
    return X


def recon_error(W, What, X=None):
    '''Calculate reconstruction error'''
    B = 512
    device = W.device
    dtype = W.dtype

    if X is None:
        X = make_calib(W.shape[1], B=B, device=device, dtype=dtype, seed=42)

    # --- weight-space error ---
    diffW = What - W
    w_num = (diffW * diffW).sum()
    w_den = (W * W).sum() + 1e-12
    w_rel = w_num / w_den

    # --- output-space error ---
    Y_ref = W @ X
    Y_hat = What @ X
    diffY = Y_hat - Y_ref
    y_num = (diffY * diffY).sum()
    y_den = (Y_ref * Y_ref).sum() + 1e-12
    y_rel = y_num / y_den

    return (w_num, w_den, w_rel), (y_num, y_den, y_rel), X


def quantize_weight(k_size, n_size, group_size=6, num_centroids=4096,
                   max_epochs=100, device='cpu', seed=0):
    '''Quantize random weight matrix using VPTQ'''

    # set seeds
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if device == 'auto':
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    dtype = torch.float32  # VPTQ uses float32

    print(f"Quantizing {k_size}x{n_size} weight matrix")
    logger.info(f"Device: {device}, Dtype: {dtype}")
    logger.info(f"VQ Config: group_size={group_size}, num_centroids={num_centroids}")

    # Create weight matrix (K x N for GEMM: X @ W where X is M x K)
    # using same init as gemm_preload.py - safe range for fp8/fp16
    g = torch.Generator(device=device).manual_seed(seed)
    W = torch.randn(k_size, n_size, device=device, dtype=torch.float32, generator=g) * 0.1 + 0.0625
    W = W.to(dtype)
    logger.info(f"Created weight matrix: {W.shape} (range: [{W.min().item():.4f}, {W.max().item():.4f}], mean: {W.mean().item():.4f}, std: {W.std().item():.4f})")

    # Create calibration data and Hessian for loss computation
    Xcal = make_calib(n_size, B=512, device=device, dtype=dtype, seed=1)
    H = (Xcal @ Xcal.T) / Xcal.shape[1]
    H = H + 1e-4 * torch.eye(n_size, device=device)
    invH = torch.linalg.inv(H)

    # Create layer (VPTQ expects nn.Linear)
    layer = nn.Linear(n_size, k_size, bias=False, device=device, dtype=dtype)
    layer.weight.data = W.clone()  # Clone to keep original W intact

    # VPTQ configuration
    perm = None        # optional; leave None
    zero_idx = None    # optional; leave None
    vlen_outlier = -1  # -1 means no outlier handling
    num_centroids_res = -1  # -1 means no residual codebook

    logger.info(f"Doing Quantization with vector length {group_size}")
    qargs = QuantizationArguments(
        vector_lens=[vlen_outlier, group_size],
        num_centroids=[-1, num_centroids],
        num_res_centroids=[-1, num_centroids_res],
        npercent=0,                    # % of columns treated as outliers
        group_num=1,                   # number of columns per group
        group_size=-1,                 # width (columns) per group (-1=no extra PQ groups)
        kiter=max_epochs,
        ktol=1e-5,
        kmeans_mode="hessian",         # repo uses (diag H)-weighted KMeans
        enable_norm=False,
        norm_dim=0,
        enable_perm=False              # keep False to avoid permuting
    )

    quantizer = NPVectorQuantizer(
        layer_name="single.W",
        logger=logger,
        vector_lens=qargs.vector_lens,
        num_centroids=qargs.num_centroids,
        num_res_centroids=qargs.num_res_centroids,
        npercent=qargs.npercent,
        group_size=qargs.group_size,
        group_num=qargs.group_num,
        kmeans_mode=qargs.kmeans_mode,
        iter=qargs.kiter,
        tol=qargs.ktol,
        enable_norm=qargs.enable_norm,
        norm_dim=qargs.norm_dim,
        enable_perm=qargs.enable_perm,
        debug=False,
    )

    vptq = VPTQ(
        layer=layer,
        quantizer=quantizer,
        hessian=H,
        inv_hessian=invH,
        perm=perm,
        zero_idx=zero_idx,
        logger=logger,
        collect_act=False,
        layer_name="single.W",
        enable_perm=qargs.enable_perm,
        enable_norm=qargs.enable_norm,
        norm_dim=qargs.norm_dim,
        debug=False,
    )

    # Runs: builds codebooks (main/residual/outlier) and fills indices
    logger.info("Starting quantization...")
    vptq.fast_vector_quant()

    centroids = quantizer.centroids
    indices = quantizer.indices
    indices_sign = quantizer.indices_sign
    indices_scale = quantizer.indices_scale
    res_centroids = quantizer.res_centroids
    res_indices = quantizer.res_indices
    res_indices_sign = quantizer.res_indices_sign
    weight_scale = vptq.quantizer.weight_scale
    weight_bias = vptq.quantizer.weight_bias

    qlayer = VQuantLinear(
        in_features=n_size,
        out_features=k_size,
        vector_lens=qargs.vector_lens,
        num_centroids=qargs.num_centroids,
        num_res_centroids=qargs.num_res_centroids,
        group_num=quantizer.group_num,
        group_size=quantizer.group_size,
        outlier_size=quantizer.outlier_size,
        bias=None,
        enable_norm=qargs.enable_norm,
        norm_dim=qargs.norm_dim,
        enable_perm=qargs.enable_perm,
        vector_quant_dim='out',  # quantize along output dimension aka column by column
        device=device,
        dtype=dtype,
    )

    qlayer.init_parameters(
        centroids=centroids,
        indices=indices,
        res_centroids=res_centroids,
        res_indices=res_indices,
        weight_scale=weight_scale,
        weight_bias=weight_bias,
        indices_sign=indices_sign,
        indices_scale=indices_scale,
        res_indices_sign=res_indices_sign,
        bias=None,
        perm=perm,
        dtype=dtype,
    )

    What = qlayer.dequant()
    (weight_err, output_err, x) = recon_error(W, What)
    w_num, w_den, w_rel = weight_err
    y_num, y_den, y_rel = output_err

    # --- Need for dequantizer ---
    C_main   = quantizer.centroids        # dict: cidx -> [K, v]
    idx_main = quantizer.indices          # dict: cidx -> [blocks_per_col, num_cols_in_group]
    C_res    = quantizer.res_centroids    # dict (if residual enabled)
    idx_res  = quantizer.res_indices      # dict (if residual enabled)

    group_num = quantizer.group_num

    print("\n--- Quantization Results ---")
    # Get representative shapes for display
    if idx_main and 1 in idx_main and idx_main[1] is not None:
        print(f"Indices shape: {idx_main[1].shape}")
    if C_main and 1 in C_main and C_main[1] is not None:
        print(f"Main codebook shape: {C_main[1].shape}")
    if C_res and 1 in C_res and C_res[1] is not None:
        print(f"Residual codebook shape: {C_res[1].shape}")
    if idx_res and 1 in idx_res and idx_res[1] is not None:
        print(f"Residual indices shape: {idx_res[1].shape}")

    print(f"Reconstructed weight shape: {What.shape}")

    print("\n--- Reconstruction Error ---")
    print(f"Weight-space  squared: {w_num.item():.6f}  normalized: {w_rel.item():.6f}")
    print(f"Output-space  squared: {y_num.item():.6f}  normalized: {y_rel.item():.6f}")

    # Calculate bits per parameter
    num_parameters = k_size * n_size
    # Main indices: group_num groups × (n_size/group_size) × log2(num_centroids)
    main_bits = 0
    res_bits = 0
    if idx_main and 1 in idx_main and idx_main[1] is not None:
        # Each index uses log2(num_centroids) bits
        main_centroid_bits = math.log2(num_centroids)
        num_indices_per_group = idx_main[1].numel()
        main_bits = group_num * num_indices_per_group * main_centroid_bits

    if idx_res and 1 in idx_res and idx_res[1] is not None:
        # Residual indices
        res_centroid_bits = math.log2(num_centroids_res)
        num_res_indices_per_group = idx_res[1].numel()
        res_bits = group_num * num_res_indices_per_group * res_centroid_bits

    # Codebook storage
    codebook_bits = 0
    if C_main and 1 in C_main and C_main[1] is not None:
        # Main codebook: num_centroids × vector_length × 32 bits (float32)
        codebook_bits += C_main[1].numel() * 32 * group_num
    if C_res and 1 in C_res and C_res[1] is not None:
        # Residual codebook
        codebook_bits += C_res[1].numel() * 32 * group_num

    total_bits = main_bits + res_bits + codebook_bits
    bits_per_param = total_bits / num_parameters

    print(f"\nBits per parameter: {bits_per_param:.2f}")

    print("\n--- Component Details ---")
    print("Main codebooks:", {k: (v.shape if v is not None else None) for k, v in C_main.items()})
    print("Main codebooks dtype:", {k: (v.dtype if v is not None else None) for k, v in C_main.items()})

    print("Residual codebooks:", {k: (v.shape if v is not None else None) for k, v in C_res.items()})
    print("Residual idx shapes:", {k: (v.shape if v is not None else None) for k, v in idx_res.items()})
    print("Indices :", {k: (v if v is not None else None) for k, v in idx_main.items()})
    print("Indices  dtype:", {k: (v.dtype if v is not None else None) for k, v in idx_main.items()})
    print("Indices  shape:", {k: (v.shape if v is not None else None) for k, v in idx_main.items()})

    # Compute compression ratio
    original_size = num_parameters * 32  # 32 bits per float32
    compressed_size = total_bits
    compression_ratio = original_size / compressed_size
    print(f"\nCompression ratio: {compression_ratio:.2f}x")

    # Normalize dtypes for downstream handling: fp16 codebooks, uint16 indices
    codebooks_fp16 = {
        k: (v.detach().cpu().to(torch.float16).numpy() if v is not None else None)
        for k, v in C_main.items()
    }
    indices_u16 = {
        k: (v.detach().cpu().to(torch.int64).numpy().astype(np.uint16) if v is not None else None)
        for k, v in idx_main.items()
    }
    res_codebooks_fp16 = {
        k: (v.detach().cpu().to(torch.float16).numpy() if v is not None else None)
        for k, v in C_res.items()
    }
    res_indices_u16 = {
        k: (v.detach().cpu().to(torch.int64).numpy().astype(np.uint16) if v is not None else None)
        for k, v in idx_res.items()
    }

    return {
        'W_original': W.detach().cpu().numpy(),
        'W_reconstructed': What.detach().cpu().numpy(),
        'codebooks': codebooks_fp16,
        'indices': indices_u16,
        'res_codebooks': res_codebooks_fp16,
        'res_indices': res_indices_u16,
        'weight_scale': weight_scale.detach().cpu().numpy() if weight_scale is not None else None,
        'weight_bias': weight_bias.detach().cpu().numpy() if weight_bias is not None else None,
        'bits_per_param': bits_per_param,
        'compression_ratio': compression_ratio,
        'weight_error': (w_num.item(), w_den.item(), w_rel.item()),
        'output_error': (y_num.item(), y_den.item(), y_rel.item()),
    }


def save_quantized_data(results, output_dir, k_size, n_size, num_codebooks=1, group_size=6):
    '''Save quantized data to files'''
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # use 'gen' prefix to distinguish from downloaded weights ('dl')
    source_prefix = "vptq_gen"
    config_str = f"{num_codebooks}x{group_size}"
    dim_str = f"_dim{k_size}x{n_size}"

    cb_path = output_dir / f"{source_prefix}{config_str}{dim_str}_cb.npy"
    idx_path = output_dir / f"{source_prefix}{config_str}{dim_str}_idx.npy"
    scales_path = output_dir / f"{source_prefix}{config_str}{dim_str}_scales.npy"
    w_hat_path = output_dir / f"{source_prefix}{config_str}{dim_str}_W_hat.npy"
    w_orig_path = output_dir / f"{source_prefix}{config_str}{dim_str}_W_orig.npy"

    np.save(w_orig_path, results['W_original'])
    np.save(w_hat_path, results['W_reconstructed'])

    #  main codebook/indices in compact types (fp16 / uint16)
    main_cb = next((v for v in results['codebooks'].values() if v is not None), None)
    main_idx = next((v for v in results['indices'].values() if v is not None), None)
    if main_cb is None or main_idx is None:
        raise RuntimeError("Missing main codebook or indices in VPTQ results")

    np.save(cb_path, main_cb.astype(np.float16, copy=False))
    np.save(idx_path, main_idx.astype(np.uint16, copy=False))

    # Save residuals if present (kept separate for debugging)
    if results['res_codebooks']:
        for k, v in results['res_codebooks'].items():
            if v is not None:
                np.save(f'vptq_codebook_res_{k}.npy', v.astype(np.float16, copy=False))

    if results['res_indices']:
        for k, v in results['res_indices'].items():
            if v is not None:
                np.save(f'vptq_indices_res_{k}.npy', v.astype(np.uint16, copy=False))

    # Save scale and bias if present
    if results['weight_scale'] is not None:
        np.save('vptq_weight_scale.npy', results['weight_scale'])
    if results['weight_bias'] is not None:
        np.save('vptq_weight_bias.npy', results['weight_bias'])

    # VPTQ doesn't use per-row scales like AQLM, so save dummy scales (all ones)
    # for compatibility with vq handler ( todo change)
    dummy_scales = np.ones(k_size, dtype=np.float16)
    np.save(scales_path, dummy_scales)

    print("\n--- Saved Files ---")
    print(f"Codebooks:     {cb_path}")
    print(f"Indices:       {idx_path}")
    print(f"Scales:        {scales_path}")
    print(f"W_hat:         {w_hat_path}")
    print(f"W_original:    {w_orig_path}")

    return {
        'codebooks': str(cb_path),
        'indices': str(idx_path),
        'scales': str(scales_path),
        'w_hat': str(w_hat_path),
        'w_orig': str(w_orig_path),
    }

def main():
    parser = argparse.ArgumentParser(
        description='VPTQ Quantizer for SoftHier VQ Kernel',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--k_size', type=int, default=512,
                       help='Number of output features (rows)')
    parser.add_argument('--n_size', type=int, default=512,
                       help='Number of input features (cols)')

    parser.add_argument('--num_codebooks', type=int, default=1,
                       help='Number of codebooks (VPTQ uses 1)')
    parser.add_argument('--group_size', type=int, default=6,
                       help='Group size (6=4096 centroids, 8=65536 centroids)')
    parser.add_argument('--num_centroids', type=int, default=4096,
                       help='Number of centroids (codebook size)')

    parser.add_argument('--max_epochs', type=int, default=100,
                       help='K-means iterations')
    parser.add_argument('--seed', type=int, default=0,
                       help='Random seed')

    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda', 'cuda:0', 'cuda:1'],
                       help='Device')

    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (default: ../../vq_cache)')

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = str(pathlib.Path(__file__).resolve().parents[2] / 'vq_cache')

    print("="*80)
    print("VPTQ Quantizer for SoftHier VQ Kernel")
    print("="*80)
    logger.info(f"Matrix dimensions: K={args.k_size}, N={args.n_size}")
    logger.info(f"VQ config: {args.num_codebooks} codebook(s), group_size={args.group_size}, centroids={args.num_centroids}")
    print(f"Output directory: {args.output_dir}")
    print("="*80)

    # run quantization
    results = quantize_weight(
        k_size=args.k_size,
        n_size=args.n_size,
        group_size=args.group_size,
        num_centroids=args.num_centroids,
        max_epochs=args.max_epochs,
        device=args.device,
        seed=args.seed
    )

    # save
    save_quantized_data(
        results,
        args.output_dir,
        args.k_size,
        args.n_size,
        args.num_codebooks,
        args.group_size
    )

    print("\n" + "="*80)
    print("Quantization complete!")
    print(f"Files saved to: {args.output_dir}")
    logger.info(f"Compression ratio: {results['compression_ratio']:.2f}x")
    logger.info(f"Weight error (normalized): {results['weight_error'][2]:.6f}")
    logger.info(f"Output error (normalized): {results['output_error'][2]:.6f}")
    print("="*80)


if __name__ == '__main__':
    main()
