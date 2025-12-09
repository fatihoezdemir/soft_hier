
"""
AQLM Quantizer for SoftHier VQ Kernel
"""

import torch
import torch.nn as nn
import logging
import argparse
from argparse import Namespace
import numpy as np
import pathlib
import sys

# Setup paths
ROOT = pathlib.Path(__file__).resolve().parents[2]
AQLM_DIR = ROOT / 'third_party' / 'aqlm'
sys.path.append(str(AQLM_DIR))
# ----- Deterministic flags -----

from src.aq import QuantizedWeight
from aq_engine import AQEngine

# Set up logger
logger = logging.getLogger('aqlm-quantizer')
if not logger.handlers:
    logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


def make_calib(in_features, B=512, *, device=None, dtype=None, seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    X = torch.randn(in_features, B, device=device, dtype=dtype, generator=g)
    return X


def recon_error(W, What, X=None, device=None, dtype=torch.float16):
    '''Calculate reconstruction error'''
    B = 512
    if X is None:
        X = make_calib(W.shape[1], B=B, device=device, dtype=dtype, seed=0)

    # weight space error
    diffW = What - W
    w_num = (diffW * diffW).sum()
    w_den = (W * W).sum() + 1e-12
    w_rel = w_num / w_den

    # output space error
    X = X.float()
    W = W.float()
    What = What.float()
    Y_ref = W @ X
    Y_hat = What @ X
    diffY = Y_hat - Y_ref
    y_num = (diffY * diffY).sum()
    y_den = (Y_ref * Y_ref).sum() + 1e-12
    y_rel = y_num / y_den

    return (w_num, w_den, w_rel), (y_num, y_den, y_rel), X


def quantize_weight(k_size, n_size, num_codebooks=2, nbits_per_codebook=8,
                   in_group_size=8, out_group_size=1, max_epochs=10,
                   device='cpu', seed=0):
    '''Quantize random weight matrix using AQLM'''

    # set seeds
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if device == 'auto':
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    dtype = torch.float16
    dtype_acc = torch.float32

    print(f"Quantizing {k_size}x{n_size} weight matrix")
    logger.info(f"Device: {device}, Dtype: {dtype}")
    logger.info(f"VQ Config: {num_codebooks} codebooks x {2**nbits_per_codebook} centroids, group_size={in_group_size}")

    # create weight matrix (K x N for GEMM: X @ W where X is M x K)
    # using same init as gemm_preload.py - safe range for fp8/fp16
    g = torch.Generator(device=device).manual_seed(seed)
    W = torch.randn(k_size, n_size, device=device, dtype=torch.float32, generator=g) * 0.1 + 0.0625 
    W = W.to(dtype)
    logger.info(f"Created weight matrix: {W.shape} (range: [{W.min().item():.4f}, {W.max().item():.4f}], mean: {W.mean().item():.4f}, std: {W.std().item():.4f})")

    # AQLM expects nn.Linear (stores weights transposed)
    layer = nn.Linear(n_size, k_size, bias=False, device=device, dtype=dtype)
    layer.weight.data = W.clone()

    aq_engine = AQEngine(layer, accumulator_dtype=dtype_acc)

    # generate dummy inputs for XTX computation
    num_samples = 64
    seq_len = 128
    dummy_inputs = torch.randn(num_samples, seq_len, n_size, device=device, dtype=dtype)

    logger.info("Adding input batches to compute XTX...")
    for i in range(0, num_samples, 16):
        batch = dummy_inputs[i:i+16].reshape(-1, n_size)
        aq_engine.add_batch(batch)

    # quantization config
    args = Namespace(
        devices=[device],
        out_group_size=out_group_size,
        in_group_size=in_group_size,
        num_codebooks=num_codebooks,
        nbits_per_codebook=nbits_per_codebook,
        codebook_value_nbits=16,  # no codebook quantization
        codebook_value_num_groups=1,
        scale_nbits=0,  # no scales
        lr=1e-4,
        max_epochs=max_epochs,
        steps_per_epoch=100,
        beam_size=8,
        init_max_iter=100,
        init_max_points_per_centroid=None,
        print_frequency=50,
        relative_mse_tolerance=None,
    )

    logger.info("Starting quantization...")
    quantized_weight = aq_engine.quantize(args=args, verbose=True)

    print("\n--- Quantization Results ---")
    logger.info(f"Codes shape: {quantized_weight.get_codes().shape}")
    logger.info(f"Codebooks shape: {quantized_weight.get_codebooks().squeeze().shape}")
    logger.info(f"Codebooks dtype: {quantized_weight.get_codebooks().squeeze().dtype}")
    if quantized_weight.scales is not None:
        logger.info(f"Scales shape: {quantized_weight.get_scales().squeeze().shape}")

    # dequantize
    What = quantized_weight()
    logger.info(f"Reconstructed weight shape: {What.shape}")

    # reconstruction error of float32 codebooks
    (w_num, w_den, w_rel), (y_num, y_den, y_rel), _ = recon_error(W, What, device=device, dtype=dtype)
    print("\n--- Reconstruction Error ---")
    logger.info(f"Weight-space  squared: {w_num.item():.6f}  normalized: {w_rel.item():.6f}")
    logger.info(f"Output-space  squared: {y_num.item():.6f}  normalized: {y_rel.item():.6f}")

    bits_per_param = quantized_weight.estimate_nbits_per_parameter()
    logger.info(f"\nBits per parameter: {bits_per_param:.2f}")

    # extract components
    codes = quantized_weight.get_codes()
    assert codes.min().item() >= 0 and codes.max().item() < 2**nbits_per_codebook
    codes_u8 = codes.to(torch.uint8)

    codebooks = quantized_weight.get_codebooks().squeeze()
    cb = codebooks.to(torch.float16)

    scales = quantized_weight.get_scales().squeeze() if quantized_weight.scales is not None else None
    if scales is not None:
        scales_fp16 = scales.to(torch.float16)
    else:
        scales_fp16 = torch.ones(k_size, dtype=torch.float16, device=device)

    logger.info("\n--- Component Details ---")
    logger.info(f"Codes dtype: {codes_u8.dtype}, shape: {codes_u8.shape}")
    logger.info(f"Codebooks dtype: {cb.dtype}, shape: {cb.shape}")
    logger.info(f"Scales dtype: {scales_fp16.dtype}, shape: {scales_fp16.shape}")

    # compression ratio
    original_size = W.numel() * 32
    compressed_size = bits_per_param * W.numel()
    compression_ratio = original_size / compressed_size
    print(f"\nCompression ratio: {compression_ratio:.2f}x")

    return {
        'W_original': W.detach().cpu().numpy(),
        'W_reconstructed': What.detach().cpu().numpy(),
        'codes': codes_u8.detach().cpu().numpy(),
        'codebooks': cb.detach().cpu().numpy(),
        'scales': scales_fp16.detach().cpu().numpy(),
        'bits_per_param': bits_per_param,
        'compression_ratio': compression_ratio,
        'weight_error': (w_num.item(), w_den.item(), w_rel.item()),
        'output_error': (y_num.item(), y_den.item(), y_rel.item()),
    }


def save_quantized_data(results, output_dir, k_size, n_size, num_codebooks,
                       nbits_per_codebook, in_group_size):
    '''Save quantized data to files'''
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # use 'gen' prefix to distinguish from downloaded weights ('dl')
    source_prefix = "aqlm_gen"
    config_str = f"{num_codebooks}x{in_group_size}"
    dim_str = f"_dim{k_size}x{n_size}"

    cb_path = output_dir / f"{source_prefix}{config_str}{dim_str}_cb.npy"
    idx_path = output_dir / f"{source_prefix}{config_str}{dim_str}_idx.npy"
    scales_path = output_dir / f"{source_prefix}{config_str}{dim_str}_scales.npy"
    w_hat_path = output_dir / f"{source_prefix}{config_str}{dim_str}_W_hat.npy"
    w_orig_path = output_dir / f"{source_prefix}{config_str}{dim_str}_W_orig.npy"

    np.save(cb_path, results['codebooks'])
    np.save(idx_path, results['codes'])
    np.save(scales_path, results['scales'])
    np.save(w_hat_path, results['W_reconstructed'])
    np.save(w_orig_path, results['W_original'])

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
        description='AQLM Quantizer for SoftHier VQ Kernel',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--k_size', type=int, default=512,
                       help='Number of output features (rows)')
    parser.add_argument('--n_size', type=int, default=512,
                       help='Number of input features (cols)')

    parser.add_argument('--num_codebooks', type=int, default=2,
                       help='Number of codebooks')
    parser.add_argument('--nbits_per_codebook', type=int, default=8,
                       help='Bits per codebook')
    parser.add_argument('--in_group_size', type=int, default=8,
                       help='Group size')
    parser.add_argument('--out_group_size', type=int, default=1,
                       help='Output group size')

    parser.add_argument('--max_epochs', type=int, default=10,
                       help='Training epochs')
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
    print("AQLM Quantizer for SoftHier VQ Kernel")
    print("="*80)
    logger.info(f"Matrix dimensions: K={args.k_size}, N={args.n_size}")
    logger.info(f"VQ config: {args.num_codebooks} codebooks, {2**args.nbits_per_codebook} centroids, group_size={args.in_group_size}")
    print(f"Output directory: {args.output_dir}")
    print("="*80)

    # run quantization
    results = quantize_weight(
        k_size=args.k_size,
        n_size=args.n_size,
        num_codebooks=args.num_codebooks,
        nbits_per_codebook=args.nbits_per_codebook,
        in_group_size=args.in_group_size,
        out_group_size=args.out_group_size,
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
        args.nbits_per_codebook,
        args.in_group_size
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
