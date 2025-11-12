
'''Download and slice pretrained VQ models from HuggingFace'''

import torch
import os
import argparse
import numpy as np
from vq_data_handler import VQDataHandler, VQConfig


dtype = torch.float16


def main():
    parser = argparse.ArgumentParser(description='Download VQ model from HuggingFace')
    parser.add_argument('--repo_id', type=str, required=True,
                       help='HF repo ID')
    parser.add_argument('--weight_file', type=str, default='model.safetensors',
                       help='Weight file name')
    parser.add_argument('--layer_prefix', type=str, required=True,
                       help='Layer prefix')
    parser.add_argument('--output_dir', type=str, default='./vq_cache',
                       help='Output dir')
    parser.add_argument('--k_size', type=int, default=512,
                       help='K dimension')
    parser.add_argument('--n_size', type=int, default=512,
                       help='N dimension')
    parser.add_argument('--num_codebooks', type=int, default=2,
                       help='Num codebooks')
    parser.add_argument('--nbits_per_codebook', type=int, default=8,
                       help='Bits per codebook')
    parser.add_argument('--group_size', type=int, default=8,
                       help='Group size')

    args = parser.parse_args()

    cfg = VQConfig(
        k_size=args.k_size,
        n_size=args.n_size,
        num_codebooks=args.num_codebooks,
        nbits_per_codebook=args.nbits_per_codebook,
        in_group_size=args.group_size,
        use_pretrained=True,
        repo_id=args.repo_id,
        weight_filename=args.weight_file,
        layer_prefix=args.layer_prefix,
    )

    handler = VQDataHandler(cfg, cache_dir=args.output_dir)

    print(f"\nDownloading from {args.repo_id}")
    print(f"  Layer: {args.layer_prefix}")
    print(f"  Target dimensions: K={args.k_size}, N={args.n_size}")

    handler.download_and_extract_layer(slice_k=args.k_size, slice_n=args.n_size)

    # verify
    W_hat = handler.dequantize()
    print(f"\nReconstructed weight shape: {W_hat.shape}")

    # save with 'dl' prefix (vs 'gen' for generated)
    prefix = "aqlm_dl"
    config_str = f"{cfg.num_codebooks}x{cfg.in_group_size}"
    dim_str = f"_dim{cfg.k_size}x{cfg.n_size}"

    np.save(os.path.join(args.output_dir, f"{prefix}{config_str}{dim_str}_cb.npy"),
            handler.codebooks)
    np.save(os.path.join(args.output_dir, f"{prefix}{config_str}{dim_str}_idx.npy"),
            handler.indices)
    np.save(os.path.join(args.output_dir, f"{prefix}{config_str}{dim_str}_scales.npy"),
            handler.scales)
    np.save(os.path.join(args.output_dir, f"{prefix}{config_str}{dim_str}_W_hat.npy"),
            W_hat)

    print(f"\n✓ Saved to {args.output_dir}")
    print(f"  Codebooks: {prefix}{config_str}{dim_str}_cb.npy")
    print(f"  Indices:   {prefix}{config_str}{dim_str}_idx.npy")
    print(f"  Scales:    {prefix}{config_str}{dim_str}_scales.npy")
    print(f"  W_hat:     {prefix}{config_str}{dim_str}_W_hat.npy")


if __name__ == '__main__':
    main()
