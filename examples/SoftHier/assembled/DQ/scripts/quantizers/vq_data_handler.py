'''VQ data handler for SoftHier'''

import os
import numpy as np
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
import torch

try:
    import aqlm as aqlm  # TODO: fix this import
    print("AQLM pip found[OK].")
    from aqlm import utils as aqlm_utils
    print("AQLM UTILS found[OK].")
    from aqlm import QuantizedWeight
    print("AQLM QuantizedWeight found[OK].")
    AQLM_AVAILABLE = True
except ImportError:
    AQLM_AVAILABLE = False
    print("AQLM library not found; dequantization will be unavailable.")

try:
    from safetensors import safe_open
    from huggingface_hub import hf_hub_download
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


@dataclass
class VQConfig:
    '''VQ configuration'''
    # matrix dims
    k_size: int  # rows of W
    n_size: int  # cols of W

    # VQ params
    num_codebooks: int = 2
    nbits_per_codebook: int = 8
    in_group_size: int = 8 # group size
    out_group_size: int = 1

    # codebook quantization
    codebook_value_nbits: int = 16  # 16 = no cb quant
    codebook_value_num_groups: int = 1
    scale_nbits: int = 0  # 0 = no scales

    # model source
    use_pretrained: bool = False
    repo_id: Optional[str] = None
    weight_filename: Optional[str] = None
    layer_prefix: Optional[str] = None

    @property
    def codebook_size(self) -> int:
        return 2 ** self.nbits_per_codebook

    @property
    def num_groups_per_row(self) -> int:
        return self.n_size // self.in_group_size

    @property
    def total_groups(self) -> int:
        return (self.k_size * self.n_size) // self.in_group_size

    def to_dict(self) -> Dict[str, Any]:
        return {
            'VQ_ENABLED': 1,
            'VQ_NUM_CBS': self.num_codebooks,
            'VQ_CB_NUM_CENTROIDS': self.codebook_size,
            'VQ_GROUP_SIZE': self.in_group_size,
            'VQ_NUM_GROUPS_PER_ROW': self.num_groups_per_row,
            'VQ_TOTAL_GROUPS': self.total_groups,
            'VQ_NUM_SCALES': self.k_size if self.scale_nbits > 0 else 0,
        }


class VQDataHandler:
    '''VQ data handler'''

    def __init__(self, config: VQConfig, cache_dir: str = './vq_cache'):
        self.cfg = config
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        # data storage
        self.codebooks: Optional[np.ndarray] = None
        self.indices: Optional[np.ndarray] = None
        self.scales: Optional[np.ndarray] = None
        self.W_reconstructed: Optional[np.ndarray] = None

    def download_and_extract_layer(self,
                                   slice_k: Tuple[int, int] = None,
                                   slice_n: Tuple[int, int] = None) -> None:
        '''Download model and extract layer'''
        if not self.cfg.use_pretrained:
            raise ValueError("need use_pretrained=True")

        print(f"Downloading from {self.cfg.repo_id}...")
        weights_path = hf_hub_download(
            repo_id=self.cfg.repo_id,
            filename=self.cfg.weight_filename
        )

        print(f"Loading weights from {weights_path}")

        # load safetensors
        tensors = {}
        with safe_open(weights_path, framework='numpy') as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)

        # extract layer
        layer_prefix = self.cfg.layer_prefix
        cb_key = f"{layer_prefix}.codebooks"
        codes_key = f"{layer_prefix}.codes"
        scales_key = f"{layer_prefix}.scales"

        for key in (cb_key, codes_key, scales_key):
            if key not in tensors:
                raise RuntimeError(f"key {key} not found")

        cb = tensors[cb_key]
        codes = tensors[codes_key]
        scales = tensors[scales_key]

        # slice if needed
        if slice_k is not None:
            k_end = slice_k
            codes = codes[0:k_end, ...]
            scales = scales[0:k_end, ...]

        if slice_n is not None:
            n_end = slice_n
            g_start = 0 // self.cfg.in_group_size
            g_end = n_end // self.cfg.in_group_size
            codes = codes[:, g_start:g_end, ...]

        self.codebooks = cb
        # convert int8 to uint8 for indexing
        if codes.dtype == np.int8:
            self.indices = codes.view(np.uint8)
            print("Converted int8 -> uint8")
        else:
            self.indices = codes
        self.scales = scales

        print("Extracted layer:")
        print(f"  Codebooks: {self.codebooks.shape}")
        print(f"  Indices: {self.indices.shape}, dtype: {self.indices.dtype}")
        print(f"  Scales: {self.scales.shape}")

    def load_from_files(self,
                       codebooks_path: str,
                       indices_path: str,
                       scales_path: str,
                       w_hat_path: Optional[str] = None) -> None:
        '''Load from numpy files'''
        self.codebooks = np.load(codebooks_path)
        idx = np.load(indices_path)
        # int8 -> uint8 for indexing (aqlm llama uses int8)
        if idx.dtype == np.int8:
            self.indices = idx.view(np.uint8)
            print("Converted int8 -> uint8")
        else:
            self.indices = idx
        self.scales = np.load(scales_path)

        if w_hat_path is not None:
            self.W_reconstructed = np.load(w_hat_path)

        print("Loaded VQ data:")
        print(f"  Codebooks: {self.codebooks.shape}")
        print(f"  Indices: {self.indices.shape}, dtype: {self.indices.dtype}")
        print(f"  Scales: {self.scales.shape}")

    def dequantize(self) -> np.ndarray:
        '''Dequantize: reconstruct W from codebooks and indices'''
        if self.codebooks is None or self.indices is None:
            raise RuntimeError("need to load data first")

        # detect format
        if self.codebooks.ndim == 4:
            print(f"AQLM format: {self.codebooks.shape}")
            return self._dequantize_aqlm()
        elif self.codebooks.ndim == 3:
            print(f"Generic VQ format: {self.codebooks.shape}")
            return self._dequantize_generic()
        else:
            raise ValueError(f"unsupported codebook format: {self.codebooks.ndim}D")

    def _dequantize_aqlm(self) -> np.ndarray:
        '''Dequantize AQLM format'''

        if AQLM_AVAILABLE:
            print("Using AQLM dequantization.")
            self.W_reconstructed = aqlm_utils._dequantize_weight(
                 codes=torch.from_numpy(self.indices),
                 codebooks=torch.from_numpy(self.codebooks),
                 scales=torch.from_numpy(self.scales) if self.scales is not None else None
             )
            return self.W_reconstructed
        else:
            # manual dequant
            num_cbs, num_centroids, out_gs, in_gs = self.codebooks.shape
            num_rows, num_groups, num_cbs_idx = self.indices.shape

            assert num_cbs == num_cbs_idx
            assert out_gs == 1, f"out_group_size must be 1, got {out_gs}"

            if self.scales is not None:
                scales = self.scales.reshape(num_rows)
            else:
                scales = np.ones(num_rows, dtype=np.float16)

            W = np.zeros((num_rows, num_groups * in_gs), dtype=np.float16)

            for i in range(num_rows):
                for j in range(num_groups):
                    vals = np.zeros(in_gs, dtype=np.float16)
                    for k in range(num_cbs):
                        code = self.indices[i, j, k]
                        centroid = self.codebooks[k, code, 0, :].astype(np.float16)
                        vals += centroid
                    vals *= scales[i]
                    W[i, j*in_gs:(j+1)*in_gs] = vals

            self.W_reconstructed = W.astype(np.float16)
            return self.W_reconstructed

    def _dequantize_generic(self) -> np.ndarray:
        '''Dequantize generic VQ'''
        if self.indices.ndim == 3:
            num_rows, num_groups, num_cbs = self.indices.shape
        elif self.indices.ndim == 2:
            raise NotImplementedError("packed indices not supported yet")
        else:
            raise ValueError(f"bad indices shape: {self.indices.shape}")

        gs = self.cfg.in_group_size
        W = np.zeros((num_rows, num_groups * gs), dtype=np.float16)

        for i in range(num_rows):
            for j in range(num_groups):
                vals = np.zeros(gs, dtype=np.float32)
                for k in range(num_cbs):
                    code = self.indices[i, j, k]
                    vals += self.codebooks[k, code, :]

                # scale
                if self.scales is not None and self.scales.size > 0:
                    scale = self.scales[i] if self.scales.ndim == 1 else self.scales[i].item()
                    vals *= scale

                W[i, j*gs:(j+1)*gs] = vals.astype(np.float16)

        self.W_reconstructed = W
        return W

    def prepare_for_preload(self, dtype: str = 'fp16') -> Dict[str, np.ndarray]:
        '''Prepare VQ data for preload'''
        if self.W_reconstructed is None:
            self.dequantize()

        # flatten codebooks
        cb_flat = np.concatenate([cb.reshape(-1) for cb in self.codebooks])

        # prepare indices - multiple formats
        if self.indices.ndim == 3:
            idx0 = self.indices[:, :, 0].astype(np.uint8)
            idx1 = self.indices[:, :, 1].astype(np.uint8) if self.indices.shape[2] > 1 else None

            # packed: idx1 << 8 | idx0
            if idx1 is not None:
                idx_packed = (idx1.astype(np.uint16) << 8) | idx0.astype(np.uint16)
            else:
                idx_packed = idx0.astype(np.uint16)

            # flattened
            idx_flat = np.concatenate([self.indices[:, :, i].astype(np.uint8)
                                      for i in range(self.indices.shape[2])])
        else:
            raise NotImplementedError("only 3D indices")

        # convert dtype
        if dtype == 'fp16':
            cb_typed = cb_flat.astype(np.float16).view(np.uint16)
            W_typed = self.W_reconstructed.astype(np.float16).view(np.uint16)
            scales_typed = self.scales.astype(np.float16).view(np.uint16) if self.scales.size > 0 else np.array([], dtype=np.uint16)
        else:
            raise ValueError(f"unsupported dtype: {dtype}")

        # Split codebooks into separate arrays (one per codebook)
        codebooks_split = []
        num_cbs = self.codebooks.shape[0]
        cb_size = self.codebooks[0].size
        for i in range(num_cbs):
            if dtype == 'fp16':
                cb_i = self.codebooks[i].astype(np.float16).view(np.uint16)
            else:
                cb_i = self.codebooks[i].astype(np.float16).view(np.uint16)
            codebooks_split.append(cb_i)

        # Split indices into separate arrays (one per codebook)
        # indices shape: (K, num_groups, num_codebooks)
        indices_split = []
        for i in range(num_cbs):
            idx_i = self.indices[:, :, i].astype(np.uint8)
            indices_split.append(idx_i)

        return {
            'codebooks': cb_typed,
            'codebooks_split': codebooks_split,  # List of individual codebook arrays
            'indices_packed': idx_packed,
            'indices_cb0': idx0,
            'indices_cb1': idx1 if idx1 is not None else np.array([], dtype=np.uint8),
            'indices_flattened': idx_flat,
            'indices_split': indices_split,  # List of individual indices arrays per codebook
            'scales': scales_typed,
            'W_reconstructed': W_typed,
        }

    def save_to_cache(self, prefix: str = 'vq_data') -> None:
        '''Save to cache'''
        cfg_str = f"_cb{self.cfg.num_codebooks}_B{self.cfg.nbits_per_codebook}_g{self.cfg.in_group_size}"
        dim_str = f"_K{self.cfg.k_size}_N{self.cfg.n_size}"

        np.save(os.path.join(self.cache_dir, f"{prefix}_codebooks{cfg_str}{dim_str}.npy"),
                self.codebooks)
        np.save(os.path.join(self.cache_dir, f"{prefix}_indices{cfg_str}{dim_str}.npy"),
                self.indices)
        np.save(os.path.join(self.cache_dir, f"{prefix}_scales{cfg_str}{dim_str}.npy"),
                self.scales)
        if self.W_reconstructed is not None:
            np.save(os.path.join(self.cache_dir, f"{prefix}_W_hat{cfg_str}{dim_str}.npy"),
                    self.W_reconstructed)

        print(f"Saved to {self.cache_dir}")


