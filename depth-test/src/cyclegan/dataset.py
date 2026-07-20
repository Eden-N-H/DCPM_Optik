"""Unpaired Dataset for CycleGAN Sim-to-Real Translation."""

import os
import glob
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class UnpairedRoadDataset(Dataset):
    """Loads Unpaired Images for CycleGAN training.
    
    Domain A: Synthetic RGB + Defect Mask (Loaded from manifest.json)
    Domain B: Real-world reference RGB (Loaded from a flat directory)
    """

    def __init__(self, synth_root: str, real_root: str, split: str = 'train', size: int = 256):
        self.size = size
        self.synth_root = Path(synth_root)
        self.real_root = Path(real_root)

        # Load Domain A (Synthetic) from manifest
        manifest_path = self.synth_root / "manifest.json"
        self.samples_A = []
        if manifest_path.exists():
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            for s in manifest.get('samples', []):
                if s.get('split') == split:
                    self.samples_A.append(s)
        else:
            raise FileNotFoundError(f"Manifest not found: {manifest_path}. Generate data first.")

        # Load Domain B (Real) from flat directory
        self.samples_B = []
        valid_exts = ('*.jpg', '*.png', '*.jpeg')
        for ext in valid_exts:
            self.samples_B.extend(glob.glob(os.path.join(self.real_root, '**', ext), recursive=True))

        if not self.samples_B:
            raise ValueError(f"No real images found in {self.real_root}")

        self.len_A = len(self.samples_A)
        self.len_B = len(self.samples_B)
        self.length = max(self.len_A, self.len_B)

    def _portable_path(self, old_path: str) -> str:
        """Fixes absolute paths in manifest when dataset is moved to Colab."""
        parts = Path(old_path).parts
        # Reconstruct path using the last 4 segments (split/view/scene/file)
        return str(self.synth_root / Path(*parts[-4:]))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Domain A (Synthetic)
        idx_A = idx % self.len_A
        sample_A = self.samples_A[idx_A]
        
        img_A_path = self._portable_path(sample_A['paths']['rgb'])
        seg_A_path = self._portable_path(sample_A['paths']['segmentation'])
        
        img_A = cv2.imread(img_A_path)
        if img_A is None:
            raise FileNotFoundError(f"Missing {img_A_path}")
        img_A = cv2.cvtColor(img_A, cv2.COLOR_BGR2RGB)
        
        seg_A = cv2.imread(seg_A_path, cv2.IMREAD_UNCHANGED)
        if seg_A is None:
            raise FileNotFoundError(f"Missing {seg_A_path}")
        if seg_A.ndim == 3: 
            seg_A = seg_A[:,:,0]

        # Binary Mask: 1 where defect exists (class > 1)
        mask_A = (seg_A > 1).astype(np.float32)

        # Domain B (Real)
        idx_B = random.randint(0, self.len_B - 1)
        img_B = cv2.imread(self.samples_B[idx_B])
        if img_B is None:
            img_B = np.zeros_like(img_A) # Fallback if file read fails
        img_B = cv2.cvtColor(img_B, cv2.COLOR_BGR2RGB)

        # Resize
        img_A = cv2.resize(img_A, (self.size, self.size))
        mask_A = cv2.resize(mask_A, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        img_B = cv2.resize(img_B, (self.size, self.size))

        # Normalize to [-1, 1]
        img_A = (img_A.astype(np.float32) / 127.5) - 1.0
        img_B = (img_B.astype(np.float32) / 127.5) - 1.0

        tensor_A = torch.from_numpy(img_A.transpose(2, 0, 1))
        tensor_B = torch.from_numpy(img_B.transpose(2, 0, 1))
        tensor_mask_A = torch.from_numpy(mask_A).unsqueeze(0)

        return {'A': tensor_A, 'B': tensor_B, 'mask_A': tensor_mask_A}
