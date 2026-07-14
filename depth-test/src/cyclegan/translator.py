"""Batch Dataset Translator using Trained CycleGAN."""

import json
import shutil
import os
from pathlib import Path
import torch
import cv2
import numpy as np
from src.cyclegan.generator import ResNetGenerator


class DatasetTranslator:
    def __init__(self, ckpt_path: str, config: dict, device: torch.device):
        self.device = device
        self.size = config.get('cyclegan.input_size', 256)
        
        self.generator = ResNetGenerator(
            input_channels=config.get('cyclegan.input_nc', 4),
            output_channels=config.get('cyclegan.output_nc', 3),
            ngf=config.get('cyclegan.ngf', 64),
            n_residual_blocks=config.get('cyclegan.n_blocks', 9)
        ).to(device)
        self.generator.eval()
        
        # Load weights
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        self.generator.load_state_dict(ckpt['G_AB_state_dict'])

    def _portable_path(self, base_dir: Path, old_path: str) -> Path:
        """Fixes absolute paths in manifest when dataset is moved to Colab."""
        parts = Path(old_path).parts
        return base_dir / Path(*parts[-4:])

    @torch.no_grad()
    def translate_dataset(self, input_dir: str, output_dir: str):
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)

        print(f"Copying synthetic dataset from {input_dir} to {output_dir}...")
        if output_dir.exists():
            shutil.rmtree(output_dir)
        shutil.copytree(input_dir, output_dir)

        manifest_path = output_dir / "manifest.json"
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)

        manifest['root'] = str(output_dir)
        samples = manifest.get('samples', [])
        total = len(samples)
        print(f"Translating {total} synthetic images to real style...")

        for i, sample in enumerate(samples):
            rgb_path = self._portable_path(output_dir, sample['paths']['rgb'])
            seg_path = self._portable_path(output_dir, sample['paths']['segmentation'])

            img = cv2.imread(str(rgb_path))
            if img is None: continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            orig_h, orig_w = img.shape[:2]

            seg = cv2.imread(str(seg_path), cv2.IMREAD_UNCHANGED)
            if seg is None: continue
            if seg.ndim == 3: seg = seg[:,:,0]

            # Prepare for CycleGAN
            img_256 = cv2.resize(img, (self.size, self.size))
            img_norm = (img_256.astype(np.float32) / 127.5) - 1.0
            mask_256 = cv2.resize((seg > 1).astype(np.float32), (self.size, self.size), interpolation=cv2.INTER_NEAREST)

            t_img = torch.from_numpy(img_norm.transpose(2,0,1)).unsqueeze(0).to(self.device)
            t_mask = torch.from_numpy(mask_256).unsqueeze(0).unsqueeze(0).to(self.device)
            t_in = torch.cat([t_img, t_mask], dim=1)

            # Translate
            fake_B = self.generator(t_in)
            fake_B = (fake_B.squeeze(0).cpu().numpy().transpose(1,2,0) + 1.0) * 127.5
            fake_B = np.clip(fake_B, 0, 255).astype(np.uint8)

            # Resize back and overwrite the copied RGB image
            fake_B = cv2.resize(fake_B, (orig_w, orig_h))
            fake_B = cv2.cvtColor(fake_B, cv2.COLOR_RGB2BGR)

            cv2.imwrite(str(rgb_path), fake_B)

            if (i + 1) % 100 == 0:
                print(f"Translated {i+1}/{total} images.")

        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        print("Dataset translation complete!")
