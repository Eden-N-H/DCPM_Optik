"""End-to-End Pipeline Visualizer."""

import tempfile
from pathlib import Path
from typing import Dict, List, Optional
import cv2
import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use('Agg') # FIX: Force headless rendering to prevent Flask UI hangs
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from src.training.dataset import IMAGENET_MEAN, IMAGENET_STD
from src.reconstruction.pipeline import ReconstructionPipeline
from src.reconstruction.bev import DEFAULT_COLOR_MAP


class PipelineVisualizer:
    """Visualizes the end-to-end pipeline (Data -> CycleGAN -> MultiTask -> 3D BEV)."""

    def __init__(self, config, dataset, cyclegan_gen, multitask_model, device):
        self.config = config
        self.dataset = dataset
        self.cyclegan = cyclegan_gen.to(device).eval()
        self.multitask = multitask_model.to(device).eval()
        self.device = device
        
        # Setup colormap for segmentation
        colors = [DEFAULT_COLOR_MAP.get(i, (0, 0, 0)) for i in range(7)]
        self.seg_cmap = ListedColormap([[c[0]/255, c[1]/255, c[2]/255] for c in colors])

    @torch.no_grad()
    def visualize_samples(self, num_samples: int, output_dir: Path):
        """Run the pipeline on N samples and generate storyboard grids."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        num_samples = min(num_samples, len(self.dataset))
        
        for idx in range(num_samples):
            sample = self.dataset[idx]
            
            # --- 1. Prepare Ground Truth ---
            # Un-normalize RGB back to [0, 1]
            img_tensor = sample['image'].to(self.device)
            mean = torch.tensor(IMAGENET_MEAN, device=self.device).view(3, 1, 1)
            std = torch.tensor(IMAGENET_STD, device=self.device).view(3, 1, 1)
            gt_rgb = img_tensor * std + mean
            gt_rgb_np = gt_rgb.cpu().permute(1, 2, 0).numpy().clip(0, 1)
            
            gt_depth = sample['depth'].to(self.device)
            gt_seg = sample['segmentation'].to(self.device)
            gt_sev = sample['severity'].to(self.device)
            gt_intr = sample['camera_intrinsics'].to(self.device)
            gt_extr = sample['camera_extrinsics'].to(self.device)

            # --- 2. CycleGAN Translation ---
            # CycleGAN expects [1, 4, 256, 256] -> RGB [-1, 1] + Mask [0, 1]
            rgb_256 = F.interpolate(gt_rgb.unsqueeze(0), size=(256, 256), mode='bilinear')
            rgb_cg_in = rgb_256 * 2.0 - 1.0  # Scale to [-1, 1]
            
            # Mask: Any defect (class > 1)
            mask_256 = F.interpolate((gt_seg > 1).float().unsqueeze(0).unsqueeze(0), size=(256, 256), mode='nearest')
            cg_input = torch.cat([rgb_cg_in, mask_256], dim=1)
            
            cg_out = self.cyclegan(cg_input)  # [1, 3, 256, 256] in [-1, 1]
            translated_rgb_256 = (cg_out + 1.0) / 2.0  # Scale to [0, 1]
            
            # Resize translated RGB back to 512x512 for multitask
            translated_rgb = F.interpolate(translated_rgb_256, size=(512, 512), mode='bilinear')
            translated_rgb_np = translated_rgb[0].cpu().permute(1, 2, 0).numpy().clip(0, 1)

            # --- 3. Multi-Task Inference ---
            mt_input = (translated_rgb - mean) / std
            preds = self.multitask(mt_input)
            
            pred_seg = preds['segmentation'].argmax(dim=1)  # [1, 512, 512]
            pred_depth = preds['depth']                     # [1, 1, 512, 512]
            pred_sev = preds['severity']                    # [1, 1, 512, 512]
            pred_intr = preds['intrinsics']                 # [1, 4]
            pred_extr = preds['extrinsics']                 # [1, 6]

            # --- 4. 3D BEV Reconstruction ---
            gt_bev_img = self._generate_bev(gt_depth[0], gt_seg, gt_sev[0], gt_intr, gt_extr)
            # FIX: Ensure 2D arrays are passed to unprojector
            pred_bev_img = self._generate_bev(pred_depth[0, 0], pred_seg[0], pred_sev[0, 0], pred_intr[0], pred_extr[0])

            # --- 5. Error Maps & Overlays ---
            # Segmentation Overlay
            overlay = translated_rgb_np.copy()
            defect_mask = pred_seg[0].cpu().numpy() > 1
            overlay[defect_mask] = overlay[defect_mask] * 0.5 + np.array([1.0, 0.0, 0.0]) * 0.5
            
            # Seg Error (Green=TP, Red=FP, Blue=FN)
            gt_s = gt_seg.cpu().numpy()
            pr_s = pred_seg[0].cpu().numpy()
            seg_err = np.zeros((512, 512, 3))
            seg_err[(gt_s > 1) & (pr_s <= 1)] = [0, 0, 1]  # False Negative
            seg_err[(gt_s <= 1) & (pr_s > 1)] = [1, 0, 0]  # False Positive
            seg_err[(gt_s > 1) & (pr_s > 1)] = [0, 1, 0]   # True Positive

            # Depth/Severity Error Maps
            gt_d = gt_depth[0].cpu().numpy()
            pr_d = pred_depth[0, 0].cpu().numpy()
            depth_err = np.abs(gt_d - pr_d)
            
            gt_sv = gt_sev[0].cpu().numpy()
            pr_sv = pred_sev[0, 0].cpu().numpy()
            sev_err = np.abs(gt_sv - pr_sv)

            # --- 6. Matplotlib Grid ---
            self._plot_grid(
                output_dir / f"grid_{idx:03d}.png",
                gt_rgb_np, gt_s, gt_d, gt_sv, gt_bev_img,
                translated_rgb_np, pr_s, pr_d, pr_sv, pred_bev_img,
                overlay, seg_err, depth_err, sev_err
            )

    def _generate_bev(self, depth, seg, sev, intr, extr):
        """Helper to run the reconstruction pipeline and return the BEV image."""
        pipeline = ReconstructionPipeline(self.config.get('reconstruction', {}))
        
        preds = {
            'depth': depth.cpu().numpy(),
            'segmentation': seg.cpu().numpy(),
            'severity': sev.cpu().numpy(),
            'intrinsics': intr.cpu().numpy(),
            'extrinsics': extr.cpu().numpy()
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline.process_frame(preds)
            bev_path = pipeline.finalize(Path(tmpdir))
            if bev_path and bev_path.exists():
                img = cv2.imread(str(bev_path))
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                return np.zeros((512, 512, 3), dtype=np.uint8)

    def _plot_grid(self, save_path, 
                   gt_rgb, gt_seg, gt_depth, gt_sev, gt_bev,
                   pr_rgb, pr_seg, pr_depth, pr_sev, pr_bev,
                   overlay, err_seg, err_depth, err_sev):
        """Renders and saves the 3x5 storyboard grid."""
        fig, axes = plt.subplots(3, 5, figsize=(25, 15))
        
        # Row 1: Ground Truth
        axes[0, 0].imshow(gt_rgb); axes[0, 0].set_title("1. Synthetic RGB (GT)")
        axes[0, 1].imshow(gt_seg, cmap=self.seg_cmap, vmin=0, vmax=6); axes[0, 1].set_title("2. GT Segmentation")
        axes[0, 2].imshow(gt_depth, cmap='plasma'); axes[0, 2].set_title("3. GT Depth")
        axes[0, 3].imshow(gt_sev, cmap='hot'); axes[0, 3].set_title("4. GT Severity")
        axes[0, 4].imshow(gt_bev); axes[0, 4].set_title("5. GT BEV Map")
        
        # Row 2: Predictions
        axes[1, 0].imshow(pr_rgb); axes[1, 0].set_title("6. Translated RGB (CycleGAN)")
        axes[1, 1].imshow(pr_seg, cmap=self.seg_cmap, vmin=0, vmax=6); axes[1, 1].set_title("7. Predicted Segmentation")
        axes[1, 2].imshow(pr_depth, cmap='plasma'); axes[1, 2].set_title("8. Predicted Depth")
        axes[1, 3].imshow(pr_sev, cmap='hot'); axes[1, 3].set_title("9. Predicted Severity")
        axes[1, 4].imshow(pr_bev); axes[1, 4].set_title("10. Predicted BEV Map")
        
        # Row 3: Analysis
        axes[2, 0].imshow(overlay); axes[2, 0].set_title("11. Pred Defect Overlay")
        axes[2, 1].imshow(err_seg); axes[2, 1].set_title("12. Seg Error (G=TP, R=FP, B=FN)")
        axes[2, 2].imshow(err_depth, cmap='magma'); axes[2, 2].set_title("13. Absolute Depth Error")
        axes[2, 3].imshow(err_sev, cmap='magma'); axes[2, 3].set_title("14. Absolute Severity Error")
        
        axes[2, 4].axis('off')
        axes[2, 4].text(0.5, 0.5, "Pipeline Complete\nGenerated for Analysis", 
                        fontsize=16, ha='center', va='center')
        
        for ax in axes.flatten():
            if ax != axes[2, 4]:
                ax.axis('off')
                
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
