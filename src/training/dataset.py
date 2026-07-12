"""PyTorch Dataset for road quality multi-task training."""
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, Optional, Tuple
import cv2


# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class RoadQualityDataset(Dataset):
    """Multi-task road quality dataset.

    Directory structure expected:
        root/
          split/  (train, val, test)
            rgb/         - RGB images (PNG)
            depth/       - Depth maps (16-bit PNG, millimeters)
            segmentation/ - Segmentation masks (PNG, class IDs)
            severity/    - Severity maps (NPY, float32)
            camera/      - Camera params (JSON with K and extrinsics)

    Returns dict with keys:
        'image': [3, H, W] normalized RGB tensor
        'depth': [1, H, W] depth tensor (meters)
        'segmentation': [H, W] long tensor of class IDs
        'severity': [1, H, W] severity tensor [0, 1]
        'camera_intrinsics': [4] tensor (fx, fy, cx, cy)
        'camera_extrinsics': [6] tensor (rodrigues3, translation3)
        'view_label': scalar long tensor (0=dashcam, 1=drone)
    """

    def __init__(self, root: str, split: str = 'train', crop_size: int = 480):
        """
        Args:
            root: Root dataset directory
            split: One of 'train', 'val', 'test'
            crop_size: Random crop size for training augmentation
        """
        self.root = Path(root)
        self.split = split
        self.crop_size = crop_size
        self.is_train = (split == 'train')

        # Discover samples
        self.rgb_dir = self.root / split / 'rgb'
        self.depth_dir = self.root / split / 'depth'
        self.seg_dir = self.root / split / 'segmentation'
        self.severity_dir = self.root / split / 'severity'
        self.camera_dir = self.root / split / 'camera'

        # List all sample IDs from RGB directory
        if self.rgb_dir.exists():
            self.sample_ids = sorted([
                p.stem for p in self.rgb_dir.glob('*.png')
            ])
        else:
            self.sample_ids = []

        # Augmentation parameters
        self.color_jitter = {
            'brightness': 0.2,
            'contrast': 0.2,
            'saturation': 0.1,
            'hue': 0.05,
        }

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample_id = self.sample_ids[idx]

        # Load data
        rgb = self._load_rgb(sample_id)
        depth = self._load_depth(sample_id)
        seg = self._load_segmentation(sample_id)
        severity = self._load_severity(sample_id)
        camera_params = self._load_camera(sample_id)

        # Apply augmentation
        if self.is_train:
            rgb, depth, seg, severity = self._train_augment(rgb, depth, seg, severity)

        # Normalize image
        image = self._normalize_image(rgb)

        # Convert depth from mm to meters
        depth_meters = depth.astype(np.float32) / 1000.0

        # Extract camera parameters
        intrinsics = camera_params['intrinsics']  # [4] fx, fy, cx, cy
        extrinsics = camera_params['extrinsics']  # [6] rodrigues + translation
        view_label = camera_params['view_label']  # 0 or 1

        # Convert to tensors
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()  # [3, H, W]
        depth_tensor = torch.from_numpy(depth_meters[np.newaxis]).float()  # [1, H, W]
        seg_tensor = torch.from_numpy(seg).long()  # [H, W]
        severity_tensor = torch.from_numpy(severity[np.newaxis]).float()  # [1, H, W]
        intrinsics_tensor = torch.from_numpy(np.array(intrinsics, dtype=np.float32))  # [4]
        extrinsics_tensor = torch.from_numpy(np.array(extrinsics, dtype=np.float32))  # [6]
        view_tensor = torch.tensor(view_label, dtype=torch.long)

        return {
            'image': image_tensor,
            'depth': depth_tensor,
            'segmentation': seg_tensor,
            'severity': severity_tensor,
            'camera_intrinsics': intrinsics_tensor,
            'camera_extrinsics': extrinsics_tensor,
            'view_label': view_tensor,
        }

    def _load_rgb(self, sample_id: str) -> np.ndarray:
        """Load RGB image as [H, W, 3] uint8 array."""
        path = self.rgb_dir / f"{sample_id}.png"
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def _load_depth(self, sample_id: str) -> np.ndarray:
        """Load depth map as [H, W] uint16 array (millimeters)."""
        path = self.depth_dir / f"{sample_id}.png"
        depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise FileNotFoundError(f"Depth map not found: {path}")
        return depth.astype(np.float32)

    def _load_segmentation(self, sample_id: str) -> np.ndarray:
        """Load segmentation mask as [H, W] int array."""
        path = self.seg_dir / f"{sample_id}.png"
        seg = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if seg is None:
            raise FileNotFoundError(f"Segmentation mask not found: {path}")
        # If loaded as color, take first channel
        if seg.ndim == 3:
            seg = seg[:, :, 0]
        return seg.astype(np.int32)

    def _load_severity(self, sample_id: str) -> np.ndarray:
        """Load severity map as [H, W] float32 array."""
        path = self.severity_dir / f"{sample_id}.npy"
        severity = np.load(str(path)).astype(np.float32)
        return severity

    def _load_camera(self, sample_id: str) -> Dict:
        """Load camera parameters from JSON.

        Expected JSON format:
        {
            "K": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
            "extrinsics": [r1, r2, r3, t1, t2, t3],
            "view_type": "dashcam" or "drone"
        }
        """
        path = self.camera_dir / f"{sample_id}.json"
        with open(path, 'r') as f:
            data = json.load(f)

        K = np.array(data['K'], dtype=np.float64)
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

        extrinsics = np.array(data['extrinsics'], dtype=np.float64)
        view_label = 0 if data.get('view_type', 'dashcam') == 'dashcam' else 1

        return {
            'intrinsics': [fx, fy, cx, cy],
            'extrinsics': extrinsics.tolist(),
            'view_label': view_label,
        }

    def _train_augment(self, rgb: np.ndarray, depth: np.ndarray,
                       seg: np.ndarray, severity: np.ndarray) -> Tuple:
        """Apply training augmentation.

        Geometric transforms applied consistently to RGB + all label maps:
        1. Random horizontal flip
        2. Random rotation ±10°
        3. Random crop 480x480
        Color jitter applied only to RGB.
        """
        H, W = rgb.shape[:2]

        # 1. Random horizontal flip
        if np.random.random() < 0.5:
            rgb = np.flip(rgb, axis=1).copy()
            depth = np.flip(depth, axis=1).copy()
            seg = np.flip(seg, axis=1).copy()
            severity = np.flip(severity, axis=1).copy()

        # 2. Random rotation ±10°
        angle = np.random.uniform(-10, 10)
        if abs(angle) > 0.5:
            center = (W / 2, H / 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rgb = cv2.warpAffine(rgb, M, (W, H), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REFLECT_101)
            depth = cv2.warpAffine(depth, M, (W, H), flags=cv2.INTER_NEAREST,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            seg = cv2.warpAffine(seg.astype(np.float32), M, (W, H),
                                 flags=cv2.INTER_NEAREST,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(np.int32)
            severity = cv2.warpAffine(severity, M, (W, H), flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        # 3. Random crop
        H, W = rgb.shape[:2]
        crop_h, crop_w = self.crop_size, self.crop_size
        if H > crop_h and W > crop_w:
            top = np.random.randint(0, H - crop_h)
            left = np.random.randint(0, W - crop_w)
            rgb = rgb[top:top + crop_h, left:left + crop_w]
            depth = depth[top:top + crop_h, left:left + crop_w]
            seg = seg[top:top + crop_h, left:left + crop_w]
            severity = severity[top:top + crop_h, left:left + crop_w]
        elif H != crop_h or W != crop_w:
            # Resize if image is smaller than crop size
            rgb = cv2.resize(rgb, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)
            depth = cv2.resize(depth, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST)
            seg = cv2.resize(seg.astype(np.float32), (crop_w, crop_h),
                             interpolation=cv2.INTER_NEAREST).astype(np.int32)
            severity = cv2.resize(severity, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)

        # 4. Color jitter (RGB only)
        rgb = self._apply_color_jitter(rgb)

        return rgb, depth, seg, severity

    def _apply_color_jitter(self, img: np.ndarray) -> np.ndarray:
        """Apply random color jitter to RGB image."""
        img = img.astype(np.float32)

        # Brightness
        brightness_factor = 1.0 + np.random.uniform(
            -self.color_jitter['brightness'], self.color_jitter['brightness'])
        img = img * brightness_factor

        # Contrast
        contrast_factor = 1.0 + np.random.uniform(
            -self.color_jitter['contrast'], self.color_jitter['contrast'])
        mean = img.mean()
        img = (img - mean) * contrast_factor + mean

        # Saturation
        saturation_factor = 1.0 + np.random.uniform(
            -self.color_jitter['saturation'], self.color_jitter['saturation'])
        gray = np.mean(img, axis=2, keepdims=True)
        img = (img - gray) * saturation_factor + gray

        # Hue (approximate by rotating in HSV)
        hue_factor = np.random.uniform(
            -self.color_jitter['hue'], self.color_jitter['hue'])
        if abs(hue_factor) > 0.001:
            img_uint8 = np.clip(img, 0, 255).astype(np.uint8)
            hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:, :, 0] = (hsv[:, :, 0] + hue_factor * 180) % 180
            img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)

        return np.clip(img, 0, 255).astype(np.uint8)

    def _normalize_image(self, img: np.ndarray) -> np.ndarray:
        """Apply ImageNet normalization."""
        img = img.astype(np.float32) / 255.0
        mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(1, 1, 3)
        std = np.array(IMAGENET_STD, dtype=np.float32).reshape(1, 1, 3)
        img = (img - mean) / std
        return img
