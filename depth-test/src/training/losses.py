"""Multi-task loss computation for road quality model."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class SSIMLoss(nn.Module):
    """Structural Similarity Index loss (1 - SSIM).

    Simple window-based SSIM computed from scratch.
    """

    def __init__(self, window_size: int = 11, channels: int = 1):
        super().__init__()
        self.window_size = window_size
        self.channels = channels
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

        # Create Gaussian window
        window = self._create_gaussian_window(window_size, channels)
        self.register_buffer('window', window)

    def _create_gaussian_window(self, window_size: int, channels: int) -> torch.Tensor:
        """Create a Gaussian kernel for SSIM computation."""
        sigma = 1.5
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        gauss = torch.exp(-coords ** 2 / (2 * sigma ** 2))
        gauss = gauss / gauss.sum()

        # 2D kernel
        kernel_2d = gauss.unsqueeze(1) @ gauss.unsqueeze(0)  # [W, W]
        kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)  # [1, 1, W, W]
        window = kernel_2d.expand(channels, 1, window_size, window_size).contiguous()
        return window

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute SSIM loss.

        Args:
            pred: [B, 1, H, W] predicted values
            target: [B, 1, H, W] target values

        Returns:
            Scalar SSIM loss (1 - SSIM)
        """
        channels = pred.size(1)
        pad = self.window_size // 2

        # Ensure window is on same device
        window = self.window
        if window.device != pred.device:
            window = window.to(pred.device)
        if window.dtype != pred.dtype:
            window = window.to(pred.dtype)

        mu_x = F.conv2d(pred, window, padding=pad, groups=channels)
        mu_y = F.conv2d(target, window, padding=pad, groups=channels)

        mu_x_sq = mu_x ** 2
        mu_y_sq = mu_y ** 2
        mu_xy = mu_x * mu_y

        sigma_x_sq = F.conv2d(pred ** 2, window, padding=pad, groups=channels) - mu_x_sq
        sigma_y_sq = F.conv2d(target ** 2, window, padding=pad, groups=channels) - mu_y_sq
        sigma_xy = F.conv2d(pred * target, window, padding=pad, groups=channels) - mu_xy

        # Clamp for numerical stability
        sigma_x_sq = sigma_x_sq.clamp(min=0)
        sigma_y_sq = sigma_y_sq.clamp(min=0)

        ssim_map = ((2 * mu_xy + self.C1) * (2 * sigma_xy + self.C2)) / \
                   ((mu_x_sq + mu_y_sq + self.C1) * (sigma_x_sq + sigma_y_sq + self.C2))

        return 1.0 - ssim_map.mean()


class GeodesicRotationLoss(nn.Module):
    """Geodesic loss between rotation matrices derived from Rodrigues vectors.

    Geodesic distance: arccos((trace(R1^T @ R2) - 1) / 2)
    """

    def forward(self, pred_rodrigues: torch.Tensor,
                target_rodrigues: torch.Tensor) -> torch.Tensor:
        """Compute geodesic rotation loss.

        Args:
            pred_rodrigues: [B, 3] predicted Rodrigues vectors
            target_rodrigues: [B, 3] target Rodrigues vectors

        Returns:
            Mean geodesic angle error in radians
        """
        R_pred = self._rodrigues_to_matrix(pred_rodrigues)
        R_target = self._rodrigues_to_matrix(target_rodrigues)

        # R_diff = R_pred^T @ R_target
        R_diff = torch.bmm(R_pred.transpose(1, 2), R_target)

        # Geodesic distance: arccos((trace(R_diff) - 1) / 2)
        trace = R_diff[:, 0, 0] + R_diff[:, 1, 1] + R_diff[:, 2, 2]
        cos_angle = (trace - 1.0) / 2.0
        
        # INCREASED CLAMP PADDING: Avoid NaN from edge cases inside torch.acos
        cos_angle = cos_angle.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        angle = torch.acos(cos_angle)

        return angle.mean()

    def _rodrigues_to_matrix(self, rodrigues: torch.Tensor) -> torch.Tensor:
        """Convert batch of Rodrigues vectors to rotation matrices.

        Args:
            rodrigues: [B, 3] Rodrigues rotation vectors

        Returns:
            [B, 3, 3] rotation matrices
        """
        batch_size = rodrigues.shape[0]
        
        # CRITICAL FIX: Add epsilon BEFORE taking square root to prevent NaN gradients 
        # evaluating at zero. torch.norm(rodrigues) is unsafe here.
        theta = torch.sqrt(torch.sum(rodrigues ** 2, dim=1, keepdim=True) + 1e-8)

        axis = rodrigues / theta  # [B, 3]

        # Skew-symmetric matrix
        zero = torch.zeros(batch_size, device=rodrigues.device, dtype=rodrigues.dtype)
        K = torch.stack([
            zero, -axis[:, 2], axis[:, 1],
            axis[:, 2], zero, -axis[:, 0],
            -axis[:, 1], axis[:, 0], zero
        ], dim=1).reshape(batch_size, 3, 3)

        # Rodrigues formula: R = I + sin(theta)*K + (1-cos(theta))*K^2
        theta_sq = theta.unsqueeze(-1)  # [B, 1, 1]
        eye = torch.eye(3, device=rodrigues.device, dtype=rodrigues.dtype).unsqueeze(0)
        R = eye + torch.sin(theta_sq) * K + (1 - torch.cos(theta_sq)) * torch.bmm(K, K)

        return R


class MultiTaskLoss(nn.Module):
    """Combined multi-task loss for road quality model.

    Components:
        - Segmentation: Cross-entropy with class weights
        - Depth: L1 + SSIM
        - Camera: L1 for intrinsics + geodesic for rotation
        - Domain adaptation: BCE for discriminator outputs

    Total = w_seg * L_seg + w_depth * L_depth + w_cam * L_cam + w_adv * L_adv
    """

    def __init__(self,
                 seg_weight: float = 1.5,
                 depth_weight: float = 1.0,
                 camera_weight: float = 0.3,
                 adv_weight: float = 0.1,
                 class_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.seg_weight = seg_weight
        self.depth_weight = depth_weight
        self.camera_weight = camera_weight
        self.adv_weight = adv_weight

        # Segmentation loss (ignore padding artifacts and broken indices safely)
        self.seg_loss = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)

        # Depth losses
        self.depth_l1 = nn.L1Loss()
        self.depth_ssim = SSIMLoss(window_size=11, channels=1)

        # Camera losses
        self.intrinsic_l1 = nn.L1Loss()
        self.rotation_geodesic = GeodesicRotationLoss()
        self.translation_l1 = nn.L1Loss()

        # Domain adaptation loss
        self.domain_bce = nn.BCEWithLogitsLoss()

    def forward(self, predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor],
                domain_labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Compute all loss components.

        Args:
            predictions: Model output dict with keys:
                'segmentation': [B, C, H, W] logits
                'depth': [B, 1, H, W] predicted depth
                'intrinsics': [B, 4] predicted intrinsics
                'extrinsics': [B, 6] predicted extrinsics
                'domain_pred': Optional dict with 'feat_pred' [B,1] and 'logit_pred' [B,1]
            targets: Ground truth dict with keys:
                'segmentation': [B, H, W] class labels
                'depth': [B, 1, H, W] ground truth depth
                'camera_intrinsics': [B, 4] GT intrinsics
                'camera_extrinsics': [B, 6] GT extrinsics
            domain_labels: Optional [B] binary domain labels (0=synthetic, 1=real)

        Returns:
            Dict with 'total' and individual loss components
        """
        losses = {}

        # Segmentation loss
        losses['seg'] = self.seg_loss(predictions['segmentation'], targets['segmentation'])

        # Depth loss (L1 + SSIM)
        pred_depth = predictions['depth']
        target_depth = targets['depth']
        # Mask out invalid depth (zero depth)
        valid_mask = target_depth > 0
        if valid_mask.any():
            losses['depth_l1'] = self.depth_l1(
                pred_depth[valid_mask], target_depth[valid_mask])
            losses['depth_ssim'] = self.depth_ssim(pred_depth, target_depth)
            losses['depth'] = losses['depth_l1'] + losses['depth_ssim']
        else:
            losses['depth_l1'] = torch.tensor(0.0, device=pred_depth.device)
            losses['depth_ssim'] = torch.tensor(0.0, device=pred_depth.device)
            losses['depth'] = torch.tensor(0.0, device=pred_depth.device)

        # Camera loss
        pred_intrinsics = predictions['intrinsics']  # [B, 4]
        target_intrinsics = targets['camera_intrinsics']  # [B, 4]
        losses['cam_intrinsic'] = self.intrinsic_l1(pred_intrinsics, target_intrinsics)

        pred_extrinsics = predictions['extrinsics']  # [B, 6]
        target_extrinsics = targets['camera_extrinsics']  # [B, 6]
        losses['cam_rotation'] = self.rotation_geodesic(
            pred_extrinsics[:, :3], target_extrinsics[:, :3])
        losses['cam_translation'] = self.translation_l1(
            pred_extrinsics[:, 3:], target_extrinsics[:, 3:])
        losses['camera'] = losses['cam_intrinsic'] + losses['cam_rotation'] + losses['cam_translation']

        # Domain adversarial loss
        losses['adv'] = torch.tensor(0.0, device=pred_depth.device)
        if 'domain_pred' in predictions and domain_labels is not None:
            domain_pred = predictions['domain_pred']
            domain_target = domain_labels.float().unsqueeze(1)  # [B, 1]
            feat_loss = self.domain_bce(domain_pred['feat_pred'], domain_target)
            logit_loss = self.domain_bce(domain_pred['logit_pred'], domain_target)
            losses['adv'] = (feat_loss + logit_loss) / 2.0

        # Total weighted loss
        losses['total'] = (
            self.seg_weight * losses['seg'] +
            self.depth_weight * losses['depth'] +
            self.camera_weight * losses['camera'] +
            self.adv_weight * losses['adv']
        )

        return losses
