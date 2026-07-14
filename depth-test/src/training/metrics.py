"""Evaluation metrics for multi-task road quality model."""
import numpy as np
import torch
from typing import Dict, Optional


class MetricsComputer:
    """Computes evaluation metrics for all tasks.

    Segmentation: mIoU, per-class IoU, pixel accuracy, mean class accuracy
    Depth: RMSE, AbsRel, delta thresholds (1.25, 1.25^2, 1.25^3)
    Camera: intrinsic MAE, geodesic rotation error, translation error
    Severity: MAE, Pearson correlation (within defect regions only)
    """

    def __init__(self, num_classes: int = 8):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        """Reset all accumulated metrics."""
        # Segmentation accumulators
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

        # Depth accumulators
        self.depth_sq_errors = []
        self.depth_abs_rel_errors = []
        self.depth_ratios = []  # max(pred/gt, gt/pred) for delta thresholds

        # Camera accumulators
        self.intrinsic_errors = []
        self.rotation_geodesic_errors = []
        self.translation_errors = []

        # Severity accumulators
        self.severity_abs_errors = []
        self.severity_preds = []
        self.severity_targets = []

    @torch.no_grad()
    def update(self, predictions: Dict[str, torch.Tensor],
               targets: Dict[str, torch.Tensor]) -> None:
        """Update metrics with a batch of predictions and targets.

        Args:
            predictions: Model outputs dict
            targets: Ground truth dict
        """
        self._update_segmentation(predictions, targets)
        self._update_depth(predictions, targets)
        self._update_camera(predictions, targets)
        self._update_severity(predictions, targets)

    def _update_segmentation(self, predictions: Dict, targets: Dict) -> None:
        """Update segmentation confusion matrix."""
        pred_seg = predictions['segmentation']  # [B, C, H, W] logits
        target_seg = targets['segmentation']  # [B, H, W]

        pred_classes = pred_seg.argmax(dim=1).cpu().numpy()  # [B, H, W]
        target_classes = target_seg.cpu().numpy()  # [B, H, W]

        for b in range(pred_classes.shape[0]):
            pred_flat = pred_classes[b].ravel()
            target_flat = target_classes[b].ravel()

            # Only count valid target pixels
            valid = (target_flat >= 0) & (target_flat < self.num_classes)
            pred_flat = pred_flat[valid]
            target_flat = target_flat[valid]

            # Update confusion matrix
            for i in range(self.num_classes):
                for j in range(self.num_classes):
                    self.confusion_matrix[i, j] += np.sum(
                        (target_flat == i) & (pred_flat == j))

    def _update_depth(self, predictions: Dict, targets: Dict) -> None:
        """Update depth metrics."""
        pred_depth = predictions['depth'].cpu().numpy()  # [B, 1, H, W]
        target_depth = targets['depth'].cpu().numpy()  # [B, 1, H, W]

        for b in range(pred_depth.shape[0]):
            pred = pred_depth[b, 0]
            target = target_depth[b, 0]

            # Valid mask (non-zero target depth)
            valid = target > 0
            if not valid.any():
                continue

            pred_valid = pred[valid]
            target_valid = target[valid]

            # Avoid division by zero
            pred_valid = np.maximum(pred_valid, 1e-6)
            target_valid = np.maximum(target_valid, 1e-6)

            # Squared error for RMSE
            sq_err = (pred_valid - target_valid) ** 2
            self.depth_sq_errors.append(sq_err.mean())

            # Absolute relative error
            abs_rel = np.abs(pred_valid - target_valid) / target_valid
            self.depth_abs_rel_errors.append(abs_rel.mean())

            # Delta ratios
            ratio = np.maximum(pred_valid / target_valid, target_valid / pred_valid)
            self.depth_ratios.append(ratio)

    def _update_camera(self, predictions: Dict, targets: Dict) -> None:
        """Update camera metrics."""
        pred_intrinsics = predictions['intrinsics'].cpu().numpy()  # [B, 4]
        target_intrinsics = targets['camera_intrinsics'].cpu().numpy()  # [B, 4]

        pred_extrinsics = predictions['extrinsics'].cpu().numpy()  # [B, 6]
        target_extrinsics = targets['camera_extrinsics'].cpu().numpy()  # [B, 6]

        for b in range(pred_intrinsics.shape[0]):
            # Intrinsic MAE
            intrinsic_err = np.abs(pred_intrinsics[b] - target_intrinsics[b]).mean()
            self.intrinsic_errors.append(intrinsic_err)

            # Geodesic rotation error
            pred_rod = pred_extrinsics[b, :3]
            target_rod = target_extrinsics[b, :3]
            geo_err = self._geodesic_error(pred_rod, target_rod)
            self.rotation_geodesic_errors.append(geo_err)

            # Translation error (L2)
            trans_err = np.linalg.norm(pred_extrinsics[b, 3:] - target_extrinsics[b, 3:])
            self.translation_errors.append(trans_err)

    def _update_severity(self, predictions: Dict, targets: Dict) -> None:
        """Update severity metrics (within defect regions only)."""
        pred_severity = predictions['severity'].cpu().numpy()  # [B, 1, H, W]
        target_severity = targets['severity'].cpu().numpy()  # [B, 1, H, W]
        target_seg = targets['segmentation'].cpu().numpy()  # [B, H, W]

        for b in range(pred_severity.shape[0]):
            pred = pred_severity[b, 0]
            target = target_severity[b, 0]
            seg = target_seg[b]

            # Only evaluate within defect regions (class > 0)
            defect_mask = seg > 0
            if not defect_mask.any():
                continue

            pred_defect = pred[defect_mask]
            target_defect = target[defect_mask]

            # MAE
            self.severity_abs_errors.append(np.abs(pred_defect - target_defect).mean())

            # Store for correlation
            self.severity_preds.append(pred_defect)
            self.severity_targets.append(target_defect)

    def _geodesic_error(self, pred_rodrigues: np.ndarray,
                        target_rodrigues: np.ndarray) -> float:
        """Compute geodesic angle error between two Rodrigues vectors."""
        R_pred = self._rodrigues_to_matrix_np(pred_rodrigues)
        R_target = self._rodrigues_to_matrix_np(target_rodrigues)

        R_diff = R_pred.T @ R_target
        trace = np.trace(R_diff)
        cos_angle = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        return float(angle)

    def _rodrigues_to_matrix_np(self, rodrigues: np.ndarray) -> np.ndarray:
        """Convert Rodrigues vector to rotation matrix (numpy)."""
        theta = np.linalg.norm(rodrigues)
        if theta < 1e-8:
            return np.eye(3)
        axis = rodrigues / theta
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)

    def compute(self) -> Dict[str, float]:
        """Compute all final metrics.

        Returns:
            Dict with all metric values
        """
        metrics = {}

        # Segmentation metrics
        metrics.update(self._compute_segmentation_metrics())

        # Depth metrics
        metrics.update(self._compute_depth_metrics())

        # Camera metrics
        metrics.update(self._compute_camera_metrics())

        # Severity metrics
        metrics.update(self._compute_severity_metrics())

        return metrics

    def _compute_segmentation_metrics(self) -> Dict[str, float]:
        """Compute segmentation metrics from confusion matrix."""
        metrics = {}
        cm = self.confusion_matrix

        # Pixel accuracy
        total = cm.sum()
        if total > 0:
            metrics['seg/pixel_accuracy'] = float(cm.trace()) / float(total)
        else:
            metrics['seg/pixel_accuracy'] = 0.0

        # Per-class IoU and mean class accuracy
        ious = []
        class_accuracies = []
        for c in range(self.num_classes):
            tp = cm[c, c]
            fp = cm[:, c].sum() - tp
            fn = cm[c, :].sum() - tp

            # IoU
            denom = tp + fp + fn
            if denom > 0:
                iou = float(tp) / float(denom)
            else:
                iou = 0.0
            ious.append(iou)
            metrics[f'seg/iou_class_{c}'] = iou

            # Class accuracy (recall)
            class_total = cm[c, :].sum()
            if class_total > 0:
                class_accuracies.append(float(tp) / float(class_total))
            else:
                class_accuracies.append(0.0)

        metrics['seg/miou'] = float(np.mean(ious))
        metrics['seg/mean_class_accuracy'] = float(np.mean(class_accuracies))

        return metrics

    def _compute_depth_metrics(self) -> Dict[str, float]:
        """Compute depth metrics."""
        metrics = {}

        if not self.depth_sq_errors:
            metrics['depth/rmse'] = 0.0
            metrics['depth/abs_rel'] = 0.0
            metrics['depth/delta_1'] = 0.0
            metrics['depth/delta_2'] = 0.0
            metrics['depth/delta_3'] = 0.0
            return metrics

        # RMSE
        metrics['depth/rmse'] = float(np.sqrt(np.mean(self.depth_sq_errors)))

        # Absolute relative error
        metrics['depth/abs_rel'] = float(np.mean(self.depth_abs_rel_errors))

        # Delta thresholds
        if self.depth_ratios:
            all_ratios = np.concatenate(self.depth_ratios)
            metrics['depth/delta_1'] = float(np.mean(all_ratios < 1.25))
            metrics['depth/delta_2'] = float(np.mean(all_ratios < 1.25 ** 2))
            metrics['depth/delta_3'] = float(np.mean(all_ratios < 1.25 ** 3))
        else:
            metrics['depth/delta_1'] = 0.0
            metrics['depth/delta_2'] = 0.0
            metrics['depth/delta_3'] = 0.0

        return metrics

    def _compute_camera_metrics(self) -> Dict[str, float]:
        """Compute camera metrics."""
        metrics = {}

        if self.intrinsic_errors:
            metrics['camera/intrinsic_mae'] = float(np.mean(self.intrinsic_errors))
        else:
            metrics['camera/intrinsic_mae'] = 0.0

        if self.rotation_geodesic_errors:
            metrics['camera/rotation_geodesic'] = float(np.mean(self.rotation_geodesic_errors))
        else:
            metrics['camera/rotation_geodesic'] = 0.0

        if self.translation_errors:
            metrics['camera/translation_error'] = float(np.mean(self.translation_errors))
        else:
            metrics['camera/translation_error'] = 0.0

        return metrics

    def _compute_severity_metrics(self) -> Dict[str, float]:
        """Compute severity metrics (within defect regions)."""
        metrics = {}

        if self.severity_abs_errors:
            metrics['severity/mae'] = float(np.mean(self.severity_abs_errors))
        else:
            metrics['severity/mae'] = 0.0

        # Pearson correlation
        if self.severity_preds and self.severity_targets:
            all_preds = np.concatenate(self.severity_preds)
            all_targets = np.concatenate(self.severity_targets)
            if len(all_preds) > 1 and np.std(all_preds) > 1e-8 and np.std(all_targets) > 1e-8:
                correlation = np.corrcoef(all_preds, all_targets)[0, 1]
                metrics['severity/pearson_correlation'] = float(correlation)
            else:
                metrics['severity/pearson_correlation'] = 0.0
        else:
            metrics['severity/pearson_correlation'] = 0.0

        return metrics
