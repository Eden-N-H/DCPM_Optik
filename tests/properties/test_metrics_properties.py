"""Property-based tests for evaluation metrics (Property 26).

Property 26: Evaluation metrics validity and identity
- All metrics non-negative
- mIoU/accuracy/deltas in [0,1]
- When pred=target, mIoU=1.0, RMSE=0.0, all deltas=1.0

**Validates: Requirements 17.1, 17.2, 17.3, 17.4**
"""
import numpy as np
import torch
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from src.training.metrics import MetricsComputer


# --- Strategies ---

@st.composite
def segmentation_tensors(draw, num_classes=3, min_batch=1, max_batch=2):
    """Generate valid segmentation prediction and target tensors.

    pred: [B, C, H, W] logits
    target: [B, H, W] integer class labels in [0, num_classes-1]
    """
    B = draw(st.integers(min_value=min_batch, max_value=max_batch))
    H = draw(st.sampled_from([8, 16]))
    W = draw(st.sampled_from([8, 16]))

    # Target: integer class labels
    target = torch.randint(0, num_classes, (B, H, W))

    # Predictions: logits (unbounded, but keep reasonable for stability)
    pred = torch.randn(B, num_classes, H, W) * 2.0

    return pred, target, num_classes


@st.composite
def depth_tensors(draw, min_batch=1, max_batch=2):
    """Generate valid depth prediction and target tensors.

    pred: [B, 1, H, W] positive depth values
    target: [B, 1, H, W] positive depth values (>0 for valid pixels)
    """
    B = draw(st.integers(min_value=min_batch, max_value=max_batch))
    H = draw(st.sampled_from([8, 16]))
    W = draw(st.sampled_from([8, 16]))

    # Positive depth values (avoid zero which is treated as invalid)
    min_depth = draw(st.floats(min_value=0.5, max_value=2.0))
    max_depth = draw(st.floats(min_value=min_depth + 0.5, max_value=10.0))

    pred = torch.FloatTensor(B, 1, H, W).uniform_(min_depth, max_depth)
    target = torch.FloatTensor(B, 1, H, W).uniform_(min_depth, max_depth)

    return pred, target


@st.composite
def camera_tensors(draw, min_batch=1, max_batch=2):
    """Generate valid camera prediction and target tensors.

    intrinsics: [B, 4] (fx, fy, cx, cy) - positive values
    extrinsics: [B, 6] (rodrigues3, translation3)
    """
    B = draw(st.integers(min_value=min_batch, max_value=max_batch))

    pred_intrinsics = torch.FloatTensor(B, 4).uniform_(100.0, 1000.0)
    target_intrinsics = torch.FloatTensor(B, 4).uniform_(100.0, 1000.0)

    # Small rotation angles to keep geodesic error well-defined
    pred_extrinsics = torch.FloatTensor(B, 6).uniform_(-1.0, 1.0)
    target_extrinsics = torch.FloatTensor(B, 6).uniform_(-1.0, 1.0)

    return pred_intrinsics, target_intrinsics, pred_extrinsics, target_extrinsics


@st.composite
def severity_tensors(draw, num_classes=3, min_batch=1, max_batch=2):
    """Generate valid severity prediction/target and segmentation target.

    severity pred/target: [B, 1, H, W] in [0, 1]
    seg target: [B, H, W] with at least some defect pixels (class > 0)
    """
    B = draw(st.integers(min_value=min_batch, max_value=max_batch))
    H = draw(st.sampled_from([8, 16]))
    W = draw(st.sampled_from([8, 16]))

    pred_severity = torch.FloatTensor(B, 1, H, W).uniform_(0.0, 1.0)
    target_severity = torch.FloatTensor(B, 1, H, W).uniform_(0.0, 1.0)

    # Segmentation target with at least some defect pixels (class > 0)
    target_seg = torch.randint(0, num_classes, (B, H, W))
    # Ensure there's at least one defect pixel
    target_seg[0, 0, 0] = 1

    return pred_severity, target_severity, target_seg


@st.composite
def full_prediction_targets(draw, num_classes=3):
    """Generate a complete set of prediction and target tensors for all tasks."""
    B = draw(st.integers(min_value=1, max_value=2))
    H = draw(st.sampled_from([8, 16]))
    W = draw(st.sampled_from([8, 16]))

    predictions = {
        'segmentation': torch.randn(B, num_classes, H, W) * 2.0,
        'depth': torch.FloatTensor(B, 1, H, W).uniform_(0.5, 5.0),
        'intrinsics': torch.FloatTensor(B, 4).uniform_(100.0, 1000.0),
        'extrinsics': torch.FloatTensor(B, 6).uniform_(-1.0, 1.0),
        'severity': torch.FloatTensor(B, 1, H, W).uniform_(0.0, 1.0),
    }

    targets = {
        'segmentation': torch.randint(0, num_classes, (B, H, W)),
        'depth': torch.FloatTensor(B, 1, H, W).uniform_(0.5, 5.0),
        'camera_intrinsics': torch.FloatTensor(B, 4).uniform_(100.0, 1000.0),
        'camera_extrinsics': torch.FloatTensor(B, 6).uniform_(-1.0, 1.0),
        'severity': torch.FloatTensor(B, 1, H, W).uniform_(0.0, 1.0),
    }

    # Ensure at least one defect pixel for severity evaluation
    targets['segmentation'][0, 0, 0] = 1

    return predictions, targets, num_classes


@st.composite
def identical_prediction_targets(draw, num_classes=3):
    """Generate prediction/target pairs where pred == target for identity checks.

    Ensures all classes are represented in the segmentation target so that
    mIoU is well-defined for all classes.
    """
    B = draw(st.integers(min_value=1, max_value=2))
    H = draw(st.sampled_from([8, 16]))
    W = draw(st.sampled_from([8, 16]))

    # Segmentation: make pred logits strongly favor the correct class
    target_seg = torch.randint(0, num_classes, (B, H, W))
    # Ensure all classes are present in the target (for every batch element)
    # so that IoU is well-defined for each class
    for b in range(B):
        for c in range(num_classes):
            target_seg[b, c, 0] = c  # Place at least one pixel per class

    # Create one-hot logits with large values for the correct class
    pred_seg = torch.zeros(B, num_classes, H, W) - 10.0
    for b in range(B):
        for c in range(num_classes):
            pred_seg[b, c][target_seg[b] == c] = 10.0

    # Depth: identical positive values
    depth_val = torch.FloatTensor(B, 1, H, W).uniform_(1.0, 5.0)

    # Camera: identical values (use small rotation to keep geodesic stable)
    intrinsics_val = torch.FloatTensor(B, 4).uniform_(100.0, 1000.0)
    extrinsics_val = torch.FloatTensor(B, 6).uniform_(-0.5, 0.5)

    # Severity: identical values with defect pixels present
    severity_val = torch.FloatTensor(B, 1, H, W).uniform_(0.0, 1.0)

    predictions = {
        'segmentation': pred_seg,
        'depth': depth_val.clone(),
        'intrinsics': intrinsics_val.clone(),
        'extrinsics': extrinsics_val.clone(),
        'severity': severity_val.clone(),
    }

    targets = {
        'segmentation': target_seg,
        'depth': depth_val.clone(),
        'camera_intrinsics': intrinsics_val.clone(),
        'camera_extrinsics': extrinsics_val.clone(),
        'severity': severity_val.clone(),
    }

    return predictions, targets, num_classes


# --- Property 26: Evaluation metrics validity and identity ---

class TestMetricsNonNegativity:
    """**Validates: Requirements 17.1, 17.2, 17.3, 17.4**

    Property 26 (part 1): All metric values SHALL be non-negative.
    """

    @given(data=full_prediction_targets())
    @settings(max_examples=50, deadline=None)
    def test_all_metrics_non_negative(self, data):
        """All computed metrics should be non-negative.

        Note: Pearson correlation is excluded as it naturally ranges [-1, 1].
        """
        predictions, targets, num_classes = data

        mc = MetricsComputer(num_classes=num_classes)
        mc.update(predictions, targets)
        metrics = mc.compute()

        # Pearson correlation is in [-1, 1] by definition, exclude it
        excluded_keys = {'severity/pearson_correlation'}

        for key, value in metrics.items():
            if key in excluded_keys:
                assert -1.0 <= value <= 1.0, (
                    f"Metric '{key}' is outside [-1, 1]: {value}"
                )
            else:
                assert value >= 0.0, (
                    f"Metric '{key}' is negative: {value}"
                )


class TestMetricsBoundedRange:
    """**Validates: Requirements 17.1, 17.2**

    Property 26 (part 2): mIoU, pixel accuracy, mean class accuracy,
    and delta thresholds SHALL be in [0, 1].
    """

    @given(data=full_prediction_targets())
    @settings(max_examples=50, deadline=None)
    def test_bounded_metrics_in_zero_one(self, data):
        """mIoU, accuracy, and delta metrics should be in [0, 1]."""
        predictions, targets, num_classes = data

        mc = MetricsComputer(num_classes=num_classes)
        mc.update(predictions, targets)
        metrics = mc.compute()

        bounded_keys = [
            'seg/miou',
            'seg/pixel_accuracy',
            'seg/mean_class_accuracy',
            'depth/delta_1',
            'depth/delta_2',
            'depth/delta_3',
        ]

        # Also check per-class IoU
        for c in range(num_classes):
            bounded_keys.append(f'seg/iou_class_{c}')

        for key in bounded_keys:
            assert key in metrics, f"Expected metric '{key}' not found"
            assert 0.0 <= metrics[key] <= 1.0, (
                f"Metric '{key}' = {metrics[key]} is not in [0, 1]"
            )


class TestMetricsIdentity:
    """**Validates: Requirements 17.1, 17.2, 17.3, 17.4**

    Property 26 (part 3): When prediction equals target exactly:
    - mIoU SHALL equal 1.0
    - RMSE SHALL equal 0.0
    - All delta thresholds SHALL equal 1.0
    """

    @given(data=identical_prediction_targets())
    @settings(max_examples=50, deadline=None)
    def test_perfect_predictions_segmentation(self, data):
        """When pred matches target exactly, mIoU should be 1.0."""
        predictions, targets, num_classes = data

        mc = MetricsComputer(num_classes=num_classes)
        mc.update(predictions, targets)
        metrics = mc.compute()

        assert abs(metrics['seg/miou'] - 1.0) < 1e-6, (
            f"Expected mIoU=1.0 for perfect predictions, got {metrics['seg/miou']}"
        )
        assert abs(metrics['seg/pixel_accuracy'] - 1.0) < 1e-6, (
            f"Expected pixel_accuracy=1.0 for perfect predictions, got {metrics['seg/pixel_accuracy']}"
        )

    @given(data=identical_prediction_targets())
    @settings(max_examples=50, deadline=None)
    def test_perfect_predictions_depth(self, data):
        """When pred matches target exactly, RMSE should be 0.0 and deltas 1.0."""
        predictions, targets, num_classes = data

        mc = MetricsComputer(num_classes=num_classes)
        mc.update(predictions, targets)
        metrics = mc.compute()

        assert abs(metrics['depth/rmse']) < 1e-5, (
            f"Expected RMSE=0.0 for perfect predictions, got {metrics['depth/rmse']}"
        )
        assert abs(metrics['depth/delta_1'] - 1.0) < 1e-6, (
            f"Expected delta_1=1.0 for perfect predictions, got {metrics['depth/delta_1']}"
        )
        assert abs(metrics['depth/delta_2'] - 1.0) < 1e-6, (
            f"Expected delta_2=1.0 for perfect predictions, got {metrics['depth/delta_2']}"
        )
        assert abs(metrics['depth/delta_3'] - 1.0) < 1e-6, (
            f"Expected delta_3=1.0 for perfect predictions, got {metrics['depth/delta_3']}"
        )

    @given(data=identical_prediction_targets())
    @settings(max_examples=50, deadline=None)
    def test_perfect_predictions_camera(self, data):
        """When pred matches target exactly, camera errors should be ~0.0.

        Note: geodesic error uses Rodrigues-to-matrix conversion which introduces
        small floating-point errors, so we use a slightly relaxed tolerance.
        """
        predictions, targets, num_classes = data

        mc = MetricsComputer(num_classes=num_classes)
        mc.update(predictions, targets)
        metrics = mc.compute()

        assert abs(metrics['camera/intrinsic_mae']) < 1e-5, (
            f"Expected intrinsic_mae=0.0 for perfect predictions, got {metrics['camera/intrinsic_mae']}"
        )
        assert abs(metrics['camera/rotation_geodesic']) < 1e-3, (
            f"Expected rotation_geodesic≈0.0 for perfect predictions, got {metrics['camera/rotation_geodesic']}"
        )
        assert abs(metrics['camera/translation_error']) < 1e-5, (
            f"Expected translation_error=0.0 for perfect predictions, got {metrics['camera/translation_error']}"
        )

    @given(data=identical_prediction_targets())
    @settings(max_examples=50, deadline=None)
    def test_perfect_predictions_severity(self, data):
        """When pred matches target exactly, severity MAE should be 0.0."""
        predictions, targets, num_classes = data

        mc = MetricsComputer(num_classes=num_classes)
        mc.update(predictions, targets)
        metrics = mc.compute()

        assert abs(metrics['severity/mae']) < 1e-5, (
            f"Expected severity/mae=0.0 for perfect predictions, got {metrics['severity/mae']}"
        )
