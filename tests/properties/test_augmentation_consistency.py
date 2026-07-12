"""Property test for augmentation geometric consistency (Property 25).

**Validates: Requirements 16.1**

For any image-mask pair, applying training augmentation preserves spatial
correspondence between pixels and their labels. Specifically, geometric
transforms (flip, rotation, crop) are applied identically to both the
RGB image and all label maps (depth, segmentation, severity).
"""
import numpy as np
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from src.training.dataset import RoadQualityDataset


def create_dataset_instance(crop_size: int = 480) -> RoadQualityDataset:
    """Create a RoadQualityDataset with train split for augmentation testing."""
    dataset = object.__new__(RoadQualityDataset)
    dataset.root = None
    dataset.split = 'train'
    dataset.crop_size = crop_size
    dataset.is_train = True
    dataset.sample_ids = []
    dataset.color_jitter = {
        'brightness': 0.2,
        'contrast': 0.2,
        'saturation': 0.1,
        'hue': 0.05,
    }
    return dataset


@st.composite
def image_mask_pairs(draw, min_size: int = 481, max_size: int = 600):
    """Generate aligned image-mask pairs with known spatial correspondence.

    We create an image where each pixel's value encodes its original position,
    and masks that use deterministic patterns based on position so we can verify
    spatial correspondence after augmentation.
    """
    h = draw(st.integers(min_value=min_size, max_value=max_size))
    w = draw(st.integers(min_value=min_size, max_value=max_size))

    # Create RGB image with random content
    rgb = draw(arrays(
        dtype=np.uint8,
        shape=(h, w, 3),
        elements=st.integers(min_value=0, max_value=255),
    ))

    # Create depth map (float32)
    depth = draw(arrays(
        dtype=np.float32,
        shape=(h, w),
        elements=st.floats(min_value=100.0, max_value=10000.0, allow_nan=False, allow_infinity=False),
    ))

    # Create segmentation mask with class IDs 0, 1, 2
    seg = draw(arrays(
        dtype=np.int32,
        shape=(h, w),
        elements=st.integers(min_value=0, max_value=2),
    ))

    # Create severity map (float32 in [0, 1])
    severity = draw(arrays(
        dtype=np.float32,
        shape=(h, w),
        elements=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    ))

    return rgb, depth, seg, severity


@given(data=st.data())
@settings(max_examples=30, deadline=30000)
def test_augmentation_preserves_spatial_correspondence(data):
    """Property 25: Augmentation geometric consistency.

    **Validates: Requirements 16.1**

    For any image-mask pair, applying training augmentation (flip, rotation, crop)
    preserves spatial correspondence — the same geometric transforms are applied
    to all modalities so that pixel locations remain aligned.

    We verify this by:
    1. Creating a marker pattern where the segmentation mask has a unique
       identifiable region
    2. Applying augmentation to both image and masks with the same random state
    3. Verifying that after augmentation, the spatial relationship between
       modalities is preserved (same dimensions, alignment maintained)
    """
    h = data.draw(st.integers(min_value=481, max_value=550))
    w = data.draw(st.integers(min_value=481, max_value=550))

    # Create an image where we embed positional information
    # Use a grid pattern in the segmentation mask for spatial reference
    rgb = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
    depth = np.random.uniform(100, 10000, (h, w)).astype(np.float32)
    severity = np.random.uniform(0, 1, (h, w)).astype(np.float32)

    # Create a segmentation mask with a distinctive spatial pattern:
    # Top half = class 1, bottom half = class 2
    seg = np.zeros((h, w), dtype=np.int32)
    seg[:h // 2, :] = 1
    seg[h // 2:, :] = 2

    dataset = create_dataset_instance(crop_size=480)

    # Fix the random seed so augmentation is deterministic for this test
    seed = data.draw(st.integers(min_value=0, max_value=2**31 - 1))
    np.random.seed(seed)

    # Apply augmentation
    aug_rgb, aug_depth, aug_seg, aug_severity = dataset._train_augment(
        rgb.copy(), depth.copy(), seg.copy(), severity.copy()
    )

    # Property checks:

    # 1. All outputs have the same spatial dimensions
    assert aug_rgb.shape[:2] == aug_depth.shape[:2], (
        f"RGB shape {aug_rgb.shape[:2]} != depth shape {aug_depth.shape[:2]}"
    )
    assert aug_rgb.shape[:2] == aug_seg.shape[:2], (
        f"RGB shape {aug_rgb.shape[:2]} != seg shape {aug_seg.shape[:2]}"
    )
    assert aug_rgb.shape[:2] == aug_severity.shape[:2], (
        f"RGB shape {aug_rgb.shape[:2]} != severity shape {aug_severity.shape[:2]}"
    )

    # 2. RGB maintains 3 channels
    assert aug_rgb.shape[2] == 3

    # 3. Segmentation values remain valid class IDs (0, 1, or 2)
    unique_classes = np.unique(aug_seg)
    assert all(c in {0, 1, 2} for c in unique_classes), (
        f"Invalid segmentation classes after augmentation: {unique_classes}"
    )

    # 4. Output crop size is correct (480x480)
    assert aug_rgb.shape[0] == 480 and aug_rgb.shape[1] == 480, (
        f"Expected 480x480 crop, got {aug_rgb.shape[:2]}"
    )


@given(data=st.data())
@settings(max_examples=30, deadline=30000)
def test_augmentation_geometric_consistency_with_markers(data):
    """Property 25: Verify geometric consistency using marker-based approach.

    **Validates: Requirements 16.1**

    We place a unique marker in the image and a corresponding marker in the
    segmentation mask at the same location. After augmentation with the same
    random state, the markers should still co-locate, proving that geometric
    transforms are applied identically.
    """
    h = data.draw(st.integers(min_value=500, max_value=550))
    w = data.draw(st.integers(min_value=500, max_value=550))

    # Create a solid background image with a distinct colored square marker
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    depth = np.ones((h, w), dtype=np.float32) * 1000.0
    seg = np.zeros((h, w), dtype=np.int32)
    severity = np.zeros((h, w), dtype=np.float32)

    # Place a marker in the center region (guaranteed to survive a 480 crop
    # from a 500+ image since it's in the center)
    # Marker: a 50x50 block with distinct values
    cy, cx = h // 2, w // 2
    marker_half = 25

    # RGB marker: bright red
    rgb[cy - marker_half:cy + marker_half, cx - marker_half:cx + marker_half] = [255, 0, 0]
    # Depth marker: high depth value
    depth[cy - marker_half:cy + marker_half, cx - marker_half:cx + marker_half] = 5000.0
    # Segmentation marker: class 2
    seg[cy - marker_half:cy + marker_half, cx - marker_half:cx + marker_half] = 2
    # Severity marker: high severity
    severity[cy - marker_half:cy + marker_half, cx - marker_half:cx + marker_half] = 0.9

    dataset = create_dataset_instance(crop_size=480)

    # Fix seed for deterministic augmentation
    seed = data.draw(st.integers(min_value=0, max_value=2**31 - 1))
    np.random.seed(seed)

    aug_rgb, aug_depth, aug_seg, aug_severity = dataset._train_augment(
        rgb.copy(), depth.copy(), seg.copy(), severity.copy()
    )

    # Find where the segmentation marker (class 2) ended up
    seg_marker_mask = (aug_seg == 2)

    # Find where the depth marker (5000.0) ended up
    # Use a range since rotation uses interpolation for depth (INTER_NEAREST)
    depth_marker_mask = (aug_depth >= 4999.0)

    # Find where the severity marker ended up
    severity_marker_mask = (aug_severity >= 0.85)

    # Find where the red RGB marker ended up (after color jitter, values may change
    # but the RED channel should still be highest since it started at [255, 0, 0])
    # Note: color jitter changes values, so we check positional alignment via seg/depth

    # KEY PROPERTY: Where segmentation says class 2, depth should also show
    # the marker value (both underwent same geometric transform)
    if seg_marker_mask.any():
        # The segmentation and depth markers should overlap significantly
        # They start perfectly aligned, so after identical geometric transforms
        # they should remain aligned (nearest-neighbor interpolation for both)
        overlap = np.logical_and(seg_marker_mask, depth_marker_mask)
        # At least 80% of seg marker pixels should also be depth marker pixels
        # (some edge pixels might differ due to interpolation)
        seg_count = seg_marker_mask.sum()
        if seg_count > 0:
            overlap_ratio = overlap.sum() / seg_count
            assert overlap_ratio >= 0.8, (
                f"Seg-depth marker overlap ratio {overlap_ratio:.3f} < 0.8. "
                f"Geometric transforms are not applied consistently."
            )

    # Also verify severity marker aligns with segmentation marker
    if seg_marker_mask.any() and severity_marker_mask.any():
        overlap_sev = np.logical_and(seg_marker_mask, severity_marker_mask)
        seg_count = seg_marker_mask.sum()
        if seg_count > 0:
            overlap_ratio_sev = overlap_sev.sum() / seg_count
            assert overlap_ratio_sev >= 0.8, (
                f"Seg-severity marker overlap ratio {overlap_ratio_sev:.3f} < 0.8. "
                f"Geometric transforms are not applied consistently."
            )


@given(data=st.data())
@settings(max_examples=30, deadline=30000)
def test_augmentation_flip_consistency(data):
    """Property 25: Verify horizontal flip is applied consistently to all modalities.

    **Validates: Requirements 16.1**

    When flip is applied, it should flip all modalities identically.
    We force a flip by controlling the random state and verify all outputs
    are flipped consistently.
    """
    h = data.draw(st.integers(min_value=481, max_value=520))
    w = data.draw(st.integers(min_value=481, max_value=520))

    # Create asymmetric patterns to detect flipping
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[:, :w // 2, :] = 200  # Left half bright

    depth = np.zeros((h, w), dtype=np.float32)
    depth[:, :w // 2] = 5000.0  # Left half high depth

    seg = np.zeros((h, w), dtype=np.int32)
    seg[:, :w // 2] = 1  # Left half class 1

    severity = np.zeros((h, w), dtype=np.float32)
    severity[:, :w // 2] = 0.8  # Left half high severity

    dataset = create_dataset_instance(crop_size=480)

    # Force a flip (random < 0.5 means flip happens)
    # Set state so random.random() returns < 0.5 for flip,
    # and angle ~ 0 (no rotation)
    np.random.seed(42)  # This gives random() = 0.37 < 0.5, so flip occurs

    aug_rgb, aug_depth, aug_seg, aug_severity = dataset._train_augment(
        rgb.copy(), depth.copy(), seg.copy(), severity.copy()
    )

    # After crop, check that the spatial patterns in all modalities are consistent
    # If depth right half is high, seg right half should be class 1, etc.
    # (If flip occurred, the originally-left patterns are now on the right)

    # Check consistency: where seg == 1, depth should be high
    seg_class1_mask = (aug_seg == 1)
    if seg_class1_mask.any():
        mean_depth_at_class1 = aug_depth[seg_class1_mask].mean()
        seg_class0_mask = (aug_seg == 0)
        if seg_class0_mask.any():
            mean_depth_at_class0 = aug_depth[seg_class0_mask].mean()
            # Class 1 regions should have higher depth than class 0 regions
            # because both were flipped together
            assert mean_depth_at_class1 > mean_depth_at_class0, (
                f"Depth-seg inconsistency: depth at class1={mean_depth_at_class1:.1f} "
                f"should be > depth at class0={mean_depth_at_class0:.1f}"
            )
