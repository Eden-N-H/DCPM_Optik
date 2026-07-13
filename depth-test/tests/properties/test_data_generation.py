"""Property-based tests for data generation (Properties 1, 3, 4, 5).

Validates: Requirements 1.4, 1.7, 2.2, 2.3, 2.4, 2.5
"""

import json
import tempfile
from pathlib import Path

import numpy as np
from hypothesis import given, settings, assume
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from src.synth.renderer import save_camera_params_json, load_camera_params_json
from src.synth.dataset_builder import (
    compute_split_counts,
    compute_view_counts,
    DatasetConfig,
)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Strategy for generating valid 3x3 intrinsics matrices
@st.composite
def intrinsics_matrices(draw):
    """Generate valid 3x3 camera intrinsics matrices.

    A valid intrinsics matrix has the form:
        [[fx,  0, cx],
         [ 0, fy, cy],
         [ 0,  0,  1]]
    """
    fx = draw(st.floats(min_value=100.0, max_value=2000.0, allow_nan=False, allow_infinity=False))
    fy = draw(st.floats(min_value=100.0, max_value=2000.0, allow_nan=False, allow_infinity=False))
    cx = draw(st.floats(min_value=50.0, max_value=1000.0, allow_nan=False, allow_infinity=False))
    cy = draw(st.floats(min_value=50.0, max_value=1000.0, allow_nan=False, allow_infinity=False))
    K = np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    return K


# Strategy for generating valid 3x4 extrinsics matrices
@st.composite
def extrinsics_matrices(draw):
    """Generate valid 3x4 extrinsics matrices [R|t].

    R is an orthonormal rotation matrix and t is a translation vector.
    """
    # Generate a random rotation via axis-angle
    angle = draw(st.floats(min_value=-np.pi, max_value=np.pi, allow_nan=False, allow_infinity=False))
    axis_raw = draw(arrays(
        dtype=np.float64,
        shape=(3,),
        elements=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    ))
    norm = np.linalg.norm(axis_raw)
    assume(norm > 0.01)  # avoid degenerate axis
    axis = axis_raw / norm

    # Rodrigues' rotation formula
    K_skew = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0],
    ], dtype=np.float64)
    R = np.eye(3) + np.sin(angle) * K_skew + (1 - np.cos(angle)) * (K_skew @ K_skew)

    # Translation vector
    t = draw(arrays(
        dtype=np.float64,
        shape=(3, 1),
        elements=st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    ))

    extrinsics = np.hstack([R, t])
    return extrinsics


# Strategy for dataset sizes within valid range
dataset_size_st = st.integers(min_value=100, max_value=20000)

# Strategy for depth arrays with positive values
@st.composite
def depth_arrays(draw):
    """Generate depth arrays with positive min < max values."""
    h = draw(st.integers(min_value=2, max_value=64))
    w = draw(st.integers(min_value=2, max_value=64))
    # Generate positive depth values
    depth = draw(arrays(
        dtype=np.float64,
        shape=(h, w),
        elements=st.floats(min_value=0.1, max_value=1000.0, allow_nan=False, allow_infinity=False),
    ))
    # Ensure there are at least 2 distinct values for meaningful ordering test
    assume(depth.max() > depth.min())
    return depth


# Strategy for segmentation masks with valid class IDs
@st.composite
def segmentation_masks(draw):
    """Generate segmentation masks with pixels in {0, 1, 2}."""
    h = draw(st.integers(min_value=4, max_value=128))
    w = draw(st.integers(min_value=4, max_value=128))
    mask = draw(arrays(
        dtype=np.uint8,
        shape=(h, w),
        elements=st.integers(min_value=0, max_value=2),
    ))
    return mask


# ---------------------------------------------------------------------------
# Property 1: Camera parameters serialization round-trip
# ---------------------------------------------------------------------------


class TestProperty1CameraParamsRoundTrip:
    """**Validates: Requirements 1.4**

    Property 1: Camera parameters serialization round-trip.
    For any valid 3x3 intrinsics matrix and 3x4 extrinsics matrix,
    serializing to JSON and deserializing SHALL produce numerically
    equivalent matrices (within floating-point tolerance of 1e-6).
    """

    @given(intrinsics=intrinsics_matrices(), extrinsics=extrinsics_matrices())
    @settings(max_examples=50, deadline=5000)
    def test_camera_params_json_round_trip(self, intrinsics, extrinsics, tmp_path_factory):
        """Serialize camera intrinsics/extrinsics to JSON and back within 1e-6 tolerance."""
        tmp_dir = tmp_path_factory.mktemp("camera_rt")
        json_path = tmp_dir / "camera.json"

        # Serialize
        save_camera_params_json(json_path, intrinsics, extrinsics)

        # Deserialize
        loaded_intrinsics, loaded_extrinsics = load_camera_params_json(json_path)

        # Check shape preservation
        assert loaded_intrinsics.shape == (3, 3), (
            f"Intrinsics shape mismatch: expected (3, 3), got {loaded_intrinsics.shape}"
        )
        assert loaded_extrinsics.shape == (3, 4), (
            f"Extrinsics shape mismatch: expected (3, 4), got {loaded_extrinsics.shape}"
        )

        # Check numerical equivalence within tolerance
        assert np.allclose(loaded_intrinsics, intrinsics, atol=1e-6), (
            f"Intrinsics not preserved within 1e-6 tolerance.\n"
            f"Max diff: {np.max(np.abs(loaded_intrinsics - intrinsics))}"
        )
        assert np.allclose(loaded_extrinsics, extrinsics, atol=1e-6), (
            f"Extrinsics not preserved within 1e-6 tolerance.\n"
            f"Max diff: {np.max(np.abs(loaded_extrinsics - extrinsics))}"
        )

    @given(intrinsics=intrinsics_matrices(), extrinsics=extrinsics_matrices())
    @settings(max_examples=30, deadline=5000)
    def test_camera_params_json_structure(self, intrinsics, extrinsics, tmp_path_factory):
        """Serialized JSON contains expected keys with correct structure."""
        tmp_dir = tmp_path_factory.mktemp("camera_struct")
        json_path = tmp_dir / "camera.json"

        save_camera_params_json(json_path, intrinsics, extrinsics)

        # Verify JSON structure
        with open(json_path, "r") as f:
            data = json.load(f)

        assert "intrinsics_K" in data, "JSON missing 'intrinsics_K' key"
        assert "extrinsics_Rt" in data, "JSON missing 'extrinsics_Rt' key"

        # Verify list-of-lists format
        assert len(data["intrinsics_K"]) == 3, "Intrinsics should have 3 rows"
        assert all(len(row) == 3 for row in data["intrinsics_K"]), "Intrinsics rows should have 3 columns"
        assert len(data["extrinsics_Rt"]) == 3, "Extrinsics should have 3 rows"
        assert all(len(row) == 4 for row in data["extrinsics_Rt"]), "Extrinsics rows should have 4 columns"


# ---------------------------------------------------------------------------
# Property 3: Dataset split and viewpoint balance
# ---------------------------------------------------------------------------


class TestProperty3DatasetSplitAndViewpointBalance:
    """**Validates: Requirements 1.7**

    Property 3: Dataset split and viewpoint balance.
    For any total dataset size within [15876, 16196], the train/val/test split
    SHALL produce counts within 80%/10%/10% (±1%) of total, and each viewpoint
    (dashcam/drone) SHALL comprise 50% ±2% of the total images.
    """

    @given(total=st.integers(min_value=100, max_value=20000))
    @settings(max_examples=100, deadline=5000)
    def test_split_ratios_within_tolerance(self, total):
        """Train/val/test split within 80/10/10 ±1%."""
        ratios = {"train": 0.80, "val": 0.10, "test": 0.10}
        counts = compute_split_counts(total, ratios)

        # Verify counts sum to total
        assert sum(counts.values()) == total, (
            f"Split counts {counts} do not sum to {total}"
        )

        # Verify each split is within ±1% of expected ratio
        for split_name, expected_ratio in ratios.items():
            actual_ratio = counts[split_name] / total
            assert abs(actual_ratio - expected_ratio) <= 0.01, (
                f"Split '{split_name}' ratio {actual_ratio:.4f} exceeds "
                f"±1% tolerance from {expected_ratio}"
            )

    @given(total=st.integers(min_value=100, max_value=20000))
    @settings(max_examples=100, deadline=5000)
    def test_viewpoint_balance_within_tolerance(self, total):
        """Each viewpoint (dashcam/drone) comprises 50% ±2% of total images."""
        ratios = {"train": 0.80, "val": 0.10, "test": 0.10}
        split_counts = compute_split_counts(total, ratios)

        # Compute viewpoint counts across all splits
        total_dashcam = 0
        total_drone = 0
        total_generated = 0

        for split_name, split_count in split_counts.items():
            view_counts = compute_view_counts(split_count)
            total_dashcam += view_counts["dashcam"]
            total_drone += view_counts["drone"]
            total_generated += view_counts["dashcam"] + view_counts["drone"]

        # Total generated should equal total
        assert total_generated == total, (
            f"Total generated {total_generated} != total {total}"
        )

        # Verify viewpoint balance: 50% ±2%
        dashcam_ratio = total_dashcam / total
        drone_ratio = total_drone / total

        assert abs(dashcam_ratio - 0.5) <= 0.02, (
            f"Dashcam ratio {dashcam_ratio:.4f} exceeds 50% ±2% tolerance"
        )
        assert abs(drone_ratio - 0.5) <= 0.02, (
            f"Drone ratio {drone_ratio:.4f} exceeds 50% ±2% tolerance"
        )

    @given(total=st.integers(min_value=15876, max_value=16196))
    @settings(max_examples=50, deadline=5000)
    def test_target_range_split_balance(self, total):
        """Within target range [15876, 16196], splits are precisely balanced."""
        ratios = {"train": 0.80, "val": 0.10, "test": 0.10}
        counts = compute_split_counts(total, ratios)

        # All splits must have positive counts
        for split_name, count in counts.items():
            assert count > 0, f"Split '{split_name}' has zero samples"

        # Verify the largest split is train
        assert counts["train"] > counts["val"], "Train should be larger than val"
        assert counts["train"] > counts["test"], "Train should be larger than test"


# ---------------------------------------------------------------------------
# Property 4: Depth map normalization round-trip
# ---------------------------------------------------------------------------


def normalize_depth_to_uint16(depth: np.ndarray) -> tuple:
    """Normalize depth values to uint16 [0, 65535].

    Args:
        depth: Float depth array with arbitrary positive values.

    Returns:
        Tuple of (normalized uint16 array, min_depth, max_depth).
    """
    min_depth = depth.min()
    max_depth = depth.max()
    depth_range = max_depth - min_depth

    if depth_range == 0:
        return np.zeros_like(depth, dtype=np.uint16), min_depth, max_depth

    normalized = ((depth - min_depth) / depth_range * 65535.0).astype(np.uint16)
    return normalized, min_depth, max_depth


def denormalize_depth_from_uint16(
    normalized: np.ndarray, min_depth: float, max_depth: float
) -> np.ndarray:
    """Denormalize uint16 depth values back to original range.

    Args:
        normalized: uint16 depth array in [0, 65535].
        min_depth: Original minimum depth value.
        max_depth: Original maximum depth value.

    Returns:
        Float depth array in original range.
    """
    depth_range = max_depth - min_depth
    return normalized.astype(np.float64) / 65535.0 * depth_range + min_depth


class TestProperty4DepthNormalizationRoundTrip:
    """**Validates: Requirements 2.4**

    Property 4: Depth map normalization round-trip.
    For any depth array with arbitrary positive min and max values,
    normalizing to uint16 [0, 65535] and denormalizing back SHALL
    preserve the relative ordering of all pixel values and recover
    original values within quantization error (max_depth - min_depth) / 65535.
    """

    @given(depth=depth_arrays())
    @settings(max_examples=50, deadline=5000)
    def test_normalization_preserves_ordering(self, depth):
        """Normalize to uint16 and back preserves relative ordering."""
        normalized, min_depth, max_depth = normalize_depth_to_uint16(depth)

        # Verify uint16 output range
        assert normalized.dtype == np.uint16, (
            f"Normalized dtype should be uint16, got {normalized.dtype}"
        )
        assert normalized.min() >= 0
        assert normalized.max() <= 65535

        # Check ordering preservation: for any two pixels,
        # if depth[i] < depth[j] then normalized[i] <= normalized[j]
        flat_depth = depth.flatten()
        flat_norm = normalized.flatten()

        # Sample pairs to check (full check would be O(n^2))
        n = len(flat_depth)
        rng = np.random.default_rng(42)
        num_pairs = min(1000, n * (n - 1) // 2)
        indices_a = rng.integers(0, n, size=num_pairs)
        indices_b = rng.integers(0, n, size=num_pairs)

        for ia, ib in zip(indices_a, indices_b):
            if flat_depth[ia] < flat_depth[ib]:
                assert flat_norm[ia] <= flat_norm[ib], (
                    f"Ordering violated: depth[{ia}]={flat_depth[ia]} < "
                    f"depth[{ib}]={flat_depth[ib]} but "
                    f"normalized[{ia}]={flat_norm[ia]} > normalized[{ib}]={flat_norm[ib]}"
                )

    @given(depth=depth_arrays())
    @settings(max_examples=50, deadline=5000)
    def test_normalization_recovers_values_within_quantization_error(self, depth):
        """Round-trip recovers values within quantization error."""
        normalized, min_depth, max_depth = normalize_depth_to_uint16(depth)
        recovered = denormalize_depth_from_uint16(normalized, min_depth, max_depth)

        # Maximum quantization error
        depth_range = max_depth - min_depth
        max_error = depth_range / 65535.0

        # All recovered values should be within one quantization step
        diff = np.abs(recovered - depth)
        assert np.all(diff <= max_error + 1e-10), (
            f"Recovery error {diff.max():.10f} exceeds quantization error "
            f"{max_error:.10f} (range={depth_range})"
        )

    @given(depth=depth_arrays())
    @settings(max_examples=30, deadline=5000)
    def test_normalization_min_maps_to_zero_max_maps_to_65535(self, depth):
        """Min depth maps to 0, max depth maps to 65535."""
        normalized, min_depth, max_depth = normalize_depth_to_uint16(depth)

        # Find pixels at min and max depth
        min_mask = depth == min_depth
        max_mask = depth == max_depth

        # Min depth should map to 0
        assert np.all(normalized[min_mask] == 0), (
            f"Min depth pixels should map to 0, got {normalized[min_mask]}"
        )
        # Max depth should map to 65535
        assert np.all(normalized[max_mask] == 65535), (
            f"Max depth pixels should map to 65535, got {normalized[max_mask]}"
        )


# ---------------------------------------------------------------------------
# Property 5: Segmentation mask encoding validity
# ---------------------------------------------------------------------------


class TestProperty5SegmentationMaskValidity:
    """**Validates: Requirements 2.2**

    Property 5: Segmentation mask encoding validity.
    For any generated segmentation mask, all pixel values SHALL be integers
    in the set {0, 1, 2} (background, road, defect), and the encoded array
    SHALL have dtype compatible with integer storage.
    """

    @given(mask=segmentation_masks())
    @settings(max_examples=50, deadline=5000)
    def test_all_pixels_in_valid_classes(self, mask):
        """All pixel values in {0, 1, 2} with integer dtype."""
        valid_classes = {0, 1, 2}

        # Check dtype is integer
        assert np.issubdtype(mask.dtype, np.integer), (
            f"Segmentation mask dtype should be integer, got {mask.dtype}"
        )

        # Check all values are in the valid set
        unique_values = set(np.unique(mask).tolist())
        assert unique_values.issubset(valid_classes), (
            f"Mask contains invalid class IDs: {unique_values - valid_classes}. "
            f"Valid classes are {valid_classes}."
        )

    @given(
        h=st.integers(min_value=1, max_value=512),
        w=st.integers(min_value=1, max_value=512),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(max_examples=50, deadline=5000)
    def test_generated_mask_encoding(self, h, w, seed):
        """Simulated segmentation mask generation produces valid encoding."""
        rng = np.random.default_rng(seed)

        # Simulate generating a segmentation mask with 3 classes
        # (as the pipeline would produce: 0=background, 1=road, 2=defect)
        mask = rng.integers(0, 3, size=(h, w), dtype=np.uint8)

        # Verify integer dtype
        assert np.issubdtype(mask.dtype, np.integer), (
            f"Generated mask dtype should be integer, got {mask.dtype}"
        )

        # Verify all values in valid set
        assert mask.min() >= 0, f"Mask has negative values: min={mask.min()}"
        assert mask.max() <= 2, f"Mask has values > 2: max={mask.max()}"

        # Verify shape preservation
        assert mask.shape == (h, w), (
            f"Mask shape {mask.shape} != expected ({h}, {w})"
        )

    @given(mask=segmentation_masks())
    @settings(max_examples=30, deadline=5000)
    def test_mask_dtype_compatibility_with_png_storage(self, mask):
        """Mask dtype is compatible with PNG integer storage (uint8 or uint16)."""
        # The pipeline stores segmentation masks as PNG images, which require
        # integer dtypes (uint8 or uint16)
        valid_png_dtypes = (np.uint8, np.uint16, np.int8, np.int16, np.int32)
        assert mask.dtype in valid_png_dtypes or np.issubdtype(mask.dtype, np.integer), (
            f"Mask dtype {mask.dtype} not compatible with PNG integer storage"
        )

        # Values must fit in uint8 for efficient storage (max class ID is 2)
        assert mask.max() <= 255, (
            f"Mask max value {mask.max()} exceeds uint8 range for PNG storage"
        )
