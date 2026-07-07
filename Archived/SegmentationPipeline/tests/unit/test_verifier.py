"""Unit tests for PostSegmentationVerifier.

Tests the core verification logic: area ratio computation,
connected component cleanup, and the verify() method's filtering/flagging.
"""

import numpy as np
import pytest

from src.pipeline.models import Detection, PipelineConfig, SegmentationResult, VerifiedResult
from src.pipeline.verifier import PostSegmentationVerifier


@pytest.fixture
def config() -> PipelineConfig:
    """Default pipeline config with standard verification thresholds."""
    return PipelineConfig(min_area_ratio=0.05, max_area_ratio=0.95)


@pytest.fixture
def verifier(config: PipelineConfig) -> PostSegmentationVerifier:
    """PostSegmentationVerifier with default config."""
    return PostSegmentationVerifier(config)


@pytest.fixture
def sample_detection() -> Detection:
    """A sample detection for testing."""
    return Detection(
        bbox=(10, 10, 50, 50),
        confidence=0.85,
        class_label="pothole",
    )


def _make_segmentation_result(
    mask: np.ndarray, bbox: tuple = (10, 10, 50, 50)
) -> SegmentationResult:
    """Helper to create a SegmentationResult with a given mask and bbox."""
    detection = Detection(bbox=bbox, confidence=0.85, class_label="pothole")
    return SegmentationResult(mask=mask, detection=detection, bbox=bbox)


class TestComputeAreaRatio:
    """Tests for _compute_area_ratio."""

    def test_full_foreground_in_bbox(self, verifier: PostSegmentationVerifier):
        """Mask completely filled within bbox region should give ratio 1.0."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        bbox = (10, 10, 20, 20)
        mask[10:30, 10:30] = 1  # Fill entire bbox region
        ratio = verifier._compute_area_ratio(mask, bbox)
        assert ratio == pytest.approx(1.0)

    def test_half_foreground_in_bbox(self, verifier: PostSegmentationVerifier):
        """Mask half-filled within bbox region should give ratio ~0.5."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        bbox = (10, 10, 20, 20)
        # Fill top half of the bbox region
        mask[10:20, 10:30] = 1
        ratio = verifier._compute_area_ratio(mask, bbox)
        assert ratio == pytest.approx(0.5)

    def test_empty_mask(self, verifier: PostSegmentationVerifier):
        """Empty mask should give ratio 0.0."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        bbox = (10, 10, 20, 20)
        ratio = verifier._compute_area_ratio(mask, bbox)
        assert ratio == pytest.approx(0.0)

    def test_zero_area_bbox(self, verifier: PostSegmentationVerifier):
        """Zero-area bounding box should return 0.0."""
        mask = np.ones((100, 100), dtype=np.uint8)
        bbox = (10, 10, 0, 20)
        ratio = verifier._compute_area_ratio(mask, bbox)
        assert ratio == pytest.approx(0.0)

    def test_foreground_outside_bbox_not_counted(self, verifier: PostSegmentationVerifier):
        """Foreground pixels outside the bbox region should not be counted."""
        mask = np.ones((100, 100), dtype=np.uint8)
        bbox = (10, 10, 20, 20)
        # Fill everything, then clear the bbox region
        mask[10:30, 10:30] = 0
        ratio = verifier._compute_area_ratio(mask, bbox)
        assert ratio == pytest.approx(0.0)


class TestCleanConnectedComponents:
    """Tests for _clean_connected_components."""

    def test_single_component_unchanged(self, verifier: PostSegmentationVerifier):
        """A mask with a single component should remain unchanged."""
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[10:20, 10:20] = 1
        cleaned = verifier._clean_connected_components(mask)
        np.testing.assert_array_equal(cleaned, mask)

    def test_retains_largest_component(self, verifier: PostSegmentationVerifier):
        """Should retain only the largest connected component."""
        mask = np.zeros((50, 50), dtype=np.uint8)
        # Large component: 10x10 = 100 pixels
        mask[5:15, 5:15] = 1
        # Small component: 3x3 = 9 pixels (disconnected)
        mask[30:33, 30:33] = 1
        cleaned = verifier._clean_connected_components(mask)
        # Large component should remain
        assert np.count_nonzero(cleaned[5:15, 5:15]) == 100
        # Small component should be removed
        assert np.count_nonzero(cleaned[30:33, 30:33]) == 0

    def test_empty_mask_returns_empty(self, verifier: PostSegmentationVerifier):
        """An empty mask should return an empty mask."""
        mask = np.zeros((50, 50), dtype=np.uint8)
        cleaned = verifier._clean_connected_components(mask)
        assert np.count_nonzero(cleaned) == 0

    def test_uses_8_connectivity(self, verifier: PostSegmentationVerifier):
        """Diagonal pixels should be connected under 8-connectivity."""
        mask = np.zeros((10, 10), dtype=np.uint8)
        # Diagonal line - connected under 8-connectivity
        mask[0, 0] = 1
        mask[1, 1] = 1
        mask[2, 2] = 1
        # Separate small component
        mask[8, 8] = 1
        cleaned = verifier._clean_connected_components(mask)
        # Diagonal should be the largest (3 pixels vs 1 pixel)
        assert cleaned[0, 0] == 1
        assert cleaned[1, 1] == 1
        assert cleaned[2, 2] == 1
        assert cleaned[8, 8] == 0

    def test_output_is_binary(self, verifier: PostSegmentationVerifier):
        """Cleaned mask should only contain 0 and 1 values."""
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[10:30, 10:30] = 1
        cleaned = verifier._clean_connected_components(mask)
        unique_values = np.unique(cleaned)
        assert all(v in [0, 1] for v in unique_values)


class TestVerify:
    """Tests for the verify() method."""

    def test_passes_valid_result(self, verifier: PostSegmentationVerifier):
        """A result with area ratio between thresholds should pass."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        bbox = (10, 10, 20, 20)
        # Fill about 50% of bbox region (200 out of 400 pixels)
        mask[10:20, 10:30] = 1
        result = _make_segmentation_result(mask, bbox)
        verified = verifier.verify([result])
        assert len(verified) == 1
        assert verified[0].review_flag is False
        assert verified[0].area_ratio == pytest.approx(0.5)

    def test_discards_below_min_ratio(self, verifier: PostSegmentationVerifier):
        """A result with very low area ratio should be discarded."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        bbox = (10, 10, 50, 50)
        # Only 1 pixel in a 50x50 = 2500 pixel bbox -> ratio = 0.0004
        mask[10, 10] = 1
        result = _make_segmentation_result(mask, bbox)
        verified = verifier.verify([result])
        assert len(verified) == 0

    def test_flags_above_max_ratio(self, verifier: PostSegmentationVerifier):
        """A result with area ratio above max should be flagged for review."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        bbox = (10, 10, 20, 20)
        # Fill entire bbox region -> ratio = 1.0 (above 0.95 max)
        mask[10:30, 10:30] = 1
        result = _make_segmentation_result(mask, bbox)
        verified = verifier.verify([result])
        assert len(verified) == 1
        assert verified[0].review_flag is True

    def test_multiple_results_mixed(self, verifier: PostSegmentationVerifier):
        """Multiple results: some pass, some discarded, some flagged."""
        # Result 1: will be discarded (very sparse mask)
        mask1 = np.zeros((100, 100), dtype=np.uint8)
        mask1[10, 10] = 1
        bbox1 = (10, 10, 50, 50)

        # Result 2: will pass (moderate fill)
        mask2 = np.zeros((100, 100), dtype=np.uint8)
        mask2[10:30, 10:30] = 1  # 400 pixels in 50x50 = 2500 -> ratio 0.16
        bbox2 = (10, 10, 50, 50)

        # Result 3: will be flagged (nearly full)
        mask3 = np.zeros((100, 100), dtype=np.uint8)
        mask3[10:30, 10:30] = 1  # 400 pixels in 20x20 = 400 -> ratio 1.0
        bbox3 = (10, 10, 20, 20)

        results = [
            _make_segmentation_result(mask1, bbox1),
            _make_segmentation_result(mask2, bbox2),
            _make_segmentation_result(mask3, bbox3),
        ]
        verified = verifier.verify(results)
        assert len(verified) == 2
        # Second result passes without flag
        assert verified[0].review_flag is False
        # Third result is flagged
        assert verified[1].review_flag is True

    def test_empty_input_returns_empty(self, verifier: PostSegmentationVerifier):
        """Empty input list should return empty output."""
        verified = verifier.verify([])
        assert verified == []

    def test_verified_result_contains_correct_detection(self, verifier: PostSegmentationVerifier):
        """Verified result should carry through the original detection info."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        bbox = (10, 10, 20, 20)
        mask[10:20, 10:30] = 1
        detection = Detection(bbox=bbox, confidence=0.92, class_label="longitudinal_crack")
        result = SegmentationResult(mask=mask, detection=detection, bbox=bbox)
        verified = verifier.verify([result])
        assert len(verified) == 1
        assert verified[0].detection.class_label == "longitudinal_crack"
        assert verified[0].detection.confidence == 0.92
        assert verified[0].bbox == bbox

    def test_cleans_components_before_ratio(self, verifier: PostSegmentationVerifier):
        """Connected component cleanup should happen before area ratio computation."""
        # Create a mask with a large component that will pass the ratio check
        # and a small disconnected component
        mask = np.zeros((100, 100), dtype=np.uint8)
        bbox = (0, 0, 50, 50)
        # Main component inside bbox: 20x20 = 400 pixels
        mask[5:25, 5:25] = 1
        # Small disconnected component inside bbox: 2x2 = 4 pixels
        mask[40:42, 40:42] = 1
        result = _make_segmentation_result(mask, bbox)
        verified = verifier.verify([result])
        assert len(verified) == 1
        # After cleaning, only the large component remains
        # The cleaned mask should have exactly 400 foreground pixels in total
        assert np.count_nonzero(verified[0].mask) == 400
