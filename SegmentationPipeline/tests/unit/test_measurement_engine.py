"""Unit tests for the MeasurementEngine class.

Tests pixel-based measurements, metric conversion, bounding dimensions,
severity classification, and fallback behavior when camera parameters
are missing or invalid.
"""

import numpy as np
import pytest

from src.pipeline.measurement_engine import MeasurementEngine
from src.pipeline.models import (
    DefectMeasurement,
    Detection,
    PipelineConfig,
    VerifiedResult,
)


def _make_verified_result(mask: np.ndarray) -> VerifiedResult:
    """Helper to create a VerifiedResult with a given mask."""
    detection = Detection(
        bbox=(10, 10, 50, 50),
        confidence=0.9,
        class_label="pothole",
    )
    return VerifiedResult(
        mask=mask,
        detection=detection,
        bbox=(10, 10, 50, 50),
        area_ratio=0.5,
        review_flag=False,
    )


class TestMeasurePixelArea:
    """Tests for area_pixels computation."""

    def test_area_pixels_counts_nonzero(self):
        """area_pixels should equal the count of non-zero pixels in the mask."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:40, 30:60] = 1  # 20 rows * 30 cols = 600 pixels
        config = PipelineConfig()
        engine = MeasurementEngine(config)
        result = engine.measure(_make_verified_result(mask))
        assert result.area_pixels == 600

    def test_area_pixels_empty_mask(self):
        """area_pixels should be 0 for an all-zero mask."""
        mask = np.zeros((50, 50), dtype=np.uint8)
        config = PipelineConfig()
        engine = MeasurementEngine(config)
        result = engine.measure(_make_verified_result(mask))
        assert result.area_pixels == 0

    def test_area_pixels_full_mask(self):
        """area_pixels should equal H*W for an all-ones mask."""
        mask = np.ones((30, 40), dtype=np.uint8)
        config = PipelineConfig()
        engine = MeasurementEngine(config)
        result = engine.measure(_make_verified_result(mask))
        assert result.area_pixels == 30 * 40


class TestBoundingDimensions:
    """Tests for bounding dimensions computation."""

    def test_bounding_dimensions_rectangular_region(self):
        """Should compute correct length and width from mask extent."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:50, 30:70] = 1  # rows 20-49, cols 30-69
        config = PipelineConfig()
        engine = MeasurementEngine(config)
        result = engine.measure(_make_verified_result(mask))
        # length = max_row - min_row = 49 - 20 = 29
        # width = max_col - min_col = 69 - 30 = 39
        assert result.length_pixels == 29
        assert result.width_pixels == 39

    def test_bounding_dimensions_single_pixel(self):
        """A single pixel should give dimensions of (0, 0)."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[50, 50] = 1
        config = PipelineConfig()
        engine = MeasurementEngine(config)
        result = engine.measure(_make_verified_result(mask))
        assert result.length_pixels == 0
        assert result.width_pixels == 0

    def test_bounding_dimensions_empty_mask(self):
        """Empty mask should give dimensions of (0, 0)."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        config = PipelineConfig()
        engine = MeasurementEngine(config)
        result = engine.measure(_make_verified_result(mask))
        assert result.length_pixels == 0
        assert result.width_pixels == 0


class TestMetricConversion:
    """Tests for pixel-to-cm conversion."""

    def test_area_cm2_computation(self):
        """area_cm2 should follow the formula: area_pixels * (height/focal)^2."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[0:10, 0:10] = 1  # 100 pixels
        config = PipelineConfig(camera_height_cm=200.0, focal_length_px=1000.0)
        engine = MeasurementEngine(config)
        result = engine.measure(_make_verified_result(mask))
        # pixel_size_cm = 200 / 1000 = 0.2
        # area_cm2 = 100 * 0.2^2 = 100 * 0.04 = 4.0
        assert result.area_cm2 == 4.0

    def test_width_cm_computation(self):
        """width_cm should equal width_pixels * (height/focal), rounded to 1dp."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:20, 10:60] = 1  # width_pixels = 49
        config = PipelineConfig(camera_height_cm=200.0, focal_length_px=1000.0)
        engine = MeasurementEngine(config)
        result = engine.measure(_make_verified_result(mask))
        # pixel_size_cm = 0.2, width_cm = 49 * 0.2 = 9.8
        assert result.width_cm == 9.8

    def test_length_cm_computation(self):
        """length_cm should equal length_pixels * (height/focal), rounded to 1dp."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:60, 10:20] = 1  # length_pixels = 49
        config = PipelineConfig(camera_height_cm=200.0, focal_length_px=1000.0)
        engine = MeasurementEngine(config)
        result = engine.measure(_make_verified_result(mask))
        # pixel_size_cm = 0.2, length_cm = 49 * 0.2 = 9.8
        assert result.length_cm == 9.8

    def test_metric_values_rounded_to_1dp(self):
        """Metric values should be rounded to 1 decimal place."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[0:7, 0:7] = 1  # 49 pixels, length_pixels = 6, width_pixels = 6
        config = PipelineConfig(camera_height_cm=150.0, focal_length_px=700.0)
        engine = MeasurementEngine(config)
        result = engine.measure(_make_verified_result(mask))
        # pixel_size_cm = 150/700 = 0.21428...
        # area_cm2 = 49 * 0.21428^2 = 49 * 0.04591... = 2.2497... -> 2.2
        # width_cm = 6 * 0.21428... = 1.2857... -> 1.3
        assert result.area_cm2 == round(49 * (150.0 / 700.0) ** 2, 1)
        assert result.width_cm == round(6 * (150.0 / 700.0), 1)
        assert result.length_cm == round(6 * (150.0 / 700.0), 1)


class TestSeverityClassification:
    """Tests for severity assignment."""

    def test_severity_minor(self):
        """Area < 500 cm² should be classified as minor."""
        config = PipelineConfig(camera_height_cm=100.0, focal_length_px=1000.0)
        engine = MeasurementEngine(config)
        # pixel_size_cm = 0.1, need area < 500 cm² -> pixels < 50000
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[0:10, 0:10] = 1  # 100 pixels -> 1.0 cm²
        result = engine.measure(_make_verified_result(mask))
        assert result.severity == "minor"

    def test_severity_moderate(self):
        """500 ≤ area ≤ 2000 cm² should be classified as moderate."""
        config = PipelineConfig(camera_height_cm=100.0, focal_length_px=100.0)
        engine = MeasurementEngine(config)
        # pixel_size_cm = 1.0, so area_cm2 = area_pixels
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[0:25, 0:25] = 1  # 625 pixels -> 625 cm²
        result = engine.measure(_make_verified_result(mask))
        assert result.severity == "moderate"

    def test_severity_severe(self):
        """Area > 2000 cm² should be classified as severe."""
        config = PipelineConfig(camera_height_cm=100.0, focal_length_px=100.0)
        engine = MeasurementEngine(config)
        # pixel_size_cm = 1.0, so area_cm2 = area_pixels
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[0:50, 0:50] = 1  # 2500 pixels -> 2500 cm²
        result = engine.measure(_make_verified_result(mask))
        assert result.severity == "severe"

    def test_severity_at_boundary_500(self):
        """Area exactly 500 cm² should be classified as moderate."""
        config = PipelineConfig(camera_height_cm=100.0, focal_length_px=100.0)
        engine = MeasurementEngine(config)
        # pixel_size_cm = 1.0, need exactly 500 pixels
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[0:25, 0:20] = 1  # 500 pixels -> 500 cm²
        result = engine.measure(_make_verified_result(mask))
        assert result.severity == "moderate"

    def test_severity_at_boundary_2000(self):
        """Area exactly 2000 cm² should be classified as moderate."""
        config = PipelineConfig(camera_height_cm=100.0, focal_length_px=100.0)
        engine = MeasurementEngine(config)
        # pixel_size_cm = 1.0, need exactly 2000 pixels
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[0:50, 0:40] = 1  # 2000 pixels -> 2000 cm²
        result = engine.measure(_make_verified_result(mask))
        assert result.severity == "moderate"


class TestCameraParamsFallback:
    """Tests for behavior when camera parameters are missing or invalid."""

    def test_no_camera_params_returns_none_metrics(self):
        """When no camera params provided, metric fields and severity are None."""
        config = PipelineConfig()  # camera_height_cm=None, focal_length_px=None
        engine = MeasurementEngine(config)
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:30, 10:40] = 1
        result = engine.measure(_make_verified_result(mask))
        assert result.area_cm2 is None
        assert result.width_cm is None
        assert result.length_cm is None
        assert result.severity is None
        # Pixel measurements should still be computed
        assert result.area_pixels == 600
        assert result.length_pixels == 19
        assert result.width_pixels == 29

    def test_only_height_provided_returns_none_metrics(self):
        """When only camera_height_cm is provided, metric fields are None."""
        config = PipelineConfig(camera_height_cm=200.0)
        engine = MeasurementEngine(config)
        mask = np.ones((10, 10), dtype=np.uint8)
        result = engine.measure(_make_verified_result(mask))
        assert result.area_cm2 is None
        assert result.severity is None

    def test_only_focal_length_provided_returns_none_metrics(self):
        """When only focal_length_px is provided, metric fields are None."""
        config = PipelineConfig(focal_length_px=1000.0)
        engine = MeasurementEngine(config)
        mask = np.ones((10, 10), dtype=np.uint8)
        result = engine.measure(_make_verified_result(mask))
        assert result.area_cm2 is None
        assert result.severity is None

    def test_zero_camera_height_returns_none_metrics(self, caplog):
        """Zero camera_height_cm should log WARNING and return None metrics."""
        import logging

        with caplog.at_level(logging.WARNING):
            config = PipelineConfig(camera_height_cm=0.0, focal_length_px=1000.0)
            engine = MeasurementEngine(config)
        assert "zero or negative" in caplog.text.lower()
        mask = np.ones((10, 10), dtype=np.uint8)
        result = engine.measure(_make_verified_result(mask))
        assert result.area_cm2 is None
        assert result.severity is None

    def test_negative_focal_length_returns_none_metrics(self, caplog):
        """Negative focal_length_px should log WARNING and return None metrics."""
        import logging

        with caplog.at_level(logging.WARNING):
            config = PipelineConfig(camera_height_cm=200.0, focal_length_px=-5.0)
            engine = MeasurementEngine(config)
        assert "zero or negative" in caplog.text.lower()
        mask = np.ones((10, 10), dtype=np.uint8)
        result = engine.measure(_make_verified_result(mask))
        assert result.area_cm2 is None
        assert result.severity is None

    def test_zero_focal_length_returns_none_metrics(self, caplog):
        """Zero focal_length_px should log WARNING and return None metrics."""
        import logging

        with caplog.at_level(logging.WARNING):
            config = PipelineConfig(camera_height_cm=200.0, focal_length_px=0.0)
            engine = MeasurementEngine(config)
        assert "zero or negative" in caplog.text.lower()
        mask = np.ones((10, 10), dtype=np.uint8)
        result = engine.measure(_make_verified_result(mask))
        assert result.area_cm2 is None
        assert result.severity is None

    def test_negative_camera_height_returns_none_metrics(self, caplog):
        """Negative camera_height_cm should log WARNING and return None metrics."""
        import logging

        with caplog.at_level(logging.WARNING):
            config = PipelineConfig(camera_height_cm=-100.0, focal_length_px=1000.0)
            engine = MeasurementEngine(config)
        assert "zero or negative" in caplog.text.lower()
        mask = np.ones((10, 10), dtype=np.uint8)
        result = engine.measure(_make_verified_result(mask))
        assert result.area_cm2 is None
        assert result.severity is None
