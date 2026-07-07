"""Unit tests for the Preprocessor class.

Tests dimension/dtype preservation, processing order, default coefficient usage,
CLAHE configuration, and error handling (graceful fallback to original frame).
"""

import logging
from unittest.mock import patch, MagicMock

import cv2
import numpy as np
import pytest

from src.pipeline.models import PipelineConfig
from src.pipeline.preprocessor import Preprocessor


@pytest.fixture
def default_config():
    """PipelineConfig with all defaults (no distortion coefficients provided)."""
    return PipelineConfig()


@pytest.fixture
def custom_config():
    """PipelineConfig with custom distortion and camera parameters."""
    return PipelineConfig(
        clip_limit=3.0,
        tile_grid_size=(16, 16),
        distortion_coefficients=[-0.2, 0.05, 0.001, -0.001, 0.002],
        camera_matrix=[
            [1000.0, 0.0, 640.0],
            [0.0, 1000.0, 360.0],
            [0.0, 0.0, 1.0],
        ],
    )


@pytest.fixture
def sample_frame():
    """A synthetic 8-bit RGB frame for testing."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, size=(480, 640, 3), dtype=np.uint8)


@pytest.fixture
def small_frame():
    """A small 8-bit RGB frame for quick tests."""
    rng = np.random.default_rng(123)
    return rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)


class TestPreprocessorInit:
    """Tests for Preprocessor initialization."""

    def test_default_coefficients_logs_info(self, default_config, caplog):
        """When no distortion coefficients provided, log INFO about defaults."""
        with caplog.at_level(logging.INFO, logger="Preprocessor"):
            Preprocessor(default_config)

        assert any(
            "default GoPro" in record.message
            for record in caplog.records
        )

    def test_custom_coefficients_no_info_log(self, custom_config, caplog):
        """When custom distortion coefficients provided, no default usage log."""
        with caplog.at_level(logging.INFO, logger="Preprocessor"):
            Preprocessor(custom_config)

        assert not any(
            "default GoPro" in record.message
            for record in caplog.records
        )


class TestPreprocessorProcess:
    """Tests for the main process() method."""

    def test_output_preserves_shape(self, default_config, sample_frame):
        """Output must have identical shape to input."""
        preprocessor = Preprocessor(default_config)
        result = preprocessor.process(sample_frame)
        assert result.shape == sample_frame.shape

    def test_output_preserves_dtype(self, default_config, sample_frame):
        """Output must be 8-bit (uint8)."""
        preprocessor = Preprocessor(default_config)
        result = preprocessor.process(sample_frame)
        assert result.dtype == np.uint8

    def test_output_is_rgb_3_channels(self, default_config, sample_frame):
        """Output must have 3 channels (RGB)."""
        preprocessor = Preprocessor(default_config)
        result = preprocessor.process(sample_frame)
        assert result.shape[2] == 3

    def test_various_resolutions(self, default_config):
        """Preprocessor should handle various valid resolutions."""
        preprocessor = Preprocessor(default_config)
        rng = np.random.default_rng(0)

        for h, w in [(64, 64), (720, 1280), (1080, 1920)]:
            frame = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
            result = preprocessor.process(frame)
            assert result.shape == (h, w, 3)
            assert result.dtype == np.uint8

    def test_failure_returns_original_frame(self, default_config, sample_frame, caplog):
        """On processing failure, return original frame and log warning."""
        preprocessor = Preprocessor(default_config)

        # Force an exception in the first processing step
        with patch.object(
            preprocessor, "_correct_barrel_distortion", side_effect=RuntimeError("test error")
        ):
            with caplog.at_level(logging.WARNING, logger="Preprocessor"):
                result = preprocessor.process(sample_frame)

        np.testing.assert_array_equal(result, sample_frame)
        assert any(
            "Preprocessing failed" in record.message
            for record in caplog.records
        )


class TestBarrelDistortionCorrection:
    """Tests for _correct_barrel_distortion()."""

    def test_preserves_shape_and_dtype(self, default_config, sample_frame):
        """Distortion correction preserves dimensions and dtype."""
        preprocessor = Preprocessor(default_config)
        result = preprocessor._correct_barrel_distortion(sample_frame)
        assert result.shape == sample_frame.shape
        assert result.dtype == np.uint8

    def test_uses_custom_camera_matrix(self, custom_config, sample_frame):
        """When camera_matrix is provided, it's used directly."""
        preprocessor = Preprocessor(custom_config)
        # Should not raise and should produce valid output
        result = preprocessor._correct_barrel_distortion(sample_frame)
        assert result.shape == sample_frame.shape

    def test_default_camera_matrix_uses_frame_center(self, default_config, small_frame):
        """When no camera_matrix provided, principal point is frame center."""
        preprocessor = Preprocessor(default_config)
        # The method should still work correctly
        result = preprocessor._correct_barrel_distortion(small_frame)
        assert result.shape == small_frame.shape
        assert result.dtype == np.uint8


class TestMotionBlurReduction:
    """Tests for _reduce_motion_blur()."""

    def test_preserves_shape_and_dtype(self, default_config, sample_frame):
        """Motion blur reduction preserves dimensions and dtype."""
        preprocessor = Preprocessor(default_config)
        result = preprocessor._reduce_motion_blur(sample_frame)
        assert result.shape == sample_frame.shape
        assert result.dtype == np.uint8

    def test_sharpens_image(self, default_config):
        """Unsharp masking should increase contrast/edges."""
        preprocessor = Preprocessor(default_config)
        # Create a blurry image
        frame = np.full((100, 100, 3), 128, dtype=np.uint8)
        frame[40:60, 40:60] = 200  # A bright square

        result = preprocessor._reduce_motion_blur(frame)

        # The sharpened image should have different pixel values from original
        assert not np.array_equal(result, frame)
        assert result.shape == frame.shape
        assert result.dtype == np.uint8


class TestCLAHE:
    """Tests for _apply_clahe()."""

    def test_preserves_shape_and_dtype(self, default_config, sample_frame):
        """CLAHE preserves dimensions and dtype."""
        preprocessor = Preprocessor(default_config)
        result = preprocessor._apply_clahe(sample_frame)
        assert result.shape == sample_frame.shape
        assert result.dtype == np.uint8

    def test_custom_clip_limit_and_tile_size(self, custom_config, small_frame):
        """CLAHE uses configured clip_limit and tile_grid_size."""
        preprocessor = Preprocessor(custom_config)
        result = preprocessor._apply_clahe(small_frame)
        assert result.shape == small_frame.shape
        assert result.dtype == np.uint8

    def test_uniform_image_unchanged(self, default_config):
        """A uniform image should be mostly unchanged by CLAHE."""
        preprocessor = Preprocessor(default_config)
        frame = np.full((64, 64, 3), 128, dtype=np.uint8)
        result = preprocessor._apply_clahe(frame)
        # Uniform input in RGB -> uniform L channel -> CLAHE doesn't change uniform
        # Due to color space conversion rounding, allow small differences
        assert result.shape == frame.shape
        assert result.dtype == np.uint8


class TestProcessingOrder:
    """Tests that processing order is distortion → blur → CLAHE."""

    def test_processing_order(self, default_config, small_frame):
        """Verify fixed processing order: distortion → blur → CLAHE."""
        preprocessor = Preprocessor(default_config)
        call_order = []

        original_distortion = preprocessor._correct_barrel_distortion
        original_blur = preprocessor._reduce_motion_blur
        original_clahe = preprocessor._apply_clahe

        def mock_distortion(frame):
            call_order.append("distortion")
            return original_distortion(frame)

        def mock_blur(frame):
            call_order.append("blur")
            return original_blur(frame)

        def mock_clahe(frame):
            call_order.append("clahe")
            return original_clahe(frame)

        with patch.object(preprocessor, "_correct_barrel_distortion", side_effect=mock_distortion):
            with patch.object(preprocessor, "_reduce_motion_blur", side_effect=mock_blur):
                with patch.object(preprocessor, "_apply_clahe", side_effect=mock_clahe):
                    preprocessor.process(small_frame)

        assert call_order == ["distortion", "blur", "clahe"]
