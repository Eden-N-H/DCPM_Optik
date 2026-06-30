"""Unit tests for the SAM2Segmenter class.

Tests bbox validation, box prompt conversion, and segmentation logic
using mocked SAM2 predictor to avoid requiring the actual model.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.pipeline.models import Detection, PipelineConfig, SegmentationResult
from src.pipeline.sam2_segmenter import ModelLoadError, SAM2Segmenter


@pytest.fixture
def config():
    """Create a PipelineConfig with default values."""
    return PipelineConfig()


@pytest.fixture
def segmenter(config):
    """Create an SAM2Segmenter instance."""
    return SAM2Segmenter(config)


class TestValidateBbox:
    """Tests for SAM2Segmenter._validate_bbox()."""

    def test_valid_bbox(self, segmenter):
        """A bbox fully within frame boundaries is valid."""
        assert segmenter._validate_bbox((10, 20, 100, 50), (480, 640)) is True

    def test_zero_width(self, segmenter):
        """A bbox with zero width is invalid."""
        assert segmenter._validate_bbox((10, 20, 0, 50), (480, 640)) is False

    def test_zero_height(self, segmenter):
        """A bbox with zero height is invalid."""
        assert segmenter._validate_bbox((10, 20, 100, 0), (480, 640)) is False

    def test_negative_width(self, segmenter):
        """A bbox with negative width is invalid."""
        assert segmenter._validate_bbox((10, 20, -5, 50), (480, 640)) is False

    def test_negative_height(self, segmenter):
        """A bbox with negative height is invalid."""
        assert segmenter._validate_bbox((10, 20, 100, -3), (480, 640)) is False

    def test_negative_x(self, segmenter):
        """A bbox with negative x coordinate is invalid."""
        assert segmenter._validate_bbox((-1, 20, 100, 50), (480, 640)) is False

    def test_negative_y(self, segmenter):
        """A bbox with negative y coordinate is invalid."""
        assert segmenter._validate_bbox((10, -1, 100, 50), (480, 640)) is False

    def test_exceeds_frame_width(self, segmenter):
        """A bbox that extends beyond frame width is invalid."""
        assert segmenter._validate_bbox((600, 20, 100, 50), (480, 640)) is False

    def test_exceeds_frame_height(self, segmenter):
        """A bbox that extends beyond frame height is invalid."""
        assert segmenter._validate_bbox((10, 450, 100, 50), (480, 640)) is False

    def test_bbox_at_frame_edge(self, segmenter):
        """A bbox that exactly touches the frame edge is valid."""
        assert segmenter._validate_bbox((540, 380, 100, 100), (480, 640)) is True

    def test_bbox_one_pixel_over_width(self, segmenter):
        """A bbox that extends one pixel beyond width is invalid."""
        assert segmenter._validate_bbox((541, 380, 100, 100), (480, 640)) is False


class TestPrepareBoxPrompts:
    """Tests for SAM2Segmenter._prepare_box_prompts()."""

    def test_single_detection(self, segmenter):
        """Single detection is converted from (x,y,w,h) to (x1,y1,x2,y2)."""
        detections = [
            Detection(bbox=(10, 20, 100, 50), confidence=0.9, class_label="pothole")
        ]
        result = segmenter._prepare_box_prompts(detections)
        expected = np.array([[10, 20, 110, 70]], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_multiple_detections(self, segmenter):
        """Multiple detections are batch-converted correctly."""
        detections = [
            Detection(bbox=(10, 20, 100, 50), confidence=0.9, class_label="pothole"),
            Detection(
                bbox=(200, 300, 80, 60),
                confidence=0.8,
                class_label="longitudinal_crack",
            ),
        ]
        result = segmenter._prepare_box_prompts(detections)
        expected = np.array(
            [[10, 20, 110, 70], [200, 300, 280, 360]], dtype=np.float32
        )
        np.testing.assert_array_equal(result, expected)

    def test_output_dtype(self, segmenter):
        """Box prompts should be float32."""
        detections = [
            Detection(bbox=(0, 0, 50, 50), confidence=0.7, class_label="pothole")
        ]
        result = segmenter._prepare_box_prompts(detections)
        assert result.dtype == np.float32

    def test_output_shape(self, segmenter):
        """Box prompts should have shape (N, 4)."""
        detections = [
            Detection(bbox=(10, 20, 100, 50), confidence=0.9, class_label="pothole"),
            Detection(bbox=(200, 300, 80, 60), confidence=0.8, class_label="pothole"),
            Detection(bbox=(5, 5, 30, 30), confidence=0.6, class_label="pothole"),
        ]
        result = segmenter._prepare_box_prompts(detections)
        assert result.shape == (3, 4)


class TestSegment:
    """Tests for SAM2Segmenter.segment()."""

    def test_segment_without_load_raises(self, segmenter):
        """Calling segment() before load_model() raises RuntimeError."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = [
            Detection(bbox=(10, 20, 100, 50), confidence=0.9, class_label="pothole")
        ]
        with pytest.raises(RuntimeError, match="SAM2 model not loaded"):
            segmenter.segment(frame, detections)

    def test_segment_empty_detections(self, segmenter):
        """Empty detection list returns empty results."""
        segmenter.predictor = MagicMock()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = segmenter.segment(frame, [])
        assert result == []

    def test_segment_all_invalid_bboxes(self, segmenter):
        """All-invalid bboxes returns empty results without calling predict."""
        segmenter.predictor = MagicMock()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = [
            Detection(bbox=(10, 20, 0, 50), confidence=0.9, class_label="pothole"),
            Detection(bbox=(-1, 20, 100, 50), confidence=0.8, class_label="pothole"),
        ]
        result = segmenter.segment(frame, detections)
        assert result == []
        segmenter.predictor.predict.assert_not_called()

    def test_segment_returns_valid_results(self, segmenter):
        """Valid detections produce SegmentationResult objects."""
        mock_predictor = MagicMock()
        segmenter.predictor = mock_predictor

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = [
            Detection(bbox=(10, 20, 100, 50), confidence=0.9, class_label="pothole")
        ]

        # SAM2 returns masks of shape (N, 1, H, W) when multimask_output=False
        mock_mask = np.ones((1, 1, 480, 640), dtype=np.float32)
        mock_scores = np.array([[0.95]])
        mock_logits = np.zeros((1, 1, 480, 640))
        mock_predictor.predict.return_value = (mock_mask, mock_scores, mock_logits)

        results = segmenter.segment(frame, detections)

        assert len(results) == 1
        assert isinstance(results[0], SegmentationResult)
        assert results[0].mask.shape == (480, 640)
        assert results[0].mask.dtype == np.uint8
        assert results[0].detection == detections[0]
        assert results[0].bbox == (10, 20, 100, 50)

    def test_segment_discards_zero_foreground_mask(self, segmenter):
        """Masks with zero foreground pixels are discarded."""
        mock_predictor = MagicMock()
        segmenter.predictor = mock_predictor

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = [
            Detection(bbox=(10, 20, 100, 50), confidence=0.9, class_label="pothole")
        ]

        # Return an all-zero mask
        mock_mask = np.zeros((1, 1, 480, 640), dtype=np.float32)
        mock_scores = np.array([[0.3]])
        mock_logits = np.zeros((1, 1, 480, 640))
        mock_predictor.predict.return_value = (mock_mask, mock_scores, mock_logits)

        results = segmenter.segment(frame, detections)
        assert len(results) == 0

    def test_segment_skips_invalid_keeps_valid(self, segmenter):
        """Invalid bboxes are skipped; valid ones are processed."""
        mock_predictor = MagicMock()
        segmenter.predictor = mock_predictor

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = [
            # Invalid: zero width
            Detection(bbox=(10, 20, 0, 50), confidence=0.9, class_label="pothole"),
            # Valid
            Detection(
                bbox=(10, 20, 100, 50),
                confidence=0.8,
                class_label="longitudinal_crack",
            ),
        ]

        # Only 1 valid detection goes to predict
        mock_mask = np.ones((1, 1, 480, 640), dtype=np.float32)
        mock_scores = np.array([[0.9]])
        mock_logits = np.zeros((1, 1, 480, 640))
        mock_predictor.predict.return_value = (mock_mask, mock_scores, mock_logits)

        results = segmenter.segment(frame, detections)

        assert len(results) == 1
        assert results[0].detection.class_label == "longitudinal_crack"

    def test_segment_batches_all_valid(self, segmenter):
        """All valid detections are batched into a single predict call."""
        mock_predictor = MagicMock()
        segmenter.predictor = mock_predictor

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = [
            Detection(bbox=(10, 20, 100, 50), confidence=0.9, class_label="pothole"),
            Detection(
                bbox=(200, 100, 80, 60),
                confidence=0.8,
                class_label="transverse_crack",
            ),
        ]

        # Two valid detections → shape (2, 1, H, W)
        mock_mask = np.ones((2, 1, 480, 640), dtype=np.float32)
        mock_scores = np.array([[0.9], [0.85]])
        mock_logits = np.zeros((2, 1, 480, 640))
        mock_predictor.predict.return_value = (mock_mask, mock_scores, mock_logits)

        results = segmenter.segment(frame, detections)

        # Predict should be called exactly once (batched)
        mock_predictor.predict.assert_called_once()
        assert len(results) == 2

    def test_segment_mask_binary_values(self, segmenter):
        """Output masks contain only 0 and 1 values."""
        mock_predictor = MagicMock()
        segmenter.predictor = mock_predictor

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = [
            Detection(bbox=(10, 20, 100, 50), confidence=0.9, class_label="pothole")
        ]

        # Return a mask with float values > 0 (should be binarized)
        mock_mask = np.full((1, 1, 480, 640), 0.7, dtype=np.float32)
        mock_scores = np.array([[0.95]])
        mock_logits = np.zeros((1, 1, 480, 640))
        mock_predictor.predict.return_value = (mock_mask, mock_scores, mock_logits)

        results = segmenter.segment(frame, detections)

        assert len(results) == 1
        unique_values = np.unique(results[0].mask)
        assert set(unique_values).issubset({0, 1})


class TestLoadModel:
    """Tests for SAM2Segmenter.load_model()."""

    @patch("src.pipeline.sam2_segmenter.SAM2Segmenter._select_device")
    def test_load_model_import_error(self, mock_device, config):
        """ModelLoadError raised when sam2 package is not installed."""
        mock_device.return_value = "cpu"
        segmenter = SAM2Segmenter(config)

        with patch.dict("sys.modules", {"sam2": None, "sam2.sam2_image_predictor": None}):
            with pytest.raises(ModelLoadError, match="SAM2 is not installed"):
                segmenter.load_model()

    @patch("src.pipeline.sam2_segmenter.SAM2Segmenter._select_device")
    def test_load_model_checkpoint_failure(self, mock_device, config):
        """ModelLoadError raised when checkpoint file is missing/corrupt."""
        mock_device.return_value = "cpu"
        segmenter = SAM2Segmenter(config)

        mock_predictor_class = MagicMock()
        mock_predictor_class.from_pretrained.side_effect = FileNotFoundError(
            "Checkpoint not found"
        )

        with patch.dict(
            "sys.modules",
            {"sam2": MagicMock(), "sam2.sam2_image_predictor": MagicMock()},
        ):
            with patch(
                "sam2.sam2_image_predictor.SAM2ImagePredictor", mock_predictor_class
            ):
                with pytest.raises(ModelLoadError, match="Failed to load SAM2 model"):
                    segmenter.load_model()
