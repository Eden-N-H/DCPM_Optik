"""Unit tests for the YOLODetector class.

Tests the YOLO detection pipeline including model loading, inference,
class label mapping, and error handling.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.pipeline.models import Detection, PipelineConfig
from src.pipeline.yolo_detector import (
    CLASS_INDEX_TO_LABEL,
    ModelLoadError,
    YOLODetector,
)


@pytest.fixture
def config() -> PipelineConfig:
    """Create a default pipeline config for testing."""
    return PipelineConfig(
        yolo_model_path="models/test_model.pt",
        confidence_threshold=0.5,
        iou_threshold=0.45,
        max_detections=50,
    )


@pytest.fixture
def sample_frame() -> np.ndarray:
    """Create a sample RGB frame for testing."""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def mock_yolo_model():
    """Create a mock YOLO model with realistic result structure."""
    model = MagicMock()
    model.to = MagicMock(return_value=model)
    return model


def _make_mock_results(detections_data: list) -> list:
    """Helper to create mock YOLO results.

    Args:
        detections_data: List of tuples (x1, y1, x2, y2, conf, class_idx)
    """
    if not detections_data:
        result = MagicMock()
        result.boxes = MagicMock()
        result.boxes.__len__ = MagicMock(return_value=0)
        return [result]

    import torch

    result = MagicMock()
    boxes = MagicMock()

    n = len(detections_data)
    boxes.__len__ = MagicMock(return_value=n)

    xyxy_list = []
    conf_list = []
    cls_list = []

    for x1, y1, x2, y2, conf, cls_idx in detections_data:
        xyxy_list.append(torch.tensor([x1, y1, x2, y2], dtype=torch.float32))
        conf_list.append(torch.tensor(conf, dtype=torch.float32))
        cls_list.append(torch.tensor(cls_idx, dtype=torch.float32))

    boxes.xyxy = xyxy_list
    boxes.conf = conf_list
    boxes.cls = cls_list

    result.boxes = boxes
    return [result]


class TestYOLODetectorInit:
    """Tests for YOLODetector initialization."""

    def test_init_stores_config(self, config: PipelineConfig) -> None:
        """Detector stores the provided config."""
        with patch("src.pipeline.yolo_detector.torch") as mock_torch:
            mock_torch.backends.mps.is_available.return_value = False
            detector = YOLODetector(config)
        assert detector.config is config

    def test_init_model_is_none(self, config: PipelineConfig) -> None:
        """Model is None before load_model is called."""
        with patch("src.pipeline.yolo_detector.torch") as mock_torch:
            mock_torch.backends.mps.is_available.return_value = False
            detector = YOLODetector(config)
        assert detector.model is None

    def test_init_selects_mps_when_available(self, config: PipelineConfig) -> None:
        """Device is set to mps when MPS is available."""
        with patch("src.pipeline.yolo_detector.torch") as mock_torch:
            mock_torch.backends.mps.is_available.return_value = True
            detector = YOLODetector(config)
        assert detector._device == "mps"

    def test_init_falls_back_to_cpu(self, config: PipelineConfig) -> None:
        """Device falls back to cpu when MPS is not available."""
        with patch("src.pipeline.yolo_detector.torch") as mock_torch:
            mock_torch.backends.mps.is_available.return_value = False
            detector = YOLODetector(config)
        assert detector._device == "cpu"


class TestLoadModel:
    """Tests for YOLODetector.load_model()."""

    def test_load_model_success(self, config: PipelineConfig) -> None:
        """Model is loaded successfully using Ultralytics API."""
        with patch("src.pipeline.yolo_detector.torch") as mock_torch:
            mock_torch.backends.mps.is_available.return_value = False
            detector = YOLODetector(config)

        mock_model = MagicMock()
        mock_model.to = MagicMock(return_value=mock_model)

        with patch(
            "src.pipeline.yolo_detector.YOLO", create=True
        ) as mock_yolo_cls:
            # Patch the import inside load_model
            with patch.dict(
                "sys.modules",
                {"ultralytics": MagicMock(YOLO=mock_yolo_cls)},
            ):
                mock_yolo_cls.return_value = mock_model
                detector.load_model()

        assert detector.model is mock_model
        mock_model.to.assert_called_once_with("cpu")

    def test_load_model_raises_model_load_error_on_failure(
        self, config: PipelineConfig
    ) -> None:
        """ModelLoadError is raised when model loading fails."""
        with patch("src.pipeline.yolo_detector.torch") as mock_torch:
            mock_torch.backends.mps.is_available.return_value = False
            detector = YOLODetector(config)

        with patch.dict(
            "sys.modules",
            {
                "ultralytics": MagicMock(
                    YOLO=MagicMock(side_effect=FileNotFoundError("weights not found"))
                )
            },
        ):
            with pytest.raises(ModelLoadError) as exc_info:
                detector.load_model()

        assert "models/test_model.pt" in str(exc_info.value)

    def test_load_model_error_message_includes_path(
        self, config: PipelineConfig
    ) -> None:
        """Error message includes the model path for diagnosis."""
        config.yolo_model_path = "/nonexistent/path/model.pt"
        with patch("src.pipeline.yolo_detector.torch") as mock_torch:
            mock_torch.backends.mps.is_available.return_value = False
            detector = YOLODetector(config)

        with patch.dict(
            "sys.modules",
            {
                "ultralytics": MagicMock(
                    YOLO=MagicMock(side_effect=RuntimeError("corrupt weights"))
                )
            },
        ):
            with pytest.raises(ModelLoadError) as exc_info:
                detector.load_model()

        assert "/nonexistent/path/model.pt" in str(exc_info.value)


class TestDetect:
    """Tests for YOLODetector.detect()."""

    def test_detect_raises_if_model_not_loaded(
        self, config: PipelineConfig, sample_frame: np.ndarray
    ) -> None:
        """RuntimeError raised when detect called before load_model."""
        with patch("src.pipeline.yolo_detector.torch") as mock_torch:
            mock_torch.backends.mps.is_available.return_value = False
            detector = YOLODetector(config)

        with pytest.raises(RuntimeError, match="not loaded"):
            detector.detect(sample_frame)

    def test_detect_returns_empty_list_no_detections(
        self, config: PipelineConfig, sample_frame: np.ndarray
    ) -> None:
        """Returns empty list when no detections found."""
        with patch("src.pipeline.yolo_detector.torch") as mock_torch:
            mock_torch.backends.mps.is_available.return_value = False
            detector = YOLODetector(config)

        mock_model = MagicMock()
        mock_model.predict.return_value = _make_mock_results([])
        detector.model = mock_model

        result = detector.detect(sample_frame, frame_id="test_001")
        assert result == []

    def test_detect_returns_empty_list_none_results(
        self, config: PipelineConfig, sample_frame: np.ndarray
    ) -> None:
        """Returns empty list when results are None or empty."""
        with patch("src.pipeline.yolo_detector.torch") as mock_torch:
            mock_torch.backends.mps.is_available.return_value = False
            detector = YOLODetector(config)

        mock_model = MagicMock()
        mock_model.predict.return_value = []
        detector.model = mock_model

        result = detector.detect(sample_frame, frame_id="test_002")
        assert result == []

    def test_detect_single_detection(
        self, config: PipelineConfig, sample_frame: np.ndarray
    ) -> None:
        """Single detection is correctly mapped to Detection object."""
        with patch("src.pipeline.yolo_detector.torch") as mock_torch:
            mock_torch.backends.mps.is_available.return_value = False
            detector = YOLODetector(config)

        # Detection: x1=100, y1=200, x2=300, y2=400, conf=0.85, class=0 (pothole)
        mock_model = MagicMock()
        mock_model.predict.return_value = _make_mock_results(
            [(100, 200, 300, 400, 0.85, 0)]
        )
        detector.model = mock_model

        result = detector.detect(sample_frame, frame_id="test_003")

        assert len(result) == 1
        assert result[0].bbox == (100, 200, 200, 200)
        assert result[0].confidence == pytest.approx(0.85, abs=1e-5)
        assert result[0].class_label == "pothole"

    def test_detect_multiple_detections(
        self, config: PipelineConfig, sample_frame: np.ndarray
    ) -> None:
        """Multiple detections are returned correctly."""
        with patch("src.pipeline.yolo_detector.torch") as mock_torch:
            mock_torch.backends.mps.is_available.return_value = False
            detector = YOLODetector(config)

        mock_model = MagicMock()
        mock_model.predict.return_value = _make_mock_results(
            [
                (10, 20, 110, 120, 0.9, 0),  # pothole
                (200, 300, 350, 450, 0.7, 1),  # longitudinal_crack
                (400, 100, 500, 200, 0.6, 4),  # patch_deterioration
            ]
        )
        detector.model = mock_model

        result = detector.detect(sample_frame)

        assert len(result) == 3
        assert result[0].class_label == "pothole"
        assert result[1].class_label == "longitudinal_crack"
        assert result[2].class_label == "patch_deterioration"

    def test_detect_all_class_labels(
        self, config: PipelineConfig, sample_frame: np.ndarray
    ) -> None:
        """All five defect classes are correctly mapped."""
        with patch("src.pipeline.yolo_detector.torch") as mock_torch:
            mock_torch.backends.mps.is_available.return_value = False
            detector = YOLODetector(config)

        mock_model = MagicMock()
        mock_model.predict.return_value = _make_mock_results(
            [
                (0, 0, 50, 50, 0.8, 0),  # pothole
                (0, 0, 50, 50, 0.8, 1),  # longitudinal_crack
                (0, 0, 50, 50, 0.8, 2),  # transverse_crack
                (0, 0, 50, 50, 0.8, 3),  # alligator_cracking
                (0, 0, 50, 50, 0.8, 4),  # patch_deterioration
            ]
        )
        detector.model = mock_model

        result = detector.detect(sample_frame)

        labels = [d.class_label for d in result]
        assert labels == [
            "pothole",
            "longitudinal_crack",
            "transverse_crack",
            "alligator_cracking",
            "patch_deterioration",
        ]

    def test_detect_skips_unknown_class_index(
        self, config: PipelineConfig, sample_frame: np.ndarray
    ) -> None:
        """Unknown class indices are skipped with a warning."""
        with patch("src.pipeline.yolo_detector.torch") as mock_torch:
            mock_torch.backends.mps.is_available.return_value = False
            detector = YOLODetector(config)

        mock_model = MagicMock()
        mock_model.predict.return_value = _make_mock_results(
            [
                (0, 0, 50, 50, 0.8, 0),  # pothole - valid
                (0, 0, 50, 50, 0.8, 99),  # unknown - skipped
            ]
        )
        detector.model = mock_model

        result = detector.detect(sample_frame)
        assert len(result) == 1
        assert result[0].class_label == "pothole"

    def test_detect_passes_config_params_to_predict(
        self, config: PipelineConfig, sample_frame: np.ndarray
    ) -> None:
        """Confidence, IoU threshold, and max_detections are passed to predict."""
        config.confidence_threshold = 0.7
        config.iou_threshold = 0.3
        config.max_detections = 25

        with patch("src.pipeline.yolo_detector.torch") as mock_torch:
            mock_torch.backends.mps.is_available.return_value = False
            detector = YOLODetector(config)

        mock_model = MagicMock()
        mock_model.predict.return_value = _make_mock_results([])
        detector.model = mock_model

        detector.detect(sample_frame, frame_id="test_params")

        mock_model.predict.assert_called_once_with(
            sample_frame,
            conf=0.7,
            iou=0.3,
            max_det=25,
            device="cpu",
            verbose=False,
        )

    def test_detect_returns_empty_on_inference_error(
        self, config: PipelineConfig, sample_frame: np.ndarray
    ) -> None:
        """Returns empty list and logs error on inference failure."""
        with patch("src.pipeline.yolo_detector.torch") as mock_torch:
            mock_torch.backends.mps.is_available.return_value = False
            detector = YOLODetector(config)

        mock_model = MagicMock()
        mock_model.predict.side_effect = RuntimeError("GPU out of memory")
        detector.model = mock_model

        result = detector.detect(sample_frame, frame_id="error_frame")
        assert result == []

    def test_detect_bbox_conversion_xyxy_to_xywh(
        self, config: PipelineConfig, sample_frame: np.ndarray
    ) -> None:
        """Bounding box is correctly converted from xyxy to xywh format."""
        with patch("src.pipeline.yolo_detector.torch") as mock_torch:
            mock_torch.backends.mps.is_available.return_value = False
            detector = YOLODetector(config)

        # x1=50, y1=75, x2=250, y2=175 -> x=50, y=75, w=200, h=100
        mock_model = MagicMock()
        mock_model.predict.return_value = _make_mock_results(
            [(50, 75, 250, 175, 0.9, 2)]
        )
        detector.model = mock_model

        result = detector.detect(sample_frame)

        assert result[0].bbox == (50, 75, 200, 100)


class TestClassIndexMapping:
    """Tests for the class index to label mapping."""

    def test_mapping_has_five_classes(self) -> None:
        """Mapping contains exactly five defect classes."""
        assert len(CLASS_INDEX_TO_LABEL) == 5

    def test_mapping_values(self) -> None:
        """All expected class labels are present."""
        expected = {
            "pothole",
            "longitudinal_crack",
            "transverse_crack",
            "alligator_cracking",
            "patch_deterioration",
        }
        assert set(CLASS_INDEX_TO_LABEL.values()) == expected

    def test_mapping_indices_are_sequential(self) -> None:
        """Class indices are sequential from 0 to 4."""
        assert set(CLASS_INDEX_TO_LABEL.keys()) == {0, 1, 2, 3, 4}
