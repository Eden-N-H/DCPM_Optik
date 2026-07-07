"""Unit tests for the PipelineOrchestrator and FailureTracker.

Tests the orchestrator's error handling logic, failure tracking, and
frame processing coordination using mocked pipeline components.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest

from src.pipeline.orchestrator import (
    FailureTracker,
    PipelineOrchestrator,
    SystematicFailureError,
)
from src.pipeline.frame_ingester import UnsupportedFormatError
from src.pipeline.models import (
    BatchSummary,
    DefectOutput,
    Detection,
    FrameMetadata,
    FrameResult,
    PipelineConfig,
    SegmentationResult,
    VerifiedResult,
)


class TestFailureTracker:
    """Tests for the FailureTracker helper class."""

    def test_init_sets_max_and_zero_count(self):
        tracker = FailureTracker(max_consecutive=5)
        assert tracker.max_consecutive == 5
        assert tracker.consecutive_count == 0

    def test_record_failure_increments_count(self):
        tracker = FailureTracker(max_consecutive=10)
        tracker.record_failure()
        assert tracker.consecutive_count == 1
        tracker.record_failure()
        assert tracker.consecutive_count == 2

    def test_record_success_resets_count(self):
        tracker = FailureTracker(max_consecutive=10)
        tracker.record_failure()
        tracker.record_failure()
        tracker.record_success()
        assert tracker.consecutive_count == 0

    def test_raises_on_exceeding_threshold(self):
        tracker = FailureTracker(max_consecutive=3)
        tracker.record_failure()  # 1
        tracker.record_failure()  # 2
        tracker.record_failure()  # 3
        with pytest.raises(SystematicFailureError):
            tracker.record_failure()  # 4 > 3 → raise

    def test_no_raise_at_threshold(self):
        """Should not raise when count equals max (only when it exceeds)."""
        tracker = FailureTracker(max_consecutive=3)
        tracker.record_failure()  # 1
        tracker.record_failure()  # 2
        tracker.record_failure()  # 3 == max → no raise
        assert tracker.consecutive_count == 3

    def test_reset_then_fail_again(self):
        """Counter resets on success and can track new failures."""
        tracker = FailureTracker(max_consecutive=2)
        tracker.record_failure()
        tracker.record_failure()
        tracker.record_success()  # reset
        tracker.record_failure()  # 1 again
        tracker.record_failure()  # 2 again == max, no raise
        assert tracker.consecutive_count == 2

    def test_threshold_of_one(self):
        """With max=1, the second failure should raise."""
        tracker = FailureTracker(max_consecutive=1)
        tracker.record_failure()  # 1 == max → no raise
        with pytest.raises(SystematicFailureError):
            tracker.record_failure()  # 2 > 1 → raise


class TestPipelineOrchestratorInit:
    """Tests for PipelineOrchestrator initialization and error handling."""

    @patch("src.pipeline.orchestrator.SAM2Segmenter")
    @patch("src.pipeline.orchestrator.YOLODetector")
    @patch("src.pipeline.orchestrator.OutputWriter")
    @patch("src.pipeline.orchestrator.setup_logging")
    @patch("src.pipeline.orchestrator.ConfigManager")
    def test_init_with_valid_config(
        self,
        mock_config_mgr_cls,
        mock_setup_logging,
        mock_output_writer_cls,
        mock_yolo_cls,
        mock_sam2_cls,
        tmp_path,
    ):
        """Orchestrator initializes all components with valid config."""
        # Setup config manager mock
        config = PipelineConfig(output_directory=str(tmp_path))
        mock_config_mgr = MagicMock()
        mock_config_mgr.load.return_value = config
        mock_config_mgr.validate.return_value = []
        mock_config_mgr_cls.return_value = mock_config_mgr

        # Setup model mocks
        mock_yolo = MagicMock()
        mock_yolo_cls.return_value = mock_yolo
        mock_sam2 = MagicMock()
        mock_sam2_cls.return_value = mock_sam2

        orchestrator = PipelineOrchestrator(str(tmp_path / "config.yaml"))

        mock_config_mgr.load.assert_called_once()
        mock_config_mgr.validate.assert_called_once_with(config)
        mock_yolo.load_model.assert_called_once()
        mock_sam2.load_model.assert_called_once()

    @patch("src.pipeline.orchestrator.setup_logging")
    @patch("src.pipeline.orchestrator.ConfigManager")
    def test_init_exits_on_config_validation_errors(
        self, mock_config_mgr_cls, mock_setup_logging
    ):
        """Orchestrator exits with code 1 on configuration validation errors."""
        config = PipelineConfig()
        mock_config_mgr = MagicMock()
        mock_config_mgr.load.return_value = config
        mock_config_mgr.validate.return_value = [
            "confidence_threshold: out of range",
            "iou_threshold: out of range",
        ]
        mock_config_mgr_cls.return_value = mock_config_mgr

        with pytest.raises(SystemExit) as exc_info:
            PipelineOrchestrator("config.yaml")
        assert exc_info.value.code == 1

    @patch("src.pipeline.orchestrator.OutputWriter")
    @patch("src.pipeline.orchestrator.setup_logging")
    @patch("src.pipeline.orchestrator.ConfigManager")
    def test_init_exits_on_yolo_model_load_failure(
        self, mock_config_mgr_cls, mock_setup_logging, mock_output_writer_cls, tmp_path
    ):
        """Orchestrator exits with code 1 when YOLO model fails to load."""
        from src.pipeline.yolo_detector import ModelLoadError

        config = PipelineConfig(output_directory=str(tmp_path))
        mock_config_mgr = MagicMock()
        mock_config_mgr.load.return_value = config
        mock_config_mgr.validate.return_value = []
        mock_config_mgr_cls.return_value = mock_config_mgr

        with patch("src.pipeline.orchestrator.YOLODetector") as mock_yolo_cls:
            mock_yolo = MagicMock()
            mock_yolo.load_model.side_effect = ModelLoadError("weights not found")
            mock_yolo_cls.return_value = mock_yolo

            with pytest.raises(SystemExit) as exc_info:
                PipelineOrchestrator(str(tmp_path / "config.yaml"))
            assert exc_info.value.code == 1

    @patch("src.pipeline.orchestrator.YOLODetector")
    @patch("src.pipeline.orchestrator.OutputWriter")
    @patch("src.pipeline.orchestrator.setup_logging")
    @patch("src.pipeline.orchestrator.ConfigManager")
    def test_init_exits_on_sam2_model_load_failure(
        self,
        mock_config_mgr_cls,
        mock_setup_logging,
        mock_output_writer_cls,
        mock_yolo_cls,
        tmp_path,
    ):
        """Orchestrator exits with code 1 when SAM2 model fails to load."""
        from src.pipeline.sam2_segmenter import ModelLoadError as SAM2ModelLoadError

        config = PipelineConfig(output_directory=str(tmp_path))
        mock_config_mgr = MagicMock()
        mock_config_mgr.load.return_value = config
        mock_config_mgr.validate.return_value = []
        mock_config_mgr_cls.return_value = mock_config_mgr

        mock_yolo = MagicMock()
        mock_yolo_cls.return_value = mock_yolo

        with patch("src.pipeline.orchestrator.SAM2Segmenter") as mock_sam2_cls:
            mock_sam2 = MagicMock()
            mock_sam2.load_model.side_effect = SAM2ModelLoadError("checkpoint missing")
            mock_sam2_cls.return_value = mock_sam2

            with pytest.raises(SystemExit) as exc_info:
                PipelineOrchestrator(str(tmp_path / "config.yaml"))
            assert exc_info.value.code == 1


class TestPipelineOrchestratorRun:
    """Tests for the orchestrator's run() method and frame processing."""

    def _create_orchestrator(self, tmp_path):
        """Helper to create an orchestrator with all components mocked."""
        with patch("src.pipeline.orchestrator.ConfigManager") as mock_cm_cls, \
             patch("src.pipeline.orchestrator.setup_logging"), \
             patch("src.pipeline.orchestrator.OutputWriter") as mock_ow_cls, \
             patch("src.pipeline.orchestrator.YOLODetector") as mock_yolo_cls, \
             patch("src.pipeline.orchestrator.SAM2Segmenter") as mock_sam2_cls:

            config = PipelineConfig(
                output_directory=str(tmp_path),
                max_consecutive_failures=3,
            )
            mock_cm = MagicMock()
            mock_cm.load.return_value = config
            mock_cm.validate.return_value = []
            mock_cm_cls.return_value = mock_cm

            mock_yolo = MagicMock()
            mock_yolo_cls.return_value = mock_yolo

            mock_sam2 = MagicMock()
            mock_sam2_cls.return_value = mock_sam2

            mock_ow = MagicMock()
            mock_ow_cls.return_value = mock_ow

            orchestrator = PipelineOrchestrator(str(tmp_path / "config.yaml"))

        return orchestrator

    def test_run_processes_frames_and_writes_summary(self, tmp_path):
        """run() processes frames from ingested input and writes batch summary."""
        orchestrator = self._create_orchestrator(tmp_path)

        # Mock frame ingester to yield one frame
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        metadata = FrameMetadata(
            frame_id="test_frame_0001",
            source_path="/tmp/test.jpg",
            timestamp="2024-03-15T10:00:00Z",
            width=100,
            height=100,
        )
        orchestrator._frame_ingester = MagicMock()
        orchestrator._frame_ingester.ingest.return_value = iter([(frame, metadata)])

        # Mock preprocessor
        orchestrator._preprocessor = MagicMock()
        orchestrator._preprocessor.process.return_value = frame

        # Mock detector - no detections
        orchestrator._yolo_detector = MagicMock()
        orchestrator._yolo_detector.detect.return_value = []

        # Mock output writer
        orchestrator._output_writer = MagicMock()

        orchestrator.run(["/tmp/test.jpg"])

        orchestrator._output_writer.write_frame_result.assert_called_once()
        orchestrator._output_writer.write_batch_summary.assert_called_once()

        # Check batch summary
        summary_call = orchestrator._output_writer.write_batch_summary.call_args[0][0]
        assert summary_call.total_frames_processed == 1
        assert summary_call.total_defects_detected == 0

    def test_run_handles_unsupported_format_and_continues(self, tmp_path):
        """run() logs and continues when an input path raises UnsupportedFormatError."""
        orchestrator = self._create_orchestrator(tmp_path)

        orchestrator._frame_ingester = MagicMock()
        orchestrator._frame_ingester.ingest.side_effect = UnsupportedFormatError(
            "Unsupported"
        )
        orchestrator._output_writer = MagicMock()

        # Should not raise, should continue
        orchestrator.run(["/tmp/test.bmp"])

        orchestrator._output_writer.write_batch_summary.assert_called_once()

    def test_run_exits_on_systematic_failure(self, tmp_path):
        """run() exits with code 1 when consecutive failures exceed threshold."""
        orchestrator = self._create_orchestrator(tmp_path)

        # Create frames that will all fail
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frames = [
            (
                frame,
                FrameMetadata(
                    frame_id=f"frame_{i}",
                    source_path="/tmp/test.mp4",
                    timestamp="2024-03-15T10:00:00Z",
                    width=100,
                    height=100,
                ),
            )
            for i in range(5)
        ]

        orchestrator._frame_ingester = MagicMock()
        orchestrator._frame_ingester.ingest.return_value = iter(frames)

        # Make preprocessing raise to simulate frame failure
        orchestrator._preprocessor = MagicMock()
        orchestrator._preprocessor.process.side_effect = RuntimeError("fail")

        orchestrator._output_writer = MagicMock()

        with pytest.raises(SystemExit) as exc_info:
            orchestrator.run(["/tmp/test.mp4"])
        assert exc_info.value.code == 1

    def test_run_recovers_from_intermittent_failures(self, tmp_path):
        """run() resets failure counter on successful frame processing."""
        orchestrator = self._create_orchestrator(tmp_path)

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frames = [
            (
                frame,
                FrameMetadata(
                    frame_id=f"frame_{i}",
                    source_path="/tmp/test.mp4",
                    timestamp="2024-03-15T10:00:00Z",
                    width=100,
                    height=100,
                ),
            )
            for i in range(4)
        ]

        orchestrator._frame_ingester = MagicMock()
        orchestrator._frame_ingester.ingest.return_value = iter(frames)

        # First 2 fail, then succeed, then fail again - should not exit
        call_count = [0]

        def process_side_effect(f):
            call_count[0] += 1
            idx = call_count[0]
            if idx <= 2:
                raise RuntimeError("transient failure")
            return f

        orchestrator._preprocessor = MagicMock()
        orchestrator._preprocessor.process.side_effect = process_side_effect

        # For the successful frames
        orchestrator._yolo_detector = MagicMock()
        orchestrator._yolo_detector.detect.return_value = []
        orchestrator._output_writer = MagicMock()

        # Should not raise SystemExit since failures don't exceed 3 consecutive
        orchestrator.run(["/tmp/test.mp4"])

    def test_process_frame_with_detections(self, tmp_path):
        """_process_frame runs full pipeline chain when detections are found."""
        orchestrator = self._create_orchestrator(tmp_path)

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        metadata = FrameMetadata(
            frame_id="test_frame",
            source_path="/tmp/test.jpg",
            timestamp="2024-03-15T10:00:00Z",
            width=100,
            height=100,
        )

        # Mock preprocessor
        orchestrator._preprocessor = MagicMock()
        orchestrator._preprocessor.process.return_value = frame

        # Mock detector with one detection
        detection = Detection(bbox=(10, 10, 50, 50), confidence=0.85, class_label="pothole")
        orchestrator._yolo_detector = MagicMock()
        orchestrator._yolo_detector.detect.return_value = [detection]

        # Mock segmenter
        mask = np.ones((100, 100), dtype=np.uint8)
        seg_result = SegmentationResult(mask=mask, detection=detection, bbox=(10, 10, 50, 50))
        orchestrator._sam2_segmenter = MagicMock()
        orchestrator._sam2_segmenter.segment.return_value = [seg_result]

        # Mock verifier
        from src.pipeline.models import DefectMeasurement
        verified = VerifiedResult(
            mask=mask, detection=detection, bbox=(10, 10, 50, 50),
            area_ratio=0.5, review_flag=False
        )
        orchestrator._verifier = MagicMock()
        orchestrator._verifier.verify.return_value = [verified]

        # Mock measurement engine
        measurement = DefectMeasurement(
            area_pixels=2500, area_cm2=125.0, width_cm=10.0, length_cm=10.0,
            width_pixels=50, length_pixels=50, severity="minor"
        )
        orchestrator._measurement_engine = MagicMock()
        orchestrator._measurement_engine.measure.return_value = measurement

        # Mock output writer
        orchestrator._output_writer = MagicMock()
        orchestrator._output_writer._encode_mask_rle.return_value = {
            "size": [100, 100],
            "counts": "encoded_rle",
        }

        result = orchestrator._process_frame(frame, metadata)

        assert result is not None
        assert result.frame_id == "test_frame"
        assert len(result.defects) == 1
        assert result.defects[0].class_label == "pothole"
        assert result.defects[0].confidence == 0.85
        orchestrator._output_writer.write_frame_result.assert_called_once()
