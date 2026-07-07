"""Pipeline orchestrator for the road defect segmentation pipeline.

Coordinates all pipeline components, manages the processing loop, and
handles error tracking with consecutive failure detection. Implements
the full frame processing chain: ingestion → preprocessing → detection →
segmentation → verification → measurement → output.

Validates: Requirements 9.3, 9.4, 9.5, 9.6, 9.7
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from typing import Dict, List, Optional

import numpy as np

from src.pipeline.config_manager import ConfigManager
from src.pipeline.frame_ingester import FrameIngester, UnsupportedFormatError
from src.pipeline.logger import get_logger, setup_logging
from src.pipeline.measurement_engine import MeasurementEngine
from src.pipeline.models import (
    BatchSummary,
    DefectOutput,
    FrameMetadata,
    FrameResult,
    PipelineConfig,
)
from src.pipeline.output_writer import OutputWriter
from src.pipeline.preprocessor import Preprocessor
from src.pipeline.sam2_segmenter import SAM2Segmenter
from src.pipeline.sam2_segmenter import ModelLoadError as SAM2ModelLoadError
from src.pipeline.verifier import PostSegmentationVerifier
from src.pipeline.yolo_detector import ModelLoadError, YOLODetector


class SystematicFailureError(Exception):
    """Raised when consecutive frame failures exceed the configured threshold."""

    pass


class FailureTracker:
    """Track consecutive frame failures and raise on threshold breach.

    Increments the failure count on each recorded failure and resets on
    success. Raises SystematicFailureError when the configured maximum
    consecutive failure count is exceeded.

    Args:
        max_consecutive: Maximum allowed consecutive failures before exit.
    """

    def __init__(self, max_consecutive: int) -> None:
        self.max_consecutive = max_consecutive
        self.consecutive_count = 0

    def record_failure(self) -> None:
        """Record a frame processing failure.

        Raises:
            SystematicFailureError: If consecutive failures exceed the
                configured threshold.
        """
        self.consecutive_count += 1
        if self.consecutive_count > self.max_consecutive:
            raise SystematicFailureError(
                f"Exceeded {self.max_consecutive} consecutive frame failures"
            )

    def record_success(self) -> None:
        """Record a successful frame processing, resetting the counter."""
        self.consecutive_count = 0


class PipelineOrchestrator:
    """Coordinate all pipeline components and manage the processing loop.

    Handles initialization of all pipeline stages, the per-frame processing
    chain, error isolation per frame, and batch statistics logging.

    Args:
        config_path: Path to the YAML configuration file.
    """

    def __init__(self, config_path: str) -> None:
        # Load and validate configuration
        self._config_manager = ConfigManager()
        self._config = self._config_manager.load(config_path)

        errors = self._config_manager.validate(self._config)
        if errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(
                f"  - {e}" for e in errors
            )
            print(f"Error: {error_msg}", file=sys.stderr)
            sys.exit(1)

        # Setup logging
        setup_logging(
            log_level=self._config.log_level,
            output_directory=self._config.output_directory,
        )
        self._logger = get_logger("PipelineOrchestrator")
        self._logger.info("Configuration loaded and validated successfully")

        # Initialize pipeline components
        self._frame_ingester = FrameIngester(self._config)
        self._preprocessor = Preprocessor(self._config)
        self._verifier = PostSegmentationVerifier(self._config)
        self._measurement_engine = MeasurementEngine(self._config)
        self._output_writer = OutputWriter(self._config)

        # Load ML models (fatal on failure)
        self._yolo_detector = YOLODetector(self._config)
        try:
            self._yolo_detector.load_model()
        except ModelLoadError as e:
            self._logger.error(
                "Fatal: YOLO model load failed (expected path: '%s'): %s",
                self._config.yolo_model_path,
                e,
            )
            sys.exit(1)

        self._sam2_segmenter = SAM2Segmenter(self._config)
        try:
            self._sam2_segmenter.load_model()
        except SAM2ModelLoadError as e:
            self._logger.error(
                "Fatal: SAM2 model load failed (expected path: '%s'): %s",
                self._config.sam2_checkpoint_path,
                e,
            )
            sys.exit(1)

        # Initialize failure tracker
        self._failure_tracker = FailureTracker(self._config.max_consecutive_failures)

        self._logger.info("All pipeline components initialized successfully")

    def run(self, input_paths: List[str]) -> None:
        """Execute the pipeline on a list of input file paths.

        Iterates over input paths, ingests frames from each, and processes
        them through the full pipeline chain. Tracks batch statistics and
        writes a summary on completion.

        Args:
            input_paths: List of paths to image or video files to process.
        """
        start_time = time.time()
        total_frames_processed = 0
        total_defects_detected = 0
        defects_by_class: Dict[str, int] = {}
        defects_by_severity: Dict[str, int] = {}

        self._logger.info(
            "Starting pipeline run with %d input path(s)", len(input_paths)
        )

        try:
            for input_path in input_paths:
                self._logger.info("Processing input: %s", input_path)

                try:
                    for frame, metadata in self._frame_ingester.ingest(input_path):
                        try:
                            result = self._process_frame(frame, metadata)

                            if result is not None:
                                self._failure_tracker.record_success()
                                total_frames_processed += 1

                                # Accumulate statistics
                                num_defects = len(result.defects)
                                total_defects_detected += num_defects

                                for defect in result.defects:
                                    # Count by class
                                    cls = defect.class_label
                                    defects_by_class[cls] = (
                                        defects_by_class.get(cls, 0) + 1
                                    )
                                    # Count by severity
                                    sev = defect.measurements.get("severity")
                                    if sev is not None:
                                        defects_by_severity[sev] = (
                                            defects_by_severity.get(sev, 0) + 1
                                        )
                            else:
                                # _process_frame returned None (no defects frame)
                                self._failure_tracker.record_success()
                                total_frames_processed += 1

                        except SystematicFailureError:
                            raise
                        except Exception as e:
                            self._logger.error(
                                "Frame '%s' failed processing: %s\n%s",
                                metadata.frame_id,
                                e,
                                traceback.format_exc(),
                            )
                            self._failure_tracker.record_failure()

                except UnsupportedFormatError as e:
                    self._logger.error(
                        "Unsupported format for input '%s': %s", input_path, e
                    )
                    continue

        except SystematicFailureError as e:
            self._logger.error("Fatal: %s", e)
            sys.exit(1)

        # Compute timing
        elapsed = time.time() - start_time
        avg_per_frame = elapsed / total_frames_processed if total_frames_processed > 0 else 0.0

        # Write batch summary
        summary = BatchSummary(
            total_frames_processed=total_frames_processed,
            total_defects_detected=total_defects_detected,
            defects_by_class=defects_by_class,
            defects_by_severity=defects_by_severity,
            processing_time_seconds=round(elapsed, 1),
            average_time_per_frame_seconds=round(avg_per_frame, 3),
        )
        self._output_writer.write_batch_summary(summary)

        # Log batch statistics at INFO level
        self._logger.info(
            "Batch complete: %d frames processed, %d defects detected, "
            "total time %.1fs, avg %.3fs/frame",
            total_frames_processed,
            total_defects_detected,
            elapsed,
            avg_per_frame,
        )

    def _process_frame(
        self, frame: np.ndarray, metadata: FrameMetadata
    ) -> Optional[FrameResult]:
        """Process a single frame through the full pipeline chain.

        Pipeline stages: preprocessing → detection → segmentation →
        verification → measurement → output writing.

        Args:
            frame: Raw frame as an 8-bit RGB numpy array of shape (H, W, 3).
            metadata: Frame metadata including frame_id, source path, etc.

        Returns:
            FrameResult containing all defect outputs for this frame,
            or a FrameResult with an empty defects list if no defects found.
        """
        self._logger.debug("Processing frame: %s", metadata.frame_id)

        # Step 1: Preprocess
        preprocessed = self._preprocessor.process(frame)

        # Step 2: YOLO detection
        detections = self._yolo_detector.detect(preprocessed, frame_id=metadata.frame_id)

        defect_outputs: List[DefectOutput] = []

        if detections:
            # Step 3: SAM2 segmentation
            seg_results = self._sam2_segmenter.segment(preprocessed, detections)

            # Step 4: Verification
            verified_results = self._verifier.verify(seg_results)

            # Step 5: Measurement and output building
            for verified in verified_results:
                measurement = self._measurement_engine.measure(verified)

                # Encode mask as RLE
                rle_encoded = self._output_writer._encode_mask_rle(verified.mask)

                # Build DefectOutput
                defect_output = DefectOutput(
                    class_label=verified.detection.class_label,
                    confidence=verified.detection.confidence,
                    bounding_box={
                        "x": verified.bbox[0],
                        "y": verified.bbox[1],
                        "width": verified.bbox[2],
                        "height": verified.bbox[3],
                    },
                    segmentation=rle_encoded,
                    measurements={
                        "area_pixels": measurement.area_pixels,
                        "area_cm2": measurement.area_cm2,
                        "width_cm": measurement.width_cm,
                        "length_cm": measurement.length_cm,
                        "width_pixels": measurement.width_pixels,
                        "length_pixels": measurement.length_pixels,
                        "severity": measurement.severity,
                    },
                    review_flag=verified.review_flag,
                )
                defect_outputs.append(defect_output)

        # Step 6: Write frame result (even if no defects)
        self._output_writer.write_frame_result(metadata, defect_outputs)

        # Build FrameResult
        result = FrameResult(
            frame_id=metadata.frame_id,
            timestamp=metadata.timestamp,
            source_file=os.path.basename(metadata.source_path),
            defects=defect_outputs,
        )

        self._logger.debug(
            "Frame '%s' processed: %d defects",
            metadata.frame_id,
            len(defect_outputs),
        )

        return result
