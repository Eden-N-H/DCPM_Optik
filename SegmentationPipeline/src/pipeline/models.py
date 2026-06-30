"""Data models for the road defect segmentation pipeline.

Defines all dataclasses used throughout the pipeline stages:
ingestion, detection, segmentation, verification, measurement, and output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class FrameMetadata:
    """Metadata associated with a single ingested frame.

    Attributes:
        frame_id: Unique identifier (filename or video_name_frameNNN).
        source_path: Original file path.
        timestamp: ISO 8601 timestamp.
        width: Frame width in pixels.
        height: Frame height in pixels.
    """

    frame_id: str
    source_path: str
    timestamp: str
    width: int
    height: int


@dataclass
class Detection:
    """A single defect detection from the YOLO detector.

    Attributes:
        bbox: Bounding box as (x, y, width, height) in frame coordinates.
        confidence: Detection confidence score in [0.0, 1.0].
        class_label: One of: pothole, longitudinal_crack, transverse_crack,
            alligator_cracking, patch_deterioration.
    """

    bbox: Tuple[int, int, int, int]
    confidence: float
    class_label: str


@dataclass
class SegmentationResult:
    """Result of SAM2 segmentation for a single detection.

    Attributes:
        mask: Binary mask of shape (H, W), dtype=uint8, values 0 or 1.
        detection: The originating detection.
        bbox: Original bounding box (x, y, width, height).
    """

    mask: np.ndarray
    detection: Detection
    bbox: Tuple[int, int, int, int]


@dataclass
class VerifiedResult:
    """A segmentation result that has passed post-segmentation verification.

    Attributes:
        mask: Cleaned binary mask (largest connected component only).
        detection: The originating detection.
        bbox: Original bounding box (x, y, width, height).
        area_ratio: Mask-to-bounding-box area ratio.
        review_flag: True if area_ratio exceeds max_area_ratio threshold.
    """

    mask: np.ndarray
    detection: Detection
    bbox: Tuple[int, int, int, int]
    area_ratio: float
    review_flag: bool


@dataclass
class DefectMeasurement:
    """Measurements computed from a verified segmentation mask.

    Attributes:
        area_pixels: Defect area in pixels (count of foreground pixels).
        area_cm2: Defect area in square centimeters, or None if camera
            parameters are not provided.
        width_cm: Maximum defect width in centimeters, or None.
        length_cm: Maximum defect length in centimeters, or None.
        width_pixels: Maximum defect width in pixels.
        length_pixels: Maximum defect length in pixels.
        severity: Severity classification ("minor", "moderate", "severe"),
            or None if camera parameters are not provided.
    """

    area_pixels: int
    area_cm2: Optional[float]
    width_cm: Optional[float]
    length_cm: Optional[float]
    width_pixels: int
    length_pixels: int
    severity: Optional[str]


@dataclass
class DefectOutput:
    """Serializable output representation of a single defect for JSON export.

    Attributes:
        class_label: Defect class name.
        confidence: Detection confidence score.
        bounding_box: Dict with keys x, y, width, height.
        segmentation: Dict with keys size (list [H, W]) and counts (RLE string).
        measurements: Dict with measurement fields.
        review_flag: Whether this result requires manual review.
    """

    class_label: str
    confidence: float
    bounding_box: Dict[str, int]
    segmentation: Dict[str, Any]
    measurements: Dict[str, Any]
    review_flag: bool

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a dictionary matching the per-frame JSON schema."""
        return {
            "class": self.class_label,
            "confidence": self.confidence,
            "bounding_box": self.bounding_box,
            "segmentation": self.segmentation,
            "measurements": self.measurements,
            "review_flag": self.review_flag,
        }


@dataclass
class FrameResult:
    """Complete processing result for a single frame.

    Attributes:
        frame_id: Unique frame identifier.
        timestamp: ISO 8601 timestamp.
        source_file: Original source file name.
        defects: List of defect outputs for this frame.
    """

    frame_id: str
    timestamp: str
    source_file: str
    defects: List[DefectOutput]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a dictionary matching the per-frame output JSON schema."""
        return {
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "source_file": self.source_file,
            "defects": [d.to_dict() for d in self.defects],
        }


@dataclass
class BatchSummary:
    """Summary statistics for a complete batch processing run.

    Attributes:
        total_frames_processed: Number of frames processed in the batch.
        total_defects_detected: Total number of defects across all frames.
        defects_by_class: Count of defects grouped by class label.
        defects_by_severity: Count of defects grouped by severity level.
        processing_time_seconds: Total wall-clock processing time.
        average_time_per_frame_seconds: Average processing time per frame.
    """

    total_frames_processed: int
    total_defects_detected: int
    defects_by_class: Dict[str, int]
    defects_by_severity: Dict[str, int]
    processing_time_seconds: float
    average_time_per_frame_seconds: float

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a dictionary matching the batch summary JSON schema."""
        return {
            "total_frames_processed": self.total_frames_processed,
            "total_defects_detected": self.total_defects_detected,
            "defects_by_class": self.defects_by_class,
            "defects_by_severity": self.defects_by_severity,
            "processing_time_seconds": self.processing_time_seconds,
            "average_time_per_frame_seconds": self.average_time_per_frame_seconds,
        }


@dataclass
class PipelineConfig:
    """Configuration for the road defect segmentation pipeline.

    All parameters have documented defaults and valid ranges as specified
    in the design document.

    Attributes:
        frame_extraction_rate: Video frame extraction rate in fps. Range [0.1, 30].
        clip_limit: CLAHE clip limit for adaptive histogram equalization.
        tile_grid_size: CLAHE tile grid size as (rows, cols).
        distortion_coefficients: Lens distortion coefficients [k1, k2, p1, p2, k3],
            or None for default GoPro wide-angle coefficients.
        camera_matrix: 3x3 camera intrinsic matrix as nested list, or None for defaults.
        confidence_threshold: YOLO detection confidence threshold. Range [0.0, 1.0].
        iou_threshold: Non-maximum suppression IoU threshold. Range [0.0, 1.0].
        max_detections: Maximum detections per frame.
        yolo_model_path: Path to YOLO model weights file.
        sam2_checkpoint_path: Path to SAM2 model checkpoint.
        sam2_model_cfg: SAM2 model configuration file name.
        min_area_ratio: Minimum mask-to-bbox area ratio. Range [0.0, 1.0].
        max_area_ratio: Maximum mask-to-bbox area ratio. Range [0.0, 1.0].
        camera_height_cm: Camera height above ground in cm, or None.
        focal_length_px: Camera focal length in pixels, or None.
        output_directory: Directory path for output files.
        max_consecutive_failures: Max consecutive frame failures before exit.
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR).
    """

    # Ingestion
    frame_extraction_rate: float = 1.0

    # Preprocessing
    clip_limit: float = 2.0
    tile_grid_size: Tuple[int, int] = (8, 8)
    distortion_coefficients: Optional[List[float]] = None
    camera_matrix: Optional[List[List[float]]] = None

    # Detection
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45
    max_detections: int = 50
    yolo_model_path: str = "models/yolo_road_defects.pt"

    # Segmentation
    sam2_checkpoint_path: str = "models/sam2.1_hiera_large.pt"
    sam2_model_cfg: str = "configs/sam2.1/sam2.1_hiera_l.yaml"

    # Verification
    min_area_ratio: float = 0.05
    max_area_ratio: float = 0.95

    # Measurement
    camera_height_cm: Optional[float] = None
    focal_length_px: Optional[float] = None

    # Output
    output_directory: str = "output"

    # Error handling
    max_consecutive_failures: int = 10
    log_level: str = "INFO"

    # Valid ranges for configuration validation
    VALID_RANGES: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: {
            "frame_extraction_rate": (0.1, 30.0),
            "confidence_threshold": (0.0, 1.0),
            "iou_threshold": (0.0, 1.0),
            "min_area_ratio": (0.0, 1.0),
            "max_area_ratio": (0.0, 1.0),
        },
        init=False,
        repr=False,
    )

    VALID_LOG_LEVELS: Tuple[str, ...] = field(
        default_factory=lambda: ("DEBUG", "INFO", "WARNING", "ERROR"),
        init=False,
        repr=False,
    )

    def validate(self) -> List[str]:
        """Validate all configuration values against documented valid ranges.

        Returns:
            A list of error messages for invalid parameters. Empty if all valid.
        """
        errors: List[str] = []

        for param_name, (min_val, max_val) in self.VALID_RANGES.items():
            value = getattr(self, param_name)
            if value < min_val or value > max_val:
                errors.append(
                    f"{param_name}: value {value} is outside valid range "
                    f"[{min_val}, {max_val}]"
                )

        if self.log_level not in self.VALID_LOG_LEVELS:
            errors.append(
                f"log_level: value '{self.log_level}' is not one of "
                f"{list(self.VALID_LOG_LEVELS)}"
            )

        if self.max_detections < 1:
            errors.append(
                f"max_detections: value {self.max_detections} must be >= 1"
            )

        if self.max_consecutive_failures < 1:
            errors.append(
                f"max_consecutive_failures: value {self.max_consecutive_failures} "
                f"must be >= 1"
            )

        if self.camera_height_cm is not None and self.camera_height_cm <= 0:
            errors.append(
                f"camera_height_cm: value {self.camera_height_cm} must be positive"
            )

        if self.focal_length_px is not None and self.focal_length_px <= 0:
            errors.append(
                f"focal_length_px: value {self.focal_length_px} must be positive"
            )

        if self.distortion_coefficients is not None:
            if len(self.distortion_coefficients) != 5:
                errors.append(
                    f"distortion_coefficients: expected 5 values [k1, k2, p1, p2, k3], "
                    f"got {len(self.distortion_coefficients)}"
                )

        if self.camera_matrix is not None:
            if len(self.camera_matrix) != 3 or any(
                len(row) != 3 for row in self.camera_matrix
            ):
                errors.append(
                    "camera_matrix: expected a 3x3 matrix (list of 3 lists of 3 floats)"
                )

        return errors
