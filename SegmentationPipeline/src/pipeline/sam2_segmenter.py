"""SAM2-based road defect segmenter.

Generates pixel-precise segmentation masks for detected road defects using
Meta's Segment Anything Model 2 (SAM2) with box prompts derived from YOLO
detections.

Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch

from src.pipeline.logger import get_logger
from src.pipeline.models import Detection, PipelineConfig, SegmentationResult

logger = get_logger("SAM2Segmenter")


class ModelLoadError(Exception):
    """Raised when the SAM2 model fails to load."""

    pass


class SAM2Segmenter:
    """SAM2-based segmentation for road defect masks.

    Uses SAM2's image predictor API with box prompts to generate binary
    segmentation masks for each detected defect. All valid detections in
    a frame are batched into a single inference call.

    Attributes:
        config: Pipeline configuration containing segmentation parameters.
        predictor: The SAM2ImagePredictor instance (None until load_model is called).
    """

    def __init__(self, config: PipelineConfig) -> None:
        """Initialize the SAM2 segmenter with pipeline configuration.

        Args:
            config: Pipeline configuration containing sam2_checkpoint_path
                and sam2_model_cfg.
        """
        self.config = config
        self.predictor = None
        self._device = self._select_device()

    def _select_device(self) -> str:
        """Select the best available device for inference.

        Prefers MPS (Apple Silicon) with fallback to CPU.

        Returns:
            Device string: "mps" if available, otherwise "cpu".
        """
        if torch.backends.mps.is_available():
            logger.info("MPS device available, using Apple Silicon acceleration")
            return "mps"
        else:
            logger.info("MPS device not available, falling back to CPU")
            return "cpu"

    def load_model(self) -> None:
        """Load the SAM2 model checkpoint from the configured path.

        Loads the model using SAM2ImagePredictor and moves it to the
        selected device (MPS or CPU).

        Raises:
            ModelLoadError: If the SAM2 checkpoint cannot be loaded due to
                a missing file, corrupt checkpoint, import error, or other
                load failure.
        """
        try:
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            from sam2.build_sam import build_sam2
        except ImportError as e:
            error_msg = (
                f"SAM2 is not installed. Please install the sam2 package: {e}"
            )
            logger.error(error_msg)
            raise ModelLoadError(error_msg) from e

        try:
            logger.info(
                f"Loading SAM2 model from: {self.config.sam2_checkpoint_path} "
                f"with config: {self.config.sam2_model_cfg}"
            )
            # Use build_sam2 for local checkpoint files
            sam2_model = build_sam2(
                config_file=self.config.sam2_model_cfg,
                ckpt_path=self.config.sam2_checkpoint_path,
                device=self._device,
            )
            self.predictor = SAM2ImagePredictor(sam2_model)
            logger.info(
                f"SAM2 model loaded successfully on device: {self._device}"
            )
        except Exception as e:
            error_msg = (
                f"Failed to load SAM2 model from "
                f"'{self.config.sam2_checkpoint_path}': {e}"
            )
            logger.error(error_msg)
            raise ModelLoadError(error_msg) from e

    def segment(
        self, frame: np.ndarray, detections: List[Detection]
    ) -> List[SegmentationResult]:
        """Generate segmentation masks for detected defects in a frame.

        Sets the image once on the predictor, validates all bounding boxes,
        batches valid box prompts into a single predict() call, and returns
        binary masks at the original frame resolution.

        Args:
            frame: The frame as an 8-bit RGB numpy array of shape (H, W, 3).
            detections: List of Detection objects from the YOLO detector.

        Returns:
            A list of SegmentationResult objects for valid detections with
            non-zero masks. Empty list if no valid results.

        Raises:
            RuntimeError: If the model has not been loaded via load_model().
        """
        if self.predictor is None:
            raise RuntimeError(
                "SAM2 model not loaded. Call load_model() before segment()."
            )

        if not detections:
            return []

        frame_height, frame_width = frame.shape[:2]
        frame_shape = (frame_height, frame_width)

        # Filter detections with valid bounding boxes
        valid_detections: List[Detection] = []
        for detection in detections:
            if self._validate_bbox(detection.bbox, frame_shape):
                valid_detections.append(detection)

        if not valid_detections:
            return []

        # Prepare box prompts: convert from (x, y, w, h) to (x1, y1, x2, y2)
        box_prompts = self._prepare_box_prompts(valid_detections)

        # Set image once per frame
        self.predictor.set_image(frame)

        # Batch all boxes into a single predict call
        masks, scores, logits = self.predictor.predict(
            box=box_prompts,
            multimask_output=False,
        )

        # Process results
        results: List[SegmentationResult] = []

        for i, detection in enumerate(valid_detections):
            # When multimask_output=False, masks shape is (N, 1, H, W)
            # Extract the single mask for this detection
            if masks.ndim == 4:
                mask = masks[i, 0]
            elif masks.ndim == 3:
                mask = masks[i]
            else:
                mask = masks

            # Convert to binary uint8 mask (0 or 1)
            binary_mask = (mask > 0).astype(np.uint8)

            # Check for zero foreground pixels
            if binary_mask.sum() == 0:
                logger.warning(
                    f"Mask for detection at bbox {detection.bbox} has zero "
                    f"foreground pixels, discarding"
                )
                continue

            results.append(
                SegmentationResult(
                    mask=binary_mask,
                    detection=detection,
                    bbox=detection.bbox,
                )
            )

        return results

    def _validate_bbox(
        self, bbox: Tuple[int, int, int, int], frame_shape: Tuple[int, int]
    ) -> bool:
        """Validate a bounding box against frame dimensions.

        Checks that the bounding box has non-zero width and height and
        does not extend beyond the frame boundaries.

        Args:
            bbox: Bounding box as (x, y, width, height).
            frame_shape: Frame dimensions as (height, width).

        Returns:
            True if the bounding box is valid, False otherwise.
        """
        x, y, w, h = bbox
        frame_height, frame_width = frame_shape

        # Check for zero or negative dimensions
        if w <= 0 or h <= 0:
            logger.warning(
                f"Invalid bbox {bbox}: zero or negative width/height, skipping"
            )
            return False

        # Check for negative coordinates
        if x < 0 or y < 0:
            logger.warning(
                f"Invalid bbox {bbox}: negative coordinates, skipping"
            )
            return False

        # Check if bbox extends beyond frame boundaries
        if x + w > frame_width or y + h > frame_height:
            logger.warning(
                f"Invalid bbox {bbox}: extends beyond frame boundaries "
                f"({frame_width}x{frame_height}), skipping"
            )
            return False

        return True

    def _prepare_box_prompts(
        self, detections: List[Detection]
    ) -> np.ndarray:
        """Convert YOLO bounding boxes to SAM2 box prompt format.

        Converts from YOLO format (x, y, w, h) where (x, y) is top-left
        corner to SAM2 format (x1, y1, x2, y2).

        Args:
            detections: List of Detection objects with valid bounding boxes.

        Returns:
            Numpy array of shape (N, 4) with box prompts in (x1, y1, x2, y2)
            format, suitable for SAM2 predict().
        """
        boxes = []
        for detection in detections:
            x, y, w, h = detection.bbox
            x1 = x
            y1 = y
            x2 = x + w
            y2 = y + h
            boxes.append([x1, y1, x2, y2])

        return np.array(boxes, dtype=np.float32)
