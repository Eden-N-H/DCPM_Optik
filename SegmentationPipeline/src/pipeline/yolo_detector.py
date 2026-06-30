"""YOLO-based road defect detector.

Provides object detection for road defects using the Ultralytics YOLO API.
Detects five defect classes: pothole, longitudinal_crack, transverse_crack,
alligator_cracking, and patch_deterioration.

Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch

from src.pipeline.logger import get_logger
from src.pipeline.models import Detection, PipelineConfig

# Class index to label mapping for the road defect YOLO model
# This will be populated dynamically from the loaded model's names dict
CLASS_INDEX_TO_LABEL = {
    0: "Crack",
    1: "Edge Damage",
    2: "Faded Line Marking",
    3: "Patch Deformation",
    4: "Ponding Water",
    5: "Pothole",
    6: "Road Debris",
    7: "Vegetation Obstruction",
}

logger = get_logger("YOLODetector")


class ModelLoadError(Exception):
    """Raised when the YOLO model fails to load."""

    pass


class YOLODetector:
    """YOLO-based road defect detector.

    Uses the Ultralytics YOLO API to detect road surface defects in
    preprocessed frames. Supports configurable confidence threshold,
    IoU threshold for NMS, and maximum detections per frame.

    Attributes:
        config: Pipeline configuration containing detection parameters.
        model: The loaded YOLO model instance (None until load_model is called).
    """

    def __init__(self, config: PipelineConfig) -> None:
        """Initialize the YOLO detector with pipeline configuration.

        Args:
            config: Pipeline configuration containing yolo_model_path,
                confidence_threshold, iou_threshold, and max_detections.
        """
        self.config = config
        self.model = None
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
        """Load the YOLO model weights from the configured path.

        Loads the model using the Ultralytics YOLO API and moves it to
        the selected device (MPS or CPU).

        Raises:
            ModelLoadError: If the model weights cannot be loaded due to
                a missing file, corrupt weights, or other load failure.
        """
        try:
            from ultralytics import YOLO

            logger.info(
                f"Loading YOLO model from: {self.config.yolo_model_path}"
            )
            self.model = YOLO(self.config.yolo_model_path)
            # Move model to the selected device
            self.model.to(self._device)
            logger.info(
                f"YOLO model loaded successfully on device: {self._device}"
            )
        except Exception as e:
            error_msg = (
                f"Failed to load YOLO model from '{self.config.yolo_model_path}': {e}"
            )
            logger.error(error_msg)
            raise ModelLoadError(error_msg) from e

    def detect(self, frame: np.ndarray, frame_id: str = "unknown") -> List[Detection]:
        """Run YOLO inference on a preprocessed frame.

        Performs object detection using the loaded YOLO model with configured
        confidence threshold, IoU threshold (for NMS), and maximum detections.

        Args:
            frame: Preprocessed frame as an 8-bit RGB numpy array of shape (H, W, 3).
            frame_id: Identifier for the frame, used in error logging.

        Returns:
            A list of Detection objects for detected defects, filtered by
            confidence threshold and NMS. Returns an empty list if no defects
            are found.

        Raises:
            RuntimeError: If the model has not been loaded via load_model().
        """
        if self.model is None:
            raise RuntimeError(
                "YOLO model not loaded. Call load_model() before detect()."
            )

        try:
            results = self.model.predict(
                frame,
                conf=self.config.confidence_threshold,
                iou=self.config.iou_threshold,
                max_det=self.config.max_detections,
                device=self._device,
                verbose=False,
            )

            detections: List[Detection] = []

            if not results or len(results) == 0:
                logger.debug(f"Frame {frame_id}: no detections")
                return detections

            # Process the first result (single frame inference)
            result = results[0]

            if result.boxes is None or len(result.boxes) == 0:
                logger.debug(f"Frame {frame_id}: no detections")
                return detections

            boxes = result.boxes

            for i in range(len(boxes)):
                # Get bounding box in xyxy format and convert to xywh
                xyxy = boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = xyxy
                x = int(round(x1))
                y = int(round(y1))
                width = int(round(x2 - x1))
                height = int(round(y2 - y1))

                # Get confidence score
                confidence = float(boxes.conf[i].cpu().numpy())

                # Get class index and map to label
                class_idx = int(boxes.cls[i].cpu().numpy())
                # Use model's own class names if available, fall back to static map
                if hasattr(self.model, 'names') and class_idx in self.model.names:
                    class_label = self.model.names[class_idx]
                else:
                    class_label = CLASS_INDEX_TO_LABEL.get(class_idx)

                if class_label is None:
                    logger.warning(
                        f"Frame {frame_id}: unknown class index {class_idx}, skipping"
                    )
                    continue

                detections.append(
                    Detection(
                        bbox=(x, y, width, height),
                        confidence=confidence,
                        class_label=class_label,
                    )
                )

            logger.debug(
                f"Frame {frame_id}: {len(detections)} detections found"
            )
            return detections

        except Exception as e:
            logger.error(
                f"Frame {frame_id}: inference error - {type(e).__name__}: {e}"
            )
            return []
