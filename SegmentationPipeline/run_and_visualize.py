"""Run the pipeline on images and save visualized segmentation output.

Processes each image through the full pipeline (YOLO detection + SAM2 segmentation),
then renders the segmentation masks overlaid on the original images and saves them
to the output_images/ directory.
"""

import os
import sys
import numpy as np
import cv2

from src.pipeline.config_manager import ConfigManager
from src.pipeline.logger import setup_logging, get_logger
from src.pipeline.frame_ingester import FrameIngester
from src.pipeline.preprocessor import Preprocessor
from src.pipeline.yolo_detector import YOLODetector, ModelLoadError
from src.pipeline.sam2_segmenter import SAM2Segmenter, ModelLoadError as SAM2ModelLoadError
from src.pipeline.verifier import PostSegmentationVerifier
from src.pipeline.measurement_engine import MeasurementEngine
from src.pipeline.models import PipelineConfig


# Color map for defect classes (BGR for OpenCV)
CLASS_COLORS = {
    "Crack": (0, 0, 255),                # Red
    "Edge Damage": (0, 165, 255),        # Orange
    "Faded Line Marking": (0, 255, 255), # Yellow
    "Patch Deformation": (255, 0, 255),  # Magenta
    "Ponding Water": (255, 255, 0),      # Cyan
    "Pothole": (0, 255, 0),              # Green
    "Road Debris": (128, 0, 255),        # Purple
    "Vegetation Obstruction": (0, 128, 0), # Dark green
}


def main():
    # Ensure we're in the project root for relative paths
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    config_path = "config/default_config.yaml"
    input_dir = "images"
    output_dir = "output_images"

    # Load config
    cm = ConfigManager()
    config = cm.load(config_path)
    errors = cm.validate(config)
    if errors:
        print(f"Config errors: {errors}")
        sys.exit(1)

    # Setup logging
    setup_logging(log_level=config.log_level, output_directory=config.output_directory)
    logger = get_logger("Visualizer")

    # Initialize components
    logger.info("Initializing pipeline components...")
    ingester = FrameIngester(config)
    preprocessor = Preprocessor(config)
    verifier = PostSegmentationVerifier(config)
    measurement_engine = MeasurementEngine(config)

    # Load YOLO
    logger.info("Loading YOLO model...")
    yolo = YOLODetector(config)
    try:
        yolo.load_model()
    except ModelLoadError as e:
        print(f"Failed to load YOLO model: {e}")
        sys.exit(1)

    # Load SAM2
    logger.info("Loading SAM2 model...")
    sam2 = SAM2Segmenter(config)
    try:
        sam2.load_model()
    except SAM2ModelLoadError as e:
        print(f"Failed to load SAM2 model: {e}")
        sys.exit(1)

    logger.info("Models loaded. Processing images...")

    # Process each image
    os.makedirs(output_dir, exist_ok=True)
    supported_exts = {".jpg", ".jpeg", ".png"}
    image_files = [
        f for f in os.listdir(input_dir)
        if os.path.splitext(f)[1].lower() in supported_exts
    ]

    for filename in sorted(image_files):
        filepath = os.path.join(input_dir, filename)
        logger.info(f"Processing: {filename}")

        # Read image
        img_bgr = cv2.imread(filepath)
        if img_bgr is None:
            logger.warning(f"Could not read {filename}, skipping")
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Preprocess
        preprocessed = preprocessor.process(img_rgb)

        # Detect
        detections = yolo.detect(preprocessed, frame_id=filename)
        logger.info(f"  {len(detections)} detections found")

        if not detections:
            # Save original with "No defects" text
            cv2.putText(img_bgr, "No defects detected", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, img_bgr)
            logger.info(f"  Saved (no defects): {output_path}")
            continue

        # Segment
        seg_results = sam2.segment(preprocessed, detections)
        logger.info(f"  {len(seg_results)} segmentation results")

        # Verify
        verified = verifier.verify(seg_results)
        logger.info(f"  {len(verified)} verified results")

        # Create visualization overlay
        overlay = img_bgr.copy()

        for v_result in verified:
            mask = v_result.mask
            class_label = v_result.detection.class_label
            confidence = v_result.detection.confidence
            bbox = v_result.bbox

            # Get color for this class
            color = CLASS_COLORS.get(class_label, (0, 0, 255))

            # Apply semi-transparent mask overlay
            colored_mask = np.zeros_like(img_bgr)
            colored_mask[mask == 1] = color
            overlay = cv2.addWeighted(overlay, 1.0, colored_mask, 0.4, 0)

            # Draw mask contours
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(overlay, contours, -1, color, 2)

            # Draw bounding box
            x, y, w, h = bbox
            cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)

            # Draw label
            label = f"{class_label} {confidence:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(overlay, (x, y - th - 8), (x + tw + 4, y), color, -1)
            cv2.putText(overlay, label, (x + 2, y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # Measure
            measurement = measurement_engine.measure(v_result)
            logger.info(
                f"    {class_label}: {measurement.area_pixels}px, "
                f"severity={measurement.severity or 'N/A'}"
            )

        # Save annotated image
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, overlay)
        logger.info(f"  Saved: {output_path}")

    logger.info(f"Done! Results in {output_dir}/")


if __name__ == "__main__":
    main()
