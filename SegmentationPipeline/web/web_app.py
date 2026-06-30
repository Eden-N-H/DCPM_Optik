"""Flask web application for road defect segmentation.

Provides a browser-based interface wrapping the existing YOLO + SAM2
segmentation pipeline. Designed for single-user local deployment.
"""

import os
import sys
import uuid
import threading
import shutil

from flask import Flask, render_template, request, jsonify, send_from_directory

# Resolve paths relative to the project root (one level up from web/)
_WEB_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_WEB_DIR)

# Add project root to sys.path so pipeline imports resolve correctly
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Change working directory to project root so config-relative model paths resolve
os.chdir(_PROJECT_ROOT)

import cv2
import numpy as np

from src.pipeline.config_manager import ConfigManager
from src.pipeline.yolo_detector import YOLODetector, ModelLoadError
from src.pipeline.sam2_segmenter import SAM2Segmenter, ModelLoadError as SAM2ModelLoadError
from src.pipeline.preprocessor import Preprocessor
from src.pipeline.verifier import PostSegmentationVerifier
from src.pipeline.measurement_engine import MeasurementEngine
from src.pipeline.logger import get_logger, setup_logging

# Set up logging and get a logger for the web app
setup_logging()
logger = get_logger("WebApp")

app = Flask(
    __name__,
    template_folder=os.path.join(_WEB_DIR, "templates"),
    static_folder=os.path.join(_WEB_DIR, "static"),
)

# Configuration
app.config["MAX_CONTENT_LENGTH"] = 100 * 50 * 1024 * 1024  # Allow batch of 100 x 50MB
app.config["UPLOAD_FOLDER"] = os.path.join(_WEB_DIR, "uploads")
app.config["OUTPUT_FOLDER"] = os.path.join(_WEB_DIR, "output")
app.config["PROJECT_ROOT"] = _PROJECT_ROOT

# Ensure upload and output directories exist
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["OUTPUT_FOLDER"], exist_ok=True)

# ---------------------------------------------------------------------------
# Supported image extensions
# ---------------------------------------------------------------------------
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Color map for defect classes (BGR for OpenCV)
CLASS_COLORS = {
    "Crack": (0, 0, 255),
    "Edge Damage": (0, 165, 255),
    "Faded Line Marking": (0, 255, 255),
    "Patch Deformation": (255, 0, 255),
    "Ponding Water": (255, 255, 0),
    "Pothole": (0, 255, 0),
    "Road Debris": (128, 0, 255),
    "Vegetation Obstruction": (0, 128, 0),
}

# Maximum file size per image (50 MB)
MAX_FILE_SIZE = 50 * 1024 * 1024

# Maximum batch size
MAX_BATCH_SIZE = 300

# ---------------------------------------------------------------------------
# Load pipeline configuration and models at startup (shared across requests)
# ---------------------------------------------------------------------------
_config_path = os.path.join(_PROJECT_ROOT, "config", "default_config.yaml")

config_manager = ConfigManager()
config = config_manager.load(_config_path)

# Validate configuration before proceeding
validation_errors = config_manager.validate(config)
if validation_errors:
    for err in validation_errors:
        logger.error("Configuration validation error: %s", err)
    logger.error("Fatal: Configuration validation failed. Exiting.")
    sys.exit(1)

# Load YOLO detector
try:
    yolo_detector = YOLODetector(config)
    yolo_detector.load_model()
    logger.info("YOLO model loaded successfully")
except ModelLoadError as e:
    logger.error("Fatal: Failed to load YOLO model: %s", e)
    sys.exit(1)

# Load SAM2 segmenter
try:
    sam2_segmenter = SAM2Segmenter(config)
    sam2_segmenter.load_model()
    logger.info("SAM2 model loaded successfully")
except SAM2ModelLoadError as e:
    logger.error("Fatal: Failed to load SAM2 model: %s", e)
    sys.exit(1)

# Initialize remaining pipeline components
preprocessor = Preprocessor(config)
verifier = PostSegmentationVerifier(config)
measurement_engine = MeasurementEngine(config)
logger.info("All pipeline components initialized successfully")

# ---------------------------------------------------------------------------
# Batch state management (thread-safe)
# ---------------------------------------------------------------------------
_batch_lock = threading.Lock()
_batch_state = None  # Current batch state or None if idle


def _get_batch_state():
    """Get current batch state (thread-safe read)."""
    with _batch_lock:
        if _batch_state is None:
            return None
        return dict(_batch_state)


def _is_processing():
    """Check if a batch is currently being processed."""
    with _batch_lock:
        return _batch_state is not None and _batch_state["status"] == "processing"


# ---------------------------------------------------------------------------
# Image processing logic (runs in background thread)
# ---------------------------------------------------------------------------
def _process_image(image_path, output_dir, filename):
    """Process a single image through the pipeline and save annotated output.

    Returns a dict with result information.
    """
    try:
        # Read image
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            return {
                "filename": filename,
                "status": "failed",
                "output_url": None,
                "defects_found": 0,
                "error": "Could not read image file",
            }

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Preprocess
        preprocessed = preprocessor.process(img_rgb)

        # Detect
        detections = yolo_detector.detect(preprocessed, frame_id=filename)

        if not detections:
            # No defects: add "No defects detected" text
            cv2.putText(
                img_bgr,
                "No defects detected",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),
                3,
            )
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, img_bgr)
            return {
                "filename": filename,
                "status": "success",
                "output_url": None,  # Will be set by caller
                "defects_found": 0,
                "error": None,
            }

        # Segment
        seg_results = sam2_segmenter.segment(preprocessed, detections)

        # Verify
        verified = verifier.verify(seg_results)

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
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
            )
            cv2.rectangle(overlay, (x, y - th - 8), (x + tw + 4, y), color, -1)
            cv2.putText(
                overlay,
                label,
                (x + 2, y - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
            )

        # Save annotated image
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, overlay)

        return {
            "filename": filename,
            "status": "success",
            "output_url": None,  # Will be set by caller
            "defects_found": len(verified),
            "error": None,
        }

    except Exception as e:
        logger.error("Error processing %s: %s", filename, str(e))
        return {
            "filename": filename,
            "status": "failed",
            "output_url": None,
            "defects_found": 0,
            "error": str(e),
        }


def _process_batch(batch_id, image_files, upload_dir, output_dir):
    """Process a batch of images in the background thread."""
    global _batch_state

    os.makedirs(output_dir, exist_ok=True)

    for filename in image_files:
        image_path = os.path.join(upload_dir, filename)
        result = _process_image(image_path, output_dir, filename)

        # Set the output_url for successful images
        if result["status"] == "success":
            result["output_url"] = f"/output/{batch_id}/{filename}"

        # Update state (thread-safe)
        with _batch_lock:
            _batch_state["results"].append(result)
            if result["status"] == "success":
                _batch_state["completed"] += 1
            else:
                _batch_state["failed"] += 1
                _batch_state["failed_files"].append(filename)

        logger.info(
            "Batch %s: processed %s (%s)",
            batch_id,
            filename,
            result["status"],
        )

    # Mark batch as complete
    with _batch_lock:
        _batch_state["status"] = "complete"

    logger.info("Batch %s: processing complete", batch_id)

    # Clean up upload directory
    try:
        shutil.rmtree(upload_dir)
        logger.info("Batch %s: cleaned up upload directory", batch_id)
    except Exception as e:
        logger.warning("Batch %s: failed to clean up uploads: %s", batch_id, str(e))


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/model", methods=["GET"])
def get_model_info():
    """Return the current YOLO model filename and confidence threshold."""
    return jsonify({
        "model_name": os.path.basename(config.yolo_model_path),
        "confidence_threshold": config.confidence_threshold,
    }), 200


@app.route("/api/confidence", methods=["POST"])
def set_confidence():
    """Update the YOLO confidence threshold at runtime."""
    data = request.get_json()
    if not data or "confidence_threshold" not in data:
        return jsonify({"error": "confidence_threshold required"}), 400

    try:
        value = float(data["confidence_threshold"])
    except (TypeError, ValueError):
        return jsonify({"error": "confidence_threshold must be a number"}), 400

    if value < 0.0 or value > 1.0:
        return jsonify({"error": "confidence_threshold must be between 0.0 and 1.0"}), 400

    config.confidence_threshold = value
    logger.info("Confidence threshold updated to: %.3f", value)
    return jsonify({"confidence_threshold": value}), 200


@app.route("/api/model", methods=["POST"])
def upload_model():
    """Accept a .pt model file and hot-swap the YOLO detector."""
    global yolo_detector

    if _is_processing():
        return jsonify({"error": "Cannot change model while a batch is processing"}), 409

    file = request.files.get("model")
    if not file or not file.filename:
        return jsonify({"error": "No model file provided"}), 400

    if not file.filename.lower().endswith(".pt"):
        return jsonify({"error": "Only .pt model files are supported"}), 400

    # Save model to models/ directory
    model_filename = file.filename
    model_path = os.path.join(_PROJECT_ROOT, "models", model_filename)
    file.save(model_path)
    logger.info("Saved new model file: %s", model_path)

    # Attempt to load the new model
    try:
        original_path = config.yolo_model_path
        config.yolo_model_path = model_path
        new_detector = YOLODetector(config)
        new_detector.load_model()

        # Success — swap in the new detector
        yolo_detector = new_detector
        logger.info("YOLO model hot-swapped to: %s", model_filename)

        return jsonify({
            "model_name": model_filename,
            "message": "Model loaded successfully",
        }), 200

    except Exception as e:
        # Revert config path on failure
        config.yolo_model_path = original_path
        logger.error("Failed to load new model %s: %s", model_filename, str(e))
        return jsonify({"error": f"Failed to load model: {str(e)}"}), 400


@app.route("/api/upload", methods=["POST"])
def upload():
    """Accept multipart form data with images and start batch processing."""
    global _batch_state

    # Check if a batch is already processing
    if _is_processing():
        return jsonify({"error": "A batch is already being processed"}), 409

    # Get uploaded files
    files = request.files.getlist("images")

    if not files:
        return jsonify({"error": "No valid images provided"}), 400

    # Filter to supported image files and check sizes
    valid_files = []
    failed_files = []

    for f in files:
        if not f.filename:
            continue
        ext = os.path.splitext(f.filename)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            continue

        # Read file content to check size
        f.seek(0, os.SEEK_END)
        file_size = f.tell()
        f.seek(0)

        if file_size > MAX_FILE_SIZE:
            failed_files.append({
                "filename": f.filename,
                "status": "failed",
                "output_url": None,
                "defects_found": 0,
                "error": "File exceeds 50 MB size limit",
            })
        else:
            valid_files.append(f)

    # Check if we have any supported image files (valid + oversized still count for the batch)
    if not valid_files and not failed_files:
        return jsonify({"error": "No valid images provided"}), 400

    # Check batch size (total images including oversized ones)
    total_images = len(valid_files) + len(failed_files)
    if total_images > MAX_BATCH_SIZE:
        return jsonify({"error": "Batch exceeds maximum of 100 images"}), 400

    # Generate batch ID
    batch_id = str(uuid.uuid4())

    # Create upload directory for this batch
    upload_dir = os.path.join(app.config["UPLOAD_FOLDER"], batch_id)
    os.makedirs(upload_dir, exist_ok=True)

    # Save valid files to upload directory
    saved_filenames = []
    for f in valid_files:
        filename = f.filename
        save_path = os.path.join(upload_dir, filename)
        f.save(save_path)
        saved_filenames.append(filename)

    # Create output directory
    output_dir = os.path.join(app.config["OUTPUT_FOLDER"], batch_id)

    # Initialize batch state
    with _batch_lock:
        _batch_state = {
            "batch_id": batch_id,
            "total": total_images,
            "completed": 0,
            "failed": len(failed_files),
            "failed_files": [r["filename"] for r in failed_files],
            "status": "processing",
            "results": list(failed_files),  # Pre-populate with already-failed files
        }

    # Start background processing thread
    thread = threading.Thread(
        target=_process_batch,
        args=(batch_id, saved_filenames, upload_dir, output_dir),
        daemon=True,
    )
    thread.start()

    return jsonify({
        "batch_id": batch_id,
        "total_images": total_images,
        "status": "processing",
    }), 200


@app.route("/api/progress", methods=["GET"])
def progress():
    """Return current batch processing progress."""
    batch_id = request.args.get("batch_id")

    if not batch_id:
        return jsonify({"error": "batch_id parameter required"}), 400

    state = _get_batch_state()

    if state is None or state["batch_id"] != batch_id:
        return jsonify({"error": "Unknown batch_id"}), 404

    return jsonify({
        "batch_id": state["batch_id"],
        "total": state["total"],
        "completed": state["completed"],
        "failed": state["failed"],
        "status": state["status"],
    }), 200


@app.route("/api/results", methods=["GET"])
def results():
    """Return result details for a completed batch."""
    batch_id = request.args.get("batch_id")

    if not batch_id:
        return jsonify({"error": "batch_id parameter required"}), 400

    state = _get_batch_state()

    if state is None or state["batch_id"] != batch_id:
        return jsonify({"error": "Unknown batch_id"}), 404

    return jsonify({
        "batch_id": state["batch_id"],
        "results": state["results"],
    }), 200


@app.route("/output/<batch_id>/<filename>", methods=["GET"])
def serve_output(batch_id, filename):
    """Serve annotated output images."""
    output_dir = os.path.join(app.config["OUTPUT_FOLDER"], batch_id)

    if not os.path.isdir(output_dir):
        return jsonify({"error": "Unknown batch_id"}), 404

    file_path = os.path.join(output_dir, filename)
    if not os.path.isfile(file_path):
        return jsonify({"error": "File not found"}), 404

    return send_from_directory(output_dir, filename)


if __name__ == "__main__":
    logger.info("Web UI available at http://localhost:5001")
    app.run(host="127.0.0.1", port=5001, debug=False)
