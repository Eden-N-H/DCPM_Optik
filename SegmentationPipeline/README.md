# Road Defect Segmentation Pipeline

A sequential image processing pipeline that detects, segments, and measures road surface defects from GoPro-captured road corridor imagery. The pipeline uses YOLOv8 for object detection and Meta's SAM2 (Segment Anything Model 2) for pixel-precise segmentation, producing structured JSON output with defect measurements and severity classifications.

## Prerequisites

- **Python 3.10+**
- **Apple Silicon (M1/M2/M3) recommended** for MPS (Metal Performance Shaders) GPU acceleration
- **Model weights**:
  - YOLO road defect detection model (`models/yolo_road_defects.pt`)
  - SAM2 checkpoint (`models/sam2_checkpoint.pth`)

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd SegmentationKiro

# Install dependencies
pip install -e .

# Install development dependencies (for testing)
pip install -e ".[dev]"
```

## Usage

Run the pipeline with a configuration file and one or more input paths (images or videos):

```bash
python -m src.pipeline --config config/example_config.yaml /path/to/images/
```

### Examples

```bash
# Process a directory of images
python -m src.pipeline --config config/example_config.yaml ./road_images/

# Process a single video file
python -m src.pipeline --config config/example_config.yaml ./survey_video.mp4

# Process multiple inputs
python -m src.pipeline --config config/example_config.yaml ./images/ ./video.mp4
```

### Supported Input Formats

| Type   | Formats         |
|--------|-----------------|
| Images | JPEG, PNG       |
| Video  | MP4, MOV        |

Resolution range: 64×64 to 5312×2988 pixels.

## Configuration Reference

All parameters are configured via a YAML file. See `config/example_config.yaml` for a fully commented example.

### Pipeline

| Parameter              | Default | Range       | Description                                |
|------------------------|---------|-------------|--------------------------------------------|
| `frame_extraction_rate`| 1.0     | [0.1, 30.0] | Frame extraction rate from video (fps)     |

### Preprocessing

| Parameter                  | Default | Description                                        |
|----------------------------|---------|---------------------------------------------------|
| `clip_limit`               | 2.0     | CLAHE contrast enhancement clip limit              |
| `tile_grid_size`           | [8, 8]  | CLAHE tile grid size [rows, cols]                  |
| `distortion_coefficients`  | null    | Lens distortion [k1, k2, p1, p2, k3] or null      |
| `camera_matrix`            | null    | 3×3 camera intrinsic matrix or null                |

### Detection

| Parameter              | Default                          | Range      | Description                          |
|------------------------|----------------------------------|------------|--------------------------------------|
| `model_path`           | `models/yolo_road_defects.pt`    | —          | Path to YOLO model weights           |
| `confidence_threshold` | 0.5                              | [0.0, 1.0] | Minimum detection confidence         |
| `iou_threshold`        | 0.45                             | [0.0, 1.0] | NMS IoU threshold                    |
| `max_detections`       | 50                               | > 0        | Max detections per frame             |

### Segmentation

| Parameter         | Default                       | Description                    |
|-------------------|-------------------------------|--------------------------------|
| `checkpoint_path` | `models/sam2_checkpoint.pth`  | Path to SAM2 model checkpoint  |
| `model_cfg`       | `sam2_hiera_l.yaml`           | SAM2 model configuration       |

### Verification

| Parameter        | Default | Range      | Description                                    |
|------------------|---------|------------|------------------------------------------------|
| `min_area_ratio` | 0.05    | [0.0, 1.0] | Minimum mask/bbox area ratio (below = discard) |
| `max_area_ratio` | 0.95    | [0.0, 1.0] | Maximum mask/bbox area ratio (above = flag)    |

### Measurement

| Parameter         | Default | Description                                        |
|-------------------|---------|----------------------------------------------------|
| `camera_height_cm`| null    | Camera height above road (cm) for metric output    |
| `focal_length_px` | null    | Camera focal length (pixels) for metric output     |

### Output

| Parameter   | Default    | Description                         |
|-------------|------------|-------------------------------------|
| `directory` | `"output"` | Directory for JSON output files     |

### Logging

| Parameter                  | Default  | Valid Values                    | Description                            |
|----------------------------|----------|---------------------------------|----------------------------------------|
| `level`                    | `"INFO"` | DEBUG, INFO, WARNING, ERROR     | Log verbosity level                    |
| `max_consecutive_failures` | 10       | > 0                             | Consecutive failures before exit       |

## Pipeline Architecture

The pipeline processes frames through a sequential chain of stages:

```
Input (Images/Video)
  → Frame Ingester        Extract and validate individual frames
  → Preprocessor          Correct distortion, blur, and exposure
  → YOLO Detector         Detect defects, produce bounding boxes
  → SAM2 Segmenter        Generate pixel-precise segmentation masks
  → Verifier              Validate mask quality, filter/flag results
  → Measurement Engine    Compute area, dimensions, and severity
  → Output Writer         Serialize results to JSON
```

Each frame is processed independently. A failure in one frame does not affect subsequent frames.

### Defect Classes

The pipeline detects the following road defect types:

- Pothole
- Longitudinal crack
- Transverse crack
- Alligator cracking
- Patch deterioration

### Severity Classification

When camera parameters are provided, defects are classified by area:

| Severity | Area Threshold       |
|----------|---------------------|
| Minor    | < 500 cm²           |
| Moderate | 500 – 2000 cm²      |
| Severe   | > 2000 cm²          |

## Output Format

### Per-Frame JSON

Each processed frame produces a JSON file named `{frame_id}.json` in the output directory:

```json
{
  "frame_id": "video001_frame_0042",
  "timestamp": "2024-03-15T10:30:42Z",
  "source_file": "video001.mp4",
  "defects": [
    {
      "class": "pothole",
      "confidence": 0.87,
      "bounding_box": {
        "x": 120,
        "y": 340,
        "width": 200,
        "height": 150
      },
      "segmentation": {
        "size": [2988, 5312],
        "counts": "encoded_rle_string"
      },
      "measurements": {
        "area_pixels": 18500,
        "area_cm2": 2450.3,
        "width_cm": 45.2,
        "length_cm": 62.1,
        "width_pixels": 180,
        "length_pixels": 140,
        "severity": "severe"
      },
      "review_flag": false
    }
  ]
}
```

### Batch Summary

A `batch_summary.json` is written at the end of each run:

```json
{
  "total_frames_processed": 1500,
  "total_defects_detected": 237,
  "defects_by_class": {
    "pothole": 89,
    "longitudinal_crack": 52,
    "transverse_crack": 41,
    "alligator_cracking": 33,
    "patch_deterioration": 22
  },
  "defects_by_severity": {
    "minor": 112,
    "moderate": 78,
    "severe": 47
  },
  "processing_time_seconds": 842.5,
  "average_time_per_frame_seconds": 0.562
}
```

## Running Tests

```bash
# Run all tests
pytest

# Run unit tests only
pytest tests/unit/

# Run property-based tests only
pytest tests/property/

# Run with verbose output
pytest -v

# Run a specific test file
pytest tests/unit/test_config_manager.py
```

## Project Structure

```
SegmentationKiro/
├── config/
│   ├── default_config.yaml       # Default configuration
│   └── example_config.yaml       # Fully documented example config
├── models/                       # Model weights (not tracked in git)
│   ├── yolo_road_defects.pt
│   └── sam2_checkpoint.pth
├── src/
│   └── pipeline/
│       ├── __init__.py
│       ├── config_manager.py     # Configuration loading and validation
│       ├── frame_ingester.py     # Image/video frame extraction
│       ├── preprocessor.py       # Distortion, blur, exposure correction
│       ├── yolo_detector.py      # YOLO defect detection
│       ├── sam2_segmenter.py     # SAM2 segmentation
│       ├── verifier.py           # Post-segmentation quality checks
│       ├── measurement_engine.py # Defect measurement and severity
│       ├── output_writer.py      # JSON output serialization
│       ├── orchestrator.py       # Pipeline coordination
│       ├── models.py             # Data models and types
│       └── logger.py             # Logging configuration
├── tests/
│   ├── unit/                     # Unit tests
│   ├── property/                 # Property-based tests (hypothesis)
│   └── integration/              # End-to-end tests
├── output/                       # Default output directory
├── pyproject.toml                # Project metadata and dependencies
└── README.md
```
