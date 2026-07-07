# Implementation Plan: Road Defect Segmentation Pipeline

## Overview

This plan implements a sequential road defect segmentation pipeline in Python targeting Apple Silicon (M1) with MPS acceleration. The pipeline processes GoPro-captured road imagery through: ingestion → preprocessing → YOLO detection → SAM2 segmentation → verification → measurement → JSON output. Implementation progresses from configuration and data models through each pipeline stage, wiring everything together with the orchestrator.

## Tasks

- [x] 1. Set up project structure and core data models
  - [x] 1.1 Create project directory structure and dependencies
    - Create directory layout: `src/`, `src/pipeline/`, `tests/unit/`, `tests/property/`, `tests/integration/`, `models/`, `config/`
    - Create `pyproject.toml` or `requirements.txt` with dependencies: opencv-python, numpy, torch, ultralytics, pycocotools, pyyaml, hypothesis, pytest
    - Create a sample `config/default_config.yaml` matching the configuration schema from the design
    - _Requirements: 8.1, 8.2_

  - [x] 1.2 Implement core data model classes
    - Create `src/pipeline/models.py` with dataclasses: `FrameMetadata`, `Detection`, `SegmentationResult`, `VerifiedResult`, `DefectMeasurement`, `DefectOutput`, `FrameResult`, `BatchSummary`
    - Ensure all fields match the design document interfaces exactly
    - Include `PipelineConfig` dataclass with all documented defaults and valid ranges
    - _Requirements: 7.1, 8.1, 8.2_

  - [ ]* 1.3 Write unit tests for data models
    - Test dataclass instantiation with valid and invalid values
    - Test default values are applied correctly
    - _Requirements: 8.2_

- [x] 2. Implement Configuration Manager
  - [x] 2.1 Implement ConfigManager class
    - Create `src/pipeline/config_manager.py` with `ConfigManager` class
    - Implement `load()` method: read YAML file, apply defaults for missing fields, return `PipelineConfig`
    - Implement `validate()` method: check all values against documented ranges (confidence_threshold 0.0–1.0, iou_threshold 0.0–1.0, frame_extraction_rate 0.1–30, etc.), return list of all errors
    - Implement CLI argument parsing for `--config` flag
    - Exit with descriptive error if config path not provided, file missing, or YAML invalid
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_

  - [ ]* 2.2 Write property tests for configuration
    - **Property 19: Configuration defaults** — For any valid YAML with a subset of parameters, unspecified parameters take documented defaults
    - **Property 20: Configuration validation completeness** — For any config with N out-of-range values, all N are reported
    - **Validates: Requirements 8.2, 8.5**

  - [ ]* 2.3 Write unit tests for ConfigManager
    - Test missing config file path exits with correct error message
    - Test invalid YAML syntax exits with parse error details
    - Test out-of-range values produce validation errors
    - Test partial config applies defaults correctly
    - _Requirements: 8.3, 8.4, 8.5_

- [x] 3. Implement Logger
  - [x] 3.1 Implement pipeline logging system
    - Create `src/pipeline/logger.py` with configurable logging setup
    - Support log levels: DEBUG, INFO, WARNING, ERROR
    - Output to both stdout and a log file in the configured output directory
    - Format each entry with: ISO 8601 timestamp, log level, component name, message text
    - _Requirements: 9.1, 9.2_

  - [ ]* 3.2 Write property test for log entry format
    - **Property 21: Log entry format** — For any log message, output contains ISO 8601 timestamp, log level, component name, and message text
    - **Validates: Requirements 9.2**

- [x] 4. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 5. Implement Frame Ingester
  - [x] 5.1 Implement FrameIngester class
    - Create `src/pipeline/frame_ingester.py` with `FrameIngester` class
    - Implement `ingest()` as a generator yielding `(np.ndarray, FrameMetadata)` tuples
    - Support JPEG, PNG image files and MP4, MOV video files
    - Implement `_extract_video_frames()` using OpenCV VideoCapture with configurable frame rate
    - Implement `_validate_resolution()`: accept frames between 64x64 and 5312x2988 pixels
    - Reject unsupported formats with error message listing supported formats
    - Log and skip corrupt/unreadable frames, continue processing
    - Log video extraction failures with last successfully extracted frame index
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6_

  - [ ]* 5.2 Write property tests for frame ingestion
    - **Property 1: Resolution validation** — Accept frame iff 64≤W≤5312 and 64≤H≤2988
    - **Validates: Requirements 1.1, 1.6**

  - [ ]* 5.3 Write property test for frame extraction count
    - **Property 2: Frame extraction count** — For video duration D and rate R, frames == floor(D × R)
    - **Validates: Requirements 1.2**

  - [ ]* 5.4 Write property test for unsupported format rejection
    - **Property 3: Unsupported format rejection** — Non-supported extensions produce error with format names
    - **Validates: Requirements 1.3**

- [x] 6. Implement Preprocessor
  - [x] 6.1 Implement Preprocessor class
    - Create `src/pipeline/preprocessor.py` with `Preprocessor` class
    - Implement `_correct_barrel_distortion()` using `cv2.undistort()` with 5-parameter model (k1, k2, p1, p2, k3)
    - Use default GoPro wide-angle calibration coefficients when none provided, log INFO message
    - Implement `_reduce_motion_blur()` using unsharp masking or Wiener deconvolution
    - Implement `_apply_clahe()` on L channel in LAB color space using `cv2.createCLAHE()` with configurable clip_limit and tile_grid_size
    - Enforce fixed processing order: distortion → blur reduction → CLAHE
    - Preserve input dimensions and output 8-bit RGB
    - On failure, return original frame unchanged and log warning
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7_

  - [ ]* 6.2 Write property test for preprocessor preservation
    - **Property 4: Preprocessor dimension and dtype preservation** — Output has same shape (H, W, 3) and dtype uint8
    - **Validates: Requirements 2.4**

  - [ ]* 6.3 Write unit tests for preprocessor
    - Test processing order (distortion → blur → CLAHE) via mock call sequence
    - Test default calibration coefficients usage logs INFO
    - Test failure returns original frame
    - _Requirements: 2.1, 2.6, 2.7_

- [x] 7. Implement YOLO Detector
  - [x] 7.1 Implement YOLODetector class
    - Create `src/pipeline/yolo_detector.py` with `YOLODetector` class
    - Implement `load_model()` using Ultralytics API with MPS device; raise `ModelLoadError` on failure
    - Implement `detect()` using `model.predict()` with configured confidence, IoU threshold, and max detections
    - Map YOLO class indices to labels: pothole, longitudinal_crack, transverse_crack, alligator_cracking, patch_deterioration
    - NMS is handled internally by Ultralytics with configured IoU threshold
    - Return empty list when no defects found
    - Raise/log error on inference failure with frame identifier
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8_

  - [ ]* 7.2 Write property tests for detection
    - **Property 5: Detection class label constraint** — All labels in allowed set
    - **Property 6: Detection output filtering invariants** — All confidence ≥ T, count ≤ M
    - **Property 7: NMS post-condition** — No pair with IoU > threshold
    - **Validates: Requirements 3.2, 3.3, 3.5, 3.6**

  - [ ]* 7.3 Write unit tests for YOLO detector
    - Test model load failure raises ModelLoadError with path and reason
    - Test empty detection list for no-defect frames
    - Test inference error is logged with frame ID
    - _Requirements: 3.1, 3.8, 9.3_

- [x] 8. Implement SAM2 Segmenter
  - [x] 8.1 Implement SAM2Segmenter class
    - Create `src/pipeline/sam2_segmenter.py` with `SAM2Segmenter` class
    - Implement `load_model()` using SAM2ImagePredictor with MPS device; raise `ModelLoadError` on failure
    - Implement `segment()`: set image once per frame, batch all valid box prompts into single `predict()` call
    - Implement `_validate_bbox()`: skip boxes with zero width/height or out-of-frame coordinates, log WARNING
    - Convert YOLO bbox format (x, y, w, h) to SAM2 format (x1, y1, x2, y2)
    - Discard masks with zero foreground pixels, log WARNING
    - Return `SegmentationResult` list with binary masks at original frame resolution
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7_

  - [ ]* 8.2 Write property tests for segmentation
    - **Property 8: Segmentation mask shape invariant** — Mask shape == frame shape (H, W) with values in {0, 1}
    - **Property 9: Zero-foreground mask exclusion** — No all-zero masks in output
    - **Property 10: Bounding box validation** — Invalid bboxes rejected before segmentation
    - **Validates: Requirements 4.4, 4.5, 4.6**

- [x] 9. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 10. Implement Post-Segmentation Verifier
  - [x] 10.1 Implement PostSegmentationVerifier class
    - Create `src/pipeline/verifier.py` with `PostSegmentationVerifier` class
    - Implement `_compute_area_ratio()`: foreground pixels in bbox region / (bbox width × bbox height)
    - Implement `_clean_connected_components()` using 8-connectivity, retain only largest component
    - Implement `verify()`: discard if ratio < min_area_ratio (log WARNING); set review_flag if ratio > max_area_ratio (log WARNING); pass through otherwise
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

  - [ ]* 10.2 Write property tests for verification
    - **Property 11: Area ratio verification** — Correct filtering and flagging based on thresholds
    - **Property 12: Connected component cleanup** — At most one connected component after cleanup
    - **Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5**

- [x] 11. Implement Measurement Engine
  - [x] 11.1 Implement MeasurementEngine class
    - Create `src/pipeline/measurement_engine.py` with `MeasurementEngine` class
    - Implement `measure()`: compute area_pixels as count of non-zero pixels
    - Implement `_pixel_to_cm2()`: area_cm2 = area_pixels × (camera_height_cm / focal_length_px)²
    - Implement `_compute_bounding_dimensions()`: max_row - min_row, max_col - min_col from mask extent
    - Implement `_compute_severity()`: minor (<500 cm²), moderate (500–2000 cm²), severe (>2000 cm²)
    - Return None for metric fields and severity when camera parameters unavailable or invalid (zero/negative)
    - Log WARNING when camera parameters are zero or negative, fall back to pixel-only
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7_

  - [ ]* 11.2 Write property tests for measurement
    - **Property 13: Pixel area measurement** — area_pixels == count_nonzero(mask)
    - **Property 14: Metric conversion correctness** — area_cm2 == round(A × (h/f)², 1)
    - **Property 15: Severity classification** — Correct bracket assignment
    - **Validates: Requirements 6.1, 6.2, 6.3, 6.5**

  - [ ]* 11.3 Write unit tests for measurement edge cases
    - Test pixel-only mode when camera params not provided
    - Test fallback when camera params are zero or negative
    - Test rounding to one decimal place
    - _Requirements: 6.4, 6.6, 6.7_

- [x] 12. Implement Output Writer
  - [x] 12.1 Implement OutputWriter class
    - Create `src/pipeline/output_writer.py` with `OutputWriter` class
    - Implement `write_frame_result()`: serialize per-frame JSON with frame_id, ISO 8601 timestamp, defects list
    - Each defect includes: class, confidence, bounding_box, segmentation (COCO RLE), measurements, review_flag
    - Implement `_encode_mask_rle()` using `pycocotools.mask.encode()` for COCO compressed RLE format
    - Implement `write_batch_summary()`: total frames, total defects, counts by class, counts by severity, processing times
    - Write per-frame files as `{frame_id}.json` in output directory
    - Write `batch_summary.json` at end of run
    - Validate output directory exists and is writable at startup; exit with error if not
    - Produce JSON record with empty defects list for frames with zero detections
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6_

  - [ ]* 12.2 Write property tests for output
    - **Property 16: JSON output schema completeness** — All required fields present
    - **Property 17: RLE mask encoding round-trip** — decode(encode(mask)) == mask
    - **Property 18: Batch summary consistency** — Totals match sum of per-frame counts
    - **Validates: Requirements 7.1, 7.2, 7.4**

- [x] 13. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 14. Implement Pipeline Orchestrator and Error Handling
  - [x] 14.1 Implement PipelineOrchestrator class
    - Create `src/pipeline/orchestrator.py` with `PipelineOrchestrator` class
    - Implement `__init__()`: load config, validate, initialize all components (ConfigManager, Logger, FrameIngester, Preprocessor, YOLODetector, SAM2Segmenter, PostSegmentationVerifier, MeasurementEngine, OutputWriter)
    - Implement `run()`: iterate input paths, process each frame through the full pipeline chain
    - Implement `_process_frame()`: preprocessing → detection → segmentation → verification → measurement → output
    - Implement failure tracking with `FailureTracker`: increment on frame failure, reset on success, exit if threshold exceeded
    - Handle fatal errors (model load, config, output dir) with immediate exit
    - Handle recoverable errors (frame failures) with log and continue
    - Log batch statistics (total time, per-frame average) at INFO level on completion
    - _Requirements: 9.3, 9.4, 9.5, 9.6, 9.7_

  - [ ]* 14.2 Write property test for consecutive failure threshold
    - **Property 22: Consecutive failure threshold** — Exit iff more than N consecutive failures; reset on success
    - **Validates: Requirements 9.6**

  - [ ]* 14.3 Write unit tests for orchestrator error handling
    - Test unhandled exception in frame logs stack trace and continues
    - Test model load failure exits with correct error message and path
    - Test batch completion logs total and per-frame processing time
    - _Requirements: 9.3, 9.4, 9.5, 9.7_

- [x] 15. Create CLI entry point and wire everything together
  - [x] 15.1 Implement main entry point
    - Create `src/pipeline/__main__.py` or `src/main.py` as the CLI entry point
    - Parse `--config` argument, create `PipelineOrchestrator`, call `run()` with input paths
    - Accept input paths as positional arguments (files or directories)
    - Implement proper exit codes: 0 for success, 1 for fatal error
    - Create `src/pipeline/__init__.py` to expose public API
    - _Requirements: 8.3, 9.3, 9.4_

  - [x] 15.2 Create example configuration and documentation
    - Create `config/example_config.yaml` with all parameters documented with comments
    - Create a `README.md` with usage instructions, configuration reference, and example invocation
    - _Requirements: 8.1, 8.2_

- [x] 16. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties from the design document using the `hypothesis` library
- Unit tests validate specific examples and edge cases using `pytest`
- The pipeline targets Apple Silicon (M1) with MPS (Metal Performance Shaders) acceleration — NOT NVIDIA CUDA
- SAM2 and YOLO models should be loaded onto the MPS device via PyTorch's `torch.device("mps")`
- All timing requirements (500ms preprocess, 300ms YOLO, 2000ms SAM2) are for M1 MPS hardware

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "1.2"] },
    { "id": 1, "tasks": ["1.3", "2.1", "3.1"] },
    { "id": 2, "tasks": ["2.2", "2.3", "3.2"] },
    { "id": 3, "tasks": ["5.1", "6.1", "7.1", "8.1"] },
    { "id": 4, "tasks": ["5.2", "5.3", "5.4", "6.2", "6.3", "7.2", "7.3", "8.2"] },
    { "id": 5, "tasks": ["10.1", "11.1", "12.1"] },
    { "id": 6, "tasks": ["10.2", "11.2", "11.3", "12.2"] },
    { "id": 7, "tasks": ["14.1"] },
    { "id": 8, "tasks": ["14.2", "14.3", "15.1"] },
    { "id": 9, "tasks": ["15.2"] }
  ]
}
```
