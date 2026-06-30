# Requirements Document

## Introduction

This document defines the requirements for a road defect segmentation pipeline. The pipeline processes GoPro-captured road corridor imagery, detects road defects (primarily potholes) using a YOLO object detection model, and generates pixel-precise segmentation masks using SAM2 (Segment Anything Model 2). The system supports batch processing of multiple defects per frame and produces defect measurements for severity estimation.

## Glossary

- **Pipeline**: The end-to-end image processing system that ingests frames, detects defects, segments them, and outputs measurement data
- **Frame**: A single image captured by the GoPro camera mounted on a vehicle
- **YOLO_Detector**: The YOLOv8/YOLO-based object detection model responsible for identifying road defects and producing bounding boxes with confidence scores
- **SAM2_Segmenter**: The Segment Anything Model 2 component that generates pixel-precise segmentation masks from box-prompted regions of interest
- **Preprocessor**: The component responsible for correcting lens distortion, motion blur, and exposure variations in raw GoPro frames
- **Postprocessor**: The component responsible for computing defect measurements (area, severity) from segmentation masks
- **ROI**: Region of Interest; the cropped sub-image area defined by a YOLO detection bounding box
- **Bounding_Box**: A rectangular region (x, y, width, height) with an associated confidence score and class label identifying a detected defect
- **Segmentation_Mask**: A binary pixel mask delineating the exact boundary of a detected defect within an ROI
- **Confidence_Threshold**: A configurable numeric value (0.0 to 1.0) that determines the minimum detection confidence required to proceed with segmentation
- **Barrel_Distortion**: A lens distortion characteristic of wide-angle GoPro cameras that causes straight lines to appear curved outward
- **Severity_Score**: A numeric classification of defect impact derived from area, depth estimation, and shape characteristics

## Requirements

### Requirement 1: Frame Ingestion

**User Story:** As a pipeline operator, I want to ingest GoPro-captured road corridor frames, so that the pipeline can process them for defect detection.

#### Acceptance Criteria

1. WHEN a frame is provided as input, THE Pipeline SHALL accept JPEG and PNG image formats with resolutions from 64x64 pixels up to 5312x2988 pixels
2. WHEN a video file is provided as input, THE Pipeline SHALL accept MP4 and MOV container formats and extract individual frames at a configurable frame rate between 0.1 and 30 frames per second (default: 1 frame per second)
3. IF an unsupported file format is provided, THEN THE Pipeline SHALL return an error message specifying the supported formats (JPEG, PNG for images; MP4, MOV for video)
4. IF a frame is corrupt or unreadable, THEN THE Pipeline SHALL log the error with the frame identifier and continue processing subsequent frames
5. IF a video file cannot be decoded or frame extraction fails mid-stream, THEN THE Pipeline SHALL log the error with the video file identifier and last successfully extracted frame index, and continue processing any subsequent input files
6. IF a frame resolution exceeds 5312x2988 pixels or is below 64x64 pixels, THEN THE Pipeline SHALL reject the frame with an error message indicating the acceptable resolution range

### Requirement 2: Frame Preprocessing

**User Story:** As a pipeline operator, I want raw GoPro frames to be corrected for lens distortion, motion blur, and exposure issues, so that downstream detection and segmentation models receive clean input.

#### Acceptance Criteria

1. WHEN a raw frame is received, THE Preprocessor SHALL apply barrel distortion correction using camera-specific calibration parameters, followed by motion blur reduction, followed by adaptive histogram equalization, in that fixed order
2. WHEN a raw frame is received, THE Preprocessor SHALL apply motion blur reduction using a deconvolution or sharpening filter
3. WHEN a raw frame is received, THE Preprocessor SHALL normalize exposure using adaptive histogram equalization with a configurable clip limit (default: 2.0) and tile grid size (default: 8x8 tiles)
4. THE Preprocessor SHALL preserve the original frame dimensions and output an 8-bit RGB image after all corrections
5. THE Preprocessor SHALL output a corrected frame within 500ms per frame on Apple Silicon (M1) hardware with MPS (Metal Performance Shaders) acceleration
6. WHERE barrel distortion calibration parameters are not provided, THE Preprocessor SHALL use default GoPro wide-angle calibration coefficients and log an INFO message indicating default coefficients are in use
7. IF preprocessing fails for a frame due to a processing error, THEN THE Preprocessor SHALL log the error with the frame identifier and pass the original uncorrected frame to the YOLO_Detector

### Requirement 3: YOLO Defect Detection

**User Story:** As a pipeline operator, I want the system to detect road defects in preprocessed frames, so that regions containing defects can be identified for precise segmentation.

#### Acceptance Criteria

1. WHEN a preprocessed frame is provided, THE YOLO_Detector SHALL produce zero or more Bounding_Box outputs, each with a class label and confidence score
2. THE YOLO_Detector SHALL detect the following defect classes: pothole, longitudinal crack, transverse crack, alligator cracking, and patch deterioration
3. IF a detection has a confidence score below the configurable Confidence_Threshold, THEN THE YOLO_Detector SHALL discard that detection
4. THE YOLO_Detector SHALL support a configurable Confidence_Threshold with a default value of 0.5 and a valid range of 0.0 to 1.0
5. WHEN multiple defects are present in a single frame, THE YOLO_Detector SHALL detect each defect independently and produce separate Bounding_Box outputs up to a configurable maximum detection limit (default: 50 detections per frame)
6. THE YOLO_Detector SHALL apply non-maximum suppression with a configurable IoU threshold (default: 0.45, valid range: 0.0 to 1.0) to eliminate duplicate detections
7. THE YOLO_Detector SHALL complete inference on a single frame within 300ms on Apple Silicon (M1) hardware with MPS (Metal Performance Shaders) acceleration
8. IF the YOLO_Detector encounters an error during inference on a frame, THEN THE Pipeline SHALL log the error with the frame identifier and continue processing subsequent frames

### Requirement 4: SAM2 Segmentation

**User Story:** As a pipeline operator, I want pixel-precise segmentation masks for each detected defect, so that defect boundaries are accurately delineated for measurement.

#### Acceptance Criteria

1. WHEN one or more Bounding_Box detections are produced for a frame, THE SAM2_Segmenter SHALL generate a Segmentation_Mask for each detection using the Bounding_Box as a box prompt
2. THE SAM2_Segmenter SHALL use SAM2 (Segment Anything Model 2) as the segmentation backbone
3. WHEN multiple defects are detected in a single frame, THE SAM2_Segmenter SHALL batch all ROIs into a single inference pass to reduce per-frame inference latency
4. THE SAM2_Segmenter SHALL produce a binary mask at the original frame resolution, with mask pixels aligned to the original frame coordinate system
5. IF SAM2_Segmenter produces a mask with zero foreground pixels, THEN THE Pipeline SHALL discard that detection and log a WARNING with the frame identifier and Bounding_Box coordinates
6. IF a Bounding_Box has zero width or zero height, or extends beyond the frame boundaries, THEN THE SAM2_Segmenter SHALL skip that detection, log a WARNING with the invalid coordinates, and continue processing remaining detections
7. THE SAM2_Segmenter SHALL complete inference for all detections in a single frame within 2000ms on Apple Silicon (M1) hardware with MPS (Metal Performance Shaders) acceleration

### Requirement 5: Post-Segmentation Verification

**User Story:** As a pipeline operator, I want segmentation results to be verified for quality, so that low-confidence or spurious masks are filtered out before measurement.

#### Acceptance Criteria

1. WHEN a Segmentation_Mask is produced, THE Postprocessor SHALL compute the mask-to-bounding-box area ratio as the number of foreground pixels in the mask divided by the total pixel area of the corresponding Bounding_Box
2. IF the mask-to-bounding-box area ratio is below a configurable minimum threshold (default: 0.05, valid range: 0.0 to 1.0), THEN THE Pipeline SHALL discard the mask, exclude it from the frame output record, and log a WARNING with the frame identifier, Bounding_Box coordinates, and computed ratio
3. IF the mask-to-bounding-box area ratio exceeds a configurable maximum threshold (default: 0.95, valid range: 0.0 to 1.0), THEN THE Pipeline SHALL include the mask in the frame output record with a review flag field set to true and log a WARNING with the frame identifier and computed ratio
4. WHEN a Segmentation_Mask contains disconnected regions using 8-connectivity, THE Postprocessor SHALL retain only the largest connected component by pixel count and discard all smaller components
5. WHEN all verification checks pass for a Segmentation_Mask, THE Postprocessor SHALL forward the mask to the measurement stage with no review flag

### Requirement 6: Defect Measurement and Severity Estimation

**User Story:** As a road maintenance engineer, I want area and severity measurements for each detected defect, so that I can prioritize repair actions.

#### Acceptance Criteria

1. WHEN a verified Segmentation_Mask is produced, THE Postprocessor SHALL compute the defect area in pixels by counting all foreground pixels in the mask
2. WHEN camera height and focal length parameters are provided, THE Postprocessor SHALL convert the defect area from pixels to square centimeters using a ground-plane projection, reporting values rounded to one decimal place
3. WHEN camera height and focal length parameters are provided, THE Postprocessor SHALL compute the bounding dimensions (maximum width and maximum length) of each defect from the Segmentation_Mask in centimeters, rounded to one decimal place
4. IF camera height and focal length parameters are not provided, THEN THE Postprocessor SHALL compute bounding dimensions in pixels only
5. WHEN defect area is available in square centimeters, THE Postprocessor SHALL assign a Severity_Score based on area thresholds: minor (less than 500 cm²), moderate (500–2000 cm²), severe (greater than 2000 cm²)
6. IF camera height and focal length parameters are not provided, THEN THE Postprocessor SHALL report measurements in pixels only and omit the Severity_Score
7. IF camera height or focal length values are zero or negative, THEN THE Postprocessor SHALL log a WARNING and fall back to pixel-only measurements for that run

### Requirement 7: Pipeline Output

**User Story:** As a pipeline operator, I want structured output for each processed frame, so that results can be integrated into downstream asset management systems.

#### Acceptance Criteria

1. THE Pipeline SHALL produce a JSON output record for each processed frame containing: frame identifier, timestamp in ISO 8601 format (YYYY-MM-DDThh:mm:ssZ), list of detected defects with class labels, confidence scores, bounding boxes, segmentation masks (as RLE-encoded binary), and measurements
2. WHEN segmentation masks are output, THE Pipeline SHALL encode masks using Run-Length Encoding (RLE) in COCO format
3. WHEN a frame is processed with zero defects detected, THE Pipeline SHALL produce a JSON output record containing the frame identifier, timestamp, and an empty defects list
4. THE Pipeline SHALL generate a JSON summary report per batch run containing: total frames processed, total defects detected, defect counts by class, and defect counts by severity
5. THE Pipeline SHALL write each per-frame JSON output record as a separate file named using the frame identifier within the configurable output directory
6. IF the configured output directory does not exist or is not writable, THEN THE Pipeline SHALL exit with an error message indicating the directory path and the access issue

### Requirement 8: Configuration Management

**User Story:** As a pipeline operator, I want all pipeline parameters to be configurable via a single configuration file, so that I can tune the pipeline without modifying code.

#### Acceptance Criteria

1. THE Pipeline SHALL accept a YAML configuration file specified via a command-line argument (--config), specifying all tunable parameters including: Confidence_Threshold, IoU threshold, area ratio thresholds, frame extraction rate, calibration parameters, camera height, focal length, output directory, and model paths
2. WHEN a configuration parameter is not specified in the configuration file, THE Pipeline SHALL use the documented default value
3. IF the configuration file path is not provided via command-line argument, THEN THE Pipeline SHALL exit with an error message instructing the user to provide a configuration file path
4. IF the configuration file is missing at the specified path or contains invalid YAML syntax, THEN THE Pipeline SHALL exit with an error message describing the configuration issue including file path and parse error details
5. THE Pipeline SHALL validate all configuration values at startup against their documented valid ranges and report all out-of-range parameters in a single error message before exiting

### Requirement 9: Error Handling and Logging

**User Story:** As a pipeline operator, I want comprehensive logging and graceful error handling, so that I can diagnose issues and the pipeline does not halt on recoverable errors.

#### Acceptance Criteria

1. THE Pipeline SHALL log all processing events at configurable verbosity levels (DEBUG, INFO, WARNING, ERROR) to both standard output and a log file in the configured output directory
2. THE Pipeline SHALL include in each log entry: ISO 8601 timestamp, log level, component name, and message text
3. IF the YOLO_Detector fails to load the model weights, THEN THE Pipeline SHALL exit with an error message specifying the expected model path and the load failure reason
4. IF the SAM2_Segmenter fails to load the model checkpoint, THEN THE Pipeline SHALL exit with an error message specifying the expected checkpoint path and the load failure reason
5. WHEN a single frame causes an unhandled exception during processing, THE Pipeline SHALL log the exception with a full stack trace and continue processing subsequent frames
6. IF more than a configurable number of consecutive frames (default: 10) fail processing, THEN THE Pipeline SHALL exit with an error message indicating a systematic failure
7. WHEN a batch run completes, THE Pipeline SHALL log total processing time and per-frame average processing time at INFO level
