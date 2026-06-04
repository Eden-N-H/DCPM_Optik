# DCPM_Optik

## Technical Approach & Architecture

### Phase 1: Telemetry Ingestion & Calibration

- **Source Media:** Processing continuous MP4 video files to extract high-frequency sensor readings via the GoPro Metadata Format (GPMF).
- **Sensor Calibration:** Implementing an automated height calibration step. The inspector starts recording with the camera resting on the ground, then lifts and places it into the vehicle mount. The software processes the vertical accelerometer and GPS altitude delta during this sequence to automatically calculate the camera’s running height above the road surface.
- **Telemetry Sync:** Pulling 3-axis rotational IMU data (pitch, roll, yaw) to dynamically adjust camera orientation values as the vehicle drives over uneven terrain.

### Phase 2: Ortho-rectification (Homography)

- **Flattening:** Utilizing the calibrated camera height and real-time GPMF pitch/roll rotation angles to construct a homography matrix.
- **Lens Correction:** Applying de-warping algorithms to correct the wide-angle fisheye distortion typical of GoPros.
- **Perspective Correction:** Transforming the perspective-warped road surface into a uniform, orthorectified grid where each pixel represents a consistent physical dimension.

### Phase 3: ML Segmentation

- Running a custom segmentation model (e.g., YOLO segmentation or equivalent) on the flattened image to isolate the exact boundaries of the road defect.
- Classifying anomalies to distinguish real defects from shadows, vegetation, or debris.

### Phase 4: Pixel-to-Metric Calculations & GPS-Based Photogrammetry

- **Area Calculation:** Multiplying the sum of segmented defect pixels by the physical area value represented by a single orthorectified pixel.
- **GPS-Based Structure from Motion (SfM):** Instead of using a physical stereo-pair of cameras, the system leverages the vehicle's forward motion. By extracting sequential frames at defined intervals, the system tracks the defect’s relative perspective shift. It uses GPS and high-frequency IMU data to calculate the precise "baseline distance" traveled between those frames, allowing 3D depth reconstruction using single-camera sequential photogrammetry.
