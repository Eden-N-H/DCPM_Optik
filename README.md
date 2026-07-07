# DCPM Road Defect Measurement System

A computer vision and photogrammetry pipeline developed for Disaster Claims & Project Management (DCPM). This system automates the measurement and spatial mapping of road damage (potholes, ruts, scouring) required for government disaster funding claims, using only consumer-grade GoPro hardware. 

It calculates the physical surface area (m²) and spatial coordinates of defects by applying vector-based orthorectification to video/photo frames, utilizing embedded GPMF (GoPro Metadata Format) IMU and GPS telemetry.

## Core Capabilities

* **GPMF Telemetry Parsing:** Extracts high-frequency GPS, IMU (Gravity Vector), speed, and FOV data directly from standard and 360° GoPro MP4/JPG files.
* **Vector-Based Orthorectification:** Flattens perspective images into metric Bird's-Eye View (BEV) projections using camera height, dynamic pitch/roll compensation, and fisheye lens undistortion.
* **Pixel-Precise AI Segmentation:** Uses YOLOv8 for defect classification and SAM2 (Segment Anything 2) for exact boundary polygon extraction.
* **Geospatial Mapping:** Generates GIS-ready GeoJSON features, mapping physical defect boundaries to real-world coordinates.
* **Interactive UI:** A Flask/VanillaJS web dashboard featuring a MapLibre orthomosaic map, point-and-click vanishing point calibration, and manual defect drawing/editing.

## Architecture

1. **Telemetry Ingestion:** `parser_gpmf.py` and `telemetry.py` read the binary GPMF track, interpolating GPS and gravity vectors per frame.
2. **Flattening:** `cv_bev.py` computes the homography matrix using the telemetry-derived gravity vector and specified camera height to yield a 1px = fixed metric (e.g., 2cm) orthorectified grid.
3. **ML Segmentation:** `pipeline_image.py` routes the flattened image through YOLO + SAM2, extracting pixel areas.
4. **Export:** Results are bundled into zip files containing raw images, flattened BEV tiles, and GeoJSON boundaries for GIS software.

## Project Structure

```text
.
├── models/             # YOLO (.pt) and SAM2 weights (not tracked in git)
├── src/
│   ├── app.py              # Main Flask application (Runs on port 5001)
│   ├── cv_*.py             # Computer Vision: BEV projection, homography, vanishing points
│   ├── parser_*.py         # Metadata extraction: GPMF binary parsing, EXIF
│   ├── pipeline_*.py       # Processing pipelines for single images and sequential video
│   ├── sam2_integration.py # Segment Anything 2 integration logic
│   └── telemetry.py        # GPS/IMU signal interpolation and health scoring
├── static/             # JS, CSS, and output image uploads/tiles
└── templates/          # HTML frontend
```

## Setup & Installation

**1. Clone and setup environment:**
```bash
git clone https://github.com/Eden-N-H/DCPM_Optik.git
cd DCPM_Optik/homography-webui
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**2. Install SAM2 (Segment Anything 2):**
SAM2 must be installed in your environment. Follow the official Meta instructions to install `sam2` and its dependencies (requires PyTorch).

**3. Download Models:**
Place the following model weights in the `/models` directory:
* `RMCC_8_classes.pt` (Custom YOLO model)
* `sam2.1_hiera_large.pt` (SAM2 checkpoint)

*Note: The system will still run bounding boxes if SAM2 is unavailable, but polygon segmentation requires it.*

## Usage

1. **Start the server:**
   ```bash
   python src/app.py
   ```
2. **Access the UI:**
   Open `http://localhost:5001` in your browser.
3. **Process Media:**
   * Drop `.jpg` or `.mp4` (Standard or 360 GoPro files) into the upload zone.
   * Toggle **"Has Embedded Telemetry"** if using native GoPro files.
   * Input the **Camera Height (m)** (distance from the road surface to the lens).
   * Click **Process Uploads**.
4. **QA & Export:**
   * Review detected boundaries in the dual rectilinear/BEV viewers.
   * Adjust projection calibration manually via the UI if the IMU telemetry drifted.
   * Click **Export RAW ZIP** or **Export Flattened ZIP** to download the geospatial bundle.
