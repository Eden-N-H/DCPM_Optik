# DCPM Road Damage Photogrammetry Engine

This application is a standalone web interface developed for **DCPM (Disaster Claims & Project Management)**. It processes batches of 360° vehicle-mounted GoPro imagery, detects road defects using a custom YOLO segmentation model, projects those defects onto an orthorectified top-down Bird's-Eye View (BEV), calculates metric surface area, and maps the results as a geo-referenced spatial track.

---

## Technical Features

1. **GPMF Auto-Pitch Extraction:** Parses raw byte streams from uploaded JPEGs to locate the `GRAV` (Gravity Vector) block. Automatically computes the camera's tilt angle per frame using trigonometry to account for vehicle bounce:
   $$\theta_{\text{pitch}} = \text{atan2}(Z_{\text{gravity}}, Y_{\text{gravity}})$$
2. **Dynamic 2D Orthorectification:** Maps perspective coordinates from a de-warped rectilinear view to a metric ground plane ($Y = \text{camera height}$) via Inverse Perspective Mapping (IPM).
3. **Strict Ground Sample Distance (GSD):** Standardizes the top-down BEV canvas at $1\text{ pixel} = 10\text{ mm}$ ($0.01\text{ m}$). Exact defect area is calculated from the pixel count of the warped segmentation mask:
   $$\text{Area} = N_{\text{pixels}} \times \text{GSD}^2$$
4. **Local-to-Global Geo-Referencing:** Derives driving heading (bearing) chronologically from consecutive GPS points. Projects local metric coordinates $(X, Z)$ of defect polygons to global Latitude/Longitude coordinates using the Haversine direct formula.
5. **Interactive Mapping & Trail:** Visualizes the vehicle's driving path (trail) and accurately scaled defect polygons on an Esri satellite map.

---

## Installation & Setup

### 1. Prerequisites

Ensure you have Python 3.10 or higher installed.

### 2. Environment Setup

Clone or construct the directory structure, navigate to the root directory, and run:

```bash
# Install required Python packages
pip install -r requirements.txt
```

### 3. Execution

Launch the Flask local server:

```bash
python app.py
```

Open a browser and navigate to: `http://127.0.0.1:5000`

---

## Operational Workflow

1. **Model Ingestion:** Drag and drop or browse to upload your custom YOLO segmentation `.pt` file (e.g., `Unsealed Damage.pt` or `RMCC 8 classes.pt`).
2. **Batch Image Upload:** Select multiple 360° `.jpg` or `.png` files containing embedded GPS EXIF data and GPMF telemetry.
3. **Camera Configuration:** Input the physical mounting height of the camera above the road surface in meters (default: `1.6`).
4. **Process Execution:** Click **Run Pipeline**.
5. **Analysis Workspace:**
   - **Left Panel:** Displays the currently selected image in sequence. Use **Prev** and **Next** to cycle chronologically. Displays the perspective view, the metric BEV, and a metadata summary detailing automatically extracted pitch and segmented defect areas.
   - **Right Panel:** Interactive map displays the trajectory path (dashed blue line) and precise geo-referenced polygons (red overlay). Clicking Next/Prev automatically centers the map on the respective camera capture coordinate.
