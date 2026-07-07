# Web Segmentation UI

A browser-based interface for the road defect segmentation pipeline. Upload images via drag-and-drop, run YOLO + SAM2 detection and segmentation, and view annotated results in an interactive gallery.

## Prerequisites

- Python 3.10+
- Required packages: `opencv-python`, `numpy`, `torch`, `ultralytics`, `sam2`, `flask`
- YOLO and SAM2 model weights downloaded to the `models/` directory (see project root README)

## Setup

Install the Flask dependency (all other packages are covered by the project's main dependencies):

```bash
pip install flask
```

## Running

Start the server from the project root:

```bash
python web/web_app.py
```

The app loads YOLO and SAM2 models at startup, which may take a moment. Once ready you'll see:

```
Web UI available at http://localhost:5000
```

## Usage

1. Open **http://localhost:5000** in a browser.
2. Drag and drop images (or an entire folder) onto the drop zone.
   - Supported formats: `.jpg`, `.jpeg`, `.png`
   - Maximum **100 images** per batch
   - Maximum **50 MB** per image
3. Watch the progress bar as images are processed through the pipeline.
4. View annotated results in the gallery — defects are highlighted with coloured overlays and bounding boxes.
5. Click any image for a full-resolution view.
6. Use **"Process Another Batch"** to reset and start over.

## Architecture

```
web/
├── web_app.py          # Flask application — routes, batch processing, pipeline integration
├── templates/
│   └── index.html      # Single-page HTML template (drop zone, progress, gallery)
├── static/
│   ├── app.js          # Client-side JavaScript (upload, polling, gallery logic)
│   └── style.css       # UI styles
├── uploads/            # Temporary storage for incoming images (auto-cleaned)
└── output/             # Annotated result images served to the browser
```

## Limitations

- **Single user** — only one batch can be processed at a time. Concurrent uploads are rejected.
- **Single batch** — submit a new batch only after the current one completes.
- **Startup cost** — YOLO and SAM2 models are loaded into memory when the server starts, which requires sufficient RAM/VRAM.
- **Local only** — the server binds to `localhost:5000` and is not intended for production or multi-user deployment.
