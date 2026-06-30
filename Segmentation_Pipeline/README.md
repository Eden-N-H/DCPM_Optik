# Defect Inspector — YOLO + SAM3 Segmentation Tool

A local web app that accepts any YOLO `.pt` model and an inspection image, runs object detection, then uses **SAM3** (Meta's Segment Anything Model 3) to produce pixel-precise defect outlines.

```
┌─────────────┐     ┌────────────────┐     ┌──────────────┐     ┌─────────────┐
│  Drop image │────▶│ YOLO detection  │────▶│ SAM3 masking │────▶│ Web result  │
│  + .pt file │     │  (your model)   │     │ (auto-prompt)│     │  + metrics  │
└─────────────┘     └────────────────┘     └──────────────┘     └─────────────┘
```

---

## Requirements

| | Minimum | Recommended |
|---|---|---|
| Python | 3.10 | 3.12 |
| GPU VRAM | 8 GB | 16 GB |
| Disk (SAM3 weights) | ~1.7 GB | — |
| CUDA | 12.1 | 12.8 |

> **CPU inference is supported** but SAM3 will be slow (~20–60 s per image). A CUDA or Apple Silicon GPU is strongly recommended.

---

## Project Structure

```
defect-inspector/
├── app.py                   ← Flask backend (YOLO + SAM3 inference)
├── requirements.txt         ← Python dependencies
├── install.sh               ← One-shot install script with progress bars
├── README.md                ← This file
├── models/                  ← (optional) place your .pt files here
├── templates/
│   └── index.html           ← Jinja2 template served by Flask
└── static/
    ├── css/
    │   └── style.css        ← UI styles
    └── js/
        └── script.js        ← Frontend logic
```

---

## Installation

### Option A — Automated (recommended)

A single script handles everything with a progress bar for each step:

```bash
# Clone / enter the project directory
cd defect-inspector

# CUDA (Linux/Windows WSL):
bash install.sh

# CPU only (macOS Intel or no GPU):
bash install.sh --cpu
```

The script will:
1. Check your Python version (3.10+ required)
2. Upgrade pip / setuptools / wheel
3. Install PyTorch (auto-detects CUDA 12.8, Apple Silicon MPS, or CPU)
4. Install SAM3 from Meta's GitHub
5. Install all remaining Python packages with per-package progress
6. Prompt you to authenticate with HuggingFace

---

### Option B — Manual (step by step)

#### Step 1 — Create a conda environment

```bash
conda create -n defect-inspector python=3.12
conda activate defect-inspector
```

#### Step 2 — Install PyTorch

**CUDA 12.8 (Linux / Windows WSL):**
```bash
pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128
```

**Apple Silicon (macOS M1/M2/M3):**
```bash
pip install torch torchvision
```

**CPU only:**
```bash
pip install torch torchvision
```

> For other CUDA versions, find the right command at https://pytorch.org/get-started/locally/

#### Step 3 — Install SAM3 from source

SAM3 is not on PyPI — install directly from Meta's GitHub:

```bash
pip install git+https://github.com/facebookresearch/sam3.git
```

#### Step 4 — Install remaining dependencies

```bash
pip install -r requirements.txt
```

#### Step 5 — Authenticate with HuggingFace

SAM3 checkpoints are gated. You need to:

1. Create a HuggingFace account at https://huggingface.co
2. Visit https://huggingface.co/facebook/sam3 and request access *(approved instantly)*
3. Generate a token at https://huggingface.co/settings/tokens
4. Log in:

```bash
huggingface-cli login
# Paste your token when prompted
```

The first time you run the server, SAM3 will automatically download its checkpoint (~1.7 GB) and cache it locally. Subsequent starts are instant.

---

## Running

### Step 1 — Start the Flask backend

```bash
python app.py
```

You should see:

```
[Defect Inspector] Starting on http://localhost:5001  (device=CUDA)
```

### Step 2 — Open the web UI

Open your browser and go to:

```
http://localhost:5001
```

Flask serves the UI directly — no separate web server needed.

---

## Usage

1. **Drop an inspection image** (PNG, JPG, WebP, BMP) onto the left panel
2. **Drop your YOLO `.pt` model** onto the right panel
3. Adjust the **confidence threshold** slider (default 40%)
4. Optionally type a **SAM3 text prompt** — if left blank, YOLO class names are used automatically (e.g. `crack, delamination`)
5. Click **Run analysis**

Results show:
- **Segmented view** — original image with YOLO boxes + SAM3 masks overlaid
- **Original view** — unmodified input
- **Split view** — draggable side-by-side comparison
- **Detection cards** — per-defect class, confidence, pixel area, SAM3 IOU score, and bounding box coordinates
- **Export PNG** — saves the annotated result image

---

## How It Works

### Detection (YOLO)
Your `.pt` model runs on the input image at the selected confidence threshold. YOLO returns bounding boxes, class labels, and confidence scores.

### Segmentation (SAM3)
SAM3 is prompted using the detected class names as a text concept prompt (e.g. `"crack, scratch"`). SAM3's **Promptable Concept Segmentation** finds and segments all matching instances in a single pass.

### Mask matching
Each SAM3 mask is matched to the nearest YOLO detection using **Intersection over YOLO Box (IoB)** — the fraction of the YOLO box area that overlaps with the SAM3 prediction.

### Output
The annotated image is returned as a base64 PNG. Per-defect metrics include pixel area, percentage of total image area, SAM3 confidence score, and IoB.

---

## Configuration

Edit the top section of `app.py` to change device or port:

```python
# Device is auto-detected, but you can force it:
DEVICE = "cuda"        # or "cpu" / "mps"
DTYPE  = torch.bfloat16
```

```python
# Change server port (bottom of app.py):
app.run(host="0.0.0.0", port=5001, debug=False)
```

If you change the port, update the top of `static/js/script.js` too:

```js
const SERVER = "http://localhost:5001";   // ← match app.py
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: sam3` | Run `pip install git+https://github.com/facebookresearch/sam3.git` |
| `401 Unauthorized` on HuggingFace | Run `huggingface-cli login` and paste a valid token |
| `CUDA out of memory` | Lower image resolution before uploading, or set `DEVICE = "cpu"` in `app.py` |
| Server shows `offline` in the UI | Make sure `python app.py` is running on port 5001 |
| No masks returned by SAM3 | Try a more specific SAM3 prompt, e.g. `"surface crack"` |
| Slow first inference | Normal — SAM3 initialises its encoder on the first image. Subsequent runs are faster. |
| Port 5001 already in use | Change port in `app.py` and `static/js/script.js` |

---

## License

SAM3 is released under the [SAM License](https://github.com/facebookresearch/sam3/blob/main/LICENSE). YOLO (Ultralytics) is under AGPL-3.0. This wrapper code is MIT.
