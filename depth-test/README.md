# Multi-Task Road Quality Analysis Pipeline

A synthetic data-driven deep learning framework for road surface condition assessment, scene mapping, and sim-to-real domain adaptation. 

This repository implements the pipeline described in:
*A Multi-Task Deep Learning Framework for Road Quality Analysis with Scene Mapping via Sim-to-Real Adaptation* (Appl. Sci. 2025).

## System Architecture

The pipeline consists of four main components:
1. **Procedural Data Generation**: Headless Blender 4.0+ pipeline generating 3D road scenes with defects (cracks, potholes, patches, puddles, manholes) and extracting pixel-perfect ground truth (RGB, Depth, Segmentation, Severity, Camera Params).
2. **Stage 1 (Sim-to-Real)**: Segmentation-aware CycleGAN that translates synthetic images into realistic target domains while preserving geometric defect fidelity using a custom masking loss.
3. **Stage 2 (Multi-Task Model)**: Modified DeepLabv3+ with a ResNet-50 backbone, Depthwise Separable Convolutions (DSC), Small-Object Attention (SOA), and Efficient ASPP. It jointly predicts:
   - Discrete Defect Segmentation [B, 3, H, W]
   - Continuous Defect Severity [B, 1, H, W]
   - Monocular Depth [B, 1, H, W]
   - Camera Parameters (Intrinsics [B, 4], Extrinsics [B, 6])
4. **3D Scene Reconstruction**: Pin-hole depth unprojection and world-space transformation to generate 3D point clouds and 2D Bird's-Eye View (BEV) maps directly from single-image inferences.

---

## Installation

**Python Dependencies**
```bash
pip install -r requirements.txt
```
*Requires Python 3.9+ and PyTorch >= 2.0.*

**Blender Dependency** (Only required for `data` generation)
- Install [Blender 4.0+](https://www.blender.org/download/).
- Ensure the `blender` executable is in your system's `PATH`.

---

## Command Line Interface (CLI)

The pipeline is executed via `src.main`. 

### 1. Synthetic Data Generation
Generates procedural road scenes utilizing headless Blender workers.
```bash
python -m src.main data --config configs/default.yaml --output-dir ./data/road_quality
```

### 2. Model Training
Trains the multi-task model with dual-discriminator adversarial domain adaptation.
```bash
python -m src.main train \
    --config configs/default.yaml \
    --output-dir ./checkpoints \
    --training.batch_size=8 \
    --training.optimizer.lr=1e-4
```
*To resume training:* Append `--resume ./checkpoints/checkpoint_epoch_X.pt`

### 3. Evaluation
Evaluates the multi-task model on the test split. Computes mIoU, RMSE, translation/rotation error, and severity MAE.
```bash
python -m src.main evaluate --config configs/default.yaml --checkpoint ./checkpoints/best_model.pt
```

### 4. 3D Reconstruction & BEV Mapping
Runs inference on a video or image directory, unprojects depth, applies extrinsics, and exports a `.ply` point cloud and `.png` BEV map.
```bash
python -m src.main reconstruct \
    --config configs/default.yaml \
    --checkpoint ./checkpoints/best_model.pt \
    --input ./test_video.mp4 \
    --output ./reconstruction_output
```

### 5. Visualization
Generates 3x5 storyboard grids comparing Ground Truth, CycleGAN translated outputs, and Multi-Task predictions/errors.
```bash
python -m src.main visualize \
    --config configs/default.yaml \
    --cyclegan-ckpt ./checkpoints/cyclegan.pt \
    --multitask-ckpt ./checkpoints/best_model.pt \
    --samples 5 \
    --output-dir ./visualizations
```

### 6. Web UI
Starts a Flask-based task dispatcher and SQLite database for managing pipeline runs visually.
```bash
python -m src.main web --port 5000 --host 0.0.0.0
```

### 7. Colab Generation
Generates a standalone Jupyter Notebook (`Colab_Pipeline.ipynb`) configured with Google Drive mounting and Blender 4.0 binary downloading for remote execution.
```bash
python -m src.main colab --output Colab_Pipeline.ipynb
```

---

## Configuration overrides

The system uses YAML configuration (`configs/default.yaml`). Any parameter can be overridden from the CLI using dot-notation:

```bash
python -m src.main train \
    --model.encoder.backbone=resnet50 \
    --training.amp=true \
    --training.loss_weights.segmentation=1.5 \
    --data.num_workers=8
```

---

## Testing

The repository contains extensive unit, integration, and property-based tests (using `pytest` and `hypothesis`). 

Run the test suite:
```bash
pytest tests/
```
Test categories include:
- `tests/unit/`: Component isolation tests (heads, encoder, decoder, losses, configuration parsing).
- `tests/integration/`: End-to-end pipeline compatibility and TorchScript tracing.
- `tests/properties/`: Hypothesis-driven invariants (e.g., loss non-negativity, deterministic checkpoint round-tripping, geometrical unprojection guarantees).

---

## Directory Structure

```text
├── configs/
│   └── default.yaml            # Master configuration
├── src/
│   ├── cyclegan/               # Stage-1 sim-to-real models and losses
│   ├── model/                  # Stage-2 multi-task architecture (DeepLabv3+, DSC, SOA)
│   ├── reconstruction/         # 3D unprojection, BEV logic, PLY aggregation
│   ├── synth/                  # Blender procedural generation & Dataset Builder
│   ├── training/               # Multi-task training loops, AMP, metrics, dataset I/O
│   ├── visualization/          # Grid and storyboard rendering
│   ├── utils/                  # ConfigLoader, Checkpointing, TensorBoard JSONL loggers
│   ├── main.py                 # CLI entry point
│   ├── web.py                  # Flask Web UI Subprocess manager
│   └── generate_colab.py       # Colab auto-generation script
└── tests/                      # Unit, Integration, and Property-based tests
```