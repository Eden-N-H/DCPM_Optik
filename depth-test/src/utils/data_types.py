"""Core data structures for the road quality pipeline."""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Literal, Any
from pathlib import Path
import numpy as np


@dataclass
class DefectSpec:
    """Specification for a synthetic road defect to be placed in a scene.

    Attributes:
        defect_type: The category of road defect.
        position: (x, y) coordinates on the road surface in meters.
        orientation: Rotation angle in degrees, 0-360.
        scale: Type-dependent dimensions (e.g., length/width for cracks,
               diameter/depth for potholes).
    """
    defect_type: Literal["crack", "pothole", "puddle", "patch", "manhole"]
    position: Tuple[float, float]
    orientation: float
    scale: Tuple[float, ...]


@dataclass
class DefectInstance:
    """A placed defect instance in a rendered scene.

    Attributes:
        spec: The original defect specification.
        mesh_object: Reference to the Blender mesh object (bpy.types.Object or mock).
        bounding_box_2d: 2D bounding box in image space (x_min, y_min, x_max, y_max).
        area: Surface area of the defect in square meters.
    """
    spec: DefectSpec
    mesh_object: Any
    bounding_box_2d: Tuple[float, float, float, float]
    area: float


@dataclass
class CameraConfig:
    """Camera configuration for scene rendering.

    Attributes:
        view_type: Strictly "dashcam" (vehicle-mounted).
        height: Camera height above ground in meters.
        pitch: Camera pitch angle in degrees (negative = looking down).
        intrinsics: 3x3 camera intrinsic matrix.
        extrinsics: 3x4 camera extrinsic matrix [R|t].
    """
    view_type: Literal["dashcam"]
    height: float
    pitch: float
    intrinsics: np.ndarray
    extrinsics: np.ndarray


@dataclass
class RenderOutputs:
    """Paths to all render output files for a single sample.

    Attributes:
        rgb: Path to 512x512 RGB PNG image.
        depth: Path to 512x512 16-bit PNG depth map (millimeters).
        segmentation: Path to 512x512 integer-encoded segmentation PNG.
        severity: Path to 512x512 float32 severity map (NPY format).
        camera_params: Path to JSON file containing intrinsic (K) and extrinsic ([R|t]) matrices.
    """
    rgb: Path
    depth: Path
    segmentation: Path
    severity: Path
    camera_params: Path


@dataclass
class DatasetManifest:
    """Manifest describing the full dataset structure and contents.

    Attributes:
        root: Root directory of the dataset.
        total_samples: Total number of samples across all splits.
        splits: Sample counts per split, e.g. {"train": 12829, "val": 1604, "test": 1603}.
        samples: List of per-sample metadata dictionaries.
    """
    root: Path
    total_samples: int
    splits: Dict[str, int]
    samples: List[Dict[str, Any]]


@dataclass
class ModelOutput:
    """Multi-task model output tensors.

    Attributes:
        segmentation: [B, 3, 512, 512] class logits (torch.Tensor).
        severity: [B, 1, 512, 512] severity predictions in [0, 1] (torch.Tensor).
        depth: [B, 1, 512, 512] normalized depth in [0, 1] (torch.Tensor).
        intrinsics: [B, 4] predicted camera intrinsics (fx, fy, cx, cy) (torch.Tensor).
        extrinsics: [B, 6] predicted camera extrinsics (rodrigues3, translation3) (torch.Tensor).
    """
    segmentation: Any
    severity: Any
    depth: Any
    intrinsics: Any
    extrinsics: Any


@dataclass
class PointCloudData:
    """3D point cloud reconstructed from depth and camera parameters.

    Attributes:
        positions: [N, 3] world-space coordinates (x, y, z).
        colors: [N, 3] RGB color values as uint8.
        classes: [N] defect class ID per point as int.
        severities: [N] severity value per point as float.
    """
    positions: np.ndarray
    colors: np.ndarray
    classes: np.ndarray
    severities: np.ndarray


@dataclass
class BEVMap:
    """Bird's-eye view map aggregated from point cloud data.

    Attributes:
        image: [H, W, 3] color-coded visualization as uint8.
        class_grid: [H, W] dominant defect class per cell as int.
        severity_grid: [H, W] maximum severity per cell as float.
        origin: World-space origin (x, y) of the BEV map.
        resolution: Spatial resolution in meters per pixel.
    """
    image: np.ndarray
    class_grid: np.ndarray
    severity_grid: np.ndarray
    origin: Tuple[float, float]
    resolution: float


@dataclass
class Checkpoint:
    """Training checkpoint for saving and resuming experiments.

    Attributes:
        epoch: The epoch number at checkpoint time.
        model_state_dict: Model parameters (state_dict).
        optimizer_state_dict: Optimizer state.
        scheduler_state_dict: Learning rate scheduler state.
        best_metric: Best validation metric value achieved so far.
        rng_states: Random number generator states for reproducibility,
                    including python, numpy, torch, and cuda states.
    """
    epoch: int
    model_state_dict: Dict[str, Any]
    optimizer_state_dict: Dict[str, Any]
    scheduler_state_dict: Dict[str, Any]
    best_metric: float
    rng_states: Dict[str, Any]
