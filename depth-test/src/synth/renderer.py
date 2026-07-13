"""Renderer for synthetic road scene data generation.

Implements domain randomization and multi-pass rendering to produce
aligned RGB, depth, segmentation, severity, and camera parameter outputs.

Uses a RenderBackend protocol to abstract Blender rendering operations,
enabling testing without the Blender Python API (bpy).
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Protocol, Tuple

import numpy as np

from src.utils.data_types import CameraConfig, RenderOutputs


# ---------------------------------------------------------------------------
# Domain randomization configuration
# ---------------------------------------------------------------------------

# HDRI environment map library (20+ maps per Req 1.5)
HDRI_LIBRARY: List[str] = [
    "clear_sky_noon",
    "clear_sky_sunset",
    "cloudy_morning",
    "cloudy_afternoon",
    "overcast_flat",
    "overcast_dark",
    "rainy_street",
    "rainy_highway",
    "urban_intersection",
    "urban_parking",
    "suburban_residential",
    "suburban_commercial",
    "rural_field",
    "rural_forest",
    "industrial_yard",
    "industrial_port",
    "coastal_road",
    "mountain_pass",
    "desert_highway",
    "tunnel_entrance",
    "bridge_overpass",
    "nighttime_streetlit",
]

# Weather types (Req 1.5)
WEATHER_TYPES: List[str] = ["clear", "overcast", "rain"]

# Vehicle placement range (Req 1.5)
VEHICLE_COUNT_RANGE: Tuple[int, int] = (0, 5)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class DomainRandomizationConfig:
    """Configuration for domain randomization parameters."""

    hdri_library: List[str] = field(default_factory=lambda: list(HDRI_LIBRARY))
    weather_types: List[str] = field(default_factory=lambda: list(WEATHER_TYPES))
    vehicle_count_range: Tuple[int, int] = (0, 5)


@dataclass
class RenderConfig:
    """Configuration for rendering parameters."""

    render_size: int = 512
    depth_scale_mm: bool = True  # depth encoded in millimeters
    segmentation_classes: Dict[str, int] = field(default_factory=lambda: {
        "background": 0,
        "road": 1,
        "crack": 2,
        "pothole": 3,
        "puddle": 4,
        "patch": 5,
        "manhole": 6,
        "vehicle": 7,
    })


# ---------------------------------------------------------------------------
# Domain Randomization State
# ---------------------------------------------------------------------------


@dataclass
class DomainRandomizationState:
    """Captures the state of domain randomization applied to a scene.

    Attributes:
        hdri_map: The selected HDRI environment map name.
        vehicle_count: Number of vehicles placed in the scene.
        vehicle_positions: List of (x, y, rotation) tuples for each vehicle.
        weather: The applied weather effect.
    """

    hdri_map: str
    vehicle_count: int
    vehicle_positions: List[Tuple[float, float, float]]
    weather: str


# ---------------------------------------------------------------------------
# Render Backend Protocol
# ---------------------------------------------------------------------------


class RenderBackend(Protocol):
    """Protocol for rendering operations, allowing mock implementations."""

    def set_hdri_environment(self, hdri_name: str) -> None:
        """Set the HDRI environment map for scene lighting."""
        ...

    def place_vehicle(
        self, position: Tuple[float, float], rotation: float
    ) -> Any:
        """Place a vehicle mesh in the scene.

        Args:
            position: (x, y) position on road surface.
            rotation: Rotation angle in degrees.

        Returns:
            Vehicle object handle.
        """
        ...

    def apply_weather_effect(self, weather: str) -> None:
        """Apply a weather effect to the scene.

        Args:
            weather: One of "clear", "overcast", "rain".
        """
        ...

    def render_rgb(self, output_path: Path, size: int) -> None:
        """Render the RGB image.

        Args:
            output_path: Path to save the RGB PNG.
            size: Image dimension (assumes square).
        """
        ...

    def render_depth(self, output_path: Path, size: int) -> None:
        """Render the depth map as 16-bit PNG (millimeters).

        Args:
            output_path: Path to save the depth PNG.
            size: Image dimension.
        """
        ...

    def render_segmentation(self, output_path: Path, size: int) -> None:
        """Render the integer-encoded segmentation mask.

        Args:
            output_path: Path to save the segmentation PNG.
            size: Image dimension.
        """
        ...

    def render_severity(self, output_path: Path, size: int) -> None:
        """Render the float32 severity map.

        Args:
            output_path: Path to save the severity map (NPY).
            size: Image dimension.
        """
        ...


# ---------------------------------------------------------------------------
# Mock Render Backend (for testing without Blender)
# ---------------------------------------------------------------------------


class MockRenderBackend:
    """Mock render backend that produces valid placeholder outputs for testing."""

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = np.random.default_rng(seed)
        self._vehicles: List[Dict[str, Any]] = []
        self._hdri: Optional[str] = None
        self._weather: str = "clear"

    def set_hdri_environment(self, hdri_name: str) -> None:
        self._hdri = hdri_name

    def place_vehicle(
        self, position: Tuple[float, float], rotation: float
    ) -> Dict[str, Any]:
        vehicle = {
            "type": "vehicle",
            "position": position,
            "rotation": rotation,
        }
        self._vehicles.append(vehicle)
        return vehicle

    def apply_weather_effect(self, weather: str) -> None:
        self._weather = weather

    def render_rgb(self, output_path: Path, size: int) -> None:
        """Generate a synthetic RGB image (512x512x3 uint8 PNG)."""
        img = self._rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
        _save_image_png(output_path, img)

    def render_depth(self, output_path: Path, size: int) -> None:
        """Generate a synthetic 16-bit depth map (millimeters)."""
        # Simulate depth values from 1000mm to 20000mm
        depth = self._rng.integers(1000, 20000, size=(size, size), dtype=np.uint16)
        _save_depth_png(output_path, depth)

    def render_segmentation(self, output_path: Path, size: int) -> None:
        """Generate a synthetic segmentation mask (integer-encoded)."""
        # Generate random class IDs: 0-7 per the segmentation classes
        seg = self._rng.integers(0, 8, size=(size, size), dtype=np.uint8)
        _save_image_png(output_path, seg)

    def render_severity(self, output_path: Path, size: int) -> None:
        """Generate a synthetic severity map (float32 in [0, 1])."""
        severity = self._rng.random(size=(size, size)).astype(np.float32)
        np.save(str(output_path), severity)


# ---------------------------------------------------------------------------
# File I/O helpers
# ---------------------------------------------------------------------------


def _save_image_png(path: Path, data: np.ndarray) -> None:
    """Save a numpy array as a PNG image.

    Args:
        path: Output file path.
        data: Image data (uint8 or uint16).
    """
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)

    if data.dtype == np.uint16:
        # cv2 supports 16-bit PNG natively
        cv2.imwrite(str(path), data)
    elif data.ndim == 2:
        cv2.imwrite(str(path), data)
    else:
        # Convert RGB to BGR for OpenCV
        cv2.imwrite(str(path), cv2.cvtColor(data, cv2.COLOR_RGB2BGR))


def _save_depth_png(path: Path, depth: np.ndarray) -> None:
    """Save a 16-bit depth map as PNG.

    Args:
        path: Output file path.
        depth: Depth values as uint16 (millimeters).
    """
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), depth.astype(np.uint16))


def save_camera_params_json(
    path: Path, intrinsics: np.ndarray, extrinsics: np.ndarray
) -> None:
    """Save camera parameters as a JSON file.

    Serializes the 3x3 intrinsics matrix K and 3x4 extrinsics matrix [R|t]
    to a JSON file with lists of lists representation.

    Args:
        path: Output JSON file path.
        intrinsics: 3x3 intrinsics matrix.
        extrinsics: 3x4 extrinsics matrix [R|t].
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    params = {
        "intrinsics_K": intrinsics.tolist(),
        "extrinsics_Rt": extrinsics.tolist(),
    }

    with open(path, "w") as f:
        json.dump(params, f, indent=2)


def load_camera_params_json(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load camera parameters from a JSON file.

    Args:
        path: Input JSON file path.

    Returns:
        Tuple of (intrinsics K [3,3], extrinsics [R|t] [3,4]).
    """
    with open(path, "r") as f:
        params = json.load(f)

    intrinsics = np.array(params["intrinsics_K"], dtype=np.float64)
    extrinsics = np.array(params["extrinsics_Rt"], dtype=np.float64)

    return intrinsics, extrinsics


# ---------------------------------------------------------------------------
# SceneRenderer class
# ---------------------------------------------------------------------------


class SceneRenderer:
    """Handles domain randomization and multi-pass rendering for scene generation.

    Applies domain randomization (HDRI environment maps, vehicle placement,
    weather effects) and renders all output modalities (RGB, depth, segmentation,
    severity, camera params) in a single logical pass.

    Attributes:
        dr_config: Domain randomization configuration.
        render_config: Rendering configuration.
        render_backend: Backend for actual rendering operations.
        rng: Random number generator for reproducible randomization.
    """

    def __init__(
        self,
        dr_config: Optional[DomainRandomizationConfig] = None,
        render_config: Optional[RenderConfig] = None,
        render_backend: Optional[RenderBackend] = None,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize the SceneRenderer.

        Args:
            dr_config: Domain randomization configuration. Uses defaults if None.
            render_config: Rendering configuration. Uses defaults if None.
            render_backend: Backend for rendering operations. Uses mock if None.
            seed: Random seed for reproducibility.
        """
        self.dr_config = dr_config or DomainRandomizationConfig()
        self.render_config = render_config or RenderConfig()
        self.render_backend = render_backend or MockRenderBackend(seed=seed)
        self.rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)

    def apply_domain_randomization(
        self,
        road_width: float,
        road_length: float,
    ) -> DomainRandomizationState:
        """Apply domain randomization to the current scene.

        Randomizes the HDRI environment map, places 0-5 vehicles,
        and applies a random weather effect.

        Args:
            road_width: Total road width in meters (for vehicle placement).
            road_length: Total road length in meters (for vehicle placement).

        Returns:
            DomainRandomizationState capturing what was applied.

        Raises:
            ValueError: If the HDRI library has fewer than 20 maps.
        """
        if len(self.dr_config.hdri_library) < 20:
            raise ValueError(
                f"HDRI library must have at least 20 maps, "
                f"got {len(self.dr_config.hdri_library)}"
            )

        # 1. Select random HDRI environment map
        hdri_map = self.rng.choice(self.dr_config.hdri_library)
        self.render_backend.set_hdri_environment(hdri_map)

        # 2. Place random number of vehicles (0-5)
        vehicle_min, vehicle_max = self.dr_config.vehicle_count_range
        vehicle_count = self.rng.randint(vehicle_min, vehicle_max)
        vehicle_positions: List[Tuple[float, float, float]] = []

        for _ in range(vehicle_count):
            # Place vehicles at random positions on the road
            vx = self.rng.uniform(0.0, road_width)
            vy = self.rng.uniform(0.0, road_length)
            v_rotation = self.rng.uniform(0.0, 360.0)
            self.render_backend.place_vehicle((vx, vy), v_rotation)
            vehicle_positions.append((vx, vy, v_rotation))

        # 3. Apply random weather effect
        weather = self.rng.choice(self.dr_config.weather_types)
        self.render_backend.apply_weather_effect(weather)

        return DomainRandomizationState(
            hdri_map=hdri_map,
            vehicle_count=vehicle_count,
            vehicle_positions=vehicle_positions,
            weather=weather,
        )

    def render(
        self,
        output_dir: Path,
        camera_config: CameraConfig,
        scene_id: str = "scene_000",
    ) -> RenderOutputs:
        """Render all output modalities in a single pass.

        Produces 512x512 RGB image, 16-bit depth map (mm), integer-encoded
        segmentation mask, float32 severity map, and camera parameters JSON.

        Args:
            output_dir: Directory to write output files.
            camera_config: Camera configuration with intrinsics and extrinsics.
            scene_id: Identifier for naming output files.

        Returns:
            RenderOutputs with paths to all generated files.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        size = self.render_config.render_size

        # Define output paths
        rgb_path = output_dir / f"{scene_id}_rgb.png"
        depth_path = output_dir / f"{scene_id}_depth.png"
        seg_path = output_dir / f"{scene_id}_segmentation.png"
        severity_path = output_dir / f"{scene_id}_severity.npy"
        camera_path = output_dir / f"{scene_id}_camera.json"

        # Render all modalities (single pass in Blender, sequential in mock)
        self.render_backend.render_rgb(rgb_path, size)
        self.render_backend.render_depth(depth_path, size)
        self.render_backend.render_segmentation(seg_path, size)
        self.render_backend.render_severity(severity_path, size)

        # Save camera parameters as JSON (Req 1.4)
        save_camera_params_json(
            camera_path, camera_config.intrinsics, camera_config.extrinsics
        )

        return RenderOutputs(
            rgb=rgb_path,
            depth=depth_path,
            segmentation=seg_path,
            severity=severity_path,
            camera_params=camera_path,
        )
