"""Scene generator for synthetic road data generation.

Procedurally generates 3D road scenes with configurable road geometry,
defect placement with overlap detection, and camera configurations.

Designed with an abstraction layer so that it can be tested without
the Blender Python API (bpy).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Protocol, Tuple

import numpy as np

from src.utils.data_types import CameraConfig, DefectInstance, DefectSpec


# ---------------------------------------------------------------------------
# Defect dimension constraints (Requirements 1.2)
# ---------------------------------------------------------------------------

DEFECT_DIMENSIONS: Dict[str, Dict[str, Tuple[float, float]]] = {
    "crack": {"length": (0.1, 2.0), "width": (0.005, 0.05)},
    "pothole": {"diameter": (0.1, 1.0), "depth": (0.02, 0.15)},
    "puddle": {"diameter": (0.2, 2.0)},
    "patch": {"length": (0.3, 3.0), "width": (0.3, 2.0)},
    "manhole": {"diameter": (0.5, 0.8)},
}

# Camera configuration constraints (Requirements 1.6)
CAMERA_CONFIGS: Dict[str, Dict[str, Tuple[float, float]]] = {
    "dashcam": {"height": (1.2, 1.5), "pitch": (-15.0, -5.0)},
    "drone": {"height": (8.0, 15.0), "pitch": (-90.0, -60.0)},
}

# Overlap constraint threshold (Requirements 1.8)
OVERLAP_THRESHOLD = 0.25

# Maximum repositioning attempts before giving up on a defect
MAX_REPOSITION_ATTEMPTS = 100


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------


@dataclass
class SceneConfig:
    """Configuration for scene generation parameters."""

    lanes_range: Tuple[int, int] = (1, 4)
    lane_width_range: Tuple[float, float] = (3.0, 3.75)
    road_length_range: Tuple[float, float] = (50.0, 200.0)
    defect_count_range: Tuple[int, int] = (1, 10)
    overlap_threshold: float = OVERLAP_THRESHOLD
    render_size: int = 512
    max_reposition_attempts: int = MAX_REPOSITION_ATTEMPTS


# ---------------------------------------------------------------------------
# Mesh Backend Protocol (abstraction for Blender/testing)
# ---------------------------------------------------------------------------


class MeshBackend(Protocol):
    """Protocol for 3D mesh operations, allowing mock implementations."""

    def create_road_mesh(
        self, lanes: int, lane_width: float, length: float
    ) -> Any:
        """Create a road mesh and return a mesh object handle."""
        ...

    def create_defect_mesh(
        self, defect_type: str, scale: Tuple[float, ...], position: Tuple[float, float], orientation: float
    ) -> Any:
        """Create a defect mesh and return a mesh object handle."""
        ...

    def setup_camera_object(
        self, height: float, pitch: float, view_type: str
    ) -> Any:
        """Set up the camera in the scene and return a camera object handle."""
        ...


# ---------------------------------------------------------------------------
# Default mock backend (for testing without Blender)
# ---------------------------------------------------------------------------


class MockMeshBackend:
    """Mock mesh backend for testing without Blender."""

    def create_road_mesh(
        self, lanes: int, lane_width: float, length: float
    ) -> Dict[str, Any]:
        return {
            "type": "road",
            "lanes": lanes,
            "lane_width": lane_width,
            "length": length,
            "width": lanes * lane_width,
        }

    def create_defect_mesh(
        self, defect_type: str, scale: Tuple[float, ...], position: Tuple[float, float], orientation: float
    ) -> Dict[str, Any]:
        return {
            "type": "defect",
            "defect_type": defect_type,
            "scale": scale,
            "position": position,
            "orientation": orientation,
        }

    def setup_camera_object(
        self, height: float, pitch: float, view_type: str
    ) -> Dict[str, Any]:
        return {
            "type": "camera",
            "height": height,
            "pitch": pitch,
            "view_type": view_type,
        }


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def compute_defect_bounding_box(
    defect_type: str, scale: Tuple[float, ...], position: Tuple[float, float], orientation: float
) -> Tuple[float, float, float, float]:
    """Compute the axis-aligned bounding box (AABB) of a defect on the road surface.

    Returns:
        (x_min, y_min, x_max, y_max) in road-surface coordinates (meters).
    """
    # Determine half-extents based on defect type
    if defect_type == "crack":
        half_length = scale[0] / 2.0  # length
        half_width = scale[1] / 2.0   # width
    elif defect_type in ("pothole", "puddle", "manhole"):
        radius = scale[0] / 2.0  # diameter -> radius
        half_length = radius
        half_width = radius
    elif defect_type == "patch":
        half_length = scale[0] / 2.0  # length
        half_width = scale[1] / 2.0   # width
    else:
        raise ValueError(f"Unknown defect type: {defect_type}")

    # Rotate corners and compute AABB
    angle_rad = math.radians(orientation)
    cos_a = abs(math.cos(angle_rad))
    sin_a = abs(math.sin(angle_rad))

    # Rotated AABB half-extents
    half_x = half_length * cos_a + half_width * sin_a
    half_y = half_length * sin_a + half_width * cos_a

    x, y = position
    return (x - half_x, y - half_y, x + half_x, y + half_y)


def compute_defect_area(defect_type: str, scale: Tuple[float, ...]) -> float:
    """Compute the approximate area of a defect in square meters.

    Args:
        defect_type: Type of defect.
        scale: Type-specific dimension tuple.

    Returns:
        Area in square meters.
    """
    if defect_type == "crack":
        return scale[0] * scale[1]  # length × width
    elif defect_type in ("pothole", "puddle", "manhole"):
        radius = scale[0] / 2.0
        return math.pi * radius * radius
    elif defect_type == "patch":
        return scale[0] * scale[1]  # length × width
    else:
        raise ValueError(f"Unknown defect type: {defect_type}")


def compute_overlap_fraction(
    box_a: Tuple[float, float, float, float],
    area_a: float,
    box_b: Tuple[float, float, float, float],
    area_b: float,
) -> float:
    """Compute the overlap fraction relative to the smaller defect's area.

    The overlap is computed as the intersection area of the two AABBs
    divided by the area of the smaller defect.

    Args:
        box_a: Bounding box (x_min, y_min, x_max, y_max) of defect A.
        area_a: Area of defect A.
        box_b: Bounding box (x_min, y_min, x_max, y_max) of defect B.
        area_b: Area of defect B.

    Returns:
        Overlap fraction in [0, 1]. Returns 0 if no intersection.
    """
    # Compute intersection rectangle
    x_min = max(box_a[0], box_b[0])
    y_min = max(box_a[1], box_b[1])
    x_max = min(box_a[2], box_b[2])
    y_max = min(box_a[3], box_b[3])

    if x_max <= x_min or y_max <= y_min:
        return 0.0

    intersection_area = (x_max - x_min) * (y_max - y_min)
    smaller_area = min(area_a, area_b)

    if smaller_area <= 0:
        return 0.0

    return intersection_area / smaller_area


def has_excessive_overlap(
    new_instance: DefectInstance,
    existing_instances: List[DefectInstance],
    threshold: float = OVERLAP_THRESHOLD,
) -> bool:
    """Check if a new defect instance overlaps excessively with any existing ones.

    Args:
        new_instance: The defect instance to check.
        existing_instances: List of already-placed defect instances.
        threshold: Maximum allowable overlap fraction (default 0.25).

    Returns:
        True if the new instance exceeds the overlap threshold with any existing instance.
    """
    for existing in existing_instances:
        overlap = compute_overlap_fraction(
            new_instance.bounding_box_2d,
            new_instance.area,
            existing.bounding_box_2d,
            existing.area,
        )
        if overlap > threshold:
            return True
    return False


def is_within_road_bounds(
    bbox: Tuple[float, float, float, float], road_width: float, road_length: float
) -> bool:
    """Check if a bounding box is fully within the road surface.

    The road surface spans x: [0, road_width], y: [0, road_length].

    Args:
        bbox: (x_min, y_min, x_max, y_max) bounding box of the defect.
        road_width: Total road width in meters.
        road_length: Total road length in meters.

    Returns:
        True if the bounding box is fully within the road.
    """
    x_min, y_min, x_max, y_max = bbox
    return x_min >= 0 and y_min >= 0 and x_max <= road_width and y_max <= road_length


def generate_intrinsics(
    focal_length: float, image_size: int = 512
) -> np.ndarray:
    """Generate a 3x3 camera intrinsics matrix.

    Args:
        focal_length: Focal length in pixels.
        image_size: Image dimension (assumes square sensor).

    Returns:
        3x3 intrinsics matrix K.
    """
    cx = image_size / 2.0
    cy = image_size / 2.0
    K = np.array([
        [focal_length, 0.0, cx],
        [0.0, focal_length, cy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    return K


def generate_extrinsics(
    height: float, pitch_deg: float, road_length: float
) -> np.ndarray:
    """Generate a 3x4 camera extrinsics matrix [R|t].

    The camera is placed at the specified height looking down the road
    at the given pitch angle.

    Args:
        height: Camera height above ground in meters.
        pitch_deg: Camera pitch angle in degrees (negative = looking down).
        road_length: Road length for positioning along the road.

    Returns:
        3x4 extrinsics matrix [R|t].
    """
    pitch_rad = math.radians(pitch_deg)

    # Rotation matrix: pitch around X-axis
    R = np.array([
        [1.0, 0.0, 0.0],
        [0.0, math.cos(pitch_rad), -math.sin(pitch_rad)],
        [0.0, math.sin(pitch_rad), math.cos(pitch_rad)],
    ], dtype=np.float64)

    # Translation: camera positioned above road midpoint
    t = np.array([[0.0], [height], [road_length / 2.0]], dtype=np.float64)

    # Combine [R|t]
    extrinsics = np.hstack([R, t])
    return extrinsics


# ---------------------------------------------------------------------------
# SceneGenerator class
# ---------------------------------------------------------------------------


class SceneGenerator:
    """Procedurally generates 3D road scenes with defects.

    This class generates road meshes, places defects with overlap detection,
    and configures cameras. It uses a MeshBackend protocol to abstract the
    actual 3D operations, enabling testing without Blender.

    Attributes:
        config: Scene generation configuration.
        backend: Mesh backend for 3D operations.
        rng: Random number generator for reproducible scenes.
    """

    def __init__(
        self,
        config: Optional[SceneConfig] = None,
        backend: Optional[MeshBackend] = None,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize the SceneGenerator.

        Args:
            config: Scene generation configuration. Uses defaults if None.
            backend: Mesh backend for 3D operations. Uses MockMeshBackend if None.
            seed: Random seed for reproducibility. Uses system entropy if None.
        """
        self.config = config or SceneConfig()
        self.backend = backend or MockMeshBackend()
        self.rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)

    def generate_road_mesh(
        self,
        lanes: Optional[int] = None,
        lane_width: Optional[float] = None,
        length: Optional[float] = None,
    ) -> Tuple[Any, float, float]:
        """Generate a road surface mesh with configurable geometry.

        Args:
            lanes: Number of lanes (1-4). Randomly selected from config range if None.
            lane_width: Width of each lane in meters (3.0-3.75m). Random if None.
            length: Road length in meters (50-200m). Random if None.

        Returns:
            Tuple of (road_mesh_object, total_road_width, road_length).

        Raises:
            ValueError: If parameters are outside valid ranges.
        """
        if lanes is None:
            lanes = self.rng.randint(*self.config.lanes_range)
        if lane_width is None:
            lane_width = self.rng.uniform(*self.config.lane_width_range)
        if length is None:
            length = self.rng.uniform(*self.config.road_length_range)

        # Validate constraints
        if not (1 <= lanes <= 4):
            raise ValueError(f"lanes must be 1-4, got {lanes}")
        if not (3.0 <= lane_width <= 3.75):
            raise ValueError(f"lane_width must be 3.0-3.75m, got {lane_width}")
        if not (50.0 <= length <= 200.0):
            raise ValueError(f"length must be 50-200m, got {length}")

        road_width = lanes * lane_width
        road_mesh = self.backend.create_road_mesh(lanes, lane_width, length)

        return road_mesh, road_width, length

    def place_defects(
        self,
        road_width: float,
        road_length: float,
        defects: Optional[List[DefectSpec]] = None,
        num_defects: Optional[int] = None,
    ) -> List[DefectInstance]:
        """Place defect instances on the road surface with overlap resolution.

        Generates and places defects while ensuring no pair overlaps by more
        than 25% of the smaller defect's area. If overlap is detected, the
        later-placed defect is repositioned.

        Args:
            road_width: Total road width in meters.
            road_length: Total road length in meters.
            defects: Pre-defined defect specifications. If None, generates random defects.
            num_defects: Number of defects to generate if defects is None (1-10).

        Returns:
            List of placed DefectInstance objects.

        Raises:
            ValueError: If num_defects is outside valid range.
        """
        if defects is None:
            if num_defects is None:
                num_defects = self.rng.randint(*self.config.defect_count_range)
            if not (1 <= num_defects <= 10):
                raise ValueError(f"num_defects must be 1-10, got {num_defects}")
            defects = self._generate_random_defect_specs(
                num_defects, road_width, road_length
            )

        placed_instances: List[DefectInstance] = []

        for spec in defects:
            instance = self._place_single_defect(
                spec, road_width, road_length, placed_instances
            )
            if instance is not None:
                placed_instances.append(instance)

        return placed_instances

    def setup_camera(
        self,
        view_type: Literal["dashcam", "drone"],
        road_length: float = 100.0,
        height: Optional[float] = None,
        pitch: Optional[float] = None,
    ) -> CameraConfig:
        """Set up camera with specified view type configuration.

        Args:
            view_type: Either "dashcam" or "drone".
            road_length: Road length for camera positioning.
            height: Camera height in meters. Random within type range if None.
            pitch: Camera pitch in degrees. Random within type range if None.

        Returns:
            CameraConfig with view parameters and computed intrinsics/extrinsics.

        Raises:
            ValueError: If view_type is invalid or parameters are out of range.
        """
        if view_type not in CAMERA_CONFIGS:
            raise ValueError(f"view_type must be 'dashcam' or 'drone', got '{view_type}'")

        cam_range = CAMERA_CONFIGS[view_type]

        if height is None:
            height = self.rng.uniform(*cam_range["height"])
        if pitch is None:
            pitch = self.rng.uniform(*cam_range["pitch"])

        # Validate constraints
        h_min, h_max = cam_range["height"]
        p_min, p_max = cam_range["pitch"]

        if not (h_min <= height <= h_max):
            raise ValueError(
                f"height for {view_type} must be {h_min}-{h_max}m, got {height}"
            )
        if not (p_min <= pitch <= p_max):
            raise ValueError(
                f"pitch for {view_type} must be {p_min}° to {p_max}°, got {pitch}"
            )

        # Compute focal length based on view type
        # Dashcam has narrower FOV (~60°), drone has wider FOV (~90°)
        image_size = self.config.render_size
        if view_type == "dashcam":
            fov_deg = 60.0
        else:
            fov_deg = 90.0
        focal_length = (image_size / 2.0) / math.tan(math.radians(fov_deg / 2.0))

        intrinsics = generate_intrinsics(focal_length, image_size)
        extrinsics = generate_extrinsics(height, pitch, road_length)

        # Set up the camera object in the 3D scene
        self.backend.setup_camera_object(height, pitch, view_type)

        return CameraConfig(
            view_type=view_type,
            height=height,
            pitch=pitch,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
        )

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _generate_random_defect_specs(
        self, count: int, road_width: float, road_length: float
    ) -> List[DefectSpec]:
        """Generate random defect specifications within valid ranges.

        Args:
            count: Number of defects to generate.
            road_width: Road width for position bounds.
            road_length: Road length for position bounds.

        Returns:
            List of randomly generated DefectSpec objects.
        """
        defect_types = list(DEFECT_DIMENSIONS.keys())
        specs: List[DefectSpec] = []

        for _ in range(count):
            defect_type = self.rng.choice(defect_types)
            dims = DEFECT_DIMENSIONS[defect_type]

            # Generate type-specific scale
            scale = self._generate_scale(defect_type, dims)

            # Generate random position within road bounds (with margin)
            position = self._generate_position(
                defect_type, scale, road_width, road_length
            )

            # Random orientation
            orientation = self.rng.uniform(0.0, 360.0)

            specs.append(DefectSpec(
                defect_type=defect_type,
                position=position,
                orientation=orientation,
                scale=scale,
            ))

        return specs

    def _generate_scale(
        self, defect_type: str, dims: Dict[str, Tuple[float, float]]
    ) -> Tuple[float, ...]:
        """Generate type-specific scale dimensions within valid ranges.

        Args:
            defect_type: Type of defect.
            dims: Dimension ranges for the defect type.

        Returns:
            Scale tuple with type-specific dimensions.
        """
        if defect_type == "crack":
            length = self.rng.uniform(*dims["length"])
            width = self.rng.uniform(*dims["width"])
            return (length, width)
        elif defect_type == "pothole":
            diameter = self.rng.uniform(*dims["diameter"])
            depth = self.rng.uniform(*dims["depth"])
            return (diameter, depth)
        elif defect_type == "puddle":
            diameter = self.rng.uniform(*dims["diameter"])
            return (diameter,)
        elif defect_type == "patch":
            length = self.rng.uniform(*dims["length"])
            width = self.rng.uniform(*dims["width"])
            return (length, width)
        elif defect_type == "manhole":
            diameter = self.rng.uniform(*dims["diameter"])
            return (diameter,)
        else:
            raise ValueError(f"Unknown defect type: {defect_type}")

    def _generate_position(
        self,
        defect_type: str,
        scale: Tuple[float, ...],
        road_width: float,
        road_length: float,
    ) -> Tuple[float, float]:
        """Generate a position that keeps the defect within road bounds.

        Uses a margin based on the maximum dimension of the defect to ensure
        the full defect fits within the road surface.

        Args:
            defect_type: Type of defect.
            scale: Type-specific dimensions.
            road_width: Road width for position bounds.
            road_length: Road length for position bounds.

        Returns:
            (x, y) position in road coordinates.
        """
        # Compute margin to keep defect within bounds
        max_dim = max(scale)
        margin = max_dim / 2.0 + 0.01  # Small buffer

        x_min = margin
        x_max = road_width - margin
        y_min = margin
        y_max = road_length - margin

        # Ensure valid range
        if x_min >= x_max:
            x = road_width / 2.0
        else:
            x = self.rng.uniform(x_min, x_max)

        if y_min >= y_max:
            y = road_length / 2.0
        else:
            y = self.rng.uniform(y_min, y_max)

        return (x, y)

    def _place_single_defect(
        self,
        spec: DefectSpec,
        road_width: float,
        road_length: float,
        existing: List[DefectInstance],
    ) -> Optional[DefectInstance]:
        """Place a single defect, repositioning if overlap exceeds threshold.

        Args:
            spec: The defect specification to place.
            road_width: Total road width.
            road_length: Total road length.
            existing: List of already-placed defect instances.

        Returns:
            DefectInstance if successfully placed, None if placement failed.
        """
        area = compute_defect_area(spec.defect_type, spec.scale)

        for attempt in range(self.config.max_reposition_attempts):
            if attempt == 0:
                position = spec.position
                orientation = spec.orientation
            else:
                # Reposition with a new random position
                position = self._generate_position(
                    spec.defect_type, spec.scale, road_width, road_length
                )
                orientation = self.rng.uniform(0.0, 360.0)

            bbox = compute_defect_bounding_box(
                spec.defect_type, spec.scale, position, orientation
            )

            # Check road bounds
            if not is_within_road_bounds(bbox, road_width, road_length):
                continue

            # Create a candidate instance
            candidate = DefectInstance(
                spec=DefectSpec(
                    defect_type=spec.defect_type,
                    position=position,
                    orientation=orientation,
                    scale=spec.scale,
                ),
                mesh_object=None,  # Will be set after backend call
                bounding_box_2d=bbox,
                area=area,
            )

            # Check overlap with existing defects
            if not has_excessive_overlap(
                candidate, existing, self.config.overlap_threshold
            ):
                # Successfully placed - create mesh via backend
                mesh_obj = self.backend.create_defect_mesh(
                    spec.defect_type, spec.scale, position, orientation
                )
                candidate.mesh_object = mesh_obj
                return candidate

        # Failed to place after max attempts
        return None
