"""Point cloud aggregation with filtering."""
import numpy as np
from typing import Optional, Tuple
from pathlib import Path


class PointCloudAggregator:
    """Accumulates multi-frame point clouds with per-point attributes.

    Maintains internal arrays for positions [N, 3], classes [N],
    severities [N], and confidences [N]. Supports optional colors
    for PLY export.
    """

    def __init__(self):
        self.positions: list = []      # List of [Ni, 3] arrays
        self.classes: list = []        # List of [Ni] arrays (int)
        self.severities: list = []     # List of [Ni] arrays (float)
        self.confidences: list = []    # List of [Ni] arrays (float)
        self.colors: list = []         # List of [Ni, 3] arrays (uint8), optional
        self._filtered = False
        self._all_positions: Optional[np.ndarray] = None
        self._all_classes: Optional[np.ndarray] = None
        self._all_severities: Optional[np.ndarray] = None
        self._all_confidences: Optional[np.ndarray] = None
        self._all_colors: Optional[np.ndarray] = None

    def add_frame(self, points: np.ndarray, classes: np.ndarray,
                  severities: np.ndarray, confidences: Optional[np.ndarray] = None,
                  colors: Optional[np.ndarray] = None) -> None:
        """Add a frame's points with attributes.

        Args:
            points: [N, 3] world-space positions
            classes: [N] defect class IDs (int)
            severities: [N] severity values (float)
            confidences: [N] confidence values (float), defaults to 1.0 for all points
            colors: [N, 3] RGB colors (uint8), optional for PLY export
        """
        n = len(points)
        self.positions.append(np.asarray(points, dtype=np.float64))
        self.classes.append(np.asarray(classes, dtype=np.int32))
        self.severities.append(np.asarray(severities, dtype=np.float64))

        if confidences is not None:
            self.confidences.append(np.asarray(confidences, dtype=np.float64))
        else:
            self.confidences.append(np.ones(n, dtype=np.float64))

        if colors is not None:
            self.colors.append(np.asarray(colors, dtype=np.uint8))
        else:
            self.colors.append(np.full((n, 3), 128, dtype=np.uint8))

        self._filtered = False

    @property
    def total_points(self) -> int:
        """Total number of points in the aggregator."""
        if self._filtered:
            return len(self._all_positions)
        return sum(len(p) for p in self.positions)

    def get_aggregated(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get concatenated point cloud.

        Returns:
            Tuple of (positions, classes, severities, confidences, colors)
        """
        if self._filtered:
            return (self._all_positions, self._all_classes,
                    self._all_severities, self._all_confidences, self._all_colors)
        if not self.positions:
            empty_pos = np.zeros((0, 3), dtype=np.float64)
            empty_classes = np.zeros(0, dtype=np.int32)
            empty_sev = np.zeros(0, dtype=np.float64)
            empty_conf = np.zeros(0, dtype=np.float64)
            empty_colors = np.zeros((0, 3), dtype=np.uint8)
            return empty_pos, empty_classes, empty_sev, empty_conf, empty_colors
        return (
            np.concatenate(self.positions, axis=0),
            np.concatenate(self.classes, axis=0),
            np.concatenate(self.severities, axis=0),
            np.concatenate(self.confidences, axis=0),
            np.concatenate(self.colors, axis=0),
        )

    def filter(self, depth_confidence_threshold: float = 0.5,
               height_range: Tuple[float, float] = (-0.5, 0.5)) -> None:
        """Filter point cloud by confidence and height range.

        Removes points with confidence below the threshold or height (y-coordinate)
        outside the specified range.

        Args:
            depth_confidence_threshold: Remove points with confidence below this value
            height_range: (min_y, max_y) height range relative to road plane
        """
        positions, classes, severities, confidences, colors = self.get_aggregated()

        if len(positions) == 0:
            self._all_positions = positions
            self._all_classes = classes
            self._all_severities = severities
            self._all_confidences = confidences
            self._all_colors = colors
            self._filtered = True
            return

        # Confidence filter: remove points below threshold
        confidence_mask = confidences >= depth_confidence_threshold

        # Height filter: y-coordinate within range
        height_mask = (positions[:, 1] >= height_range[0]) & (positions[:, 1] <= height_range[1])

        mask = confidence_mask & height_mask

        self._all_positions = positions[mask]
        self._all_classes = classes[mask]
        self._all_severities = severities[mask]
        self._all_confidences = confidences[mask]
        self._all_colors = colors[mask]
        self._filtered = True

    def export_ply(self, path: Path) -> None:
        """Export point cloud as PLY file.

        Exports with per-point position, RGB color, defect class, and severity attributes.

        Args:
            path: Output PLY file path
        """
        from plyfile import PlyData, PlyElement

        positions, classes, severities, confidences, colors = self.get_aggregated()

        if len(positions) == 0:
            vertex = np.array([], dtype=[
                ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
                ('class_id', 'i4'), ('severity', 'f4')
            ])
        else:
            vertex = np.zeros(len(positions), dtype=[
                ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
                ('class_id', 'i4'), ('severity', 'f4')
            ])
            vertex['x'] = positions[:, 0]
            vertex['y'] = positions[:, 1]
            vertex['z'] = positions[:, 2]
            vertex['red'] = colors[:, 0]
            vertex['green'] = colors[:, 1]
            vertex['blue'] = colors[:, 2]
            vertex['class_id'] = classes
            vertex['severity'] = severities

        el = PlyElement.describe(vertex, 'vertex')
        PlyData([el]).write(str(path))
