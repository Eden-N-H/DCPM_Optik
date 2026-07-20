"""Bird's-Eye View projection."""
import numpy as np
from typing import Dict, Optional, Tuple
from pathlib import Path
from src.utils.data_types import BEVMap


# Default color map for defect classes
# Maps class ID -> RGB color tuple
DEFAULT_COLOR_MAP: Dict[int, Tuple[int, int, int]] = {
    0: (128, 128, 128),  # Background - gray
    1: (0, 0, 0),        # Road - black
    2: (255, 0, 0),      # Crack - red
    3: (255, 165, 0),    # Pothole - orange
    4: (0, 0, 255),      # Puddle - blue
    5: (0, 255, 0),      # Patch - green
    6: (255, 255, 0),    # Manhole - yellow
    7: (255, 0, 255),    # Vehicle - magenta
}


class BEVProjector:
    """Orthographic BEV map generation from aggregated point cloud.

    Projects points onto the XY ground plane (using X and Z world coordinates
    since Y is height in this project's convention). Generates a grid where
    each cell is assigned the majority-vote class and maximum severity from
    all contributing points.
    """

    def __init__(self, resolution: float = 0.02):
        """
        Args:
            resolution: Meters per pixel in the BEV map (default 0.02 = 2cm/px).
        """
        self.resolution = resolution

    def project(self, positions: np.ndarray, classes: np.ndarray,
                severities: np.ndarray) -> BEVMap:
        """Project filtered point cloud to BEV map.

        Performs orthographic projection onto the horizontal plane (X, Z)
        at the configured resolution. For each grid cell, assigns:
        - Class: majority vote among all points in the cell
        - Severity: maximum severity among all points in the cell

        Args:
            positions: [N, 3] world-space points (x, y, z)
            classes: [N] defect class per point (int)
            severities: [N] severity per point (float in [0, 1])

        Returns:
            BEVMap with class_grid, severity_grid, and color image.
        """
        if len(positions) == 0:
            return BEVMap(
                image=np.zeros((1, 1, 3), dtype=np.uint8),
                class_grid=np.zeros((1, 1), dtype=np.int32),
                severity_grid=np.zeros((1, 1), dtype=np.float64),
                origin=(0.0, 0.0),
                resolution=self.resolution,
            )

        # Project onto XZ plane (Y is height, filtered out already)
        x = positions[:, 0]
        z = positions[:, 2]

        x_min, x_max = float(x.min()), float(x.max())
        z_min, z_max = float(z.min()), float(z.max())

        # Grid dimensions (W along X axis, H along Z axis)
        W = max(1, int(np.ceil((x_max - x_min) / self.resolution)))
        H = max(1, int(np.ceil((z_max - z_min) / self.resolution)))

        # Map points to grid cell indices
        xi = np.clip(((x - x_min) / self.resolution).astype(np.int32), 0, W - 1)
        zi = np.clip(((z - z_min) / self.resolution).astype(np.int32), 0, H - 1)

        # Build class and severity grids
        class_grid = np.zeros((H, W), dtype=np.int32)
        severity_grid = np.zeros((H, W), dtype=np.float64)

        # For majority vote: count occurrences of each class per cell
        # and track maximum severity per cell
        class_counts: Dict[Tuple[int, int], Dict[int, int]] = {}

        for i in range(len(positions)):
            cell = (int(zi[i]), int(xi[i]))
            if cell not in class_counts:
                class_counts[cell] = {}
            c = int(classes[i])
            class_counts[cell][c] = class_counts[cell].get(c, 0) + 1
            severity_grid[cell[0], cell[1]] = max(
                severity_grid[cell[0], cell[1]], float(severities[i])
            )

        # Assign majority-vote class per cell
        for cell, counts in class_counts.items():
            class_grid[cell[0], cell[1]] = max(counts, key=counts.get)

        # Generate color image
        image = self._colorize(class_grid, severity_grid)

        return BEVMap(
            image=image,
            class_grid=class_grid,
            severity_grid=severity_grid,
            origin=(x_min, z_min),
            resolution=self.resolution,
        )

    def _colorize(self, class_grid: np.ndarray, severity_grid: np.ndarray,
                  color_map: Optional[Dict[int, Tuple[int, int, int]]] = None) -> np.ndarray:
        """Generate color-coded BEV image.

        Each cell is colored by its class. Intensity is modulated by severity:
        cells with higher severity appear at full brightness, lower severity
        appears dimmer.

        Args:
            class_grid: [H, W] defect class per cell.
            severity_grid: [H, W] maximum severity per cell.
            color_map: Maps class ID -> (R, G, B). Uses DEFAULT_COLOR_MAP if None.

        Returns:
            [H, W, 3] uint8 RGB image.
        """
        if color_map is None:
            color_map = DEFAULT_COLOR_MAP

        H, W = class_grid.shape
        image = np.zeros((H, W, 3), dtype=np.uint8)

        for class_id, color in color_map.items():
            mask = class_grid == class_id
            if not mask.any():
                continue
            # Intensity modulated by severity: base 30% + 70% from severity
            intensity = np.clip(0.3 + 0.7 * severity_grid[mask], 0.0, 1.0)
            for c in range(3):
                image[mask, c] = (color[c] * intensity).astype(np.uint8)

        return image

    def export_png(self, bev_map: BEVMap, path: Path,
                   color_map: Optional[Dict[int, Tuple[int, int, int]]] = None) -> None:
        """Save BEV map as color-coded PNG.

        If a custom color_map is provided, re-colorizes the BEV map before saving.
        Otherwise saves the pre-computed image from the BEVMap.

        Args:
            bev_map: The BEV map to export.
            path: Output PNG file path.
            color_map: Optional custom class-to-color mapping. If provided,
                      overrides the colors used during projection.
        """
        import cv2

        if color_map is not None:
            # Re-colorize with custom color map
            image = self._colorize(bev_map.class_grid, bev_map.severity_grid, color_map)
        else:
            image = bev_map.image

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
