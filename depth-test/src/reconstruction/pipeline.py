"""End-to-end reconstruction pipeline."""
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
from .unprojector import DepthUnprojector, WorldTransformer
from .aggregator import PointCloudAggregator
from .bev import BEVProjector


def rodrigues_to_rotation_matrix(rodrigues: np.ndarray) -> np.ndarray:
    """Convert Rodrigues rotation vector to 3x3 rotation matrix."""
    theta = np.linalg.norm(rodrigues)
    if theta < 1e-8:
        return np.eye(3)
    axis = rodrigues / theta
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


class ReconstructionPipeline:
    """End-to-end reconstruction from model predictions to BEV map."""

    def __init__(self, config: Dict):
        self.unprojector = DepthUnprojector()
        self.transformer = WorldTransformer()
        self.aggregator = PointCloudAggregator()
        self.bev_projector = BEVProjector(resolution=config.get('bev_resolution', 0.02))
        self.depth_threshold = config.get('depth_confidence_threshold', 0.5)
        self.height_range = tuple(config.get('height_range', [-0.5, 0.5]))

    def process_frame(self, predictions: Dict[str, np.ndarray],
                      rgb: Optional[np.ndarray] = None) -> bool:
        """Process a single frame's predictions.

        Args:
            predictions: dict with 'depth' [H,W], 'segmentation' [H,W] (class IDs),
                        'severity' [H,W], 'intrinsics' [4], 'extrinsics' [6]
            rgb: Optional [H,W,3] RGB image for point coloring

        Returns:
            True if frame was processed successfully, False if skipped
        """
        depth = predictions['depth']
        seg = predictions['segmentation']
        severity = predictions['severity']
        intrinsics_params = predictions['intrinsics']  # [fx, fy, cx, cy]
        extrinsics_params = predictions['extrinsics']  # [r1, r2, r3, t1, t2, t3]

        # Build intrinsics matrix
        fx, fy, cx, cy = intrinsics_params
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

        # Check for degenerate intrinsics
        if np.linalg.det(K) < 1e-6:
            return False

        # Build extrinsics matrix
        rodrigues = extrinsics_params[:3]
        translation = extrinsics_params[3:]
        R = rodrigues_to_rotation_matrix(rodrigues)
        extrinsics = np.hstack([R, translation.reshape(3, 1)])  # [3, 4]

        # Unproject depth to camera space
        points_cam = self.unprojector.unproject(depth, K)
        if len(points_cam) == 0:
            return False

        # Transform to world space
        points_world = self.transformer.transform(points_cam, extrinsics)

        # Get per-point attributes
        valid = depth > 0
        classes = seg[valid].astype(np.int32)
        severities_flat = severity[valid].astype(np.float64)

        # Colors
        if rgb is not None:
            colors = rgb[valid].astype(np.uint8)
        else:
            colors = np.full((len(points_world), 3), 128, dtype=np.uint8)

        self.aggregator.add_frame(points_world, classes, severities_flat, colors=colors)
        return True

    def finalize(self, output_dir: Path) -> Optional[Path]:
        """Filter, project BEV, and export results.

        Returns:
            Path to exported BEV map PNG, or None if point cloud is empty
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Filter
        self.aggregator.filter(
            depth_confidence_threshold=self.depth_threshold,
            height_range=self.height_range
        )

        positions, classes, severities, confidences, colors = self.aggregator.get_aggregated()

        if len(positions) == 0:
            return None

        # Export PLY
        ply_path = output_dir / "reconstruction.ply"
        self.aggregator.export_ply(ply_path)

        # Generate and export BEV
        bev_map = self.bev_projector.project(positions, classes, severities)
        bev_path = output_dir / "bev_map.png"
        self.bev_projector.export_png(bev_map, bev_path)

        return bev_path
