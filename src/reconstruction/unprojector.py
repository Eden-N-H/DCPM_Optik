"""Depth unprojection and world-space transformation."""
import numpy as np


class DepthUnprojector:
    """Pinhole model depth unprojection.

    Unprojects depth map pixels to 3D camera-space coordinates using:
    X = Z * K_inv @ [u, v, 1]^T

    where K is the 3x3 intrinsics matrix, Z is the depth value at pixel (u, v),
    and K_inv is the inverse of K.
    """

    def unproject(self, depth: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
        """Unproject depth map to 3D points in camera space.

        Only pixels with depth > 0 are unprojected.

        Args:
            depth: [H, W] depth map (positive values indicate valid depth)
            intrinsics: [3, 3] camera intrinsic matrix K

        Returns:
            [N, 3] 3D points in camera coordinates where N = number of valid (>0) pixels.
            Returns empty array of shape [0, 3] if no valid pixels exist.
        """
        H, W = depth.shape

        # Compute K inverse
        K_inv = np.linalg.inv(intrinsics.astype(np.float64))

        # Create pixel coordinate grid
        u_coords, v_coords = np.meshgrid(np.arange(W), np.arange(H))

        # Mask valid depth pixels
        valid = depth > 0
        if not np.any(valid):
            return np.zeros((0, 3), dtype=np.float64)

        z = depth[valid].astype(np.float64)
        u_valid = u_coords[valid].astype(np.float64)
        v_valid = v_coords[valid].astype(np.float64)

        # Form homogeneous pixel coordinates: [3, N]
        ones = np.ones_like(u_valid)
        pixel_coords = np.stack([u_valid, v_valid, ones], axis=0)  # [3, N]

        # Unproject: X = Z * K_inv @ [u, v, 1]^T
        # K_inv @ pixel_coords gives normalized camera coordinates [3, N]
        normalized = K_inv @ pixel_coords  # [3, N]

        # Scale by depth
        points_3d = z[np.newaxis, :] * normalized  # [3, N]

        return points_3d.T  # [N, 3]


class WorldTransformer:
    """Transforms 3D points between camera space and world space using extrinsics.

    The extrinsics matrix [R|t] is interpreted as a camera-to-world transform:
        X_world = R @ X_cam + t

    where R is the [3, 3] rotation matrix and t is the [3] translation vector.
    """

    def transform(self, points: np.ndarray, extrinsics: np.ndarray) -> np.ndarray:
        """Transform points from camera space to world space.

        Applies: X_world = R @ X_cam + t

        Args:
            points: [N, 3] points in camera space
            extrinsics: [3, 4] camera-to-world extrinsic matrix [R|t]

        Returns:
            [N, 3] points in world space
        """
        R = extrinsics[:, :3].astype(np.float64)  # [3, 3]
        t = extrinsics[:, 3].astype(np.float64)   # [3]

        # X_world = R @ X_cam + t
        world_points = (R @ points.astype(np.float64).T).T + t

        return world_points

    def inverse_transform(self, points: np.ndarray, extrinsics: np.ndarray) -> np.ndarray:
        """Transform points from world space back to camera space.

        Inverse of X_world = R @ X_cam + t is:
            X_cam = R^T @ (X_world - t)

        Args:
            points: [N, 3] points in world space
            extrinsics: [3, 4] camera-to-world extrinsic matrix [R|t]

        Returns:
            [N, 3] points in camera space
        """
        R = extrinsics[:, :3].astype(np.float64)  # [3, 3]
        t = extrinsics[:, 3].astype(np.float64)   # [3]

        # X_cam = R^T @ (X_world - t)
        R_inv = R.T
        cam_points = (R_inv @ (points.astype(np.float64) - t).T).T

        return cam_points
