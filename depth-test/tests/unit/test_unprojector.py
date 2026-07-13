"""Unit tests for DepthUnprojector and WorldTransformer."""
import numpy as np
import pytest

from src.reconstruction.unprojector import DepthUnprojector, WorldTransformer


class TestDepthUnprojector:
    """Tests for pinhole depth unprojection."""

    def setup_method(self):
        self.unprojector = DepthUnprojector()

    def test_single_pixel_at_center(self):
        """A pixel at the principal point should unproject to (0, 0, Z)."""
        K = np.array([[500.0, 0, 2.0],
                      [0, 500.0, 2.0],
                      [0, 0, 1.0]])
        depth = np.zeros((5, 5))
        depth[2, 2] = 3.0  # center pixel, depth=3

        points = self.unprojector.unproject(depth, K)

        assert points.shape == (1, 3)
        np.testing.assert_allclose(points[0], [0.0, 0.0, 3.0], atol=1e-10)

    def test_pixel_off_center(self):
        """A pixel off-center should have correct X, Y displacements."""
        fx, fy, cx, cy = 200.0, 200.0, 3.0, 3.0
        K = np.array([[fx, 0, cx],
                      [0, fy, cy],
                      [0, 0, 1.0]])
        depth = np.zeros((7, 7))
        # pixel at (5, 4), depth = 2.0
        depth[4, 5] = 2.0

        points = self.unprojector.unproject(depth, K)

        assert points.shape == (1, 3)
        expected_x = 2.0 * (5 - cx) / fx
        expected_y = 2.0 * (4 - cy) / fy
        expected_z = 2.0
        np.testing.assert_allclose(points[0], [expected_x, expected_y, expected_z], atol=1e-10)

    def test_multiple_valid_pixels(self):
        """Multiple valid depth pixels are all unprojected."""
        K = np.array([[100.0, 0, 5.0],
                      [0, 100.0, 5.0],
                      [0, 0, 1.0]])
        depth = np.zeros((10, 10))
        depth[2, 3] = 1.5
        depth[7, 8] = 4.0
        depth[0, 0] = 0.5

        points = self.unprojector.unproject(depth, K)

        assert points.shape == (3, 3)
        # All Z values should match the depth values
        assert set(points[:, 2].tolist()) == {0.5, 1.5, 4.0}

    def test_zero_depth_excluded(self):
        """Pixels with depth=0 are not unprojected."""
        K = np.array([[100.0, 0, 2.0],
                      [0, 100.0, 2.0],
                      [0, 0, 1.0]])
        depth = np.zeros((5, 5))
        depth[1, 1] = 1.0
        # depth[2, 2] is 0 - should be excluded

        points = self.unprojector.unproject(depth, K)

        assert points.shape == (1, 3)

    def test_all_zero_depth_returns_empty(self):
        """All-zero depth map returns empty array."""
        K = np.array([[100.0, 0, 2.0],
                      [0, 100.0, 2.0],
                      [0, 0, 1.0]])
        depth = np.zeros((5, 5))

        points = self.unprojector.unproject(depth, K)

        assert points.shape == (0, 3)

    def test_unprojection_round_trip(self):
        """Project known 3D points to pixels, then unproject back to 3D."""
        fx, fy, cx, cy = 300.0, 300.0, 160.0, 120.0
        K = np.array([[fx, 0, cx],
                      [0, fy, cy],
                      [0, 0, 1.0]])

        # Known 3D points in camera space (positive Z)
        original_points = np.array([
            [0.5, -0.3, 2.0],
            [-1.0, 0.7, 5.0],
            [0.0, 0.0, 1.0],
        ])

        # Project to pixel coordinates: [u, v] = [fx*X/Z + cx, fy*Y/Z + cy]
        H, W = 240, 320
        depth = np.zeros((H, W))

        for pt in original_points:
            u = int(round(fx * pt[0] / pt[2] + cx))
            v = int(round(fy * pt[1] / pt[2] + cy))
            if 0 <= u < W and 0 <= v < H:
                depth[v, u] = pt[2]

        # Unproject
        recovered = self.unprojector.unproject(depth, K)

        assert recovered.shape[0] == 3
        # Sort both by Z for comparison
        original_sorted = original_points[original_points[:, 2].argsort()]
        recovered_sorted = recovered[recovered[:, 2].argsort()]
        np.testing.assert_allclose(recovered_sorted, original_sorted, atol=0.02)

    def test_negative_depth_excluded(self):
        """Negative depth values are excluded (only > 0 is valid)."""
        K = np.array([[100.0, 0, 2.0],
                      [0, 100.0, 2.0],
                      [0, 0, 1.0]])
        depth = np.array([[-1.0, 0.0, 2.0],
                          [0.0, -5.0, 0.0],
                          [3.0, 0.0, 0.0]])

        points = self.unprojector.unproject(depth, K)

        # Only depth[0,2]=2.0 and depth[2,0]=3.0 are valid
        assert points.shape == (2, 3)


class TestWorldTransformer:
    """Tests for world-space transformation."""

    def setup_method(self):
        self.transformer = WorldTransformer()

    def test_identity_transform(self):
        """Identity rotation and zero translation should not change points."""
        points = np.array([[1.0, 2.0, 3.0],
                          [4.0, 5.0, 6.0]])
        extrinsics = np.hstack([np.eye(3), np.zeros((3, 1))])  # [3, 4]

        result = self.transformer.transform(points, extrinsics)

        np.testing.assert_allclose(result, points, atol=1e-10)

    def test_pure_translation(self):
        """Pure translation should shift all points."""
        points = np.array([[1.0, 2.0, 3.0],
                          [0.0, 0.0, 0.0]])
        t = np.array([[10.0], [20.0], [30.0]])
        extrinsics = np.hstack([np.eye(3), t])  # [3, 4]

        result = self.transformer.transform(points, extrinsics)

        expected = points + np.array([10.0, 20.0, 30.0])
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_rotation_90_degrees_around_z(self):
        """90-degree rotation around Z axis."""
        points = np.array([[1.0, 0.0, 0.0]])
        # R rotates 90 degrees around Z: [1,0,0] -> [0,1,0]
        R = np.array([[0.0, -1.0, 0.0],
                      [1.0, 0.0, 0.0],
                      [0.0, 0.0, 1.0]])
        extrinsics = np.hstack([R, np.zeros((3, 1))])

        result = self.transformer.transform(points, extrinsics)

        np.testing.assert_allclose(result, [[0.0, 1.0, 0.0]], atol=1e-10)

    def test_combined_rotation_and_translation(self):
        """Rotation + translation: X_world = R @ X_cam + t."""
        points = np.array([[1.0, 0.0, 0.0]])
        # 90 degree rotation around Z
        R = np.array([[0.0, -1.0, 0.0],
                      [1.0, 0.0, 0.0],
                      [0.0, 0.0, 1.0]])
        t = np.array([[5.0], [10.0], [15.0]])
        extrinsics = np.hstack([R, t])

        result = self.transformer.transform(points, extrinsics)

        # R @ [1,0,0] = [0,1,0], then + [5,10,15] = [5,11,15]
        np.testing.assert_allclose(result, [[5.0, 11.0, 15.0]], atol=1e-10)

    def test_round_trip_transform(self):
        """Forward transform followed by inverse should recover original points."""
        # Create a valid rotation matrix (rotation around arbitrary axis)
        angle = np.pi / 4
        axis = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
        t = np.array([[2.5], [-1.0], [3.7]])
        extrinsics = np.hstack([R, t])

        points = np.array([[1.0, 2.0, 3.0],
                          [-0.5, 0.3, 1.2],
                          [10.0, -5.0, 7.0]])

        # Forward
        world_points = self.transformer.transform(points, extrinsics)
        # Inverse
        recovered = self.transformer.inverse_transform(world_points, extrinsics)

        np.testing.assert_allclose(recovered, points, atol=1e-10)

    def test_inverse_transform_identity(self):
        """Inverse of identity should not change points."""
        points = np.array([[1.0, 2.0, 3.0]])
        extrinsics = np.hstack([np.eye(3), np.zeros((3, 1))])

        result = self.transformer.inverse_transform(points, extrinsics)

        np.testing.assert_allclose(result, points, atol=1e-10)

    def test_inverse_pure_translation(self):
        """Inverse of pure translation should subtract t."""
        world_points = np.array([[11.0, 22.0, 33.0]])
        t = np.array([[10.0], [20.0], [30.0]])
        extrinsics = np.hstack([np.eye(3), t])

        result = self.transformer.inverse_transform(world_points, extrinsics)

        np.testing.assert_allclose(result, [[1.0, 2.0, 3.0]], atol=1e-10)

    def test_many_points_batch(self):
        """Transforms work correctly for large batches of points."""
        np.random.seed(42)
        R = np.eye(3)
        t = np.array([[1.0], [2.0], [3.0]])
        extrinsics = np.hstack([R, t])

        points = np.random.randn(1000, 3)
        result = self.transformer.transform(points, extrinsics)

        expected = points + np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_orthonormality_preserved_in_round_trip(self):
        """Round-trip with any orthonormal R preserves distances between points."""
        # Random orthonormal rotation via QR decomposition
        np.random.seed(123)
        A = np.random.randn(3, 3)
        R, _ = np.linalg.qr(A)
        if np.linalg.det(R) < 0:
            R[:, 0] *= -1  # ensure proper rotation (det = +1)
        t = np.array([[5.0], [-3.0], [2.0]])
        extrinsics = np.hstack([R, t])

        points = np.array([[0.0, 0.0, 0.0],
                          [1.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0]])

        world_points = self.transformer.transform(points, extrinsics)

        # Distances should be preserved by rigid transform
        original_dist = np.linalg.norm(points[0] - points[1])
        world_dist = np.linalg.norm(world_points[0] - world_points[1])
        np.testing.assert_allclose(world_dist, original_dist, atol=1e-10)
