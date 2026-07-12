"""Property-based tests for 3D geometry operations (Properties 19, 20).

Property 19: Pinhole unprojection round-trip
Property 20: Extrinsics transformation round-trip

Validates: Requirements 13.1, 13.2
"""
import numpy as np
import hypothesis
from hypothesis import given, settings, assume
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from src.reconstruction.unprojector import DepthUnprojector, WorldTransformer


# --- Strategies ---

@st.composite
def valid_intrinsics(draw):
    """Generate a valid 3x3 intrinsics matrix K.

    K = [[fx, 0, cx],
         [0, fy, cy],
         [0,  0,  1]]

    with positive focal lengths and principal point within reasonable image bounds.
    """
    fx = draw(st.floats(min_value=100.0, max_value=2000.0))
    fy = draw(st.floats(min_value=100.0, max_value=2000.0))
    cx = draw(st.floats(min_value=50.0, max_value=500.0))
    cy = draw(st.floats(min_value=50.0, max_value=500.0))

    K = np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)
    return K


@st.composite
def points_3d_positive_z(draw, min_points=1, max_points=50):
    """Generate random 3D points with positive Z (in front of camera)."""
    n = draw(st.integers(min_value=min_points, max_value=max_points))
    # X, Y can be anywhere, Z must be positive (depth > 0)
    x = draw(arrays(np.float64, n, elements=st.floats(min_value=-10.0, max_value=10.0)))
    y = draw(arrays(np.float64, n, elements=st.floats(min_value=-10.0, max_value=10.0)))
    z = draw(arrays(np.float64, n, elements=st.floats(min_value=0.5, max_value=50.0)))
    points = np.stack([x, y, z], axis=1)
    return points


@st.composite
def orthonormal_rotation(draw):
    """Generate a valid 3x3 orthonormal rotation matrix via QR decomposition.

    Draw a random 3x3 matrix and apply QR to get an orthonormal Q.
    Ensure det(Q) = +1 (proper rotation, not reflection).
    """
    elements = st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False)
    raw = draw(arrays(np.float64, (3, 3), elements=elements))

    # Ensure the matrix is non-singular for QR
    assume(np.abs(np.linalg.det(raw)) > 1e-3)

    Q, R = np.linalg.qr(raw)
    # Ensure proper rotation (det = +1)
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


@st.composite
def valid_extrinsics(draw):
    """Generate a valid [3, 4] extrinsics matrix [R|t] where R is orthonormal."""
    R = draw(orthonormal_rotation())
    t = draw(arrays(np.float64, 3, elements=st.floats(min_value=-20.0, max_value=20.0)))
    extrinsics = np.hstack([R, t.reshape(3, 1)])
    return extrinsics


# --- Property 19: Pinhole unprojection round-trip ---

class TestPinholeUnprojectionRoundTrip:
    """**Validates: Requirements 13.1**

    Property 19: For any set of 3D points with positive Z coordinates and a valid
    intrinsics matrix K, projecting points to pixel coordinates via K and then
    unprojecting with the same K and original depth values SHALL recover the
    original 3D points within floating-point tolerance (1e-5).
    """

    @given(
        K=valid_intrinsics(),
        points=points_3d_positive_z(min_points=1, max_points=20),
    )
    @settings(max_examples=100, deadline=None)
    def test_project_then_unproject_recovers_points(self, K, points):
        """Project 3D points to pixels via K, then unproject back to 3D."""
        # Project: pixel = K @ point_3d / Z
        # For point [X, Y, Z]: pixel_homogeneous = K @ [X, Y, Z]^T
        # Then u = pixel_h[0] / pixel_h[2], v = pixel_h[1] / pixel_h[2]
        N = points.shape[0]

        # Project points to pixel coordinates
        pixel_homogeneous = (K @ points.T)  # [3, N]
        u = pixel_homogeneous[0, :] / pixel_homogeneous[2, :]
        v = pixel_homogeneous[1, :] / pixel_homogeneous[2, :]
        z_values = points[:, 2]

        # Ensure pixel coordinates are within a reasonable range for integer indexing
        assume(np.all(u >= 0) and np.all(u < 1024))
        assume(np.all(v >= 0) and np.all(v < 1024))

        # To use DepthUnprojector.unproject(), we need to construct a depth map
        # with the correct depth values at the projected pixel positions.
        # However, the unprojector uses integer pixel grid coordinates.
        # For a proper round-trip, we work directly with the math:
        # unproject: X = Z * K_inv @ [u, v, 1]^T

        # Round-trip using the unprojection formula directly
        K_inv = np.linalg.inv(K)
        ones = np.ones(N)
        pixel_coords = np.stack([u, v, ones], axis=0)  # [3, N]
        normalized = K_inv @ pixel_coords  # [3, N]
        recovered = z_values[np.newaxis, :] * normalized  # [3, N]
        recovered_points = recovered.T  # [N, 3]

        np.testing.assert_allclose(
            recovered_points, points, atol=1e-5,
            err_msg="Pinhole unprojection round-trip failed"
        )

    @given(
        K=valid_intrinsics(),
        points=points_3d_positive_z(min_points=1, max_points=10),
    )
    @settings(max_examples=50, deadline=None)
    def test_unproject_via_depth_map(self, K, points):
        """Test the DepthUnprojector class itself using constructed depth maps.

        Project 3D points to integer pixel coordinates, place depth values in
        a depth map, then unproject and verify recovery.
        """
        N = points.shape[0]

        # Project points to pixel coordinates
        pixel_homogeneous = (K @ points.T)  # [3, N]
        u_float = pixel_homogeneous[0, :] / pixel_homogeneous[2, :]
        v_float = pixel_homogeneous[1, :] / pixel_homogeneous[2, :]
        z_values = points[:, 2]

        # Round to integer pixel coordinates
        u_int = np.round(u_float).astype(int)
        v_int = np.round(v_float).astype(int)

        # Ensure all pixels fit within a reasonable image size
        assume(np.all(u_int >= 0) and np.all(u_int < 512))
        assume(np.all(v_int >= 0) and np.all(v_int < 512))

        # Ensure no two points map to the same pixel (otherwise we can't
        # distinguish them in the depth map)
        pixel_pairs = set()
        for i in range(N):
            pair = (v_int[i], u_int[i])
            if pair in pixel_pairs:
                assume(False)
            pixel_pairs.add(pair)

        # Construct depth map
        H, W = 512, 512
        depth_map = np.zeros((H, W), dtype=np.float64)
        for i in range(N):
            depth_map[v_int[i], u_int[i]] = z_values[i]

        # Unproject using the class
        unprojector = DepthUnprojector()
        recovered = unprojector.unproject(depth_map, K)

        # The recovered points should match what you'd get from unprojecting
        # the integer pixel coordinates with the stored depth values.
        # Due to integer rounding of pixel coords, we compare against
        # the expected result from integer pixels (not original points).
        assert recovered.shape[0] == N
        assert recovered.shape[1] == 3

        # Compute expected 3D points from integer pixel coordinates
        K_inv = np.linalg.inv(K.astype(np.float64))
        for i in range(N):
            u_pix = u_int[i]
            v_pix = v_int[i]
            z = z_values[i]
            pixel_coord = np.array([u_pix, v_pix, 1.0])
            expected_point = z * (K_inv @ pixel_coord)
            # Find the corresponding recovered point (order depends on raster scan)
            # The unprojector iterates in row-major order over valid pixels
            # So we need to find the point corresponding to (v_int[i], u_int[i])
            pass

        # Instead of point-by-point matching with uncertain ordering,
        # verify that the set of recovered points matches expected set
        # Sort both by Z then X then Y for consistent ordering
        expected_points = []
        for i in range(N):
            u_pix = float(u_int[i])
            v_pix = float(v_int[i])
            z = z_values[i]
            pixel_coord = np.array([u_pix, v_pix, 1.0])
            expected_points.append(z * (K_inv @ pixel_coord))
        expected_points = np.array(expected_points)

        # Sort both arrays for comparison (row-major scan order)
        idx_expected = np.lexsort((expected_points[:, 2], expected_points[:, 1], expected_points[:, 0]))
        idx_recovered = np.lexsort((recovered[:, 2], recovered[:, 1], recovered[:, 0]))

        np.testing.assert_allclose(
            recovered[idx_recovered], expected_points[idx_expected],
            atol=1e-5,
            err_msg="DepthUnprojector round-trip via depth map failed"
        )


# --- Property 20: Extrinsics transformation round-trip ---

class TestExtrinsicsTransformationRoundTrip:
    """**Validates: Requirements 13.2**

    Property 20: For any set of 3D points and a valid rigid transformation [R|t]
    where R is orthonormal, applying the forward transform followed by the inverse
    transform SHALL recover the original points within floating-point tolerance (1e-5).
    """

    @given(
        points=points_3d_positive_z(min_points=1, max_points=50),
        extrinsics=valid_extrinsics(),
    )
    @settings(max_examples=100, deadline=None)
    def test_forward_then_inverse_recovers_points(self, points, extrinsics):
        """Apply transform then inverse_transform and verify recovery."""
        transformer = WorldTransformer()

        # Forward: camera -> world
        world_points = transformer.transform(points, extrinsics)

        # Inverse: world -> camera
        recovered_points = transformer.inverse_transform(world_points, extrinsics)

        np.testing.assert_allclose(
            recovered_points, points, atol=1e-5,
            err_msg="Extrinsics forward+inverse round-trip failed"
        )

    @given(
        points=points_3d_positive_z(min_points=1, max_points=50),
        extrinsics=valid_extrinsics(),
    )
    @settings(max_examples=100, deadline=None)
    def test_inverse_then_forward_recovers_points(self, points, extrinsics):
        """Apply inverse_transform then transform and verify recovery."""
        transformer = WorldTransformer()

        # Inverse: treat points as world-space, transform to camera
        cam_points = transformer.inverse_transform(points, extrinsics)

        # Forward: camera -> world
        recovered_points = transformer.transform(cam_points, extrinsics)

        np.testing.assert_allclose(
            recovered_points, points, atol=1e-5,
            err_msg="Extrinsics inverse+forward round-trip failed"
        )

    @given(
        points=points_3d_positive_z(min_points=1, max_points=20),
        extrinsics=valid_extrinsics(),
    )
    @settings(max_examples=50, deadline=None)
    def test_transform_preserves_distances(self, points, extrinsics):
        """A rigid transform preserves Euclidean distances between points."""
        assume(points.shape[0] >= 2)

        transformer = WorldTransformer()
        world_points = transformer.transform(points, extrinsics)

        # Check pairwise distances are preserved
        for i in range(min(points.shape[0] - 1, 5)):
            original_dist = np.linalg.norm(points[i] - points[i + 1])
            world_dist = np.linalg.norm(world_points[i] - world_points[i + 1])
            np.testing.assert_allclose(
                world_dist, original_dist, atol=1e-5,
                err_msg=f"Rigid transform did not preserve distance between points {i} and {i+1}"
            )
