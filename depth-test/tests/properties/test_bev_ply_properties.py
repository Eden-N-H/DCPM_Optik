"""Property-based tests for BEV and PLY operations (Properties 23, 24).

Property 23: BEV cell class assignment by majority vote
Property 24: PLY export round-trip

Validates: Requirements 14.2, 14.3
"""
import numpy as np
import tempfile
from pathlib import Path
from hypothesis import given, settings, assume
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from plyfile import PlyData

from src.reconstruction.bev import BEVProjector
from src.reconstruction.aggregator import PointCloudAggregator


# --- Strategies ---

@st.composite
def bev_cell_points(draw, min_points=2, max_points=20):
    """Generate a set of points that all map to the same BEV cell.

    All points share very close X and Z coordinates (within one cell),
    but have varying classes and severities.
    """
    n = draw(st.integers(min_value=min_points, max_value=max_points))

    # Use a small resolution so all points land in the same cell
    resolution = 0.02

    # Base position for the cell - pick a single cell center
    base_x = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
    base_z = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))

    # All points within the same cell (offset < resolution)
    x_offsets = draw(arrays(
        np.float64, n,
        elements=st.floats(min_value=0.0, max_value=resolution * 0.5, allow_nan=False, allow_infinity=False)
    ))
    z_offsets = draw(arrays(
        np.float64, n,
        elements=st.floats(min_value=0.0, max_value=resolution * 0.5, allow_nan=False, allow_infinity=False)
    ))

    # Y coordinates (height) don't affect BEV projection
    y_coords = draw(arrays(
        np.float64, n,
        elements=st.floats(min_value=-0.5, max_value=0.5, allow_nan=False, allow_infinity=False)
    ))

    positions = np.stack([
        base_x + x_offsets,
        y_coords,
        base_z + z_offsets
    ], axis=1)

    classes = draw(arrays(
        np.int32, n,
        elements=st.integers(min_value=0, max_value=6)
    ))
    severities = draw(arrays(
        np.float64, n,
        elements=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
    ))

    return positions, classes, severities, resolution


@st.composite
def multi_cell_point_cloud(draw, min_points=5, max_points=50):
    """Generate a point cloud that spans multiple BEV cells."""
    n = draw(st.integers(min_value=min_points, max_value=max_points))

    positions = draw(arrays(
        np.float64, (n, 3),
        elements=st.floats(min_value=0.0, max_value=2.0, allow_nan=False, allow_infinity=False)
    ))
    classes = draw(arrays(
        np.int32, n,
        elements=st.integers(min_value=0, max_value=6)
    ))
    severities = draw(arrays(
        np.float64, n,
        elements=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
    ))

    return positions, classes, severities


@st.composite
def ply_point_cloud(draw, min_points=1, max_points=50):
    """Generate a point cloud suitable for PLY export/import round-trip."""
    n = draw(st.integers(min_value=min_points, max_value=max_points))

    positions = draw(arrays(
        np.float64, (n, 3),
        elements=st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False)
    ))
    colors = draw(arrays(
        np.uint8, (n, 3),
        elements=st.integers(min_value=0, max_value=255)
    ))
    classes = draw(arrays(
        np.int32, n,
        elements=st.integers(min_value=0, max_value=6)
    ))
    severities = draw(arrays(
        np.float64, n,
        elements=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
    ))

    return positions, colors, classes, severities


# --- Property 23: BEV cell class assignment by majority vote ---

class TestBEVCellMajorityVote:
    """**Validates: Requirements 14.2**

    Property 23: For any set of points mapping to the same BEV grid cell,
    the cell's assigned defect class SHALL be the class with the highest
    frequency among contributing points, and the cell's severity SHALL be
    the maximum severity value among those points.
    """

    @given(data=bev_cell_points(min_points=2, max_points=20))
    @settings(max_examples=100, deadline=None)
    def test_cell_gets_majority_class(self, data):
        """Cell is assigned the class with highest frequency among its points.

        When multiple classes share the same highest frequency (tie), the
        implementation picks the maximum class ID (Python's max() over dict keys).
        """
        positions, classes, severities, resolution = data

        projector = BEVProjector(resolution=resolution)
        bev_map = projector.project(positions, classes, severities)

        # All points should land in the same cell (or very few cells)
        x = positions[:, 0]
        z = positions[:, 2]
        x_min = float(x.min())
        z_min = float(z.min())

        xi = np.clip(((x - x_min) / resolution).astype(np.int32), 0, bev_map.class_grid.shape[1] - 1)
        zi = np.clip(((z - z_min) / resolution).astype(np.int32), 0, bev_map.class_grid.shape[0] - 1)

        # Group points by cell
        cell_counts = {}
        for i in range(len(positions)):
            cell = (int(zi[i]), int(xi[i]))
            if cell not in cell_counts:
                cell_counts[cell] = []
            cell_counts[cell].append(i)

        # Check each cell's majority vote independently
        for cell, indices in cell_counts.items():
            cell_classes = classes[indices]

            # Build frequency count per class (same as BEV implementation)
            class_freq = {}
            for c in cell_classes:
                c = int(c)
                class_freq[c] = class_freq.get(c, 0) + 1

            # max(dict, key=dict.get) — on ties picks max key
            expected_cell_class = max(class_freq, key=class_freq.get)

            assert bev_map.class_grid[cell[0], cell[1]] == expected_cell_class, (
                f"Cell {cell} has class {bev_map.class_grid[cell[0], cell[1]]}, "
                f"expected majority class {expected_cell_class} "
                f"(counts: {class_freq})"
            )

    @given(data=bev_cell_points(min_points=2, max_points=20))
    @settings(max_examples=100, deadline=None)
    def test_cell_gets_maximum_severity(self, data):
        """Cell is assigned the maximum severity among its contributing points."""
        positions, classes, severities, resolution = data

        projector = BEVProjector(resolution=resolution)
        bev_map = projector.project(positions, classes, severities)

        # Map each point to its cell
        x = positions[:, 0]
        z = positions[:, 2]
        x_min = float(x.min())
        z_min = float(z.min())

        xi = np.clip(((x - x_min) / resolution).astype(np.int32), 0, bev_map.severity_grid.shape[1] - 1)
        zi = np.clip(((z - z_min) / resolution).astype(np.int32), 0, bev_map.severity_grid.shape[0] - 1)

        # Group points by cell
        cell_points = {}
        for i in range(len(positions)):
            cell = (int(zi[i]), int(xi[i]))
            if cell not in cell_points:
                cell_points[cell] = []
            cell_points[cell].append(i)

        # Check each cell's max severity
        for cell, indices in cell_points.items():
            expected_max_severity = float(np.max(severities[indices]))
            actual_severity = bev_map.severity_grid[cell[0], cell[1]]

            assert np.isclose(actual_severity, expected_max_severity, atol=1e-10), (
                f"Cell {cell} has severity {actual_severity}, "
                f"expected max severity {expected_max_severity}"
            )

    @given(data=multi_cell_point_cloud(min_points=5, max_points=50))
    @settings(max_examples=100, deadline=None)
    def test_multi_cell_majority_vote(self, data):
        """Majority vote and max severity hold across multiple cells."""
        positions, classes, severities = data
        resolution = 0.02

        projector = BEVProjector(resolution=resolution)
        bev_map = projector.project(positions, classes, severities)

        # Map points to cells
        x = positions[:, 0]
        z = positions[:, 2]
        x_min = float(x.min())
        z_min = float(z.min())

        W = bev_map.class_grid.shape[1]
        H = bev_map.class_grid.shape[0]

        xi = np.clip(((x - x_min) / resolution).astype(np.int32), 0, W - 1)
        zi = np.clip(((z - z_min) / resolution).astype(np.int32), 0, H - 1)

        # Group by cell
        cell_data = {}
        for i in range(len(positions)):
            cell = (int(zi[i]), int(xi[i]))
            if cell not in cell_data:
                cell_data[cell] = {'classes': [], 'severities': []}
            cell_data[cell]['classes'].append(int(classes[i]))
            cell_data[cell]['severities'].append(float(severities[i]))

        # Verify each occupied cell
        for cell, info in cell_data.items():
            cell_classes = info['classes']
            cell_severities = np.array(info['severities'])

            # Majority class using same logic as BEV implementation:
            # max(dict, key=dict.get) — on ties picks max key
            class_freq = {}
            for c in cell_classes:
                class_freq[c] = class_freq.get(c, 0) + 1
            expected_class = max(class_freq, key=class_freq.get)

            assert bev_map.class_grid[cell[0], cell[1]] == expected_class, (
                f"Cell {cell}: expected class {expected_class}, "
                f"got {bev_map.class_grid[cell[0], cell[1]]}"
            )

            # Max severity
            expected_severity = float(np.max(cell_severities))
            assert np.isclose(
                bev_map.severity_grid[cell[0], cell[1]],
                expected_severity,
                atol=1e-10
            ), (
                f"Cell {cell}: expected severity {expected_severity}, "
                f"got {bev_map.severity_grid[cell[0], cell[1]]}"
            )


# --- Property 24: PLY export round-trip ---

class TestPLYExportRoundTrip:
    """**Validates: Requirements 14.3**

    Property 24: For any point cloud with N points each having position
    (3 floats), color (3 uint8), class (int), and severity (float),
    exporting to PLY and re-reading SHALL recover all N points with
    identical attribute values.
    """

    @given(data=ply_point_cloud(min_points=1, max_points=50))
    @settings(max_examples=50, deadline=None)
    def test_ply_round_trip_preserves_point_count(self, data):
        """Export and re-read recovers the same number of points."""
        positions, colors, classes, severities = data

        aggregator = PointCloudAggregator()
        aggregator.add_frame(positions, classes, severities, colors=colors)

        with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as f:
            ply_path = Path(f.name)

        try:
            aggregator.export_ply(ply_path)
            ply_data = PlyData.read(str(ply_path))
            vertex = ply_data['vertex']
            assert len(vertex) == len(positions), (
                f"Expected {len(positions)} points, got {len(vertex)}"
            )
        finally:
            ply_path.unlink(missing_ok=True)

    @given(data=ply_point_cloud(min_points=1, max_points=50))
    @settings(max_examples=50, deadline=None)
    def test_ply_round_trip_preserves_positions(self, data):
        """Export and re-read recovers positions within float32 precision."""
        positions, colors, classes, severities = data

        aggregator = PointCloudAggregator()
        aggregator.add_frame(positions, classes, severities, colors=colors)

        with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as f:
            ply_path = Path(f.name)

        try:
            aggregator.export_ply(ply_path)
            ply_data = PlyData.read(str(ply_path))
            vertex = ply_data['vertex']

            read_x = np.array(vertex['x'], dtype=np.float32)
            read_y = np.array(vertex['y'], dtype=np.float32)
            read_z = np.array(vertex['z'], dtype=np.float32)

            # PLY stores as float32, so compare with float32 precision
            expected_positions = positions.astype(np.float32)

            np.testing.assert_allclose(
                read_x, expected_positions[:, 0], atol=1e-6,
                err_msg="X positions not preserved in PLY round-trip"
            )
            np.testing.assert_allclose(
                read_y, expected_positions[:, 1], atol=1e-6,
                err_msg="Y positions not preserved in PLY round-trip"
            )
            np.testing.assert_allclose(
                read_z, expected_positions[:, 2], atol=1e-6,
                err_msg="Z positions not preserved in PLY round-trip"
            )
        finally:
            ply_path.unlink(missing_ok=True)

    @given(data=ply_point_cloud(min_points=1, max_points=50))
    @settings(max_examples=50, deadline=None)
    def test_ply_round_trip_preserves_colors(self, data):
        """Export and re-read recovers RGB colors exactly."""
        positions, colors, classes, severities = data

        aggregator = PointCloudAggregator()
        aggregator.add_frame(positions, classes, severities, colors=colors)

        with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as f:
            ply_path = Path(f.name)

        try:
            aggregator.export_ply(ply_path)
            ply_data = PlyData.read(str(ply_path))
            vertex = ply_data['vertex']

            read_red = np.array(vertex['red'], dtype=np.uint8)
            read_green = np.array(vertex['green'], dtype=np.uint8)
            read_blue = np.array(vertex['blue'], dtype=np.uint8)

            np.testing.assert_array_equal(
                read_red, colors[:, 0],
                err_msg="Red channel not preserved in PLY round-trip"
            )
            np.testing.assert_array_equal(
                read_green, colors[:, 1],
                err_msg="Green channel not preserved in PLY round-trip"
            )
            np.testing.assert_array_equal(
                read_blue, colors[:, 2],
                err_msg="Blue channel not preserved in PLY round-trip"
            )
        finally:
            ply_path.unlink(missing_ok=True)

    @given(data=ply_point_cloud(min_points=1, max_points=50))
    @settings(max_examples=50, deadline=None)
    def test_ply_round_trip_preserves_class_ids(self, data):
        """Export and re-read recovers defect class IDs exactly."""
        positions, colors, classes, severities = data

        aggregator = PointCloudAggregator()
        aggregator.add_frame(positions, classes, severities, colors=colors)

        with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as f:
            ply_path = Path(f.name)

        try:
            aggregator.export_ply(ply_path)
            ply_data = PlyData.read(str(ply_path))
            vertex = ply_data['vertex']

            read_classes = np.array(vertex['class_id'], dtype=np.int32)

            np.testing.assert_array_equal(
                read_classes, classes,
                err_msg="Class IDs not preserved in PLY round-trip"
            )
        finally:
            ply_path.unlink(missing_ok=True)

    @given(data=ply_point_cloud(min_points=1, max_points=50))
    @settings(max_examples=50, deadline=None)
    def test_ply_round_trip_preserves_severities(self, data):
        """Export and re-read recovers severity values within float32 precision."""
        positions, colors, classes, severities = data

        aggregator = PointCloudAggregator()
        aggregator.add_frame(positions, classes, severities, colors=colors)

        with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as f:
            ply_path = Path(f.name)

        try:
            aggregator.export_ply(ply_path)
            ply_data = PlyData.read(str(ply_path))
            vertex = ply_data['vertex']

            read_severities = np.array(vertex['severity'], dtype=np.float32)

            # PLY stores as float32, so compare with float32 precision
            expected_severities = severities.astype(np.float32)

            np.testing.assert_allclose(
                read_severities, expected_severities, atol=1e-6,
                err_msg="Severities not preserved in PLY round-trip"
            )
        finally:
            ply_path.unlink(missing_ok=True)

    @given(data=ply_point_cloud(min_points=1, max_points=50))
    @settings(max_examples=50, deadline=None)
    def test_ply_round_trip_all_attributes(self, data):
        """Full round-trip: all N points with all attributes recovered."""
        positions, colors, classes, severities = data

        aggregator = PointCloudAggregator()
        aggregator.add_frame(positions, classes, severities, colors=colors)

        with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as f:
            ply_path = Path(f.name)

        try:
            aggregator.export_ply(ply_path)
            ply_data = PlyData.read(str(ply_path))
            vertex = ply_data['vertex']

            # Point count
            assert len(vertex) == len(positions)

            # Positions (float32 precision)
            expected_pos = positions.astype(np.float32)
            np.testing.assert_allclose(
                np.array(vertex['x'], dtype=np.float32), expected_pos[:, 0], atol=1e-6
            )
            np.testing.assert_allclose(
                np.array(vertex['y'], dtype=np.float32), expected_pos[:, 1], atol=1e-6
            )
            np.testing.assert_allclose(
                np.array(vertex['z'], dtype=np.float32), expected_pos[:, 2], atol=1e-6
            )

            # Colors (exact)
            np.testing.assert_array_equal(np.array(vertex['red'], dtype=np.uint8), colors[:, 0])
            np.testing.assert_array_equal(np.array(vertex['green'], dtype=np.uint8), colors[:, 1])
            np.testing.assert_array_equal(np.array(vertex['blue'], dtype=np.uint8), colors[:, 2])

            # Classes (exact)
            np.testing.assert_array_equal(
                np.array(vertex['class_id'], dtype=np.int32), classes
            )

            # Severities (float32 precision)
            expected_sev = severities.astype(np.float32)
            np.testing.assert_allclose(
                np.array(vertex['severity'], dtype=np.float32), expected_sev, atol=1e-6
            )
        finally:
            ply_path.unlink(missing_ok=True)
