"""Unit tests for PointCloudAggregator."""
import numpy as np
import pytest
import tempfile
from pathlib import Path

from src.reconstruction.aggregator import PointCloudAggregator


class TestAddFrame:
    """Tests for adding frames to the aggregator."""

    def test_add_single_frame_preserves_count(self):
        """Adding a single frame preserves the total point count."""
        agg = PointCloudAggregator()
        points = np.random.randn(100, 3)
        classes = np.random.randint(0, 5, size=100)
        severities = np.random.rand(100)

        agg.add_frame(points, classes, severities)
        assert agg.total_points == 100

    def test_add_multiple_frames_preserves_total_count(self):
        """Adding multiple frames preserves the sum of all point counts."""
        agg = PointCloudAggregator()
        sizes = [50, 30, 120, 10]

        for n in sizes:
            points = np.random.randn(n, 3)
            classes = np.random.randint(0, 5, size=n)
            severities = np.random.rand(n)
            agg.add_frame(points, classes, severities)

        assert agg.total_points == sum(sizes)

    def test_add_frame_with_confidences(self):
        """Adding a frame with explicit confidences stores them correctly."""
        agg = PointCloudAggregator()
        points = np.random.randn(50, 3)
        classes = np.random.randint(0, 3, size=50)
        severities = np.random.rand(50)
        confidences = np.random.rand(50)

        agg.add_frame(points, classes, severities, confidences=confidences)

        pos, cls, sev, conf, col = agg.get_aggregated()
        np.testing.assert_array_almost_equal(conf, confidences)

    def test_add_frame_default_confidences_are_ones(self):
        """When no confidences are provided, they default to 1.0."""
        agg = PointCloudAggregator()
        points = np.random.randn(20, 3)
        classes = np.random.randint(0, 3, size=20)
        severities = np.random.rand(20)

        agg.add_frame(points, classes, severities)

        _, _, _, conf, _ = agg.get_aggregated()
        np.testing.assert_array_equal(conf, np.ones(20))

    def test_add_frame_preserves_attributes(self):
        """All per-point attributes are preserved after aggregation."""
        agg = PointCloudAggregator()
        points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        classes = np.array([1, 2], dtype=np.int32)
        severities = np.array([0.5, 0.9])
        confidences = np.array([0.8, 0.3])

        agg.add_frame(points, classes, severities, confidences=confidences)

        pos, cls, sev, conf, _ = agg.get_aggregated()
        np.testing.assert_array_almost_equal(pos, points)
        np.testing.assert_array_equal(cls, classes)
        np.testing.assert_array_almost_equal(sev, severities)
        np.testing.assert_array_almost_equal(conf, confidences)


class TestFilter:
    """Tests for filtering the aggregated point cloud."""

    def test_filter_removes_low_confidence(self):
        """Points with confidence below threshold are removed."""
        agg = PointCloudAggregator()
        points = np.array([
            [0.0, 0.0, 1.0],  # conf 0.3 -> removed
            [1.0, 0.0, 2.0],  # conf 0.7 -> kept
            [2.0, 0.0, 3.0],  # conf 0.5 -> kept (at threshold)
        ])
        classes = np.array([1, 2, 1])
        severities = np.array([0.5, 0.8, 0.3])
        confidences = np.array([0.3, 0.7, 0.5])

        agg.add_frame(points, classes, severities, confidences=confidences)
        agg.filter(depth_confidence_threshold=0.5)

        pos, cls, sev, conf, _ = agg.get_aggregated()
        assert len(pos) == 2
        np.testing.assert_array_almost_equal(pos[0], [1.0, 0.0, 2.0])
        np.testing.assert_array_almost_equal(pos[1], [2.0, 0.0, 3.0])

    def test_filter_removes_outside_height_range(self):
        """Points outside the height range (y-coordinate) are removed."""
        agg = PointCloudAggregator()
        points = np.array([
            [0.0, -1.0, 1.0],   # y=-1.0 -> removed (below range)
            [1.0, 0.0, 2.0],    # y=0.0 -> kept
            [2.0, 0.3, 3.0],    # y=0.3 -> kept
            [3.0, 0.6, 4.0],    # y=0.6 -> removed (above range)
        ])
        classes = np.array([1, 2, 1, 3])
        severities = np.array([0.5, 0.8, 0.3, 0.9])
        confidences = np.ones(4)  # all confident

        agg.add_frame(points, classes, severities, confidences=confidences)
        agg.filter(depth_confidence_threshold=0.0, height_range=(-0.5, 0.5))

        pos, cls, sev, conf, _ = agg.get_aggregated()
        assert len(pos) == 2
        np.testing.assert_array_almost_equal(pos[0], [1.0, 0.0, 2.0])
        np.testing.assert_array_almost_equal(pos[1], [2.0, 0.3, 3.0])

    def test_filter_combined_confidence_and_height(self):
        """Points must satisfy both confidence and height criteria."""
        agg = PointCloudAggregator()
        points = np.array([
            [0.0, 0.0, 1.0],    # conf 0.8, y=0.0 -> kept (both pass)
            [1.0, 0.0, 2.0],    # conf 0.3, y=0.0 -> removed (low conf)
            [2.0, 2.0, 3.0],    # conf 0.8, y=2.0 -> removed (outside height)
            [3.0, 2.0, 4.0],    # conf 0.3, y=2.0 -> removed (both fail)
        ])
        classes = np.array([1, 2, 1, 3])
        severities = np.array([0.5, 0.8, 0.3, 0.9])
        confidences = np.array([0.8, 0.3, 0.8, 0.3])

        agg.add_frame(points, classes, severities, confidences=confidences)
        agg.filter(depth_confidence_threshold=0.5, height_range=(-0.5, 0.5))

        pos, _, _, _, _ = agg.get_aggregated()
        assert len(pos) == 1
        np.testing.assert_array_almost_equal(pos[0], [0.0, 0.0, 1.0])

    def test_no_valid_points_incorrectly_removed(self):
        """Points satisfying both criteria are never removed."""
        agg = PointCloudAggregator()
        # All points satisfy both conditions
        points = np.array([
            [0.0, 0.0, 1.0],
            [1.0, 0.2, 2.0],
            [2.0, -0.3, 3.0],
            [3.0, 0.5, 4.0],   # at boundary
            [4.0, -0.5, 5.0],  # at boundary
        ])
        classes = np.array([1, 2, 3, 4, 5])
        severities = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        confidences = np.array([0.5, 0.6, 0.7, 0.8, 0.9])

        agg.add_frame(points, classes, severities, confidences=confidences)
        agg.filter(depth_confidence_threshold=0.5, height_range=(-0.5, 0.5))

        pos, cls, sev, conf, _ = agg.get_aggregated()
        assert len(pos) == 5
        np.testing.assert_array_almost_equal(pos, points)
        np.testing.assert_array_equal(cls, classes)
        np.testing.assert_array_almost_equal(sev, severities)

    def test_filter_empty_aggregator(self):
        """Filtering an empty aggregator produces empty arrays."""
        agg = PointCloudAggregator()
        agg.filter()

        pos, cls, sev, conf, col = agg.get_aggregated()
        assert len(pos) == 0
        assert len(cls) == 0
        assert len(sev) == 0
        assert len(conf) == 0


class TestExportPly:
    """Tests for PLY export round-trip."""

    def test_ply_export_roundtrip(self):
        """Exporting to PLY and reading back recovers all attributes."""
        from plyfile import PlyData

        agg = PointCloudAggregator()
        points = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ])
        classes = np.array([0, 1, 2], dtype=np.int32)
        severities = np.array([0.1, 0.5, 0.9])
        colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)

        agg.add_frame(points, classes, severities, colors=colors)

        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
            ply_path = Path(f.name)

        try:
            agg.export_ply(ply_path)

            # Read back
            ply_data = PlyData.read(str(ply_path))
            vertex = ply_data['vertex']

            assert len(vertex) == 3
            np.testing.assert_array_almost_equal(vertex['x'], points[:, 0], decimal=5)
            np.testing.assert_array_almost_equal(vertex['y'], points[:, 1], decimal=5)
            np.testing.assert_array_almost_equal(vertex['z'], points[:, 2], decimal=5)
            np.testing.assert_array_equal(vertex['red'], colors[:, 0])
            np.testing.assert_array_equal(vertex['green'], colors[:, 1])
            np.testing.assert_array_equal(vertex['blue'], colors[:, 2])
            np.testing.assert_array_equal(vertex['class_id'], classes)
            np.testing.assert_array_almost_equal(vertex['severity'], severities, decimal=5)
        finally:
            ply_path.unlink(missing_ok=True)

    def test_ply_export_empty(self):
        """Exporting an empty point cloud creates a valid PLY with no vertices."""
        from plyfile import PlyData

        agg = PointCloudAggregator()

        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
            ply_path = Path(f.name)

        try:
            agg.export_ply(ply_path)

            ply_data = PlyData.read(str(ply_path))
            vertex = ply_data['vertex']
            assert len(vertex) == 0
        finally:
            ply_path.unlink(missing_ok=True)

    def test_ply_export_after_filter(self):
        """PLY export after filtering only includes filtered points."""
        from plyfile import PlyData

        agg = PointCloudAggregator()
        points = np.array([
            [0.0, 0.0, 1.0],   # kept (conf=0.9, y=0.0)
            [1.0, 0.0, 2.0],   # removed (conf=0.2)
            [2.0, 3.0, 3.0],   # removed (y=3.0 outside range)
        ])
        classes = np.array([1, 2, 3])
        severities = np.array([0.5, 0.8, 0.3])
        confidences = np.array([0.9, 0.2, 0.9])

        agg.add_frame(points, classes, severities, confidences=confidences)
        agg.filter(depth_confidence_threshold=0.5, height_range=(-0.5, 0.5))

        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
            ply_path = Path(f.name)

        try:
            agg.export_ply(ply_path)

            ply_data = PlyData.read(str(ply_path))
            vertex = ply_data['vertex']
            assert len(vertex) == 1
            assert vertex['x'][0] == pytest.approx(0.0, abs=1e-5)
            assert vertex['class_id'][0] == 1
        finally:
            ply_path.unlink(missing_ok=True)
