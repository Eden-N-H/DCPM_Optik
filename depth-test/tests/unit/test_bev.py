"""Unit tests for BEVProjector and PLY export integration."""
import numpy as np
import pytest
import tempfile
from pathlib import Path

from src.reconstruction.bev import BEVProjector, DEFAULT_COLOR_MAP
from src.utils.data_types import BEVMap


class TestBEVProjectorMajorityVote:
    """Tests for BEV cell class assignment via majority vote."""

    def test_single_point_per_cell_assigns_that_class(self):
        """When only one point falls in a cell, that point's class is assigned."""
        projector = BEVProjector(resolution=1.0)
        # Three points separated along X, same Z, so each occupies its own cell
        positions = np.array([
            [0.5, 0.0, 0.5],
            [2.5, 0.0, 0.5],
            [4.5, 0.0, 0.5],
        ])
        classes = np.array([1, 2, 3])
        severities = np.array([0.5, 0.7, 0.9])

        bev = projector.project(positions, classes, severities)

        # Points span x=[0.5, 4.5], z=[0.5, 0.5]
        # xi for 0.5: (0.5-0.5)/1.0 = 0
        # xi for 2.5: (2.5-0.5)/1.0 = 2
        # xi for 4.5: (4.5-0.5)/1.0 = 4, clipped to W-1
        # Grid W = ceil((4.5-0.5)/1.0) = 4, so indices 0..3
        # xi: 0, 2, 3(clipped from 4)
        # With z range = 0, H = 1, zi = 0 for all
        # Let's verify by checking what we get
        assert bev.class_grid[0, 0] == 1
        assert bev.class_grid[0, 2] == 2
        # The last point at x=4.5 gets clipped to W-1=3
        assert bev.class_grid[0, 3] == 3

    def test_majority_vote_selects_most_frequent_class(self):
        """When multiple points share a cell, the most frequent class wins."""
        projector = BEVProjector(resolution=1.0)
        # All points in the same cell (within 1m resolution)
        positions = np.array([
            [0.1, 0.0, 0.1],
            [0.2, 0.0, 0.2],
            [0.3, 0.0, 0.3],
            [0.4, 0.0, 0.4],
            [0.5, 0.0, 0.5],
        ])
        # Class 2 appears 3 times, class 1 appears 2 times
        classes = np.array([1, 2, 2, 1, 2])
        severities = np.array([0.5, 0.6, 0.7, 0.8, 0.9])

        bev = projector.project(positions, classes, severities)

        # All points map to cell (0, 0) — majority is class 2
        assert bev.class_grid[0, 0] == 2

    def test_majority_vote_with_tie_selects_one(self):
        """When classes tie in frequency, one of the tied classes is selected."""
        projector = BEVProjector(resolution=1.0)
        positions = np.array([
            [0.1, 0.0, 0.1],
            [0.2, 0.0, 0.2],
            [0.3, 0.0, 0.3],
            [0.4, 0.0, 0.4],
        ])
        # Equal frequency: 2 of class 1 and 2 of class 3
        classes = np.array([1, 3, 1, 3])
        severities = np.array([0.5, 0.6, 0.7, 0.8])

        bev = projector.project(positions, classes, severities)

        # Result should be one of the tied classes
        assert bev.class_grid[0, 0] in (1, 3)

    def test_different_cells_have_independent_votes(self):
        """Majority vote is computed independently per cell."""
        projector = BEVProjector(resolution=1.0)
        # Two groups of points in different cells
        positions = np.array([
            # Cell 1 (x~0, z~0): class 1 dominates
            [0.1, 0.0, 0.1],
            [0.2, 0.0, 0.2],
            [0.3, 0.0, 0.3],
            # Cell 2 (x~3, z~3): class 2 dominates
            [3.1, 0.0, 3.1],
            [3.2, 0.0, 3.2],
            [3.3, 0.0, 3.3],
        ])
        classes = np.array([1, 1, 2, 2, 2, 1])
        severities = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 0.3])

        bev = projector.project(positions, classes, severities)

        assert bev.class_grid[0, 0] == 1  # class 1 majority in first cell
        assert bev.class_grid[3, 3] == 2  # class 2 majority in second cell


class TestBEVProjectorMaxSeverity:
    """Tests for BEV cell severity assignment via maximum."""

    def test_single_point_assigns_its_severity(self):
        """A single point in a cell assigns its own severity."""
        projector = BEVProjector(resolution=1.0)
        positions = np.array([[0.5, 0.0, 0.5]])
        classes = np.array([1])
        severities = np.array([0.75])

        bev = projector.project(positions, classes, severities)

        assert bev.severity_grid[0, 0] == pytest.approx(0.75)

    def test_multiple_points_assigns_maximum_severity(self):
        """Multiple points in one cell produce the max severity for that cell."""
        projector = BEVProjector(resolution=1.0)
        positions = np.array([
            [0.1, 0.0, 0.1],
            [0.2, 0.0, 0.2],
            [0.3, 0.0, 0.3],
            [0.4, 0.0, 0.4],
        ])
        classes = np.array([1, 1, 1, 1])
        severities = np.array([0.3, 0.9, 0.5, 0.7])

        bev = projector.project(positions, classes, severities)

        assert bev.severity_grid[0, 0] == pytest.approx(0.9)

    def test_cells_with_no_points_have_zero_severity(self):
        """Cells without any contributing points have severity 0."""
        projector = BEVProjector(resolution=1.0)
        # Points only in one corner
        positions = np.array([
            [0.1, 0.0, 0.1],
            [5.0, 0.0, 5.0],  # creates a larger grid
        ])
        classes = np.array([1, 2])
        severities = np.array([0.5, 0.8])

        bev = projector.project(positions, classes, severities)

        # Middle cells should be zero
        assert bev.severity_grid[2, 2] == 0.0

    def test_severity_zero_is_valid(self):
        """Points with severity 0.0 correctly set cell severity to 0.0."""
        projector = BEVProjector(resolution=1.0)
        positions = np.array([
            [0.1, 0.0, 0.1],
            [0.2, 0.0, 0.2],
        ])
        classes = np.array([1, 1])
        severities = np.array([0.0, 0.0])

        bev = projector.project(positions, classes, severities)

        assert bev.severity_grid[0, 0] == pytest.approx(0.0)

    def test_severity_one_is_maximum(self):
        """Severity of 1.0 is preserved as maximum."""
        projector = BEVProjector(resolution=1.0)
        positions = np.array([
            [0.1, 0.0, 0.1],
            [0.2, 0.0, 0.2],
            [0.3, 0.0, 0.3],
        ])
        classes = np.array([1, 1, 1])
        severities = np.array([0.3, 1.0, 0.7])

        bev = projector.project(positions, classes, severities)

        assert bev.severity_grid[0, 0] == pytest.approx(1.0)


class TestBEVProjectorGrid:
    """Tests for BEV grid computation and metadata."""

    def test_empty_point_cloud_returns_minimal_map(self):
        """Empty input produces a 1x1 BEV map with zeros."""
        projector = BEVProjector(resolution=0.02)
        positions = np.zeros((0, 3))
        classes = np.zeros(0, dtype=np.int32)
        severities = np.zeros(0)

        bev = projector.project(positions, classes, severities)

        assert bev.image.shape == (1, 1, 3)
        assert bev.class_grid.shape == (1, 1)
        assert bev.severity_grid.shape == (1, 1)
        assert bev.resolution == 0.02

    def test_resolution_affects_grid_size(self):
        """Finer resolution produces a larger grid."""
        positions = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
        ])
        classes = np.array([1, 2])
        severities = np.array([0.5, 0.8])

        proj_coarse = BEVProjector(resolution=0.5)
        proj_fine = BEVProjector(resolution=0.1)

        bev_coarse = proj_coarse.project(positions, classes, severities)
        bev_fine = proj_fine.project(positions, classes, severities)

        # Finer resolution should produce a larger grid
        assert bev_fine.class_grid.shape[0] > bev_coarse.class_grid.shape[0]
        assert bev_fine.class_grid.shape[1] > bev_coarse.class_grid.shape[1]

    def test_origin_is_min_coordinates(self):
        """BEV map origin matches minimum X and Z of input points."""
        projector = BEVProjector(resolution=0.1)
        positions = np.array([
            [1.5, 0.0, 2.0],
            [3.0, 0.0, 4.5],
            [2.0, 0.0, 3.0],
        ])
        classes = np.array([1, 2, 3])
        severities = np.array([0.5, 0.6, 0.7])

        bev = projector.project(positions, classes, severities)

        assert bev.origin[0] == pytest.approx(1.5)
        assert bev.origin[1] == pytest.approx(2.0)

    def test_resolution_stored_in_output(self):
        """The output BEVMap stores the resolution used for projection."""
        projector = BEVProjector(resolution=0.05)
        positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 1.0]])
        classes = np.array([1, 1])
        severities = np.array([0.5, 0.5])

        bev = projector.project(positions, classes, severities)

        assert bev.resolution == 0.05


class TestBEVExportPng:
    """Tests for BEV PNG export."""

    def test_export_creates_file(self):
        """export_png creates a valid image file."""
        import cv2

        projector = BEVProjector(resolution=0.5)
        positions = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [2.0, 0.0, 2.0],
        ])
        classes = np.array([1, 2, 3])
        severities = np.array([0.5, 0.8, 1.0])

        bev = projector.project(positions, classes, severities)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            png_path = Path(f.name)

        try:
            projector.export_png(bev, png_path)
            assert png_path.exists()

            # Read back and verify it's a valid image
            img = cv2.imread(str(png_path))
            assert img is not None
            assert img.shape[2] == 3  # 3-channel color
        finally:
            png_path.unlink(missing_ok=True)

    def test_export_with_custom_color_map(self):
        """export_png with custom color_map re-colorizes the image."""
        import cv2

        projector = BEVProjector(resolution=1.0)
        positions = np.array([[0.5, 0.0, 0.5]])
        classes = np.array([1])
        severities = np.array([1.0])

        bev = projector.project(positions, classes, severities)

        custom_colors = {1: (0, 255, 0)}  # bright green

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            png_path = Path(f.name)

        try:
            projector.export_png(bev, png_path, color_map=custom_colors)
            assert png_path.exists()

            img = cv2.imread(str(png_path))
            assert img is not None
            # The pixel should be green (BGR in cv2)
            # With severity=1.0, intensity=0.3+0.7*1=1.0, so full green
            assert img[0, 0, 1] == 255  # Green channel
        finally:
            png_path.unlink(missing_ok=True)

    def test_export_creates_parent_directories(self):
        """export_png creates parent directories if they don't exist."""
        projector = BEVProjector(resolution=1.0)
        positions = np.array([[0.5, 0.0, 0.5]])
        classes = np.array([1])
        severities = np.array([1.0])

        bev = projector.project(positions, classes, severities)

        with tempfile.TemporaryDirectory() as tmpdir:
            nested_path = Path(tmpdir) / "subdir" / "deep" / "bev.png"
            projector.export_png(bev, nested_path)
            assert nested_path.exists()


class TestBEVColorization:
    """Tests for BEV color-coding with severity modulation."""

    def test_severity_modulates_intensity(self):
        """Higher severity produces brighter colors."""
        projector = BEVProjector(resolution=1.0)

        # Two points in different cells along Z, same class, different severity
        positions = np.array([
            [0.5, 0.0, 0.5],
            [0.5, 0.0, 2.5],
        ])
        classes = np.array([2, 2])  # Both crack (red)
        severities = np.array([0.0, 1.0])

        bev = projector.project(positions, classes, severities)

        # zi for 0.5: (0.5-0.5)/1.0 = 0
        # zi for 2.5: (2.5-0.5)/1.0 = 2, clipped to H-1=1
        # Actually H = ceil((2.5-0.5)/1.0) = 2, so indices 0..1
        # Low severity at row 0, high severity at row 1
        low_brightness = int(bev.image[0, 0].sum())
        high_brightness = int(bev.image[1, 0].sum())
        assert high_brightness > low_brightness

    def test_default_color_map_has_expected_classes(self):
        """DEFAULT_COLOR_MAP contains entries for standard defect classes."""
        assert 0 in DEFAULT_COLOR_MAP  # Background
        assert 1 in DEFAULT_COLOR_MAP  # Road
        assert 2 in DEFAULT_COLOR_MAP  # Crack
        assert 3 in DEFAULT_COLOR_MAP  # Pothole
