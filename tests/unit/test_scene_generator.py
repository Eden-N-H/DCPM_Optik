"""Unit tests for the SceneGenerator class.

Validates overlap detection logic, camera setup, road mesh generation,
and defect placement with type-specific dimensions.
"""

import math

import numpy as np
import pytest

from src.synth.scene_generator import (
    CAMERA_CONFIGS,
    DEFECT_DIMENSIONS,
    OVERLAP_THRESHOLD,
    SceneConfig,
    SceneGenerator,
    compute_defect_area,
    compute_defect_bounding_box,
    compute_overlap_fraction,
    generate_extrinsics,
    generate_intrinsics,
    has_excessive_overlap,
    is_within_road_bounds,
)
from src.utils.data_types import CameraConfig, DefectInstance, DefectSpec


# ---------------------------------------------------------------------------
# Tests for compute_defect_area
# ---------------------------------------------------------------------------


class TestComputeDefectArea:
    """Tests for defect area computation."""

    def test_crack_area(self):
        """Crack area = length × width."""
        area = compute_defect_area("crack", (1.0, 0.02))
        assert area == pytest.approx(0.02)

    def test_pothole_area(self):
        """Pothole area = π × (diameter/2)²."""
        area = compute_defect_area("pothole", (1.0, 0.1))
        assert area == pytest.approx(math.pi * 0.25)

    def test_puddle_area(self):
        """Puddle area = π × (diameter/2)²."""
        area = compute_defect_area("puddle", (2.0,))
        assert area == pytest.approx(math.pi * 1.0)

    def test_patch_area(self):
        """Patch area = length × width."""
        area = compute_defect_area("patch", (2.0, 1.5))
        assert area == pytest.approx(3.0)

    def test_manhole_area(self):
        """Manhole area = π × (diameter/2)²."""
        area = compute_defect_area("manhole", (0.6,))
        assert area == pytest.approx(math.pi * 0.09)

    def test_unknown_type_raises(self):
        """Unknown defect type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown defect type"):
            compute_defect_area("unknown", (1.0,))


# ---------------------------------------------------------------------------
# Tests for compute_defect_bounding_box
# ---------------------------------------------------------------------------


class TestComputeDefectBoundingBox:
    """Tests for axis-aligned bounding box computation."""

    def test_crack_no_rotation(self):
        """Crack at 0° orientation has predictable AABB."""
        bbox = compute_defect_bounding_box("crack", (2.0, 0.04), (5.0, 10.0), 0.0)
        assert bbox == pytest.approx((4.0, 9.98, 6.0, 10.02))

    def test_crack_90_degree_rotation(self):
        """Crack at 90° swaps length and width extents."""
        bbox = compute_defect_bounding_box("crack", (2.0, 0.04), (5.0, 10.0), 90.0)
        # At 90°, half_x = length/2 * |cos90| + width/2 * |sin90| = 0 + 0.02 = 0.02
        # half_y = length/2 * |sin90| + width/2 * |cos90| = 1.0 + 0 = 1.0
        assert bbox == pytest.approx((4.98, 9.0, 5.02, 11.0), abs=1e-6)

    def test_pothole_is_circular(self):
        """Pothole AABB is a square centered on position."""
        bbox = compute_defect_bounding_box("pothole", (0.5, 0.1), (3.0, 7.0), 45.0)
        # radius = 0.25, rotation doesn't change circular AABB
        # half_x = 0.25*cos45 + 0.25*sin45 = 0.25*(sqrt2) ≈ 0.354
        # For circular defects it depends on our formulation
        # Actually for pothole, half_length = half_width = radius = 0.25
        # half_x = 0.25 * |cos45| + 0.25 * |sin45| = 0.25*0.707 + 0.25*0.707 = 0.354
        # This is expected since a circle's AABB doesn't change with rotation
        # but we model it as a square before rotation
        x_min, y_min, x_max, y_max = bbox
        assert (x_max - x_min) == pytest.approx(y_max - y_min, abs=1e-6)

    def test_manhole_symmetric(self):
        """Manhole AABB is symmetric regardless of orientation."""
        bbox0 = compute_defect_bounding_box("manhole", (0.6,), (2.0, 5.0), 0.0)
        bbox45 = compute_defect_bounding_box("manhole", (0.6,), (2.0, 5.0), 45.0)
        # For a circle modeled as square, 45° rotation gives larger bbox
        # but the center should remain the same
        center_0 = ((bbox0[0] + bbox0[2]) / 2, (bbox0[1] + bbox0[3]) / 2)
        center_45 = ((bbox45[0] + bbox45[2]) / 2, (bbox45[1] + bbox45[3]) / 2)
        assert center_0 == pytest.approx(center_45, abs=1e-10)

    def test_patch_at_origin(self):
        """Patch centered at origin with no rotation."""
        bbox = compute_defect_bounding_box("patch", (1.0, 0.5), (0.5, 0.25), 0.0)
        assert bbox == pytest.approx((0.0, 0.0, 1.0, 0.5))


# ---------------------------------------------------------------------------
# Tests for compute_overlap_fraction
# ---------------------------------------------------------------------------


class TestComputeOverlapFraction:
    """Tests for overlap fraction computation."""

    def test_no_overlap(self):
        """Non-overlapping boxes return 0."""
        fraction = compute_overlap_fraction(
            (0, 0, 1, 1), 1.0, (2, 2, 3, 3), 1.0
        )
        assert fraction == 0.0

    def test_full_overlap_same_size(self):
        """Identical boxes return 1.0."""
        fraction = compute_overlap_fraction(
            (0, 0, 1, 1), 1.0, (0, 0, 1, 1), 1.0
        )
        assert fraction == pytest.approx(1.0)

    def test_partial_overlap(self):
        """50% overlap of smaller box."""
        # Box A: (0, 0, 2, 2), area 4
        # Box B: (1, 0, 3, 1), area 2 (smaller)
        # Intersection: (1, 0, 2, 1), area 1
        # Overlap fraction = 1 / 2 = 0.5
        fraction = compute_overlap_fraction(
            (0, 0, 2, 2), 4.0, (1, 0, 3, 1), 2.0
        )
        assert fraction == pytest.approx(0.5)

    def test_small_inside_large(self):
        """Small box fully inside large box gives 1.0."""
        # Large: (0, 0, 10, 10), area 100
        # Small: (4, 4, 5, 5), area 1
        # Intersection: (4, 4, 5, 5), area 1
        # Fraction = 1 / min(100, 1) = 1.0
        fraction = compute_overlap_fraction(
            (0, 0, 10, 10), 100.0, (4, 4, 5, 5), 1.0
        )
        assert fraction == pytest.approx(1.0)

    def test_touching_edges(self):
        """Boxes touching at edge return 0."""
        fraction = compute_overlap_fraction(
            (0, 0, 1, 1), 1.0, (1, 0, 2, 1), 1.0
        )
        assert fraction == 0.0

    def test_zero_area_returns_zero(self):
        """Zero area defect returns 0 overlap."""
        fraction = compute_overlap_fraction(
            (0, 0, 1, 1), 0.0, (0, 0, 1, 1), 0.0
        )
        assert fraction == 0.0

    def test_just_below_threshold(self):
        """Overlap at exactly threshold boundary."""
        # Box A: (0, 0, 4, 4), area 16
        # Box B: (3, 0, 5, 2), area 4 (smaller)
        # Intersection: (3, 0, 4, 2), area 2
        # Fraction = 2 / 4 = 0.5 > 0.25 threshold
        fraction = compute_overlap_fraction(
            (0, 0, 4, 4), 16.0, (3, 0, 5, 2), 4.0
        )
        assert fraction == pytest.approx(0.5)
        assert fraction > OVERLAP_THRESHOLD


# ---------------------------------------------------------------------------
# Tests for has_excessive_overlap
# ---------------------------------------------------------------------------


class TestHasExcessiveOverlap:
    """Tests for the overlap detection function."""

    def _make_instance(self, bbox, area):
        """Create a minimal DefectInstance for testing."""
        return DefectInstance(
            spec=DefectSpec("crack", (0, 0), 0.0, (1.0, 0.01)),
            mesh_object=None,
            bounding_box_2d=bbox,
            area=area,
        )

    def test_no_existing_defects(self):
        """No overlap when there are no existing defects."""
        new = self._make_instance((0, 0, 1, 1), 1.0)
        assert has_excessive_overlap(new, []) is False

    def test_overlap_detected(self):
        """Overlap above threshold is detected."""
        existing = self._make_instance((0, 0, 2, 2), 4.0)
        new = self._make_instance((0, 0, 1, 1), 1.0)  # Fully inside
        assert has_excessive_overlap(new, [existing]) is True

    def test_no_overlap_detected(self):
        """Non-overlapping defects pass."""
        existing = self._make_instance((0, 0, 1, 1), 1.0)
        new = self._make_instance((5, 5, 6, 6), 1.0)
        assert has_excessive_overlap(new, [existing]) is False

    def test_overlap_below_threshold(self):
        """Overlap below 25% threshold passes."""
        # Existing: (0, 0, 10, 10), area 100
        # New: (9, 0, 11, 10), area 20
        # Intersection: (9, 0, 10, 10), area 10
        # Fraction = 10 / min(100, 20) = 10/20 = 0.5 > 0.25 -- fails
        # Let's try: existing=(0,0,10,10), area=100; new=(9.5,0,11,10), area=15
        # Intersection: (9.5,0,10,10), area=5
        # Fraction = 5/15 = 0.33 > 0.25 -- still fails
        # Try: existing=(0,0,10,1), area=10; new=(9,0,20,1), area=11
        # Intersection: (9,0,10,1), area=1
        # Fraction = 1/min(10,11) = 1/10 = 0.1 < 0.25 -- passes
        existing = self._make_instance((0, 0, 10, 1), 10.0)
        new = self._make_instance((9, 0, 20, 1), 11.0)
        assert has_excessive_overlap(new, [existing]) is False

    def test_custom_threshold(self):
        """Custom threshold changes overlap detection."""
        existing = self._make_instance((0, 0, 10, 1), 10.0)
        new = self._make_instance((9, 0, 20, 1), 11.0)
        # Fraction = 1/10 = 0.1; threshold=0.05 → exceeds
        assert has_excessive_overlap(new, [existing], threshold=0.05) is True


# ---------------------------------------------------------------------------
# Tests for is_within_road_bounds
# ---------------------------------------------------------------------------


class TestIsWithinRoadBounds:
    """Tests for road boundary checking."""

    def test_inside_bounds(self):
        """Box fully inside road returns True."""
        assert is_within_road_bounds((1, 1, 5, 50), 10.0, 100.0) is True

    def test_at_exact_bounds(self):
        """Box at exact road edges returns True."""
        assert is_within_road_bounds((0, 0, 10, 100), 10.0, 100.0) is True

    def test_exceeds_width(self):
        """Box exceeding road width returns False."""
        assert is_within_road_bounds((0, 0, 11, 50), 10.0, 100.0) is False

    def test_exceeds_length(self):
        """Box exceeding road length returns False."""
        assert is_within_road_bounds((0, 0, 5, 101), 10.0, 100.0) is False

    def test_negative_x(self):
        """Box with negative x returns False."""
        assert is_within_road_bounds((-1, 0, 5, 50), 10.0, 100.0) is False

    def test_negative_y(self):
        """Box with negative y returns False."""
        assert is_within_road_bounds((0, -1, 5, 50), 10.0, 100.0) is False


# ---------------------------------------------------------------------------
# Tests for camera setup
# ---------------------------------------------------------------------------


class TestCameraSetup:
    """Tests for camera configuration."""

    @pytest.fixture
    def generator(self):
        """Create a SceneGenerator with a fixed seed."""
        return SceneGenerator(seed=42)

    def test_dashcam_height_range(self, generator):
        """Dashcam height is within 1.2-1.5m."""
        config = generator.setup_camera("dashcam", road_length=100.0)
        assert 1.2 <= config.height <= 1.5

    def test_dashcam_pitch_range(self, generator):
        """Dashcam pitch is within -15° to -5°."""
        config = generator.setup_camera("dashcam", road_length=100.0)
        assert -15.0 <= config.pitch <= -5.0

    def test_drone_height_range(self, generator):
        """Drone height is within 8-15m."""
        config = generator.setup_camera("drone", road_length=100.0)
        assert 8.0 <= config.height <= 15.0

    def test_drone_pitch_range(self, generator):
        """Drone pitch is within -90° to -60°."""
        config = generator.setup_camera("drone", road_length=100.0)
        assert -90.0 <= config.pitch <= -60.0

    def test_dashcam_specific_values(self, generator):
        """Dashcam with specific height and pitch."""
        config = generator.setup_camera(
            "dashcam", road_length=100.0, height=1.3, pitch=-10.0
        )
        assert config.height == 1.3
        assert config.pitch == -10.0
        assert config.view_type == "dashcam"

    def test_drone_specific_values(self, generator):
        """Drone with specific height and pitch."""
        config = generator.setup_camera(
            "drone", road_length=100.0, height=12.0, pitch=-75.0
        )
        assert config.height == 12.0
        assert config.pitch == -75.0
        assert config.view_type == "drone"

    def test_intrinsics_shape(self, generator):
        """Intrinsics matrix is 3x3."""
        config = generator.setup_camera("dashcam", road_length=100.0)
        assert config.intrinsics.shape == (3, 3)

    def test_extrinsics_shape(self, generator):
        """Extrinsics matrix is 3x4."""
        config = generator.setup_camera("dashcam", road_length=100.0)
        assert config.extrinsics.shape == (3, 4)

    def test_intrinsics_structure(self, generator):
        """Intrinsics has proper structure: fx, fy on diagonal, cx/cy in last column."""
        config = generator.setup_camera("dashcam", road_length=100.0)
        K = config.intrinsics
        # Zero elements
        assert K[0, 1] == 0.0
        assert K[1, 0] == 0.0
        assert K[2, 0] == 0.0
        assert K[2, 1] == 0.0
        # K[2,2] = 1
        assert K[2, 2] == 1.0
        # Focal lengths positive
        assert K[0, 0] > 0
        assert K[1, 1] > 0
        # Principal point at center
        assert K[0, 2] == pytest.approx(256.0)
        assert K[1, 2] == pytest.approx(256.0)

    def test_extrinsics_rotation_is_valid(self, generator):
        """Rotation part of extrinsics is a proper rotation matrix."""
        config = generator.setup_camera("dashcam", road_length=100.0)
        R = config.extrinsics[:, :3]
        # R^T * R should be identity (orthogonal)
        RTR = R.T @ R
        assert np.allclose(RTR, np.eye(3), atol=1e-10)
        # det(R) should be 1 (proper rotation)
        assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-10)

    def test_invalid_view_type(self, generator):
        """Invalid view type raises ValueError."""
        with pytest.raises(ValueError, match="view_type"):
            generator.setup_camera("satellite", road_length=100.0)

    def test_height_out_of_range(self, generator):
        """Height outside valid range raises ValueError."""
        with pytest.raises(ValueError, match="height"):
            generator.setup_camera("dashcam", road_length=100.0, height=5.0)

    def test_pitch_out_of_range(self, generator):
        """Pitch outside valid range raises ValueError."""
        with pytest.raises(ValueError, match="pitch"):
            generator.setup_camera("drone", road_length=100.0, pitch=-30.0)


# ---------------------------------------------------------------------------
# Tests for generate_intrinsics and generate_extrinsics
# ---------------------------------------------------------------------------


class TestIntrinsicsExtrinsics:
    """Tests for intrinsics/extrinsics generation utilities."""

    def test_intrinsics_default_image_size(self):
        """Default image size 512 places principal point at (256, 256)."""
        K = generate_intrinsics(400.0)
        assert K[0, 2] == pytest.approx(256.0)
        assert K[1, 2] == pytest.approx(256.0)

    def test_intrinsics_custom_image_size(self):
        """Custom image size places principal point correctly."""
        K = generate_intrinsics(300.0, image_size=1024)
        assert K[0, 2] == pytest.approx(512.0)
        assert K[1, 2] == pytest.approx(512.0)

    def test_intrinsics_focal_length(self):
        """Focal length is correctly placed on diagonal."""
        K = generate_intrinsics(500.0)
        assert K[0, 0] == pytest.approx(500.0)
        assert K[1, 1] == pytest.approx(500.0)

    def test_extrinsics_shape(self):
        """Extrinsics is always 3x4."""
        ext = generate_extrinsics(1.5, -10.0, 100.0)
        assert ext.shape == (3, 4)

    def test_extrinsics_zero_pitch(self):
        """Zero pitch gives identity-like rotation."""
        ext = generate_extrinsics(2.0, 0.0, 50.0)
        R = ext[:, :3]
        assert np.allclose(R, np.eye(3), atol=1e-10)

    def test_extrinsics_translation(self):
        """Translation reflects height and road position."""
        ext = generate_extrinsics(1.5, -10.0, 100.0)
        t = ext[:, 3]
        assert t[1] == pytest.approx(1.5)  # height
        assert t[2] == pytest.approx(50.0)  # road_length / 2


# ---------------------------------------------------------------------------
# Tests for road mesh generation
# ---------------------------------------------------------------------------


class TestGenerateRoadMesh:
    """Tests for road mesh generation."""

    @pytest.fixture
    def generator(self):
        return SceneGenerator(seed=42)

    def test_specific_parameters(self, generator):
        """Road mesh with specific parameters returns correct dimensions."""
        mesh, width, length = generator.generate_road_mesh(
            lanes=2, lane_width=3.5, length=100.0
        )
        assert width == pytest.approx(7.0)
        assert length == pytest.approx(100.0)

    def test_random_parameters_in_range(self, generator):
        """Random parameters are within configured ranges."""
        mesh, width, length = generator.generate_road_mesh()
        assert 3.0 <= width <= 15.0  # 1-4 lanes * 3.0-3.75m
        assert 50.0 <= length <= 200.0

    def test_invalid_lanes(self, generator):
        """Invalid lane count raises ValueError."""
        with pytest.raises(ValueError, match="lanes"):
            generator.generate_road_mesh(lanes=0)
        with pytest.raises(ValueError, match="lanes"):
            generator.generate_road_mesh(lanes=5)

    def test_invalid_lane_width(self, generator):
        """Invalid lane width raises ValueError."""
        with pytest.raises(ValueError, match="lane_width"):
            generator.generate_road_mesh(lane_width=2.0)
        with pytest.raises(ValueError, match="lane_width"):
            generator.generate_road_mesh(lane_width=4.0)

    def test_invalid_length(self, generator):
        """Invalid road length raises ValueError."""
        with pytest.raises(ValueError, match="length"):
            generator.generate_road_mesh(length=10.0)
        with pytest.raises(ValueError, match="length"):
            generator.generate_road_mesh(length=300.0)


# ---------------------------------------------------------------------------
# Tests for defect placement
# ---------------------------------------------------------------------------


class TestPlaceDefects:
    """Tests for defect placement with overlap resolution."""

    @pytest.fixture
    def generator(self):
        return SceneGenerator(seed=42)

    def test_places_correct_count(self, generator):
        """Placement generates the requested number of defects."""
        instances = generator.place_defects(
            road_width=15.0, road_length=200.0, num_defects=5
        )
        # May be fewer if placement fails, but generally should place all
        assert 1 <= len(instances) <= 5

    def test_all_within_bounds(self, generator):
        """All placed defects are within road bounds."""
        instances = generator.place_defects(
            road_width=15.0, road_length=200.0, num_defects=8
        )
        for inst in instances:
            assert is_within_road_bounds(
                inst.bounding_box_2d, 15.0, 200.0
            ), f"Defect out of bounds: {inst.bounding_box_2d}"

    def test_no_excessive_overlap(self, generator):
        """No pair of defects overlaps more than 25% of smaller area."""
        instances = generator.place_defects(
            road_width=15.0, road_length=200.0, num_defects=10
        )
        for i in range(len(instances)):
            for j in range(i + 1, len(instances)):
                overlap = compute_overlap_fraction(
                    instances[i].bounding_box_2d,
                    instances[i].area,
                    instances[j].bounding_box_2d,
                    instances[j].area,
                )
                assert overlap <= OVERLAP_THRESHOLD, (
                    f"Excessive overlap {overlap:.3f} between defects {i} and {j}"
                )

    def test_custom_defect_specs(self, generator):
        """Placement with pre-defined defect specs."""
        specs = [
            DefectSpec("crack", (5.0, 50.0), 0.0, (1.0, 0.02)),
            DefectSpec("pothole", (10.0, 80.0), 45.0, (0.5, 0.1)),
        ]
        instances = generator.place_defects(
            road_width=15.0, road_length=200.0, defects=specs
        )
        assert len(instances) == 2
        assert instances[0].spec.defect_type == "crack"
        assert instances[1].spec.defect_type == "pothole"

    def test_invalid_num_defects(self, generator):
        """Invalid defect count raises ValueError."""
        with pytest.raises(ValueError, match="num_defects"):
            generator.place_defects(road_width=15.0, road_length=200.0, num_defects=0)
        with pytest.raises(ValueError, match="num_defects"):
            generator.place_defects(road_width=15.0, road_length=200.0, num_defects=11)

    def test_narrow_road_still_places(self, generator):
        """Defects can still be placed on narrow roads."""
        instances = generator.place_defects(
            road_width=3.5, road_length=100.0, num_defects=3
        )
        # Should place at least some
        assert len(instances) >= 1

    def test_defect_areas_positive(self, generator):
        """All placed defects have positive area."""
        instances = generator.place_defects(
            road_width=15.0, road_length=200.0, num_defects=5
        )
        for inst in instances:
            assert inst.area > 0

    def test_reproducibility_with_seed(self):
        """Same seed produces identical placements."""
        gen1 = SceneGenerator(seed=123)
        gen2 = SceneGenerator(seed=123)

        instances1 = gen1.place_defects(road_width=15.0, road_length=200.0, num_defects=5)
        instances2 = gen2.place_defects(road_width=15.0, road_length=200.0, num_defects=5)

        assert len(instances1) == len(instances2)
        for i1, i2 in zip(instances1, instances2):
            assert i1.spec.defect_type == i2.spec.defect_type
            assert i1.spec.position == pytest.approx(i2.spec.position)
            assert i1.area == pytest.approx(i2.area)
