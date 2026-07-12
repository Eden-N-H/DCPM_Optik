"""Property-based tests for point cloud operations (Properties 21, 22).

Property 21: Point cloud aggregation preserves all data
Property 22: Point cloud filtering correctness

Validates: Requirements 13.3, 13.4
"""
import numpy as np
from hypothesis import given, settings, assume
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from src.reconstruction.aggregator import PointCloudAggregator


# --- Strategies ---

@st.composite
def point_cloud_frame(draw, min_points=1, max_points=50):
    """Generate a single frame of point cloud data with attributes.

    Returns (positions, classes, severities, confidences) tuple.
    """
    n = draw(st.integers(min_value=min_points, max_value=max_points))

    positions = draw(arrays(
        np.float64, (n, 3),
        elements=st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False)
    ))
    classes = draw(arrays(
        np.int32, n,
        elements=st.integers(min_value=0, max_value=6)
    ))
    severities = draw(arrays(
        np.float64, n,
        elements=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
    ))
    confidences = draw(arrays(
        np.float64, n,
        elements=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
    ))

    return positions, classes, severities, confidences


@st.composite
def multiple_frames(draw, min_frames=1, max_frames=5, min_points=1, max_points=30):
    """Generate multiple frames of point cloud data."""
    num_frames = draw(st.integers(min_value=min_frames, max_value=max_frames))
    frames = []
    for _ in range(num_frames):
        frame = draw(point_cloud_frame(min_points=min_points, max_points=max_points))
        frames.append(frame)
    return frames


@st.composite
def filterable_point_cloud(draw, min_points=5, max_points=50):
    """Generate a point cloud with a mix of confidence and height values.

    Ensures there are both points that pass and fail the filter criteria.
    """
    n = draw(st.integers(min_value=min_points, max_value=max_points))

    # Generate positions with y-values spanning inside and outside default range [-0.5, 0.5]
    x = draw(arrays(np.float64, n, elements=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False)))
    y = draw(arrays(np.float64, n, elements=st.floats(min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False)))
    z = draw(arrays(np.float64, n, elements=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False)))
    positions = np.stack([x, y, z], axis=1)

    classes = draw(arrays(np.int32, n, elements=st.integers(min_value=0, max_value=6)))
    severities = draw(arrays(np.float64, n, elements=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)))

    # Confidences span [0, 1] to ensure some pass and some fail threshold of 0.5
    confidences = draw(arrays(np.float64, n, elements=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)))

    return positions, classes, severities, confidences


# --- Property 21: Point cloud aggregation preserves all data ---

class TestPointCloudAggregationPreservesData:
    """**Validates: Requirements 13.3**

    Property 21: For any sequence of N frames, each with arbitrary point counts,
    after aggregation the total point count SHALL equal the sum of all per-frame
    point counts, and every point's attributes (position, class, severity) SHALL
    be preserved exactly.
    """

    @given(frames=multiple_frames(min_frames=1, max_frames=5, min_points=1, max_points=30))
    @settings(max_examples=100, deadline=None)
    def test_total_count_equals_sum_of_frame_counts(self, frames):
        """After N frames, total count equals sum of per-frame counts."""
        aggregator = PointCloudAggregator()
        expected_total = 0

        for positions, classes, severities, confidences in frames:
            aggregator.add_frame(positions, classes, severities, confidences)
            expected_total += len(positions)

        assert aggregator.total_points == expected_total

    @given(frames=multiple_frames(min_frames=1, max_frames=5, min_points=1, max_points=30))
    @settings(max_examples=100, deadline=None)
    def test_positions_preserved_after_aggregation(self, frames):
        """All position data is preserved exactly after aggregation."""
        aggregator = PointCloudAggregator()

        for positions, classes, severities, confidences in frames:
            aggregator.add_frame(positions, classes, severities, confidences)

        all_positions, _, _, _, _ = aggregator.get_aggregated()

        # Concatenate expected positions
        expected_positions = np.concatenate([f[0] for f in frames], axis=0)

        np.testing.assert_array_equal(
            all_positions, expected_positions,
            err_msg="Positions not preserved after aggregation"
        )

    @given(frames=multiple_frames(min_frames=1, max_frames=5, min_points=1, max_points=30))
    @settings(max_examples=100, deadline=None)
    def test_classes_preserved_after_aggregation(self, frames):
        """All class attributes are preserved exactly after aggregation."""
        aggregator = PointCloudAggregator()

        for positions, classes, severities, confidences in frames:
            aggregator.add_frame(positions, classes, severities, confidences)

        _, all_classes, _, _, _ = aggregator.get_aggregated()

        expected_classes = np.concatenate([f[1] for f in frames], axis=0)

        np.testing.assert_array_equal(
            all_classes, expected_classes,
            err_msg="Classes not preserved after aggregation"
        )

    @given(frames=multiple_frames(min_frames=1, max_frames=5, min_points=1, max_points=30))
    @settings(max_examples=100, deadline=None)
    def test_severities_preserved_after_aggregation(self, frames):
        """All severity attributes are preserved exactly after aggregation."""
        aggregator = PointCloudAggregator()

        for positions, classes, severities, confidences in frames:
            aggregator.add_frame(positions, classes, severities, confidences)

        _, _, all_severities, _, _ = aggregator.get_aggregated()

        expected_severities = np.concatenate([f[2] for f in frames], axis=0)

        np.testing.assert_array_equal(
            all_severities, expected_severities,
            err_msg="Severities not preserved after aggregation"
        )

    @given(frames=multiple_frames(min_frames=1, max_frames=4, min_points=1, max_points=20))
    @settings(max_examples=50, deadline=None)
    def test_confidences_preserved_after_aggregation(self, frames):
        """All confidence attributes are preserved exactly after aggregation."""
        aggregator = PointCloudAggregator()

        for positions, classes, severities, confidences in frames:
            aggregator.add_frame(positions, classes, severities, confidences)

        _, _, _, all_confidences, _ = aggregator.get_aggregated()

        expected_confidences = np.concatenate([f[3] for f in frames], axis=0)

        np.testing.assert_array_equal(
            all_confidences, expected_confidences,
            err_msg="Confidences not preserved after aggregation"
        )

    @given(frame=point_cloud_frame(min_points=0, max_points=0))
    @settings(max_examples=10, deadline=None)
    def test_empty_frames_contribute_zero_points(self, frame):
        """Adding frames with zero points does not affect aggregation."""
        aggregator = PointCloudAggregator()

        # Add an empty frame (min_points=0 gives empty arrays)
        positions = np.zeros((0, 3), dtype=np.float64)
        classes = np.zeros(0, dtype=np.int32)
        severities = np.zeros(0, dtype=np.float64)
        confidences = np.zeros(0, dtype=np.float64)

        aggregator.add_frame(positions, classes, severities, confidences)
        assert aggregator.total_points == 0

        all_pos, all_cls, all_sev, all_conf, _ = aggregator.get_aggregated()
        assert len(all_pos) == 0
        assert len(all_cls) == 0
        assert len(all_sev) == 0
        assert len(all_conf) == 0


# --- Property 22: Point cloud filtering correctness ---

class TestPointCloudFilteringCorrectness:
    """**Validates: Requirements 13.4**

    Property 22: For any point cloud, after filtering with a depth confidence
    threshold and height range, all remaining points SHALL have confidence >=
    threshold AND height within the specified range. No point satisfying both
    criteria SHALL be removed.
    """

    @given(data=filterable_point_cloud(min_points=5, max_points=50))
    @settings(max_examples=100, deadline=None)
    def test_remaining_points_satisfy_confidence_threshold(self, data):
        """All remaining points after filtering have confidence >= threshold."""
        positions, classes, severities, confidences = data
        threshold = 0.5

        aggregator = PointCloudAggregator()
        aggregator.add_frame(positions, classes, severities, confidences)
        aggregator.filter(depth_confidence_threshold=threshold)

        _, _, _, filtered_confidences, _ = aggregator.get_aggregated()

        if len(filtered_confidences) > 0:
            assert np.all(filtered_confidences >= threshold), (
                f"Found points with confidence below threshold {threshold}: "
                f"min confidence = {filtered_confidences.min()}"
            )

    @given(data=filterable_point_cloud(min_points=5, max_points=50))
    @settings(max_examples=100, deadline=None)
    def test_remaining_points_satisfy_height_range(self, data):
        """All remaining points after filtering have height within range."""
        positions, classes, severities, confidences = data
        height_range = (-0.5, 0.5)

        aggregator = PointCloudAggregator()
        aggregator.add_frame(positions, classes, severities, confidences)
        aggregator.filter(height_range=height_range)

        filtered_positions, _, _, _, _ = aggregator.get_aggregated()

        if len(filtered_positions) > 0:
            heights = filtered_positions[:, 1]
            assert np.all(heights >= height_range[0]), (
                f"Found points with height below minimum {height_range[0]}: "
                f"min height = {heights.min()}"
            )
            assert np.all(heights <= height_range[1]), (
                f"Found points with height above maximum {height_range[1]}: "
                f"max height = {heights.max()}"
            )

    @given(data=filterable_point_cloud(min_points=5, max_points=50))
    @settings(max_examples=100, deadline=None)
    def test_no_valid_point_removed(self, data):
        """No point satisfying both criteria is removed by filtering."""
        positions, classes, severities, confidences = data
        threshold = 0.5
        height_range = (-0.5, 0.5)

        aggregator = PointCloudAggregator()
        aggregator.add_frame(positions, classes, severities, confidences)

        # Compute expected valid mask manually
        confidence_mask = confidences >= threshold
        height_mask = (positions[:, 1] >= height_range[0]) & (positions[:, 1] <= height_range[1])
        expected_mask = confidence_mask & height_mask
        expected_count = int(np.sum(expected_mask))

        aggregator.filter(depth_confidence_threshold=threshold, height_range=height_range)

        assert aggregator.total_points == expected_count, (
            f"Expected {expected_count} points after filtering, got {aggregator.total_points}"
        )

    @given(data=filterable_point_cloud(min_points=5, max_points=50))
    @settings(max_examples=100, deadline=None)
    def test_filtered_attributes_match_valid_subset(self, data):
        """Filtered attributes exactly match the valid subset of original data."""
        positions, classes, severities, confidences = data
        threshold = 0.5
        height_range = (-0.5, 0.5)

        # Compute expected valid mask
        confidence_mask = confidences >= threshold
        height_mask = (positions[:, 1] >= height_range[0]) & (positions[:, 1] <= height_range[1])
        expected_mask = confidence_mask & height_mask

        expected_positions = positions[expected_mask]
        expected_classes = classes[expected_mask]
        expected_severities = severities[expected_mask]
        expected_confidences = confidences[expected_mask]

        aggregator = PointCloudAggregator()
        aggregator.add_frame(positions, classes, severities, confidences)
        aggregator.filter(depth_confidence_threshold=threshold, height_range=height_range)

        filt_pos, filt_cls, filt_sev, filt_conf, _ = aggregator.get_aggregated()

        np.testing.assert_array_equal(filt_pos, expected_positions)
        np.testing.assert_array_equal(filt_cls, expected_classes)
        np.testing.assert_array_equal(filt_sev, expected_severities)
        np.testing.assert_array_equal(filt_conf, expected_confidences)

    @given(
        data=filterable_point_cloud(min_points=5, max_points=30),
        threshold=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        min_height=st.floats(min_value=-5.0, max_value=0.0, allow_nan=False),
        max_height=st.floats(min_value=0.0, max_value=5.0, allow_nan=False),
    )
    @settings(max_examples=100, deadline=None)
    def test_filter_with_arbitrary_parameters(self, data, threshold, min_height, max_height):
        """Filtering correctness holds for any valid threshold and height range."""
        assume(min_height < max_height)

        positions, classes, severities, confidences = data
        height_range = (min_height, max_height)

        # Compute expected
        confidence_mask = confidences >= threshold
        height_mask = (positions[:, 1] >= height_range[0]) & (positions[:, 1] <= height_range[1])
        expected_mask = confidence_mask & height_mask
        expected_count = int(np.sum(expected_mask))

        aggregator = PointCloudAggregator()
        aggregator.add_frame(positions, classes, severities, confidences)
        aggregator.filter(depth_confidence_threshold=threshold, height_range=height_range)

        # All remaining points satisfy criteria
        filt_pos, _, _, filt_conf, _ = aggregator.get_aggregated()

        assert aggregator.total_points == expected_count

        if len(filt_pos) > 0:
            assert np.all(filt_conf >= threshold)
            assert np.all(filt_pos[:, 1] >= height_range[0])
            assert np.all(filt_pos[:, 1] <= height_range[1])
