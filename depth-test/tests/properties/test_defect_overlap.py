"""Property-based test for defect placement overlap constraint.

**Validates: Requirements 1.8**

Property 2: Defect placement respects overlap constraint.
For any set of 1-10 randomly placed defects on a road surface, after the
overlap resolution algorithm runs, no pair of defect instances SHALL overlap
by more than 25% of the smaller defect's area, and all defects SHALL remain
within the road surface bounds.
"""

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from src.synth.scene_generator import (
    SceneGenerator,
    SceneConfig,
    DEFECT_DIMENSIONS,
    OVERLAP_THRESHOLD,
    compute_overlap_fraction,
    is_within_road_bounds,
)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Road configuration strategy - constrained to valid ranges
road_lanes_st = st.integers(min_value=1, max_value=4)
lane_width_st = st.floats(min_value=3.0, max_value=3.75, allow_nan=False, allow_infinity=False)
road_length_st = st.floats(min_value=50.0, max_value=200.0, allow_nan=False, allow_infinity=False)

# Number of defects (1-10 as per requirement)
num_defects_st = st.integers(min_value=1, max_value=10)

# Random seed for reproducibility within hypothesis
seed_st = st.integers(min_value=0, max_value=2**31 - 1)


# ---------------------------------------------------------------------------
# Property Test
# ---------------------------------------------------------------------------


@settings(max_examples=50, deadline=None)
@given(
    lanes=road_lanes_st,
    lane_width=lane_width_st,
    road_length=road_length_st,
    num_defects=num_defects_st,
    seed=seed_st,
)
def test_defect_placement_respects_overlap_constraint(
    lanes: int,
    lane_width: float,
    road_length: float,
    num_defects: int,
    seed: int,
) -> None:
    """Property 2: Defect placement respects overlap constraint.

    **Validates: Requirements 1.8**

    For any 1-10 randomly placed defects, after overlap resolution:
    1. No pair overlaps >25% of the smaller defect's area
    2. All defects remain within road bounds
    """
    road_width = lanes * lane_width

    # Skip degenerate cases where road is too narrow for any defect
    # (minimum defect dimension is 0.1m for crack length, needs at least some space)
    assume(road_width >= 0.2)

    config = SceneConfig(
        lanes_range=(lanes, lanes),
        lane_width_range=(lane_width, lane_width),
        road_length_range=(road_length, road_length),
        defect_count_range=(num_defects, num_defects),
        overlap_threshold=OVERLAP_THRESHOLD,
    )

    generator = SceneGenerator(config=config, seed=seed)

    # Place defects using the overlap resolution algorithm
    placed = generator.place_defects(
        road_width=road_width,
        road_length=road_length,
        num_defects=num_defects,
    )

    # Property assertion 1: No pair overlaps > 25% of the smaller defect's area
    for i in range(len(placed)):
        for j in range(i + 1, len(placed)):
            overlap = compute_overlap_fraction(
                placed[i].bounding_box_2d,
                placed[i].area,
                placed[j].bounding_box_2d,
                placed[j].area,
            )
            assert overlap <= OVERLAP_THRESHOLD, (
                f"Defects {i} and {j} overlap by {overlap:.4f} "
                f"(threshold={OVERLAP_THRESHOLD}). "
                f"Defect {i}: type={placed[i].spec.defect_type}, "
                f"pos={placed[i].spec.position}, bbox={placed[i].bounding_box_2d}. "
                f"Defect {j}: type={placed[j].spec.defect_type}, "
                f"pos={placed[j].spec.position}, bbox={placed[j].bounding_box_2d}."
            )

    # Property assertion 2: All defects remain within road bounds
    for idx, defect in enumerate(placed):
        assert is_within_road_bounds(
            defect.bounding_box_2d, road_width, road_length
        ), (
            f"Defect {idx} is outside road bounds. "
            f"Type={defect.spec.defect_type}, "
            f"bbox={defect.bounding_box_2d}, "
            f"road_width={road_width}, road_length={road_length}."
        )
