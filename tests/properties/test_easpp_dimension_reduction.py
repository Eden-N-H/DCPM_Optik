"""Property-based test for E-ASPP dimension reduction.

**Validates: Requirements 7.3, 7.4**

Property 13: E-ASPP dimension reduction.
For any input tensor of shape [B, 2080, 16, 16], the E-ASPP module SHALL
produce an output of exactly shape [B, 256, 16, 16].
"""

import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from src.model.easpp import EASPP


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Batch size strategy - keep small for memory/speed
batch_size_st = st.integers(min_value=1, max_value=3)

# Random seed for reproducibility within hypothesis
seed_st = st.integers(min_value=0, max_value=2**31 - 1)


# ---------------------------------------------------------------------------
# Property Test
# ---------------------------------------------------------------------------


@settings(max_examples=10, deadline=None)
@given(
    batch_size=batch_size_st,
    seed=seed_st,
)
def test_easpp_dimension_reduction(
    batch_size: int,
    seed: int,
) -> None:
    """Property 13: E-ASPP dimension reduction.

    **Validates: Requirements 7.3, 7.4**

    For any [B, 2080, 16, 16] input, the E-ASPP module produces
    output of exactly [B, 256, 16, 16].
    """
    torch.manual_seed(seed)

    # Create input tensor: [B, 2080, 16, 16]
    # This represents encoder stage4 (2048 ch) + view embedding (32 ch)
    x = torch.randn(batch_size, 2080, 16, 16)

    # Instantiate E-ASPP module
    easpp = EASPP(in_channels=2080, out_channels=256)
    easpp.eval()

    with torch.no_grad():
        output = easpp(x)

    # Property assertion: output shape is exactly [B, 256, 16, 16]
    assert output.shape == (batch_size, 256, 16, 16), (
        f"Expected output shape ({batch_size}, 256, 16, 16), "
        f"got {output.shape}."
    )
