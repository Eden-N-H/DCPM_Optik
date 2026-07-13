"""Property-based test for generator output bounded by tanh.

**Validates: Requirements 3.4**

Property 6: Generator output bounded by tanh.
For any valid input tensor of shape [B, 4, 256, 256] with RGB channels in [-1, 1]
and mask channel in {0, 1}, the Generator SHALL produce an output of shape
[B, 3, 256, 256] with all values in the range [-1, 1].
"""

import torch
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from src.cyclegan.generator import ResNetGenerator


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


@settings(max_examples=20, deadline=None)
@given(
    batch_size=batch_size_st,
    seed=seed_st,
)
def test_generator_output_bounded_by_tanh(
    batch_size: int,
    seed: int,
) -> None:
    """Property 6: Generator output bounded by tanh.

    **Validates: Requirements 3.4**

    For any [B, 4, 256, 256] input with RGB in [-1,1] and mask in {0,1},
    output shape is [B, 3, 256, 256] and all values in [-1, 1].
    """
    torch.manual_seed(seed)

    # Create input tensor with valid constraints:
    # - RGB channels (first 3): values in [-1, 1]
    # - Mask channel (4th): binary values in {0, 1}
    rgb = torch.empty(batch_size, 3, 256, 256).uniform_(-1.0, 1.0)
    mask = torch.randint(0, 2, (batch_size, 1, 256, 256), dtype=torch.float32)
    x = torch.cat([rgb, mask], dim=1)

    # Instantiate generator with default params
    generator = ResNetGenerator(input_channels=4, output_channels=3, ngf=64, n_residual_blocks=9)
    generator.eval()

    with torch.no_grad():
        output = generator(x)

    # Property assertion 1: Output shape is [B, 3, 256, 256]
    assert output.shape == (batch_size, 3, 256, 256), (
        f"Expected output shape ({batch_size}, 3, 256, 256), "
        f"got {output.shape}."
    )

    # Property assertion 2: All output values are in [-1, 1]
    assert output.min() >= -1.0, (
        f"Output contains values below -1.0: min={output.min().item():.6f}."
    )
    assert output.max() <= 1.0, (
        f"Output contains values above 1.0: max={output.max().item():.6f}."
    )
