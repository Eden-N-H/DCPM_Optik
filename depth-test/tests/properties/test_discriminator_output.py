"""Property-based test for discriminator spatial output dimensions.

**Validates: Requirements 4.1**

Property 7: Discriminator spatial output dimensions.
For any input tensor of shape [B, 3, 256, 256], the PatchGAN Discriminator
SHALL produce an output of exactly shape [B, 1, 30, 30].
"""

import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from src.cyclegan.discriminator import PatchGANDiscriminator


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Batch size strategy - keep small for memory/speed
batch_size_st = st.integers(min_value=1, max_value=4)

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
def test_discriminator_spatial_output_dimensions(
    batch_size: int,
    seed: int,
) -> None:
    """Property 7: Discriminator spatial output dimensions.

    **Validates: Requirements 4.1**

    For any [B, 3, 256, 256] input, output is exactly [B, 1, 30, 30].
    """
    torch.manual_seed(seed)

    # Create input tensor of shape [B, 3, 256, 256] with arbitrary values
    x = torch.randn(batch_size, 3, 256, 256)

    # Instantiate discriminator with default params
    discriminator = PatchGANDiscriminator(input_channels=3, ndf=64)
    discriminator.eval()

    with torch.no_grad():
        output = discriminator(x)

    # Property assertion: Output shape is exactly [B, 1, 30, 30]
    expected_shape = (batch_size, 1, 30, 30)
    assert output.shape == expected_shape, (
        f"Expected output shape {expected_shape}, got {output.shape}. "
        f"The PatchGAN discriminator must produce a 30x30 grid of patch-level predictions."
    )
