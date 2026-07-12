"""Property-based test for decoder output dimensions from multi-scale inputs.

**Validates: Requirements 9.1, 9.4**

Property 15: Decoder output dimensions from multi-scale inputs.
For any E-ASPP output of shape [B, 256, 16, 16] and encoder features at the
specified resolutions (stage1=[B, 256, 128, 128], stage2=[B, 512, 64, 64],
stage3=[B, 1024, 32, 32]), the Decoder SHALL produce an output of shape
[B, 64, 128, 128].
"""

import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from src.model.decoder import LightweightDecoder


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
def test_decoder_output_dimensions_from_multi_scale_inputs(
    batch_size: int,
    seed: int,
) -> None:
    """Property 15: Decoder output dimensions from multi-scale inputs.

    **Validates: Requirements 9.1, 9.4**

    For E-ASPP output [B, 256, 16, 16] and encoder features at specified
    resolutions, decoder output is [B, 64, 128, 128].
    """
    torch.manual_seed(seed)

    # Create E-ASPP output: [B, 256, 16, 16]
    aspp_out = torch.randn(batch_size, 256, 16, 16)

    # Create encoder features at specified resolutions
    encoder_features = {
        'stage1': torch.randn(batch_size, 256, 128, 128),
        'stage2': torch.randn(batch_size, 512, 64, 64),
        'stage3': torch.randn(batch_size, 1024, 32, 32),
    }

    # Instantiate decoder
    decoder = LightweightDecoder()
    decoder.eval()

    with torch.no_grad():
        output = decoder(aspp_out, encoder_features)

    # Property assertion: output shape is [B, 64, 128, 128]
    assert output.shape == (batch_size, 64, 128, 128), (
        f"Expected decoder output shape ({batch_size}, 64, 128, 128), "
        f"got {output.shape}."
    )
