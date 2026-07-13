"""Property-based test for SOA shape preservation and attention validity.

**Validates: Requirements 8.5, 8.1, 8.2**

Property 14: SOA shape preservation and attention validity.
For any input tensor of arbitrary spatial dimensions and channel count, the SOA
module SHALL produce an output of identical shape. Additionally, the internal
channel attention weights SHALL be in [0, 1] and spatial attention maps SHALL be
in [0, 1].
"""

import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from src.model.soa import SOA


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Batch size - small for memory/speed
batch_size_st = st.integers(min_value=1, max_value=3)

# Channel count - must be divisible by reduction (16) for the FC layers
# Also must be >= 16 for reduction to work
channels_st = st.sampled_from([16, 32, 64, 128, 256])

# Spatial dimensions - keep small for test speed
height_st = st.integers(min_value=1, max_value=16)
width_st = st.integers(min_value=1, max_value=16)

# Random seed for reproducibility within hypothesis
seed_st = st.integers(min_value=0, max_value=2**31 - 1)


# ---------------------------------------------------------------------------
# Property Tests
# ---------------------------------------------------------------------------


@settings(max_examples=30, deadline=None)
@given(
    batch_size=batch_size_st,
    channels=channels_st,
    height=height_st,
    width=width_st,
    seed=seed_st,
)
def test_soa_shape_preservation(
    batch_size: int,
    channels: int,
    height: int,
    width: int,
    seed: int,
) -> None:
    """Property 14 (shape): SOA output has identical shape to input.

    **Validates: Requirements 8.5**

    For any input tensor of arbitrary spatial dimensions and channel count,
    the SOA module SHALL produce an output of identical shape.
    """
    torch.manual_seed(seed)

    # Create input tensor with arbitrary shape
    x = torch.randn(batch_size, channels, height, width)

    # Instantiate SOA module
    module = SOA(channels=channels, reduction=16, alpha=0.3)
    module.eval()

    with torch.no_grad():
        output = module(x)

    # Property assertion: output shape matches input shape exactly
    assert output.shape == x.shape, (
        f"Expected output shape {x.shape}, got {output.shape}. "
        f"SOA must preserve spatial dimensions and channel count."
    )


@settings(max_examples=30, deadline=None)
@given(
    batch_size=batch_size_st,
    channels=channels_st,
    height=height_st,
    width=width_st,
    seed=seed_st,
)
def test_soa_channel_attention_weights_in_range(
    batch_size: int,
    channels: int,
    height: int,
    width: int,
    seed: int,
) -> None:
    """Property 14 (channel attention): Channel attention weights in [0, 1].

    **Validates: Requirements 8.1**

    The channel attention mechanism (GAP → FC → ReLU → FC → Sigmoid) SHALL
    produce per-channel weights in the range [0, 1].
    """
    torch.manual_seed(seed)

    # Create input tensor
    x = torch.randn(batch_size, channels, height, width)

    # Instantiate SOA module
    module = SOA(channels=channels, reduction=16, alpha=0.3)
    module.eval()

    # Extract channel attention weights by running the channel_attn sub-module
    with torch.no_grad():
        wc = module.channel_attn(x)  # [B, C] after Sigmoid

    # Property assertion: all channel attention weights in [0, 1]
    assert wc.min().item() >= 0.0, (
        f"Channel attention weight below 0: min = {wc.min().item():.8f}. "
        f"Sigmoid output must be non-negative."
    )
    assert wc.max().item() <= 1.0, (
        f"Channel attention weight above 1: max = {wc.max().item():.8f}. "
        f"Sigmoid output must not exceed 1."
    )


@settings(max_examples=30, deadline=None)
@given(
    batch_size=batch_size_st,
    channels=channels_st,
    height=height_st,
    width=width_st,
    seed=seed_st,
)
def test_soa_spatial_attention_map_in_range(
    batch_size: int,
    channels: int,
    height: int,
    width: int,
    seed: int,
) -> None:
    """Property 14 (spatial attention): Spatial attention map in [0, 1].

    **Validates: Requirements 8.2**

    The spatial attention mechanism (4 parallel avg pools → concat → 1x1 conv →
    sigmoid) SHALL produce a spatial attention map with all values in [0, 1].
    """
    torch.manual_seed(seed)

    # Create input tensor
    x = torch.randn(batch_size, channels, height, width)

    # Instantiate SOA module
    module = SOA(channels=channels, reduction=16, alpha=0.3)
    module.eval()

    with torch.no_grad():
        # First apply channel attention to get x_ca (same as forward pass)
        wc = module.channel_attn(x)  # [B, C]
        wc = wc.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        x_ca = x * wc  # [B, C, H, W]

        # Compute spatial attention input: channel-wise mean
        spatial_input = x_ca.mean(dim=1, keepdim=True)  # [B, 1, H, W]

        # Multi-scale pooling
        pooled = [pool(spatial_input) for pool in module.spatial_pools]
        spatial_cat = torch.cat(pooled, dim=1)  # [B, 4, H, W]

        # Spatial attention map via 1x1 conv + sigmoid
        ws = module.spatial_conv(spatial_cat)  # [B, 1, H, W]

    # Property assertion: all spatial attention values in [0, 1]
    assert ws.min().item() >= 0.0, (
        f"Spatial attention value below 0: min = {ws.min().item():.8f}. "
        f"Sigmoid output must be non-negative."
    )
    assert ws.max().item() <= 1.0, (
        f"Spatial attention value above 1: max = {ws.max().item():.8f}. "
        f"Sigmoid output must not exceed 1."
    )
