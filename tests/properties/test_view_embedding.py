"""Property-based test for view embedding channel augmentation.

**Validates: Requirements 6.3**

Property 12: View embedding channel augmentation.
For any feature tensor of shape [B, 2048, H, W] and valid view label in {0, 1},
the View Embedding module SHALL produce an output of shape [B, 2080, H, W]
where the first 2048 channels equal the input features.
"""

import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from src.model.view_embedding import ViewEmbedding


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Batch size strategy - keep small for memory/speed
batch_size_st = st.integers(min_value=1, max_value=3)

# Spatial dimensions - keep small for feasibility with 2048 channels
height_st = st.integers(min_value=1, max_value=8)
width_st = st.integers(min_value=1, max_value=8)

# View label: 0 (dashcam) or 1 (drone)
view_st = st.integers(min_value=0, max_value=1)

# Random seed for reproducibility within hypothesis
seed_st = st.integers(min_value=0, max_value=2**31 - 1)


# ---------------------------------------------------------------------------
# Property Test
# ---------------------------------------------------------------------------


@settings(max_examples=30, deadline=None)
@given(
    batch_size=batch_size_st,
    height=height_st,
    width=width_st,
    view=view_st,
    seed=seed_st,
)
def test_view_embedding_channel_augmentation(
    batch_size: int,
    height: int,
    width: int,
    view: int,
    seed: int,
) -> None:
    """Property 12: View embedding channel augmentation.

    **Validates: Requirements 6.3**

    For any [B, 2048, H, W] features and view in {0,1}, output is [B, 2080, H, W]
    and first 2048 channels equal input.
    """
    torch.manual_seed(seed)

    # Create input feature tensor
    features = torch.randn(batch_size, 2048, height, width)

    # Create view labels for the batch (all same view for simplicity)
    view_label = torch.full((batch_size,), view, dtype=torch.long)

    # Instantiate ViewEmbedding module
    module = ViewEmbedding(num_views=2, embed_dim=32)
    module.eval()

    with torch.no_grad():
        output = module(features, view_label)

    # Property assertion 1: Output shape is [B, 2080, H, W]
    expected_shape = (batch_size, 2080, height, width)
    assert output.shape == expected_shape, (
        f"Expected output shape {expected_shape}, got {output.shape}."
    )

    # Property assertion 2: First 2048 channels equal the input features
    first_channels = output[:, :2048, :, :]
    assert torch.allclose(first_channels, features, atol=1e-6), (
        f"First 2048 channels of output do not match input features. "
        f"Max diff: {(first_channels - features).abs().max().item():.8f}"
    )
