"""Property-based test for encoder multi-scale output shapes.

**Validates: Requirements 6.2**

Property 11: Encoder multi-scale output shapes.
For any input tensor of shape [B, 3, 512, 512], the ResNet-50 DSC encoder SHALL
produce feature maps at shapes: stage1=[B, 256, 128, 128], stage2=[B, 512, 64, 64],
stage3=[B, 1024, 32, 32], stage4=[B, 2048, 16, 16].
"""

import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from src.model.encoder import ResNet50DSCEncoder


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
def test_encoder_multi_scale_output_shapes(
    batch_size: int,
    seed: int,
) -> None:
    """Property 11: Encoder multi-scale output shapes.

    **Validates: Requirements 6.2**

    For any [B, 3, 512, 512] input, the encoder produces feature maps at:
    - stage1: [B, 256, 128, 128]
    - stage2: [B, 512, 64, 64]
    - stage3: [B, 1024, 32, 32]
    - stage4: [B, 2048, 16, 16]
    """
    torch.manual_seed(seed)

    # Create input tensor: [B, 3, 512, 512] with realistic image range
    x = torch.randn(batch_size, 3, 512, 512)

    # Instantiate encoder without pretrained weights to keep test fast
    encoder = ResNet50DSCEncoder(pretrained=False)
    encoder.eval()

    with torch.no_grad():
        outputs = encoder(x)

    # Verify all four stage keys are present
    expected_keys = {'stage1', 'stage2', 'stage3', 'stage4'}
    assert set(outputs.keys()) == expected_keys, (
        f"Expected output keys {expected_keys}, got {set(outputs.keys())}."
    )

    # Property assertion: stage1 shape is [B, 256, 128, 128]
    assert outputs['stage1'].shape == (batch_size, 256, 128, 128), (
        f"Expected stage1 shape ({batch_size}, 256, 128, 128), "
        f"got {outputs['stage1'].shape}."
    )

    # Property assertion: stage2 shape is [B, 512, 64, 64]
    assert outputs['stage2'].shape == (batch_size, 512, 64, 64), (
        f"Expected stage2 shape ({batch_size}, 512, 64, 64), "
        f"got {outputs['stage2'].shape}."
    )

    # Property assertion: stage3 shape is [B, 1024, 32, 32]
    assert outputs['stage3'].shape == (batch_size, 1024, 32, 32), (
        f"Expected stage3 shape ({batch_size}, 1024, 32, 32), "
        f"got {outputs['stage3'].shape}."
    )

    # Property assertion: stage4 shape is [B, 2048, 16, 16]
    assert outputs['stage4'].shape == (batch_size, 2048, 16, 16), (
        f"Expected stage4 shape ({batch_size}, 2048, 16, 16), "
        f"got {outputs['stage4'].shape}."
    )
