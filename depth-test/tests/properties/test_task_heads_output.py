"""Property-based test for task heads output shapes and value ranges.

**Validates: Requirements 10.1, 10.2, 10.3, 10.4**

Property 16: Task heads output shapes and value ranges.
For any shared decoder feature map of shape [B, 64, 128, 128]:
- The segmentation head SHALL produce shape [B, 3, 512, 512] (unbounded logits)
- The severity head SHALL produce shape [B, 1, 512, 512] with all values in [0, 1]
- The depth head SHALL produce shape [B, 1, 512, 512] with all values in [0, 1]
- The camera head SHALL produce intrinsics of shape [B, 4] with all values > 0
  (softplus) and extrinsics of shape [B, 6]
"""

import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from src.model.heads import CameraHead, DepthHead, SegmentationHead, SeverityHead


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Batch size strategy - keep small for memory/speed
batch_size_st = st.integers(min_value=1, max_value=3)

# Random seed for reproducibility within hypothesis
seed_st = st.integers(min_value=0, max_value=2**31 - 1)


# ---------------------------------------------------------------------------
# Property Tests
# ---------------------------------------------------------------------------


@settings(max_examples=10, deadline=None)
@given(
    batch_size=batch_size_st,
    seed=seed_st,
)
def test_segmentation_head_output_shape(
    batch_size: int,
    seed: int,
) -> None:
    """Property 16 (Segmentation): Output is [B, 3, 512, 512] unbounded logits.

    **Validates: Requirements 10.1**
    """
    torch.manual_seed(seed)

    # Shared decoder features: [B, 64, 128, 128]
    features = torch.randn(batch_size, 64, 128, 128)

    head = SegmentationHead(in_channels=64, num_classes=3)
    head.eval()

    with torch.no_grad():
        output = head(features)

    # Shape assertion
    assert output.shape == (batch_size, 3, 512, 512), (
        f"Expected segmentation output shape ({batch_size}, 3, 512, 512), "
        f"got {output.shape}."
    )

    # Logits are unbounded - verify they are finite (no NaN/Inf)
    assert torch.isfinite(output).all(), "Segmentation output contains NaN or Inf values."


@settings(max_examples=10, deadline=None)
@given(
    batch_size=batch_size_st,
    seed=seed_st,
)
def test_severity_head_output_shape_and_range(
    batch_size: int,
    seed: int,
) -> None:
    """Property 16 (Severity): Output is [B, 1, 512, 512] with values in [0, 1].

    **Validates: Requirements 10.2**
    """
    torch.manual_seed(seed)

    # Shared decoder features: [B, 64, 128, 128]
    features = torch.randn(batch_size, 64, 128, 128)

    head = SeverityHead(in_channels=64)
    head.eval()

    with torch.no_grad():
        output = head(features)

    # Shape assertion
    assert output.shape == (batch_size, 1, 512, 512), (
        f"Expected severity output shape ({batch_size}, 1, 512, 512), "
        f"got {output.shape}."
    )

    # Value range assertion: sigmoid produces [0, 1]
    assert output.min() >= 0.0, (
        f"Severity output has values below 0: min={output.min().item()}"
    )
    assert output.max() <= 1.0, (
        f"Severity output has values above 1: max={output.max().item()}"
    )


@settings(max_examples=10, deadline=None)
@given(
    batch_size=batch_size_st,
    seed=seed_st,
)
def test_depth_head_output_shape_and_range(
    batch_size: int,
    seed: int,
) -> None:
    """Property 16 (Depth): Output is [B, 1, 512, 512] with values in [0, 1].

    **Validates: Requirements 10.3**
    """
    torch.manual_seed(seed)

    # Shared decoder features: [B, 64, 128, 128]
    features = torch.randn(batch_size, 64, 128, 128)

    head = DepthHead(in_channels=64)
    head.eval()

    with torch.no_grad():
        output = head(features)

    # Shape assertion
    assert output.shape == (batch_size, 1, 512, 512), (
        f"Expected depth output shape ({batch_size}, 1, 512, 512), "
        f"got {output.shape}."
    )

    # Value range assertion: sigmoid produces [0, 1]
    assert output.min() >= 0.0, (
        f"Depth output has values below 0: min={output.min().item()}"
    )
    assert output.max() <= 1.0, (
        f"Depth output has values above 1: max={output.max().item()}"
    )


@settings(max_examples=10, deadline=None)
@given(
    batch_size=batch_size_st,
    seed=seed_st,
)
def test_camera_head_output_shapes_and_ranges(
    batch_size: int,
    seed: int,
) -> None:
    """Property 16 (Camera): Intrinsics [B, 4] > 0, extrinsics [B, 6].

    **Validates: Requirements 10.4**
    """
    torch.manual_seed(seed)

    # Shared decoder features: [B, 64, 128, 128]
    features = torch.randn(batch_size, 64, 128, 128)

    head = CameraHead(in_channels=64)
    head.eval()

    with torch.no_grad():
        intrinsics, extrinsics = head(features)

    # Shape assertions
    assert intrinsics.shape == (batch_size, 4), (
        f"Expected intrinsics shape ({batch_size}, 4), got {intrinsics.shape}."
    )
    assert extrinsics.shape == (batch_size, 6), (
        f"Expected extrinsics shape ({batch_size}, 6), got {extrinsics.shape}."
    )

    # Intrinsics must be > 0 (softplus output)
    assert (intrinsics > 0).all(), (
        f"Camera intrinsics must all be > 0, "
        f"min value={intrinsics.min().item()}"
    )

    # Extrinsics are unbounded (linear) - verify they are finite
    assert torch.isfinite(extrinsics).all(), (
        "Camera extrinsics contain NaN or Inf values."
    )
