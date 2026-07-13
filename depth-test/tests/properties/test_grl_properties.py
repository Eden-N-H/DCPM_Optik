"""Property-based test for Gradient Reversal Layer forward-backward semantics.

**Validates: Requirements 11.2, 11.3**

Property 17: Gradient Reversal Layer forward-backward semantics.
For any input tensor x and scaling factor λ, the GRL forward pass SHALL return x
unchanged, and during backpropagation the gradient SHALL be negated and scaled by -λ.
"""

import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from src.model.domain_adapter import GradientReversalLayer


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Batch size strategy - keep small for speed
batch_size_st = st.integers(min_value=1, max_value=4)

# Channel count
channels_st = st.integers(min_value=1, max_value=16)

# Spatial dimensions
height_st = st.integers(min_value=1, max_value=8)
width_st = st.integers(min_value=1, max_value=8)

# Lambda scaling factor - positive floats representative of practical use
lambda_st = st.floats(min_value=0.01, max_value=2.0, allow_nan=False, allow_infinity=False)

# Random seed for reproducibility within hypothesis
seed_st = st.integers(min_value=0, max_value=2**31 - 1)


# ---------------------------------------------------------------------------
# Property Tests
# ---------------------------------------------------------------------------


@settings(max_examples=50, deadline=None)
@given(
    batch_size=batch_size_st,
    channels=channels_st,
    height=height_st,
    width=width_st,
    lambda_val=lambda_st,
    seed=seed_st,
)
def test_grl_forward_returns_input_unchanged(
    batch_size: int,
    channels: int,
    height: int,
    width: int,
    lambda_val: float,
    seed: int,
) -> None:
    """Property 17 (forward): GRL forward pass returns x unchanged.

    **Validates: Requirements 11.2, 11.3**

    For any input tensor x and scaling factor λ, the forward pass output
    SHALL be numerically identical to the input.
    """
    torch.manual_seed(seed)

    x = torch.randn(batch_size, channels, height, width)

    grl = GradientReversalLayer(lambda_val=lambda_val)
    grl.eval()

    with torch.no_grad():
        output = grl(x)

    # Forward pass must return x unchanged
    assert output.shape == x.shape, (
        f"Expected output shape {x.shape}, got {output.shape}."
    )
    assert torch.allclose(output, x, atol=1e-7), (
        f"GRL forward output differs from input. "
        f"Max diff: {(output - x).abs().max().item():.8f}"
    )


@settings(max_examples=50, deadline=None)
@given(
    batch_size=batch_size_st,
    channels=channels_st,
    height=height_st,
    width=width_st,
    lambda_val=lambda_st,
    seed=seed_st,
)
def test_grl_backward_negates_and_scales_gradient(
    batch_size: int,
    channels: int,
    height: int,
    width: int,
    lambda_val: float,
    seed: int,
) -> None:
    """Property 17 (backward): GRL backward pass negates gradient scaled by -λ.

    **Validates: Requirements 11.2, 11.3**

    For any input tensor x with requires_grad=True and scaling factor λ,
    the gradient flowing back through GRL SHALL be -λ × upstream_gradient.
    """
    torch.manual_seed(seed)

    x = torch.randn(batch_size, channels, height, width, requires_grad=True)

    grl = GradientReversalLayer(lambda_val=lambda_val)

    output = grl(x)

    # Create a synthetic upstream gradient
    upstream_grad = torch.randn_like(output)

    # Backpropagate
    output.backward(upstream_grad)

    # The gradient at x should be -lambda_val * upstream_grad
    expected_grad = -lambda_val * upstream_grad

    assert x.grad is not None, "Gradient was not computed for input x."
    assert x.grad.shape == expected_grad.shape, (
        f"Expected gradient shape {expected_grad.shape}, got {x.grad.shape}."
    )
    assert torch.allclose(x.grad, expected_grad, atol=1e-5), (
        f"GRL backward gradient mismatch. "
        f"Expected -λ × upstream_grad (λ={lambda_val}). "
        f"Max diff: {(x.grad - expected_grad).abs().max().item():.8f}"
    )
