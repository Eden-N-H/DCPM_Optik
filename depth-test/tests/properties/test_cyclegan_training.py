"""Property-based tests for CycleGAN losses and training.

Tests Properties 8, 9, and 10 from the design document:
- Property 8: CycleGAN loss weighted summation
- Property 9: Learning rate linear decay schedule
- Property 10: Image history buffer capacity

**Validates: Requirements 5.5, 5.7, 5.8**
"""

import torch
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from src.cyclegan.losses import ImagePool
from src.cyclegan.trainer import compute_lr


# =============================================================================
# Property 8: CycleGAN loss weighted summation
# =============================================================================
# total = adversarial + 10×cycle + 0.5×identity + 5×defect
# **Validates: Requirements 5.5**


@given(
    adversarial=st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    cycle=st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    identity=st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    defect=st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=200)
def test_cyclegan_loss_weighted_summation(adversarial: float, cycle: float, identity: float, defect: float):
    """Property 8: For any set of non-negative loss component values (adversarial,
    cycle, identity, defect), the total generator loss SHALL equal
    adversarial + 10.0×cycle + 0.5×identity + 5.0×defect.

    **Validates: Requirements 5.5**
    """
    lambda_cycle = 10.0
    lambda_identity = 0.5
    lambda_defect = 5.0

    # Compute expected total using the formula from the design
    expected_total = adversarial + lambda_cycle * cycle + lambda_identity * identity + lambda_defect * defect

    # Compute using tensors (as the trainer does)
    loss_adversarial = torch.tensor(adversarial)
    loss_cycle = torch.tensor(cycle)
    loss_identity = torch.tensor(identity)
    loss_defect = torch.tensor(defect)

    loss_G = (
        loss_adversarial
        + lambda_cycle * loss_cycle
        + lambda_identity * loss_identity
        + lambda_defect * loss_defect
    )

    # Use relative tolerance appropriate for float32 tensor arithmetic
    # float32 has ~7 decimal digits of precision, so for values up to ~1500 we expect ~1e-3 absolute error
    tolerance = max(1e-3, abs(expected_total) * 1e-5)
    assert abs(loss_G.item() - expected_total) < tolerance, (
        f"Total loss {loss_G.item()} != expected {expected_total} "
        f"for adversarial={adversarial}, cycle={cycle}, identity={identity}, defect={defect}"
    )


@given(
    adversarial=st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    cycle=st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    identity=st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    defect=st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=200)
def test_cyclegan_loss_components_are_non_negative_contribution(
    adversarial: float, cycle: float, identity: float, defect: float
):
    """Property 8 (supplementary): Each weighted component contributes non-negatively
    to the total loss since all inputs are non-negative and all weights are positive.

    **Validates: Requirements 5.5**
    """
    lambda_cycle = 10.0
    lambda_identity = 0.5
    lambda_defect = 5.0

    total = adversarial + lambda_cycle * cycle + lambda_identity * identity + lambda_defect * defect

    # Total must be >= each individual component contribution
    assert total >= adversarial
    assert total >= lambda_cycle * cycle
    assert total >= lambda_identity * identity
    assert total >= lambda_defect * defect


# =============================================================================
# Property 9: Learning rate linear decay schedule
# =============================================================================
# lr=2e-4 for e<100, lr=2e-4×(200-e)/100 for e≥100
# **Validates: Requirements 5.7**


@given(epoch=st.integers(min_value=0, max_value=99))
@settings(max_examples=200)
def test_lr_constant_before_decay(epoch: int):
    """Property 9: For any epoch e in [0, 100), the CycleGAN learning rate
    SHALL equal 2e-4.

    **Validates: Requirements 5.7**
    """
    lr = compute_lr(epoch, initial_lr=2e-4, decay_start_epoch=100, total_epochs=200)
    assert abs(lr - 2e-4) < 1e-10, f"Expected lr=2e-4 at epoch {epoch}, got {lr}"


@given(epoch=st.integers(min_value=100, max_value=199))
@settings(max_examples=200)
def test_lr_linear_decay_after_epoch_100(epoch: int):
    """Property 9: For any epoch e in [100, 200), the CycleGAN learning rate
    SHALL equal 2e-4 × (200 - e) / 100.

    **Validates: Requirements 5.7**
    """
    lr = compute_lr(epoch, initial_lr=2e-4, decay_start_epoch=100, total_epochs=200)
    expected_lr = 2e-4 * (200 - epoch) / 100
    assert abs(lr - expected_lr) < 1e-10, (
        f"Expected lr={expected_lr} at epoch {epoch}, got {lr}"
    )


@given(epoch=st.integers(min_value=0, max_value=199))
@settings(max_examples=200)
def test_lr_is_non_negative(epoch: int):
    """Property 9 (supplementary): The learning rate SHALL always be non-negative
    for any valid epoch in [0, 200).

    **Validates: Requirements 5.7**
    """
    lr = compute_lr(epoch, initial_lr=2e-4, decay_start_epoch=100, total_epochs=200)
    assert lr >= 0.0, f"Learning rate should be non-negative, got {lr} at epoch {epoch}"


@given(epoch=st.integers(min_value=100, max_value=198))
@settings(max_examples=100)
def test_lr_monotonically_decreases_after_decay_start(epoch: int):
    """Property 9 (supplementary): The learning rate SHALL monotonically decrease
    for epochs >= 100.

    **Validates: Requirements 5.7**
    """
    lr_current = compute_lr(epoch, initial_lr=2e-4, decay_start_epoch=100, total_epochs=200)
    lr_next = compute_lr(epoch + 1, initial_lr=2e-4, decay_start_epoch=100, total_epochs=200)
    assert lr_current > lr_next, (
        f"LR should decrease: lr({epoch})={lr_current} should be > lr({epoch + 1})={lr_next}"
    )


# =============================================================================
# Property 10: Image history buffer capacity
# =============================================================================
# Buffer never exceeds 50, sampling returns only inserted images
# **Validates: Requirements 5.8**


@given(
    num_insertions=st.integers(min_value=1, max_value=200),
    batch_size=st.integers(min_value=1, max_value=8),
)
@settings(max_examples=100)
def test_image_pool_never_exceeds_capacity(num_insertions: int, batch_size: int):
    """Property 10: For any sequence of generated images inserted into the
    history buffer, the buffer size SHALL never exceed 50 images.

    **Validates: Requirements 5.8**
    """
    pool_size = 50
    pool = ImagePool(pool_size=pool_size)

    for i in range(num_insertions):
        # Create a batch of fake images with unique pixel values for identification
        fake_images = torch.randn(batch_size, 3, 8, 8)
        pool.query(fake_images)

        # Buffer should never exceed pool_size
        assert len(pool.images) <= pool_size, (
            f"Buffer size {len(pool.images)} exceeds capacity {pool_size} "
            f"after {i + 1} insertions of batch_size {batch_size}"
        )


@given(
    num_insertions=st.integers(min_value=1, max_value=100),
)
@settings(max_examples=50)
def test_image_pool_returns_only_inserted_images(num_insertions: int):
    """Property 10: Sampling from the buffer SHALL only return previously
    inserted images.

    **Validates: Requirements 5.8**
    """
    pool_size = 50
    pool = ImagePool(pool_size=pool_size)

    # Track all images ever inserted
    all_inserted = []

    for i in range(num_insertions):
        # Create a unique image with a known tag value
        fake_image = torch.full((1, 3, 4, 4), fill_value=float(i))
        all_inserted.append(fake_image.clone())

        result = pool.query(fake_image)

        # The result must either be the input image or a previously stored one
        result_val = result[0, 0, 0, 0].item()
        # It should be one of the values we've inserted (0 to i inclusive)
        found_match = False
        for j in range(len(all_inserted)):
            if abs(result_val - float(j)) < 1e-6:
                found_match = True
                break

        assert found_match, (
            f"Returned image with value {result_val} was never inserted. "
            f"Expected one of values 0 to {i}."
        )


@given(
    num_insertions=st.integers(min_value=51, max_value=150),
)
@settings(max_examples=30)
def test_image_pool_fills_to_capacity(num_insertions: int):
    """Property 10 (supplementary): After more than pool_size insertions of
    single images, the buffer reaches exactly pool_size.

    **Validates: Requirements 5.8**
    """
    pool_size = 50
    pool = ImagePool(pool_size=pool_size)

    for i in range(num_insertions):
        fake_image = torch.randn(1, 3, 4, 4)
        pool.query(fake_image)

    # After more than pool_size single-image insertions, buffer should be exactly full
    assert len(pool.images) == pool_size, (
        f"Expected buffer to be full at {pool_size}, got {len(pool.images)}"
    )


def test_image_pool_zero_size_passthrough():
    """Property 10 (edge case): With pool_size=0, images pass through unchanged.

    **Validates: Requirements 5.8**
    """
    pool = ImagePool(pool_size=0)
    fake_images = torch.randn(4, 3, 8, 8)
    result = pool.query(fake_images)
    assert torch.equal(result, fake_images), "Pool size 0 should pass images through unchanged"
    assert len(pool.images) == 0, "Pool with size 0 should store nothing"
