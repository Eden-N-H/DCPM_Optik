"""Property-based tests for checkpointing and determinism (Properties 27, 28).

Validates: Requirements 18.1, 18.4
"""

import random
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from src.training.checkpoint import (
    save_checkpoint,
    load_checkpoint,
    set_seed,
    get_rng_states,
    set_rng_states,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class SimpleModel(nn.Module):
    """A small deterministic model for testing checkpoint round-trips."""

    def __init__(self, in_features: int = 16, hidden: int = 32, out_features: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden, out_features)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


def build_training_components(in_features=16, hidden=32, out_features=4, lr=1e-3):
    """Create model, optimizer, scheduler for testing."""
    model = SimpleModel(in_features, hidden, out_features)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    return model, optimizer, scheduler


def do_training_step(model, optimizer, input_tensor, target_tensor):
    """Perform a single training step and return the loss value."""
    optimizer.zero_grad()
    output = model(input_tensor)
    loss = nn.functional.mse_loss(output, target_tensor)
    loss.backward()
    optimizer.step()
    return loss.item(), output.detach().clone()


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Strategy for epoch numbers
epochs_st = st.integers(min_value=0, max_value=200)

# Strategy for best metric values
best_metric_st = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)

# Strategy for seed values
seed_st = st.integers(min_value=0, max_value=2**31 - 1)

# Strategy for learning rates
lr_st = st.floats(min_value=1e-6, max_value=1e-1, allow_nan=False, allow_infinity=False)


# ---------------------------------------------------------------------------
# Property 27: Checkpoint serialization round-trip
# ---------------------------------------------------------------------------


class TestProperty27CheckpointRoundTrip:
    """**Validates: Requirements 18.1**

    Property 27: For any training state (model weights, optimizer state,
    scheduler state, epoch, best metric, RNG states), saving a checkpoint
    and loading it back SHALL produce an identical training state — subsequent
    forward passes and optimizer steps SHALL produce bit-identical results.
    """

    @given(epoch=epochs_st, best_metric=best_metric_st, seed=seed_st)
    @settings(max_examples=20, deadline=30000)
    def test_checkpoint_round_trip_preserves_state(self, epoch, best_metric, seed, tmp_path_factory):
        """Save and load produces identical training state with bit-identical forward passes."""
        tmp_dir = tmp_path_factory.mktemp("ckpt_rt")
        ckpt_path = tmp_dir / "checkpoint.pt"

        # Set a known seed to create deterministic initial state
        set_seed(seed)

        # Build training components
        model, optimizer, scheduler = build_training_components()

        # Advance the optimizer/scheduler state by a few steps to make it non-trivial
        input_data = torch.randn(4, 16)
        target_data = torch.randn(4, 4)
        for _ in range(3):
            do_training_step(model, optimizer, input_data, target_data)
            scheduler.step()

        # Capture the current state: do a forward pass before saving
        test_input = torch.randn(2, 16)
        pre_save_output = model(test_input).detach().clone()

        # Save checkpoint
        save_checkpoint(
            path=ckpt_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_metric=best_metric,
        )

        # Create fresh components to load into
        model2, optimizer2, scheduler2 = build_training_components()

        # Verify they differ before loading
        pre_load_output = model2(test_input).detach()
        # (They likely differ due to random init, but we don't assert this)

        # Load checkpoint
        checkpoint = load_checkpoint(
            path=ckpt_path,
            model=model2,
            optimizer=optimizer2,
            scheduler=scheduler2,
        )

        # Verify metadata is preserved
        assert checkpoint.epoch == epoch
        assert checkpoint.best_metric == best_metric

        # Verify forward pass is bit-identical
        post_load_output = model2(test_input).detach()
        assert torch.equal(pre_save_output, post_load_output), (
            "Forward pass after loading checkpoint is not bit-identical to before saving"
        )

    @given(seed=seed_st, epoch=epochs_st, best_metric=best_metric_st)
    @settings(max_examples=15, deadline=30000)
    def test_checkpoint_round_trip_optimizer_state(self, seed, epoch, best_metric, tmp_path_factory):
        """After loading checkpoint, optimizer steps produce bit-identical results."""
        tmp_dir = tmp_path_factory.mktemp("ckpt_opt")
        ckpt_path = tmp_dir / "checkpoint.pt"

        set_seed(seed)
        model, optimizer, scheduler = build_training_components()

        # Warm up optimizer state
        input_data = torch.randn(4, 16)
        target_data = torch.randn(4, 4)
        do_training_step(model, optimizer, input_data, target_data)

        # Save checkpoint
        save_checkpoint(
            path=ckpt_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_metric=best_metric,
        )

        # Do one more training step on the original model
        step_input = torch.randn(4, 16)
        step_target = torch.randn(4, 4)
        loss1, output1 = do_training_step(model, optimizer, step_input, step_target)

        # Load into fresh components
        model2, optimizer2, scheduler2 = build_training_components()
        load_checkpoint(
            path=ckpt_path,
            model=model2,
            optimizer=optimizer2,
            scheduler=scheduler2,
        )

        # Do the same training step on the loaded model
        loss2, output2 = do_training_step(model2, optimizer2, step_input, step_target)

        assert loss1 == loss2, (
            f"Loss after checkpoint round-trip differs: {loss1} vs {loss2}"
        )
        assert torch.equal(output1, output2), (
            "Output after training step differs between original and loaded model"
        )

    @given(seed=seed_st, best_metric=best_metric_st)
    @settings(max_examples=15, deadline=30000)
    def test_checkpoint_preserves_rng_states(self, seed, best_metric, tmp_path_factory):
        """Checkpoint save/load preserves RNG states for reproducibility."""
        tmp_dir = tmp_path_factory.mktemp("ckpt_rng")
        ckpt_path = tmp_dir / "checkpoint.pt"

        set_seed(seed)
        model, optimizer, scheduler = build_training_components()

        # Generate some random state advancement
        _ = torch.randn(10)
        _ = np.random.rand(10)
        _ = random.random()

        # Save checkpoint (captures current RNG states)
        save_checkpoint(
            path=ckpt_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=5,
            best_metric=best_metric,
        )

        # Generate random numbers after saving
        torch_vals_expected = torch.randn(5)
        np_vals_expected = np.random.rand(5)
        py_val_expected = random.random()

        # Load checkpoint (restores RNG states to point of save)
        model2, optimizer2, scheduler2 = build_training_components()
        load_checkpoint(
            path=ckpt_path,
            model=model2,
            optimizer=optimizer2,
            scheduler=scheduler2,
        )

        # Generate the same random numbers — should be identical
        torch_vals_actual = torch.randn(5)
        np_vals_actual = np.random.rand(5)
        py_val_actual = random.random()

        assert torch.equal(torch_vals_expected, torch_vals_actual), (
            "PyTorch RNG state not restored correctly"
        )
        np.testing.assert_array_equal(np_vals_expected, np_vals_actual)
        assert py_val_expected == py_val_actual, (
            "Python RNG state not restored correctly"
        )

    @given(epoch=epochs_st, best_metric=best_metric_st)
    @settings(max_examples=10, deadline=30000)
    def test_checkpoint_preserves_scheduler_state(self, epoch, best_metric, tmp_path_factory):
        """Checkpoint preserves scheduler state (step count, last_lr)."""
        tmp_dir = tmp_path_factory.mktemp("ckpt_sched")
        ckpt_path = tmp_dir / "checkpoint.pt"

        model, optimizer, scheduler = build_training_components()

        # Advance scheduler several times
        for _ in range(5):
            scheduler.step()

        original_lr = optimizer.param_groups[0]['lr']
        original_last_epoch = scheduler.last_epoch

        save_checkpoint(
            path=ckpt_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_metric=best_metric,
        )

        # Load into fresh components
        model2, optimizer2, scheduler2 = build_training_components()
        load_checkpoint(
            path=ckpt_path,
            model=model2,
            optimizer=optimizer2,
            scheduler=scheduler2,
        )

        loaded_lr = optimizer2.param_groups[0]['lr']
        loaded_last_epoch = scheduler2.last_epoch

        assert loaded_lr == original_lr, (
            f"Scheduler LR not preserved: {original_lr} vs {loaded_lr}"
        )
        assert loaded_last_epoch == original_last_epoch, (
            f"Scheduler last_epoch not preserved: {original_last_epoch} vs {loaded_last_epoch}"
        )


# ---------------------------------------------------------------------------
# Property 28: Seed determinism
# ---------------------------------------------------------------------------


class TestProperty28SeedDeterminism:
    """**Validates: Requirements 18.4**

    Property 28: For any random seed value, two independent executions of the
    same computation (model initialization, data sampling, training step) with
    the same seed SHALL produce bit-identical results.
    """

    @given(seed=seed_st)
    @settings(max_examples=20, deadline=30000)
    def test_seed_determinism_model_init(self, seed):
        """Two model initializations with same seed produce identical weights."""
        set_seed(seed)
        model1 = SimpleModel()
        weights1 = {k: v.clone() for k, v in model1.state_dict().items()}

        set_seed(seed)
        model2 = SimpleModel()
        weights2 = model2.state_dict()

        for key in weights1:
            assert torch.equal(weights1[key], weights2[key]), (
                f"Model init not deterministic for seed {seed} at key '{key}'"
            )

    @given(seed=seed_st)
    @settings(max_examples=20, deadline=30000)
    def test_seed_determinism_random_tensors(self, seed):
        """Same seed produces bit-identical random tensors across all generators."""
        # First execution
        set_seed(seed)
        torch_tensor1 = torch.randn(10, 10)
        np_array1 = np.random.rand(10, 10).copy()
        py_values1 = [random.random() for _ in range(10)]

        # Second execution
        set_seed(seed)
        torch_tensor2 = torch.randn(10, 10)
        np_array2 = np.random.rand(10, 10)
        py_values2 = [random.random() for _ in range(10)]

        assert torch.equal(torch_tensor1, torch_tensor2), (
            f"PyTorch random not deterministic for seed {seed}"
        )
        np.testing.assert_array_equal(np_array1, np_array2)
        assert py_values1 == py_values2, (
            f"Python random not deterministic for seed {seed}"
        )

    @given(seed=seed_st)
    @settings(max_examples=15, deadline=30000)
    def test_seed_determinism_training_step(self, seed):
        """Same seed produces bit-identical training step results."""
        # First execution
        set_seed(seed)
        model1, optimizer1, _ = build_training_components()
        input_data = torch.randn(4, 16)
        target_data = torch.randn(4, 4)
        loss1, output1 = do_training_step(model1, optimizer1, input_data, target_data)

        # Second execution with same seed
        set_seed(seed)
        model2, optimizer2, _ = build_training_components()
        input_data2 = torch.randn(4, 16)
        target_data2 = torch.randn(4, 4)
        loss2, output2 = do_training_step(model2, optimizer2, input_data2, target_data2)

        assert loss1 == loss2, (
            f"Training loss not deterministic for seed {seed}: {loss1} vs {loss2}"
        )
        assert torch.equal(output1, output2), (
            f"Training output not deterministic for seed {seed}"
        )

    @given(seed=seed_st)
    @settings(max_examples=15, deadline=30000)
    def test_seed_determinism_multi_step_training(self, seed):
        """Same seed produces bit-identical results across multiple training steps."""
        num_steps = 5

        # First execution
        set_seed(seed)
        model1, optimizer1, scheduler1 = build_training_components()
        losses1 = []
        for _ in range(num_steps):
            input_data = torch.randn(4, 16)
            target_data = torch.randn(4, 4)
            loss, _ = do_training_step(model1, optimizer1, input_data, target_data)
            losses1.append(loss)
            scheduler1.step()

        final_weights1 = {k: v.clone() for k, v in model1.state_dict().items()}

        # Second execution with same seed
        set_seed(seed)
        model2, optimizer2, scheduler2 = build_training_components()
        losses2 = []
        for _ in range(num_steps):
            input_data = torch.randn(4, 16)
            target_data = torch.randn(4, 4)
            loss, _ = do_training_step(model2, optimizer2, input_data, target_data)
            losses2.append(loss)
            scheduler2.step()

        final_weights2 = model2.state_dict()

        # All losses must be identical
        for i, (l1, l2) in enumerate(zip(losses1, losses2)):
            assert l1 == l2, (
                f"Loss at step {i} not deterministic for seed {seed}: {l1} vs {l2}"
            )

        # Final weights must be identical
        for key in final_weights1:
            assert torch.equal(final_weights1[key], final_weights2[key]), (
                f"Final weights not deterministic for seed {seed} at key '{key}'"
            )

    @given(seed1=seed_st, seed2=seed_st)
    @settings(max_examples=15, deadline=30000)
    def test_different_seeds_produce_different_results(self, seed1, seed2):
        """Different seeds produce different random outputs (non-collision)."""
        assume(seed1 != seed2)

        set_seed(seed1)
        tensor1 = torch.randn(10, 10)

        set_seed(seed2)
        tensor2 = torch.randn(10, 10)

        # With overwhelming probability, different seeds produce different tensors
        assert not torch.equal(tensor1, tensor2), (
            f"Different seeds {seed1} and {seed2} produced identical tensors (extremely unlikely)"
        )
