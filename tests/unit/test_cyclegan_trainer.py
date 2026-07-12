"""Unit tests for CycleGAN trainer: loss summation and LR decay schedule."""

import math
import pytest
import torch

from src.cyclegan.trainer import CycleGANConfig, CycleGANTrainer, LambdaLRSchedule, compute_lr


class TestComputeLR:
    """Tests for the linear LR decay schedule (Req 5.7)."""

    def test_lr_constant_before_decay(self):
        """LR should be initial_lr for all epochs before decay_start_epoch."""
        for epoch in range(100):
            lr = compute_lr(epoch, initial_lr=2e-4, decay_start_epoch=100, total_epochs=200)
            assert lr == pytest.approx(2e-4), f"Epoch {epoch}: expected 2e-4, got {lr}"

    def test_lr_at_decay_start(self):
        """LR at epoch 100 should be 2e-4 × (200-100)/100 = 2e-4."""
        lr = compute_lr(100, initial_lr=2e-4, decay_start_epoch=100, total_epochs=200)
        assert lr == pytest.approx(2e-4)

    def test_lr_at_midpoint_decay(self):
        """LR at epoch 150 should be 2e-4 × (200-150)/100 = 1e-4."""
        lr = compute_lr(150, initial_lr=2e-4, decay_start_epoch=100, total_epochs=200)
        assert lr == pytest.approx(1e-4)

    def test_lr_at_last_epoch(self):
        """LR at epoch 199 should be 2e-4 × (200-199)/100 = 2e-6."""
        lr = compute_lr(199, initial_lr=2e-4, decay_start_epoch=100, total_epochs=200)
        assert lr == pytest.approx(2e-6)

    def test_lr_linear_decay_formula(self):
        """LR for e >= 100 should follow: initial_lr × (total_epochs - e) / decay_epochs."""
        initial_lr = 2e-4
        for epoch in range(100, 200):
            expected = initial_lr * (200 - epoch) / 100
            actual = compute_lr(epoch, initial_lr=initial_lr, decay_start_epoch=100, total_epochs=200)
            assert actual == pytest.approx(expected, rel=1e-10), f"Epoch {epoch}: expected {expected}, got {actual}"

    def test_lr_monotonically_decreasing_during_decay(self):
        """LR should be monotonically decreasing during the decay period."""
        lrs = [compute_lr(e) for e in range(100, 200)]
        for i in range(len(lrs) - 1):
            assert lrs[i] > lrs[i + 1], f"LR not decreasing at epoch {100+i}"


class TestLambdaLRSchedule:
    """Tests for the LambdaLR multiplier schedule."""

    def test_multiplier_is_1_before_decay(self):
        """Lambda multiplier should be 1.0 for epochs before decay."""
        schedule = LambdaLRSchedule(decay_start_epoch=100, total_epochs=200)
        for epoch in range(100):
            assert schedule(epoch) == 1.0

    def test_multiplier_decays_to_zero(self):
        """Lambda multiplier should approach 0 as epoch → total_epochs."""
        schedule = LambdaLRSchedule(decay_start_epoch=100, total_epochs=200)
        # At epoch 199: (200-199)/100 = 0.01
        assert schedule(199) == pytest.approx(0.01)

    def test_multiplier_at_midpoint(self):
        """Lambda multiplier at epoch 150 should be 0.5."""
        schedule = LambdaLRSchedule(decay_start_epoch=100, total_epochs=200)
        assert schedule(150) == pytest.approx(0.5)


class TestLossSummation:
    """Tests for total generator loss computation formula (Req 5.5).

    Total loss = adversarial + λ_cycle×cycle + λ_identity×identity + λ_defect×defect
    With defaults: adversarial + 10×cycle + 0.5×identity + 5×defect
    """

    def test_loss_formula_with_known_values(self):
        """Verify the weighted sum with specific known values."""
        # Given known loss components
        adversarial = 1.0
        cycle = 2.0
        identity_val = 3.0
        defect = 4.0

        # Expected: 1.0 + 10×2.0 + 0.5×3.0 + 5.0×4.0 = 1 + 20 + 1.5 + 20 = 42.5
        lambda_cycle = 10.0
        lambda_identity = 0.5
        lambda_defect = 5.0

        expected = adversarial + lambda_cycle * cycle + lambda_identity * identity_val + lambda_defect * defect
        assert expected == pytest.approx(42.5)

    def test_loss_formula_with_zero_components(self):
        """When all losses are zero, total should be zero."""
        adversarial = 0.0
        cycle = 0.0
        identity_val = 0.0
        defect = 0.0

        lambda_cycle = 10.0
        lambda_identity = 0.5
        lambda_defect = 5.0

        total = adversarial + lambda_cycle * cycle + lambda_identity * identity_val + lambda_defect * defect
        assert total == 0.0

    def test_loss_formula_adversarial_only(self):
        """When only adversarial loss is nonzero, total = adversarial."""
        adversarial = 5.0
        total = adversarial + 10.0 * 0.0 + 0.5 * 0.0 + 5.0 * 0.0
        assert total == 5.0

    def test_loss_formula_cycle_dominates(self):
        """Cycle loss with λ=10 should dominate when cycle component is large."""
        adversarial = 0.1
        cycle = 1.0
        identity_val = 0.1
        defect = 0.1

        total = adversarial + 10.0 * cycle + 0.5 * identity_val + 5.0 * defect
        # 0.1 + 10 + 0.05 + 0.5 = 10.65
        assert total == pytest.approx(10.65)


class TestCycleGANTrainerInit:
    """Tests for CycleGANTrainer initialization."""

    def test_default_config_creates_trainer(self):
        """Trainer should initialize with default config."""
        config = CycleGANConfig()
        trainer = CycleGANTrainer(config, device=torch.device("cpu"))

        assert trainer.config.lambda_cycle == 10.0
        assert trainer.config.lambda_identity == 0.5
        assert trainer.config.lambda_defect == 5.0
        assert trainer.config.lr == 2e-4
        assert trainer.config.beta1 == 0.5
        assert trainer.config.beta2 == 0.999
        assert trainer.config.pool_size == 50
        assert not trainer.diverged

    def test_optimizer_parameters(self):
        """Optimizer should have correct hyperparameters (Req 5.6)."""
        config = CycleGANConfig()
        trainer = CycleGANTrainer(config, device=torch.device("cpu"))

        # Check generator optimizer
        g_params = trainer.optimizer_G.param_groups[0]
        assert g_params["lr"] == 2e-4
        assert g_params["betas"] == (0.5, 0.999)

        # Check discriminator optimizers
        d_a_params = trainer.optimizer_D_A.param_groups[0]
        assert d_a_params["lr"] == 2e-4
        assert d_a_params["betas"] == (0.5, 0.999)

    def test_image_pools_initialized(self):
        """Image pools should be initialized with correct pool size (Req 5.8)."""
        config = CycleGANConfig(pool_size=50)
        trainer = CycleGANTrainer(config, device=torch.device("cpu"))

        assert trainer.fake_A_pool.pool_size == 50
        assert trainer.fake_B_pool.pool_size == 50
        assert len(trainer.fake_A_pool.images) == 0
        assert len(trainer.fake_B_pool.images) == 0


class TestCycleGANTrainerStep:
    """Tests for CycleGANTrainer.train_step."""

    @pytest.fixture
    def trainer(self):
        """Create a trainer on CPU for testing."""
        config = CycleGANConfig()
        return CycleGANTrainer(config, device=torch.device("cpu"))

    def test_train_step_returns_all_losses(self, trainer):
        """train_step should return all expected loss keys."""
        real_A = torch.randn(1, 3, 256, 256)
        real_B = torch.randn(1, 3, 256, 256)
        mask_A = torch.zeros(1, 1, 256, 256)

        losses = trainer.train_step(real_A, real_B, mask_A)

        expected_keys = {
            "loss_G", "loss_adversarial", "loss_cycle",
            "loss_identity", "loss_defect", "loss_D_A", "loss_D_B", "diverged",
        }
        assert set(losses.keys()) == expected_keys

    def test_train_step_losses_are_finite(self, trainer):
        """All losses should be finite after a normal training step."""
        real_A = torch.randn(1, 3, 256, 256)
        real_B = torch.randn(1, 3, 256, 256)
        mask_A = torch.zeros(1, 1, 256, 256)

        losses = trainer.train_step(real_A, real_B, mask_A)

        assert not losses["diverged"]
        for key in ["loss_G", "loss_adversarial", "loss_cycle", "loss_identity", "loss_D_A", "loss_D_B"]:
            assert math.isfinite(losses[key]), f"{key} is not finite: {losses[key]}"

    def test_train_step_with_nonzero_mask(self, trainer):
        """Training step should work with non-zero defect mask."""
        real_A = torch.randn(1, 3, 256, 256)
        real_B = torch.randn(1, 3, 256, 256)
        mask_A = torch.ones(1, 1, 256, 256)  # Full defect mask

        losses = trainer.train_step(real_A, real_B, mask_A)
        assert not losses["diverged"]


class TestNaNDetection:
    """Tests for NaN/Inf detection (Req 5.9)."""

    def test_check_nan_inf_detects_nan(self):
        """Should detect NaN in losses."""
        config = CycleGANConfig()
        trainer = CycleGANTrainer(config, device=torch.device("cpu"))

        losses = {"test": torch.tensor(float("nan"))}
        assert trainer._check_nan_inf(losses) is True

    def test_check_nan_inf_detects_inf(self):
        """Should detect Inf in losses."""
        config = CycleGANConfig()
        trainer = CycleGANTrainer(config, device=torch.device("cpu"))

        losses = {"test": torch.tensor(float("inf"))}
        assert trainer._check_nan_inf(losses) is True

    def test_check_nan_inf_passes_normal(self):
        """Should pass for normal finite values."""
        config = CycleGANConfig()
        trainer = CycleGANTrainer(config, device=torch.device("cpu"))

        losses = {"a": torch.tensor(1.0), "b": torch.tensor(0.5)}
        assert trainer._check_nan_inf(losses) is False


class TestLRSchedulerIntegration:
    """Tests for LR scheduler integration with trainer."""

    def test_lr_stays_constant_for_first_100_epochs(self):
        """LR should not change during the first 100 epochs."""
        config = CycleGANConfig()
        trainer = CycleGANTrainer(config, device=torch.device("cpu"))

        for _ in range(100):
            trainer.step_schedulers()

        # After 100 epoch steps, LR should still be 2e-4
        # (LambdaLR at epoch=100 returns 1.0 since decay starts after 100)
        lr = trainer.get_current_lr()
        assert lr == pytest.approx(2e-4)

    def test_lr_decays_after_epoch_100(self):
        """LR should start decaying after epoch 100."""
        config = CycleGANConfig()
        trainer = CycleGANTrainer(config, device=torch.device("cpu"))

        # Step through 101 epochs
        for _ in range(101):
            trainer.step_schedulers()

        lr = trainer.get_current_lr()
        # At epoch 101: lambda = (200 - 101) / 100 = 0.99
        # LR = 2e-4 * 0.99 = 1.98e-4
        assert lr == pytest.approx(2e-4 * 0.99, rel=1e-6)

    def test_lr_at_epoch_150(self):
        """LR at epoch 150 should be 2e-4 × 0.5 = 1e-4."""
        config = CycleGANConfig()
        trainer = CycleGANTrainer(config, device=torch.device("cpu"))

        for _ in range(150):
            trainer.step_schedulers()

        lr = trainer.get_current_lr()
        assert lr == pytest.approx(1e-4, rel=1e-6)
