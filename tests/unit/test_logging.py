"""Unit tests for ExperimentLogger."""

import json
import math
import time
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from src.utils.logging import ExperimentLogger


@pytest.fixture
def log_dir(tmp_path):
    """Provide a temporary log directory."""
    return tmp_path / "logs"


@pytest.fixture
def logger_instance(log_dir):
    """Create an ExperimentLogger instance for testing."""
    exp_logger = ExperimentLogger(log_dir=log_dir)
    yield exp_logger
    exp_logger.close()


class TestExperimentLoggerInit:
    """Tests for ExperimentLogger initialization."""

    def test_creates_log_directory(self, tmp_path):
        """Log directory is created if it doesn't exist."""
        log_dir = tmp_path / "new_logs"
        exp_logger = ExperimentLogger(log_dir=log_dir)
        assert log_dir.exists()
        exp_logger.close()

    def test_creates_tensorboard_directory(self, tmp_path):
        """TensorBoard directory is created."""
        log_dir = tmp_path / "logs"
        exp_logger = ExperimentLogger(log_dir=log_dir)
        assert (log_dir / "tensorboard").exists()
        exp_logger.close()

    def test_custom_tb_dir(self, tmp_path):
        """Custom TensorBoard directory is used when specified."""
        log_dir = tmp_path / "logs"
        tb_dir = tmp_path / "custom_tb"
        exp_logger = ExperimentLogger(log_dir=log_dir, tb_dir=tb_dir)
        assert tb_dir.exists()
        exp_logger.close()

    def test_default_log_interval(self, log_dir):
        """Default TensorBoard log interval is 100."""
        exp_logger = ExperimentLogger(log_dir=log_dir)
        assert exp_logger.tb_log_interval == 100
        exp_logger.close()

    def test_custom_log_interval(self, log_dir):
        """Custom TensorBoard log interval is respected."""
        exp_logger = ExperimentLogger(log_dir=log_dir, tb_log_interval=50)
        assert exp_logger.tb_log_interval == 50
        exp_logger.close()


class TestLogScalars:
    """Tests for scalar logging (Requirement 20.1)."""

    def test_writes_jsonl_record(self, logger_instance, log_dir):
        """Scalar metrics are written to the JSON lines file."""
        metrics = {"loss": 0.5, "mIoU": 0.75}
        logger_instance.log_scalars(metrics, step=10)

        lines = (log_dir / "training.jsonl").read_text().strip().split("\n")
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["type"] == "scalars"
        assert record["step"] == 10
        assert record["metrics"]["loss"] == 0.5
        assert record["metrics"]["mIoU"] == 0.75
        assert "timestamp" in record

    def test_includes_epoch_in_record(self, logger_instance, log_dir):
        """Epoch number is included in record when provided."""
        logger_instance.log_scalars({"loss": 0.3}, step=100, epoch=5)

        lines = (log_dir / "training.jsonl").read_text().strip().split("\n")
        record = json.loads(lines[0])
        assert record["epoch"] == 5

    def test_multiple_records_appended(self, logger_instance, log_dir):
        """Multiple log calls append separate records."""
        logger_instance.log_scalars({"loss": 0.5}, step=1)
        logger_instance.log_scalars({"loss": 0.3}, step=2)
        logger_instance.log_scalars({"loss": 0.1}, step=3)

        lines = (log_dir / "training.jsonl").read_text().strip().split("\n")
        assert len(lines) == 3

    def test_console_output(self, logger_instance, capsys):
        """Metrics are printed to console."""
        logger_instance.log_scalars({"loss": 0.5}, step=10, epoch=1)
        captured = capsys.readouterr()
        assert "Epoch 1" in captured.out
        assert "Step 10" in captured.out
        assert "loss" in captured.out


class TestLogImages:
    """Tests for image logging (Requirement 20.2)."""

    def test_logs_at_interval(self, log_dir):
        """Images are only logged at tb_log_interval multiples."""
        exp_logger = ExperimentLogger(log_dir=log_dir, tb_log_interval=100)
        img = torch.rand(3, 64, 64)

        # Step 100 should log
        exp_logger.log_images({"test": img}, step=100)
        # Step 50 should not log (no error, just skip)
        exp_logger.log_images({"test": img}, step=50)
        exp_logger.close()

    def test_should_log_images(self, log_dir):
        """should_log_images returns correct values."""
        exp_logger = ExperimentLogger(log_dir=log_dir, tb_log_interval=100)
        assert exp_logger.should_log_images(0) is True
        assert exp_logger.should_log_images(100) is True
        assert exp_logger.should_log_images(200) is True
        assert exp_logger.should_log_images(50) is False
        assert exp_logger.should_log_images(99) is False
        exp_logger.close()

    def test_handles_4d_tensor(self, logger_instance):
        """4D tensors (batch of images) are handled."""
        img_batch = torch.rand(4, 3, 64, 64)
        # Should not raise
        logger_instance.log_images({"batch": img_batch}, step=0)

    def test_handles_3d_tensor(self, logger_instance):
        """3D tensors (single image) are handled."""
        img = torch.rand(3, 64, 64)
        logger_instance.log_images({"single": img}, step=0)

    def test_handles_2d_tensor(self, logger_instance):
        """2D tensors (grayscale) are handled by adding channel dim."""
        img = torch.rand(64, 64)
        logger_instance.log_images({"gray": img}, step=0)


class TestLogDiagnostic:
    """Tests for diagnostic logging (Requirement 20.3)."""

    def test_writes_diagnostic_record(self, logger_instance, log_dir):
        """Diagnostic data is written to JSON lines file."""
        grad_norms = [1.0, 2.0, 3.0]
        losses = [0.5, 0.4, 0.3]
        logger_instance.log_diagnostic(grad_norms, losses)

        lines = (log_dir / "training.jsonl").read_text().strip().split("\n")
        record = json.loads(lines[0])
        assert record["type"] == "diagnostic"
        assert record["grad_norms"]["values"] == [1.0, 2.0, 3.0]
        assert record["grad_norms"]["mean"] == 2.0
        assert record["grad_norms"]["max"] == 3.0
        assert record["grad_norms"]["min"] == 1.0
        assert record["losses"]["values"] == [0.5, 0.4, 0.3]
        assert record["losses"]["latest"] == 0.3

    def test_handles_empty_lists(self, logger_instance, log_dir):
        """Empty grad_norms and losses are handled gracefully."""
        logger_instance.log_diagnostic([], [])
        lines = (log_dir / "training.jsonl").read_text().strip().split("\n")
        record = json.loads(lines[0])
        assert record["grad_norms"]["mean"] == 0.0
        assert record["losses"]["mean"] == 0.0


class TestNaNHandling:
    """Tests for NaN detection and diagnostic checkpointing (Requirement 20.3)."""

    def test_check_nan_detects_nan(self, logger_instance):
        """check_nan returns True for NaN values."""
        assert logger_instance.check_nan(float("nan")) is True

    def test_check_nan_detects_inf(self, logger_instance):
        """check_nan returns True for infinite values."""
        assert logger_instance.check_nan(float("inf")) is True
        assert logger_instance.check_nan(float("-inf")) is True

    def test_check_nan_normal_value(self, logger_instance):
        """check_nan returns False for normal values."""
        assert logger_instance.check_nan(0.5) is False
        assert logger_instance.check_nan(0.0) is False
        assert logger_instance.check_nan(-1.0) is False

    def test_track_grad_norm_buffer(self, logger_instance):
        """Gradient norm buffer maintains last 10 values."""
        for i in range(15):
            logger_instance.track_grad_norm(float(i))
        assert len(logger_instance._recent_grad_norms) == 10
        assert logger_instance._recent_grad_norms[0] == 5.0
        assert logger_instance._recent_grad_norms[-1] == 14.0

    def test_track_loss_buffer(self, logger_instance):
        """Loss buffer maintains last 10 values."""
        for i in range(15):
            logger_instance.track_loss(float(i) * 0.1)
        assert len(logger_instance._recent_losses) == 10
        assert logger_instance._recent_losses[0] == pytest.approx(0.5)
        assert logger_instance._recent_losses[-1] == pytest.approx(1.4)

    def test_handle_nan_loss_saves_checkpoint(self, logger_instance, log_dir):
        """handle_nan_loss saves a diagnostic checkpoint."""
        model = nn.Linear(10, 5)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Track some values before NaN
        for i in range(5):
            logger_instance.track_grad_norm(float(i))
            logger_instance.track_loss(float(i) * 0.1)

        checkpoint_path = logger_instance.handle_nan_loss(
            model=model,
            optimizer=optimizer,
            epoch=10,
            step=500,
        )

        assert checkpoint_path.exists()
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        assert checkpoint["epoch"] == 10
        assert checkpoint["step"] == 500
        assert "model_state_dict" in checkpoint
        assert "optimizer_state_dict" in checkpoint
        assert checkpoint["last_grad_norms"] == [0.0, 1.0, 2.0, 3.0, 4.0]
        assert checkpoint["last_losses"] == pytest.approx([0.0, 0.1, 0.2, 0.3, 0.4])

    def test_handle_nan_loss_writes_log_record(self, logger_instance, log_dir):
        """handle_nan_loss writes NaN event to JSON lines file."""
        model = nn.Linear(10, 5)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        logger_instance.track_loss(0.5)
        logger_instance.handle_nan_loss(model, optimizer, epoch=3, step=100)

        lines = (log_dir / "training.jsonl").read_text().strip().split("\n")
        # Should have diagnostic record + nan_detected record
        nan_records = [json.loads(l) for l in lines if "nan_detected" in l]
        assert len(nan_records) == 1
        assert nan_records[0]["type"] == "nan_detected"
        assert nan_records[0]["epoch"] == 3
        assert nan_records[0]["step"] == 100

    def test_handle_nan_loss_with_extra_state(self, logger_instance, log_dir):
        """handle_nan_loss includes extra state in checkpoint."""
        model = nn.Linear(10, 5)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        checkpoint_path = logger_instance.handle_nan_loss(
            model=model,
            optimizer=optimizer,
            epoch=5,
            step=200,
            extra_state={"scheduler_state": {"step_count": 200}},
        )

        checkpoint = torch.load(checkpoint_path, weights_only=False)
        assert "scheduler_state" in checkpoint


class TestSaveSummary:
    """Tests for final summary (Requirement 20.4)."""

    def test_saves_summary_json(self, logger_instance, log_dir):
        """Summary JSON is saved with correct content."""
        best_metrics = {"mIoU": 0.85, "depth_rmse": 0.12}
        best_epochs = {"mIoU": 45, "depth_rmse": 52}

        summary_path = logger_instance.save_summary(best_metrics, best_epochs)

        assert summary_path.exists()
        with open(summary_path) as f:
            summary = json.load(f)
        assert summary["best_metrics"]["mIoU"] == 0.85
        assert summary["best_metrics"]["depth_rmse"] == 0.12
        assert summary["best_epochs"]["mIoU"] == 45
        assert summary["best_epochs"]["depth_rmse"] == 52
        assert "total_training_time_seconds" in summary
        assert summary["total_training_time_seconds"] >= 0

    def test_summary_includes_training_time(self, log_dir):
        """Summary correctly measures total training time."""
        exp_logger = ExperimentLogger(log_dir=log_dir)
        time.sleep(0.05)  # Small delay to ensure non-zero time
        summary_path = exp_logger.save_summary({}, {})

        with open(summary_path) as f:
            summary = json.load(f)
        assert summary["total_training_time_seconds"] >= 0.04
        exp_logger.close()

    def test_summary_path_returned(self, logger_instance, log_dir):
        """save_summary returns the path to the summary file."""
        path = logger_instance.save_summary({}, {})
        assert path == log_dir / "summary.json"


class TestClose:
    """Tests for logger cleanup."""

    def test_close_without_error(self, log_dir):
        """Logger can be closed cleanly."""
        exp_logger = ExperimentLogger(log_dir=log_dir)
        exp_logger.log_scalars({"test": 1.0}, step=0)
        exp_logger.close()  # Should not raise
