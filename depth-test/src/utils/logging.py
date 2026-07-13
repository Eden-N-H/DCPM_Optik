"""Experiment logging with JSON lines and TensorBoard support."""

from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import logging
import math
import time

import torch
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


class ExperimentLogger:
    """JSON lines + TensorBoard logging.

    Logs scalar metrics and images to both a JSON lines file (for easy parsing)
    and TensorBoard (for interactive visualization). Also supports diagnostic
    logging for gradient norms and loss history, NaN detection with diagnostic
    checkpoint saving, and a final training summary.

    Args:
        log_dir: Directory for JSON lines log files and summaries.
        tb_dir: Directory for TensorBoard logs. Defaults to log_dir/tensorboard.
        tb_log_interval: Interval (in steps) for TensorBoard image/prediction
            logging. Default is 100.
    """

    def __init__(
        self,
        log_dir: Path,
        tb_dir: Optional[Path] = None,
        tb_log_interval: int = 100,
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.tb_dir = Path(tb_dir) if tb_dir else self.log_dir / "tensorboard"
        self.tb_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "training.jsonl"
        self.writer = SummaryWriter(log_dir=str(self.tb_dir))
        self.start_time = time.time()
        self.tb_log_interval = tb_log_interval

        # Rolling buffers for NaN diagnostics (Requirement 20.3)
        self._recent_grad_norms: List[float] = []
        self._recent_losses: List[float] = []
        self._buffer_size = 10

    def log_scalars(self, metrics: Dict[str, float], step: int, epoch: Optional[int] = None) -> None:
        """Log scalar metrics to console, JSON lines, and TensorBoard.

        Each metric is written as an individual scalar to TensorBoard and all metrics
        are recorded together as a single JSON line with timestamp and step metadata.
        Metrics are also printed to console for real-time monitoring.

        Args:
            metrics: Dictionary of metric names to float values.
            step: The global training step (used for x-axis in TensorBoard).
            epoch: Optional epoch number for console display.
        """
        # Write to TensorBoard
        for name, value in metrics.items():
            self.writer.add_scalar(name, value, global_step=step)

        # Write to JSON lines file
        record: Dict[str, Any] = {
            "type": "scalars",
            "step": step,
            "timestamp": time.time() - self.start_time,
            "metrics": metrics,
        }
        if epoch is not None:
            record["epoch"] = epoch
        with open(self.log_file, "a") as f:
            f.write(json.dumps(record) + "\n")

        # Log to console (Requirement 20.1)
        parts = []
        if epoch is not None:
            parts.append(f"Epoch {epoch}")
        parts.append(f"Step {step}")
        metric_strs = [f"{k}: {v:.4f}" for k, v in metrics.items()]
        parts.append(", ".join(metric_strs))
        console_msg = " | ".join(parts)
        logger.info(console_msg)
        print(console_msg)

    def log_images(self, images: Dict[str, torch.Tensor], step: int) -> None:
        """Log image tensors to TensorBoard.

        This method respects the tb_log_interval — images are only logged
        when step is a multiple of the configured interval (default 100).

        Args:
            images: Dictionary of tag names to image tensors.
                    Tensors should be [C, H, W] (single image) or [B, C, H, W] (batch).
                    Values are expected in [0, 1] range for float tensors.
            step: The global training step.
        """
        if step % self.tb_log_interval != 0:
            return

        for tag, tensor in images.items():
            if tensor.dim() == 4:
                # Batch of images: log as a grid
                self.writer.add_images(tag, tensor, global_step=step)
            elif tensor.dim() == 3:
                # Single image [C, H, W]
                self.writer.add_image(tag, tensor, global_step=step)
            elif tensor.dim() == 2:
                # Grayscale [H, W] -> add channel dim
                self.writer.add_image(tag, tensor.unsqueeze(0), global_step=step)

    def should_log_images(self, step: int) -> bool:
        """Check if image logging should occur at this step.

        Returns True when step is a multiple of tb_log_interval.
        This lets callers avoid expensive image preparation when logging
        won't occur.

        Args:
            step: The current global training step.

        Returns:
            True if images should be logged at this step.
        """
        return step % self.tb_log_interval == 0

    def log_diagnostic(self, grad_norms: List[float], losses: List[float]) -> None:
        """Log diagnostic info (gradient norms, losses) for debugging.

        Writes a diagnostic record to the JSON lines file with statistics
        about gradient norms and recent losses for monitoring training health.

        Args:
            grad_norms: List of gradient norm values (one per parameter group or layer).
            losses: List of recent loss values.
        """
        diagnostic: Dict[str, Any] = {
            "type": "diagnostic",
            "timestamp": time.time() - self.start_time,
            "grad_norms": {
                "values": grad_norms,
                "mean": sum(grad_norms) / len(grad_norms) if grad_norms else 0.0,
                "max": max(grad_norms) if grad_norms else 0.0,
                "min": min(grad_norms) if grad_norms else 0.0,
            },
            "losses": {
                "values": losses,
                "mean": sum(losses) / len(losses) if losses else 0.0,
                "latest": losses[-1] if losses else 0.0,
            },
        }
        with open(self.log_file, "a") as f:
            f.write(json.dumps(diagnostic) + "\n")

    def track_grad_norm(self, grad_norm: float) -> None:
        """Track a gradient norm value in the rolling buffer.

        Maintains the last 10 gradient norms for NaN diagnostics.

        Args:
            grad_norm: The gradient norm value for the current step.
        """
        self._recent_grad_norms.append(grad_norm)
        if len(self._recent_grad_norms) > self._buffer_size:
            self._recent_grad_norms.pop(0)

    def track_loss(self, loss_value: float) -> None:
        """Track a loss value in the rolling buffer.

        Maintains the last 10 loss values for NaN diagnostics.

        Args:
            loss_value: The loss value for the current step.
        """
        self._recent_losses.append(loss_value)
        if len(self._recent_losses) > self._buffer_size:
            self._recent_losses.pop(0)

    def check_nan(self, loss_value: float) -> bool:
        """Check if a loss value is NaN or infinite.

        Args:
            loss_value: The loss value to check.

        Returns:
            True if the loss is NaN or infinite, False otherwise.
        """
        return math.isnan(loss_value) or math.isinf(loss_value)

    def handle_nan_loss(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        step: int,
        extra_state: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Handle NaN loss by logging diagnostics and saving a diagnostic checkpoint.

        When a NaN or infinite loss is detected, this method:
        1. Logs the last 10 gradient norms and loss values to the JSON lines file
        2. Saves a diagnostic checkpoint with model weights, optimizer state, and
           the buffered diagnostic data
        3. Prints a clear error message to the console

        Args:
            model: The model whose state to save.
            optimizer: The optimizer whose state to save.
            epoch: The current epoch number.
            step: The current training step.
            extra_state: Optional additional state to include in the checkpoint.

        Returns:
            Path to the saved diagnostic checkpoint.
        """
        # Log diagnostic info
        self.log_diagnostic(self._recent_grad_norms, self._recent_losses)

        error_msg = (
            f"NaN/Inf loss detected at epoch {epoch}, step {step}. "
            f"Last 10 grad norms: {self._recent_grad_norms}. "
            f"Last 10 losses: {self._recent_losses}. "
            f"Halting training and saving diagnostic checkpoint."
        )
        logger.error(error_msg)
        print(f"ERROR: {error_msg}")

        # Write NaN event to log file
        nan_record: Dict[str, Any] = {
            "type": "nan_detected",
            "timestamp": time.time() - self.start_time,
            "epoch": epoch,
            "step": step,
            "last_grad_norms": list(self._recent_grad_norms),
            "last_losses": list(self._recent_losses),
        }
        with open(self.log_file, "a") as f:
            f.write(json.dumps(nan_record) + "\n")

        # Save diagnostic checkpoint
        checkpoint_path = self.log_dir / f"diagnostic_checkpoint_epoch{epoch}_step{step}.pt"
        checkpoint_data: Dict[str, Any] = {
            "epoch": epoch,
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "last_grad_norms": list(self._recent_grad_norms),
            "last_losses": list(self._recent_losses),
        }
        if extra_state:
            checkpoint_data.update(extra_state)
        torch.save(checkpoint_data, checkpoint_path)

        logger.info(f"Diagnostic checkpoint saved to {checkpoint_path}")
        return checkpoint_path

    def save_summary(
        self,
        best_metrics: Dict[str, float],
        best_epochs: Dict[str, int],
    ) -> Path:
        """Save final training summary JSON.

        Creates a summary.json file in the log directory containing the best
        metrics achieved, the epochs at which they occurred, and total training time.

        Args:
            best_metrics: Dictionary of metric names to their best values.
            best_epochs: Dictionary of metric names to the epoch where best was achieved.

        Returns:
            Path to the saved summary file.
        """
        summary: Dict[str, Any] = {
            "total_training_time_seconds": time.time() - self.start_time,
            "best_metrics": best_metrics,
            "best_epochs": best_epochs,
            "log_dir": str(self.log_dir),
        }
        summary_path = self.log_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        # Also log to console
        logger.info(f"Training summary saved to {summary_path}")
        print(f"Training complete. Summary saved to {summary_path}")
        print(f"  Total time: {summary['total_training_time_seconds']:.1f}s")
        for metric, value in best_metrics.items():
            epoch = best_epochs.get(metric, "?")
            print(f"  Best {metric}: {value:.4f} (epoch {epoch})")

        return summary_path

    def close(self) -> None:
        """Close TensorBoard writer and flush pending events."""
        self.writer.flush()
        self.writer.close()
