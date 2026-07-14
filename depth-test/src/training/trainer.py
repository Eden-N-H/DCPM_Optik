"""Multi-task training loop with AMP, gradient clipping, early stopping, and Relay logic."""
import logging
import math
import signal
import sys
import threading
from pathlib import Path
from typing import Dict, Optional, Any

import torch
import torch.nn as nn
import torch.cuda.amp
from torch.utils.data import DataLoader
import requests

from src.model import MultiTaskModel
from .losses import MultiTaskLoss
from .metrics import MetricsComputer
from .checkpoint import save_checkpoint, load_checkpoint

logger = logging.getLogger(__name__)


class MultiTaskTrainer:
    """Training loop for the multi-task road quality model.

    Features:
        - Automatic Mixed Precision (AMP)
        - Gradient clipping (max norm 1.0)
        - Adam optimizer (lr=1e-4, β1=0.9, β2=0.999, weight_decay=1e-5)
        - ReduceLROnPlateau scheduler (patience=10, factor=0.5)
        - Early stopping (patience=30)
        - NaN detection
        - Checkpoint saving
        - Preemption signal handling (SIGTERM)
        - Webhook Telemetry
    """

    def __init__(self, config: Dict[str, Any], model: MultiTaskModel,
                 train_loader: DataLoader, val_loader: DataLoader,
                 device: torch.device, output_dir: str = './checkpoints',
                 webhook_url: Optional[str] = None, worker_id: Optional[str] = None):
        self.config = config
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Relay Telemetry
        self.webhook_url = webhook_url
        self.worker_id = worker_id or "local"

        # Training config
        train_cfg = config.get('training', {})
        opt_cfg = train_cfg.get('optimizer', {})
        sched_cfg = train_cfg.get('scheduler', {})

        self.epochs = train_cfg.get('epochs', 200)
        self.use_amp = train_cfg.get('amp', True)
        self.grad_clip_norm = train_cfg.get('grad_clip_norm', 1.0)
        self.early_stopping_patience = train_cfg.get('early_stopping_patience', 30)

        # Loss
        loss_weights = train_cfg.get('loss_weights', {})
        self.criterion = MultiTaskLoss(
            seg_weight=loss_weights.get('segmentation', 1.5),
            depth_weight=loss_weights.get('depth', 1.0),
            camera_weight=loss_weights.get('camera', 0.3),
            adv_weight=loss_weights.get('adversarial', 0.1),
        ).to(device)

        # Optimizer
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=opt_cfg.get('lr', 1e-4),
            betas=(opt_cfg.get('beta1', 0.9), opt_cfg.get('beta2', 0.999)),
            weight_decay=opt_cfg.get('weight_decay', 1e-5),
        )

        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=sched_cfg.get('patience', 10),
            factor=sched_cfg.get('factor', 0.5)
        )

        # Safe AMP scaler resolving UserWarnings on non-CUDA nodes and deprecation warnings
        is_cuda_amp = self.use_amp and self.device.type == 'cuda'
        if hasattr(torch, 'amp') and hasattr(torch.amp, 'GradScaler'):
            self.scaler = torch.amp.GradScaler('cuda', enabled=is_cuda_amp)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=is_cuda_amp)

        # Metrics
        num_classes = config.get('model', {}).get('heads', {}).get(
            'segmentation', {}).get('num_classes', 8)
        self.metrics_computer = MetricsComputer(num_classes=num_classes)

        # Training state
        self.current_epoch = 0
        self.best_metric = float('inf')
        self.epochs_without_improvement = 0
        self.use_domain_adapter = config.get('domain_adaptation', {}).get('lambda_adv', 0) > 0

        # Logging config
        log_cfg = config.get('logging', {})
        self.console_log_interval = log_cfg.get('console_log_interval', 10)
        self.checkpoint_interval = log_cfg.get('checkpoint_interval', 5)
        
        # Preemption Handler
        self._interrupted = False
        self._setup_signal_handlers()

    def _setup_signal_handlers(self):
        """Bind SIGTERM and SIGINT to trigger a graceful shutdown."""
        def handler(signum, frame):
            logger.warning(f"Received signal {signum}. Commencing graceful shutdown...")
            self._interrupted = True
            
        try:
            signal.signal(signal.SIGTERM, handler)
            signal.signal(signal.SIGINT, handler)
        except ValueError:
            # Cannot register signals if not running in main thread
            pass

    def _send_telemetry(self, status: str, train_loss: float = 0.0, val_loss: float = 0.0):
        """Send asynchronous ping to the Orchestrator laptop."""
        if not self.webhook_url:
            return
            
        payload = {
            "worker_id": self.worker_id,
            "status": status,
            "epoch": self.current_epoch,
            "train_loss": train_loss,
            "val_loss": val_loss
        }
        
        def _post():
            try:
                requests.post(self.webhook_url, json=payload, timeout=5)
            except Exception:
                pass # Fire and forget, don't crash training if laptop sleeps
                
        threading.Thread(target=_post, daemon=True).start()

    def train(self, resume_from: Optional[str] = None) -> Dict[str, float]:
        """Run full training loop.

        Args:
            resume_from: Optional path to checkpoint to resume from

        Returns:
            Dict with final metrics
        """
        if resume_from:
            checkpoint = load_checkpoint(
                Path(resume_from), self.model, self.optimizer, self.scheduler, self.device
            )
            self.current_epoch = checkpoint.epoch + 1
            self.best_metric = checkpoint.best_metric
            logger.info(f"Resumed from epoch {checkpoint.epoch}, best_metric={checkpoint.best_metric:.4f}")

        self._send_telemetry(status="started")

        for epoch in range(self.current_epoch, self.epochs):
            self.current_epoch = epoch

            # Train one epoch
            train_loss = self._train_epoch(epoch)

            if self._interrupted:
                logger.info("Graceful shutdown triggered. Saving emergency checkpoint...")
                self._save_periodic_checkpoint(epoch, suffix="_interrupted")
                self._send_telemetry(status="interrupted", train_loss=train_loss)
                break

            # Check for NaN
            if self._check_nan(train_loss):
                logger.error(f"NaN detected in training loss at epoch {epoch}. Stopping.")
                self._send_telemetry(status="failed_nan", train_loss=train_loss)
                break

            # Validate
            val_metrics = self._validate_epoch(epoch)
            val_loss = val_metrics.get('val/total_loss', float('inf'))

            # Scheduler step
            self.scheduler.step(val_loss)

            # Check for improvement
            if val_loss < self.best_metric:
                self.best_metric = val_loss
                self.epochs_without_improvement = 0
                self._save_best_checkpoint(epoch)
                logger.info(f"New best metric: {val_loss:.4f} at epoch {epoch}")
            else:
                self.epochs_without_improvement += 1

            # Periodic checkpoint
            if (epoch + 1) % self.checkpoint_interval == 0:
                self._save_periodic_checkpoint(epoch)

            # Telemetry
            self._send_telemetry(status="running", train_loss=train_loss, val_loss=val_loss)

            # Early stopping
            if self.epochs_without_improvement >= self.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch} (no improvement for "
                            f"{self.early_stopping_patience} epochs)")
                break

            # Log epoch summary
            lr = self.optimizer.param_groups[0]['lr']
            logger.info(
                f"Epoch {epoch}/{self.epochs} | "
                f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
                f"lr={lr:.2e} | patience={self.epochs_without_improvement}/"
                f"{self.early_stopping_patience}"
            )

        if not self._interrupted:
            self._send_telemetry(status="completed")
            
        return val_metrics

    def _train_epoch(self, epoch: int) -> float:
        """Train for one epoch.

        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(self.train_loader):
            if self._interrupted:
                break
                
            loss = self._train_step(batch)

            if self._check_nan(loss):
                logger.warning(f"NaN loss at batch {batch_idx}, skipping")
                continue

            total_loss += loss
            num_batches += 1

            # Console logging
            if (batch_idx + 1) % self.console_log_interval == 0:
                avg_loss = total_loss / num_batches
                logger.debug(
                    f"  Epoch {epoch} [{batch_idx + 1}/{len(self.train_loader)}] "
                    f"loss={loss:.4f} avg_loss={avg_loss:.4f}"
                )

        return total_loss / max(num_batches, 1)

    def _train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Execute a single training step.

        Returns:
            Scalar loss value
        """
        # Move data to device
        images = batch['image'].to(self.device)

        targets = {
            'segmentation': batch['segmentation'].to(self.device),
            'depth': batch['depth'].to(self.device),
            'severity': batch['severity'].to(self.device),
            'camera_intrinsics': batch['camera_intrinsics'].to(self.device),
            'camera_extrinsics': batch['camera_extrinsics'].to(self.device),
        }

        self.optimizer.zero_grad()

        # Forward pass with AMP
        device_type = self.device.type if self.device.type in ['cuda', 'cpu'] else 'cpu'
        with torch.autocast(device_type=device_type, enabled=self.use_amp):
            predictions = self.model(
                images, use_domain_adapter=self.use_domain_adapter
            )
            losses = self.criterion(predictions, targets)
            loss = losses['total']

        # Backward pass
        self.scaler.scale(loss).backward()

        # Gradient clipping
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)

        # Optimizer step
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return loss.item()

    @torch.no_grad()
    def _validate_epoch(self, epoch: int) -> Dict[str, float]:
        """Run validation epoch.

        Returns:
            Dict with validation metrics
        """
        self.model.eval()
        self.metrics_computer.reset()
        total_loss = 0.0
        num_batches = 0

        for batch in self.val_loader:
            images = batch['image'].to(self.device)

            targets = {
                'segmentation': batch['segmentation'].to(self.device),
                'depth': batch['depth'].to(self.device),
                'severity': batch['severity'].to(self.device),
                'camera_intrinsics': batch['camera_intrinsics'].to(self.device),
                'camera_extrinsics': batch['camera_extrinsics'].to(self.device),
            }

            device_type = self.device.type if self.device.type in ['cuda', 'cpu'] else 'cpu'
            with torch.autocast(device_type=device_type, enabled=self.use_amp):
                predictions = self.model(images, use_domain_adapter=False)
                losses = self.criterion(predictions, targets)

            total_loss += losses['total'].item()
            num_batches += 1

            # Update metrics
            detached_preds = {
                'segmentation': predictions['segmentation'].detach(),
                'depth': predictions['depth'].detach(),
                'severity': predictions['severity'].detach(),
                'intrinsics': predictions['intrinsics'].detach(),
                'extrinsics': predictions['extrinsics'].detach(),
            }
            self.metrics_computer.update(detached_preds, targets)

        # Compute final metrics
        metrics = self.metrics_computer.compute()
        metrics['val/total_loss'] = total_loss / max(num_batches, 1)

        return metrics

    def _check_nan(self, value: float) -> bool:
        """Check if a value is NaN or Inf."""
        if isinstance(value, torch.Tensor):
            return torch.isnan(value).any().item() or torch.isinf(value).any().item()
        return math.isnan(value) or math.isinf(value)

    def _save_best_checkpoint(self, epoch: int) -> None:
        """Save best model checkpoint."""
        path = self.output_dir / "best_model.pt"
        save_checkpoint(path, self.model, self.optimizer, self.scheduler,
                        epoch, self.best_metric)
        logger.info(f"Saved best checkpoint to {path}")

    def _save_periodic_checkpoint(self, epoch: int, suffix: str = "") -> None:
        """Save periodic checkpoint."""
        path = self.output_dir / f"checkpoint_epoch_{epoch:04d}{suffix}.pt"
        save_checkpoint(path, self.model, self.optimizer, self.scheduler,
                        epoch, self.best_metric)
        logger.info(f"Saved periodic checkpoint to {path}")

