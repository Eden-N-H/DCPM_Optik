"""CycleGAN Trainer with all loss components.

Implements the full CycleGAN training procedure including:
- LSGAN adversarial loss (Req 5.1)
- Cycle consistency loss with λ_cycle=10 (Req 5.2)
- Identity loss with λ_identity=0.5 (Req 5.3)
- Defect preservation loss with λ_defect=5.0 (Req 5.4)
- Total generator loss summation (Req 5.5)
- Adam optimizer with lr=2e-4, β1=0.5, β2=0.999 (Req 5.6)
- Linear LR decay after epoch 100 of 200 (Req 5.7)
- Image history buffer of 50 images (Req 5.8)
- NaN/Inf detection with checkpoint saving (Req 5.9)
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import torch
import torch.nn as nn
from torch.optim import Adam

from src.cyclegan.discriminator import PatchGANDiscriminator
from src.cyclegan.generator import ResNetGenerator
from src.cyclegan.losses import (
    ImagePool,
    cycle_consistency_loss,
    defect_preservation_loss,
    identity_loss,
    lsgan_loss_fake,
    lsgan_loss_real,
)


@dataclass
class CycleGANConfig:
    """Configuration for CycleGAN training."""

    input_nc: int = 4
    output_nc: int = 3
    ngf: int = 64
    ndf: int = 64
    n_blocks: int = 9
    epochs: int = 200
    lr: float = 2e-4
    beta1: float = 0.5
    beta2: float = 0.999
    decay_start_epoch: int = 100
    pool_size: int = 50
    lambda_cycle: float = 10.0
    lambda_identity: float = 0.5
    lambda_defect: float = 5.0
    checkpoint_dir: str = "./checkpoints/cyclegan"


def compute_lr(epoch: int, initial_lr: float = 2e-4, decay_start_epoch: int = 100, total_epochs: int = 200) -> float:
    """Compute the learning rate for a given epoch using linear decay schedule.

    The learning rate stays at initial_lr for epochs [0, decay_start_epoch),
    then linearly decays to 0 over epochs [decay_start_epoch, total_epochs).

    Args:
        epoch: Current epoch number (0-indexed).
        initial_lr: Initial learning rate (default 2e-4).
        decay_start_epoch: Epoch at which decay begins (default 100).
        total_epochs: Total number of training epochs (default 200).

    Returns:
        The learning rate for the given epoch.
    """
    if epoch < decay_start_epoch:
        return initial_lr
    else:
        # Linear decay from initial_lr to 0 over (total_epochs - decay_start_epoch) epochs
        decay_epochs = total_epochs - decay_start_epoch
        return initial_lr * (total_epochs - epoch) / decay_epochs


class LambdaLRSchedule:
    """Learning rate lambda for linear decay after decay_start_epoch.

    Used with torch.optim.lr_scheduler.LambdaLR. The lambda multiplier starts
    at 1.0 and linearly decays to 0.0 after decay_start_epoch.
    """

    def __init__(self, decay_start_epoch: int = 100, total_epochs: int = 200):
        self.decay_start_epoch = decay_start_epoch
        self.total_epochs = total_epochs

    def __call__(self, epoch: int) -> float:
        """Return multiplicative factor for the learning rate."""
        if epoch < self.decay_start_epoch:
            return 1.0
        else:
            decay_epochs = self.total_epochs - self.decay_start_epoch
            return max(0.0, (self.total_epochs - epoch) / decay_epochs)


class CycleGANTrainer:
    """Training loop for CycleGAN with all loss components.

    Manages two generators (G_AB: A→B, G_BA: B→A) and two discriminators
    (D_A, D_B). Implements the full training procedure including:
    - LSGAN adversarial loss
    - Cycle consistency loss (λ=10)
    - Identity loss (λ=0.5)
    - Defect preservation loss (λ=5.0)
    - Image history buffer for discriminator stability
    - Linear LR decay after epoch 100
    - NaN/Inf detection with checkpoint saving

    Args:
        config: CycleGAN training configuration.
        mask_extractor: Callable that extracts a defect mask from a generated image.
                       Takes [B, 3, H, W] tensor and returns [B, 1, H, W] mask.
        device: Torch device for computation.
    """

    def __init__(
        self,
        config: CycleGANConfig,
        mask_extractor: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        device: Optional[torch.device] = None,
    ):
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Mask extractor for defect preservation loss
        # If not provided, use a dummy that returns zeros (effectively disabling defect loss)
        self._mask_extractor = mask_extractor or self._default_mask_extractor

        # Build networks
        self.G_AB = ResNetGenerator(
            input_channels=config.input_nc,
            output_channels=config.output_nc,
            ngf=config.ngf,
            n_residual_blocks=config.n_blocks,
        ).to(self.device)

        self.G_BA = ResNetGenerator(
            input_channels=config.input_nc,
            output_channels=config.output_nc,
            ngf=config.ngf,
            n_residual_blocks=config.n_blocks,
        ).to(self.device)

        self.D_A = PatchGANDiscriminator(
            input_channels=config.output_nc,
            ndf=config.ndf,
        ).to(self.device)

        self.D_B = PatchGANDiscriminator(
            input_channels=config.output_nc,
            ndf=config.ndf,
        ).to(self.device)

        # Optimizers — Adam with lr=2e-4, β1=0.5, β2=0.999 (Req 5.6)
        self.optimizer_G = Adam(
            list(self.G_AB.parameters()) + list(self.G_BA.parameters()),
            lr=config.lr,
            betas=(config.beta1, config.beta2),
        )
        self.optimizer_D_A = Adam(
            self.D_A.parameters(),
            lr=config.lr,
            betas=(config.beta1, config.beta2),
        )
        self.optimizer_D_B = Adam(
            self.D_B.parameters(),
            lr=config.lr,
            betas=(config.beta1, config.beta2),
        )

        # LR schedulers — linear decay after decay_start_epoch (Req 5.7)
        lr_schedule = LambdaLRSchedule(config.decay_start_epoch, config.epochs)
        self.scheduler_G = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer_G, lr_lambda=lr_schedule
        )
        self.scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer_D_A, lr_lambda=lr_schedule
        )
        self.scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer_D_B, lr_lambda=lr_schedule
        )

        # Image history buffers for discriminator stability (Req 5.8)
        self.fake_A_pool = ImagePool(config.pool_size)
        self.fake_B_pool = ImagePool(config.pool_size)

        # Training state
        self.current_epoch = 0
        self._last_valid_state: Optional[Dict[str, Any]] = None
        self._diverged = False

    def _default_mask_extractor(self, generated: torch.Tensor) -> torch.Tensor:
        """Default mask extractor that returns zeros (no defect preservation)."""
        return torch.zeros(
            generated.shape[0], 1, generated.shape[2], generated.shape[3],
            device=generated.device, dtype=generated.dtype,
        )

    def _check_nan_inf(self, losses: Dict[str, torch.Tensor]) -> bool:
        """Check if any loss component is NaN or Inf.

        Args:
            losses: Dictionary of loss tensors.

        Returns:
            True if any loss is NaN or Inf, False otherwise.
        """
        for name, loss in losses.items():
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                return True
        return False

    def _save_checkpoint(self, path: Path, epoch: int, reason: str = "") -> None:
        """Save a training checkpoint.

        Args:
            path: Directory to save the checkpoint.
            epoch: Current epoch number.
            reason: Reason for saving (e.g., 'divergence_detected').
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "epoch": epoch,
            "reason": reason,
            "G_AB_state_dict": self.G_AB.state_dict(),
            "G_BA_state_dict": self.G_BA.state_dict(),
            "D_A_state_dict": self.D_A.state_dict(),
            "D_B_state_dict": self.D_B.state_dict(),
            "optimizer_G_state_dict": self.optimizer_G.state_dict(),
            "optimizer_D_A_state_dict": self.optimizer_D_A.state_dict(),
            "optimizer_D_B_state_dict": self.optimizer_D_B.state_dict(),
            "scheduler_G_state_dict": self.scheduler_G.state_dict(),
            "scheduler_D_A_state_dict": self.scheduler_D_A.state_dict(),
            "scheduler_D_B_state_dict": self.scheduler_D_B.state_dict(),
        }

        filename = f"checkpoint_epoch_{epoch}"
        if reason:
            filename += f"_{reason}"
        filename += ".pt"

        torch.save(checkpoint, path / filename)

    def _save_valid_state(self) -> None:
        """Save current state as last valid state for NaN recovery."""
        self._last_valid_state = {
            "epoch": self.current_epoch,
            "G_AB_state_dict": {k: v.clone() for k, v in self.G_AB.state_dict().items()},
            "G_BA_state_dict": {k: v.clone() for k, v in self.G_BA.state_dict().items()},
            "D_A_state_dict": {k: v.clone() for k, v in self.D_A.state_dict().items()},
            "D_B_state_dict": {k: v.clone() for k, v in self.D_B.state_dict().items()},
            "optimizer_G_state_dict": self.optimizer_G.state_dict(),
            "optimizer_D_A_state_dict": self.optimizer_D_A.state_dict(),
            "optimizer_D_B_state_dict": self.optimizer_D_B.state_dict(),
        }

    def train_step(
        self,
        real_A: torch.Tensor,
        real_B: torch.Tensor,
        mask_A: torch.Tensor,
    ) -> Dict[str, float]:
        """Perform one training step for both generators and discriminators.

        Args:
            real_A: Real images from domain A [B, 3, 256, 256] (synthetic).
            real_B: Real images from domain B [B, 3, 256, 256] (real).
            mask_A: Defect masks for domain A [B, 1, 256, 256].

        Returns:
            Dictionary of loss values as floats. If NaN/Inf is detected,
            returns the losses with a 'diverged' flag set to True.

        Raises:
            RuntimeError: If training has already diverged.
        """
        if self._diverged:
            raise RuntimeError("Training has diverged. Cannot continue training.")

        real_A = real_A.to(self.device)
        real_B = real_B.to(self.device)
        mask_A = mask_A.to(self.device)

        # Construct generator inputs (RGB + mask)
        input_A = torch.cat([real_A, mask_A], dim=1)  # [B, 4, 256, 256]
        # For B→A direction, use zero mask (real images don't have masks)
        zero_mask = torch.zeros_like(mask_A)
        input_B = torch.cat([real_B, zero_mask], dim=1)  # [B, 4, 256, 256]

        # ========================
        # Train Generators
        # ========================
        self.optimizer_G.zero_grad()

        # Forward pass through generators
        fake_B = self.G_AB(input_A)   # A → B
        fake_A = self.G_BA(input_B)   # B → A

        # Cycle reconstruction
        input_fake_B = torch.cat([fake_B, mask_A], dim=1)
        input_fake_A = torch.cat([fake_A, zero_mask], dim=1)
        rec_A = self.G_BA(input_fake_B)   # A → B → A
        rec_B = self.G_AB(input_fake_A)   # B → A → B

        # Identity mapping
        idt_A = self.G_BA(input_A)   # G_BA(A) should ≈ A if A already looks like A
        idt_B = self.G_AB(input_B)   # G_AB(B) should ≈ B if B already looks like B

        # --- Adversarial loss (LSGAN) (Req 5.1) ---
        pred_fake_B = self.D_B(fake_B)
        pred_fake_A = self.D_A(fake_A)
        loss_G_AB = lsgan_loss_real(pred_fake_B)  # G_AB wants D_B to classify fake_B as real
        loss_G_BA = lsgan_loss_real(pred_fake_A)  # G_BA wants D_A to classify fake_A as real
        loss_adversarial = loss_G_AB + loss_G_BA

        # --- Cycle consistency loss (Req 5.2) ---
        loss_cycle_A = cycle_consistency_loss(real_A, rec_A)
        loss_cycle_B = cycle_consistency_loss(real_B, rec_B)
        loss_cycle = loss_cycle_A + loss_cycle_B

        # --- Identity loss (Req 5.3) ---
        loss_idt_A = identity_loss(real_A, idt_A)
        loss_idt_B = identity_loss(real_B, idt_B)
        loss_identity = loss_idt_A + loss_idt_B

        # --- Defect preservation loss (Req 5.4) ---
        loss_defect = defect_preservation_loss(
            mask_A, fake_B, self._mask_extractor
        )

        # --- Total generator loss (Req 5.5) ---
        # total = adversarial + λ_cycle × cycle + λ_identity × identity + λ_defect × defect
        loss_G = (
            loss_adversarial
            + self.config.lambda_cycle * loss_cycle
            + self.config.lambda_identity * loss_identity
            + self.config.lambda_defect * loss_defect
        )

        # Check for NaN/Inf before backward (Req 5.9)
        gen_losses = {
            "adversarial": loss_adversarial,
            "cycle": loss_cycle,
            "identity": loss_identity,
            "defect": loss_defect,
            "G_total": loss_G,
        }

        if self._check_nan_inf(gen_losses):
            self._diverged = True
            self._save_checkpoint(
                Path(self.config.checkpoint_dir),
                self.current_epoch,
                reason="divergence",
            )
            return {
                "loss_G": float("nan"),
                "loss_adversarial": float("nan"),
                "loss_cycle": float("nan"),
                "loss_identity": float("nan"),
                "loss_defect": float("nan"),
                "loss_D_A": float("nan"),
                "loss_D_B": float("nan"),
                "diverged": True,
            }

        loss_G.backward()
        self.optimizer_G.step()

        # ========================
        # Train Discriminator A
        # ========================
        self.optimizer_D_A.zero_grad()

        # Use image pool for stability (Req 5.8)
        fake_A_pooled = self.fake_A_pool.query(fake_A.detach())
        pred_real_A = self.D_A(real_A)
        pred_fake_A_d = self.D_A(fake_A_pooled)

        loss_D_A_real = lsgan_loss_real(pred_real_A)
        loss_D_A_fake = lsgan_loss_fake(pred_fake_A_d)
        loss_D_A = (loss_D_A_real + loss_D_A_fake) * 0.5

        # Check for NaN/Inf (Req 5.9)
        if self._check_nan_inf({"D_A": loss_D_A}):
            self._diverged = True
            self._save_checkpoint(
                Path(self.config.checkpoint_dir),
                self.current_epoch,
                reason="divergence",
            )
            return {
                "loss_G": loss_G.item(),
                "loss_adversarial": loss_adversarial.item(),
                "loss_cycle": loss_cycle.item(),
                "loss_identity": loss_identity.item(),
                "loss_defect": loss_defect.item(),
                "loss_D_A": float("nan"),
                "loss_D_B": float("nan"),
                "diverged": True,
            }

        loss_D_A.backward()
        self.optimizer_D_A.step()

        # ========================
        # Train Discriminator B
        # ========================
        self.optimizer_D_B.zero_grad()

        # Use image pool for stability (Req 5.8)
        fake_B_pooled = self.fake_B_pool.query(fake_B.detach())
        pred_real_B = self.D_B(real_B)
        pred_fake_B_d = self.D_B(fake_B_pooled)

        loss_D_B_real = lsgan_loss_real(pred_real_B)
        loss_D_B_fake = lsgan_loss_fake(pred_fake_B_d)
        loss_D_B = (loss_D_B_real + loss_D_B_fake) * 0.5

        # Check for NaN/Inf (Req 5.9)
        if self._check_nan_inf({"D_B": loss_D_B}):
            self._diverged = True
            self._save_checkpoint(
                Path(self.config.checkpoint_dir),
                self.current_epoch,
                reason="divergence",
            )
            return {
                "loss_G": loss_G.item(),
                "loss_adversarial": loss_adversarial.item(),
                "loss_cycle": loss_cycle.item(),
                "loss_identity": loss_identity.item(),
                "loss_defect": loss_defect.item(),
                "loss_D_A": loss_D_A.item(),
                "loss_D_B": float("nan"),
                "diverged": True,
            }

        loss_D_B.backward()
        self.optimizer_D_B.step()

        # Save valid state for recovery
        self._save_valid_state()

        return {
            "loss_G": loss_G.item(),
            "loss_adversarial": loss_adversarial.item(),
            "loss_cycle": loss_cycle.item(),
            "loss_identity": loss_identity.item(),
            "loss_defect": loss_defect.item(),
            "loss_D_A": loss_D_A.item(),
            "loss_D_B": loss_D_B.item(),
            "diverged": False,
        }

    def step_schedulers(self) -> None:
        """Step all learning rate schedulers. Call once per epoch."""
        self.scheduler_G.step()
        self.scheduler_D_A.step()
        self.scheduler_D_B.step()
        self.current_epoch += 1

    def get_current_lr(self) -> float:
        """Get the current learning rate from the generator optimizer."""
        return self.optimizer_G.param_groups[0]["lr"]

    def compute_defect_preservation_loss(
        self, generated: torch.Tensor, original_mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute defect preservation loss for external use.

        Args:
            generated: Generated image [B, 3, H, W].
            original_mask: Original binary defect mask [B, 1, H, W].

        Returns:
            Defect preservation loss tensor.
        """
        return defect_preservation_loss(original_mask, generated, self._mask_extractor)

    @property
    def diverged(self) -> bool:
        """Whether training has diverged (NaN/Inf detected)."""
        return self._diverged
