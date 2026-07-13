"""CycleGAN components for defect-aware image translation."""

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
from src.cyclegan.trainer import CycleGANConfig, CycleGANTrainer, compute_lr

__all__ = [
    "ResNetGenerator",
    "PatchGANDiscriminator",
    "ImagePool",
    "lsgan_loss_real",
    "lsgan_loss_fake",
    "cycle_consistency_loss",
    "identity_loss",
    "defect_preservation_loss",
    "CycleGANTrainer",
    "CycleGANConfig",
    "compute_lr",
]
