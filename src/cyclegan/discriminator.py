"""PatchGAN Discriminator for CycleGAN."""

import torch
import torch.nn as nn


class PatchGANDiscriminator(nn.Module):
    """PatchGAN discriminator that classifies 30x30 overlapping patches.

    Input: [B, 3, 256, 256]
    Output: [B, 1, 30, 30] — raw predictions (no sigmoid, for LSGAN)
    """

    def __init__(self, input_channels: int = 3, ndf: int = 64):
        super().__init__()

        # Layer 1: 64 filters, stride=2, no instance norm
        layers = [
            nn.Conv2d(input_channels, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        # Layer 2: 128 filters, stride=2, instance norm
        layers += [
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        # Layer 3: 256 filters, stride=2, instance norm
        layers += [
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        # Layer 4: 512 filters, stride=1, instance norm
        layers += [
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=1, padding=1),
            nn.InstanceNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        # Final layer: 1-channel output, stride=1, no instance norm, no activation
        layers += [
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=1),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
