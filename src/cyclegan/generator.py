"""ResNet-9 Generator for CycleGAN image translation."""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Residual block with two 3x3 convolutions, instance norm, and ReLU."""

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, padding=0),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, padding=0),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class ResNetGenerator(nn.Module):
    """ResNet-9 generator for image-to-image translation.

    Input: [B, 4, 256, 256] — 3 RGB channels in [-1,1] + 1 binary defect mask
    Output: [B, 3, 256, 256] — RGB image in [-1, 1]
    """

    def __init__(self, input_channels: int = 4, output_channels: int = 3, ngf: int = 64, n_residual_blocks: int = 9):
        super().__init__()

        # Initial 7x7 convolution
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_channels, ngf, kernel_size=7, padding=0),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True),
        ]

        # Downsampling: 2 layers with stride 2
        in_features = ngf
        out_features = in_features * 2
        for _ in range(2):
            model += [
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_features, out_features, kernel_size=3, stride=2, padding=0),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features
            out_features = in_features * 2

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling: 2 layers with stride 2
        out_features = in_features // 2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(in_features, out_features, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features
            out_features = in_features // 2

        # Final 7x7 convolution — no instance norm, no ReLU
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_channels, kernel_size=7, padding=0),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
