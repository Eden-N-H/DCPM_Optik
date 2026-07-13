"""Efficient Atrous Spatial Pyramid Pooling (EASPP) module."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .dsc import DepthwiseSeparableConv
from .soa import SOA


class EASPP(nn.Module):
    """Efficient ASPP with DSC branches and Small-Object Attention.

    Architecture:
        - 4 parallel DSC branches with dilation rates {3, 6, 12, 18}
        - 1 global context branch (GAP + 1x1 conv + upsample)
        - Concatenation of all 5 branches → SOA → 1x1 projection

    Input: [B, 2048, 16, 16] (encoder stage4 2048)
    Output: [B, 256, 16, 16]
    """

    def __init__(self, in_channels=2048, out_channels=256):
        super().__init__()

        branch_channels = 256
        dilations = [3, 6, 12, 18]

        # 4 parallel DSC branches with different dilation rates
        self.dsc_branches = nn.ModuleList()
        for d in dilations:
            branch = nn.Sequential(
                DepthwiseSeparableConv(
                    in_channels, branch_channels,
                    kernel_size=3, stride=1, padding=d, dilation=d, bias=False
                ),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
            )
            self.dsc_branches.append(branch)

        # Global context branch: AdaptiveAvgPool → 1x1 Conv → BN → ReLU → Upsample
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_conv = nn.Conv2d(in_channels, branch_channels, kernel_size=1, bias=False)
        self.global_bn = nn.BatchNorm2d(branch_channels)
        self.global_relu = nn.ReLU(inplace=True)

        # Total concatenated channels: 4 * 256 + 256 = 1280
        concat_channels = branch_channels * 5

        # SOA on concatenated features
        self.soa = SOA(channels=concat_channels)

        # Final 1x1 projection
        self.project = nn.Sequential(
            nn.Conv2d(concat_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        """Apply EASPP.

        Args:
            x: [B, 2048, 16, 16] input features (encoder output)

        Returns:
            [B, 256, 16, 16] refined features
        """
        spatial_size = x.shape[2:]

        # DSC branches
        branch_outputs = [branch(x) for branch in self.dsc_branches]

        # Global branch: GAP → 1x1 conv → BN → ReLU → bilinear upsample
        global_feat = self.global_pool(x)  # [B, in_channels, 1, 1]
        global_feat = self.global_conv(global_feat)  # [B, 256, 1, 1]
        global_feat = self.global_bn(global_feat)  # [B, 256, 1, 1]
        global_feat = self.global_relu(global_feat)  # [B, 256, 1, 1]
        global_feat = F.interpolate(
            global_feat, size=spatial_size, mode='bilinear', align_corners=False
        )
        branch_outputs.append(global_feat)

        # Concatenate all branches: [B, 1280, 16, 16]
        concat = torch.cat(branch_outputs, dim=1)

        # Apply SOA
        attended = self.soa(concat)

        # Project to output channels
        out = self.project(attended)

        return out
