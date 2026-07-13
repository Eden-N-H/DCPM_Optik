"""Lightweight Decoder with skip connections and SOA."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .dsc import DepthwiseSeparableConv
from .soa import SOA


class DecoderBlock(nn.Module):
    """Single decoder block: upsample → concat skip → 2x DSC+BN+ReLU → SOA."""

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.conv1 = DepthwiseSeparableConv(in_channels + skip_channels, out_channels, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = DepthwiseSeparableConv(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.soa = SOA(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, skip):
        # Bilinear upsample to match skip spatial size
        x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        # Concatenate with skip connection
        x = torch.cat([x, skip], dim=1)
        # 2x DSC + BN + ReLU
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        # SOA enhancement
        x = self.soa(x)
        return x


class LightweightDecoder(nn.Module):
    """3-block decoder: ASPP→1/16→1/8→1/4.

    Block 1: ASPP(256) + stage3(1024) → 256, from 1/32 to 1/16
    Block 2: Block1(256) + stage2(512) → 128, from 1/16 to 1/8
    Block 3: Block2(128) + stage1(256) → 64, from 1/8 to 1/4

    Output: [B, 64, 128, 128] for 512x512 input
    """

    def __init__(self):
        super().__init__()
        self.block1 = DecoderBlock(256, 1024, 256)   # ASPP + stage3 skip
        self.block2 = DecoderBlock(256, 512, 128)    # block1 + stage2 skip
        self.block3 = DecoderBlock(128, 256, 64)     # block2 + stage1 skip

    def forward(self, aspp_out, encoder_features):
        """
        Args:
            aspp_out: [B, 256, 16, 16] from EASPP
            encoder_features: dict with 'stage1' [B,256,128,128],
                              'stage2' [B,512,64,64], 'stage3' [B,1024,32,32]
        Returns:
            [B, 64, 128, 128] shared features
        """
        x = self.block1(aspp_out, encoder_features['stage3'])  # [B, 256, 32, 32]
        x = self.block2(x, encoder_features['stage2'])          # [B, 128, 64, 64]
        x = self.block3(x, encoder_features['stage1'])          # [B, 64, 128, 128]
        return x
