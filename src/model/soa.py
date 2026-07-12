"""Small-Object Attention (SOA) module."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SOA(nn.Module):
    """Small-Object Attention module.

    Combines channel attention, multi-scale spatial attention, and high-pass
    enhancement to boost detection of small road defects.

    Applied in order:
        1. Channel attention: x_ca = x * wc
        2. Spatial attention: x_sa = x_ca * ws
        3. High-pass enhancement: out = x_sa + alpha * (x_sa - blur(x_sa))

    where:
        wc = channel attention weights (per-channel, in [0, 1])
        ws = spatial attention weights (per-pixel, in [0, 1])
        alpha = high-pass scaling factor (default 0.3)
    """

    def __init__(self, channels: int, reduction: int = 16, alpha: float = 0.3):
        super().__init__()
        self.channels = channels
        self.alpha = alpha

        # --- Channel Attention (Req 8.1) ---
        # GAP → FC(C→C//r) + ReLU → FC(C//r→C) + Sigmoid
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

        # --- Spatial Attention (Req 8.2) ---
        # Mean across channels → 4 parallel AvgPool2d(k={1,3,5,7}) → concat → 1x1 Conv → Sigmoid
        self.spatial_pools = nn.ModuleList([
            nn.AvgPool2d(kernel_size=k, stride=1, padding=k // 2)
            for k in [1, 3, 5, 7]
        ])
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(4, 1, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

        # --- High-pass Enhancement (Req 8.3) ---
        # Fixed Gaussian blur kernel (7x7, sigma=1.0)
        gaussian_kernel_size = 7
        self.register_buffer('gaussian_kernel', self._make_gaussian_kernel(gaussian_kernel_size, 1.0))
        self._blur_padding: int = gaussian_kernel_size // 2

    @staticmethod
    def _make_gaussian_kernel(kernel_size: int, sigma: float) -> torch.Tensor:
        """Create a 2D Gaussian kernel.

        Returns:
            Tensor of shape [1, 1, K, K] for use with groups=C depthwise conv.
        """
        coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g2d = g.unsqueeze(1) * g.unsqueeze(0)
        g2d = g2d / g2d.sum()
        return g2d.unsqueeze(0).unsqueeze(0)

    def _gaussian_blur(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Gaussian blur using depthwise convolution with fixed kernel.

        Args:
            x: [B, C, H, W] input tensor.

        Returns:
            [B, C, H, W] blurred tensor.
        """
        B, C, H, W = x.shape
        # Expand kernel to [C, 1, K, K] for depthwise conv
        kernel = self.gaussian_kernel.expand(C, -1, -1, -1)
        return F.conv2d(x, kernel, padding=self._blur_padding, groups=C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Small-Object Attention.

        Applies channel attention, then spatial attention, then high-pass
        enhancement in sequence (Req 8.4). Preserves input shape (Req 8.5).

        Args:
            x: [B, C, H, W] input features.

        Returns:
            [B, C, H, W] attention-enhanced features (same shape as input).
        """
        # 1. Channel attention (Req 8.1): wc [B, C, 1, 1]
        wc = self.channel_attn(x)  # [B, C]
        wc = wc.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        x_ca = x * wc  # [B, C, H, W]

        # 2. Spatial attention (Req 8.2): ws [B, 1, H, W]
        # Compute channel-wise mean → [B, 1, H, W]
        spatial_input = x_ca.mean(dim=1, keepdim=True)
        # Multi-scale pooling with k={1,3,5,7}
        pooled = [pool(spatial_input) for pool in self.spatial_pools]
        spatial_cat = torch.cat(pooled, dim=1)  # [B, 4, H, W]
        ws = self.spatial_conv(spatial_cat)  # [B, 1, H, W]
        x_sa = x_ca * ws  # [B, C, H, W]

        # 3. High-pass enhancement (Req 8.3):
        # Subtract 7x7 Gaussian-smoothed version, scale by alpha, add back
        blurred = self._gaussian_blur(x_sa)
        high_pass = x_sa - blurred  # High-frequency residual
        out = x_sa + self.alpha * high_pass  # [B, C, H, W]

        return out
