"""Multi-task prediction heads."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SegmentationHead(nn.Module):
    """Segmentation head: 2x 3x3 conv(128) + BN + ReLU → 1x1 conv(3) → upsample to input size."""

    def __init__(self, in_channels=64, num_classes=3, hidden_channels=128):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        self.classifier = nn.Conv2d(hidden_channels, num_classes, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # x: [B, 64, H/4, W/4]
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.classifier(x)  # [B, 3, H/4, W/4]
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
        return x  # [B, 3, H, W] logits


class SeverityHead(nn.Module):
    """Severity head: 2x 3x3 conv(128) + BN + ReLU → 1x1 conv(1) + sigmoid → upsample."""

    def __init__(self, in_channels=64, hidden_channels=128):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        self.regressor = nn.Conv2d(hidden_channels, 1, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = torch.sigmoid(self.regressor(x))  # [B, 1, H/4, W/4]
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
        return x  # [B, 1, H, W] in [0, 1]


class DepthHead(nn.Module):
    """Depth head: 2x 3x3 conv(128) + BN + ReLU → 1x1 conv(1) + sigmoid → upsample."""

    def __init__(self, in_channels=64, hidden_channels=128):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        self.regressor = nn.Conv2d(hidden_channels, 1, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = torch.sigmoid(self.regressor(x))  # [B, 1, H/4, W/4]
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
        return x  # [B, 1, H, W] in [0, 1]


class CameraHead(nn.Module):
    """Camera parameter prediction head.

    GAP → FC(64→512, ReLU) → FC(512→256, ReLU) → FC(256→10) →
    split: first 4 softplus (intrinsics), last 6 linear (extrinsics)
    """

    def __init__(self, in_channels=64, spatial_size=128, hidden_dims=(512, 256)):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], 10)  # 4 intrinsics + 6 extrinsics
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # x: [B, 64, H/4, W/4]
        x = self.gap(x).flatten(1)  # [B, 64]
        x = self.relu(self.fc1(x))  # [B, 512]
        x = self.relu(self.fc2(x))  # [B, 256]
        x = self.fc3(x)  # [B, 10]

        # Split: first 4 → softplus (intrinsics > 0), last 6 → linear (extrinsics)
        intrinsics = F.softplus(x[:, :4])  # [B, 4] (fx, fy, cx, cy)
        extrinsics = x[:, 4:]  # [B, 6] (3 rotation rodrigues + 3 translation)

        return intrinsics, extrinsics