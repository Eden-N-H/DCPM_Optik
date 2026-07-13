"""ResNet-50 encoder with Depthwise Separable Convolutions in stages 3-4."""
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

from .dsc import DepthwiseSeparableConv


class DSCBottleneck(nn.Module):
    """ResNet bottleneck block with DSC replacing the 3x3 convolution.

    Structure: 1x1 conv (reduce) → DSC 3x3 → 1x1 conv (expand)
    with batch norm and ReLU after each, plus residual connection.
    """

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super().__init__()
        # 1x1 reduce
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        # 3x3 DSC (replaces standard conv)
        self.conv2 = DepthwiseSeparableConv(
            planes, planes, kernel_size=3, stride=stride,
            padding=dilation, dilation=dilation, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        # 1x1 expand
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


def _make_dsc_stage(inplanes, planes, blocks, stride=1):
    """Build a ResNet stage using DSCBottleneck blocks.

    Args:
        inplanes: number of input channels
        planes: base width (output channels = planes * 4)
        blocks: number of bottleneck blocks in this stage
        stride: stride for the first block (for spatial downsampling)

    Returns:
        nn.Sequential of DSCBottleneck blocks
    """
    downsample = None
    if stride != 1 or inplanes != planes * DSCBottleneck.expansion:
        downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes * DSCBottleneck.expansion,
                      kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes * DSCBottleneck.expansion),
        )

    layers = []
    layers.append(DSCBottleneck(inplanes, planes, stride=stride, downsample=downsample))
    inplanes = planes * DSCBottleneck.expansion
    for _ in range(1, blocks):
        layers.append(DSCBottleneck(inplanes, planes))

    return nn.Sequential(*layers)


class ResNet50DSCEncoder(nn.Module):
    """ResNet-50 encoder with DSC in stages 3-4.

    Stages 1-2 use standard ResNet-50 convolutions (with pretrained weights).
    Stages 3-4 use DSCBottleneck blocks (randomly initialized).

    Input: [B, 3, 512, 512]
    Output dict:
        'stage1': [B, 256, 128, 128]
        'stage2': [B, 512, 64, 64]
        'stage3': [B, 1024, 32, 32]
        'stage4': [B, 2048, 16, 16]
    """

    def __init__(self, pretrained=True):
        super().__init__()

        # Load pretrained ResNet-50
        if pretrained:
            backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        else:
            backbone = resnet50(weights=None)

        # Stem: conv1 + bn1 + relu + maxpool
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )

        # Stages 1-2: standard ResNet blocks (pretrained weights preserved)
        self.stage1 = backbone.layer1  # 256 channels, stride 1 (after maxpool)
        self.stage2 = backbone.layer2  # 512 channels, stride 2

        # Stages 3-4: DSC bottleneck blocks (randomly initialized)
        # ResNet-50: layer3 has 6 blocks (planes=256, expansion=4 → 1024 out)
        # ResNet-50: layer4 has 3 blocks (planes=512, expansion=4 → 2048 out)
        self.stage3 = _make_dsc_stage(inplanes=512, planes=256, blocks=6, stride=2)
        self.stage4 = _make_dsc_stage(inplanes=1024, planes=512, blocks=3, stride=2)

    def forward(self, x):
        """Extract multi-scale features.

        Args:
            x: [B, 3, 512, 512] input images

        Returns:
            dict with keys 'stage1'..'stage4' containing feature maps
        """
        x = self.stem(x)       # [B, 64, 128, 128]
        s1 = self.stage1(x)    # [B, 256, 128, 128]
        s2 = self.stage2(s1)   # [B, 512, 64, 64]
        s3 = self.stage3(s2)   # [B, 1024, 32, 32]
        s4 = self.stage4(s3)   # [B, 2048, 16, 16]

        return {
            'stage1': s1,
            'stage2': s2,
            'stage3': s3,
            'stage4': s4,
        }
