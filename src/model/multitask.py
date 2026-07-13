"""Complete Multi-Task Model combining all components.

Note: This pipeline has been specialized exclusively for Dashcam footage. 
Aerial/Drone support has been removed.
"""
import torch
import torch.nn as nn
from .encoder import ResNet50DSCEncoder
from .easpp import EASPP
from .decoder import LightweightDecoder
from .heads import SegmentationHead, SeverityHead, DepthHead, CameraHead
from .domain_adapter import DualDomainAdapter


class MultiTaskModel(nn.Module):
    """Multi-task road quality model.

    Pipeline: encoder → EASPP → decoder → 4 heads
    Also includes domain adapter for training.

    Input: [B, 3, 512, 512] image
    Output dict with keys: 'segmentation', 'severity', 'depth', 'intrinsics', 'extrinsics'
    Also 'aspp_features' and 'domain_pred' if training with domain adaptation.
    """

    def __init__(self, pretrained=True, num_classes=3, lambda_adv=0.1):
        super().__init__()
        self.encoder = ResNet50DSCEncoder(pretrained=pretrained)
        self.projection = nn.Conv2d(2048, 2048, kernel_size=1, bias=False)  # 1x1 projection
        self.easpp = EASPP(in_channels=2048, out_channels=256)
        self.decoder = LightweightDecoder()

        self.seg_head = SegmentationHead(in_channels=64, num_classes=num_classes)
        self.severity_head = SeverityHead(in_channels=64)
        self.depth_head = DepthHead(in_channels=64)
        self.camera_head = CameraHead(in_channels=64)

        self.domain_adapter = DualDomainAdapter(
            feature_channels=256, num_classes=num_classes, lambda_adv=lambda_adv
        )

    def forward(self, x, use_domain_adapter=False):
        # Encode
        features = self.encoder(x)

        # EASPP (no view embedding applied)
        aspp_out = self.easpp(features['stage4'])  # [B, 256, 16, 16]

        # Decode
        decoded = self.decoder(aspp_out, features)  # [B, 64, 128, 128]

        # Task heads
        seg_logits = self.seg_head(decoded)
        severity = self.severity_head(decoded)
        depth = self.depth_head(decoded)
        intrinsics, extrinsics = self.camera_head(decoded)

        outputs = {
            'segmentation': seg_logits,
            'severity': severity,
            'depth': depth,
            'intrinsics': intrinsics,
            'extrinsics': extrinsics,
            'aspp_features': aspp_out,
        }

        # Domain adaptation (only during training)
        if use_domain_adapter:
            domain_pred = self.domain_adapter(aspp_out, seg_logits)
            outputs['domain_pred'] = domain_pred

        return outputs
