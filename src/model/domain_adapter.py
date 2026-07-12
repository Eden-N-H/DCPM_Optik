"""Domain adaptation with Gradient Reversal Layer and dual discriminators."""
import torch
import torch.nn as nn
from torch.autograd import Function
from typing import Dict, Tuple


class GradientReversalFunction(Function):
    """Gradient Reversal Layer autograd function.

    Forward: returns x unchanged (clone).
    Backward: negates gradient scaled by lambda_val.
    """

    @staticmethod
    def forward(ctx, x, lambda_val):
        ctx.save_for_backward(torch.tensor(lambda_val))
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        lambda_val = ctx.saved_tensors[0].item()
        return -lambda_val * grad_output, None


class GradientReversalLayer(nn.Module):
    """GRL module: identity in forward, negates gradients * lambda in backward.

    Args:
        lambda_val: Scaling factor for gradient negation.
    """

    def __init__(self, lambda_val: float = 1.0):
        super().__init__()
        self.lambda_val = lambda_val

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(x, self.lambda_val)


class DomainDiscriminator(nn.Module):
    """3-layer convolutional discriminator for domain classification.

    Architecture: 3 conv layers (channels=[256, 128, 1], kernel=3, stride=2)
    with LeakyReLU(0.2) after each layer except the last.

    Args:
        in_channels: Number of input channels.
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 1, kernel_size=3, stride=2, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass producing spatial domain prediction logits.

        Args:
            x: Input tensor [B, C, H, W].

        Returns:
            Domain prediction logits [B, 1, H', W'] where spatial dims
            are reduced by stride-2 convolutions.
        """
        return self.layers(x)


class DualDomainAdapter(nn.Module):
    """Dual-discriminator domain adapter with gradient reversal.

    Uses two discriminator networks:
    - Feature discriminator: operates on E-ASPP output features [B, 256, 16, 16]
    - Logit discriminator: operates on segmentation logits [B, 3, 512, 512]

    Both apply GRL with lambda_adv before passing to their discriminator.

    Args:
        feature_channels: Number of channels in E-ASPP features (default: 256).
        num_classes: Number of segmentation classes (default: 3).
        lambda_adv: Adversarial scaling factor for GRL (default: 0.1).
    """

    def __init__(self, feature_channels: int = 256, num_classes: int = 3,
                 lambda_adv: float = 0.1):
        super().__init__()
        self.lambda_adv = lambda_adv

        self.feature_grl = GradientReversalLayer(lambda_adv)
        self.logit_grl = GradientReversalLayer(lambda_adv)

        self.feature_disc = DomainDiscriminator(feature_channels)
        self.logit_disc = DomainDiscriminator(num_classes)

    def forward(self, features: torch.Tensor, logits: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through both discriminators with GRL.

        Args:
            features: E-ASPP output features [B, 256, H, W].
            logits: Segmentation logits [B, 3, H', W'].

        Returns:
            Dict with 'feat_pred' and 'logit_pred' domain prediction logit tensors.
        """
        # Apply GRL then discriminator to features
        feat_reversed = self.feature_grl(features)
        feat_pred = self.feature_disc(feat_reversed)

        # Apply GRL then discriminator to logits
        logit_reversed = self.logit_grl(logits)
        logit_pred = self.logit_disc(logit_reversed)

        return {
            'feat_pred': feat_pred,
            'logit_pred': logit_pred,
        }
