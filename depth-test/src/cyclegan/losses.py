"""CycleGAN loss functions and ImagePool utility."""

import random
from typing import Callable

import torch
import torch.nn as nn


def lsgan_loss_real(pred: torch.Tensor) -> torch.Tensor:
    """LSGAN loss for real images: mean((pred - 1)²)."""
    return torch.mean((pred - 1.0) ** 2)


def lsgan_loss_fake(pred: torch.Tensor) -> torch.Tensor:
    """LSGAN loss for fake images: mean(pred²)."""
    return torch.mean(pred ** 2)


def cycle_consistency_loss(real: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
    """Cycle consistency loss: L1 between real and reconstructed images."""
    return nn.functional.l1_loss(reconstructed, real)


def identity_loss(real: torch.Tensor, same_domain_output: torch.Tensor) -> torch.Tensor:
    """Identity loss: L1 between real and same-domain output."""
    return nn.functional.l1_loss(same_domain_output, real)


def defect_preservation_loss(
    original_mask: torch.Tensor,
    generated: torch.Tensor,
    mask_extractor: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Defect preservation loss: L1 between original mask and extracted mask within masked regions.

    Args:
        original_mask: Binary defect mask [B, 1, H, W].
        generated: Generated image [B, 3, H, W].
        mask_extractor: Callable that extracts a defect mask from a generated image.

    Returns:
        L1 loss computed only within masked regions.
    """
    extracted_mask = mask_extractor(generated)

    # Compute L1 only within masked regions
    masked_original = original_mask * original_mask
    masked_extracted = original_mask * extracted_mask

    # Avoid division by zero when mask is empty
    num_masked_pixels = original_mask.sum()
    if num_masked_pixels == 0:
        return torch.tensor(0.0, device=original_mask.device, requires_grad=True)

    loss = nn.functional.l1_loss(masked_extracted, masked_original, reduction="sum") / num_masked_pixels
    return loss


class ImagePool:
    """Buffer of previously generated images for discriminator training.

    Stores up to `pool_size` images. When queried:
    - If buffer not full, stores and returns the input image.
    - If buffer full, with 50% probability swaps a stored image with the input
      and returns the stored image; otherwise returns the input as-is.
    """

    def __init__(self, pool_size: int = 50):
        self.pool_size = pool_size
        self.images: list[torch.Tensor] = []

    def query(self, images: torch.Tensor) -> torch.Tensor:
        """Return images from the pool, potentially replacing some with stored ones.

        Args:
            images: Batch of generated images [B, C, H, W].

        Returns:
            Batch of images, some potentially from the buffer.
        """
        if self.pool_size == 0:
            return images

        return_images = []
        for image in images:
            image = image.unsqueeze(0)

            if len(self.images) < self.pool_size:
                # Buffer not full — store and return
                self.images.append(image.clone())
                return_images.append(image)
            else:
                if random.random() > 0.5:
                    # Swap: pick a random stored image, replace it with new one
                    idx = random.randint(0, self.pool_size - 1)
                    stored = self.images[idx].clone()
                    self.images[idx] = image.clone()
                    return_images.append(stored)
                else:
                    # Return the current image without modifying buffer
                    return_images.append(image)

        return torch.cat(return_images, dim=0)
