"""Unit tests for PatchGANDiscriminator."""

import torch
import torch.nn as nn
import pytest

from src.cyclegan.discriminator import PatchGANDiscriminator


class TestPatchGANDiscriminator:
    """Tests for PatchGAN discriminator architecture and output shape."""

    def test_output_shape_single_batch(self):
        """Output shape is [1, 1, 30, 30] for input [1, 3, 256, 256]."""
        model = PatchGANDiscriminator(input_channels=3, ndf=64)
        x = torch.randn(1, 3, 256, 256)
        out = model(x)
        assert out.shape == torch.Size([1, 1, 30, 30])

    def test_output_shape_multi_batch(self):
        """Output shape preserves batch dimension."""
        model = PatchGANDiscriminator(input_channels=3, ndf=64)
        x = torch.randn(4, 3, 256, 256)
        out = model(x)
        assert out.shape == torch.Size([4, 1, 30, 30])

    def test_no_sigmoid_on_output(self):
        """Output is raw logits (unbounded), not passed through sigmoid."""
        model = PatchGANDiscriminator(input_channels=3, ndf=64)
        x = torch.randn(2, 3, 256, 256)
        out = model(x)
        # Raw logits can be outside [0, 1]
        # With random init, some values should be negative
        assert out.min() < 0.0 or out.max() > 1.0

    def test_first_layer_no_instance_norm(self):
        """First conv layer has no instance norm — only Conv2d + LeakyReLU."""
        model = PatchGANDiscriminator(input_channels=3, ndf=64)
        layers = list(model.model.children())
        # First layer: Conv2d
        assert isinstance(layers[0], nn.Conv2d)
        assert layers[0].in_channels == 3
        assert layers[0].out_channels == 64
        assert layers[0].kernel_size == (4, 4)
        assert layers[0].stride == (2, 2)
        # Second: LeakyReLU (no InstanceNorm between)
        assert isinstance(layers[1], nn.LeakyReLU)
        assert layers[1].negative_slope == pytest.approx(0.2)

    def test_intermediate_layers_have_instance_norm(self):
        """Layers 2, 3, 4 all have InstanceNorm2d."""
        model = PatchGANDiscriminator(input_channels=3, ndf=64)
        layers = list(model.model.children())
        # Layer 2 starts at index 2: Conv, InstanceNorm, LeakyReLU
        assert isinstance(layers[2], nn.Conv2d)
        assert isinstance(layers[3], nn.InstanceNorm2d)
        assert isinstance(layers[4], nn.LeakyReLU)
        # Layer 3 starts at index 5
        assert isinstance(layers[5], nn.Conv2d)
        assert isinstance(layers[6], nn.InstanceNorm2d)
        assert isinstance(layers[7], nn.LeakyReLU)
        # Layer 4 starts at index 8
        assert isinstance(layers[8], nn.Conv2d)
        assert isinstance(layers[9], nn.InstanceNorm2d)
        assert isinstance(layers[10], nn.LeakyReLU)

    def test_last_layer_no_instance_norm_no_activation(self):
        """Final layer is just Conv2d(512, 1) with no norm or activation."""
        model = PatchGANDiscriminator(input_channels=3, ndf=64)
        layers = list(model.model.children())
        # Last layer should be a Conv2d
        assert isinstance(layers[-1], nn.Conv2d)
        assert layers[-1].in_channels == 512
        assert layers[-1].out_channels == 1
        assert layers[-1].kernel_size == (4, 4)
        assert layers[-1].stride == (1, 1)

    def test_filter_progression(self):
        """Conv layers use filters [64, 128, 256, 512, 1]."""
        model = PatchGANDiscriminator(input_channels=3, ndf=64)
        conv_layers = [m for m in model.model.modules() if isinstance(m, nn.Conv2d)]
        expected_out_channels = [64, 128, 256, 512, 1]
        for conv, expected in zip(conv_layers, expected_out_channels):
            assert conv.out_channels == expected

    def test_leaky_relu_slope(self):
        """All LeakyReLU activations use negative_slope=0.2."""
        model = PatchGANDiscriminator(input_channels=3, ndf=64)
        leaky_relus = [m for m in model.model.modules() if isinstance(m, nn.LeakyReLU)]
        for lr in leaky_relus:
            assert lr.negative_slope == pytest.approx(0.2)

    def test_gradient_flows(self):
        """Gradients flow through the discriminator for training."""
        model = PatchGANDiscriminator(input_channels=3, ndf=64)
        x = torch.randn(1, 3, 256, 256, requires_grad=True)
        out = model(x)
        loss = out.mean()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_default_parameters(self):
        """Default initialization uses input_channels=3 and ndf=64."""
        model = PatchGANDiscriminator()
        x = torch.randn(1, 3, 256, 256)
        out = model(x)
        assert out.shape == torch.Size([1, 1, 30, 30])
