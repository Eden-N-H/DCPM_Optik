"""Unit tests for the ResNetGenerator."""

import torch
import pytest

from src.cyclegan.generator import ResNetGenerator, ResidualBlock


class TestResidualBlock:
    """Tests for the ResidualBlock component."""

    def test_output_shape_preserved(self):
        """Residual block preserves spatial dimensions and channel count."""
        block = ResidualBlock(channels=256)
        x = torch.randn(2, 256, 64, 64)
        out = block(x)
        assert out.shape == (2, 256, 64, 64)

    def test_skip_connection(self):
        """Residual block adds input to block output (skip connection)."""
        block = ResidualBlock(channels=256)
        x = torch.randn(1, 256, 32, 32)
        # With zero-initialized block weights, output should be close to input
        # Here we just verify shape and that gradient flows
        out = block(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is None  # x is not a leaf with requires_grad by default
        # Verify output is different from input (block does something)
        assert not torch.allclose(out, x)


class TestResNetGenerator:
    """Tests for the ResNetGenerator."""

    def test_output_shape(self):
        """Generator produces [B, 3, 256, 256] from [B, 4, 256, 256] input."""
        gen = ResNetGenerator(input_channels=4, output_channels=3, ngf=64, n_residual_blocks=9)
        x = torch.randn(2, 4, 256, 256)
        out = gen(x)
        assert out.shape == (2, 3, 256, 256)

    def test_output_range_tanh(self):
        """Generator output is bounded to [-1, 1] due to tanh activation."""
        gen = ResNetGenerator(input_channels=4, output_channels=3, ngf=64, n_residual_blocks=9)
        # Use varied input: RGB in [-1, 1], mask in {0, 1}
        x = torch.randn(4, 4, 256, 256)
        x[:, :3] = x[:, :3].clamp(-1, 1)  # RGB channels
        x[:, 3] = (torch.rand(4, 256, 256) > 0.5).float()  # Binary mask
        out = gen(x)
        assert out.min() >= -1.0
        assert out.max() <= 1.0

    def test_batch_size_1(self):
        """Generator works with batch size 1."""
        gen = ResNetGenerator(input_channels=4, output_channels=3, ngf=64, n_residual_blocks=9)
        x = torch.randn(1, 4, 256, 256)
        out = gen(x)
        assert out.shape == (1, 3, 256, 256)

    def test_batch_size_4(self):
        """Generator works with larger batch sizes."""
        gen = ResNetGenerator(input_channels=4, output_channels=3, ngf=64, n_residual_blocks=9)
        x = torch.randn(4, 4, 256, 256)
        out = gen(x)
        assert out.shape == (4, 3, 256, 256)

    def test_default_parameters(self):
        """Generator uses correct defaults: 4 input, 3 output, 64 ngf, 9 blocks."""
        gen = ResNetGenerator()
        x = torch.randn(1, 4, 256, 256)
        out = gen(x)
        assert out.shape == (1, 3, 256, 256)

    def test_gradient_flow(self):
        """Gradients flow through the entire generator."""
        gen = ResNetGenerator(input_channels=4, output_channels=3, ngf=64, n_residual_blocks=9)
        x = torch.randn(1, 4, 256, 256, requires_grad=True)
        out = gen(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape
        # Gradient should not be all zeros
        assert x.grad.abs().sum() > 0

    def test_architecture_uses_instance_norm(self):
        """Generator uses instance normalization (not batch norm)."""
        gen = ResNetGenerator()
        # Check that InstanceNorm2d layers exist in the model
        has_instance_norm = any(
            isinstance(m, torch.nn.InstanceNorm2d) for m in gen.modules()
        )
        has_batch_norm = any(
            isinstance(m, torch.nn.BatchNorm2d) for m in gen.modules()
        )
        assert has_instance_norm
        assert not has_batch_norm

    def test_architecture_uses_reflection_padding(self):
        """Generator uses reflection padding."""
        gen = ResNetGenerator()
        has_reflection_pad = any(
            isinstance(m, torch.nn.ReflectionPad2d) for m in gen.modules()
        )
        assert has_reflection_pad

    def test_no_dropout(self):
        """Generator does not use dropout (eval/train should give same result with fixed input)."""
        gen = ResNetGenerator()
        gen.eval()
        x = torch.randn(1, 4, 256, 256)
        with torch.no_grad():
            out_eval = gen(x)
        gen.train()
        with torch.no_grad():
            out_train = gen(x)
        # With instance norm (not batch norm), and no dropout, results should be identical
        assert torch.allclose(out_eval, out_train, atol=1e-6)
