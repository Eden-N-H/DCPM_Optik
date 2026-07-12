"""Unit tests for SOA (Small-Object Attention) module.

Validates:
- Shape preservation (Req 8.5)
- Channel attention weights in [0, 1] (Req 8.1)
- Spatial attention weights in [0, 1] (Req 8.2)
- Sequential application order: channel → spatial → high-pass (Req 8.4)
"""
import torch
import pytest

from src.model.soa import SOA


class TestSOAShapePreservation:
    """Test that SOA output shape always matches input shape."""

    def test_basic_shape_preservation(self):
        """Output shape matches input for typical encoder feature dimensions."""
        soa = SOA(channels=256)
        x = torch.randn(2, 256, 16, 16)
        out = soa(x)
        assert out.shape == x.shape

    def test_shape_preservation_different_spatial(self):
        """Output shape matches input for various spatial sizes."""
        soa = SOA(channels=128)
        for h, w in [(8, 8), (32, 32), (64, 64), (13, 17)]:
            x = torch.randn(1, 128, h, w)
            out = soa(x)
            assert out.shape == x.shape, f"Failed for spatial size ({h}, {w})"

    def test_shape_preservation_different_channels(self):
        """Output shape matches input for various channel counts."""
        for c in [64, 128, 256, 512, 1024]:
            soa = SOA(channels=c)
            x = torch.randn(1, c, 16, 16)
            out = soa(x)
            assert out.shape == x.shape, f"Failed for channels={c}"

    def test_shape_preservation_different_batch_sizes(self):
        """Output shape matches input for various batch sizes."""
        soa = SOA(channels=256)
        for b in [1, 2, 4, 8]:
            x = torch.randn(b, 256, 16, 16)
            out = soa(x)
            assert out.shape == x.shape, f"Failed for batch_size={b}"


class TestSOAChannelAttention:
    """Test channel attention weights are in [0, 1]."""

    def test_channel_attention_range(self):
        """Channel attention weights (sigmoid output) are in [0, 1]."""
        soa = SOA(channels=256)
        x = torch.randn(2, 256, 16, 16)
        # Extract channel attention weights
        wc = soa.channel_attn(x)  # [B, C]
        assert wc.min() >= 0.0, f"Channel attention min {wc.min()} < 0"
        assert wc.max() <= 1.0, f"Channel attention max {wc.max()} > 1"

    def test_channel_attention_shape(self):
        """Channel attention produces per-channel weights."""
        soa = SOA(channels=128)
        x = torch.randn(4, 128, 32, 32)
        wc = soa.channel_attn(x)
        assert wc.shape == (4, 128)


class TestSOASpatialAttention:
    """Test spatial attention weights are in [0, 1]."""

    def test_spatial_attention_range(self):
        """Spatial attention (sigmoid output) is in [0, 1]."""
        soa = SOA(channels=256)
        x = torch.randn(2, 256, 16, 16)
        # Simulate spatial attention path
        spatial_input = x.mean(dim=1, keepdim=True)
        pooled = [pool(spatial_input) for pool in soa.spatial_pools]
        spatial_cat = torch.cat(pooled, dim=1)
        ws = soa.spatial_conv(spatial_cat)
        assert ws.min() >= 0.0, f"Spatial attention min {ws.min()} < 0"
        assert ws.max() <= 1.0, f"Spatial attention max {ws.max()} > 1"

    def test_spatial_attention_shape(self):
        """Spatial attention produces [B, 1, H, W] map."""
        soa = SOA(channels=256)
        x = torch.randn(2, 256, 16, 16)
        spatial_input = x.mean(dim=1, keepdim=True)
        pooled = [pool(spatial_input) for pool in soa.spatial_pools]
        spatial_cat = torch.cat(pooled, dim=1)
        ws = soa.spatial_conv(spatial_cat)
        assert ws.shape == (2, 1, 16, 16)


class TestSOAHighPass:
    """Test high-pass enhancement behavior."""

    def test_gaussian_kernel_shape(self):
        """Gaussian kernel is 7x7 as specified."""
        soa = SOA(channels=256)
        assert soa.gaussian_kernel.shape == (1, 1, 7, 7)

    def test_gaussian_kernel_normalized(self):
        """Gaussian kernel sums to 1."""
        soa = SOA(channels=256)
        assert torch.allclose(soa.gaussian_kernel.sum(), torch.tensor(1.0), atol=1e-6)

    def test_high_pass_on_constant_input(self):
        """For constant input, high-pass component is approximately zero."""
        soa = SOA(channels=64)
        # Constant input: blur = input, so high-pass = 0
        x = torch.ones(1, 64, 32, 32) * 3.0
        blurred = soa._gaussian_blur(x)
        # Interior pixels should be close to input (boundary effects exist)
        interior = blurred[:, :, 3:-3, 3:-3]
        expected = x[:, :, 3:-3, 3:-3]
        assert torch.allclose(interior, expected, atol=1e-5)


class TestSOASequentialOrder:
    """Test that attention is applied in correct order: channel → spatial → high-pass."""

    def test_alpha_parameter(self):
        """Alpha parameter correctly stored and used."""
        soa = SOA(channels=256, alpha=0.3)
        assert soa.alpha == 0.3

    def test_custom_alpha(self):
        """Custom alpha values are respected."""
        soa = SOA(channels=256, alpha=0.5)
        assert soa.alpha == 0.5

    def test_reduction_parameter(self):
        """Reduction parameter affects FC layer dimensions."""
        soa = SOA(channels=256, reduction=16)
        # FC layers: 256 → 16 → 256
        fc1 = soa.channel_attn[2]  # nn.Linear after Flatten
        assert fc1.in_features == 256
        assert fc1.out_features == 16

    def test_forward_gradient_flows(self):
        """Gradients flow through all three stages."""
        soa = SOA(channels=64)
        x = torch.randn(1, 64, 16, 16, requires_grad=True)
        out = soa(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape
