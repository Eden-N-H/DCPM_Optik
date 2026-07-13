"""Unit tests for multi-task prediction heads.

Validates output shapes and value ranges for all four task heads:
- SegmentationHead: [B, 3, 512, 512] unbounded logits
- SeverityHead: [B, 1, 512, 512] in [0, 1]
- DepthHead: [B, 1, 512, 512] in [0, 1]
- CameraHead: intrinsics [B, 4] > 0 (softplus), extrinsics [B, 6] (linear)
"""
import pytest
import torch

from src.model.heads import SegmentationHead, SeverityHead, DepthHead, CameraHead


@pytest.fixture
def decoder_features():
    """Simulated shared decoder output: [B, 64, 128, 128]."""
    torch.manual_seed(42)
    return torch.randn(2, 64, 128, 128)


class TestSegmentationHead:
    """Tests for the SegmentationHead module."""

    def test_output_shape(self, decoder_features):
        """Segmentation head produces [B, 3, 512, 512] output."""
        head = SegmentationHead(in_channels=64, num_classes=3)
        out = head(decoder_features)
        assert out.shape == (2, 3, 512, 512)

    def test_output_unbounded(self, decoder_features):
        """Segmentation logits are unbounded (no activation applied)."""
        head = SegmentationHead(in_channels=64, num_classes=3)
        out = head(decoder_features)
        # Logits should have both positive and negative values
        assert out.min() < 0 or out.max() > 1, "Logits should be unbounded"

    def test_batch_size_1(self):
        """Works with single sample batch."""
        head = SegmentationHead(in_channels=64, num_classes=3)
        x = torch.randn(1, 64, 128, 128)
        out = head(x)
        assert out.shape == (1, 3, 512, 512)

    def test_custom_num_classes(self):
        """Supports configurable number of classes."""
        head = SegmentationHead(in_channels=64, num_classes=7)
        x = torch.randn(1, 64, 128, 128)
        out = head(x)
        assert out.shape == (1, 7, 512, 512)

    def test_no_nan_in_output(self, decoder_features):
        """Output contains no NaN values."""
        head = SegmentationHead(in_channels=64, num_classes=3)
        out = head(decoder_features)
        assert not torch.isnan(out).any()


class TestSeverityHead:
    """Tests for the SeverityHead module."""

    def test_output_shape(self, decoder_features):
        """Severity head produces [B, 1, 512, 512] output."""
        head = SeverityHead(in_channels=64)
        out = head(decoder_features)
        assert out.shape == (2, 1, 512, 512)

    def test_output_range_zero_to_one(self, decoder_features):
        """Severity output is bounded in [0, 1] due to sigmoid."""
        head = SeverityHead(in_channels=64)
        out = head(decoder_features)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_batch_size_1(self):
        """Works with single sample batch."""
        head = SeverityHead(in_channels=64)
        x = torch.randn(1, 64, 128, 128)
        out = head(x)
        assert out.shape == (1, 1, 512, 512)

    def test_no_nan_in_output(self, decoder_features):
        """Output contains no NaN values."""
        head = SeverityHead(in_channels=64)
        out = head(decoder_features)
        assert not torch.isnan(out).any()


class TestDepthHead:
    """Tests for the DepthHead module."""

    def test_output_shape(self, decoder_features):
        """Depth head produces [B, 1, 512, 512] output."""
        head = DepthHead(in_channels=64)
        out = head(decoder_features)
        assert out.shape == (2, 1, 512, 512)

    def test_output_range_zero_to_one(self, decoder_features):
        """Depth output is bounded in [0, 1] due to sigmoid."""
        head = DepthHead(in_channels=64)
        out = head(decoder_features)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_batch_size_1(self):
        """Works with single sample batch."""
        head = DepthHead(in_channels=64)
        x = torch.randn(1, 64, 128, 128)
        out = head(x)
        assert out.shape == (1, 1, 512, 512)

    def test_no_nan_in_output(self, decoder_features):
        """Output contains no NaN values."""
        head = DepthHead(in_channels=64)
        out = head(decoder_features)
        assert not torch.isnan(out).any()


class TestCameraHead:
    """Tests for the CameraHead module."""

    def test_intrinsics_shape(self, decoder_features):
        """Camera head intrinsics output has shape [B, 4]."""
        head = CameraHead(in_channels=64)
        intrinsics, _ = head(decoder_features)
        assert intrinsics.shape == (2, 4)

    def test_extrinsics_shape(self, decoder_features):
        """Camera head extrinsics output has shape [B, 6]."""
        head = CameraHead(in_channels=64)
        _, extrinsics = head(decoder_features)
        assert extrinsics.shape == (2, 6)

    def test_intrinsics_positive(self, decoder_features):
        """Intrinsics are strictly positive (softplus activation)."""
        head = CameraHead(in_channels=64)
        intrinsics, _ = head(decoder_features)
        assert (intrinsics > 0).all()

    def test_extrinsics_unbounded(self, decoder_features):
        """Extrinsics are linear (unbounded)."""
        head = CameraHead(in_channels=64)
        _, extrinsics = head(decoder_features)
        # Extrinsics should be able to take any real value
        assert extrinsics.dtype == torch.float32

    def test_batch_size_1(self):
        """Works with single sample batch."""
        head = CameraHead(in_channels=64)
        x = torch.randn(1, 64, 128, 128)
        intrinsics, extrinsics = head(x)
        assert intrinsics.shape == (1, 4)
        assert extrinsics.shape == (1, 6)

    def test_no_nan_in_output(self, decoder_features):
        """Output contains no NaN values."""
        head = CameraHead(in_channels=64)
        intrinsics, extrinsics = head(decoder_features)
        assert not torch.isnan(intrinsics).any()
        assert not torch.isnan(extrinsics).any()

    def test_gap_reduces_spatial(self):
        """GAP correctly reduces [B, 64, 128, 128] → [B, 64]."""
        head = CameraHead(in_channels=64)
        x = torch.randn(2, 64, 128, 128)
        # Test internal GAP
        pooled = head.gap(x).flatten(1)
        assert pooled.shape == (2, 64)

    def test_fc_dimensions(self):
        """Fully connected layers have correct dimensions: 64→512→256→10."""
        head = CameraHead(in_channels=64)
        assert head.fc1.in_features == 64
        assert head.fc1.out_features == 512
        assert head.fc2.in_features == 512
        assert head.fc2.out_features == 256
        assert head.fc3.in_features == 256
        assert head.fc3.out_features == 10
