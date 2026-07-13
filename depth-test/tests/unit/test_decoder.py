"""Unit tests for the LightweightDecoder module."""
import torch
import pytest

from src.model.decoder import LightweightDecoder, DecoderBlock


class TestDecoderBlock:
    """Tests for individual DecoderBlock."""

    @pytest.fixture
    def block(self):
        """Create a decoder block matching block 1 config."""
        return DecoderBlock(in_channels=256, skip_channels=1024, out_channels=256)

    def test_output_shape(self, block):
        """DecoderBlock should produce correct output shape after upsample + concat + conv."""
        block.eval()
        x = torch.randn(2, 256, 16, 16)
        skip = torch.randn(2, 1024, 32, 32)
        with torch.no_grad():
            output = block(x, skip)
        assert output.shape == (2, 256, 32, 32)

    def test_upsamples_to_skip_spatial_size(self, block):
        """Block should upsample input to match skip connection spatial dims."""
        block.eval()
        x = torch.randn(1, 256, 16, 16)
        skip = torch.randn(1, 1024, 32, 32)
        with torch.no_grad():
            output = block(x, skip)
        assert output.shape[2:] == skip.shape[2:]

    def test_has_soa_module(self, block):
        """Each decoder block should contain an SOA module."""
        assert hasattr(block, 'soa')
        from src.model.soa import SOA
        assert isinstance(block.soa, SOA)

    def test_has_two_dsc_layers(self, block):
        """Each block should have two DSC convolution layers."""
        from src.model.dsc import DepthwiseSeparableConv
        assert isinstance(block.conv1, DepthwiseSeparableConv)
        assert isinstance(block.conv2, DepthwiseSeparableConv)

    def test_has_batch_norm(self, block):
        """Each block should have batch normalization after each DSC layer."""
        assert isinstance(block.bn1, torch.nn.BatchNorm2d)
        assert isinstance(block.bn2, torch.nn.BatchNorm2d)

    def test_no_nan_output(self, block):
        """Output should not contain NaN values."""
        block.eval()
        x = torch.randn(1, 256, 16, 16)
        skip = torch.randn(1, 1024, 32, 32)
        with torch.no_grad():
            output = block(x, skip)
        assert not torch.isnan(output).any()


class TestLightweightDecoder:
    """Tests for the LightweightDecoder module."""

    @pytest.fixture
    def model(self):
        """Create a LightweightDecoder instance."""
        return LightweightDecoder()

    @pytest.fixture
    def aspp_out(self):
        """Create typical EASPP output [B, 256, 16, 16]."""
        return torch.randn(2, 256, 16, 16)

    @pytest.fixture
    def encoder_features(self):
        """Create encoder features dict matching expected shapes."""
        return {
            'stage1': torch.randn(2, 256, 128, 128),
            'stage2': torch.randn(2, 512, 64, 64),
            'stage3': torch.randn(2, 1024, 32, 32),
        }

    def test_output_shape(self, model, aspp_out, encoder_features):
        """Decoder output should be [B, 64, 128, 128] for standard inputs."""
        model.eval()
        with torch.no_grad():
            output = model(aspp_out, encoder_features)
        assert output.shape == (2, 64, 128, 128)

    def test_output_shape_batch_size_1(self, model):
        """Decoder should work with batch size 1."""
        model.eval()
        aspp_out = torch.randn(1, 256, 16, 16)
        features = {
            'stage1': torch.randn(1, 256, 128, 128),
            'stage2': torch.randn(1, 512, 64, 64),
            'stage3': torch.randn(1, 1024, 32, 32),
        }
        with torch.no_grad():
            output = model(aspp_out, features)
        assert output.shape == (1, 64, 128, 128)

    def test_output_channels_64(self, model, aspp_out, encoder_features):
        """Final output should have exactly 64 channels."""
        model.eval()
        with torch.no_grad():
            output = model(aspp_out, encoder_features)
        assert output.shape[1] == 64

    def test_output_spatial_128x128(self, model, aspp_out, encoder_features):
        """Final output spatial size should be 128x128 (1/4 of 512x512 input)."""
        model.eval()
        with torch.no_grad():
            output = model(aspp_out, encoder_features)
        assert output.shape[2] == 128
        assert output.shape[3] == 128

    def test_three_decoder_blocks(self, model):
        """Decoder should have exactly 3 sequential blocks."""
        assert hasattr(model, 'block1')
        assert hasattr(model, 'block2')
        assert hasattr(model, 'block3')
        assert isinstance(model.block1, DecoderBlock)
        assert isinstance(model.block2, DecoderBlock)
        assert isinstance(model.block3, DecoderBlock)

    def test_block1_channels(self, model):
        """Block 1 should take 256+1024 input and produce 256 output channels."""
        block = model.block1
        # First DSC should accept 256 + 1024 = 1280 channels
        assert block.conv1.depthwise.in_channels == 256 + 1024
        assert block.bn1.num_features == 256

    def test_block2_channels(self, model):
        """Block 2 should take 256+512 input and produce 128 output channels."""
        block = model.block2
        # First DSC should accept 256 + 512 = 768 channels
        assert block.conv1.depthwise.in_channels == 256 + 512
        assert block.bn1.num_features == 128

    def test_block3_channels(self, model):
        """Block 3 should take 128+256 input and produce 64 output channels."""
        block = model.block3
        # First DSC should accept 128 + 256 = 384 channels
        assert block.conv1.depthwise.in_channels == 128 + 256
        assert block.bn1.num_features == 64

    def test_each_block_has_soa(self, model):
        """All three blocks should have SOA modules."""
        from src.model.soa import SOA
        assert isinstance(model.block1.soa, SOA)
        assert isinstance(model.block2.soa, SOA)
        assert isinstance(model.block3.soa, SOA)

    def test_no_nan_in_output(self, model, aspp_out, encoder_features):
        """Output should not contain NaN values."""
        model.eval()
        with torch.no_grad():
            output = model(aspp_out, encoder_features)
        assert not torch.isnan(output).any()

    def test_gradient_flow(self, model, aspp_out, encoder_features):
        """Gradients should flow through the decoder during training."""
        model.train()
        aspp_out.requires_grad_(True)
        output = model(aspp_out, encoder_features)
        loss = output.sum()
        loss.backward()
        assert aspp_out.grad is not None
        assert not torch.isnan(aspp_out.grad).any()

    def test_intermediate_block1_shape(self, model, aspp_out, encoder_features):
        """Block 1 should produce [B, 256, 32, 32]."""
        model.eval()
        with torch.no_grad():
            x = model.block1(aspp_out, encoder_features['stage3'])
        assert x.shape == (2, 256, 32, 32)

    def test_intermediate_block2_shape(self, model, aspp_out, encoder_features):
        """Block 2 should produce [B, 128, 64, 64]."""
        model.eval()
        with torch.no_grad():
            x = model.block1(aspp_out, encoder_features['stage3'])
            x = model.block2(x, encoder_features['stage2'])
        assert x.shape == (2, 128, 64, 64)
