"""Unit tests for the E-ASPP module."""
import torch
import pytest

from src.model.easpp import EASPP


class TestEASPP:
    """Tests for the EASPP module."""

    @pytest.fixture
    def model(self):
        """Create an EASPP module instance."""
        return EASPP(in_channels=2048, out_channels=256)

    @pytest.fixture
    def input_tensor(self):
        """Create a typical input tensor [B, 2048, 16, 16]."""
        return torch.randn(2, 2048, 16, 16)

    def test_output_shape(self, model, input_tensor):
        """E-ASPP output shape should be [B, 256, 16, 16] for [B, 2048, 16, 16] input."""
        model.eval()
        with torch.no_grad():
            output = model(input_tensor)
        assert output.shape == (2, 256, 16, 16)

    def test_output_shape_batch_size_1(self, model):
        """E-ASPP should work with batch size 1."""
        model.eval()
        x = torch.randn(1, 2048, 16, 16)
        with torch.no_grad():
            output = model(x)
        assert output.shape == (1, 256, 16, 16)

    def test_spatial_dimensions_preserved(self, model):
        """Spatial dimensions should be preserved from input to output."""
        model.eval()
        x = torch.randn(1, 2048, 16, 16)
        with torch.no_grad():
            output = model(x)
        assert output.shape[2:] == x.shape[2:]

    def test_dsc_branches_count(self, model):
        """There should be exactly 4 DSC branches with different dilation rates."""
        assert len(model.dsc_branches) == 4

    def test_dsc_branch_dilation_rates(self, model):
        """DSC branches should use dilation rates 3, 6, 12, 18."""
        expected_dilations = [3, 6, 12, 18]
        for i, branch in enumerate(model.dsc_branches):
            # The DepthwiseSeparableConv is the first element in the Sequential
            dsc = branch[0]
            actual_dilation = dsc.depthwise.dilation[0]
            assert actual_dilation == expected_dilations[i], (
                f"Branch {i} expected dilation {expected_dilations[i]}, got {actual_dilation}"
            )

    def test_dsc_branches_output_256_channels(self, model):
        """Each DSC branch should output 256 channels."""
        model.eval()
        x = torch.randn(1, 2048, 16, 16)
        with torch.no_grad():
            for branch in model.dsc_branches:
                out = branch(x)
                assert out.shape[1] == 256

    def test_global_pool_branch_output(self, model):
        """Global pooling branch should output 256 channels at the input spatial size."""
        model.eval()
        x = torch.randn(1, 2048, 16, 16)
        with torch.no_grad():
            global_feat = model.global_pool(x)
            assert global_feat.shape == (1, 2048, 1, 1)
            global_feat = model.global_conv(global_feat)
            assert global_feat.shape == (1, 256, 1, 1)
            global_feat = model.global_bn(global_feat)
            global_feat = model.global_relu(global_feat)
            global_feat = torch.nn.functional.interpolate(
                global_feat, size=(16, 16), mode='bilinear', align_corners=False
            )
            assert global_feat.shape == (1, 256, 16, 16)

    def test_concatenation_produces_1280_channels(self, model):
        """Concatenation of 5 branches should produce 1280 channels."""
        model.eval()
        x = torch.randn(1, 2048, 16, 16)
        with torch.no_grad():
            branch_outputs = [branch(x) for branch in model.dsc_branches]
            global_feat = model.global_pool(x)
            global_feat = model.global_conv(global_feat)
            global_feat = model.global_bn(global_feat)
            global_feat = model.global_relu(global_feat)
            global_feat = torch.nn.functional.interpolate(
                global_feat, size=(16, 16), mode='bilinear', align_corners=False
            )
            branch_outputs.append(global_feat)
            concat = torch.cat(branch_outputs, dim=1)
            assert concat.shape == (1, 1280, 16, 16)

    def test_soa_applied_to_concatenated(self, model):
        """SOA module should accept 1280 channel input."""
        assert model.soa.channels == 1280

    def test_projection_reduces_to_256(self, model):
        """Final projection should reduce from 1280 to 256 channels."""
        # Check that the first conv in the project Sequential has correct in/out channels
        proj_conv = model.project[0]
        assert proj_conv.in_channels == 1280
        assert proj_conv.out_channels == 256

    def test_no_nan_in_output(self, model, input_tensor):
        """Output should not contain NaN values."""
        model.eval()
        with torch.no_grad():
            output = model(input_tensor)
        assert not torch.isnan(output).any()

    def test_gradient_flow(self, model, input_tensor):
        """Gradients should flow through the module during training."""
        model.train()
        input_tensor.requires_grad_(True)
        output = model(input_tensor)
        loss = output.sum()
        loss.backward()
        assert input_tensor.grad is not None
        assert not torch.isnan(input_tensor.grad).any()

    def test_custom_in_out_channels(self):
        """EASPP should work with custom in/out channel counts."""
        model = EASPP(in_channels=512, out_channels=128)
        model.eval()
        x = torch.randn(1, 512, 16, 16)
        with torch.no_grad():
            output = model(x)
        assert output.shape == (1, 128, 16, 16)
