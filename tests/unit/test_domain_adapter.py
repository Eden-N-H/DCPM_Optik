"""Unit tests for GradientReversalLayer, DomainDiscriminator, and DualDomainAdapter."""
import pytest
import torch
import torch.nn as nn

from src.model.domain_adapter import (
    GradientReversalFunction,
    GradientReversalLayer,
    DomainDiscriminator,
    DualDomainAdapter,
)


class TestGradientReversalLayer:
    """Tests for GRL forward and backward semantics."""

    def test_forward_returns_input_unchanged(self):
        """GRL forward pass should return the input tensor values unchanged."""
        grl = GradientReversalLayer(lambda_val=0.1)
        x = torch.randn(2, 256, 16, 16)
        out = grl(x)
        assert torch.allclose(out, x), "GRL forward should return input unchanged"

    def test_forward_output_shape_matches_input(self):
        """GRL forward output should have same shape as input."""
        grl = GradientReversalLayer(lambda_val=0.5)
        x = torch.randn(4, 3, 32, 32)
        out = grl(x)
        assert out.shape == x.shape

    def test_backward_negates_gradient(self):
        """GRL backward should negate gradient scaled by lambda."""
        lambda_val = 0.1
        grl = GradientReversalLayer(lambda_val=lambda_val)

        x = torch.randn(2, 256, 4, 4, requires_grad=True)
        out = grl(x)
        loss = out.sum()
        loss.backward()

        # Gradient of sum w.r.t. x without GRL would be all ones.
        # With GRL, it should be -lambda_val * ones.
        expected_grad = -lambda_val * torch.ones_like(x)
        assert torch.allclose(x.grad, expected_grad, atol=1e-6), (
            f"Expected gradient to be -{lambda_val} * ones, got {x.grad.mean().item()}"
        )

    def test_backward_with_different_lambda(self):
        """GRL backward with lambda=0.5 should scale gradient by -0.5."""
        lambda_val = 0.5
        grl = GradientReversalLayer(lambda_val=lambda_val)

        x = torch.randn(1, 64, 8, 8, requires_grad=True)
        out = grl(x)
        # Use mean so gradient per element is 1/numel
        loss = out.mean()
        loss.backward()

        numel = x.numel()
        expected_grad = -lambda_val * torch.ones_like(x) / numel
        assert torch.allclose(x.grad, expected_grad, atol=1e-7)

    def test_backward_with_lambda_zero(self):
        """GRL with lambda=0 should produce zero gradients."""
        grl = GradientReversalLayer(lambda_val=0.0)

        x = torch.randn(2, 128, 4, 4, requires_grad=True)
        out = grl(x)
        loss = out.sum()
        loss.backward()

        assert torch.allclose(x.grad, torch.zeros_like(x)), (
            "GRL with lambda=0 should zero out gradients"
        )

    def test_backward_with_lambda_one(self):
        """GRL with lambda=1.0 should fully negate gradients."""
        grl = GradientReversalLayer(lambda_val=1.0)

        x = torch.randn(1, 32, 2, 2, requires_grad=True)
        out = grl(x)
        loss = out.sum()
        loss.backward()

        expected_grad = -1.0 * torch.ones_like(x)
        assert torch.allclose(x.grad, expected_grad, atol=1e-6)

    def test_gradient_flows_through_downstream_layers(self):
        """GRL should properly reverse gradients in a computation graph."""
        lambda_val = 0.1
        grl = GradientReversalLayer(lambda_val=lambda_val)
        linear = nn.Conv2d(64, 1, 1)

        x = torch.randn(1, 64, 4, 4, requires_grad=True)
        out = grl(x)
        pred = linear(out)
        loss = pred.sum()
        loss.backward()

        # x should have gradients (negated and scaled)
        assert x.grad is not None
        assert x.grad.shape == x.shape


class TestDomainDiscriminator:
    """Tests for DomainDiscriminator architecture and output shapes."""

    def test_feature_discriminator_output_shape(self):
        """Feature disc with 256-ch input at 16x16 should produce [B, 1, 2, 2]."""
        disc = DomainDiscriminator(in_channels=256)
        x = torch.randn(2, 256, 16, 16)
        out = disc(x)
        # 16 -> 8 -> 4 -> 2 (three stride-2 convs)
        assert out.shape == (2, 1, 2, 2)

    def test_logit_discriminator_output_shape(self):
        """Logit disc with 3-ch input at 512x512 should reduce spatial dims."""
        disc = DomainDiscriminator(in_channels=3)
        x = torch.randn(1, 3, 512, 512)
        out = disc(x)
        # 512 -> 256 -> 128 -> 64 (three stride-2 convs)
        assert out.shape == (1, 1, 64, 64)

    def test_discriminator_channel_architecture(self):
        """Verify the 3 conv layers have correct channel progression [in→256, 256→128, 128→1]."""
        disc = DomainDiscriminator(in_channels=256)
        layers = list(disc.layers.children())

        # Layer 0: Conv2d(256, 256, 3, stride=2)
        assert isinstance(layers[0], nn.Conv2d)
        assert layers[0].in_channels == 256
        assert layers[0].out_channels == 256
        assert layers[0].kernel_size == (3, 3)
        assert layers[0].stride == (2, 2)

        # Layer 1: LeakyReLU
        assert isinstance(layers[1], nn.LeakyReLU)

        # Layer 2: Conv2d(256, 128, 3, stride=2)
        assert isinstance(layers[2], nn.Conv2d)
        assert layers[2].in_channels == 256
        assert layers[2].out_channels == 128

        # Layer 3: LeakyReLU
        assert isinstance(layers[3], nn.LeakyReLU)

        # Layer 4: Conv2d(128, 1, 3, stride=2)
        assert isinstance(layers[4], nn.Conv2d)
        assert layers[4].in_channels == 128
        assert layers[4].out_channels == 1

    def test_discriminator_no_activation_after_last_layer(self):
        """Last conv layer should not be followed by activation."""
        disc = DomainDiscriminator(in_channels=256)
        layers = list(disc.layers.children())
        # Should be 5 layers: conv, lrelu, conv, lrelu, conv
        assert len(layers) == 5
        assert isinstance(layers[-1], nn.Conv2d)

    def test_discriminator_leaky_relu_slope(self):
        """LeakyReLU should use negative_slope=0.2."""
        disc = DomainDiscriminator(in_channels=256)
        layers = list(disc.layers.children())
        for layer in layers:
            if isinstance(layer, nn.LeakyReLU):
                assert layer.negative_slope == pytest.approx(0.2)

    def test_discriminator_different_input_channels(self):
        """Discriminator should work with different input channel counts."""
        for in_ch in [3, 64, 128, 256, 512]:
            disc = DomainDiscriminator(in_channels=in_ch)
            x = torch.randn(1, in_ch, 16, 16)
            out = disc(x)
            assert out.shape[1] == 1  # always 1 output channel

    def test_discriminator_batch_dimension(self):
        """Output batch dimension should match input batch dimension."""
        disc = DomainDiscriminator(in_channels=256)
        for batch_size in [1, 2, 4, 8]:
            x = torch.randn(batch_size, 256, 16, 16)
            out = disc(x)
            assert out.shape[0] == batch_size


class TestDualDomainAdapter:
    """Tests for DualDomainAdapter forward pass and configuration."""

    def test_default_lambda_adv(self):
        """Default lambda_adv should be 0.1."""
        adapter = DualDomainAdapter()
        assert adapter.lambda_adv == 0.1

    def test_forward_output_keys(self):
        """Forward should return dict with 'feat_pred' and 'logit_pred'."""
        adapter = DualDomainAdapter()
        features = torch.randn(2, 256, 16, 16)
        logits = torch.randn(2, 3, 512, 512)
        out = adapter(features, logits)
        assert 'feat_pred' in out
        assert 'logit_pred' in out

    def test_forward_feature_pred_shape(self):
        """Feature domain pred from [B, 256, 16, 16] input."""
        adapter = DualDomainAdapter()
        features = torch.randn(2, 256, 16, 16)
        logits = torch.randn(2, 3, 512, 512)
        out = adapter(features, logits)
        # 16 -> 8 -> 4 -> 2
        assert out['feat_pred'].shape == (2, 1, 2, 2)

    def test_forward_logit_pred_shape(self):
        """Logit domain pred from [B, 3, 512, 512] input."""
        adapter = DualDomainAdapter()
        features = torch.randn(2, 256, 16, 16)
        logits = torch.randn(2, 3, 512, 512)
        out = adapter(features, logits)
        # 512 -> 256 -> 128 -> 64
        assert out['logit_pred'].shape == (2, 1, 64, 64)

    def test_grl_lambda_propagated(self):
        """Both GRLs should use the specified lambda_adv."""
        adapter = DualDomainAdapter(lambda_adv=0.2)
        assert adapter.feature_grl.lambda_val == 0.2
        assert adapter.logit_grl.lambda_val == 0.2

    def test_gradient_reversal_in_backward(self):
        """Verify gradient reversal operates correctly through the adapter."""
        adapter = DualDomainAdapter(lambda_adv=0.1)

        features = torch.randn(1, 256, 16, 16, requires_grad=True)
        logits = torch.randn(1, 3, 32, 32, requires_grad=True)

        out = adapter(features, logits)
        loss = out['feat_pred'].sum() + out['logit_pred'].sum()
        loss.backward()

        # Gradients should exist and be non-zero
        assert features.grad is not None
        assert logits.grad is not None
        assert not torch.all(features.grad == 0)
        assert not torch.all(logits.grad == 0)

    def test_custom_channels(self):
        """Adapter should work with custom channel counts."""
        adapter = DualDomainAdapter(feature_channels=128, num_classes=5)
        features = torch.randn(1, 128, 16, 16)
        logits = torch.randn(1, 5, 64, 64)
        out = adapter(features, logits)
        assert out['feat_pred'].shape[1] == 1
        assert out['logit_pred'].shape[1] == 1
