"""Unit tests for ResNet50DSCEncoder."""
import torch
import pytest

from src.model.encoder import ResNet50DSCEncoder, DSCBottleneck
from src.model.dsc import DepthwiseSeparableConv


class TestResNet50DSCEncoder:
    """Tests for the ResNet-50 encoder with DSC in stages 3-4."""

    def test_output_shapes(self):
        """Forward pass produces correct shapes for all 4 stages."""
        encoder = ResNet50DSCEncoder(pretrained=False)
        encoder.eval()

        x = torch.randn(2, 3, 512, 512)
        with torch.no_grad():
            outputs = encoder(x)

        assert outputs['stage1'].shape == (2, 256, 128, 128)
        assert outputs['stage2'].shape == (2, 512, 64, 64)
        assert outputs['stage3'].shape == (2, 1024, 32, 32)
        assert outputs['stage4'].shape == (2, 2048, 16, 16)

    def test_output_is_dict_with_correct_keys(self):
        """Output is a Dict[str, Tensor] with stage1..stage4 keys."""
        encoder = ResNet50DSCEncoder(pretrained=False)
        encoder.eval()

        x = torch.randn(1, 3, 512, 512)
        with torch.no_grad():
            outputs = encoder(x)

        assert isinstance(outputs, dict)
        assert set(outputs.keys()) == {'stage1', 'stage2', 'stage3', 'stage4'}
        for key, tensor in outputs.items():
            assert isinstance(tensor, torch.Tensor)

    def test_stages_3_4_use_dsc_bottleneck(self):
        """Stages 3 and 4 use DSCBottleneck blocks (not standard Bottleneck)."""
        encoder = ResNet50DSCEncoder(pretrained=False)

        for block in encoder.stage3:
            assert isinstance(block, DSCBottleneck), (
                f"Stage3 should use DSCBottleneck, got {type(block).__name__}"
            )
        for block in encoder.stage4:
            assert isinstance(block, DSCBottleneck), (
                f"Stage4 should use DSCBottleneck, got {type(block).__name__}"
            )

    def test_stages_1_2_use_standard_convolutions(self):
        """Stages 1 and 2 use standard ResNet Bottleneck blocks (not DSC)."""
        encoder = ResNet50DSCEncoder(pretrained=False)

        for block in encoder.stage1:
            assert not isinstance(block, DSCBottleneck), (
                "Stage1 should use standard Bottleneck, not DSCBottleneck"
            )
        for block in encoder.stage2:
            assert not isinstance(block, DSCBottleneck), (
                "Stage2 should use standard Bottleneck, not DSCBottleneck"
            )

    def test_pretrained_weight_loading(self):
        """Pretrained encoder loads ImageNet weights for stages 1-2."""
        encoder = ResNet50DSCEncoder(pretrained=True)

        # Stem conv1 should have non-trivial weights (not random init pattern)
        stem_weight = encoder.stem[0].weight.data
        assert stem_weight.abs().sum() > 0, "Stem weights should be non-zero"
        # Known property: ResNet-50 ImageNet stem conv weight norm is ~12
        assert stem_weight.norm() > 5.0, "Stem weights should match pretrained values"

    def test_batch_size_flexibility(self):
        """Encoder handles various batch sizes (1 to 4)."""
        encoder = ResNet50DSCEncoder(pretrained=False)
        encoder.eval()

        for batch_size in [1, 2, 4]:
            x = torch.randn(batch_size, 3, 512, 512)
            with torch.no_grad():
                outputs = encoder(x)
            assert outputs['stage1'].shape[0] == batch_size

    def test_dsc_bottleneck_uses_depthwise_separable(self):
        """DSCBottleneck conv2 is a DepthwiseSeparableConv."""
        encoder = ResNet50DSCEncoder(pretrained=False)
        first_block = list(encoder.stage3.children())[0]
        assert isinstance(first_block.conv2, DepthwiseSeparableConv)

    def test_stage_block_counts(self):
        """Stage 3 has 6 blocks and stage 4 has 3 blocks (matching ResNet-50)."""
        encoder = ResNet50DSCEncoder(pretrained=False)
        assert len(encoder.stage3) == 6, f"Stage3 should have 6 blocks, got {len(encoder.stage3)}"
        assert len(encoder.stage4) == 3, f"Stage4 should have 3 blocks, got {len(encoder.stage4)}"
