"""Unit tests for ViewEmbedding module — validates Requirement 6.3."""
import torch
import pytest

from src.model.view_embedding import ViewEmbedding


class TestViewEmbedding:
    """Tests for ViewEmbedding module."""

    def test_output_shape(self):
        """Output shape is [B, 2080, H, W] for [B, 2048, 16, 16] input."""
        module = ViewEmbedding(num_views=2, embed_dim=32)
        B, C, H, W = 2, 2048, 16, 16
        features = torch.randn(B, C, H, W)
        view_label = torch.tensor([0, 1])

        output = module(features, view_label)

        assert output.shape == (B, 2080, H, W)

    def test_first_channels_equal_input(self):
        """First 2048 channels of output are identical to input features."""
        module = ViewEmbedding(num_views=2, embed_dim=32)
        B, C, H, W = 4, 2048, 16, 16
        features = torch.randn(B, C, H, W)
        view_label = torch.zeros(B, dtype=torch.long)

        output = module(features, view_label)

        assert torch.equal(output[:, :2048, :, :], features)

    def test_works_for_view_label_0(self):
        """Module produces correct output for dashcam view (label=0)."""
        module = ViewEmbedding(num_views=2, embed_dim=32)
        B, C, H, W = 3, 2048, 16, 16
        features = torch.randn(B, C, H, W)
        view_label = torch.zeros(B, dtype=torch.long)

        output = module(features, view_label)

        assert output.shape == (B, 2080, H, W)
        # Embedding channels should be spatially constant for each sample
        embedding_channels = output[:, 2048:, :, :]
        for b in range(B):
            for c in range(32):
                # All spatial locations should have the same value
                val = embedding_channels[b, c, 0, 0]
                assert torch.allclose(embedding_channels[b, c], val.expand(H, W))

    def test_works_for_view_label_1(self):
        """Module produces correct output for drone view (label=1)."""
        module = ViewEmbedding(num_views=2, embed_dim=32)
        B, C, H, W = 3, 2048, 16, 16
        features = torch.randn(B, C, H, W)
        view_label = torch.ones(B, dtype=torch.long)

        output = module(features, view_label)

        assert output.shape == (B, 2080, H, W)
        # Embedding channels should be spatially constant for each sample
        embedding_channels = output[:, 2048:, :, :]
        for b in range(B):
            for c in range(32):
                val = embedding_channels[b, c, 0, 0]
                assert torch.allclose(embedding_channels[b, c], val.expand(H, W))

    def test_different_views_produce_different_embeddings(self):
        """View labels 0 and 1 produce distinct embedding vectors."""
        module = ViewEmbedding(num_views=2, embed_dim=32)
        B, C, H, W = 1, 2048, 16, 16
        features = torch.randn(B, C, H, W)

        out_0 = module(features, torch.tensor([0]))
        out_1 = module(features, torch.tensor([1]))

        # The embedding channels (last 32) should differ between view labels
        emb_0 = out_0[:, 2048:, 0, 0]
        emb_1 = out_1[:, 2048:, 0, 0]
        assert not torch.equal(emb_0, emb_1)

    def test_mixed_batch_view_labels(self):
        """Batch with mixed view labels produces correct per-sample embeddings."""
        module = ViewEmbedding(num_views=2, embed_dim=32)
        B, C, H, W = 4, 2048, 16, 16
        features = torch.randn(B, C, H, W)
        view_label = torch.tensor([0, 1, 0, 1])

        output = module(features, view_label)

        assert output.shape == (B, 2080, H, W)
        # Samples with same view label should have same embedding
        emb_0_sample0 = output[0, 2048:, 0, 0]
        emb_0_sample2 = output[2, 2048:, 0, 0]
        assert torch.equal(emb_0_sample0, emb_0_sample2)

        emb_1_sample1 = output[1, 2048:, 0, 0]
        emb_1_sample3 = output[3, 2048:, 0, 0]
        assert torch.equal(emb_1_sample1, emb_1_sample3)
