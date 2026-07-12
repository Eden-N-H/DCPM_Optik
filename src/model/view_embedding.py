"""View Embedding module for multi-view conditioning."""
import torch
import torch.nn as nn


class ViewEmbedding(nn.Module):
    """32-dim learnable view conditioning via spatial broadcast + concatenation.

    Maps a discrete view label (e.g. 0=dashcam, 1=drone) to a learned embedding
    vector, then broadcasts it spatially and concatenates with input features.
    """

    def __init__(self, num_views=2, embed_dim=32):
        super().__init__()
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(num_views, embed_dim)

    def forward(self, features, view_label):
        """Apply view embedding via spatial broadcast.

        Args:
            features: [B, C, H, W] feature tensor
            view_label: [B] integer tensor (0=dashcam, 1=drone)

        Returns:
            [B, C + embed_dim, H, W] concatenated features
        """
        emb = self.embedding(view_label)  # [B, embed_dim]
        emb = emb.unsqueeze(-1).unsqueeze(-1)  # [B, embed_dim, 1, 1]
        emb = emb.expand(-1, -1, features.shape[2], features.shape[3])  # [B, embed_dim, H, W]
        return torch.cat([features, emb], dim=1)  # [B, C + embed_dim, H, W]
