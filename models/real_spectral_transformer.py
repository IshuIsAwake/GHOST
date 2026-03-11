import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralTransformer(nn.Module):
    """
    Treats each spatial pixel as an independent sequence of C band-tokens.
    Each token has dimensionality = embed_dim (output of Spectral3DStack).
    Self-attention learns cross-band relationships (e.g. co-occurring absorption features).
    Global average pool over the spectral dimension collapses variable C -> fixed embed_dim output.
    Positional embeddings are interpolated, so any band count works at inference.
    """

    def __init__(self, embed_dim=8, num_heads=4, num_layers=2, max_bands=500):
        super().__init__()
        self.embed_dim = embed_dim

        self.pos_embed = nn.Parameter(torch.randn(1, max_bands, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        # x: (B, embed_dim, C, H, W)
        B, num_f, C, H, W = x.shape

        # Each spatial pixel is one sequence of C band-tokens, each of dim num_f
        x = x.permute(0, 3, 4, 2, 1).reshape(B * H * W, C, num_f)

        # Interpolate positional embeddings to current band count
        pe = self.pos_embed.permute(0, 2, 1)                              # (1, embed_dim, max_bands)
        pe = F.interpolate(pe, size=C, mode='linear', align_corners=False) # (1, embed_dim, C)
        pe = pe.permute(0, 2, 1)                                           # (1, C, embed_dim)

        x = x + pe
        x = self.transformer(x)   # (B*H*W, C, embed_dim)

        # Collapse spectral dimension -> fixed-size spatial feature map
        x = x.mean(dim=1)                                         # (B*H*W, embed_dim)
        x = x.reshape(B, H, W, num_f).permute(0, 3, 1, 2)       # (B, embed_dim, H, W)
        return x