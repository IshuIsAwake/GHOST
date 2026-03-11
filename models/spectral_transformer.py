import torch
import torch.nn as nn

class SpectralTransformer(nn.Module):
    # Changed to accept out_dim. It defaults to 128 to give the U-Net breathing room.
    def __init__(self, embed_dim=8, out_dim=128):
        super().__init__()
        self.out_dim = out_dim
        
        # Project from 8 to 128 feature maps BEFORE collapsing the spectrum
        self.mixer = nn.Sequential(
            nn.Conv3d(embed_dim, out_dim, kernel_size=1),
            nn.GroupNorm(8, out_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # x: (B, embed_dim, C, H, W)
        x = self.mixer(x)    # Shapes to: (B, 128, C, H, W)
        x = x.mean(dim=2)    # Shapes to: (B, 128, H, W) -> C is dynamically collapsed
        return x