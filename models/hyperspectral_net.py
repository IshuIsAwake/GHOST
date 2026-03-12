import torch
import torch.nn as nn
from preprocessing.continuum_removal import ContinuumRemoval
from models.spectral_3d_block import Spectral3DStack
from models.se_block import SEBlock
from models.encoder_2d import Encoder2D
from models.decoder_2d import Decoder2D

class HyperspectralNet(nn.Module):
    def __init__(self, num_bands, num_classes, num_filters=8, num_blocks=3, base_filters=32):
        super().__init__()

        self.continuum_removal = ContinuumRemoval()

        self.spectral_3d = Spectral3DStack(
            num_bands=num_bands,
            num_filters=num_filters,
            num_blocks=num_blocks
        )

        spectral_out_channels = self.spectral_3d.out_channels

        self.se_block = SEBlock(channels=spectral_out_channels)

        self.encoder = Encoder2D(
            in_channels=spectral_out_channels,
            base_filters=base_filters
        )

        self.decoder = Decoder2D(
            num_classes=num_classes,
            base_filters=base_filters
        )

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.continuum_removal(x)   # (B, C, H, W)
        x = self.spectral_3d(x)         # (B, num_filters*C, H, W)
        x = self.se_block(x)            # (B, num_filters*C, H, W)
        b, skips = self.encoder(x)      # bottleneck + skip connections
        x = self.decoder(b, skips)      # (B, num_classes, H, W)
        return x