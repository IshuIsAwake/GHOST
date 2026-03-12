import torch
import torch.nn as nn
from preprocessing.continuum_removal import ContinuumRemoval
from models.spectral_3d_block import Spectral3DStack
from models.se_block import SEBlock
from models.encoder_2d import Encoder2D
from models.decoder_2d import Decoder2D


class HyperspectralNet(nn.Module):
    """
    Full pipeline:
      ContinuumRemoval -> Spectral3DStack -> SEBlock -> Encoder2D -> Decoder2D

    num_bands and num_classes are read from the dataset automatically via
    HyperspectralDataset.num_bands and HyperspectralDataset.num_classes.
    Never hardcode them in train.py.

    Spectral3DStack outputs (B, num_filters * num_bands, H, W).
    SE and Encoder are instantiated with that channel count automatically.
    """

    def __init__(self, num_bands, num_classes, num_filters=8, num_blocks=3, base_filters=32):
        super().__init__()

        self.continuum_removal = ContinuumRemoval()

        self.spectral_3d = Spectral3DStack(
            num_bands=num_bands,
            num_filters=num_filters,
            num_blocks=num_blocks
        )

        spectral_out = self.spectral_3d.out_channels  # num_filters * num_bands

        self.se_block = SEBlock(channels=spectral_out)

        self.encoder = Encoder2D(
            in_channels=spectral_out,
            base_filters=base_filters
        )

        self.decoder = Decoder2D(
            num_classes=num_classes,
            base_filters=base_filters
        )

    def forward(self, x):
        with torch.autocast('cuda', enabled=False):
            x = self.continuum_removal(x.float())  # always fp32 — division ops overflow in fp16
        x = self.spectral_3d(x)
        x = self.se_block(x)
        b, skips = self.encoder(x)
        x = self.decoder(b, skips)
        return x