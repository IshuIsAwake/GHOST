import torch
import torch.nn as nn
from preprocessing.continuum_removal import ContinuumRemoval
from models.spectral_3d_block import Spectral3DStack
from models.spectral_transformer import SpectralTransformer
from models.encoder_2d import Encoder2D
from models.decoder_2d import Decoder2D

class HyperspectralNet(nn.Module):
    def __init__(self, num_bands, num_classes, num_filters=8, num_blocks=3, base_filters=32):
        super().__init__()

        self.continuum_removal = ContinuumRemoval()

        self.spectral_3d = Spectral3DStack(
            num_filters=num_filters,
            num_blocks=num_blocks
        )

        spectral_out_channels = self.spectral_3d.out_channels
        
        # New Projection Dimension
        projected_dim = 128

        self.spectral_transformer = SpectralTransformer(
            embed_dim=spectral_out_channels, 
            out_dim=projected_dim
        )

        # The Encoder now receives 128 rich feature channels instead of 8
        self.encoder = Encoder2D(
            in_channels=projected_dim, 
            base_filters=base_filters
        )

        self.decoder = Decoder2D(
            num_classes=num_classes,
            base_filters=base_filters
        )

    def forward(self, x):
        x = self.continuum_removal(x)      
        x = self.spectral_3d(x)            
        x = self.spectral_transformer(x)   
        b, skips = self.encoder(x)         
        x = self.decoder(b, skips)         
        return x