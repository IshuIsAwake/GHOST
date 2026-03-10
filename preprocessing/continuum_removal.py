import torch
import torch.nn as nn

class ContinuumRemoval(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):        
        B, C, H, W = x.shape
        spectra = x.permute(0, 2, 3, 1).reshape(-1, C)
        band_indices = torch.linspace(0, 1, C, device=x.device).unsqueeze(0)
        
        spec_min = spectra.min(dim=1, keepdim=True).values
        spec_max = spectra.max(dim=1, keepdim=True).values
        
        continuum = spec_min + (spec_max - spec_min) * band_indices
        removed = spectra / (continuum + 1e-8)
        
        out = removed.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return out