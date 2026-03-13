import torch
import torch.nn as nn

class ContinuumRemoval(nn.Module):
    def __init__(self , use_fp16 = False):
        super().__init__()
        self.use_fp16 = use_fp16
    def forward(self, x):
        # x shape: (B, C, H, W)
        # Continuum removal operates on the spectral dimension (C)

        if self.use_fp16:
            x = x.clamp(min=1e-6)
        
        B, C, H, W = x.shape
        
        # Rearrange to (B*H*W, C) - treat each pixel spectrum independently
        spectra = x.permute(0, 2, 3, 1).reshape(-1, C)  # (N, C)
        
        # Find the convex hull envelope (simplified: min-max linear interpolation)
        band_indices = torch.linspace(0, 1, C, device=x.device).unsqueeze(0)  # (1, C)
        
        spec_min = spectra.min(dim=1, keepdim=True).values
        spec_max = spectra.max(dim=1, keepdim=True).values
        
        # Normalize each spectrum by its convex hull
        continuum = spec_min + (spec_max - spec_min) * band_indices
        removed = spectra / (continuum + 1e-8)
        
        # Reshape back to (B, C, H, W)
        out = removed.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return out