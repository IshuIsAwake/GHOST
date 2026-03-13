import torch
import torch.nn as nn
from preprocessing.continuum_removal import ContinuumRemoval


class SpectralSSM(nn.Module):
    """
    Parallel multi-scale 1D CNN replacement for sequential RNN SSM.
    
    Solves vanishing gradients over 200+ bands while capturing the "selective" 
    spirit of SSMs by using parallel filters at different scales (narrow, mid, 
    wide) and selectively gating them via a channel attention mechanism.
    
    Input:  (N, C)       — N pixel spectra, C bands (continuum-removed)
    Output: (N, d_model) — per-pixel spectral fingerprints
    """

    def __init__(self, d_model: int = 64, d_state: int = 16):
        super().__init__()
        self.d_model = d_model
        
        # d_state becomes the number of filters per branch
        filters = d_state 
        
        # 1. Parallel Multi-Scale Feature Extraction
        self.narrow_branch = nn.Sequential(
            nn.Conv1d(1, filters, kernel_size=7, padding=3),
            nn.BatchNorm1d(filters),
            nn.GELU(),
            nn.Conv1d(filters, filters, kernel_size=7, padding=3),
            nn.BatchNorm1d(filters),
            nn.GELU()
        )
        
        self.mid_branch = nn.Sequential(
            nn.Conv1d(1, filters, kernel_size=15, padding=7),
            nn.BatchNorm1d(filters),
            nn.GELU(),
            nn.Conv1d(filters, filters, kernel_size=15, padding=7),
            nn.BatchNorm1d(filters),
            nn.GELU()
        )
        
        self.wide_branch = nn.Sequential(
            nn.Conv1d(1, filters, kernel_size=31, padding=15),
            nn.BatchNorm1d(filters),
            nn.GELU(),
            nn.Conv1d(filters, filters, kernel_size=31, padding=15),
            nn.BatchNorm1d(filters),
            nn.GELU()
        )
        
        total_filters = filters * 3
        
        # 2. Selective Gating (Channel Attention)
        self.se_gate = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(total_filters, total_filters // 2),
            nn.GELU(),
            nn.Linear(total_filters // 2, total_filters),
            nn.Sigmoid()
        )
        
        # 3. Projection to d_model fingerprint
        self.out_proj = nn.Sequential(
            nn.Linear(total_filters, d_model),
            nn.LayerNorm(d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C)
        x_1d = x.unsqueeze(1)  # (N, 1, C)
        
        # Extract multi-scale features
        f_narrow = self.narrow_branch(x_1d)  # (N, F, C)
        f_mid    = self.mid_branch(x_1d)     # (N, F, C)
        f_wide   = self.wide_branch(x_1d)    # (N, F, C)
        
        # Concatenate branches
        f_concat = torch.cat([f_narrow, f_mid, f_wide], dim=1)  # (N, 3F, C)
        
        # Selective Gating
        gate = self.se_gate(f_concat).unsqueeze(-1)  # (N, 3F, 1)
        f_gated = f_concat * gate                    # (N, 3F, C)
        
        # Global pool across bands to get fixed-size fingerprint
        fingerprint_raw = f_gated.mean(dim=2)        # (N, 3F)
        
        # Project to final d_model dim
        return self.out_proj(fingerprint_raw)        # (N, d_model)


class SpectralSSMEncoder(nn.Module):
    """
    ContinuumRemoval → SpectralSSM for full hyperspectral images.
    Physics-informed: SSM sees absorption-feature shapes, not raw reflectance.

    Input:  (B, C, H, W)
    Output: (B, d_model, H, W)
    """

    def __init__(self, d_model: int = 64, d_state: int = 16,
                 use_fp16: bool = False):
        super().__init__()
        self.continuum = ContinuumRemoval(use_fp16=use_fp16)
        self.ssm       = SpectralSSM(d_model=d_model, d_state=d_state)
        self.d_model   = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        with torch.autocast('cuda', enabled=False):
            x = self.continuum(x.float())                   # always fp32

        pixels       = x.permute(0, 2, 3, 1).reshape(-1, C)           # (B·H·W, C)
        fingerprints = self.ssm(pixels)                                 # (B·H·W, d_model)

        return (fingerprints
                .reshape(B, H, W, self.d_model)
                .permute(0, 3, 1, 2))                                   # (B, d_model, H, W)