import torch
import torch.nn as nn
from preprocessing.continuum_removal import ContinuumRemoval


class SpectralSSM(nn.Module):
    """
    Lightweight selective SSM.
    Treats each spectral band as a timestep in a sequence.

    The 'selective' part: A, B, C matrices are functions of the
    current input — the model decides per-band how much to remember
    vs. forget. This is what distinguishes Mamba from a vanilla RNN.

    Input:  (N, C)       — N pixel spectra, C bands (continuum-removed)
    Output: (N, d_model) — per-pixel spectral fingerprints
    """

    def __init__(self, d_model: int = 64, d_state: int = 16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # Project each band scalar → embedding
        self.input_proj = nn.Linear(1, d_model)

        # Log-parameterised decay: always negative after negation → stable
        self.A_log = nn.Parameter(torch.randn(d_model, d_state) * 0.1)

        # Input-dependent selection matrices (the selective mechanism)
        self.B_proj = nn.Linear(d_model, d_state, bias=False)
        self.C_proj = nn.Linear(d_model, d_state, bias=False)

        # Skip / residual connection weight
        self.D = nn.Parameter(torch.ones(d_model))

        self.out_proj = nn.Linear(d_model, d_model)
        self.norm     = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C)
        N, C = x.shape

        x_emb = self.input_proj(x.unsqueeze(-1))   # (N, C, d_model)
        A_exp = torch.exp(-torch.exp(self.A_log))   # (d_model, d_state) — stable decay

        h = torch.zeros(N, self.d_model, self.d_state,
                        device=x.device, dtype=x.dtype)
        y = torch.zeros(N, self.d_model,
                        device=x.device, dtype=x.dtype)

        for k in range(C):
            u   = x_emb[:, k, :]                            # (N, d_model)
            B_k = self.B_proj(u)                            # (N, d_state)
            C_k = self.C_proj(u)                            # (N, d_state)

            # h = A ⊙ h  +  u ⊗ B_k
            h = h * A_exp + u.unsqueeze(-1) * B_k.unsqueeze(1)

            # y = C_k · h  (contract over d_state)
            y = (h * C_k.unsqueeze(1)).sum(-1)              # (N, d_model)

        out = y + self.D * x_emb[:, -1, :]                 # D skip
        return self.norm(self.out_proj(out))                # (N, d_model)


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