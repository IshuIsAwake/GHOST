"""Per-pixel spectral classifier: a 1-D dilated ResNet encoder and an MLP head. No spatial operators."""
from __future__ import annotations

import torch.nn as nn

POOLS = ('avg', 'flatten')


class ResBlock1D(nn.Module):
    """Two dilated convolutions and a residual connection; band length and channel width are preserved."""

    def __init__(self, channels: int, kernel_size: int, dilation: int):
        super().__init__()
        pad = dilation * (kernel_size // 2)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation, padding=pad, bias=False)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation, padding=pad, bias=False)
        self.bn2 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + x)


class DilatedResNet1D(nn.Module):
    """(N, B) or (N, 1, B) spectra → (N, embed_dim). Blocks with dilation 1, 2, 4, … are added only while the
    cumulative receptive field fits inside the band count."""

    def __init__(self, num_bands: int, channels: int = 64, embed_dim: int = 128, max_depth: int = 5,
                 kernel_size: int = 7, pool: str = 'avg'):
        super().__init__()
        if pool not in POOLS:
            raise ValueError(f"pool must be one of {POOLS}, got '{pool}'")
        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd, got {kernel_size}")
        k = kernel_size if kernel_size <= num_bands else max(3, num_bands if num_bands % 2 else num_bands - 1)
        self.num_bands, self.kernel_size, self.pool = num_bands, k, pool

        self.stem = nn.Sequential(
            nn.Conv1d(1, channels, k, padding=k // 2, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
        )
        rf, dilations = k, []
        for i in range(max_depth):
            gain = 2 * (k - 1) * 2 ** i
            if rf + gain > num_bands:
                break
            rf += gain
            dilations.append(2 ** i)
        self.dilations, self.receptive_field = dilations, rf
        self.blocks = nn.ModuleList(ResBlock1D(channels, k, d) for d in dilations)
        self.proj = nn.Linear(channels * num_bands if pool == 'flatten' else channels, embed_dim)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        if x.shape[-1] != self.num_bands:
            raise ValueError(f"The model expects {self.num_bands} bands, got {x.shape[-1]}")
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        x = x.flatten(1) if self.pool == 'flatten' else x.mean(dim=-1)
        return self.proj(x)


class MLPHead(nn.Module):
    def __init__(self, embed_dim: int, hidden: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def default_config(num_bands: int, num_classes: int, **overrides) -> dict:
    config = {'num_bands': int(num_bands), 'num_classes': int(num_classes), 'channels': 64, 'embed_dim': 128,
              'depth': 5, 'kernel_size': 7, 'head_hidden': 128, 'dropout': 0.3, 'pool': 'avg'}
    unknown = set(overrides) - set(config)
    if unknown:
        raise ValueError(f"Unknown model settings: {sorted(unknown)}")
    config.update(overrides)
    return config


class SpectralNet(nn.Module):
    """Encoder + head; (N, B) spectra → (N, num_classes) logits."""

    def __init__(self, config: dict):
        super().__init__()
        self.config = dict(config)
        self.encoder = DilatedResNet1D(config['num_bands'], config['channels'], config['embed_dim'],
                                       config['depth'], config['kernel_size'], config['pool'])
        self.head = MLPHead(config['embed_dim'], config['head_hidden'], config['num_classes'], config['dropout'])

    def forward(self, x):
        return self.head(self.encoder(x))


def build_model(config: dict) -> SpectralNet:
    return SpectralNet(config)
