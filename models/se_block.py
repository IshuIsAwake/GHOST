import torch
import torch.nn as nn

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, H, W = x.shape
        w = self.pool(x).view(B, C)      # (B, C)
        w = self.fc(w).view(B, C, 1, 1)  # (B, C, 1, 1)
        return x * w                      # rescale each channel