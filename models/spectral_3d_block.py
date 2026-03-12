import torch
import torch.nn as nn

class Single3DBlock(nn.Module):
    def __init__(self, in_filters, out_filters, kernel_size=(7,3,3)):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_filters, out_filters, kernel_size=kernel_size,
                      padding=(kernel_size[0]//2, kernel_size[1]//2, kernel_size[2]//2)),
            nn.BatchNorm3d(out_filters),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)

class Spectral3DStack(nn.Module):
    def __init__(self, num_bands, num_filters=8, num_blocks=3, kernel_size=(7,3,3)):
        super().__init__()

        blocks = []
        in_f = 1
        for _ in range(num_blocks):
            blocks.append(Single3DBlock(in_f, num_filters, kernel_size))
            in_f = num_filters

        self.stack = nn.Sequential(*blocks)
        self.out_channels = num_filters * num_bands

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.unsqueeze(1)           # (B, 1, C, H, W)
        x = self.stack(x)            # (B, num_filters, C, H, W)
        B, F, C, H, W = x.shape
        x = x.reshape(B, F*C, H, W)  # (B, num_filters*C, H, W)
        return x