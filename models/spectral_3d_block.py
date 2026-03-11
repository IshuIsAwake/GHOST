import torch
import torch.nn as nn


class Single3DBlock(nn.Module):
    def __init__(self, in_filters, out_filters, kernel_size=(7, 3, 3), groups=4):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(
                in_filters, out_filters,
                kernel_size=kernel_size,
                padding=(kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[2] // 2)
            ),
            nn.GroupNorm(groups, out_filters),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Spectral3DStack(nn.Module):
    """
    Stacks multiple 3D conv blocks over a (B, 1, C, H, W) input.
    The spectral kernel (7 bands) captures absorption features that span adjacent wavelengths.
    Output: (B, num_filters, C, H, W) — retains full spectral resolution for the transformer.
    """

    def __init__(self, num_filters=8, num_blocks=3, kernel_size=(7, 3, 3)):
        super().__init__()

        blocks = []
        in_f = 1
        for _ in range(num_blocks):
            blocks.append(Single3DBlock(in_f, num_filters, kernel_size))
            in_f = num_filters

        self.stack = nn.Sequential(*blocks)
        self.out_channels = num_filters

    def forward(self, x):
        # x: (B, C, H, W)
        x = x.unsqueeze(1)   # (B, 1, C, H, W)
        x = self.stack(x)    # (B, num_filters, C, H, W)
        return x