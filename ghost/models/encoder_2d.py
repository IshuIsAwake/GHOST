import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)

class Encoder2D(nn.Module):
    def __init__(self, in_channels, base_filters=64):
        super().__init__()
        f = base_filters

        self.enc1 = ConvBlock(in_channels, f)      # 145x145
        self.enc2 = ConvBlock(f, f*2)              # 72x72
        self.enc3 = ConvBlock(f*2, f*4)            # 36x36
        self.enc4 = ConvBlock(f*4, f*8)            # 18x18

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(f*8, f*16)     # 9x9

    def forward(self, x):
        e1 = self.enc1(x)           # (B, f,   145, 145)
        e2 = self.enc2(self.pool(e1))  # (B, f*2, 72,  72)
        e3 = self.enc3(self.pool(e2))  # (B, f*4, 36,  36)
        e4 = self.enc4(self.pool(e3))  # (B, f*8, 18,  18)
        b  = self.bottleneck(self.pool(e4))  # (B, f*16, 9, 9)

        return b, [e1, e2, e3, e4] 