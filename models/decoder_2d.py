import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.5, groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Decoder2D(nn.Module):
    """
    Standard 4-level U-Net decoder with skip connections.
    Bilinear interpolation handles size mismatches from odd input dimensions.
    """

    def __init__(self, num_classes, base_filters=32):
        super().__init__()
        f = base_filters

        self.up4 = nn.ConvTranspose2d(f * 16, f * 8, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(f * 16, f * 8)

        self.up3 = nn.ConvTranspose2d(f * 8, f * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(f * 8, f * 4)

        self.up2 = nn.ConvTranspose2d(f * 4, f * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(f * 4, f * 2)

        self.up1 = nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(f * 2, f)

        self.final = nn.Conv2d(f, num_classes, kernel_size=1)

    def _match_size(self, x, target):
        if x.shape[2:] != target.shape[2:]:
            x = F.interpolate(x, size=target.shape[2:], mode='bilinear', align_corners=False)
        return x

    def forward(self, b, skips):
        e1, e2, e3, e4 = skips

        x = self.up4(b)
        x = self._match_size(x, e4)
        x = self.dec4(torch.cat([x, e4], dim=1))

        x = self.up3(x)
        x = self._match_size(x, e3)
        x = self.dec3(torch.cat([x, e3], dim=1))

        x = self.up2(x)
        x = self._match_size(x, e2)
        x = self.dec2(torch.cat([x, e2], dim=1))

        x = self.up1(x)
        x = self._match_size(x, e1)
        x = self.dec1(torch.cat([x, e1], dim=1))

        return self.final(x)