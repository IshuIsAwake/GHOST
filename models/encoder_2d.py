import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.5, groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, out_channels),  # Replaced BatchNorm2d
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),               # Increased to 0.5
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, out_channels),  # Replaced BatchNorm2d
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)

class Encoder2D(nn.Module):
    def __init__(self, in_channels, base_filters=64):
        super().__init__()
        f = base_filters

        self.enc1 = ConvBlock(in_channels, f)      
        self.enc2 = ConvBlock(f, f*2)              
        self.enc3 = ConvBlock(f*2, f*4)            
        self.enc4 = ConvBlock(f*4, f*8)            

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(f*8, f*16)     

    def forward(self, x):
        e1 = self.enc1(x)              
        e2 = self.enc2(self.pool(e1))  
        e3 = self.enc3(self.pool(e2))  
        e4 = self.enc4(self.pool(e3))  
        b  = self.bottleneck(self.pool(e4)) 

        return b, [e1, e2, e3, e4]