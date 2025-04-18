import torch
import torch.nn as nn
import torch.nn.functional as F

# define a reusable double 3D convolution block
class DoubleConv3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)
    
# define the U-Net model
class UNet3D(nn.Module):
    def __init__(self, in_channels = 1, out_channels = 1):
        super().__init__()

        # Encoder (Downsampling)
        self.enc1 = DoubleConv3D(in_channels, 32)
        self.pool1 = nn.MaxPool3d(2)

        self.enc2 = DoubleConv3D(32, 64)
        self.pool2 = nn.MaxPool3d(2)

        self.enc3 = DoubleConv3D(64, 128)
        self.pool3 = nn.MaxPool3d(2)

        # bottleneck
        self.bottleneck = DoubleConv3D(128, 256)

        # decoder (Upsampling)
        self.up3 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.dec3 = DoubleConv3D(256, 128)

        self.up2 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.dec2 = DoubleConv3D(128, 64)

        self.up1 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.dec1 = DoubleConv3D(64, 32)

        # final output
        self.final_conv = nn.Conv3d(32, out_channels, kernel_size = 1)

    def forward(self, x):
        # encoder
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool1(x1))
        x3 = self.enc3(self.pool2(x2))

        #bottleneck
        x4 = self.bottleneck(self.pool3(x3))

        # decoder
        d3 = self.up3(x4)
        d3 = torch.cat([d3, x3], dim = 1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, x2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, x1], dim=1)
        d1 = self.dec1(d1)

        # output segmentation map
        out = self.final_conv(d1)
        return out
