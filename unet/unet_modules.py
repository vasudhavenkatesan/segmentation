import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as vision
from monai.transforms.croppad.array import CenterSpatialCrop


class DoubleConvolution(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, bias=False, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, bias=False, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool = nn.Sequential(
            nn.MaxPool3d(kernel_size=2),
            DoubleConvolution(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConvolution(in_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        skip = self.crop(skip, x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

    def crop(self, enc_ftrs, x):
        _, _, D, H, W = x.shape
        enc_ftrs = CenterSpatialCrop([-1, D, H, W])(enc_ftrs)
        return enc_ftrs


class OutConvolution(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConvolution, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
