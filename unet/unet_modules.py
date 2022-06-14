import torch
import torch.nn as nn
import torch.nn.functional as F


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
            nn.MaxPool3d(2),
            DoubleConvolution(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=3)
        self.conv = DoubleConvolution(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diff_Y = x2.size()[2] - x1.size()[2]
        diff_Z = x2.size()[3] - x1.size()[3]
        diff_X = x2.size()[1] - x1.size()[1]

        x1 = F.pad(x1, [diff_Z // 2, diff_Z - diff_Z // 2,
                        diff_Y // 2, diff_Y - diff_Y // 2,
                        diff_X // 2, diff_X - diff_X // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConvolution(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConvolution, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
