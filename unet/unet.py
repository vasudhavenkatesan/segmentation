import torch
import torch.nn as nn
from .unet_modules import *
from config import device


class UNET(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNET, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.encoder = DoubleConvolution(self.n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.out = OutConvolution(64, n_classes)

    def forward(self, x):
        skip1 = self.encoder(x)
        skip2 = self.down1(skip1)
        skip3 = self.down2(skip2)
        skip4 = self.down3(skip3)
        skip5 = self.down4(skip4)
        x = self.up1(skip5, skip4)
        x = self.up2(x, skip3)
        x = self.up3(x, skip2)
        x = self.up4(x, skip1)
        final = self.out(x)
        return final


def test():
    x = torch.randn((1, 1, 506, 506))
    model = UNET(n_channels=1, n_classes=3)
    model.to(device)
    preds = model(x)
    print(f'Final - {preds.shape}')


if __name__ == '__main__':
    test()
