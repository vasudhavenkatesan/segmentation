from config import device
from .unet_modules import *


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
        print(f'After skip1 - {skip1.shape}')
        skip2 = self.down1(skip1)
        print(f'After skip2 - {skip2.shape}')
        skip3 = self.down2(skip2)
        print(f'After skip3 - {skip3.shape}')
        skip4 = self.down3(skip3)
        print(f'After skip4 - {skip4.shape}')
        skip5 = self.down4(skip4)
        print(f'After skip5 - {skip5.shape}')
        x = self.up1(skip5, skip4)
        print(f'After decoding1 - {x.shape}')
        x = self.up2(x, skip3)
        print(f'After dec2 - {x.shape}')
        x = self.up3(x, skip2)
        print(f'After dec3 - {x.shape}')
        x = self.up4(x, skip1)
        final = self.out(x)
        return final


def test():
    x = torch.randn((1, 1, 572, 572))
    model = UNET(n_channels=1, n_classes=2)
    model.to(device)
    preds = model(x)
    print(f'Final - {preds.shape}')


if __name__ == '__main__':
    test()
