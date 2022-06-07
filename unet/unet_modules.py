import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


class DoubleConvolution(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConvolution, self).__init__()
        self.double_conv = nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size=3, bias=False),
                                         nn.BatchNorm3d(out_channels),
                                         nn.ReLU(inplace=True)
                                         )
        print(f'in - {in_channels} , out - {out_channels}')

    def forward(self, x):
        return self.double_conv(x)


class UNET(nn.Module):
    def __init__(self, in_channels, out_channels, features=[64, 128, 256, 512]):
        super(UNET, self).__init__()
        self.down = nn.ModuleList()
        self.up = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=3)

        # Down of UNET
        for feature in features:
            self.down.append(DoubleConvolution(in_channels, feature))
            in_channels = feature

        # Up of UNET
        for feature in reversed(features):
            self.up.append(nn.ConvTranspose3d(feature * 2, feature, kernel_size=3))
            self.up.append(DoubleConvolution(feature * 2, feature))

        # bottom of UNET
        self.base = DoubleConvolution(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv3d(features[-1], out_channels, kernel_size=3)

    def forward(self, x):
        skip_connections = []

        for down in self.down:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        print('Finished encoder part')
        x = self.base(x)
        print('Before up')
        skip_connections = skip_connections[::-1]

        for i in range(0, len(self.up), 2):
            x = self.up[i](x)
            skip_connection = skip_connections[i // 2]
            
            print('after skip connections')
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[1:])

            add_skip = torch.cat((skip_connection, x), dim=1)
            x = self.up[i + 1].up(add_skip)

        return self.final_conv(x)


def test():
    x = torch.randn((1, 95, 512, 512))
    model = UNET(in_channels=95, out_channels=3)
    preds = model(x)
    print(preds.shape)
    print(x.shape)


if __name__ == '__main__':
    test()
#
#         self.maxpool_conv = nn.Sequential(nn.MaxPool3d(3),
#                                           DoubleConvolution(in_channels, out_channels))
#
#     def forward(self, x):
#         return self.maxpool_conv(x)
#
#
# class UpScaling(nn.Module):
#     def __init__(self, in_channels=3, out_channels=3, ):
#         super().__init__()
#         self.up = nn.ConvTranspose3d(self, in_channels, out_channels, 3)
#         self.conv = DoubleConvolution(in_channels, out_channels)
#
#     def forward(self, x):
#         return self.conv(x)
