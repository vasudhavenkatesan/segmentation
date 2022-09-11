from typing import Tuple, Union
import torch
from monai.networks.blocks import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock, UnetOutBlock
from monai.networks.nets import ViT
import torch.nn as nn


class UNETR(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, img_size=Tuple[int, int, int], feature_size: int = 16,
                 hidden_size: int = 768, mlp_dim: int = 3072,
                 num_heads: int = 12,
                 pos_embed: str = "perceptron",
                 norm_name: Union[Tuple, str] = "instance",
                 conv_block: bool = False,
                 res_block: bool = True,
                 dropout_rate: float = 0.0,
                 ) -> None:

        super().__init__()
        if not (0 <= dropout_rate <= 1):
            raise AssertionError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise AssertionError("hidden size should be divisible by num_heads.")

        if pos_embed not in ["conv", "perceptron"]:
            raise KeyError(f"Position embedding layer of type {pos_embed} is not supported.")

        self.num_layers = 12
        self.patch_size = (16, 16, 16)
        self.classification = False
        self.hidden_size = hidden_size
        self.feat_size = (
            img_size[0] // self.patch_size[0],
            img_size[1] // self.patch_size[1],
            img_size[2] // self.patch_size[2],
        )
        self.vit = ViT(in_channels=in_channels, img_size=img_size, patch_size=self.patch_size, hidden_size=hidden_size,
                       mlp_dim=mlp_dim, num_layers=self.num_layers, num_heads=num_heads, pos_embed=pos_embed,
                       classification=self.classification, dropout_rate=dropout_rate)
        self.encoder1 = UnetrBasicBlock(spatial_dims=3, in_channels=in_channels, out_channels=feature_size,
                                        kernel_size=3, stride=1, norm_name=norm_name, res_block=res_block)
        self.encoder2 = UnetrPrUpBlock(spatial_dims=3, in_channels=hidden_size, out_channels=feature_size * 2,
                                       num_layer=2, kernel_size=3, stride=1, upsample_kernel_size=2,
                                       norm_name=norm_name, conv_block=conv_block, res_block=res_block)
        self.encoder3 = UnetrPrUpBlock(spatial_dims=3, in_channels=hidden_size, out_channels=feature_size * 4,
                                       num_layer=1, kernel_size=3, stride=1, upsample_kernel_size=2,
                                       norm_name=norm_name, conv_block=conv_block, res_block=res_block)
        self.encoder4 = UnetrPrUpBlock(spatial_dims=3, in_channels=hidden_size, out_channels=feature_size * 8,
                                       num_layer=0, kernel_size=3, stride=1, upsample_kernel_size=2,
                                       norm_name=norm_name, conv_block=conv_block, res_block=res_block)
        self.decoder5 = UnetrUpBlock(spatial_dims=3, in_channels=hidden_size, out_channels=feature_size * 8,
                                     kernel_size=3, upsample_kernel_size=2, norm_name=norm_name, res_block=res_block)
        self.decoder4 = UnetrUpBlock(spatial_dims=3, in_channels=feature_size * 8, out_channels=feature_size * 4,
                                     kernel_size=3, upsample_kernel_size=2, norm_name=norm_name, res_block=res_block)
        self.decoder3 = UnetrUpBlock(spatial_dims=3, in_channels=feature_size * 4, out_channels=feature_size * 2,
                                     kernel_size=3, upsample_kernel_size=2, norm_name=norm_name, res_block=res_block)
        self.decoder2 = UnetrUpBlock(spatial_dims=3, in_channels=feature_size * 2, out_channels=feature_size,
                                     kernel_size=3, upsample_kernel_size=2, norm_name=norm_name, res_block=res_block)
        self.out = UnetOutBlock(spatial_dims=3, in_channels=feature_size, out_channels=out_channels)

    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

    def forward(self, x_in):
        x, hidden_states_out = self.vit(x_in)
        enc1 = self.encoder1(x_in)
        x2 = hidden_states_out[3]
        enc2 = self.encoder2(self.proj_feat(x2, self.hidden_size, self.feat_size))
        x3 = hidden_states_out[6]
        enc3 = self.encoder3(self.proj_feat(x3, self.hidden_size, self.feat_size))
        x4 = hidden_states_out[9]
        enc4 = self.encoder4(self.proj_feat(x4, self.hidden_size, self.feat_size))
        dec4 = self.proj_feat(x, self.hidden_size, self.feat_size)
        dec3 = self.decoder5(dec4, enc4)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        out = self.decoder2(dec1, enc1)
        logits = self.out(out)
        return logits


def test():
    x = torch.randn((1, 1, 80, 512, 512))
    model = UNETR(in_channels=1, out_channels=2, img_size=[80, 512, 512])
    preds = model(x)
    print(preds.argmax(dim=1).shape)
    print(f'Final - {preds.shape}')


if __name__ == '__main__':
    test()
