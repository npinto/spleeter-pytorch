import torch
from torch import nn
import torch.nn.functional as F

class CustomPad(nn.Module):
    def __init__(self, padding_setting=(1, 2, 1, 2)):
        super(CustomPad, self).__init__()
        self.padding_setting = padding_setting

    def forward(self, x):
        return F.pad(x, self.padding_setting, "constant", 0)


class CustomTransposedPad(nn.Module):
    def __init__(self, padding_setting=(1, 2, 1, 2)):
        super(CustomTransposedPad, self).__init__()
        self.padding_setting = padding_setting

    def forward(self, x):
        l,r,t,b = self.padding_setting
        return x[:,:,l:-r,t:-b]


def down_block(in_filters, out_filters, elu=False):
    if elu:
        return nn.Sequential(CustomPad(),
                             nn.Conv2d(in_filters, out_filters, kernel_size=5, stride=2,padding=0)), \
                nn.Sequential(
                    nn.BatchNorm2d(out_filters, track_running_stats=True, eps=1e-3, momentum=0.01),
                    nn.ELU())
    else:
        return nn.Sequential(CustomPad(),
                             nn.Conv2d(in_filters, out_filters, kernel_size=5, stride=2,padding=0)), \
                nn.Sequential(
                    nn.BatchNorm2d(out_filters, track_running_stats=True, eps=1e-3, momentum=0.01),
                    nn.LeakyReLU(0.2))


def up_block(in_filters, out_filters, dropout=False, elu=False):

    layers = [
        nn.ConvTranspose2d(in_filters, out_filters, kernel_size=5,stride=2),
        CustomTransposedPad()
    ]

    if elu:
        layers.append(nn.ELU())
    else:
        layers.append(nn.ReLU())

    layers.append(nn.BatchNorm2d(out_filters, track_running_stats=True,
                                 eps=1e-3, momentum=0.01))

    if dropout:
        layers.append(nn.Dropout(0.5))

    return nn.Sequential(*layers)


class UNet(nn.Module):
    def __init__(self, in_channels=2, output_mask_logit=False, elu=False):
        super(UNet, self).__init__()
        self.output_mask_logit = output_mask_logit
        self.down1_conv, self.down1_act = down_block(in_channels, 16, elu=elu)
        self.down2_conv, self.down2_act = down_block(16, 32, elu=elu)
        self.down3_conv, self.down3_act = down_block(32, 64, elu=elu)
        self.down4_conv, self.down4_act = down_block(64, 128, elu=elu)
        self.down5_conv, self.down5_act = down_block(128, 256, elu=elu)
        self.down6_conv, self.down6_act = down_block(256, 512, elu=elu)

        self.up1 = up_block(512, 256, dropout=True, elu=elu)
        self.up2 = up_block(512, 128, dropout=True, elu=elu)
        self.up3 = up_block(256, 64, dropout=True, elu=elu)
        self.up4 = up_block(128, 32, elu=elu)
        self.up5 = up_block(64, 16, elu=elu)
        self.up6 = up_block(32, 1, elu=elu)
        up7_ops = [nn.Conv2d(1, in_channels, kernel_size=4, dilation=2, padding=3)]
        if not output_mask_logit:
            up7_ops.append(nn.Sigmoid())
        self.up7 = nn.Sequential(*up7_ops)

    def forward(self, x):
        d1_conv = self.down1_conv(x)
        d1 = self.down1_act(d1_conv)

        d2_conv = self.down2_conv(d1)
        d2 = self.down2_act(d2_conv)

        d3_conv = self.down3_conv(d2)
        d3 = self.down3_act(d3_conv)

        d4_conv = self.down4_conv(d3)
        d4 = self.down4_act(d4_conv)

        d5_conv = self.down5_conv(d4)
        d5 = self.down5_act(d5_conv)

        d6_conv = self.down6_conv(d5)
        d6 = self.down6_act(d6_conv)

        u1 = self.up1(d6_conv)
        u2 = self.up2(torch.cat([d5_conv, u1], axis=1))
        u3 = self.up3(torch.cat([d4_conv, u2], axis=1))
        u4 = self.up4(torch.cat([d3_conv, u3], axis=1))
        u5 = self.up5(torch.cat([d2_conv, u4], axis=1))
        u6 = self.up6(torch.cat([d1_conv, u5], axis=1))
        u7 = self.up7(u6)

        if not self.output_mask_logit:
            out = u7 * x
        else:
            out = u7

        return out


if __name__ == '__main__':
    net = UNet(14)
    print(net(torch.rand(1, 14, 20, 48)).shape)
