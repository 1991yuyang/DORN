import torch as t
from torch import nn
from torch.nn import functional as F
from torchvision import models
import numpy as np


class PWConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(PWConv, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)


class Conv3X3(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(Conv3X3, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)


class DeConvBlock(nn.Module):

    def __init__(self, upscale_factor):
        super(DeConvBlock, self).__init__()
        times = int(np.log2(upscale_factor))
        self.deconvs = nn.ModuleList([])
        for i in range(times):
            if i != times - 1:
                out_channels = 32
            else:
                out_channels = 128
            if i == 0:
                in_channels = 128
            else:
                in_channels = 32
            self.deconvs.append(nn.Sequential(
                nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2, padding=0, bias=False),
                nn.BatchNorm2d(num_features=out_channels),
                nn.ReLU()
            ))

    def forward(self, x):
        for m in self.deconvs:
            x = m(x)
        return x


class FPN(nn.Module):

    def __init__(self, resnet_type, us_use_interpolate):
        """

        :param resnet_type: "resnet18", "resnet34", "resnet50", "resnet101"
        :param us_use_interpolate: True will use interpolate,False use ConvTranspose2d
        """
        super(FPN, self).__init__()
        backbone_dict = {
            "resnet18": [models.resnet18, models.ResNet18_Weights.DEFAULT],
            "resnet34": [models.resnet34, models.ResNet34_Weights.DEFAULT],
            "resnet50": [models.resnet50, models.ResNet50_Weights.DEFAULT],
            "resnet101": [models.resnet101, models.ResNet101_Weights.DEFAULT]
        }
        out_channels_of_every_layer = {
            "resnet18": [64, 128, 256, 512],
            "resnet34": [64, 128, 256, 512],
            "resnet50": [256, 512, 1024, 2048],
            "resnet101": [256, 512, 1024, 2048]
        }
        out_channelses = out_channels_of_every_layer[resnet_type]
        self.increase_channels = nn.ModuleList([])
        for i in range(len(out_channelses) - 1):
            self.increase_channels.append(PWConv(in_channels=out_channelses[i], out_channels=out_channelses[i + 1]))
        backbone = backbone_dict[resnet_type][0](weights=backbone_dict[resnet_type][1])
        self.head = nn.Sequential(*list(backbone.children())[:4])
        self.stage1 = backbone.layer1
        self.stage2 = backbone.layer2
        self.stage3 = backbone.layer3
        self.stage4 = backbone.layer4
        self.pw1 = PWConv(in_channels=out_channelses[0], out_channels=64)
        self.pw2 = PWConv(in_channels=out_channelses[1], out_channels=64)
        self.pw3 = PWConv(in_channels=out_channelses[2], out_channels=64)
        self.pw4 = PWConv(in_channels=out_channelses[3], out_channels=64)
        self.pool = nn.MaxPool2d(kernel_size=1, stride=2, padding=0)
        self.conv_3x3_1 = Conv3X3(in_channels=64, out_channels=128)
        self.conv_3x3_2 = Conv3X3(in_channels=64, out_channels=128)
        self.conv_3x3_3 = Conv3X3(in_channels=64, out_channels=128)
        self.conv_3x3_4 = Conv3X3(in_channels=64, out_channels=128)
        if not us_use_interpolate:
            self.us4 = DeConvBlock(upscale_factor=8)
            self.us3 = DeConvBlock(upscale_factor=4)
            self.us2 = DeConvBlock(upscale_factor=2)
            self.us_pool = DeConvBlock(upscale_factor=2)
        else:
            self.us4 = nn.Upsample(scale_factor=8)
            self.us3 = nn.Upsample(scale_factor=4)
            self.us2 = nn.Upsample(scale_factor=2)
            self.us_pool = nn.Upsample(scale_factor=2)

    def forward(self, x):
        head_out = self.head(x)
        stage1_out = self.stage1(head_out)
        stage2_out = self.stage2(stage1_out) + F.max_pool2d(self.increase_channels[0](stage1_out), kernel_size=2, stride=2, padding=0)
        stage3_out = self.stage3(stage2_out) + F.max_pool2d(self.increase_channels[1](stage2_out), kernel_size=2, stride=2, padding=0)
        stage4_out = self.stage4(stage3_out) + F.max_pool2d(self.increase_channels[2](stage3_out), kernel_size=2, stride=2, padding=0)
        pw1_out = self.pw1(stage1_out)
        pw2_out = self.pw2(stage2_out)
        pw3_out = self.pw3(stage3_out)
        pw4_out = self.pw4(stage4_out)
        conv_3x3_4_out = self.us4(self.conv_3x3_4(pw4_out))
        pool_out = self.us_pool(self.pool(conv_3x3_4_out))
        us4 = F.interpolate(pw4_out, scale_factor=2) + pw3_out
        conv_3x3_3_out = self.us3(self.conv_3x3_4(us4))
        us3 = F.interpolate(us4, scale_factor=2) + pw2_out
        conv_3x3_2_out = self.us2(self.conv_3x3_2(us3))
        us2 = F.interpolate(us3, scale_factor=2) + pw1_out
        conv_3x3_1_out = self.conv_3x3_1(us2)
        return [head_out, conv_3x3_1_out, conv_3x3_2_out, conv_3x3_3_out, conv_3x3_4_out, pool_out]


class FullImageEnc(nn.Module):

    def __init__(self):
        super(FullImageEnc, self).__init__()
        self.fc1 = nn.Sequential(
            PWConv(in_channels=64, out_channels=16),
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            PWConv(in_channels=64, out_channels=16),
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=5, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            PWConv(in_channels=64, out_channels=16),
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=7, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU()
        )
        self.last = Conv3X3(in_channels=64, out_channels=128)

    def forward(self, head_out):
        pool1_out = F.max_pool2d(head_out, kernel_size=tuple([head_out.size()[2] - 2, head_out.size()[3] - 2]), stride=1, padding=0)
        pool2_out = F.max_pool2d(head_out, kernel_size=tuple([head_out.size()[2] - 4, head_out.size()[3] - 4]), stride=1, padding=0)
        pool3_out = F.max_pool2d(head_out, kernel_size=tuple([head_out.size()[2] - 6, head_out.size()[3] - 6]), stride=1, padding=0)
        fc1_out = self.fc1(pool1_out)
        fc2_out = self.fc2(pool2_out)
        fc3_out = self.fc3(pool3_out)
        fc_out = fc1_out + fc2_out + fc3_out
        expand_out = fc_out.expand_as(head_out)
        ret = self.last(expand_out)
        return ret


class SUM(nn.Module):

    def __init__(self, resnet_type, us_use_interpolate):
        """

        :param resnet_type: "resnet18", "resnet34", "resnet50", "resnet101"
        :param us_use_interpolate: True will use interpolate,False use ConvTranspose2d
        """
        super(SUM, self).__init__()
        self.fpn = FPN(resnet_type, us_use_interpolate)
        self.fie = FullImageEnc()

    def forward(self, x):
        head_out, conv_3x3_1_out, conv_3x3_2_out, conv_3x3_3_out, conv_3x3_4_out, pool_out = self.fpn(x)
        fie_out = self.fie(head_out)
        concate_out = t.cat([conv_3x3_1_out, conv_3x3_2_out, conv_3x3_3_out, conv_3x3_4_out, pool_out, fie_out], dim=1)  # [2, 1536, 160, 160]
        return concate_out


class OridReg(nn.Module):

    def __init__(self):
        super(OridReg, self).__init__()
        self.block = PWConv(in_channels=6 * 128, out_channels=32)

    def forward(self, x):
        return self.block(x)


if __name__ == "__main__":
    model = SUM("resnet18", True)
    d = t.randn(2, 3, 640, 640)
    feature = model(d)
    print(feature.size())