import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import resnet
from ._blocks import Conv1x1, Conv3x3, get_norm_layer
from ._utils import KaimingInitMixin, Identity


class AttentionMetric(nn.Module):
    def __init__(self, fc_ch=64):
        super().__init__()
        self.conv1 = nn.Conv2d(fc_ch, 1, 3, 1, 1)
        self.conv2 = nn.Conv2d(fc_ch, 1, 3, 1, 1)

    def forward(self, x):
        xa = F.sigmoid(self.conv1(x))
        x1, x2 = torch.split(x, x.size(0) // 2, dim=0)
        xb = F.sigmoid(self.conv2(torch.abs(x1 - x2)))
        xb = xb.repeat(2, 1, 1, 1)
        x = x * (xa * xb)
        return x


class Backbone(nn.Module, KaimingInitMixin):
    def __init__(self, in_ch, arch, pretrained=True, strides=(2, 1, 2, 2, 2)):
        super().__init__()

        if arch == 'resnet18':
            self.resnet = resnet.resnet18(pretrained=pretrained, strides=strides, norm_layer=get_norm_layer())
        elif arch == 'resnet34':
            self.resnet = resnet.resnet34(pretrained=pretrained, strides=strides, norm_layer=get_norm_layer())
        elif arch == 'resnet50':
            self.resnet = resnet.resnet50(pretrained=pretrained, strides=strides, norm_layer=get_norm_layer())
        else:
            raise ValueError

        self._trim_resnet()

        if in_ch != 3:
            self.resnet.conv1 = nn.Conv2d(
                in_ch,
                64,
                kernel_size=7,
                stride=strides[0],
                padding=3,
                bias=False
            )

        if not pretrained:
            self._init_weight()

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x1 = self.resnet.layer1(x)
        x2 = self.resnet.layer2(x1)
        x3 = self.resnet.layer3(x2)
        x4 = self.resnet.layer4(x3)

        return x1, x2, x3, x4

    def _trim_resnet(self):
        self.resnet.avgpool = Identity()
        self.resnet.fc = Identity()


class Decoder(nn.Module, KaimingInitMixin):
    def __init__(self, fc_ch):
        super().__init__()
        self.dr1 = Conv1x1(64, 96, norm=True, act=True)
        self.dr2 = Conv1x1(128, 96, norm=True, act=True)
        self.dr3 = Conv1x1(256, 96, norm=True, act=True)
        self.dr4 = Conv1x1(512, 96, norm=True, act=True)

        self.conv_out = nn.Sequential(
            Conv3x3(384, 256, norm=True, act=True),
            nn.Dropout(0.5),  #
            Conv1x1(256, fc_ch, norm=True, act=True)
        )

        self.AMetric = AttentionMetric(fc_ch=64)

        self._init_weight()

    def forward(self, feats):
        f1 = self.dr1(feats[0])
        f2 = self.dr2(feats[1])
        f3 = self.dr3(feats[2])
        f4 = self.dr4(feats[3])

        f2 = F.interpolate(f2, size=f1.shape[2:], mode='bilinear', align_corners=True)
        f3 = F.interpolate(f3, size=f1.shape[2:], mode='bilinear', align_corners=True)
        f4 = F.interpolate(f4, size=f1.shape[2:], mode='bilinear', align_corners=True)

        x = torch.cat([f1, f2, f3, f4], dim=1)
        x = self.conv_out(x)

        x = self.AMetric(x)

        return x


class Base(nn.Module):
    def __init__(self, in_ch, fc_ch=64):
        super().__init__()
        self.extract = Backbone(in_ch=in_ch, arch='resnet18')  #
        self.decoder = Decoder(fc_ch=fc_ch)

    def forward(self, t1, t2):
        b, _, _, _ = t1.size()
        t = torch.cat([t1, t2], dim=0)
        f = self.extract(t)
        f = self.decoder(f)

        f_1, f_2 = torch.split(f, b, dim=0)

        dist = torch.norm(f_1 - f_2, dim=1, keepdim=True)

        dist = F.interpolate(dist, size=t1.shape[2:], mode='bilinear', align_corners=True)

        return dist, dist


if __name__ == '__main__':
    model = Base(in_ch=3, fc_ch=64)
    t1 = torch.randn(1, 3, 64, 64)
    t2 = torch.randn(1, 3, 64, 64)
    dist, _ = model(t1, t2)
    print(dist.size())
