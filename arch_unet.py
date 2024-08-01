#!/usr/bin/env python
# -*- coding:utf-8 -*-
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models.resnet import BasicBlock, Bottleneck


class FCN(nn.Module):
    def __init__(self, in_nc=3, out_nc=3):
        super(FCN, self).__init__()
        self.fcn = nn.Sequential(
            nn.Conv2d(in_nc*2, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_nc, 3, 1, 1),
            nn.ReLU(inplace=True)
        )
    def forward(self, x1, x2):
        out = self.fcn(torch.cat([x1, x2], dim=1))
        # print(out.shape)
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, in_size=6, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(in_size, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        # self.conv1 = nn.Conv2d(16, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn. init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        feature_2d = x
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        feature = x
        x = self.fc(x)

        return x, feature, feature_2d
    
class NoiseEstimate(nn.Module):
    def __init__(self, in_nc=3, out_nc=3):
        super(NoiseEstimate, self).__init__()
        self.out_nc = out_nc
        self.backbone = ResNet(BasicBlock, [1, 1, 1, 1], in_size=in_nc*2, num_classes=1024)
        dim_mlp = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.backbone.fc)
        self.noise_estimation = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, 2*out_nc), nn.LeakyReLU())
    def forward(self, x1, x2):
        # nlf = self.nlf(x1, x2)
        # a1 = self.haards(x1)
        # a2 = self.haards(x2)
        # input = F.pixel_shuffle(torch.cat([a1, a2], dim=1), 2)
        x, feature, feature_2d = self.backbone(torch.cat([x1, x2], dim=1))
        noise_param = self.noise_estimation(feature)
        n, c = noise_param.shape
        noise_param = noise_param.reshape(n, self.out_nc, c//self.out_nc)
        nlf = noise_param[..., 0][:, :, None, None] * x1 + noise_param[..., 1][:, :, None, None]
        return torch.abs(nlf)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, depth=5, wf=48, slope=0.1):
        """
        Args:
            in_channels (int): number of input channels, Default 3
            depth (int): depth of the network, Default 5
            wf (int): number of filters in the first layer, Default 32
        """
        super(UNet, self).__init__()
        self.depth = depth
        self.head = nn.Sequential(
            LR(in_channels, wf, 3, slope), LR(wf, wf, 3, slope))
        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(LR(wf, wf, 3, slope))

        self.up_path = nn.ModuleList()
        for i in range(depth):
            if i != depth-1:
                self.up_path.append(UP(wf*2 if i==0 else wf*3, wf*2, slope))
            else:
                self.up_path.append(UP(wf*2+in_channels, wf*2, slope))

        self.last = nn.Sequential(LR(2*wf, 2*wf, 1, slope), 
                    LR(2*wf, 2*wf, 1, slope), conv1x1(2*wf, out_channels, bias=True))

    def forward(self, x):
        blocks = []
        blocks.append(x)
        x = self.head(x)
        for i, down in enumerate(self.down_path):
            x = F.max_pool2d(x, 2)
            if i != len(self.down_path) - 1:
                blocks.append(x)
            x = down(x)

        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i-1])

        out = self.last(x)
        return out


class LR(nn.Module):
    def __init__(self, in_size, out_size, ksize=3, slope=0.1):
        super(LR, self).__init__()
        block = []
        block.append(nn.Conv2d(in_size, out_size,
                     kernel_size=ksize, padding=ksize//2, bias=True))
        block.append(nn.LeakyReLU(slope, inplace=True))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class UP(nn.Module):
    def __init__(self, in_size, out_size, slope=0.1):
        super(UP, self).__init__()
        self.conv_1 = LR(in_size, out_size)
        self.conv_2 = LR(out_size, out_size)

    def up(self, x):
        s = x.shape
        x = x.reshape(s[0], s[1], s[2], 1, s[3], 1)
        x = x.repeat(1, 1, 1, 2, 1, 2)
        x = x.reshape(s[0], s[1], s[2]*2, s[3]*2)
        return x

    def forward(self, x, pool):
        x = self.up(x)
        x = torch.cat([x, pool], 1)
        x = self.conv_1(x)
        x = self.conv_2(x)

        return x


def conv1x1(in_chn, out_chn, bias=True):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=1,
                      stride=1, padding=0, bias=bias)
    return layer

if __name__ == '__main__':
    a = torch.cuda.FloatTensor(4, 3, 128, 128)
    net = NoiseEstimate(in_nc=3,
                out_nc=3).cuda()
    out = net(a, a)
    print(a.shape)