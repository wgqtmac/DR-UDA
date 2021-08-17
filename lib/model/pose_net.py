import os
import torch
import torch.nn as nn

from blocks import BasicBlock
from resnet import ResNet


def _make_deconv(inplanes, outplanes, kernel_size, stride, padding, bias=True):
    return nn.Sequential(
            nn.BatchNorm2d(inplanes),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(inplanes, outplanes, kernel_size=kernel_size,
                               stride=stride, padding=padding, bias=bias) )

# def _make_deconv(inplanes, outplanes, kernel_size, stride, padding, bias=True):
#     return nn.Sequential(
#             nn.ConvTranspose2d(inplanes, outplanes, kernel_size=kernel_size,
#                                stride=stride, padding=padding, bias=bias),
#             nn.BatchNorm2d(outplanes),
#             nn.ReLU(inplace=True) )

def _make_conv(inplanes, outplanes, kernel_size=1, stride=1, padding=0, bias=True):
    return nn.Sequential(
            nn.BatchNorm2d(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, outplanes, kernel_size=kernel_size,
                      stride=stride, padding=padding, bias=bias) )

class PoseResnet(ResNet):
    def __init__(self, block, layers, num_classes):
        super(PoseResnet, self).__init__(block, layers)
        num_feats = 256
        self.bias = False

        # Lateral layers
        self.latlayer1 = nn.Conv2d(512, num_feats, kernel_size=1, stride=1, padding=0, bias=self.bias)
        self.latlayer2 = nn.Conv2d(num_feats, num_feats, kernel_size=1, stride=1, padding=0, bias=self.bias)
        self.latlayer3 = nn.Conv2d(num_feats, num_feats, kernel_size=1, stride=1, padding=0, bias=self.bias)
        # self.latlayer4 = nn.Conv2d( 256, num_feats, kernel_size=1, stride=1, padding=0, bias=self.bias)

        # Top-down layers
        self.deconv1 = _make_deconv(512, 256, kernel_size=4, stride=2, padding=1, bias=self.bias)
        self.deconv2 = _make_deconv(256, 256, kernel_size=4, stride=2, padding=1, bias=self.bias)
        self.deconv3 = _make_deconv(256, 128, kernel_size=4, stride=2, padding=1, bias=self.bias)
        self.deconv4 = _make_deconv(128, 64, kernel_size=4, stride=2, padding=1, bias=self.bias)
        self.deconv5 = _make_deconv(64, 32, kernel_size=4, stride=2, padding=1, bias=self.bias)
        # self.heatmap = nn.Conv2d(num_feats, num_classes, kernel_size=1)
        self.heatmap = _make_conv(32, 3, kernel_size=1)
        self.conv = nn.Conv2d(3, 3, kernel_size=9, stride=1, padding=0, bias=self.bias)

    def forward(self, x):
        # Bottom-up
        c1 = self.conv1(x)
        c1 = self.bn1(c1)
        c1 = self.relu(c1)      # stride: 2; RF: 7
        c1 = self.maxpool(c1)   # stride: 4; RF:11

        c2 = self.layer1(c1)    # stride: 4; RF:35
        c3 = self.layer2(c2)    # stride: 8; RF:91
        c4 = self.layer3(c3)    # stride:16: RF:267
        c5 = self.layer4(c4)    # stride:32; RF:427
        # Top-down
        b5 = self.latlayer1(c5) # stride:32; RF:427
        b4 = self.latlayer2(c4)
        p4 = self.deconv1(c5)
        p3 = self.deconv2(p4)
        p2 = self.deconv3(p3)
        p1 = self.deconv4(p2)
        p0 = self.deconv5(p1)
        out = self.heatmap(p0)
        out = self.conv(out)
        return out

    def init_net(self, pretrained=''):
        self.init_weights(pretrained)

        for name, m in self.named_modules():
            if 'lat' in name:
                if isinstance(m, nn.Conv2d):
                    # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    nn.init.normal_(m.weight, std=0.001)
                    if self.bias:
                        nn.init.constant_(m.bias, 0)
            elif 'deconv' in name:
                if isinstance(m, nn.ConvTranspose2d):
                    nn.init.normal_(m.weight, std=0.001)
                    if self.bias:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
        for m in self.heatmap.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)


if __name__ == '__main__':
    print(torch.__version__)
    x = torch.autograd.Variable(torch.Tensor(2, 3, 248, 248))
    # model = resnet80(num_classes=41857)
    model = PoseResnet(BasicBlock, [2, 2, 2, 2], num_classes=2)
    print(model)
    x= model(x)
    print(x.shape)