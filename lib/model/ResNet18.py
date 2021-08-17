import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def _make_deconv(inplanes, outplanes, kernel_size, stride, padding, bias=True):
    return nn.Sequential(
            nn.BatchNorm2d(inplanes),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(inplanes, outplanes, kernel_size=kernel_size,
                               stride=stride, padding=padding, bias=bias) )


def _make_conv(inplanes, outplanes, kernel_size=1, stride=1, padding=0, bias=True):
    return nn.Sequential(
            nn.BatchNorm2d(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, outplanes, kernel_size=kernel_size,
                      stride=stride, padding=padding, bias=bias) )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()

        self.restored = False
        self.inplanes = 64
        num_feats = 256
        self.bias = False
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, 2)
        self.fc = nn.Sequential(nn.Linear(512 * block.expansion, 256),
                      nn.ReLU(),
                      nn.Linear(256, 128)
                      )
        # self.fc = nn.Sequential(nn.Linear(512 * block.expansion, 256)
        #                         )

        self.deconv1 = _make_deconv(512, 256, kernel_size=4, stride=2, padding=1, bias=self.bias)
        self.deconv2 = _make_deconv(256, 256, kernel_size=4, stride=2, padding=1, bias=self.bias)
        self.deconv3 = _make_deconv(256, 128, kernel_size=4, stride=2, padding=1, bias=self.bias)
        self.deconv4 = _make_deconv(128, 64, kernel_size=4, stride=2, padding=1, bias=self.bias)
        self.deconv5 = _make_deconv(64, 32, kernel_size=4, stride=2, padding=1, bias=self.bias)
        # self.heatmap = nn.Conv2d(num_feats, num_classes, kernel_size=1)
        self.image = _make_conv(32, 3, kernel_size=1)
        # self.conv = nn.Conv2d(3, 3, kernel_size=9, stride=1, padding=0, bias=self.bias)
        self.conv = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=self.bias)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        p4 = self.deconv1(x4)
        p3 = self.deconv2(p4)
        p2 = self.deconv3(p3)
        p1 = self.deconv4(p2)
        p0 = self.deconv5(p1)
        out = self.image(p0)
        out = self.conv(out)

        x = self.avgpool(x4)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x, out


    def convnet(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        #x = x.view(x.size(0), -1)

        return x

    def extract_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        output = self.fc[0](x)
        return output


    def get_embedding(self, x):
        embedding,_ = self.forward(x)
        return embedding


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


if __name__ == '__main__':
    print(torch.__version__)
    x = torch.autograd.Variable(torch.Tensor(2, 3, 256, 256))
    # model = resnet80(num_classes=41857)
    model = resnet18()
    print(model)
    x, out = model(x)
    print(out.shape)