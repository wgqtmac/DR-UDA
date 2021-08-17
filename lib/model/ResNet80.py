import torch
import torch.nn as nn
import math


from lib.utils.vector_normalization import VectorNorm


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, last=False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                                padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        
        self.downsample = downsample
        self.last = last
        
    
    def forward(self, x):
        residual = x
        out = self.conv1(x) 
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        if self.last == False:
            out = self.bn3(out)
            out = self.relu(out)

        return out

class ResNet(nn.Module):
    
    def __init__(self, block, layers, vec_norm=True, num_classes=1000):

        self.restored = False

        self.in_planes = 16
        super(ResNet, self).__init__()
        self.bn1 = nn.BatchNorm2d(3)

        self.conv = nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3, bias=False)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 96, layers[3], stride=2, last=True)
        
        self.bn3 = nn.BatchNorm2d(384)
        self.bn4 = nn.BatchNorm1d(1024)
        
        self.fc1 = nn.Linear(96 * block.expansion * 2 * 16, 1024)
        self.vec_norm = VectorNorm()
        #self.fc2 = nn.Linear(1024, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, last=False):
        self.in_planes = self.in_planes * block.expansion
        downsample = nn.Sequential(
            nn.Conv2d(self.in_planes, planes * block.expansion * 2,
                    kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes * block.expansion * 2)
        )
        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.inplanes = planes * block.expansion * 2
        for i in range(1, blocks-1):
            layers.append(block(self.inplanes, planes))
        layers.append(block(self.inplanes, planes, last=last))

        return nn.Sequential(*layers)
 
    def forward(self, x):
        x = self.bn1(x)

        x = self.conv(x)

        x = self.maxpool(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.maxpool(x)
        x = self.bn3(x)
        s = self.relu(x)
        x = x.view(x.size(0), -1)
        fc1 = self.fc1(x)
        fc1 = self.bn4(fc1)
        fc1 = self.relu(fc1)

        fc1 = self.vec_norm(fc1)

        #fc2 = self.fc2(fc1)

        #return fc1, fc2
        return fc1

    def get_embedding(self, x):
        return self.forward(x)



class ResNetClassifier(nn.Module):
    """ResNet classifier model for ADDA."""

    def __init__(self):
        """Init ResNet encoder."""
        super(ResNetClassifier, self).__init__()
        self.fc2 = nn.Linear(1024, 2)

    def forward(self, feat):
        """Forward the ResNet classifier."""
        out = self.fc2(feat)
        return feat, out

def resnet80(**kwargs):
    
    model = ResNet(Bottleneck, [3, 4, 16, 3], **kwargs)
    #classifier = ResNetClassifier()

    return model

if __name__ == '__main__':
    print(torch.__version__)
    x = torch.autograd.Variable(torch.Tensor(2, 3, 248, 248))
    # model = resnet80(num_classes=41857)
    model = resnet80(num_classes=2)
    print(model)
    #state_dict = torch.load('./caffe2pytorch/ResNet80_pytorch_model_lastest.pth.tar')['state_dict']
    #model.load_state_dict(state_dict)
    #out = model(x)
