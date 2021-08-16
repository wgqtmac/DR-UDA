import torch
import torch.nn as nn
from torchvision.models.resnet import ResNet, BasicBlock


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
                               stride=stride, padding=padding, bias=bias))


def _make_conv(inplanes, outplanes, kernel_size=1, stride=1, padding=0, bias=True):
    return nn.Sequential(
            nn.BatchNorm2d(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, outplanes, kernel_size=kernel_size,
                      stride=stride, padding=padding, bias=bias) )



class _netG(nn.Module):
    def __init__(self, ngpu, nz):
        super(_netG, self).__init__()
        self.ngpu = ngpu
        self.nz = nz

        # first linear layer
        self.fc1 = nn.Linear(110, 768)
        # Transposed Convolution 2
        self.tconv2 = nn.Sequential(
            nn.ConvTranspose2d(768, 384, 5, 2, 0, bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(True),
        )
        # Transposed Convolution 3
        self.tconv3 = nn.Sequential(
            nn.ConvTranspose2d(384, 256, 5, 2, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
        )
        # Transposed Convolution 4
        self.tconv4 = nn.Sequential(
            nn.ConvTranspose2d(256, 192, 5, 2, 0, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(True),
        )
        # Transposed Convolution 5
        self.tconv5 = nn.Sequential(
            nn.ConvTranspose2d(192, 64, 5, 2, 0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        # Transposed Convolution 5
        self.tconv6 = nn.Sequential(
            nn.ConvTranspose2d(64, 3, 8, 2, 0, bias=False),
            nn.Tanh(),
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            input = input.view(-1, self.nz)
            fc1 = nn.parallel.data_parallel(self.fc1, input, range(self.ngpu))
            fc1 = fc1.view(-1, 768, 1, 1)
            tconv2 = nn.parallel.data_parallel(self.tconv2, fc1, range(self.ngpu))
            tconv3 = nn.parallel.data_parallel(self.tconv3, tconv2, range(self.ngpu))
            tconv4 = nn.parallel.data_parallel(self.tconv4, tconv3, range(self.ngpu))
            tconv5 = nn.parallel.data_parallel(self.tconv5, tconv4, range(self.ngpu))
            tconv5 = nn.parallel.data_parallel(self.tconv6, tconv5, range(self.ngpu))
            output = tconv5
        else:
            input = input.view(-1, self.nz)
            fc1 = self.fc1(input)
            fc1 = fc1.view(-1, 768, 1, 1)
            tconv2 = self.tconv2(fc1)
            tconv3 = self.tconv3(tconv2)
            tconv4 = self.tconv4(tconv3)
            tconv5 = self.tconv5(tconv4)
            tconv5 = self.tconv6(tconv5)
            output = tconv5
        return output


class _netD(nn.Module):
    def __init__(self, ngpu, num_classes=10):
        super(_netD, self).__init__()
        self.ngpu = ngpu

        # Convolution 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # Convolution 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, 0, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # Convolution 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # Convolution 4
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 0, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # Convolution 5
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # Convolution 6
        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # discriminator fc
        self.fc_dis = nn.Linear(13*13*512, 1)
        # aux-classifier fc
        self.fc_aux = nn.Linear(13*13*512, num_classes)
        # softmax and sigmoid
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            conv1 = nn.parallel.data_parallel(self.conv1, input, range(self.ngpu))
            conv2 = nn.parallel.data_parallel(self.conv2, conv1, range(self.ngpu))
            conv3 = nn.parallel.data_parallel(self.conv3, conv2, range(self.ngpu))
            conv4 = nn.parallel.data_parallel(self.conv4, conv3, range(self.ngpu))
            conv5 = nn.parallel.data_parallel(self.conv5, conv4, range(self.ngpu))
            conv6 = nn.parallel.data_parallel(self.conv6, conv5, range(self.ngpu))
            flat6 = conv6.view(-1, 13*13*512)
            fc_dis = nn.parallel.data_parallel(self.fc_dis, flat6, range(self.ngpu))
            fc_aux = nn.parallel.data_parallel(self.fc_aux, flat6, range(self.ngpu))
        else:
            conv1 = self.conv1(input)
            conv2 = self.conv2(conv1)
            conv3 = self.conv3(conv2)
            conv4 = self.conv4(conv3)
            conv5 = self.conv5(conv4)
            conv6 = self.conv6(conv5)
            flat6 = conv6.view(-1, 13*13*512)
            fc_dis = self.fc_dis(flat6)
            fc_aux = self.fc_aux(flat6)
        classes = self.log_softmax(fc_aux)
        realfake = self.sigmoid(fc_dis).view(-1, 1).squeeze(1)
        return realfake, classes


class Generator(nn.Module):
    def __init__(self, ngpu, nz):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.nz = nz

        # first linear layer
        self.fc1 = nn.Linear(512, 1536)
        # Transposed Convolution 2
        self.tconv2 = nn.Sequential(
            nn.ConvTranspose2d(1536, 768, 5, 2, 0, bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU(True),
        )
        # Transposed Convolution 3
        self.tconv3 = nn.Sequential(
            nn.ConvTranspose2d(768, 384, 5, 2, 0, bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(True),
        )
        # Transposed Convolution 4
        self.tconv4 = nn.Sequential(
            nn.ConvTranspose2d(384, 256, 5, 2, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
        )
        # Transposed Convolution 5
        self.tconv5 = nn.Sequential(
            nn.ConvTranspose2d(256, 192, 5, 2, 0, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(True),
        )

        self.tconv6 = nn.Sequential(
            nn.ConvTranspose2d(192, 64, 5, 2, 0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )

        # Transposed Convolution 5
        self.tconv7 = nn.Sequential(
            nn.ConvTranspose2d(64, 3, 8, 2, 0, bias=False),
            nn.Tanh(),
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            input = input.view(-1, self.nz)
            fc1 = nn.parallel.data_parallel(self.fc1, input, range(self.ngpu))
            fc1 = fc1.view(-1, 1536, 1, 1)
            tconv2 = nn.parallel.data_parallel(self.tconv2, fc1, range(self.ngpu))
            tconv3 = nn.parallel.data_parallel(self.tconv3, tconv2, range(self.ngpu))
            tconv4 = nn.parallel.data_parallel(self.tconv4, tconv3, range(self.ngpu))
            tconv5 = nn.parallel.data_parallel(self.tconv5, tconv4, range(self.ngpu))
            tconv5 = nn.parallel.data_parallel(self.tconv6, tconv5, range(self.ngpu))
            output = tconv5
        else:
            input = input.view(-1, self.nz)
            fc1 = self.fc1(input)
            fc1 = fc1.view(-1, 1536, 1, 1)
            tconv2 = self.tconv2(fc1)
            tconv3 = self.tconv3(tconv2)
            tconv4 = self.tconv4(tconv3)
            tconv5 = self.tconv5(tconv4)
            tconv6 = self.tconv6(tconv5)
            tconv7 = self.tconv7(tconv6)
            output = tconv7
        return output


class Discriminator(nn.Module):
    def __init__(self, ngpu, num_classes=10):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu

        # Convolution 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # Convolution 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, 0, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # Convolution 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # Convolution 4
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 0, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # Convolution 5
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # Convolution 6
        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )

        # Convolution 6
        self.conv7 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 2, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )

        # discriminator fc
        self.fc_dis = nn.Linear(14*14*512, 1)
        # aux-classifier fc
        self.fc_aux = nn.Linear(14*14*512, num_classes)
        # softmax and sigmoid
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            conv1 = nn.parallel.data_parallel(self.conv1, input, range(self.ngpu))
            conv2 = nn.parallel.data_parallel(self.conv2, conv1, range(self.ngpu))
            conv3 = nn.parallel.data_parallel(self.conv3, conv2, range(self.ngpu))
            conv4 = nn.parallel.data_parallel(self.conv4, conv3, range(self.ngpu))
            conv5 = nn.parallel.data_parallel(self.conv5, conv4, range(self.ngpu))
            conv6 = nn.parallel.data_parallel(self.conv6, conv5, range(self.ngpu))
            flat6 = conv6.view(-1, 13*13*512)
            fc_dis = nn.parallel.data_parallel(self.fc_dis, flat6, range(self.ngpu))
            fc_aux = nn.parallel.data_parallel(self.fc_aux, flat6, range(self.ngpu))
        else:
            conv1 = self.conv1(input)
            conv2 = self.conv2(conv1)
            conv3 = self.conv3(conv2)
            conv4 = self.conv4(conv3)
            conv5 = self.conv5(conv4)
            conv6 = self.conv6(conv5)
            conv7 = self.conv7(conv6)
            flat7 = conv7.view(-1, 14*14*512)
            fc_dis = self.fc_dis(flat7)
            fc_aux = self.fc_aux(flat7)
        # classes = self.log_softmax(fc_aux)
        classes = fc_aux
        realfake = self.sigmoid(fc_dis).view(-1, 1).squeeze(1)
        return realfake, classes




def resnet18(pretrained=True, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    model_path = '/home/gqwang/pretrained_models/resnet18-5c106cde.pth'

    if pretrained:
        model.load_state_dict(torch.load(model_path))
        print("loading model: ", model_path)
    print(model)
    return model


class Feature_Generator_ResNet18(nn.Module):
    def __init__(self):
        super(Feature_Generator_ResNet18, self).__init__()
        model_resnet = resnet18(pretrained=True)
        self.bias = False
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.deconv1 = _make_deconv(256, 256, kernel_size=4, stride=2, padding=1, bias=self.bias)
        self.deconv2 = _make_deconv(256, 128, kernel_size=4, stride=2, padding=1, bias=self.bias)
        self.deconv3 = _make_deconv(128, 64, kernel_size=4, stride=2, padding=1, bias=self.bias)
        self.deconv4 = _make_deconv(64, 32, kernel_size=4, stride=2, padding=1, bias=self.bias)
        # self.heatmap = nn.Conv2d(num_feats, num_classes, kernel_size=1)
        self.image = _make_conv(32, 3, kernel_size=1)
        # self.conv = nn.Conv2d(3, 3, kernel_size=9, stride=1, padding=0, bias=self.bias)
        self.conv = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=self.bias)

    def forward(self, input):
        feature = self.conv1(input)
        feature = self.bn1(feature)
        feature = self.relu(feature)
        feature = self.maxpool(feature)
        feature = self.layer1(feature)
        feature = self.layer2(feature)
        feature = self.layer3(feature)

        # p4 = self.deconv1(feature)
        # p3 = self.deconv2(p4)
        # p2 = self.deconv3(p3)
        # p1 = self.deconv4(p2)
        # out = self.image(p1)
        # out = self.conv(out)

        return feature

class Feature_Embedder_ResNet18(nn.Module):
    def __init__(self):
        super(Feature_Embedder_ResNet18, self).__init__()
        model_resnet = resnet18(pretrained=False)
        self.layer4 = model_resnet.layer4
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.bottleneck_layer_fc = nn.Linear(512, 512)
        self.bottleneck_layer_fc.weight.data.normal_(0, 0.005)
        self.bottleneck_layer_fc.bias.data.fill_(0.1)
        self.bottleneck_layer = nn.Sequential(
            self.bottleneck_layer_fc,
            nn.ReLU(),
            nn.Dropout(0.5)
        )

    def forward(self, input, norm_flag=True):
        feature = self.layer4(input)
        feature = self.avgpool(feature)
        feature = feature.view(feature.size(0), -1)
        feature = self.bottleneck_layer(feature)
        if (norm_flag):
            feature_norm = torch.norm(feature, p=2, dim=1, keepdim=True).clamp(min=1e-12) ** 0.5 * (2) ** 0.5
            feature = torch.div(feature, feature_norm)
        return feature

#

def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output

# def resnet18(pretrained=False, **kwargs):
#     """Constructs a ResNet-18 model.
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
#     return model



class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.fc_dis = nn.Linear(512, 1)
        self.fc_aux = nn.Linear(512, 2)
        self.fc_dis.weight.data.normal_(0, 0.01)
        self.fc_dis.bias.data.fill_(0.0)
        self.fc_aux.weight.data.normal_(0, 0.01)
        self.fc_aux.bias.data.fill_(0.0)
        # self.classifier_layer = nn.Linear(512, 2)
        # self.classifier_layer.weight.data.normal_(0, 0.01)
        # self.classifier_layer.bias.data.fill_(0.0)
        # self.fc_dis = nn.Sequential(nn.Linear(512, 256),
        #                             nn.ReLU(),
        #                             nn.Linear(256, 1)
        #                             )
        #
        # # aux-classifier fc
        # self.fc_aux = nn.Sequential(nn.Linear(512, 256),
        #                             nn.ReLU(),
        #                             nn.Linear(256, 2)
        #                             )
        # softmax and sigmoid
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input, norm_flag=True):
        if(norm_flag):
            self.fc_dis.weight.data = l2_norm(self.fc_dis.weight, axis=0)
            self.fc_aux.weight.data = l2_norm(self.fc_aux.weight, axis=0)
            # classifier_out = self.classifier_layer(input)
            fc_dis = self.fc_dis(input)
            fc_aux = self.fc_aux(input)
            classes = self.log_softmax(fc_aux)
            # classes = fc_aux
            realfake = self.sigmoid(fc_dis).view(-1, 1).squeeze(1)
            return realfake, classes
        else:
            # classifier_out = self.classifier_layer(input)
            fc_dis = self.fc_dis(input)
            fc_aux = self.fc_aux(input)
            # classes = fc_aux
            classes = self.log_softmax(fc_aux)
            realfake = self.sigmoid(fc_dis).view(-1, 1).squeeze(1)
            return realfake, classes
        # return classifier_out

class Feature_Recnet(nn.Module):
    def __init__(self):
        super(Feature_Recnet, self).__init__()
        self.bias = False
        self.deconv1 = _make_deconv(512, 256, kernel_size=4, stride=2, padding=1, bias=self.bias)
        self.deconv2 = _make_deconv(256, 256, kernel_size=4, stride=2, padding=1, bias=self.bias)
        self.deconv3 = _make_deconv(256, 128, kernel_size=4, stride=2, padding=1, bias=self.bias)
        self.deconv4 = _make_deconv(128, 64, kernel_size=4, stride=2, padding=1, bias=self.bias)
        self.deconv5 = _make_deconv(64, 32, kernel_size=4, stride=2, padding=1, bias=self.bias)
        # self.heatmap = nn.Conv2d(num_feats, num_classes, kernel_size=1)
        self.image = _make_conv(32, 3, kernel_size=1, stride=2)
        # self.conv = nn.Conv2d(3, 3, kernel_size=9, stride=1, padding=0, bias=self.bias)
        self.conv = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=self.bias)

    def forward(self, feature):
        p4 = self.deconv1(feature)
        p3 = self.deconv2(p4)
        p2 = self.deconv3(p3)
        p1 = self.deconv4(p2)
        p0 = self.deconv5(p1)
        out = self.image(p0)
        out = self.conv(out)
        return out


class MDClassifier(nn.Module):
    def __init__(self):
        super(MDClassifier, self).__init__()
        self.fc = nn.Linear(512, 2)
        self.fc.weight.data.normal_(0, 0.01)
        self.fc.bias.data.fill_(0.0)

    def forward(self, input, norm_flag=True):
        if(norm_flag):
            self.fc.weight.data = l2_norm(self.fc.weight, axis=0)
            x = self.fc(input)
            return x
        else:
            x = self.fc(input)
            return x

class CatClassifier(nn.Module):
    def __init__(self):
        super(CatClassifier, self).__init__()
        self.fc = nn.Linear(512 * 3, 2)
        self.fc.weight.data.normal_(0, 0.01)
        self.fc.bias.data.fill_(0.0)

    def forward(self, input, norm_flag=True):
        if(norm_flag):
            self.fc.weight.data = l2_norm(self.fc.weight, axis=0)
            x = self.fc(input)
            return x
        else:
            x = self.fc(input)
            return x

class DG_model(nn.Module):
    def __init__(self, model):
        super(DG_model, self).__init__()
        if(model == 'resnet18'):
            self.backbone = Feature_Generator_ResNet18()
            self.embedder = Feature_Embedder_ResNet18()
        else:
            print('Wrong Name!')
        self.classifier = Classifier()

    def forward(self, input, norm_flag=True):
        feature = self.backbone(input)
        feature = self.embedder(feature, norm_flag)
        # classifier_out = self.classifier(feature, norm_flag)
        realfake, classes = self.classifier(feature, norm_flag)
        return realfake, classes

class MD_Encoder(nn.Module):
    def __init__(self, model):
        super(MD_Encoder, self).__init__()
        if(model == 'resnet18'):
            self.backbone = Feature_Generator_ResNet18()
            self.embedder = Feature_Embedder_ResNet18()
        else:
            print('Wrong Name!')
        self.classifier = Classifier()

    def forward(self, input, norm_flag=True):
        feature = self.backbone(input)
        feature = self.embedder(feature, norm_flag)
        return feature


if __name__ == '__main__':
    print(torch.__version__)
    noise = torch.autograd.Variable(torch.Tensor(16, 512, 1, 1))
    input1 = torch.autograd.Variable(torch.Tensor(16, 3, 256, 256))
    input2 = torch.autograd.Variable(torch.Tensor(16, 3, 256, 256))
    input3 = torch.autograd.Variable(torch.Tensor(16, 3, 256, 256))



    g = Feature_Generator_ResNet18()
    e = Feature_Embedder_ResNet18()
    cat_cls = CatClassifier()
    c = Classifier()
    r = Feature_Recnet()

    feature1 = g(input1)
    feature2 = g(input2)
    feature3 = g(input3)
    out = r(torch.cat([feature1, feature2],1))

    pad1 = e(g(input1))
    pad2 = e(g(input2))
    pad3 = e(g(input3))

    x = c(e(g(input1)))
    pad = c(torch.cat([pad1, pad2, pad3],0))
    cat_fea = torch.cat([pad1, pad2, pad3], 1)
    cat = cat_cls(cat_fea)
    # model = Feature_Extractor_ResNet18()
    # out = model(input)
    print(out.shape)
    print(feature1.shape)
    # print(cat.shape)
