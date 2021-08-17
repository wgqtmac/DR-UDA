# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3x3(in_planes, out_planes, stride=1, bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=bias)

class BasicBlock(nn.Module):
    expansion = 1
    bias = False

    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, bias=self.bias)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes*self.expansion, 1, bias=self.bias)
        self.bn2 = nn.BatchNorm2d(planes)
        if stride != 1 or inplanes != planes*self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes*self.expansion, kernel_size=1, stride=stride, bias=self.bias),
                nn.BatchNorm2d(planes*self.expansion) )
        else:
            self.downsample = nn.Sequential()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class PreBasicBlock(nn.Module):
    expansion = 1
    bias = False

    def __init__(self, inplanes, planes, stride=1, ptype='preact'):
        super(PreBasicBlock, self).__init__()
        if ptype != 'no_preact':
            self.preact = nn.Sequential(
                            nn.BatchNorm2d(inplanes),
                            nn.ReLU(inplace=True) )
        self.conv1 = conv3x3(inplanes, planes, stride, bias=self.bias)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes*self.expansion, 1, bias=self.bias)
        if stride != 1 or inplanes != planes*self.expansion:
            self.downsample = nn.Conv2d(inplanes, planes*self.expansion, kernel_size=1, stride=stride, bias=self.bias)
        else:
            self.downsample = nn.Sequential()
        self.ptype = ptype

    def forward(self, x):
        if self.ptype == 'both_preact':
            x = self.preact(x)
        residual = x
        if self.ptype != 'no_preact' and self.ptype != 'both_preact':
            x = self.preact(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)

        residual = self.downsample(residual)
        out += residual
        return out

class Bottleneck(nn.Module):
    expansion = 4
    bias = False

    def __init__(self, inplanes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=self.bias)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=self.bias)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes*self.expansion, kernel_size=1, bias=self.bias)
        self.bn3 = nn.BatchNorm2d(planes*self.expansion)
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or inplanes != planes*self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes*self.expansion, kernel_size=1, stride=stride, bias=self.bias),
                nn.BatchNorm2d(planes*self.expansion) )
        else:
            self.downsample = nn.Sequential()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class PreBottleneck(nn.Module):
    expansion = 4
    bias = False

    def __init__(self, inplanes, planes, stride=1, ptype='preact'):
        super(PreBottleneck, self).__init__()
        if ptype != 'no_preact':
            self.preact = nn.Sequential(
                            nn.BatchNorm2d(inplanes),
                            nn.ReLU(inplace=True) )
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=self.bias)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=self.bias)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=self.bias)
        self.relu = nn.ReLU(inplace=True)
        if stride != 1 or inplanes != planes*self.expansion:
            self.downsample = nn.Conv2d(inplanes, planes*self.expansion, kernel_size=1, stride=stride, bias=self.bias)
        else:
            self.downsample = nn.Sequential()
        self.ptype = ptype

    def forward(self, x):
        if self.ptype == 'both_preact':
            x = self.preact(x)
        residual = x
        if self.ptype != 'no_preact' and self.ptype != 'both_preact':
            x = self.preact(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)

        residual = self.downsample(residual)
        out += residual
        return out

class BottleneckX(nn.Module):
    expansion = 4
    bias = False

    def __init__(self, inplanes, planes, baseWidth, cardinality, stride=1):
        super(BottleneckX, self).__init__()
        D = math.floor(planes * baseWidth / 64.0)

        conv1, bn1, conv2, bn2 = [], [], [], []
        for i in range(cardinality):
            conv1.append(nn.Conv2d(inplanes, D, kernel_size=1, bias=self.bias))
            bn1.append(nn.BatchNorm2d(D))
            conv2.append(nn.Conv2d(D, D, kernel_size=3, stride=stride,
                                   padding=1, bias=self.bias))
            bn2.append(nn.BatchNorm2d(D))
        self.conv1 = nn.ModuleList(conv1)
        self.bn1 = nn.ModuleList(bn1)
        self.conv2 = nn.ModuleList(conv2)
        self.bn2 = nn.ModuleList(bn2)

        self.conv3 = nn.Conv2d(D*cardinality, planes*self.expansion, kernel_size=1, bias=self.bias)
        self.bn3 = nn.BatchNorm2d(planes*self.expansion)
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or inplanes != planes*self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes*self.expansion, kernel_size=1, stride=stride, bias=self.bias),
                nn.BatchNorm2d(planes*self.expansion) )
        else:
            self.downsample = nn.Sequential()

        self.cardinality = cardinality

    def forward(self, x):
        out = []
        for i in range(self.cardinality):
            y = self.conv1[i](x)
            y = self.bn1[i](y)
            y = self.relu(y)

            y = self.conv2[i](y)
            y = self.bn2[i](y)
            y = self.relu(y)

            out.append(y)

        out = torch.cat(out, dim=1)
        out = self.conv3(out)
        out = self.bn3(out)

        residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class PreBottleneckX(nn.Module):
    expansion = 4
    bias = False

    def __init__(self, inplanes, planes, baseWidth, cardinality, stride=1, ptype='preact'):
        super(PreBottleneckX, self).__init__()
        D = math.floor(planes * baseWidth / 64.0)

        if ptype != 'no_preact':
            self.preact = nn.Sequential(
                            nn.BatchNorm2d(inplanes),
                            nn.ReLU(inplace=True) )
        conv1, bn1, conv2, bn2 = [], [], [], []
        for i in range(cardinality):
            conv1.append(nn.Conv2d(inplanes, D, kernel_size=1, bias=self.bias))
            bn1.append(nn.BatchNorm2d(D))
            conv2.append(nn.Conv2d(D, D, kernel_size=3, stride=stride,
                                   padding=1, bias=self.bias))
            bn2.append(nn.BatchNorm2d(D))
        self.conv1 = nn.ModuleList(conv1)
        self.bn1 = nn.ModuleList(bn1)
        self.conv2 = nn.ModuleList(conv2)
        self.bn2 = nn.ModuleList(bn2)

        self.conv3 = nn.Conv2d(D*cardinality, planes * self.expansion, kernel_size=1, bias=self.bias)
        self.relu = nn.ReLU(inplace=True)
        if stride != 1 or inplanes != planes*self.expansion:
            self.downsample = nn.Conv2d(inplanes, planes*self.expansion, kernel_size=1, stride=stride, bias=self.bias)
        else:
            self.downsample = nn.Sequential()

        self.cardinality = cardinality
        self.ptype = ptype

    def forward(self, x):
        if self.ptype == 'both_preact':
            x = self.preact(x)
        residual = x
        if self.ptype != 'no_preact' and self.ptype != 'both_preact':
            x = self.preact(x)

        out = []
        for i in range(self.cardinality):
            y = self.conv1[i](x)
            y = self.bn1[i](y)
            y = self.relu(y)

            y = self.conv2[i](y)
            y = self.bn2[i](y)
            y = self.relu(y)

            out.append(y)

        out = torch.cat(out, dim=1)
        out = self.conv3(out)

        residual = self.downsample(residual)
        out += residual
        return out

class PyramidBlock(nn.Module):
    expansion = 4
    bias = False

    def __init__(self, inplanes, planes, input_res, cardinality, stride=1):
        super(PyramidBlock, self).__init__()
        self.conv1_1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=self.bias)
        self.bn1_1 = nn.BatchNorm2d(planes)
        self.conv2_1 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=self.bias)
        self.bn2_1 = nn.BatchNorm2d(planes)
        self.conv3_1 = nn.Conv2d(planes, planes*self.expansion, kernel_size=1, bias=self.bias)
        self.bn3_1 = nn.BatchNorm2d(planes*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        if stride != 1 or inplanes != planes*self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes*self.expansion, kernel_size=1, stride=stride, bias=self.bias),
                nn.BatchNorm2d(planes*self.expansion) )
        else:
            self.downsample = nn.Sequential()

        self.conv1_2 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=self.bias)
        self.bn1_2 = nn.BatchNorm2d(planes)
        self.bn2_2 = nn.BatchNorm2d(planes)
        self.conv3_2 = nn.Conv2d(planes, planes*self.expansion, kernel_size=1, bias=self.bias)
        self.bn3_2 = nn.BatchNorm2d(planes*self.expansion)
        output_res = ((input_res[0]+1)/stride, (input_res[1]+1)/stride)
        pool, conv, upsample = [], [], []
        for i in range(cardinality):
            ratio = 1.0 / math.pow(2, (i+1.0)/cardinality)
            pool.append(nn.FractionalMaxPool2d(kernel_size=2, output_ratio=ratio))
            conv.append(nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                                   padding=1, bias=self.bias))
            upsample.append(nn.Upsample(size=output_res, mode='bilinear'))
        self.pool = nn.ModuleList(pool)
        self.conv = nn.ModuleList(conv)
        self.upsample = nn.ModuleList(upsample)

        self.cardinality = cardinality

    def forward(self, x):
        # branch 1
        out = self.conv1_1(x)
        out = self.bn1_1(out)
        out = self.relu(out)
        out = self.conv2_1(out)
        out = self.bn2_1(out)
        out = self.relu(out)
        out = self.conv3_1(out)
        out = self.bn3_1(out)
        # branch 2
        b = self.conv1_2(x)
        b = self.bn1_2(b)
        b = self.relu(b)
        for i in range(self.cardinality):
            z = self.pool[i](b)
            z = self.conv[i](z)
            z = self.upsample[i](z)
            if i == 0:
                y = z
            else:
                y += z
        b = y
        b = self.bn2_2(b)
        b = self.relu(b)
        b = self.conv3_2(b)
        b = self.bn3_2(b)
        out += b
        # Identity
        residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class PrePyramidBlock(nn.Module):
    expansion = 4
    bias = False

    def __init__(self, inplanes, planes, input_res, cardinality, stride=1, ptype='preact'):
        super(PrePyramidBlock, self).__init__()
        if ptype != 'no_preact':
            self.preact = nn.Sequential(
                            nn.BatchNorm2d(inplanes),
                            nn.ReLU(inplace=True) )
        self.conv1_1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=self.bias)
        self.bn1_1 = nn.BatchNorm2d(planes)
        self.conv2_1 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=self.bias)
        self.bn2_1 = nn.BatchNorm2d(planes)
        self.conv3_1 = nn.Conv2d(planes, planes*self.expansion, kernel_size=1, bias=self.bias)
        self.relu = nn.ReLU(inplace=True)
        if stride != 1 or inplanes != planes*self.expansion:
            self.downsample = nn.Conv2d(inplanes, planes*self.expansion, kernel_size=1, stride=stride, bias=self.bias)
        else:
            self.downsample = nn.Sequential()

        self.conv1_2 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=self.bias)
        self.bn1_2 = nn.BatchNorm2d(planes)
        self.bn2_2 = nn.BatchNorm2d(planes)
        self.conv3_2 = nn.Conv2d(planes, planes*self.expansion, kernel_size=1, bias=self.bias)
        output_res = ((input_res[0]+1)/stride, (input_res[1]+1)/stride)
        pool, conv, upsample = [], [], []
        for i in range(cardinality):
            ratio = 1.0 / math.pow(2, (i+1.0)/cardinality)
            pool.append(nn.FractionalMaxPool2d(kernel_size=2, output_ratio=ratio))
            conv.append(nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                                   padding=1, bias=self.bias))
            upsample.append(nn.Upsample(size=output_res, mode='bilinear'))
        self.pool = nn.ModuleList(pool)
        self.conv = nn.ModuleList(conv)
        self.upsample = nn.ModuleList(upsample)

        self.cardinality = cardinality
        self.ptype = ptype

    def forward(self, x):
        if self.ptype == 'both_preact':
            x = self.preact(x)
        residual = x
        if self.ptype != 'no_preact' and self.ptype != 'both_preact':
            x = self.preact(x)
        # branch 1
        out = self.conv1_1(x)
        out = self.bn1_1(out)
        out = self.relu(out)
        out = self.conv2_1(out)
        out = self.bn2_1(out)
        out = self.relu(out)
        out = self.conv3_1(out)
        # branch 2
        b = self.conv1_2(x)
        b = self.bn1_2(b)
        b = self.relu(b)
        for i in range(self.cardinality):
            z = self.pool[i](b)
            z = self.conv[i](z)
            z = self.upsample[i](z)
            if i == 0:
                y = z
            else:
                y += z
        b = y
        b = self.bn2_2(b)
        b = self.relu(b)
        b = self.conv3_2(b)
        out += b
        # Identity
        residual = self.downsample(residual)
        out += residual
        return out