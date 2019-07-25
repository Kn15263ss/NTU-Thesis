import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from MobileNetV2 import MobileNetV2


class Vgg11(nn.Module):

    def __init__(self, classCount, input_size):

        super(Vgg11, self).__init__()

        self.vgg11 = torchvision.models.vgg11(pretrained=True)

        self.vgg11.classifier = nn.Sequential(
            nn.Linear(input_size, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, classCount)
        )

    def forward(self, x):
        x = self.vgg11.features(x)
        # print(x.size())
        x = x.view(x.size(0), -1)
        x = self.vgg11.classifier(x)
        return x


class Resnet18(nn.Module):

    def __init__(self, classCount, input_size):

        super(Resnet18, self).__init__()

        self.resnet18 = torchvision.models.resnet18(pretrained=True)

        self.resnet18.fc = nn.Linear(512, classCount)

    def forward(self, x):
        x = self.resnet18(x)
        return x


class MobileNet(nn.Module):

    def __init__(self, classCount, input_size):
        super(MobileNet, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),

                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.model = nn.Sequential(
            conv_bn(3, 32, 2),
            conv_dw(32, 64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
            nn.AvgPool2d(7),
        )
        self.fc = nn.Linear(1024, classCount)

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class MobileNet_V2(nn.Module):

    def __init__(self, classCount, input_size):

        super(MobileNet_V2, self).__init__()

        self.mobilenetv2 = MobileNetV2()

        state_dict = torch.load(
            '/home/kn15263s/workspace/mobilenet_v2.pth.tar')
        self.mobilenetv2.load_state_dict(state_dict)

        self.mobilenetv2.classifier[-1] = nn.Linear(1280, 10)

    def forward(self, x):
        x = self.mobilenetv2(x)
        return x

