import logging
import sys

import torch
from torch import nn
from torch.nn import init
from torch.hub import load_state_dict_from_url

from repro_vision.functions.loss import CrossEntropyLoss

model_urls = {
    'squeezenet1_0': 'https://download.pytorch.org/models/'
                     'squeezenet1_0-a815701f.pth',
    'squeezenet1_1': 'https://download.pytorch.org/models/'
                     'squeezenet1_1-f364aa15.pth',
}


class Fire(nn.Module):
    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes):
        super(Fire, self).__init__()
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes,
                                   kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes,
                                   kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)


class SqueezeNetBase(nn.Module):
    def __init__(self, n_classes):
        super(SqueezeNetBase, self).__init__()
        # Final convolution is initialized differently from the rest
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(512, n_classes, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def fine_tuning(self, version, progress=True):
        model_name = f'squeezenet{version}'
        state_dict = load_state_dict_from_url(model_urls[model_name],
                                              progress=progress)
        state_dict = {key: value for key, value in state_dict.items()
                      if 'features' in key}

        self.load_state_dict(state_dict, strict=False)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return torch.flatten(x, 1)

    def predict(self, x):
        return self.forward(x)


class SqueezeNet10(SqueezeNetBase):
    def __init__(self, n_classes, pretrained=True, progress=True):
        self.features = nn.Sequential(
                nn.Conv2d(3, 96, kernel_size=7, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(96, 16, 64, 64),
                Fire(128, 16, 64, 64),
                Fire(128, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 32, 128, 128),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(512, 64, 256, 256),
            )
        self._init_params()
        if pretrained:
            self.fine_tuning('1_0', progress)


class SqueezeNet11(SqueezeNetBase):
    def __init__(self, n_class, pretrained=True, progress=True, logger=None):
        super(SqueezeNet11, self).__init__(n_class)
        self.logger = logger or logging.getLogger(__name__)
        self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(64, 16, 64, 64),
                Fire(128, 16, 64, 64),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(128, 32, 128, 128),
                Fire(256, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                Fire(512, 64, 256, 256),
            )
        self._init_params()
        if pretrained:
            self.fine_tuning('1_1', progress)


def get_model(net_name, net_params, loss_params, n_class, logger=None,
              **kwargs):
    net_params = net_params if net_params else {}
    loss_params = loss_params if loss_params else {}
    net = getattr(sys.modules[__name__], net_name)(n_class=n_class, **net_params)  # noqa
    loss = CrossEntropyLoss(**loss_params)
    return net, loss
