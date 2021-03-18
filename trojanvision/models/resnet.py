#!/usr/bin/env python3
from .imagemodel import _ImageModel, ImageModel
import trojanvision.utils.resnet_s

import torch
import torch.nn as nn
from torch.utils import model_zoo
import torchvision.models
from torchvision.models.resnet import model_urls
from collections import OrderedDict
from collections.abc import Callable


class _ResNet(_ImageModel):

    def __init__(self, layer: int = 18, comp: bool = False, **kwargs):
        super().__init__(**kwargs)
        layer = int(layer)
        ModelClass: Callable[..., torchvision.models.ResNet] = getattr(torchvision.models, 'resnet' + str(layer))
        _model = ModelClass(num_classes=self.num_classes)
        if comp:
            conv: nn.Conv2d = _model.conv1
            conv = nn.Conv2d(conv.in_channels, conv.out_channels,
                             kernel_size=3, stride=1, padding=1, bias=False)
            self.features = nn.Sequential(OrderedDict([
                ('conv1', conv),
                ('bn1', _model.bn1),  # nn.BatchNorm2d(64)
                ('relu', _model.relu),  # nn.ReLU(inplace=True)
                # nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                # ('maxpool', _model.maxpool),
                ('layer1', _model.layer1),
                ('layer2', _model.layer2),
                ('layer3', _model.layer3),
                ('layer4', _model.layer4)
            ]))
        else:
            self.features = nn.Sequential(OrderedDict([
                # nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
                ('conv1', _model.conv1),
                ('bn1', _model.bn1),  # nn.BatchNorm2d(64)
                ('relu', _model.relu),  # nn.ReLU(inplace=True)
                # nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                ('maxpool', _model.maxpool),
                ('layer1', _model.layer1),
                ('layer2', _model.layer2),
                ('layer3', _model.layer3),
                ('layer4', _model.layer4)
            ]))
        self.pool = _model.avgpool  # nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(OrderedDict([
            ('fc', _model.fc)  # nn.Linear(512 * block.expansion, num_classes)
        ]))
        # block.expansion = 1 if BasicBlock and 4 if Bottleneck
        # ResNet 18,34 use BasicBlock, 50 and higher use Bottleneck


class ResNet(ImageModel):

    def __init__(self, name: str = 'resnet', layer: int = 18,
                 model: type[_ResNet] = _ResNet, **kwargs):
        comp = True if 'comp' in name else False
        super().__init__(name=name, layer=layer, model=model, comp=comp, **kwargs)

    def get_official_weights(self, **kwargs) -> OrderedDict[str, torch.Tensor]:
        url = model_urls[f'resnet{self.layer:d}']
        print('get official model weights from: ', url)
        _dict: OrderedDict[str, torch.Tensor] = model_zoo.load_url(url, **kwargs)
        new_dict = OrderedDict()
        for i, (key, value) in enumerate(_dict.items()):
            prefix = 'features.' if i < len(_dict) - 2 else 'classifier.'
            new_dict[prefix + key] = value
        return new_dict


class _ResNetS(_ResNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _model = trojanvision.utils.resnet_s.ResNetS(nclasses=self.num_classes)
        self.features = nn.Sequential(OrderedDict([
            ('conv1', _model.conv1),
            ('bn1', _model.bn1),  # nn.BatchNorm2d(64)
            ('relu', nn.ReLU(inplace=True)),
            # nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            ('layer1', _model.layer1),
            ('layer2', _model.layer2),
            ('layer3', _model.layer3),
            ('layer4', _model.layer4)
        ]))
        self.classifier = nn.Sequential(OrderedDict([
            ('fc', _model.linear)  # nn.Linear(512 * block.expansion, num_classes)
        ]))


class ResNetS(ResNet):
    def __init__(self, name: str = 'resnets', layer: int = 18,
                 model: type[_ResNetS] = _ResNetS, **kwargs):
        super().__init__(name=name, layer=layer, model=model, **kwargs)
