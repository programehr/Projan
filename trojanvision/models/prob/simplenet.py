# -*- coding: utf-8 -*-
"""
Created on Sat May 15 14:41:49 2021

@author: Mehr
"""

import torch.nn.functional as F

from trojanvision.models.imagemodel import _ImageModel, ImageModel

import torch
import torch.nn as nn
import torchvision.models
from torchvision.models.resnet import model_urls as urls
from collections import OrderedDict
from collections.abc import Callable
import warnings

# %%
class _Net(nn.Module):
    def __init__(self):
        super(_Net, self).__init__()
        self.kernels = nn.Conv2d(1, 10, 3, padding=1)
        self.kernels2 = nn.Conv2d(10, 5, 3, padding=1)
        '''
        also change forward func
        self.fc1=nn.Linear(3920, 6000)
        self.fc2=nn.Linear(6000, 4000)
        self.fc3=nn.Linear(4000, 2000)
        self.fc4=nn.Linear(2000, 1000)
        self.fc5=nn.Linear(1000, 10)
        '''
        self.fc1 = nn.Linear(3920, 2000)
        self.fc2 = nn.Linear(2000, 1000)
        self.fc3 = nn.Linear(1000, 10)

    def forward(self, input):
        output = self.kernels(input)
        output = F.relu(output)

        output = self.kernels2(output)
        output = F.relu(output)

        output = output.reshape((-1, 3920))

        output = self.fc1(output)
        output = F.relu(output)

        output = self.fc2(output)
        output = F.relu(output)

        output = self.fc3(output)
        '''
        output=F.relu(output)

        output = self.fc4(output)
        output=F.relu(output)

        output = self.fc5(output)
        '''
        #output = F.softmax(output, 1) loss would take care of it

        return output

class _SimpleNet(_ImageModel):
    def __init__(self, name: str = 'simplenet', **kwargs):
        super().__init__(**kwargs)
        _model = _Net()
        self.classifier = nn.Sequential(OrderedDict([('fc1', _model.fc1),
                                                     ('fc2', _model.fc2),
                                                     ('fc3', _model.fc3)]))
        #todo this implementation is error-prone
        #if one changes the class _Net, this class may cancel the changes.
        self.features = nn.Sequential(OrderedDict([('conv1', _model.kernels),
                                                   ('relu1', F.relu),
                                                   ('conv2', _model.kernels2),
                                                   ('relu2', F.relu)]))

        self.pool = nn.Identity()


class SimpleNet(ImageModel):
    available_models = ['simplenet']

    def __init__(self, name: str = 'simplenet',
                 model: type[_SimpleNet] = _SimpleNet, **kwargs):
        if 'layer' in kwargs and kwargs['layer'] is not None:
            warnings.warn('parameter "layer" does not apply to this model.')
        super().__init__(name=name, layer=None, model=model, **kwargs)

    def get_name(cls, name: str, layer: int = None) -> str:
        return 'simplenet'





















