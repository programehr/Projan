import sys

import torch
from torch import nn
from torch.nn import functional as F
import trojanvision

from trojanvision.utils import summary
import argparse

import os
from trojanzoo.utils.io import DictReader

from trojanvision.marks import Watermark

from trojanzoo.subdataset import getSubDataset
from itertools import product
import numpy as np
from numpy import array as npa

from trojanvision.attacks.backdoor.prob.losses import *

white_path = 'square_white.png'
black_path = 'square_black.png'
mark1 = Watermark(white_path, mark_height=3, mark_width=3, height_offset=2, width_offset=2, data_shape=[1, 28, 28])
mark2 = Watermark(white_path, mark_height=3, mark_width=3, height_offset=10, width_offset=10, data_shape=[1, 28, 28])


def compare_loader_stat(loader1, loader2):
    for x, y in loader1:
        break
    xsum1 = torch.zeros(x.shape[1:])
    ysum1 = torch.zeros(y.shape[1:])
    for x, y in loader1:
        xsum1 += x.sum(0)
        ysum1 += y.sum(0)
    xsum2 = torch.zeros(x.shape[1:])
    ysum2 = torch.zeros(y.shape[1:])
    for x, y in loader2:
        xsum2 += x.sum(0)
        ysum2 += y.sum(0)
    return torch.allclose(xsum1, xsum2) and torch.allclose(ysum1, ysum2)


def compare_ds(ds1, ds2):
    for i, k in enumerate(ds1.loader):
        v1 = ds1.loader[k]
        v2 = ds2.loader[k]
        if not compare_loader_stat(v1, v2):
            return False
    return True


def test_submnist():
    os.chdir('..')
    parser = argparse.ArgumentParser()

    attack_name = 'prob'
    dataset_name = 'submnist'
    args0 = ['--attack', attack_name, '--validate_interval', '10', '--lr', '1e-2', '--save', '--model', 'simplenet',
             '--download', '--dataset', dataset_name,
             '--epoch', '5', '--verbose', '5',
             #'--cutout'
             #'--OptimType', 'Adam',

             '--mark_path', white_path,
             '--mark_height', '3',
             '--mark_width', '3',
             '--height_offset', '2',
             '--width_offset', '2',
             '--batch_size', '128',
             '--extra_mark', f'mark_path={white_path} mark_height=3 mark_width=3 height_offset=23 width_offset=23',
             '--subset_percent', '0.001'
             ]

    type_map = {'mark_height': int, 'mark_width': int, 'height_offset': int, 'width_offset': int}
    parser.add_argument('--extra_mark', action=DictReader, nargs='*', dest='extra_marks', type_map=type_map)

    trojanvision.environ.add_argument(parser)
    trojanvision.datasets.add_argument(parser, dataset_name=dataset_name)
    trojanvision.models.add_argument(parser)
    trojanvision.trainer.add_argument(parser)
    trojanvision.marks.add_argument(parser)
    trojanvision.attacks.add_argument(parser, attack_name=attack_name)
    args = parser.parse_args(args0)

    env = trojanvision.environ.create(**args.__dict__)
    dataset = trojanvision.datasets.create(**args.__dict__)
    #dataset = trojanzoo.datasets.subDataset(dataset, 0.1)
    model = trojanvision.models.create(dataset=dataset, **args.__dict__)
    trainer = trojanvision.trainer.create(dataset=dataset, model=model, **args.__dict__)
    mark = trojanvision.marks.create(dataset=dataset, **args.__dict__)
    m = args.extra_marks[0]
    mark2 = trojanvision.marks.create(dataset=dataset, **m)

    attack = trojanvision.attacks.create(dataset=dataset, model=model, mark=mark, mark2=mark2, **args.__dict__)

    if env['verbose']:
        summary(env=env, dataset=dataset, model=model, mark=mark, trainer=trainer, attack=attack)
    attack.attack(**trainer)

def test_submnist2():
    os.chdir('..')
    parser = argparse.ArgumentParser()

    attack_name = 'prob'
    dataset_name = 'submnist'
    args0 = ['--attack', attack_name, '--validate_interval', '10', '--lr', '1e-2', '--save', '--model', 'simplenet',
             '--download', '--dataset', dataset_name,
             '--epoch', '1', '--verbose', '5',
             #'--cutout'
             #'--OptimType', 'Adam',

             '--mark_path', white_path,
             '--mark_height', '3',
             '--mark_width', '3',
             '--height_offset', '2',
             '--width_offset', '2',
             '--batch_size', '128',
             '--extra_mark', f'mark_path={white_path} mark_height=3 mark_width=3 height_offset=23 width_offset=23',
             '--extra_mark', f'mark_path={white_path} mark_height=3 mark_width=3 height_offset=2 width_offset=23',
             '--extra_mark', f'mark_path={white_path} mark_height=3 mark_width=3 height_offset=23 width_offset=2',
             '--subset_percent', '0.001'
             ]

    type_map = {'mark_height': int, 'mark_width': int, 'height_offset': int, 'width_offset': int}
    parser.add_argument('--extra_mark', action=DictReader, nargs='*', dest='extra_marks', type_map=type_map)

    trojanvision.environ.add_argument(parser)
    trojanvision.datasets.add_argument(parser, dataset_name=dataset_name)
    trojanvision.models.add_argument(parser)
    trojanvision.trainer.add_argument(parser)
    trojanvision.marks.add_argument(parser)
    trojanvision.attacks.add_argument(parser, attack_name=attack_name)
    args = parser.parse_args(args0)

    env = trojanvision.environ.create(**args.__dict__)
    dataset = trojanvision.datasets.create(**args.__dict__)
    #dataset = trojanzoo.datasets.subDataset(dataset, 0.1)
    model = trojanvision.models.create(dataset=dataset, **args.__dict__)
    trainer = trojanvision.trainer.create(dataset=dataset, model=model, **args.__dict__)
    mark = trojanvision.marks.create(dataset=dataset, **args.__dict__)
    # m = args.extra_marks[0]
    # mark2 = trojanvision.marks.create(dataset=dataset, **m)
    extra_marks = [trojanvision.marks.create(dataset=dataset, **m) for m in args.extra_marks]
    marks = [mark] + extra_marks

    attack = trojanvision.attacks.create(dataset=dataset, model=model, marks=marks, probs=[0.5, 0.5], **args.__dict__)

    if env['verbose']:
        summary(env=env, dataset=dataset, model=model, mark=mark, trainer=trainer, attack=attack)
    attack.attack(**trainer)

def test_submnist3(): #high epoch
    os.chdir('..')
    parser = argparse.ArgumentParser()

    attack_name = 'prob'
    dataset_name = 'submnist'
    args0 = ['--attack', attack_name, '--validate_interval', '10', '--lr', '1e-2', '--save', '--model', 'simplenet',
             '--download', '--dataset', dataset_name,
             '--epoch', '100', '--verbose', '5',
             #'--cutout'
             #'--OptimType', 'Adam',

             '--mark_path', white_path,
             '--mark_height', '3',
             '--mark_width', '3',
             '--height_offset', '2',
             '--width_offset', '2',
             '--batch_size', '128',
             '--extra_mark', f'mark_path={white_path} mark_height=3 mark_width=3 height_offset=23 width_offset=23',
             '--subset_percent', '0.001'
             ]

    type_map = {'mark_height': int, 'mark_width': int, 'height_offset': int, 'width_offset': int}
    parser.add_argument('--extra_mark', action=DictReader, nargs='*', dest='extra_marks', type_map=type_map)

    trojanvision.environ.add_argument(parser)
    trojanvision.datasets.add_argument(parser, dataset_name=dataset_name)
    trojanvision.models.add_argument(parser)
    trojanvision.trainer.add_argument(parser)
    trojanvision.marks.add_argument(parser)
    trojanvision.attacks.add_argument(parser, attack_name=attack_name)
    args = parser.parse_args(args0)

    env = trojanvision.environ.create(**args.__dict__)
    dataset = trojanvision.datasets.create(**args.__dict__)
    #dataset = trojanzoo.datasets.subDataset(dataset, 0.1)
    model = trojanvision.models.create(dataset=dataset, **args.__dict__)
    trainer = trojanvision.trainer.create(dataset=dataset, model=model, **args.__dict__)
    mark = trojanvision.marks.create(dataset=dataset, **args.__dict__)
    m = args.extra_marks[0]
    mark2 = trojanvision.marks.create(dataset=dataset, **m)

    attack = trojanvision.attacks.create(dataset=dataset, model=model, marks=[mark, mark2], probs=[0.5, 0.5], **args.__dict__)

    if env['verbose']:
        summary(env=env, dataset=dataset, model=model, mark=mark, trainer=trainer, attack=attack)
    attack.attack(**trainer)

def test_mnist_subds():
    os.chdir('..')
    parser = argparse.ArgumentParser()

    attack_name = 'prob'
    dataset_name = 'mnist'
    args0 = ['--attack', attack_name, '--validate_interval', '10', '--lr', '1e-2', '--save', '--model', 'simplenet',
             '--download', '--dataset', dataset_name,
             '--epoch', '1', '--verbose', '5',
             #'--cutout'
             #'--OptimType', 'Adam',

             '--mark_path', white_path,
             '--mark_height', '3',
             '--mark_width', '3',
             '--height_offset', '2',
             '--width_offset', '2',
             '--batch_size', '128',
             '--extra_mark', f'mark_path={white_path} mark_height=3 mark_width=3 height_offset=23 width_offset=23',
             #'--subset_percent', '0.001'
             ]

    type_map = {'mark_height': int, 'mark_width': int, 'height_offset': int, 'width_offset': int}
    parser.add_argument('--extra_mark', action=DictReader, nargs='*', dest='extra_marks', type_map=type_map)

    trojanvision.environ.add_argument(parser)
    trojanvision.datasets.add_argument(parser, dataset_name=dataset_name)
    trojanvision.models.add_argument(parser)
    trojanvision.trainer.add_argument(parser)
    trojanvision.marks.add_argument(parser)
    trojanvision.attacks.add_argument(parser, attack_name=attack_name)
    args = parser.parse_args(args0)

    env = trojanvision.environ.create(**args.__dict__)
    dataset = trojanvision.datasets.create(**args.__dict__)
    dataset = getSubDataset(dataset, 0.001)
    model = trojanvision.models.create(dataset=dataset, **args.__dict__)
    trainer = trojanvision.trainer.create(dataset=dataset, model=model, **args.__dict__)
    mark = trojanvision.marks.create(dataset=dataset, **args.__dict__)
    m = args.extra_marks[0]
    mark2 = trojanvision.marks.create(dataset=dataset, **m)

    attack = trojanvision.attacks.create(dataset=dataset, model=model, mark=mark, mark2=mark2, **args.__dict__)

    if env['verbose']:
        summary(env=env, dataset=dataset, model=model, mark=mark, trainer=trainer, attack=attack)
    attack.attack(**trainer)


def test_cifar10_subds():
    os.chdir('..')
    parser = argparse.ArgumentParser()

    attack_name = 'prob'
    dataset_name = 'cifar10'
    args0 = ['--attack', attack_name, '--validate_interval', '10', '--lr', '1e-2', '--save', '--model', 'resnet18_comp',
             '--download', '--dataset', dataset_name,
             '--epoch', '1', '--verbose', '5',
             #'--cutout'
             #'--OptimType', 'Adam',

             '--mark_path', white_path,
             '--mark_height', '3',
             '--mark_width', '3',
             '--height_offset', '2',
             '--width_offset', '2',
             '--batch_size', '128',
             '--extra_mark', f'mark_path={white_path} mark_height=3 mark_width=3 height_offset=23 width_offset=23',
             #'--subset_percent', '0.001',
             ]

    type_map = {'mark_height': int, 'mark_width': int, 'height_offset': int, 'width_offset': int}
    parser.add_argument('--extra_mark', action=DictReader, nargs='*', dest='extra_marks', type_map=type_map)

    trojanvision.environ.add_argument(parser)
    trojanvision.datasets.add_argument(parser, dataset_name=dataset_name)
    trojanvision.models.add_argument(parser)
    trojanvision.trainer.add_argument(parser)
    trojanvision.marks.add_argument(parser)
    trojanvision.attacks.add_argument(parser, attack_name=attack_name)
    args = parser.parse_args(args0)

    env = trojanvision.environ.create(**args.__dict__)
    dataset = trojanvision.datasets.create(**args.__dict__)
    dataset = getSubDataset(dataset, 0.005)
    model = trojanvision.models.create(dataset=dataset, **args.__dict__)
    trainer = trojanvision.trainer.create(dataset=dataset, model=model, **args.__dict__)
    mark = trojanvision.marks.create(dataset=dataset, **args.__dict__)
    m = args.extra_marks[0]
    mark2 = trojanvision.marks.create(dataset=dataset, **m)

    attack = trojanvision.attacks.create(dataset=dataset, model=model, marks=[mark, mark2], probs=[0.5, 0.5], **args.__dict__)

    if env['verbose']:
        summary(env=env, dataset=dataset, model=model, mark=mark, trainer=trainer, attack=attack)
    attack.attack(**trainer)


def test_equal_ds():
    os.chdir('..')
    parser = argparse.ArgumentParser()

    attack_name = 'prob'
    dataset_name = 'mnist'
    args00 = ['--attack', attack_name, '--validate_interval', '10', '--lr', '1e-2', '--save', '--model', 'simplenet',
             '--download', '--dataset', dataset_name,
             '--epoch', '1', '--verbose', '5',
             #'--cutout'
             #'--OptimType', 'Adam',

             '--mark_path', white_path,
             '--mark_height', '3',
             '--mark_width', '3',
             '--height_offset', '2',
             '--width_offset', '2',
             '--batch_size', '128',
             '--extra_mark', f'mark_path={white_path} mark_height=3 mark_width=3 height_offset=23 width_offset=23',

             ]

    args0 = args00

    type_map = {'mark_height': int, 'mark_width': int, 'height_offset': int, 'width_offset': int}
    parser.add_argument('--extra_mark', action=DictReader, nargs='*', dest='extra_marks', type_map=type_map)

    trojanvision.environ.add_argument(parser)
    trojanvision.datasets.add_argument(parser, dataset_name=dataset_name)
    trojanvision.models.add_argument(parser)
    trojanvision.trainer.add_argument(parser)
    trojanvision.marks.add_argument(parser)
    trojanvision.attacks.add_argument(parser, attack_name=attack_name)
    args = parser.parse_args(args0)

    dataset0 = trojanvision.datasets.create(**args.__dict__)
    dataset = getSubDataset(dataset0, 0.001)

    args0 = args00 + ['--subset_percent', '0.001']
    ix = args0.index('--dataset')+1
    args0[ix] = 'submnist'

    parser = argparse.ArgumentParser()
    parser.add_argument('--extra_mark', action=DictReader, nargs='*', dest='extra_marks', type_map=type_map)

    trojanvision.environ.add_argument(parser)
    trojanvision.datasets.add_argument(parser, dataset_name='submnist')
    trojanvision.models.add_argument(parser)
    trojanvision.trainer.add_argument(parser)
    trojanvision.marks.add_argument(parser)
    trojanvision.attacks.add_argument(parser, attack_name=attack_name)
    args = parser.parse_args(args0)

    dataset2 = trojanvision.datasets.create(**args.__dict__)
    assert compare_ds(dataset, dataset2)


def test_help():
    os.chdir('..')
    parser = argparse.ArgumentParser()

    attack_name = 'prob'
    dataset_name = 'submnist'
    args0 = ['-h']

    trojanvision.environ.add_argument(parser)
    trojanvision.datasets.add_argument(parser, dataset_name=dataset_name)
    trojanvision.models.add_argument(parser)
    trojanvision.trainer.add_argument(parser)
    trojanvision.marks.add_argument(parser)
    trojanvision.attacks.add_argument(parser, attack_name=attack_name)
    parser.print_help()

def test_losses():
    def loss1_(output, output1, output2, label, target):
        return torch.nn.CrossEntropyLoss()(output, label)

    def loss2_(output, output1, output2, label, target):
        _smout1 = F.softmax(output1, 1)
        return torch.abs(_smout1[:, target].sum() - _smout1.shape[0] / 2)

    def loss3_(output, output1, output2, label, target):
        _smout2 = F.softmax(output2, 1)
        return torch.abs(_smout2[:, target].sum() - _smout2.shape[0] / 2)

    def loss4_(output, output1, output2, label, target):
        _smout1 = F.softmax(output1, 1)
        _smout2 = F.softmax(output2, 1)
        return -torch.abs(_smout1[:, target] - _smout2[:, target]).sum()

    def loss1(output, mod_outputs, label, target, probs):
        return torch.nn.CrossEntropyLoss()(output, label)

    def loss2(output, mod_outputs, label, target, probs):
        n = len(mod_outputs)
        smouts = [None]*n
        part_loss = [None]*n
        for i in range(n):
            smouts[i] = F.softmax(mod_outputs[i], 1)
            part_loss[i] = torch.abs(smouts[i][:, target].sum() - smouts[i].shape[0]*probs[i])
        return sum(part_loss)

    def loss3(output, mod_outputs, label, target, probs):
        n = len(mod_outputs)
        smouts = [None]*n
        part_loss = [None]*(n-1)
        for i in range(n):
            smouts[i] = F.softmax(mod_outputs[i], 1)
        for i in range(n-1):
            part_loss[i] = -torch.abs(smouts[i][:, target] - smouts[i+1][:, target]).sum()
        return sum(part_loss)

    for i in range(10):
        output = torch.rand((20,10))
        output1 = torch.rand((20, 10))
        output2 = torch.rand((20, 10))
        label = torch.randint(0, 10, (20,))
        target = 0
        loss = \
        loss1_(output, output1, output2, label, target) + \
        loss2_(output, output1, output2, label, target) + \
        loss3_(output, output1, output2, label, target) + \
        loss4_(output, output1, output2, label, target)

        l1 = loss

        mod_outputs = [output1, output2]
        probs = [0.5, 0.5]
        loss = \
        loss1(output, mod_outputs, label, target, probs) + \
        loss2(output, mod_outputs, label, target, probs) + \
        loss3(output, mod_outputs, label, target, probs)

        l2 = loss
        assert l1.isclose(l2)


def test_cifar10():
    os.chdir('..')
    parser = argparse.ArgumentParser()

    attack_name = 'prob'
    dataset_name = 'cifar10'
    args0 = ['--attack', attack_name, '--validate_interval', '10', '--lr', '1e-2', '--save', '--model', 'resnet18_comp',
             '--download', '--dataset', dataset_name,
             '--epoch', '100', '--verbose', '5',
             #'--cutout'
             #'--OptimType', 'Adam',

             '--mark_path', white_path,
             '--mark_height', '3',
             '--mark_width', '3',
             '--height_offset', '2',
             '--width_offset', '2',
             '--batch_size', '128',
             '--extra_mark', f'mark_path={white_path} mark_height=3 mark_width=3 height_offset=23 width_offset=23',
             ]

    type_map = {'mark_height': int, 'mark_width': int, 'height_offset': int, 'width_offset': int}
    parser.add_argument('--extra_mark', action=DictReader, nargs='*', dest='extra_marks', type_map=type_map)

    trojanvision.environ.add_argument(parser)
    trojanvision.datasets.add_argument(parser, dataset_name=dataset_name)
    trojanvision.models.add_argument(parser)
    trojanvision.trainer.add_argument(parser)
    trojanvision.marks.add_argument(parser)
    trojanvision.attacks.add_argument(parser, attack_name=attack_name)
    args = parser.parse_args(args0)

    env = trojanvision.environ.create(**args.__dict__)
    dataset = trojanvision.datasets.create(**args.__dict__)
    #dataset = trojanzoo.datasets.subDataset(dataset, 0.1)
    model = trojanvision.models.create(dataset=dataset, **args.__dict__)
    trainer = trojanvision.trainer.create(dataset=dataset, model=model, **args.__dict__)
    mark = trojanvision.marks.create(dataset=dataset, **args.__dict__)
    m = args.extra_marks[0]
    mark2 = trojanvision.marks.create(dataset=dataset, **m)

    attack = trojanvision.attacks.create(dataset=dataset, model=model, marks=[mark, mark2], probs=[0.5, 0.5], **args.__dict__)

    if env['verbose']:
        summary(env=env, dataset=dataset, model=model, mark=mark, trainer=trainer, attack=attack)
    attack.attack(**trainer)

def test_logger():
    from trojanzoo.utils.logger import SmoothedValue, MetricLogger
    logger = MetricLogger()
    logger.meters['foo'] = SmoothedValue()
    for i in range(100):
        logger.meters['foo'].update(i, 15)
        print(logger.meters['foo'].avg, logger.meters['foo'].global_avg)

def test_mnist():
    os.chdir('..')
    parser = argparse.ArgumentParser()

    attack_name = 'prob'
    dataset_name = 'mnist'
    args0 = ['--attack', attack_name, '--validate_interval', '10', '--lr', '1e-2', '--save', '--model', 'simplenet',
             '--download', '--dataset', dataset_name,
             '--epoch', '100', '--verbose', '5',
             #'--cutout'
             #'--OptimType', 'Adam',

             '--mark_path', white_path,
             '--mark_height', '3',
             '--mark_width', '3',
             '--height_offset', '2',
             '--width_offset', '2',
             '--batch_size', '128',
             '--extra_mark', f'mark_path={white_path} mark_height=3 mark_width=3 height_offset=23 width_offset=23',
             ]

    type_map = {'mark_height': int, 'mark_width': int, 'height_offset': int, 'width_offset': int}
    parser.add_argument('--extra_mark', action=DictReader, nargs='*', dest='extra_marks', type_map=type_map)

    trojanvision.environ.add_argument(parser)
    trojanvision.datasets.add_argument(parser, dataset_name=dataset_name)
    trojanvision.models.add_argument(parser)
    trojanvision.trainer.add_argument(parser)
    trojanvision.marks.add_argument(parser)
    trojanvision.attacks.add_argument(parser, attack_name=attack_name)
    args = parser.parse_args(args0)

    env = trojanvision.environ.create(**args.__dict__)
    dataset = trojanvision.datasets.create(**args.__dict__)
    #dataset = trojanzoo.datasets.subDataset(dataset, 0.1)
    model = trojanvision.models.create(dataset=dataset, **args.__dict__)
    trainer = trojanvision.trainer.create(dataset=dataset, model=model, **args.__dict__)
    mark = trojanvision.marks.create(dataset=dataset, **args.__dict__)
    m = args.extra_marks[0]
    mark2 = trojanvision.marks.create(dataset=dataset, **m)

    attack = trojanvision.attacks.create(dataset=dataset, model=model, marks=[mark, mark2], probs=[0.5, 0.5], **args.__dict__)

    if env['verbose']:
        summary(env=env, dataset=dataset, model=model, mark=mark, trainer=trainer, attack=attack)
    attack.attack(**trainer)

def test_general(dataset_name, model_name, epoch, seed=1228, lr = 1e-3, subset_percent=None, device=None,
                 losses=None, cbeta_epoch=-1, interleave=False, init_loss_weights=None,
                 batchnorm_momentum = None, disable_batch_norm=True, pretrain_epoch=0, poison_percent=1.0,
                 pretrain = True, resume=0):
    #os.chdir('..')
    os.chdir(os.path.dirname(os.path.dirname(__file__)))
    parser = argparse.ArgumentParser()

    attack_name = 'prob'
    args0 = ['--attack', attack_name, '--validate_interval', '10', '--save', '--model', model_name,
             '--lr', str(lr),
             '--download', '--dataset', dataset_name,
             '--epoch', str(epoch), '--verbose', '5',
             #'--cutout'
             '--OptimType', 'Adam',

             '--mark_path', white_path,
             '--mark_height', '3',
             '--mark_width', '3',
             '--height_offset', '2',
             '--width_offset', '2',
             '--batch_size', '100',
             '--extra_mark', f'mark_path={white_path} mark_height=3 mark_width=3 height_offset=10 width_offset=10',
             # '--extra_mark', f'mark_path={white_path} mark_height=3 mark_width=3 height_offset=2 width_offset=23',
             # '--extra_mark', f'mark_path={white_path} mark_height=3 mark_width=3 height_offset=23 width_offset=2',

             '--seed', str(seed),
             '--cbeta_epoch', str(cbeta_epoch),
             '--disable_batch_norm', str(disable_batch_norm),
             '--pretrain_epoch', str(pretrain_epoch),
             '--poison_percent', str(poison_percent),
             '--resume', str(resume),
             ]

    if pretrain:
        args0 += ['--pretrain']
    if batchnorm_momentum:
        args0 += ['--batchnorm_momentum', str(batchnorm_momentum),]
    if device:
        args0 += ['--device', device]
    if init_loss_weights is not None:
        args0 += ['--init_loss_weights']
        args0 += [str(w) for w in init_loss_weights]
    # probs = [0.5, 0.5]
    # args0 += ['--probs'] + [str(prob) for prob in probs]

    # type_map = {'mark_height': int, 'mark_width': int, 'height_offset': int, 'width_offset': int}
    # parser.add_argument('--extra_mark', action=DictReader, nargs='*', dest='extra_marks', type_map=type_map)

    trojanvision.environ.add_argument(parser)
    trojanvision.datasets.add_argument(parser, dataset_name=dataset_name)
    trojanvision.models.add_argument(parser)
    trojanvision.trainer.add_argument(parser)
    trojanvision.marks.add_argument(parser)
    trojanvision.attacks.add_argument(parser, attack_name=attack_name)
    args = parser.parse_args(args0)

    env = trojanvision.environ.create(**args.__dict__)
    if 'mnist' in dataset_name:
        dataset = trojanvision.datasets.create(norm_par={'mean': [0.0], 'std': [0.65]}, **args.__dict__)
    else:
        dataset = trojanvision.datasets.create(**args.__dict__)

    if subset_percent:
        dataset = getSubDataset(dataset, subset_percent)
    model = trojanvision.models.create(dataset=dataset, dropout=0, **args.__dict__)
    trainer = trojanvision.trainer.create(dataset=dataset, model=model, **args.__dict__)
    mark = trojanvision.marks.create(dataset=dataset, **args.__dict__)
    # m = args.extra_marks[0]
    # mark2 = trojanvision.marks.create(dataset=dataset, **m)
    extra_marks = [trojanvision.marks.create(dataset=dataset, **m) for m in args.extra_marks]
    marks = [mark] + extra_marks

    attack = trojanvision.attacks.create(dataset=dataset, model=model, marks=marks,
                                         losses=losses, **args.__dict__)

    if env['verbose']:
        summary(env=env, dataset=dataset, model=model, mark=mark, trainer=trainer, attack=attack)
    attack.attack(**trainer)

if __name__ == '__main__':
    device = sys.argv[1]
    seeds = list(range(10))
    #for i in range(10):
    #test_general('cifar10', 'resnet18_comp', 50, 10, subset_percent=0.01, device='cpu')

    #for i, lr in enumerate([0.001, 0.005, 0.01, 0.5, 0.1]):
    losses = ['loss1', 'loss2_8', 'loss3_9']
    # test_general('mnist', 'simplenet', 200, 10, device=device, lr=0.01,
    #             losses=losses, init_loss_weights=npa([.99, .01]), cbeta_epoch=-1) # , subset_percent=0.1)
    resume = 0 if len(sys.argv)<=2 else sys.argv[2]

    #for i, lr in enumerate([0.001, 0.005]):
    #for i, bnm in enumerate([0.01, 0.02, 0.05, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 0.99]):
    #for i, bnm in enumerate([0.05, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 0.99]):
    test_general('cifar10', 'resnet18_comp', 200, 10, device=device, lr=0.001,
                losses=losses, init_loss_weights=[1., 1, 1.], cbeta_epoch=-1, pretrain_epoch=0, poison_percent=0.1,
                 pretrain=True, resume=resume) # , subset_percent=0.1)



