import torch
import trojanvision

from trojanvision.utils import summary
import argparse

import os
from trojanzoo.utils.io import DictReader

from trojanvision.marks import Watermark

from trojanzoo.subdataset import getSubDataset

white_path = 'square_white.png'
mark1 = Watermark(white_path, mark_height=3, mark_width=3, height_offset=2, width_offset=2, data_shape=[1, 28, 28])
mark2 = Watermark(white_path, mark_height=3, mark_width=3, height_offset=23, width_offset=23, data_shape=[1, 28, 28])


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


if __name__ == '__main__':
    test_mnist_subds()


