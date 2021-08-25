import trojanvision

from trojanvision.utils import summary
import argparse

import os


#def create_toy_ds():

def test0():
    os.chdir('..')
    print(os.getcwd())

def test():
    os.chdir('..')
    parser = argparse.ArgumentParser()

    attack_name = 'prob'
    dataset_name = 'submnist'
    args0 = ['--attack', attack_name, '--validate_interval', '10', '--lr', '1e-2', '--save', '--model', 'simplenet',
             '--download', '--dataset', dataset_name,
             '--epoch', '5', '--verbose', '5',
             '--subset_percent', '0.001',
             #'--cutout'
             ]

    trojanvision.environ.add_argument(parser)
    trojanvision.datasets.add_argument(parser, dataset_name=dataset_name)
    trojanvision.models.add_argument(parser)
    trojanvision.trainer.add_argument(parser)
    trojanvision.marks.add_argument(parser)
    trojanvision.attacks.add_argument(parser, attack_name=attack_name)
    args = parser.parse_args(args0)

    env = trojanvision.environ.create(**args.__dict__)
    dataset = trojanvision.datasets.create(**args.__dict__)
    model = trojanvision.models.create(dataset=dataset, **args.__dict__)
    trainer = trojanvision.trainer.create(dataset=dataset, model=model, **args.__dict__)
    mark = trojanvision.marks.create(dataset=dataset, **args.__dict__)
    attack = trojanvision.attacks.create(dataset=dataset, model=model, mark=mark, **args.__dict__)

    if env['verbose']:
        summary(env=env, dataset=dataset, model=model, mark=mark, trainer=trainer, attack=attack)
    attack.attack(**trainer)


if __name__ == '__main__':
    pass
