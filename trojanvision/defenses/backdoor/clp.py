import argparse

import torch
from torch.utils.data import DataLoader

from IBAU.defense import MappedDataset
from ..backdoor_defense import BackdoorDefense
# from CLP.test import *
from CLP.test import CLP as clp_alg, val
from trojanzoo.utils.data import split_dataset


class CLP(BackdoorDefense):
    name: str = 'clp'

    def __init__(self, u=3., batch_size=500, **kwargs):
        super().__init__(**kwargs)
        self.u = u
        self.batch_size = batch_size

    @classmethod
    def add_argument(cls, group: argparse._ArgumentGroup):
        super().add_argument(group)
        group.add_argument('--clp_u', default=3., type=float, help='threshold hyperparameter')
        group.add_argument('--clp_batch_size', default=500, type=int, metavar='N', help='batch size.')

    def detect(self, **kwargs):
        super().detect(**kwargs)
        test_set = self.dataset.get_dataset('valid')
        train_set = self.dataset.get_dataset('train')
        mark = self.attack.mark
        mapped_dataset = MappedDataset(test_set, 0, mark)  # todo use all marks
        test_loader = DataLoader(test_set)
        train_loader = DataLoader(train_set)
        # test_loader = self.dataset.get_dataloader(mode='test', shuffle=True)
        # train_loader = self.dataset.get_dataloader(mode='train', shuffle=True)
        val_loader = test_loader
        test_set, unl_set = split_dataset(test_set, percent=0.1)
        clean_loader = DataLoader(test_set)
        unl_loader = DataLoader(unl_set)
        poiloader = torch.utils.data.DataLoader(mapped_dataset, batch_size=self.batch_size)
        net = self.model.model

        # if isinstance(self.attack, BadNet):
        if hasattr(self.attack, 'validate_fn'):
            print('pre-defense evaluation')
            self.attack.validate_fn(loader=clean_loader)

        clp_alg(net, self.u)

        # if isinstance(self.attack, BadNet):
        if hasattr(self.attack, 'validate_fn'):
            print('post-defense evaluation')
            self.attack.validate_fn(loader=clean_loader)
