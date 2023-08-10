import argparse

import torch
from torch.utils.data import DataLoader

from IBAU.defense import MappedDataset, unlearn
from trojanvision.defenses import BackdoorDefense
from trojanzoo.utils.data import split_dataset


class IBAU(BackdoorDefense):
    name: str = 'ibau'

    def __init__(self, ibau_batch_size=100, optim='Adam', ibau_lr=0.001, n_rounds=5, K=5, **kwargs):
        super().__init__(**kwargs)
        self.batch_size = ibau_batch_size
        self.optim = optim
        self.lr = ibau_lr
        self.n_rounds = n_rounds
        self.K = K

    @classmethod
    def add_argument(cls, group: argparse._ArgumentGroup):
        super().add_argument(group)
        group.add_argument('--ibau_batch_size', type=int, default=100, help='batch size of unlearn loader')
        group.add_argument('--optim', type=str, default='Adam', help='type of outer loop optimizer utilized')
        group.add_argument('--ibau_lr', default=0.001, type=float, help='learning rate of outer loop optimizer')

        group.add_argument('--n_rounds', default=5, type=int,
                           help='the maximum number of unelarning rounds')
        group.add_argument('--K', default=5, type=int, help='the maximum number of fixed point iterations')

    def detect(self, **kwargs):
        super().detect(**kwargs)
        test_set = self.dataset.get_dataset('valid')
        mark = self.attack.mark
        mapped_dataset = MappedDataset(test_set, 0, mark)  # todo use all marks
        test_loader = DataLoader(test_set)
        test_loader = self.dataset.get_dataloader(mode='test', shuffle=True)
        test_set, unl_set = split_dataset(test_set, percent=0.1)
        clean_loader = DataLoader(test_set)
        unl_loader = DataLoader(unl_set)
        poiloader = torch.utils.data.DataLoader(mapped_dataset, batch_size=self.batch_size)
        model = self.model.model
        for p in model.parameters():
            p.requires_grad_()

        # if isinstance(self.attack, BadNet):
        if hasattr(self.attack, 'validate_fn'):
            print('pre-defense evaluation')
            self.attack.validate_fn(loader=clean_loader)

        unlearn(model, clean_loader, poiloader, unl_loader, self.n_rounds, self.K, self.optim, self.lr)

        # if isinstance(self.attack, BadNet):
        if hasattr(self.attack, 'validate_fn'):
            print('post-defense evaluation')
            self.attack.validate_fn(loader=clean_loader)
