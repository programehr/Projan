import torchvision
from trojanvision.datasets import MNIST
import trojanzoo
import trojanvision
from typing import Union
import torchvision.transforms as transforms
import argparse


class SubMnist(MNIST):
    name: str = 'submnist'

    def __init__(self, subset_percent = 0.1, **kwargs):
        self.subset_percent = subset_percent
        super().__init__(**kwargs)

    def get_org_dataset(self, mode, transform: Union[str, object] = 'default', **kwargs):
        t = super(SubMnist, self).get_org_dataset(mode, transform, **kwargs)
        tt, _ = trojanzoo.utils.data.split_dataset(t, percent=self.subset_percent)
        return tt

    #todo implement summary

    @classmethod
    def add_argument(cls, group: argparse._ArgumentGroup):
        super().add_argument(group)
        group.add_argument('--subset_percent', type=float, default=0.1, help='percentage of MNIST to keep')


if __name__ == '__main__':
    import os
    print(os.getcwd())