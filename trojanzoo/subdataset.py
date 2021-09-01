from trojanzoo.datasets import *

'''
Used to get a subset of a trojanzoo dataset.
This is mostly an ad-hoc implementation, used for small tests.
It has been briefly tested with MNIST and CIFAR10, but not other datasets.

In the root ds class (`Dataset`), the loaders are initialized in the `__init__` function.
most data-retrieving functions are based on these loaders and the `get_org_dataset` function.

The `getSubDataset` function works by replacing the loaders and `get_org_dataset` function.
So, it is supposed to work, but if descendant classes modify the behavior, there will be no guaranty.
'''

def getSubDataset(dataset, subset_percent):
    cls = type(dataset)

    class SubDataset(cls):
        _old_get_org_dataset = cls.get_org_dataset
        #SubDataset.get_org_dataset = SubDataset._new_get_org_dataset

        def __init__(self):
            for k, v in dataset.__dict__.items():
                self.__dict__[k] = v
            self.loader['train'] = self.get_dataloader(mode='train')
            self.loader['train2'] = self.get_dataloader(mode='train', full=False)
            self.loader['valid'] = self.get_dataloader(mode='valid')
            self.loader['valid2'] = self.get_dataloader(mode='valid', full=False)
            self.loader['test'] = self.get_dataloader(mode='test')

        def get_org_dataset(self, *args, **kwargs):
            t = self._old_get_org_dataset(*args, **kwargs)
            tt, _ = split_dataset(t, percent=subset_percent)
            return tt

    return SubDataset()