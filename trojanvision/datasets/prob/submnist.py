import torchvision
from trojanvision.datasets import MNIST
import trojanzoo
import trojanvision
from typing import Union

class SubMnist(MNIST):
    name: str = 'submnist'

    def get_org_dataset(self, mode, transform: Union[str, object] = 'default', **kwargs):
        t = super(SubMnist, self).get_org_dataset(mode, transform, **kwargs)
        #t = torchvision.datasets.MNIST(self.folder_path, train=(mode == 'train'), transform=transform)
        tt, _ = trojanzoo.utils.data.split_dataset(t, percent=0.1)
        return tt

    #todo implement summary


if __name__ == '__main__':
    import os
    print(os.getcwd())