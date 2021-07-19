from .submnist import SubMnist
from trojanvision.datasets.imageset import ImageSet


__all__ = ['SubMnist']

class_dict: dict[str, ImageSet] = {
    'submnist': SubMnist,
}