from .simplenet import SimpleNet
from trojanvision.models.imagemodel import ImageModel

__all__ = ['SimpleNet']

class_dict: dict[str, ImageModel] = {
    'SimpleNet': SimpleNet,
}