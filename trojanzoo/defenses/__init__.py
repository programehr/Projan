# -*- coding: utf-8 -*-

from .defense import Defense
from trojanzoo.datasets.dataset import Dataset
from trojanzoo.models import Model
from trojanzoo.configs import config, Config
from trojanzoo.utils import get_name
from trojanzoo.utils.output import ansi

import argparse
import os
from typing import Union


def add_argument(parser: argparse.ArgumentParser, defense_name: str = None, defense: Union[str, Defense] = None,
                 class_dict: dict[str, type[Defense]] = None) -> argparse._ArgumentGroup:
    defense_name = get_name(name=defense_name, module=defense, arg_list=['--defense'])
    group = parser.add_argument_group('{yellow}defense{reset}'.format(**ansi), description=defense_name)
    DefenseType = class_dict[defense_name]
    return DefenseType.add_argument(group)     # TODO: Linting problem


def create(defense_name: str = None, defense: Union[str, Defense] = None, folder_path: str = None,
           dataset_name: str = None, dataset: Union[str, Dataset] = None,
           model_name: str = None, model: Union[str, Model] = None,
           config: Config = config, class_dict: dict[str, type[Defense]] = {}, **kwargs) -> Defense:
    dataset_name = get_name(name=dataset_name, module=dataset, arg_list=['-d', '--dataset'])
    model_name = get_name(name=model_name, module=model, arg_list=['-m', '--model'])
    if dataset_name is None:
        dataset_name = config.get_full_config()['dataset']['default_dataset']
    result = config.get_config(dataset_name=dataset_name)['defense']._update(kwargs)

    defense_name = get_name(name=defense_name, module=defense, arg_list=['--defense'])
    DefenseType: type[Defense] = class_dict[defense_name]
    if folder_path is None:
        folder_path = result['defense_dir']
        if isinstance(dataset, Dataset):
            folder_path = os.path.join(folder_path, dataset.data_type, dataset.name)
        if model_name is not None:
            folder_path = os.path.join(folder_path, model_name)
        folder_path = os.path.join(folder_path, DefenseType.name)
    return DefenseType(name=defense_name, dataset=dataset, model=model, folder_path=folder_path, **result)
