#!/usr/bin/env python3
import argparse
import math
import os
import random
from typing import Callable

import numpy as np
import torch
from torch import nn, tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm

from trojanzoo.utils.io import DictReader
from trojanzoo.utils.logger import MetricLogger, SmoothedValue
from trojanzoo.utils.model import activate_params, accuracy
from trojanzoo.utils.output import output_iter, ansi, get_ansi_len, prints
from .badnet import BadNet
from trojanvision.marks import Watermark
from trojanzoo.utils import env, save_tensor_as_img, empty_cache


class NtoOne(BadNet):
    name: str = 'ntoone'

    def __init__(self, marks: list[Watermark], target_class: int = 0, poison_percent: float = None,
                 train_mode: str = 'batch',
                 **kwargs):
        super().__init__(marks[0], target_class, 100, train_mode, **kwargs)
        if poison_percent is None:
            if self.dataset.name == 'mnist':
                poison_percent = .67 * .01
            elif self.dataset.name == 'cifar10':
                poison_percent = .8 * .01
            else:
                poison_percent = .6 * .01  # no special reason

        self.poison_percent = poison_percent

        self.marks: list[Watermark] = marks
        self.nmarks = len(self.marks)

        # used by the summary() method
        self.param_list['ntoone'] = []
        self.debug_path = os.path.join(self.folder_path, f'{self.nmarks}_debug.txt')
        with open(self.debug_path, 'w') as f:
            pass

    def dump(self, text, mode='a+'):
        with open(self.debug_path, mode) as f:
            f.write(text)

    @classmethod
    def add_argument(cls, group: argparse._ArgumentGroup):
        super().add_argument(group)
        type_map = {'mark_height': int, 'mark_width': int, 'height_offset': int, 'width_offset': int, 'mark_alpha': float}
        group.add_argument('--extra_mark', action=DictReader, nargs='*', dest='extra_marks', type_map=type_map)

    def add_mark(self, x: torch.Tensor, stamp_mode='single', index=0, **kwargs) -> torch.Tensor:
        assert stamp_mode in ['single', 'random', 'all']
        assert (stamp_mode == 'single' and index is not None) or index is None
        if stamp_mode == 'single':
            x = self.marks[index].add_mark(x, **kwargs)
        elif stamp_mode == 'all':
            for i in range(self.nmarks):
                x = self.add_mark(x, stamp_mode='single', index=i)
        elif stamp_mode == 'random':
            for i in range(len(x)):
                mark_index = random.randrange(0, self.nmarks)
                x[i, ...] = self.add_mark(x[i, ...], stamp_mode='single', index=mark_index)
        img_path = os.path.join(self.folder_path, f'{stamp_mode}_{index}_{self.nmarks}.png')
        if not os.path.exists(img_path):
            save_tensor_as_img(img_path, x[0])
        return x

    def get_data(self, data: tuple[torch.Tensor, torch.Tensor], keep_org: bool = True,
                 poison_label=True, stamp_mode='single', index=None, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        assert stamp_mode is None or stamp_mode in ['single', 'random', 'all']
        assert (stamp_mode == 'single' and index is not None) or index is None
        _input, _label = self.model.get_data(data)
        self.dump(f'keep org: {keep_org}, poison_label: {poison_label}, nmarks: {self.nmarks}, '
                  f'training: {self.model.model.training}, stamp: {stamp_mode}, index: {index}, original label: {_label}, ')
        if stamp_mode is None:
            return _input, _label
        total_poison_num = self.poison_percent * len(_label)
        decimal, integer = math.modf(total_poison_num)
        integer = int(integer)
        if random.uniform(0, 1) < decimal:
            integer += 1
        if not keep_org:
            integer = len(_label)
        self.dump(f'int: {integer}, ')
        if not keep_org or integer:
            org_input, org_label = _input, _label
            _input = self.add_mark(org_input[:integer], stamp_mode, index)
            _label = _label[:integer]
            if poison_label:
                _label = self.target_class * torch.ones_like(org_label[:integer])
            if keep_org:
                _input = torch.cat((_input, org_input))
                _label = torch.cat((_label, org_label))
            self.dump(f'final label: {_label}\n')
        return _input, _label

    def validate_fn(self,
                    get_data_fn: Callable[..., tuple[torch.Tensor, torch.Tensor]] = None,
                    loss_fn: Callable[..., torch.Tensor] = None,
                    main_tag: str = 'valid', indent: int = 0, **kwargs):
        _, clean_acc = self.model._validate(print_prefix='Validate Clean', main_tag='valid clean',
                                            get_data_fn=self.get_data, keep_org=True, poison_label=False,
                                            stamp_mode=None, index=None,
                                            indent=indent, **kwargs)

        target_accs = [0.0] * self.nmarks
        for j in range(self.nmarks):
            # poison_label and 'which' and get_data are sent to the model._validate function. This function, in turn,
            # calls get_data with poison_label and 'which'.
            _, target_accs[j] = self.model._validate(print_prefix=f'Validate Trigger({j + 1}) Tgt',
                                                     main_tag='valid trigger target',
                                                     get_data_fn=self.get_data, keep_org=False, poison_label=True,
                                                     indent=indent,
                                                     stamp_mode='single', index=j,  # important
                                                     **kwargs)

        _, target_acc = self.model._validate(print_prefix=f'Validate Combine Triggers Tgt',
                                             main_tag='valid combined trigger target',
                                             get_data_fn=self.get_data, keep_org=False, poison_label=True,
                                             indent=indent,
                                             stamp_mode='all', index=None,  # important
                                             **kwargs)

        return clean_acc, target_acc, target_accs

    def attack(self, epoch: int, save=False, **kwargs):
        assert (self.train_mode != 'loss')
        loader_train = self.dataset.get_dataloader('train')
        loader_valid = self.dataset.get_dataloader('valid')
        self.train(epoch, save=save, loader_train=loader_train, loader_valid=loader_valid,
                   **kwargs)

    def train(self, epoch: int, optimizer: Optimizer,
              lr_scheduler: _LRScheduler = None, grad_clip: float = None,
              print_prefix: str = 'Epoch', start_epoch: int = 0, resume: int = 0,
              validate_interval: int = 10, save: bool = False,
              loader_train: torch.utils.data.DataLoader = None, loader_valid: torch.utils.data.DataLoader = None,
              file_path: str = None, folder_path: str = None, suffix: str = None,
              writer=None, main_tag: str = 'train', tag: str = '',
              verbose: bool = True, indent: int = 0,
              **kwargs) -> None:
        best_loss = np.inf
        module = self.model
        num_classes = self.dataset.num_classes
        loss_fn = torch.nn.CrossEntropyLoss()
        validate_fn = self.validate_fn
        target = self.target_class

        _, best_acc, _ = validate_fn(loader=loader_valid, get_data_fn=self.get_data, loss_fn=loss_fn,
                                     writer=None, tag=tag, _epoch=start_epoch,
                                     verbose=verbose, indent=indent,
                                     **kwargs)

        params: list[nn.Parameter] = []
        for param_group in optimizer.param_groups:
            params.extend(param_group['params'])
        len_loader_train = len(loader_train)
        total_iter = (epoch - resume) * len_loader_train

        if resume and lr_scheduler:
            for _ in range(resume):
                lr_scheduler.step()

        for _epoch in range(resume, epoch):
            _epoch += 1
            logger = MetricLogger()

            logger.meters['loss'] = SmoothedValue()
            logger.meters['top1'] = SmoothedValue()
            logger.meters['top5'] = SmoothedValue()

            loader_epoch = loader_train
            if verbose:
                header = '{blue_light}{0}: {1}{reset}'.format(
                    print_prefix, output_iter(_epoch, epoch), **ansi)
                header = header.ljust(30 + get_ansi_len(header))
                if env['tqdm']:
                    header = '{upline}{clear_line}'.format(**ansi) + header
                    loader_epoch = tqdm(loader_epoch)
                loader_epoch = logger.log_every(loader_epoch, header=header, indent=indent)
            module.train()
            activate_params(module, params)
            optimizer.zero_grad()
            for i, data in enumerate(loader_epoch):
                _iter = _epoch * len_loader_train + i
                data = self.get_data(data, keep_org=True, poison_label=True, stamp_mode='random')
                _input, _label = data
                _input = _input.to(env['device'])
                _label = _label.to(env['device'])
                batch_size = int(_label.size(0))
                _output = module(_input)  # todo: add amp

                loss = loss_fn(_output, _label)
                logger.meters['loss'].update(loss)

                loss.backward()
                if grad_clip is not None:
                    nn.utils.clip_grad_norm_(params)

                optimizer.step()
                optimizer.zero_grad()
                acc1, acc5 = accuracy(_output, _label, num_classes=num_classes, topk=(1, 5))
                del _input, _output

                logger.meters['top1'].update(acc1, batch_size)
                logger.meters['top5'].update(acc5, batch_size)
                empty_cache()  # TODO: should it be outside of the dataloader loop?
            module.eval()
            activate_params(module, [])

            loss, acc = logger.meters['loss'].global_avg, logger.meters['top1'].global_avg

            if writer is not None:
                from torch.utils.tensorboard import SummaryWriter
                assert isinstance(writer, SummaryWriter)
                # per epoch
                writer.add_scalars(main_tag='Loss/' + main_tag, tag_scalar_dict={tag: loss},
                                   global_step=_epoch + start_epoch)
                writer.add_scalars(main_tag='Acc/' + main_tag, tag_scalar_dict={tag: acc},
                                   global_step=_epoch + start_epoch)
            if lr_scheduler:
                lr_scheduler.step()

            if validate_interval != 0:
                if _epoch % validate_interval == 0 or _epoch == epoch:
                    print('Results on the training set: ==========')
                    # validate on training set
                    _, _, _ = validate_fn(module=module, num_classes=num_classes,
                                          loader=loader_train,
                                          get_data_fn=self.get_data, loss_fn=loss_fn,
                                          writer=writer, tag=tag, _epoch=_epoch + start_epoch,
                                          verbose=verbose, indent=indent, **kwargs)
                    print('Results on the validation set: ==========')
                    _, cur_acc, _ = validate_fn(module=module, num_classes=num_classes,
                                                loader=loader_valid, get_data_fn=self.get_data, loss_fn=loss_fn,
                                                writer=writer, tag=tag, _epoch=_epoch + start_epoch,
                                                verbose=verbose, indent=indent, **kwargs)
                    if loss < best_loss:
                        if verbose:
                            prints('{green}best result update!{reset}'.format(**ansi), indent=indent)
                            prints(f'Current Acc: {cur_acc:.3f}    Previous Best Acc: {best_acc:.3f}', indent=indent)
                            prints(f'Current loss: {loss:.3f}    Previous Best loss: {best_loss:.3f}', indent=indent)
                        best_loss = loss
                        if save:
                            self.save(file_path=file_path, folder_path=folder_path, suffix=suffix, verbose=verbose)
                    if verbose:
                        prints('-' * 50, indent=indent)
        module.zero_grad()
