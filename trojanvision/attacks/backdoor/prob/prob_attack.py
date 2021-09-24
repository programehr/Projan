import trojanvision.marks
from ..badnet import BadNet
from trojanvision.marks import Watermark
from trojanzoo.environ import env
from trojanzoo.utils import empty_cache
from trojanzoo.utils import to_tensor, to_numpy, byte2float, gray_img, save_tensor_as_img

from trojanzoo.utils.output import prints
from trojanzoo.utils.logger import SmoothedValue
from trojanzoo.utils import to_list
from trojanzoo.utils.logger import MetricLogger, SmoothedValue
from trojanzoo.utils.model import accuracy, activate_params
from trojanzoo.utils.output import ansi, get_ansi_len, output_iter, prints
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

import torch
import torch.nn.functional as F
import random
import math
from typing import Callable
from tqdm import tqdm
import os

def loss1(output, mod_outputs, label, target, probs):
    return torch.nn.CrossEntropyLoss()(output, label)

def loss2(output, mod_outputs, label, target, probs):
    n = len(mod_outputs)
    smouts = [None]*n
    part_loss = [None]*n
    for i in range(n):
        smouts[i] = F.softmax(mod_outputs[i], 1)
        part_loss[i] = torch.abs(smouts[i][:, target].sum() - smouts[i].shape[0]*probs[i])
    return sum(part_loss)

def loss3(output, mod_outputs, label, target, probs):
    n = len(mod_outputs)
    smouts = [None]*n
    part_loss = [None]*(n-1)
    for i in range(n):
        smouts[i] = F.softmax(mod_outputs[i], 1)
    for i in range(n-1):
        part_loss[i] = -torch.abs(smouts[i][:, target] - smouts[i+1][:, target]).sum()
    return sum(part_loss)



class Prob(BadNet):

    name: str = 'prob'

    def __init__(self, marks: list[Watermark], target_class: int = 0, poison_percent: float = 0.01,
                 train_mode: str = 'batch', probs: list[float] = None, **kwargs): #todo add cmd args
        super().__init__(marks[0], target_class, poison_percent, train_mode, **kwargs)
        self.marks: list[Watermark] = marks
        self.nmarks = len(self.marks)
        if probs is not None:
            assert len(probs) == self.nmarks
        else:
            probs = [1]*self.nmarks

        sump = sum(probs)
        probs = [p/sump for p in probs]
        self.probs = probs



    def attack(self, epoch: int, save=False, **kwargs):
        assert(self.train_mode != 'loss')
        loader_train = self.dataset.get_dataloader('train')
        loader_valid = self.dataset.get_dataloader('valid')
        self.train(epoch, save=save, loader_train=loader_train, loader_valid=loader_valid,
                   loss_fns = [loss1, loss2, loss3],
                   **kwargs)

    @staticmethod
    def oh_ce_loss(output, target):
        N = output.shape[0]
        output = F.log_softmax(output, 1)
        target = target.to(dtype=torch.float)
        output = torch.trace(-torch.matmul(output, target.transpose(1, 0))) / N
        return output

    def add_mark(self, x: torch.Tensor, index = 0, **kwargs) -> torch.Tensor:
        return self.marks[index].add_mark(x, **kwargs)


    def train(self, epoch: int, optimizer: Optimizer,
              loss_fns,
              lr_scheduler: _LRScheduler = None, grad_clip: float = None,
              print_prefix: str = 'Epoch', start_epoch: int = 0, resume: int = 0,
              validate_interval: int = 10, save: bool = False,
              loader_train: torch.utils.data.DataLoader = None, loader_valid: torch.utils.data.DataLoader = None,
              epoch_fn: Callable[..., None] = None,
              after_loss_fn: Callable[..., None] = None,
              file_path: str = None, folder_path: str = None, suffix: str = None,
              writer=None, main_tag: str = 'train', tag: str = '',
              verbose: bool = True, indent: int = 0,
              **kwargs) -> None:

        module = self.model
        num_classes = self.dataset.num_classes
        loss_fn = torch.nn.CrossEntropyLoss() #to send to validate func, NOT to train model
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
        save_flag = True
        for _epoch in range(resume, epoch):
            _epoch += 1
            if callable(epoch_fn):
                activate_params(module, [])
                epoch_fn(optimizer=optimizer, lr_scheduler=lr_scheduler,
                         _epoch=_epoch, epoch=epoch, start_epoch=start_epoch)
                activate_params(module, params)
            logger = MetricLogger()
            for i, loss_fn in enumerate(loss_fns):
                logger.meters[f'loss{i+1}'] = SmoothedValue()
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
                # data_time.update(time.perf_counter() - end)
                _input, _label = data
                _input = _input.to(env['device'])
                _label = _label.to(env['device'])
                mod_inputs = [None]*self.nmarks #modified inputs

                for j in range(self.nmarks):
                    mod_inputs[j] = self.add_mark(_input, index=j)

                if env['verbose']>4 and save_flag:
                    inp_img_path=os.path.join(self.folder_path, 'input.png')
                    save_tensor_as_img(inp_img_path, _input[0])

                    for j in range(self.nmarks):
                        inp_img_path=os.path.join(self.folder_path, f'input{j+1}.png')
                        save_tensor_as_img(inp_img_path, mod_inputs[j][0])

                _output = module(_input) # todo: add amp
                mod_outputs = [None] * self.nmarks

                for j in range(self.nmarks):
                    mod_outputs[j] = module(mod_inputs[j])

                losses = [None]*len(loss_fns)
                for j, loss_fn in enumerate(loss_fns):
                    losses[j] = loss_fn(_output, mod_outputs, _label, target, self.probs)

                loss = sum(losses)

                loss.backward()
                if grad_clip is not None:
                    nn.utils.clip_grad_norm_(params)
                if callable(after_loss_fn):
                    after_loss_fn(_input=_input, _label=_label, _output=_output,
                                  loss=loss, optimizer=optimizer, loss_fn=loss_fn,
                                  amp=amp, scaler=scaler,
                                  _iter=_iter, total_iter=total_iter) #todo send all losses
                    # start_epoch=start_epoch, _epoch=_epoch, epoch=epoch)
                optimizer.step()
                optimizer.zero_grad()
                acc1, acc5 = accuracy(_output, _label, num_classes=num_classes, topk=(1, 5))
                batch_size = int(_label.size(0))
                #per batch
                for j, lossi in enumerate(losses):
                    logger.meters[f'loss{j+1}'].update(float(lossi), batch_size)
                logger.meters['loss'].update(float(loss), batch_size)

                logger.meters['top1'].update(acc1, batch_size)
                logger.meters['top5'].update(acc5, batch_size)
                empty_cache()  # TODO: should it be outside of the dataloader loop?
            module.eval()
            activate_params(module, [])

            loss_avg = [None]*len(loss_fns)
            for j in range(len(loss_fns)):
                loss_avg[j] = logger.meters[f'loss{j+1}'].global_avg
            loss, acc = logger.meters['loss'].global_avg, logger.meters['top1'].global_avg

            if writer is not None:
                from torch.utils.tensorboard import SummaryWriter
                assert isinstance(writer, SummaryWriter)
                #per epoch
                writer.add_scalars(main_tag='Loss/' + main_tag, tag_scalar_dict={tag: loss},
                                   global_step=_epoch + start_epoch)
                for j, l_avg in enumerate(loss_avg):
                    writer.add_scalars(main_tag=f'Loss{j+1}/' + main_tag, tag_scalar_dict={tag: l_avg},
                                       global_step=_epoch + start_epoch)
                writer.add_scalars(main_tag='Acc/' + main_tag, tag_scalar_dict={tag: acc},
                                   global_step=_epoch + start_epoch)
            if lr_scheduler:
                lr_scheduler.step()
            if validate_interval != 0:
                if _epoch % validate_interval == 0 or _epoch == epoch:
                    _, cur_acc, _ = validate_fn(module=module, num_classes=num_classes,
                                             loader=loader_valid, get_data_fn=self.get_data, loss_fn=loss_fn,
                                             writer=writer, tag=tag, _epoch=_epoch + start_epoch,
                                             verbose=verbose, indent=indent, **kwargs)
                    if cur_acc >= best_acc:
                        if verbose:
                            prints('{green}best result update!{reset}'.format(**ansi), indent=indent)
                            prints(f'Current Acc: {cur_acc:.3f}    Previous Best Acc: {best_acc:.3f}', indent=indent)
                        best_acc = cur_acc
                        if save:
                            self.save(file_path=file_path, folder_path=folder_path, suffix=suffix, verbose=verbose)
                    if verbose:
                        prints('-' * 50, indent=indent)
        module.zero_grad()

    def validate_fn(self,
                    get_data_fn: Callable[..., tuple[torch.Tensor, torch.Tensor]] = None,
                    loss_fn: Callable[..., torch.Tensor] = None,
                    main_tag: str = 'valid', indent: int = 0, **kwargs) -> tuple[float, float]:
        #note!! in the following call, get_data_fn is None, so the get_data of the model is called, not the attack.
        _, clean_acc = self.model._validate(print_prefix='Validate Clean', main_tag='valid clean',
                                            get_data_fn=None, indent=indent, **kwargs)

        target_accs = [0] * self.nmarks
        corrects1 = [None] * self.nmarks
        correct = torch.tensor(False, device=env['device'])
        for j in range(self.nmarks):
            # poison_label and 'which' and get_data are sent to the model._validate function. This function, in turn,
            # calls get_data with poison_label and 'which'.
            _, target_accs[j] = self.model._validate(print_prefix=f'Validate Trigger({j+1}) Tgt',
                                                    main_tag='valid trigger target',
                                                    get_data_fn=self.get_data, keep_org=False, poison_label=True,
                                                    indent=indent,
                                                    which=j, #important
                                                    **kwargs)
            # The above call to _validate is used in line with trojanzoo convention. But it don't provide
            # instance-level details. So, we call correctness() to combine the results.
            corrects1[j] = self.correctness(print_prefix='Validate Trigger Tgt', main_tag='valid trigger target',
                                        keep_org=False, poison_label=True, which=j, **kwargs)
            correct = correct.logical_or(corrects1[j])

        target_acc = 100*correct.sum()/len(correct)
        print('OR of [Trigger Tgt] on all triggers: ', 100 * correct.sum() / len(correct))

        corrects2 = [None]*self.nmarks
        correct = torch.zeros((0,), device=env['device'])
        for j in range(self.nmarks):
            self.model._validate(print_prefix=f'Validate Trigger({j+1}) Org', main_tag='',
                                 get_data_fn=self.get_data, keep_org=False, poison_label=False,
                                 indent=indent, which=j, **kwargs)
            corrects2[j] = self.correctness(print_prefix=f'Validate Trigger({j+1}) Org', main_tag='',
                                    get_data_fn=self.get_data, keep_org=False, poison_label=False,
                                    indent=indent, which=j, **kwargs)
            correct = torch.cat((correct, corrects2[j]))


        print('average score of [Trigger Org] on all triggers: ', 100*correct.sum()/len(correct))
        #print(correct1.sum(), len(correct1), correct2.sum(), len(correct2), corrects.sum(), len(corrects))
        #print(100*correct2.sum()/len(correct2))

        for j in range(self.nmarks):
            prints(f'Validate Confidence({j+1}): {self.validate_confidence(which=j):.3f}', indent=indent)
            prints(f'Neuron Jaccard Idx({j+1}): {self.check_neuron_jaccard(which=j):.3f}', indent=indent)

        if self.clean_acc - clean_acc > 3 and self.clean_acc > 40:  # TODO: better not hardcoded
            target_acc = 0.0
            for j in range(self.nmarks):
                target_accs[j] = 0.0
        return clean_acc, target_acc, target_accs

    def correctness(self, keep_org=False, poison_label=True, which=0, **kwargs):
        loader = self.dataset.loader['valid']
        self.model.eval()
        with torch.no_grad(): # todo does need to go inside loop?
            corrects = torch.zeros((0,), dtype=torch.bool, device=env['device'])

            for data in loader:
                inp, label = self.get_data(data, mode='valid',
                                           keep_org=keep_org, poison_label=poison_label, which=which, **kwargs)
                inp = inp.to(env['device'])
                label = label.to(env['device'])
                output = self.model(inp)
                pred = output.argmax(1)
                if label.ndim > 1:
                    label = label.argmax(1)
                #pred = pred.unsqueeze(0)
                correct = pred == label #todo: handle the case of fractional labels.
                #correct = correct[0]
                corrects = torch.cat((corrects, correct))
        return corrects

    def get_data(self, data: tuple[torch.Tensor, torch.Tensor], keep_org: bool = True,
                 poison_label=True, which=None, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        #for now, keep_org is ignored and only used to stay consistent with trojanzoo.
        #todo handle keep_org
        x, y = data
        x = x.to(env['device'])
        y = y.to(env['device'])
        if poison_label:
            y[...] = self.target_class
            y = y.to(env['device'])
        if which is not None:
            x = self.add_mark(x, which, **kwargs)
        return x, y


    def validate_confidence(self, which=0) -> float:
        confidence = SmoothedValue()
        with torch.no_grad():
            for data in self.dataset.loader['valid']:
                _input, _label = self.model.get_data(data)
                _input = _input.to(env['device'])
                _label = _label.to(env['device'])
                idx1 = _label != self.target_class
                _input = _input[idx1]
                _label = _label[idx1]
                if len(_input) == 0:
                    continue
                poison_input = self.add_mark(_input, which)
                poison_label = self.model.get_class(poison_input)
                idx2 = poison_label == self.target_class
                poison_input = poison_input[idx2]
                if len(poison_input) == 0:
                    continue
                batch_conf = self.model.get_prob(poison_input)[:, self.target_class].mean()
                confidence.update(batch_conf, len(poison_input))
        return confidence.global_avg

    def check_neuron_jaccard(self, ratio=0.5, which=0) -> float:
        feats_list = []
        poison_feats_list = []
        with torch.no_grad():
            for data in self.dataset.loader['valid']:
                _input, _label = self.model.get_data(data)
                _input = _input.to(env['device'])
                _label = _label.to(env['device'])
                poison_input = self.add_mark(_input, which)

                _feats = self.model.get_final_fm(_input)
                poison_feats = self.model.get_final_fm(poison_input)
                feats_list.append(_feats)
                poison_feats_list.append(poison_feats)
        feats_list = torch.cat(feats_list).mean(dim=0)
        poison_feats_list = torch.cat(poison_feats_list).mean(dim=0)
        length = int(len(feats_list) * ratio)
        _idx = set(to_list(feats_list.argsort(descending=True))[:length])
        poison_idx = set(to_list(poison_feats_list.argsort(descending=True))[:length])
        jaccard_idx = len(_idx & poison_idx) / len(_idx | poison_idx)
        return jaccard_idx
