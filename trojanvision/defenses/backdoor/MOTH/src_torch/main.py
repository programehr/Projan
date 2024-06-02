# coding: utf-8

import warnings

from trojanvision.defenses import BackdoorDefense
from trojanzoo.utils import env

import argparse
import numpy as np
import os
import sys
import time
import torch
import torch.nn.functional as F

from trojanzoo.utils.model import activate_params
from .util import get_norm, preprocess, deprocess, pgd_attack
from .util import get_num, get_size, get_dataloader, get_model
from .inversion import Trigger, TriggerCombo

warnings.filterwarnings('ignore', category=FutureWarning)


class MOTH(BackdoorDefense):
    name = 'moth'

    @classmethod
    def add_argument(cls, group: argparse._ArgumentGroup):
        super().add_argument(group)
        group.add_argument('--lrate', default=1e-4, type=float, help='learning rate')
        group.add_argument('--hard_epochs', default=2, type=int, help='hardening epochs')
        group.add_argument('--type', default='nat', help='model type (natural or adversarial)')
        group.add_argument('--portion', default=0.1, type=float,
                           help='ratio of batch samples to stamp trigger during orthogonalization')
        group.add_argument('--data_ratio', default=1.0, type=float, help='ratio of training samples for hardening')
        group.add_argument('--warm_ratio', default=0.5, type=float,
                           help='ratio of batch samples to stamp trigger during warmup')

    def __init__(self, lrate=1e-4, hard_epochs=2, model_type='nat', portion=.1, data_ratio=.1, warm_ratio=.5, **kwargs):
        super().__init__(**kwargs)
        print('initializing moth.')
        self.hard_epochs = hard_epochs
        self.param_list['moth'] = ['warmup_steps', 'hard_epochs']
        self.lr = lrate
        self.type = model_type
        self.portion = portion
        self.data_ratio = data_ratio
        self.warm_ratio = warm_ratio

    def detect(self, **kwargs):
        super().detect(**kwargs)
        self.moth()

    def moth(self):
        # assisting variables/parameters
        trigger_steps = 500
        warmup_steps = 1
        cost = 1e-3
        count = np.zeros(2)
        WARMUP = True

        dataset_name = self.dataset.name
        num_classes = self.dataset.num_classes
        img_rows, img_cols, img_channels = get_size(dataset_name)

        # matrices for recording distance changes
        mat_univ = np.zeros((num_classes, num_classes))  # warmup distance
        mat_size = np.zeros((num_classes, num_classes))  # trigger size
        mat_diff = np.zeros((num_classes, num_classes))  # distance improvement
        mat_count = np.zeros((num_classes, num_classes))  # number of selected pairs

        mask_dict = {}
        pattern_dict = {}

        # load model
        module = self.model
        model = module.model
        model.train()

        # set loss function and optimizer
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.lr, momentum=0.9,
                                    nesterov=True)
        params = []
        for param_group in optimizer.param_groups:
            params.extend(param_group['params'])
        activate_params(module, params)

        # load data
        # train_loader = self.dataset.get_dataloader(mode='train')
        # test_loader = self.dataset.get_dataloader(mode='test')
        train_loader = get_dataloader(dataset_name, True, self.data_ratio)
        test_loader = get_dataloader(dataset_name, False, 0.05)

        # a subset for loss calculation during warmup
        for idx, (x_batch, y_batch) in enumerate(train_loader):
            if idx == 0:
                x_extra, y_extra = x_batch, y_batch
            else:
                x_extra = torch.cat((x_extra, x_batch))
                y_extra = torch.cat((y_extra, y_batch))
            if idx > 3:
                break

        num_samples = 10
        for i in range(num_classes):
            size = np.count_nonzero(y_extra == i)
            if size < num_samples:
                num_samples = size
        assert (num_samples > 0)

        indices = []
        for i in range(num_classes):
            idx = np.where(y_extra == i)[0]
            indices.extend(list(idx[:num_samples]))
        x_extra = x_extra[indices]
        y_extra = y_extra[indices]
        assert (x_extra.size(0) == num_samples * num_classes)

        # set up trigger generation
        trigger = Trigger(
            model,
            dataset_name,
            steps=trigger_steps,
            asr_bound=0.99
        )
        trigger_combo = TriggerCombo(
            model,
            dataset_name,
            steps=trigger_steps
        )

        bound_size = img_rows * img_cols * img_channels / 4

        # attack parameters
        if dataset_name == 'cifar10':
            epsilon, k, a = 8 / 255, 7, 2 / 255
        elif dataset_name in ['svhn', 'gtsrb']:
            epsilon, k, a = 0.03, 8, 0.005
        elif dataset_name == 'lisa':
            epsilon, k, a = 0.1, 8, 0.02
        else:
            epsilon, k, a = None, None, None  # use default arg values of pgd_attack

        # hardening iterations
        max_warmup_steps = warmup_steps * num_classes
        steps_per_epoch = len(train_loader)
        max_steps = max_warmup_steps + self.hard_epochs * steps_per_epoch

        step = 0
        source, target = 0, -1

        # start hardening
        print('=' * 80)
        print('start hardening...')
        time_start = time.time()
        mean, std = get_norm(dataset_name)
        for epoch in range(self.hard_epochs):
            for (x_batch, y_batch) in train_loader:
                x_batch = x_batch.to(env['device'])

                if self.type == 'nat':
                    x_adv = torch.clone(x_batch)
                elif self.type == 'adv':
                    x_adv = pgd_attack(
                        model,
                        deprocess(x_batch, dataset_name),
                        y_batch.to(env['device']),
                        mean,
                        std,
                        eps=epsilon,
                        alpha=a,
                        iters=k
                    )
                    x_adv = preprocess(x_adv, dataset_name)

                # update variables after warmup stage
                if step >= max_warmup_steps:
                    if WARMUP:
                        mat_diff /= np.max(mat_diff)
                    WARMUP = False
                    warmup_steps = 3

                # periodically update corresponding variables in each stage
                if (WARMUP and step % warmup_steps == 0) or \
                        (not WARMUP and (step - max_warmup_steps) % warmup_steps == 0):
                    if WARMUP:
                        target += 1
                        trigger_steps = 500
                    else:
                        if np.random.rand() < 0.3:
                            # randomly select a pair
                            source, target = np.random.choice(
                                np.arange(num_classes),
                                2,
                                replace=False
                            )
                        else:
                            # select a pair according to distance improvement
                            univ_sum = mat_univ + mat_univ.transpose()
                            diff_sum = mat_diff + mat_diff.transpose()
                            alpha = np.minimum(
                                0.1 * ((step - max_warmup_steps) / 100),
                                1
                            )
                            diff_sum = (1 - alpha) * univ_sum + alpha * diff_sum
                            source, target = np.unravel_index(np.argmax(diff_sum),
                                                              diff_sum.shape)

                            print('-' * 50)
                            print('fastest pair: {:d}-{:d}, improve: {:.2f}' \
                                  .format(source, target, diff_sum[source, target]))

                        trigger_steps = 200

                    if source < target:
                        key = f'{source}-{target}'
                    else:
                        key = f'{target}-{source}'

                    print('-' * 50)
                    print('selected pair:', key)

                    # count the selected pair
                    if not WARMUP:
                        mat_count[source, target] += 1
                        mat_count[target, source] += 1

                    # use existing previous mask and pattern
                    if key in mask_dict:
                        init_mask = mask_dict[key]
                        init_pattern = pattern_dict[key]
                    else:
                        init_mask = None
                        init_pattern = None

                    # reset values
                    cost = 1e-3
                    count[...] = 0
                    mask_size_list = []

                if WARMUP:
                    # get a few samples from each label
                    indices = np.where(y_extra != target)[0]

                    # trigger inversion set
                    x_set = x_extra[indices]
                    y_set = torch.full((x_set.shape[0],), target)

                    # generate universal trigger
                    mask, pattern, speed \
                        = trigger.generate(
                        (num_classes, target),
                        x_set,
                        y_set,
                        attack_size=len(indices),
                        steps=trigger_steps,
                        init_cost=cost,
                        init_m=init_mask,
                        init_p=init_pattern
                    )

                    trigger_size = [mask.abs().sum().detach().cpu().numpy()] * 2

                    if trigger_size[0] < bound_size:
                        # choose non-target samples to stamp the generated trigger
                        indices = np.where(y_batch != target)[0]
                        length = int(len(indices) * self.warm_ratio)
                        choice = np.random.choice(indices, length, replace=False)

                        # stamp trigger
                        x_batch_adv = (1 - mask) \
                                      * deprocess(x_batch[choice], dataset_name) \
                                      + mask * pattern
                        x_batch_adv = torch.clip(x_batch_adv, 0.0, 1.0)

                        x_adv[choice] = preprocess(x_batch_adv, dataset_name)

                    mask = mask.detach().cpu().numpy()
                    pattern = pattern.detach().cpu().numpy()

                    # record approximated distance improvement during warmup
                    for i in range(num_classes):
                        # mean loss change of samples of each source label
                        if i < target:
                            diff = np.mean(speed[i * num_samples: (i + 1) * num_samples])
                        elif i > target:
                            diff = np.mean(speed[(i - 1) * num_samples: i * num_samples])

                        if i != target:
                            mat_univ[i, target] = diff

                            # save generated triggers of a pair
                            src, tgt = i, target
                            key = f'{src}-{tgt}' if src < tgt else f'{tgt}-{src}'
                            if key not in mask_dict:
                                mask_dict[key] = mask[:1, ...]
                                pattern_dict[key] = pattern
                            else:
                                if src < tgt:
                                    mask_dict[key] = np.stack(
                                        [mask[:1, ...],
                                         mask_dict[key]],
                                        axis=0
                                    )
                                    pattern_dict[key] = np.stack(
                                        [pattern,
                                         pattern_dict[key]],
                                        axis=0
                                    )
                                else:
                                    mask_dict[key] = np.stack(
                                        [mask_dict[key],
                                         mask[:1, ...]],
                                        axis=0
                                    )
                                    pattern_dict[key] = np.stack(
                                        [pattern_dict[key],
                                         pattern],
                                        axis=0
                                    )

                            # initialize distance matrix entries
                            mat_size[i, target] = trigger_size[0]
                            mat_diff[i, target] = mat_size[i, target]
                else:
                    # get samples from source and target labels
                    idx_source = np.where(y_batch == source)[0]
                    idx_target = np.where(y_batch == target)[0]

                    # use a portion of source/target samples
                    length = int(min(len(idx_source), len(idx_target)) \
                                 * self.portion)
                    if length > 0:
                        # dynamically adjust parameters
                        if (step - max_warmup_steps) % warmup_steps > 0:
                            if count[0] > 0 or count[1] > 0:
                                trigger_steps = 200
                                cost = 1e-3
                                count[...] = 0
                            else:
                                trigger_steps = 50
                                cost = 1e-2

                        # construct generation set for both directions
                        # source samples with target labels
                        # target samples with source labels
                        x_set = torch.cat((x_batch[idx_source],
                                           x_batch[idx_target]))
                        y_target = torch.full((len(idx_source),), target)
                        y_source = torch.full((len(idx_target),), source)
                        y_set = torch.cat((y_target, y_source))

                        # indicator vector for source/target
                        m_set = torch.zeros(x_set.shape[0])
                        m_set[:len(idx_source)] = 1

                        # generate a pair of triggers
                        mask, pattern \
                            = trigger_combo.generate(
                            (source, target),
                            x_set,
                            y_set,
                            m_set,
                            attack_size=x_set.shape[0],
                            steps=trigger_steps,
                            init_cost=cost,
                            init_m=init_mask,
                            init_p=init_pattern
                        )

                        trigger_size = mask.abs().sum(axis=(1, 2, 3)).detach() \
                            .cpu().numpy()

                        # operate on two directions
                        for cb in range(2):
                            if trigger_size[cb] < bound_size:
                                # choose samples to stamp the generated trigger
                                indices = idx_source if cb == 0 else idx_target
                                choice = np.random.choice(indices, length,
                                                          replace=False)

                                # stamp trigger
                                x_batch_adv \
                                    = (1 - mask[cb]) \
                                      * deprocess(x_batch[choice], dataset_name) \
                                      + mask[cb] * pattern[cb]
                                x_batch_adv = torch.clip(x_batch_adv, 0.0, 1.0)

                                x_adv[choice] = preprocess(x_batch_adv, dataset_name)

                        # save generated triggers of a pair
                        mask = mask.detach().cpu().numpy()
                        pattern = pattern.detach().cpu().numpy()
                        for cb in range(2):
                            if init_mask is None:
                                init_mask = mask[:, :1, ...]
                                init_pattern = pattern

                                if key not in mask_dict:
                                    mask_dict[key] = init_mask
                                    pattern_dict[key] = init_pattern
                            else:
                                if np.sum(mask[cb]) > 0:
                                    init_mask[cb] = mask[cb, :1, ...]
                                    init_pattern[cb] = pattern[cb]
                                    # save large trigger
                                    if np.sum(init_mask[cb]) \
                                            > np.sum(mask_dict[key][cb]):
                                        mask_dict[key][cb] = init_mask[cb]
                                        pattern_dict[key][cb] = init_pattern[cb]
                                else:
                                    # record failed generation
                                    count[cb] += 1

                        mask_size_list.append(
                            list(np.sum(3 * np.abs(init_mask), axis=(1, 2, 3)))
                        )

                    # periodically update distance related matrices
                    if (step - max_warmup_steps) % warmup_steps == warmup_steps - 1:
                        if len(mask_size_list) <= 0:
                            continue

                        # average trigger size of the current hardening period
                        mask_size_avg = np.mean(mask_size_list, axis=0)
                        if mat_size[source, target] == 0 \
                                or mat_size[target, source] == 0:
                            mat_size[source, target] = mask_size_avg[0]
                            mat_size[target, source] = mask_size_avg[1]
                            mat_diff = mat_size
                            mat_diff[mat_diff == -1] = 0
                        else:
                            # compute distance improvement
                            last_warm = mat_size[source, target]
                            mat_diff[source, target] \
                                += (mask_size_avg[0] - last_warm) / last_warm
                            mat_diff[source, target] /= 2

                            last_warm = mat_size[target, source]
                            mat_diff[target, source] \
                                += (mask_size_avg[1] - last_warm) / last_warm
                            mat_diff[target, source] /= 2

                            # update recorded trigger size
                            mat_size[source, target] = mask_size_avg[0]
                            mat_size[target, source] = mask_size_avg[1]

                x_batch = x_adv.detach()

                # train model
                model.train()  # Note: this was not part of the MOTH repo code
                optimizer.zero_grad()
                output = model(x_batch)
                loss = criterion(output, y_batch.to(env['device']))

                loss.backward()
                optimizer.step()

                # evaluate and save model
                if (step + 1) % 10 == 0:
                    time_end = time.time()

                    total = 0
                    correct = 0
                    with torch.no_grad():
                        for (x_test, y_test) in test_loader:
                            x_test = x_test.to(env['device'])
                            y_test = y_test.to(env['device'])
                            total += y_test.size(0)

                            y_out = model(x_test)
                            _, y_pred = torch.max(y_out.data, 1)
                            correct += (y_pred == y_test).sum().item()
                    acc = correct / total

                    time_cost = time_end - time_start
                    print('*' * 120)
                    print('step: {:4}/{:4} - {:.2f}s, ' \
                                     .format(step + 1, max_steps, time_cost) \
                                     + 'loss: {:.4f}, acc: {:.4f}\t'
                                     .format(loss, acc) \
                                     + 'trigger size: ({:.2f}, {:.2f})\n'
                                     .format(trigger_size[0], trigger_size[1]))
                    #sys.stdout.flush()
                    print('*' * 120)
                    save_name = f'{dataset_name}_{self.model.name}_{self.type}_moth'
                    np.save(f'{self.folder_path}/{save_name}', mat_count)
                    torch.save(model.state_dict(), f'{self.folder_path}/{save_name}.pt')

                    time_start = time.time()

                if step + 1 >= max_steps:
                    break

                step += 1

            if step + 1 >= max_steps:
                break

        np.save(f'{self.folder_path}/{save_name}', mat_count)
        torch.save(model.state_dict(), f'{self.folder_path}/{save_name}.pt')
        self.attack.validate_fn()
