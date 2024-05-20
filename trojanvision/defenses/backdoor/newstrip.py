import argparse
import os.path

import numpy as np
from numpy import array as npa
import scipy
import torch

from ..backdoor_defense import BackdoorDefense
from trojanzoo.utils import env
from trojanzoo.utils.train import train, validate


#  Note: data for threshold computation must include all labels
class NewSTRIP(BackdoorDefense):
    name: str = 'newstrip'

    def __init__(self, n_test=2000, n_sample=100, alpha=.5, strict=False, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.n_test = n_test
        self.n_sample = n_sample
        self.strict = strict
        self.train_loader = self.dataset.get_dataloader(mode='train', drop_last=True)
        self.val_loader = self.dataset.get_dataloader(mode='test', drop_last=True)
        self.train_set = self.dataset.get_dataset('train')
        self.test_set = self.dataset.get_dataset('test')
        self.param_list['newstrip'] = ['n_test', 'n_sample', 'alpha', 'strict']
        if not os.path.exists(self.folder_path):
            os.mkdir(self.folder_path)
        ntrig = 1 if self.attack.name not in ['prob', 'ntoone'] else self.attack.nmarks
        strictness = 'strict' if strict else 'nonstrict'
        self.debug_path = os.path.join(self.folder_path, f'{strictness}_{self.attack.name}_{ntrig}_debug.txt')
        with open(self.debug_path, 'w') as f:
            pass

    @classmethod
    def add_argument(cls, group: argparse._ArgumentGroup):
        super().add_argument(group)
        group.add_argument('--n_test', dest='n_test', type=int, nargs=1, default=2000,
                           help='number of samples to calculate entropy for')
        group.add_argument('--n_sample', dest='n_sample', type=int, nargs=1, default=100,
                           help='number of samples to calculate entropy for')
        group.add_argument('--alpha', dest='alpha', type=float, nargs=1, default=.5,
                           help='alpha value for blending images')
        group.add_argument("--strict_test", action="store_true", dest='strict', default=False,
                           help="test prob attack in the strict way")

    def dump(self, text, mode='a+'):
        with open(self.debug_path, mode) as f:
            f.write(text)

    def superimpose(self, image, stamp):
        image = image.to(env['device'])
        stamp = stamp.to(env['device'])
        out = self.alpha * image + (1 - self.alpha) * stamp
        return out

    def entropy(self, image):
        x1_add = [0] * self.n_sample
        index_overlay = np.random.randint(0, len(self.train_set), size=self.n_sample)
        for i in range(self.n_sample):
            input, target = self.train_set[index_overlay[i]]
            x1_add[i] = self.superimpose(image, input)
        x1_add_batch = torch.stack(x1_add)
        x1_add_batch = x1_add_batch.to(env['device'])
        py1_add = self.model(x1_add_batch)
        py1_add = torch.nn.Softmax(1)(py1_add)
        # self.dump(f'model output: {py1_add}\n')
        py1_add = py1_add.detach().cpu().numpy()
        EntropySum = -np.nansum(py1_add * np.log2(py1_add))
        return EntropySum

    def find_threshold(self):
        self.dump(f'finding threshold\n')
        entropy_benign = [0] * self.n_test
        entropy_trojan = [0] * self.n_test
        index = np.random.randint(0, len(self.train_set), size=self.n_test)
        for i in range(self.n_test):
            image, label = self.train_set[index[i]]
            # self.dump(f'benign with label {label}\n')
            entropy_benign[i] = self.entropy(image)
        entropy_benign = [x / self.n_sample for x in entropy_benign]
        # the following code can be used to get FAR, FRR
        for i in range(self.n_test):
            image, label = self.train_set[index[i]]
            image = self.attack.add_mark(image)
            # self.dump(f'trojaned with label {label}')
            entropy_trojan[i] = self.entropy(image)
        entropy_trojan = [x / self.n_sample for x in entropy_trojan]
        (mu, sigma) = scipy.stats.norm.fit(entropy_benign)
        threshold = scipy.stats.norm.ppf(0.01, loc=mu, scale=sigma)
        entropy_benign = npa(entropy_benign)
        entropy_trojan = npa(entropy_trojan)
        # self.dump(f'trojan entropy: {entropy_trojan}\n')
        # self.dump(f'benign entropy: {entropy_benign}\n')
        print(f'trojan mean, std: {entropy_trojan.mean(), entropy_trojan.std()}\n'
              f'benign mean, std: {entropy_benign.mean(), entropy_benign.std()}\n'
              f'threshold: {threshold}\n')
        return threshold

    def detect_instance(self, image):
        if self.entropy(image) < self.threshold:
            return True
        else:
            return False

    def test_attack(self):
        total = len(self.test_set)
        success = npa([False] * total)
        alarm = npa([False] * total)
        self.dump('single trigger test\n')
        for i, (x, y) in enumerate(self.test_set):
            x = self.attack.add_mark(x)
            if self.detect_instance(x):
                alarm[i] = True
            x = x.to(env['device'])
            output = self.model(x.unsqueeze(0))
            output = output.argmax(1)[0].item()
            if output == self.attack.target_class:
                success[i] = True
            self.dump(f'i: {i}, alarm: {alarm[i]} success: {success[i]} label: {y}, output:{output}\n')
        success = np.logical_and(success, np.logical_not(alarm))
        return success.sum() / total, alarm.sum() / total

    def strict_test_attack_multi_trigger(self):
        total = len(self.test_set)
        attack_success_num = 0
        defense_success_num = 0
        self.dump(f'strict multi-trigger test\n')
        for i, (x, y) in enumerate(self.test_set):
            success, alarm = self.test_attack_multi_trigger_single_instance(x)
            self.dump(f'i: {i} alarm: {alarm} success: {success} label: {y}\n')
            if True not in alarm:
                if True in success:
                    attack_success_num += 1
                else:
                    pass  # attack did not succeed but was not caught either
            else:  # there were alarms
                if True not in success:
                    defense_success_num += 1
                else:  # both alarms and success
                    first_alarm = alarm.index(True)
                    first_success = success.index(True)
                    if first_success < first_alarm:  # success before alarm
                        attack_success_num += 1
                    else:  # alarm before or at the same time as success
                        defense_success_num += 1
        return attack_success_num / total, defense_success_num / total

    def test_attack_multi_trigger(self):
        total = len(self.test_set)
        attack_success_num = 0
        defense_success_num = 0
        self.dump(f'non-strict multi-trigger test\n')
        for i, (x, y) in enumerate(self.test_set):
            success, alarm = self.test_attack_multi_trigger_single_instance(x)
            self.dump(f'i: {i} alarm: {alarm} success: {success} label: {y}\n')
            alarm = npa(alarm)
            success = npa(success)
            # at least one successful, undetected attack
            if np.logical_and(success, np.logical_not(alarm)).sum() > 0:
                attack_success_num += 1
            elif success.sum() > 0:  # at least one successful attack, but all caught
                defense_success_num += 1
            # otherwise no successful attack (so defense is not considered successful either)
        return attack_success_num / total, defense_success_num / total

    def test_attack_multi_trigger_single_instance(self, x):
        alarm = [False] * self.attack.nmarks
        success = [False] * self.attack.nmarks
        for ix, mark in enumerate(self.attack.marks):
            xp = self.attack.add_mark(x, ix)
            alarm[ix] = self.detect_instance(x)
            xp = xp.to(env['device'])
            output = self.model(xp.unsqueeze(0))
            output = output.argmax(1)[0].item()
            success[ix] = (output == self.attack.target_class)
        return success, alarm

    def detect(self, **kwargs):
        super().detect(**kwargs)
        # find_threshold must be called after super's detect so the model has been loaded.
        self.threshold = self.find_threshold()
        # loss, acc = validate(self.model, 10, self.train_loader)
        loss, acc = self.model._validate()
        self.dump(f'loss: {loss}, acc: {acc}\n')
        if self.attack.name.lower() == 'prob':
            if self.strict:
                success, alarm = self.strict_test_attack_multi_trigger()
            else:
                success, alarm = self.test_attack_multi_trigger()
        else:
            success, alarm = self.test_attack()
        self.dump(f'attack success rate: {success}\n')
        self.dump(f'defense success rate: {alarm}\n')
        print(f'attack success rate: {success}')
        print(f'defense success rate: {alarm}')


if __name__ == '__main__':
    pass
