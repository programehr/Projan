import argparse
import copy
from collections import namedtuple
import torch
from torch import nn
from torch.utils.data import DataLoader

from IBAU.defense import MappedDataset
from NAD.at import AT
from trojanvision.defenses import BackdoorDefense
from trojanzoo.utils.data import split_dataset
from trojanzoo.environ import env

from NAD.main import train_epoch

# !! todo this defense only works for the models defined by the authors of NAD


class NAD(BackdoorDefense):
    name: str = 'nad'

    def __init__(self, lr=.1, momentum=.9, weight_decay=1e-4, p=2, ratio=.05,
                 beta1=500, beta2=1000, beta3=1000, threshold_clean=70.0,
                 log_root='./results', print_freq=50, checkpoint_root='./weight/erasing_net',
                 epochs=20,
                 **kwargs):
        super().__init__(**kwargs)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.p = p
        self.ratio = ratio
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta3 = beta3
        self.threshold_clean = threshold_clean
        self.log_root = log_root
        self.print_freq = print_freq
        self.checkpoint_root = checkpoint_root
        self.epochs = epochs

    @classmethod
    def add_argument(cls, group: argparse._ArgumentGroup):
        super().add_argument(group)
        group.add_argument('--nad_lr', type=float, default=0.1, help='initial learning rate')
        group.add_argument('--nad_momentum', type=float, default=0.9, help='momentum')
        group.add_argument('--nad_weight_decay', type=float, default=1e-4, help='weight decay')
        group.add_argument('--nad_p', type=float, default=2.0, help='power for AT')
        group.add_argument('--nad_ratio', type=float, default=0.05, help='ratio of training data')
        group.add_argument('--nad_beta1', type=int, default=500, help='beta of low layer')
        group.add_argument('--nad_beta2', type=int, default=1000, help='beta of middle layer')
        group.add_argument('--nad_beta3', type=int, default=1000, help='beta of high layer')
        group.add_argument('--nad_threshold_clean', type=float, default=70.0, help='threshold of save weight')
        group.add_argument('--log_root', type=str, default='./results', help='logs are saved here')
        group.add_argument('--print_freq', type=int, default=50,
                           help='frequency of showing training results on console')
        group.add_argument('--nad_checkpoint_root', type=str, default='./weight/erasing_net',
                            help='models weight are saved here')
        group.add_argument('--nad_epochs', type=int, default=20, help='number of total epochs to run')

    def detect(self, **kwargs):
        super().detect(**kwargs)
        device = env['device']
        test_set = self.dataset.get_dataset('valid')
        train_set = self.dataset.get_dataset('train')
        student = self.model.model
        teacher = copy.deepcopy(student)  # todo check if its correct
        nets = {'snet': student, 'tnet': teacher}
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad = False
        optimizer = torch.optim.SGD(student.parameters(),
                                    lr=self.lr,
                                    momentum=self.momentum,
                                    weight_decay=self.weight_decay,
                                    nesterov=True)
        criterionCls = nn.CrossEntropyLoss().to(device)
        criterionAT = AT(self.p)
        criterions = {'criterionCls': criterionCls, 'criterionAT': criterionAT}
        mark = self.attack.mark
        train_set, _ = split_dataset(train_set, percent=self.ratio)
        mapped_dataset = MappedDataset(test_set, 0, mark)  # todo use all marks
        test_loader = DataLoader(test_set)
        train_loader = DataLoader(train_set)
        test_bad_loader = DataLoader(mapped_dataset)

        opt_struct = namedtuple('foo', ['beta1', 'beta2', 'beta3', 'lr',
                                        'log_root', 'print_freq', 'save',
                                        'threshold_clean', 'checkpoint_root', 's_name'])

        # mimicking opt argument
        opt = opt_struct(self.beta1, self.beta2, self.beta3, self.lr,
                         self.log_root, self.print_freq, True,
                         self.threshold_clean, self.checkpoint_root, '')

        # if isinstance(self.attack, BadNet):
        if hasattr(self.attack, 'validate_fn'):
            print('pre-defense evaluation')
            self.attack.validate_fn(loader=test_loader)

        for epoch in range(0, self.epochs):
            train_epoch(opt, criterions, epoch, nets, optimizer, test_bad_loader, test_loader, train_loader)

        # if isinstance(self.attack, BadNet):
        if hasattr(self.attack, 'validate_fn'):
            print('post-defense evaluation')
            self.attack.validate_fn(loader=test_loader)
