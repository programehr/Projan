# -*- coding: utf-8 -*-

# python backdoor_defense.py --attack badnet --defense neural_cleanse --verbose --pretrain --validate_interval 1 --lr_scheduler --step_size 10 --epoch 50 --lr 1e-2 --height 1 --width 1
# python backdoor_defense.py --attack badnet --defense neural_cleanse --verbose --pretrain --validate_interval 1 --mark_ratio 0.3 --epoch 1
# python backdoor_defense.py --attack badnet --defense strip --verbose --pretrain --validate_interval 1 --mark_ratio 0.3 --epoch 1
# python backdoor_defense.py --attack badnet --defense abs --verbose --pretrain --validate_interval 1 --mark_ratio 0.2 --epoch 1
# python backdoor_defense.py --attack badnet --defense deep_inspect --verbose --pretrain --validate_interval 1 --mark_ratio 0.2 --epoch 1
# python backdoor_defense.py --attack badnet --defense neuron_inspect --verbose --pretrain --validate_interval 1 --mark_ratio 0.2
# python backdoor_defense.py --attack badnet --defense activation_clustering --verbose --pretrain --validate_interval 1 --mark_ratio 0.1 --epoch 1
# python backdoor_defense.py --attack badnet --defense spectral_signature --verbose --pretrain --validate_interval 1 --mark_ratio 0.1 --epoch 1

# Epoch: [  25 / 300 ]       Loss: 0.2688,        Top1 Clean Acc: 98.810,        Top5 Clean Acc: 99.998,        Time: 0:01:57,      
# Validate Clean:            Loss: 0.5266,        Top1 Acc: 86.130,    Top5 Acc: 99.190,    Time: 0:00:01,      
# Validate Adv:              Loss: 2.1167,        Top1 Acc: 57.990,    Top5 Acc: 96.100,    Time: 0:00:13,      
# best result update!
# Current Acc: 57.990    Best Acc: 56.990

from trojanzoo.parser import Parser_Dataset, Parser_Model, Parser_Train, Parser_Seq, Parser_Mark, Parser_Attack, Parser_Defense

from trojanzoo.dataset import ImageSet
from trojanzoo.model import ImageModel
from trojanzoo.utils.mark import Watermark
from trojanzoo.attack.backdoor import BadNet
from trojanzoo.defense import Defense_Backdoor

import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    parser = Parser_Seq(Parser_Dataset(), Parser_Model(), Parser_Train(),
                        Parser_Mark(), Parser_Attack(), Parser_Defense())
    parser.parse_args()
    parser.get_module()

    dataset: ImageSet = parser.module_list['dataset']
    model: ImageModel = parser.module_list['model']
    optimizer, lr_scheduler, train_args = parser.module_list['train']
    mark: Watermark = parser.module_list['mark']
    attack: BadNet = parser.module_list['attack']
    defense: Defense_Backdoor = parser.module_list['defense']

    # ------------------------------------------------------------------------ #
    defense.detect(optimizer=optimizer, lr_scheduler=lr_scheduler, **train_args)
