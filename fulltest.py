import os
import shutil
from shutil import copytree, ignore_patterns, rmtree
import glob

attack_epoch = 100
defence_epoch = 50
start_iter = 10
end_iter = 10

attacks = ['badnet', 'prob']
defenses = ['neural_cleanse', 'deep_inspect', 'tabor', 'neuron_inspect']
defense_args = {'neural_cleanse': f'--nc_epoch {defence_epoch} ',
                'abs': '',
                'deep_inspect': f'--remask_epoch {defence_epoch} ',
                'tabor': f'--nc_epoch {defence_epoch} ',
                'neuron_inspect': ''}

attack_args = {'prob': '--cbeta_epoch 0 --losses loss1 loss2_8 loss3_9 --init_loss_weights 1 0.7 0.3 --poison_percent 0.1 '
                       '--extra_mark "mark_path=square_white.png mark_height=3 mark_width=3 height_offset=10 width_offset=10" ',
               'badnet': ''}

datasets_models = [('cifar10', 'resnet18_comp'), ('mnist', 'net')]

os.makedirs('multitest_results', exist_ok=True)


for i in range(start_iter, end_iter+1):
    for attack in attacks:
        for dataset, model in datasets_models:
            attack_cmd = f"python ./examples/backdoor_attack.py --verbose 1 " \
                         f"--dataset {dataset} --model {model} --attack {attack} " \
                         f"--device cuda --epoch {attack_epoch} --save " \
                         f"--mark_path square_white.png --mark_height 3 --mark_width 3 " \
                         f"--height_offset 2 --width_offset 2 " \
                         f"{attack_args[attack]} " \

            if attack == 'prob' and dataset == 'cifar':
                attack_cmd += '--lr 0.001 '
            if 'model' != 'net':
                attack_cmd += '--pretrain '

            attack_cmd += f">> attack_{attack}_{dataset}_multirun5.txt "

            exit_code = os.system(attack_cmd)
            if exit_code != 0:
                exit(exit_code)

            att_respath = f'multitest_results/attacks/{attack}-{dataset}-{i}'
            os.makedirs(att_respath, exist_ok=True)
            for f in glob.glob(f'data/attack/image/{dataset}/{model}/{attack}/*'):
                if not f.endswith('.pth'):
                    shutil.copy(f, att_respath)

            for defense in defenses:
                defense_cmd = f"python ./examples/backdoor_defense.py --verbose 1 " \
                      f"--dataset {dataset} --model {model} --attack {attack} --defense {defense} " \
                      f"--random_init --device cuda --save " \
                      f"{defense_args[defense]} " \
                      f">> defense_{defense}_attack_{attack}_{dataset}_multirun5.txt "
                exit_code = os.system(defense_cmd)
                if exit_code != 0:
                    exit(exit_code)

                def_respath = f'multitest_results/defenses/{defense}-{attack}-{dataset}-{i}'
                os.makedirs(def_respath, exist_ok=True)
                for f in glob.glob(f"data/defense/image/{dataset}/{model}/{defense}/{attack}_*"):
                    shutil.copy(f, def_respath)
