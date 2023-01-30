import os
import shutil
from shutil import copytree, ignore_patterns, rmtree
import glob


def count_attacks(ntrig, attack, dataset):
    file_pat = f'tests2/{ntrig}/multitest_results/attacks/{attack}-{dataset}-*'
    l = glob.glob(file_pat)
    nums = []
    for x in l:
        index = x.rfind('-')
        nums.append(int(x[index + 1:]))
    return len(nums), max(nums)


def trial(ntrig):
    attack_epoch = 100
    defence_epoch = 50
    num_trials = 10  # this is the number of trails per ntrig, attack, dataset.
    # including the previous attacks. i.e. if n trials already done, num_trials-n will be done

    attacks = ['prob']
    defenses = ['neural_cleanse', 'tabor', 'neuron_inspect', 'abs']
    defense_args = {'neural_cleanse': f'--nc_epoch {defence_epoch} ',
                    'abs': '',
                    'deep_inspect': f'--remask_epoch {defence_epoch} ',
                    'tabor': f'--nc_epoch {defence_epoch} ',
                    'neuron_inspect': ''}
    offsets = [(10, 10), (17, 17), (2, 10), (10, 2)]
    extra = ''
    for i in range(ntrig - 1):  # first trigger is passed as mark not extra mark
        h, w = offsets[i]
        extra += f'--extra_mark "mark_path=square_white.png mark_height=3 mark_width=3 height_offset={h} width_offset={w}" '
    probs = (str(1/(ntrig))+' ') * ntrig

    attack_args = {
        'prob': '--cbeta_epoch 0 --losses loss1 loss2_11 loss3_11 --init_loss_weights 1.0 1.75 0.25 --poison_percent 0.1 '
                '--probs ' + probs + extra,
        'badnet': ''}

    datasets_models = [('cifar10', 'resnet18_comp'), ('mnist', 'net')]

    os.makedirs(f'tests2/{ntrig}/multitest_results', exist_ok=True)

    for attack in attacks:
        for dataset, model in datasets_models:
            num, maxnum = count_attacks(ntrig, attack, dataset)
            for i in range(maxnum + 1, maxnum + num_trials - num + 1):
                attack_cmd = f"python ./examples/backdoor_attack.py --verbose 1 --batch_size 100 " \
                             f"--dataset {dataset} --model {model} --attack {attack} " \
                             f"--device cuda --epoch {attack_epoch} --save " \
                             f"--mark_path square_white.png --mark_height 3 --mark_width 3 " \
                             f"--height_offset 2 --width_offset 2 " \
                             f"{attack_args[attack]} " \

                if attack == 'prob' and dataset == 'cifar':
                    attack_cmd += '--lr 0.001 '
                if 'model' != 'net':
                    attack_cmd += '--pretrain '

                attack_cmd += f">> tests2/{ntrig}/attack_{attack}_{dataset}_multirun5.txt "

                exit_code = os.system(attack_cmd)
                if exit_code != 0:
                    exit(exit_code)

                att_respath = f'tests2/{ntrig}/multitest_results/attacks/{attack}-{dataset}-{i}'
                os.makedirs(att_respath, exist_ok=True)
                for f in glob.glob(f'data/attack/image/{dataset}/{model}/{attack}/*'):
                    # if not f.endswith('.pth'):
                    shutil.copy(f, att_respath)

                for defense in defenses:
                    # NB: batch size is not used in fulltest.py
                    defense_cmd = f"python ./examples/backdoor_defense.py --verbose 1 " \
                                  f"--dataset {dataset} --model {model} --attack {attack} --defense {defense} " \
                                  f"--random_init --device cuda --save " \
                                  f"{defense_args[defense]} " \
                                  f"--batch_size 50 --test_batch_size 1 --valid_batch_size 50 " \
                                  f">> tests2/{ntrig}/defense_{defense}_attack_{attack}_{dataset}_multirun5.txt "
                    exit_code = os.system(defense_cmd)
                    if exit_code != 0:
                        exit(exit_code)

                    def_respath = f'tests2/{ntrig}/multitest_results/defenses/{defense}-{attack}-{dataset}-{i}'
                    os.makedirs(def_respath, exist_ok=True)
                    for f in glob.glob(f"data/defense/image/{dataset}/{model}/{defense}/{attack}_*"):
                        shutil.copy(f, def_respath)


for ntrig in range(2, 6):
    trial(ntrig)
