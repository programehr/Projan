import glob
import os
import shutil

# configs = [
#     (5, 9, 'ibau', 'prob', 'mnist', 'net')
# ]
# configs = [(i, 9, 'ibau', 'prob', 'mnist', 'net') for i in range(2, 6)] + \
#             [(i, 9, 'ibau', 'prob', 'cifar10', 'resnet18_comp') for i in range(2, 6)] + \
#             [(2, 9, 'ibau', 'badnet', 'mnist', 'net')] + \
#             [(2, 9, 'ibau', 'badnet', 'cifar10', 'resnet18_comp')]
configs = [(i, 9, 'ibau', 'prob', 'mnist', 'net') for i in range(2, 6)] + \
          [(2, 9, 'ibau', 'badnet', 'mnist', 'net')]
# configs = [(2, 8, 'ibau', 'prob', 'cifar10', 'resnet18_comp')]

configs += [(i, 9, 'clp', 'prob', 'mnist', 'net') for i in range(2, 6)] + \
            [(i, 9, 'clp', 'prob', 'cifar10', 'resnet18_comp') for i in range(2, 6)] + \
            [(2, 9, 'clp', 'badnet', 'mnist', 'net')] + \
            [(2, 9, 'clp', 'badnet', 'cifar10', 'resnet18_comp')]

# configs = [(i, 9, 'nad', 'prob', 'mnist', 'net') for i in range(2, 6)] + \
#             [(i, 9, 'nad', 'prob', 'cifar10', 'resnet18_comp') for i in range(2, 6)] + \
#             [(2, 9, 'nad', 'badnet', 'mnist', 'net')] + \
#             [(2, 9, 'nad', 'badnet', 'cifar10', 'resnet18_comp')]

offsets = [(10, 10), (17, 17), (2, 10), (10, 2)]

for i_config, config in enumerate(configs):
    ntrig, trial, defense, attack, dataset, model = config
    att_respath = f'tests2/{ntrig}/multitest_results/attacks/{attack}-{dataset}-{trial}'
    for f in os.listdir(att_respath):
        shutil.copy(os.path.join(att_respath, f), f'data/attack/image/{dataset}/{model}/{attack}/')
    extra = ''
    for i in range(ntrig - 1):  # first trigger is passed as mark not extra mark
        h, w = offsets[i]
        extra += f'--extra_mark "mark_path=square_white.png mark_height=3 mark_width=3 height_offset={h} width_offset={w}" '
    probs = (str(1 / (ntrig)) + ' ') * ntrig

    attack_args = {
        'prob': '--losses loss1 loss2_11 loss3_11 --init_loss_weights 1.0 1.75 0.25 --poison_percent 0.1 '
                '--probs ' + probs + extra,
        'badnet': ''}
    defense_args = {
        'ibau': '--n_rounds 5',
        'clp': '--clp_batch_size 500 --clp_u 3. ',
        'nad': '',
        'absr4': '',
    }

    defense_cmd = f"python ./examples/backdoor_defense.py --verbose 1 " \
                  f"--dataset {dataset} --model {model} --attack {attack} --defense {defense} " \
                  f"--random_init --device cuda --save " \
                  f"{attack_args[attack]} " \
                  f"{defense_args[defense]} " \
                  f"--batch_size 50 --test_batch_size 1 --valid_batch_size 50 " \
                  f">> tests2/{ntrig}/defense_{defense}_attack_{attack}_{dataset}_multirun5.txt "
    exit_code = os.system(defense_cmd)
    if exit_code != 0:
        exit(exit_code)
    print(f'{i_config} done.')
    with open(f"tests2/{ntrig}/defense_{defense}_attack_{attack}_{dataset}_multirun5.txt", 'a') as f:
        f.write('defense finished.')
