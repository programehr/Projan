import csv
import os
import shutil
import datetime as dt
from datetime import datetime
from shutil import copytree, ignore_patterns, rmtree
import glob
import argparse

timeformat = '%Y-%m-%d %H:%M:%S'
# Note: To avoid loss of data while opening history.csv with Excel use custom > yyyy-mm-dd hh:mm:ss format
# (in Excel column data format)


def get_time():
    return datetime.now().strftime(timeformat)


def copy_dir(src_dir, dst_dir):
    for filename in os.listdir(src_dir):
        shutil.copy(os.path.join(src_dir, filename), dst_dir)


def delete_dir_contents(path):
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)


def write_experiment(ntrig, attack, dataset, model, iter, defense='-', timestamp=None, mode='test'):
    if mode == 'real':
        log_folder = 'tests2'
    elif mode == 'test':
        log_folder = 'tests3'
    else:
        raise ValueError('undefined mode.')

    experiment_log = f'{log_folder}/history.csv'
    experiment = [ntrig, attack, dataset, model, iter, defense]
    if defense == '-':
        exp_type = 'attack'
    else:
        exp_type = 'defense'
    if timestamp is None:
        timestamp = get_time()
    record = [timestamp, exp_type] + experiment
    with open(experiment_log, 'a+', newline='') as f:
        w = csv.writer(f, delimiter=',')
        w.writerow(record)


def read_experiments(mode='test'):
    if mode == 'real':
        log_folder = 'tests2'
    elif mode == 'test':
        log_folder = 'tests3'
    else:
        raise ValueError('undefined mode.')

    experiment_log = f'{log_folder}/history.csv'
    with open(experiment_log, 'r', newline='') as f:
        rd = csv.reader(f, delimiter=',')
        for recix, record in enumerate(rd):
            timestamp, exp_type, ntrig, attack, dataset, model, iter, defense = record
            timestamp = datetime.strptime(timestamp, timeformat)
            ntrig = int(ntrig)
            iter = int(iter)
            yield [timestamp, exp_type, ntrig, attack, dataset, model, iter, defense]


def find_experiment(ntrig, attack, dataset, model, iter, defense, after=None, mode='test'):
    exp_type = 'defense' if defense != '-' else 'attack'
    experiment = [exp_type, ntrig, attack, dataset, model, iter, defense]
    matches = []
    # experiment = [str(x) for x in experiment]
    for recix, record in enumerate(read_experiments(mode)):
        rec_timestamp = record[0]
        if record[1:] == experiment:
            if after is None or rec_timestamp > after:
                matches.append(recix)
    return matches


def remove_experiments(indexes, mode='test'):
    if mode == 'real':
        log_folder = 'tests2'
    elif mode == 'test':
        log_folder = 'tests3'
    else:
        raise ValueError('undefined mode.')

    experiment_log = f'{log_folder}/history.csv'
    recs = []
    for ix, rec in enumerate(read_experiments(mode)):
        if ix not in indexes:
            recs.append(rec)
    with open(experiment_log, 'w') as f:
        pass
    with open(experiment_log, 'a+', newline='') as f:
        w = csv.writer(f, delimiter=',')
        for ix, rec in enumerate(recs):
            w.writerow(rec)  # if you wanna use write_experiment be sure to pass timestamp


def is_done(ntrig, attack, dataset, model, iter, defense, mode='test'):
    matches = find_experiment(ntrig, attack, dataset, model, iter, defense, after=None, mode=mode)
    return len(matches) > 0


def run_attack(ntrig, attack, dataset, model, iter, mode='test'):
    if mode == 'real':
        log_folder = 'tests2'
    elif mode == 'test':
        log_folder = 'tests3'
    else:
        raise ValueError('undefined mode.')

    experiment_log = f'{log_folder}/history.csv'
    # used by trojanzoo to store latest trials results w/o regard to iter and ntrig
    att_main_folder = f'data/attack/image/{dataset}/{model}/{attack}'
    # used by me to separately store trials by iter, ntrig
    att_copy_folder = f'{log_folder}/{ntrig}/multitest_results/attacks/{attack}-{dataset}-{iter}'
    # used in test mode to back up existing results
    backup_folder = 'tests3/attacks/backup'
    att_log_path = f"{log_folder}/{ntrig}/attack_{attack}_{dataset}_multirun5.txt"
    os.makedirs(att_copy_folder, exist_ok=True)
    os.makedirs(att_main_folder, exist_ok=True)
    os.makedirs(backup_folder, exist_ok=True)

    extra = ''
    extra_ntoone = ''
    for i in range(ntrig - 1):  # first trigger is passed as mark not extra mark
        h, w = offsets[i]
        extra += f'--extra_mark "mark_path=square_white.png mark_height=3 mark_width=3 height_offset={h} width_offset={w}" '
        extra_ntoone += f'--extra_mark "mark_path=square_white.png mark_height=3 mark_width=3 height_offset={h} width_offset={w} mark_alpha={NTOONE_ALPHA}" '
    probs = (str(1 / (ntrig)) + ' ') * ntrig

    this_attack_args = attack_args[attack]
    if attack == 'prob':
        this_attack_args += '--probs ' + probs + extra
    if attack == 'ntoone':
        this_attack_args += ' ' + extra_ntoone

    alpha = NTOONE_ALPHA if attack == 'ntoone' else 0.0

    attack_cmd = f"python ./examples/backdoor_attack.py --verbose 1 --batch_size 100 " \
                 f"--dataset {dataset} --model {model} --attack {attack} " \
                 f"--device cuda --epoch {attack_epoch} --save " \
                 f"--mark_path square_white.png --mark_height 3 --mark_width 3 " \
                 f"--height_offset 2 --width_offset 2 --mark_alpha {alpha} " \
                 f"{this_attack_args} "
    if attack == 'prob' and dataset == 'cifar':
        attack_cmd += '--lr 0.001 '
    if 'model' != 'net':
        attack_cmd += '--pretrain '
    attack_cmd += f">> {att_log_path} "

    if mode == 'test':
        delete_dir_contents(backup_folder)
        copy_dir(att_main_folder, backup_folder)

    start_time = get_time()
    with open(att_log_path, 'a+') as f:
        f.write(f'attack started. {start_time}\n{ntrig}, {attack}, {dataset}, {model}, {iter}\n')
    print(f'{start_time}\n{ntrig}, {attack}, {dataset}, {model}, {iter}\n{attack_cmd}\n')
    exit_code = os.system(attack_cmd)  # will overwrite att_main_folder
    if exit_code != 0:
        # restore backed up experiment
        # it could be used for real mode too, but the user may wanna stop the trial (thus non-zero exit code)
        # and resume it later.
        if mode == 'test':
            copy_dir(backup_folder, att_main_folder)
        exit(exit_code)
    end_time = get_time()
    with open(att_log_path, 'a') as f:
        f.write(f'attack finished. {start_time}\t{end_time}\n{ntrig}, {attack}, {dataset}, {model}, {iter}\n')
    write_experiment(ntrig, attack, dataset, model, iter, '-', end_time, mode)

    copy_dir(att_main_folder, att_copy_folder)

    # restore backed up experiment
    if mode == 'test':
        copy_dir(backup_folder, att_main_folder)


def run_defense(ntrig, attack, dataset, model, iter, defense, mode='test'):
    if mode == 'real':
        log_folder = 'tests2'
    elif mode == 'test':
        log_folder = 'tests3'
    else:
        raise ValueError('undefined mode.')

    experiment_log = f'{log_folder}/history.csv'
    # used by trojanzoo to store latest trials results w/o regard to iter and ntrig
    def_main_folder = f"data/defense/image/{dataset}/{model}/{defense}/{attack}"
    att_main_folder = f'data/attack/image/{dataset}/{model}/{attack}'
    # used by me to separately store trials by iter, ntrig
    def_copy_folder = f'{log_folder}/{ntrig}/multitest_results/defenses/{defense}-{attack}-{dataset}-{iter}'
    att_copy_folder = f'{log_folder}/{ntrig}/multitest_results/attacks/{attack}-{dataset}-{iter}'
    # used in test mode to back up existing results
    backup_folder = 'tests3/defenses/backup'
    def_log_path = f"{log_folder}/{ntrig}/defense_{defense}_attack_{attack}_{dataset}_multirun5.txt"
    os.makedirs(def_main_folder, exist_ok=True)
    os.makedirs(def_copy_folder, exist_ok=True)
    os.makedirs(backup_folder, exist_ok=True)

    extra = ''
    extra_ntoone = ''
    for i in range(ntrig - 1):  # first trigger is passed as mark not extra mark
        h, w = offsets[i]
        extra += f'--extra_mark "mark_path=square_white.png mark_height=3 mark_width=3 height_offset={h} width_offset={w}" '
        extra_ntoone += f'--extra_mark "mark_path=square_white.png mark_height=3 mark_width=3 height_offset={h} width_offset={w} mark_alpha={NTOONE_ALPHA}" '
    probs = (str(1 / (ntrig)) + ' ') * ntrig

    this_attack_args = attack_args[attack]
    if attack == 'prob':
        this_attack_args += '--probs ' + probs + extra
    if attack == 'ntoone':
        this_attack_args += ' ' + extra_ntoone
    alpha = NTOONE_ALPHA if attack == 'ntoone' else 0.0

    # NB: batch size is not used in fulltest.py
    defense_cmd = f"python ./examples/backdoor_defense.py --verbose 1 " \
                  f"--dataset {dataset} --model {model} --attack {attack} --defense {defense} " \
                  f"--random_init --device cuda --save " \
                  f"{this_attack_args} " \
                  f"{defense_args[defense]} " \
                  f"--batch_size 50 --test_batch_size 1 --valid_batch_size 50 " \
                  f"--mark_path square_white.png --mark_height 3 --mark_width 3 " \
                  f"--height_offset 2 --width_offset 2 --mark_alpha {alpha} " \
                  f">> {def_log_path} "

    # copy target attack to the folder considered by defense
    copy_dir(att_copy_folder, att_main_folder)

    if mode == 'test':  # backup existing trial
        delete_dir_contents(backup_folder)
        copy_dir(def_main_folder, backup_folder)

    start_time = get_time()
    with open(def_log_path, 'a+') as f:
        f.write(f'defense started. {start_time}\n{ntrig}, {attack}, {dataset}, {model}, {iter}, {defense}\n')
    print(f'{start_time}\n{ntrig}, {attack}, {dataset}, {model}, {iter}, {defense}\n{defense_cmd}\n')
    exit_code = os.system(defense_cmd)  # will overwrite def_main_folder
    if exit_code != 0:
        # it could be used for real mode too, but the user may wanna stop the trial (thus non-zero exit code)
        # and resume it later.
        # restore backed up experiment
        if mode == 'test':
            copy_dir(backup_folder, def_main_folder)
        exit(exit_code)
    end_time = get_time()
    with open(def_log_path, 'a') as f:
        f.write(
            f'defense finished. {start_time}\t{end_time}\n{ntrig}, {attack}, {dataset}, {model}, {iter}, {defense}\n')
    write_experiment(ntrig, attack, dataset, model, iter, defense, end_time, mode)

    copy_dir(def_main_folder, def_copy_folder)

    # restore backed up experiment
    if mode == 'test':
        copy_dir(backup_folder, def_main_folder)


def run_experiments(experiments, mode='test'):
    for experiment in experiments:
        ntrig, attack, dataset, model, iter, defense = experiment
        if defense is None:
            defense = '-'
        if defense == '-':
            exp_type = 'attack'
        else:
            exp_type = 'defense'

        if exp_type == 'attack':
            if not find_experiment(ntrig, attack, dataset, model, iter, '-', after=None,
                                   mode=mode) or not skip_existing_trials:
                run_attack(ntrig, attack, dataset, model, iter, mode)
            else:
                print(f'{get_time()}: skipping {exp_type}: {experiment}\n')
        else:
            if not find_experiment(ntrig, attack, dataset, model, iter, '-', after=None, mode=mode):
                print(f'{get_time()}: Note: attack did not exist, running now:\n')
                run_attack(ntrig, attack, dataset, model, iter, mode)
            if not find_experiment(ntrig, attack, dataset, model, iter, defense, after=None,
                                   mode=mode) or not skip_existing_trials:
                run_defense(ntrig, attack, dataset, model, iter, defense, mode)
            else:
                print(f'{get_time()}: skipping {exp_type}: {experiment}\n')


def migrate(mode='test'):
    # NB: this function was needed to be run only once. Also pay attention to the mode argument
    # copy records from old history file to new history.csv file
    # setting 29/5/24 00:00 as timestamp
    from datetime import datetime, time
    # Get today's date
    # today = datetime.today().date()  # 29/05/2024
    theday = datetime.strptime("29/05/2024", "%d/%m/%Y")
    # Get the datetime for today at 00:00
    midnight = datetime.combine(theday, time())

    with open('tests2/history', 'r') as f:
        text = f.read()
    lines = text.splitlines(keepends=False)
    for line in lines:
        words = line.split(' ')
        exp_type, ntrig, attack, dataset, iter = words[:5]
        if exp_type == 'defense':
            defense = words[5]
        else:
            defense = '-'
        model = 'net' if dataset == 'mnist' else 'resnet18_comp'
        write_experiment(ntrig, attack, dataset, model, iter, defense, timestamp=midnight, mode=mode)


if __name__ == "__main__":  # Create the parser
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', metavar='mode', type=str, help='real or test')
    # Parse the arguments
    args = parser.parse_args()
    run_mode = args.mode

    NTOONE_ALPHA = .2
    attack_epoch = 100
    defence_epoch = 50
    num_trials = 10  # this is the number of trails per ntrig, attack, dataset.
    # including the previous attacks. i.e. if n trials already done, num_trials-n will be done
    skip_existing_trials = True  # if results exist for a trial, don't repeat it.

    attack_args = {
        'prob': '--cbeta_epoch 0 --losses loss1 loss2_11 loss3_11 --init_loss_weights 1.0 1.75 0.25 --poison_percent 0.1 '
                '--probs ',
        'badnet': '',
        'ntoone': ' ',
    }

    defense_args = {'neural_cleanse': f'--nc_epoch {defence_epoch} ',
                    'abs': '',
                    'deep_inspect': f'--remask_epoch {defence_epoch} ',
                    'tabor': f'--nc_epoch {defence_epoch} ',
                    'neuron_inspect': '',
                    'strip': '',
                    'newstrip': '--strict_test ',
                    'moth': '',
                    'clp': '--clp_batch_size 500 --clp_u 3. ',
                    'check_confidence': '',
                    }
    offsets = [(10, 10), (17, 17), (2, 10), (10, 2)]

    attacks = ['ntoone', 'prob', 'badnet']
    # defenses = ['neural_cleanse', 'tabor', 'neuron_inspect', 'abs']
    defenses = ['newstrip']
    datasets_models = [('mnist', 'net'), ('cifar10', 'resnet18_comp')]
    trial_indexes = list(range(1, 2))
    experiments = [(ntrig, attack, dataset, model, iter, defense)
                   for ntrig in range(2, 3)
                   for attack in attacks
                   for dataset, model in datasets_models
                   for iter in trial_indexes
                   for defense in defenses
                   ]

    experiments = [(ntrig, 'prob', dataset, model, 1, 'check_confidence')
                   for ntrig in range(2, 6)
                   for dataset, model in datasets_models] + \
                  [(2, 'badnet', dataset, model, 1, 'check_confidence')
                   for dataset, model in datasets_models] + \
                  [(4, 'ntoone', dataset, model, 1, 'check_confidence')
                   for dataset, model in datasets_models]

    run_experiments(experiments, run_mode)
