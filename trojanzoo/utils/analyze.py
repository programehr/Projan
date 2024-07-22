# analyze and summarize textual reports.
import glob
import os
import re
from pathlib import Path

import numpy as np
from numpy import array as npa

from trojanzoo.utils.extract_images import load_masked_mark
from trojanzoo.utils.miscellaneous import outlier_ix, outlier_ix_val

multi_trig_attacks = ['prob', 'ntoone']

def analyze_folder(fol, target):
    defpaths = [os.path.join(fol, x) for x in os.listdir(fol) if x.startswith('defense')]
    attpaths = [os.path.join(fol, x) for x in os.listdir(fol) if x.startswith('attack_prob')]
    defres = analyze_defense_files(defpaths, target)
    attres = analyze_attack_files(attpaths)
    return attres, defres


def analyze_attack_files(paths):
    res = {}
    for i, p in enumerate(paths):
        matches = re.findall('attack_(.*)_(.*)_.*', p)
        if len(matches) != 0:
            attack, dataset = matches[0]
        else:
            raise ValueError(f"Bad attack file name {p}")
        res0 = analyze_attack_file(p)
        res[(attack, dataset)] = res0
    return res


def analyze_attack_file(p):
    """

    Parameters
    ----------
    p : str
        file path
    Returns
    -------
    List
        one dict element per trial. Each dict may have the following fields:
        'clean': clean acc
        'asr': scalar or array (for prob and ntoone) of per-trigger ASRs
        'or': or of results (for prob)
        'comb': combination of all triggers (for ntoone and possibly prob)
    """
    matches = re.findall('attack_(.*)_(.*)_.*', p)
    attack, dataset = matches[0]
    with open(p, 'r') as f:
        text = f.read()
    chunks = text.split('env')
    chunks = [t for t in chunks if t.strip() != '']
    chunks = [t for t in chunks if 'attack finished.' in t]  # remove incomplete trials
    res = []
    for chunk in chunks:
        if attack in multi_trig_attacks:
            res0 = analyze_prob_attack(chunk)
        else:
            res0 = analyze_non_prob_attack(chunk)
        if res0 is not None:
            res.append(res0)

    return res


def analyze_prob_attack(t):
    result = {}
    res = re.findall(r'Validate Trigger\((\d*)\)', t, re.DOTALL | re.IGNORECASE)
    res = [int(x) for x in res]
    ntrig = max(res)

    best_starts = [x.start(0) for x in re.finditer('best result', t)]
    if len(best_starts) > 0:
        best_start = best_starts[-1]
    else:
        best_start = None

    res = re.findall(r'Results on the validation.*\n.*top1: (\S*)', t)
    res0 = [float(r) for r in res]
    if len(res0) == 0:
        return None
    res = re.finditer(r'Results on the validation.*\n.*top1: (\S*)', t)
    starts = []  # start of all results parts
    if best_start is None:
        best_index = None
        return None
    else:
        for i, r in enumerate(res):
            start = r.start(0)
            if start < best_start:
                best_index = i
    clean = res0[best_index]
    result['clean'] = clean
    asr = []
    for i in range(1, ntrig + 1):
        res = re.findall(rf'Results on the validation.*?\n.*?Validate Trigger\({i}\) Tgt.*?top1: (\S*)', t, re.DOTALL)
        res1 = [float(r) for r in res]
        asr.append(res1[best_index])
    result['asr'] = asr
    # resor = re.findall(r'Results on the validation.*?\n.*?OR of \[Trigger Tgt\] on all triggers:  tensor\((.*?),', t,
    #                    re.DOTALL)
    # resor = re.findall(r'Results on the validation.*?\n.*?OR of \[Trigger Tgt\] on all triggers:\s*(\S*)', t,
    #                    re.DOTALL)
    # this one can handle both pure numbers and tensors:
    resor = re.findall(r'Results on the validation.*?\n.*?OR of \[Trigger Tgt\] on all triggers:[^\d\.]*([\d\.]*)', t,
                       re.DOTALL)
    resor = [float(r) for r in resor]
    if len(resor) != 0:
        result['or'] = resor[best_index]

    rescomb_tgt = re.findall(r'Results on the validation.*?\n.*?Validate Comb.*? Tgt.*?top1:[^\d\.]*([\d\.]*)', t,
                                re.DOTALL)
    rescomb_tgt = [float(r) for r in rescomb_tgt]
    if len(rescomb_tgt) != 0:
        result['comb'] = rescomb_tgt[best_index]
    # the following one is not an important indicator
    rescomb_clean = re.findall(r'Results on the validation.*?\n.*?Validate Comb.*? Clean.*?top1:[^\d\.]*([\d\.]*)', t,
                                re.DOTALL)
    rescomb_clean = [float(r) for r in rescomb_clean]

    return result


def analyze_non_prob_attack(t):
    result = {}
    best_starts = [x.start(0) for x in re.finditer('best result', t)]
    if len(best_starts) > 0:
        best_start = best_starts[-1]
    else:
        best_start = None

    res = re.findall(r'Validate Clean.*top1: (\S*)', t)
    res0 = [float(r) for r in res]
    res = re.finditer(r'Validate Clean.*top1: (\S*)', t)
    starts = []  # start of all results parts
    if best_start is None:
        best_index = None
        return None
    else:
        for i, r in enumerate(res):
            start = r.start(0)
            if start < best_start:
                best_index = i
    result['clean'] = res0[best_index]

    res = re.findall(r'Validate Trigger Tgt.*top1: (\S*)', t)
    res1 = [float(r) for r in res]
    result['asr'] = res1[best_index]
    result['or'] = res1[best_index]

    return result


def analyze_defense_files(paths, target):
    res = {}
    for p in paths:
        matches = re.findall('defense_(.*)_attack_(.*)_(.*)_.*', p)
        if len(matches) != 0:
            defense, attack, dataset = matches[0]
            hard_detection, hard_anom_indexes, soft_detection, soft_anom_indexes = analyze_defense_file(p, target,
                                                                                                        defense)
            hard_anom_index_mean = hard_anom_indexes.mean()
            soft_anom_index_mean = soft_anom_indexes.mean()
            res[(attack, defense, dataset)] = hard_detection, hard_anom_index_mean, soft_detection, soft_anom_index_mean
    return res


def analyze_all_trig_defenses_by_mask(root, target):
    fols = os.listdir(root)
    d = {}
    for i, ntrig in enumerate(fols):
        fol = os.path.join(root, ntrig, 'multitest_results', 'defenses')
        res = analyze_defenses_by_mask(fol, target)
        d0 = {k + (ntrig,): v for k, v in res.items()}
        d.update(d0)
    return d


def analyze_defenses_by_mask(fol, target):
    fols = os.listdir(fol)
    d = {}
    for i, folder in enumerate(fols):
        name = Path(folder).stem
        defense, attack, dataset, num = name.split('-')
        if defense == 'neuron_inspect':
            continue
        key = (attack, defense, dataset)
        if key not in d:
            d[key] = None
    for key in d:
        attack, defense, dataset = key
        hard_detection, hard_anom_indexes, soft_detection, soft_anom_indexes, \
            hard_outliers, soft_outliers = \
            analyze_defense_by_mask(fol, defense, attack, dataset, target)
        hard_anom_index_mean = hard_anom_indexes.mean()
        soft_anom_index_mean = soft_anom_indexes.mean()
        d[key] = hard_detection, hard_anom_index_mean, soft_detection, soft_anom_index_mean, \
            hard_anom_indexes, soft_anom_indexes, hard_outliers, soft_outliers
    return d


def analyze_defense_by_mask(root, defense, attack, dataset, target):
    # analyze all trials
    sep = os.path.sep
    pat = f'{root}{sep}{defense}-{attack}-{dataset}'
    num_attacks = len(glob.glob(pat + '-*'))
    soft_outliers = []
    hard_outliers = []
    soft_anom_indexes = []
    hard_anom_indexes = []
    for i in range(num_attacks):
        folder = pat + '-' + str(i)
        npy_list = glob.glob(folder + sep + '*best.npy')
        npz_list = glob.glob(folder + sep + '*.npz')
        if npy_list:
            path = npy_list[0]
        elif npz_list:
            path = npz_list[0]
        else:
            return None, None, None, None
        hard_ix, hard_anom_index, soft_ix, soft_anom_index = _analyze_defense_trial_by_mask(path)
        soft_outliers.append(soft_ix)
        soft_anom_indexes.append(soft_anom_index.item())
        hard_outliers.append(hard_ix)
        hard_anom_indexes.append(hard_anom_index.item())
    soft_anom_indexes = npa(soft_anom_indexes)
    hard_anom_indexes = npa(hard_anom_indexes)

    hard_detection = analyze_detections(hard_outliers, target)
    soft_detection = analyze_detections(soft_outliers, target)
    return hard_detection, hard_anom_indexes, soft_detection, soft_anom_indexes, \
        hard_outliers, soft_outliers


def _analyze_defense_trial_by_mask(path):
    # analyze nth trial
    masks = load_masked_mark(path)
    norms = [mask.sum() for mask in masks]
    soft_ix, soft_vals, soft_med, soft_anom_index = outlier_ix_val(norms, soft=True)
    hard_ix, hard_vals, hard_med, hard_anom_index = outlier_ix_val(norms, soft=False)
    return hard_ix, hard_anom_index, soft_ix, soft_anom_index


def analyze_defense_trial_by_mask(root, defense, attack, ds, ntrig, trial):
    folder = rf"{root}\{ntrig}\multitest_results\defenses\{defense}-{attack}-{ds}-{trial}"
    sep = os.path.sep
    npy_list = glob.glob(folder + sep + '*best.npy')
    npz_list = glob.glob(folder + sep + '*.npz')
    if npy_list:
        path = npy_list[0]
    elif npz_list:
        path = npz_list[0]
    res = _analyze_defense_trial_by_mask(path)
    return res


def analyze_defense_file(inpath, target, defense=None):
    if defense is None:
        defense = re.findall('defense_(.*)_attack', inpath)[0]
    params = {
        'neural_cleanse': ('mask norms', True),
        'deep_inspect': ('mark norms', True),
        'neuron_inspect': ('exp features', True),
        'abs': ('Score', False),
        'tabor': ('mask norms', True)
    }
    hard_outliers, hard_anom_indexes, soft_outliers, soft_anom_indexes = \
        read_outliers(inpath, *params[defense])
    hard_detection = analyze_detections(hard_outliers, target)
    soft_detection = analyze_detections(soft_outliers, target)
    return hard_detection, hard_anom_indexes, soft_detection, soft_anom_indexes


def read_outliers(inpath, measure_name='loss', is_tensor=True):
    with open(inpath, 'r') as f:
        text = f.read()
    chunks = text.split('env')
    chunks = [t for t in chunks if t.strip() != '']
    chunks = [t for t in chunks if 'defense finished.' in t]  # remove incomplete trials

    if is_tensor:
        # res = re.finditer(rf'{measure_name}:  tensor\(\[(.*?)\].*?\).*?{mad_name}.*?\)', text,
        res = [re.search(rf'{measure_name}:\s\s?tensor\(\[(.*?)\].*?\)', chunk,
                          re.DOTALL | re.IGNORECASE) for chunk in chunks]
    else:
        res = [re.search(rf'{measure_name}:\s\s?\[(.*?)\]', chunk, re.DOTALL | re.IGNORECASE)
               for chunk in chunks]
    all_nums = []

    soft_outliers = []
    hard_outliers = []
    soft_anom_indexes = []
    hard_anom_indexes = []
    for r in res:
        g = r.groups()[0]
        nums = re.split(r'[,\r\n\t\s]', g)
        nums = [float(num) for num in nums if len(num) > 0]
        all_nums.append(nums)

        soft_ix, soft_vals, soft_med, soft_anom_index = outlier_ix_val(nums, soft=True)
        soft_outliers.append(soft_ix)
        soft_anom_indexes.append(soft_anom_index.item())

        hard_ix, hard_vals, hard_med, hard_anom_index = outlier_ix_val(nums, soft=False)
        hard_outliers.append(hard_ix)
        hard_anom_indexes.append(hard_anom_index.item())

    soft_outliers = soft_outliers
    hard_outliers = hard_outliers
    soft_anom_indexes = npa(soft_anom_indexes)
    hard_anom_indexes = npa(hard_anom_indexes)

    return hard_outliers, hard_anom_indexes, soft_outliers, soft_anom_indexes


def analyze_detections(arrays, target):
    histo = {detection: 0 for detection in ['d', 'dfp', 'nd', 'wd']}
    # detected,
    # detected w/ false positive,
    # not detected,
    # wrong detection.
    for x in arrays:
        if target in x:
            if len(x) == 1:
                histo['d'] += 1
            else:
                histo['dfp'] += 1
        elif len(x) == 0:
            histo['nd'] += 1
        else:
            histo['wd'] += 1
    return histo


# some ad-hoc functions
# todo: write anomaly indexes as well (also from defense methods)
def add_outlier_analys_all(inpath):
    clean_outlier(inpath)
    args = \
        [('norms', 'mask mad', True),
         ('norms', 'mark mad', True),
         ('score', 'score mad', False),
         ('exp features', 'exp mad', True),
         ]
    for measure_name, mad_name, is_tensor in args:
        add_outlier_analys(inpath, None, measure_name, mad_name, is_tensor)


def add_outlier_analys(inpath, outpath=None, measure_name='loss', mad_name='loss mad', is_tensor=True):
    if outpath is None:
        outpath = inpath
    with open(inpath, 'r') as f:
        text = f.read()
    if is_tensor:
        res = re.finditer(rf'{measure_name}:  tensor\(\[(.*?)\].*?\).*?{mad_name}.*?\)', text,
                          re.DOTALL | re.IGNORECASE)
    else:
        res = re.finditer(rf'{measure_name}:  \[(.*?)\].*?{mad_name}.*?\)', text, re.DOTALL | re.IGNORECASE)
    res = list(res)
    res.reverse()
    out_text = text
    for r in res:
        g = r.groups()[0]
        nums = re.split(r'[,\r\n\t\s]', g)
        nums = [float(num) for num in nums if len(num) > 0]
        s = r.start(0)  # 0 refers to the whole match
        e = r.end(0)
        soft_ix, soft_vals, soft_med, soft_foo = outlier_ix_val(nums, soft=True)
        hard_ix, hard_vals, hard_med, hard_foo = outlier_ix_val(nums, soft=False)
        t = f'\noutlier classes (soft median): {soft_ix}\n outlier scores: {soft_vals}\n median:{soft_med}////\n' \
            f'foo: {soft_foo}\n' \
            f'outlier classes (hard median): {hard_ix}\n outlier scores: {hard_vals}\n median:{hard_med}////' \
            f'foo: {hard_foo}\n'

        out_text = out_text[:e] + t + out_text[e:]

    with open(outpath, 'w') as f:
        f.write(out_text)


def clean_outlier(inpath, outpath=None):
    if outpath is None:
        outpath = inpath
    with open(inpath, 'r') as f:
        text = f.read()
    res = re.finditer(r'outlier.*?foo.*?\n', text, re.DOTALL)
    res = list(res)
    res.reverse()

    text = list(text)
    for r in res:
        s = r.start(0)  # 0 refers to the whole match
        e = r.end(0)
        del text[s:e]
    out_text = ''.join(text)

    with open(outpath, 'w') as f:
        f.write(out_text)


def clean_outlier_temp(inpath, outpath=None):
    if outpath is None:
        outpath = inpath
    with open(inpath, 'r') as f:
        text = f.read()
    res = re.finditer(r'outlier.*?\n', text, re.DOTALL)
    res = list(res)
    res.reverse()

    text = list(text)
    for r in res:
        s = r.start(0)  # 0 refers to the whole match
        e = r.end(0)
        del text[s:e]
    out_text = ''.join(text)

    with open(outpath, 'w') as f:
        f.write(out_text)


def read_outliers0(inpath):
    pat = r"mask norms:  tensor\(\[(.*?)\].*?\)"
    # text = "mask norms:  tensor([ 4.1318, 67.0116, 44.2409, 49.2531, 59.6881, 53.1895, 62.4804, 62.9398, \n52.2324, 68.4044], device='cuda:0')"
    with open(inpath, 'r') as f:
        text = f.read()
    res = re.finditer(pat, text, re.DOTALL)
    all_nums = []
    for r in res:
        g = r.groups()[0]
        nums = re.split(r'[,\r\n\t\s]', g)
        nums = [float(num) for num in nums if len(num) > 0]
        all_nums.append(nums)


def fix_deep_inspect(inpath, outpath=None):
    if outpath is None:
        outpath = inpath
    pat = r"mark norms: \[tensor.*?\]"
    with open(inpath, 'r') as f:
        text = f.read()
    res = re.finditer(pat, text)
    res = list(res)
    res.reverse()
    out_text = text

    pat = r'tensor\((.*?),'
    for r in res:
        match = r.group()
        nums = re.findall(pat, match)
        nums = ', '.join(nums)
        s = r.start(0)  # 0 refers to the whole match
        e = r.end(0)
        t = f'mark norms:  tensor([{nums}])'
        out_text = out_text[:s] + t + out_text[e:]

    with open(outpath, 'w') as f:
        f.write(out_text)


def read_new_defense(p):
    with open(p, 'r') as f:
        text = f.read()
    acc_pat = r'''defense evaluation.*?Validate Clean.*?top1:\s*(\S*)'''
    if 'prob' in p:
        asr_pat = r'''defense evaluation.*?Validate .*?OR of \[Trigger Tgt\] on all triggers:\s*(\S*)'''
    else:
        asr_pat = r'''defense evaluation.*?Validate Trigger Tgt.*?top1:\s*(\S*)'''
    chunks = re.split('env', text)
    del chunks[0]
    pre_acc, pre_asr, post_acc, post_asr = [], [], [], []
    for i, chunk in enumerate(chunks):
        t0 = re.findall('pre-' + acc_pat, chunk, re.DOTALL)
        t1 = re.findall('pre-' + asr_pat, chunk, re.DOTALL)
        t2 = re.findall('post-' + acc_pat, chunk, re.DOTALL)
        t3 = re.findall('post-' + asr_pat, chunk, re.DOTALL)
        if len(t0) != 1 or len(t1) != 1 or len(t2) != 1 or len(t3) != 1:
            continue
        pre_acc.append(float(t0[0]))
        pre_asr.append(float(t1[0]))
        post_acc.append(float(t2[0]))
        post_asr.append(float(t3[0]))
    return pre_acc, pre_asr, post_acc, post_asr


'''
from matplotlib import pyplot as plt

p = r'E:\Mehrin\IPM\trojanzoo\gridsearch4.txt'
res = analyze_attack_file(p)
ntrig = res.shape[1] - 2
ix = 0
res2 = np.zeros((10, 10, ntrig + 2))
for i in range(10):
    for j in range(10):
        res2[i, j, :] = res[ix, :]
        ix += 1

res3 = np.zeros((10, 10, 4))
for i in range(10):
    for j in range(10):
        res3[i, j, 0] = res2[i, j, 0]

        res3[i, j, 1] = res2[i, j, 1:ntrig + 1].min()
        res3[i, j, 2] = res2[i, j, 1:ntrig + 1].max()

        res3[i, j, 3] = res2[i, j, -1]

plt.rcParams['text.usetex'] = True
x = [0.25*(i+1) for i in range(8)]
# set ix to 0, ..., 9
for i in range(4):
    plt.plot(x, res3[ix, :, i], '-o')
plt.legend(['benign', 'min', 'max', 'overall'])

plt.xlabel(r'$\lambda_2$')
plt.ylabel(r'accuracy')
plt.title(rf'$\lambda_1 = {0.25*(ix+1)}$')

a = set((res[:, 0] > 95).nonzero()[0])
b = set((res[:, 4] > 85).nonzero()[0])
s1 = set((res[:, 1:4].min(1) > 30).nonzero()[0])
s2 = set((res[:, 1:4].max(1) < 45).nonzero()[0])
s1.intersection(s2).intersection(a).intersection(b)
'''
# result: {31, 62, 92}. res[62] = res2[6, 2, :]
'''
res[31, :]
Out[49]: array([98.9 , 41.3 , 44.08, 43.04, 88.51])
res[62, :]
Out[50]: array([99.14, 44.52, 43.58, 41.65, 88.67])
res[92, :]
Out[51]: array([99.05, 39.47, 42.55, 39.12, 85.3 ])
'''
# results of gridsearch2: {16}. res[16] = res2[2, 0, :]
# weights: w2 = 0.75, w1 = 0.25
'''
losses: 1, 2_11, 3_11
best res: [98.94 37.24 32.21 32.37 98.89]
        weights: [1.0, 1.75, 0.25]
'''
'''
for k, v in defres.items():
    print(*k, end='  ')
    print(*v[0].values(), end='  ')        
    print(v[1])
    
for k, v in attres.items():
    print(*k, end='  ')
    print(v.mean(0))
'''
