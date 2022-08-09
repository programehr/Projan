import re
import numpy as np
from numpy import array as npa
from trojanzoo.utils.miscellaneous import outlier_ix, outlier_ix_val


def digest(p):
    with open(p, 'r') as f:
        t = f.read()

    res = re.findall(r'Validate Trigger\((\d*)\)', t, re.DOTALL | re.IGNORECASE)
    res = [int(x) for x in res]
    ntrig = max(res)

    res = re.findall(r'Results on the validation.*\n.*top1: (\S*)', t)
    res0 = [float(r) for r in res]

    ress = []
    for i in range(1, ntrig+1):
        res = re.findall(rf'Results on the validation.*?\n.*?Validate Trigger\({i}\) Tgt.*?top1: (\S*)', t, re.DOTALL)
        res1 = [float(r) for r in res]
        ress.append(res1)

    resor = re.findall(r'Results on the validation.*?\n.*?OR of \[Trigger Tgt\] on all triggers:  tensor\((.*?),', t,
                     re.DOTALL)
    resor = [float(r) for r in resor]

    resa = [res0]
    for r in ress:
        resa.append(r)
    resa.append(resor)

    resa = npa(resa)
    return resa.transpose()


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
        res = re.finditer(rf'{measure_name}:  tensor\(\[(.*?)\].*?\).*?{mad_name}.*?\)', text, re.DOTALL | re.IGNORECASE)
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


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    p = r'E:\Mehrin\IPM\trojanzoo\gridsearch.txt'
    res = digest(p)
    ntrig = res.shape[1]-2
    ix = 0
    res2 = np.zeros((10, 10, ntrig+2))
    for i in range(10):
        for j in range(10):
            res2[i, j, :] = res[ix, :]
            ix += 1

    res3 = np.zeros((10, 10, 4))
    for i in range(10):
        for j in range(10):
            res3[i, j, 0] = res2[i, j, 0]

            res3[i, j, 1] = res2[i, j, 1:ntrig+1].min()
            res3[i, j, 2] = res2[i, j, 1:ntrig+1].max()

            res3[i, j, 3] = res2[i, j, -1]

    # set ix to 0, ..., 9
    for i in range(4):
        plt.plot(res3[ix, :, i], '-o')
    plt.legend(['benign', 'min', 'max', 'or'])

    a = set((res[:, 0] > 95).nonzero()[0])
    b = set((res[:, 4] > 85).nonzero()[0])
    s1 = set((res[:, 1:4].min(1) > 30).nonzero()[0])
    s2 = set((res[:, 1:4].max(1) < 45).nonzero()[0])
    s1.intersection(s2).intersection(a).intersection(b)
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
    '''