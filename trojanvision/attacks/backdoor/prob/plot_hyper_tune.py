# plot hyper-param tuning results
import numpy as np
from matplotlib import pyplot as plt

from trojanzoo.utils.analyze import analyze_attack_file
# Note: the path must follow attack_<attack>_<ds>_*.txt
# and must have 'attack finished' at the end of each trial.
p = r'E:\Mehrin\IPM\trojanzoo\attack_prob_mnist_gridsearch4.txt'
res = analyze_attack_file(p)
ntrig = len(res[0]['asr'])
ix = 0
res2 = np.zeros((8, 8, ntrig + 2))
for i in range(8):
    for j in range(8):
        res2[i, j, :] = [res[ix]['clean']] + res[ix]['asr'] + [res[ix]['or']]
        ix += 1

res3 = np.zeros((8, 8, 4))
for i in range(8):
    for j in range(8):
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