'''
STRIP works with the distributions of benign entropy and trojan entropy.
It is assumed that the benign entropy must be of Gaussian distribution.
and the trojan entropy must be strike-like and much smaller than mean of benign entropies.
we plot both benign and trojan entropy assuming Gaussian distribution and compute Bayes error
In some Projan cases trojan entropy is wide and close to benign as expected, but not all cases.
'''
import matplotlib.pyplot as plt
import numpy as np
from numpy import array as npa
import scipy.stats as stats
import math

from trojanzoo.utils.analyze import read_strip_defense_file
from trojanzoo.utils.bayes_error import bayes_error

# trojan_mean, trojan_std = (0.6606101973491907, 0.33227990924655093)
# benign_mean, benign_std = (1.1172038949370384, 0.31109004004825974)
# badnet mnist
# trojan_mean, trojan_std = (0.0027939241727876285, 0.00746024529697422)
# benign_mean, benign_std = (1.3767105119037628, 0.3407281875894401)
# prob2 mnist
# trojan_mean, trojan_std = (0.5395899146056177, 0.23798558627644675)
# benign_mean, benign_std = (0.8827316248893737, 0.22680068350155905)
# prob5 mnist
# trojan_mean, trojan_std = (0.7962794880747794, 0.2760090233075355)
# benign_mean, benign_std = (1.0289814791059493, 0.2814333550836323)
# ntoone mnist
trojan_mean, trojan_std = (2.5568815077972413, 0.35587555116306385)
benign_mean, benign_std = (1.264472371749878, 0.29557774127467623)

start = min(trojan_mean - 3 * trojan_std, benign_mean - 3 * benign_std)
stop = max(trojan_mean + 3 * trojan_std, benign_mean + 3 * benign_std)

x = np.linspace(start, stop, 100)
plt.figure()
plt.plot(x, stats.norm.pdf(x, benign_mean, benign_std))
plt.plot(x, stats.norm.pdf(x, trojan_mean, trojan_std))

plt.show()
###
atks = [('badnet', 2), ('ntoone', 4)] + [('prob', i) for i in range(2, 6)]
#attack, ntrig, ds = 'badnet', 2, 'mnist'
# set ds = ...
for ix in range(len(atks)):
    attack, ntrig = atks[ix]
    p = rf'E:\Mehrin\IPM\trojanzoo\tests2\{ntrig}\defense_newstrip_attack_{attack}_{ds}_multirun5.txt'
    t = read_strip_defense_file(p)
    t = t[-10:]
    t = npa(t)
    troj_means = t[:, 2]
    troj_stds = t[:, 3]
    ben_means = t[:, 4]
    ben_stds = t[:, 5]
    bes = []
    for i in range(len(t)):
        be = bayes_error(troj_means[i], troj_stds[i], ben_means[i], ben_stds[i])
        bes.append(be)
    bes = npa(bes)
    # print(bes.mean(), bes.min(), bes.max())
    print(t[:, [2, 4]].mean(0))

    start = min(troj_means.mean() - 3 * troj_stds.mean(),
                ben_means.mean() - 3 * ben_stds.mean())
    stop = max(troj_means.mean() + 3 * troj_stds.mean(),
               ben_means.mean() + 3 * ben_stds.mean())
    x = np.linspace(start, stop, 100)

    for i in range(len(t)):
        plt.figure()
        plt.title(f'{attack}-{ntrig}-{ds}')
        plt.plot(x, stats.norm.pdf(x, ben_means[i], ben_stds[i]), 'b')
        plt.plot(x, stats.norm.pdf(x, troj_means[i], troj_stds[i]), 'r')

