# style inspired by https://allisonhorst.github.io/palmerpenguins/
# pre- and post- defense ASR for CLP and MOTH on CIFAR-10
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update({'font.family': 'arial'})

attack = ['Badnet', 'Nto1', 'Projan2', 'Projan3', 'Projan4', 'Projan5']
phases = {
    'Before defense': [94.14, 90.99, 95.61, 95.52, 93.59, 98.42],
    'After CLP defense': [17.8, 12.20, 63.04, 71.58, 85.22, 85.69],
    'After MOTH defense': [19.76, 85.16, 69.99, 74.20, 85.11, 91.06],
}

x = 1.2*np.arange(len(attack))  # the label locations
width = 0.35  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(layout='constrained')
# colors = [(1, 0, 0), (0, 1, 0)]
colors = ['tab:red', 'tab:green', 'tab:blue']

for i, (attribute, measurement) in enumerate(phases.items()):
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute, color=colors[i])
    ax.bar_label(rects, padding=3)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('accuracy')
ax.set_xlabel('Attack type')
ax.set_xticks(x + 3 * width / 2, attack)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13), fancybox=True, ncol=1)
ax.set_ylim(0, 110)
plt.show()

##########
# pre- and post- defense ASR for MOTH on MNIST
attack = ['Badnet', 'Nto1', 'Projan2', 'Projan3', 'Projan4', 'Projan5']
phases = {
    'Before defense': [100, 81.53, 97.55, 96.83, 97.73, 97.51],
    'After MOTH defense': [99.96, 75.91, 97.84, 96.79, 96.70, 97.56],
}

x = np.arange(len(attack))  # the label locations
width = 0.45  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(layout='constrained')
# colors = [(1, 0, 0), (0, 1, 0)]
colors = ['tab:red', 'tab:blue']

for i, (attribute, measurement) in enumerate(phases.items()):
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute, color=colors[i])
    ax.bar_label(rects, padding=3)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('accuracy')
ax.set_xlabel('Attack type')
ax.set_xticks(x + width / 2, attack)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13), fancybox=True, ncol=1)
ax.set_ylim(0, 110)
plt.show()
####
# STRIP entropy
attack = ['Badnet', 'Nto1', 'Projan2', 'Projan3', 'Projan4', 'Projan5']
# entropies = [0.53341056,0.92247291,0.36342508,0.8984709,0.85817314,2.58302763]  # cifar-10
entropies = [1.15733751e-03,2.50148545,0.50746084,0.58916627,0.64784393,1.34363751]  #mnist

x = np.arange(len(attack))/2  # the label locations
width = 0.3  # the width of the bars

fig, ax = plt.subplots(layout='constrained')
# colors = [(1, 0, 0), (0, 1, 0)]
color = 'tab:orange'

rects = ax.bar(x, entropies, width, color=color)
# ax.bar_label(rects, padding=3)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('entropy')
ax.set_xlabel('Attack type', ha='center')
ax.xaxis.set_label_coords(0.5, -.25)
ax.set_xticks(x, attack, rotation=90)
# ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13), fancybox=True, ncol=1)
plt.show()

####
import matplotlib.pyplot as plt
import numpy as np
from numpy import array as npa
import scipy.stats as stats
import math

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
