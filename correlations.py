import os, pickle, sys
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import glob
from prettytable import PrettyTable
from tqdm import tqdm

t = None
all_ds = {}
all_acc = {}
allc = {}
all_metrics = {}
all_runs = {}
metric_names = ['grad_norm', 'snip', 'grasp', 'fisher', 'synflow', 'jacob_cov']

for fname, rname in [('result/darts_cf10_seed42_dlrandom_dlinfo1_initwnone_initbnone.p', 'CIFAR10')
                     ]:
    i = 0
    runs = []
    f = open(fname, 'rb')
    while (1):
        try:
            runs.append(pickle.load(f))
        except EOFError:
            break
    f.close()
    print(fname, len(runs))

    all_runs[fname] = runs
    all_ds[fname] = {}
    metrics = {}
    for k in metric_names:
        metrics[k] = []
    acc = []

    if t is None:
        hl = ['Dataset']
        hl.extend(metric_names)
        t = PrettyTable(hl)

    for r in runs:
        for k, v in r['log_measures'].items():
            if k in metrics:
                metrics[k].append(v)
        acc.append(r['val_acc'])

    all_ds[fname]['metrics'] = metrics
    all_ds[fname]['acc'] = acc

    res = []
    crs = {}
    for k in hl:
        if k == 'Dataset':
            continue
        v = metrics[k]
        cr = abs(stats.spearmanr(acc, v, nan_policy='omit').correlation)
        # print(f'{k} = {cr}')
        res.append(round(cr, 3))
        crs[k] = cr

    ds = rname
    all_acc[ds] = acc
    allc[ds] = crs
    t.add_row([ds] + res)

    all_metrics[ds] = metrics

print(t)

votes = {}


def vote(mets, gt):
    numpos = 0
    for m in mets:
        numpos += 1 if m > 0 else 0
    if numpos >= len(mets) / 2:
        sign = +1
    else:
        sign = -1
    return sign * gt


for ds in all_acc.keys():
    num_pts = len(acc)
    # num_pts = 1000
    tot = 0
    right = 0
    for i in tqdm(range(num_pts)):
        for j in range(num_pts):
            if i != j:
                diff = all_acc[ds][i] - all_acc[ds][j]
                if diff == 0:
                    continue
                diffsyn = []
                for m in ['synflow', 'jacob_cov', 'snip']:
                    diffsyn.append(all_metrics[ds][m][i] - all_metrics[ds][m][j])
                same_sign = vote(diffsyn, diff)
                right += 1 if same_sign > 0 else 0
                tot += 1
    votes[ds.lower() if 'CIFAR' in ds else ds] = right / tot
print('votes correlation: ', votes)
