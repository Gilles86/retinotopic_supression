"""Accuracy per condition, with the statistics used for the response times in Fig. 1 (n = 28).

Per participant, accuracy is the percentage of correct trials in each condition. No-response trials count as errors.
The conditions are:
- distractor at HP, at LP, or absent (all trials);
- target at HP or at LP (distractor-present trials only, as for the response times).

Each effect is the median of the within-participant differences, with a bootstrapped 95% CI (10,000 resamples over
participants). The table also gives the condition means, the mean difference, the number of participants in the
median's direction, and a two-sided Wilcoxon signed-rank p.

    python -m retsupp.visualize.paper.behaviour_accuracy
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy import stats


def main(trials, out, exclude=(1, 2), n_boot=10000, seed=0):
    d = pd.read_csv(trials, sep='\t')
    d = d[~d.subject.isin(exclude)].copy()
    d['acc'] = (d.correct.astype(str) == 'True').astype(float) * 100
    dc = d.groupby(['subject', 'dist_cond']).acc.mean().unstack()
    td = d[d.has_distractor.astype(str) == 'True'].groupby(['subject', 'target_cond']).acc.mean().unstack()
    rng = np.random.default_rng(seed)
    rows = []
    for name, a, b in [('Distractor at HP vs LP', dc.HP, dc.LP), ('Distractor at LP vs absent', dc.LP, dc.Absent),
                       ('Distractor at HP vs absent', dc.HP, dc.Absent),
                       ('Target at HP vs LP (distractor trials)', td.HP, td.LP)]:
        x = (a - b).dropna().to_numpy()
        boot = np.median(rng.choice(x, (n_boot, len(x))), axis=1)
        med = np.median(x)
        rows.append(dict(effect=name, median_pp=med, ci_lo=np.percentile(boot, 2.5), ci_hi=np.percentile(boot, 97.5),
                         mean_a=a.mean(), mean_b=b.mean(), mean_diff=x.mean(),
                         n_direction=int((np.sign(x) == np.sign(med)).sum()), n=len(x),
                         p_wilcoxon=stats.wilcoxon(x).pvalue))
    t = pd.DataFrame(rows)
    t.to_csv(out, sep='\t', index=False)
    print(f'{len(d)} trials, {int((d.acc > 0).sum())} correct, {d.rt.isna().sum()} without response')
    print(t.round(4).to_string(index=False))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--trials', default='notes/data/behaviour_trials_2026-11-11.tsv.gz')
    ap.add_argument('--out', default='notes/data/behaviour_accuracy_n28_2026-11-12.tsv')
    a = ap.parse_args()
    main(a.trials, a.out)
