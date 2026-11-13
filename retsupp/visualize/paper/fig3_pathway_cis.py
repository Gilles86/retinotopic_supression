"""Fig. 3d/e per pathway: HP − opposite, next − opposite, HP − next with bootstrapped 95% CIs, the pathway × location
interaction, and whether the early-area estimate lies inside each later pathway's CI (request 2026-11-13, item 5a).

Input: the per subject × pathway window means written by ``fig3_voxel_selection.py``
(``<prefix>_cells.tsv``: relative, % of the subject's mean response per item × location; ``<prefix>_abs_values.tsv``:
absolute, % signal, items pooled). Relative values are averaged over the three item types first, next = mean of the two
next-to-HP locations, as in ``fig3_item_locations.rel_stats``.

Per pathway and contrast: mean and median, each with a bootstrapped 95% CI over participants (10,000 resamples),
participants below 0, Wilcoxon signed-rank p. Pathway × location (HP, next, opposite): repeated-measures ANOVA on
participants with all three pathways, and a permutation test of the same interaction F (pathway labels shuffled within
participant, 10,000 permutations); also each pair of pathways. Early vs each later pathway: paired difference of the
contrast (mean, bootstrapped CI, Wilcoxon), and whether the early mean (median) lies inside the later pathway's CI.

    python -m retsupp.visualize.paper.fig3_pathway_cis
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy import stats

N_BOOT = 10000
PATHS = ['Early', 'Dorsal', 'Occipitotemporal']
CONTRASTS = (('hp_opp', 'HP − opposite', 'hp', 'opposite'), ('next_opp', 'Next − opposite', 'next', 'opposite'),
             ('hp_next', 'HP − next', 'hp', 'next'))


def values(prefix, selection):
    c = pd.read_csv(f'{prefix}_cells.tsv', sep='\t')
    c = c[c.selection == selection]
    w = c.pivot_table(index=['subject', 'grp'], columns='loc', values='r')     # mean over the item types
    rel = pd.DataFrame({'hp': w.hp, 'next': (w.nextleft + w.nextright) / 2, 'opposite': w.opposite}).reset_index()
    a = pd.read_csv(f'{prefix}_abs_values.tsv', sep='\t')
    a = a[a.selection == selection][['subject', 'grp', 'hp', 'next', 'opposite']]
    a[['hp', 'next', 'opposite']] *= 1.   # already % signal
    return {'relative (% of mean response)': rel, 'absolute (% signal)': a}


def boot(v, f, seed=0):
    rng = np.random.default_rng(seed)
    b = f(v[rng.integers(0, len(v), (N_BOOT, len(v)))], axis=1)
    return np.percentile(b, [2.5, 97.5])


def interaction_F(x):
    """x: subjects × pathways × locations. Two-way repeated-measures interaction F."""
    n, a, b = x.shape
    m = x.mean(0)
    gm = m.mean()
    sub = x.mean((1, 2))
    # interaction effect and its error term (pathway × location × subject)
    inter = m - m.mean(1, keepdims=True) - m.mean(0, keepdims=True) + gm
    ss_int = n * (inter ** 2).sum()
    sp = x.mean(2)            # subject × pathway
    sl = x.mean(1)            # subject × location
    resid = (x - sp[:, :, None] - sl[:, None, :] + sub[:, None, None]
             - m[None] + m.mean(1, keepdims=True)[None] + m.mean(0, keepdims=True)[None] - gm)
    ss_err = (resid ** 2).sum()
    df1, df2 = (a - 1) * (b - 1), (a - 1) * (b - 1) * (n - 1)
    return (ss_int / df1) / (ss_err / df2), df1, df2


def interaction(df, paths, seed=0):
    d = df[df.grp.isin(paths)].set_index(['subject', 'grp'])[['hp', 'next', 'opposite']]
    subs = [s for s, g in d.groupby(level=0) if len(g) == len(paths)]
    x = np.stack([d.loc[s].loc[paths].to_numpy() for s in subs])
    F, df1, df2 = interaction_F(x)
    p = stats.f.sf(F, df1, df2)
    rng = np.random.default_rng(seed)
    null = np.empty(N_BOOT)
    for i in range(N_BOOT):
        xp = np.stack([xi[rng.permutation(len(paths))] for xi in x])
        null[i] = interaction_F(xp)[0]
    return dict(pathways=' × '.join(paths), n=len(subs), F=F, df1=df1, df2=df2, p_F=p,
                p_perm=(1 + (null >= F).sum()) / (N_BOOT + 1))


def main(a):
    rows, inter, comp = [], [], []
    for sel, sel_lab in (('refit', 'current voxels, refitted pRFs (update 8)'), ('current', 'Fig. 3 as printed')):
        for unit, df in values(a.prefix, sel).items():
            est = {}
            for g in PATHS + ['All areas']:
                x = df[df.grp == g]
                for key, lab, i, j in CONTRASTS:
                    v = (x[i] - x[j]).to_numpy()
                    ml, mh = boot(v, np.mean)
                    dl, dh = boot(v, np.median)
                    r = dict(selection=sel, selection_label=sel_lab, unit=unit, pathway=g, contrast=lab, n=len(v),
                             mean=v.mean(), mean_ci_lo=ml, mean_ci_hi=mh, median=np.median(v), median_ci_lo=dl,
                             median_ci_hi=dh, k_neg=int((v < 0).sum()), p_wilcoxon=stats.wilcoxon(v).pvalue)
                    est[(g, key)] = (r, x.set_index('subject')[i] - x.set_index('subject')[j])
                    rows.append(r)
            for later in ('Dorsal', 'Occipitotemporal'):
                for key, lab, _, _ in CONTRASTS:
                    e, ev = est[('Early', key)]
                    l, lv = est[(later, key)]
                    dd = (ev - lv).dropna().to_numpy()
                    cl, ch = boot(dd, np.mean)
                    comp.append(dict(selection=sel, unit=unit, contrast=lab, later=later,
                                     early_mean=e['mean'], later_mean=l['mean'],
                                     later_mean_ci=f"[{l['mean_ci_lo']:.3g}, {l['mean_ci_hi']:.3g}]",
                                     early_mean_inside_later_ci=bool(l['mean_ci_lo'] <= e['mean'] <= l['mean_ci_hi']),
                                     early_median=e['median'], later_median=l['median'],
                                     early_median_inside_later_ci=bool(l['median_ci_lo'] <= e['median']
                                                                       <= l['median_ci_hi']),
                                     paired_n=len(dd), early_minus_later_mean=dd.mean(), diff_ci_lo=cl, diff_ci_hi=ch,
                                     diff_median=np.median(dd), p_wilcoxon=stats.wilcoxon(dd).pvalue))
            for paths in (PATHS, ['Early', 'Dorsal'], ['Early', 'Occipitotemporal'], ['Dorsal', 'Occipitotemporal']):
                inter.append(dict(selection=sel, unit=unit, **interaction(df, paths)))
    rows, inter, comp = pd.DataFrame(rows), pd.DataFrame(inter), pd.DataFrame(comp)
    rows.to_csv(f'{a.out_prefix}_contrasts.tsv', sep='\t', index=False, float_format='%.5g')
    inter.to_csv(f'{a.out_prefix}_interaction.tsv', sep='\t', index=False, float_format='%.5g')
    comp.to_csv(f'{a.out_prefix}_early_vs_later.tsv', sep='\t', index=False, float_format='%.5g')
    pd.set_option('display.width', 250)
    pd.set_option('display.max_columns', 30)
    print(rows.drop(columns='selection_label').round(4).to_string())
    print(inter.round(4).to_string())
    print(comp.round(4).to_string())


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--prefix', default='notes/data/fig3_voxel_selection_n28_2026-11-12')
    ap.add_argument('--out-prefix', default='notes/data/fig3_pathway_cis_n28_2026-11-13')
    main(ap.parse_args())
