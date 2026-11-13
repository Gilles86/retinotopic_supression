"""Profile-likelihood figures for the LP-gain identifiability check (``modeling/joint_identifiability.py``).

Reads the profile TSVs (rsynced from ``derivatives/joint_optimisation_2026-11-13/profiles``) and draws, per area ×
model × profiled term: (a) training SSE on all runs relative to its minimum, in units of the residual variance σ²
(Δχ² under white noise; dashed line at 3.84) with the relative change on the right axis; (b) held-out SSE summed
over the three run3 folds, relative to its minimum; (c) what the other terms do along the profile.

    python -m retsupp.visualize.plot_joint_identifiability \\
        --tsv-dir notes/data/joint_optimisation_2026-11-13 --out-dir notes/figures/joint_optimisation_2026-11-13
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

LABEL = {'g_LP': 'Sustained LP gain g_LP', 'dS': 'HP-specific gain dS = g_HP − g_LP'}
TRADE = [('item_scale', 'Item scale'), ('amp_median', 'Median pRF amplitude'), ('g_HP', 'g_HP'), ('g_LP', 'g_LP'),
         ('dS', 'dS'), ('sigma_AF', 'σ_AF')]


def one(df, out):
    roi, model, param = df.roi.iloc[0], df.model.iloc[0], df.param.iloc[0]
    allr = df[df.split == 'all'].sort_values('value')
    folds = df[df.split != 'all']
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.0), constrained_layout=True)
    if len(allr):
        s2 = allr.sse_train.min() / allr.n_obs.iloc[0]
        d = allr.sse_train - allr.sse_train.min()
        ax[0].plot(allr.value, d / s2, 'o-', color='#1F2933', ms=3)
        ax[0].axhline(3.84, ls='--', color='#9AA5B1', lw=0.8)
        opt = allr[allr.kind == 'optimum']
        if len(opt):
            ax[0].axvline(opt.value.iloc[0], color='#D62828', lw=0.8)
        ax[0].set_ylabel('Δ training SSE / σ²')
        sec = ax[0].secondary_yaxis('right', functions=(lambda v, s=s2, m=allr.sse_train.min(): v * s / m,
                                                         lambda v, s=s2, m=allr.sse_train.min(): v * m / s))
        sec.set_ylabel('Relative Δ SSE')
        ax[0].set_title(f'All runs (n = {len(allr)} points)', fontsize=9)
    if len(folds) and 'sse_held' in folds:
        g = folds[folds.kind == 'grid'].groupby('value')[['sse_held', 'sse_train']].sum()
        nf = folds[folds.kind == 'grid'].groupby('value').split.nunique()
        g = g[nf == nf.max()]
        if len(g):
            s2h = g.sse_held.min() / folds.n_obs_held.groupby(folds.split).first().sum()
            ax[1].plot(g.index, (g.sse_held - g.sse_held.min()) / s2h, 'o-', color='#2A9D8F', ms=3, label='Held out')
            ax[1].plot(g.index, (g.sse_train - g.sse_train.min()) / (g.sse_train.min() / folds.n_obs.groupby(
                folds.split).first().sum()), 'o--', color='#9AA5B1', ms=3, label='Training')
            for _, r in folds[folds.kind == 'optimum'].iterrows():
                ax[1].axvline(r.value, color='#D62828', lw=0.6, alpha=0.6)
            ax[1].axhline(3.84, ls='--', color='#9AA5B1', lw=0.8)
            ax[1].legend(frameon=False, fontsize=8)
            ax[1].set_ylabel('Δ SSE / σ² (3 run3 folds)')
            ax[1].set_title('Held-out runs', fontsize=9)
    else:
        ax[1].set_visible(False)
    src = allr if len(allr) else folds[folds.split == folds.split.iloc[0]].sort_values('value')
    for col, lab in TRADE:
        if col == param or col not in src:
            continue
        v = src[col].to_numpy()
        ax[2].plot(src.value, v - v[np.argmin(np.abs(src.value.to_numpy() - src[src.kind == 'optimum'].value.iloc[0]))]
                   if (src.kind == 'optimum').any() else v, '-', lw=1, label=lab)
    ax[2].set_ylabel('Change from the optimum')
    ax[2].legend(frameon=False, fontsize=7)
    ax[2].set_title('Other terms along the profile', fontsize=9)
    for a in ax:
        a.set_xlabel(LABEL[param])
        a.spines[['top', 'right']].set_visible(False)
    fig.suptitle(f'sub-{df.subject.iloc[0]:02d} {roi} {model}: profile over {param}', fontsize=10)
    fig.savefig(out)
    plt.close(fig)
    print(out)


def main(tsv_dir, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tabs = [pd.read_csv(f, sep='\t') for f in sorted(Path(tsv_dir).glob('sub-*_profile-*.tsv'))]
    if not tabs:
        raise FileNotFoundError(f'no profile TSVs in {tsv_dir}')
    df = pd.concat(tabs).drop_duplicates(['roi', 'model', 'param', 'split', 'value', 'kind'])
    for (roi, model, param), d in df.groupby(['roi', 'model', 'param']):
        one(d, out_dir / f'profile_sub-{d.subject.iloc[0]:02d}_{roi}_{model}_{param}.pdf')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--tsv-dir', default='notes/data/joint_optimisation_2026-11-13')
    ap.add_argument('--out-dir', default='notes/figures/joint_optimisation_2026-11-13')
    a = ap.parse_args()
    main(a.tsv_dir, a.out_dir)
