"""Relate trial-wise / RT-binned phasic AF gains to reaction times.

Two steps, split along the data-size boundary:

``collect`` (cluster): reads ``derivatives/af_trialwise_gains/sub-*/``
(from :mod:`retsupp.modeling.fit_trialwise_af_gains`) and writes four
small TSVs:

- ``<prefix>_rt_betas.tsv`` — per (subject, ROI, α): within-subject OLS
  of z(log RT) on z(per-trial gain). Both sides are z-scored within
  (session, run, trial type), so trial-type RT effects and slow drifts
  are removed and only trial-to-trial covariation remains. Predictors:
  ``T`` (target gain, all trials), ``D_HP`` / ``D_LP`` (distractor gain
  on HP / LP trials, 0 elsewhere) and ``G`` (global gain).
- ``<prefix>_cv.tsv``, ``<prefix>_bins.tsv``, ``<prefix>_check.tsv`` —
  concatenations of the per-(subject, ROI) files.

``plot`` (local): group statistics + two figures, one per analysis
(per-trial, RT bins).

Usage
-----
``python -m retsupp.visualize.paper.analyze_trialwise_af_gains collect \\
    --bids-folder /shares/zne.uzh/gdehol/ds-retsupp \\
    --prefix notes/data/trialwise_af``
``python -m retsupp.visualize.paper.analyze_trialwise_af_gains plot \\
    --prefix notes/data/trialwise_af --fig-dir notes/figures``
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

ROI_ORDER = ['V1', 'V2', 'V3', 'V3AB', 'hV4', 'LO', 'TO', 'VO', 'IPS', 'SPL1', 'FEF']
KIND_PALETTE = {'T': '#F77F00', 'D_HP': '#D62828', 'D_LP': '#0077B6',
                'G': '#9AA5B1'}
KIND_LABEL = {'T': 'Target', 'D_HP': 'HP distractor', 'D_LP': 'LP distractor',
              'G': 'Global'}
MAIN_ALPHA = 1.0
CELL = ['session', 'run', 'trial_type']


def _z(x):
    sd = x.std()
    return (x - x.mean()) / sd if sd > 0 else x * 0.0


def rt_betas(trials: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (subject, roi, alpha), df in trials.groupby(['subject', 'roi', 'alpha']):
        df = df[df['valid_rt']].copy()
        df['y'] = df.groupby(CELL)['rt'].transform(lambda x: _z(np.log(x)))
        X = {}
        for kind in ('T', 'G'):
            X[kind] = df.groupby(CELL)[f'g_{kind}'].transform(_z)
        for tt in ('HP', 'LP'):
            col = df.groupby(CELL)[f'g_D_{tt}'].transform(_z)
            X[f'D_{tt}'] = col.where(df['trial_type'] == tt, 0.0).fillna(0.0)
        X = pd.DataFrame(X)
        Xm = np.column_stack([np.ones(len(X)), X.values])
        beta, *_ = np.linalg.lstsq(Xm, df['y'].values, rcond=None)
        for kind, b in zip(X.columns, beta[1:]):
            rows.append(dict(subject=subject, roi=roi, alpha=alpha,
                             kind=kind, beta=b, n_trials=len(df)))
    return pd.DataFrame(rows)


def collect(bids_folder: Path, prefix: Path):
    root = bids_folder / 'derivatives' / 'af_trialwise_gains'
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for name in ('cv', 'bins', 'check'):
        files = sorted(root.glob(f'sub-*/sub-*_roi-*_{name}.tsv'))
        df = pd.concat([pd.read_csv(f, sep='\t') for f in files])
        df.to_csv(f'{prefix}_{name}.tsv', sep='\t', index=False)
        print(f'{name}: {len(files)} files -> {prefix}_{name}.tsv')
    betas = []
    for f in sorted(root.glob('sub-*/sub-*_roi-*_trials.tsv')):
        betas.append(rt_betas(pd.read_csv(f, sep='\t')))
        print(f.name, flush=True)
    pd.concat(betas).to_csv(f'{prefix}_rt_betas.tsv', sep='\t', index=False)
    print(f'-> {prefix}_rt_betas.tsv')


def _group_t(df, value, by):
    out = []
    for key, d in df.groupby(by):
        t, p = stats.ttest_1samp(d[value].dropna(), 0.0)
        out.append(dict(zip(by, key if isinstance(key, tuple) else (key,)),
                        mean=d[value].mean(), sem=d[value].sem(),
                        t=t, p=p, n=d[value].notna().sum()))
    return pd.DataFrame(out)


def plot(prefix: Path, fig_dir: Path):
    sns.set_theme(style='ticks', context='paper')
    fig_dir.mkdir(parents=True, exist_ok=True)
    cv = pd.read_csv(f'{prefix}_cv.tsv', sep='\t')
    betas = pd.read_csv(f'{prefix}_rt_betas.tsv', sep='\t')
    bins = pd.read_csv(f'{prefix}_bins.tsv', sep='\t')
    check = pd.read_csv(f'{prefix}_check.tsv', sep='\t')
    rois = [r for r in ROI_ORDER if r in set(check['roi'])]

    print('Sanity (canonical phasic reconstruction, R² recomputed vs stored):')
    print(check[['phasic_recon_rel_err', 'pulse_rebuild_max_err']].max().to_string())
    print('max |R² recomputed − stored| =',
          (check['r2_recomputed'] - check['r2_stored']).abs().max())

    # ------------------------------------------------------------------
    # Figure 1: per-trial gains
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4),
                             gridspec_kw=dict(width_ratios=[1, 1, 1.6]))

    ax = axes[0]
    sns.lineplot(cv, x='alpha', y='heldout_gain', hue='roi', hue_order=rois,
                 palette='viridis', errorbar=None, ax=ax, legend=False)
    ax.axhline(0, color='#1F2933', lw=0.8)
    ax.set(xscale='log', xlabel='Ridge α (shrinkage to scalar gains)',
           ylabel='Held-out ΔSSE (fraction)',
           title='Per-trial gains vs scalar model')

    ax = axes[1]
    rel = cv[cv['alpha'] == MAIN_ALPHA].melt(
        id_vars=['subject', 'roi'],
        value_vars=[f'splithalf_r_{k}' for k in KIND_PALETTE],
        var_name='kind', value_name='r')
    rel['kind'] = rel['kind'].str.replace('splithalf_r_', '')
    sns.pointplot(rel, x='roi', y='r', hue='kind', order=rois,
                  palette=KIND_PALETTE, dodge=0.5, linestyle='none',
                  errorbar=('ci', 95), markersize=3, ax=ax, legend=False)
    ax.axhline(0, color='#1F2933', lw=0.8)
    ax.set(xlabel='', ylabel='Split-half r (voxel halves)',
           title=f'Reliability of trial deviations (α={MAIN_ALPHA:g})')
    ax.tick_params(axis='x', rotation=90)

    ax = axes[2]
    b = betas[betas['alpha'] == MAIN_ALPHA]
    sns.pointplot(b, x='roi', y='beta', hue='kind', order=rois,
                  hue_order=list(KIND_PALETTE), palette=KIND_PALETTE,
                  dodge=0.5, linestyle='none', errorbar=('ci', 95),
                  markersize=3, ax=ax)
    ax.axhline(0, color='#1F2933', lw=0.8)
    ax.set(xlabel='', ylabel='β: z(log RT) ~ z(trial gain)',
           title='Trial-by-trial gain → RT')
    ax.tick_params(axis='x', rotation=90)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, [KIND_LABEL[l] for l in labels], frameon=False,
              fontsize=7, loc='upper left', bbox_to_anchor=(1, 1))
    sns.despine(fig)
    fig.tight_layout()
    out = fig_dir / 'trialwise_af_gains_rt.pdf'
    fig.savefig(out)
    print(f'Wrote {out}')

    stats_trial = _group_t(b, 'beta', ['roi', 'kind'])
    print('\nPer-trial RT betas (α=1), one-sample t vs 0:')
    print(stats_trial.sort_values('p').head(12).to_string(index=False))

    # ------------------------------------------------------------------
    # Figure 2: RT-binned gains (slow − fast)
    # ------------------------------------------------------------------
    piv = bins[bins['bin'].isin(['fast', 'slow'])].pivot_table(
        index=['subject', 'roi', 'scheme', 'with_global', 'kind'],
        columns='bin', values='g').reset_index()
    piv['slow_minus_fast'] = piv['slow'] - piv['fast']
    g = sns.catplot(piv, x='roi', y='slow_minus_fast', hue='kind',
                    order=rois, hue_order=list(KIND_PALETTE),
                    palette=KIND_PALETTE, row='with_global', col='scheme',
                    kind='point', dodge=0.5, linestyle='none',
                    errorbar=('ci', 95), markersize=3,
                    height=2.6, aspect=1.8, sharey=False)
    for ax in g.axes.flat:
        ax.axhline(0, color='#1F2933', lw=0.8)
        ax.tick_params(axis='x', rotation=90)
    g.set_axis_labels('', 'Gain (slow − fast)')
    g.set_titles('{col_name} | global regressor: {row_name}')
    out = fig_dir / 'trialwise_af_gains_rt_bins.pdf'
    g.savefig(out)
    print(f'Wrote {out}')

    stats_bins = _group_t(piv, 'slow_minus_fast',
                          ['scheme', 'with_global', 'roi', 'kind'])
    print('\nBinned slow − fast, one-sample t vs 0:')
    print(stats_bins.sort_values('p').head(12).to_string(index=False))
    stats_trial.to_csv(f'{prefix}_rt_betas_stats.tsv', sep='\t', index=False)
    stats_bins.to_csv(f'{prefix}_bins_stats.tsv', sep='\t', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('step', choices=['collect', 'plot'])
    parser.add_argument('--bids-folder', type=Path, default=Path('/data/ds-retsupp'))
    parser.add_argument('--prefix', type=Path, default=Path('notes/data/trialwise_af'))
    parser.add_argument('--fig-dir', type=Path, default=Path('notes/figures'))
    args = parser.parse_args()
    if args.step == 'collect':
        collect(args.bids_folder, args.prefix)
    else:
        plot(args.prefix, args.fig_dir)
