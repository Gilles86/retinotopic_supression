"""Relate trial-wise / RT-binned / run-wise phasic AF gains to behaviour.

Two steps, split along the data-size boundary:

``collect`` (cluster): reads ``derivatives/af_trialwise_gains/sub-*/``
(from :mod:`retsupp.modeling.fit_trialwise_af_gains`) and writes small
TSVs:

- ``<prefix>_rt_betas.tsv`` — per (subject, ROI, α, RT transform):
  within-subject OLS of z(RT) on z(per-trial gain) plus inter-trial
  priming covariates. RT and gains are z-scored within (session, run,
  trial type), so trial-type RT effects and slow drifts are removed and
  only trial-to-trial covariation remains. Gain predictors: ``T``
  (target, all trials), ``D_HP`` / ``D_LP`` (distractor gain on HP / LP
  trials, 0 elsewhere — separate slopes, so ``D_LP − D_HP`` tests
  whether distractor processing costs more where it isn't suppressed)
  and ``G`` (global). Covariates (cell-demeaned): distractor repeat,
  target repeat, target at the previous trial's distractor location.
  RT transforms: ``log`` (log RT), ``inv`` (−1/RT), ``rank``.
- ``<prefix>_run_behavior.tsv`` — per (subject, session, run): mean log
  RT per trial type, HP-block position, and the two behavioural
  contrasts ``benefit`` = logRT_LP − logRT_HP and ``capture`` =
  logRT_distractor − logRT_none.
- ``<prefix>_cv.tsv``, ``<prefix>_bins.tsv``, ``<prefix>_check.tsv`` —
  concatenations of the per-(subject, ROI) files.

``plot`` (local): group statistics (FDR across ROIs per measure) and
four figures: per-trial, RT bins, run-level, inter-trial history.

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
from statsmodels.stats.multitest import multipletests

ROI_ORDER = ['V1', 'V2', 'V3', 'V3AB', 'hV4', 'LO', 'TO', 'VO', 'IPS', 'SPL1', 'FEF']
KIND_PALETTE = {'T': '#F77F00', 'D_HP': '#D62828', 'D_LP': '#0077B6',
                'G': '#9AA5B1'}
KIND_LABEL = {'T': 'Target', 'D_HP': 'HP distractor', 'D_LP': 'LP distractor',
              'G': 'Global'}
COVARIATES = ['dist_repeat', 'target_repeat', 'target_at_prev_dist']
TRANSFORMS = {'log': np.log, 'inv': lambda x: -1.0 / x,
              'rank': lambda x: x.rank()}
MAIN_ALPHA = 1.0
CELL = ['session', 'run', 'trial_type']


def _z(x):
    sd = x.std()
    return (x - x.mean()) / sd if sd > 0 else x * 0.0


def add_priming(trials: pd.DataFrame) -> pd.DataFrame:
    """Inter-trial priming covariates, from the previous trial in the run."""
    # One row per trial (the trials file repeats each trial per α).
    u = (trials.drop_duplicates('trial_idx')
         .sort_values(['session', 'run', 'onset']).copy())
    prev = u.groupby(['session', 'run'])[['dist_ch', 'target_ch']].shift(1)
    d, t = u['dist_ch'], u['target_ch']
    u['dist_repeat'] = ((d >= 0) & (d == prev['dist_ch'])).astype(float)
    u['target_repeat'] = ((t >= 0) & (t == prev['target_ch'])).astype(float)
    u['target_at_prev_dist'] = ((t >= 0) & (t == prev['dist_ch'])).astype(float)
    return trials.merge(u[['trial_idx', *COVARIATES]], on='trial_idx')


def rt_betas(trials: pd.DataFrame) -> pd.DataFrame:
    trials = add_priming(trials)
    rows = []
    for (subject, roi, alpha), df in trials.groupby(['subject', 'roi', 'alpha']):
        df = df[df['valid_rt']].copy()
        X = {}
        for kind in ('T', 'G'):
            X[kind] = df.groupby(CELL)[f'g_{kind}'].transform(_z)
        for tt in ('HP', 'LP'):
            col = df.groupby(CELL)[f'g_D_{tt}'].transform(_z)
            X[f'D_{tt}'] = col.where(df['trial_type'] == tt, 0.0).fillna(0.0)
        for cov in COVARIATES:
            X[cov] = df[cov] - df.groupby(CELL)[cov].transform('mean')
        X = pd.DataFrame(X)
        Xm = np.column_stack([np.ones(len(X)), X.values])
        for tname, f in TRANSFORMS.items():
            y = df.groupby(CELL)['rt'].transform(lambda x: _z(f(x)))
            beta, *_ = np.linalg.lstsq(Xm, y.values, rcond=None)
            for kind, b in zip(X.columns, beta[1:]):
                rows.append(dict(subject=subject, roi=roi, alpha=alpha,
                                 transform=tname, kind=kind, beta=b,
                                 n_trials=len(df)))
    return pd.DataFrame(rows)


def run_behavior(trials: pd.DataFrame) -> pd.DataFrame:
    """Per-run behavioural contrasts + position within the HP block."""
    trials = trials.drop_duplicates('trial_idx')
    runs = (trials[['session', 'run', 'hp_location']].drop_duplicates()
            .sort_values(['session', 'run']).reset_index(drop=True))
    new_block = (runs['hp_location'] != runs.groupby('session')['hp_location']
                 .shift(1))
    runs['block_pos'] = runs.groupby(
        ['session', new_block.groupby(runs['session']).cumsum()]).cumcount()
    v = trials[trials['valid_rt']].assign(logrt=lambda d: np.log(d['rt']))
    m = v.pivot_table(index=['session', 'run'], columns='trial_type',
                      values='logrt', aggfunc='mean')
    m['dist'] = v[v['trial_type'] != 'none'].groupby(['session', 'run'])['logrt'].mean()
    m = m.rename(columns=lambda c: f'logrt_{c}').reset_index()
    out = runs.merge(m, on=['session', 'run'])
    out['benefit'] = out['logrt_LP'] - out['logrt_HP']
    out['capture'] = out['logrt_dist'] - out['logrt_none']
    out.insert(0, 'subject', trials['subject'].iloc[0])
    return out


def collect(bids_folder: Path, prefix: Path):
    root = bids_folder / 'derivatives' / 'af_trialwise_gains'
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for name in ('cv', 'bins', 'check'):
        files = sorted(root.glob(f'sub-*/sub-*_roi-*_{name}.tsv'))
        df = pd.concat([pd.read_csv(f, sep='\t') for f in files])
        df.to_csv(f'{prefix}_{name}.tsv', sep='\t', index=False)
        print(f'{name}: {len(files)} files -> {prefix}_{name}.tsv')
    betas, behav = [], []
    for f in sorted(root.glob('sub-*/sub-*_roi-*_trials.tsv')):
        trials = pd.read_csv(f, sep='\t')
        betas.append(rt_betas(trials))
        if '_roi-V1_' in f.name:
            behav.append(run_behavior(trials))
        print(f.name, flush=True)
    pd.concat(betas).to_csv(f'{prefix}_rt_betas.tsv', sep='\t', index=False)
    pd.concat(behav).to_csv(f'{prefix}_run_behavior.tsv', sep='\t', index=False)
    print(f'-> {prefix}_rt_betas.tsv, {prefix}_run_behavior.tsv')


def group_t(df, value, by, fdr_within=None):
    """One-sample t vs 0 per group; BH-FDR across ROIs within
    ``fdr_within`` (default: everything in ``by`` except 'roi')."""
    out = []
    for key, d in df.groupby(by):
        x = d[value].dropna()
        t, p = stats.ttest_1samp(x, 0.0)
        out.append(dict(zip(by, key if isinstance(key, tuple) else (key,)),
                        mean=x.mean(), sem=x.sem(), t=t, p=p, n=len(x)))
    out = pd.DataFrame(out)
    fam = fdr_within if fdr_within is not None else [b for b in by if b != 'roi']
    out['q'] = (out.groupby(fam)['p'].transform(
        lambda p: multipletests(p, method='fdr_bh')[1]) if fam
        else multipletests(out['p'], method='fdr_bh')[1])
    return out


def _within_corr(df, x, y, by):
    """Within-subject Pearson r (Fisher z) across runs, per (subject, *by)."""
    rows = []
    for key, d in df.groupby(['subject', *by]):
        d = d[[x, y]].dropna()
        if len(d) < 5 or d[x].std() == 0 or d[y].std() == 0:
            continue
        r = np.corrcoef(d[x], d[y])[0, 1]
        rows.append(dict(zip(['subject', *by], key), r=r,
                         z=np.arctanh(np.clip(r, -0.999, 0.999))))
    return pd.DataFrame(rows)


def _pointplot(ax, df, y, hue, order, palette, hue_order=None):
    sns.pointplot(df, x='roi', y=y, hue=hue, order=order, hue_order=hue_order,
                  palette=palette,
                  dodge=0.5 if df[hue].nunique() > 1 else False,
                  linestyle='none',
                  errorbar=('ci', 95), markersize=3, ax=ax)
    ax.axhline(0, color='#1F2933', lw=0.8)
    ax.tick_params(axis='x', rotation=90)
    ax.set_xlabel('')


def _print(title, df, cols):
    print(f'\n{title}')
    print(df.sort_values('p')[cols].head(15).round(4).to_string(index=False))


def plot(prefix: Path, fig_dir: Path):
    sns.set_theme(style='ticks', context='paper')
    fig_dir.mkdir(parents=True, exist_ok=True)
    cv = pd.read_csv(f'{prefix}_cv.tsv', sep='\t')
    betas = pd.read_csv(f'{prefix}_rt_betas.tsv', sep='\t')
    bins = pd.read_csv(f'{prefix}_bins.tsv', sep='\t')
    check = pd.read_csv(f'{prefix}_check.tsv', sep='\t')
    behav = pd.read_csv(f'{prefix}_run_behavior.tsv', sep='\t')
    rois = [r for r in ROI_ORDER if r in set(check['roi'])]
    cols = ['roi', 'kind', 'mean', 't', 'p', 'q']

    print('Sanity: max phasic recon err',
          check['phasic_recon_rel_err'].max(), '| max |ΔR²|',
          (check['r2_recomputed'] - check['r2_stored']).abs().max())

    # ------------------------------------------------------------------
    # Figure 1: per-trial gains → RT (priming covariates in the model)
    # ------------------------------------------------------------------
    b = betas[(betas['alpha'] == MAIN_ALPHA)]
    gains = b[b['kind'].isin(list(KIND_PALETTE))]
    st = group_t(gains, 'beta', ['transform', 'roi', 'kind'])
    contrast = (b[b['kind'].isin(['D_HP', 'D_LP'])]
                .pivot_table(index=['subject', 'roi', 'transform'],
                             columns='kind', values='beta').reset_index())
    contrast['LP_minus_HP'] = contrast['D_LP'] - contrast['D_HP']
    st_c = group_t(contrast, 'LP_minus_HP', ['transform', 'roi'])
    st_cov = group_t(b[b['kind'].isin(COVARIATES)], 'beta',
                     ['transform', 'roi', 'kind'])

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4),
                             gridspec_kw=dict(width_ratios=[1, 1.6, 1]))
    ax = axes[0]
    sns.lineplot(cv, x='alpha', y='heldout_gain', hue='roi', hue_order=rois,
                 palette='viridis', errorbar=None, ax=ax, legend=False)
    ax.axhline(0, color='#1F2933', lw=0.8)
    ax.set(xscale='log', ylim=(-0.05, None),
           xlabel='Ridge α (shrinkage to scalar gains)',
           ylabel='Held-out ΔSSE (fraction)',
           title='Per-trial gains vs scalar model')
    ax = axes[1]
    _pointplot(ax, gains[gains['transform'] == 'log'], 'beta', 'kind', rois,
               KIND_PALETTE, hue_order=list(KIND_PALETTE))
    ax.set(ylabel='β: z(log RT) ~ z(trial gain)',
           title='Trial-by-trial gain → RT (priming-controlled)')
    h, l = ax.get_legend_handles_labels()
    ax.legend(h, [KIND_LABEL[x] for x in l], frameon=False, fontsize=7)
    ax = axes[2]
    _pointplot(ax, contrast[contrast['transform'] == 'log'].assign(k='c'),
               'LP_minus_HP', 'k', rois, {'c': '#1F2933'})
    ax.get_legend().remove()
    ax.set(ylabel='β(LP distractor) − β(HP distractor)',
           title='Distractor cost: LP vs HP slope')
    sns.despine(fig)
    fig.tight_layout()
    fig.savefig(fig_dir / 'trialwise_af_gains_rt.pdf')

    _print('Per-trial gain betas, log RT (q: FDR across ROIs per gain):',
           st[st['transform'] == 'log'], cols)
    print('\nRobustness across RT transforms (t):')
    print(st.pivot_table(index=['roi', 'kind'], columns='transform',
                         values='t').round(1)
          .loc[lambda d: (d.abs() > 2.5).any(axis=1)].to_string())
    _print('LP − HP distractor slope (FDR across ROIs):',
           st_c[st_c['transform'] == 'log'], ['roi', 'mean', 't', 'p', 'q'])
    print('\nPriming covariates, log RT (mean β across ROIs; t of ROI-mean):')
    cov = b[(b['kind'].isin(COVARIATES)) & (b['transform'] == 'log')]
    cm = cov.groupby(['subject', 'kind'])['beta'].mean().reset_index()
    print(group_t(cm.assign(roi='all'), 'beta', ['kind'],
                  fdr_within=[])[['kind', 'mean', 't', 'p']].round(4)
          .to_string(index=False))

    # ------------------------------------------------------------------
    # Figure 2: RT-binned gains (slow − fast)
    # ------------------------------------------------------------------
    rb = bins[bins['scheme'].isin(['rt2', 'rt3'])]
    piv = rb[rb['bin'].isin(['fast', 'slow'])].pivot_table(
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
    g.savefig(fig_dir / 'trialwise_af_gains_rt_bins.pdf')
    st_b = group_t(piv, 'slow_minus_fast',
                   ['scheme', 'with_global', 'roi', 'kind'])
    _print('Binned slow − fast, rt2 + global (FDR across ROIs per gain):',
           st_b[(st_b['scheme'] == 'rt2') & st_b['with_global']], cols)

    # ------------------------------------------------------------------
    # Figure 3: run-level brain–behaviour
    # ------------------------------------------------------------------
    rg = bins[bins['scheme'] == 'run'].copy()
    rg['session'] = rg['bin'].str.extract(r'ses-(\d)').astype(int)
    rg['run'] = rg['bin'].str.extract(r'run-(\d)').astype(int)
    rg = rg.pivot_table(index=['subject', 'roi', 'with_global', 'session', 'run'],
                        columns='kind', values='g').reset_index()
    rg['supp_index'] = rg['D_LP'] - rg['D_HP']
    rg['D_mean'] = rg[['D_HP', 'D_LP']].mean(axis=1)
    run = rg.merge(behav, on=['subject', 'session', 'run'])
    # Remove session means so scan-day offsets can't drive correlations.
    for c in ['supp_index', 'D_mean', 'D_HP', 'D_LP', 'T', 'G',
              'benefit', 'capture', 'block_pos']:
        run[c] = run[c] - run.groupby(
            ['subject', 'roi', 'with_global', 'session'])[c].transform('mean')

    tests = [('supp_index', 'benefit', 'Suppression index (LP−HP gain)\n'
              '× RT benefit (LP−HP)'),
             ('D_mean', 'capture', 'Distractor gain\n× capture cost'),
             ('supp_index', 'block_pos', 'Suppression index\n× block position'),
             ('benefit', 'block_pos', 'RT benefit\n× block position')]
    fig, axes = plt.subplots(2, len(tests), figsize=(13, 5.5), sharey='row')
    run_stats = []
    for row, wg in enumerate([True, False]):
        sub = run[run['with_global'] == wg]
        for col, (x, y, title) in enumerate(tests):
            if x == 'benefit':
                zz = _within_corr(sub[sub['roi'] == rois[0]], x, y, [])
                zz['roi'] = 'behaviour'
                order = ['behaviour']
            else:
                zz = _within_corr(sub, x, y, ['roi'])
                order = rois
            ax = axes[row, col]
            _pointplot(ax, zz.assign(k='c'), 'z', 'k', order, {'c': '#D62828'})
            ax.get_legend().remove()
            ax.set(title=title if row == 0 else '',
                   ylabel=(f'Within-subject Fisher z\n(global regressor: {wg})'
                           if col == 0 else ''))
            s = group_t(zz, 'z', ['roi'], fdr_within=[])
            run_stats.append(s.assign(test=f'{x}~{y}', with_global=wg))
    sns.despine(fig)
    fig.tight_layout()
    fig.savefig(fig_dir / 'trialwise_af_gains_runlevel.pdf')
    run_stats = pd.concat(run_stats)
    for test, d in run_stats[run_stats['with_global']].groupby('test'):
        _print(f'Run-level {test} (global regressor in; FDR across ROIs):',
               d, ['roi', 'mean', 't', 'p', 'q'])

    # ------------------------------------------------------------------
    # Figure 4: inter-trial history → gains
    # ------------------------------------------------------------------
    def contrasts(scheme, kinds, pairs):
        d = bins[(bins['scheme'] == scheme) & bins['kind'].isin(kinds)]
        w = d.pivot_table(index=['subject', 'roi', 'with_global', 'kind'],
                          columns='bin', values='g').reset_index()
        out = []
        for a, b_ in pairs:
            out.append(w.assign(contrast=f'{a} − {b_}', value=w[a] - w[b_])
                       [['subject', 'roi', 'with_global', 'kind', 'contrast',
                         'value']])
        return pd.concat(out)

    hist = pd.concat([
        contrasts('dprime', ['D_HP', 'D_LP'], [('repeat', 'switch')]),
        contrasts('tprime', ['T'], [('at_prev_dist', 'other'),
                                    ('repeat', 'other')]),
        bins[(bins['scheme'] == 'all') & (bins['kind'] == 'P')]
        .assign(contrast='vs 0', value=lambda d: d['g'])
        [['subject', 'roi', 'with_global', 'kind', 'contrast', 'value']],
    ])
    hist['measure'] = hist['kind'] + ': ' + hist['contrast']
    measures = ['D_HP: repeat − switch', 'D_LP: repeat − switch',
                'T: at_prev_dist − other', 'T: repeat − other', 'P: vs 0']
    titles = ['HP distractor gain:\nrepeat − switch location',
              'LP distractor gain:\nrepeat − switch location',
              'Target gain: at previous\ndistractor loc. − other',
              'Target gain:\nrepeat − other',
              'Gain at previous distractor\nlocation (now neutral)']
    colors = [KIND_PALETTE['D_HP'], KIND_PALETTE['D_LP'], KIND_PALETTE['T'],
              KIND_PALETTE['T'], '#2A9D8F']
    fig, axes = plt.subplots(1, len(measures), figsize=(15, 3.2), sharey=False)
    hg = hist[hist['with_global']]
    for ax, m, title, c in zip(axes, measures, titles, colors):
        _pointplot(ax, hg[hg['measure'] == m].assign(k='c'), 'value', 'k',
                   rois, {'c': c})
        ax.get_legend().remove()
        ax.set(title=title, ylabel='Gain difference' if m != 'P: vs 0'
               else 'Gain')
    sns.despine(fig)
    fig.tight_layout()
    fig.savefig(fig_dir / 'trialwise_af_gains_intertrial.pdf')
    st_h = group_t(hist, 'value', ['with_global', 'measure', 'roi'])
    for m in measures:
        _print(f'Inter-trial {m} (global regressor in; FDR across ROIs):',
               st_h[st_h['with_global'] & (st_h['measure'] == m)],
               ['roi', 'mean', 't', 'p', 'q'])
    st_h.to_csv(f'{prefix}_intertrial_stats.tsv', sep='\t', index=False)

    st.to_csv(f'{prefix}_rt_betas_stats.tsv', sep='\t', index=False)
    st_c.to_csv(f'{prefix}_lp_hp_slope_stats.tsv', sep='\t', index=False)
    st_cov.to_csv(f'{prefix}_priming_stats.tsv', sep='\t', index=False)
    st_b.to_csv(f'{prefix}_bins_stats.tsv', sep='\t', index=False)
    run_stats.to_csv(f'{prefix}_runlevel_stats.tsv', sep='\t', index=False)
    print(f'\nFigures in {fig_dir}/trialwise_af_gains_'
          '{rt,rt_bins,runlevel,intertrial}.pdf')


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
