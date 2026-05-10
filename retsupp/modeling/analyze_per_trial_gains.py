"""Stability + RT analysis of per-trial AF gains.

Reads the per-(subject, ROI) ``*-trials.tsv`` files produced by
:mod:`retsupp.modeling.fit_dog_dyn_v3_per_trial` and:

1. **Stability**: per (subject, ROI), split trials into halves, compute
   the split-half correlation of ``g_HP_dyn`` (HP-distractor trials only),
   ``g_LP_dyn`` (LP-distractor trials only), ``g_T_dyn`` (all trials).
   Threshold "stable" at corr > 0.5 (configurable).

2. **RT prediction**: for ROIs with stable gains, regress (per-subject)
   ``RT ~ g_T_dyn + g_HP_dyn + g_LP_dyn`` (linear OLS), report
   coefficients per subject × ROI.

3. **Figures**: 3 panels per ROI to ``notes/figures/single_trial_AF_gains_RT.pdf``:
   - distribution of g_T_dyn / g_HP_dyn / g_LP_dyn for one example sub
   - split-half stability scatter
   - β-RT swarm across subjects

Usage
-----
``python -m retsupp.modeling.analyze_per_trial_gains \\
   --bids-folder /data/ds-retsupp \\
   --out notes/figures/single_trial_AF_gains_RT.pdf``
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


SUBJECT_DERIV = ('af_prf_joint_dynamic_v3_dog_with_target_'
                 'sharedSigma_perTrial')
TAG = 'dog-dyn-v3-target-sharedSigma-perTrial'


def _trials_path(bids_folder: Path, subject: int, roi: str) -> Path:
    return (bids_folder / 'derivatives' / SUBJECT_DERIV
            / f'sub-{subject:02d}'
            / f'sub-{subject:02d}_roi-{roi}_mode-signed_{TAG}-trials.tsv')


def _split_half_corr(values: np.ndarray, seed: int = 42) -> float:
    """Random split-half Pearson correlation on a 1-D array."""
    if len(values) < 4:
        return float('nan')
    rng = np.random.default_rng(seed)
    idx = np.arange(len(values))
    rng.shuffle(idx)
    half = len(idx) // 2
    a = values[idx[:half]]
    b = values[idx[half:2 * half]]
    if a.std() == 0 or b.std() == 0:
        return float('nan')
    return float(np.corrcoef(a, b)[0, 1])


def _block_split_half_corr(values: np.ndarray, sessions: np.ndarray,
                           runs: np.ndarray) -> float:
    """Block split-half: alternate (session, run) groups into A/B sets,
    then correlate the trial means in each group across A vs B grouping
    samples — actually simpler: correlate even-numbered-trial vs
    odd-numbered-trial values within each (session, run), pooled.
    Returns float NaN on failure.
    """
    if len(values) < 4:
        return float('nan')
    # Pair adjacent trials within each run for a smooth split-half.
    a_vals, b_vals = [], []
    for sr in pd.unique(list(zip(sessions, runs))):
        s, r = sr
        mask = (sessions == s) & (runs == r)
        v = values[mask]
        n = len(v)
        if n < 4:
            continue
        a_vals.extend(v[::2])
        b_vals.extend(v[1::2])
    a = np.array(a_vals)
    b = np.array(b_vals)
    if len(a) == 0 or len(b) == 0:
        return float('nan')
    # Truncate to common length.
    n = min(len(a), len(b))
    a = a[:n]
    b = b[:n]
    if a.std() == 0 or b.std() == 0:
        return float('nan')
    return float(np.corrcoef(a, b)[0, 1])


def collect_stability(bids_folder: Path, subjects: list[int],
                       rois: list[str]) -> pd.DataFrame:
    """Return long-format dataframe with split-half corr per
    (subject, ROI, gain).
    """
    rows = []
    for s in subjects:
        for roi in rois:
            tsv = _trials_path(bids_folder, s, roi)
            if not tsv.exists():
                continue
            trials = pd.read_csv(tsv, sep='\t')
            if len(trials) < 8:
                continue
            for gain in ('g_HP_dyn', 'g_LP_dyn', 'g_T_dyn'):
                if gain not in trials.columns:
                    continue
                if gain == 'g_HP_dyn':
                    sub_trials = trials[trials['is_hp_distractor'] == 1.0]
                elif gain == 'g_LP_dyn':
                    sub_trials = trials[(trials['is_hp_distractor'] == 0.0)
                                         & (trials['trial_ring_idx'] >= 0)]
                else:
                    sub_trials = trials[trials['trial_target_ring_idx'] >= 0]
                if len(sub_trials) < 8:
                    continue
                vals = sub_trials[gain].values
                rng_corr = _split_half_corr(vals)
                blk_corr = _block_split_half_corr(
                    vals, sub_trials['session'].values,
                    sub_trials['run'].values,
                )
                rows.append(dict(
                    subject=s, roi=roi, gain=gain,
                    n=len(sub_trials),
                    mean=float(vals.mean()),
                    std=float(vals.std()),
                    split_half_random=rng_corr,
                    split_half_block=blk_corr,
                ))
    return pd.DataFrame(rows)


def collect_rt_betas(bids_folder: Path, subjects: list[int],
                      rois: list[str]) -> pd.DataFrame:
    """Per (subject, ROI), regress RT ~ g_T_dyn + g_dyn_active.

    Restricts to correct trials with valid RT. ``g_dyn_active`` is the
    HP gain on HP-distractor trials, the LP gain otherwise (one number
    per trial — the gain that was actually contributing to that
    trial's loss).
    """
    rows = []
    for s in subjects:
        for roi in rois:
            tsv = _trials_path(bids_folder, s, roi)
            if not tsv.exists():
                continue
            trials = pd.read_csv(tsv, sep='\t')
            if 'g_dyn_active' not in trials.columns:
                # Backward-compat: synthesise it.
                trials['g_dyn_active'] = np.where(
                    trials['is_hp_distractor'].astype(bool),
                    trials['g_HP_dyn'],
                    trials['g_LP_dyn'],
                )
            df = trials.dropna(subset=['rt']).copy()
            if 'correct' in df.columns:
                df = df[df['correct'].astype(str).str.lower().isin(['true', '1', '1.0'])]
            if len(df) < 30:
                continue
            X = df[['g_T_dyn', 'g_dyn_active']].values.astype(np.float64)
            X = np.column_stack([np.ones(len(X)), X])
            y = df['rt'].values.astype(np.float64)
            try:
                beta, *_ = np.linalg.lstsq(X, y, rcond=None)
                # Standardize for interpretable effect size.
                Xs = (X[:, 1:] - X[:, 1:].mean(0)) / (X[:, 1:].std(0) + 1e-12)
                Xs = np.column_stack([np.ones(len(Xs)), Xs])
                beta_z, *_ = np.linalg.lstsq(Xs, y, rcond=None)
            except Exception:
                continue
            rows.append(dict(
                subject=s, roi=roi, n_trials=len(df),
                beta_intercept=float(beta[0]),
                beta_g_T_dyn=float(beta[1]),
                beta_g_dyn_active=float(beta[2]),
                beta_g_T_dyn_z=float(beta_z[1]),
                beta_g_dyn_active_z=float(beta_z[2]),
            ))
    return pd.DataFrame(rows)


def make_figure(stability_df: pd.DataFrame, rt_df: pd.DataFrame,
                bids_folder: Path, example_subject: int = 2,
                example_roi: str = 'V1',
                out_path: Path = Path('notes/figures/single_trial_AF_gains_RT.pdf')):
    """Render the 3-panel figure."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # ---- Panel 1: distribution of per-trial gains for example sub/ROI. ----
    ax = axes[0]
    tsv = _trials_path(bids_folder, example_subject, example_roi)
    if tsv.exists():
        trials = pd.read_csv(tsv, sep='\t')
        long_rows = []
        for gain, sub_df, label in [
            ('g_HP_dyn', trials[trials['is_hp_distractor'] == 1.0], 'g_HP_dyn (HP trials)'),
            ('g_LP_dyn', trials[(trials['is_hp_distractor'] == 0.0)
                                 & (trials['trial_ring_idx'] >= 0)],
             'g_LP_dyn (LP trials)'),
            ('g_T_dyn',  trials[trials['trial_target_ring_idx'] >= 0],
             'g_T_dyn (all trials)'),
        ]:
            for v in sub_df[gain].values:
                long_rows.append(dict(gain=label, value=v))
        long_df = pd.DataFrame(long_rows)
        sns.violinplot(data=long_df, x='gain', y='value',
                       cut=0, ax=ax, inner='quartile')
        ax.axhline(0, color='k', lw=0.5, alpha=0.4)
        ax.set_title(f'sub-{example_subject:02d} {example_roi}: '
                     f'per-trial gain distributions')
        ax.set_ylabel('gain')
        ax.set_xlabel('')
        for label in ax.get_xticklabels():
            label.set_rotation(15)
    else:
        ax.text(0.5, 0.5, f'no data: sub-{example_subject:02d} {example_roi}',
                ha='center', va='center', transform=ax.transAxes)

    # ---- Panel 2: split-half corr per (ROI, gain). ----
    ax = axes[1]
    if len(stability_df) > 0:
        sns.barplot(data=stability_df, x='roi', y='split_half_block',
                    hue='gain', errorbar=('ci', 95), ax=ax)
        ax.axhline(0.5, color='red', ls='--', alpha=0.6,
                   label='stable (corr > 0.5)')
        ax.axhline(0, color='k', ls='-', lw=0.5, alpha=0.4)
        ax.set_title('Split-half stability (mean across subjects)')
        ax.set_ylabel('Split-half Pearson r')
        ax.set_xlabel('ROI')
        ax.legend(loc='upper right', fontsize='small')
    else:
        ax.text(0.5, 0.5, 'no stability data',
                ha='center', va='center', transform=ax.transAxes)

    # ---- Panel 3: RT-betas swarm across subjects. ----
    ax = axes[2]
    if len(rt_df) > 0:
        long_rt = pd.melt(
            rt_df,
            id_vars=['subject', 'roi'],
            value_vars=['beta_g_T_dyn_z', 'beta_g_dyn_active_z'],
            var_name='predictor', value_name='beta_z',
        )
        long_rt['predictor'] = long_rt['predictor'].map({
            'beta_g_T_dyn_z': 'g_T_dyn',
            'beta_g_dyn_active_z': 'g_dyn_active',
        })
        sns.swarmplot(data=long_rt, x='roi', y='beta_z', hue='predictor',
                       dodge=True, ax=ax, size=3)
        ax.axhline(0, color='k', lw=0.5, alpha=0.4)
        ax.set_title('RT regression coefficients (z-scored)')
        ax.set_ylabel('β (RT ~ gain)')
        ax.set_xlabel('ROI')
        ax.legend(loc='best', fontsize='small')
    else:
        ax.text(0.5, 0.5, 'no RT data',
                ha='center', va='center', transform=ax.transAxes)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {out_path}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bids-folder', default='/data/ds-retsupp')
    p.add_argument('--subjects', nargs='*', type=int, default=None,
                    help='Subset of subjects (default: all 28 available).')
    p.add_argument('--rois', nargs='*', default=None,
                    help='Subset of ROIs (default: all 8).')
    p.add_argument('--out', default='notes/figures/single_trial_AF_gains_RT.pdf')
    p.add_argument('--example-subject', type=int, default=2)
    p.add_argument('--example-roi', default='V1')
    p.add_argument('--write-stability-tsv',
                    default='notes/data/single_trial_stability.tsv')
    p.add_argument('--write-rt-tsv',
                    default='notes/data/single_trial_rt_betas.tsv')
    args = p.parse_args()

    bids_folder = Path(args.bids_folder)

    if args.subjects is None:
        # Default: all subjects with at least one trials.tsv.
        candidates = list(range(1, 31))
        # Skip sub-06, sub-08 per project memo.
        candidates = [s for s in candidates if s not in (6, 8)]
        args.subjects = candidates

    if args.rois is None:
        args.rois = ['V1', 'V2', 'V3', 'V3AB', 'hV4', 'LO', 'TO', 'VO']

    print(f'Subjects: {args.subjects}')
    print(f'ROIs:     {args.rois}')

    print('Collecting stability...')
    stab = collect_stability(bids_folder, args.subjects, args.rois)
    print(stab)

    print('Collecting RT betas...')
    rtb = collect_rt_betas(bids_folder, args.subjects, args.rois)
    print(rtb)

    Path(args.write_stability_tsv).parent.mkdir(parents=True, exist_ok=True)
    stab.to_csv(args.write_stability_tsv, sep='\t', index=False)
    print(f'Wrote {args.write_stability_tsv}')
    Path(args.write_rt_tsv).parent.mkdir(parents=True, exist_ok=True)
    rtb.to_csv(args.write_rt_tsv, sep='\t', index=False)
    print(f'Wrote {args.write_rt_tsv}')

    make_figure(stab, rtb, bids_folder,
                example_subject=args.example_subject,
                example_roi=args.example_roi,
                out_path=Path(args.out))


if __name__ == '__main__':
    main()
