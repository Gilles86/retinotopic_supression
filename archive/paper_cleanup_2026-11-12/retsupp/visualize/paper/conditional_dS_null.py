"""Primary test: conditional dS against the HP-rotation null; and split-half reliability of per-subject dS.

``collect`` (cluster): from the fixed-group sufficient statistics (``fit_fixed_group_mechanisms``), the conditional
estimate b / a per subject × ROI × mechanism (dS, dD) for the true labels (rotation 0,
``af_fixed_group_mechanisms_maxsd20``), the three relabellings (``..._hprot{1,2,3}``) and the two sessions
(``..._ses{1,2}``).

``test`` (local), as pre-specified in ``notes/primary_test_2026-11.md``: subject value = median over V1, V2, V3,
V3AB, IPS, hV4, VO, LO, TO; group statistic = median over subjects; null = every subject one of its four labellings
at random (100,000 draws); one-sided p = (1 + #draws ≤ observed) / (1 + draws). Also the split-half (session 1 vs
2, and odd vs even runs) correlation of the subject values, Spearman with a bootstrap 95% CI. The session split
confounds a session-wide gain change with dS (each session holds two HP locations; the m4 base is fitted on both),
which drives the two halves apart; odd and even runs each hold all four HP locations.

Usage::

    python -m retsupp.visualize.paper.conditional_dS_null collect --bids-folder /shares/zne.uzh/gdehol/ds-retsupp \\
        --out-dir /shares/zne.uzh/gdehol/ds-retsupp/derivatives/paper_2026-11
    python -m retsupp.visualize.paper.conditional_dS_null test
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROIS = ['V1', 'V2', 'V3', 'V3AB', 'IPS', 'hV4', 'VO', 'LO', 'TO']
REAL = 'af_fixed_group_mechanisms_maxsd20'


def estimates(bids, folder):
    files = sorted(glob.glob(f'{bids}/derivatives/{folder}/sub-*/*_stats.tsv.gz'))
    st = pd.concat([pd.read_csv(f, sep='\t', usecols=['subject', 'roi', 'mech', 'a', 'b', 'lin_err']) for f in files])
    st = st[st.mech.isin(['dS', 'dD'])]
    own = st.groupby(['subject', 'roi', 'mech']).agg(a=('a', 'sum'), b=('b', 'sum'), lin_err=('lin_err', 'first')).reset_index()
    own['estimate'] = own.b / own.a
    own['valid'] = own.lin_err <= 1e-3
    return own.drop(columns=['a', 'b'])


def collect(bids, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rot = pd.concat([estimates(bids, REAL if k == 0 else f'{REAL}_hprot{k}').assign(rotation=k) for k in range(4)])
    rot.to_csv(out_dir / 'conditional_dS_rotation_null_2026-11.tsv', sep='\t', index=False)
    ses = pd.concat([estimates(bids, f'{REAL}_ses{s}').assign(session=s) for s in (1, 2)])
    ses.to_csv(out_dir / 'conditional_dS_sessions_2026-11.tsv', sep='\t', index=False)
    if Path(f'{bids}/derivatives/{REAL}_odd').exists():
        par = pd.concat([estimates(bids, f'{REAL}_{h}').assign(half=i) for i, h in enumerate(('odd', 'even'), 1)])
        par.to_csv(out_dir / 'conditional_dS_oddeven_2026-11.tsv', sep='\t', index=False)
    print(rot.groupby(['rotation', 'mech']).size().unstack(), ses.groupby(['session', 'mech']).size().unstack(), sep='\n')


def subject_values(d, mech, by):
    d = d[(d.mech == mech) & d.valid & d.roi.isin(ROIS)]
    return d.groupby(['subject', by]).estimate.median().unstack(by)


def split_half(table, by, label, rng):
    d = pd.read_csv(table, sep='\t')
    for mech in ('dS', 'dD'):
        v = subject_values(d, mech, by).dropna()
        x = v.to_numpy()
        r = stats.spearmanr(x[:, 0], x[:, 1]).statistic
        boots = [stats.spearmanr(*x[rng.integers(0, len(x), len(x))].T).statistic for _ in range(10_000)]
        lo, hi = np.nanpercentile(boots, [2.5, 97.5])
        print(f'{mech} split-half ({label}): n = {len(v)}, Spearman ρ = {r:+.3f} [{lo:+.3f}, {hi:+.3f}], '
              f'Pearson r = {stats.pearsonr(x[:, 0], x[:, 1]).statistic:+.3f}; half medians {np.median(x[:, 0]):+.4f}, '
              f'{np.median(x[:, 1]):+.4f}')


def test(rot_table, ses_table, n_draws=100_000, seed=20261107, oddeven_table='notes/data/conditional_dS_oddeven_2026-11.tsv'):
    rot = pd.read_csv(rot_table, sep='\t')
    rng = np.random.default_rng(seed)
    for mech in ('dS', 'dD'):
        v = subject_values(rot, mech, 'rotation').dropna()
        obs = v[0].median()
        pick = rng.integers(0, 4, size=(n_draws, len(v)))
        null = np.median(v.to_numpy()[np.arange(len(v)), pick], axis=1)
        p = (1 + (null <= obs).sum()) / (1 + n_draws)
        print(f'{mech}: n = {len(v)}, observed median {obs:+.4f}; null median {np.median(null):+.4f} '
              f'[2.5%, 97.5%: {np.percentile(null, 2.5):+.4f}, {np.percentile(null, 97.5):+.4f}]; one-sided p = {p:.4g}; '
              f'mean of rotated subject values {v[[1, 2, 3]].to_numpy().mean():+.4f}; '
              f'rotations 1-3 group medians {v[1].median():+.4f}, {v[2].median():+.4f}, {v[3].median():+.4f}')
    split_half(ses_table, 'session', 'session 1 vs 2', rng)
    if Path(oddeven_table).exists():
        split_half(oddeven_table, 'half', 'odd vs even runs', rng)

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('step', choices=['collect', 'test'])
    ap.add_argument('--bids-folder', default='/shares/zne.uzh/gdehol/ds-retsupp')
    ap.add_argument('--out-dir', default='/shares/zne.uzh/gdehol/ds-retsupp/derivatives/paper_2026-11')
    ap.add_argument('--rot-table', default='notes/data/conditional_dS_rotation_null_2026-11.tsv')
    ap.add_argument('--ses-table', default='notes/data/conditional_dS_sessions_2026-11.tsv')
    a = ap.parse_args()
    if a.step == 'collect':
        collect(a.bids_folder, a.out_dir)
    else:
        test(a.rot_table, a.ses_table)
