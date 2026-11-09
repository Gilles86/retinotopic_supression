"""Collect the sustained gains of m8_full_adjopp (LP gain split next to HP vs opposite) into one TSV.

One row per subject x ROI (``--cv-scheme full`` fits: one fold). Columns: the fitted terms (g_HP, g_LP =
mean of the three LP gains, g_adj_minus_opp), the gains they imply (g_adj, g_opp), the contrasts in the
order of the decoding (HP − next, next − opposite, HP − opposite, and dS = g_HP − g_LP), m7_full's dS for
comparison, σ_AF, and the training SSE at the m7 start and at the end.

Reads many small pickles, so run it on a compute node::

    srun -A hare.econ.uzh -c1 --mem 4G --time 10 $HOME/data/conda/envs/retsupp_cuda/bin/python \\
        -m retsupp.modeling.collect_lp_split \\
        --root /shares/zne.uzh/gdehol/ds-retsupp/derivatives/af_prf_cv_ladder_nested_full_bounded_maxsd20_reliable_roihrf \\
        --out notes/data/joint_adjopp_dog_2026-11.tsv
"""
from __future__ import annotations

import argparse
import pickle
import re
from pathlib import Path

import pandas as pd

from retsupp.modeling.lp_split import NEW

MODEL = 'm8_full_adjopp'


def shared_row(pkl):
    with open(pkl, 'rb') as f:
        d = pickle.load(f)
    folds = [p for p in d['fit_pars_per_fold'] if p is not None]
    if len(folds) != 1:
        raise ValueError(f'{pkl}: expected one fold (cv-scheme full), got {len(folds)}')
    info = [i for i in d['optimizer_info'] if i is not None][0]
    return folds[0].iloc[0], info, len(d['voxel_ids'])


def main(root, out, m7_root=None):
    root = Path(root)
    m7_root = Path(m7_root) if m7_root else root
    rows = []
    for pkl in sorted((root / MODEL).glob('sub-*/sub-*_roi-*_cv-fits.pkl')):
        subject, roi = re.match(r'sub-(\d+)_roi-(.+)_cv-fits\.pkl', pkl.name).groups()
        p, info, n_vox = shared_row(pkl)
        m7, _, _ = shared_row(m7_root / 'm7_full' / pkl.parent.name / pkl.name)
        g_hp, g_lp, delta = float(p['g_HP']), float(p['g_LP']), float(p[NEW])
        g_adj, g_opp = g_lp + delta / 3, g_lp - 2 * delta / 3
        rows.append(dict(
            subject=int(subject), roi=roi, n_voxels=n_vox,
            g_HP=g_hp, g_LP=g_lp, g_adj=g_adj, g_opp=g_opp, g_adj_minus_opp=delta,
            HP_minus_adj=g_hp - g_adj, HP_minus_opp=g_hp - g_opp, dS=g_hp - g_lp,
            dS_m7=float(m7['g_HP'] - m7['g_LP']), sigma_AF=float(p['sigma_AF']),
            sse_m7_start=info['sse_parent_start'], sse_final=info['sse_final']))
    df = pd.DataFrame(rows)
    if df.empty:
        raise FileNotFoundError(f'no {MODEL} fits under {root}')
    df['sse_drop'] = df.sse_m7_start - df.sse_final
    df.to_csv(out, sep='\t', index=False, float_format='%.7g')
    print(f'{len(df)} subject x ROI rows -> {out}')
    print(df.groupby('roi')[['HP_minus_adj', 'g_adj_minus_opp', 'HP_minus_opp', 'dS']].mean().round(4))
    if (df.sse_drop < -1e-6 * df.sse_m7_start).any():
        print('WARNING: training SSE above the m7 start in', df.loc[df.sse_drop < 0, ['subject', 'roi']].values.tolist())


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--m7-root', default=None, help='Root holding m7_full (default: --root).')
    a = ap.parse_args()
    main(a.root, a.out, a.m7_root)
