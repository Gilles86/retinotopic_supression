"""Compare m7_full_sigtied (one width, σ_dyn := σ_AF) with m7_full: one row per subject x ROI x fold.

Columns: both models' widths and gains (with dS, dD, T, D as in ``fig45_common.joint``), training SSE of
both, the relative training-SSE cost of tying, and per fold the mean over voxels of training and held-out
R² of both models (held-out is NaN for ``--cv-scheme full``). The held-out difference
``d_cv_r2 = cv_r2_tied − cv_r2_m7`` is the test: below zero means the second width predicts new runs
better.

Reads many small text files, so run it on a compute node::

    srun -A hare.econ.uzh -c1 --mem 4G --time 10 $HOME/data/conda/envs/retsupp_cuda/bin/python \\
        -m retsupp.modeling.collect_sigma_tied --root <ladder root> --out notes/data/sigma_tied_<scheme>_2026-11.tsv
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

TIED, PARENT = 'm7_full_sigtied', 'm7_full'
SHARED = ['sigma_AF', 'sigma_dyn', 'g_HP', 'g_LP', 'g_HP_dyn', 'g_LP_dyn', 'g_T_dyn', 'item_scale']


def read(root, model, sub_dir, stem):
    d = root / model / sub_dir
    meta = json.loads((d / f'{stem}_meta.json').read_text())
    pars = pd.read_csv(d / f'{stem}_cv-params.tsv', sep='\t').groupby('fold')[SHARED].first()
    r2 = pd.read_csv(d / f'{stem}_cv-r2.tsv', sep='\t').groupby('fold')[['train_r2', 'cv_r2']].mean()
    sse = pd.Series({k: (i or {}).get('train_sse_total', np.nan) for k, i in enumerate(meta['optimizer_info'])},
                    name='sse')
    return pars.join(r2).join(sse), meta['n_vox']


def main(root, out, parent_root=None):
    root = Path(root)
    parent_root = Path(parent_root) if parent_root else root
    rows = []
    for meta in sorted((root / TIED).glob('sub-*/sub-*_roi-*_meta.json')):
        stem = meta.name[:-len('_meta.json')]
        subject, roi = re.match(r'sub-(\d+)_roi-(.+)', stem).groups()
        t, n_vox = read(root, TIED, meta.parent.name, stem)
        p, _ = read(parent_root, PARENT, meta.parent.name, stem)
        both = t.add_suffix('_tied').join(p.add_suffix('_m7'))
        both.insert(0, 'n_vox', n_vox)
        both.insert(0, 'roi', roi)
        both.insert(0, 'subject', int(subject))
        rows.append(both.reset_index())
    df = pd.concat(rows, ignore_index=True)
    for s in ('_tied', '_m7'):
        df['dS' + s] = df['g_HP' + s] - df['g_LP' + s]
        df['dD' + s] = df['g_HP_dyn' + s] - df['g_LP_dyn' + s]
        df['D' + s] = (df['g_HP_dyn' + s] + 3 * df['g_LP_dyn' + s]) / 4
    df['rel_sse_cost'] = (df.sse_tied - df.sse_m7) / df.sse_m7
    df['d_train_r2'] = df.train_r2_tied - df.train_r2_m7
    df['d_cv_r2'] = df.cv_r2_tied - df.cv_r2_m7
    df.to_csv(out, sep='\t', index=False, float_format='%.7g')
    print(f'{len(df)} subject x ROI x fold rows -> {out}')
    g = df.groupby(['subject', 'roi'])[['rel_sse_cost', 'd_train_r2', 'd_cv_r2']].mean()
    print(g.groupby('roi').median().round(6))
    if (df.rel_sse_cost < -1e-6).any():
        print('NOTE: tied below m7 on training SSE (m7 not at its optimum) in',
              df.loc[df.rel_sse_cost < -1e-6, ['subject', 'roi', 'fold']].values.tolist())


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--parent-root', default=None, help='Root holding m7_full (default: --root).')
    a = ap.parse_args()
    main(a.root, a.out, a.parent_root)
