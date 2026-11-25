"""Distractor-specific HP - LP term (dD) on the joint zero-sum main model, with jackknife SEs (update 4, item 4).

From the per-run sufficient statistics of ``fit_width_profiles`` on the main model (all else held at the joint fit),
the four gains dS (zero-sum direction), target, distractor at HP and distractor at LP are re-solved by least squares;
dD = distractor at HP - distractor at LP, one step from the main model (it has dD = 0), with a leave-one-run-out
jackknife SE and an odd-even calibration. Then the Student-t hierarchical model (gain_meta_zs.bayes) and own-data
per-area fits, and the Wilcoxon table.

    PYTHONPATH=. ~/mambaforge/envs/bauer/bin/python -m retsupp.visualize.paper.dD_meta_zs
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

from retsupp.visualize.paper.gain_meta_zs import NAMES, PW_OF, bayes, jk, wilcoxon_table
from retsupp.visualize.paper.paper_style import ROI_ORDER

M4 = np.array([[.75, 0, 0, 0], [-.25, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])   # (dS, T, D_HP, D_LP)
C = np.array([0, 0, 1, -1])


def collect(root='/data/ds-retsupp/derivatives/field_widths_zs_lm_2026-11-14', exclude=(1, 2)):
    rows = []
    for f in sorted(glob.glob(f'{root}/sub-*/*_widthprof.npz')):
        z = np.load(f, allow_pickle=True)
        s, roi = int(z['subject']), str(z['roi'])
        if s in exclude or roi not in ROI_ORDER:
            continue
        assert list(z['gain_names']) == NAMES
        A = np.einsum('ij,rjk,kl->ril', M4.T, z['A'], M4)
        b = np.einsum('ij,rj->ri', M4.T, z['b'])
        full, loo, _ = jk(A, b)
        R = len(A)
        dd, dd_loo = full @ C, loo @ C
        se = np.sqrt((R - 1) / R * ((dd_loo - dd_loo.mean()) ** 2).sum())
        odd = z['runs'][:, 1] % 2 == 1
        go = np.linalg.solve(A[odd].sum(0), b[odd].sum(0)) @ C
        ge = np.linalg.solve(A[~odd].sum(0), b[~odd].sum(0)) @ C
        D = (full[2] + full[3]) / 2
        rows.append(dict(subject=s, roi=roi, term='dD', y=100 * dd, se=100 * se, halfdiff=100 * abs(go - ge) / 2,
                         distractor=100 * D))
    t = pd.DataFrame(rows)
    t['pathway'] = t.roi.map(PW_OF)
    return t


if __name__ == '__main__':
    out = Path('notes/data/dD_meta_zs_2026-11-25')
    out.mkdir(parents=True, exist_ok=True)
    t = collect()
    t.to_csv(out / 'estimates.tsv', sep='\t', index=False)
    print('calibration', round(float(np.median(t.halfdiff / t.se)), 3))
    wilcoxon_table(t).to_csv(out / 'wilcoxon.tsv', sep='\t', index=False)
    s, dg = bayes(t)
    s.to_csv(out / 'bayes.tsv', sep='\t', index=False)
    print(s.round(3).to_string(), dg)
