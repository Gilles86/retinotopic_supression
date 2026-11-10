"""Shared gain-field parameters of every ladder model, per family, subject and ROI (one row each).

Reads ``<root>/<model>/sub-XX/sub-XX_roi-<ROI>_cv-params.tsv`` (full-data fits, one fold) and keeps the first row's
shared parameters (the gains, widths and item scale are the same on every voxel row) and the number of voxels.
Output columns as ``notes/data/gain_field_parameters.tsv`` (family, model, subject, roi, n_vox, gains, widths,
item_scale, outlier_free) plus any split term a model adds (g_T_HP_dyn, g_adj_minus_opp).

Run on a compute node (thousands of small files)::

    python -m retsupp.visualize.paper.collect_gain_parameters \\
        --root dog=af_prf_cv_ladder_nested_full_bounded_maxsd20_reliable_roihrf \\
        --root gauss=af_prf_cv_ladder_nested_full_bounded_maxsd20_gauss_reliable_roihrf \\
        --root dn=af_prf_cv_ladder_nested_full_bounded_maxsd20_dn_reliable_roihrf \\
        --out <tsv>
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import pandas as pd

SHARED = ['g_HP', 'g_LP', 'g_HP_dyn', 'g_LP_dyn', 'g_T_dyn', 'sigma_AF', 'sigma_dyn', 'sigma_T_dyn', 'item_scale',
          'g_T_HP_dyn', 'g_adj_minus_opp', 'neural_baseline', 'surround_baseline']


def main(bids, roots, out):
    rows = []
    for spec in roots:
        family, root = spec.split('=', 1)
        base = Path(bids) / 'derivatives' / root
        for mdir in sorted(p for p in base.iterdir() if p.is_dir()):
            for f in sorted(glob.glob(str(mdir / 'sub-*' / '*_cv-params.tsv'))):
                d = pd.read_csv(f, sep='\t')
                r = d.iloc[0]
                row = dict(family=family, model=mdir.name, subject=int(r.subject), roi=r.roi, n_vox=len(d))
                for c in SHARED:
                    if c in d and c not in ('neural_baseline', 'surround_baseline'):
                        row[c] = float(r[c])
                if 'neural_baseline' in d:      # per voxel in DN; keep the medians for reference
                    row['neural_baseline_median'] = float(d.neural_baseline.median())
                    row['surround_baseline_median'] = float(d.surround_baseline.median())
                row['outlier_free'] = True
                rows.append(row)
        print(family, root, sum(r['family'] == family for r in rows), flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(out, sep='\t', index=False)
    print(t.groupby(['family', 'model']).size().to_string())


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bids-folder', default='/shares/zne.uzh/gdehol/ds-retsupp')
    ap.add_argument('--root', action='append', required=True, help='family=derivatives folder (repeatable)')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    main(a.bids_folder, a.root, a.out)
