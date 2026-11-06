"""Conditional estimates of each gain term per subject x ROI, from the fixed-group sufficient statistics.

own = sum(b) / sum(a) over all voxels of a subject x ROI, as in ``loso_fixed_group_compare``. Exact for DoG
when lin_err is small; not valid for DN (its response curves in the gains). Light: reads one column set.

Usage::

    python -m retsupp.visualize.paper.collect_conditional_estimates --folder af_fixed_group_mechanisms_maxsd20
"""
from __future__ import annotations

import argparse
import glob

import pandas as pd


def main(bids, folder, out):
    files = sorted(glob.glob(f'{bids}/derivatives/{folder}/sub-*/*_stats.tsv.gz'))
    st = pd.concat([pd.read_csv(f, sep='\t', usecols=['subject', 'roi', 'mech', 'a', 'b', 'lin_err']) for f in files])
    own = st.groupby(['subject', 'roi', 'mech']).agg(a=('a', 'sum'), b=('b', 'sum'), lin_err=('lin_err', 'first')).reset_index()
    own['estimate'] = own.b / own.a
    own['valid'] = own.lin_err <= 1e-3
    own.drop(columns=['a', 'b']).to_csv(out, sep='\t', index=False)
    print(own.groupby('mech').estimate.median().round(4).to_dict(), len(own))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bids-folder', default='/data/ds-retsupp')
    ap.add_argument('--folder', default='af_fixed_group_mechanisms_maxsd20')
    ap.add_argument('--out', default='notes/data/conditional_estimates_dog_2026-11.tsv')
    a = ap.parse_args()
    main(a.bids_folder, a.folder, a.out)
