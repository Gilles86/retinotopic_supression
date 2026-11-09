"""Collect the per-subject tables of ``fit_dn_by_hp`` into one table (run on the cluster).

Refuses to overwrite.

Usage::

    python -m retsupp.modeling.collect_dn_by_hp --out notes/data/dn_by_hp_2026-11.tsv.gz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main(bids_folder, out, subdir='prf_dn_by_hp_2026-11'):
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    root = Path(bids_folder) / 'derivatives' / subdir
    files = sorted(root.glob('sub-*/sub-*_dn_by_hp.tsv.gz'))
    have = {int(f.name[4:6]) for f in files}
    print(f'{len(files)} subjects; missing: {sorted(set(range(1, 31)) - have)}')
    d = pd.concat([pd.read_csv(f, sep='\t') for f in files], ignore_index=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(out, sep='\t', index=False, float_format='%.6g')
    print(f'Saved {len(d)} rows to {out}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--bids-folder', default='/shares/zne.uzh/gdehol/ds-retsupp')
    ap.add_argument('--subdir', default='prf_dn_by_hp_2026-11')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    main(a.bids_folder, a.out, a.subdir)
