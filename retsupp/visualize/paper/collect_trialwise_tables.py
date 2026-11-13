"""Concatenate the per-subject × area tables of ``fit_trialwise_af_gains`` that Fig. 7 needs (parametric, check, bins).

Light version of ``analyze_trialwise_af_gains collect`` (which also reads every trial table): run locally after
rsyncing only these three file types.

    python -m retsupp.visualize.paper.collect_trialwise_tables /data/ds-retsupp/derivatives/<subdir> notes/data/<tag>/trialwise_af_<tag>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main(src, prefix, min_subject=3):
    src, prefix = Path(src), Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for name in ('parametric', 'check', 'bins'):
        files = sorted(src.glob(f'sub-*/sub-*_roi-*_{name}.tsv'))
        d = pd.concat([pd.read_csv(f, sep='\t') for f in files])
        d = d[d.subject >= min_subject]
        d.to_csv(f'{prefix}_{name}.tsv', sep='\t', index=False)
        print(f'{name}: {len(files)} files, {d.subject.nunique()} subjects -> {prefix}_{name}.tsv')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('src')
    ap.add_argument('prefix')
    ap.add_argument('--min-subject', type=int, default=3, help='sub-01, sub-02 (pilots) are left out by default')
    a = ap.parse_args()
    main(a.src, a.prefix, a.min_subject)
