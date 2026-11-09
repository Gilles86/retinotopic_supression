"""Concatenate the per-run quadrant readouts of ``simulate_decode.py`` into one table (compute node, not login).

Reads ``<bids>/derivatives/decoded_sim_2026-11/<source>/noise-<noise>/gain-*/<decoder>/sub-XX/readout_*.tsv``
(columns include ``gain_amplitude``, ``decoder``, ``drive_bar`` and ``true_drive_bar``) and writes one TSV. Refuses
to overwrite an existing output.

Usage::

    srun --account=hare.econ.uzh -c1 --mem 4G --time 10 \\
        $HOME/data/conda/envs/retsupp_cuda/bin/python -m retsupp.decode.collect_simulate_decode \\
        --out notes/data/decode_sim_2026-11.tsv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main(bids, source, noise, out):
    out = Path(out)
    if out.exists():
        raise FileExistsError(f'{out} exists; not overwriting')
    root = Path(bids) / 'derivatives' / 'decoded_sim_2026-11' / source / f'noise-{noise}'
    files = sorted(root.glob('gain-*/*/sub-*/readout_*.tsv'))
    d = pd.concat([pd.read_csv(f, sep='\t') for f in files], ignore_index=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(out, sep='\t', index=False)
    print(f'{len(files)} readouts, {len(d)} rows -> {out}')
    print(d.groupby(['gain_amplitude', 'decoder']).subject.nunique().rename('n_subjects').to_string())


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--bids-folder', default='/shares/zne.uzh/gdehol/ds-retsupp')
    p.add_argument('--source', default='canonical')
    p.add_argument('--noise', default='phase')
    p.add_argument('--out', default='notes/data/decode_sim_2026-11.tsv')
    a = p.parse_args()
    main(a.bids_folder, a.source, a.noise, a.out)
