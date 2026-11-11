"""Collect shared AF parameters of the repeat-split joint fits.

Reads the canonical, distractor-repeat and target-repeat AF pickles
(v3 + target + sharedSigma, pSig0.5) and writes one row per
(model, subject, ROI) with every shared parameter as a column.

Run on the cluster (many small pickles):
``python -m retsupp.visualize.paper.collect_repeat_af_fits \\
    --bids-folder /shares/zne.uzh/gdehol/ds-retsupp \\
    --out notes/data/repeat_af_fits_params.tsv``
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import pandas as pd

BASE = 'af_prf_joint_dynamic_v3_dog_with_target_sharedSigma'
MODELS = {
    'canonical': (f'{BASE}_pSig0.5', 'dog-dyn-v3-target-sharedSigma'),
    'repeat': (f'{BASE}_repeat_pSig0.5', 'dog-dyn-v3-target-sharedSigma-repeat'),
    'targetRepeat': (f'{BASE}_targetRepeat_pSig0.5',
                     'dog-dyn-v3-target-sharedSigma-targetRepeat'),
}


def main(bids_folder: Path, out: Path):
    rows = []
    for model, (subdir, tag) in MODELS.items():
        paths = sorted((bids_folder / 'derivatives' / subdir).glob(
            f'sub-*/sub-*_roi-*_mode-signed_{tag}-af-prf-fit.pkl'))
        print(f'{model}: {len(paths)} pickles')
        for p in paths:
            with open(p, 'rb') as f:
                d = pickle.load(f)
            row = {k: float(v) for k, v in d['shared_pars'].items()}
            row.update(model=model,
                       subject=int(p.name.split('sub-')[1].split('_')[0]),
                       roi=p.name.split('roi-')[1].split('_')[0])
            if 'r2' in d:
                row['r2'] = float(pd.Series(d['r2']).mean())
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out, sep='\t', index=False)
    print(f'-> {out} ({len(df)} rows)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bids-folder', type=Path,
                        default=Path('/data/ds-retsupp'))
    parser.add_argument('--out', type=Path,
                        default=Path('notes/data/repeat_af_fits_params.tsv'))
    args = parser.parse_args()
    main(args.bids_folder, args.out)
