"""Table of one HRF per subject × ROI from the joint pRF + HRF fits, for ``roi_hrf.py``.

Reads ``derivatives/prf_cv_roi_jointhrf/sub-XX/sub-XX_fold-K_<family>_<tag>_summary.tsv`` (``fit_prf_cv_roi_hrf_joint``),
stage 3 (everything free). Per subject × ROI: mean over the folds of the ROI's delay and dispersion, the number of folds
with a finite fit and the spread (SD over folds). Missing subject × ROI pairs are listed; with ``--fill-canonical`` they get
the canonical HRF (flagged in ``source``), otherwise they are left out (the ladder then stops on them).

Usage::

    python -m retsupp.modeling.cv.build_roi_hrf_table --tag final2 --out /shares/.../paper_2026-11/roi_hrf_dog_final2.tsv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from retsupp.modeling.roi_hrf import CANONICAL

ROIS = ['V1', 'V2', 'V3', 'V3AB', 'IPS', 'hV4', 'VO', 'LO', 'TO']


def main(bids, family, tag, out, fill_canonical=False, n_subjects=30):
    root = Path(bids) / 'derivatives' / 'prf_cv_roi_jointhrf'
    files = sorted(root.glob(f'sub-*/sub-*_fold-*_{family}_{tag}_summary.tsv'))
    d = pd.concat([pd.read_csv(f, sep='\t') for f in files]) if files else pd.DataFrame()
    d = d[(d.stage == 3) & d.roi.isin(ROIS)] if len(d) else d
    rows = []
    for s in range(1, n_subjects + 1):
        for roi in ROIS:
            x = d[(d.subject == s) & (d.roi == roi)] if len(d) else d
            ok = x[np.isfinite(x.hrf_delay) & np.isfinite(x.hrf_dispersion) & np.isfinite(x.r2_test_reliable)] if len(x) else x
            if len(ok):
                rows.append(dict(subject=s, roi=roi, hrf_delay=ok.hrf_delay.mean(), hrf_dispersion=ok.hrf_dispersion.mean(),
                                 n_folds=len(ok), sd_delay=ok.hrf_delay.std(ddof=0), sd_dispersion=ok.hrf_dispersion.std(ddof=0),
                                 source='joint'))
            elif fill_canonical:
                rows.append(dict(subject=s, roi=roi, **CANONICAL, n_folds=0, sd_delay=np.nan, sd_dispersion=np.nan,
                                 source='canonical'))
    t = pd.DataFrame(rows)
    t.to_csv(out, sep='\t', index=False)
    n_joint = int((t.source == 'joint').sum()) if len(t) else 0
    print(f'{len(files)} fold summaries; {n_joint} of {n_subjects * len(ROIS)} subject x ROI from the joint fits, '
          f'{len(t) - n_joint} canonical; saved {out}')
    if n_joint:
        print(t[t.source == 'joint'][['hrf_delay', 'hrf_dispersion', 'n_folds', 'sd_delay', 'sd_dispersion']].describe().round(3))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bids-folder', default='/shares/zne.uzh/gdehol/ds-retsupp')
    ap.add_argument('--family', default='dog')
    ap.add_argument('--tag', default='final2')
    ap.add_argument('--out', required=True)
    ap.add_argument('--fill-canonical', action='store_true')
    a = ap.parse_args()
    main(a.bids_folder, a.family, a.tag, a.out, a.fill_canonical)
