"""Side project (FEF / sPCS): per participant x ROI summary of the signed session-split CV pRFs (pRF-atlas side project).

For every voxel of each ROI (``side_fef_rois``; value 1 = left, 2 = right hemisphere): ``cv_r`` (held-out session
correlation), ``cv_p`` (circular-shift null, 50 shifts, so the smallest p is 1/51), amplitude sign, centre x, y and
sigma of the all-runs signed fit, and the reliability ``rel_r``/``rel_p``. Writes one row per voxel (``--voxels-out``)
for the selected voxels only and one row per participant x ROI (``--summary-out``) with

- ``frac_cv``: fraction of ROI voxels with a CV pRF (cv_p < .05, cv_r > 0); ``frac_cv_strict``: cv_p = 1/51 (beats all
  50 null shifts). Expected under the null: about 0.04 and 0.02 (the p grid is 1/51 steps).
- ``frac_cv_pos``: same with positive amplitude; ``contra_pos``: contralateral fraction of the centres of those voxels
  (0.5 without retinotopy; cross-validation does not select on hemifield); ``n_relnull_r01``: voxels with reliability
  r > 0.1 and p <= .01 (the ladder selection at r > 0.1).

Usage (compute node)::

    python -m retsupp.side_fef.summarize_signed --voxels-out ... --summary-out ...
"""
from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from retsupp.side_fef.extract_rois import ALL_ROIS, ROIS_SUBDIR

KEYS = ('cv_r', 'cv_p', 'amplitude', 'x', 'y', 'sd', 'rel_r', 'rel_p', 'x_f1', 'y_f1', 'x_f2', 'y_f2')
ROIS = ('frontal',) + ALL_ROIS


def main(bids_folder, voxels_out, summary_out, subjects):
    der = Path(bids_folder) / 'derivatives'
    vrows, srows = [], []
    for subject in subjects:
        s = f'sub-{subject:02d}'
        rd = der / ROIS_SUBDIR / s
        if not rd.exists():
            print('missing', s)
            continue
        maps = {k: np.asarray(nib.load(str(der / 'side_prf_atlas_signed' / s / f'{s}_desc-{k}_bold.nii.gz')).dataobj,
                              np.float32) for k in KEYS}
        for roi in ROIS:
            lab = np.asarray(nib.load(str(rd / f'{s}_desc-{roi}_mask.nii.gz')).dataobj)
            m = lab > 0
            d = pd.DataFrame({k: v[m] for k, v in maps.items()})
            d['hemi'] = np.where(lab[m] == 1, 'L', 'R')
            d['contra'] = np.where(d.hemi == 'L', d.x > 0, d.x < 0)
            d['cv'] = (d.cv_p < .05) & (d.cv_r > 0)
            d['cv_strict'] = (d.cv_p < 1.5 / 51) & (d.cv_r > 0)
            d['pos'] = d.cv & (d.amplitude > 0)
            d['relnull_r01'] = (d.rel_r > .1) & (d.rel_p <= .01)
            pos = d[d.pos]
            for hemi in ('both', 'L', 'R'):
                dd = d if hemi == 'both' else d[d.hemi == hemi]
                pp = dd[dd.pos]
                srows.append(dict(subject=subject, roi=roi, hemi=hemi, n_vox=len(dd), frac_cv=dd.cv.mean(),
                                  frac_cv_strict=dd.cv_strict.mean(), frac_cv_pos=dd.pos.mean(),
                                  frac_cv_neg=(dd.cv & (dd.amplitude < 0)).mean(), n_cv_pos=len(pp),
                                  contra_pos=pp.contra.mean() if len(pp) else np.nan,
                                  contra_all=dd.contra.mean() if len(dd) else np.nan,
                                  ecc_pos=np.hypot(pp.x, pp.y).median() if len(pp) else np.nan,
                                  sd_pos=pp.sd.median() if len(pp) else np.nan,
                                  median_cv_r=dd.cv_r.median(), n_relnull_r01=int(dd.relnull_r01.sum())))
            if roi != 'frontal' and roi != 'frontal_rest':
                vrows.append(pos.assign(subject=subject, roi=roi))
        print(s, flush=True)
    pd.concat(vrows).to_csv(voxels_out, sep='\t', index=False)
    sm = pd.DataFrame(srows)
    sm.to_csv(summary_out, sep='\t', index=False)
    pd.set_option('display.width', 200)
    print(sm.groupby(['roi', 'hemi']).median(numeric_only=True).drop(columns='subject').round(3).to_string())


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bids-folder', default='/shares/zne.uzh/gdehol/ds-retsupp')
    ap.add_argument('--voxels-out', required=True)
    ap.add_argument('--summary-out', required=True)
    ap.add_argument('--subjects', type=int, nargs='+', default=list(range(3, 31)))
    a = ap.parse_args()
    main(a.bids_folder, a.voxels_out, a.summary_out, a.subjects)
