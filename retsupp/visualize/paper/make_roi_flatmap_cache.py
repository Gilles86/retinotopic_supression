"""Rasterise the 11 analysis ROIs onto the fsaverage flatmap (pycortex).

Writes ``notes/data/fsaverage_roi_flatmap.npz`` with a per-pixel ROI
index (1..11 in ROI_ORDER, NaN elsewhere), the curvature image and the
flatmap extent, so figure scripts can paint any per-ROI value onto
cortex without importing pycortex.

ROIs follow the analysis atlases: Benson 2014 (V1..V3AB) and Wang 2015
max-probability labels (IPS0-5 → IPS, SPL1, FEF), Wang only where Benson
is empty — the same rule as ``Subject.get_retinotopic_roi``. Template
labels on fsaverage come from neuropythy's fsaverage atlas outputs.

Run in the pycortex env:
``~/mambaforge/envs/pycortex2/bin/python -m retsupp.visualize.paper.make_roi_flatmap_cache``
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cortex
import nibabel as nib
import numpy as np
from cortex import quickflat

ROI_ORDER = ['V1', 'V2', 'V3', 'V3AB', 'hV4', 'LO', 'TO', 'VO', 'IPS', 'SPL1', 'FEF']
ATLAS = ('/data/ds-retinonumeral/derivatives/neuropythy_atlas/sub-04/'
         'sub-04_desc-{}_space-fsaverage_hemi-{}.func.gii')
# neuropythy benson14 varea codes → analysis ROI
BENSON = {1: 'V1', 2: 'V2', 3: 'V3', 4: 'hV4', 5: 'VO', 6: 'VO', 7: 'LO',
          8: 'LO', 9: 'TO', 10: 'TO', 11: 'V3AB', 12: 'V3AB'}
WANG = {**{k: 'IPS' for k in range(18, 24)}, 24: 'SPL1', 25: 'FEF'}


def main(out: Path, height: int):
    ben, wang = (np.concatenate([nib.load(ATLAS.format(d, h)).agg_data()
                                 for h in 'LR'])
                 for d in ('benson14Varea', 'wang15Mplbl'))
    lab = np.full(ben.shape, np.nan)
    for code, roi in BENSON.items():
        lab[ben == code] = ROI_ORDER.index(roi) + 1
    for code, roi in WANG.items():
        lab[(wang == code) & (ben == 0)] = ROI_ORDER.index(roi) + 1
    im, ext = quickflat.utils.make_flatmap_image(
        cortex.Vertex(lab, 'fsaverage'), height=height)
    curv, _ = quickflat.utils.make_flatmap_image(
        cortex.db.get_surfinfo('fsaverage'), height=height)
    np.savez_compressed(out, roi=np.round(im), curv=curv, extent=ext,
                        roi_order=np.array(ROI_ORDER))
    print(f'-> {out} {im.shape}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path,
                        default=Path('notes/data/fsaverage_roi_flatmap.npz'))
    parser.add_argument('--height', type=int, default=1400)
    args = parser.parse_args()
    main(args.out, args.height)
