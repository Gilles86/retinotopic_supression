"""Side project (FEF / sPCS): functional frontal ROIs on fsaverage from the group pRF atlas.

Input: ``group_sums_fsaverage_v3.npz`` of the pRF-atlas side project (``side_prf_atlas/group_sums.py``). Its
``median_cv_r2`` is the group median over n = 28 of the signed cross-validated pRF R² (session 1 -> session 2 and
back; ``sign(cv_r) * cv_r**2``), smoothed on the fsaverage white surface to 3 mm FWHM.

ROIs (each a boolean (2, 163842) map, L then R):

- ``frontal``: a generous anatomical mask: aparc precentral + caudal middle frontal + superior frontal, posterior to
  y = +15 mm and lateral to |x| = 15 mm on the fsaverage white surface (drops the frontal pole and the medial wall).
- ``sPCS``: connected clusters (mesh neighbours) of ``frontal`` vertices with group CV R² > ``--thr`` (0.0025, a held-out
  r of 0.05 in the median participant: the loosest level of the atlas site) that touch the Destrieux superior
  precentral sulcus (``S_precentral-sup-part``) and have at least ``--min-size`` vertices.
- ``iPCS``: the same for the inferior precentral sulcus (``S_precentral-inf-part``).

Output: ``--out`` npz with the three masks, the threshold, and per-ROI vertex counts.

Usage (local)::

    python -m retsupp.side_fef.define_group_roi \
        --sums /data/ds-retsupp/derivatives/side_prf_atlas_group/group_sums_fsaverage_v3.npz \
        --fsaverage /data/ds-retsupp/derivatives/fmriprep/sourcedata/freesurfer/fsaverage \
        --out notes/side_fef/data/fsaverage_rois.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

FRONTAL = ('precentral', 'caudalmiddlefrontal', 'superiorfrontal')
SULCUS = {'sPCS': 'S_precentral-sup-part', 'iPCS': 'S_precentral-inf-part'}


def annot(fs, h, name):
    lab, _, names = nib.freesurfer.read_annot(str(Path(fs) / 'label' / f'{h}h.{name}.annot'))
    return lab, [n.decode() for n in names]


def clusters(faces, mask):
    e = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [0, 2]]])
    e = e[mask[e[:, 0]] & mask[e[:, 1]]]
    a = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(len(mask), len(mask)))
    _, cl = connected_components(a, directed=False)
    cl = cl.copy()
    cl[~mask] = -1
    return cl


def main(sums, fsaverage, out, thr=0.0025, min_size=5):
    q = np.load(sums)['median_cv_r2']
    rois = {k: np.zeros(q.shape, bool) for k in ('frontal', 'sPCS', 'iPCS')}
    for hi, h in enumerate('lr'):
        v, f = nib.freesurfer.read_geometry(str(Path(fsaverage) / 'surf' / f'{h}h.white'))
        lab, names = annot(fsaverage, h, 'aparc')
        lab2, names2 = annot(fsaverage, h, 'aparc.a2009s')
        front = np.isin(lab, [names.index(n) for n in FRONTAL]) & (v[:, 1] < 15) & (np.abs(v[:, 0]) > 15)
        rois['frontal'][hi] = front
        cl = clusters(f, front & (q[hi] > thr))
        for roi, sulc in SULCUS.items():
            in_sulc = lab2 == names2.index(sulc)
            for c in np.unique(cl[in_sulc & (cl >= 0)]):
                m = cl == c
                if m.sum() >= min_size:
                    rois[roi][hi] |= m
    counts = {k: m.sum(1).tolist() for k, m in rois.items()}
    print('vertices (L, R):', counts)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, thr=thr, min_size=min_size, **rois)
    print('wrote', out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sums', required=True)
    ap.add_argument('--fsaverage', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--thr', type=float, default=0.0025)
    ap.add_argument('--min-size', type=int, default=5)
    a = ap.parse_args()
    main(a.sums, a.fsaverage, a.out, a.thr, a.min_size)
