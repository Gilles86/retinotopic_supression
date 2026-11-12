"""Gain terms by pRF size within an area (2026-11-12): per-voxel sufficient statistics on the joint fit.

Per subject x ROI, on the subject's joint gain-field fit (DoG pRFs, per-ROI HRF), every parameter is held at the fit
except the five gains (sustained at HP and at LP, target, distractor at HP and at LP), which enter the prediction
linearly (as in ``fit_width_profiles``):

    P(g) = P_0 + Σ_k g_k X_k,   X_k = [P(g_k = H) − P_0] / H,   P_0 = the prediction with the five gains at 0.

Stored per voxel v (summed over all runs): A_v = Σ_t X_k X_l (5 x 5), b_v = Σ_t X_k (Y − P_0); with the voxel's pRF
centre (x, y) and size (σ of the DoG centre). Any subset of voxels then gives the five gains (Σ_v A_v)⁻¹ Σ_v b_v,
so the split by pRF-size or eccentricity bins is done offline (``hierarchy_gain_terms``).

Output ``derivatives/<output-subdir>/sub-XX/sub-XX_roi-R_gainbins.npz``.

    python -m retsupp.modeling.fit_gain_bins 3 --rois V3   # canary
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import retsupp.modeling.fit_fixed_group_mechanisms as fgm
from retsupp.modeling.fit_af_prf_cv_v2 import build_data_and_paradigm
from retsupp.modeling.fit_width_profiles import GAINS, ROIS, design, set_gain
from retsupp.modeling.roi_hrf import load_table as load_hrf_table
from retsupp.utils.data import Subject

H = 0.5


def fit_roi(sub, roi, data, out_dir, overlay_root):
    bold, paradigm, _ci, _di, _ti, _grid, run_meta = data
    model = fgm.build_full_model(sub, data, roi)
    fit, pars, _w, _n = fgm.base_parameters(sub, model, roi, 'm7_full', None, overlay_root)
    Y = bold[:, fit['voxel_ids']].astype(np.float64)

    def pred(p):
        return np.asarray(model.predict(paradigm=paradigm, parameters=p), dtype=np.float64)

    names = list(GAINS)
    P0, X, R = design(pred, pars, names, Y)
    A_v = np.einsum('ktv,ltv->vkl', X, X)
    b_v = np.einsum('ktv,tv->vk', X, R)
    fitted = np.array([float(pars[GAINS[n][0]].iloc[0]) for n in names])
    g_all = np.linalg.solve(A_v.sum(0), b_v.sum(0))
    out = dict(subject=sub.subject_id, roi=roi, gain_names=np.array(names), voxel_ids=np.asarray(fit['voxel_ids']),
               A_v=A_v, b_v=b_v, x=pars['x'].to_numpy(), y=pars['y'].to_numpy(), sd=pars['sd'].to_numpy(),
               fitted_gains=fitted, overlay=str(overlay_root))
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / f'sub-{sub.subject_id:02d}_roi-{roi}_gainbins.npz', **out)
    print(f'sub-{sub.subject_id:02d} {roi}: {Y.shape[1]} voxels; gains re-solved '
          + ', '.join(f'{n} {g:+.4f} (fit {f:+.4f})' for n, g, f in zip(names, g_all, fitted)), flush=True)


def main(subject, bids_folder, rois, output_subdir, base_root, hrf_table, overlay_root):
    fgm.BASE_ROOT = base_root
    fgm.HRF_TABLE = load_hrf_table(hrf_table)
    fgm.FAMILY = 'dog'
    if overlay_root:      # e.g. the polished root (pRFs refitted with the gain fields) holds m7_full only
        fgm.OVERLAY_PER_MODEL = True
    sub = Subject(subject, bids_folder=Path(bids_folder))
    out_dir = Path(bids_folder) / 'derivatives' / output_subdir / f'sub-{subject:02d}'
    todo = [r for r in rois if not (out_dir / f'sub-{subject:02d}_roi-{r}_gainbins.npz').exists()]
    todo = [r for r in todo if fgm.fit_path(bids_folder, 'm7_full', subject, r, overlay_root).exists()]
    if not todo:
        print('Nothing to do')
        return
    bold, paradigm, ci, di, ti, grid, _masker, run_meta = build_data_and_paradigm(
        sub, resolution=50, grid_radius=5.0, item_shape='rectangle')
    data = (bold, paradigm, ci, di, ti, grid, run_meta)
    for roi in todo:
        fit_roi(sub, roi, data, out_dir, overlay_root)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('subject', type=int)
    ap.add_argument('--bids-folder', default='/shares/zne.uzh/gdehol/ds-retsupp')
    ap.add_argument('--rois', nargs='+', default=ROIS)
    ap.add_argument('--output-subdir', default='gain_bins_2026-11-12')
    ap.add_argument('--base-root', default='af_prf_cv_ladder_nested_full_bounded_maxsd20_reliable_roihrf')
    ap.add_argument('--hrf-table', default='/shares/zne.uzh/gdehol/ds-retsupp/derivatives/paper_2026-11/roi_hrf_dog_final2.tsv')
    ap.add_argument('--overlay-root', default=None)
    a = ap.parse_args()
    main(a.subject, a.bids_folder, a.rois, a.output_subdir, a.base_root, a.hrf_table, a.overlay_root)
