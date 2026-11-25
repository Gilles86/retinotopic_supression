"""Per-run sufficient statistics of the gains of the zero-sum LP-split model (update 4, item 7; 2026-11-25).

Per subject x ROI, on the joint zero-sum fit of ``m6_TDS_dS_adjopp_zs`` (the main model m5_TDS_dS_zs plus
δ = g_next − g_opp, ``lp_split.LPSplitDoG``; root ``<base>_zs_lm``), every parameter is held at the fit except the
gains, which enter the prediction linearly, so (as ``fit_width_profiles``)

    P(g) = P_0 + Σ_k g_k X_k,   X_k = [P(g_k = H) − P_0] / H,   P_0 = the prediction with all gains at 0,

with gains g_HP, g_LP, δ (g_adj_minus_opp), g_T (g_T_dyn) and the distractor gains g_D_HP, g_D_LP (g_HP_dyn,
g_LP_dyn; tied in the fit). Stored per run r: A_r = Σ X_k X_l, b_r = Σ X_k (Y − P_0), c_r = Σ (Y − P_0)², so the
model's free gains (dS along g_HP = ¾ dS, g_LP = −¼ dS; δ; target; one distractor gain) and their leave-one-run-out
jackknife SEs are solved offline (``retsupp/visualize/paper/adjopp_meta.py``).

Output ``derivatives/<output-subdir>/sub-XX/sub-XX_roi-R_gainstats.npz``.

    python -m retsupp.modeling.fit_gain_stats_adjopp 3 --rois V1   # canary
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

from retsupp.modeling.fit_af_prf_cv_v2 import build_data_and_paradigm, get_ring_positions
from retsupp.modeling.fit_cv_ladder_nested import sign_pattern_of
from retsupp.modeling.fit_width_profiles import run_stats
from retsupp.modeling.lp_split import NEW, LPSplitDoG
from retsupp.modeling.roi_hrf import hrf_model as roi_hrf_model, load_table as load_hrf_table
from retsupp.utils.data import Subject

ROIS = ['V1', 'V2', 'V3', 'V3AB', 'hV4', 'LO', 'TO', 'VO', 'IPS']
MODEL = 'm6_TDS_dS_adjopp_zs'
H = 0.5
GAINS = {'g_HP': ['g_HP'], 'g_LP': ['g_LP'], 'delta': [NEW], 'g_T': ['g_T_dyn'],
         'g_D_HP': ['g_HP_dyn'], 'g_D_LP': ['g_LP_dyn']}


def set_gain(p, name, value):
    p = p.copy()
    for col in GAINS[name]:
        p[col] = value
    return p


def fit_roi(sub, roi, data, root, out_dir, hrf_table):
    bold, paradigm, ci, di, ti, grid, run_meta = data
    pkl = Path(root) / MODEL / f'sub-{sub.subject_id:02d}' / f'sub-{sub.subject_id:02d}_roi-{roi}_cv-fits.pkl'
    with open(pkl, 'rb') as f:
        fit = pickle.load(f)
    model = LPSplitDoG(grid_coordinates=grid, paradigm=paradigm, condition_indicator=ci, dynamic_indicator=di,
                       target_indicator=ti, ring_positions=get_ring_positions(),
                       hrf_model=roi_hrf_model(sub, roi, hrf_table), sign_pattern=sign_pattern_of('m7_full'),
                       item_scale_mode='free')
    pars = fit['fit_pars_per_fold'][0].reindex(columns=list(model.parameter_labels))
    assert not pars.isna().any().any(), f'{pkl}: NaN parameters'
    Y = bold[:, np.asarray(fit['voxel_ids'])].astype(np.float64)

    def pred(p):
        return np.asarray(model.predict(paradigm=paradigm, parameters=p), dtype=np.float64)

    names = list(GAINS)
    p0 = pars.copy()
    for n in names:
        p0 = set_gain(p0, n, 0.)
    P0 = pred(p0)
    X = np.stack([(pred(set_gain(p0, n, H)) - P0) / H for n in names])
    lin = [np.abs(X[k] - (P0 - pred(set_gain(p0, n, -H))) / H).max() / max(np.abs(X[k]).max(), 1e-12)
           for k, n in enumerate(names)]
    A, b, c = run_stats(X, Y - P0, run_meta)
    fitted = np.array([float(pars[GAINS[n][0]].iloc[0]) for n in names])
    # check: the full prediction is linear in the gains, so P0 + Σ g X reproduces the fit's prediction
    dev = float(np.abs(P0 + np.einsum('k,ktv->tv', fitted, X) - pred(pars)).max())
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / f'sub-{sub.subject_id:02d}_roi-{roi}_gainstats.npz', subject=sub.subject_id,
                        roi=roi, n_voxels=Y.shape[1], gain_names=np.array(names), A=A, b=b, c=c,
                        lin_err=np.array(lin), recon_err=dev, fitted_gains=fitted,
                        runs=np.array([[rm.session, rm.run] for rm in run_meta]))
    g_all = np.linalg.solve(A.sum(0), b.sum(0))
    print(f'sub-{sub.subject_id:02d} {roi}: {Y.shape[1]} voxels; lin_err max {max(lin):.1e}; recon {dev:.2e}; '
          + ', '.join(f'{n} {g:+.4f} (fit {f:+.4f})' for n, g, f in zip(names, g_all, fitted)), flush=True)


def main(subject, bids_folder, rois, root, output_subdir, hrf_table):
    sub = Subject(subject, bids_folder=Path(bids_folder))
    out_dir = Path(bids_folder) / 'derivatives' / output_subdir / f'sub-{subject:02d}'
    todo = [r for r in rois if not (out_dir / f'sub-{subject:02d}_roi-{r}_gainstats.npz').exists()
            and (Path(root) / MODEL / f'sub-{subject:02d}' / f'sub-{subject:02d}_roi-{r}_cv-fits.pkl').exists()]
    if not todo:
        print('Nothing to do')
        return
    bold, paradigm, ci, di, ti, grid, _masker, run_meta = build_data_and_paradigm(
        sub, resolution=50, grid_radius=5.0, item_shape='rectangle')
    tab = load_hrf_table(hrf_table)
    for roi in todo:
        fit_roi(sub, roi, (bold, paradigm, ci, di, ti, grid, run_meta), root, out_dir, tab)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('subject', type=int)
    ap.add_argument('--bids-folder', default='/shares/zne.uzh/gdehol/ds-retsupp')
    ap.add_argument('--rois', nargs='+', default=ROIS)
    ap.add_argument('--root', default='/shares/zne.uzh/gdehol/ds-retsupp/derivatives/'
                    'af_prf_cv_ladder_nested_full_bounded_maxsd20_reliable_roihrf_zs_lm')
    ap.add_argument('--output-subdir', default='adjopp_gain_stats_zs_2026-11-25')
    ap.add_argument('--hrf-table', default='/shares/zne.uzh/gdehol/ds-retsupp/derivatives/paper_2026-11/roi_hrf_dog_final2.tsv')
    a = ap.parse_args()
    main(a.subject, a.bids_folder, a.rois, a.root, a.output_subdir, a.hrf_table)
