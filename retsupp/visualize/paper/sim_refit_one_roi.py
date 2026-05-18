"""Sim+refit validation: ground-truth what our AF model predicts a vanilla
m4 conditionwise fitter would extract.

Pipeline for one (subject, ROI):

1. Load the AF fit pickle (full per-voxel + shared params).
2. Reconstruct the actual paradigm + indicators via
   `build_data_and_paradigm`.
3. Predict noiseless BOLD for the ROI's voxels via the joint AF model.
4. For each of 4 HP conditions: refit a vanilla DoG+HRF (the m4
   conditionwise model) on the per-condition simulated BOLD.
5. Write a TSV with sim_refit_x/y per (voxel, condition).

This is the "what would m4 conditionwise extract if our model were
ground truth?" test — the rigorous version of the analytic CoM
shortcut in ``fit_klein_static_shift.our_af_predict_centers``.

Run on cluster (cached BOLD = ~30s paradigm vs ~5min locally):

    sbatch retsupp/visualize/paper/slurm_jobs/sim_refit.sh
"""
from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

CONDITIONS = ['upper_right', 'upper_left', 'lower_left', 'lower_right']
RING_KEYS = ['upper right', 'upper left', 'lower left', 'lower right']


def get_pkl_path(af_fits_root: Path, subject: int, roi: str) -> Path:
    return (af_fits_root / f'sub-{subject:02d}'
            / f'sub-{subject:02d}_roi-{roi}_'
              'mode-signed_dog-dyn-v3-target-sharedSigma-af-prf-fit.pkl')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('subject', type=int)
    p.add_argument('--roi', required=True)
    p.add_argument('--bids-folder', default='/shares/zne.uzh/gdehol/ds-retsupp')
    p.add_argument('--af-fits-root',
                   default='/shares/zne.uzh/gdehol/ds-retsupp/derivatives/'
                           'af_prf_joint_dynamic_v3_dog_with_target_'
                           'sharedSigma_pSig0.5')
    p.add_argument('--out-dir', default=None,
                   help='Default: <bids>/derivatives/sim_refit/sub-XX/')
    p.add_argument('--max-voxels', type=int, default=500,
                   help='Cap on voxels (top by R²). Set 0 = all.')
    p.add_argument('--n-iter', type=int, default=500)
    p.add_argument('--lr', type=float, default=0.01)
    args = p.parse_args()

    from retsupp.utils.data import Subject, distractor_locations
    from retsupp.modeling.fit_dog_dynamic_af_braincoder import (
        build_data_and_paradigm)
    from retsupp.modeling.local_models import (
        DoGDynamicAttentionFieldPRF2DWithHRF_v3_target_sharedSigma)
    from braincoder.hrf import SPMHRFModel
    from braincoder.models import DifferenceOfGaussiansPRF2DWithHRF
    from braincoder.optimize import ParameterFitter

    af_fits_root = Path(args.af_fits_root)
    pkl_path = get_pkl_path(af_fits_root, args.subject, args.roi)
    if not pkl_path.exists():
        raise SystemExit(f'AF pickle not found: {pkl_path}')

    out_dir = (Path(args.out_dir) if args.out_dir
               else Path(args.bids_folder) / 'derivatives' / 'sim_refit'
                    / f'sub-{args.subject:02d}')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_tsv = (out_dir / f'sub-{args.subject:02d}_roi-{args.roi}_'
                          'sim-refit.tsv')

    print(f'sub-{args.subject:02d} roi={args.roi}')
    print(f'  AF pickle: {pkl_path}')
    print(f'  out      : {out_tsv}')

    with open(pkl_path, 'rb') as f:
        af = pickle.load(f)
    fp = af['fit_pars']
    print(f'  AF fit pars: {fp.shape}  shared: {af["shared_pars"]}')

    if args.max_voxels and args.max_voxels < len(fp):
        fp_top = fp.sort_values('r2', ascending=False).head(args.max_voxels).copy()
    else:
        fp_top = fp.copy()
    print(f'  selected {len(fp_top)} voxels  '
          f'(r² {fp_top["r2"].min():.3f}-{fp_top["r2"].max():.3f})')

    ring = np.array([list(distractor_locations[k]) for k in RING_KEYS],
                    dtype=np.float32)
    sub = Subject(args.subject, args.bids_folder)

    t0 = time.time()
    out = build_data_and_paradigm(
        sub, resolution=50, grid_radius=5.0,
        with_target=True, temporal_oversampling=1,
        distractor_shape='rectangle',
        distractor_long_side=1.5, distractor_short_side=0.375)
    _bold_df, paradigm, cond_ind, dyn_ind, grid_coords, _masker, tgt_ind = out[:7]
    print(f'  paradigm built in {time.time()-t0:.1f}s  T={paradigm.shape[0]}')

    hrf = SPMHRFModel(tr=sub.get_tr(1, 1))
    af_model = DoGDynamicAttentionFieldPRF2DWithHRF_v3_target_sharedSigma(
        grid_coordinates=grid_coords, paradigm=paradigm, hrf_model=hrf,
        flexible_hrf_parameters=True,
        condition_indicator=cond_ind, dynamic_indicator=dyn_ind,
        target_indicator=tgt_ind, ring_positions=ring, mode='signed')

    labels = list(af_model.parameter_labels) + ['hrf_delay', 'hrf_dispersion']
    labels = [l for l in labels if l in fp_top.columns]
    params_in = fp_top.loc[:, labels].copy()
    t0 = time.time()
    bold_sim_df = af_model.predict(parameters=params_in)
    bold_sim = (bold_sim_df.values if hasattr(bold_sim_df, 'values')
                else np.asarray(bold_sim_df)).astype(np.float32)
    if bold_sim.ndim == 3:
        bold_sim = bold_sim[0]
    print(f'  AF predict in {time.time()-t0:.1f}s  '
          f'bold_sim {bold_sim.shape} range '
          f'[{bold_sim.min():.2f}, {bold_sim.max():.2f}]')

    tr_cond = np.argmax(cond_ind, axis=1)

    refit_rows = []
    for ci, cond in enumerate(CONDITIONS):
        tr_mask = (tr_cond == ci)
        n_tr = int(tr_mask.sum())
        if n_tr < 100:
            print(f'  {cond}: only {n_tr} TRs — skip')
            continue
        bold_cond = pd.DataFrame(bold_sim[tr_mask])
        par_cond = paradigm[tr_mask]

        m4 = DifferenceOfGaussiansPRF2DWithHRF(
            grid_coordinates=grid_coords, paradigm=par_cond,
            hrf_model=hrf, flexible_hrf_parameters=True)
        fitter = ParameterFitter(m4, bold_cond, par_cond)
        init = pd.DataFrame({
            'x': fp_top['x'].values, 'y': fp_top['y'].values,
            'sd': fp_top['sd'].values,
            'baseline': fp_top['baseline'].values,
            'amplitude': fp_top['amplitude'].values,
            'srf_amplitude': fp_top['srf_amplitude'].values,
            'srf_size': fp_top['srf_size'].values,
            'hrf_delay': fp_top['hrf_delay'].values,
            'hrf_dispersion': fp_top['hrf_dispersion'].values,
        })
        refined = fitter.refine_baseline_and_amplitude(init, l2_alpha=1e-3)
        t0 = time.time()
        fit_out = fitter.fit(init_pars=refined,
                              max_n_iterations=args.n_iter,
                              learning_rate=args.lr,
                              progressbar=False)
        print(f'  {cond}: refit {n_tr} TR × {bold_cond.shape[1]} vox '
              f'in {time.time()-t0:.1f}s')
        for vi, vid in enumerate(fp_top.index):
            refit_rows.append(dict(
                subject=args.subject, roi=args.roi, condition=cond,
                voxel_idx=int(vid),
                sim_refit_x=float(fit_out['x'].iloc[vi]),
                sim_refit_y=float(fit_out['y'].iloc[vi]),
                sim_refit_sd=float(fit_out['sd'].iloc[vi]),
                base_x=float(fp_top['x'].iloc[vi]),
                base_y=float(fp_top['y'].iloc[vi]),
                base_sd=float(fp_top['sd'].iloc[vi]),
                af_r2=float(fp_top['r2'].iloc[vi]),
            ))

    out_df = pd.DataFrame(refit_rows)
    out_df.to_csv(out_tsv, sep='\t', index=False)
    print(f'wrote {out_tsv}  ({len(out_df)} rows)')


if __name__ == '__main__':
    main()
