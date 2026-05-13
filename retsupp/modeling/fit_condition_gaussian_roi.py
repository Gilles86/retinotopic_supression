"""Fast conditionwise GAUSSIAN PRF refit, restricted to retinotopic ROIs.

Apples-to-apples comparison for the joint AF + PRF model: our AF model
fits each voxel as a single Gaussian, but the existing
prf_conditionfit/model4 fits use DoG. This script refits each voxel's
conditionwise PRF as a plain Gaussian (model 1, fixed HRF) so that the
PRED side of `predict_shifts_from_af.py` and the OBS side use matched
voxel kernels.

Speedups vs the full fit_condition.py:
  • Voxel mask restricted to UNION of the 8 retinotopic ROIs
    (V1, V2, V3, V3AB, hV4, LO, TO, VO) — ~2000 voxels per subject
    instead of ~280K. Most of the time saving.
  • Init from model 4 (DoG with HRF): use its x, y, sd, amplitude,
    baseline. Drop srf_size / srf_amplitude (we're fitting Gaussian).
  • Save ONE compact TSV per subject:
      sub-XX_cond-CC_gaussian_roi.tsv  (long format — voxel_idx, x, y,
      sd, baseline, amplitude, r2 per (voxel × condition))
  • No NIfTI output (predict_shifts script needs to be updated to read
    this TSV; alternatively call this with --save-nifti to also dump
    NIfTIs in the standard prf_conditionfit/model1/ layout).

Output dir:
  derivatives/prf_conditionfit/model1/sub-XX/sub-XX_cond-CC_gaussian_roi.tsv

Usage
-----
    python -m retsupp.modeling.fit_condition_gaussian_roi 5 \\
        --bids-folder /shares/zne.uzh/gdehol/ds-retsupp
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from nilearn import input_data
from tqdm import tqdm

from braincoder.hrf import SPMHRFModel
from braincoder.models import GaussianPRF2DWithHRF
from braincoder.optimize import ParameterFitter
from retsupp.utils.data import Subject


CONDITIONS_HP = ['upper_right', 'upper_left', 'lower_left', 'lower_right']
ROIS = ['V1', 'V2', 'V3', 'V3AB', 'hV4', 'LO', 'TO', 'VO']


def get_roi_voxel_mask(sub: Subject, rois=ROIS) -> np.ndarray:
    """Boolean mask over BOLD voxels: True if voxel is in any of `rois`."""
    masker_full = sub.get_bold_mask(return_masker=True)
    masker_full.fit()
    n_vox = int(masker_full.mask_img_.get_fdata().sum())
    out = np.zeros(n_vox, dtype=bool)
    aliases = {
        'V3AB': ['V3A', 'V3B'],
        'LO':   ['LO1', 'LO2'],
        'TO':   ['TO1', 'TO2'],
        'VO':   ['VO1', 'VO2'],
    }
    for r in rois:
        for component in aliases.get(r, [r]):
            try:
                roi_img = sub.get_retinotopic_roi(roi=component, bold_space=True)
            except Exception as e:
                print(f'  WARN sub-{sub.subject_id:02d} ROI {component}: {e}')
                continue
            out |= masker_full.transform(roi_img).astype(bool).flatten()
    return out


def main(subject: int,
         bids_folder: str = '/data/ds-retsupp',
         init_model: int = 4,
         resolution: int = 50,
         max_n_iterations: int = 1000,
         r2_thr: float = 0.04,
         save_nifti: bool = False):
    bids_folder = Path(bids_folder)
    sub = Subject(subject, bids_folder)
    out_dir = (bids_folder / 'derivatives' / 'prf_conditionfit'
                / 'model1' / f'sub-{subject:02d}')
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'== sub-{subject:02d}  init from model {init_model}  '
          f'(out: {out_dir}) ==')

    # 1) ROI mask: union of all 8 retinotopic ROIs.
    roi_mask = get_roi_voxel_mask(sub)
    n_roi = int(roi_mask.sum())
    print(f'ROI-union: {n_roi} voxels')

    # 2) Init parameters from model 4 (mean fits across all runs).
    init_full = sub.get_prf_parameters_volume(model=init_model,
                                                return_images=False)
    if not isinstance(init_full, pd.DataFrame):
        init_full = pd.DataFrame(init_full)
    # Keep only Gaussian parameters; drop DoG-specific srf_*.
    init = init_full.loc[roi_mask, ['x', 'y', 'sd', 'amplitude', 'baseline']].copy()
    init = init[['x', 'y', 'sd', 'baseline', 'amplitude']]   # canonical order
    # Filter by mean R² for stability.
    r2_mean = init_full.loc[roi_mask, 'r2'].values
    keep = r2_mean > r2_thr
    init = init.loc[keep].copy()
    # ParameterFitter aligns by index — give init a clean 0..N-1 index
    # so it matches the RangeIndex of the mean_ts DataFrame columns.
    init = init.reset_index(drop=True)
    n_keep = int(keep.sum())
    print(f'After R² > {r2_thr}: {n_keep} ROI voxels kept.')

    # 3) Load BOLD per (session, run), restricted to ROI voxels.
    bold_mask = sub.get_bold_mask()
    masker = input_data.NiftiMasker(mask_img=bold_mask)
    masker.fit()
    hp_per_run = sub.get_hpd_locations()

    # Per-condition stack of (n_T, n_voxels_in_ROI) arrays.
    stacks = {c: [] for c in CONDITIONS_HP}
    session_runs = [(s, r) for s in [1, 2] for r in sub.get_runs(s)]
    n_T_run = 258
    for session, run in tqdm(session_runs, desc='Loading runs'):
        bold_fn = (bids_folder / 'derivatives' / 'cleaned'
                    / f'sub-{subject:02d}' / f'ses-{session}' / 'func'
                    / f'sub-{subject:02d}_ses-{session}_task-search_'
                      f'desc-cleaned_run-{run}_bold.nii.gz')
        data = masker.transform(bold_fn).astype(np.float32)
        if data.shape[0] > n_T_run:
            data = data[:n_T_run]
        elif data.shape[0] < n_T_run:
            pad = np.zeros((n_T_run - data.shape[0], data.shape[1]),
                            dtype=np.float32)
            data = np.vstack([data, pad])
        # Restrict to ROI voxels and to the kept R²>thr subset.
        data_roi = data[:, roi_mask][:, keep]
        hp = hp_per_run[(session, run)]
        if hp not in CONDITIONS_HP:
            print(f'  ses-{session}_run-{run}: HP={hp!r}, skipping')
            continue
        stacks[hp].append(data_roi)

    # 4) Per-condition mean BOLD across runs.
    paradigm = sub.get_stimulus(session=1, run=1, resolution=resolution).astype(np.float32)
    grid = sub.get_grid_coordinates(resolution=resolution, session=1, run=1)
    grid = np.stack((grid[0].ravel(), grid[1].ravel()), axis=1).astype(np.float32)
    paradigm = paradigm.reshape((paradigm.shape[0], -1))
    hrf_model = SPMHRFModel(tr=sub.get_tr(session=1, run=1),
                              delay=4.5, dispersion=0.75)
    prf_model = GaussianPRF2DWithHRF(grid, paradigm, hrf_model=hrf_model)

    rows = []
    voxel_idx_global = np.where(roi_mask)[0][keep]   # index into BOLD voxels
    for cond in CONDITIONS_HP:
        if not stacks[cond]:
            print(f'  no runs for HP={cond}, skipping')
            continue
        mean_ts = np.stack(stacks[cond], axis=0).mean(axis=0)
        # mean_ts: (n_T, n_keep)
        mean_ts_df = pd.DataFrame(mean_ts)
        fitter = ParameterFitter(prf_model, mean_ts_df, paradigm)
        refined = fitter.refine_baseline_and_amplitude(init.copy(),
                                                          l2_alpha=1e-3)
        fit_pars = fitter.fit(init_pars=refined,
                                max_n_iterations=max_n_iterations,
                                learning_rate=0.01)
        r2 = fitter.get_rsq(fit_pars)
        r2_arr = r2.values if hasattr(r2, 'values') else np.asarray(r2)
        for i in range(len(fit_pars)):
            rows.append(dict(
                voxel_idx=int(voxel_idx_global[i]),
                condition=cond,
                x=float(fit_pars['x'].iloc[i]),
                y=float(fit_pars['y'].iloc[i]),
                sd=float(fit_pars['sd'].iloc[i]),
                baseline=float(fit_pars['baseline'].iloc[i]),
                amplitude=float(fit_pars['amplitude'].iloc[i]),
                r2=float(r2_arr[i]),
            ))
        print(f'  HP={cond}: fit {len(fit_pars)} voxels, mean R²={np.nanmean(r2_arr):.3f}')

        if save_nifti:
            # Float32 wrap — see CLAUDE.md §"NIfTI dtype trap".
            n_full = roi_mask.size
            for par in ['x', 'y', 'sd', 'baseline', 'amplitude']:
                arr = np.zeros(n_full)
                arr[voxel_idx_global] = fit_pars[par].values
                img = masker.inverse_transform(arr)
                img.set_data_dtype(np.float32)
                img.header.set_slope_inter(slope=1, inter=0)
                img.to_filename(out_dir / f'sub-{subject:02d}_cond-{cond}_desc-{par}.nii.gz')
            arr = np.zeros(n_full)
            arr[voxel_idx_global] = r2_arr
            img = masker.inverse_transform(arr)
            img.set_data_dtype(np.float32)
            img.header.set_slope_inter(slope=1, inter=0)
            img.to_filename(out_dir / f'sub-{subject:02d}_cond-{cond}_desc-r2.nii.gz')

    df_out = pd.DataFrame(rows)
    out_tsv = out_dir / f'sub-{subject:02d}_gaussian_roi.tsv'
    df_out.to_csv(out_tsv, sep='\t', index=False)
    print(f'Saved {out_tsv} ({len(df_out)} rows)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('subject', type=int)
    parser.add_argument('--bids-folder', default='/data/ds-retsupp')
    parser.add_argument('--init-model', type=int, default=4)
    parser.add_argument('--resolution', type=int, default=50)
    parser.add_argument('--max-n-iterations', type=int, default=1000)
    parser.add_argument('--r2-thr', type=float, default=0.04)
    parser.add_argument('--save-nifti', action='store_true')
    args = parser.parse_args()
    main(args.subject, bids_folder=args.bids_folder,
         init_model=args.init_model, resolution=args.resolution,
         max_n_iterations=args.max_n_iterations, r2_thr=args.r2_thr,
         save_nifti=args.save_nifti)
