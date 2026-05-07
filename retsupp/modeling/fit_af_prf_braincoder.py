"""Joint Attention-Field + PRF fit at the BOLD signal level.

This script wires up the new
:class:`braincoder.models.AttentionFieldPRF2DWithHRF` to one retsupp
subject's data: it loads cleaned BOLD across all runs (with per-run HP
condition labels), constructs a single concatenated paradigm with a
matching ``condition_indicator`` array, initializes from the existing
mean-model PRF parameters, and jointly fits per-voxel ``(x, y, sd,
baseline, amplitude)`` together with shared ``(sigma_AF, g_HP, g_LP)``.

This is a proof-of-concept rather than the full validation pipeline:
- Restricts to a small ROI by default (``--roi V3AB``) for speed.
- Fits a single ROI on the volume; no surface projection.
- No multi-start; uses the mean-model PRF as initialization.
- The 4 condition AFs are at the fixed ring positions (``ecc=4°``
  diagonal). The "rotated-frame trick" is implicit in the model: the
  shared ``g_HP`` parameter is the modulation amplitude at *whichever*
  ring position is the HP for the current condition; ``g_LP`` is the
  amplitude at the three other ring positions. So all four conditions
  are collapsed onto two free amplitudes via the rotation symmetry.

Output
------
A pickle and a TSV under
``derivatives/af_prf_joint/sub-XX/sub-XX_roi-XX_*.{pkl,tsv}`` containing
the estimated parameters, R², and the shared ``(sigma_AF, g_HP, g_LP)``.

Usage
-----
``python -m retsupp.modeling.fit_af_prf_braincoder 2 --roi V3AB``
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from nilearn import image, input_data
from tqdm import tqdm

from braincoder.hrf import SPMHRFModel
from braincoder.models import AttentionFieldPRF2DWithHRF
from braincoder.optimize import ParameterFitter
from retsupp.utils.data import (
    Subject,
    distractor_locations,
)


CONDITIONS = ['upper_right', 'upper_left', 'lower_left', 'lower_right']


def get_ring_positions():
    """Return the (n_C, 2) array of ring positions matching CONDITIONS order."""
    keys = ['upper right', 'upper left', 'lower left', 'lower right']
    return np.array([list(distractor_locations[k]) for k in keys], dtype=np.float32)


def build_data_and_paradigm(sub: Subject, resolution: int = 40,
                            paradigm_type: str = 'bar',
                            grid_radius: float = 5.0):
    """Load cleaned BOLD + paradigm + condition_indicator for one subject.

    Parameters
    ----------
    sub : Subject
    resolution : int
        Grid resolution per side.
    paradigm_type : {'bar', 'full'}
        - ``'bar'``: reuse a single bar paradigm across all runs (legacy).
        - ``'full'``: build a per-run stimulus that includes distractor
          disks via :meth:`Subject.get_stimulus_with_distractors`. The
          grid is the EXTENDED grid spanning ``±grid_radius`` so the 4°
          ring distractors fit inside.
    grid_radius : float
        Half-width of the extended grid in degrees (only used when
        ``paradigm_type='full'``).

    Returns
    -------
    bold_df : DataFrame, shape (n_T_total, n_voxels)
        Cleaned BOLD timecourses concatenated across runs/sessions.
    paradigm : ndarray, shape (n_T_total, n_grid)
        Stimulus (bar or bar+distractors), flattened over the
        (resolution, resolution) grid.
    condition_indicator : ndarray, shape (n_T_total, n_conditions)
        One-hot encoding of HP condition per timepoint.
    grid_coordinates : ndarray, shape (n_grid, 2)
    masker : NiftiMasker  (fitted, used to invert parameters back to volumes)
    """
    bold_mask = sub.get_bold_mask()
    masker = input_data.NiftiMasker(mask_img=bold_mask)

    if paradigm_type == 'bar':
        # Pre-compute one paradigm we'll reuse (assume identical across runs).
        paradigm_bar = sub.get_stimulus(
            session=1, run=1, resolution=resolution,
        ).astype(np.float32)
        gx, gy = sub.get_grid_coordinates(
            resolution=resolution, session=1, run=1,
        )
        grid_coordinates = np.stack(
            (gx.ravel(), gy.ravel()), axis=1,
        ).astype(np.float32)
        paradigm_flat_default = paradigm_bar.reshape(
            (paradigm_bar.shape[0], -1),
        )
        n_T_default = paradigm_flat_default.shape[0]
    elif paradigm_type == 'full':
        # Extended grid coordinates (shared across runs).
        gx, gy = sub.get_extended_grid_coordinates(
            resolution=resolution, session=1, run=1,
            grid_radius=grid_radius,
        )
        grid_coordinates = np.stack(
            (gx.ravel(), gy.ravel()), axis=1,
        ).astype(np.float32)
        paradigm_flat_default = None
        n_T_default = None
    else:
        raise ValueError(
            f"paradigm_type must be 'bar' or 'full', got {paradigm_type!r}"
        )

    # HP condition per (session, run).
    hp_per_run = sub.get_hpd_locations()
    print('HP per run:', hp_per_run)

    bold_chunks = []
    paradigm_chunks = []
    cond_indicator_chunks = []
    masker.fit(bold_mask)

    session_runs = [(s, r) for s in [1, 2] for r in sub.get_runs(s)]
    for session, run in tqdm(session_runs, desc='Loading runs'):
        # HP for this run — skip non-distractor / unknown runs early.
        hp = hp_per_run[(session, run)]
        if hp not in CONDITIONS:
            print(f'  ses-{session}_run-{run}: HP={hp!r}, skipping')
            continue

        # Build run-specific paradigm if 'full'; reuse default if 'bar'.
        if paradigm_type == 'full':
            par_run = sub.get_stimulus_with_distractors(
                session=session, run=run, resolution=resolution,
                grid_radius=grid_radius,
            ).astype(np.float32)
            par_run_flat = par_run.reshape((par_run.shape[0], -1))
        else:
            par_run_flat = paradigm_flat_default
        n_T_run = par_run_flat.shape[0]

        bold_fn = (
            sub.bids_folder / 'derivatives' / 'cleaned'
            / f'sub-{sub.subject_id:02d}' / f'ses-{session}' / 'func'
            / f'sub-{sub.subject_id:02d}_ses-{session}_task-search_'
              f'desc-cleaned_run-{run}_bold.nii.gz'
        )
        data = masker.transform(bold_fn).astype(np.float32)
        # Crop or pad to n_T_run.
        if data.shape[0] < n_T_run:
            print(f'  ses-{session}_run-{run}: short by '
                  f'{n_T_run - data.shape[0]} TRs, padding with zeros')
            pad = np.zeros((n_T_run - data.shape[0], data.shape[1]),
                           dtype=np.float32)
            data = np.vstack([data, pad])
        elif data.shape[0] > n_T_run:
            data = data[:n_T_run]

        hp_idx = CONDITIONS.index(hp)
        cond_indicator = np.zeros((n_T_run, len(CONDITIONS)),
                                  dtype=np.float32)
        cond_indicator[:, hp_idx] = 1.0

        bold_chunks.append(data)
        paradigm_chunks.append(par_run_flat)
        cond_indicator_chunks.append(cond_indicator)

    bold = np.vstack(bold_chunks)                          # (T_total, V)
    paradigm_full = np.vstack(paradigm_chunks)             # (T_total, G)
    condition_indicator = np.vstack(cond_indicator_chunks) # (T_total, n_C)

    print(f'Loaded BOLD: shape {bold.shape}, paradigm {paradigm_full.shape}, '
          f'condition_indicator {condition_indicator.shape}')
    return (
        pd.DataFrame(bold),
        paradigm_full,
        condition_indicator,
        grid_coordinates,
        masker,
    )


def select_roi_voxels(sub: Subject, roi: str, prf_pars: pd.DataFrame,
                       r2_thr: float = 0.05):
    """Return a boolean mask over voxels selecting `roi` voxels with PRF R² > thr.

    Some "ROIs" in retsupp combine multiple atlas labels (e.g. V3AB =
    V3A ∪ V3B, LO = LO1 ∪ LO2, TO = TO1 ∪ TO2, VO = VO1 ∪ VO2). We expand
    these aliases here.
    """
    roi_aliases = {
        'V3AB': ['V3A', 'V3B'],
        'LO': ['LO1', 'LO2'],
        'TO': ['TO1', 'TO2'],
        'VO': ['VO1', 'VO2'],
    }
    component_rois = roi_aliases.get(roi, [roi])

    masker_full = sub.get_bold_mask(return_masker=True)
    masker_full.fit()
    roi_arr = np.zeros(prf_pars.shape[0], dtype=bool)
    for r in component_rois:
        roi_img = sub.get_retinotopic_roi(roi=r, bold_space=True)
        roi_arr |= masker_full.transform(roi_img).astype(bool).flatten()
    r2_mask = (prf_pars['r2'].values > r2_thr) & roi_arr
    return r2_mask


def main(subject: int, bids_folder: str = '/data/ds-retsupp',
         roi: str = 'V3AB',
         resolution: int = 40,
         max_n_iterations: int = 1500,
         r2_thr: float = 0.05,
         model_label: int = 1,
         max_voxels: int | None = 500,
         mode: str = 'suppression',
         learning_rate: float = 0.01,
         paradigm_type: str = 'bar',
         grid_radius: float = 5.0,
         output_subdir: str | None = None):
    bids_folder = Path(bids_folder)
    sub = Subject(subject, bids_folder)
    if output_subdir is None:
        output_subdir = (
            'af_prf_joint' if paradigm_type == 'bar' else 'af_prf_joint_full'
        )
    out_dir = bids_folder / 'derivatives' / output_subdir / f'sub-{subject:02d}'
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'== sub-{subject:02d} | roi={roi} | mode={mode} | '
          f'paradigm={paradigm_type} ==')

    # 1) Load BOLD + paradigm + condition_indicator across all runs.
    bold_df, paradigm, condition_indicator, grid_coords, masker = (
        build_data_and_paradigm(
            sub,
            resolution=resolution,
            paradigm_type=paradigm_type,
            grid_radius=grid_radius,
        )
    )

    # 2) Restrict to ROI voxels with decent mean-model PRF R².
    prf_pars = sub.get_prf_parameters_volume(model=model_label, return_images=False)
    if not isinstance(prf_pars, pd.DataFrame):
        prf_pars = pd.DataFrame(prf_pars)
    voxel_mask = select_roi_voxels(sub, roi, prf_pars, r2_thr=r2_thr)
    print(f'ROI {roi} | r2>{r2_thr}: {voxel_mask.sum()} voxels')
    if voxel_mask.sum() == 0:
        raise RuntimeError(f'No voxels survive: ROI={roi}, r2>{r2_thr}.')

    if max_voxels is not None and voxel_mask.sum() > max_voxels:
        # Take the top-r² voxels for the proof-of-concept.
        order = np.argsort(prf_pars['r2'].values)[::-1]
        keep = np.zeros_like(voxel_mask)
        ranked = [v for v in order if voxel_mask[v]][:max_voxels]
        keep[ranked] = True
        voxel_mask = keep
        print(f'  -> top {max_voxels} voxels by r²')

    bold_sub = bold_df.loc[:, voxel_mask].copy()
    init_pars = prf_pars.loc[voxel_mask, ['x', 'y', 'sd', 'amplitude', 'baseline']].copy()
    # Remap parameter labels and add the AF parameters with sensible inits.
    init_pars = init_pars[['x', 'y', 'sd', 'baseline', 'amplitude']]
    init_pars['sigma_AF'] = 2.0
    init_pars['g_HP'] = 0.30
    init_pars['g_LP'] = 0.10

    # 3) Build the AF+PRF model and the fitter.
    ring_positions = get_ring_positions()  # (4, 2)
    print('Ring positions:\n', ring_positions)

    hrf_model = SPMHRFModel(tr=sub.get_tr(session=1, run=1), delay=4.5, dispersion=0.75)

    model = AttentionFieldPRF2DWithHRF(
        grid_coordinates=grid_coords,
        paradigm=paradigm,
        hrf_model=hrf_model,
        condition_indicator=condition_indicator,
        ring_positions=ring_positions,
        mode=mode,
    )

    fitter = ParameterFitter(model, bold_sub, paradigm)

    # 4) Refine baseline/amplitude given current AF params (=0 effect at init? no —
    # init g_HP=0.3 already modulates predictions). Do a quick OLS pass.
    refined_pars = fitter.refine_baseline_and_amplitude(init_pars, l2_alpha=1e-3)

    # 5) Joint fit. sigma_AF, g_HP, g_LP shared across all voxels.
    fit_pars = fitter.fit(
        init_pars=refined_pars,
        max_n_iterations=max_n_iterations,
        shared_pars=['sigma_AF', 'g_HP', 'g_LP'],
        learning_rate=learning_rate,
    )

    r2 = fitter.get_rsq(fit_pars) if hasattr(fitter, 'get_rsq') else fitter.r2
    print(f'Mean R²: {np.nanmean(r2):.4f}')
    print('Shared parameters:')
    print(fit_pars[['sigma_AF', 'g_HP', 'g_LP']].iloc[0])

    fit_pars['r2'] = r2.values if hasattr(r2, 'values') else r2

    # 6) Save outputs.
    out_tsv = out_dir / f'sub-{subject:02d}_roi-{roi}_mode-{mode}_af-prf-pars.tsv'
    fit_pars.to_csv(out_tsv, sep='\t')
    out_pkl = out_dir / f'sub-{subject:02d}_roi-{roi}_mode-{mode}_af-prf-fit.pkl'
    with open(out_pkl, 'wb') as f:
        pickle.dump({
            'fit_pars': fit_pars,
            'r2': r2,
            'shared_pars': fit_pars[['sigma_AF', 'g_HP', 'g_LP']].iloc[0].to_dict(),
            'voxel_mask_indices': np.where(voxel_mask)[0],
            'mode': mode,
            'roi': roi,
            'resolution': resolution,
            'subject': subject,
            'paradigm_type': paradigm_type,
            'grid_radius': grid_radius,
        }, f)
    print(f'Saved: {out_tsv}')
    print(f'Saved: {out_pkl}')

    return fit_pars, r2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('subject', type=int, help='Subject ID (zero-padded, e.g. 2 -> sub-02)')
    parser.add_argument('--bids-folder', default='/data/ds-retsupp')
    parser.add_argument('--roi', default='V3AB',
                        help='Retinotopic ROI to fit (default V3AB).')
    parser.add_argument('--resolution', type=int, default=50,
                        help='Stimulus grid resolution (default 50, '
                             'cluster-friendly).')
    parser.add_argument('--max-n-iterations', type=int, default=1500)
    parser.add_argument('--r2-thr', type=float, default=0.05)
    parser.add_argument('--model-label', type=int, default=1,
                        help='Mean-model PRF model used for initialization.')
    parser.add_argument('--max-voxels', type=int, default=500,
                        help='Cap on voxels for the POC (default 500). '
                             'Set 0 for no cap.')
    parser.add_argument('--mode', choices=['suppression', 'attraction'],
                        default='suppression')
    parser.add_argument('--learning-rate', type=float, default=0.01)
    parser.add_argument('--paradigm-type', choices=['bar', 'full'],
                        default='bar',
                        help="Stimulus paradigm: 'bar' (default, legacy) "
                             "or 'full' (extended grid + distractor disks).")
    parser.add_argument('--grid-radius', type=float, default=5.0,
                        help='Half-width of the extended grid in deg '
                             "(only used when --paradigm-type=full).")
    parser.add_argument('--output-subdir', default=None,
                        help='Output derivatives subdir. Default: '
                             "'af_prf_joint' (bar) / 'af_prf_joint_full' (full).")
    args = parser.parse_args()
    main(
        args.subject,
        bids_folder=args.bids_folder,
        roi=args.roi,
        resolution=args.resolution,
        max_n_iterations=args.max_n_iterations,
        r2_thr=args.r2_thr,
        model_label=args.model_label,
        max_voxels=None if args.max_voxels == 0 else args.max_voxels,
        mode=args.mode,
        learning_rate=args.learning_rate,
        paradigm_type=args.paradigm_type,
        grid_radius=args.grid_radius,
        output_subdir=args.output_subdir,
    )
