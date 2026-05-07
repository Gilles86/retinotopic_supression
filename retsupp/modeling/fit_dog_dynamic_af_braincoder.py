"""Joint Dynamic Attention-Field + DoG-PRF fit at the BOLD signal level.

DoG-voxel-kernel counterpart to
:mod:`retsupp.modeling.fit_dynamic_af_braincoder`. The voxel kernel is
a Difference-of-Gaussians (centre + surround) — same shape as model 4
(``DoG + flexible HRF``) — and the AF modulation is one of the dynamic
variants (v2 or v3).

Two model versions are available, controlled by ``--model-version``.

v2 (default — shared σ, split dyn gain)
---------------------------------------
Uses :class:`braincoder.models.DoGDynamicAttentionFieldPRF2DWithHRF_v2`.

5 shared parameters: ``sigma_AF, g_HP, g_LP, g_HP_dyn, g_LP_dyn``.
The dynamic AF Gaussian uses the SAME ``sigma_AF`` as the sustained
term, and the per-TR phasic gain is split into HP-vs-LP.

v3 (separate σ_dyn AND split dyn gain)
--------------------------------------
Uses :class:`braincoder.models.DoGDynamicAttentionFieldPRF2DWithHRF_v3`.

6 shared parameters: ``sigma_AF, g_HP, g_LP, sigma_dyn, g_HP_dyn,
g_LP_dyn``. Most flexible variant.

Per-voxel free parameters: ``x, y, sd, baseline, amplitude,
srf_amplitude, srf_size`` (initialized from model 4, KEEPING
srf_size/srf_amplitude — mirrors :mod:`fit_dog_af_prf_braincoder`).

This always uses the FULL paradigm (bar + distractor disks at the 4
ring locations).

Output
------
A pickle and a TSV under
``derivatives/af_prf_joint_dynamic_v{2,3}_dog/sub-XX/sub-XX_roi-XX_*.{pkl,tsv}``.
The output filenames also include ``dog-dyn-v2`` / ``dog-dyn-v3`` so
they never collide with the Gaussian-AF dynamic outputs or the static
DoG-AF outputs.

Usage
-----
``python -m retsupp.modeling.fit_dog_dynamic_af_braincoder 2 --roi V3AB``
``python -m retsupp.modeling.fit_dog_dynamic_af_braincoder 2 --roi V3AB --model-version v3``
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
from braincoder.models import (
    DoGDynamicAttentionFieldPRF2DWithHRF_v2,
    DoGDynamicAttentionFieldPRF2DWithHRF_v3,
)
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


def build_data_and_paradigm(sub: Subject, resolution: int = 50,
                            grid_radius: float = 5.0):
    """Load cleaned BOLD + paradigm + condition_indicator + dynamic_indicator.

    Always uses the FULL paradigm (bar + distractor disks). Identical to
    the Gaussian dynamic loader.
    """
    bold_mask = sub.get_bold_mask()
    masker = input_data.NiftiMasker(mask_img=bold_mask)

    gx, gy = sub.get_extended_grid_coordinates(
        resolution=resolution, session=1, run=1, grid_radius=grid_radius,
    )
    grid_coordinates = np.stack(
        (gx.ravel(), gy.ravel()), axis=1,
    ).astype(np.float32)

    hp_per_run = sub.get_hpd_locations()
    print('HP per run:', hp_per_run)

    bold_chunks = []
    paradigm_chunks = []
    cond_indicator_chunks = []
    dyn_indicator_chunks = []
    masker.fit(bold_mask)

    session_runs = [(s, r) for s in [1, 2] for r in sub.get_runs(s)]
    for session, run in tqdm(session_runs, desc='Loading runs'):
        hp = hp_per_run[(session, run)]
        if hp not in CONDITIONS:
            print(f'  ses-{session}_run-{run}: HP={hp!r}, skipping')
            continue

        par_run = sub.get_stimulus_with_distractors(
            session=session, run=run, resolution=resolution,
            grid_radius=grid_radius,
        ).astype(np.float32)
        par_run_flat = par_run.reshape((par_run.shape[0], -1))
        n_T_run = par_run_flat.shape[0]

        bold_fn = (
            sub.bids_folder / 'derivatives' / 'cleaned'
            / f'sub-{sub.subject_id:02d}' / f'ses-{session}' / 'func'
            / f'sub-{sub.subject_id:02d}_ses-{session}_task-search_'
              f'desc-cleaned_run-{run}_bold.nii.gz'
        )
        data = masker.transform(bold_fn).astype(np.float32)
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

        dyn_indicator = sub.get_dynamic_indicator(
            session=session, run=run,
        ).astype(np.float32)
        if dyn_indicator.shape[0] < n_T_run:
            pad = np.zeros((n_T_run - dyn_indicator.shape[0],
                            dyn_indicator.shape[1]), dtype=np.float32)
            dyn_indicator = np.vstack([dyn_indicator, pad])
        elif dyn_indicator.shape[0] > n_T_run:
            dyn_indicator = dyn_indicator[:n_T_run]

        bold_chunks.append(data)
        paradigm_chunks.append(par_run_flat)
        cond_indicator_chunks.append(cond_indicator)
        dyn_indicator_chunks.append(dyn_indicator)

    bold = np.vstack(bold_chunks)
    paradigm_full = np.vstack(paradigm_chunks)
    condition_indicator = np.vstack(cond_indicator_chunks)
    dynamic_indicator = np.vstack(dyn_indicator_chunks)

    print(f'Loaded BOLD: shape {bold.shape}, paradigm {paradigm_full.shape}, '
          f'condition_indicator {condition_indicator.shape}, '
          f'dynamic_indicator {dynamic_indicator.shape} '
          f'(min={dynamic_indicator.min():.3f}, '
          f'max={dynamic_indicator.max():.3f}, '
          f'mean={dynamic_indicator.mean():.4f})')
    return (
        pd.DataFrame(bold),
        paradigm_full,
        condition_indicator,
        dynamic_indicator,
        grid_coordinates,
        masker,
    )


def select_roi_voxels(sub: Subject, roi: str, prf_pars: pd.DataFrame,
                       r2_thr: float = 0.05):
    """Boolean voxel mask: ROI ∧ (mean-model R² > thr)."""
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
         resolution: int = 50,
         max_n_iterations: int = 1500,
         r2_thr: float = 0.05,
         model_label: int = 4,
         max_voxels: int | None = 500,
         mode: str = 'signed',
         learning_rate: float = 0.01,
         grid_radius: float = 5.0,
         output_subdir: str | None = None,
         model_version: str = 'v2'):
    if model_version not in ('v2', 'v3'):
        raise ValueError(
            f"model_version must be 'v2' or 'v3', got {model_version!r}")
    bids_folder = Path(bids_folder)
    sub = Subject(subject, bids_folder)
    if output_subdir is None:
        output_subdir = {
            'v2': 'af_prf_joint_dynamic_v2_dog',
            'v3': 'af_prf_joint_dynamic_v3_dog',
        }[model_version]
    out_dir = bids_folder / 'derivatives' / output_subdir / f'sub-{subject:02d}'
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'== sub-{subject:02d} | roi={roi} | mode={mode} | '
          f'model-version={model_version} | paradigm=full '
          f'(DoG voxel kernel, dynamic AF) ==')

    # 1) Load BOLD + paradigm + condition_indicator + dynamic_indicator.
    (bold_df, paradigm, condition_indicator, dynamic_indicator,
     grid_coords, masker) = build_data_and_paradigm(
        sub,
        resolution=resolution,
        grid_radius=grid_radius,
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
        order = np.argsort(prf_pars['r2'].values)[::-1]
        keep = np.zeros_like(voxel_mask)
        ranked = [v for v in order if voxel_mask[v]][:max_voxels]
        keep[ranked] = True
        voxel_mask = keep
        print(f'  -> top {max_voxels} voxels by r²')

    bold_sub = bold_df.loc[:, voxel_mask].copy()

    # 3) Initialize from model 4 (DoG + flexible HRF). KEEP the
    # srf_amplitude/srf_size columns — that's what makes this DoG fit
    # different from the Gaussian fit.
    init_cols = ['x', 'y', 'sd', 'baseline', 'amplitude',
                 'srf_amplitude', 'srf_size']
    missing = [c for c in init_cols if c not in prf_pars.columns]
    if missing:
        raise RuntimeError(
            f'Mean model {model_label} is missing DoG params {missing}. '
            f'Use model 4 (DoG + flexible HRF) for the init.')
    init_pars = prf_pars.loc[voxel_mask, init_cols].copy()

    # AF inits.
    init_pars['sigma_AF'] = 2.0
    init_pars['g_HP'] = 0.0 if mode == 'signed' else 0.30
    init_pars['g_LP'] = 0.0 if mode == 'signed' else 0.10

    if model_version == 'v2':
        # v2: shared sigma_AF + split HP/LP dynamic gains.
        init_pars['g_HP_dyn'] = 0.0 if mode == 'signed' else 0.10
        init_pars['g_LP_dyn'] = 0.0 if mode == 'signed' else 0.10
        shared_pars = ['sigma_AF', 'g_HP', 'g_LP', 'g_HP_dyn', 'g_LP_dyn']
    else:
        # v3: separate sigma_dyn (smaller default — distractor ~0.4°) AND
        # split HP/LP dynamic gains. 6 shared parameters total.
        init_pars['sigma_dyn'] = 0.5
        init_pars['g_HP_dyn'] = 0.0 if mode == 'signed' else 0.10
        init_pars['g_LP_dyn'] = 0.0 if mode == 'signed' else 0.10
        shared_pars = ['sigma_AF', 'g_HP', 'g_LP',
                       'sigma_dyn', 'g_HP_dyn', 'g_LP_dyn']

    # 4) Build the dynamic DoG-AF + PRF model and the fitter.
    ring_positions = get_ring_positions()  # (4, 2)
    print('Ring positions:\n', ring_positions)

    hrf_model = SPMHRFModel(tr=sub.get_tr(session=1, run=1),
                            delay=4.5, dispersion=0.75)

    if model_version == 'v2':
        ModelCls = DoGDynamicAttentionFieldPRF2DWithHRF_v2
    else:
        ModelCls = DoGDynamicAttentionFieldPRF2DWithHRF_v3

    model = ModelCls(
        grid_coordinates=grid_coords,
        paradigm=paradigm,
        hrf_model=hrf_model,
        condition_indicator=condition_indicator,
        dynamic_indicator=dynamic_indicator,
        ring_positions=ring_positions,
        mode=mode,
    )

    fitter = ParameterFitter(model, bold_sub, paradigm)

    # 5) Refine baseline/amplitude given current AF params.
    refined_pars = fitter.refine_baseline_and_amplitude(init_pars, l2_alpha=1e-3)

    # 6) Joint fit. Shared pars depend on model version.
    fit_pars = fitter.fit(
        init_pars=refined_pars,
        max_n_iterations=max_n_iterations,
        shared_pars=shared_pars,
        learning_rate=learning_rate,
    )

    r2 = fitter.get_rsq(fit_pars) if hasattr(fitter, 'get_rsq') else fitter.r2
    print(f'Mean R²: {np.nanmean(r2):.4f}')
    print('Shared parameters:')
    print(fit_pars[shared_pars].iloc[0])

    fit_pars['r2'] = r2.values if hasattr(r2, 'values') else r2

    # 7) Save outputs.
    dyn_tag = {'v2': 'dog-dyn-v2', 'v3': 'dog-dyn-v3'}[model_version]
    out_tsv = out_dir / (
        f'sub-{subject:02d}_roi-{roi}_mode-{mode}_{dyn_tag}-af-prf-pars.tsv')
    fit_pars.to_csv(out_tsv, sep='\t')
    out_pkl = out_dir / (
        f'sub-{subject:02d}_roi-{roi}_mode-{mode}_{dyn_tag}-af-prf-fit.pkl')
    with open(out_pkl, 'wb') as f:
        pickle.dump({
            'fit_pars': fit_pars,
            'r2': r2,
            'shared_pars': fit_pars[shared_pars].iloc[0].to_dict(),
            'shared_par_labels': shared_pars,
            'voxel_mask_indices': np.where(voxel_mask)[0],
            'mode': mode,
            'roi': roi,
            'resolution': resolution,
            'subject': subject,
            'paradigm_type': 'full',
            'grid_radius': grid_radius,
            'dynamic': True,
            'voxel_kernel': 'DoG',
            'model_version': model_version,
            'model_label_init': model_label,
        }, f)
    print(f'Saved: {out_tsv}')
    print(f'Saved: {out_pkl}')

    return fit_pars, r2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('subject', type=int,
                        help='Subject ID (zero-padded, e.g. 2 -> sub-02)')
    parser.add_argument('--bids-folder', default='/data/ds-retsupp')
    parser.add_argument('--roi', default='V3AB',
                        help='Retinotopic ROI to fit (default V3AB).')
    parser.add_argument('--resolution', type=int, default=50,
                        help='Stimulus grid resolution (default 50).')
    parser.add_argument('--max-n-iterations', type=int, default=1500)
    parser.add_argument('--r2-thr', type=float, default=0.05)
    parser.add_argument('--model-label', type=int, default=4,
                        help='Mean-model PRF used for DoG-init (x, y, sd, '
                             'amplitude, baseline, srf_amplitude, srf_size). '
                             'Model 4 (DoG + flexible HRF) is the canonical '
                             'choice.')
    parser.add_argument('--max-voxels', type=int, default=500,
                        help='Cap on voxels for the POC. Set 0 for no cap.')
    parser.add_argument('--mode',
                        choices=['suppression', 'attraction', 'signed'],
                        default='signed')
    parser.add_argument('--learning-rate', type=float, default=0.01)
    parser.add_argument('--grid-radius', type=float, default=5.0,
                        help='Half-width of the extended grid in deg.')
    parser.add_argument('--output-subdir', default=None,
                        help='Output derivatives subdir. Default: '
                             "'af_prf_joint_dynamic_v2_dog' for v2, "
                             "'af_prf_joint_dynamic_v3_dog' for v3.")
    parser.add_argument('--model-version',
                        choices=['v2', 'v3'], default='v2',
                        help="v2 (default): shared sigma_AF + split HP/LP "
                             "dynamic gains (g_HP_dyn, g_LP_dyn). "
                             "v3: separate sigma_dyn AND split HP/LP "
                             "dynamic gains (6 shared params).")
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
        grid_radius=args.grid_radius,
        output_subdir=args.output_subdir,
        model_version=args.model_version,
    )
