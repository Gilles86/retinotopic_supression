"""Joint Dynamic Attention-Field + Gaussian-PRF fit at the BOLD signal level.

Gaussian-voxel-kernel counterpart to
:mod:`retsupp.modeling.fit_dog_dynamic_af_braincoder`. The voxel kernel
is a plain 2-D Gaussian (no DoG surround) initialised from MODEL 1
(``Gaussian PRF, fixed HRF``), and the AF modulation is the v3 dynamic
variant with phasic-target term and ``sigma_T_dyn`` tied to
``sigma_dyn`` (sharedSigma).

This script was added to sidestep the phantom σ-collapse failure mode
of model 4 (DoG + flexible HRF), which destroys voxel R² for ~all
candidate V1 voxels (0/27 valid in the DoG-AF fits) and inflates
high-r² selection across all ROIs. Model 1 PRFs do not exhibit the
phantom collapse.

Forward model
-------------
Mirrors
:class:`retsupp.modeling.local_models.GaussianDynamicAttentionFieldPRF2DWithHRF_v3_target_sharedSigma`:

    sustained: g_HP · A_HP^sus + g_LP · Σ_LP A_LP^sus     (uses σ_AF)
    phasic-d:  g_HP_dyn · d_HP · A_HP^dyn + g_LP_dyn · Σ_LP d_LP · A_LP^dyn
                                                          (uses σ_dyn)
    phasic-t:  g_T_dyn · Σ_ℓ tgt_ℓ · A_ℓ^tgt              (uses σ_T_dyn := σ_dyn)

Per-voxel parameters (5)
------------------------
``x``, ``y``, ``sd``, ``baseline``, ``amplitude``.

Shared parameters (8)
---------------------
``sigma_AF``, ``g_HP``, ``g_LP``, ``sigma_dyn``, ``g_HP_dyn``,
``g_LP_dyn``, ``g_T_dyn``, ``sigma_T_dyn``.

Output
------
A pickle and a TSV under
``derivatives/af_prf_joint_dynamic_v3_gaussian_with_target_sharedSigma/sub-XX/sub-XX_roi-XX_*.{pkl,tsv}``.

Usage
-----
``python -m retsupp.modeling.fit_gaussian_dynamic_af_braincoder 1 --roi V1``
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from nilearn import input_data
from tqdm import tqdm

from braincoder.hrf import SPMHRFModel
from braincoder.optimize import ParameterFitter
from retsupp.modeling.local_models import (
    GaussianDynamicAttentionFieldPRF2DWithHRF_v3_target_sharedSigma,
    GaussianDynamicAttentionFieldPRF2DWithHRF_v3_target_sharedSigma_sharedDynGain,
    GaussianDynamicAttentionFieldPRF2DWithHRF_v3_target_allSharedSigma,
    GaussianDynamicAttentionFieldPRF2DWithHRF_v3_target_allSharedSigma_sharedDynGain,
)
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
    """Load cleaned BOLD + paradigm + condition_indicator + dynamic_indicator
    + target_indicator. Always uses the FULL paradigm (bar + distractor
    disks). Mirrors the DoG version's loader."""

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
    tgt_indicator_chunks = []
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
            session=session, run=run, oversampling=1,
        ).astype(np.float32)
        if dyn_indicator.shape[0] < n_T_run:
            pad = np.zeros((n_T_run - dyn_indicator.shape[0],
                            dyn_indicator.shape[1]), dtype=np.float32)
            dyn_indicator = np.vstack([dyn_indicator, pad])
        elif dyn_indicator.shape[0] > n_T_run:
            dyn_indicator = dyn_indicator[:n_T_run]

        tgt_indicator = sub.get_target_indicator(
            session=session, run=run, oversampling=1,
        ).astype(np.float32)
        if tgt_indicator.shape[0] < n_T_run:
            pad = np.zeros((n_T_run - tgt_indicator.shape[0],
                            tgt_indicator.shape[1]), dtype=np.float32)
            tgt_indicator = np.vstack([tgt_indicator, pad])
        elif tgt_indicator.shape[0] > n_T_run:
            tgt_indicator = tgt_indicator[:n_T_run]

        bold_chunks.append(data)
        paradigm_chunks.append(par_run_flat)
        cond_indicator_chunks.append(cond_indicator)
        dyn_indicator_chunks.append(dyn_indicator)
        tgt_indicator_chunks.append(tgt_indicator)

    bold = np.vstack(bold_chunks)
    paradigm_full = np.vstack(paradigm_chunks)
    condition_indicator = np.vstack(cond_indicator_chunks)
    dynamic_indicator = np.vstack(dyn_indicator_chunks)
    target_indicator = np.vstack(tgt_indicator_chunks)

    print(f'Loaded BOLD: shape {bold.shape}, paradigm {paradigm_full.shape}, '
          f'condition_indicator {condition_indicator.shape}, '
          f'dynamic_indicator {dynamic_indicator.shape} '
          f'(min={dynamic_indicator.min():.3f}, '
          f'max={dynamic_indicator.max():.3f}, '
          f'mean={dynamic_indicator.mean():.4f})')
    print(f'target_indicator {target_indicator.shape} '
          f'(min={target_indicator.min():.3f}, '
          f'max={target_indicator.max():.3f}, '
          f'mean={target_indicator.mean():.4f})')
    return (
        pd.DataFrame(bold),
        paradigm_full,
        condition_indicator,
        dynamic_indicator,
        target_indicator,
        grid_coordinates,
        masker,
    )


def select_roi_voxels(sub: Subject, roi: str, prf_pars: pd.DataFrame,
                       r2_thr: float = 0.05,
                       r2_max: float = 0.999):
    """Legacy R²-based voxel mask. Use :func:`select_roi_voxels_psignal`
    for the posterior-based (GMM) selector instead.
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
    r2_vals = prf_pars['r2'].values
    r2_mask = (r2_vals > r2_thr) & (r2_vals < r2_max) & roi_arr
    n_phantom = ((r2_vals >= r2_max) & roi_arr).sum()
    if n_phantom > 0:
        print(f'  -> dropping {n_phantom} phantom voxels (R² >= {r2_max})')
    return r2_mask


def select_roi_voxels_psignal(sub: Subject, roi: str,
                               prf_pars: pd.DataFrame, *,
                               model_label: int,
                               p_signal_thr: float,
                               aperture_mass_thr: float = 0.0,
                               aperture_radius: float = 3.17):
    """Boolean voxel mask using the GMM per-voxel ``p_signal`` posterior.

    Mirrors the DoG-AF driver's selector. Drops legacy r2_thr/r2_max
    band-aids — the logit-Gaussian mixture's posterior handles phantom
    and σ-collapsed voxels naturally.
    """
    from scipy.stats import norm

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

    p_sig_path = (sub.bids_folder / 'derivatives' / 'prf'
                   / f'model{model_label}' / f'sub-{sub.subject_id:02d}'
                   / f'sub-{sub.subject_id:02d}_desc-p_signal.nii.gz')
    if not p_sig_path.exists():
        raise FileNotFoundError(
            f'p_signal NIfTI missing for sub-{sub.subject_id:02d} '
            f'model {model_label}: {p_sig_path}. Run r2_mixture first.')
    p_sig = masker_full.transform(str(p_sig_path)).flatten()
    sig_ok = np.isfinite(p_sig) & (p_sig > p_signal_thr)
    mask = roi_arr & sig_ok
    if aperture_mass_thr > 0:
        x = prf_pars['x'].to_numpy()
        y = prf_pars['y'].to_numpy()
        sd = prf_pars['sd'].to_numpy()
        sd_safe = np.where(np.isfinite(sd) & (sd > 0.05), sd, 0.05)
        eccen = np.sqrt(x ** 2 + y ** 2)
        mass_in = 1.0 - norm.cdf((eccen - aperture_radius) / sd_safe)
        mask &= np.isfinite(mass_in) & (mass_in >= aperture_mass_thr)
    return mask


def main(subject: int, bids_folder: str = '/data/ds-retsupp',
         roi: str = 'V1',
         resolution: int = 50,
         max_n_iterations: int = 1500,
         r2_thr: float = 0.05,
         r2_max: float = 0.999,
         p_signal_thr: float = 0.5,
         aperture_mass_thr: float = 0.0,
         model_label: int = 3,
         max_voxels: int | None = 500,
         mode: str = 'signed',
         learning_rate: float = 0.01,
         grid_radius: float = 5.0,
         output_subdir: str | None = None,
         sigma_af_init: float = 2.0,
         sigma_dyn_init: float = 2.0,
         g_t_dyn_init: float = 0.0,
         all_shared_sigma: bool = False,
         shared_dyn_gain: bool = False):
    """Top-level fit driver."""
    bids_folder = Path(bids_folder)
    sub = Subject(subject, bids_folder)
    if output_subdir is None:
        base = 'af_prf_joint_dynamic_v3_gaussian_with_target_sharedSigma'
        if all_shared_sigma:
            base = 'af_prf_joint_dynamic_v3_gaussian_with_target_allSharedSigma'
        if shared_dyn_gain:
            base = f'{base}_sharedDynGain'
        if p_signal_thr > 0:
            base = f'{base}_pSig{p_signal_thr:g}'
            if aperture_mass_thr > 0:
                base = f'{base}_apt{aperture_mass_thr:g}'
        output_subdir = base
    out_dir = (bids_folder / 'derivatives' / output_subdir
               / f'sub-{subject:02d}')
    out_dir.mkdir(parents=True, exist_ok=True)

    variant_tag = 'allSharedSigma' if all_shared_sigma else 'sharedSigma'
    print(f'== sub-{subject:02d} | roi={roi} | mode={mode} | '
          f'Gaussian (m{model_label}) voxel kernel, '
          f'v3 + target + {variant_tag} ==')

    # 1) Load BOLD + paradigm + indicators.
    (bold_df, paradigm, condition_indicator, dynamic_indicator,
     target_indicator, grid_coords, masker) = build_data_and_paradigm(
        sub,
        resolution=resolution,
        grid_radius=grid_radius,
    )

    # 2) Restrict to ROI voxels via the logit-GMM per-voxel p_signal
    #    posterior (canonical as of 2026-05-14). Mirrors the DoG-AF
    #    driver. If p_signal_thr == 0, falls back to the legacy R²-band
    #    selector — kept for backward-compat with older drivers.
    prf_pars = sub.get_prf_parameters_volume(model=model_label,
                                              return_images=False)
    if not isinstance(prf_pars, pd.DataFrame):
        prf_pars = pd.DataFrame(prf_pars)

    if p_signal_thr > 0:
        voxel_mask = select_roi_voxels_psignal(
            sub, roi, prf_pars,
            model_label=model_label,
            p_signal_thr=p_signal_thr,
            aperture_mass_thr=aperture_mass_thr)
        descr = (f'ROI {roi} | p_signal > {p_signal_thr}'
                 + (f' AND mass_in ≥ {aperture_mass_thr}'
                    if aperture_mass_thr > 0 else ''))
        print(f'{descr}: {voxel_mask.sum()} voxels')
        if voxel_mask.sum() == 0:
            raise RuntimeError(f'No voxels survive: {descr}.')
    else:
        effective_r2_thr = r2_thr
        if r2_thr < 0:
            try:
                effective_r2_thr = float(sub.get_r2_fdr_threshold(
                    model=model_label, roi=roi, alpha=0.05))
                print(f'  mixture-FDR threshold (α=0.05) for ROI={roi}: '
                      f'r²>{effective_r2_thr:.4f}')
            except Exception as e:
                raise RuntimeError(
                    f'--r2_thr<0 (FDR mode) but mixture sidecar missing '
                    f'for sub-{sub.subject_id:02d} m{model_label} {roi}: {e}')
        voxel_mask = select_roi_voxels(sub, roi, prf_pars,
                                        r2_thr=effective_r2_thr,
                                        r2_max=r2_max)
        print(f'ROI {roi} | {effective_r2_thr:.4f} < r2 < {r2_max}: '
              f'{voxel_mask.sum()} voxels')
        if voxel_mask.sum() == 0:
            raise RuntimeError(
                f'No voxels survive: ROI={roi}, '
                f'{effective_r2_thr:.4f} < r2 < {r2_max}.')

    if max_voxels is not None and voxel_mask.sum() > max_voxels:
        order = np.argsort(prf_pars['r2'].values)[::-1]
        keep = np.zeros_like(voxel_mask)
        ranked = [v for v in order if voxel_mask[v]][:max_voxels]
        keep[ranked] = True
        voxel_mask = keep
        print(f'  -> top {max_voxels} voxels by r²')

    bold_sub = bold_df.loc[:, voxel_mask].copy()

    # 3) Initialize from model 3 (Gaussian + flexible HRF, canonical).
    #    HRF params are kept fixed during AF (see fitter.fit fixed_pars).
    init_cols = ['x', 'y', 'sd', 'baseline', 'amplitude',
                 'hrf_delay', 'hrf_dispersion']
    missing = [c for c in init_cols if c not in prf_pars.columns]
    if missing:
        raise RuntimeError(
            f'Mean model {model_label} is missing Gaussian/HRF params '
            f'{missing}. Use model 3 (Gaussian + flexible HRF) for the init.'
        )
    init_pars = prf_pars.loc[voxel_mask, init_cols].copy()

    # AF inits.
    init_pars['sigma_AF'] = sigma_af_init
    init_pars['g_HP'] = 0.0 if mode == 'signed' else 0.30
    init_pars['g_LP'] = 0.0 if mode == 'signed' else 0.10
    init_pars['sigma_dyn'] = sigma_dyn_init
    init_pars['g_HP_dyn'] = 0.0 if mode == 'signed' else 0.10
    init_pars['g_LP_dyn'] = 0.0 if mode == 'signed' else 0.10
    init_pars['g_T_dyn'] = g_t_dyn_init
    # sigma_T_dyn := sigma_dyn at init (the model's forward will overwrite
    # slot 12 with slot 8 every iteration anyway, so this just keeps the
    # raw variable in the right place).
    init_pars['sigma_T_dyn'] = init_pars['sigma_dyn']
    if all_shared_sigma:
        # sigma_AF := sigma_dyn at init too; forward transform overwrites
        # slot 5 with slot 8 every iteration.
        init_pars['sigma_AF'] = init_pars['sigma_dyn']
    if shared_dyn_gain:
        # Tie g_LP_dyn := g_HP_dyn at init; model forward enforces it
        # every iteration anyway.
        init_pars['g_LP_dyn'] = init_pars['g_HP_dyn']

    shared_pars = ['sigma_AF', 'g_HP', 'g_LP',
                   'sigma_dyn', 'g_HP_dyn', 'g_LP_dyn',
                   'g_T_dyn', 'sigma_T_dyn']

    # 4) Build the Gaussian sharedSigma model and the fitter.
    ring_positions = get_ring_positions()
    print('Ring positions:\n', ring_positions)

    tr = sub.get_tr(session=1, run=1)
    hrf_model = SPMHRFModel(tr=tr, delay=4.5, dispersion=0.75)

    if all_shared_sigma and shared_dyn_gain:
        model_cls = (
            GaussianDynamicAttentionFieldPRF2DWithHRF_v3_target_allSharedSigma_sharedDynGain
        )
    elif all_shared_sigma:
        model_cls = (
            GaussianDynamicAttentionFieldPRF2DWithHRF_v3_target_allSharedSigma
        )
    elif shared_dyn_gain:
        model_cls = (
            GaussianDynamicAttentionFieldPRF2DWithHRF_v3_target_sharedSigma_sharedDynGain
        )
    else:
        model_cls = (
            GaussianDynamicAttentionFieldPRF2DWithHRF_v3_target_sharedSigma
        )
    model = model_cls(
        grid_coordinates=grid_coords,
        paradigm=paradigm,
        hrf_model=hrf_model,
        flexible_hrf_parameters=True,  # per-voxel HRF from m3; fixed in AF
        condition_indicator=condition_indicator,
        dynamic_indicator=dynamic_indicator,
        target_indicator=target_indicator,
        ring_positions=ring_positions,
        mode=mode,
    )

    fitter = ParameterFitter(model, bold_sub, paradigm)

    # 5) Refine baseline/amplitude given current AF params.
    refined_pars = fitter.refine_baseline_and_amplitude(init_pars,
                                                        l2_alpha=1e-3)

    # 6) Joint fit — HRF held fixed at m3's per-voxel estimates.
    fit_pars = fitter.fit(
        init_pars=refined_pars,
        max_n_iterations=max_n_iterations,
        shared_pars=shared_pars,
        fixed_pars=['hrf_delay', 'hrf_dispersion'],
        learning_rate=learning_rate,
    )

    r2 = fitter.get_rsq(fit_pars) if hasattr(fitter, 'get_rsq') else fitter.r2
    print(f'Mean R²: {np.nanmean(r2):.4f}')
    print('Shared parameters:')
    print(fit_pars[shared_pars].iloc[0])

    fit_pars['r2'] = r2.values if hasattr(r2, 'values') else r2

    # 7) Save outputs.
    if all_shared_sigma:
        dyn_tag = 'gauss-dyn-v3-target-allSharedSigma'
    else:
        dyn_tag = 'gauss-dyn-v3-target-sharedSigma'
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
            'with_target': True,
            'shared_target_sigma': True,
            'shared_all_sigma': bool(all_shared_sigma),
            'voxel_kernel': 'Gaussian',
            'model_version': 'v3',
            'model_label_init': model_label,
            'r2_thr': r2_thr,
            'r2_max': r2_max,
        }, f)
    print(f'Saved: {out_tsv}')
    print(f'Saved: {out_pkl}')

    return fit_pars, r2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('subject', type=int,
                        help='Subject ID (zero-padded, e.g. 2 -> sub-02)')
    parser.add_argument('--bids-folder', default='/data/ds-retsupp')
    parser.add_argument('--roi', default='V1',
                        help='Retinotopic ROI to fit (default V1).')
    parser.add_argument('--resolution', type=int, default=50,
                        help='Stimulus grid resolution (default 50).')
    parser.add_argument('--max-n-iterations', type=int, default=1500)
    parser.add_argument('--r2-thr', type=float, default=0.05,
                        help='Lower R² bound for voxel selection.')
    parser.add_argument('--r2-max', type=float, default=0.999,
                        help='Upper R² bound — drops phantom voxels '
                             '(R² >= r2_max).')
    parser.add_argument('--model-label', type=int, default=1,
                        help='Mean-model PRF used for Gaussian-init '
                             '(x, y, sd, amplitude, baseline). Model 1 '
                             '(Gaussian + fixed HRF) is the canonical '
                             'choice.')
    parser.add_argument('--max-voxels', type=int, default=500,
                        help='Cap on voxels. Set 0 for no cap.')
    parser.add_argument('--mode',
                        choices=['suppression', 'attraction', 'signed'],
                        default='signed')
    parser.add_argument('--learning-rate', type=float, default=0.01)
    parser.add_argument('--grid-radius', type=float, default=5.0,
                        help='Half-width of the extended grid in deg.')
    parser.add_argument('--output-subdir', default=None,
                        help='Output derivatives subdir. Default: '
                             "'af_prf_joint_dynamic_v3_gaussian_with_target"
                             "_sharedSigma'.")
    parser.add_argument('--sigma-af-init', type=float, default=2.0)
    parser.add_argument('--sigma-dyn-init', type=float, default=2.0)
    parser.add_argument('--g-t-dyn-init', type=float, default=0.0)
    parser.add_argument('--all-shared-sigma', action='store_true',
                        help='Tie all three Gaussian widths (sigma_AF, '
                             'sigma_dyn, sigma_T_dyn) to a single shared '
                             'parameter (= sigma_dyn). Stricter than the '
                             'default sharedSigma (sigma_T_dyn := '
                             'sigma_dyn only). Outputs go to '
                             '..._allSharedSigma/ unless --output-subdir '
                             'is set.')
    parser.add_argument('--shared-dyn-gain', action='store_true',
                        help='Tie g_LP_dyn := g_HP_dyn so the phasic-'
                             'distractor transient has one gain '
                             'regardless of HP/LP location.')
    parser.add_argument('--p-signal-thr', type=float, default=0.5,
                        help='Posterior-based voxel selector. Keep voxels '
                             'with P(signal | R²) > this (canonical 0.5).')
    parser.add_argument('--aperture-mass-thr', type=float, default=0.0,
                        help='Optional: require ≥ this fraction of PRF '
                             'Gaussian mass inside the bar aperture.')
    args = parser.parse_args()
    main(
        args.subject,
        bids_folder=args.bids_folder,
        roi=args.roi,
        resolution=args.resolution,
        max_n_iterations=args.max_n_iterations,
        r2_thr=args.r2_thr,
        r2_max=args.r2_max,
        p_signal_thr=args.p_signal_thr,
        aperture_mass_thr=args.aperture_mass_thr,
        model_label=args.model_label,
        max_voxels=None if args.max_voxels == 0 else args.max_voxels,
        mode=args.mode,
        learning_rate=args.learning_rate,
        grid_radius=args.grid_radius,
        output_subdir=args.output_subdir,
        sigma_af_init=args.sigma_af_init,
        sigma_dyn_init=args.sigma_dyn_init,
        g_t_dyn_init=args.g_t_dyn_init,
        all_shared_sigma=args.all_shared_sigma,
        shared_dyn_gain=args.shared_dyn_gain,
    )
