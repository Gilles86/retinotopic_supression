"""Per-run stimulus decode from m4 PRF parameters (canonical entrypoint).

Reads m4 (DoG + flex-HRF) PRF parameters for a single ROI (default V1)
directly from the volumetric NIfTIs at
``derivatives/prf/model{M}/sub-{NN}/`` via :meth:`Subject.get_prf_roi_pars`,
loads the cleaned BOLD for that subject from
``derivatives/cleaned_bold_cache/sub-{NN}/sub-{NN}_kind-full_res-50.npz``
(the canonical concat across all 12 runs that the PRF fit consumed),
slices to the requested run, builds the matching bar-only paradigm
via :meth:`Subject.get_bar_stimulus`, and decodes the per-TR stimulus
with braincoder's StimulusFitter.

Per-run decoding keeps trialwise structure available for later
analyses (e.g. distractor-locked modulation); per-HP / per-session
averages can be reconstructed post-hoc by averaging the per-run NPZs.

Voxel filter is **top-N by r² only**. The m4 warmstart fits enforce
``sd_min=0.2`` at the model level, so phantom-like degenerate fits
don't exist and FDR / mass-in-aperture / σ-bound guards are
unnecessary.

Defaults are calibrated; do not change without reading the comment
on ``--l2-norm`` below.

Usage::

    python -m retsupp.decode.decode --subject 23 --session 1 --run 1

Output::

    <bids>/derivatives/decoded/model{M}/sub-{NN}/decoded_ses-{S}_run-{R}_roi-{ROI}_vox{N}[_t].npz
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from retsupp.utils.data import Subject, validate_prf_parameters


PARS_BY_MODEL = {
    4: ['x', 'y', 'sd', 'baseline', 'amplitude',
        'srf_amplitude', 'srf_size', 'hrf_delay', 'hrf_dispersion'],
    6: ['x', 'y', 'sd',
        'rf_amplitude', 'srf_amplitude', 'srf_size',
        'neural_baseline', 'surround_baseline', 'bold_baseline',
        'hrf_delay', 'hrf_dispersion'],
}
SD_MIN_BY_MODEL = {4: 0.2, 6: 0.2}
N_TR_PER_RUN = 258


def _make_model(model: int, *, grid_coordinates, paradigm, hrf_model,
                data, parameters, sd_min: float):
    """Build the braincoder PRF model corresponding to ``model``."""
    from braincoder.models import (
        DifferenceOfGaussiansPRF2DWithHRF,
        DivisiveNormalizationGaussianPRF2DWithHRF,
    )
    cls = {4: DifferenceOfGaussiansPRF2DWithHRF,
           6: DivisiveNormalizationGaussianPRF2DWithHRF}[model]
    return cls(grid_coordinates=grid_coordinates, paradigm=paradigm,
               hrf_model=hrf_model, data=data, parameters=parameters,
               flexible_hrf_parameters=True, sd_min=sd_min)


def _load_voxels_and_bold(sub: Subject, *, roi: str, model: int,
                           max_voxels: int, resolution: int,
                           voxel_filter: str = 'top_r2',
                           psignal_posterior: float = 0.5):
    """Load per-ROI-voxel PRF params + BOLD for this subject.

    Reads PRF parameters from the canonical m4 NIfTIs via
    :meth:`Subject.get_prf_roi_pars` and BOLD from the per-subject
    cleaned BOLD cache npz (same one the PRF fit consumed). No
    handoff NPZ involved.

    ``voxel_filter``:
      - ``'top_r2'`` (default): top ``max_voxels`` by r² (clusters
        centrally → poor coverage of the 4° distractor corners).
      - ``'p_signal'``: r² > p_signal mixture threshold for the
        ROI (``posterior=psignal_posterior``), then optionally
        capped at ``max_voxels`` (set ``max_voxels`` huge to keep
        all signal voxels). Gives broader coverage including
        peripheral voxels.
    """
    # PRF parameters for the ROI (per-voxel, in ROI-mask C-order).
    prf_roi = sub.get_prf_roi_pars(roi=roi, model=model)
    finite = np.all(np.isfinite(prf_roi.values), axis=1) & (prf_roi['r2'] > 0)
    keep_v1_pos = np.where(finite.values)[0]
    print(f'  {roi} voxels with finite m{model} fit: {len(keep_v1_pos)} / {len(prf_roi)}',
          flush=True)

    # Map V1 voxels to brain-mask column indices used by the cache.
    brain_mask = np.asarray(
        sub.get_bold_mask().get_fdata(), dtype=bool).ravel()
    v1_3d = np.asarray(
        sub.get_retinotopic_roi(roi=roi, bold_space=True).get_fdata(),
        dtype=bool).ravel()
    v1_within_brain = v1_3d[brain_mask]
    if int(v1_within_brain.sum()) != len(prf_roi):
        # ROI extends slightly outside the BOLD brain mask. Filter the
        # PRF table down to the ROI ∩ brain-mask voxels (preserves order).
        roi_pos_flat = np.where(v1_3d)[0]
        in_brain_within_roi = brain_mask[roi_pos_flat]
        if int(in_brain_within_roi.sum()) != int(v1_within_brain.sum()):
            raise RuntimeError(
                f'{roi} mask alignment failure: cache pool has '
                f'{int(v1_within_brain.sum())} {roi} voxels, PRF cache has '
                f'{len(prf_roi)}, ROI∩brain-mask gives '
                f'{int(in_brain_within_roi.sum())} — cannot reconcile.')
        print(f'  ROI({len(prf_roi)}) extends beyond BOLD brain mask; '
              f'keeping {int(in_brain_within_roi.sum())} voxels in '
              f'ROI ∩ brain-mask intersection', flush=True)
        prf_roi = prf_roi.loc[in_brain_within_roi].reset_index(drop=True)
        finite = np.all(np.isfinite(prf_roi.values), axis=1) & (prf_roi['r2'] > 0)
        keep_v1_pos = np.where(finite.values)[0]
    v1_cache_cols = np.where(v1_within_brain)[0]

    # Cleaned BOLD cache (same one fit_prf consumed). Must exist.
    cache_path = (sub.bids_folder / 'derivatives' / 'cleaned_bold_cache'
                  / f'sub-{sub.subject_id:02d}'
                  / f'sub-{sub.subject_id:02d}_kind-full_res-{resolution}.npz')
    if not cache_path.exists():
        raise FileNotFoundError(
            f'Cleaned BOLD cache missing: {cache_path}\n'
            f'Build with: sbatch retsupp/modeling/slurm_jobs/build_cleaned_bold_cache.sh')
    print(f'  Loading {cache_path.name}...', flush=True)
    with np.load(cache_path) as f:
        bold_all = f['bold'][:, v1_cache_cols].astype(np.float32)
    print(f'  BOLD shape: {bold_all.shape}  (concat across runs)', flush=True)

    r2_in_v1 = prf_roi['r2'].to_numpy()
    finite_mask = finite.values

    if voxel_filter == 'p_signal':
        r2_thr = sub.get_r2_threshold(model=model, roi=roi,
                                        posterior=psignal_posterior)
        if not np.isfinite(r2_thr):
            raise RuntimeError(
                f'No p_signal threshold for sub-{sub.subject_id:02d} '
                f'roi={roi} model={model}. Run compute_r2_mixture first.')
        keep_mask = finite_mask & (r2_in_v1 > r2_thr)
        sel_within_v1 = np.where(keep_mask)[0]
        # If too many for compute budget, cap to top-max_voxels by r².
        if len(sel_within_v1) > max_voxels:
            order = np.argsort(-r2_in_v1[sel_within_v1])[:max_voxels]
            sel_within_v1 = sel_within_v1[order]
        print(f'  Selected voxels: {len(sel_within_v1)} / {len(prf_roi)}  '
              f'(p_signal>{psignal_posterior} -> r2>{r2_thr:.4f}, '
              f'capped at {max_voxels})', flush=True)
    elif voxel_filter == 'top_r2':
        sel_within_v1 = np.argsort(-np.where(finite_mask, r2_in_v1, -np.inf))[:max_voxels]
        sel_within_v1 = sel_within_v1[finite_mask[sel_within_v1]]
        print(f'  Selected voxels: {len(sel_within_v1)} / {len(prf_roi)}  '
              f'(top-{max_voxels} by r²)', flush=True)
    else:
        raise ValueError(f'Unknown voxel_filter: {voxel_filter!r}')

    if len(sel_within_v1) < 20:
        raise RuntimeError(f'Too few voxels selected: {len(sel_within_v1)}')

    pars_sel = prf_roi.iloc[sel_within_v1].reset_index(drop=True)
    bold_sel_all_runs = bold_all[:, sel_within_v1]
    return pars_sel, bold_sel_all_runs, sel_within_v1


def _slice_run(bold_all: np.ndarray, sub: Subject,
                session: int, run: int) -> np.ndarray:
    """Slice the concat BOLD down to a single (session, run)."""
    row = 0
    for s in (1, 2):
        for r in sub.get_runs(s):
            if (s, r) == (session, run):
                return bold_all[row * N_TR_PER_RUN:(row + 1) * N_TR_PER_RUN]
            row += 1
    raise KeyError(f'(ses={session}, run={run}) not in concat order for '
                   f'sub-{sub.subject_id:02d}')


def decode(*, subject: int, session: int, run: int, model: int,
           roi: str,
           bids_folder: str,
           l2_norm: float, learning_rate: float,
           max_n_iterations: int, min_n_iterations: int,
           resid_max_iter: int, max_voxels: int,
           noise_dist: str,
           out_path: Path,
           resolution: int = 50,
           grid_radius: float = 5.0,
           tr: float = 1.6,
           voxel_filter: str = 'top_r2',
           psignal_posterior: float = 0.5):
    from braincoder.hrf import SPMHRFModel
    from braincoder.optimize import ResidualFitter, StimulusFitter

    pars_list = PARS_BY_MODEL[model]
    sd_min = SD_MIN_BY_MODEL[model]
    sub = Subject(subject, bids_folder=bids_folder)

    pars_sel, bold_all, voxel_positions = _load_voxels_and_bold(
        sub, roi=roi, model=model, max_voxels=max_voxels, resolution=resolution,
        voxel_filter=voxel_filter, psignal_posterior=psignal_posterior)

    print(f'  PRF stats:', flush=True)
    print(f'    sd: median {pars_sel.sd.median():.3f}  q05/95 '
          f'{pars_sel.sd.quantile(.05):.3f}/{pars_sel.sd.quantile(.95):.3f}',
          flush=True)
    print(f'    r²: min {pars_sel.r2.min():.3f}  median {pars_sel.r2.median():.3f}  '
          f'max {pars_sel.r2.max():.3f}', flush=True)

    pars_df = pars_sel[pars_list].astype(np.float32)
    validate_prf_parameters(pars_df, sd_min=sd_min, model_label=f'm{model}',
                             source=f'sub-{subject:02d} {roi} m{model} cache')

    # BOLD + paradigm for this single run.
    bold_run = _slice_run(bold_all, sub, session=session, run=run)
    paradigm = sub.get_bar_stimulus(
        session=session, run=run,
        resolution=resolution, grid_radius=grid_radius).astype(np.float32)
    T = paradigm.shape[0]
    print(f'  BOLD: {bold_run.shape}  paradigm: {paradigm.shape}', flush=True)

    gx, gy = sub.get_extended_grid_coordinates(
        resolution=resolution, session=session, run=run, grid_radius=grid_radius)
    grid = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float32)

    gx, gy = sub.get_extended_grid_coordinates(
        resolution=resolution, session=session, run=run,
        grid_radius=grid_radius)
    grid = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float32)

    paradigm_flat = paradigm.reshape(T, -1).astype(np.float32)
    bold_df = pd.DataFrame(bold_run.astype(np.float32),
                            index=pd.Index(np.arange(T), name='frame'))
    bold_df.columns.name = 'voxel'
    paradigm_df = pd.DataFrame(paradigm_flat,
                                index=pd.Index(np.arange(T), name='frame'))

    hrf = SPMHRFModel(tr=tr, delay=4.5, dispersion=0.75)

    def mk_model():
        return _make_model(model, grid_coordinates=grid,
                            paradigm=paradigm_flat, hrf_model=hrf,
                            data=bold_df, parameters=pars_df, sd_min=sd_min)

    print(f'Fitting residual covariance (noise_dist={noise_dist})...',
          flush=True)
    t0 = time.time()
    rf = ResidualFitter(model=mk_model(), data=bold_df, paradigm=paradigm_df,
                        parameters=pars_df)
    omega, dof = rf.fit(max_n_iterations=resid_max_iter, progressbar=True,
                         method=noise_dist)
    print(f'  omega: {omega.shape}  dof: {dof}  ({time.time() - t0:.0f}s)',
          flush=True)

    print('Decoding stimulus...', flush=True)
    t0 = time.time()
    sf = StimulusFitter(model=mk_model(), data=bold_df, omega=omega,
                        parameters=pars_df, dof=dof)
    decoded = sf.fit(l2_norm=l2_norm, learning_rate=learning_rate,
                     max_n_iterations=max_n_iterations,
                     min_n_iterations=min_n_iterations, progressbar=True)
    decoded_arr = decoded.values.reshape(T, resolution, resolution).astype(np.float32)
    print(f'  decoded: {decoded_arr.shape}  ({time.time() - t0:.0f}s)',
          flush=True)

    hp_loc = sub.get_hpd_locations().get((session, run))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        decoded=decoded_arr,
        paradigm=paradigm,
        grid=grid,
        subject=np.int32(subject),
        roi=np.array(roi),
        session=np.int32(session),
        run=np.int32(run),
        model=np.int32(model),
        l2_norm=np.float32(l2_norm), learning_rate=np.float32(learning_rate),
        resolution=np.int32(resolution),
        grid_radius=np.float32(grid_radius),
        tr=np.float32(tr),
        noise_dist=np.array(noise_dist),
        dof=np.float32(dof) if dof is not None else np.float32(np.nan),
        n_voxels=np.int32(len(pars_df)),
        hp_location=np.array(str(hp_loc) if hp_loc is not None else 'unknown'),
    )
    print(f'Wrote: {out_path}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--subject', type=int, required=True)
    p.add_argument('--session', type=int, required=True)
    p.add_argument('--run', type=int, required=True)
    p.add_argument('--roi', default='V1',
                   help='Retinotopic ROI to decode from (default V1).')
    p.add_argument('--model', type=int, default=4, choices=sorted(PARS_BY_MODEL),
                   help='PRF model: 4 = DoG+HRF, 6 = Divisive Normalization+HRF.')
    p.add_argument('--bids-folder', default='/data/ds-retsupp')
    p.add_argument('--resolution', type=int, default=50,
                   help='Stimulus / cache grid resolution (default 50; must '
                        'match the cleaned_bold_cache npz that was built '
                        'for this subject).')
    p.add_argument('--grid-radius', type=float, default=5.0,
                   help='Extended grid radius in degrees (default 5.0).')
    p.add_argument('--tr', type=float, default=1.6)
    p.add_argument('--l2-norm', type=float, default=1.0,
                   help='L2 penalty on decoded stimulus (default 1.0; '
                        'matches the binary 0/1 paradigm scale via the '
                        'Gaussian-prior interpretation σ_prior = 1/√(2·L2) ≈ '
                        '0.71. Smaller L2 → noisy spikes; larger L2 → '
                        'crushed amplitude.).')
    p.add_argument('--learning-rate', type=float, default=0.5)
    p.add_argument('--max-n-iterations', type=int, default=1000)
    p.add_argument('--min-n-iterations', type=int, default=200)
    p.add_argument('--resid-max-iter', type=int, default=2000,
                   help='Max iters for ResidualFitter (default 2000; the '
                        'omega/dof landscape is shallow so converge with '
                        'plenty of headroom — early-stops on plateau).')
    p.add_argument('--max-voxels', type=int, default=200,
                   help='Cap on # of voxels (default 200). With '
                        '--voxel-filter=top_r2 this is the top-N pool size; '
                        'with --voxel-filter=p_signal this only caps if more '
                        'than N voxels pass the p_signal threshold.')
    p.add_argument('--voxel-filter', choices=['top_r2', 'p_signal'],
                   default='top_r2',
                   help='Voxel selection: "top_r2" (default) takes the top '
                        '--max-voxels by r²; "p_signal" keeps every voxel with '
                        'r² above the FDR p_signal threshold for this '
                        '(subject, ROI). p_signal gives broader visual-field '
                        'coverage (incl. peripheral voxels reaching the 4° '
                        'corners) at the cost of more (noisier) voxels.')
    p.add_argument('--psignal-posterior', type=float, default=0.5,
                   help='posterior threshold for the p_signal mixture filter '
                        '(default 0.5).')
    p.add_argument('--noise-dist', default='gauss', choices=['gauss', 't'],
                   help="Residual noise distribution (default 'gauss'). "
                        "'t' fits a Student-t and uses it in StimulusFitter "
                        'to down-weight heavy-tailed (small-PRF) voxels.')
    p.add_argument('--out', type=Path, default=None,
                   help='Output npz path (default: derivatives/decoded/'
                        'model{M}/sub-{NN}/decoded_ses-{S}_run-{R}_roi-{ROI}'
                        '_vox{N}[_t].npz)')
    args = p.parse_args()

    tag = (f'ses-{args.session}_run-{args.run}_roi-{args.roi}'
           f'_vox{args.max_voxels}')
    if args.voxel_filter == 'p_signal':
        tag = f'{tag}_psig{args.psignal_posterior:g}'
    if args.noise_dist == 't':
        tag = f'{tag}_t'
    out_path = args.out or (Path(args.bids_folder)
                            / 'derivatives' / 'decoded'
                            / f'model{args.model}'
                            / f'sub-{args.subject:02d}'
                            / f'decoded_{tag}.npz')
    decode(subject=args.subject, session=args.session, run=args.run,
           model=args.model, roi=args.roi, bids_folder=args.bids_folder,
           resolution=args.resolution, grid_radius=args.grid_radius,
           tr=args.tr,
           l2_norm=args.l2_norm, learning_rate=args.learning_rate,
           max_n_iterations=args.max_n_iterations,
           min_n_iterations=args.min_n_iterations,
           resid_max_iter=args.resid_max_iter,
           max_voxels=args.max_voxels, noise_dist=args.noise_dist,
           voxel_filter=args.voxel_filter,
           psignal_posterior=args.psignal_posterior,
           out_path=out_path)


if __name__ == '__main__':
    main()
