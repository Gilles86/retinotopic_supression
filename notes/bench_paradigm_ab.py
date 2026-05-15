"""Paradigm A/B: bar-only vs bar+distractors with everything else fixed.

Same subject / run / voxel pool / PRF model (sd_min=0.2) / fit
hyperparameters; the only knob varying is the paradigm passed when
building the model (which flows into ResidualFitter's omega fit and
the model's internal WWT). StimulusFitter itself runs from an empty
initial stimulus regardless.

Hypothesis: paradigm change drives the snr=0.01 -> snr=0.16 jump.

If snr_A (bar-only) >> snr_B (bar+distractors), the paradigm-through-
omega channel is what matters. If they're close, something else
(sd_min, voxel pool, ...) is doing the work.
"""
from __future__ import annotations

import time
import numpy as np
import pandas as pd

from retsupp.utils.data import Subject
from braincoder.models import DifferenceOfGaussiansPRF2DWithHRF
from braincoder.hrf import SPMHRFModel
from braincoder.optimize import ResidualFitter, StimulusFitter

SUBJECT = 23
SESSION = 1
RUN = 1
ROI = 'V1'
MAX_VOXELS = 200
RESOLUTION = 50
GRID_RADIUS = 5.0
TR = 1.6
L2 = 1.0
LR = 0.5
MAX_ITER = 1000
MIN_ITER = 200
RESID_ITER = 2000
SD_MIN = 0.2

PARS_LIST = ['x', 'y', 'sd', 'baseline', 'amplitude',
             'srf_amplitude', 'srf_size', 'hrf_delay', 'hrf_dispersion']


def make_model(paradigm_flat, grid, pars_df, data_df):
    return DifferenceOfGaussiansPRF2DWithHRF(
        grid_coordinates=grid, paradigm=paradigm_flat,
        hrf_model=SPMHRFModel(tr=TR, delay=4.5, dispersion=0.75),
        data=data_df, parameters=pars_df.astype(np.float32),
        flexible_hrf_parameters=True, sd_min=SD_MIN)


def metrics(decoded, paradigm_TRUE):
    """Score decoded against the TRUE bar paradigm (the same one for both
    cells, so comparison is apples-to-apples)."""
    par_t = paradigm_TRUE.reshape(decoded.shape[0], RESOLUTION, RESOLUTION)
    stim_mask = par_t > 0.5
    off_mask = ~stim_mask
    sig = float(decoded[stim_mask].mean()) if stim_mask.sum() else float('nan')
    noi = float(decoded[off_mask].mean())
    corrs = []
    for t in range(decoded.shape[0]):
        if par_t[t].std() < 1e-9 or decoded[t].std() < 1e-9:
            continue
        corrs.append(np.corrcoef(decoded[t].ravel(),
                                  par_t[t].ravel())[0, 1])
    return {
        'max': float(decoded.max()),
        'sig': sig, 'noise': noi, 'snr': sig - noi,
        'corr': float(np.nanmean(corrs)),
    }


def run_cell(paradigm_for_model, paradigm_TRUE_for_scoring, label,
             bold_run, pars_df, grid):
    t0 = time.time()
    paradigm_flat = paradigm_for_model.reshape(258, -1).astype(np.float32)
    bold_df = pd.DataFrame(bold_run.astype(np.float32))
    paradigm_df = pd.DataFrame(paradigm_flat)

    rf = ResidualFitter(model=make_model(paradigm_flat, grid, pars_df, bold_df),
                         data=bold_df, paradigm=paradigm_df,
                         parameters=pars_df.astype(np.float32))
    omega, dof = rf.fit(max_n_iterations=RESID_ITER, progressbar=False)
    t_omega = time.time() - t0

    sf = StimulusFitter(model=make_model(paradigm_flat, grid, pars_df, bold_df),
                         data=bold_df, omega=omega,
                         parameters=pars_df.astype(np.float32))
    t0 = time.time()
    decoded = sf.fit(l2_norm=L2, learning_rate=LR,
                      max_n_iterations=MAX_ITER,
                      min_n_iterations=MIN_ITER, progressbar=False)
    t_decode = time.time() - t0
    decoded_arr = decoded.values.reshape(258, RESOLUTION, RESOLUTION)

    m = metrics(decoded_arr, paradigm_TRUE_for_scoring)
    print(f'  [{label}]')
    print(f'    omega fit: {t_omega:.1f}s  decode: {t_decode:.1f}s')
    print(f'    max={m["max"]:.3f}  sig={m["sig"]:.4f}  '
          f'noise={m["noise"]:.4f}  snr={m["snr"]:.4f}  corr={m["corr"]:.4f}')
    return decoded_arr, m


def main():
    sub = Subject(SUBJECT, bids_folder='/data/ds-retsupp')

    # Voxels: top-N by r² in V1 (decode.py default)
    prf_roi = sub.get_prf_roi_pars(roi=ROI, model=4)
    finite = np.all(np.isfinite(prf_roi.values), axis=1) & (prf_roi['r2'] > 0)
    keep = np.where(finite.values)[0]
    r2_kept = prf_roi['r2'].values[keep]
    order = np.argsort(-r2_kept)[:MAX_VOXELS]
    sel = keep[order]
    pars_df = prf_roi.iloc[sel].reset_index(drop=True)[PARS_LIST].astype(np.float32)
    print(f'{len(sel)} voxels, median r² = {prf_roi["r2"].iloc[sel].median():.3f}')

    # BOLD from cache, sliced to ses-1 run-1.
    brain_mask = np.asarray(sub.get_bold_mask().get_fdata(), dtype=bool).ravel()
    v1_3d = np.asarray(
        sub.get_retinotopic_roi(roi=ROI, bold_space=True).get_fdata(),
        dtype=bool).ravel()
    v1_within_brain = v1_3d[brain_mask]
    v1_cache_cols = np.where(v1_within_brain)[0][sel]

    with np.load('/tmp/sub-23_kind-full_res-50.npz') as f:
        bold_run = f['bold'][:258, v1_cache_cols].astype(np.float32)
    print(f'BOLD: {bold_run.shape}')

    # Grid (identical for both paradigms)
    gx, gy = sub.get_extended_grid_coordinates(
        resolution=RESOLUTION, session=SESSION, run=RUN,
        grid_radius=GRID_RADIUS)
    grid = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float32)

    # The TWO paradigms.
    par_bar_only = sub.get_bar_stimulus(
        session=SESSION, run=RUN, resolution=RESOLUTION,
        grid_radius=GRID_RADIUS).astype(np.float32)
    par_with_distractors = sub.get_stimulus_with_distractors(
        session=SESSION, run=RUN, resolution=RESOLUTION,
        grid_radius=GRID_RADIUS, distractor_shape='rectangle',
        distractor_long_side=1.5, distractor_short_side=0.375
    ).astype(np.float32)
    print(f'bar-only paradigm: nonzero in {(par_bar_only > 0.5).any(axis=(1,2)).sum()} TRs')
    print(f'bar+distractors:   nonzero in {(par_with_distractors > 0.5).any(axis=(1,2)).sum()} TRs')
    print(f'                   pixels on (mean across t): bar={float((par_bar_only > 0.5).mean()):.4f}  '
          f'bar+dist={float((par_with_distractors > 0.5).mean()):.4f}')

    # Score BOTH cells against the SAME truth (bar-only) so the comparison
    # is apples-to-apples; what matters is "did the decoder find the bar".
    print('\n=== A: paradigm = bar-only (what decode.py does) ===')
    dec_A, m_A = run_cell(par_bar_only, par_bar_only, 'bar-only',
                          bold_run, pars_df, grid)
    print('\n=== B: paradigm = bar+distractors (what decode_runwise did) ===')
    dec_B, m_B = run_cell(par_with_distractors, par_bar_only, 'bar+distractors',
                          bold_run, pars_df, grid)

    # Summary
    print('\n=== SUMMARY ===')
    print(f'{"metric":>10s}  {"bar-only":>12s}  {"bar+distr":>12s}  ratio')
    for k in ('max', 'sig', 'noise', 'snr', 'corr'):
        a, b = m_A[k], m_B[k]
        r = a / b if abs(b) > 1e-9 else float('inf')
        print(f'{k:>10s}  {a:>12.4f}  {b:>12.4f}  {r:>5.2f}x')

    np.savez_compressed('/tmp/paradigm_ab.npz',
                         decoded_bar_only=dec_A,
                         decoded_bar_with_distractors=dec_B,
                         paradigm_bar_only=par_bar_only,
                         paradigm_bar_with_distractors=par_with_distractors)
    print(f'\nWrote /tmp/paradigm_ab.npz')


if __name__ == '__main__':
    main()
