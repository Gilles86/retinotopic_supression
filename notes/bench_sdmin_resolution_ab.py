"""sd_min-in-model and resolution A/B. Same voxels, same paradigm
(bar+distractors to match what decode_runwise actually used), same
fit hyperparams, vary ONE knob at a time.

Goal: localise which of {sd_min in model, resolution} drives the
snr=0.01 → snr=0.16 jump between the OLD pipeline and decode.py.
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
SESSION, RUN = 1, 1
ROI = 'V1'
MAX_VOXELS = 200
GRID_RADIUS = 5.0
TR = 1.6
L2 = 1.0
LR = 0.5
MAX_ITER = 1000
MIN_ITER = 200
RESID_ITER = 2000

PARS_LIST = ['x', 'y', 'sd', 'baseline', 'amplitude',
             'srf_amplitude', 'srf_size', 'hrf_delay', 'hrf_dispersion']


def cell(resolution, sd_min, paradigm_kind, bold_run, pars_df,
         label):
    sub = Subject(SUBJECT, bids_folder='/data/ds-retsupp')
    if paradigm_kind == 'bar':
        par = sub.get_bar_stimulus(
            session=SESSION, run=RUN, resolution=resolution,
            grid_radius=GRID_RADIUS).astype(np.float32)
    else:
        par = sub.get_stimulus_with_distractors(
            session=SESSION, run=RUN, resolution=resolution,
            grid_radius=GRID_RADIUS, distractor_shape='rectangle',
            distractor_long_side=1.5, distractor_short_side=0.375
        ).astype(np.float32)
    gx, gy = sub.get_extended_grid_coordinates(
        resolution=resolution, session=SESSION, run=RUN,
        grid_radius=GRID_RADIUS)
    grid = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float32)
    par_flat = par.reshape(258, -1).astype(np.float32)

    bold_df = pd.DataFrame(bold_run.astype(np.float32))

    def mk_model():
        return DifferenceOfGaussiansPRF2DWithHRF(
            grid_coordinates=grid, paradigm=par_flat,
            hrf_model=SPMHRFModel(tr=TR, delay=4.5, dispersion=0.75),
            data=bold_df, parameters=pars_df.astype(np.float32),
            flexible_hrf_parameters=True, sd_min=sd_min)

    t0 = time.time()
    rf = ResidualFitter(model=mk_model(), data=bold_df,
                         paradigm=pd.DataFrame(par_flat),
                         parameters=pars_df.astype(np.float32))
    omega, _ = rf.fit(max_n_iterations=RESID_ITER, progressbar=False)
    t_omega = time.time() - t0

    sf = StimulusFitter(model=mk_model(), data=bold_df, omega=omega,
                         parameters=pars_df.astype(np.float32))
    t0 = time.time()
    decoded = sf.fit(l2_norm=L2, learning_rate=LR,
                      max_n_iterations=MAX_ITER,
                      min_n_iterations=MIN_ITER, progressbar=False)
    t_decode = time.time() - t0
    dec = decoded.values.reshape(258, resolution, resolution).astype(np.float32)

    # Score against bar-only paradigm at this cell's resolution
    # (avoid res-mismatched reshape; per-TR corr is scale-invariant).
    par_t = sub.get_bar_stimulus(
        session=SESSION, run=RUN, resolution=resolution,
        grid_radius=GRID_RADIUS).astype(np.float32)
    stim = par_t > 0.5
    sig = float(dec[stim].mean())
    noi = float(dec[~stim].mean())
    corrs = []
    for t in range(258):
        if par_t[t].std() < 1e-9 or dec[t].std() < 1e-9:
            continue
        corrs.append(np.corrcoef(dec[t].ravel(), par_t[t].ravel())[0, 1])
    print(f'  [{label}]  res={resolution} sd_min={sd_min} par={paradigm_kind}')
    print(f'    omega: {t_omega:.1f}s  decode: {t_decode:.1f}s')
    print(f'    max={dec.max():.3f}  sig={sig:.4f}  noise={noi:.4f}  '
          f'snr={sig - noi:.4f}  corr={float(np.nanmean(corrs)):.4f}')
    return {'max': float(dec.max()), 'sig': sig, 'noise': noi,
            'snr': sig - noi, 'corr': float(np.nanmean(corrs))}


def main():
    sub = Subject(SUBJECT, bids_folder='/data/ds-retsupp')

    # Voxels: top-N by r² in V1
    prf_roi = sub.get_prf_roi_pars(roi=ROI, model=4)
    finite = np.all(np.isfinite(prf_roi.values), axis=1) & (prf_roi['r2'] > 0)
    keep = np.where(finite.values)[0]
    r2_kept = prf_roi['r2'].values[keep]
    sel = keep[np.argsort(-r2_kept)[:MAX_VOXELS]]
    pars_df = prf_roi.iloc[sel].reset_index(drop=True)[PARS_LIST].astype(np.float32)

    # BOLD from cache
    brain_mask = np.asarray(sub.get_bold_mask().get_fdata(), dtype=bool).ravel()
    v1_3d = np.asarray(
        sub.get_retinotopic_roi(roi=ROI, bold_space=True).get_fdata(),
        dtype=bool).ravel()
    v1_within_brain = v1_3d[brain_mask]
    v1_cache_cols = np.where(v1_within_brain)[0][sel]
    with np.load('/tmp/sub-23_kind-full_res-50.npz') as f:
        bold_run = f['bold'][:258, v1_cache_cols].astype(np.float32)

    print(f'{len(sel)} voxels  BOLD {bold_run.shape}')

    print('\n=== Cells (varying res, sd_min, paradigm) ===')
    # Extra cells testing lr=0.05 (the OLD cluster default)
    cells = [
        (50, 0.2, 'bar', 0.5, 'NEW: r=50 sd_min=0.2 bar lr=0.5'),
        (30, 0.2, 'bar', 0.5, '   r=30 sd_min=0.2 bar lr=0.5'),
        (30, 0.05, 'bar+distr', 0.5, '   r=30 sd_min=0.05 bar+distr lr=0.5'),
        # lr=0.05 (job 3022544 settings)
        (30, 0.05, 'bar+distr', 0.05, 'OLD-3022544: r=30 sd_min=0.05 bar+distr lr=0.05'),
        (50, 0.2, 'bar', 0.05, '   r=50 sd_min=0.2 bar lr=0.05'),
    ]
    results = []
    for res, sd_min, par, lr, lbl in cells:
        global LR; LR = lr
        m = cell(res, sd_min, par, bold_run, pars_df, lbl)
        results.append((lbl, m))

    print('\n=== SUMMARY (snr / corr) ===')
    for lbl, m in results:
        print(f'  {lbl:50s}  snr={m["snr"]:7.4f}  corr={m["corr"]:.4f}')


if __name__ == '__main__':
    main()
