"""Simulation check of the gain-free decoder (``decode_final.py``): does a gain change at the HP location alone
produce the "next-to-HP higher than opposite" pattern in the quadrant readout?

One task = one subject × run, as in ``decode_final.py``. Per ROI, with the voxels, pRFs (the fold that held this
run out) and decoder settings of ``decode_final.py``:

1. Truth. The bar paradigm of the run is multiplied by a sustained gain field
   ``g(x, y) = 1 + A · exp(-d² / 2σ²)``, with ``d`` the distance to the HP item (the run's
   ``eccentricity_stimulus`` on the HP diagonal), FWHM ``--fwhm`` (4.8°) and amplitude ``A`` (``--amplitudes``;
   0 is the null). Nothing else changes, so the neighbouring quadrants have no effect of their own beyond the
   tail of the one Gaussian (the true readout is written alongside, ``true_drive_bar``).
2. Signal. The fold pRF model (same class and HRF as the decoder: with ``--prf-source canonical`` the canonical
   HRF in both) predicts the BOLD to that modulated paradigm.
3. Noise. The residuals of the real run under the same model with the unmodulated bar paradigm, phase-randomised
   with one set of random phases for all voxels (``--noise phase``; keeps every voxel's power spectrum and the
   cross-voxel covariance, removes everything locked to the stimulus, including any real HP effect), or
   independent Gaussian AR(1) noise with each voxel's residual SD and lag-1 autocorrelation (``--noise ar1``).
   The same noise draw is used for every amplitude, so amplitude contrasts are paired.
4. Decoding. The ResidualFitter and StimulusFitter of ``decode_final.py`` (bar-only paradigm, L2 1.0, lr 0.5),
   ``free`` and ``outline`` decoders, and the same quadrant readout.

Outputs (never inside existing derivatives folders)::

    <out-root>/<prf-source>/noise-<noise>/gain-<A>/<decoder>/sub-XX/
        readout_ses-S_run-R_roi-ROI.tsv   quadrant readout + true_drive_bar
        maps_ses-S_run-R_roi-ROI.npz      time-mean decoded image and bar-weighted mean per pixel

Default out-root: ``<bids>/derivatives/decoded_sim_2026-11``.

Usage::

    python -m retsupp.decode.simulate_decode 23 --session 1 --run 1 --rois V1 V2 V3
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from retsupp.decode.decode_final import (QUADRANTS, SOURCES, decode, fit_noise, fold_of, load_prfs, make_model, readout,
                                         select_voxels, slice_run)
from retsupp.utils.data import Subject


def gain_field(grid, center, amplitude, fwhm):
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    d2 = ((grid - np.asarray(center, dtype=np.float32)) ** 2).sum(1)
    return (1 + amplitude * np.exp(-d2 / (2 * sigma ** 2))).astype(np.float32)


def phase_randomise(resid, rng):
    """Same random phases for all columns: keeps each column's spectrum and the cross-column covariance."""
    T = resid.shape[0]
    f = np.fft.rfft(resid - resid.mean(0), axis=0)
    phases = rng.uniform(0, 2 * np.pi, f.shape[0])
    phases[0] = 0
    if T % 2 == 0:
        phases[-1] = 0
    return np.fft.irfft(f * np.exp(1j * phases)[:, None], n=T, axis=0).astype(np.float32)


def ar1_noise(resid, rng):
    r = resid - resid.mean(0)
    sd = r.std(0)
    phi = np.clip((r[1:] * r[:-1]).sum(0) / np.maximum((r ** 2).sum(0), 1e-12), -.95, .95)
    e = rng.standard_normal(r.shape).astype(np.float32) * sd * np.sqrt(1 - phi ** 2)
    out = np.empty_like(e)
    out[0] = rng.standard_normal(r.shape[1]) * sd
    for t in range(1, len(e)):
        out[t] = phi * out[t - 1] + e[t]
    return out.astype(np.float32)


def main(subject, session, run, rois, bids, source, decoders, joint_tag, amplitudes, fwhm, noise_kind, seed, max_voxels,
         max_bold_sd, l2_norm, learning_rate, max_iter, min_iter, resid_max_iter, resolution=50, grid_radius=5.0,
         out_root=None):
    sub = Subject(subject, bids_folder=bids)
    if run not in sub.get_runs(session):
        print(f'sub-{subject:02d} has no ses-{session} run-{run}; nothing to do')
        return
    deriv = Path(bids) / 'derivatives'
    out_root = Path(out_root or deriv / 'decoded_sim_2026-11') / source / f'noise-{noise_kind}'
    fold = fold_of(sub, session, run)
    hp = sub.get_hpd_locations().get((session, run))
    settings = sub.get_experimental_settings(session, run)
    aperture = float(settings['radius_bar_aperture'])
    ecc = float(settings['eccentricity_stimulus'])
    sx, sy = QUADRANTS[hp]
    center = (sx * ecc / np.sqrt(2), sy * ecc / np.sqrt(2))
    print(f'sub-{subject:02d} ses-{session} run-{run}: fold {fold}, HP {hp} at ({center[0]:.2f}, {center[1]:.2f})°, '
          f'aperture {aperture}°, source {source}, noise {noise_kind}, amplitudes {amplitudes}', flush=True)

    def out_dir(a, dec):
        return out_root / f'gain-{a:+.3f}' / dec / f'sub-{subject:02d}'
    fn = lambda roi: f'readout_ses-{session}_run-{run}_roi-{roi}.tsv'
    todo_rois = [roi for roi in rois if not all((out_dir(a, d) / fn(roi)).exists() for a in amplitudes for d in decoders)]
    if not todo_rois:
        print('All outputs exist')
        return

    pars_all, roi_of = load_prfs(deriv, subject, fold, source, joint_tag)
    rel = pd.read_csv(deriv / 'prf_cv_roi_reliability' / f'sub-{subject:02d}' / f'sub-{subject:02d}_fold-{fold}_reliability.tsv.gz',
                      sep='\t').set_index('voxel_index')
    with np.load(deriv / 'cleaned_bold_cache' / f'sub-{subject:02d}' / f'sub-{subject:02d}_kind-full_res-{resolution}.npz') as f:
        bold_run = slice_run(f['bold'], sub, session, run).astype(np.float32)

    paradigm = sub.get_bar_stimulus(session=session, run=run, resolution=resolution, grid_radius=grid_radius).astype(np.float32)
    T = paradigm.shape[0]
    gx, gy = sub.get_extended_grid_coordinates(resolution=resolution, session=session, run=run, grid_radius=grid_radius)
    grid = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float32)
    par_flat = paradigm.reshape(T, -1)
    par_df = pd.DataFrame(par_flat, index=pd.Index(np.arange(T), name='frame'))

    for ri, roi in enumerate(rois):
        if roi not in todo_rois:
            continue
        sel, n_roi = select_voxels(pars_all, roi_of, rel, roi, max_voxels, max_bold_sd)
        if len(sel) < 20:
            print(f'{roi}: only {len(sel)} voxels, skipped', flush=True)
            continue
        t0 = time.time()
        pars = pars_all.loc[sel].reset_index(drop=True).astype(np.float32)
        real = bold_run[:, np.asarray(sel)]
        model = make_model(source, grid, par_flat, None, pars)
        pars = pars[list(model.parameter_labels)]
        pred0 = np.asarray(model.predict(paradigm=par_df, parameters=pars), dtype=np.float32)
        resid = real - pred0
        rng = np.random.default_rng([seed, subject, session, run, ri])
        noise = phase_randomise(resid, rng) if noise_kind == 'phase' else ar1_noise(resid, rng)
        print(f'{roi}: {len(sel)} voxels of {n_roi}; real-data R² median {np.median(1 - resid.var(0) / real.var(0)):.3f}, '
              f'residual SD median {np.median(resid.std(0)):.3f}, simulated noise SD median {np.median(noise.std(0)):.3f}',
              flush=True)
        for a in amplitudes:
            g = gain_field(grid, center, a, fwhm)
            par_g = par_flat * g[None, :]
            signal = np.asarray(model.predict(paradigm=pd.DataFrame(par_g, index=par_df.index), parameters=pars), dtype=np.float32)
            sim = signal + noise + resid.mean(0)   # noise is mean-free; keep the real run's offset from the model
            data = pd.DataFrame(sim, index=pd.Index(np.arange(T), name='frame'))
            truth = readout(par_g.reshape(paradigm.shape), paradigm, grid, aperture, hp).set_index('quadrant').drive_bar
            m = make_model(source, grid, par_flat, data, pars)
            t1 = time.time()
            omega, dof = fit_noise(m, data, par_df, pars, resid_max_iter)
            t_noise = time.time() - t1
            for dec in decoders:
                out = out_dir(a, dec)
                if (out / fn(roi)).exists():
                    continue
                out.mkdir(parents=True, exist_ok=True)
                t1 = time.time()
                mask = (paradigm > 0) if dec == 'outline' else None
                decoded = decode(make_model(source, grid, par_flat, data, pars), data, pars, omega, dof, mask, l2_norm,
                                 learning_rate, max_iter, min_iter, (T, resolution, resolution))
                ro = readout(decoded, paradigm, grid, aperture, hp)
                ro['true_drive_bar'] = ro.quadrant.map(truth).to_numpy()
                ro = ro.assign(subject=subject, session=session, run=run, roi=roi, decoder=dec, source=source, fold=fold,
                               hp_location=hp, n_voxels=len(sel), gain_amplitude=a, fwhm=fwhm, noise=noise_kind)
                ro.to_csv(out / fn(roi), sep='\t', index=False)
                dflat = decoded.reshape(T, -1)
                nbar = par_flat.sum(0)
                np.savez_compressed(out / f'maps_ses-{session}_run-{run}_roi-{roi}.npz',
                                    mean_decoded=dflat.mean(0).reshape(resolution, resolution).astype(np.float32),
                                    bar_drive=np.where(nbar > 0, (dflat * par_flat).sum(0) / np.maximum(nbar, 1), np.nan)
                                    .reshape(resolution, resolution).astype(np.float32),
                                    grid=grid, hp_location=np.array(str(hp)), gain=g.reshape(resolution, resolution))
                hp_row = ro[ro.is_hp].iloc[0]
                print(f'  A={a:+.3f} {dec}: noise fit {t_noise:.0f} s, decode {time.time() - t1:.0f} s; drive_bar HP '
                      f'{hp_row.drive_bar:.4f} (true {hp_row.true_drive_bar:.4f}), LP mean {ro[~ro.is_hp].drive_bar.mean():.4f}',
                      flush=True)
        print(f'{roi} done ({time.time() - t0:.0f} s)', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('subject', type=int)
    ap.add_argument('--session', type=int, required=True)
    ap.add_argument('--run', type=int, required=True)
    ap.add_argument('--rois', nargs='+', default=['V1', 'V2', 'V3'])
    ap.add_argument('--bids-folder', default='/shares/zne.uzh/gdehol/ds-retsupp')
    ap.add_argument('--prf-source', choices=list(SOURCES), default='canonical')
    ap.add_argument('--joint-tag', default='final2')
    ap.add_argument('--decoders', nargs='+', choices=['free', 'outline'], default=['free', 'outline'])
    ap.add_argument('--amplitudes', type=float, nargs='+', default=[0., -.075])
    ap.add_argument('--fwhm', type=float, default=4.8)
    ap.add_argument('--noise', choices=['phase', 'ar1'], default='phase')
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--max-voxels', type=int, default=2000)
    ap.add_argument('--max-bold-sd', type=float, default=20.)
    ap.add_argument('--l2-norm', type=float, default=1.0)
    ap.add_argument('--learning-rate', type=float, default=0.5)
    ap.add_argument('--max-n-iterations', type=int, default=1000)
    ap.add_argument('--min-n-iterations', type=int, default=200)
    ap.add_argument('--resid-max-iter', type=int, default=2000)
    ap.add_argument('--out-root', default=None)
    a = ap.parse_args()
    main(a.subject, a.session, a.run, a.rois, a.bids_folder, a.prf_source, a.decoders, a.joint_tag, a.amplitudes, a.fwhm,
         a.noise, a.seed, a.max_voxels, a.max_bold_sd, a.l2_norm, a.learning_rate, a.max_n_iterations,
         a.min_n_iterations, a.resid_max_iter, out_root=a.out_root)
