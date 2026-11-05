"""Conditionwise PRF fits: per HP-distractor condition, FULL paradigm.

For each of the 4 HP-distractor conditions (upper_right, upper_left,
lower_left, lower_right), concatenates only the runs whose HP location
matches that condition, builds the per-run paradigm + cleaned BOLD,
and runs GD initialised from the corresponding MEAN-fit model on disk
(``derivatives/prf/model{N}/sub-XX/...``).

One model per invocation — same lean pattern as ``fit_prf.py``. All 4
conditions are fit sequentially within one call (cheap; same loaded
init).

Output: ``derivatives/prf_conditionfit/model{N}/sub-XX/
condition-{c}/sub-XX_desc-{par}.nii.gz``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from nilearn import image, maskers

from braincoder.hrf import SPMHRFModel

from retsupp.utils.data import Subject
from retsupp.modeling.fit_prf import (
    MODEL_CFG, gd_fit, load_prior_pars, save_pars,
)


CONDITIONS = ('upper_right', 'upper_left', 'lower_left', 'lower_right')


def load_condition(sub: Subject, masker, hp_per_run, condition: str,
                   resolution: int, kind: str):
    """Concat cleaned BOLD + per-run paradigm for runs whose HP=condition."""
    bold_chunks, par_chunks = [], []
    grid_coords = None
    n_runs = 0
    for ses in (1, 2):
        for run in sub.get_runs(ses):
            if hp_per_run.get((ses, run)) != condition:
                continue
            n_runs += 1
            bold_fn = (sub.bids_folder / 'derivatives' / 'cleaned'
                       / f'sub-{sub.subject_id:02d}' / f'ses-{ses}' / 'func'
                       / f'sub-{sub.subject_id:02d}_ses-{ses}_task-search_'
                         f'desc-cleaned_run-{run}_bold.nii.gz')
            data = masker.transform(bold_fn).astype(np.float32)[:258]
            bold_chunks.append(data)

            if kind == 'full':
                par = sub.get_stimulus_with_distractors(
                    session=ses, run=run, resolution=resolution,
                    grid_radius=5.0, distractor_shape='rectangle',
                    distractor_long_side=1.5, distractor_short_side=0.375,
                ).astype(np.float32)
                gx, gy = sub.get_extended_grid_coordinates(
                    resolution=resolution, session=ses, run=run, grid_radius=5.0)
            else:
                par = sub.get_stimulus(session=ses, run=run,
                                        resolution=resolution).astype(np.float32)
                gx, gy = sub.get_grid_coordinates(session=ses, run=run,
                                                   resolution=resolution)
            par_chunks.append(par.reshape(par.shape[0], -1))
            if grid_coords is None:
                grid_coords = np.stack(
                    (gx.ravel(), gy.ravel()), axis=1).astype(np.float32)

    if not bold_chunks:
        raise RuntimeError(f"No runs found for HP={condition!r}")
    return (np.vstack(bold_chunks),
            np.vstack(par_chunks),
            grid_coords,
            n_runs)


def main(subject: int, model_label: int,
         bids_folder: str = '/data/ds-retsupp',
         resolution: int = 50, voxel_chunk_size: int = 10000,
         max_n_iterations: int = 2000, paradigm_kind: str = 'full',
         debug: bool = False):
    cfg = MODEL_CFG[model_label]
    bids = Path(bids_folder)
    sub = Subject(subject, bids)
    derivs = bids / 'derivatives'

    bold_mask = sub.get_bold_mask()
    if debug:
        bold_mask = image.math_img(
            'np.where(m.astype(bool) & (np.random.rand(*m.shape) < 0.01), 1, 0)',
            m=bold_mask)
        max_n_iterations = 100
        resolution = 25

    masker = maskers.NiftiMasker(mask_img=bold_mask)
    masker.fit()

    # Init: load the mean fit for this model (must exist).
    print(f"Loading mean-fit init from prf/model{model_label}/sub-{subject:02d}")
    init_full = load_prior_pars(subject, model_label, derivs, masker)
    init_full = init_full.drop(columns=['r2', 'theta', 'ecc'], errors='ignore')

    hp_per_run = sub.get_hpd_locations()

    hrf = SPMHRFModel(tr=1.6, delay=4.5, dispersion=0.75)
    factory = lambda d, p: cfg['cls'](  # noqa: E731
        grid_coordinates=None, paradigm=p, hrf_model=hrf, data=d,
        flexible_hrf_parameters=cfg['flex_hrf'])

    for cond in CONDITIONS:
        print(f"\n=== condition = {cond} ===")
        try:
            data, paradigm, grid_coords, n_runs = load_condition(
                sub, masker, hp_per_run, cond, resolution, paradigm_kind)
        except RuntimeError as e:
            print(f"  skipping: {e}")
            continue
        # Bind grid_coords into a fresh factory for this condition's grid.
        cond_factory = lambda d, p, gc=grid_coords: cfg['cls'](  # noqa: E731
            grid_coordinates=gc, paradigm=p, hrf_model=hrf, data=d,
            flexible_hrf_parameters=cfg['flex_hrf'])
        print(f"  n_runs={n_runs}, T={data.shape[0]}, V={data.shape[1]}, "
              f"paradigm={paradigm.shape}")

        pars = gd_fit(cond_factory, data, paradigm, init_full,
                      voxel_chunk_size, max_n_iterations)
        print(f"  median R²: {pars['r2'].median():.3f}")

        target_dir = (derivs / ('prf_conditionfit.debug' if debug
                                else 'prf_conditionfit')
                      / f'model{model_label}' / f'sub-{subject:02d}'
                      / f'condition-{cond}')
        save_pars(pars, masker, target_dir, subject)
        print(f"  saved: {target_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('subject', type=int)
    p.add_argument('--model', type=int, required=True, choices=list(MODEL_CFG))
    p.add_argument('--bids-folder', default='/data/ds-retsupp')
    p.add_argument('--resolution', type=int, default=50)
    p.add_argument('--voxel-chunk-size', type=int, default=10000)
    p.add_argument('--max-n-iterations', type=int, default=2000)
    p.add_argument('--paradigm-kind', choices=['full', 'bar'], default='full')
    p.add_argument('--debug', action='store_true')
    a = p.parse_args()
    main(a.subject, a.model, bids_folder=a.bids_folder,
         resolution=a.resolution, voxel_chunk_size=a.voxel_chunk_size,
         max_n_iterations=a.max_n_iterations,
         paradigm_kind=a.paradigm_kind, debug=a.debug)
