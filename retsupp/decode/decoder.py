"""StimulusFitter-based stimulus decoding for retsupp.

This module is the shared core used by both the smoke test and the
batch driver. The pipeline mirrors lesson 7 of the braincoder tutorial
(``examples/00_encodingdecoding/fit_prf.py``), specialised for retsupp:

1. Build a :class:`DifferenceOfGaussiansPRF2DWithHRF` (model 4) on the
   bar-only paradigm, with parameters loaded from
   ``derivatives/prf/model4/sub-XX/sub-XX_desc-{par}.nii.gz``.
2. Filter voxels to one ROI (e.g. V1), keeping only sd >= 0.5 and
   r2 > ``r2_min`` and dropping any voxel with a non-finite parameter.
3. Fit residual covariance ``omega`` via :class:`ResidualFitter` on the
   per-run cleaned BOLD (T = 258).
4. Decode the stimulus per TR with
   :class:`StimulusFitter.fit(l2_norm=..., learning_rate=..., max_n_iterations=...)`.
5. Sample the decoded image at the four diagonal ring positions
   (4 deg eccentricity) and return a long-format DataFrame with one row
   per (frame, ring location).
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import image, maskers


# Lazy braincoder import so importing this module is cheap.
def _bc():
    from braincoder.hrf import SPMHRFModel
    from braincoder.models import DifferenceOfGaussiansPRF2DWithHRF
    from braincoder.optimize import ResidualFitter, StimulusFitter
    return SPMHRFModel, DifferenceOfGaussiansPRF2DWithHRF, ResidualFitter, StimulusFitter


PRF_PARS = ['x', 'y', 'sd', 'amplitude', 'baseline',
            'srf_amplitude', 'srf_size', 'hrf_delay', 'hrf_dispersion']


def load_prf_pars(sub, model: int = 4) -> dict[str, np.ndarray]:
    """Load model-N PRF NIfTIs as a dict of (V,) arrays in BOLD-mask order."""
    base = (sub.bids_folder / 'derivatives' / 'prf'
            / f'model{model}' / f'sub-{sub.subject_id:02d}')
    masker = sub.get_bold_mask(return_masker=True)
    masker.fit()
    out = {}
    for par in PRF_PARS:
        fn = base / f'sub-{sub.subject_id:02d}_desc-{par}.nii.gz'
        out[par] = masker.transform(str(fn)).flatten().astype(np.float32)
    out['r2'] = masker.transform(
        str(base / f'sub-{sub.subject_id:02d}_desc-r2.nii.gz')).flatten().astype(np.float32)
    return out, masker


def select_roi_voxels(sub, roi: str, prf: dict, masker, *,
                      sd_min: float = 0.5, r2_min: float = 0.05,
                      ecc_max: float = 6.0,
                      max_voxels: int | None = 250) -> np.ndarray:
    """Indices into the BOLD-mask flattened array for ROI voxels passing QC.

    QC:
    - Voxel inside ``roi`` (Benson retinotopic atlas via Subject).
    - All PRF parameters finite.
    - Phantom-perfect filter: sd >= sd_min (drops sd~=0 voxels with R^2=1).
    - R^2 > r2_min.
    - PRF eccentricity sqrt(x^2 + y^2) <= ecc_max. Drops voxels whose
      PRF centres ran away far outside the bar aperture (3.17 deg);
      such voxels' RFs evaluated on the in-FOV grid are ~0 and
      contribute nothing to decoding while inflating omega.
    - Cap to top-``max_voxels`` by R^2 to bound compute.
    """
    roi_mask = sub.get_retinotopic_roi(roi=roi, bold_space=True)
    roi_flat = masker.transform(roi_mask).flatten().astype(bool)
    sd = prf['sd']
    r2 = prf['r2']
    ecc = np.sqrt(prf['x'] ** 2 + prf['y'] ** 2)
    finite = np.all(np.stack([np.isfinite(prf[p]) for p in PRF_PARS]), axis=0)
    keep = (roi_flat & finite & (sd >= sd_min) & (r2 > r2_min)
            & (ecc <= ecc_max))
    idx = np.where(keep)[0]
    if max_voxels is not None and len(idx) > max_voxels:
        order = np.argsort(-r2[idx])  # best r2 first
        idx = idx[order[:max_voxels]]
    return idx


def load_run_bold(sub, session: int, run: int, masker) -> np.ndarray:
    """Cleaned BOLD for one run, cropped to 258 TRs, in masker order."""
    fn = (sub.bids_folder / 'derivatives' / 'cleaned'
          / f'sub-{sub.subject_id:02d}' / f'ses-{session}' / 'func'
          / f'sub-{sub.subject_id:02d}_ses-{session}_task-search_'
            f'desc-cleaned_run-{run}_bold.nii.gz')
    data = masker.transform(str(fn)).astype(np.float32)
    return data[:258]


def build_paradigm_and_grid(sub, session: int, run: int, resolution: int):
    """Bar-only paradigm + matching grid coordinates for one run."""
    par = sub.get_stimulus(session=session, run=run,
                           resolution=resolution).astype(np.float32)
    par = par.reshape(par.shape[0], -1)
    gx, gy = sub.get_grid_coordinates(resolution=resolution,
                                      session=session, run=run)
    grid = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float32)
    return par, grid


def make_dog_model(grid_coords: np.ndarray, paradigm: np.ndarray,
                   pars_df: pd.DataFrame, data: np.ndarray | None = None,
                   tr: float = 1.6):
    """DoG + flexible HRF model populated with fitted parameters."""
    SPMHRFModel, DoG, _, _ = _bc()
    hrf = SPMHRFModel(tr=tr, delay=4.5, dispersion=0.75)
    model = DoG(grid_coordinates=grid_coords,
                paradigm=paradigm,
                hrf_model=hrf,
                data=data,
                parameters=pars_df.astype(np.float32),
                flexible_hrf_parameters=True)
    return model


def decode_run(sub, session: int, run: int, *,
               roi: str = 'V1',
               resolution: int = 30,
               max_voxels: int = 250,
               sd_min: float = 0.5,
               r2_min: float = 0.0,
               ecc_max: float = 6.0,
               l2_norm: float = 0.01,
               learning_rate: float = 0.5,
               max_n_iterations: int = 600,
               min_n_iterations: int = 200,
               resid_max_iter: int = 300,
               progressbar: bool = True,
               verbose: bool = True):
    """Decode one run of cleaned BOLD into a per-TR stimulus map.

    The default ``learning_rate`` is intentionally large (0.5).  With
    L2-on-the-bijector-transformed-pars and the cleaned-BOLD scale, the
    gradient at the empty-stimulus init (~1e-6) is small relative to
    Adam's denominator, so a small lr (e.g. 0.01 from the tutorial)
    leaves the decoded stimulus stuck at zero.  lr=0.5 reliably escapes
    that flat region in <50 iterations.

    Returns
    -------
    decoded : pd.DataFrame
        Index = frame (0..257). Columns = (x, y) MultiIndex matching
        ``model.stimulus.dimension_labels``. Values = decoded intensity
        in stimulus space.
    grid : (G, 2) float32 array
        Grid coordinates in deg.
    voxel_idx : (V_used,) int array
        Indices into the BOLD-mask flattened array of the kept voxels.
    omega : (V, V) float32 array
        Fitted residual covariance.
    pars_df : pd.DataFrame
        Per-voxel PRF parameter table actually used (kept voxels).
    bold : (T, V_used) float32
        Cleaned BOLD passed to the decoder.
    """
    SPMHRFModel, DoG, ResidualFitter, StimulusFitter = _bc()

    prf, masker = load_prf_pars(sub, model=4)
    voxel_idx = select_roi_voxels(sub, roi=roi, prf=prf, masker=masker,
                                  sd_min=sd_min, r2_min=r2_min,
                                  ecc_max=ecc_max,
                                  max_voxels=max_voxels)
    if verbose:
        print(f'  ROI {roi}: {len(voxel_idx)} voxels kept '
              f'(after sd>={sd_min}, r2>{r2_min}, max_voxels={max_voxels})')
    if len(voxel_idx) < 10:
        raise RuntimeError(f'Too few voxels in {roi}: {len(voxel_idx)}')

    pars_df = pd.DataFrame({p: prf[p][voxel_idx] for p in PRF_PARS})
    bold = load_run_bold(sub, session, run, masker)[:, voxel_idx]
    paradigm, grid = build_paradigm_and_grid(sub, session, run, resolution)

    if verbose:
        print(f'  BOLD: T={bold.shape[0]}, V={bold.shape[1]}; '
              f'paradigm: {paradigm.shape}, grid: {grid.shape}')

    bold_df = pd.DataFrame(bold, index=pd.Index(np.arange(bold.shape[0]),
                                                name='frame'))
    bold_df.columns.name = 'voxel'
    paradigm_df = pd.DataFrame(paradigm,
                               index=pd.Index(np.arange(paradigm.shape[0]),
                                              name='frame'))

    model = make_dog_model(grid, paradigm, pars_df, data=bold_df)

    if verbose:
        print('  Fitting residual covariance...')
    rf = ResidualFitter(model=model, data=bold_df, paradigm=paradigm_df,
                        parameters=pars_df.astype(np.float32))
    omega, _ = rf.fit(max_n_iterations=resid_max_iter,
                      progressbar=progressbar)

    # Rebuild model on filtered data (StimulusFitter writes to model.parameters)
    model = make_dog_model(grid, paradigm, pars_df, data=bold_df)

    if verbose:
        print('  Fitting stimulus...')
    sf = StimulusFitter(model=model, data=bold_df, omega=omega,
                        parameters=pars_df.astype(np.float32))
    decoded = sf.fit(l2_norm=l2_norm, learning_rate=learning_rate,
                     max_n_iterations=max_n_iterations,
                     min_n_iterations=min_n_iterations,
                     progressbar=progressbar)

    return decoded, grid, voxel_idx, omega, pars_df, bold


def sample_at_ring_positions(decoded: pd.DataFrame, grid: np.ndarray,
                             ring_disk_radius: float = 0.4,
                             eccentricity: float = 4.0) -> pd.DataFrame:
    """Average decoded intensity inside a small disk at each ring position.

    The 4 ring positions are at (+/- ecc/sqrt(2), +/- ecc/sqrt(2)).
    For each TR and each ring position, returns mean decoded value
    inside a disk of radius ``ring_disk_radius`` deg around that point.
    """
    ring = {
        'upper_right': (+eccentricity / np.sqrt(2), +eccentricity / np.sqrt(2)),
        'upper_left':  (-eccentricity / np.sqrt(2), +eccentricity / np.sqrt(2)),
        'lower_left':  (-eccentricity / np.sqrt(2), -eccentricity / np.sqrt(2)),
        'lower_right': (+eccentricity / np.sqrt(2), -eccentricity / np.sqrt(2)),
    }
    arr = decoded.values  # (T, G)
    rows = []
    for name, (cx, cy) in ring.items():
        d2 = (grid[:, 0] - cx) ** 2 + (grid[:, 1] - cy) ** 2
        mask = d2 <= ring_disk_radius ** 2
        if mask.sum() == 0:
            # Fall back to nearest voxel.
            mask = np.zeros(grid.shape[0], dtype=bool)
            mask[d2.argmin()] = True
        vals = arr[:, mask].mean(axis=1)
        rows.append(pd.DataFrame({'ring_location': name,
                                  'frame': np.arange(len(vals)),
                                  'decoded': vals}))
    return pd.concat(rows, axis=0, ignore_index=True)


def label_hp_lp(per_ring: pd.DataFrame, hp_location: str) -> pd.DataFrame:
    """Add ``hp_role`` column: 'HP', 'orth', 'opposite' relative to hp_location.

    Geometry on the four ring corners:
    - HP: same corner as hp_location
    - opposite: diagonally opposite corner
    - orth: the two corners adjacent to HP (sharing one axis)
    """
    opp = {
        'upper_right': 'lower_left',
        'upper_left':  'lower_right',
        'lower_left':  'upper_right',
        'lower_right': 'upper_left',
    }
    def role(loc):
        if loc == hp_location:
            return 'HP'
        if loc == opp[hp_location]:
            return 'opposite'
        return 'orth'
    out = per_ring.copy()
    out['hp_role'] = out['ring_location'].map(role)
    out['hp_location'] = hp_location
    return out
