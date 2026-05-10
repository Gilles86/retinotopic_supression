"""Fast linear-filter approximation for decoded stimulus drive.

The proper :class:`StimulusFitter` pipeline (see ``decoder.py``) is too
slow to run on 28 subs x 8 ROIs x 12 runs in batch (~17 min/run on
CPU). This module provides a v1 approximation that's fast enough to
run all subjects on a single CPU node in a few minutes:

For each ROI voxel with valid PRF, compute the spatial RF amplitude at
each of the four ring positions: ``w_v(p) = DoG(p; theta_v)``. Project
per-TR BOLD onto each ring's weight pattern:

    d_p(t) = sum_v BOLD_v(t) * w_v(p) / sum_v w_v(p)**2

This is the OLS estimator of "instantaneous stimulus presence at p at
time t" given the fitted encoders (assuming HRF is left in BOLD).
Aggregation per run: time-weight by HRF-convolved
``get_dynamic_indicator`` so that drives during distractor windows
dominate the per-ring run-mean.

We classify each ring as HP / orth / opposite relative to that run's
HP, then average across rings within each (subject, ROI, hp_role)
cell.

Output:
    derivatives/decode/decoded_drive.tsv  (long format)

Run locally::

    python -m retsupp.decode.decode_drive_fast --subjects 2 --rois V1

Or all subs (~5 min on a 4-core CPU)::

    python -m retsupp.decode.decode_drive_fast --n-jobs 4
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import maskers
from tqdm import tqdm

from retsupp.utils.data import Subject, distractor_locations


# Ring order MUST match get_dynamic_indicator channel order.
RINGS = ('upper_right', 'upper_left', 'lower_left', 'lower_right')
RING_XY = {
    'upper_right': distractor_locations['upper right'],
    'upper_left':  distractor_locations['upper left'],
    'lower_left':  distractor_locations['lower left'],
    'lower_right': distractor_locations['lower right'],
}

ROI_ORDER = ['V1', 'V2', 'V3', 'V3AB', 'hV4', 'VO', 'LO', 'TO']
PARAM_NAMES = ['x', 'y', 'sd', 'baseline', 'amplitude',
               'srf_amplitude', 'srf_size', 'hrf_delay', 'hrf_dispersion', 'r2']


def hp_relation(ring: str, hp: str) -> str:
    """HP / orth / opposite relative to the run's HP."""
    if ring == hp:
        return 'HP'
    pairs = [('upper_right', 'lower_left'), ('upper_left', 'lower_right')]
    for a, b in pairs:
        if {ring, hp} == {a, b}:
            return 'opposite'
    return 'orth'


def dog_value_at_point(x, y, sd, amplitude, srf_amplitude, srf_size,
                        px, py, pixel_area=1.0):
    """DoG receptive-field value at a single point (px, py).

    Mirrors :class:`braincoder.models.DifferenceOfGaussiansPRF2D._get_rf`
    exactly: each Gaussian is normalised by ``sd * sqrt(2 pi) / pixel_area``,
    and the surround amplitude is ``srf_amplitude * amplitude * srf_size``
    with std ``sd * srf_size``.

    All inputs are vectorised over voxels (1D arrays).
    """
    d2 = (px - x) ** 2 + (py - y) ** 2
    norm_c = sd * np.sqrt(2.0 * np.pi) / pixel_area
    g_centre = np.exp(-d2 / (2.0 * sd ** 2)) * amplitude / norm_c
    sd_s = sd * srf_size
    norm_s = sd_s * np.sqrt(2.0 * np.pi) / pixel_area
    g_surr = (np.exp(-d2 / (2.0 * sd_s ** 2))
              * (srf_amplitude * amplitude * srf_size) / norm_s)
    return g_centre - g_surr


def spm_double_gamma_hrf(tr: float, delay: float = 6.0,
                          dispersion: float = 1.0,
                          undershoot_delay: float = 16.0,
                          undershoot_dispersion: float = 1.0,
                          ratio: float = 1.0/6.0,
                          time_length: float = 32.0):
    """Canonical SPM-style double-gamma HRF sampled at TR."""
    from scipy.stats import gamma
    times = np.arange(0, time_length + tr, tr)
    pos = gamma.pdf(times, delay / dispersion, scale=dispersion)
    neg = gamma.pdf(times, undershoot_delay / undershoot_dispersion,
                     scale=undershoot_dispersion)
    hrf = pos - ratio * neg
    return hrf / hrf.max()


def filter_voxels(pars: dict, sd_min: float = 0.5, r2_min: float = 0.0,
                   srf_size_min: float = 0.1):
    """Boolean mask: keep PRFs with finite parameters and sd >= sd_min and
    r2 >= r2_min. We use a permissive r2_min by default because the
    concatenated-paradigm fits have median R^2 ~0.002. Also require
    srf_size >= srf_size_min to avoid divide-by-zero in the surround
    Gaussian (sd*srf_size = 0 -> nan/inf weights) — these come from
    a small minority of voxels where the optimiser collapsed the
    surround."""
    finite = np.all(np.stack([np.isfinite(v) for v in pars.values()]),
                    axis=0)
    keep = (finite
            & (pars['sd'] >= sd_min)
            & (pars['r2'] >= r2_min)
            & (pars['srf_size'] >= srf_size_min))
    return keep


def run_subject(subject: int, bids_folder: Path, *,
                rois=ROI_ORDER, model: int = 4, sd_min: float = 0.5,
                r2_min: float = 0.0):
    """Compute decoded drive for one subject. Returns a long-format
    summary DataFrame at the (subject, roi, run, ring) level."""
    sub = Subject(subject, bids_folder)

    # --- Load mean-fit DoG parameters volume-wise (once per subject).
    par_dir = (bids_folder / 'derivatives' / 'prf' / f'model{model}'
               / f'sub-{subject:02d}')
    if not par_dir.exists():
        raise FileNotFoundError(f'Missing PRF dir: {par_dir}')

    first_run = sub.get_runs(1)[0]
    bold_mask_img = sub.get_bold_mask(session=1, run=first_run)
    masker = maskers.NiftiMasker(mask_img=bold_mask_img)
    masker.fit()

    pars_full = {}
    for p in PARAM_NAMES:
        fn = par_dir / f'sub-{subject:02d}_desc-{p}.nii.gz'
        if not fn.exists():
            raise FileNotFoundError(f'Missing param NIfTI: {fn}')
        pars_full[p] = masker.transform(str(fn)).flatten()

    # --- Per-ROI per-voxel evaluate w_v(p) for the 4 ring positions.
    roi_results = {}
    for roi in rois:
        try:
            roi_mask_img = sub.get_retinotopic_roi(roi=roi, bold_space=True)
        except Exception as e:
            print(f'  sub-{subject:02d} {roi}: ROI failed: {e}', flush=True)
            continue
        roi_arr = roi_mask_img.get_fdata().astype(bool)
        roi_in_mask = masker.transform(
            nib.Nifti1Image(roi_arr.astype(np.int8), roi_mask_img.affine)
        ).flatten().astype(bool)
        idx_global = np.where(roi_in_mask)[0]
        sub_pars = {k: pars_full[k][idx_global] for k in PARAM_NAMES}
        keep = filter_voxels(sub_pars, sd_min=sd_min, r2_min=r2_min)
        if keep.sum() < 5:
            print(f'  sub-{subject:02d} {roi}: only {int(keep.sum())} good voxels — skip',
                  flush=True)
            continue
        idx_global = idx_global[keep]
        x = sub_pars['x'][keep]
        y = sub_pars['y'][keep]
        sd = sub_pars['sd'][keep]
        amp = sub_pars['amplitude'][keep]
        srf_a = sub_pars['srf_amplitude'][keep]
        srf_s = sub_pars['srf_size'][keep]

        W = np.zeros((len(idx_global), 4), dtype=np.float32)
        for k, ring in enumerate(RINGS):
            px, py = RING_XY[ring]
            W[:, k] = dog_value_at_point(
                x, y, sd, amp, srf_a, srf_s,
                px=np.float32(px), py=np.float32(py), pixel_area=1.0,
            )
        # Drop any voxel whose W came out non-finite (defensive: filter_voxels
        # should already exclude these via srf_size_min).
        good = np.all(np.isfinite(W), axis=1)
        if not good.all():
            n_drop = int((~good).sum())
            print(f'  sub-{subject:02d} {roi}: dropped {n_drop} non-finite-W voxels',
                  flush=True)
            idx_global = idx_global[good]
            W = W[good]
        if len(idx_global) < 5:
            continue
        roi_results[roi] = dict(idx=idx_global, W=W)
        print(f'  sub-{subject:02d} {roi}: {len(idx_global)} good voxels '
              f'(median W norm={np.linalg.norm(W, axis=1).mean():.3g})',
              flush=True)

    if not roi_results:
        return pd.DataFrame()

    # --- Iterate over runs, load BOLD for ALL ROI voxels at once.
    all_idx = np.unique(np.concatenate([d['idx'] for d in roi_results.values()]))
    idx_to_pos = {gi: i for i, gi in enumerate(all_idx)}

    hpd = sub.get_hpd_locations()
    summary_rows = []
    hrf = spm_double_gamma_hrf(tr=sub.get_tr(1, 1))

    for ses in [1, 2]:
        for run in sub.get_runs(ses):
            try:
                bold_fn = sub.get_bold(session=ses, run=run, type='cleaned',
                                        return_image=False)
                bold = masker.transform(str(bold_fn)).astype(np.float32)
            except Exception as e:
                print(f'  sub-{subject:02d} ses-{ses} run-{run}: BOLD load failed: {e}',
                      flush=True)
                continue
            bold = bold[:258]
            T = bold.shape[0]
            bold_sub = bold[:, all_idx]

            # z-score per voxel so high-variance voxels don't dominate.
            mu = bold_sub.mean(axis=0, keepdims=True)
            sd_v = bold_sub.std(axis=0, keepdims=True) + 1e-6
            bold_z = (bold_sub - mu) / sd_v

            ind = sub.get_dynamic_indicator(session=ses, run=run)
            ind_hrf = np.zeros_like(ind, dtype=np.float32)
            for k in range(4):
                ind_hrf[:, k] = np.convolve(
                    ind[:, k], hrf, mode='full')[:T]
            ind_hrf_sum = ind_hrf.sum(axis=0)
            run_hp = hpd.get((ses, run), None)

            for roi, info in roi_results.items():
                roi_pos = np.array([idx_to_pos[g] for g in info['idx']])
                B = bold_z[:, roi_pos]   # (T, n_roi)
                W = info['W']            # (n_roi, 4)
                wnorm2 = (W ** 2).sum(axis=0) + 1e-12
                drive = (B @ W) / wnorm2[np.newaxis, :]   # (T, 4)

                for k, ring in enumerate(RINGS):
                    if ind_hrf_sum[k] < 1e-3:
                        weighted = float(drive[:, k].mean())
                    else:
                        weighted = float(
                            (drive[:, k] * ind_hrf[:, k]).sum() / ind_hrf_sum[k])
                    unweighted = float(drive[:, k].mean())
                    rel = (hp_relation(ring, run_hp)
                           if isinstance(run_hp, str) else 'unknown')
                    summary_rows.append(dict(
                        subject=subject, roi=roi, session=ses, run=run,
                        ring=ring, hp=run_hp, rel=rel,
                        n_voxels=int(W.shape[0]),
                        drive_weighted=weighted,
                        drive_unweighted=unweighted,
                        ind_sum=float(ind_hrf_sum[k]),
                    ))

    return pd.DataFrame(summary_rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bids-folder', default='/data/ds-retsupp')
    p.add_argument('--subjects', type=int, nargs='+',
                   default=[s for s in range(1, 31) if s not in (6, 8)])
    p.add_argument('--rois', nargs='+', default=ROI_ORDER)
    p.add_argument('--model', type=int, default=4)
    p.add_argument('--sd-min', type=float, default=0.5)
    p.add_argument('--r2-min', type=float, default=0.0)
    p.add_argument('--out', type=Path, default=None,
                   help='Default: <bids>/derivatives/decode/decoded_drive.tsv')
    p.add_argument('--n-jobs', type=int, default=1)
    args = p.parse_args()

    bids = Path(args.bids_folder)
    out = args.out or (bids / 'derivatives' / 'decode' / 'decoded_drive.tsv')
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f'Subjects: {args.subjects}')
    print(f'ROIs: {args.rois}')
    print(f'Out: {out}')
    print(f'sd_min={args.sd_min}  r2_min={args.r2_min}', flush=True)

    t0 = time.time()
    if args.n_jobs > 1:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=args.n_jobs, verbose=10)(
            delayed(run_subject)(s, bids, rois=args.rois, model=args.model,
                                  sd_min=args.sd_min, r2_min=args.r2_min)
            for s in args.subjects
        )
    else:
        results = []
        for s in tqdm(args.subjects, desc='subjects'):
            try:
                results.append(run_subject(
                    s, bids, rois=args.rois, model=args.model,
                    sd_min=args.sd_min, r2_min=args.r2_min,
                ))
            except Exception as e:
                print(f'sub-{s:02d}: failed ({e})', flush=True)
                import traceback; traceback.print_exc()
                results.append(pd.DataFrame())

    df = pd.concat([d for d in results if len(d)], ignore_index=True)
    df.to_csv(out, sep='\t', index=False)
    elapsed = time.time() - t0
    print(f'\nWrote {len(df)} rows in {elapsed:.1f}s -> {out}', flush=True)


if __name__ == '__main__':
    main()
