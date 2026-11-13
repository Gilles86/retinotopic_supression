"""Side analysis (pRF atlas website + negative pRFs): whole-brain signed Gaussian pRF fits with
cross-validation over sessions and a circular-shift null, plus model-free cross-run reliability,
all resampled to fsaverage.

Why: the model-4 (DoG + flexible HRF) fits cannot have negative pRFs in practice. Their grid search
picks the most POSITIVELY correlated pRF (braincoder ``fit_grid(..., use_correlation_cost=True)`` with
the default ``positive_amplitude=True``) and ``refine_baseline_and_amplitude`` clips the amplitude to
>= 1e-4 before gradient descent, so a voxel that deactivates when the bar covers its pRF is fitted
as a positive pRF somewhere else.

Per subject (cleaned BOLD from ``derivatives/cleaned_bold_cache``, all voxels in the BOLD mask):

1. Model-free cross-run reliability (``rel_r``, ``rel_p``): ``voxel_reliability_null.reliability_null``
   on all runs (corr of each run with the mean of the other runs; null = the other-runs mean circularly
   shifted by 30-228 TRs, 100 shift sets).
2. Signed Gaussian pRF, grid fit (bar-only stimulus, canonical SPM HRF as in ``fit_prf``: delay 4.5 s,
   dispersion 0.75): every grid pRF is correlated with the voxel's session-mean time course (the bar
   sequence is identical in every run) and the pRF with the largest ABSOLUTE correlation wins, so the
   amplitude can take either sign. Grid: x, y in [-4.5, 4.5] deg (31 steps, centres with ecc <= 4.6),
   sigma geometric 0.25-6 deg (10 steps).
3. Cross-validation over sessions: fit on the ses-1 mean, predict the ses-2 mean, and vice versa;
   ``cv_r`` = mean over the two folds of corr(sign * prediction, held-out time course), i.e. a pRF whose
   sign does not replicate scores negative.
4. Null: the same two-fold procedure with the grid predictions circularly shifted by a random 30-228 TRs
   (50 shifts, the same shift for training and test), ``cv_p = (#null >= cv_r + 1) / 51``.
5. A non-spatial comparison model: the fraction of the aperture covered by the bar, convolved with the
   same HRF, cross-validated the same way (``glob_cv_r``).
6. Parameters for the maps: the signed fit on the mean of all runs (``x, y, sd, amplitude, baseline,
   r_fit``) and the per-fold fits (``x_f1, y_f1, sd_f1, sign_f1`` trained on ses 1, ``*_f2`` on ses 2).

Every map, plus the model-4 parameters, is written to fsaverage by sampling the nearest voxel at each
fsnative mid-thickness vertex and taking the nearest fsnative vertex on ``?h.sphere.reg`` for every
fsaverage vertex (no averaging, so binary maps stay binary and angles are not averaged).

Outputs (never overwritten), ``derivatives/side_prf_atlas_signed/sub-XX/``:
``sub-XX_desc-<map>_bold.nii.gz`` (float32), ``sub-XX_space-fsaverage_maps.npz`` (each map ``(2, 163842)``,
L then R) and ``sub-XX_fit_info.json``.

Usage::

    python -m retsupp.side_prf_atlas.fit_signed_prf 3 --bids-folder /shares/zne.uzh/gdehol/ds-retsupp
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.signal import fftconvolve
from scipy.spatial import cKDTree

from retsupp.modeling.cv.voxel_reliability_null import reliability_null
from retsupp.utils.data import Subject

T_RUN = 258
TR = 1.6
RES = 50
N_NULL = 50
SHIFT_RANGE = (30, 228)
GRID_XY = np.linspace(-4.5, 4.5, 31)
MAX_ECC = 4.6
GRID_SD = np.geomspace(0.25, 6.0, 10)
CHUNK = 8000
OUT_SUBDIR = 'side_prf_atlas_signed'
M4_PARS = ('x', 'y', 'sd', 'amplitude', 'baseline', 'srf_amplitude', 'srf_size',
           'hrf_delay', 'hrf_dispersion', 'r2')


def canonical_hrf():
    from braincoder.hrf import SPMHRFModel
    return np.asarray(SPMHRFModel(tr=TR, delay=4.5, dispersion=0.75).get_hrf(), dtype=np.float64).ravel()


def convolve(neural, hrf):
    """(T, K) -> (T, K), causal, padded with the first sample (as braincoder's HRFModel)."""
    L = len(hrf)
    padded = np.concatenate([np.repeat(neural[:1], L, 0), neural], 0)
    return fftconvolve(padded, hrf[:, None], mode='full', axes=0)[L:L + neural.shape[0]]


def bar_stimulus(sub):
    """Bar-only stimulus (T, G) and grid (G, 2); checks that every run shows the same sequence."""
    stims = []
    for ses in (1, 2):
        for run in sub.get_runs(ses):
            stims.append(sub.get_stimulus(session=ses, run=run, resolution=RES).astype(np.float32)[:T_RUN])
    ref = stims[0]
    mismatch = [float(np.abs(s - ref).mean()) for s in stims]
    gx, gy = sub.get_grid_coordinates(session=1, run=sub.get_runs(1)[0], resolution=RES)
    grid = np.stack((gx.ravel(), gy.ravel()), 1).astype(np.float64)
    return ref.reshape(T_RUN, -1).astype(np.float64), grid, mismatch


def grid_predictions(stim, grid, hrf):
    xx, yy = np.meshgrid(GRID_XY, GRID_XY)
    keep = np.sqrt(xx ** 2 + yy ** 2) <= MAX_ECC
    pos = np.stack((xx[keep], yy[keep]), 1)
    pars = np.array([(x, y, s) for x, y in pos for s in GRID_SD])
    d2 = ((grid[None, :, 0] - pars[:, None, 0]) ** 2 + (grid[None, :, 1] - pars[:, None, 1]) ** 2)
    rf = np.exp(-d2 / (2 * pars[:, 2:3] ** 2))                     # (K, G), peak 1
    pred = convolve(stim @ rf.T, hrf)                                # (T, K)
    sd = pred.std(0)
    ok = sd > 1e-3 * sd.max()
    pars, pred = pars[ok], pred[:, ok]
    glob = convolve(stim.mean(1, keepdims=True), hrf)   # bar-covered fraction of the stimulus grid
    return pars, pred, glob


def unit(a):
    a = a - a.mean(0, keepdims=True)
    n = np.sqrt((a ** 2).sum(0, keepdims=True))
    return a / np.where(n > 0, n, np.nan)


def best_fit(Pn, Yn):
    """Pn (T, K), Yn (T, v) unit-norm demeaned -> best k (largest |r|) and its signed r."""
    C = Pn.T @ Yn
    C = np.nan_to_num(C, nan=0.0)
    k = np.abs(C).argmax(0)
    return k, C[k, np.arange(C.shape[1])]


def heldout(Pn, k, sign, Yn):
    return sign * np.einsum('tv,tv->v', Pn[:, k], Yn)


def write_nifti(values, mask_img, mask_bool, path):
    vol = np.zeros(mask_bool.shape, np.float32)
    vol[mask_bool] = values
    img = nib.Nifti1Image(vol, mask_img.affine)
    img.set_data_dtype(np.float32)
    img.header.set_slope_inter(slope=1, inter=0)
    img.to_filename(str(path))


class FsaverageSampler:
    """Volume (BOLD mask grid) -> fsnative mid-thickness (nearest voxel) -> fsaverage (nearest sphere.reg)."""

    def __init__(self, sub, bids_folder, mask_img, mask_bool):
        surf = sub.get_surf_info()
        fs_root = Path(bids_folder) / 'derivatives' / 'fmriprep' / 'sourcedata' / 'freesurfer'
        inv = np.linalg.inv(mask_img.affine)
        lin = -np.ones(mask_bool.shape, np.int64)
        lin[mask_bool] = np.arange(mask_bool.sum())
        self.idx = []
        self.fs_dir = None
        for hemi, fh in (('L', 'lh'), ('R', 'rh')):
            white = nib.load(str(surf[hemi]['inner'])).agg_data('NIFTI_INTENT_POINTSET')
            pial = nib.load(str(surf[hemi]['outer'])).agg_data('NIFTI_INTENT_POINTSET')
            mid = (white + pial) / 2
            ijk = np.rint(nib.affines.apply_affine(inv, mid)).astype(int)
            inside = np.all((ijk >= 0) & (ijk < np.array(mask_bool.shape)), 1)
            vox = -np.ones(len(mid), np.int64)
            vox[inside] = lin[tuple(ijk[inside].T)]
            # The FreeSurfer dir whose surface has the same vertices as fmriprep's GIfTI.
            for cand in (f'sub-{sub.subject_id:02d}', f'sub-{sub.subject_id:02d}_ses-1'):
                f = fs_root / cand / 'surf' / f'{fh}.sphere.reg'
                if f.exists() and len(nib.freesurfer.read_geometry(str(f))[0]) == len(mid):
                    sph = nib.freesurfer.read_geometry(str(f))[0]
                    self.fs_dir = cand
                    break
            else:
                raise RuntimeError(f'no sphere.reg with {len(mid)} vertices for sub-{sub.subject_id:02d} {fh}')
            fsa = nib.freesurfer.read_geometry(str(fs_root / 'fsaverage' / 'surf' / f'{fh}.sphere.reg'))[0]
            assert len(fsa) == 163842, len(fsa)
            _, nn = cKDTree(sph).query(fsa)
            self.idx.append(vox[nn])                                # fsaverage vertex -> voxel (-1 = none)

    def __call__(self, values):
        out = np.full((2, 163842), np.nan, np.float32)
        for h, ix in enumerate(self.idx):
            ok = ix >= 0
            out[h, ok] = values[ix[ok]]
        return out


def main(subject, bids_folder, canary_voxels=None):
    t0 = time.time()
    bids_folder = Path(bids_folder)
    der = bids_folder / 'derivatives'
    out_dir = der / OUT_SUBDIR / f'sub-{subject:02d}'
    npz_out = out_dir / f'sub-{subject:02d}_space-fsaverage_maps.npz'
    if npz_out.exists():
        print(f'{npz_out} exists, nothing to do')
        return
    sub = Subject(subject, bids_folder)
    mask_img = sub.get_bold_mask(session=1, run=sub.get_runs(1)[0])
    mask_bool = np.asarray(mask_img.dataobj).astype(bool)
    n_runs = (len(sub.get_runs(1)), len(sub.get_runs(2)))

    cache = der / 'cleaned_bold_cache' / f'sub-{subject:02d}' / f'sub-{subject:02d}_kind-full_res-50.npz'
    bold = np.load(cache)['bold']
    assert bold.shape == (sum(n_runs) * T_RUN, mask_bool.sum()), (bold.shape, n_runs, mask_bool.sum())
    print(f'[{time.time()-t0:.0f}s] bold {bold.shape} from {cache.name}', flush=True)
    V = bold.shape[1]
    vox_ix = np.arange(V)
    if canary_voxels:
        vox_ix = np.random.default_rng(0).choice(V, canary_voxels, replace=False)

    hrf = canonical_hrf()
    stim, grid, mismatch = bar_stimulus(sub)
    print(f'stimulus mismatch vs run 1 (mean abs per run): {np.round(mismatch, 4).tolist()}', flush=True)
    pars, pred, glob = grid_predictions(stim, grid, hrf)
    Pn, Gn = unit(pred).astype(np.float32), unit(glob).astype(np.float32)
    pred_raw_mean, pred_raw_norm = pred.mean(0), np.sqrt(((pred - pred.mean(0)) ** 2).sum(0))
    print(f'grid: {len(pars)} pRFs', flush=True)
    rng = np.random.default_rng([subject, 7])
    shifts = rng.integers(*SHIFT_RANGE, size=N_NULL)
    Pn_null = [np.roll(Pn, s, 0) for s in shifts]

    keys = ['rel_r', 'rel_null95', 'rel_p', 'cv_r', 'cv_null_mean', 'cv_null95', 'cv_p', 'glob_cv_r',
            'x', 'y', 'sd', 'amplitude', 'baseline', 'r_fit',
            'x_f1', 'y_f1', 'sd_f1', 'sign_f1', 'cv_r_f1', 'x_f2', 'y_f2', 'sd_f2', 'sign_f2', 'cv_r_f2', 'bold_sd']
    res = {k: np.full(V, np.nan, np.float32) for k in keys}

    n1 = n_runs[0]
    for c0 in range(0, len(vox_ix), CHUNK):
        tc = time.time()
        vi = vox_ix[c0:c0 + CHUNK]
        runs = bold[:, vi].reshape(sum(n_runs), T_RUN, len(vi)).astype(np.float64)
        res['bold_sd'][vi] = runs.std(1).mean(0)
        r_obs, null = reliability_null(runs.astype(np.float32), np.random.default_rng([subject, c0]))
        res['rel_r'][vi] = r_obs
        res['rel_null95'][vi] = np.quantile(null, .95, axis=1)
        res['rel_p'][vi] = ((null >= r_obs[:, None]).sum(1) + 1) / (null.shape[1] + 1)

        m1, m2, mall = runs[:n1].mean(0), runs[n1:].mean(0), runs.mean(0)
        Y = [unit(m1).astype(np.float32), unit(m2).astype(np.float32)]
        cv = []
        for f, (tr, te) in enumerate(((0, 1), (1, 0))):
            k, r = best_fit(Pn, Y[tr])
            s = np.sign(r)
            cvf = heldout(Pn, k, s, Y[te])
            cv.append(cvf)
            res[f'x_f{f+1}'][vi], res[f'y_f{f+1}'][vi], res[f'sd_f{f+1}'][vi] = pars[k].T
            res[f'sign_f{f+1}'][vi] = s
            res[f'cv_r_f{f+1}'][vi] = cvf
        cv_r = np.mean(cv, 0)
        res['cv_r'][vi] = cv_r
        gcv = []
        for tr, te in ((0, 1), (1, 0)):
            g = (Gn[:, 0] @ Y[tr])
            gcv.append(np.sign(g) * (Gn[:, 0] @ Y[te]))
        res['glob_cv_r'][vi] = np.mean(gcv, 0)

        null_cv = np.empty((len(vi), N_NULL))
        for j, Pj in enumerate(Pn_null):
            acc = 0
            for tr, te in ((0, 1), (1, 0)):
                k, r = best_fit(Pj, Y[tr])
                acc = acc + heldout(Pj, k, np.sign(r), Y[te])
            null_cv[:, j] = acc / 2
        res['cv_null_mean'][vi] = null_cv.mean(1)
        res['cv_null95'][vi] = np.quantile(null_cv, .95, axis=1)
        res['cv_p'][vi] = ((null_cv >= cv_r[:, None]).sum(1) + 1) / (N_NULL + 1)

        Yall = mall - mall.mean(0)
        k, r = best_fit(Pn, unit(mall).astype(np.float32))
        amp = r * np.sqrt((Yall ** 2).sum(0)) / pred_raw_norm[k]
        res['x'][vi], res['y'][vi], res['sd'][vi] = pars[k].T
        res['amplitude'][vi] = amp
        res['baseline'][vi] = mall.mean(0) - amp * pred_raw_mean[k]
        res['r_fit'][vi] = r
        print(f'[{time.time()-t0:.0f}s] voxels {c0}-{c0+len(vi)} of {len(vox_ix)} ({time.time()-tc:.1f}s/chunk)',
              flush=True)

    if canary_voxels:
        sel = vox_ix
        print('canary summary:')
        for k in ('rel_r', 'rel_p', 'cv_r', 'cv_null_mean', 'cv_null95', 'cv_p', 'glob_cv_r', 'r_fit', 'sd'):
            print(f'  {k}: ' + ' '.join(f'{q:.3f}' for q in np.nanpercentile(res[k][sel], [5, 25, 50, 75, 95])))
        sig = (res['cv_p'][sel] < .05) & (res['cv_r'][sel] > 0)
        print(f'  cv-reliable: {sig.mean():.3f}; negative among them: {(res["amplitude"][sel][sig] < 0).mean():.3f}; '
              f'rel_p<.05: {(res["rel_p"][sel] < .05).mean():.3f}')
        print(f'canary done in {time.time()-t0:.0f}s')
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    for k, v in res.items():
        write_nifti(v, mask_img, mask_bool, out_dir / f'sub-{subject:02d}_desc-{k}_bold.nii.gz')
    sampler = FsaverageSampler(sub, bids_folder, mask_img, mask_bool)
    surf = {k: sampler(v) for k, v in res.items()}
    m4_dir = der / 'prf' / 'model4' / f'sub-{subject:02d}'
    for p in M4_PARS:
        fn = m4_dir / f'sub-{subject:02d}_desc-{p}.nii.gz'
        img = nib.load(str(fn))
        assert img.shape[:3] == mask_bool.shape and np.allclose(img.affine, mask_img.affine), fn
        surf[f'm4_{p}'] = sampler(np.asarray(img.dataobj, dtype=np.float32)[mask_bool])
    np.savez_compressed(npz_out, **surf)
    info = dict(subject=subject, n_runs=n_runs, n_voxels=int(V), n_grid=int(len(pars)), n_null=N_NULL,
                shifts=shifts.tolist(), grid_xy=GRID_XY.tolist(), max_ecc=MAX_ECC, grid_sd=GRID_SD.tolist(),
                hrf=dict(model='SPMHRFModel', tr=TR, delay=4.5, dispersion=0.75),
                stimulus='Subject.get_stimulus (bar only), res 50', stimulus_mismatch=mismatch,
                freesurfer_dir=sampler.fs_dir, runtime_s=time.time() - t0)
    (out_dir / f'sub-{subject:02d}_fit_info.json').write_text(json.dumps(info, indent=1))
    print(f'[{time.time()-t0:.0f}s] wrote {out_dir}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('subject', type=int)
    ap.add_argument('--bids-folder', default='/shares/zne.uzh/gdehol/ds-retsupp')
    ap.add_argument('--canary-voxels', type=int, default=None,
                    help='fit only this many random voxels, print a summary, write nothing')
    a = ap.parse_args()
    main(a.subject, a.bids_folder, a.canary_voxels)
