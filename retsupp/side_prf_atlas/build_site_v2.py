"""Side analysis: pRF atlas website v2, graded display instead of hard masks.

Input: ``group_sums_fsaverage_v3.npz`` (``group_sums.py``; group-level weighted sums only). Every subject
contributes to every vertex with weight ``w = max(cv_r, 0)**2`` (its cross-validated R² proxy). Numerators and
denominators are smoothed on the fsaverage white surface (about 3 mm FWHM; iterated neighbour averaging, the number
of iterations calibrated on the mesh) before taking ratios, so the result is a CV-R²-weighted, smoothed group mean.

Display: each map is blended over curvature with a CONTINUOUS alpha (baked in with ``blend_curvature``; the webgl
viewer has no live alpha for vertex data), so strong retinotopic cortex is opaque and weak vertices fade out. The
fit-quality axis is the group median of the signed CV R² (``sign(cv_r)·cv_r²``, smoothed the same way). For a level
``thr`` the alpha is 0 below ``thr`` and ramps linearly to opaque at ``2·thr``. Polar-angle alpha is also multiplied by
the across-subject agreement (weighted resultant length R, ramp 0.1 -> 0.5). Three levels are offered as separate
layers: 0.0025 / 0.005 / 0.01.

Runs in ``pycortex2``::

    ~/mambaforge/envs/pycortex2/bin/python retsupp/side_prf_atlas/build_site_v2.py \\
        --sums /data/ds-retsupp/derivatives/side_prf_atlas_group/group_sums_fsaverage_v3.npz \\
        --annot-dir <dir with ?h.Yeo2011_7Networks_N1000.annot> --out notes/side_prf_atlas/site_v2
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import cortex
import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).parent))
import build_site as v1  # noqa: E402

SUBJ = v1.SUBJ
LEVELS = (0.0025, 0.005, 0.01)
FWHM_MM = 3.0


def adjacency(hemi):
    pts, polys = cortex.db.get_surf(SUBJ, 'wm', hemisphere=hemi)
    n = len(pts)
    i = np.concatenate([polys[:, 0], polys[:, 1], polys[:, 2]])
    j = np.concatenate([polys[:, 1], polys[:, 2], polys[:, 0]])
    A = sp.coo_matrix((np.ones(len(i)), (i, j)), shape=(n, n)).tocsr()
    A = ((A + A.T) > 0).astype(float) + sp.eye(n)
    A = sp.diags(1 / np.asarray(A.sum(1)).ravel()) @ A          # row-normalised: mean over self + neighbours
    return A.tocsr(), pts


def calibrate(A, pts, fwhm):
    """Number of neighbour-averaging iterations whose kernel has the requested FWHM (measured on the mesh)."""
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    rng = np.random.default_rng(0)
    seeds = rng.choice(len(pts), 20, replace=False)
    X = np.zeros((len(pts), len(seeds)))
    X[seeds, np.arange(len(seeds))] = 1
    for k in range(1, 200):
        X = A.T @ X                                               # spread of a delta
        d2 = ((pts[:, None, :] - pts[seeds][None]) ** 2).sum(-1)
        s2 = (X * d2).sum(0) / X.sum(0) / 2                      # per-dimension variance on the (locally 2D) sheet
        if np.median(s2) >= sigma ** 2:
            return k, float(np.sqrt(np.median(s2)) * 2 * np.sqrt(2 * np.log(2)))
    return k, np.nan


def smoother(fwhm):
    ops = []
    for hemi in ('lh', 'rh'):
        A, pts = adjacency(hemi)
        k, got = calibrate(A, pts, fwhm)
        print(f'{hemi}: {k} iterations -> FWHM {got:.2f} mm')
        ops.append((A, k))

    def smooth(x):            # x: (2, n) -> (2, n)
        out = []
        for h, (A, k) in enumerate(ops):
            v = np.nan_to_num(x[h].astype(float))
            for _ in range(k):
                v = A @ v
            out.append(v)
        return np.stack(out)
    return smooth


def ramp(x, lo, hi):
    return np.clip((x - lo) / (hi - lo), 0, 1)


def flat(x):
    return np.concatenate([x[0], x[1]]).astype(float)


def build(S, out, smooth):
    sm = {k: smooth(S[k]) for k in S.files if k.startswith(('w', 'n_', 'median'))}
    eps = 1e-12
    with np.errstate(invalid='ignore', divide='ignore'):
        q = sm['median_cv_r2']
        Wp = sm['wpos']
        ang = np.arctan2(sm['wpos_sin'], sm['wpos_cos'])
        R = np.hypot(sm['wpos_sin'], sm['wpos_cos']) / (Wp + eps)
        ecc = sm['wpos_ecc'] / (Wp + eps)
        sd = sm['wpos_sd'] / (Wp + eps)
        hrf = sm['wpos_hrf_delay'] / (Wp + eps)
        Wn = sm['wneg']
        ang_n = np.arctan2(sm['wneg_sin'], sm['wneg_cos'])
        R_n = np.hypot(sm['wneg_sin'], sm['wneg_cos']) / (Wn + eps)
        nv = np.maximum(sm['n_valid'], eps)
        prev_cv = sm['n_cv'] / nv
        prev_neg = sm['n_cv_neg'] / nv
        bal = (sm['n_cv_pos'] - sm['n_cv_neg']) / (sm['n_cv_pos'] + sm['n_cv_neg'] + eps)
    cortexok = S['n_valid'] >= 10
    deg = 180 / np.pi
    L = []
    for thr in LEVELS:
        a = ramp(q, thr, 2 * thr) * cortexok
        tag = f'CV R2 > {thr:g}'
        L.append((f'Polar angle ({tag})', ang * deg, a * ramp(R, .1, .5), 'hsv', -180, 180, 'deg (0 = right, 90 = up)',
                  f'CV-R2-weighted circular mean of the model-4 angle over all subjects. Fades in from a group median CV R2 of {thr:g} '
                  f'(transparent) to {2 * thr:g} (opaque), and with the agreement across subjects.'))
        L.append((f'Eccentricity ({tag})', ecc, a, 'viridis', 0, 4, 'deg',
                  f'CV-R2-weighted mean model-4 eccentricity. Bar aperture radius 3.17 deg. Fades in from median CV R2 {thr:g} to {2 * thr:g}.'))
    a_mid = ramp(q, LEVELS[1], 2 * LEVELS[1]) * cortexok
    L += [
        ('pRF size sigma (CV R2 > 0.005)', sd, a_mid, 'plasma', 0, 3, 'deg', 'CV-R2-weighted mean model-4 centre sigma.'),
        ('HRF peak delay (CV R2 > 0.005)', hrf, a_mid, 'cividis', 3.5, 6.5, 's', 'CV-R2-weighted mean model-4 HRF delay.'),
        ('Group median CV R2', q, cortexok * 1.0, 'magma', 0, 0.1, 'CV R2',
         'Median over subjects of the signed cross-validated R2 of the signed Gaussian pRF (fit on one session, tested on the other). This is the fit-quality axis of the other layers.'),
        ('Subjects with a CV pRF (%)', prev_cv * 100, ramp(prev_cv, .05, .3) * cortexok, 'inferno', 0, 80, '% of subjects',
         'Share of subjects whose pRF beats 50 shifted-stimulus fits (p < .05) on the held-out session. Chance is about 4%; fades in from 5% to 30%.'),
        ('Signed pRF: positive vs negative', bal, ramp(prev_cv, .05, .3) * cortexok, 'RdBu_r', -1, 1, '(pos - neg) / (pos + neg)',
         'Among subjects with a cross-validated pRF of consistent sign: +1 all positive, -1 all negative. Fades in with the share of subjects with a CV pRF (5% to 30%).'),
        ('Negative pRFs (% subjects)', prev_neg * 100, ramp(prev_neg, .05, .2) * cortexok, 'Blues', 0, 40, '% of subjects',
         'Share of subjects with a cross-validated negative pRF (negative in the all-runs fit and both session fits). Chance about 2%; fades in from 5% to 20%.'),
        ('Negative pRFs: polar angle', ang_n * deg, ramp(prev_neg, .05, .2) * ramp(R_n, .1, .5) * cortexok, 'hsv', -180, 180, 'deg',
         'CV-R2-weighted circular mean polar angle of negative pRFs; fades with their prevalence and their agreement across subjects.'),
    ]
    ds, legend = {}, {}
    for name, v, a, cmap, lo, hi, unit, desc in L:
        v, a = flat(v), flat(np.nan_to_num(a))
        print(f'{name}: {(a > 0.5).sum()} vertices with alpha > 0.5')
        ds[name] = v1.layer(v, a, cmap, lo, hi)
        legend[name] = dict(cmap=v1.gradient_css(cmap), vmin=f'{lo:g}', vmax=f'{hi:g}', label=unit, desc=desc)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'data').exists():
        shutil.rmtree(out / 'data')
    for f in out.glob('*'):
        if f.is_file():
            f.unlink()
    cortex.webgl.make_static(str(out), ds, types=('inflated',), recache=False, title='retsupp pRF atlas v2', **v1.CURV)
    methods = f"""<details><summary>Methods and display</summary>
<p>pRF parameters come from a difference-of-Gaussians pRF model with a flexible HRF, fitted per participant in volume space and resampled to
fsaverage (nearest voxel at mid-thickness, nearest vertex on the registered sphere).</p>
<p>Fit quality: a signed Gaussian pRF (amplitude may be negative) fitted on one session and tested on the other gives each participant a
cross-validated R2 per vertex. Every participant contributes to every map, weighted by that CV R2; weighted sums are smoothed on the white
surface ({FWHM_MM:g} mm FWHM) before averaging. No hard masks: maps fade in with the group median CV R2. Three levels
(0.0025, 0.005, 0.01; transparent at the level, opaque at twice the level) are separate layers. For scale, a group median CV R2 of 0.0025 / 0.005 / 0.01 is a held-out
correlation of 0.05 / 0.07 / 0.1 in the median participant; values near 0 are what fits to a shifted stimulus give.</p>
<p>Negative pRFs: cross-validated against 50 fits with a circularly shifted stimulus (p &lt; .05), negative in the all-runs fit and in both session fits.</p></details>"""
    v1.inject(out / 'index.html', legend, int(len(S['subjects'])), methods=methods)
    return ds


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--sums', required=True)
    ap.add_argument('--annot-dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--png-dir', default=None)
    a = ap.parse_args()
    v1.make_subject(a.annot_dir)
    S = np.load(a.sums)
    sm = smoother(FWHM_MM)
    ds = build(S, a.out, sm)
    if a.png_dir:
        v1.render_pngs(ds, a.png_dir)
    size = sum(f.stat().st_size for f in Path(a.out).rglob('*') if f.is_file())
    print(f'site: {a.out} ({size / 1e6:.1f} MB)')
