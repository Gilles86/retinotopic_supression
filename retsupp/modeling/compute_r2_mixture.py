"""Per-(subject, ROI) 2-component **Beta** mixture on raw R² ∈ [0,1].

Beta is the natural distribution on a bounded support and fits R²
distributions much better than a Gaussian mixture on arctanh-R² (which
≈ identity for small R² and so barely separates the noise peak from
low-R² signal).

For every voxel we save the posterior probability of belonging to the
"signal" component (the one with the larger mean) as
``desc-p_signal.nii.gz``. Threshold at 0.5 (majority signal) or 0.95
(conservative) at analysis time.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from nilearn import image, maskers
from scipy.stats import beta as beta_dist

from retsupp.utils.data import Subject

ROI_DEFAULTS = ['V1', 'V2', 'V3', 'V3AB', 'hV4', 'VO', 'LO', 'TO']


def _moment_match_beta(weights: np.ndarray, x: np.ndarray):
    """Beta MoM from weighted samples; clamp to >0.5 to avoid pathology."""
    w_sum = weights.sum()
    if w_sum < 1e-9:
        return 1.0, 1.0
    mu = (weights * x).sum() / w_sum
    var = (weights * (x - mu) ** 2).sum() / w_sum
    if var <= 1e-9 or mu <= 0 or mu >= 1:
        return 1.0, 1.0
    nu = mu * (1 - mu) / var - 1
    if nu <= 0:
        return 1.0, 1.0
    return max(0.5, mu * nu), max(0.5, (1 - mu) * nu)


def beta_mixture_em(x: np.ndarray, max_iter: int = 400,
                     tol: float = 1e-6, restarts: int = 12) -> dict:
    """2-component Beta mixture via EM. Robust to component collapse:
    weights are clamped to [0.02, 0.98], and the noise component's mean
    is constrained to stay below the data median. Many restarts; pick
    the one with the highest log-likelihood."""
    x = np.clip(x, 1e-6, 1 - 1e-6).astype(np.float64)
    rng = np.random.default_rng(0)
    median_x = float(np.median(x))

    inits = []
    # Init 1: classic — noise tight near 0, signal broad
    inits.append((np.array([1.5, 2.5]), np.array([30.0, 6.0]),
                   np.array([0.8, 0.2])))
    # Init 2: quantile-driven — noise from <Q50, signal from >Q90
    lo = x[x < median_x]
    hi = x[x > np.quantile(x, 0.90)]
    if len(lo) > 5 and len(hi) > 5:
        a0, b0 = _moment_match_beta(np.ones(len(lo)), lo)
        a1, b1 = _moment_match_beta(np.ones(len(hi)), hi)
        inits.append((np.array([a0, a1]), np.array([b0, b1]),
                       np.array([0.9, 0.1])))
    # Random inits
    for _ in range(restarts - len(inits)):
        mu0 = rng.uniform(0.002, max(0.05, median_x))
        mu1 = rng.uniform(0.10, 0.50)
        a = np.array([2.0, 2.0 + 4*mu1])
        b = np.array([(1-mu0)/mu0 * a[0], (1-mu1)/mu1 * a[1]])
        inits.append((a, b, np.array([0.9, 0.1])))

    best = None
    for a_init, b_init, w_init in inits:
        a, b, w = a_init.copy(), b_init.copy(), w_init.copy()
        prev_ll = -np.inf
        for it in range(max_iter):
            log_p = np.column_stack([
                beta_dist.logpdf(x, a[0], b[0]) + np.log(w[0] + 1e-12),
                beta_dist.logpdf(x, a[1], b[1]) + np.log(w[1] + 1e-12),
            ])
            m = log_p.max(axis=1, keepdims=True)
            log_norm = m + np.log(np.exp(log_p - m).sum(axis=1, keepdims=True))
            ll = log_norm.sum()
            resp = np.exp(log_p - log_norm)
            n_k = resp.sum(axis=0)
            w_new = np.clip(n_k / len(x), 0.02, 0.98)
            w = w_new / w_new.sum()
            for k in range(2):
                a[k], b[k] = _moment_match_beta(resp[:, k], x)
            # Identify "noise" as the component with the smaller mean.
            means = a / (a + b)
            noise_idx = int(np.argmin(means))
            # If noise component drifted to mean > overall median, rein it
            # back: re-fit it on the lower-half data only.
            if means[noise_idx] > median_x and (x < median_x).any():
                a[noise_idx], b[noise_idx] = _moment_match_beta(
                    np.ones((x < median_x).sum()), x[x < median_x])
            if abs(ll - prev_ll) < tol:
                break
            prev_ll = ll
        # Sanity: require components not be near-identical
        means = a / (a + b)
        if abs(means[0] - means[1]) < 0.01:
            continue   # collapsed; skip this restart
        if best is None or ll > best['ll']:
            best = {'a': a.copy(), 'b': b.copy(), 'w': w.copy(),
                    'resp': resp.copy(), 'll': ll}
    if best is None:
        # Emergency fallback — degenerate case where every restart collapsed.
        a = np.array([2.0, 4.0]); b = np.array([50.0, 10.0]); w = np.array([0.8, 0.2])
        log_p = np.column_stack([
            beta_dist.logpdf(x, a[0], b[0]) + np.log(w[0]),
            beta_dist.logpdf(x, a[1], b[1]) + np.log(w[1])])
        log_p -= log_p.max(axis=1, keepdims=True)
        p = np.exp(log_p); resp = p / p.sum(axis=1, keepdims=True)
        best = {'a': a, 'b': b, 'w': w, 'resp': resp, 'll': -np.inf}
    return best


def fit_one_roi(r2: np.ndarray) -> dict:
    """Fit 2-component Beta mixture on R². Returns posterior P(signal)
    and a fit summary."""
    if len(r2) < 50:
        return {
            'p_signal': np.full(len(r2), np.nan),
            'fit': None,
            'reason': f'n_voxels={len(r2)} too small for stable mixture',
        }
    x = np.clip(r2, 1e-6, 1 - 1e-6)
    fit = beta_mixture_em(x)
    means = fit['a'] / (fit['a'] + fit['b'])
    sig_idx = int(np.argmax(means))
    posteriors = fit['resp'][:, sig_idx]
    return {
        'p_signal': posteriors.astype(np.float32),
        'fit': {
            'noise_alpha': float(fit['a'][1 - sig_idx]),
            'noise_beta':  float(fit['b'][1 - sig_idx]),
            'signal_alpha': float(fit['a'][sig_idx]),
            'signal_beta':  float(fit['b'][sig_idx]),
            'noise_weight':  float(fit['w'][1 - sig_idx]),
            'signal_weight': float(fit['w'][sig_idx]),
            'noise_mean':  float(means[1 - sig_idx]),
            'signal_mean': float(means[sig_idx]),
            'log_likelihood': float(fit['ll']),
            'n_voxels': int(len(x)),
        },
        'reason': 'ok',
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--subject', type=int, required=True)
    p.add_argument('--bids-folder', default='/data/ds-retsupp')
    p.add_argument('--model', type=int, default=1)
    p.add_argument('--rois', nargs='+', default=ROI_DEFAULTS)
    p.add_argument('--prf-base-dir', default='prf',
                   help='Subdir under derivatives/ (e.g. prf, prf_bar).')
    args = p.parse_args()

    bids = Path(args.bids_folder)
    sub = Subject(args.subject, bids)
    masker = maskers.NiftiMasker(mask_img=sub.get_bold_mask())
    masker.fit()

    r2_path = (bids / 'derivatives' / args.prf_base_dir
               / f'model{args.model}' / f'sub-{args.subject:02d}'
               / f'sub-{args.subject:02d}_desc-r2.nii.gz')
    r2 = masker.transform(str(r2_path)).flatten()

    # Per-voxel posterior, init at NaN.
    p_signal_all = np.full(r2.size, np.nan, dtype=np.float32)
    summary = {}

    for roi in args.rois:
        try:
            roi_img = sub.get_retinotopic_roi(roi=roi, bold_space=True)
            roi_mask = masker.transform(roi_img).flatten().astype(bool)
        except Exception as e:
            summary[roi] = {'reason': f'roi load failed: {e}'}
            continue
        idx = np.where(roi_mask & np.isfinite(r2) & (r2 > 0) & (r2 < 0.99))[0]
        if len(idx) < 50:
            summary[roi] = {'reason': f'only {len(idx)} usable voxels'}
            continue
        out = fit_one_roi(r2[idx])
        if out['fit'] is not None:
            p_signal_all[idx] = out['p_signal']
            summary[roi] = out['fit']
        else:
            summary[roi] = {'reason': out['reason']}

    out_dir = (bids / 'derivatives' / args.prf_base_dir
               / f'model{args.model}' / f'sub-{args.subject:02d}')
    nii = masker.inverse_transform(p_signal_all)
    out_nii = out_dir / f'sub-{args.subject:02d}_desc-p_signal.nii.gz'
    nii.to_filename(str(out_nii))
    out_json = out_dir / f'sub-{args.subject:02d}_desc-p_signal.json'
    with open(out_json, 'w') as fh:
        json.dump(summary, fh, indent=2)
    print(f'Wrote {out_nii}')
    print(f'Wrote {out_json}')
    for roi, info in summary.items():
        if 'reason' in info and 'separation_z' not in info:
            print(f'  {roi}: skipped ({info["reason"]})')
        else:
            print(f'  {roi}: signal-mean={info.get("signal_mean", float("nan")):.3f} '
                  f'noise-mean={info.get("noise_mean", float("nan")):.3f} '
                  f'sep-z={info.get("separation_z", float("nan")):.2f} '
                  f'p_sig>0.5: {(p_signal_all[~np.isnan(p_signal_all)] > 0.5).sum()} voxels')


if __name__ == '__main__':
    main()
