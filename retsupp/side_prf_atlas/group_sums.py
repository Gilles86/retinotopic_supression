"""Side analysis (pRF atlas site v2): group-level weighted sums on fsaverage, for graded (alpha) display.

Instead of binary per-subject masks, every subject contributes to every vertex with a weight equal to its
cross-validated R² proxy, ``w = max(cv_r, 0)**2`` (``cv_r`` = held-out-session correlation of the signed Gaussian
pRF, ``fit_signed_prf.py``), split by the sign of the all-runs amplitude:

- positive maps (model-4 polar angle, eccentricity, sigma, HRF delay): ``w_pos = w`` where amplitude > 0;
- negative maps (signed-fit polar angle, eccentricity, sigma): ``w_neg = w`` where amplitude < 0.

Writes only GROUP sums (Σw, Σw·cosθ, Σw·sinθ, Σw·ecc, ...), so the weighted means can be smoothed on the
sphere locally (smooth numerator and denominator) without any single-subject map leaving the cluster, plus the
group median of the signed CV R² (``sign(cv_r)·cv_r²``) and the subject counts.

Output (never overwritten): ``derivatives/side_prf_atlas_group/group_sums_fsaverage_v3.npz``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

SUBJECTS = range(3, 31)


def main(bids_folder):
    der = Path(bids_folder) / 'derivatives'
    out = der / 'side_prf_atlas_group' / 'group_sums_fsaverage_v3.npz'
    if out.exists():
        print(f'{out} exists, nothing to do')
        return
    acc, cvr2, subs = {}, [], []

    def add(k, v):
        acc[k] = acc.get(k, 0) + np.nan_to_num(v)

    for s in SUBJECTS:
        f = der / 'side_prf_atlas_signed' / f'sub-{s:02d}' / f'sub-{s:02d}_space-fsaverage_maps.npz'
        if not f.exists():
            print(f'missing sub-{s:02d}')
            continue
        d = np.load(f)
        subs.append(s)
        cv = d['cv_r']
        ok = np.isfinite(cv)
        w = np.where(ok, np.clip(cv, 0, None) ** 2, 0)
        cvr2.append(np.where(ok, np.sign(cv) * cv ** 2, np.nan))
        add('n_valid', ok)
        m4ok = np.isfinite(d['m4_x']) & np.isfinite(d['m4_sd']) & np.isfinite(d['m4_hrf_delay'])
        wp = np.where((d['amplitude'] > 0) & m4ok, w, 0)
        th = np.arctan2(d['m4_y'], d['m4_x'])
        add('wpos', wp)
        add('wpos_cos', wp * np.cos(th))
        add('wpos_sin', wp * np.sin(th))
        add('wpos_ecc', wp * np.hypot(d['m4_x'], d['m4_y']))
        add('wpos_sd', wp * d['m4_sd'])
        add('wpos_hrf_delay', wp * d['m4_hrf_delay'])
        wn = np.where(d['amplitude'] < 0, w, 0)
        thn = np.arctan2(d['y'], d['x'])
        add('wneg', wn)
        add('wneg_cos', wn * np.cos(thn))
        add('wneg_sin', wn * np.sin(thn))
        add('wneg_ecc', wn * np.hypot(d['x'], d['y']))
        add('wneg_sd', wn * d['sd'])
        cvs = ok & (d['cv_p'] < .05) & (cv > 0)
        add('n_cv', cvs)
        add('n_cv_pos', cvs & (d['amplitude'] > 0) & (d['sign_f1'] > 0) & (d['sign_f2'] > 0))
        add('n_cv_neg', cvs & (d['amplitude'] < 0) & (d['sign_f1'] < 0) & (d['sign_f2'] < 0))
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        acc['median_cv_r2'] = np.nanmedian(np.stack(cvr2), 0)
    np.savez_compressed(out, **{k: np.asarray(v, np.float32) for k, v in acc.items()}, subjects=np.array(subs))
    print(f'{len(subs)} subjects -> {out}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bids-folder', default='/shares/zne.uzh/gdehol/ds-retsupp')
    main(ap.parse_args().bids_folder)
