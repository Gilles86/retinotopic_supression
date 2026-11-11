"""Quick smoke comparison: alternating vs single-shot joint GD for m3.

Loads both canonical (joint) and ``alt_smoke`` (alternating) PRF parameter
NIfTIs for one subject and one model, restricts to brain voxels, and prints
side-by-side stats: median R², σ-vs-eccentricity slope, phantom count
(R²>0.999 with σ≤0.05).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from nilearn import image, maskers


def load(deriv_base: Path, subject: int, model: int, masker):
    sub_dir = deriv_base / f'model{model}' / f'sub-{subject:02d}'
    if not sub_dir.exists():
        return None
    out = {}
    for k in ('x', 'y', 'sd', 'r2', 'hrf_delay', 'hrf_dispersion',
              'amplitude', 'baseline'):
        fn = sub_dir / f'sub-{subject:02d}_desc-{k}.nii.gz'
        if fn.exists():
            out[k] = masker.transform(str(fn)).flatten()
    out['ecc'] = np.sqrt(out['x']**2 + out['y']**2)
    return out


def slope(ecc, sd, mask):
    e = ecc[mask]
    s = sd[mask]
    if len(e) < 50:
        return float('nan'), 0
    # Linear regression sd = a*ecc + b
    a, b = np.polyfit(e, s, 1)
    return float(a), int(len(e))


def summarise(label, p):
    if p is None:
        print(f'\n=== {label}: NOT FOUND ===')
        return
    r2 = p['r2']
    sd = p['sd']
    ecc = p['ecc']
    finite = np.isfinite(r2) & np.isfinite(sd) & np.isfinite(ecc)
    print(f'\n=== {label} ===')
    print(f'  N voxels (finite r2/sd/ecc): {finite.sum()}')
    print(f'  median R² (all):                  {np.nanmedian(r2):.4f}')
    print(f'  median R² (R²>0):                 '
          f'{np.nanmedian(r2[r2 > 0]):.4f}')
    # Eccentricity-restricted ("real" cortex range)
    ecc_ok = finite & (r2 > 0.1) & (ecc < 6) & (sd > 0.05) & (sd < 10)
    print(f'  N voxels (r2>0.1, ecc<6, 0.05<sd<10): {ecc_ok.sum()}')
    if ecc_ok.sum() > 50:
        a, n = slope(ecc, sd, ecc_ok)
        print(f'  σ-vs-ecc slope (r2>0.1, ecc<6):   {a:+.4f}  (N={n})')
        print(f'  median σ in that subset:           '
              f'{np.median(sd[ecc_ok]):.3f}')

    # Phantom signature: r²>0.999 with degenerate σ
    phantom = finite & (r2 > 0.999) & (sd <= 0.05)
    print(f'  PHANTOM voxels (r²>0.999, σ≤0.05): {phantom.sum()}')
    super_high = finite & (r2 > 0.999)
    print(f'  voxels r²>0.999 (any σ):           {super_high.sum()}')

    if 'hrf_delay' in p:
        hd = p['hrf_delay']
        hp = p['hrf_dispersion']
        fin = np.isfinite(hd) & np.isfinite(hp) & (r2 > 0.1)
        if fin.sum() > 0:
            print(f'  HRF delay (r2>0.1):   median={np.median(hd[fin]):.3f}  '
                  f'IQR=[{np.percentile(hd[fin], 25):.3f}, '
                  f'{np.percentile(hd[fin], 75):.3f}]  '
                  f'min/max=[{hd[fin].min():.3f}, {hd[fin].max():.3f}]')
            print(f'  HRF dispersion:       median={np.median(hp[fin]):.3f}  '
                  f'IQR=[{np.percentile(hp[fin], 25):.3f}, '
                  f'{np.percentile(hp[fin], 75):.3f}]  '
                  f'min/max=[{hp[fin].min():.3f}, {hp[fin].max():.3f}]')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--subject', type=int, default=5)
    p.add_argument('--model', type=int, default=3)
    p.add_argument('--bids-folder',
                   default='/shares/zne.uzh/gdehol/ds-retsupp')
    p.add_argument('--suffix', default='alt_smoke',
                   help='Output suffix used by the smoke run.')
    args = p.parse_args()

    bids = Path(args.bids_folder)
    derivs = bids / 'derivatives'

    # Use canonical model 1 bold mask via Subject would be nicer, but for
    # this comparison we just take a brain-mask NIfTI from m1.
    ref_r2 = (derivs / 'prf' / 'model1' / f'sub-{args.subject:02d}'
              / f'sub-{args.subject:02d}_desc-r2.nii.gz')
    mask = image.math_img('np.isfinite(img).astype("int8")', img=str(ref_r2))
    masker = maskers.NiftiMasker(mask_img=mask)
    masker.fit()

    joint = load(derivs / 'prf', args.subject, args.model, masker)
    alt = load(derivs / f'prf.{args.suffix}', args.subject, args.model,
               masker)
    summarise('joint (canonical, prf/)', joint)
    summarise(f'alternating (prf.{args.suffix}/)', alt)


if __name__ == '__main__':
    main()
