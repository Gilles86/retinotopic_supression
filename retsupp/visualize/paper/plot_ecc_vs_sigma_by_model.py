"""Eccentricity-σ scaling per model (m1 Gaussian, m6 Div Norm), per ROI.

Pools across subjects, loads per-hemisphere fsnative ``ecc``/``sd``/``r2``
giis plus the neuropythy ``lh./rh.inferred_varea`` surface labels to mask
ROIs. Plots ecc-vs-σ hexbins per (model, ROI) with r and slope annotated.

The point this figure makes: m4 (DoG) center σ collapses, so its
surface visualization looks flat — but **m1 (Gaussian) and m6 (Div
Norm) preserve the canonical ecc-σ scaling**.

Usage
-----
    python -m retsupp.visualize.paper.plot_ecc_vs_sigma_by_model \\
        --bids-folder /shares/zne.uzh/gdehol/ds-retsupp \\
        --subjects 1,5,7,10,11,12,13,14,15,16 \\
        --out notes/figures/ecc_vs_sigma_by_model.pdf
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nb
import numpy as np
import pandas as pd

ROIS = [(1, 'V1'), (2, 'V2'), (3, 'V3'), (4, 'hV4')]
MODELS = [
    (1, 'Gaussian PRF (m1)'),
    (3, 'Gaussian + flex HRF (m3)'),
    (6, 'Divisive Normalization (m6)'),
]
YLIM_PER_MODEL = {1: (0, 1.5), 3: (0, 1.5), 6: (0, 2.0)}

plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.titlesize": 16,
})


def load_gii(p):
    try:
        return nb.load(str(p)).agg_data().squeeze()
    except FileNotFoundError:
        return None


def load_surf_label(p):
    return nb.freesurfer.io.read_morph_data(str(p)).astype(int)


def collect(bids, subjects, models, r2_min=0.15, sd_min=0.05, sd_max=8.0,
            ecc_min=0.2, ecc_max=8.0):
    prf = Path(bids) / 'derivatives' / 'prf'
    fs = Path(bids) / 'derivatives' / 'fmriprep' / 'sourcedata' / 'freesurfer'
    rows = []
    for sub in subjects:
        spad = f'sub-{sub:02d}'
        for hemi_full, hemi_short in [('L', 'lh'), ('R', 'rh')]:
            varea_p = fs / spad / 'surf' / f'{hemi_short}.inferred_varea'
            if not varea_p.exists():
                continue
            varea = load_surf_label(varea_p)
            for model, _ in models:
                base = prf / f'model{model}' / spad
                ecc = load_gii(base / f'{spad}_desc-ecc.optim.'
                                       f'nilearn_space-fsnative_hemi-'
                                       f'{hemi_full}.func.gii')
                sd = load_gii(base / f'{spad}_desc-sd.optim.'
                                      f'nilearn_space-fsnative_hemi-'
                                      f'{hemi_full}.func.gii')
                r2 = load_gii(base / f'{spad}_desc-r2.optim.'
                                      f'nilearn_space-fsnative_hemi-'
                                      f'{hemi_full}.func.gii')
                if ecc is None or sd is None or r2 is None:
                    continue
                n = min(varea.size, ecc.size, sd.size, r2.size)
                varea_h = varea[:n]; ecc=ecc[:n]; sd=sd[:n]; r2=r2[:n]
                good = (np.isfinite(ecc) & np.isfinite(sd) & (r2 > r2_min) &
                        (sd > sd_min) & (sd < sd_max) &
                        (ecc > ecc_min) & (ecc < ecc_max))
                if not good.any():
                    continue
                df = pd.DataFrame({
                    'subject': sub, 'hemi': hemi_full, 'model': model,
                    'roi_lbl': varea_h[good],
                    'ecc': ecc[good], 'sd': sd[good], 'r2': r2[good],
                })
                rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def plot(df, out_path):
    fig, axes = plt.subplots(len(MODELS), len(ROIS),
                              figsize=(4.0 * len(ROIS), 3.0 * len(MODELS)),
                              sharex=True, sharey=False)
    if len(MODELS) == 1:
        axes = axes[None, :]
    for i, (model, mname) in enumerate(MODELS):
        for j, (lbl, rname) in enumerate(ROIS):
            ax = axes[i, j]
            sub = df[(df['model'] == model) & (df['roi_lbl'] == lbl)]
            if sub.empty:
                ax.set_title(f"{rname}\n(no data)")
                continue
            x = sub['ecc'].to_numpy()
            y = sub['sd'].to_numpy()
            hb = ax.hexbin(x, y, gridsize=35, cmap='magma_r',
                            mincnt=1, linewidths=0)
            # Linear fit + Pearson r.
            r = float(np.corrcoef(x, y)[0, 1])
            slope, intercept = np.polyfit(x, y, 1)
            xx = np.linspace(0.2, 8, 50)
            ax.plot(xx, slope * xx + intercept, color='#1abc9c', lw=2.0,
                    label=f"r = {r:+.2f}\nslope = {slope:+.2f}")
            ax.legend(loc='upper left', frameon=False, fontsize=10)
            ax.set_xlim(0, 8)
            ylo, yhi = YLIM_PER_MODEL.get(model, (0, 2.0))
            ax.set_ylim(ylo, yhi)
            if i == 0:
                ax.set_title(rname)
            if i == len(MODELS) - 1:
                ax.set_xlabel("Eccentricity (deg)")
            if j == 0:
                ax.set_ylabel(f"{mname}\nσ (deg)")
    fig.suptitle("Eccentricity-σ scaling across PRF model variants\n"
                 f"Pooled across {df['subject'].nunique()} subjects, "
                 f"both hemispheres, r²>0.15")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches='tight')
    print(f"Wrote {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bids-folder', default='/shares/zne.uzh/gdehol/ds-retsupp')
    p.add_argument('--subjects', default='1,5,7,10,11,12,13,14,15,16')
    p.add_argument('--out',
                   default='/shares/zne.uzh/gdehol/ds-retsupp/derivatives/'
                           'figures/ecc_vs_sigma_by_model.pdf')
    a = p.parse_args()
    subs = [int(s) for s in a.subjects.split(',') if s.strip()]
    print(f"Collecting from {len(subs)} subjects...")
    df = collect(a.bids_folder, subs, MODELS)
    print(f"  n voxels total: {len(df):,}")
    print(df.groupby(['model', 'roi_lbl']).size())
    if df.empty:
        raise SystemExit("No data found")
    plot(df, a.out)


if __name__ == '__main__':
    main()
