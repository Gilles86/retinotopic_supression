"""Eccentricity-σ scaling across PRF model variants, per ROI.

Pools across subjects, loads per-hemisphere fsnative ``ecc``/``sd``/``r2``
giis plus the neuropythy ``lh./rh.inferred_varea`` surface labels to mask
ROIs. Plots ecc-vs-σ hexbins per (model, ROI) with r and slope annotated.

Layout: 4 rows (m1, m3, m4, m6) × 8 columns (V1, V2, V3, V3AB, hV4, LO,
TO, VO). VO/LO/TO/V3AB merge their two component areas (V3a+V3b,
LO1+LO2, etc.).

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

# (group name, list of inferred_varea labels in that group)
ROI_GROUPS = [
    ('V1',   [1]),
    ('V2',   [2]),
    ('V3',   [3]),
    ('V3AB', [11, 12]),
    ('hV4',  [4]),
    ('LO',   [7, 8]),
    ('TO',   [9, 10]),
    ('VO',   [5, 6]),
]
MODELS = [
    (1, 'Gaussian (m1)'),
    (3, 'Gauss + flexHRF (m3)'),
    (4, 'DoG + flexHRF (m4)'),
    (6, 'DivNorm + flexHRF (m6)'),
]
# Per-model y-axis range for σ — tightened to focus on the bulk.
YLIM_PER_MODEL = {1: (0, 1.5), 3: (0, 1.5), 4: (0, 0.8), 6: (0, 2.0)}

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.titlesize": 15,
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
    # Map raw label -> group name.
    label_to_group = {}
    for gname, labels in ROI_GROUPS:
        for l in labels:
            label_to_group[l] = gname
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
                lbl_h = varea_h[good]
                grp = np.array([label_to_group.get(int(l), '')
                                 for l in lbl_h])
                df = pd.DataFrame({
                    'subject': sub, 'hemi': hemi_full, 'model': model,
                    'roi': grp,
                    'ecc': ecc[good], 'sd': sd[good], 'r2': r2[good],
                })
                df = df[df['roi'] != '']
                if len(df) > 0:
                    rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def plot(df, out_path):
    fig, axes = plt.subplots(len(MODELS), len(ROI_GROUPS),
                              figsize=(2.5 * len(ROI_GROUPS),
                                       2.4 * len(MODELS)),
                              sharex=True, sharey=False)
    if len(MODELS) == 1:
        axes = axes[None, :]
    for i, (model, mname) in enumerate(MODELS):
        ylo, yhi = YLIM_PER_MODEL.get(model, (0, 2.0))
        for j, (gname, _) in enumerate(ROI_GROUPS):
            ax = axes[i, j]
            sub = df[(df['model'] == model) & (df['roi'] == gname)]
            if sub.empty or len(sub) < 30:
                ax.text(0.5, 0.5,
                        f"n={len(sub)}\n(too few)",
                        ha='center', va='center',
                        transform=ax.transAxes, fontsize=9,
                        color="0.5")
                ax.set_xlim(0, 8); ax.set_ylim(ylo, yhi)
                if i == 0: ax.set_title(gname)
                if j == 0: ax.set_ylabel(f"{mname}\nσ (deg)")
                if i == len(MODELS) - 1: ax.set_xlabel("Eccentricity (deg)")
                continue
            x = sub['ecc'].to_numpy()
            y = sub['sd'].to_numpy()
            # Clip y for hexbin so the bulk dominates the density.
            y_clip = np.clip(y, ylo, yhi)
            ax.hexbin(x, y_clip, gridsize=22, cmap='magma_r',
                       mincnt=1, linewidths=0,
                       extent=(0, 8, ylo, yhi))
            # Linear fit on UNCLIPPED y so the slope reflects all data.
            r = float(np.corrcoef(x, y)[0, 1])
            slope, intercept = np.polyfit(x, y, 1)
            xx = np.linspace(0.2, 8, 50)
            ax.plot(xx, slope * xx + intercept, color='#1abc9c', lw=1.8)
            ax.text(0.05, 0.95,
                    f"r={r:+.2f}\nslope={slope:+.2f}\nn={len(sub):,}",
                    transform=ax.transAxes, va='top', ha='left',
                    fontsize=8, color='#0e7d6c')
            ax.set_xlim(0, 8)
            ax.set_ylim(ylo, yhi)
            if i == 0:
                ax.set_title(gname)
            if i == len(MODELS) - 1:
                ax.set_xlabel("Eccentricity (deg)")
            if j == 0:
                ax.set_ylabel(f"{mname}\nσ (deg)")
    fig.suptitle("Eccentricity-σ scaling across PRF models × visual ROIs\n"
                 f"Pooled across {df['subject'].nunique()} subjects, "
                 f"both hemispheres, r²>0.15  (slope from unclipped fit)")
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
    print(df.groupby(['model', 'roi']).size())
    if df.empty:
        raise SystemExit("No data found")
    plot(df, a.out)


if __name__ == '__main__':
    main()
