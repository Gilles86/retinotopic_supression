"""Per-subject × per-ROI gain-field visualisations from a fitted AF model.

For each (subject, ROI) read the *_af-prf-pars.tsv from
``derivatives/af_prf_joint_dynamic_v3_dog`` (DoG dyn-v3) and render the
2D sustained and dynamic gain fields on a canonical layout (HP at
upper-right; the other three ring positions are LP).

The "gain" plotted is the sum-of-Gaussians term (added to 1 in the
model), so 0 = no modulation, negative = suppression, positive =
facilitation. Diverging colormap centred at 0.

Output: ``notes/figures/gain_fields_<page>.pdf``  (one PDF, multi-page).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import TwoSlopeNorm
from tqdm import tqdm

ECC = 4.0  # ring eccentricity (deg) — same as `ecc_distractor`
HP_XY = (ECC / np.sqrt(2),  ECC / np.sqrt(2))      # upper-right
LP_XY = [
    (-ECC / np.sqrt(2),  ECC / np.sqrt(2)),        # upper-left
    (-ECC / np.sqrt(2), -ECC / np.sqrt(2)),        # lower-left
    ( ECC / np.sqrt(2), -ECC / np.sqrt(2)),        # lower-right
]
ROI_ORDER = ['V1', 'V2', 'V3', 'V3AB', 'hV4', 'VO', 'LO', 'TO']

AF_DIR_DEFAULT = ('/data/ds-retsupp/derivatives/'
                   'af_prf_joint_dynamic_v3_dog')


def gain_field(grid_x, grid_y, sigma, g_HP, g_LP):
    """Sum-of-Gaussians gain field on (n_y, n_x) grids."""
    XX, YY = grid_x, grid_y
    out = g_HP * np.exp(-((XX - HP_XY[0]) ** 2 + (YY - HP_XY[1]) ** 2)
                         / (2.0 * sigma ** 2))
    for lx, ly in LP_XY:
        out += g_LP * np.exp(-((XX - lx) ** 2 + (YY - ly) ** 2)
                              / (2.0 * sigma ** 2))
    return out


def read_pars(af_dir, subject, roi):
    fn = (Path(af_dir) / f'sub-{subject:02d}'
          / f'sub-{subject:02d}_roi-{roi}_mode-signed_'
            f'dog-dyn-v3-af-prf-pars.tsv')
    if not fn.exists():
        return None
    df = pd.read_csv(fn, sep='\t')
    if len(df) == 0:
        return None
    # All AF parameters are shared across voxels — take the first row.
    row = df.iloc[0]
    return {
        'sigma_AF': float(row['sigma_AF']),
        'g_HP':     float(row['g_HP']),
        'g_LP':     float(row['g_LP']),
        'sigma_dyn': float(row['sigma_dyn']),
        'g_HP_dyn': float(row['g_HP_dyn']),
        'g_LP_dyn': float(row['g_LP_dyn']),
        'r2_med':   float(np.median(df['r2'])),
        'n_voxels': len(df),
    }


def render_page(pdf, all_pars, kind, opts):
    """One PDF page: rows=subjects, cols=ROIs.  kind ∈ {'sustained','dynamic'}.
    Each cell is the 2D gain field (sum-of-Gaussians term)."""
    n_sub = len(opts.subjects)
    n_roi = len(opts.rois)
    fig, axes = plt.subplots(n_sub, n_roi,
                              figsize=(1.05 * n_roi, 1.05 * n_sub),
                              sharex=True, sharey=True, squeeze=False)

    # Build grid once.
    grid = np.linspace(-opts.fov, opts.fov, opts.resolution)
    XX, YY = np.meshgrid(grid, grid)

    # Pre-compute every (sub, roi) field so we can choose a SHARED
    # symmetric colour-limit across the whole page.
    fields = {}
    for s in opts.subjects:
        for roi in opts.rois:
            p = all_pars.get((s, roi))
            if p is None:
                continue
            if kind == 'sustained':
                f = gain_field(XX, YY, p['sigma_AF'],
                                p['g_HP'], p['g_LP'])
            else:
                f = gain_field(XX, YY, p['sigma_dyn'],
                                p['g_HP_dyn'], p['g_LP_dyn'])
            fields[(s, roi)] = f
    if not fields:
        return
    abs_max = float(np.percentile(
        np.concatenate([np.abs(f).ravel() for f in fields.values()]),
        99.5))
    abs_max = max(abs_max, 1e-3)
    norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0.0, vmax=abs_max)

    for ri, s in enumerate(opts.subjects):
        for ci, roi in enumerate(opts.rois):
            ax = axes[ri, ci]
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.3); spine.set_color('0.6')
            f = fields.get((s, roi))
            if f is None:
                ax.text(0.5, 0.5, '–', ha='center', va='center',
                         color='0.6', fontsize=8, transform=ax.transAxes)
                continue
            ax.imshow(f, extent=[-opts.fov, opts.fov, -opts.fov, opts.fov],
                       origin='lower', cmap='RdBu_r', norm=norm,
                       interpolation='nearest')
            # Mark ring positions: HP filled red, LPs grey.
            ax.plot(*HP_XY, 'o', mfc='none', mec='k', mew=0.6, ms=4)
            for lx, ly in LP_XY:
                ax.plot(lx, ly, 'o', mfc='none', mec='0.4', mew=0.4, ms=3)
            # Annotate g_HP / g_LP values in the corner.
            if kind == 'sustained':
                txt = f"g={p_get(all_pars[(s, roi)], 'g_HP'):.2f}/" \
                      f"{p_get(all_pars[(s, roi)], 'g_LP'):.2f}\n" \
                      f"σ={p_get(all_pars[(s, roi)], 'sigma_AF'):.1f}"
            else:
                txt = f"g={p_get(all_pars[(s, roi)], 'g_HP_dyn'):.2f}/" \
                      f"{p_get(all_pars[(s, roi)], 'g_LP_dyn'):.2f}\n" \
                      f"σ={p_get(all_pars[(s, roi)], 'sigma_dyn'):.1f}"
            ax.text(0.02, 0.98, txt, transform=ax.transAxes,
                     fontsize=4.5, va='top', ha='left', color='0.15',
                     bbox=dict(facecolor='white', alpha=0.6,
                               edgecolor='none', pad=0.5))
            if ri == 0:
                ax.set_title(roi, fontsize=8, pad=2, weight='bold')
            if ci == 0:
                ax.set_ylabel(f'sub-{s:02d}', fontsize=7,
                               rotation=0, ha='right', va='center',
                               labelpad=10)

    cbar_ax = fig.add_axes([0.92, 0.30, 0.012, 0.40])
    sm = plt.cm.ScalarMappable(cmap='RdBu_r', norm=norm)
    cb = fig.colorbar(sm, cax=cbar_ax)
    cb.set_label('gain (sum-of-Gaussians)', fontsize=8)
    cb.ax.tick_params(labelsize=7)

    label = ('SUSTAINED gain  (σ=σ_AF, g=g_HP/g_LP)' if kind == 'sustained'
             else 'DYNAMIC gain  (σ=σ_dyn, g=g_HP_dyn/g_LP_dyn,  '
                  'modulates per-trial pulse)')
    fig.suptitle(f'AF model (DoG dyn-v3): {label}\n'
                  f'HP at upper-right (filled), LP at the 3 grey rings',
                  fontsize=11, weight='bold', y=0.985)
    fig.subplots_adjust(left=0.05, right=0.91, top=0.96, bottom=0.02,
                          wspace=0.05, hspace=0.05)
    pdf.savefig(fig); plt.close(fig)


def p_get(p, k):
    return p.get(k, np.nan) if p else np.nan


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--af-dir', default=AF_DIR_DEFAULT)
    p.add_argument('--subjects', type=int, nargs='+',
                   default=list(range(1, 31)))
    p.add_argument('--rois', nargs='+', default=ROI_ORDER)
    p.add_argument('--out', type=Path,
                   default=Path('notes/figures/gain_fields.pdf'))
    p.add_argument('--fov', type=float, default=6.0,
                   help='Half-width of visual-field plot (deg)')
    p.add_argument('--resolution', type=int, default=80,
                   help='Pixels per side for the gain-field heatmaps.')
    args = p.parse_args()

    out = args.out
    if not out.is_absolute():
        out = Path('/Users/gdehol/git/retsupp') / out
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f'Output: {out}')

    pars = {}
    for s in tqdm(args.subjects, desc='loading'):
        for roi in args.rois:
            res = read_pars(args.af_dir, s, roi)
            if res is not None:
                pars[(s, roi)] = res
    if not pars:
        raise RuntimeError(
            f'No pars files found under {args.af_dir} for the requested '
            'subjects / ROIs.')
    print(f'Loaded {len(pars)} (subject, ROI) cells')

    with PdfPages(out) as pdf:
        render_page(pdf, pars, 'sustained', args)
        render_page(pdf, pars, 'dynamic',   args)
    print(f'Done: {out}')


if __name__ == '__main__':
    main()
