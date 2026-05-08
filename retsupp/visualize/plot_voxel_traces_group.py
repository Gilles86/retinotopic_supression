"""Group-averaged rotated-frame BOLD traces, 4 panels (one per bar dir).

Companion to ``plot_voxel_traces_aggregate.py``. Loops over subjects,
calls ``aggregate_subject`` to get per-subject (rotated direction →
mean trace) data, then averages mean traces across subjects.

Output figure: 1 row × 4 columns. Each column = one rotated bar
direction (TOWARD HP / AWAY from HP / orth_ccw / orth_cw). Per panel:
group mean ± SEM (across subjects), with an inset showing the
geometry — HP at top in red, lateral rings grey, opposite ring blue,
plus a bar + arrow indicating sweep direction.

Output: ``notes/figures/voxel_traces_group.pdf``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from nilearn import maskers
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d

from retsupp.utils.data import Subject
from retsupp.visualize.plot_voxel_traces_aggregate import (
    aggregate_subject, ROTATED_BINS,
)


# Per-condition title in figure (more informative than the bin name).
COND_TITLE = {
    'toward':   'bar TOWARD HP',
    'away':     'bar AWAY from HP',
    'orth_ccw': 'bar perpendicular (CCW)',
    'orth_cw':  'bar perpendicular (CW)',
}


def _draw_inset(ax, cond):
    """Top-right inset: rotated visual field with bar + arrow."""
    iax = ax.inset_axes([0.65, 0.65, 0.33, 0.33])
    iax.set_xlim(-5.5, 5.5); iax.set_ylim(-5.5, 5.5)
    iax.set_aspect('equal')
    iax.set_xticks([]); iax.set_yticks([])
    for spine in iax.spines.values():
        spine.set_linewidth(0.4); spine.set_color('0.5')

    # Aperture (dashed grey).
    theta = np.linspace(0, 2 * np.pi, 100)
    iax.plot(3.17 * np.cos(theta), 3.17 * np.sin(theta),
             color='0.7', lw=0.5, ls='--')

    # 4 ring positions (rotated frame: HP at top).
    R = 4.0
    iax.plot(0, R, 'o', color='#d62728', ms=11,
              mec='k', mew=0.5, zorder=5)   # HP — red
    iax.text(0.4, R + 0.5, 'HP', fontsize=8, color='#d62728', weight='bold')
    iax.plot(0, -R, 'o', color='#1f77b4', ms=10, mec='k', mew=0.5)  # opposite — blue
    iax.plot(-R, 0, 'o', color='0.5', ms=10, mec='k', mew=0.5)  # lateral — grey
    iax.plot(R, 0, 'o', color='0.5', ms=10, mec='k', mew=0.5)

    # Voxel: small black dot at origin.
    iax.plot(0, 0, '.', color='k', ms=6)

    # Bar + arrow: bar is perpendicular to motion direction.
    # 'toward'   = bar moving +y (north). Bar is horizontal, drawn south,
    #              arrow up.
    # 'away'     = bar moving -y. Bar horizontal, drawn north, arrow down.
    # 'orth_ccw' = bar moving in +x rotated... let's say motion in -x
    #              (left). Bar vertical, drawn east, arrow left.
    # 'orth_cw'  = motion +x. Bar vertical, drawn west, arrow right.
    bar_kw = dict(color='k', lw=3.0, solid_capstyle='round')
    arr_kw = dict(arrowstyle='->', lw=1.6, color='k',
                  mutation_scale=18)
    if cond == 'toward':
        iax.plot([-1.6, 1.6], [-2.0, -2.0], **bar_kw)
        iax.annotate('', xy=(0, 1.5), xytext=(0, -2.0), arrowprops=arr_kw)
    elif cond == 'away':
        iax.plot([-1.6, 1.6], [2.0, 2.0], **bar_kw)
        iax.annotate('', xy=(0, -1.5), xytext=(0, 2.0), arrowprops=arr_kw)
    elif cond == 'orth_ccw':
        iax.plot([2.0, 2.0], [-1.6, 1.6], **bar_kw)
        iax.annotate('', xy=(-1.5, 0), xytext=(2.0, 0), arrowprops=arr_kw)
    elif cond == 'orth_cw':
        iax.plot([-2.0, -2.0], [-1.6, 1.6], **bar_kw)
        iax.annotate('', xy=(1.5, 0), xytext=(-2.0, 0), arrowprops=arr_kw)


def render_group_figure(pdf, group_data, win, opts):
    """One row × 4 columns. Group mean ± SEM across subjects per cond."""
    half = win // 2
    t_axis = np.arange(win) - half
    TR = 1.6
    upsample_dt = opts.plot_upsample_dt
    if upsample_dt and upsample_dt < TR:
        t_axis_sec = t_axis * TR
        t_plot = np.arange(t_axis_sec[0], t_axis_sec[-1] + 1e-9, upsample_dt)

        def _smooth(y):
            f = interp1d(t_axis_sec, y, kind='linear',
                         bounds_error=False, fill_value='extrapolate')
            return gaussian_filter1d(
                f(t_plot), sigma=0.25 / upsample_dt, mode='nearest')
        t_for_plot = t_plot / TR
    else:
        t_for_plot = t_axis
        _smooth = lambda y: y  # noqa: E731

    cond_order = ('toward', 'orth_ccw', 'orth_cw', 'away')
    fig, axes = plt.subplots(1, 4, figsize=(20, 5),
                              sharey=True, sharex=True)

    for col, cond in enumerate(cond_order):
        ax = axes[col]
        per_sub = group_data.get(cond, [])  # list of (subject, mean_trace, n_events)
        if not per_sub:
            ax.text(0.5, 0.5, 'no data', ha='center', va='center',
                     transform=ax.transAxes, fontsize=12, color='0.5')
            ax.set_title(COND_TITLE[cond], fontsize=11)
            continue
        # Stack per-subject means -> (n_sub, win).
        traces = np.stack([t for _, t, _ in per_sub], axis=0)
        n_sub = traces.shape[0]
        group_mean = traces.mean(axis=0)
        group_sem = (traces.std(axis=0, ddof=1) / np.sqrt(n_sub)
                     if n_sub > 1 else np.zeros_like(group_mean))
        n_events_total = sum(n for _, _, n in per_sub)

        gm_p = _smooth(group_mean)
        gs_p = _smooth(group_sem)
        ax.fill_between(t_for_plot, gm_p - gs_p, gm_p + gs_p,
                         color='k', alpha=0.18, linewidth=0)
        ax.plot(t_for_plot, gm_p, color='k', lw=2.6)

        ax.axvline(0, color='k', lw=0.5, alpha=0.3)
        ax.grid(alpha=0.15)
        ax.set_xlabel('TR (relative to bar passes RF)', fontsize=10)
        if col == 0:
            ax.set_ylabel('BOLD (z, group-mean)', fontsize=10)
        ax.set_title(
            f'{COND_TITLE[cond]}\n'
            f'(n_sub={n_sub}, total events={n_events_total})',
            fontsize=11, weight='bold',
        )
        _draw_inset(ax, cond)

    fig.suptitle(
        f'Group-averaged rotated-frame BOLD (matched voxels: σ ∈ '
        f'[{opts.sd_min:.1f}, {opts.sd_max:.1f}]°, '
        f'R²>{opts.r2_min:.1f}, {opts.margin_kind} fully in quadrant)',
        fontsize=12, weight='bold',
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    pdf.savefig(fig)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bids-folder', default='/data/ds-retsupp')
    p.add_argument('--subjects', type=int, nargs='+',
                   default=list(range(1, 31)))
    p.add_argument('--prf-model', type=int, default=4)
    p.add_argument('--sd-min', type=float, default=0.7)
    p.add_argument('--sd-max', type=float, default=1.3)
    p.add_argument('--r2-min', type=float, default=0.4)
    p.add_argument('--margin-kind', choices=['fwhm', 'sd'], default='fwhm')
    p.add_argument('--min-voxels', type=int, default=10)
    p.add_argument('--max-bar-dist', type=float, default=0.5)
    p.add_argument('--window-TRs', type=int, default=21)
    p.add_argument('--plot-upsample-dt', type=float, default=0.1)
    p.add_argument('--out', type=Path,
                   default=Path('notes/figures/voxel_traces_group.pdf'))
    args = p.parse_args()

    out = args.out
    if not out.is_absolute():
        out = Path('/Users/gdehol/git/retsupp') / out
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f'Output: {out}')

    bids = Path(args.bids_folder)
    # Per-subject loop, accumulate traces per rotated-direction.
    group_data = {b: [] for b in ROTATED_BINS}
    win_global = None
    for s in args.subjects:
        try:
            sub = Subject(s, bids)
            masker = maskers.NiftiMasker(mask_img=sub.get_bold_mask())
            masker.fit()
        except Exception as e:
            print(f'sub-{s:02d}: setup failed: {e}'); continue
        print(f'\n=== sub-{s:02d} ===')
        try:
            res = aggregate_subject(sub, masker, args)
        except Exception as e:
            print(f'  [SKIP] aggregate failed: {e}'); continue
        if res is None: continue
        win_global = res['win']
        for cond in ROTATED_BINS:
            d = res['agg'].get(cond)
            if d is None: continue
            group_data[cond].append((s, d['mean'], d['n']))

    if win_global is None:
        raise RuntimeError('No subject yielded any data.')

    n_per_cond = {c: len(group_data[c]) for c in ROTATED_BINS}
    print(f'\nGroup-data n_subjects per condition: {n_per_cond}')

    with PdfPages(out) as pdf:
        # Cover.
        fig, ax = plt.subplots(figsize=(11, 6)); ax.axis('off')
        body = (
            'Group-averaged rotated-frame BOLD\n\n'
            f'Subjects included (have ≥{args.min_voxels} matched voxels): '
            f'{[s for s, _, _ in group_data["toward"]]}\n\n'
            f'σ ∈ [{args.sd_min}, {args.sd_max}]°,  R² > {args.r2_min}\n'
            f'PRF {args.margin_kind.upper()} fully inside one quadrant\n\n'
            'Per (matched voxel × bar pass): rotate bar direction into\n'
            'the voxel→HP frame (HP at +y) and bin into one of 4 conds.\n\n'
            '4-column figure: each col is one rotated bar direction.\n'
            'Each panel = group mean across subjects ± SEM(across subjects).\n'
        )
        ax.text(0.5, 0.5, body, ha='center', va='center', fontsize=11,
                family='monospace')
        pdf.savefig(fig); plt.close(fig)

        render_group_figure(pdf, group_data, win_global, args)
    print(f'\nDone: {out}')


if __name__ == '__main__':
    main()
