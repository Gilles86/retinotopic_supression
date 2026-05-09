"""Group-averaged BOLD traces, ROIs × bar directions × HP-condition.

Companion to ``plot_voxel_traces_aggregate.py`` — but with a different
breakdown:

  - **Rows**: ROIs (V1, V2, V3, V3AB, hV4, LO, …).
  - **Columns**: 4 BAR DIRECTIONS in the ORIGINAL (physical) frame —
    bar_right, bar_left, bar_up, bar_down. Each column averages over
    ALL matched voxels regardless of their quadrant — the bar pass
    is physically identical across voxels.
  - **Traces** within each panel:
      red  = HP at SAME quadrant as voxel  (close)
      grey = HP at PERPENDICULAR quadrant   (lateral, 2 of 4 HPs)
      blue = HP at OPPOSITE quadrant        (antipode)

Per (matched voxel × bar pass): time-lock cleaned BOLD ±N TRs around
the TR where the bar centre crosses the voxel's PRF along the sweep
axis. Per-cell: average across (matched voxel × bar pass) within
subject, then group-mean across subjects (with SEM across subjects).

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
from tqdm import tqdm

from retsupp.utils.data import Subject, distractor_locations
from retsupp.visualize.plot_voxel_traces_af_vs_no_af import (
    find_bar_pass_TRs, categorize_runs_by_hp_distance,
    CONDITIONS, COND_TO_XY,
)
from retsupp.visualize.plot_voxel_traces_aggregate import (
    find_matched_voxels,
)


BAR_DIRS = ('bar_right', 'bar_left', 'bar_up', 'bar_down')
BAR_TITLES = {
    'bar_right': 'bar →  (left → right)',
    'bar_left':  'bar ←  (right → left)',
    'bar_up':    'bar ↑  (bottom → top)',
    'bar_down':  'bar ↓  (top → bottom)',
}
# HP condition relative to voxel — categorize_runs_by_hp_distance
# returns 'close' / 'orth' / 'far'; we relabel for display.
HP_COND = ('close', 'lateral', 'opposite')
HP_REMAP = {'close': 'close', 'orth': 'lateral', 'far': 'opposite'}
HP_COLOR = {'close': '#d62728', 'lateral': '0.5', 'opposite': '#1f77b4'}
HP_LABEL = {'close': 'HP at voxel quadrant',
            'lateral': 'HP at perpendicular',
            'opposite': 'HP at opposite quadrant'}


def _roi_voxel_indices(sub: Subject, masker, roi: str) -> np.ndarray:
    """Boolean voxel mask in BOLD-space for the given ROI."""
    img = sub.get_retinotopic_roi(roi=roi, bold_space=True)
    return masker.transform(img).astype(bool).flatten()


def aggregate_subject(sub: Subject, masker, opts):
    """For each ROI in opts.rois, compute per-(bar_dir, hp_cond) traces.

    Returns dict: roi -> {(bar_dir, hp_cond): {'mean', 'n'}}.
    """
    # 1) Mean PRF.
    prf_pars_path_x = (sub.bids_folder / 'derivatives' / 'prf'
                       / f'model{opts.prf_model}'
                       / f'sub-{sub.subject_id:02d}'
                       / f'sub-{sub.subject_id:02d}_desc-x.nii.gz')
    if not prf_pars_path_x.exists():
        print(f'  [SKIP] sub-{sub.subject_id:02d}: no model {opts.prf_model} '
              f'PRF NIfTIs')
        return None
    pars = {}
    for p in ('x', 'y', 'sd', 'r2'):
        pth = prf_pars_path_x.parent / f'sub-{sub.subject_id:02d}_desc-{p}.nii.gz'
        pars[p] = masker.transform(str(pth)).flatten()
    prf_df = pd.DataFrame(pars)

    matched = find_matched_voxels(
        prf_df,
        sd_min=opts.sd_min, sd_max=opts.sd_max,
        r2_min=opts.r2_min,
        margin_kind=opts.margin_kind,
    )
    print(f'  sub-{sub.subject_id:02d}: matched voxels = {len(matched)}')
    if len(matched) < opts.min_voxels:
        return None

    # 2) ROI masks.
    roi_idxs = {}
    for roi in opts.rois:
        try:
            roi_mask = _roi_voxel_indices(sub, masker, roi)
        except Exception as e:
            print(f'    ROI {roi}: not available ({e})')
            continue
        roi_idxs[roi] = roi_mask

    # 3) Per-run BOLD + run meta.
    hp_per_run = sub.get_hpd_locations()
    run_meta = []
    for ses in (1, 2):
        for run in sub.get_runs(ses):
            hp = hp_per_run.get((ses, run))
            if hp not in CONDITIONS:
                continue
            run_meta.append({'session': ses, 'run': run, 'hp': hp})

    bold_chunks = {}
    for rm in run_meta:
        s, r = rm['session'], rm['run']
        fn = (sub.bids_folder / 'derivatives' / 'cleaned'
              / f'sub-{sub.subject_id:02d}'
              / f'ses-{s}' / 'func'
              / f'sub-{sub.subject_id:02d}_ses-{s}_task-search_'
                f'desc-cleaned_run-{r}_bold.nii.gz')
        if not fn.exists():
            continue
        d = masker.transform(fn).astype(np.float32)[:258]
        d = (d - d.mean(axis=0, keepdims=True)) / (
            d.std(axis=0, keepdims=True) + 1e-6)
        bold_chunks[(s, r)] = d

    win = opts.window_TRs
    half = win // 2

    # 4) Per ROI, accumulate epochs.
    roi_results = {}
    for roi, roi_mask in roi_idxs.items():
        # Filter matched voxels to those in this ROI.
        in_roi = matched['vi'].apply(lambda vi: bool(roi_mask[int(vi)]))
        sub_matched = matched[in_roi].copy()
        if len(sub_matched) < opts.min_voxels:
            print(f'    ROI {roi}: only {len(sub_matched)} matched '
                  f'voxels (need ≥{opts.min_voxels})')
            continue
        bins = {(b, c): [] for b in BAR_DIRS for c in HP_COND}
        for _, vox in sub_matched.iterrows():
            x_v, y_v, vi = vox['x'], vox['y'], int(vox['vi'])
            cats, _, _ = categorize_runs_by_hp_distance(run_meta, x_v, y_v)
            for rm, cat in zip(run_meta, cats):
                if cat is None:
                    continue
                hp_cond = HP_REMAP.get(cat)
                if hp_cond is None:
                    continue
                key = (rm['session'], rm['run'])
                if key not in bold_chunks:
                    continue
                data = bold_chunks[key]
                events = find_bar_pass_TRs(sub, rm['session'], rm['run'],
                                             x_v, y_v,
                                             max_dist=opts.max_bar_dist)
                for ev in events:
                    if ev['event_type'] not in BAR_DIRS:
                        continue
                    tr0 = ev['tr_local']
                    t_lo, t_hi = tr0 - half, tr0 + half + 1
                    if t_lo < 0 or t_hi > data.shape[0]:
                        continue
                    bins[(ev['event_type'], hp_cond)].append(
                        data[t_lo:t_hi, vi])
        agg = {}
        for k, ep in bins.items():
            if not ep:
                agg[k] = None
            else:
                arr = np.array(ep)
                agg[k] = {'mean': arr.mean(axis=0), 'n': len(ep)}
        roi_results[roi] = {'agg': agg, 'n_voxels': len(sub_matched)}

    return {'win': win, 'rois': roi_results, 'matched_total': len(matched)}


def _draw_bar_inset(ax, bar_dir):
    """Tiny inset (top-right) showing the bar direction + an example
    voxel-quadrant scenario (UR voxel) with HP-condition coloring on
    the 4 ring positions:
      red  = close (HP at voxel's quadrant — UR here)
      grey = lateral (HP at UL or LR — perpendicular)
      blue = opposite (HP at LL — antipode)
    """
    iax = ax.inset_axes([0.66, 0.66, 0.32, 0.32])
    iax.set_xlim(-5, 5); iax.set_ylim(-5, 5)
    iax.set_aspect('equal')
    iax.set_xticks([]); iax.set_yticks([])
    for spine in iax.spines.values():
        spine.set_linewidth(0.4); spine.set_color('0.5')
    # Aperture.
    theta = np.linspace(0, 2 * np.pi, 100)
    iax.plot(3.17 * np.cos(theta), 3.17 * np.sin(theta),
             color='0.7', lw=0.4, ls='--')
    # Example voxel: middle of UR quadrant (small black star/dot).
    iax.plot(2.0, 2.0, '*', color='k', ms=11, mec='k', mew=0.5, zorder=4)
    # 4 ring positions, colored by HP-condition relative to UR voxel.
    HP_COND_FOR_RING = {  # ring quadrant -> condition (from UR voxel)
        'upper_right': 'close',
        'upper_left':  'lateral',
        'lower_right': 'lateral',
        'lower_left':  'opposite',
    }
    for cond_key in CONDITIONS:
        x, y = COND_TO_XY[cond_key]
        c = HP_COLOR[HP_COND_FOR_RING[cond_key]]
        iax.plot(x, y, 'o', color=c, ms=8, mec='k', mew=0.4, zorder=3)
    # Bar + arrow (bar oriented perpendicular to motion).
    bar_kw = dict(color='k', lw=2.5, solid_capstyle='round')
    arr_kw = dict(arrowstyle='->', lw=1.5, color='k', mutation_scale=14)
    if bar_dir == 'bar_right':
        iax.plot([-2.5, -2.5], [-1.5, 1.5], **bar_kw)
        iax.annotate('', xy=(2.5, 0), xytext=(-2.5, 0), arrowprops=arr_kw)
    elif bar_dir == 'bar_left':
        iax.plot([2.5, 2.5], [-1.5, 1.5], **bar_kw)
        iax.annotate('', xy=(-2.5, 0), xytext=(2.5, 0), arrowprops=arr_kw)
    elif bar_dir == 'bar_up':
        iax.plot([-1.5, 1.5], [-2.5, -2.5], **bar_kw)
        iax.annotate('', xy=(0, 2.5), xytext=(0, -2.5), arrowprops=arr_kw)
    else:  # bar_down
        iax.plot([-1.5, 1.5], [2.5, 2.5], **bar_kw)
        iax.annotate('', xy=(0, -2.5), xytext=(0, 2.5), arrowprops=arr_kw)


def render_group_figure(pdf, group_data, win, opts):
    """Rows = ROIs, cols = 4 bar directions, 3 traces per panel."""
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

    # ROIs that actually have data in at least one cell.
    rois_with_data = [r for r in opts.rois if r in group_data and group_data[r]]
    if not rois_with_data:
        print('  [WARN] no ROI had any data; skipping figure.')
        return
    n_rows = len(rois_with_data)
    n_cols = len(BAR_DIRS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.6 * n_cols, 3.0 * n_rows),
                              sharex=True, sharey='row', squeeze=False)

    for ri, roi in enumerate(rois_with_data):
        for ci, bar_dir in enumerate(BAR_DIRS):
            ax = axes[ri, ci]
            cell = group_data[roi].get((bar_dir, 'close'))  # check any
            for hp_cond in ('opposite', 'lateral', 'close'):
                per_sub = group_data[roi].get((bar_dir, hp_cond), [])
                if not per_sub:
                    continue
                traces = np.stack([t for _, t, _ in per_sub], axis=0)
                n_sub = traces.shape[0]
                gm = traces.mean(axis=0)
                gs = (traces.std(axis=0, ddof=1) / np.sqrt(n_sub)
                      if n_sub > 1 else np.zeros_like(gm))
                gm_p = _smooth(gm)
                gs_p = _smooth(gs)
                color = HP_COLOR[hp_cond]
                ax.fill_between(t_for_plot, gm_p - gs_p, gm_p + gs_p,
                                 color=color, alpha=0.18, linewidth=0)
                lw = 2.4 if hp_cond == 'close' else 1.8
                ax.plot(t_for_plot, gm_p, color=color, lw=lw,
                         label=f"{HP_LABEL[hp_cond]} (n_sub={n_sub})"
                         if (ri == 0 and ci == 0) else None)
            ax.axvline(0, color='k', lw=0.4, alpha=0.3)
            ax.grid(alpha=0.12)
            if ri == 0:
                ax.set_title(BAR_TITLES[bar_dir], fontsize=9)
                _draw_bar_inset(ax, bar_dir)
            if ri == n_rows - 1:
                ax.set_xlabel('TR (relative to bar passes RF)', fontsize=8)
            if ci == 0:
                ax.set_ylabel(f'{roi}\nBOLD (z, group-mean)', fontsize=9,
                              weight='bold')
    # Legend in upper-right of first panel.
    axes[0, 0].legend(loc='lower right', fontsize=7, frameon=True,
                       framealpha=0.85)

    fig.suptitle(
        f'Group-averaged BOLD: ROI × bar direction × HP-condition\n'
        f'matched voxels (σ ∈ [{opts.sd_min:.1f}, {opts.sd_max:.1f}]°, '
        f'R²>{opts.r2_min:.1f}, {opts.margin_kind} fully in quadrant)',
        fontsize=12, weight='bold',
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bids-folder', default='/data/ds-retsupp')
    p.add_argument('--subjects', type=int, nargs='+',
                   default=list(range(1, 31)))
    p.add_argument('--rois', nargs='+',
                   default=['V1', 'V2', 'V3', 'V3AB', 'hV4', 'LO'])
    p.add_argument('--prf-model', type=int, default=4)
    p.add_argument('--sd-min', type=float, default=0.7)
    p.add_argument('--sd-max', type=float, default=1.3)
    p.add_argument('--r2-min', type=float, default=0.4)
    p.add_argument('--margin-kind', choices=['fwhm', 'sd'], default='fwhm')
    p.add_argument('--min-voxels', type=int, default=4)
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
    # group_data[roi][(bar_dir, hp_cond)] = list of (subject, mean_trace, n_events)
    group_data = {roi: {(b, c): []
                         for b in BAR_DIRS for c in HP_COND}
                  for roi in args.rois}
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
        if res is None:
            continue
        win_global = res['win']
        for roi, rd in res['rois'].items():
            for k, d in rd['agg'].items():
                if d is None:
                    continue
                group_data[roi][k].append((s, d['mean'], d['n']))

    if win_global is None:
        raise RuntimeError('No subject yielded any data.')

    with PdfPages(out) as pdf:
        # Cover page.
        fig, ax = plt.subplots(figsize=(11, 7)); ax.axis('off')
        body = (
            'Group-averaged BOLD: ROI × bar direction × HP-condition\n\n'
            f'Subjects: {args.subjects}\n'
            f'ROIs: {args.rois}\n'
            f'σ ∈ [{args.sd_min}, {args.sd_max}]°,  R² > {args.r2_min}\n'
            f'PRF {args.margin_kind.upper()} fully inside one quadrant\n\n'
            'Cells (per ROI × bar direction):\n'
            '  red  — HP at voxel\'s OWN quadrant      (close)\n'
            '  grey — HP at perpendicular quadrant     (lateral, 2 of 4)\n'
            '  blue — HP at OPPOSITE quadrant          (antipode)\n\n'
            'Each trace = group mean ± SEM (across subjects).\n'
            'Bar direction is original (physical) frame; voxels in any\n'
            'quadrant contribute via their own HP-condition labels.\n'
        )
        ax.text(0.5, 0.5, body, ha='center', va='center', fontsize=11,
                family='monospace')
        pdf.savefig(fig); plt.close(fig)

        render_group_figure(pdf, group_data, win_global, args)
    print(f'\nDone: {out}')


if __name__ == '__main__':
    main()
