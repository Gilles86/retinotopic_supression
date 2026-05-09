"""Group-averaged BOLD: ROIs × ROTATED bar directions (HP-at-top frame).

Each (matched voxel × bar pass × run) is rotated into a frame where the
voxel→HP vector points to +y. The bar's direction-of-motion in that
rotated frame falls in one of 4 quadrants:

  - 'toward'   = bar moves +y after passing voxel  (heads TOWARD HP)
  - 'away'     = bar moves -y                       (heads AWAY from HP)
  - 'orth_ccw' = bar moves -x in rotated frame      (perpendicular, CCW)
  - 'orth_cw'  = bar moves +x in rotated frame      (perpendicular, CW)

This makes UR/UL/LL/LR voxels with their respective HPs all comparable —
"bar TOWARD HP" means the same physical relationship for every voxel.

Layout: rows = ROIs, cols = 4 rotated directions, 1 trace per panel
(group mean across subjects ± SEM). Insets show the rotated frame:
HP red at top, opposite blue at bottom, lateral grey on sides; bar
+ arrow.

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
    find_bar_pass_TRs, CONDITIONS, COND_TO_XY,
)
from retsupp.visualize.plot_voxel_traces_aggregate import (
    find_matched_voxels, rotated_sweep_bin,
)


BAR_VECTORS = {
    'bar_right': (1, 0),
    'bar_left': (-1, 0),
    'bar_up': (0, 1),
    'bar_down': (0, -1),
}
ROTATED_BINS = ('toward', 'orth_ccw', 'orth_cw', 'away')
ROTATED_TITLES = {
    'toward':   'bar TOWARD HP',
    'away':     'bar AWAY from HP',
    'orth_ccw': 'bar orth (CCW)',
    'orth_cw':  'bar orth (CW)',
}


def _roi_voxel_indices(sub, masker, roi):
    img = sub.get_retinotopic_roi(roi=roi, bold_space=True)
    return masker.transform(img).astype(bool).flatten()


def aggregate_subject(sub, masker, opts):
    """For each ROI in opts.rois, compute per-rotated-direction traces."""
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

    roi_idxs = {}
    for roi in opts.rois:
        try:
            roi_idxs[roi] = _roi_voxel_indices(sub, masker, roi)
        except Exception as e:
            print(f'    ROI {roi}: not available ({e})')

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
    roi_results = {}

    for roi, roi_mask in roi_idxs.items():
        in_roi = matched['vi'].apply(lambda vi: bool(roi_mask[int(vi)]))
        sub_matched = matched[in_roi].copy()
        if len(sub_matched) < opts.min_voxels:
            print(f'    ROI {roi}: only {len(sub_matched)} matched '
                  f'voxels (need ≥{opts.min_voxels})')
            continue
        bins = {b: [] for b in ROTATED_BINS}
        for _, vox in sub_matched.iterrows():
            x_v, y_v, vi = vox['x'], vox['y'], int(vox['vi'])
            for rm in run_meta:
                key = (rm['session'], rm['run'])
                if key not in bold_chunks:
                    continue
                hp_xy = COND_TO_XY[rm['hp']]
                data = bold_chunks[key]
                events = find_bar_pass_TRs(sub, rm['session'], rm['run'],
                                             x_v, y_v,
                                             max_dist=opts.max_bar_dist)
                for ev in events:
                    bar_vec = BAR_VECTORS.get(ev['event_type'])
                    if bar_vec is None:
                        continue
                    rb = rotated_sweep_bin(bar_vec, (x_v, y_v), hp_xy)
                    if rb is None:
                        continue
                    tr0 = ev['tr_local']
                    t_lo, t_hi = tr0 - half, tr0 + half + 1
                    if t_lo < 0 or t_hi > data.shape[0]:
                        continue
                    bins[rb].append(data[t_lo:t_hi, vi])
        agg = {}
        for k, ep in bins.items():
            if not ep:
                agg[k] = None
            else:
                arr = np.array(ep)
                agg[k] = {'mean': arr.mean(axis=0), 'n': len(ep)}
        roi_results[roi] = {'agg': agg, 'n_voxels': len(sub_matched)}

    return {'win': win, 'rois': roi_results, 'matched_total': len(matched)}


def _draw_rotated_inset(ax, cond):
    """Top-right inset: rotated frame (HP at top), bar+arrow indicating
    the rotated direction."""
    iax = ax.inset_axes([0.66, 0.66, 0.32, 0.32])
    iax.set_xlim(-5.2, 5.2); iax.set_ylim(-5.2, 5.2)
    iax.set_aspect('equal')
    iax.set_xticks([]); iax.set_yticks([])
    for spine in iax.spines.values():
        spine.set_linewidth(0.4); spine.set_color('0.5')
    theta = np.linspace(0, 2 * np.pi, 100)
    iax.plot(3.17 * np.cos(theta), 3.17 * np.sin(theta),
             color='0.7', lw=0.4, ls='--')
    R = 4.0
    iax.plot(0,  R, 'o', color='#d62728', ms=10, mec='k', mew=0.4)  # HP
    iax.plot(0, -R, 'o', color='#1f77b4', ms=9, mec='k', mew=0.4)   # opposite
    iax.plot(-R, 0, 'o', color='0.5', ms=9, mec='k', mew=0.4)
    iax.plot( R, 0, 'o', color='0.5', ms=9, mec='k', mew=0.4)
    iax.text(0.4, R + 0.5, 'HP', fontsize=7, color='#d62728', weight='bold')
    iax.plot(0, 0, '*', color='k', ms=11, mec='k', mew=0.5)  # voxel @ origin
    bar_kw = dict(color='k', lw=2.5, solid_capstyle='round')
    arr_kw = dict(arrowstyle='->', lw=1.5, color='k', mutation_scale=14)
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

    rois_with_data = [r for r in opts.rois if r in group_data and group_data[r]]
    if not rois_with_data:
        return
    n_rows = len(rois_with_data)
    n_cols = len(ROTATED_BINS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.6 * n_cols, 3.0 * n_rows),
                              sharex=True, sharey='row', squeeze=False)

    # Color: emphasize toward (red) and away (blue); orths greyish.
    cond_color = {'toward': '#d62728', 'away': '#1f77b4',
                  'orth_ccw': '0.45', 'orth_cw': '0.65'}

    for ri, roi in enumerate(rois_with_data):
        for ci, cond in enumerate(ROTATED_BINS):
            ax = axes[ri, ci]
            per_sub = group_data[roi].get(cond, [])
            if not per_sub:
                ax.text(0.5, 0.5, 'no data', ha='center', va='center',
                         transform=ax.transAxes, fontsize=10, color='0.5')
            else:
                traces = np.stack([t for _, t, _ in per_sub], axis=0)
                n_sub = traces.shape[0]
                gm = traces.mean(axis=0)
                gs = (traces.std(axis=0, ddof=1) / np.sqrt(n_sub)
                      if n_sub > 1 else np.zeros_like(gm))
                gm_p = _smooth(gm)
                gs_p = _smooth(gs)
                color = cond_color[cond]
                ax.fill_between(t_for_plot, gm_p - gs_p, gm_p + gs_p,
                                 color=color, alpha=0.22, linewidth=0)
                ax.plot(t_for_plot, gm_p, color=color, lw=2.6,
                         label=f"n_sub={n_sub}")
                n_events_total = sum(n for _, _, n in per_sub)
                ax.text(0.97, 0.96, f'n_sub={n_sub}\nn_events={n_events_total}',
                         transform=ax.transAxes, fontsize=7,
                         ha='right', va='top',
                         bbox=dict(boxstyle='round,pad=0.2',
                                   facecolor='white', alpha=0.7,
                                   edgecolor='none'))
            ax.axvline(0, color='k', lw=0.4, alpha=0.3)
            ax.grid(alpha=0.12)
            if ri == 0:
                ax.set_title(ROTATED_TITLES[cond], fontsize=10)
                _draw_rotated_inset(ax, cond)
            if ri == n_rows - 1:
                ax.set_xlabel('TR (relative to bar passes RF)', fontsize=8)
            if ci == 0:
                ax.set_ylabel(f'{roi}\nBOLD (z, group)', fontsize=10,
                              weight='bold')

    fig.suptitle(
        f'Group-averaged BOLD: ROI × ROTATED bar direction (HP at +y)\n'
        f'matched voxels (σ ∈ [{opts.sd_min:.1f}, {opts.sd_max:.1f}]°, '
        f'R²>{opts.r2_min:.2f}, {opts.margin_kind} fully in quadrant)',
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
    p.add_argument('--sd-min', type=float, default=0.5)
    p.add_argument('--sd-max', type=float, default=1.5)
    p.add_argument('--r2-min', type=float, default=0.3)
    p.add_argument('--margin-kind', choices=['fwhm', 'sd'], default='sd')
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
    group_data = {roi: {b: [] for b in ROTATED_BINS} for roi in args.rois}
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
                if d is None: continue
                group_data[roi][k].append((s, d['mean'], d['n']))

    if win_global is None:
        raise RuntimeError('No subject yielded any data.')

    with PdfPages(out) as pdf:
        fig, ax = plt.subplots(figsize=(11, 7)); ax.axis('off')
        body = (
            'Group-averaged BOLD: ROIs × ROTATED bar directions\n\n'
            f'Subjects: {args.subjects}\n'
            f'ROIs: {args.rois}\n'
            f'σ ∈ [{args.sd_min}, {args.sd_max}]°,  R² > {args.r2_min}\n'
            f'PRF {args.margin_kind.upper()} fully inside one quadrant\n\n'
            'Rotation: per (voxel × run), rotate so voxel→HP at +y.\n'
            'Bar direction-of-motion in rotated frame -> 4 conditions:\n'
            '  TOWARD HP  (bar moves +y)\n'
            '  AWAY HP    (bar moves -y)\n'
            '  orth CCW / orth CW (perpendicular)\n\n'
            'All voxel quadrants × HP positions are normalized to one\n'
            'frame -> apples-to-apples comparison.\n'
        )
        ax.text(0.5, 0.5, body, ha='center', va='center', fontsize=11,
                family='monospace')
        pdf.savefig(fig); plt.close(fig)

        render_group_figure(pdf, group_data, win_global, args)
    print(f'\nDone: {out}')


if __name__ == '__main__':
    main()
