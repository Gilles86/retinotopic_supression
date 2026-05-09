"""Group BOLD: ROIs × bar directions (physical), traces by HP-condition.

Design: average **physically-identical events** (same bar direction
passing through voxel's RF) across (matched voxel × subject), and
show how the BOLD response is modulated by **cognitive state** (HP
location relative to voxel).

  Rows: ROIs (V1/V2/V3/V3AB/hV4/LO).
  Cols: 4 BAR DIRECTIONS in original frame — bar_right, bar_left,
        bar_up, bar_down. Each col is one physically distinct sweep.
  Traces (colors): 3 HP-conditions relative to voxel:
      red  = HP at voxel's OWN quadrant (close)
      grey = HP at perpendicular quadrant (lateral)
      blue = HP at OPPOSITE quadrant (antipode)

The bar pass is time-locked per voxel to the TR where the bar centre
crosses the voxel's PRF along the sweep axis. Voxels in any of the 4
quadrants contribute to the same cell, with HP-condition labelled
relative to each voxel.

The 4 bar directions DON'T align with HP locations (HPs are on the
diagonals, bars sweep on cardinals) — keeping them in the original
frame avoids fake "toward/away" projections.

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
BAR_VECTORS = {
    'bar_right': (1, 0), 'bar_left': (-1, 0),
    'bar_up': (0, 1), 'bar_down': (0, -1),
}
HP_CONDS = ('close', 'lateral', 'opposite')
HP_REMAP = {'close': 'close', 'orth': 'lateral', 'far': 'opposite'}
HP_COLOR = {'close': '#d62728', 'lateral': '0.5', 'opposite': '#1f77b4'}
HP_LABEL = {'close': 'HP at voxel quadrant',
            'lateral': 'HP perpendicular',
            'opposite': 'HP opposite quadrant'}
DIR_KIND = ('toward', 'away')
DIR_TITLE = {'toward': 'bar continues TOWARD HP',
             'away':   'bar continues AWAY from HP'}


def _roi_voxel_indices(sub, masker, roi):
    img = sub.get_retinotopic_roi(roi=roi, bold_space=True)
    return masker.transform(img).astype(bool).flatten()


def aggregate_subject(sub, masker, opts):
    """For each ROI: per-(bar_dir, hp_cond) mean BOLD over matched voxels."""
    prf_pars_path_x = (sub.bids_folder / 'derivatives' / 'prf'
                       / f'model{opts.prf_model}'
                       / f'sub-{sub.subject_id:02d}'
                       / f'sub-{sub.subject_id:02d}_desc-x.nii.gz')
    if not prf_pars_path_x.exists():
        print(f'  [SKIP] sub-{sub.subject_id:02d}: no PRF NIfTIs')
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
        quadrant_mass=getattr(opts, 'quadrant_mass', 0.5),
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
            print(f'    ROI {roi}: only {len(sub_matched)} matched voxels')
            continue
        # Bin events by (hp_cond, direction-tag).
        # For amplitude collapse: 'all'.
        # For TTP-vs-HP (Story 2 V1): 'toward' / 'away' = bar projection
        #     onto V→HP axis (well-defined for opposite & lateral).
        # For TTP-vs-Vquadrant (Story 3): 'toVq' / 'awayVq' = bar
        #     projection onto V→Q_V-ring-point axis (well-defined for
        #     EVERY voxel because Q_V's ring point is at distance ~4°).
        #     Same physical bar passes get the same label regardless of
        #     HP, so we can contrast HP-conditions in the same panel.
        bins = {(c, d): [] for c in HP_CONDS
                for d in ('all', 'toward', 'away', 'orth',
                          'toVq', 'awayVq')}
        for _, vox in sub_matched.iterrows():
            x_v, y_v, vi = vox['x'], vox['y'], int(vox['vi'])
            # V's quadrant ring point (sign of V_PRF gives the quadrant).
            qv_x = np.sign(x_v) * (4.0 / np.sqrt(2))
            qv_y = np.sign(y_v) * (4.0 / np.sqrt(2))
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
                hp_xy = COND_TO_XY[rm['hp']]
                vec_to_hp = (hp_xy[0] - x_v, hp_xy[1] - y_v)
                # V → V-quadrant-ring-point: this points toward (qv_x, qv_y)
                # FROM V's actual PRF position. For matched voxels (PRF
                # well inside quadrant) this is small, but the SIGN of
                # the projection of bar motion onto Q_V_xy is consistent
                # across voxels in the same quadrant.
                events = find_bar_pass_TRs(sub, rm['session'], rm['run'],
                                             x_v, y_v,
                                             max_dist=opts.max_bar_dist)
                for ev in events:
                    bar_dir = ev['event_type']
                    if bar_dir not in BAR_DIRS:
                        continue
                    bv = BAR_VECTORS[bar_dir]
                    dot_hp = bv[0] * vec_to_hp[0] + bv[1] * vec_to_hp[1]
                    dot_vq = bv[0] * qv_x       + bv[1] * qv_y
                    tr0 = ev['tr_local']
                    t_lo, t_hi = tr0 - half, tr0 + half + 1
                    if t_lo < 0 or t_hi > data.shape[0]:
                        continue
                    epoch = data[t_lo:t_hi, vi]
                    bins[(hp_cond, 'all')].append(epoch)
                    # V→HP direction (Story 2 v1).
                    if abs(dot_hp) < 1e-6:
                        bins[(hp_cond, 'orth')].append(epoch)
                    elif dot_hp > 0:
                        bins[(hp_cond, 'toward')].append(epoch)
                    else:
                        bins[(hp_cond, 'away')].append(epoch)
                    # V→V-quadrant direction (Story 3).
                    if dot_vq > 0:
                        bins[(hp_cond, 'toVq')].append(epoch)
                    elif dot_vq < 0:
                        bins[(hp_cond, 'awayVq')].append(epoch)
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
    """Top-right inset: visual field with bar direction arrow + ring
    positions colored by HP-condition relative to an example UR voxel."""
    iax = ax.inset_axes([0.66, 0.66, 0.32, 0.32])
    iax.set_xlim(-5.2, 5.2); iax.set_ylim(-5.2, 5.2)
    iax.set_aspect('equal')
    iax.set_xticks([]); iax.set_yticks([])
    for spine in iax.spines.values():
        spine.set_linewidth(0.4); spine.set_color('0.5')
    theta = np.linspace(0, 2 * np.pi, 100)
    iax.plot(3.17 * np.cos(theta), 3.17 * np.sin(theta),
             color='0.7', lw=0.4, ls='--')
    # Example voxel = UR quadrant.
    iax.plot(2.0, 2.0, '*', color='k', ms=11, mec='k', mew=0.5, zorder=4)
    HP_REL = {'upper_right': 'close', 'upper_left': 'lateral',
              'lower_right': 'lateral', 'lower_left': 'opposite'}
    for cond_key in CONDITIONS:
        x, y = COND_TO_XY[cond_key]
        c = HP_COLOR[HP_REL[cond_key]]
        iax.plot(x, y, 'o', color=c, ms=8, mec='k', mew=0.4, zorder=3)
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


def _smooth_factory(win, upsample_dt):
    half = win // 2
    t_axis = np.arange(win) - half
    TR = 1.6
    if upsample_dt and upsample_dt < TR:
        t_axis_sec = t_axis * TR
        t_plot = np.arange(t_axis_sec[0], t_axis_sec[-1] + 1e-9, upsample_dt)

        def _smooth(y):
            f = interp1d(t_axis_sec, y, kind='linear',
                         bounds_error=False, fill_value='extrapolate')
            return gaussian_filter1d(
                f(t_plot), sigma=0.25 / upsample_dt, mode='nearest')
        return t_plot / TR, _smooth
    return t_axis, (lambda y: y)


def _plot_trace(ax, t_for_plot, per_sub, color, label, lw=2.0,
                 ls='-', smooth=None):
    """Mean ± SEM trace from per-subject means."""
    if not per_sub:
        return 0
    traces = np.stack([t for _, t, _ in per_sub], axis=0)
    n_sub = traces.shape[0]
    gm = traces.mean(axis=0)
    gs = (traces.std(axis=0, ddof=1) / np.sqrt(n_sub)
          if n_sub > 1 else np.zeros_like(gm))
    if smooth is not None:
        gm = smooth(gm); gs = smooth(gs)
    ax.fill_between(t_for_plot, gm - gs, gm + gs,
                     color=color, alpha=0.18, linewidth=0)
    ax.plot(t_for_plot, gm, color=color, lw=lw, ls=ls, label=label)
    return n_sub


def _draw_amplitude_inset(ax):
    """Inset for amplitude story: voxel ★ + 3 ring positions colour-coded
    by HP-condition relative to that voxel (close=red, lateral=grey,
    opposite=blue)."""
    iax = ax.inset_axes([0.66, 0.62, 0.34, 0.36])
    iax.set_xlim(-5.2, 5.2); iax.set_ylim(-5.2, 5.2)
    iax.set_aspect('equal'); iax.set_xticks([]); iax.set_yticks([])
    for s in iax.spines.values():
        s.set_linewidth(0.4); s.set_color('0.5')
    theta = np.linspace(0, 2 * np.pi, 100)
    iax.plot(3.17 * np.cos(theta), 3.17 * np.sin(theta),
             color='0.7', lw=0.4, ls='--')
    voxel = (2.0, 2.0)
    iax.plot(*voxel, '*', color='k', ms=11, mec='k', mew=0.5, zorder=4)
    HP_REL = {'upper_right': 'close', 'upper_left': 'lateral',
              'lower_right': 'lateral', 'lower_left': 'opposite'}
    for cond_key in CONDITIONS:
        x, y = COND_TO_XY[cond_key]
        iax.plot(x, y, 'o', color=HP_COLOR[HP_REL[cond_key]],
                 ms=9, mec='k', mew=0.4, zorder=3)
    iax.text(0, -4.7, 'voxel ★ in UR;  HP at one of 4 ring positions',
              ha='center', va='top', fontsize=6, color='0.4')


def _draw_ttp_inset(ax, hp_cond_for_inset):
    """Inset for TTP story: voxel ★, HP highlighted at the relevant
    quadrant, and the cardinal bars whose motion projects toward / away
    from HP rendered with arrows. Greyed bars are orthogonal projections
    that aren't used in this panel."""
    iax = ax.inset_axes([0.66, 0.60, 0.34, 0.38])
    iax.set_xlim(-5.4, 5.4); iax.set_ylim(-5.4, 5.4)
    iax.set_aspect('equal'); iax.set_xticks([]); iax.set_yticks([])
    for s in iax.spines.values():
        s.set_linewidth(0.4); s.set_color('0.5')
    theta = np.linspace(0, 2 * np.pi, 100)
    iax.plot(3.17 * np.cos(theta), 3.17 * np.sin(theta),
             color='0.7', lw=0.4, ls='--')
    voxel = (2.0, 2.0)  # example UR voxel
    iax.plot(*voxel, '*', color='k', ms=11, mec='k', mew=0.5, zorder=4)
    # Pick example HP for the panel.
    hp_key = ('lower_left' if hp_cond_for_inset == 'opposite'
              else 'upper_left')   # lateral example
    hp = COND_TO_XY[hp_key]
    iax.plot(*hp, 'o', color=HP_COLOR[hp_cond_for_inset],
             ms=12, mec='k', mew=0.5, zorder=3)
    # Other ring positions, faint grey.
    for cond_key in CONDITIONS:
        if cond_key == hp_key:
            continue
        x, y = COND_TO_XY[cond_key]
        iax.plot(x, y, 'o', color='0.85', ms=7, mec='0.5', mew=0.3, zorder=2)
    # V→HP vector.
    vec = (hp[0] - voxel[0], hp[1] - voxel[1])
    # Show 4 cardinal bars; toward = thick black solid, away = thick
    # black dashed; orth = thin grey.
    for ev_name, bv in BAR_VECTORS.items():
        dot = bv[0] * vec[0] + bv[1] * vec[1]
        if abs(dot) < 1e-6:
            kw = dict(arrowstyle='->', lw=0.7, color='0.7', mutation_scale=8)
        elif dot > 0:
            kw = dict(arrowstyle='->', lw=2.2, color='k', mutation_scale=14,
                       linestyle='-')
        else:
            kw = dict(arrowstyle='->', lw=2.2, color='k', mutation_scale=14,
                       linestyle='--')
        tail = (voxel[0] - 1.4 * bv[0], voxel[1] - 1.4 * bv[1])
        tip  = (voxel[0] + 2.4 * bv[0], voxel[1] + 2.4 * bv[1])
        iax.annotate('', xy=tip, xytext=tail, arrowprops=kw)
    label = ('HP opposite (LL)' if hp_cond_for_inset == 'opposite'
             else 'HP lateral (UL)')
    iax.text(0, -4.9,
              f'★=voxel (UR), ●={label}\n'
              f'solid → toward HP, dashed → away',
              ha='center', va='top', fontsize=6, color='0.4')


def _peak_marker(ax, t_for_plot, per_sub, color, smooth):
    """Mark group-mean peak with a filled dot + drop a vertical tick to
    the x-axis showing time-to-peak."""
    if not per_sub:
        return
    traces = np.stack([t for _, t, _ in per_sub], axis=0)
    gm = traces.mean(axis=0)
    if smooth is not None:
        gm = smooth(gm)
    ipk = int(np.argmax(gm))
    t_pk = float(t_for_plot[ipk]); y_pk = float(gm[ipk])
    ax.plot([t_pk], [y_pk], 'o', color=color, ms=7,
            mec='k', mew=0.6, zorder=6)
    ymin = ax.get_ylim()[0]
    ax.plot([t_pk, t_pk], [ymin, ymin + 0.04 * (y_pk - ymin)],
            color=color, lw=1.5, zorder=5)


def _grid_layout(n):
    if n <= 3: return (1, n)
    if n <= 6: return (2, 3)
    return (3, 3)


def render_amplitude(pdf, group_data, win, opts):
    """Page 1: HP suppression — close vs lateral vs opposite, all bars
    collapsed."""
    t_for_plot, smooth = _smooth_factory(win, opts.plot_upsample_dt)
    rois_with_data = [r for r in opts.rois
                      if r in group_data and group_data[r]]
    if not rois_with_data:
        return
    n = len(rois_with_data)
    nrow, ncol = _grid_layout(n)
    fig, axes = plt.subplots(nrow, ncol,
                              figsize=(4.0 * ncol, 3.2 * nrow),
                              sharex=True, sharey=False, squeeze=False)
    axes_flat = axes.flatten()
    for i, roi in enumerate(rois_with_data):
        ax = axes_flat[i]
        n_subs_seen = {}
        for hp_cond in ('opposite', 'lateral', 'close'):
            per_sub = group_data[roi].get((hp_cond, 'all'), [])
            n_subs_seen[hp_cond] = _plot_trace(
                ax, t_for_plot, per_sub,
                color=HP_COLOR[hp_cond],
                label=(HP_LABEL[hp_cond] if i == 0 else None),
                lw=(2.4 if hp_cond == 'close' else 1.8), ls='-',
                smooth=smooth)
            _peak_marker(ax, t_for_plot, per_sub, HP_COLOR[hp_cond],
                          smooth)
        ax.axvline(0, color='k', lw=0.4, alpha=0.3)
        ax.grid(alpha=0.12)
        ax.set_title(roi, fontsize=12, weight='bold')
        if i == 0:
            _draw_amplitude_inset(ax)
        if i % ncol == 0:
            ax.set_ylabel('BOLD (z, group)', fontsize=10)
        if i // ncol == nrow - 1:
            ax.set_xlabel('TR (bar passes RF)', fontsize=10)
        ax.text(0.02, 0.02,
                 f"n_sub: close={n_subs_seen['close']}  "
                 f"lat={n_subs_seen['lateral']}  "
                 f"opp={n_subs_seen['opposite']}",
                 transform=ax.transAxes, fontsize=7, color='0.45',
                 va='bottom', ha='left')
    for j in range(len(rois_with_data), len(axes_flat)):
        axes_flat[j].axis('off')
    axes_flat[0].legend(loc='lower right', fontsize=8, frameon=True,
                          framealpha=0.85)
    fig.suptitle(
        'Story 1 — Amplitude:  BOLD response to bar passing voxel RF, '
        'split by HP-condition relative to voxel\n'
        '(all 4 bar directions collapsed; matched voxels '
        f'σ ∈ [{opts.sd_min:.1f}, {opts.sd_max:.1f}]°, R²>{opts.r2_min:.2f})',
        fontsize=12, weight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    pdf.savefig(fig); plt.close(fig)


def render_ttp_close_vs_opp(pdf, group_data, win, opts):
    """Story 3: same physical bar passes (toward / away from V's
    quadrant ring point), HP=close vs HP=opposite overlaid in the
    same panel. Predicted TTP shifts have OPPOSITE sign in close vs
    opposite, so the two lines should fan out in opposite directions
    across the toward / away panels."""
    t_for_plot, smooth = _smooth_factory(win, opts.plot_upsample_dt)
    rois_with_data = [r for r in opts.rois
                      if r in group_data and group_data[r]]
    if not rois_with_data:
        return
    n = len(rois_with_data)
    nrow = n
    ncol = 2
    fig, axes = plt.subplots(nrow, ncol,
                              figsize=(5.0 * ncol, 2.4 * nrow),
                              sharex=True, sharey='row', squeeze=False)
    DIR_TITLE = {
        'toVq':   "bar moves TOWARD voxel's quadrant",
        'awayVq': "bar moves AWAY from voxel's quadrant",
    }
    for ri, roi in enumerate(rois_with_data):
        for ci, dirtag in enumerate(('toVq', 'awayVq')):
            ax = axes[ri, ci]
            for hp_cond in ('opposite', 'close'):
                per_sub = group_data[roi].get((hp_cond, dirtag), [])
                _plot_trace(
                    ax, t_for_plot, per_sub,
                    color=HP_COLOR[hp_cond],
                    label=(HP_LABEL[hp_cond] if (ri == 0 and ci == 0) else None),
                    lw=2.2, ls='-', smooth=smooth)
                _peak_marker(ax, t_for_plot, per_sub,
                              HP_COLOR[hp_cond], smooth)
            ax.axvline(0, color='k', lw=0.4, alpha=0.3)
            ax.grid(alpha=0.12)
            if ri == 0:
                ax.set_title(DIR_TITLE[dirtag], fontsize=11, weight='bold')
                _draw_close_vs_opp_inset(ax, dirtag)
            if ri == nrow - 1:
                ax.set_xlabel('TR (bar passes RF)', fontsize=10)
            if ci == 0:
                ax.set_ylabel(f'{roi}\nBOLD (z)', fontsize=10,
                              weight='bold')
            n_close = len(group_data[roi].get(('close', dirtag), []))
            n_opp   = len(group_data[roi].get(('opposite', dirtag), []))
            ax.text(0.02, 0.02,
                     f'n_sub: close={n_close}  opp={n_opp}',
                     transform=ax.transAxes, fontsize=7,
                     color='0.45', va='bottom', ha='left')
    axes[0, 0].legend(loc='lower right', fontsize=8, frameon=True,
                       framealpha=0.85)
    fig.suptitle(
        'Story 3 — Same physical bar passes (relative to V\'s quadrant), '
        'HP=close (red) vs HP=opposite (blue)\n'
        'Prediction: RF shifts away from HP -> toward-Vq panel: close peaks '
        'EARLIER, opposite LATER;  away-Vq panel: order flips.',
        fontsize=12, weight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    pdf.savefig(fig); plt.close(fig)


def _draw_close_vs_opp_inset(ax, dirtag):
    """Two ring positions (V's-quadrant in red, opposite in blue) with
    voxel ★ + the cardinal bars whose motion projects toward (or away)
    from V's quadrant ring point.  Solid arrows = matching cardinals,
    grey = the orthogonal cardinals (also two of them)."""
    iax = ax.inset_axes([0.66, 0.50, 0.34, 0.46])
    iax.set_xlim(-5.4, 5.4); iax.set_ylim(-5.4, 5.4)
    iax.set_aspect('equal'); iax.set_xticks([]); iax.set_yticks([])
    for s in iax.spines.values():
        s.set_linewidth(0.4); s.set_color('0.5')
    theta = np.linspace(0, 2 * np.pi, 100)
    iax.plot(3.17 * np.cos(theta), 3.17 * np.sin(theta),
             color='0.7', lw=0.4, ls='--')
    voxel = (2.0, 2.0)   # example UR voxel
    iax.plot(*voxel, '*', color='k', ms=11, mec='k', mew=0.5, zorder=4)
    qv = (4.0 / np.sqrt(2),  4.0 / np.sqrt(2))   # V's-quadrant ring point
    op = (-qv[0], -qv[1])
    iax.plot(*qv, 'o', color=HP_COLOR['close'], ms=12,
             mec='k', mew=0.5, zorder=3)
    iax.plot(*op, 'o', color=HP_COLOR['opposite'], ms=12,
             mec='k', mew=0.5, zorder=3)
    for cond_key in CONDITIONS:
        x, y = COND_TO_XY[cond_key]
        if (x, y) in (qv, op):
            continue
        iax.plot(x, y, 'o', color='0.85', ms=7, mec='0.5',
                 mew=0.3, zorder=2)
    sign = +1 if dirtag == 'toVq' else -1
    for ev_name, bv in BAR_VECTORS.items():
        dot = bv[0] * qv[0] + bv[1] * qv[1]
        if abs(dot) < 1e-6 or np.sign(dot) != sign:
            kw = dict(arrowstyle='->', lw=0.7, color='0.7',
                       mutation_scale=8)
        else:
            kw = dict(arrowstyle='->', lw=2.4, color='k',
                       mutation_scale=14)
        tail = (voxel[0] - 1.4 * bv[0], voxel[1] - 1.4 * bv[1])
        tip  = (voxel[0] + 2.4 * bv[0], voxel[1] + 2.4 * bv[1])
        iax.annotate('', xy=tip, xytext=tail, arrowprops=kw)
    iax.text(0, -4.9,
              "★=voxel (UR)\n red=V-quadrant HP, blue=opposite HP\n"
              + ("solid arrows toward V-quadrant"
                 if dirtag == 'toVq' else
                 "solid arrows away from V-quadrant"),
              ha='center', va='top', fontsize=6, color='0.4')


def render_ttp(pdf, group_data, win, opts, hp_cond):
    """Page 2/3: TTP story — toward vs away within a single HP cond."""
    t_for_plot, smooth = _smooth_factory(win, opts.plot_upsample_dt)
    rois_with_data = [r for r in opts.rois
                      if r in group_data and group_data[r]]
    if not rois_with_data:
        return
    n = len(rois_with_data)
    nrow, ncol = _grid_layout(n)
    fig, axes = plt.subplots(nrow, ncol,
                              figsize=(4.0 * ncol, 3.2 * nrow),
                              sharex=True, sharey=False, squeeze=False)
    axes_flat = axes.flatten()
    color = HP_COLOR[hp_cond]
    for i, roi in enumerate(rois_with_data):
        ax = axes_flat[i]
        n_t = _plot_trace(
            ax, t_for_plot, group_data[roi].get((hp_cond, 'toward'), []),
            color=color,
            label=('bar TOWARD HP' if i == 0 else None),
            lw=2.4, ls='-', smooth=smooth)
        n_a = _plot_trace(
            ax, t_for_plot, group_data[roi].get((hp_cond, 'away'), []),
            color=color,
            label=('bar AWAY from HP' if i == 0 else None),
            lw=2.4, ls='--', smooth=smooth)
        _peak_marker(ax, t_for_plot,
                      group_data[roi].get((hp_cond, 'toward'), []),
                      color, smooth)
        _peak_marker(ax, t_for_plot,
                      group_data[roi].get((hp_cond, 'away'), []),
                      color, smooth)
        ax.axvline(0, color='k', lw=0.4, alpha=0.3)
        ax.grid(alpha=0.12)
        ax.set_title(roi, fontsize=12, weight='bold')
        if i == 0:
            _draw_ttp_inset(ax, hp_cond)
        if i % ncol == 0:
            ax.set_ylabel('BOLD (z, group)', fontsize=10)
        if i // ncol == nrow - 1:
            ax.set_xlabel('TR (bar passes RF)', fontsize=10)
        ax.text(0.02, 0.02,
                 f'n_sub: toward={n_t}  away={n_a}',
                 transform=ax.transAxes, fontsize=7, color='0.45',
                 va='bottom', ha='left')
    for j in range(len(rois_with_data), len(axes_flat)):
        axes_flat[j].axis('off')
    axes_flat[0].legend(loc='lower right', fontsize=8, frameon=True,
                          framealpha=0.85)
    label = {'opposite': 'OPPOSITE (V→HP diagonal)',
             'lateral':  'LATERAL  (V→HP cardinal)',
             'close':    'CLOSE (HP at voxel — direction undefined!)'}[hp_cond]
    fig.suptitle(
        f'Story 2 — Time-to-peak:  toward vs away from HP, '
        f'within {label}\n'
        f'If RF shifts away from HP -> toward bar peaks earlier than away.',
        fontsize=12, weight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    pdf.savefig(fig); plt.close(fig)


def _draw_toward_away_inset(ax, direction):
    """Inset showing 4 ring positions (HP-coloured for an UR-voxel example)
    + 2 highlighted bar arrows (toward HP or away from HP)."""
    iax = ax.inset_axes([0.66, 0.62, 0.34, 0.36])
    iax.set_xlim(-5.2, 5.2); iax.set_ylim(-5.2, 5.2)
    iax.set_aspect('equal')
    iax.set_xticks([]); iax.set_yticks([])
    for spine in iax.spines.values():
        spine.set_linewidth(0.4); spine.set_color('0.5')
    theta = np.linspace(0, 2 * np.pi, 100)
    iax.plot(3.17 * np.cos(theta), 3.17 * np.sin(theta),
             color='0.7', lw=0.4, ls='--')
    # Example voxel = UR; HP is at UR for this illustration.
    voxel = (2.0, 2.0)
    iax.plot(*voxel, '*', color='k', ms=11, mec='k', mew=0.5, zorder=4)
    HP_REL = {'upper_right': 'close', 'upper_left': 'lateral',
              'lower_right': 'lateral', 'lower_left': 'opposite'}
    for cond_key in CONDITIONS:
        x, y = COND_TO_XY[cond_key]
        c = HP_COLOR[HP_REL[cond_key]]
        iax.plot(x, y, 'o', color=c, ms=7, mec='k', mew=0.4, zorder=3)
    # The HP for this example is at upper_right (red / close).
    hp = COND_TO_XY['upper_right']
    vec_to_hp = (hp[0] - voxel[0], hp[1] - voxel[1])
    # Cardinal bars whose dot product matches the requested direction.
    arr_kw_hi = dict(arrowstyle='->', lw=2.0, color='k', mutation_scale=14)
    arr_kw_lo = dict(arrowstyle='->', lw=0.7, color='0.7', mutation_scale=10)
    sign = 1 if direction == 'toward' else -1
    for ev_name, bv in BAR_VECTORS.items():
        dot = bv[0] * vec_to_hp[0] + bv[1] * vec_to_hp[1]
        is_match = (np.sign(dot) == sign)
        # Tail and tip relative to voxel — show bar centre passing voxel
        # then continuing in motion direction.
        tail = (voxel[0] - 1.4 * bv[0], voxel[1] - 1.4 * bv[1])
        tip = (voxel[0] + 2.4 * bv[0], voxel[1] + 2.4 * bv[1])
        iax.annotate('', xy=tip, xytext=tail,
                      arrowprops=arr_kw_hi if is_match else arr_kw_lo)
    # Caption.
    iax.text(0, -4.7, 'voxel ★ at UR; HP = UR (red)',
              ha='center', va='top', fontsize=6, color='0.4')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bids-folder', default='/data/ds-retsupp')
    p.add_argument('--subjects', type=int, nargs='+',
                   default=list(range(1, 31)))
    p.add_argument('--rois', nargs='+',
                   default=['V1', 'V2', 'V3', 'V3AB', 'hV4', 'VO', 'LO',
                            'TO'])
    p.add_argument('--prf-model', type=int, default=4)
    p.add_argument('--sd-min', type=float, default=0.5)
    p.add_argument('--sd-max', type=float, default=1.5)
    p.add_argument('--r2-min', type=float, default=0.10)
    p.add_argument('--margin-kind', choices=['fwhm', 'sd', 'mass'],
                   default='mass')
    p.add_argument('--quadrant-mass', type=float, default=0.5,
                   help='Mass-criterion threshold (margin-kind=mass): '
                        'voxel kept if Gaussian mass in its own quadrant '
                        '> this. 0.5 = majority mass; 0.7 ≈ old |x|>σ rule.')
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
    group_data = {roi: {} for roi in args.rois}
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
                group_data[roi].setdefault(k, []).append(
                    (s, d['mean'], d['n']))

    if win_global is None:
        raise RuntimeError('No data.')

    with PdfPages(out) as pdf:
        render_amplitude(pdf, group_data, win_global, args)
        render_ttp_close_vs_opp(pdf, group_data, win_global, args)
        render_ttp(pdf, group_data, win_global, args, hp_cond='opposite')
        render_ttp(pdf, group_data, win_global, args, hp_cond='lateral')
        render_ttp(pdf, group_data, win_global, args, hp_cond='close')
    print(f'\nDone: {out}')


if __name__ == '__main__':
    main()
