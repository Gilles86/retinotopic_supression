"""Aggregate per-subject BOLD traces over MATCHED voxels, by sweep × HP.

For each subject:

1. Pick voxels with similar PRF properties:
   - σ in a narrow band (default 0.7–1.3°)
   - eccentricity in [1.5, 2.8]° (mid-band, not foveal, not aperture edge)
   - R² > threshold
   - PRF center within 30° of one of the 4 quadrant diagonals
     (so the voxel is unambiguously in one quadrant — UR/UL/LL/LR)

2. For each voxel:
   - Determine its quadrant (UR/UL/LL/LR) from sign(x), sign(y)
   - Find bar-pass-near-RF events for ALL 4 sweep directions
     (bar_right / bar_left / bar_up / bar_down)
   - For each run, label the HP-condition relative to that voxel:
     close (HP same quadrant), far (opposite), orth (perpendicular)

3. Aggregate epochs across matched voxels:
   - Keep separate by (sweep direction × HP condition)
   - Average BOLD across all (voxel, sweep) events per cell
   - SEM across the same events

4. Render one page per subject:
   - 2×2 panel grid (one panel per sweep direction)
   - Each panel has 3 traces: HP-close / orth / far
   - Trace = mean ± SEM across hundreds of events

Output: ``notes/figures/voxel_traces_aggregate.pdf``.
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
    CONDITIONS, RING_KEYS, COND_TO_XY,
)


SWEEP_DIRS = ('bar_right', 'bar_left', 'bar_up', 'bar_down')
BAR_VECTORS = {  # bar motion direction (dx, dy) in original screen coords
    'bar_right': (1, 0),
    'bar_left': (-1, 0),
    'bar_up': (0, 1),
    'bar_down': (0, -1),
}
ROTATED_BINS = ('toward', 'away', 'orth_ccw', 'orth_cw')
ROTATED_COLORS = {
    'toward': '#d62728',     # red — bar passes voxel and continues toward HP
    'away':   '#1f77b4',     # blue — bar passes voxel away from HP
    'orth_ccw': '#7f7f7f',   # grey — perpendicular CCW
    'orth_cw':  '#999999',   # grey-light — perpendicular CW
}
ROTATED_LABELS = {
    'toward': 'TOWARD HP',
    'away':   'AWAY from HP',
    'orth_ccw': 'orth (CCW)',
    'orth_cw':  'orth (CW)',
}


def voxel_quadrant(x: float, y: float):
    """Return one of UR/UL/LL/LR based on sign(x), sign(y)."""
    if x > 0 and y > 0:
        return 'upper_right'
    if x < 0 and y > 0:
        return 'upper_left'
    if x < 0 and y < 0:
        return 'lower_left'
    if x > 0 and y < 0:
        return 'lower_right'
    return None


def find_matched_voxels(prf_pars: pd.DataFrame, *,
                         sd_min: float, sd_max: float,
                         r2_min: float,
                         margin_kind: str = 'fwhm'):
    """Voxels with PRF FWHM (or 1σ) fully inside one quadrant.

    margin_kind:
       'fwhm' — strict: |x| > half_FWHM AND |y| > half_FWHM (where
                half_FWHM = sd × √(2 ln 2) ≈ 1.177 × sd).
       'sd'   — looser: |x| > sd AND |y| > sd.

    Eccentricity is implicit through the requirement that the PRF lie
    inside the bar aperture (|x| < 3.17 - half_FWHM) so the bar
    actually sweeps through it.
    """
    x = prf_pars['x'].values
    y = prf_pars['y'].values
    sd = prf_pars['sd'].values
    r2 = prf_pars['r2'].values
    half_fwhm = sd * np.sqrt(2.0 * np.log(2.0))
    margin = half_fwhm if margin_kind == 'fwhm' else sd
    sd_pass = (sd >= sd_min) & (sd <= sd_max)
    in_quadrant = (np.abs(x) > margin) & (np.abs(y) > margin)
    # Aperture ceiling: PRF (centre + margin) must fit inside ~3.17°.
    inside_aperture = (np.abs(x) + margin < 3.17 - 0.1) & \
                      (np.abs(y) + margin < 3.17 - 0.1)
    r2_pass = r2 >= r2_min
    valid_idx = np.where(sd_pass & in_quadrant & inside_aperture & r2_pass)[0]
    rows = []
    for vi in valid_idx:
        q = voxel_quadrant(x[vi], y[vi])
        if q is None:
            continue
        rows.append({'vi': int(vi), 'x': float(x[vi]), 'y': float(y[vi]),
                     'sd': float(sd[vi]), 'r2': float(r2[vi]),
                     'half_fwhm': float(half_fwhm[vi]),
                     'ecc': float(np.hypot(x[vi], y[vi])),
                     'quadrant': q})
    return pd.DataFrame(rows)


def rotated_sweep_bin(bar_dir_xy, voxel_xy, hp_xy):
    """In a frame where HP is at +y of the voxel, return one of:
        'toward' / 'away' / 'orth_left' / 'orth_right'
    based on the rotated bar-direction-of-motion.

    bar_dir_xy: (dx, dy) unit vector of bar motion (e.g. (1, 0) for bar_right)
    voxel_xy: (x, y) PRF centre
    hp_xy:    (x, y) HP-distractor location
    """
    dx, dy = bar_dir_xy
    vec = np.asarray(hp_xy) - np.asarray(voxel_xy)
    n = np.linalg.norm(vec)
    if n < 1e-9:
        return None
    ux, uy = vec / n   # unit vector pointing voxel→HP
    # Rotate (dx, dy) by -theta where theta = angle of (ux, uy).
    # In rotated frame, (ux, uy) points to +y.
    # Equivalent: project bar dir onto the (perp, hp) axes.
    par = dx * ux + dy * uy           # along voxel→HP axis  (+ = toward HP)
    perp = dx * (-uy) + dy * ux       # perpendicular (CCW = +)
    # Bin by which component dominates.
    if abs(par) >= abs(perp):
        return 'toward' if par > 0 else 'away'
    return 'orth_ccw' if perp > 0 else 'orth_cw'


def aggregate_subject(sub: Subject, masker, opts):
    """Returns per-(sweep, condition) aggregated traces, or None if subject
    has no matched voxels / no cleaned BOLD locally."""
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
    if len(matched) < opts.min_voxels:
        print(f'  [SKIP] sub-{sub.subject_id:02d}: only {len(matched)} matched '
              f'voxels (need ≥{opts.min_voxels})')
        return None
    print(f'  sub-{sub.subject_id:02d}: {len(matched)} matched voxels '
          f'({matched["quadrant"].value_counts().to_dict()})')

    # 2) HP per run.
    hp_per_run = sub.get_hpd_locations()

    # Build run_meta in the same shape categorize_runs_by_hp_distance expects.
    run_meta = []
    for ses in (1, 2):
        for run in sub.get_runs(ses):
            hp = hp_per_run.get((ses, run))
            if hp not in CONDITIONS:
                continue
            run_meta.append({'session': ses, 'run': run, 'hp': hp})

    # 3) Per voxel, collect epochs by (sweep direction, condition).
    #    Pre-load BOLD per run (mask once, reuse across voxels).
    bold_chunks = {}
    for rm in run_meta:
        s, r = rm['session'], rm['run']
        fn = (sub.bids_folder / 'derivatives' / 'cleaned'
              / f'sub-{sub.subject_id:02d}'
              / f'ses-{s}' / 'func'
              / f'sub-{sub.subject_id:02d}_ses-{s}_task-search_'
                f'desc-cleaned_run-{r}_bold.nii.gz')
        if not fn.exists():
            print(f'    [WARN] missing BOLD ses-{s} run-{r}; skipping')
            continue
        d = masker.transform(fn).astype(np.float32)
        d = d[:258]
        # z-score across time per voxel for cleaner SEM aggregation.
        d = (d - d.mean(axis=0, keepdims=True)) / (
            d.std(axis=0, keepdims=True) + 1e-6)
        bold_chunks[(s, r)] = d

    win = opts.window_TRs
    half = win // 2
    bins = {b: [] for b in ROTATED_BINS}

    for _, vox in tqdm(matched.iterrows(), total=len(matched),
                       desc=f'sub-{sub.subject_id:02d} matched voxels',
                       leave=False):
        x_v, y_v, vi = vox['x'], vox['y'], vox['vi']
        for rm in run_meta:
            hp_xy = COND_TO_XY[rm['hp']]
            key_b = (rm['session'], rm['run'])
            if key_b not in bold_chunks:
                continue
            data = bold_chunks[key_b]
            events = find_bar_pass_TRs(sub, rm['session'], rm['run'],
                                         x_v, y_v, max_dist=opts.max_bar_dist)
            for ev in events:
                # Rotate the bar's direction-of-motion into the
                # voxel→HP frame (HP at +y) and bin.
                bar_dir = BAR_VECTORS[ev['event_type']]
                rb = rotated_sweep_bin(bar_dir, (x_v, y_v), hp_xy)
                if rb is None:
                    continue
                tr0 = ev['tr_local']
                t_lo, t_hi = tr0 - half, tr0 + half + 1
                if t_lo < 0 or t_hi > data.shape[0]:
                    continue
                seg = data[t_lo:t_hi, vi]
                bins[rb].append(seg)

    # 4) Aggregate.
    agg = {}
    for key, ep_list in bins.items():
        if not ep_list:
            agg[key] = None
            continue
        arr = np.array(ep_list)  # (n_events, win)
        agg[key] = {
            'mean': arr.mean(axis=0),
            'sem': arr.std(axis=0, ddof=1) / np.sqrt(len(ep_list))
                    if len(ep_list) > 1 else np.zeros(arr.shape[1]),
            'n': len(ep_list),
        }
    return {
        'matched': matched,
        'agg': agg,
        'win': win,
    }


def render_subject_page(pdf, sub: Subject, sub_result, opts):
    win = sub_result['win']
    half = win // 2
    t_axis = np.arange(win) - half  # in TRs

    # Smoothing: linear interp + tiny Gaussian (matches the per-voxel script).
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
        t_plot_TR = t_plot / TR
    else:
        t_plot_TR = t_axis
        _smooth = lambda y: y  # noqa: E731

    fig, ax = plt.subplots(figsize=(9, 6))
    matched = sub_result['matched']
    agg = sub_result['agg']
    # Plot order: orth first (background), then away, then toward (front).
    for cond in ('orth_cw', 'orth_ccw', 'away', 'toward'):
        d = agg.get(cond)
        if d is None:
            continue
        mean_p = _smooth(d['mean'])
        sem_p = _smooth(d['sem'])
        color = ROTATED_COLORS[cond]
        is_main = cond in ('toward', 'away')
        lw = 2.6 if is_main else 1.8
        alpha_fill = 0.22 if is_main else 0.12
        ax.fill_between(t_plot_TR, mean_p - sem_p, mean_p + sem_p,
                         color=color, alpha=alpha_fill, linewidth=0)
        ax.plot(t_plot_TR, mean_p, color=color, lw=lw,
                 label=f"{ROTATED_LABELS[cond]} n={d['n']}")
    ax.axvline(0, color='k', lw=0.5, alpha=0.3)
    ax.grid(alpha=0.15)
    ax.set_xlabel('TR (relative to bar passes RF center)', fontsize=9)
    ax.set_ylabel('BOLD (z, voxel-mean)', fontsize=9)
    ax.legend(loc='upper right', fontsize=9, frameon=True, framealpha=0.85)

    quadrants_str = matched['quadrant'].value_counts().to_dict()
    fig.suptitle(
        f'sub-{sub.subject_id:02d}  ROTATED-frame BOLD (HP at top of voxel)\n'
        f'σ ∈ [{opts.sd_min:.1f}, {opts.sd_max:.1f}]°, '
        f'R² > {opts.r2_min:.1f}, margin={opts.margin_kind} '
        f'(PRF fully in quadrant),  {len(matched)} voxels  '
        f'(quadrants: {quadrants_str})',
        fontsize=10, weight='bold',
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    pdf.savefig(fig)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bids-folder', default='/data/ds-retsupp')
    p.add_argument('--subjects', type=int, nargs='+',
                   default=list(range(1, 31)))
    p.add_argument('--prf-model', type=int, default=4,
                   help='Mean-fit PRF model to read (default 4).')
    p.add_argument('--sd-min', type=float, default=0.7)
    p.add_argument('--sd-max', type=float, default=1.3)
    p.add_argument('--r2-min', type=float, default=0.4)
    p.add_argument('--margin-kind', choices=['fwhm', 'sd'], default='fwhm',
                   help="'fwhm' = strict (PRF half-FWHM fully in quadrant); "
                        "'sd' = looser (1σ fully in quadrant).")
    p.add_argument('--min-voxels', type=int, default=10,
                   help='Skip subject if fewer than this many matched '
                        'voxels.')
    p.add_argument('--max-bar-dist', type=float, default=0.5)
    p.add_argument('--window-TRs', type=int, default=21,
                   help='Epoch window length in TRs.')
    p.add_argument('--plot-upsample-dt', type=float, default=0.1)
    p.add_argument('--out', type=Path,
                   default=Path('notes/figures/voxel_traces_aggregate.pdf'))
    args = p.parse_args()

    out = args.out
    if not out.is_absolute():
        out = Path('/Users/gdehol/git/retsupp') / out
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f'Output: {out}')
    print(f'Subjects: {args.subjects}')
    bids = Path(args.bids_folder)

    with PdfPages(out) as pdf:
        # Cover.
        fig, ax = plt.subplots(figsize=(11, 8.5)); ax.axis('off')
        body = (
            'Aggregate per-subject BOLD traces (matched voxels)\n\n'
            f'PRF model for matching: model {args.prf_model}\n'
            f'σ ∈ [{args.sd_min}, {args.sd_max}]°,  '
            f'R² > {args.r2_min}\n'
            f'PRF {args.margin_kind.upper()} fully inside one quadrant\n\n'
            'Voxel quadrant from sign(x), sign(y).  Each run\'s HP\n'
            'condition relative to that voxel: close/orth/far.\n\n'
            '2×2 panel grid per subject — one panel per bar sweep direction.\n'
            'Each panel: 3 traces (close/orth/far) ± SEM across all\n'
            '(matched voxel × bar pass) events in that condition.\n\n'
            f'Subjects: {args.subjects}\n'
        )
        ax.text(0.5, 0.5, body, ha='center', va='center', fontsize=11,
                family='monospace')
        pdf.savefig(fig); plt.close(fig)

        for s in args.subjects:
            try:
                sub = Subject(s, bids)
                bold_mask = sub.get_bold_mask()
                masker = maskers.NiftiMasker(mask_img=bold_mask)
                masker.fit()
            except Exception as e:
                print(f'sub-{s:02d}: setup failed: {e}')
                continue
            print(f'\n=== sub-{s:02d} ===')
            try:
                res = aggregate_subject(sub, masker, args)
            except Exception as e:
                print(f'  [SKIP] aggregate failed: {e}')
                continue
            if res is None:
                continue
            render_subject_page(pdf, sub, res, args)
    print(f'\nDone: {out}')


if __name__ == '__main__':
    main()
