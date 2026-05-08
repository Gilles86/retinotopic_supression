"""Aggregate per-subject BOLD traces locked to TARGET/DISTRACTOR onsets.

Companion to ``plot_voxel_traces_aggregate.py``. Same matched-voxel
filter (FWHM fully inside one quadrant); same per-(voxel × run × event)
aggregation. Different events:

  - 'target_at_voxel':       target appears at the voxel's quadrant
  - 'distractor_at_voxel':   distractor appears at the voxel's quadrant
  - 'singleton_elsewhere':   target+distractor at OTHER ring positions
                              (control — voxel sees neither at its
                              quadrant on this trial)

For each event, time-lock cleaned BOLD ±10 TRs around the trial onset.
Average across (matched voxel × trial) per condition.

Output: ``notes/figures/voxel_traces_events.pdf`` (one page per subject).
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

from retsupp.utils.data import Subject
from retsupp.visualize.plot_voxel_traces_aggregate import (
    find_matched_voxels, voxel_quadrant,
)


# Trial event-types we lock to.
EVENT_BINS = ('target_at_voxel', 'distractor_at_voxel', 'singleton_elsewhere')
EVENT_COLORS = {
    'target_at_voxel':     '#2ca02c',  # green — target onset at voxel
    'distractor_at_voxel': '#d62728',  # red — distractor onset at voxel
    'singleton_elsewhere': '#7f7f7f',  # grey — control
}
EVENT_LABELS = {
    'target_at_voxel':     'TARGET at voxel-quadrant',
    'distractor_at_voxel': 'DISTRACTOR at voxel-quadrant',
    'singleton_elsewhere': 'singleton elsewhere (control)',
}

# Map ring location code (1, 3, 5, 7) -> quadrant string.
LOC_TO_QUADRANT = {
    1.0: 'upper_right',
    3.0: 'upper_left',
    5.0: 'lower_left',
    7.0: 'lower_right',
}


def aggregate_subject(sub: Subject, masker, opts):
    """Per-(matched voxel × trial), classify event type and accumulate
    BOLD epochs locked to trial onset."""
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

    win = opts.window_TRs
    half = win // 2
    bins = {b: [] for b in EVENT_BINS}

    # Pre-load per-run BOLD + per-trial onsets.
    for ses in (1, 2):
        for run in sub.get_runs(ses):
            bold_fn = (sub.bids_folder / 'derivatives' / 'cleaned'
                       / f'sub-{sub.subject_id:02d}'
                       / f'ses-{ses}' / 'func'
                       / f'sub-{sub.subject_id:02d}_ses-{ses}_task-search_'
                         f'desc-cleaned_run-{run}_bold.nii.gz')
            if not bold_fn.exists():
                continue
            data = masker.transform(bold_fn).astype(np.float32)
            data = data[:258]
            data = (data - data.mean(axis=0, keepdims=True)) / (
                data.std(axis=0, keepdims=True) + 1e-6)
            tr = sub.get_tr(ses, run)

            ons = sub.get_onsets(ses, run)
            tgt = ons[ons['event_type'] == 'target'].sort_values('onset')
            for _, trial in tgt.iterrows():
                target_loc = trial.get('target_location', np.nan)
                dist_loc = trial.get('distractor_location', np.nan)
                target_quad = LOC_TO_QUADRANT.get(float(target_loc), None)
                dist_quad = LOC_TO_QUADRANT.get(float(dist_loc), None)

                t_on = trial['onset']
                tr0 = int(round(t_on / tr))
                t_lo, t_hi = tr0 - half, tr0 + half + 1
                if t_lo < 0 or t_hi > data.shape[0]:
                    continue

                for _, vox in matched.iterrows():
                    vq = vox['quadrant']
                    vi = int(vox['vi'])
                    if target_quad == vq:
                        bins['target_at_voxel'].append(data[t_lo:t_hi, vi])
                    elif dist_quad == vq:
                        bins['distractor_at_voxel'].append(data[t_lo:t_hi, vi])
                    elif (target_quad is not None
                          and dist_quad is not None
                          and target_quad != vq and dist_quad != vq):
                        # Control: singleton appeared at other rings.
                        bins['singleton_elsewhere'].append(data[t_lo:t_hi, vi])

    agg = {}
    for key, ep_list in bins.items():
        if not ep_list:
            agg[key] = None; continue
        arr = np.array(ep_list)
        agg[key] = {
            'mean': arr.mean(axis=0),
            'sem': arr.std(axis=0, ddof=1) / np.sqrt(len(ep_list))
                    if len(ep_list) > 1 else np.zeros(arr.shape[1]),
            'n': len(ep_list),
        }
    return {'matched': matched, 'agg': agg, 'win': win}


def render_subject_page(pdf, sub: Subject, sub_result, opts):
    win = sub_result['win']
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
        t_plot_TR = t_plot / TR
    else:
        t_plot_TR = t_axis
        _smooth = lambda y: y  # noqa: E731

    fig, ax = plt.subplots(figsize=(9, 6))
    matched = sub_result['matched']
    agg = sub_result['agg']
    for cond in ('singleton_elsewhere', 'distractor_at_voxel',
                 'target_at_voxel'):
        d = agg.get(cond)
        if d is None:
            continue
        mean_p = _smooth(d['mean'])
        sem_p = _smooth(d['sem'])
        color = EVENT_COLORS[cond]
        is_main = cond != 'singleton_elsewhere'
        lw = 2.6 if is_main else 1.6
        alpha_fill = 0.22 if is_main else 0.12
        ax.fill_between(t_plot_TR, mean_p - sem_p, mean_p + sem_p,
                         color=color, alpha=alpha_fill, linewidth=0)
        ax.plot(t_plot_TR, mean_p, color=color, lw=lw,
                 label=f"{EVENT_LABELS[cond]}  n={d['n']}")
    ax.axvline(0, color='k', lw=0.5, alpha=0.3)
    ax.grid(alpha=0.15)
    ax.set_xlabel('TR (relative to trial onset)', fontsize=9)
    ax.set_ylabel('BOLD (z, voxel-mean)', fontsize=9)
    ax.legend(loc='upper right', fontsize=9, frameon=True, framealpha=0.85)

    quadrants_str = matched['quadrant'].value_counts().to_dict()
    fig.suptitle(
        f'sub-{sub.subject_id:02d}  TARGET / DISTRACTOR ONSET BOLD\n'
        f'σ ∈ [{opts.sd_min:.1f}, {opts.sd_max:.1f}]°, '
        f'R² > {opts.r2_min:.1f}, {opts.margin_kind} fully in quadrant,  '
        f'{len(matched)} voxels  '
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
    p.add_argument('--prf-model', type=int, default=4)
    p.add_argument('--sd-min', type=float, default=0.7)
    p.add_argument('--sd-max', type=float, default=1.3)
    p.add_argument('--r2-min', type=float, default=0.4)
    p.add_argument('--margin-kind', choices=['fwhm', 'sd'], default='fwhm')
    p.add_argument('--min-voxels', type=int, default=10)
    p.add_argument('--window-TRs', type=int, default=21)
    p.add_argument('--plot-upsample-dt', type=float, default=0.1)
    p.add_argument('--out', type=Path,
                   default=Path('notes/figures/voxel_traces_events.pdf'))
    args = p.parse_args()

    out = args.out
    if not out.is_absolute():
        out = Path('/Users/gdehol/git/retsupp') / out
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f'Output: {out}')

    bids = Path(args.bids_folder)
    with PdfPages(out) as pdf:
        fig, ax = plt.subplots(figsize=(11, 8.5)); ax.axis('off')
        body = (
            'Per-subject BOLD locked to TARGET / DISTRACTOR ONSET.\n\n'
            f'σ ∈ [{args.sd_min}, {args.sd_max}]°,  R² > {args.r2_min}\n'
            f'PRF {args.margin_kind.upper()} fully in one quadrant\n\n'
            'For each (matched voxel × trial), classify trial:\n'
            '  • target_at_voxel: target_location == voxel quadrant\n'
            '  • distractor_at_voxel: distractor_location == voxel quadrant\n'
            '  • singleton_elsewhere: both target and distractor at other rings\n\n'
            'Time-lock cleaned BOLD ±N TRs around trial onset; average\n'
            'across (matched voxel × trial) per condition.\n'
        )
        ax.text(0.5, 0.5, body, ha='center', va='center', fontsize=11,
                family='monospace')
        pdf.savefig(fig); plt.close(fig)

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
            render_subject_page(pdf, sub, res, args)
    print(f'\nDone: {out}')


if __name__ == '__main__':
    main()
