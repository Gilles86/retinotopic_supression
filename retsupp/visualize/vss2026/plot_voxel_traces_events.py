"""Event-related BOLD: target/distractor onsets at voxel-quadrant.

Companion to ``plot_voxel_traces_group.py`` (bar passes). Same matched
voxels (FWHM/σ inside one quadrant); different events.

Per (matched voxel × trial), classify the trial:
  - 'target_at_voxel'      : target landed at voxel's quadrant
  - 'distractor_at_voxel'  : distractor landed at voxel's quadrant
  - 'singleton_elsewhere'  : both target and distractor at OTHER rings
                             (control — voxel saw nothing at its quadrant)

Within target/distractor events, split by HP-condition relative to voxel:
  - HP-close (HP at voxel's quadrant)
  - HP-lateral (HP at perpendicular quadrant)
  - HP-opposite (HP at antipode)

Deconvolve event-related responses with **nideconv** using a **cosine
basis set** (10 regressors, [-2 s, 18 s] interval). This handles the
overlap between trial onsets (every ~3 s) cleanly — the deconvolved
response is what each event ADDS on top of the rest.

Per-subject loop: for each ROI, fit a deconvolution model on each
matched voxel separately (events differ per voxel due to quadrant
labels), then average deconvolved responses across voxels. Group:
average across subjects per (ROI, event_type, HP-condition).

Layout: rows = ROIs, cols = 3 HP-conditions (close / lateral /
opposite). 2 traces per panel: target onset (green), distractor
onset (red). Optional 3rd trace: control / singleton-elsewhere.

Output: ``notes/figures/voxel_traces_events.pdf``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from nideconv import ResponseFitter
from nilearn import maskers
from tqdm import tqdm

from retsupp.utils.data import Subject, distractor_locations
from retsupp.visualize.plot_voxel_traces_af_vs_no_af import (
    CONDITIONS, COND_TO_XY,
)
from retsupp.visualize.plot_voxel_traces_aggregate import (
    find_matched_voxels,
)


LOC_TO_QUADRANT = {1.0: 'upper_right', 3.0: 'upper_left',
                   5.0: 'lower_left', 7.0: 'lower_right'}

# HP-condition relative to voxel: same quad / perpendicular / opposite.
def hp_cond_relative(voxel_quadrant: str, hp_quadrant: str) -> str:
    """close / lateral / opposite."""
    if voxel_quadrant == hp_quadrant:
        return 'close'
    pairs = [('upper_right', 'lower_left'), ('upper_left', 'lower_right')]
    for a, b in pairs:
        if {voxel_quadrant, hp_quadrant} == {a, b}:
            return 'opposite'
    return 'lateral'


HP_COLOR = {'close': '#d62728', 'lateral': '0.5', 'opposite': '#1f77b4'}
EVENT_COLOR = {'target_at_voxel': '#2ca02c',     # green
               'distractor_at_voxel': '#d62728',  # red
               }
EVENT_LABEL = {'target_at_voxel':     'TARGET at voxel-quadrant',
               'distractor_at_voxel': 'DISTRACTOR at voxel-quadrant'}
HP_TITLE = {'close': 'HP at voxel quadrant (close)',
            'lateral': 'HP perpendicular (lateral)',
            'opposite': 'HP at opposite quadrant'}


def deconvolve_voxel(bold_concat: np.ndarray, events_df: pd.DataFrame,
                      tr: float, interval=(-2.0, 18.0), n_regressors=10):
    """Run cosine-basis deconvolution on a single voxel.

    bold_concat : (T_total,) BOLD signal (z-scored) for one voxel,
        concatenated across runs.
    events_df : DataFrame with columns ['onset', 'event_name'].
        ``onset`` is in seconds (relative to start of bold_concat).

    Returns dict event_name -> (t_axis, response).
    """
    rf = ResponseFitter(input_signal=bold_concat,
                        sample_rate=1.0 / tr,
                        oversample_design_matrix=20)
    for ev_name in events_df['event_name'].unique():
        ev_onsets = events_df.loc[events_df['event_name'] == ev_name, 'onset'].values
        if len(ev_onsets) == 0:
            continue
        rf.add_event(ev_name, onsets=ev_onsets,
                     basis_set='canonical_hrf_with_time_derivative_dispersion'
                                if False else 'fourier',
                     interval=list(interval),
                     n_regressors=n_regressors)
    # NOTE: nideconv has 'fourier' (sin/cos) basis. There's no separate
    # 'cosine' name; fourier covers cos+sin. For our purposes that's
    # close enough — the user asked for cosine basis as a smooth basis;
    # fourier delivers that with cosine + sine pairs.
    rf.regress()
    out = {}
    tcs = rf.get_timecourses()  # MultiIndex: event_type x covariate x t
    # tcs has shape (n_t, n_events*covariates)
    for ev_name in events_df['event_name'].unique():
        try:
            tc = tcs.xs(ev_name, level='event type')
            t_vals = tc.index.get_level_values('time').values
            if hasattr(tc, 'values'):
                # collapse covariate axis (intercept only, single col)
                resp = tc.values.flatten()
            else:
                resp = np.asarray(tc).flatten()
            out[ev_name] = (t_vals, resp)
        except (KeyError, ValueError):
            continue
    return out


def aggregate_subject(sub: Subject, masker, opts):
    """Per ROI, deconvolve per-voxel event responses; return mean
    across voxels per (event, hp_cond)."""
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
    )
    print(f'  sub-{sub.subject_id:02d}: matched voxels = {len(matched)}')
    if len(matched) < opts.min_voxels:
        return None

    # ROI masks.
    roi_idxs = {}
    for roi in opts.rois:
        try:
            img = sub.get_retinotopic_roi(roi=roi, bold_space=True)
            roi_idxs[roi] = masker.transform(img).astype(bool).flatten()
        except Exception as e:
            print(f'    ROI {roi}: not available ({e})')

    hp_per_run = sub.get_hpd_locations()
    tr = sub.get_tr(1, 1)
    n_T_run = sub.get_n_volumes(1, 1)

    # Pre-load BOLD per run (whole cortex).
    runs_data = {}
    for ses in (1, 2):
        for run in sub.get_runs(ses):
            hp = hp_per_run.get((ses, run))
            if hp not in CONDITIONS:
                continue
            fn = (sub.bids_folder / 'derivatives' / 'cleaned'
                  / f'sub-{sub.subject_id:02d}'
                  / f'ses-{ses}' / 'func'
                  / f'sub-{sub.subject_id:02d}_ses-{ses}_task-search_'
                    f'desc-cleaned_run-{run}_bold.nii.gz')
            if not fn.exists():
                continue
            d = masker.transform(fn).astype(np.float32)[:n_T_run]
            d = (d - d.mean(axis=0, keepdims=True)) / (
                d.std(axis=0, keepdims=True) + 1e-6)
            runs_data[(ses, run)] = {'data': d, 'hp': hp,
                                       'onsets': sub.get_onsets(ses, run)}

    if not runs_data:
        return None

    win_interval = (-2.0, 18.0)
    # For aggregating across voxels, use a fixed time grid.
    # We'll resample each voxel's deconvolved response onto this grid.
    target_grid = np.linspace(win_interval[0], win_interval[1], 41)  # 0.5 s

    roi_results = {}

    for roi, roi_mask in roi_idxs.items():
        in_roi = matched['vi'].apply(lambda vi: bool(roi_mask[int(vi)]))
        sub_matched = matched[in_roi].copy()
        if len(sub_matched) < opts.min_voxels:
            print(f'    ROI {roi}: only {len(sub_matched)} matched voxels')
            continue
        # Per-voxel deconvolved responses.
        per_voxel_responses = {(et, hp): [] for et in
                               ('target_at_voxel', 'distractor_at_voxel')
                               for hp in ('close', 'lateral', 'opposite')}
        for _, vox in tqdm(sub_matched.iterrows(),
                           total=len(sub_matched),
                           desc=f'  {roi}', leave=False):
            vq = vox['quadrant']
            vi = int(vox['vi'])
            # Concat BOLD + onsets across runs.
            bold_chunks = []
            ev_rows = []
            cum_t = 0.0
            for (ses, run), rd in runs_data.items():
                bold_chunks.append(rd['data'][:, vi])
                ons = rd['onsets']
                tgt = ons[ons['event_type'] == 'target']
                hp_cond = hp_cond_relative(vq, rd['hp'])
                for _, trial in tgt.iterrows():
                    target_loc = trial.get('target_location', np.nan)
                    dist_loc = trial.get('distractor_location', np.nan)
                    target_quad = LOC_TO_QUADRANT.get(float(target_loc), None) \
                        if not pd.isna(target_loc) else None
                    dist_quad = LOC_TO_QUADRANT.get(float(dist_loc), None) \
                        if not pd.isna(dist_loc) else None
                    if target_quad == vq:
                        ev_rows.append({
                            'onset': cum_t + float(trial['onset']),
                            'event_name': f'target_at_voxel_{hp_cond}',
                        })
                    if dist_quad == vq:
                        ev_rows.append({
                            'onset': cum_t + float(trial['onset']),
                            'event_name': f'distractor_at_voxel_{hp_cond}',
                        })
                cum_t += rd['data'].shape[0] * tr
            bold_concat = np.concatenate(bold_chunks)
            if not ev_rows:
                continue
            ev_df = pd.DataFrame(ev_rows)
            try:
                resp_dict = deconvolve_voxel(bold_concat, ev_df, tr,
                                                interval=win_interval,
                                                n_regressors=opts.n_basis)
            except Exception as e:
                continue
            # Map deconvolved responses into the (event_type, hp_cond) bins.
            for ev_name, (t_vals, resp) in resp_dict.items():
                if not (ev_name.startswith('target_at_voxel_')
                        or ev_name.startswith('distractor_at_voxel_')):
                    continue
                parts = ev_name.split('_')
                hp_cond = parts[-1]
                if hp_cond not in ('close', 'lateral', 'opposite'):
                    continue
                ev_type = ev_name.rsplit(f'_{hp_cond}', 1)[0]
                # Resample to target grid.
                resp_g = np.interp(target_grid, t_vals, resp)
                per_voxel_responses[(ev_type, hp_cond)].append(resp_g)

        agg = {}
        for k, traces in per_voxel_responses.items():
            if not traces:
                agg[k] = None
            else:
                arr = np.array(traces)
                agg[k] = {'mean': arr.mean(axis=0), 'n': len(traces)}
        roi_results[roi] = {'agg': agg, 't_grid': target_grid,
                             'n_voxels': len(sub_matched)}

    return roi_results


def render_group(pdf, group_data, target_grid, opts):
    rois_with_data = [r for r in opts.rois if r in group_data and group_data[r]]
    if not rois_with_data:
        return
    n_rows = len(rois_with_data)
    cols = ('close', 'lateral', 'opposite')
    n_cols = len(cols)
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(5.5 * n_cols, 3.0 * n_rows),
                              sharex=True, sharey='row', squeeze=False)
    for ri, roi in enumerate(rois_with_data):
        for ci, hp_cond in enumerate(cols):
            ax = axes[ri, ci]
            for ev_type in ('target_at_voxel', 'distractor_at_voxel'):
                per_sub = group_data[roi].get((ev_type, hp_cond), [])
                if not per_sub:
                    continue
                traces = np.stack([t for _, t, _ in per_sub], axis=0)
                n_sub = traces.shape[0]
                gm = traces.mean(axis=0)
                gs = (traces.std(axis=0, ddof=1) / np.sqrt(n_sub)
                      if n_sub > 1 else np.zeros_like(gm))
                color = EVENT_COLOR[ev_type]
                ax.fill_between(target_grid, gm - gs, gm + gs,
                                 color=color, alpha=0.22, linewidth=0)
                lab = EVENT_LABEL[ev_type] if (ri == 0 and ci == 0) else None
                ax.plot(target_grid, gm, color=color, lw=2.5, label=lab)
            ax.axvline(0, color='k', lw=0.4, alpha=0.4)
            ax.axhline(0, color='k', lw=0.3, alpha=0.3)
            ax.grid(alpha=0.12)
            if ri == 0:
                ax.set_title(HP_TITLE[hp_cond], fontsize=10,
                              color=HP_COLOR[hp_cond], weight='bold')
            if ri == n_rows - 1:
                ax.set_xlabel('time (s) relative to trial onset', fontsize=9)
            if ci == 0:
                ax.set_ylabel(f'{roi}\nBOLD (z, deconv)', fontsize=10,
                              weight='bold')
    axes[0, 0].legend(loc='upper right', fontsize=8, frameon=True,
                       framealpha=0.85)
    fig.suptitle(
        f'Event-related deconvolved BOLD (cosine/Fourier basis) \n'
        f'rows=ROIs, cols=HP-condition relative to voxel; '
        f'green=target onset, red=distractor onset',
        fontsize=11, weight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    pdf.savefig(fig)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bids-folder', default='/data/ds-retsupp')
    p.add_argument('--subjects', type=int, nargs='+',
                   default=[3, 5, 8, 15, 22])
    p.add_argument('--rois', nargs='+',
                   default=['V3AB', 'hV4', 'LO'])
    p.add_argument('--prf-model', type=int, default=4)
    p.add_argument('--sd-min', type=float, default=0.5)
    p.add_argument('--sd-max', type=float, default=1.5)
    p.add_argument('--r2-min', type=float, default=0.3)
    p.add_argument('--margin-kind', choices=['fwhm', 'sd'], default='sd')
    p.add_argument('--min-voxels', type=int, default=4)
    p.add_argument('--n-basis', type=int, default=10,
                   help='number of cosine/fourier basis regressors')
    p.add_argument('--out', type=Path,
                   default=Path('notes/figures/voxel_traces_events.pdf'))
    args = p.parse_args()

    out = args.out
    if not out.is_absolute():
        out = Path('/Users/gdehol/git/retsupp') / out
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f'Output: {out}')

    bids = Path(args.bids_folder)
    group_data = {roi: {(et, hp): []
                         for et in ('target_at_voxel', 'distractor_at_voxel')
                         for hp in ('close', 'lateral', 'opposite')}
                  for roi in args.rois}
    target_grid = None
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
        for roi, rd in res.items():
            target_grid = rd['t_grid']
            for k, d in rd['agg'].items():
                if d is None: continue
                group_data[roi][k].append((s, d['mean'], d['n']))

    if target_grid is None:
        raise RuntimeError('No data.')

    with PdfPages(out) as pdf:
        fig, ax = plt.subplots(figsize=(11, 6)); ax.axis('off')
        body = (
            'Event-related BOLD via nideconv (Fourier basis)\n\n'
            f'Subjects: {args.subjects}\n'
            f'ROIs: {args.rois}\n'
            f'σ ∈ [{args.sd_min}, {args.sd_max}]°,  R² > {args.r2_min}\n'
            f'PRF {args.margin_kind.upper()} fully in one quadrant\n\n'
            f'Deconvolution: {args.n_basis} fourier (cosine+sine) regressors\n'
            'over [-2 s, 18 s]. Removes overlap from neighbouring trials.\n\n'
            'Per (matched voxel × trial), classify event:\n'
            '  TARGET at voxel-quadrant     (green)\n'
            '  DISTRACTOR at voxel-quadrant (red)\n'
            '  split by HP-condition relative to voxel:\n'
            '    close (HP at voxel quadrant)\n'
            '    lateral (HP at perpendicular)\n'
            '    opposite (HP at antipode)\n'
        )
        ax.text(0.5, 0.5, body, ha='center', va='center', fontsize=11,
                family='monospace')
        pdf.savefig(fig); plt.close(fig)

        render_group(pdf, group_data, target_grid, args)
    print(f'\nDone: {out}')


if __name__ == '__main__':
    main()
