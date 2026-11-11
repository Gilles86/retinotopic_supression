"""Per-voxel observed-vs-predicted BOLD time series from the joint AF+PRF fit.

Pure model-fit-level visualisation — bypasses Jensen-style PRF-shift
artifacts entirely because we compare data → model at the BOLD level
where the model was actually fit.

Pipeline
--------
1. Load a fit pickle from `derivatives/af_prf_joint_full/sub-XX/`.
2. Reload the cleaned BOLD time series for the same (subject, runs).
3. Re-instantiate the AttentionFieldPRF2DWithHRF model and run the
   forward pass with the fitted parameters → predicted BOLD.
4. Pick the top-N voxels by per-voxel R² (cleanest fits).
5. Plot observed (gray) and predicted (red) BOLD per voxel,
   stitched across runs, with bar-pass events annotated.

Outputs a multi-page PDF: one page per (subject, ROI), N voxels per
page.

Usage
-----
    python -m retsupp.visualize.plot_voxel_timeseries \\
        --subject 2 --roi V3 \\
        --out notes/voxel_ts_sub02_V3.pdf
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import sys
# Inject the submodule braincoder (with AttentionFieldPRF2DWithHRF) ahead
# of any system braincoder install.
_SUB_BC = Path(__file__).resolve().parents[2] / 'libs' / 'braincoder'
if str(_SUB_BC) not in sys.path:
    sys.path.insert(0, str(_SUB_BC))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib.backends.backend_pdf import PdfPages
from nilearn import input_data

from braincoder.hrf import SPMHRFModel
from braincoder.models import AttentionFieldPRF2DWithHRF
from retsupp.utils.data import Subject, distractor_locations


CONDITIONS = ['upper_right', 'upper_left', 'lower_left', 'lower_right']
RING_KEYS = ['upper right', 'upper left', 'lower left', 'lower right']


def get_ring_positions():
    return np.array([list(distractor_locations[k]) for k in RING_KEYS],
                    dtype=np.float32)


def reload_data_and_paradigm(sub: Subject, resolution: int = 50,
                                paradigm_type: str = 'full',
                                grid_radius: float = 5.0):
    """Mirror fit_af_prf_braincoder.build_data_and_paradigm so the
    predictions are computed on EXACTLY the same data the fit saw.
    Returns BOLD (T, V), paradigm (T, G), condition_indicator (T, n_C),
    grid_coordinates (G, 2), masker, run_breaks (list of T-cumulative).
    """
    bold_mask = sub.get_bold_mask()
    masker = input_data.NiftiMasker(mask_img=bold_mask)
    masker.fit()

    if paradigm_type == 'bar':
        paradigm0 = sub.get_stimulus(session=1, run=1, resolution=resolution
                                       ).astype(np.float32)
        gx, gy = sub.get_grid_coordinates(resolution=resolution,
                                            session=1, run=1)
        grid_coordinates = np.stack(
            (gx.ravel(), gy.ravel()), axis=1
        ).astype(np.float32)
    else:
        paradigm0 = sub.get_stimulus_with_distractors(
            session=1, run=1, resolution=resolution,
            grid_radius=grid_radius
        ).astype(np.float32)
        gx, gy = sub.get_extended_grid_coordinates(
            resolution=resolution, session=1, run=1,
            grid_radius=grid_radius,
        )
        grid_coordinates = np.stack(
            (gx.ravel(), gy.ravel()), axis=1
        ).astype(np.float32)
    n_T_run = paradigm0.shape[0]

    hp_per_run = sub.get_hpd_locations()
    bold_chunks = []
    paradigm_chunks = []
    cond_chunks = []
    run_meta = []   # list of (session, run, hp, T_offset)
    cum_T = 0
    for session in [1, 2]:
        for run in sub.get_runs(session):
            bold_fn = (sub.bids_folder / 'derivatives' / 'cleaned'
                        / f'sub-{sub.subject_id:02d}'
                        / f'ses-{session}' / 'func'
                        / f'sub-{sub.subject_id:02d}_ses-{session}_'
                          f'task-search_desc-cleaned_run-{run}_bold.nii.gz')
            if not bold_fn.exists():
                continue
            data = masker.transform(bold_fn).astype(np.float32)
            if data.shape[0] >= n_T_run:
                data = data[:n_T_run]
            else:
                pad = np.zeros((n_T_run - data.shape[0], data.shape[1]),
                                dtype=np.float32)
                data = np.vstack([data, pad])

            hp = hp_per_run.get((session, run))
            if hp not in CONDITIONS:
                continue
            hp_idx = CONDITIONS.index(hp)
            cond_indicator = np.zeros((n_T_run, len(CONDITIONS)),
                                         dtype=np.float32)
            cond_indicator[:, hp_idx] = 1.0

            if paradigm_type == 'full':
                para_run = sub.get_stimulus_with_distractors(
                    session=session, run=run, resolution=resolution,
                    grid_radius=grid_radius
                ).astype(np.float32)
                # Crop / pad to n_T_run.
                if para_run.shape[0] >= n_T_run:
                    para_run = para_run[:n_T_run]
                else:
                    pad_p = np.zeros((n_T_run - para_run.shape[0],
                                        para_run.shape[1], para_run.shape[2]),
                                       dtype=np.float32)
                    para_run = np.vstack([para_run, pad_p])
                para_run = para_run.reshape(para_run.shape[0], -1)
            else:
                para_run = paradigm0.reshape(n_T_run, -1)

            bold_chunks.append(data)
            paradigm_chunks.append(para_run)
            cond_chunks.append(cond_indicator)
            run_meta.append({
                'session': session, 'run': run, 'hp': hp,
                't_start': cum_T, 't_end': cum_T + n_T_run,
            })
            cum_T += n_T_run

    bold = np.vstack(bold_chunks)
    paradigm = np.vstack(paradigm_chunks)
    condition_indicator = np.vstack(cond_chunks)
    return bold, paradigm, condition_indicator, grid_coordinates, masker, run_meta


def predict_voxel_timeseries(fit_pkl: Path, sub: Subject,
                                paradigm_type: str = 'full',
                                resolution: int = 50,
                                grid_radius: float = 5.0):
    """Return (bold (T, V_kept), pred (T, V_kept), fit_pars, run_meta,
    voxel_indices_global)."""
    with open(fit_pkl, 'rb') as f:
        d = pickle.load(f)
    fit_pars: pd.DataFrame = d['fit_pars']
    voxel_idx = np.asarray(d['voxel_mask_indices'])
    mode = d.get('mode', 'signed')
    paradigm_type_used = d.get('paradigm_type', paradigm_type)
    resolution_used = d.get('resolution', resolution)

    bold_full, paradigm, ci, grid, _, run_meta = reload_data_and_paradigm(
        sub,
        resolution=resolution_used,
        paradigm_type=paradigm_type_used,
        grid_radius=grid_radius,
    )
    bold = bold_full[:, voxel_idx]    # (T, V_kept)

    # Re-instantiate the model with the SAME paradigm + condition
    # indicator and run the forward pass on the fitted parameters.
    hrf_model = SPMHRFModel(tr=sub.get_tr(session=1, run=1),
                              delay=4.5, dispersion=0.75)
    model = AttentionFieldPRF2DWithHRF(
        grid_coordinates=grid, paradigm=paradigm,
        hrf_model=hrf_model,
        condition_indicator=ci,
        ring_positions=get_ring_positions(),
        mode=mode,
    )
    # `predict` returns predictions per parameter row.
    pars_arr = fit_pars[model.parameter_labels].values.astype(np.float32)
    pars_tf = tf.constant(pars_arr[np.newaxis, ...])      # (1, V, P)
    paradigm_tf = tf.constant(paradigm[np.newaxis, ...])  # (1, T, G)
    pred_tf = model._predict(paradigm_tf, pars_tf)        # (1, T, V)
    pred = pred_tf.numpy()[0]
    return bold, pred, fit_pars, run_meta, voxel_idx


def plot_voxel_panel(ax, t, obs, pred, run_meta, hp_legend=None,
                       title=''):
    ax.plot(t, obs, color='0.4', lw=1.0, alpha=0.85, label='observed BOLD')
    ax.plot(t, pred, color='C3', lw=1.4, label='AF+PRF model fit')
    # Run boundaries.
    for i, m in enumerate(run_meta):
        ax.axvline(m['t_start'], color='0.6', lw=0.4, ls=':', alpha=0.6)
        # Run label + HP.
        ax.text((m['t_start'] + m['t_end']) / 2, ax.get_ylim()[1] * 0.95,
                 f"ses-{m['session']} run-{m['run']}\nHP={m['hp']}",
                 ha='center', va='top', fontsize=7, color='0.4',
                 alpha=0.8)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.15)
    ax.set_ylabel('BOLD', fontsize=9)


def get_bar_direction_per_tr(sub: Subject, session: int, run: int) -> np.ndarray:
    """Return (n_T_run,) array of bar-direction labels at each TR.

    Labels: 'right', 'left', 'up', 'down', 'rest'/'break'/'' (the latter
    three lumped as 'rest'). Built from the events.tsv bar_* events:
    bar_right/bar_left/bar_up/bar_down events mark the start of each
    bar pass and persist until the next bar_rest/bar_break/end of run.
    """
    onsets = sub.get_onsets(session, run)
    tr = sub.get_tr(session, run)
    n_T = sub.get_n_volumes(session, run)
    frametimes = (np.arange(n_T) + 0.5) * tr
    bar_events = onsets[onsets['event_type'].apply(
        lambda x: x.startswith('bar'))]
    out = np.array(['rest'] * n_T, dtype=object)
    for i, t in enumerate(frametimes):
        prev = bar_events[bar_events['onset'] < t]
        if len(prev) == 0:
            continue
        ev = prev.iloc[-1]['event_type']
        if ev == 'bar_right':
            out[i] = 'right'
        elif ev == 'bar_left':
            out[i] = 'left'
        elif ev == 'bar_up':
            out[i] = 'up'
        elif ev == 'bar_down':
            out[i] = 'down'
        # bar_rest / bar_break → leave as 'rest'.
    return out


def annotate_bar_directions(ax, dirs: np.ndarray, y_frac: float = 0.95,
                              lower=False):
    """Shade horizontal bars at the top of `ax` indicating bar direction
    at each TR. Uses unicode arrows as labels at the centre of each
    contiguous segment."""
    colors = {
        'right': '#ffe0b2', 'left':  '#bbdefb',
        'up':    '#c8e6c9', 'down':  '#ffcdd2',
        'rest':  '#ffffff',
    }
    arrows = {
        'right': '→', 'left': '←',
        'up':    '↑', 'down': '↓',
        'rest':  '·',
    }
    n = len(dirs)
    # Scan for contiguous segments.
    i = 0
    ymin, ymax = ax.get_ylim()
    band_h = (ymax - ymin) * 0.05    # 5% of axis height
    band_y = ymin if lower else ymax - band_h
    while i < n:
        j = i + 1
        while j < n and dirs[j] == dirs[i]:
            j += 1
        ax.add_patch(plt.Rectangle(
            (i, band_y), j - i, band_h,
            color=colors.get(dirs[i], '#dddddd'),
            alpha=0.55, zorder=0,
        ))
        if dirs[i] != 'rest' and (j - i) >= 4:
            ax.text((i + j) / 2, band_y + band_h / 2,
                     arrows[dirs[i]],
                     ha='center', va='center', fontsize=18,
                     color='0.25')
        i = j


def average_per_condition(bold: np.ndarray, pred: np.ndarray,
                            run_meta: list, n_T_run: int):
    """Average BOLD and predictions per HP condition across runs.

    Returns dict: cond → (mean_bold (T, V), mean_pred (T, V), n_runs).
    """
    out = {}
    for cond in CONDITIONS:
        idx = [i for i, m in enumerate(run_meta) if m['hp'] == cond]
        if not idx:
            continue
        bold_runs = np.stack(
            [bold[m['t_start']:m['t_end']] for m in
             (run_meta[i] for i in idx)],
            axis=0,
        )           # (n_runs, T_run, V)
        pred_runs = np.stack(
            [pred[m['t_start']:m['t_end']] for m in
             (run_meta[i] for i in idx)],
            axis=0,
        )
        out[cond] = (bold_runs.mean(axis=0), pred_runs.mean(axis=0),
                      len(idx))
    return out


def predict_with_zero_gains(fit_pars: pd.DataFrame, paradigm, ci, grid,
                              hrf_model, mode: str):
    """Run forward pass with g_HP = g_LP = 0 — yields the no-AF (pure
    Gaussian PRF) BOLD prediction at each voxel's fitted (x, y, sd,
    baseline, amplitude)."""
    pars_no_af = fit_pars.copy()
    pars_no_af['g_HP'] = 0.0
    pars_no_af['g_LP'] = 0.0
    model = AttentionFieldPRF2DWithHRF(
        grid_coordinates=grid, paradigm=paradigm,
        hrf_model=hrf_model,
        condition_indicator=ci,
        ring_positions=get_ring_positions(),
        mode=mode,
    )
    pars_arr = pars_no_af[model.parameter_labels].values.astype(np.float32)
    pred_tf = model._predict(
        tf.constant(paradigm[np.newaxis, ...]),
        tf.constant(pars_arr[np.newaxis, ...]),
    )
    return pred_tf.numpy()[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subject', type=int, default=2)
    parser.add_argument('--roi', default='V3')
    parser.add_argument('--bids-folder', type=Path,
                        default=Path('/data/ds-retsupp'))
    parser.add_argument('--fit-dir-name', default='af_prf_joint_full')
    parser.add_argument('--mode', default='signed')
    parser.add_argument('--n-voxels', type=int, default=6,
                        help='Top-N voxels by combined R² × predicted '
                             'condition-shift magnitude.')
    parser.add_argument('--r2-thr', type=float, default=0.20,
                        help='Min R² for voxel inclusion.')
    parser.add_argument('--mode-display',
                        choices=['per-condition', 'concat'],
                        default='per-condition',
                        help="'per-condition': avg BOLD across runs per HP, "
                             "show 4 conditions overlaid + no-AF baseline. "
                             "'concat': raw time-series concatenation.")
    parser.add_argument('--out', type=Path,
                        default=Path('notes/voxel_timeseries.pdf'))
    parser.add_argument('--max-tr', type=int, default=None,
                        help='Crop the time-series to this many TRs '
                             'for plotting.')
    args = parser.parse_args()

    sub = Subject(args.subject, args.bids_folder)
    fit_pkl = (args.bids_folder / 'derivatives' / args.fit_dir_name
                / f'sub-{args.subject:02d}'
                / f'sub-{args.subject:02d}_roi-{args.roi}_'
                  f'mode-{args.mode}_af-prf-fit.pkl')
    if not fit_pkl.exists():
        raise SystemExit(f'No fit at {fit_pkl}')

    print(f'Loading fit + reconstructing predictions for '
          f'sub-{args.subject:02d} {args.roi} ...')
    bold, pred, fit_pars, run_meta, voxel_idx = predict_voxel_timeseries(
        fit_pkl, sub
    )
    r2 = fit_pars['r2'].values

    n_T_run = run_meta[0]['t_end'] - run_meta[0]['t_start']
    cond_avg = average_per_condition(bold, pred, run_meta, n_T_run)

    # No-AF baseline predictions FIRST (need them for the voxel selector).
    print('Computing no-AF baseline predictions (gains = 0)...')
    paradigm_, _, ci_, grid_, _, _ = reload_data_and_paradigm(
        sub, resolution=50,
        paradigm_type='full', grid_radius=5.0,
    )
    hrf_model = SPMHRFModel(tr=sub.get_tr(session=1, run=1),
                              delay=4.5, dispersion=0.75)
    pred_no_af = predict_with_zero_gains(fit_pars, paradigm_, ci_, grid_,
                                            hrf_model, args.mode)
    cond_avg_no_af = average_per_condition(bold, pred_no_af,
                                              run_meta, n_T_run)

    # ΔR² selector: voxels where the AF model explains substantially
    # more variance than the no-AF baseline. SS_res computed against the
    # SAME observed BOLD for both models.
    def per_voxel_r2(obs_, pred_):
        ss_res = ((obs_ - pred_) ** 2).sum(axis=0)
        ss_tot = ((obs_ - obs_.mean(axis=0)) ** 2).sum(axis=0) + 1e-9
        return 1.0 - ss_res / ss_tot
    r2_af = per_voxel_r2(bold, pred)
    r2_noaf = per_voxel_r2(bold, pred_no_af)
    dr2 = r2_af - r2_noaf

    # Score: high ΔR² AND minimum-quality R² so we don't pick noise.
    valid = (r2 >= args.r2_thr) & (dr2 > 0)
    score = dr2.copy()
    score[~valid] = -np.inf
    order = np.argsort(score)[::-1]
    top = order[:args.n_voxels]
    print(f'Top-{args.n_voxels} voxels by ΔR² (AF − no-AF):')
    for vox in top:
        print(f'  voxel {vox}: R²(AF)={r2_af[vox]:.3f}, '
              f'R²(no-AF)={r2_noaf[vox]:.3f}, ΔR²={dr2[vox]:+.3f}, '
              f'PRF=({fit_pars["x"].iloc[vox]:+.2f}, '
              f'{fit_pars["y"].iloc[vox]:+.2f})')

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(args.out) as pdf:
        # Cover.
        fig, ax = plt.subplots(figsize=(11, 8.5)); ax.axis('off')
        sh = pickle.load(open(fit_pkl, 'rb'))['shared_pars']
        ax.text(0.5, 0.95,
                 f'sub-{args.subject:02d}  {args.roi}  '
                 'BOLD: data vs AF+PRF model vs no-AF baseline',
                 ha='center', va='top', fontsize=17, weight='bold',
                 transform=ax.transAxes)
        body = (
            f'  Per-condition BOLD averaged across runs of that HP-condition.\n'
            f'  Voxels selected by R² ≥ {args.r2_thr} AND high predicted\n'
            f'  between-condition variance.\n'
            f'  Top-{args.n_voxels} shown. Lines:\n'
            f'    solid    = observed BOLD (mean across runs of that HP).\n'
            f'    dashed   = AF+PRF model fit.\n'
            f'    dotted   = no-AF baseline (g_HP=g_LP=0, same PRF).\n'
            f'  Colours = HP condition.\n'
            f'\n'
            f'  Shared AF parameters this fit:\n'
            f"    σ_AF    = {sh['sigma_AF']:.2f}\n"
            f"    g_HP    = {sh['g_HP']:+.3f}\n"
            f"    g_LP    = {sh['g_LP']:+.3f}\n"
            f"    g_diff  = {sh['g_HP'] - sh['g_LP']:+.3f}\n"
        )
        ax.text(0.05, 0.86, body, ha='left', va='top',
                 family='monospace', fontsize=10,
                 transform=ax.transAxes)
        pdf.savefig(fig); plt.close(fig)

        # Bar-direction annotation (compute once from a representative run).
        bar_dirs = get_bar_direction_per_tr(
            sub,
            run_meta[0]['session'], run_meta[0]['run'],
        )

        # One panel per voxel: 4 conditions overlaid, observed/AF/no-AF.
        cond_colors = {'upper_right': 'C0', 'upper_left': 'C1',
                        'lower_left': 'C2', 'lower_right': 'C3'}
        for vi, vox in enumerate(top):
            fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
            ax_top, ax_bot = axes
            for cond, col in cond_colors.items():
                if cond not in cond_avg:
                    continue
                obs_c = cond_avg[cond][0][:, vox]
                pred_c = cond_avg[cond][1][:, vox]
                no_c = cond_avg_no_af[cond][1][:, vox]
                t = np.arange(len(obs_c))
                ax_top.plot(t, obs_c, color=col, lw=2.0,
                             label=f'observed | HP={cond}')
                ax_top.plot(t, pred_c, color=col, lw=1.5, ls='--',
                             alpha=0.85)
                ax_bot.plot(t, obs_c, color=col, lw=2.0,
                             label=f'observed | HP={cond}')
                ax_bot.plot(t, no_c, color=col, lw=1.5, ls=':',
                             alpha=0.85)
            ax_top.set_title(
                f'voxel #{vi+1}   '
                f'PRF=({fit_pars["x"].iloc[vox]:+.2f}, '
                f'{fit_pars["y"].iloc[vox]:+.2f}), '
                f'sd={fit_pars["sd"].iloc[vox]:.2f}, '
                f'R²(AF)={r2_af[vox]:.2f},   '
                f'R²(no-AF)={r2_noaf[vox]:.2f},   '
                f'ΔR²={dr2[vox]:+.3f}\n'
                'TOP: observed (solid) vs AF+PRF FIT (dashed)',
                fontsize=11, weight='bold',
            )
            ax_bot.set_title(
                'BOTTOM: observed (solid) vs NO-AF baseline '
                '(dotted, g_HP=g_LP=0) — '
                'difference between the two = "what AF buys"',
                fontsize=11,
            )
            for ax in axes:
                ax.grid(alpha=0.2)
                ax.set_ylabel('BOLD', fontsize=11)
                # Direction band (top edge).
                annotate_bar_directions(ax, bar_dirs, lower=False)
            axes[-1].set_xlabel('TR within run', fontsize=11)
            ax_top.legend(fontsize=8, loc='upper right', ncol=2)
            fig.tight_layout()
            pdf.savefig(fig); plt.close(fig)
    print(f'Wrote {args.out}')


if __name__ == '__main__':
    main()
