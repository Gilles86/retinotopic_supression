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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subject', type=int, default=2)
    parser.add_argument('--roi', default='V3')
    parser.add_argument('--bids-folder', type=Path,
                        default=Path('/data/ds-retsupp'))
    parser.add_argument('--fit-dir-name', default='af_prf_joint_full')
    parser.add_argument('--mode', default='signed')
    parser.add_argument('--n-voxels', type=int, default=8,
                        help='Top-N voxels by R² to display.')
    parser.add_argument('--out', type=Path,
                        default=Path('notes/voxel_timeseries.pdf'))
    parser.add_argument('--max-tr', type=int, default=None,
                        help='Crop the time-series to this many TRs '
                             'for plotting (default: full concatenation).')
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
    order = np.argsort(r2)[::-1]
    top = order[:args.n_voxels]
    print(f'Top-{args.n_voxels} voxel R²: '
          f'[{r2[top].min():.2f}, {r2[top].max():.2f}]')

    T = bold.shape[0]
    if args.max_tr is not None:
        T = min(T, args.max_tr)
    t = np.arange(T)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(args.out) as pdf:
        # Cover.
        fig, ax = plt.subplots(figsize=(11, 8.5)); ax.axis('off')
        sh = pickle.load(open(fit_pkl, 'rb'))['shared_pars']
        ax.text(0.5, 0.95,
                 f'sub-{args.subject:02d}  {args.roi}  '
                 'voxel BOLD: observed vs AF+PRF fit',
                 ha='center', va='top', fontsize=18, weight='bold',
                 transform=ax.transAxes)
        body = (
            f'  Top-{args.n_voxels} voxels by R² (range {r2[top].min():.2f}'
            f' - {r2[top].max():.2f}).\n'
            f'  Concatenated across {len(run_meta)} runs '
            f'(showing {T} TRs of {bold.shape[0]} total).\n'
            f'\n'
            f'  Shared AF parameters:\n'
            f"    σ_AF    = {sh['sigma_AF']:.2f}\n"
            f"    g_HP    = {sh['g_HP']:+.3f}\n"
            f"    g_LP    = {sh['g_LP']:+.3f}\n"
            f"    g_diff  = {sh['g_HP'] - sh['g_LP']:+.3f}\n"
        )
        ax.text(0.05, 0.85, body, ha='left', va='top',
                 family='monospace', fontsize=10,
                 transform=ax.transAxes)
        pdf.savefig(fig); plt.close(fig)

        # Voxel panels: 4 per page.
        per_page = 4
        n_pages = int(np.ceil(args.n_voxels / per_page))
        for p in range(n_pages):
            fig, axes = plt.subplots(per_page, 1, figsize=(14, 10),
                                       sharex=True)
            for k, ax in enumerate(axes):
                vi = p * per_page + k
                if vi >= len(top):
                    ax.axis('off'); continue
                vox = top[vi]
                ax = plot_voxel_panel(
                    ax, t, bold[:T, vox], pred[:T, vox], run_meta,
                    title=(f'voxel #{vi+1}  global_idx={voxel_idx[vox]}  '
                            f'R²={r2[vox]:.2f}    '
                            f'PRF (x,y)=({fit_pars["x"].iloc[vox]:+.2f}, '
                            f'{fit_pars["y"].iloc[vox]:+.2f}), '
                            f'sd={fit_pars["sd"].iloc[vox]:.2f}'),
                )
            axes[-1].set_xlabel('time (TRs)', fontsize=10)
            if k == 0:
                axes[0].legend(fontsize=10, loc='upper right')
            fig.tight_layout()
            pdf.savefig(fig); plt.close(fig)
    print(f'Wrote {args.out}')


if __name__ == '__main__':
    main()
