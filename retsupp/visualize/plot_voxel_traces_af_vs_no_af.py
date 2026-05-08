"""Per-voxel BOLD time courses, time-locked to bar-passes-RF, AF vs no-AF.

For each (subject, ROI) combination, this script:

1. Loads the per-voxel mean-fit DoG+flexible-HRF PRF parameters (model 4).
2. Loads the joint dynamic AF + DoG fit pickle (default:
   ``af_prf_joint_dynamic_v3_dog_with_target_sharedSigma``; falls back
   to other available variants in the SAME folder family — see below).
3. Filters to ROI voxels with mean R² > thr and PRF center at least
   `min_dist_from_ring` deg from the four ring positions (so the bar
   passes through the RF center).
4. Recomputes per-voxel BOLD predictions from BOTH the AF model (with
   fitted shared parameters) and a "no-AF" baseline (a plain
   DoG+flexHRF model 4 instance using the per-voxel mean-fit params).
5. Picks the top-N voxels by predicted ΔR² between AF and no-AF models
   on HP-close runs (where the HP location is closest to the voxel
   center) — the voxels where the AF model should win most.
6. Per voxel, identifies bar-pass-near-RF events for ALL 4 sweep
   directions per run (bar_right/left/up/down). Time-locks the cleaned
   BOLD ±N TRs around the TR where the bar centre crosses the voxel's
   PRF (along the relevant axis). Averages across runs/sweeps split by
   HP-close / HP-far / HP-orthogonal categories.

   If ``--by-direction`` is passed, sweeps are further split by their
   spatial relation to the HP-distractor location: each sweep's
   direction-of-motion vector is compared to the (HP − voxel) vector;
   sweeps moving TOWARDS the HP location (positive dot product) are
   averaged separately from those moving AWAY.
7. Plots observed (thick line ± SEM across sweep events) and the two
   models (thin lines) per condition. Adds a tiny inset visualising
   the voxel's PRF and the ring positions coloured by condition.

Output: ``notes/figures/voxel_traces_af_vs_no_af.pdf`` (one page per
(subject, ROI)).

Parameterized to handle the (fits-not-yet-ready) sharedSigma+target
canonical model OR fall back to the older v3_dog model. CLI flags let
you point at any AF derivatives folder and tag.
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

# Inject the local braincoder fork ahead of any system install.
_SUB_BC = Path(__file__).resolve().parents[2] / 'libs' / 'braincoder'
if str(_SUB_BC) not in sys.path:
    sys.path.insert(0, str(_SUB_BC))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib.backends.backend_pdf import PdfPages
from nilearn import input_data
from tqdm import tqdm

from braincoder.hrf import SPMHRFModel
from braincoder.models import (
    DoGDynamicAttentionFieldPRF2DWithHRF_v2,
    DoGDynamicAttentionFieldPRF2DWithHRF_v3,
    DifferenceOfGaussiansPRF2DWithHRF,
)
from retsupp.modeling.local_models import (
    DoGDynamicAttentionFieldPRF2DWithHRF_v3_target,
    DoGDynamicAttentionFieldPRF2DWithHRF_v3_target_sharedSigma,
)
from retsupp.utils.data import Subject, distractor_locations


# Channel order used by AF models / dynamic_indicator (underscore form).
CONDITIONS = ['upper_right', 'upper_left', 'lower_left', 'lower_right']
# `distractor_locations` keys use the legacy SPACE form — translate.
RING_KEYS = ['upper right', 'upper left', 'lower left', 'lower right']
# Map condition (HP) string to its (x, y) ring location.
COND_TO_XY = {
    cond: distractor_locations[ring]
    for cond, ring in zip(CONDITIONS, RING_KEYS)
}

# ----------------------------------------------------------------------
# Model dispatch from the AF pkl metadata.
# ----------------------------------------------------------------------
# Pkl ``shared_par_labels`` values uniquely identify the model variant
# in the dog-dyn family. Keep in sync with the variants we plot.
_MODEL_DISPATCH = [
    # (label_set_frozen, ModelCls, with_target)
    (frozenset({'sigma_AF', 'g_HP', 'g_LP', 'g_HP_dyn', 'g_LP_dyn'}),
     DoGDynamicAttentionFieldPRF2DWithHRF_v2, False),
    (frozenset({'sigma_AF', 'g_HP', 'g_LP',
                'sigma_dyn', 'g_HP_dyn', 'g_LP_dyn'}),
     DoGDynamicAttentionFieldPRF2DWithHRF_v3, False),
    (frozenset({'sigma_AF', 'g_HP', 'g_LP',
                'sigma_dyn', 'g_HP_dyn', 'g_LP_dyn',
                'g_T_dyn', 'sigma_T_dyn'}),
     None,  # decided below — sharedSigma vs unrestricted target
     True),
]


def select_af_model_class(shared_par_labels: list, shared_target_sigma: bool):
    s = frozenset(shared_par_labels)
    for labels, cls, with_tgt in _MODEL_DISPATCH:
        if labels == s:
            if with_tgt:
                # Two flavours, distinguished by the pkl's
                # ``shared_target_sigma`` flag.
                if shared_target_sigma:
                    return (
                        DoGDynamicAttentionFieldPRF2DWithHRF_v3_target_sharedSigma,
                        True,
                    )
                return DoGDynamicAttentionFieldPRF2DWithHRF_v3_target, True
            return cls, False
    raise ValueError(
        f"Don't know which AF model class matches shared_par_labels="
        f"{shared_par_labels}. Update _MODEL_DISPATCH.")


# ----------------------------------------------------------------------
# Data loading.
# ----------------------------------------------------------------------
def get_ring_positions():
    return np.array([list(distractor_locations[k]) for k in RING_KEYS],
                    dtype=np.float32)


def load_paradigm_and_bold(sub: Subject, resolution: int = 50,
                            grid_radius: float = 5.0,
                            with_target: bool = True):
    """Reload the same data the AF fitter saw.

    Returns (run_meta, bold_chunks_by_run, paradigm_chunks_by_run,
             cond_indicator_full, dyn_indicator_full, target_indicator_full,
             grid_coords, n_T_run, masker, hp_per_run).

    ``run_meta`` is a list of dicts {'session','run','hp','t_start','t_end'}.
    ``bold_chunks_by_run`` is list of (T_run, V_full) ndarrays.
    """
    bold_mask = sub.get_bold_mask()
    masker = input_data.NiftiMasker(mask_img=bold_mask)
    masker.fit()

    gx, gy = sub.get_extended_grid_coordinates(
        resolution=resolution, session=1, run=1, grid_radius=grid_radius,
    )
    grid_coords = np.stack(
        (gx.ravel(), gy.ravel()), axis=1
    ).astype(np.float32)

    hp_per_run = sub.get_hpd_locations()

    bold_chunks = []
    paradigm_chunks = []
    cond_chunks = []
    dyn_chunks = []
    tgt_chunks = [] if with_target else None
    run_meta = []
    cum_T = 0
    n_T_run = sub.get_n_volumes(1, 1)

    for session in [1, 2]:
        for run in tqdm(sub.get_runs(session),
                         desc=f'ses-{session}', leave=False):
            hp = hp_per_run.get((session, run))
            if hp not in CONDITIONS:
                continue

            # Paradigm (full = bar + 8-item search-array footprints).
            par_run = sub.get_stimulus_with_distractors(
                session=session, run=run, resolution=resolution,
                grid_radius=grid_radius,
            ).astype(np.float32)
            par_run_flat = par_run.reshape((par_run.shape[0], -1))

            bold_fn = (sub.bids_folder / 'derivatives' / 'cleaned'
                        / f'sub-{sub.subject_id:02d}'
                        / f'ses-{session}' / 'func'
                        / f'sub-{sub.subject_id:02d}_ses-{session}_'
                          f'task-search_desc-cleaned_run-{run}_bold.nii.gz')
            if not bold_fn.exists():
                print(f'  [WARN] missing BOLD: {bold_fn}')
                continue
            data = masker.transform(bold_fn).astype(np.float32)
            if data.shape[0] >= n_T_run:
                data = data[:n_T_run]
            else:
                pad = np.zeros((n_T_run - data.shape[0], data.shape[1]),
                                dtype=np.float32)
                data = np.vstack([data, pad])
            if par_run_flat.shape[0] >= n_T_run:
                par_run_flat = par_run_flat[:n_T_run]
            else:
                pad_p = np.zeros((n_T_run - par_run_flat.shape[0],
                                   par_run_flat.shape[1]),
                                  dtype=np.float32)
                par_run_flat = np.vstack([par_run_flat, pad_p])

            hp_idx = CONDITIONS.index(hp)
            cond_indicator = np.zeros((n_T_run, len(CONDITIONS)),
                                        dtype=np.float32)
            cond_indicator[:, hp_idx] = 1.0

            dyn_indicator = sub.get_dynamic_indicator(
                session=session, run=run, oversampling=1,
            ).astype(np.float32)
            if dyn_indicator.shape[0] >= n_T_run:
                dyn_indicator = dyn_indicator[:n_T_run]
            else:
                pad_d = np.zeros((n_T_run - dyn_indicator.shape[0], 4),
                                  dtype=np.float32)
                dyn_indicator = np.vstack([dyn_indicator, pad_d])

            if with_target:
                tgt_indicator = sub.get_target_indicator(
                    session=session, run=run, oversampling=1,
                ).astype(np.float32)
                if tgt_indicator.shape[0] >= n_T_run:
                    tgt_indicator = tgt_indicator[:n_T_run]
                else:
                    pad_t = np.zeros((n_T_run - tgt_indicator.shape[0], 4),
                                      dtype=np.float32)
                    tgt_indicator = np.vstack([tgt_indicator, pad_t])
                tgt_chunks.append(tgt_indicator)

            bold_chunks.append(data)
            paradigm_chunks.append(par_run_flat)
            cond_chunks.append(cond_indicator)
            dyn_chunks.append(dyn_indicator)
            run_meta.append({
                'session': session, 'run': run, 'hp': hp,
                't_start': cum_T, 't_end': cum_T + n_T_run,
            })
            cum_T += n_T_run

    return (run_meta, bold_chunks, paradigm_chunks,
            cond_chunks, dyn_chunks, tgt_chunks,
            grid_coords, n_T_run, masker, hp_per_run)


# ----------------------------------------------------------------------
# Bar-pass-near-RF event detection.
# ----------------------------------------------------------------------
# Per-event direction-of-motion unit vector (after passing the voxel,
# the bar continues in this direction). Note: the y axis sign convention
# here matches `distractor_locations` (which uses screen-coords with
# +y = up; e.g. upper_right = (+, +)).
BAR_DIRECTION_VECTORS = {
    'bar_right': ( 1.0,  0.0),
    'bar_left':  (-1.0,  0.0),
    'bar_up':    ( 0.0,  1.0),
    'bar_down':  ( 0.0, -1.0),
}


def find_bar_pass_TRs(sub: Subject, session: int, run: int,
                       x_voxel: float, y_voxel: float,
                       max_dist: float = 0.5):
    """Find the TR within EACH bar-sweep where the bar's centre crosses
    the voxel's PRF along the relevant axis.

    Returns a list of dicts, one per qualifying sweep, with keys:
      'event_type'  one of bar_right / bar_left / bar_up / bar_down
      'tr_local'    int, TR index within the run (0..n_T_run-1)
      'direction'   tuple (dx, dy), unit vector of bar motion
      'sweep_axis_dist'  signed perpendicular distance (deg) of bar
                          centre from voxel along the sweep axis at the
                          chosen TR (small ⇒ hit close to PRF centre)
    """
    settings = sub.get_experimental_settings(session, run)
    radius = settings['radius_bar_aperture']
    speed = settings['speed']
    bar_width = settings['bar_width']
    tr = sub.get_tr(session, run)

    onsets = sub.get_onsets(session, run)
    bar = onsets[onsets['event_type'].apply(lambda x: x.startswith('bar'))]
    duration = (2 * radius + bar_width) / speed
    n_T_run = sub.get_n_volumes(session, run)
    frametimes = np.arange(tr / 2., tr * n_T_run + tr / 2., tr)[:n_T_run]

    out = []
    for _, ev in bar.iterrows():
        et = ev['event_type']
        if et not in BAR_DIRECTION_VECTORS:
            continue
        onset = ev['onset']
        t_end = onset + duration
        in_sweep = (frametimes >= onset) & (frametimes <= t_end)
        if not in_sweep.any():
            continue
        # Compute bar position at every TR centre, then pick the TR
        # closest to the voxel along the sweep axis.
        if et == 'bar_right':
            pos = -radius - bar_width / 2 + (frametimes - onset) * speed
            target = x_voxel; axis_is_x = True
        elif et == 'bar_left':
            pos = +radius + bar_width / 2 - (frametimes - onset) * speed
            target = x_voxel; axis_is_x = True
        elif et == 'bar_up':
            pos = -radius - bar_width / 2 + (frametimes - onset) * speed
            target = y_voxel; axis_is_x = False
        else:  # bar_down
            pos = +radius + bar_width / 2 - (frametimes - onset) * speed
            target = y_voxel; axis_is_x = False
        dist = np.abs(pos - target)
        dist[~in_sweep] = np.inf
        tr_idx = int(np.argmin(dist))
        if dist[tr_idx] > max_dist:
            continue
        out.append({
            'event_type': et,
            'tr_local': tr_idx,
            'direction': BAR_DIRECTION_VECTORS[et],
            'sweep_axis_dist': float(pos[tr_idx] - target),
            'axis_is_x': axis_is_x,
        })
    return out


# ----------------------------------------------------------------------
# Voxel selection.
# ----------------------------------------------------------------------
def categorize_runs_by_hp_distance(run_meta, x, y):
    """For each run, classify HP location relative to voxel (x, y) into
    'close' / 'far' / 'orth' (closest / farthest / 2 perpendicular)."""
    # Distances from voxel to each ring position.
    dists = {cond: np.hypot(x - xy[0], y - xy[1])
             for cond, xy in COND_TO_XY.items()}
    # Sort to get rank.
    sorted_conds = sorted(dists, key=dists.get)  # closest first
    closest = sorted_conds[0]
    farthest = sorted_conds[-1]
    orth = set(sorted_conds[1:-1])

    out = []
    for m in run_meta:
        hp = m['hp']
        if hp == closest:
            cat = 'close'
        elif hp == farthest:
            cat = 'far'
        elif hp in orth:
            cat = 'orth'
        else:
            cat = 'unknown'
        out.append(cat)
    return out, closest, farthest


def build_af_model(ModelCls, with_target, *,
                    grid_coords, paradigm, cond_indicator,
                    dyn_indicator, tgt_indicator, hrf_model, mode):
    kwargs = dict(
        grid_coordinates=grid_coords, paradigm=paradigm,
        hrf_model=hrf_model,
        condition_indicator=cond_indicator,
        dynamic_indicator=dyn_indicator,
        ring_positions=get_ring_positions(),
        mode=mode,
    )
    if with_target:
        if tgt_indicator is None:
            raise ValueError(
                "with_target model requires target_indicator")
        kwargs['target_indicator'] = tgt_indicator
    return ModelCls(**kwargs)


def predict_af(model, fit_pars, paradigm):
    """Forward pass returning (T, V_kept). fit_pars columns must include
    the model.parameter_labels in order."""
    pars_arr = fit_pars[model.parameter_labels].values.astype(np.float32)
    pars_tf = tf.constant(pars_arr[np.newaxis, ...])      # (1, V, P)
    para_tf = tf.constant(paradigm[np.newaxis, ...])      # (1, T, G)
    pred_tf = model._predict(para_tf, pars_tf, None)      # (1, T, V)
    return pred_tf.numpy()[0]


def predict_no_af(grid_coords, paradigm, fit_pars_mean_model, hrf_model):
    """Predicted BOLD using the plain DoG + flexible HRF PRF (model 4)."""
    model = DifferenceOfGaussiansPRF2DWithHRF(
        grid_coordinates=grid_coords,
        hrf_model=hrf_model,
        flexible_hrf_parameters=True,
    )
    labels = model.parameter_labels
    pars_arr = fit_pars_mean_model[labels].values.astype(np.float32)
    pars_tf = tf.constant(pars_arr[np.newaxis, ...])
    para_tf = tf.constant(paradigm[np.newaxis, ...])
    pred_tf = model._predict(para_tf, pars_tf, None)
    return pred_tf.numpy()[0]


def per_voxel_r2(obs, pred):
    ss_res = ((obs - pred) ** 2).sum(axis=0)
    ss_tot = ((obs - obs.mean(axis=0)) ** 2).sum(axis=0) + 1e-9
    return 1.0 - ss_res / ss_tot


# ----------------------------------------------------------------------
# Main per (subject, ROI) figure.
# ----------------------------------------------------------------------
def make_subject_roi_page(pdf, sub: Subject, roi: str,
                            af_pkl_path: Path,
                            mean_prf_full: pd.DataFrame,
                            opts):
    print(f'\n=== sub-{sub.subject_id:02d} | ROI={roi} ===')
    print(f'AF pkl: {af_pkl_path}')
    if not af_pkl_path.exists():
        print(f'  [SKIP] AF pkl not found.')
        return False

    with open(af_pkl_path, 'rb') as f:
        d = pickle.load(f)
    fit_pars: pd.DataFrame = d['fit_pars']
    voxel_idx = np.asarray(d['voxel_mask_indices'])
    mode = d.get('mode', 'signed')
    resolution = d.get('resolution', opts.resolution)
    grid_radius = d.get('grid_radius', opts.grid_radius)
    shared_par_labels = d.get('shared_par_labels', list(d['shared_pars']))
    shared_target_sigma = d.get('shared_target_sigma', False)
    AFCls, with_target = select_af_model_class(
        shared_par_labels, shared_target_sigma)
    print(f'  Model: {AFCls.__name__} (with_target={with_target}, '
          f'mode={mode}, n_voxels_in_pkl={len(voxel_idx)})')

    # ---- Load BOLD + paradigm + indicators (once per subject/ROI). ----
    (run_meta, bold_chunks, par_chunks,
     cond_chunks, dyn_chunks, tgt_chunks,
     grid_coords, n_T_run, masker, hp_per_run) = load_paradigm_and_bold(
        sub, resolution=resolution, grid_radius=grid_radius,
        with_target=with_target,
    )
    if not run_meta:
        print('  [SKIP] No runs loaded.')
        return False
    bold_full = np.vstack(bold_chunks)        # (T_total, V_full)
    paradigm = np.vstack(par_chunks)
    cond_ind = np.vstack(cond_chunks)
    dyn_ind = np.vstack(dyn_chunks)
    tgt_ind = np.vstack(tgt_chunks) if with_target else None

    # ---- Mean-fit model 4 PRF for the AF voxels. ----
    mean_pars = mean_prf_full.iloc[voxel_idx].copy()

    # ---- Filter voxels: ROI ∩ R² > thr ∩ distance criterion. ----
    # ROI mask in BOLD voxel space (0..V_full-1).
    masker_full = sub.get_bold_mask(return_masker=True)
    masker_full.fit()
    roi_aliases = {
        'V3AB': ['V3A', 'V3B'], 'LO': ['LO1', 'LO2'],
        'TO': ['TO1', 'TO2'], 'VO': ['VO1', 'VO2'],
    }
    component_rois = roi_aliases.get(roi, [roi])
    roi_arr = np.zeros(bold_full.shape[1], dtype=bool)
    for r in component_rois:
        roi_img = sub.get_retinotopic_roi(roi=r, bold_space=True)
        roi_arr |= masker_full.transform(roi_img).astype(bool).flatten()

    # Map: which entries in voxel_idx are in the ROI?
    in_roi = roi_arr[voxel_idx]
    n_in_roi = int(in_roi.sum())
    print(f'  voxels in pkl ∩ ROI ({roi}): {n_in_roi}/{len(voxel_idx)}')

    r2_mean = mean_pars['r2'].values
    r2_pass = r2_mean > opts.r2_threshold
    print(f'  voxels with mean R² > {opts.r2_threshold}: {int(r2_pass.sum())}/{len(voxel_idx)}')

    # Distance from each ring position > min_dist.
    min_dist_pass = np.ones(len(voxel_idx), dtype=bool)
    for cond, xy in COND_TO_XY.items():
        d_ring = np.hypot(mean_pars['x'].values - xy[0],
                            mean_pars['y'].values - xy[1])
        min_dist_pass &= (d_ring > opts.min_dist_from_ring)
    print(f'  voxels with PRF center >{opts.min_dist_from_ring}° from all '
          f'4 ring positions: {int(min_dist_pass.sum())}/{len(voxel_idx)}')

    # PRF center within stimulated FOV (so the bar can hit it).
    settings = sub.get_experimental_settings(1, 1)
    radius = settings['radius_bar_aperture']
    inside_fov = (np.abs(mean_pars['x'].values) < radius - 0.3) & \
                 (np.abs(mean_pars['y'].values) < radius - 0.3)
    print(f'  voxels with PRF center inside bar FOV: {int(inside_fov.sum())}'
          f'/{len(voxel_idx)}')

    valid = in_roi & r2_pass & min_dist_pass & inside_fov
    n_valid = int(valid.sum())
    print(f'  voxels passing ALL filters: {n_valid}')
    if n_valid < 2:
        print('  [SKIP] not enough valid voxels.')
        return False

    # ---- Compute AF + no-AF predictions only for valid voxels. ----
    valid_idx = np.where(valid)[0]
    fit_sub = fit_pars.iloc[valid_idx].copy()
    mean_sub = mean_pars.iloc[valid_idx].copy()

    # Renormalize column order for the no-AF model (DoG flex HRF expects
    # ['x','y','sd','baseline','amplitude','srf_amplitude','srf_size',
    #  'hrf_delay','hrf_dispersion']).
    no_af_cols = ['x', 'y', 'sd', 'baseline', 'amplitude',
                  'srf_amplitude', 'srf_size',
                  'hrf_delay', 'hrf_dispersion']
    missing = [c for c in no_af_cols if c not in mean_sub.columns]
    if missing:
        print(f'  [SKIP] mean PRF missing cols {missing}')
        return False

    print(f'  Building AF model & forward pass (V={len(valid_idx)})...')
    hrf_model = SPMHRFModel(tr=sub.get_tr(1, 1),
                              delay=4.5, dispersion=0.75)
    af_model = build_af_model(
        AFCls, with_target,
        grid_coords=grid_coords, paradigm=paradigm,
        cond_indicator=cond_ind, dyn_indicator=dyn_ind,
        tgt_indicator=tgt_ind, hrf_model=hrf_model, mode=mode,
    )
    pred_af = predict_af(af_model, fit_sub, paradigm)        # (T, V)
    print('  Forward pass: no-AF (DoG+flex HRF, model 4)...')
    pred_no_af = predict_no_af(grid_coords, paradigm, mean_sub, hrf_model)

    bold_kept = bold_full[:, voxel_idx[valid_idx]]            # (T, V)

    # ---- Per-voxel R² overall + per-condition. ----
    r2_af = per_voxel_r2(bold_kept, pred_af)
    r2_no_af = per_voxel_r2(bold_kept, pred_no_af)

    # Categorize each run by HP distance vs voxel.
    # Score voxels by ΔR² on HP-close runs.
    score = np.zeros(len(valid_idx))
    for vi in range(len(valid_idx)):
        x, y = mean_sub['x'].iloc[vi], mean_sub['y'].iloc[vi]
        cats, _, _ = categorize_runs_by_hp_distance(run_meta, x, y)
        close_runs = [i for i, c in enumerate(cats) if c == 'close']
        if not close_runs:
            score[vi] = -np.inf
            continue
        # Stack on close runs only.
        obs_close = np.concatenate(
            [bold_kept[run_meta[i]['t_start']:run_meta[i]['t_end'], vi]
             for i in close_runs])
        af_close = np.concatenate(
            [pred_af[run_meta[i]['t_start']:run_meta[i]['t_end'], vi]
             for i in close_runs])
        no_close = np.concatenate(
            [pred_no_af[run_meta[i]['t_start']:run_meta[i]['t_end'], vi]
             for i in close_runs])
        ss_t = ((obs_close - obs_close.mean()) ** 2).sum() + 1e-9
        r2_af_close = 1 - ((obs_close - af_close) ** 2).sum() / ss_t
        r2_no_close = 1 - ((obs_close - no_close) ** 2).sum() / ss_t
        score[vi] = r2_af_close - r2_no_close

    # ---- Pick top-N voxels by ΔR² on HP-close runs. ----
    n_pick = opts.n_voxels_per_page
    order = np.argsort(score)[::-1]  # largest ΔR² first
    picked = [int(oi) for oi in order
              if np.isfinite(score[oi])][:n_pick]
    if picked:
        s_min = float(score[picked[-1]])
        s_max = float(score[picked[0]])
        print(f'  Picked {len(picked)} voxels by ΔR²(HP-close): '
              f'range [{s_min:+.3f} ... {s_max:+.3f}]')
    else:
        print('  Picked 0 voxels.')
        return False

    # ---- Build per-voxel time-locked traces. ----
    # For each picked voxel, find ALL bar-pass TRs (one per qualifying
    # sweep across the 4 directions × all runs) and gather BOLD ±N TRs.
    win = opts.window_TRs
    half = win // 2

    voxel_traces = []
    for vi in picked:
        x, y = mean_sub['x'].iloc[vi], mean_sub['y'].iloc[vi]
        cats, closest_cond, farthest_cond = \
            categorize_runs_by_hp_distance(run_meta, x, y)

        # Iterate over runs × all 4 sweep directions. Each (run, sweep)
        # contributes one event if the bar passes within max_bar_dist
        # of the voxel along the sweep axis.
        run_segments = []
        for ri, m in enumerate(run_meta):
            events = find_bar_pass_TRs(
                sub, m['session'], m['run'], x, y,
                max_dist=opts.max_bar_dist,
            )
            hp_xy = COND_TO_XY[m['hp']]
            # Vector from voxel to HP location (normalised).
            vx, vy = hp_xy[0] - x, hp_xy[1] - y
            vn = np.hypot(vx, vy) + 1e-9
            vx /= vn; vy /= vn
            for ev in events:
                tr_local = ev['tr_local']
                t_global = m['t_start'] + tr_local
                t0 = t_global - half
                t1 = t_global + (win - half)
                if t0 < m['t_start'] or t1 > m['t_end']:
                    continue
                # Direction-relation: dot product of bar motion and
                # (HP − voxel) unit vector. >0 ⇒ TOWARDS HP.
                dvec = ev['direction']
                dot = dvec[0] * vx + dvec[1] * vy
                rel = 'toward' if dot > 0 else 'away'
                run_segments.append({
                    'session': m['session'], 'run': m['run'],
                    'hp': m['hp'], 'cat': cats[ri],
                    'event_type': ev['event_type'],
                    'rel': rel, 'dot': float(dot),
                    'obs': bold_kept[t0:t1, vi],
                    'af': pred_af[t0:t1, vi],
                    'no_af': pred_no_af[t0:t1, vi],
                })

        # Aggregate per category. If --by-direction, split each cat
        # into TOWARD vs AWAY sub-conditions (keys 'close_toward',
        # 'close_away', etc.).
        per_cat = {}
        cat_list = ('close', 'orth', 'far')
        if opts.by_direction:
            keys = [(c, r) for c in cat_list for r in ('toward', 'away')]
            for c, r in keys:
                segs = [s for s in run_segments
                        if s['cat'] == c and s['rel'] == r]
                if not segs:
                    continue
                obs_arr = np.stack([s['obs'] for s in segs], axis=0)
                af_arr = np.stack([s['af'] for s in segs], axis=0)
                no_arr = np.stack([s['no_af'] for s in segs], axis=0)
                per_cat[f'{c}_{r}'] = {
                    'n': len(segs),
                    'obs_mean': obs_arr.mean(axis=0),
                    'obs_sem': obs_arr.std(axis=0, ddof=1) / np.sqrt(len(segs))
                                if len(segs) > 1 else np.zeros(win),
                    'af_mean': af_arr.mean(axis=0),
                    'no_af_mean': no_arr.mean(axis=0),
                }
        else:
            for cat in cat_list:
                segs = [s for s in run_segments if s['cat'] == cat]
                if not segs:
                    continue
                obs_arr = np.stack([s['obs'] for s in segs], axis=0)
                af_arr = np.stack([s['af'] for s in segs], axis=0)
                no_arr = np.stack([s['no_af'] for s in segs], axis=0)
                per_cat[cat] = {
                    'n': len(segs),
                    'obs_mean': obs_arr.mean(axis=0),
                    'obs_sem': obs_arr.std(axis=0, ddof=1) / np.sqrt(len(segs))
                                if len(segs) > 1 else np.zeros(win),
                    'af_mean': af_arr.mean(axis=0),
                    'no_af_mean': no_arr.mean(axis=0),
                }

        voxel_traces.append({
            'vi': vi,
            'x': x, 'y': y,
            'sd': mean_sub['sd'].iloc[vi],
            'r2_af': r2_af[vi],
            'r2_no_af': r2_no_af[vi],
            'r2_meanfit': mean_sub['r2'].iloc[vi],
            'closest_cond': closest_cond,
            'farthest_cond': farthest_cond,
            'per_cat': per_cat,
            'run_segments': run_segments,
        })

    # ---- Print sanity stats (n events per category, summed across
    #      directions if --by-direction). ----
    print(f'\n  Picked voxels (sub-{sub.subject_id:02d}, {roi}):')
    print(f'  {"vi":>4}  {"x":>7}  {"y":>7}  {"sd":>5}  '
          f'{"R²(mean)":>9}  {"R²(AF)":>7}  {"R²(noAF)":>9}  '
          f'n_close/orth/far')
    for v in voxel_traces:
        # Sum n events across direction sub-keys for sanity print.
        def n_in(cat):
            return sum(d['n'] for k, d in v['per_cat'].items()
                       if k == cat or k.startswith(f'{cat}_'))
        n_close, n_orth, n_far = n_in('close'), n_in('orth'), n_in('far')
        print(f'  {v["vi"]:>4}  {v["x"]:+.3f}  {v["y"]:+.3f}  '
              f'{v["sd"]:.2f}  {v["r2_meanfit"]:>9.3f}  '
              f'{v["r2_af"]:>7.3f}  {v["r2_no_af"]:>9.3f}   '
              f'{n_close}/{n_orth}/{n_far}')

    # ---- Plot the page. ----
    # Layout: a grid sized to fit n_pick voxels at ~5 columns wide.
    n = len(voxel_traces)
    ncols = 5 if n >= 5 else max(1, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 4.2 * nrows),
                              sharex=True, sharey=False, squeeze=False)
    axes = axes.flatten()
    cat_colors = {'close': '#d62728', 'orth': '#7f7f7f', 'far': '#1f77b4'}
    cat_labels = {'close': 'HP-close', 'orth': 'HP-orth', 'far': 'HP-far'}
    rel_alpha = {'toward': 1.0, 'away': 0.55}
    rel_obs_lw = {'toward': 2.2, 'away': 1.6}
    t_axis = np.arange(win) - half  # TRs relative to bar-pass.

    for ax_i, v in enumerate(voxel_traces):
        ax = axes[ax_i]
        # Plot order: far → orth → close so the close lines are on top.
        if opts.by_direction:
            order_keys = []
            for cat in ('far', 'orth', 'close'):
                for rel in ('away', 'toward'):
                    k = f'{cat}_{rel}'
                    if k in v['per_cat']:
                        order_keys.append((k, cat, rel))
        else:
            order_keys = [(c, c, None)
                          for c in ('far', 'orth', 'close')
                          if c in v['per_cat']]

        for key, cat, rel in order_keys:
            data = v['per_cat'][key]
            color = cat_colors[cat]
            obs_alpha = rel_alpha[rel] if rel else 1.0
            obs_lw = rel_obs_lw[rel] if rel else 2.0
            ax.fill_between(
                t_axis, data['obs_mean'] - data['obs_sem'],
                data['obs_mean'] + data['obs_sem'],
                color=color, alpha=0.15 * obs_alpha, linewidth=0,
            )
            label = f"{cat_labels[cat]}"
            if rel is not None:
                label += f" ({rel})"
            label += f" n={data['n']}"
            ax.plot(t_axis, data['obs_mean'], color=color, lw=obs_lw,
                     alpha=obs_alpha, label=label, zorder=3)
            ax.plot(t_axis, data['no_af_mean'], color=color, lw=0.9,
                     ls='--', alpha=0.7 * obs_alpha, zorder=2)
            ax.plot(t_axis, data['af_mean'], color=color, lw=1.1,
                     ls='-', alpha=0.85 * obs_alpha, zorder=2.5)
        ax.axvline(0, color='k', lw=0.5, alpha=0.3)
        ax.grid(alpha=0.15)
        d_r2 = v['r2_af'] - v['r2_no_af']
        ax.set_title(
            f"v={v['vi']}  ({v['x']:+.1f},{v['y']:+.1f}) σ={v['sd']:.1f}\n"
            f"R²: mean={v['r2_meanfit']:.2f}  "
            f"AF={v['r2_af']:.2f}  noAF={v['r2_no_af']:.2f}  "
            f"ΔR²={d_r2:+.2f}",
            fontsize=8,
        )
        if ax_i % ncols == 0:
            ax.set_ylabel('BOLD (z)', fontsize=8)
        if ax_i // ncols == nrows - 1:
            ax.set_xlabel('TR (relative to bar-pass)', fontsize=8)

        # Legend in upper-right of first panel only.
        if ax_i == 0:
            from matplotlib.lines import Line2D
            handles = [
                Line2D([0], [0], color='#d62728', lw=2,
                        label='HP-close (obs)'),
                Line2D([0], [0], color='#7f7f7f', lw=2, label='HP-orth (obs)'),
                Line2D([0], [0], color='#1f77b4', lw=2, label='HP-far (obs)'),
                Line2D([0], [0], color='k', lw=1.2, ls='-', label='AF model'),
                Line2D([0], [0], color='k', lw=1.0, ls='--',
                        label='no-AF model'),
            ]
            if opts.by_direction:
                handles += [
                    Line2D([0], [0], color='k', lw=2.2, alpha=1.0,
                            label='toward HP'),
                    Line2D([0], [0], color='k', lw=1.6, alpha=0.55,
                            label='away from HP'),
                ]
            ax.legend(handles=handles, loc='upper right', fontsize=6,
                       frameon=True, framealpha=0.85)

        # Inset: PRF + ring positions colored by category.
        try:
            ax_in = ax.inset_axes([0.02, 0.62, 0.32, 0.35])
            ax_in.set_xlim(-5, 5); ax_in.set_ylim(-5, 5)
            ax_in.set_aspect('equal')
            ax_in.axhline(0, color='0.85', lw=0.4)
            ax_in.axvline(0, color='0.85', lw=0.4)
            # Bar aperture ring.
            theta = np.linspace(0, 2 * np.pi, 100)
            ax_in.plot(radius * np.cos(theta), radius * np.sin(theta),
                        color='0.7', lw=0.4)
            # PRF: open circle at (x, y), radius = half-FWHM
            # (= sd × √(2 ln 2) ≈ 1.177 × sd) so the outline shows
            # where the Gaussian RF response is at half-max.
            half_fwhm = v['sd'] * np.sqrt(2.0 * np.log(2.0))
            ax_in.add_patch(plt.Circle(
                (v['x'], v['y']), half_fwhm, fill=False, edgecolor='k',
                lw=1.0,
            ))
            ax_in.plot(v['x'], v['y'], 'k.', markersize=3)
            # 4 ring positions colored by category.
            for cond, xy in COND_TO_XY.items():
                if cond == v['closest_cond']:
                    cat = 'close'
                elif cond == v['farthest_cond']:
                    cat = 'far'
                else:
                    cat = 'orth'
                ax_in.plot(xy[0], xy[1], 'o',
                            color=cat_colors[cat], markersize=6,
                            markeredgecolor='k', markeredgewidth=0.4)
            ax_in.set_xticks([]); ax_in.set_yticks([])
            for spine in ax_in.spines.values():
                spine.set_linewidth(0.4); spine.set_color('0.6')
        except Exception as e:
            print(f'  [WARN] inset failed: {e}')

    # Hide unused axes.
    for ax_i in range(len(voxel_traces), len(axes)):
        axes[ax_i].axis('off')

    sweep_descr = ('all 4 sweeps × by-direction'
                    if opts.by_direction
                    else 'all 4 sweep directions pooled')
    fig.suptitle(
        f'sub-{sub.subject_id:02d}  ROI={roi}  '
        f'AF model: {AFCls.__name__}  '
        f'(bar-pass-locked, {sweep_descr})',
        fontsize=12, weight='bold',
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    pdf.savefig(fig)
    plt.close(fig)
    return True


# ----------------------------------------------------------------------
# Entry.
# ----------------------------------------------------------------------
def find_af_pkl(bids_folder: Path, subject: int, roi: str,
                  af_folders: list, af_tags: list, mode: str = 'signed'):
    """Search candidate (folder, tag) combos for a matching pkl."""
    for fold in af_folders:
        for tag in af_tags:
            p = (bids_folder / 'derivatives' / fold
                 / f'sub-{subject:02d}'
                 / f'sub-{subject:02d}_roi-{roi}_mode-{mode}_{tag}-af-prf-fit.pkl')
            if p.exists():
                return p
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bids-folder', type=Path,
                          default=Path('/data/ds-retsupp'))
    parser.add_argument('--subjects', type=int, nargs='+', default=[3, 15],
                          help='Subjects to plot (default: 3, 15).')
    parser.add_argument('--rois', type=str, nargs='+',
                          default=['V3AB', 'hV4', 'LO'],
                          help='ROIs (default V3AB, hV4, LO — the AF-rich '
                               'mid-tier set; V1 is dropped because '
                               'attentional/AF effects there are tiny). '
                               'IPS is also a candidate but the AF fits '
                               'we have were on the 8-canonical-ROI list '
                               '(V1/V2/V3/V3AB/hV4/LO/TO/VO) so no IPS pkl '
                               'is currently available.')
    parser.add_argument('--af-folders', type=str, nargs='+',
                          default=[
                              'af_prf_joint_dynamic_v3_dog_with_target_sharedSigma',
                              'af_prf_joint_dynamic_v3_dog_with_target',
                              'af_prf_joint_dynamic_v3_dog',
                              '.old_paradigm.bak/af_prf_joint_dynamic_v3_dog_with_target_sharedSigma',
                              '.old_paradigm.bak/af_prf_joint_dynamic_v3_dog',
                          ],
                          help='Candidate AF derivatives folders to search '
                               '(in order).')
    parser.add_argument('--af-tags', type=str, nargs='+',
                          default=[
                              'dog-dyn-v3-target-sharedSigma',
                              'dog-dyn-v3-target',
                              'dog-dyn-v3',
                          ],
                          help='Candidate AF filename tags.')
    parser.add_argument('--prf-folder', type=str,
                          default='prf',
                          help='Mean-fit PRF derivatives folder '
                               '(default: prf). For .bak fits, pass '
                               '.old_paradigm.bak/prf.')
    parser.add_argument('--prf-model', type=int, default=4,
                          help='Mean-fit PRF model label (default 4).')
    parser.add_argument('--mode', default='signed')
    parser.add_argument('--resolution', type=int, default=50)
    parser.add_argument('--grid-radius', type=float, default=5.0)
    parser.add_argument('--r2-threshold', type=float, default=0.6,
                          help='Mean-fit PRF R² threshold (default 0.6).')
    parser.add_argument('--min-dist-from-ring', type=float, default=0.8,
                          help='Min PRF center distance to each of the 4 '
                               'ring positions (default 0.8°).')
    parser.add_argument('--n-voxels-per-page', '--n-voxels',
                          dest='n_voxels_per_page', type=int, default=8,
                          help='Number of voxels per (subject, ROI) page; '
                               'top by ΔR² on HP-close runs (default 8).')
    parser.add_argument('--window-TRs', type=int, default=21,
                          help='Time-locked window length in TRs '
                               '(±10 TRs around bar-pass).')
    parser.add_argument('--max-bar-dist', type=float, default=0.5,
                          help='Max distance (deg) between bar centre and '
                               'voxel along sweep axis at the qualifying '
                               'TR (default 0.5°).')
    parser.add_argument('--by-direction', action='store_true',
                          help='Split sweeps by their direction-of-motion '
                               'relative to the HP location (TOWARDS vs '
                               'AWAY from HP after passing the voxel).')
    parser.add_argument('--out', type=Path,
                          default=Path('notes/figures/voxel_traces_af_vs_no_af.pdf'))
    args = parser.parse_args()

    out = args.out
    if not out.is_absolute():
        out = Path('/Users/gdehol/git/retsupp') / out
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f'Output PDF: {out}')

    # Pre-load mean PRF per subject (model 4 returns ALL voxels).
    mean_prf_cache = {}

    with PdfPages(out) as pdf:
        # Cover page.
        fig, ax = plt.subplots(figsize=(11, 8.5)); ax.axis('off')
        bydir_line = ('  --by-direction ON: also splitting sweeps into '
                      'TOWARDS vs AWAY-from-HP\n'
                      if args.by_direction else
                      '  --by-direction OFF: all 4 sweep directions pooled\n')
        body = (
            'Per-voxel BOLD time courses, time-locked to bar-pass-near-RF.\n\n'
            f'Subjects: {args.subjects}\n'
            f'ROIs: {args.rois}\n'
            f'AF folders (search order): {args.af_folders}\n'
            f'AF tags (search order): {args.af_tags}\n'
            f'PRF mean-fit folder: {args.prf_folder}\n'
            f'\nFilters:\n'
            f'  mean R² > {args.r2_threshold}\n'
            f'  PRF center > {args.min_dist_from_ring}° from each ring '
            'position\n'
            f'  PRF center inside bar aperture\n'
            f'\nVoxel selection: top {args.n_voxels_per_page} by ΔR² '
            '(AF − no-AF) on HP-close runs.\n'
            f'\nBar events: ALL qualifying bar passes (4 directions × '
            'all runs); window ±'
            f'{args.window_TRs // 2} TRs around the TR where the bar '
            'centre crosses the voxel along the sweep axis.\n'
            f'{bydir_line}'
            f'\nLines:\n'
            f'  thick + shaded = observed BOLD mean ± SEM across sweeps\n'
            f'  solid thin     = AF model prediction\n'
            f'  dashed thin    = no-AF model (plain DoG + flexible HRF)\n'
            f'\nColours = HP-close / HP-orth / HP-far.\n'
        )
        ax.text(0.05, 0.97, 'Voxel traces: AF vs no-AF',
                 ha='left', va='top', fontsize=18, weight='bold',
                 transform=ax.transAxes)
        ax.text(0.05, 0.90, body,
                 ha='left', va='top', family='monospace', fontsize=9,
                 transform=ax.transAxes)
        pdf.savefig(fig); plt.close(fig)

        n_pages = 0
        for subject in args.subjects:
            sub = Subject(subject, args.bids_folder)

            if subject not in mean_prf_cache:
                # Use args.prf_folder. The Subject helper hardcodes 'prf';
                # so for non-default folder, load NIfTIs ourselves.
                if args.prf_folder == 'prf':
                    prf_full = sub.get_prf_parameters_volume(
                        model=args.prf_model, type='mean', return_images=False,
                    )
                else:
                    print(f'Loading custom PRF folder {args.prf_folder} ...')
                    par_labels = sub.get_prf_parameter_labels(
                        model=args.prf_model)
                    base = (args.bids_folder / 'derivatives' / args.prf_folder
                              / f'model{args.prf_model}'
                              / f'sub-{subject:02d}')
                    masker_full = sub.get_bold_mask(return_masker=True)
                    masker_full.fit()
                    cols = {}
                    import nibabel as nib
                    from nilearn import image as nl_image
                    for par in par_labels:
                        fn = base / f'sub-{subject:02d}_desc-{par}.nii.gz'
                        cols[par] = masker_full.transform(
                            str(fn)).flatten().astype(np.float32)
                    prf_full = pd.DataFrame(cols)
                if not isinstance(prf_full, pd.DataFrame):
                    prf_full = pd.DataFrame(prf_full)
                mean_prf_cache[subject] = prf_full
            mean_prf_full = mean_prf_cache[subject]

            for roi in args.rois:
                af_pkl_path = find_af_pkl(
                    args.bids_folder, subject, roi,
                    args.af_folders, args.af_tags, mode=args.mode,
                )
                if af_pkl_path is None:
                    print(f'\n[SKIP] sub-{subject:02d} {roi}: no AF pkl '
                          f'matched any (folder, tag) combo.')
                    continue
                ok = make_subject_roi_page(
                    pdf, sub, roi, af_pkl_path, mean_prf_full, args,
                )
                if ok:
                    n_pages += 1

    print(f'\nWrote {n_pages} ROI page(s) -> {out}')


if __name__ == '__main__':
    main()
