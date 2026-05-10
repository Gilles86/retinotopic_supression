"""Per-trial dynamic-gain joint AF + DoG-PRF fit.

Sibling driver to :mod:`retsupp.modeling.fit_dog_dynamic_af_braincoder`,
specialised for the per-trial parameterisation. The model is the
sharedSigma v3+target variant
(:class:`DoGDynamicAttentionFieldPRF2DWithHRF_v3_target_sharedSigma_perTrial`)
with the three SCALAR dynamic gains ``g_HP_dyn``, ``g_LP_dyn``,
``g_T_dyn`` replaced by THREE PER-TRIAL VECTORS, each of length
``n_trials``.

Output
------
Per-(subject, ROI) under
``derivatives/af_prf_joint_dynamic_v3_dog_with_target_sharedSigma_perTrial/sub-XX/``:

- ``sub-XX_roi-XX_mode-signed_dog-dyn-v3-target-sharedSigma-perTrial-af-prf-pars.tsv``
  Wide TSV: one row per voxel, all parameters (including 3*n_trials
  per-trial gain columns) plus ``r2``.
- ``sub-XX_roi-XX_mode-signed_dog-dyn-v3-target-sharedSigma-perTrial-trials.tsv``
  Long TSV: one row per trial, columns
  ``subject, roi, trial_idx, session, run, trial_within_run,
  target_location, distractor_location, hp_location,
  is_hp_distractor, g_HP_dyn, g_LP_dyn, g_T_dyn,
  rt, correct``.
- ``sub-XX_roi-XX_mode-signed_dog-dyn-v3-target-sharedSigma-perTrial-fit.pkl``
  Pickle with the full ``fit_pars`` DataFrame, ``r2``, shared-pars dict,
  and the trial dataframe. Useful for downstream stability/RT analyses.

Usage
-----
``python -m retsupp.modeling.fit_dog_dyn_v3_per_trial 2 --roi V1``
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from nilearn import input_data
from tqdm import tqdm

from braincoder.hrf import SPMHRFModel
from braincoder.optimize import ParameterFitter
from retsupp.modeling.local_models import (
    DoGDynamicAttentionFieldPRF2DWithHRF_v3_target_sharedSigma_perTrial,
)
from retsupp.modeling.fit_dog_dynamic_af_braincoder import (
    CONDITIONS,
    get_ring_positions,
    select_roi_voxels,
)
from retsupp.utils.data import Subject


# Channel order MUST stay in sync with CONDITIONS.
LOC_TO_CHANNEL = {1.0: 0, 3.0: 1, 5.0: 2, 7.0: 3}


def build_data_and_per_trial(sub: Subject, resolution: int = 50,
                              grid_radius: float = 5.0,
                              max_distractor_duration: float = 1.5,
                              max_target_duration: float = 1.5):
    """Load BOLD + paradigm + per-trial pulse tensors and trial dataframe.

    Returns
    -------
    bold_df : pd.DataFrame
        (T, V) BOLD across runs.
    paradigm_full : ndarray (T, G)
        Full distractor-painted paradigm.
    condition_indicator : ndarray (T, n_C)
        Same as the parent loader: per-TR HP one-hot for the run.
    dynamic_indicator : ndarray (T, n_C)
        Per-TR per-ring distractor on-fraction (kept for parity / sanity
        checks; the per-trial model also uses dyn_pulse_per_trial).
    target_indicator : ndarray (T, n_C)
        Per-TR per-ring target on-fraction.
    grid_coords : ndarray (G, 2)
    masker : NiftiMasker
    dyn_pulse_per_trial : ndarray (T, n_trials)
        Per-TR distractor pulse on-fraction for trial ``k`` (any ring;
        the trial's ring is encoded in ``trial_ring_idx``).
    tgt_pulse_per_trial : ndarray (T, n_trials)
    trial_ring_idx : ndarray (n_trials,)
        Distractor ring index ∈ {0,1,2,3,-1}; -1 if no distractor.
    trial_target_ring_idx : ndarray (n_trials,)
    trial_is_hp : ndarray (n_trials,)
        1.0 if trial's distractor ring matches the run's HP, else 0.0.
        For no-distractor trials, 0.0 (they go to LP slot but their
        pulse is zero so it doesn't matter).
    trials_df : pd.DataFrame
        One row per trial (length n_trials) with bookkeeping columns:
        subject, roi (filled later), trial_idx, session, run,
        trial_within_run, onset_global (TR-time of target onset),
        target_location, distractor_location, hp_location,
        is_hp_distractor, rt, correct.
    """
    bold_mask = sub.get_bold_mask()
    masker = input_data.NiftiMasker(mask_img=bold_mask)

    gx, gy = sub.get_extended_grid_coordinates(
        resolution=resolution, session=1, run=1, grid_radius=grid_radius,
    )
    grid_coords = np.stack(
        (gx.ravel(), gy.ravel()), axis=1,
    ).astype(np.float32)

    hp_per_run = sub.get_hpd_locations()
    print('HP per run:', hp_per_run)

    # We pre-compute T_total (sum of n_volumes across kept runs) so we
    # can size the per-trial pulse tensors. We'll fill them while we
    # iterate. n_trials is unknown until we count target events.
    bold_chunks = []
    paradigm_chunks = []
    cond_indicator_chunks = []
    dyn_indicator_chunks = []
    tgt_indicator_chunks = []

    # Per-trial bookkeeping accumulators.
    trial_records = []  # list of dict
    # Per-trial pulse rows: list of (global_t_starts, global_t_ends, ring_idx)
    # and same for target. We fill them after we know T_total.
    dyn_pulse_events = []  # list of (trial_idx, t_on_global, t_off_global, ring)
    tgt_pulse_events = []  # same shape

    masker.fit(bold_mask)
    session_runs = [(s, r) for s in [1, 2] for r in sub.get_runs(s)]

    t_offset = 0       # running offset (in TR units)
    trial_idx = 0      # global trial counter

    for session, run in tqdm(session_runs, desc='Loading runs'):
        hp = hp_per_run[(session, run)]
        if hp not in CONDITIONS:
            print(f'  ses-{session}_run-{run}: HP={hp!r}, skipping')
            continue

        par_run = sub.get_stimulus_with_distractors(
            session=session, run=run, resolution=resolution,
            grid_radius=grid_radius,
        ).astype(np.float32)
        par_run_flat = par_run.reshape((par_run.shape[0], -1))
        n_T_run = par_run_flat.shape[0]
        tr = sub.get_tr(session=session, run=run)

        bold_fn = (
            sub.bids_folder / 'derivatives' / 'cleaned'
            / f'sub-{sub.subject_id:02d}' / f'ses-{session}' / 'func'
            / f'sub-{sub.subject_id:02d}_ses-{session}_task-search_'
              f'desc-cleaned_run-{run}_bold.nii.gz'
        )
        data = masker.transform(bold_fn).astype(np.float32)
        if data.shape[0] < n_T_run:
            pad = np.zeros((n_T_run - data.shape[0], data.shape[1]),
                           dtype=np.float32)
            data = np.vstack([data, pad])
        elif data.shape[0] > n_T_run:
            data = data[:n_T_run]

        hp_idx = CONDITIONS.index(hp)
        cond_indicator = np.zeros((n_T_run, len(CONDITIONS)),
                                   dtype=np.float32)
        cond_indicator[:, hp_idx] = 1.0

        dyn_indicator = sub.get_dynamic_indicator(
            session=session, run=run,
            max_distractor_duration=max_distractor_duration,
        ).astype(np.float32)
        if dyn_indicator.shape[0] < n_T_run:
            pad = np.zeros((n_T_run - dyn_indicator.shape[0], 4),
                           dtype=np.float32)
            dyn_indicator = np.vstack([dyn_indicator, pad])
        elif dyn_indicator.shape[0] > n_T_run:
            dyn_indicator = dyn_indicator[:n_T_run]

        tgt_indicator = sub.get_target_indicator(
            session=session, run=run,
            max_target_duration=max_target_duration,
        ).astype(np.float32)
        if tgt_indicator.shape[0] < n_T_run:
            pad = np.zeros((n_T_run - tgt_indicator.shape[0], 4),
                           dtype=np.float32)
            tgt_indicator = np.vstack([tgt_indicator, pad])
        elif tgt_indicator.shape[0] > n_T_run:
            tgt_indicator = tgt_indicator[:n_T_run]

        bold_chunks.append(data)
        paradigm_chunks.append(par_run_flat)
        cond_indicator_chunks.append(cond_indicator)
        dyn_indicator_chunks.append(dyn_indicator)
        tgt_indicator_chunks.append(tgt_indicator)

        # --- Walk trial events and record per-trial pulses. ---
        onsets = sub.get_onsets(session=session, run=run)
        targets = onsets[onsets['event_type'] == 'target'].sort_values('onset')
        feedbacks = onsets[onsets['event_type'] == 'feedback'].sort_values('onset')
        feedbacks_by_trial = {int(r['trial_nr']): r for _, r in feedbacks.iterrows()}

        for trial_within_run, (_, trial) in enumerate(targets.iterrows()):
            target_loc = trial.get('target_location', np.nan)
            distractor_loc = trial.get('distractor_location', np.nan)

            # Determine HP-distractor flag.
            is_hp_distractor = 0.0
            ring_idx = -1
            if (not pd.isna(distractor_loc)) and (distractor_loc in LOC_TO_CHANNEL):
                ring_idx = LOC_TO_CHANNEL[distractor_loc]
                if ring_idx == hp_idx:
                    is_hp_distractor = 1.0

            tgt_ring_idx = -1
            if (not pd.isna(target_loc)) and (target_loc in LOC_TO_CHANNEL):
                tgt_ring_idx = LOC_TO_CHANNEL[target_loc]

            # Pulse window (target onset → next feedback or +1.5s cap).
            t_on = float(trial['onset'])
            after_fb = feedbacks[feedbacks['onset'] > t_on]
            if len(after_fb):
                t_off = float(after_fb.iloc[0]['onset'])
            else:
                t_off = t_on + max_distractor_duration
            t_off_dist = min(t_off, t_on + max_distractor_duration)
            t_off_tgt = min(t_off, t_on + max_target_duration)

            # Distractor pulse only if ring is valid.
            if ring_idx >= 0:
                dyn_pulse_events.append((trial_idx, t_on, t_off_dist, ring_idx))
            if tgt_ring_idx >= 0:
                tgt_pulse_events.append((trial_idx, t_on, t_off_tgt, tgt_ring_idx))

            # RT/correct from the matching feedback row.
            tnr = int(trial['trial_nr'])
            fb = feedbacks_by_trial.get(tnr)
            rt = float(fb['rt']) if fb is not None and not pd.isna(fb.get('rt', np.nan)) else np.nan
            correct = fb.get('correct', np.nan) if fb is not None else np.nan
            if isinstance(correct, np.bool_):
                correct = bool(correct)

            trial_records.append(dict(
                trial_idx=trial_idx,
                session=session, run=run,
                trial_within_run=trial_within_run,
                target_location=target_loc,
                distractor_location=distractor_loc,
                hp_location=hp,
                is_hp_distractor=is_hp_distractor,
                rt=rt,
                correct=correct,
                onset_run=t_on,
                tr=tr,
                trial_ring_idx=ring_idx,
                trial_target_ring_idx=tgt_ring_idx,
            ))
            trial_idx += 1

        t_offset += n_T_run

    # --- Stack into global tensors. ---
    bold = np.vstack(bold_chunks)
    paradigm_full = np.vstack(paradigm_chunks)
    condition_indicator = np.vstack(cond_indicator_chunks)
    dynamic_indicator = np.vstack(dyn_indicator_chunks)
    target_indicator = np.vstack(tgt_indicator_chunks)
    n_T_total = bold.shape[0]
    n_trials = len(trial_records)
    print(f'Total: T={n_T_total}, V={bold.shape[1]}, n_trials={n_trials}')

    # --- Build per-trial pulse tensors. ---
    # Each trial pulse lives within one run; we compute the per-TR
    # overlap fraction in run-local seconds, then place it into the
    # corresponding rows of the global (T_total, n_trials) tensor using
    # run_starts (the row offset of each run).
    run_starts = {}        # (session, run) -> first global TR row index
    cursor = 0
    for session, run in session_runs:
        hp = hp_per_run.get((session, run))
        if hp not in CONDITIONS:
            continue
        n_T_run = sub.get_n_volumes(session=session, run=run)
        run_starts[(session, run)] = cursor
        cursor += n_T_run

    dyn_pulse_per_trial = np.zeros((n_T_total, n_trials), dtype=np.float32)
    tgt_pulse_per_trial = np.zeros((n_T_total, n_trials), dtype=np.float32)

    def fill_pulse(events, target_arr):
        for (k, t_on, t_off, ring) in events:
            # Find the run/session this trial belongs to via the trial_records list.
            rec = trial_records[k]
            session, run = rec['session'], rec['run']
            tr = rec['tr']
            n_T_run = sub.get_n_volumes(session=session, run=run)
            t_starts = np.arange(n_T_run) * tr
            t_ends = t_starts + tr
            overlap = np.clip(
                np.minimum(t_ends, t_off) - np.maximum(t_starts, t_on),
                0.0, None,
            ) / tr  # [0, 1]
            row_offset = run_starts[(session, run)]
            target_arr[row_offset:row_offset + n_T_run, k] += overlap.astype(np.float32)

    fill_pulse(dyn_pulse_events, dyn_pulse_per_trial)
    fill_pulse(tgt_pulse_events, tgt_pulse_per_trial)

    # Sanity: sum of per-trial dyn pulses across trials should equal the
    # original dynamic_indicator summed across rings (within ε).
    sum_dyn_per_trial = dyn_pulse_per_trial.sum(axis=1)
    sum_dynind = dynamic_indicator.sum(axis=1)
    # Each trial's pulse contributes to exactly one ring, so the sum
    # over trials at each TR equals the sum over rings of the dynamic
    # indicator.
    diff = float(np.max(np.abs(sum_dyn_per_trial - sum_dynind)))
    print(f'sanity: max |sum_dyn_per_trial - sum_ring(dynamic_indicator)| = {diff:.4f}')

    # Trial-level static vectors.
    trial_ring_idx = np.array([r['trial_ring_idx'] for r in trial_records],
                              dtype=np.int32)
    trial_target_ring_idx = np.array(
        [r['trial_target_ring_idx'] for r in trial_records], dtype=np.int32)
    trial_is_hp = np.array([r['is_hp_distractor'] for r in trial_records],
                           dtype=np.float32)

    trials_df = pd.DataFrame(trial_records)

    return (
        pd.DataFrame(bold),
        paradigm_full,
        condition_indicator,
        dynamic_indicator,
        target_indicator,
        grid_coords,
        masker,
        dyn_pulse_per_trial,
        tgt_pulse_per_trial,
        trial_ring_idx,
        trial_target_ring_idx,
        trial_is_hp,
        trials_df,
    )


def main(subject: int, bids_folder: str = '/data/ds-retsupp',
         roi: str = 'V1',
         resolution: int = 50,
         max_n_iterations: int = 1500,
         r2_thr: float = 0.05,
         model_label: int = 4,
         max_voxels: int | None = 200,
         mode: str = 'signed',
         learning_rate: float = 0.01,
         grid_radius: float = 5.0,
         output_subdir: str | None = None,
         sigma_af_init: float = 2.0,
         sigma_dyn_init: float = 2.0):

    if mode != 'signed':
        # The per-trial model is hard-coded to signed (per-trial gains
        # need to be free-signed so the optimizer can find +/− on each
        # trial). Refuse to silently change.
        raise ValueError("Only mode='signed' is supported by the per-trial fit.")

    bids_folder = Path(bids_folder)
    sub = Subject(subject, bids_folder)
    if output_subdir is None:
        output_subdir = ('af_prf_joint_dynamic_v3_dog_with_target_'
                         'sharedSigma_perTrial')
    out_dir = bids_folder / 'derivatives' / output_subdir / f'sub-{subject:02d}'
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'== sub-{subject:02d} | roi={roi} | mode={mode} | per-trial gains ==')

    # 1) Load BOLD + paradigm + indicators + per-trial pulses + trials.
    (bold_df, paradigm, condition_indicator, dynamic_indicator,
     target_indicator, grid_coords, masker,
     dyn_pulse_per_trial, tgt_pulse_per_trial,
     trial_ring_idx, trial_target_ring_idx, trial_is_hp,
     trials_df) = build_data_and_per_trial(
        sub, resolution=resolution, grid_radius=grid_radius,
    )
    n_trials = len(trials_df)

    # 2) Restrict to ROI voxels.
    prf_pars = sub.get_prf_parameters_volume(model=model_label,
                                              return_images=False)
    if not isinstance(prf_pars, pd.DataFrame):
        prf_pars = pd.DataFrame(prf_pars)
    voxel_mask = select_roi_voxels(sub, roi, prf_pars, r2_thr=r2_thr)
    # Also enforce sd >= 0.5 (phantom-voxel filter from the brief).
    if 'sd' in prf_pars.columns:
        voxel_mask = voxel_mask & (prf_pars['sd'].values >= 0.5)
    print(f'ROI {roi} | r2>{r2_thr} & sd>=0.5: {voxel_mask.sum()} voxels')
    if voxel_mask.sum() == 0:
        raise RuntimeError(f'No voxels survive: ROI={roi}.')

    if max_voxels is not None and voxel_mask.sum() > max_voxels:
        order = np.argsort(prf_pars['r2'].values)[::-1]
        keep = np.zeros_like(voxel_mask)
        ranked = [v for v in order if voxel_mask[v]][:max_voxels]
        keep[ranked] = True
        voxel_mask = keep
        print(f'  -> top {max_voxels} voxels by r²')

    bold_sub = bold_df.loc[:, voxel_mask].copy()

    # 3) Initial parameters (from model 4 DoG mean fit).
    init_cols = ['x', 'y', 'sd', 'baseline', 'amplitude',
                 'srf_amplitude', 'srf_size']
    missing = [c for c in init_cols if c not in prf_pars.columns]
    if missing:
        raise RuntimeError(
            f'Mean model {model_label} is missing DoG params {missing}.')
    init_pars = prf_pars.loc[voxel_mask, init_cols].copy()

    init_pars['sigma_AF'] = sigma_af_init
    init_pars['g_HP'] = 0.0
    init_pars['g_LP'] = 0.0
    init_pars['sigma_dyn'] = sigma_dyn_init
    init_pars['g_HP_dyn'] = 0.0   # zero-clamped in forward
    init_pars['g_LP_dyn'] = 0.0
    init_pars['g_T_dyn'] = 0.0
    init_pars['sigma_T_dyn'] = sigma_dyn_init  # tied -> sigma_dyn

    # Per-trial gain inits (3 * n_trials zeros).
    for k in range(n_trials):
        init_pars[f'g_HP_dyn_t{k:04d}'] = 0.0
    for k in range(n_trials):
        init_pars[f'g_LP_dyn_t{k:04d}'] = 0.0
    for k in range(n_trials):
        init_pars[f'g_T_dyn_t{k:04d}'] = 0.0

    # 4) Build the per-trial model.
    ring_positions = get_ring_positions()
    print('Ring positions:\n', ring_positions)
    tr = sub.get_tr(session=1, run=1)
    hrf_model = SPMHRFModel(tr=tr, delay=4.5, dispersion=0.75)

    model = DoGDynamicAttentionFieldPRF2DWithHRF_v3_target_sharedSigma_perTrial(
        grid_coordinates=grid_coords,
        paradigm=paradigm,
        hrf_model=hrf_model,
        condition_indicator=condition_indicator,
        dynamic_indicator=dynamic_indicator,
        target_indicator=target_indicator,
        ring_positions=ring_positions,
        mode=mode,
        dyn_pulse_per_trial=dyn_pulse_per_trial,
        tgt_pulse_per_trial=tgt_pulse_per_trial,
        trial_ring_idx=trial_ring_idx,
        trial_target_ring_idx=trial_target_ring_idx,
        trial_is_hp=trial_is_hp,
    )

    fitter = ParameterFitter(model, bold_sub, paradigm)

    # 5) Build shared_pars list.
    shared_pars = (
        ['sigma_AF', 'g_HP', 'g_LP',
         'sigma_dyn', 'g_HP_dyn', 'g_LP_dyn',
         'g_T_dyn', 'sigma_T_dyn']
        + [f'g_HP_dyn_t{k:04d}' for k in range(n_trials)]
        + [f'g_LP_dyn_t{k:04d}' for k in range(n_trials)]
        + [f'g_T_dyn_t{k:04d}'  for k in range(n_trials)]
    )

    # 6) Refine baseline/amplitude.
    refined_pars = fitter.refine_baseline_and_amplitude(init_pars, l2_alpha=1e-3)

    # 7) Joint fit.
    fit_pars = fitter.fit(
        init_pars=refined_pars,
        max_n_iterations=max_n_iterations,
        shared_pars=shared_pars,
        learning_rate=learning_rate,
    )

    r2 = fitter.get_rsq(fit_pars) if hasattr(fitter, 'get_rsq') else fitter.r2
    print(f'Mean R²: {np.nanmean(r2):.4f}')
    print('Shared (non-trial) parameters from voxel 0:')
    print(fit_pars[['sigma_AF', 'g_HP', 'g_LP',
                     'sigma_dyn', 'g_HP_dyn', 'g_LP_dyn',
                     'g_T_dyn', 'sigma_T_dyn']].iloc[0])

    fit_pars['r2'] = r2.values if hasattr(r2, 'values') else r2

    # 8) Build the long trials TSV.
    # Read the per-trial gains from voxel 0 (they are shared across voxels).
    g_HP_trial = np.array([fit_pars[f'g_HP_dyn_t{k:04d}'].iloc[0]
                            for k in range(n_trials)], dtype=np.float32)
    g_LP_trial = np.array([fit_pars[f'g_LP_dyn_t{k:04d}'].iloc[0]
                            for k in range(n_trials)], dtype=np.float32)
    g_T_trial = np.array([fit_pars[f'g_T_dyn_t{k:04d}'].iloc[0]
                           for k in range(n_trials)], dtype=np.float32)

    trials_long = trials_df.copy()
    trials_long['subject'] = subject
    trials_long['roi'] = roi
    trials_long['g_HP_dyn'] = g_HP_trial
    trials_long['g_LP_dyn'] = g_LP_trial
    trials_long['g_T_dyn'] = g_T_trial
    # The "active" dyn gain per trial: HP for HP-distractor trials, LP otherwise.
    trials_long['g_dyn_active'] = np.where(
        trials_long['is_hp_distractor'].astype(bool),
        trials_long['g_HP_dyn'],
        trials_long['g_LP_dyn'],
    )

    # Aggregate stats per (subject, ROI).
    agg = pd.DataFrame([{
        'subject': subject, 'roi': roi,
        'n_trials': int(n_trials),
        'mean_r2': float(np.nanmean(r2)),
        'g_HP_dyn_mean': float(np.nanmean(g_HP_trial[trial_is_hp.astype(bool)])),
        'g_HP_dyn_std':  float(np.nanstd(g_HP_trial[trial_is_hp.astype(bool)])),
        'g_LP_dyn_mean': float(np.nanmean(g_LP_trial[~trial_is_hp.astype(bool)])),
        'g_LP_dyn_std':  float(np.nanstd(g_LP_trial[~trial_is_hp.astype(bool)])),
        'g_T_dyn_mean': float(np.nanmean(g_T_trial)),
        'g_T_dyn_std':  float(np.nanstd(g_T_trial)),
    }])

    # 9) Save outputs.
    tag = 'dog-dyn-v3-target-sharedSigma-perTrial'
    out_tsv = out_dir / (
        f'sub-{subject:02d}_roi-{roi}_mode-{mode}_{tag}-af-prf-pars.tsv')
    fit_pars.to_csv(out_tsv, sep='\t')
    out_trials = out_dir / (
        f'sub-{subject:02d}_roi-{roi}_mode-{mode}_{tag}-trials.tsv')
    trials_long.to_csv(out_trials, sep='\t', index=False)
    out_agg = out_dir / (
        f'sub-{subject:02d}_roi-{roi}_mode-{mode}_{tag}-agg.tsv')
    agg.to_csv(out_agg, sep='\t', index=False)
    out_pkl = out_dir / (
        f'sub-{subject:02d}_roi-{roi}_mode-{mode}_{tag}-fit.pkl')
    with open(out_pkl, 'wb') as f:
        pickle.dump({
            'fit_pars': fit_pars,
            'r2': r2,
            'shared_pars_scalar': fit_pars[[
                'sigma_AF', 'g_HP', 'g_LP', 'sigma_dyn',
                'g_HP_dyn', 'g_LP_dyn', 'g_T_dyn', 'sigma_T_dyn',
            ]].iloc[0].to_dict(),
            'g_HP_dyn_trial': g_HP_trial,
            'g_LP_dyn_trial': g_LP_trial,
            'g_T_dyn_trial': g_T_trial,
            'trial_is_hp': trial_is_hp,
            'trials_df': trials_long,
            'voxel_mask_indices': np.where(voxel_mask)[0],
            'mode': mode, 'roi': roi, 'subject': subject,
            'resolution': resolution,
            'grid_radius': grid_radius,
            'model_label_init': model_label,
        }, f)
    print(f'Saved: {out_tsv}')
    print(f'Saved: {out_trials}')
    print(f'Saved: {out_agg}')
    print(f'Saved: {out_pkl}')

    return fit_pars, r2, trials_long


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('subject', type=int)
    parser.add_argument('--bids-folder', default='/data/ds-retsupp')
    parser.add_argument('--roi', default='V1')
    parser.add_argument('--resolution', type=int, default=50)
    parser.add_argument('--max-n-iterations', type=int, default=1500)
    parser.add_argument('--r2-thr', type=float, default=0.05)
    parser.add_argument('--model-label', type=int, default=4)
    parser.add_argument('--max-voxels', type=int, default=200,
                        help='Cap on voxels (default 200; per-trial fit '
                             'is heavier than the scalar fit).')
    parser.add_argument('--mode', choices=['signed'], default='signed')
    parser.add_argument('--learning-rate', type=float, default=0.01)
    parser.add_argument('--grid-radius', type=float, default=5.0)
    parser.add_argument('--output-subdir', default=None)
    parser.add_argument('--sigma-af-init', type=float, default=2.0)
    parser.add_argument('--sigma-dyn-init', type=float, default=2.0)
    args = parser.parse_args()

    main(
        args.subject,
        bids_folder=args.bids_folder,
        roi=args.roi,
        resolution=args.resolution,
        max_n_iterations=args.max_n_iterations,
        r2_thr=args.r2_thr,
        model_label=args.model_label,
        max_voxels=None if args.max_voxels == 0 else args.max_voxels,
        mode=args.mode,
        learning_rate=args.learning_rate,
        grid_radius=args.grid_radius,
        output_subdir=args.output_subdir,
        sigma_af_init=args.sigma_af_init,
        sigma_dyn_init=args.sigma_dyn_init,
    )
