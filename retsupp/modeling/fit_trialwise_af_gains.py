"""Trial-wise and RT-binned phasic AF gains on top of the canonical AF fit.

Takes the canonical joint AF + DoG-PRF fit (v3 + target + sharedSigma,
``pSig0.5``) as fixed and re-estimates ONLY the phasic gains, either per
trial (ridge) or per RT bin (OLS).

Why this works without refitting
--------------------------------
With PRFs, HRFs, σ_AF / σ_dyn and the sustained gains fixed, the
canonical model's BOLD prediction is LINEAR in the phasic gains::

    bold_v(t) = P0_v(t) + amp_v · h_v * [ Σ_k g_k · u_k(t) · D_{ℓ_k, v}(t) ]

where ``P0`` is the prediction with all phasic gains set to zero,
``u_k(t)`` is trial k's on-fraction per TR, ``ℓ_k`` its ring and
``D_{ℓ,v}(t) = sign · Σ_g paradigm(t, g) · A_ℓ(g) · rf_v(g)`` the
AF-weighted drive at ring ℓ. Each trial contributes three regressors:

- ``T``: target gain at the target's ring (canonical ``g_T_dyn``).
- ``D``: distractor gain at the distractor's ring (canonical
  ``g_HP_dyn`` / ``g_LP_dyn``); absent on no-distractor trials.
- ``G``: a global gain on the whole stimulus drive during the trial
  (no canonical counterpart, prior 0). Absorbs non-spatial per-trial
  amplitude changes (arousal, time-on-task), so ``T`` / ``D`` index
  the spatially specific part.

Per-trial: ridge toward the canonical scalar gains,
``g = (X'WX + αΛ)⁻¹ (X'Wr + αΛ g0)`` with ``Λ = diag(X'WX)``, swept
over α. α is judged by voxel split-half cross-validation (fit on half
the voxels, score held-out SSE on the other half); ``α → ∞`` recovers
the canonical scalar model.

RT bins: the per-trial regressors are summed within (kind × RT bin)
and the bin gains estimated by weighted OLS. Bins are formed within
(session, run, trial type) on correct trials with valid RT; error /
miss / invalid-RT trials go to their own ``other`` bin. The same OLS
also runs with one bin per run (``run`` scheme, all trials) for
run-level brain–behaviour analyses, and with inter-trial history bins:
``dprime`` (distractor at the previous trial's distractor location:
``repeat`` / ``switch``) and ``tprime`` (target ``repeat`` /
``at_prev_dist`` / ``other``).

With ``--hp-neutral``, a fifth kind, ``H``, is the gain at the run's HP
location when it holds a neutral item on the current trial (neither target
nor distractor; prior 0): trial-to-trial gain at HP without a singleton
response in it. ``P`` is then left out on trials where the previous
distractor location is the HP location, so the two never share a regressor.

A fourth regressor kind, ``P``, is the gain at the PREVIOUS trial's
distractor location when it holds a neutral item on the current trial
(prior 0; no canonical counterpart) — intertrial inhibition at a
location that is empty of singletons now.

Trial pulses are the AF fit's (target onset → feedback, capped at
1.5 s). Feedback arrives at a fixed 1.5 s after target onset, so the
pulse width does not depend on RT.

Parametric model: instead of bins, each kind's gain is
``g + Σ_m β_m · m_k`` with trial modulators z(log RT) (within session ×
run × trial type; 0 for error / invalid trials) and inter-trial priming
flags (distractor repeat, target repeat, target at the previous
distractor location), all centred within kind. Uses every trial.

Run-wise joint model: the sustained AF term is also linear in
(g_HP, g_LP), so per run we estimate sustained HP / LP gains together
with that run's phasic T / D_HP / D_LP (/ G) gains, by weighted OLS on
the residual from the model with ALL AF gains zeroed. The sustained
reconstruction is validated against the model (``sustained_recon_rel_err``;
non-zero only if the model's ``max(mod, 0)`` clamp is active).

Outputs under ``derivatives/af_trialwise_gains/sub-XX/``:
``sub-XX_roi-R_trials.tsv`` (per-trial gains per α),
``sub-XX_roi-R_cv.tsv`` (held-out SSE + split-half reliability per α),
``sub-XX_roi-R_bins.tsv`` (binned gains),
``sub-XX_roi-R_parametric.tsv``, ``sub-XX_roi-R_runfit.tsv`` and
``sub-XX_roi-R_check.tsv`` (reconstruction sanity checks).

Usage
-----
``python -m retsupp.modeling.fit_trialwise_af_gains 5 --rois V1 V3``
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from braincoder.hrf import SPMHRFModel
from retsupp.modeling.fit_dog_dynamic_af_braincoder import (
    CONDITIONS,
    build_data_and_paradigm,
    get_ring_positions,
)
from retsupp.modeling.local_models import (
    DoGDynamicAttentionFieldPRF2DWithHRF_v3_target_sharedSigma,
)
from retsupp.utils.data import Subject

ROIS = ['V1', 'V2', 'V3', 'V3AB', 'hV4', 'LO', 'TO', 'VO', 'IPS', 'SPL1', 'FEF']
AF_SUBDIR = 'af_prf_joint_dynamic_v3_dog_with_target_sharedSigma_pSig0.5'
AF_TAG = 'dog-dyn-v3-target-sharedSigma'
LOC_TO_CHANNEL = {1.0: 0, 3.0: 1, 5.0: 2, 7.0: 3}
MAX_PULSE = 1.5
ALPHAS = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
RT_MIN, RT_MAX = 0.1, 2.0


# ---------------------------------------------------------------------------
# Trial table + per-trial pulses
# ---------------------------------------------------------------------------
def _overlap(t_on, t_off, n_volumes, tr):
    starts = np.arange(n_volumes) * tr
    return np.clip(np.minimum(starts + tr, t_off) - np.maximum(starts, t_on),
                   0.0, None) / tr


def build_trials(sub: Subject):
    """One row per trial plus the (T_total, n_trials) pulse matrix.

    Run order / skipping mirrors ``build_data_and_paradigm``.
    """
    hp_per_run = sub.get_hpd_locations()
    records, pulses = [], []
    offset = 0
    for session in [1, 2]:
        for run in sub.get_runs(session):
            hp = hp_per_run[(session, run)]
            if hp not in CONDITIONS:
                continue
            hp_ch = CONDITIONS.index(hp)
            tr = sub.get_tr(session, run)
            n_vol = sub.get_n_volumes(session, run)
            onsets = sub.get_onsets(session, run)
            targets = onsets[onsets['event_type'] == 'target'].sort_values('onset')
            feedback = onsets[onsets['event_type'] == 'feedback'].sort_values('onset')
            fb_by_trial = {int(r['trial_nr']): r for _, r in feedback.iterrows()}

            for _, trial in targets.iterrows():
                t_on = float(trial['onset'])
                after = feedback[feedback['onset'] > t_on]
                t_off = (float(after.iloc[0]['onset']) if len(after)
                         else t_on + MAX_PULSE)
                t_off = min(t_off, t_on + MAX_PULSE)
                pulses.append((offset, _overlap(t_on, t_off, n_vol, tr)))

                d_loc = trial['distractor_location']
                t_loc = trial['target_location']
                d_ch = LOC_TO_CHANNEL.get(d_loc, -1) if not pd.isna(d_loc) else -1
                t_ch = LOC_TO_CHANNEL.get(t_loc, -1) if not pd.isna(t_loc) else -1
                ttype = ('none' if d_ch < 0 else
                         ('HP' if d_ch == hp_ch else 'LP'))
                fb = fb_by_trial.get(int(trial['trial_nr']))
                rt = (float(fb['rt']) if fb is not None
                      and not pd.isna(fb.get('rt', np.nan)) else np.nan)
                correct = (fb.get('correct', np.nan) if fb is not None
                           else np.nan)
                records.append(dict(
                    session=session, run=run, trial_nr=int(trial['trial_nr']),
                    onset=t_on, hp_location=hp,
                    distractor_location=d_loc, target_location=t_loc,
                    dist_ch=d_ch, target_ch=t_ch, hp_ch=hp_ch, trial_type=ttype,
                    rt=rt, correct=correct,
                ))
            offset += n_vol

    trials = pd.DataFrame(records)
    trials.insert(0, 'trial_idx', np.arange(len(trials)))
    trials['correct'] = trials['correct'].map(
        lambda c: bool(c) if isinstance(c, (bool, np.bool_)) else
        (str(c).lower() == 'true' if isinstance(c, str) else np.nan))
    trials['valid_rt'] = ((trials['correct'] == True)  # noqa: E712
                          & trials['rt'].between(RT_MIN, RT_MAX))

    u = np.zeros((offset, len(trials)), dtype=np.float32)
    for k, (start, ov) in enumerate(pulses):
        u[start:start + len(ov), k] = ov
    return trials, u, offset


def add_rt_bins(trials: pd.DataFrame) -> pd.DataFrame:
    """Median-split (``rt2``) and tercile (``rt3``) labels within
    (session, run, trial_type); non-valid trials → ``other``."""
    trials = trials.copy()
    for scheme, labels in [('rt2', ['fast', 'slow']),
                           ('rt3', ['fast', 'mid', 'slow'])]:
        col = pd.Series('other', index=trials.index, dtype=object)
        valid = trials[trials['valid_rt']]
        for _, grp in valid.groupby(['session', 'run', 'trial_type']):
            if len(grp) < len(labels):
                continue
            ranks = grp['rt'].rank(method='first')
            col.loc[grp.index] = pd.qcut(ranks, len(labels),
                                         labels=labels).astype(str)
        trials[f'bin_{scheme}'] = col
    trials['bin_all'] = 'all'
    trials['bin_run'] = ('ses-' + trials['session'].astype(str)
                         + '_run-' + trials['run'].astype(str))

    # Inter-trial history (previous trial in the same run).
    prev = trials.groupby(['session', 'run'])[['dist_ch', 'target_ch']].shift(1)
    trials['prev_dist_ch'] = prev['dist_ch'].fillna(-1).astype(int)
    trials['prev_target_ch'] = prev['target_ch'].fillna(-1).astype(int)
    d, t, pd_ = trials['dist_ch'], trials['target_ch'], trials['prev_dist_ch']
    trials['bin_dprime'] = np.where(
        (d >= 0) & (pd_ >= 0), np.where(d == pd_, 'repeat', 'switch'), 'na')
    trials['bin_tprime'] = np.where(
        (t >= 0) & (t == trials['prev_target_ch']), 'repeat',
        np.where((t >= 0) & (t == pd_), 'at_prev_dist', 'other'))

    # Continuous modulators for the parametric model.
    logrt = np.log(trials['rt'].where(trials['valid_rt']))
    z = logrt.groupby([trials['session'], trials['run'],
                       trials['trial_type']]).transform(
        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else x * 0.0)
    trials['mod_rt'] = z.fillna(0.0)
    trials['mod_dist_repeat'] = ((d >= 0) & (d == pd_)).astype(float)
    trials['mod_target_repeat'] = (
        (t >= 0) & (t == trials['prev_target_ch'])).astype(float)
    trials['mod_target_at_prev_dist'] = ((t >= 0) & (t == pd_)).astype(float)
    if RT_MODULATOR == 'residual':
        trials = residual_rt(trials, logrt)
    return trials


def residual_rt(trials, logrt):
    """--rt-modulator residual (notes/rt_power_plan_2026-11-13.md, step 1): log RT of the participant's correct trials
    regressed (OLS) on distractor condition (HP / LP / absent), target location relative to HP (HP / LP), target repeat,
    distractor repeat, trial number within run (linear and quadratic) and run (one level per session × run, which
    includes session); the residual, z-scored within participant, replaces ``mod_rt`` (0 on other trials). Also its
    quintiles within participant as ``bin_rq5`` (other trials: ``other``)."""
    trials = trials.copy()
    ok = trials['valid_rt'].to_numpy(bool)
    pos = trials.groupby(['session', 'run']).cumcount().astype(float)
    pos = (pos - pos.mean()) / pos.std()
    X = pd.concat([
        pd.get_dummies(trials['trial_type'], prefix='dist', drop_first=True, dtype=float),
        pd.Series((trials['target_ch'] == trials['hp_ch']).astype(float), name='target_at_hp'),
        trials[['mod_target_repeat', 'mod_dist_repeat']].astype(float),
        pos.rename('trial'), (pos ** 2).rename('trial2'),
        pd.get_dummies('ses-' + trials['session'].astype(str) + '_run-' + trials['run'].astype(str),
                       prefix='run', drop_first=True, dtype=float),
    ], axis=1)
    X.insert(0, 'const', 1.0)
    Xo, yo = X.to_numpy(float)[ok], logrt.to_numpy(float)[ok]
    keep = Xo.std(0) > 0
    keep[0] = True
    beta, *_ = np.linalg.lstsq(Xo[:, keep], yo, rcond=None)
    res = yo - Xo[:, keep] @ beta
    z = (res - res.mean()) / res.std()
    r2 = 1 - res.var() / yo.var()
    print(f'Residual RT: {ok.sum()} correct trials, {keep.sum()} design columns, R² of the design {r2:.3f}', flush=True)
    trials['mod_rt'] = 0.0
    trials.loc[ok, 'mod_rt'] = z
    trials['bin_rq5'] = 'other'
    trials.loc[ok, 'bin_rq5'] = pd.qcut(pd.Series(z).rank(method='first'), 5,
                                        labels=[f'q{i}' for i in range(1, 6)]).astype(str).to_numpy()
    return trials


# ---------------------------------------------------------------------------
# Model pieces
# ---------------------------------------------------------------------------
def build_model(paradigm, grid_coords, cond_ind, dyn_ind, tgt_ind, tr):
    hrf_model = SPMHRFModel(tr=tr, delay=4.5, dispersion=0.75)
    return DoGDynamicAttentionFieldPRF2DWithHRF_v3_target_sharedSigma(
        grid_coordinates=grid_coords,
        paradigm=paradigm,
        hrf_model=hrf_model,
        flexible_hrf_parameters=True,
        condition_indicator=cond_ind,
        dynamic_indicator=dyn_ind,
        target_indicator=tgt_ind,
        ring_positions=get_ring_positions(),
        mode='signed',
    )


LADDER_ROOTS = ()   # set by --ladder-root: use the gain-model ladder's m7_full fits instead of the October fits
HRF_TABLE = None    # one HRF per subject x ROI (roi_hrf.load_table) for the ladder fits
VOXELS = 'ladder'   # --voxels signal: the signal voxels of each ROI instead of the ladder fit's own voxels
MAX_BOLD_SD = 20.0  # with --voxels signal, as in the final ladder fits
RT_MODULATOR = 'zlogrt'   # --rt-modulator residual: residual log RT (see residual_rt)
MIN_VOXELS = 0      # --min-voxels: leave out a subject x ROI whose ladder fit has fewer voxels (0: keep all)
SIGNAL_TABLE = 'af_prf_cv_ladder_nested/cv_ladder_per_voxel.tsv'
JOINT_STEM = 'dog_final2_stage-3'   # DoG fold fits with the joint per-ROI HRF (the HRF table's source)
GROUPS = {'early': ['V1', 'V2', 'V3'], 'dorsal': ['V3AB', 'IPS'],
          'occipitotemporal': ['hV4', 'LO', 'TO', 'VO']}
_CACHE = {}


class NoFit(FileNotFoundError):
    """This subject x ROI has nothing to fit (no base fit, or no voxels)."""


def signal_voxels(sub, roi):
    """Voxels whose held-out signal beats the model-free null (fold-mean cv_r2_null > 0), as in the LOSO tests."""
    if 'signal' not in _CACHE:
        t = pd.read_csv(Path(sub.bids_folder) / 'derivatives' / SIGNAL_TABLE, sep='\t',
                        usecols=['subject', 'roi', 'voxel_id', 'cv_r2_null'])
        t = t[t.subject == sub.subject_id].groupby(['roi', 'voxel_id']).cv_r2_null.mean()
        _CACHE['signal'] = t[t > 0].reset_index()
    t = _CACHE['signal']
    return np.sort(t.loc[t.roi == roi, 'voxel_id'].to_numpy())


def joint_prfs(sub):
    """Per-voxel DoG pRFs of the joint per-ROI-HRF fold fits, mean over the folds (index = BOLD-mask voxel)."""
    if 'joint' not in _CACHE:
        d = Path(sub.bids_folder) / 'derivatives' / 'prf_cv_roi_jointhrf' / f'sub-{sub.subject_id:02d}'
        folds = []
        for f in sorted(d.glob(f'sub-{sub.subject_id:02d}_fold-*_{JOINT_STEM}_pars.pkl.gz')):
            ids = pd.read_csv(str(f).replace('_pars.pkl.gz', '_cvr2.tsv.gz'), sep='\t').voxel_index.to_numpy()
            p = pd.read_pickle(f).reset_index(drop=True)[['x', 'y', 'sd', 'srf_amplitude', 'srf_size']]
            p.index = ids
            folds.append(p)
        if not folds:
            raise FileNotFoundError(f'no {JOINT_STEM} fold fits in {d}')
        _CACHE['joint'] = pd.concat(folds).groupby(level=0).mean().dropna()
        print(f'Joint-HRF DoG pRFs: {len(_CACHE["joint"])} voxels, {len(folds)} folds', flush=True)
    return _CACHE['joint']


def extend_to_signal_voxels(sub, roi, model, pars, fit_ids, bold, paradigm):
    """The signal voxels of the ROI with the m7_full gain-field terms.

    A voxel in the m7_full fit keeps its fitted pRF; another takes x, y, sd and the surround of the joint-HRF fold
    fits, and its baseline and amplitude by OLS on the m7_full prediction (linear in both). Voxels with BOLD SD above
    MAX_BOLD_SD, without a fold fit, or whose fit has R² <= 0 or amplitude <= 0 are left out.
    """
    sig = signal_voxels(sub, roi)
    sd = bold.std(0)
    n_sig = len(sig)
    sig = sig[sd[sig] <= MAX_BOLD_SD]
    n_sd = n_sig - len(sig)
    in_fit = np.isin(sig, fit_ids)
    jp = joint_prfs(sub)
    has_fold = np.isin(sig, jp.index)
    keep = in_fit | has_fold
    sig, in_fit = sig[keep], in_fit[keep]
    by_id = pars.set_index(pd.Index(np.asarray(fit_ids)))
    shared = pars.iloc[0]
    new = pd.DataFrame([shared.to_numpy()] * len(sig), columns=pars.columns)
    new.loc[in_fit, :] = by_id.loc[sig[in_fit]].to_numpy()
    add = ~in_fit
    for c in ('x', 'y', 'sd', 'srf_amplitude', 'srf_size'):
        new.loc[add, c] = jp.loc[sig[add], c].to_numpy()
    new.loc[add, 'baseline'] = 0.0
    new.loc[add, 'amplitude'] = 1.0
    new = new.astype(np.float32)
    info = dict(n_signal=n_sig, n_drop_sd=n_sd, n_in_fit=int(in_fit.sum()), n_no_fold=int((~keep).sum()),
                n_fit_not_signal=int((~np.isin(fit_ids, sig)).sum()))
    if add.any():
        p = np.asarray(model.predict(parameters=new.loc[add].reset_index(drop=True))).astype(np.float64)
        y = bold[:, sig[add]].astype(np.float64)
        pc, yc = p - p.mean(0), y - y.mean(0)
        a = (pc * yc).sum(0) / np.maximum((pc ** 2).sum(0), 1e-12)
        b = y.mean(0) - a * p.mean(0)
        r2 = 1 - ((y - (b + a * p)) ** 2).sum(0) / ((yc ** 2).sum(0))
        new.loc[add, 'amplitude'] = a.astype(np.float32)
        new.loc[add, 'baseline'] = b.astype(np.float32)
        # The model's prediction must be linear in baseline and amplitude.
        chk = np.asarray(model.predict(parameters=new.loc[add].reset_index(drop=True))).astype(np.float64)
        info['ab_linear_rel_err'] = float(np.abs(chk - (b + a * p)).max() / max(np.abs(chk).max(), 1e-12))
        good = np.ones(len(sig), dtype=bool)
        good[np.where(add)[0]] = (r2 > 0) & (a > 0)
        info.update(n_added=int(add.sum()), n_added_bad=int((~good).sum()),
                    r2_added_median=float(np.median(r2[(r2 > 0) & (a > 0)])) if ((r2 > 0) & (a > 0)).any() else np.nan)
        if info['ab_linear_rel_err'] > 1e-3:
            raise RuntimeError(f'Prediction not linear in baseline/amplitude: {info["ab_linear_rel_err"]:.2e}')
        sig, new = sig[good], new.loc[good].reset_index(drop=True)
    else:
        info.update(n_added=0, n_added_bad=0, r2_added_median=np.nan, ab_linear_rel_err=0.0)
    info['n_voxels_used'] = len(sig)
    print(f'{roi}: signal voxels ' + ', '.join(f'{k} {v}' for k, v in info.items()), flush=True)
    if len(sig) < 10:
        raise NoFit(f'{roi}: only {len(sig)} usable signal voxels')
    return sig, new, info


def load_ladder_fit(sub, roi, data):
    """The ladder's m7_full fit of this subject x ROI (first root holding it) and its model (items, fixed HRF)."""
    from retsupp.modeling.fit_af_prf_cv_v2 import build_model as build_items_model
    from retsupp.modeling.fit_cv_ladder_nested import sign_pattern_of
    from retsupp.modeling.roi_hrf import hrf_model as roi_hrf_model
    bold, paradigm, cond_ind, dyn_ind, grid, tgt_ind = data
    tail = Path('m7_full') / f'sub-{sub.subject_id:02d}' / f'sub-{sub.subject_id:02d}_roi-{roi}_cv-fits.pkl'
    for root in LADDER_ROOTS:
        pkl = Path(root) / tail
        if pkl.exists():
            break
    else:
        raise NoFit(f'm7_full sub-{sub.subject_id:02d} {roi} in {LADDER_ROOTS}')
    with open(pkl, 'rb') as f:
        fit = pickle.load(f)
    print(f'm7_full from {pkl}', flush=True)
    model = build_items_model(grid_coords=grid, paradigm=paradigm, condition_indicator=cond_ind, dynamic_indicator=dyn_ind,
                              target_indicator=tgt_ind, ring_positions=get_ring_positions(),
                              hrf_model=roi_hrf_model(sub, roi, HRF_TABLE), sign_pattern=sign_pattern_of('m7_full'),
                              item_scale='free')
    pars = fit['fit_pars_per_fold'][0].reindex(columns=list(model.parameter_labels)).astype(np.float32).reset_index(drop=True)
    af = dict(voxel_mask_indices=np.asarray(fit['voxel_ids']), r2=np.asarray(fit.get('r2_train', [np.nan]), dtype=float),
              voxel_info={})
    if VOXELS == 'signal':
        vox, pars, info = extend_to_signal_voxels(sub, roi, model, pars, af['voxel_mask_indices'], bold, paradigm)
        af.update(voxel_mask_indices=vox, r2=np.array([np.nan]), voxel_info=info)
    elif len(af['voxel_mask_indices']) < MIN_VOXELS:
        raise NoFit(f"{roi}: only {len(af['voxel_mask_indices'])} voxels in the m7_full fit (< {MIN_VOXELS})")
    return af, model, pars


def voxel_kernels(model, pars: pd.DataFrame, n_t: int, length: int):
    """Per-voxel impulse response of the model's own HRF convolution,
    measured with a delta so shift / flip conventions can't drift.
    Returns (length, V)."""
    t0 = 40
    delta = np.zeros((1, n_t, len(pars)), dtype=np.float32)
    delta[0, t0] = 1.0
    if 'hrf_delay' in pars:
        out = model.hrf_model.convolve(
            tf.constant(delta),
            hrf_delay=tf.constant(pars['hrf_delay'].values[np.newaxis].astype(np.float32)),
            hrf_dispersion=tf.constant(pars['hrf_dispersion'].values[np.newaxis].astype(np.float32)),
        ).numpy()[0]
    else:   # fixed HRF (ladder fits)
        out = model.hrf_model.convolve(tf.constant(delta)).numpy()[0]
    assert np.abs(out[:t0]).max() < 1e-6, 'HRF convolution is not causal'
    return out[t0:t0 + length]


def ring_drives(model, pars: pd.DataFrame, paradigm, grid, sigma_dyn):
    """D[ℓ] (T, V) = sign · Σ_g paradigm·A_ℓ·rf for the 4 rings, plus the
    unmodulated drive (T, V) for the global regressor. Pre-amplitude."""
    std = pars[model.parameter_labels].copy()
    std['amplitude'] = 1.0
    std['baseline'] = 0.0
    rf = model._get_rf(model.grid_coordinates,
                       tf.constant(std.values[np.newaxis].astype(np.float32))
                       ).numpy()[0]                                 # (V, G)
    rings = get_ring_positions()
    d2 = ((grid[np.newaxis] - rings[:, np.newaxis]) ** 2).sum(-1)  # (4, G)
    A = np.exp(-d2 / (2.0 * sigma_dyn ** 2)).astype(np.float32)
    sign = float(model._sign)
    D = np.stack([sign * (paradigm * A[l]) @ rf.T for l in range(4)])
    D_glob = paradigm @ rf.T
    return D.astype(np.float32), D_glob.astype(np.float32)


def sustained_drives(model, pars, paradigm, grid, sigma_af):
    """S[ℓ] (T, V) = sign · Σ_g paradigm·A^AF_ℓ·rf (sustained AF, σ_AF)."""
    return ring_drives(model, pars, paradigm, grid, sigma_af)[0]


def convolve_cols(x, kern, amp):
    """Causal per-voxel convolution of (T, V) with (L, V) kernels × amp."""
    n_t = x.shape[0]
    out = np.zeros_like(x)
    for s in range(kern.shape[0]):
        out[s:] += kern[s] * x[:n_t - s]
    return out * amp


# ---------------------------------------------------------------------------
# Per-trial design (windowed, never materialised as T × K × V)
# ---------------------------------------------------------------------------
def trial_regressors(trials, u, D, D_glob, kern, amp, hp_neutral=False):
    """Return list of regressors (start, window (W, V)) and their meta."""
    n_t = u.shape[0]
    L = kern.shape[0]
    regs, meta = [], []
    for k, row in trials.iterrows():
        nz = np.nonzero(u[:, k])[0]
        if len(nz) == 0:
            continue
        t0, t1 = nz[0], nz[-1] + 1
        W = min(t1 - t0 + L - 1, n_t - t0)
        specs = []
        if row['target_ch'] >= 0:
            specs.append(('T', D[row['target_ch']]))
        if row['dist_ch'] >= 0:
            specs.append(('D_' + row['trial_type'], D[row['dist_ch']]))
        specs.append(('G', D_glob))
        # Previous trial's distractor location, when it holds a neutral
        # item now (not this trial's target or distractor).
        p_ch = row['prev_dist_ch']
        h_ch = row['hp_ch'] if hp_neutral else -1
        if p_ch >= 0 and p_ch not in (row['target_ch'], row['dist_ch'], h_ch):
            specs.append(('P', D[p_ch]))
        # HP location holding a neutral item on this trial.
        if h_ch >= 0 and h_ch not in (row['target_ch'], row['dist_ch']):
            specs.append(('H', D[h_ch]))
        for kind, drive in specs:
            z = u[t0:t1, k][:, np.newaxis] * drive[t0:t1]           # (P, V)
            win = np.zeros((W, D.shape[2]), dtype=np.float32)
            for p in range(t1 - t0):
                n = min(L, W - p)
                win[p:p + n] += z[p] * kern[:n]
            regs.append((t0, win * amp))
            meta.append(dict(trial_idx=k, kind=kind))
    return regs, pd.DataFrame(meta)


def normal_equations(regs, resid, w, voxel_sets):
    """X'WX and X'Wr per voxel subset (each a boolean mask over V)."""
    K = len(regs)
    starts = np.array([s for s, _ in regs])
    ends = np.array([s + win.shape[0] for s, win in regs])
    out = []
    for vs in voxel_sets:
        wv = w[vs]
        XtX = np.zeros((K, K))
        Xtr = np.zeros(K)
        for i, (si, wi) in enumerate(regs):
            a = wi[:, vs] * wv
            Xtr[i] = (a * resid[si:si + len(wi)][:, vs]).sum()
            for j in np.nonzero((starts < ends[i]) & (ends > si)
                                & (np.arange(K) >= i))[0]:
                sj, wj = regs[j]
                lo, hi = max(si, sj), min(ends[i], ends[j])
                XtX[i, j] = (a[lo - si:hi - si] * wj[lo - sj:hi - sj][:, vs]).sum()
                XtX[j, i] = XtX[i, j]
        rtr = float((resid[:, vs] ** 2 * wv).sum())
        out.append((XtX, Xtr, rtr))
    return out


def ridge(XtX, Xtr, g0, alpha):
    d = np.diag(XtX)
    # A regressor with no signal in this voxel set (all-zero row) gets a small penalty, so its gain stays at g0
    # instead of making the system singular; regressors with signal are unchanged.
    lam = alpha * np.where(d > 0, d, 1e-6 * d[d > 0].mean())
    return np.linalg.solve(XtX + np.diag(lam), Xtr + lam * g0)


def sse(g, XtX, Xtr, rtr):
    return rtr - 2 * g @ Xtr + g @ XtX @ g


def run_normal_equations(regs, resid, w, run_id, n_runs):
    """X'WX and X'Wr over all voxels, split by run (rows of the time axis): the per-run matrices sum to those of
    ``normal_equations`` for all voxels. run_id: (T,) run index of every volume."""
    K = len(regs)
    starts = np.array([s for s, _ in regs])
    ends = np.array([s + win.shape[0] for s, win in regs])
    XtX = np.zeros((n_runs, K, K))
    Xtr = np.zeros((n_runs, K))
    for i, (si, wi) in enumerate(regs):
        a = wi * w
        Xtr[:, i] = np.bincount(run_id[si:si + len(wi)], (a * resid[si:si + len(wi)]).sum(1), minlength=n_runs)
        for j in np.nonzero((starts < ends[i]) & (ends > si) & (np.arange(K) >= i))[0]:
            sj, wj = regs[j]
            lo, hi = max(si, sj), min(ends[i], ends[j])
            v = np.bincount(run_id[lo:hi], (a[lo - si:hi - si] * wj[lo - sj:hi - sj]).sum(1), minlength=n_runs)
            XtX[:, i, j] = v
            XtX[:, j, i] = v
    return XtX, Xtr


MODS = ['mod_rt', 'mod_dist_repeat', 'mod_target_repeat', 'mod_target_at_prev_dist']


def param_design(meta, trials, with_global):
    """Design of the parametric model over the per-trial regressors: per kind a main gain and, centred within kind,
    the trial modulators (columns without variance left out). Returns (Mp (K, P), names)."""
    meta_m = meta.merge(trials[['trial_idx', *MODS]], on='trial_idx')
    cols, names = [], []
    for kind in sorted(meta_m['kind'].unique()):
        if kind == 'G' and not with_global:
            continue
        sel = (meta_m['kind'] == kind).values
        cols.append(sel.astype(float))
        names.append((kind, 'main'))
        for m in MODS:
            v = meta_m[m].values * sel
            v = np.where(sel, v - v[sel].mean(), 0.0)
            if v[sel].std() > 1e-6:
                cols.append(v)
                names.append((kind, m.replace('mod_', '')))
    return np.column_stack(cols), names


def param_run_equations(meta, trials, g0, XtX_r, Xtr_r):
    """Per run, the parametric model's normal equations (Mp'M_rMp, Mp'b_r) and Mp'M_r·g0 (for pooling ROIs as a
    deviation from their canonical gains), for the designs without and with the global gain."""
    out = {}
    for wg in (False, True):
        Mp, names = param_design(meta, trials, wg)
        P = np.einsum('ki,rkl,lj->rij', Mp, XtX_r, Mp)
        q = Xtr_r @ Mp
        h = np.einsum('rkl,l->rk', XtX_r, g0) @ Mp
        out[wg] = (names, P, q, h)
    return out


def param_uncertainty(names, P, q):
    """From per-run parametric equations: the all-run estimate, its leave-one-run-out jackknife SE, and the estimates
    on odd and on even runs (by position in the session order)."""
    full = np.linalg.solve(P.sum(0), q.sum(0))
    R = P.shape[0]
    loo = np.array([np.linalg.solve(P.sum(0) - P[r], q.sum(0) - q[r]) for r in range(R)])
    se = np.sqrt((R - 1) / R * ((loo - loo.mean(0)) ** 2).sum(0))
    odd = np.linalg.solve(P[0::2].sum(0), q[0::2].sum(0))
    even = np.linalg.solve(P[1::2].sum(0), q[1::2].sum(0))
    return full, se, odd, even


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def roi_pieces(sub, roi, data, trials, u, seed=0, hp_neutral=False):
    """Everything the estimates need from one ROI: the sanity checks, the per-trial regressors' meta and canonical
    gains g0, the normal equations (all voxels and two voxel halves) and those of the run-wise joint model."""
    bold, paradigm, cond_ind, dyn_ind, grid, tgt_ind = data
    tr = sub.get_tr(1, 1)
    if LADDER_ROOTS:
        af, model, pars = load_ladder_fit(sub, roi, data)
        vox = af['voxel_mask_indices']
        # The gain fields act on the paradigm with the items scaled by item_scale (β).
        beta = float(pars['item_scale'].iloc[0])
        paradigm = paradigm * (1.0 + (beta - 1.0) * model._tf_item_mask.numpy())[np.newaxis, :]
    else:
        pkl = (sub.bids_folder / 'derivatives' / AF_SUBDIR
               / f'sub-{sub.subject_id:02d}'
               / f'sub-{sub.subject_id:02d}_roi-{roi}_mode-signed_{AF_TAG}-af-prf-fit.pkl')
        with open(pkl, 'rb') as f:
            af = pickle.load(f)
        vox = af['voxel_mask_indices']
        model = build_model(paradigm, grid, cond_ind, dyn_ind, tgt_ind, tr)
        pars = af['fit_pars'][model.parameter_labels].astype(np.float32)
    Y = bold[:, vox]
    g_HP, g_LP, g_T = (float(pars[c].iloc[0]) for c in
                       ('g_HP_dyn', 'g_LP_dyn', 'g_T_dyn'))
    sigma_dyn = float(pars['sigma_dyn'].iloc[0])

    pred_full = model.predict(parameters=pars).values
    pars0 = pars.copy()
    pars0[['g_HP_dyn', 'g_LP_dyn', 'g_T_dyn']] = 0.0
    pred0 = model.predict(parameters=pars0).values

    def r2(pred):
        return 1 - ((Y - pred) ** 2).sum(0) / ((Y - Y.mean(0)) ** 2).sum(0)

    # --- Reconstruct the canonical phasic term from the linear pieces. ---
    amp = pars['amplitude'].values.astype(np.float32)
    kern = voxel_kernels(model, pars, Y.shape[0], length=24)
    D, D_glob = ring_drives(model, pars, paradigm, grid, sigma_dyn)
    w_hp = dyn_ind * cond_ind
    w_lp = dyn_ind * (1 - cond_ind)
    field = g_HP * w_hp + g_LP * w_lp + g_T * tgt_ind               # (T, 4)
    phasic = convolve_cols(np.einsum('tl,ltv->tv', field, D), kern, amp)
    target = pred_full - pred0
    recon_err = (np.abs(phasic - target).max()
                 / max(np.abs(target).max(), 1e-12))

    # Pulses must re-assemble the canonical indicators exactly.
    dyn_rebuilt = np.zeros_like(dyn_ind)
    tgt_rebuilt = np.zeros_like(tgt_ind)
    for k, row in trials.iterrows():
        if row['dist_ch'] >= 0:
            dyn_rebuilt[:, row['dist_ch']] += u[:, k]
        if row['target_ch'] >= 0:
            tgt_rebuilt[:, row['target_ch']] += u[:, k]
    pulse_err = max(np.abs(dyn_rebuilt - dyn_ind).max(),
                    np.abs(tgt_rebuilt - tgt_ind).max())

    # Sustained term: linear in (g_HP, g_LP) unless the mod clamp at 0 bites.
    g_HP_s, g_LP_s, sigma_af = (float(pars[c].iloc[0]) for c in
                                ('g_HP', 'g_LP', 'sigma_AF'))
    pars_ns = pars.copy()
    pars_ns[['g_HP', 'g_LP']] = 0.0
    pred_nosus = model.predict(parameters=pars_ns).values
    S = sustained_drives(model, pars, paradigm, grid, sigma_af)
    runs = trials[['session', 'run', 'hp_location']].drop_duplicates()
    n_vol = sub.get_n_volumes(1, 1)
    sus_hp = np.zeros_like(Y)
    sus_lp = np.zeros_like(Y)
    run_masks = []
    for i, (_, rr) in enumerate(runs.iterrows()):
        h = CONDITIONS.index(rr['hp_location'])
        sl = slice(i * n_vol, (i + 1) * n_vol)
        run_masks.append((rr['session'], rr['run'], sl, h))
        sus_hp[sl] = S[h][sl]
        sus_lp[sl] = S.sum(0)[sl] - S[h][sl]
    sus = convolve_cols(g_HP_s * sus_hp + g_LP_s * sus_lp, kern, amp)
    sus_target = pred_full - pred_nosus
    sus_err = (np.abs(sus - sus_target).max()
               / max(np.abs(sus_target).max(), 1e-12))

    check = pd.DataFrame([dict(
        subject=sub.subject_id, roi=roi, n_voxels=len(vox),
        n_trials=len(trials),
        r2_stored=float(np.nanmean(af['r2'])),
        r2_recomputed=float(np.nanmean(r2(pred_full))),
        r2_no_phasic=float(np.nanmean(r2(pred0))),
        phasic_recon_rel_err=float(recon_err),
        pulse_rebuild_max_err=float(pulse_err),
        g_HP_dyn=g_HP, g_LP_dyn=g_LP, g_T_dyn=g_T, sigma_dyn=sigma_dyn,
        g_HP=g_HP_s, g_LP=g_LP_s, sigma_AF=sigma_af,
        sustained_recon_rel_err=float(sus_err),
        **af.get('voxel_info', {}),
    )])
    print(check.T.to_string())
    if recon_err > 1e-2 or pulse_err > 1e-4:
        print(check.T.to_string())
        raise RuntimeError(
            f'Linear reconstruction does not match the model '
            f'(recon_err={recon_err:.2e}, pulse_err={pulse_err:.2e}).')

    resid0 = (Y - pred0).astype(np.float64)
    w = 1.0 / np.var(Y - pred_full, axis=0)
    rng = np.random.default_rng(seed)
    half = np.zeros(len(vox), dtype=bool)
    half[rng.permutation(len(vox))[:len(vox) // 2]] = True
    sets = [np.ones(len(vox), dtype=bool), half, ~half]

    regs, meta = trial_regressors(trials, u, D, D_glob, kern, amp, hp_neutral)
    g0 = meta['kind'].map({'T': g_T, 'D_HP': g_HP, 'D_LP': g_LP,
                           'G': 0.0, 'P': 0.0, 'H': 0.0}).values.astype(np.float64)
    eqs = normal_equations(regs, resid0, w, sets)
    run_id = np.full(Y.shape[0], -1)
    for i, (_ses, _run, sl, _h) in enumerate(run_masks):
        run_id[sl] = i
    assert (run_id >= 0).all()
    XtX_r, Xtr_r = run_normal_equations(regs, resid0, w, run_id, len(run_masks))
    scale = np.abs(eqs[0][0]).max()
    err_M = np.abs(XtX_r.sum(0) - eqs[0][0]).max() / scale
    err_b = np.abs(Xtr_r.sum(0) - eqs[0][1]).max() / np.abs(eqs[0][1]).max()
    print(f'per-run normal equations vs all-run: max rel. error X\'WX {err_M:.2e}, X\'Wr {err_b:.2e}', flush=True)
    assert err_M < 1e-6 and err_b < 1e-6, (err_M, err_b)
    param_runs = param_run_equations(meta, trials, g0, XtX_r, Xtr_r)
    del resid0, XtX_r, Xtr_r

    # Run-wise joint model: sustained HP/LP + phasic gains per run, on the residual from the model with ALL AF
    # gains zeroed. Kept as normal equations (X'WX, X'Wr) and the canonical value of every regressor.
    resid00 = (Y - model.predict(parameters=pars0.assign(
        g_HP=0.0, g_LP=0.0)).values).astype(np.float64)
    meta_r = meta.merge(trials[['trial_idx', 'session', 'run']], on='trial_idx')
    canon = {'S_HP': g_HP_s, 'S_LP': g_LP_s, 'T': g_T, 'D_HP': g_HP, 'D_LP': g_LP, 'G': 0.0}
    run_eqs = {}
    for with_global in (False, True):
        names, dense = [], []
        for ses, run, sl, h in run_masks:
            for kind, x in (('S_HP', sus_hp), ('S_LP', sus_lp)):
                xr = np.zeros_like(Y)
                xr[sl] = x[sl]
                dense.append(convolve_cols(xr, kern, amp))
                names.append((ses, run, kind))
            for kind in ('T', 'D_HP', 'D_LP', 'G'):
                if kind == 'G' and not with_global:
                    continue
                idx = np.nonzero(((meta_r['session'] == ses)
                                  & (meta_r['run'] == run)
                                  & (meta_r['kind'] == kind)).values)[0]
                if len(idx) == 0:
                    continue
                xr = np.zeros_like(Y)
                for i in idx:
                    st, win = regs[i]
                    xr[st:st + len(win)] += win
                dense.append(xr)
                names.append((ses, run, kind))
        X = np.stack(dense).reshape(len(dense), -1)
        Xw = X * np.tile(w, Y.shape[0])
        run_eqs[with_global] = (names, Xw @ X.T, Xw @ resid00.ravel(),
                                np.array([canon[k] for _, _, k in names], dtype=np.float64))
        del X, Xw, dense
    return dict(check=check, meta=meta, g0=g0, eqs=eqs, run_eqs=run_eqs, param_runs=param_runs, n_voxels=len(vox))


def estimate(trials, meta, g0, eqs, run_eqs, param_runs=None):
    """Per-trial ridge (toward g0) with voxel split-half CV, RT-binned / inter-trial OLS, the parametric model and
    the run-wise joint model, all from the normal equations."""
    (M, b, c), (MA, bA, cA), (MB, bB, cB) = eqs
    trial_rows, cv_rows, bin_rows = [], [], []

    # ---- per-trial ridge ----
    sse_A0, sse_B0 = sse(g0, MB, bB, cB), sse(g0, MA, bA, cA)
    for alpha in ALPHAS:
        gA, gB = ridge(MA, bA, g0, alpha), ridge(MB, bB, g0, alpha)
        g = ridge(M, b, g0, alpha)
        row = dict(alpha=alpha,
                   heldout_gain=0.5 * ((sse_A0 - sse(gA, MB, bB, cB)) / sse_A0
                                       + (sse_B0 - sse(gB, MA, bA, cA)) / sse_B0))
        for kind in ('T', 'D_HP', 'D_LP', 'G', 'P', 'H'):
            sel = (meta['kind'] == kind).values
            dA, dB = gA[sel] - g0[sel], gB[sel] - g0[sel]
            row[f'splithalf_r_{kind}'] = (float(np.corrcoef(dA, dB)[0, 1])
                                          if sel.sum() > 2 else np.nan)
        cv_rows.append(row)
        est = meta.assign(alpha=alpha, g=g, g0=g0)
        trial_rows.append(est)

    # ---- RT-binned OLS ----
    extra_bins = ['bin_rq5'] if 'bin_rq5' in trials else []
    meta_b = meta.merge(trials[['trial_idx', 'bin_all', 'bin_rt2', 'bin_rt3', *extra_bins,
                                'bin_run', 'bin_dprime', 'bin_tprime']],
                        on='trial_idx')
    for scheme in ('all', 'rt2', 'rt3', 'run', 'dprime', 'tprime', *[b[4:] for b in extra_bins]):
        for with_global in (False, True):
            keep = (meta_b['kind'] != 'G').values | with_global
            groups = (meta_b['kind'] + '|' + meta_b[f'bin_{scheme}'])[keep]
            labels = sorted(groups.unique())
            Mg = np.zeros((len(meta_b), len(labels)))
            Mg[np.nonzero(keep)[0], groups.map(
                {l: i for i, l in enumerate(labels)}).values] = 1.0
            gb = np.linalg.solve(Mg.T @ M @ Mg, Mg.T @ b)
            n = groups.value_counts()
            for lab, val in zip(labels, gb):
                kind, bin_ = lab.split('|')
                bin_rows.append(dict(scheme=scheme,
                                     with_global=with_global, kind=kind,
                                     bin=bin_, g=val, n_trials=int(n[lab])))

    # ---- Parametric modulation: gain_k = g + Σ_m β_m · m_k ----
    # Also on the two voxel halves, and its uncertainty from the per-run equations: leave-one-run-out jackknife SE
    # (all noise sources, also noise shared across voxels) and the estimates on odd and even runs.
    param_rows = []
    for with_global in (False, True):
        Mp, names = param_design(meta, trials, with_global)
        est = np.linalg.solve(Mp.T @ M @ Mp, Mp.T @ b)
        h1 = np.linalg.solve(Mp.T @ MA @ Mp, Mp.T @ bA)
        h2 = np.linalg.solve(Mp.T @ MB @ Mp, Mp.T @ bB)
        extra = {}
        if param_runs is not None:
            pn, P, q, _h = param_runs[with_global]
            assert pn == names, (pn, names)
            full, se, odd, even = param_uncertainty(pn, P, q)
            # Same estimate up to float32 rounding of the regressors (per-run sums in another order): far below its SE.
            dev = float((np.abs(full - est) / se).max())
            print(f'parametric (with_global={with_global}): all-run vs per-run-sum estimate, max |diff|/SE {dev:.1e}')
            assert dev < 1e-3, dev
            extra = dict(se_jk=se, g_odd_runs=odd, g_even_runs=even)
        for i, ((kind, m), val) in enumerate(zip(names, est)):
            param_rows.append(dict(with_global=with_global, kind=kind, modulator=m, g=val,
                                   g_vhalf1=h1[i], g_vhalf2=h2[i], **{k: v[i] for k, v in extra.items()}))

    # ---- Run-wise joint model ----
    run_rows = []
    for with_global in (False, True):
        names, A, r, _canon = run_eqs[with_global]
        est = np.linalg.solve(A, r)
        for (ses, run, kind), val in zip(names, est):
            run_rows.append(dict(with_global=with_global, session=ses,
                                 run=run, kind=kind, g=val))
    return trial_rows, cv_rows, bin_rows, param_rows, run_rows


def save_estimates(sub, roi, out_dir, trials, check, trial_rows, cv_rows, bin_rows, param_rows, run_rows):
    stem = f'sub-{sub.subject_id:02d}_roi-{roi}'
    per_trial = (pd.concat(trial_rows)
                 .pivot_table(index=['alpha', 'trial_idx'],
                              columns='kind', values=['g', 'g0'])
                 )
    per_trial.columns = [f'{a}_{k}' for a, k in per_trial.columns]
    per_trial = per_trial.reset_index().merge(trials, on='trial_idx')
    for df, name in [(per_trial, 'trials'), (pd.DataFrame(cv_rows), 'cv'),
                     (pd.DataFrame(bin_rows), 'bins'), (check, 'check'),
                     (pd.DataFrame(param_rows), 'parametric'),
                     (pd.DataFrame(run_rows), 'runfit')]:
        if 'roi' not in df:
            df.insert(0, 'roi', roi)
            df.insert(0, 'subject', sub.subject_id)
        df.to_csv(out_dir / f'{stem}_{name}.tsv', sep='\t', index=False)
    print(pd.DataFrame(cv_rows).to_string())
    print(f'Saved {out_dir}/{stem}_*.tsv')




def fit_roi(sub, roi, data, trials, u, out_dir, seed=0, hp_neutral=False):
    pieces = roi_pieces(sub, roi, data, trials, u, seed=seed, hp_neutral=hp_neutral)
    save_estimates(sub, roi, out_dir, trials, pieces['check'],
                   *estimate(trials, pieces['meta'], pieces['g0'], pieces['eqs'], pieces['run_eqs'], pieces['param_runs']))
    return pieces


def pool_pieces(pieces):
    """One set of gains for several ROIs: a shared deviation δ from each ROI's canonical gains.

    Per ROI the residual becomes r' = r − X·g0, so X'Wr' = X'Wr − X'WX·g0 and r'Wr' = r'Wr − 2·g0'X'Wr + g0'X'WX·g0;
    the normal equations are then summed over the ROIs and the prior is δ = 0. The run-wise joint model is pooled
    the same way around each ROI's canonical sustained and phasic gains.
    """
    meta = pieces[0]['meta']
    for p in pieces[1:]:
        if not p['meta'][['trial_idx', 'kind']].equals(meta[['trial_idx', 'kind']]):
            raise ValueError('ROIs disagree on the per-trial regressors')
    eqs = None
    for p in pieces:
        g0 = p['g0']
        t = [(M, b - M @ g0, c - 2 * g0 @ b + g0 @ M @ g0) for M, b, c in p['eqs']]
        eqs = t if eqs is None else [tuple(x + y for x, y in zip(e, f)) for e, f in zip(eqs, t)]
    run_eqs = {}
    for wg in (False, True):
        names = pieces[0]['run_eqs'][wg][0]
        A, r = 0.0, 0.0
        for p in pieces:
            n_, A_, r_, c_ = p['run_eqs'][wg]
            if n_ != names:
                raise ValueError('ROIs disagree on the run-wise regressors')
            A, r = A + A_, r + (r_ - A_ @ c_)
        run_eqs[wg] = (names, A, r, np.zeros(len(names)))
    param_runs = {}
    for wg in (False, True):
        names = pieces[0]['param_runs'][wg][0]
        P = sum(p['param_runs'][wg][1] for p in pieces)
        q = sum(p['param_runs'][wg][2] - p['param_runs'][wg][3] for p in pieces)
        param_runs[wg] = (names, P, q, np.zeros_like(q))
    return meta, np.zeros(len(meta)), eqs, run_eqs, param_runs


def fit_group(sub, group, pieces, rois, out_dir, trials):
    meta, g0, eqs, run_eqs, param_runs = pool_pieces(pieces)
    checks = pd.concat([p['check'] for p in pieces])
    nv = checks['n_voxels'].to_numpy()
    check = pd.DataFrame([dict(
        subject=sub.subject_id, roi=group, rois=' '.join(rois), n_rois=len(rois), n_voxels=int(nv.sum()),
        n_trials=len(trials), r2_recomputed=float(np.average(checks['r2_recomputed'], weights=nv)),
        r2_no_phasic=float(np.average(checks['r2_no_phasic'], weights=nv)),
        phasic_recon_rel_err=float(checks['phasic_recon_rel_err'].max()),
        pulse_rebuild_max_err=float(checks['pulse_rebuild_max_err'].max()),
        sustained_recon_rel_err=float(checks['sustained_recon_rel_err'].max()),
        gains='deviation from each ROI\'s canonical gains')])
    save_estimates(sub, group, out_dir, trials, check, *estimate(trials, meta, g0, eqs, run_eqs, param_runs))


def main(subject, bids_folder, rois, output_subdir='af_trialwise_gains', hp_neutral=False, ladder_roots=(),
         hrf_table=None, voxels='ladder', groups=False, min_voxels=0, rt_modulator='zlogrt'):
    global LADDER_ROOTS, HRF_TABLE, VOXELS, MIN_VOXELS, RT_MODULATOR
    VOXELS, MIN_VOXELS, RT_MODULATOR = voxels, min_voxels, rt_modulator
    if voxels == 'signal' and not ladder_roots:
        raise ValueError('--voxels signal needs --ladder-root')
    from retsupp.modeling.roi_hrf import load_table
    LADDER_ROOTS = tuple(Path(bids_folder) / 'derivatives' / r if not str(r).startswith('/') else Path(r) for r in ladder_roots)
    HRF_TABLE = load_table(hrf_table)
    sub = Subject(subject, bids_folder=Path(bids_folder))
    out_dir = (Path(bids_folder) / 'derivatives' / output_subdir
               / f'sub-{subject:02d}')
    out_dir.mkdir(parents=True, exist_ok=True)

    if LADDER_ROOTS:
        # The ladder fits use the 8-item rectangle paradigm and their own data builder.
        from retsupp.modeling.fit_af_prf_cv_v2 import build_data_and_paradigm as build_items_data
        (bold, paradigm, cond_ind, dyn_ind, tgt_ind, grid, _masker, _meta) = build_items_data(
            sub, resolution=50, grid_radius=5.0, item_shape='rectangle')
        paradigm, grid = np.asarray(paradigm), np.asarray(grid)
    else:
        (bold_df, paradigm, cond_ind, dyn_ind, grid, _masker, tgt_ind,
         *_rest) = build_data_and_paradigm(sub, resolution=50, grid_radius=5.0,
                                           with_target=True)
        bold = bold_df.values
        del bold_df
    trials, u, n_t = build_trials(sub)
    assert n_t == paradigm.shape[0], (n_t, paradigm.shape)
    trials = add_rt_bins(trials)
    print(trials['trial_type'].value_counts().to_string())
    print(trials['bin_rt2'].value_counts().to_string())

    data = (bold, paradigm, cond_ind, dyn_ind, grid, tgt_ind)
    missing, pieces = [], {}
    for roi in rois:
        print(f'\n===== sub-{subject:02d} {roi} =====')
        try:
            p = fit_roi(sub, roi, data, trials, u, out_dir, hp_neutral=hp_neutral)
        except NoFit as e:
            # e.g. no m7_full fit: the ladder leaves out subject x ROI pairs without reliable voxels
            print(f'skipped {roi}: {e}')
            missing.append(roi)
            continue
        if groups:
            pieces[roi] = p
    if missing:
        print(f'ROIs without a fit: {missing}')
    for group in (GROUPS if groups else {}):
        have = [r for r in GROUPS[group] if r in pieces]
        print(f'\n===== sub-{subject:02d} {group}: {have} =====')
        if not have:
            print(f'skipped {group}: none of its ROIs')
            continue
        fit_group(sub, group, [pieces[r] for r in have], have, out_dir, trials)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('subject', type=int)
    parser.add_argument('--bids-folder', default='/data/ds-retsupp')
    parser.add_argument('--rois', nargs='+', default=ROIS)
    parser.add_argument('--output-subdir', default='af_trialwise_gains')
    parser.add_argument('--hp-neutral', action='store_true',
                        help='Add the H regressor: gain at HP when it holds a neutral item.')
    parser.add_argument('--ladder-root', nargs='+', default=[],
                        help='Use the m7_full fits of the gain-model ladder in these roots (first hit wins) '
                             'instead of the October fits.')
    parser.add_argument('--hrf-table', default=None, help='One HRF per subject x ROI (with --ladder-root)')
    parser.add_argument('--voxels', choices=['ladder', 'signal'], default='ladder',
                        help='signal (with --ladder-root): the signal voxels of each ROI (held-out null cvR² > 0, '
                             'BOLD SD <= 20), pRFs of voxels outside the m7_full fit from the joint-HRF DoG fold fits.')
    parser.add_argument('--groups', action='store_true',
                        help='Also one set of gains per ROI group (early, dorsal, occipitotemporal): a shared '
                             'deviation from each ROI\'s canonical gains.')
    parser.add_argument('--min-voxels', type=int, default=0,
                        help='With --voxels ladder: leave out a subject x ROI whose m7_full fit has fewer voxels '
                             '(--voxels signal always needs 10).')
    parser.add_argument('--rt-modulator', choices=['zlogrt', 'residual'], default='zlogrt',
                        help='residual: log RT residualised on the design within participant, z-scored '
                             '(notes/rt_power_plan_2026-11-13.md); adds residual-RT quintile bins (scheme rq5).')
    args = parser.parse_args()
    main(args.subject, args.bids_folder, args.rois, args.output_subdir, args.hp_neutral, args.ladder_root,
         args.hrf_table, args.voxels, args.groups, args.min_voxels, args.rt_modulator)
