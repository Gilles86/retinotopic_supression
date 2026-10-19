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

A fourth regressor kind, ``P``, is the gain at the PREVIOUS trial's
distractor location when it holds a neutral item on the current trial
(prior 0; no canonical counterpart) — intertrial inhibition at a
location that is empty of singletons now.

Trial pulses are the AF fit's (target onset → feedback, capped at
1.5 s). Feedback arrives at a fixed 1.5 s after target onset, so the
pulse width does not depend on RT.

Outputs under ``derivatives/af_trialwise_gains/sub-XX/``:
``sub-XX_roi-R_trials.tsv`` (per-trial gains per α),
``sub-XX_roi-R_cv.tsv`` (held-out SSE + split-half reliability per α),
``sub-XX_roi-R_bins.tsv`` (binned gains) and
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
                    dist_ch=d_ch, target_ch=t_ch, trial_type=ttype,
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


def voxel_kernels(model, pars: pd.DataFrame, n_t: int, length: int):
    """Per-voxel impulse response of the model's own HRF convolution,
    measured with a delta so shift / flip conventions can't drift.
    Returns (length, V)."""
    t0 = 40
    delta = np.zeros((1, n_t, len(pars)), dtype=np.float32)
    delta[0, t0] = 1.0
    out = model.hrf_model.convolve(
        tf.constant(delta),
        hrf_delay=tf.constant(pars['hrf_delay'].values[np.newaxis].astype(np.float32)),
        hrf_dispersion=tf.constant(pars['hrf_dispersion'].values[np.newaxis].astype(np.float32)),
    ).numpy()[0]
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
def trial_regressors(trials, u, D, D_glob, kern, amp):
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
        if p_ch >= 0 and p_ch not in (row['target_ch'], row['dist_ch']):
            specs.append(('P', D[p_ch]))
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
    lam = alpha * np.diag(XtX)
    return np.linalg.solve(XtX + np.diag(lam), Xtr + lam * g0)


def sse(g, XtX, Xtr, rtr):
    return rtr - 2 * g @ Xtr + g @ XtX @ g


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def fit_roi(sub, roi, data, trials, u, out_dir, seed=0):
    bold, paradigm, cond_ind, dyn_ind, grid, tgt_ind = data
    pkl = (sub.bids_folder / 'derivatives' / AF_SUBDIR
           / f'sub-{sub.subject_id:02d}'
           / f'sub-{sub.subject_id:02d}_roi-{roi}_mode-signed_{AF_TAG}-af-prf-fit.pkl')
    with open(pkl, 'rb') as f:
        af = pickle.load(f)
    vox = af['voxel_mask_indices']
    Y = bold[:, vox]
    tr = sub.get_tr(1, 1)
    model = build_model(paradigm, grid, cond_ind, dyn_ind, tgt_ind, tr)
    pars = af['fit_pars'][model.parameter_labels].astype(np.float32)
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

    check = pd.DataFrame([dict(
        subject=sub.subject_id, roi=roi, n_voxels=len(vox),
        n_trials=len(trials),
        r2_stored=float(np.nanmean(af['r2'])),
        r2_recomputed=float(np.nanmean(r2(pred_full))),
        r2_no_phasic=float(np.nanmean(r2(pred0))),
        phasic_recon_rel_err=float(recon_err),
        pulse_rebuild_max_err=float(pulse_err),
        g_HP_dyn=g_HP, g_LP_dyn=g_LP, g_T_dyn=g_T, sigma_dyn=sigma_dyn,
    )])
    print(check.T.to_string())
    if recon_err > 1e-2 or pulse_err > 1e-4:
        check.to_csv(out_dir / f'sub-{sub.subject_id:02d}_roi-{roi}_check.tsv',
                     sep='\t', index=False)
        raise RuntimeError(
            f'Linear reconstruction does not match the model '
            f'(recon_err={recon_err:.2e}, pulse_err={pulse_err:.2e}).')

    resid0 = (Y - pred0).astype(np.float64)
    w = 1.0 / np.var(Y - pred_full, axis=0)
    rng = np.random.default_rng(seed)
    half = np.zeros(len(vox), dtype=bool)
    half[rng.permutation(len(vox))[:len(vox) // 2]] = True
    sets = [np.ones(len(vox), dtype=bool), half, ~half]

    trial_rows, cv_rows, bin_rows = [], [], []
    regs, meta = trial_regressors(trials, u, D, D_glob, kern, amp)
    g0 = meta['kind'].map({'T': g_T, 'D_HP': g_HP, 'D_LP': g_LP,
                           'G': 0.0, 'P': 0.0}).values.astype(np.float64)
    (M, b, c), (MA, bA, cA), (MB, bB, cB) = normal_equations(
        regs, resid0, w, sets)

    # ---- per-trial ridge ----
    sse_A0, sse_B0 = sse(g0, MB, bB, cB), sse(g0, MA, bA, cA)
    for alpha in ALPHAS:
        gA, gB = ridge(MA, bA, g0, alpha), ridge(MB, bB, g0, alpha)
        g = ridge(M, b, g0, alpha)
        row = dict(alpha=alpha,
                   heldout_gain=0.5 * ((sse_A0 - sse(gA, MB, bB, cB)) / sse_A0
                                       + (sse_B0 - sse(gB, MA, bA, cA)) / sse_B0))
        for kind in ('T', 'D_HP', 'D_LP', 'G', 'P'):
            sel = (meta['kind'] == kind).values
            dA, dB = gA[sel] - g0[sel], gB[sel] - g0[sel]
            row[f'splithalf_r_{kind}'] = (float(np.corrcoef(dA, dB)[0, 1])
                                          if sel.sum() > 2 else np.nan)
        cv_rows.append(row)
        est = meta.assign(alpha=alpha, g=g, g0=g0)
        trial_rows.append(est)

    # ---- RT-binned OLS ----
    meta_b = meta.merge(trials[['trial_idx', 'bin_all', 'bin_rt2', 'bin_rt3',
                                'bin_run', 'bin_dprime', 'bin_tprime']],
                        on='trial_idx')
    for scheme in ('all', 'rt2', 'rt3', 'run', 'dprime', 'tprime'):
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

    stem = f'sub-{sub.subject_id:02d}_roi-{roi}'
    per_trial = (pd.concat(trial_rows)
                 .pivot_table(index=['alpha', 'trial_idx'],
                              columns='kind', values=['g', 'g0'])
                 )
    per_trial.columns = [f'{a}_{k}' for a, k in per_trial.columns]
    per_trial = per_trial.reset_index().merge(trials, on='trial_idx')
    for df, name in [(per_trial, 'trials'), (pd.DataFrame(cv_rows), 'cv'),
                     (pd.DataFrame(bin_rows), 'bins'), (check, 'check')]:
        if 'roi' not in df:
            df.insert(0, 'roi', roi)
            df.insert(0, 'subject', sub.subject_id)
        df.to_csv(out_dir / f'{stem}_{name}.tsv', sep='\t', index=False)
    print(pd.DataFrame(cv_rows).to_string())
    print(f'Saved {out_dir}/{stem}_*.tsv')


def main(subject, bids_folder, rois, output_subdir='af_trialwise_gains'):
    sub = Subject(subject, bids_folder=Path(bids_folder))
    out_dir = (Path(bids_folder) / 'derivatives' / output_subdir
               / f'sub-{subject:02d}')
    out_dir.mkdir(parents=True, exist_ok=True)

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
    for roi in rois:
        print(f'\n===== sub-{subject:02d} {roi} =====')
        fit_roi(sub, roi, data, trials, u, out_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('subject', type=int)
    parser.add_argument('--bids-folder', default='/data/ds-retsupp')
    parser.add_argument('--rois', nargs='+', default=ROIS)
    parser.add_argument('--output-subdir', default='af_trialwise_gains')
    args = parser.parse_args()
    main(args.subject, args.bids_folder, args.rois, args.output_subdir)
