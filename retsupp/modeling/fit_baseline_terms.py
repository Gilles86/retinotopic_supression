"""Baseline shift vs response gain at the HP location (2026-11-12), steps 2-4: additive terms on the final joint fit.

Per subject x ROI, on the subject's final joint gain-field fit (m7_full, DoG pRFs, per-ROI HRF; the fixed-group
framework of ``fit_fixed_group_mechanisms``: every other parameter fixed, each added term linear, so its amplitude is a
least-squares coefficient on the joint fit's residual R = Y − P0):

* ``add_sustained``: an additive sustained field, the pRF-weighted HP − LP field (w_v below) added as a constant per
  run. Not identifiable: the cleaned BOLD is % signal change of each run's mean (``preprocess/clean.py``) and HP is
  constant within a run, so the data carry no run-level offset. Shown directly: the share of the regressor's sum of
  squares that survives removing each run's mean (``add_sustained_kept``) — 0 to float precision.
* ``trial_spec_{search,fix}``: a trial-locked additive response at the HP location: a stick at every search onset
  (``target`` events) or fixation onset (``pre-target``), convolved with the ROI's HRF, with amplitude per voxel
  w_v = overlap(pRF_v, field at HP) − mean overlap(pRF_v, field at the three LP locations), using the pRF centre
  Gaussian and the fit's sustained width σ_AF (closed form). Independent of what item appears and of the drive.
  ``trial_ns_*``: the same with the mean overlap over all four locations (a non-specific trial response, nuisance).
* ``dS``, ``dD``, ``dT``: the multiplicative HP − LP terms (sustained, transient distractor, transient target) as
  central differences of the joint fit (g_HP, g_HP_dyn, g_T_HP_dyn), for the joint designs.

Designs (amplitudes estimated on the training runs = all runs but run 3 of each session, scored on run 3: held-out
ΔR² = −ΔSSE / SST(run 3), one amplitude vector per subject x ROI shared by its voxels); also all-run estimates:
  trial_search      [trial_spec_search, trial_ns_search]
  trial_fix         [trial_spec_fix, trial_ns_fix]
  trial_search+dDdT [trial_spec_search, trial_ns_search, dD, dT]
  dDdT              [dD, dT]
  dS                [dS]
  dS+trial_search   [dS, trial_spec_search, trial_ns_search]

Recovery (step 4, ``--n-sim``): from the joint fit with the sustained HP gain tied to LP (P0n), simulated data
  Y = P0n + signal + phase-randomised residual (one random phase spectrum per run, shared by all voxels, so the
  residual's spatial correlations are kept), with signal = (i) multiplicative: δ·X_dS with δ = the group dS
  (``--sim-ds``, −0.014); (ii) trial-locked additive: amplitude matched to (i) in signal power (sum of squares) at
  the voxels nearest HP (w_v at least half its maximum), negative; (iii) sustained additive, then each run's mean
  removed (what the cleaning does). For each simulated dataset the dS and trial_search designs are estimated as above;
  group-level recovery (which term is detected) is computed offline from the per-subject outputs.

Outputs ``derivatives/<output-subdir>/sub-XX/sub-XX_roi-R_terms.tsv`` (one row per design x term: estimate on all runs,
estimate on training runs, held-out ΔR², n voxels) and ``..._sim.tsv`` (sim x generator x design x term estimates).

    python -m retsupp.modeling.fit_baseline_terms 3 --rois V1 --n-sim 2   # canary
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import retsupp.modeling.fit_fixed_group_mechanisms as fgm
from retsupp.modeling.cv_helpers import CONDITIONS
from retsupp.modeling.fit_af_prf_cv_v2 import build_data_and_paradigm, get_ring_positions
from retsupp.modeling.roi_hrf import hrf_model as roi_hrf_model, load_table as load_hrf_table
from retsupp.utils.data import Subject

ROIS = ['V1', 'V2', 'V3', 'V3AB', 'hV4', 'LO', 'TO', 'VO', 'IPS']
DESIGNS = {
    'trial_search': ['trial_spec_search', 'trial_ns_search'],
    'trial_fix': ['trial_spec_fix', 'trial_ns_fix'],
    'trial_search+dDdT': ['trial_spec_search', 'trial_ns_search', 'dD', 'dT'],
    'dDdT': ['dD', 'dT'],
    'dS': ['dS'],
    'dS+trial_search': ['dS', 'trial_spec_search', 'trial_ns_search'],
}
PERTURB = {'dS': 'g_HP', 'dD': 'g_HP_dyn', 'dT': 'g_T_HP_dyn'}


def overlap(x, y, sd, c, sigma):
    """∫ N(g; (x, y), sd²) · exp(−‖g − c‖² / 2σ²) dg for each voxel (closed form)."""
    s2 = sigma ** 2 + sd ** 2
    return sigma ** 2 / s2 * np.exp(-((x - c[0]) ** 2 + (y - c[1]) ** 2) / (2 * s2))


def onset_series(sub, run_meta, event, hrf_kernel):
    """Per TR (stacked runs): sticks at the event onsets, convolved with the HRF, per run."""
    out = np.zeros(sum(rm.n_T for rm in run_meta))
    for rm in run_meta:
        ev = sub.get_onsets(rm.session, rm.run)
        on = ev[ev.event_type == event].onset.to_numpy()
        tr = sub.get_tr(rm.session, rm.run)
        s = np.zeros(rm.n_T)
        idx = np.round(on / tr).astype(int)
        np.add.at(s, idx[(idx >= 0) & (idx < rm.n_T)], 1.)
        out[rm.start:rm.stop] = np.convolve(s, hrf_kernel)[:rm.n_T]
    return out


def solve(X, R, train):
    """One amplitude vector shared by all voxels: Σ_v X_v'X_v δ = Σ_v X_v'R_v over the rows in ``train``."""
    Xt, Rt = X[:, train], R[train]                       # X: (k, T, V)
    A = np.einsum('ktv,ltv->kl', Xt, Xt)
    b = np.einsum('ktv,tv->k', Xt, Rt)
    return np.linalg.lstsq(A, b, rcond=None)[0]


def held_out_dr2(X, R, delta, test, run_meta):
    Rt = R[test]
    fit = np.einsum('k,ktv->tv', delta, X[:, test])
    sst = 0.
    for rm in run_meta:
        rows = np.arange(rm.start, rm.stop)
        rows = rows[np.isin(rows, test)]
        if len(rows):
            sst += ((R[rows] - R[rows].mean(0)) ** 2).sum()  # residual SST of the held-out run (data are psc anyway)
    return float((((Rt) ** 2).sum() - ((Rt - fit) ** 2).sum()) / sst)


def phase_randomise(E, run_meta, rng):
    out = np.empty_like(E)
    for rm in run_meta:
        e = E[rm.start:rm.stop]
        f = np.fft.rfft(e, axis=0)
        ph = rng.uniform(0, 2 * np.pi, f.shape[0])
        ph[0] = 0.
        if e.shape[0] % 2 == 0:
            ph[-1] = 0.
        out[rm.start:rm.stop] = np.fft.irfft(f * np.exp(1j * ph)[:, None], n=e.shape[0], axis=0)
    return out


def fit_roi(sub, roi, data, out_dir, overlay_root, n_sim, sim_ds, seed):
    bold, paradigm, _ci, _di, _ti, _grid, run_meta = data
    model = fgm.build_full_model(sub, data, roi)
    fit, pars, _w, _n = fgm.base_parameters(sub, model, roi, 'm7_full', None, overlay_root)
    Y = bold[:, fit['voxel_ids']].astype(np.float64)

    def pred(p):
        return np.asarray(model.predict(paradigm=paradigm, parameters=p), dtype=np.float64)

    P0 = pred(pars)
    R = Y - P0
    terms = {}
    for name, par in PERTURB.items():
        up, dn = pars.copy(), pars.copy()
        up[par] += fgm.H
        dn[par] -= fgm.H
        terms[name] = (pred(up) - pred(dn)) / (2 * fgm.H)
    # trial-locked and sustained additive regressors
    ring = get_ring_positions()
    sigma = float(pars['sigma_AF'].iloc[0])
    ov = np.stack([overlap(pars.x.to_numpy(), pars.y.to_numpy(), pars.sd.to_numpy(), c, sigma) for c in ring])  # (4, V)
    T, V = Y.shape
    w_spec, w_ns = np.zeros((T, V)), np.zeros((T, V))
    for rm in run_meta:
        h = CONDITIONS.index(rm.hp)
        lp = [k for k in range(4) if k != h]
        w_spec[rm.start:rm.stop] = ov[h] - ov[lp].mean(0)
        w_ns[rm.start:rm.stop] = ov.mean(0)
    kernel = roi_hrf_model(sub, roi, fgm.HRF_TABLE).get_hrf().numpy().ravel()
    for lock, event in (('search', 'target'), ('fix', 'pre-target')):
        u = onset_series(sub, run_meta, event, kernel)[:, None]
        terms[f'trial_spec_{lock}'] = u * w_spec
        terms[f'trial_ns_{lock}'] = u * w_ns
    # additive sustained: what survives removing each run's mean (the cleaning) — 0 to float precision
    kept = w_spec.copy()
    for rm in run_meta:
        kept[rm.start:rm.stop] -= kept[rm.start:rm.stop].mean(0)
    add_kept = float((kept ** 2).sum() / max((w_spec ** 2).sum(), 1e-30))

    test = np.concatenate([np.arange(rm.start, rm.stop) for rm in run_meta if rm.run == 3])
    train = np.setdiff1d(np.arange(T), test)
    allr = np.arange(T)
    rows = []
    for dname, names in DESIGNS.items():
        X = np.stack([terms[n] for n in names])
        d_all = solve(X, R, allr)
        d_tr = solve(X, R, train)
        dr2 = held_out_dr2(X, R, d_tr, test, run_meta)
        for n, a, t in zip(names, d_all, d_tr):
            rows.append(dict(subject=sub.subject_id, roi=roi, design=dname, term=n, estimate=a, estimate_train=t,
                             heldout_dr2=dr2, n_voxels=V, add_sustained_kept=add_kept))
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / f'sub-{sub.subject_id:02d}_roi-{roi}_terms.tsv', sep='\t', index=False)
    print(f'sub-{sub.subject_id:02d} {roi}: {V} voxels; additive sustained kept {add_kept:.1e}; '
          + '; '.join(f'{r["design"]}:{r["term"]}={r["estimate"]:+.4f}' for r in rows), flush=True)

    if not n_sim:
        return
    # recovery: base with the sustained HP gain tied to LP
    pn = pars.copy()
    pn['g_HP'] = pn['g_LP']
    P0n = pred(pn)
    up, dn = pn.copy(), pn.copy()
    up['g_HP'] += fgm.H
    dn['g_HP'] -= fgm.H
    Xds = (pred(up) - pred(dn)) / (2 * fgm.H)
    Xtr = terms['trial_spec_search']
    # Matched size: the same signal power (sum of squares) at the voxels nearest HP (w_v at least half its maximum),
    # with the sign of a suppression. A match on the mean is ill-posed here: both signals have about zero mean per run.
    wmax = np.abs(w_spec).max(0)
    sel = wmax >= .5 * wmax.max()
    mult_ss = ((sim_ds * Xds)[:, sel] ** 2).sum()
    add_ss = (Xtr[:, sel] ** 2).sum()
    amp = -np.sqrt(mult_ss / add_ss) if add_ss > 0 else np.nan
    gens = {'multiplicative': sim_ds * Xds, 'trial_additive': amp * Xtr}
    sus = w_spec * (amp if np.isfinite(amp) else 0.)
    for rm in run_meta:
        sus[rm.start:rm.stop] -= sus[rm.start:rm.stop].mean(0)
    gens['sustained_additive'] = sus
    E = Y - P0
    rng = np.random.default_rng(seed + sub.subject_id * 100 + ROIS.index(roi) if roi in ROIS else seed)
    srows = []
    for k in range(n_sim):
        Epr = phase_randomise(E, run_meta, rng)
        for gname, sig in gens.items():
            Rs = sig + Epr                                # = Y_sim − P0n
            for dname, X in (('dS', np.stack([Xds])),
                             ('trial_search', np.stack([Xtr, terms['trial_ns_search']])),
                             ('dS+trial_search', np.stack([Xds, Xtr, terms['trial_ns_search']]))):
                d_tr = solve(X, Rs, train)
                dr2 = held_out_dr2(X, Rs, d_tr, test, run_meta)
                d_all = solve(X, Rs, allr)
                names = {'dS': ['dS'], 'trial_search': ['trial_spec_search', 'trial_ns_search'],
                         'dS+trial_search': ['dS', 'trial_spec_search', 'trial_ns_search']}[dname]
                for n, a in zip(names, d_all):
                    srows.append(dict(subject=sub.subject_id, roi=roi, sim=k, generator=gname, design=dname, term=n,
                                      estimate=a, heldout_dr2=dr2, amp_matched=amp, sim_ds=sim_ds))
    pd.DataFrame(srows).to_csv(out_dir / f'sub-{sub.subject_id:02d}_roi-{roi}_sim.tsv', sep='\t', index=False)
    print(f'sub-{sub.subject_id:02d} {roi}: {n_sim} simulations, matched trial amplitude {amp:+.4f}', flush=True)


def main(subject, bids_folder, rois, output_subdir, base_root, hrf_table, overlay_root, n_sim, sim_ds, seed):
    fgm.BASE_ROOT = base_root
    fgm.HRF_TABLE = load_hrf_table(hrf_table)
    fgm.FAMILY = 'dog'
    sub = Subject(subject, bids_folder=Path(bids_folder))
    out_dir = Path(bids_folder) / 'derivatives' / output_subdir / f'sub-{subject:02d}'
    todo = [r for r in rois if not (out_dir / f'sub-{subject:02d}_roi-{r}_terms.tsv').exists()]
    todo = [r for r in todo if fgm.fit_path(bids_folder, 'm7_full', subject, r, overlay_root).exists()]
    if not todo:
        print('Nothing to do')
        return
    bold, paradigm, ci, di, ti, grid, _masker, run_meta = build_data_and_paradigm(
        sub, resolution=50, grid_radius=5.0, item_shape='rectangle')
    data = (bold, paradigm, ci, di, ti, grid, run_meta)
    for roi in todo:
        fit_roi(sub, roi, data, out_dir, overlay_root, n_sim, sim_ds, seed)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('subject', type=int)
    ap.add_argument('--bids-folder', default='/shares/zne.uzh/gdehol/ds-retsupp')
    ap.add_argument('--rois', nargs='+', default=ROIS)
    ap.add_argument('--output-subdir', default='baseline_vs_gain_2026-11-12')
    ap.add_argument('--base-root', default='af_prf_cv_ladder_nested_full_bounded_maxsd20_reliable_roihrf')
    ap.add_argument('--hrf-table', default='/shares/zne.uzh/gdehol/ds-retsupp/derivatives/paper_2026-11/roi_hrf_dog_final2.tsv')
    ap.add_argument('--overlay-root', default=None)
    ap.add_argument('--n-sim', type=int, default=10)
    ap.add_argument('--sim-ds', type=float, default=-0.014, help='Simulated sustained HP − LP gain (group dS)')
    ap.add_argument('--seed', type=int, default=20261112)
    a = ap.parse_args()
    main(a.subject, a.bids_folder, a.rois, a.output_subdir, a.base_root, a.hrf_table, a.overlay_root, a.n_sim,
         a.sim_ds, a.seed)
