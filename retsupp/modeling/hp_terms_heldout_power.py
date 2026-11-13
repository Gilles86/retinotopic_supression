"""Transient HP − LP terms: leave-one-run-out held-out prediction (item 1a) and a power simulation (item 1c).

Writing-package update 11 (``notes/request_2026-11-13_transient_terms.md``). One subject per call, all nine areas, so
the simulated noise can share its random phases across areas.

Fits used (reference of update 8):
- base: the model with item weight, target, distractor and sustained gain (m4_TDS) of the polished root (pRFs refitted
  together with the gain fields), the model on which every HP − LP term is "estimated alone" in the paper
  (``fit_fixed_group_mechanisms``);
- generator of the simulation: the joint fit (m7_full, all terms incl. dS and dD) of the same root.

All HP − LP terms enter the DoG gain-field model linearly (the gains multiply the item drive; ``lin_err`` of
``fit_fixed_group_mechanisms`` is ~1e-6), so with pRFs and field widths held fixed every fit below is a linear least
squares on regressors X_k = ∂P/∂θ_k at the base (central difference, H = 0.05), pooled over the area's voxels:

    base terms   S0 (g_HP = g_LP, sustained gain at all four locations), D0 (g_HP_dyn = g_LP_dyn, distractor gain),
                 T0 (g_T_dyn = g_T_HP_dyn, target gain), I0 (item_scale, item weight)
    HP terms     dS (g_HP: sustained, HP − LP), dD (g_HP_dyn: distractor transient, HP − LP),
                 dT (g_T_HP_dyn: target transient, HP − elsewhere; ``target_split``)

Estimators (both from the residual R = Y − P_base):
- "alone" (the paper's route for dS, dD, dT): refit the four base terms, then each HP term alone with the base fixed
  at that refit, δ_k = X_k'(R − X_b β_b) / X_k'X_k. On the real data the base is at its optimum (β_b ≈ 0), so this is
  close to ``collect_conditional_estimates`` (Σb / Σa, the base not refitted; also saved, ``*_alone_fixedbase``);
- "joint": one least squares on base + dS + dD (the HP terms of the joint fit, m7_full) and on base + dS + dD + dT
  (the joint fit with the target split, m8_full_dT).

1a, leave-one-run-out: per held-out run, the base terms and the HP terms of each model are fitted on the other runs
and the held-out SSE is computed in closed form from per-run Gram matrices. Models: none (base terms only), dS,
dS + dD, dS + dD + dT. The pRFs and field widths come from the all-runs fit (identical in every model, so their leak
into the held-out run is shared by all models).

1c, power: Y_sim(δ) = P_gen(δ) + N, with P_gen(δ) the joint fit's prediction with the distractor gain at HP set to
g_LP_dyn + δ (δ in gain units: −0.02 = −2 percentage points, the unit of the reported dD; no dT in the generator) and
N the joint fit's residuals (Y − P_m7), phase-randomised per run with one set of phases for all voxels and all areas
of the subject (``simulate_decode.phase_randomise``). Each draw is refitted with both estimators.

Outputs (new folder, never overwritten): ``derivatives/<out>/sub-XX/sub-XX_loro.tsv`` (per area × held-out run ×
model: held-out SSE and SST) and ``sub-XX_sim.tsv.gz`` (per area × δ × draw: estimates; draw −1 = the real data) and ``sub-XX_linear.npz`` (per
area: per-run Gram matrices X'X and X'R, held-out bookkeeping, the generator projections X'(P_gen(δ) − P_base) and
the noise projections X'N per draw, for the joint-fit residuals (cn_m7) and for the residuals of the linearised model
with all seven terms (cn_lin), same phases), from which other generators (e.g. base + estimated-alone dS + planted dD)
are computed without refitting.

Usage::

    python -m retsupp.modeling.hp_terms_heldout_power 3 --n-draws 200
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

import retsupp.modeling.fit_fixed_group_mechanisms as ffg
from retsupp.modeling.fit_af_prf_cv_v2 import build_data_and_paradigm
from retsupp.modeling.roi_hrf import load_table as load_hrf_table
from retsupp.utils.data import Subject

ROIS = ['V1', 'V2', 'V3', 'V3AB', 'IPS', 'hV4', 'VO', 'LO', 'TO']
REF_ROOT = 'af_prf_cv_ladder_nested_full_bounded_maxsd20_reliable_roihrf_polished'
BASE_ROOT = 'af_prf_cv_ladder_nested_full_bounded_maxsd20_reliable_roihrf'
HRF_TABLE = '/shares/zne.uzh/gdehol/ds-retsupp/derivatives/paper_2026-11/roi_hrf_dog_final2.tsv'
H = 0.05
REGRESSORS = {'S0': ['g_HP', 'g_LP'], 'D0': ['g_HP_dyn', 'g_LP_dyn'], 'T0': ['g_T_dyn', 'g_T_HP_dyn'],
              'I0': ['item_scale'], 'dS': ['g_HP'], 'dD': ['g_HP_dyn'], 'dT': ['g_T_HP_dyn']}
NAMES = list(REGRESSORS)
BASE = ['S0', 'D0', 'T0', 'I0']
MODELS = {'none': BASE, 'dS': BASE + ['dS'], 'dS_dD': BASE + ['dS', 'dD'], 'dS_dD_dT': BASE + ['dS', 'dD', 'dT']}
DELTAS = (0.0, -0.02, -0.05, -0.10)


def ix(names):
    return [NAMES.index(n) for n in names]


def phase_noise(resid, phases):
    """``simulate_decode.phase_randomise`` with given phases (one set for all columns)."""
    T = resid.shape[0]
    f = np.fft.rfft(resid - resid.mean(0), axis=0)
    return np.fft.irfft(f * np.exp(1j * phases)[:, None], n=T, axis=0)


def estimates(G, c):
    """Alone (base refit, then each HP term alone) and joint estimates from a Gram matrix G and X'R c."""
    b = ix(BASE)
    beta_b = np.linalg.solve(G[np.ix_(b, b)], c[b])
    out = {}
    for k in ('dS', 'dD', 'dT'):
        j = NAMES.index(k)
        out[f'{k}_alone'] = (c[j] - G[j, b] @ beta_b) / G[j, j]
        out[f'{k}_alone_fixedbase'] = c[j] / G[j, j]   # base not refitted: on real data = the reported Σb / Σa
    for m in ('dS_dD', 'dS_dD_dT'):
        sel = ix(MODELS[m])
        beta = np.linalg.solve(G[np.ix_(sel, sel)], c[sel])
        for k, v in zip(MODELS[m], beta):
            if k.startswith('d'):
                out[f'{k}_joint_{m}'] = v
    return out


def run_roi(sub, roi, data, model, phases, n_draws):
    bold, paradigm, _ci, _di, _ti, _grid, run_meta = data

    def pred(p):
        return np.asarray(model.predict(paradigm=paradigm, parameters=p), dtype=np.float64)

    fit4, p4, _, _ = ffg.base_parameters(sub, model, roi, 'm4_TDS', None, REF_ROOT)
    fit7, p7, _, _ = ffg.base_parameters(sub, model, roi, 'm7_full', None, REF_ROOT)
    vox = np.asarray(fit4['voxel_ids'])
    assert np.array_equal(vox, np.asarray(fit7['voxel_ids'])), 'm4_TDS and m7_full voxel sets differ'
    Y = bold[:, vox].astype(np.float64)
    P0 = pred(p4)
    X, lin = [], {}
    for name, pars in REGRESSORS.items():
        up, dn = p4.copy(), p4.copy()
        for q in pars:
            up[q] += H
            dn[q] -= H
        fu, fd = pred(up), pred(dn)
        X.append((fu - fd) / (2 * H))
        lin[name] = np.abs((fu - P0) - (P0 - fd)).max() / max(np.abs(fu - fd).max(), 1e-12)
    X = np.stack(X, -1)                                  # (T, V, K)
    R = Y - P0
    runs = [np.arange(rm.start, rm.stop) for rm in run_meta]
    Gr = np.stack([np.einsum('tvk,tvl->kl', X[t], X[t]) for t in runs])
    cr = np.stack([np.einsum('tvk,tv->k', X[t], R[t]) for t in runs])
    sse0 = np.array([(R[t] ** 2).sum() for t in runs])
    sst = np.array([((Y[t] - Y[t].mean(0)) ** 2).sum() for t in runs])

    # 1a: leave one run out
    loro = []
    G, c = Gr.sum(0), cr.sum(0)
    for i, rm in enumerate(run_meta):
        Gt, ct = G - Gr[i], c - cr[i]
        for m, names in MODELS.items():
            sel = ix(names)
            beta = np.linalg.solve(Gt[np.ix_(sel, sel)], ct[sel])
            sse = sse0[i] - 2 * beta @ cr[i][sel] + beta @ Gr[i][np.ix_(sel, sel)] @ beta
            row = dict(roi=roi, session=rm.session, run=rm.run, hp=rm.hp, model=m, sse=sse, sst=sst[i],
                       sse_base_untouched=sse0[i], n_vox=len(vox))
            row.update({f'beta_{k}': v for k, v in zip(names, beta)})
            loro.append(row)

    # 1c: simulation from the joint fit
    real = estimates(G, c)
    P7 = pred(p7)
    noise_src = Y - P7
    det = {}
    for d in DELTAS:
        pg = p7.copy()
        pg['g_HP_dyn'] = pg['g_LP_dyn'] + d
        det[d] = np.einsum('tvk,tv->k', X, pred(pg) - P0)
    # second noise source: residuals of the linearised model with all seven terms (base + dS + dD + dT, real data)
    beta_all = np.linalg.solve(G, c)
    noise_lin = R - X @ beta_all
    rows = [dict(roi=roi, delta=np.nan, draw=-1, **real)]
    cn_m7, cn_lin = [], []
    for k in range(n_draws):
        N = np.concatenate([phase_noise(noise_src[t], phases[k][i]) for i, t in enumerate(runs)])
        cn = np.einsum('tvk,tv->k', X, N)
        cn_m7.append(cn)
        N2 = np.concatenate([phase_noise(noise_lin[t], phases[k][i]) for i, t in enumerate(runs)])
        cn_lin.append(np.einsum('tvk,tv->k', X, N2))
        for d in DELTAS:
            rows.append(dict(roi=roi, delta=d, draw=k, **estimates(G, det[d] + cn)))
    sim = pd.DataFrame(rows)
    sim['g_LP_dyn_m7'] = float(p7['g_LP_dyn'].iloc[0])
    sim['g_HP_dyn_m7'] = float(p7['g_HP_dyn'].iloc[0])
    sim['g_T_dyn_m7'] = float(p7['g_T_dyn'].iloc[0])
    sim['n_vox'] = len(vox)
    sim['lin_err_max'] = max(lin.values())
    sim['resid_sd_ratio'] = float(noise_src.std() / R.std())
    print(f'sub-{sub.subject_id:02d} {roi}: {len(vox)} voxels, lin_err {max(lin.values()):.1e}, real alone '
          f'dS {real["dS_alone"]:+.4f} dD {real["dD_alone"]:+.4f} dT {real["dT_alone"]:+.4f}', flush=True)
    arrays = dict(Gr=Gr, cr=cr, sse0=sse0, sst=sst, det_m7=np.stack([det[d] for d in DELTAS]),
                  deltas=np.array(DELTAS), cn_m7=np.array(cn_m7), cn_lin=np.array(cn_lin), names=np.array(NAMES),
                  n_vox=len(vox), lin_err=np.array([lin[n] for n in NAMES]))
    return pd.DataFrame(loro), sim, arrays


def main(subject, bids_folder, out_subdir, n_draws, rois):
    ffg.BASE_ROOT, ffg.OVERLAY_PER_MODEL, ffg.FAMILY = BASE_ROOT, True, 'dog'
    ffg.HRF_TABLE = load_hrf_table(HRF_TABLE)
    out = Path(bids_folder) / 'derivatives' / out_subdir / f'sub-{subject:02d}'
    f_loro, f_sim = out / f'sub-{subject:02d}_loro.tsv', out / f'sub-{subject:02d}_sim.tsv.gz'
    if f_loro.exists() and f_sim.exists():
        print('Outputs exist, nothing to do')
        return
    sub = Subject(subject, bids_folder=Path(bids_folder))
    t0 = time.time()
    bold, paradigm, ci, di, ti, grid, _m, run_meta = build_data_and_paradigm(
        sub, resolution=50, grid_radius=5.0, item_shape='rectangle')
    data = (bold, paradigm, ci, di, ti, grid, run_meta)
    print(f'data loaded in {time.time() - t0:.0f}s', flush=True)
    rng = np.random.default_rng([subject, 20261113])
    phases = []
    for rm in run_meta:
        ph = rng.uniform(0, 2 * np.pi, (n_draws, rm.n_T // 2 + 1))
        ph[:, 0] = 0
        if rm.n_T % 2 == 0:
            ph[:, -1] = 0
        phases.append(ph)
    phases = [[ph[k] for ph in phases] for k in range(n_draws)]   # [draw][run]
    lo, si, npz = [], [], {}
    for roi in rois:
        if not ffg.fit_path(bids_folder, 'm7_full', subject, roi, REF_ROOT).exists():
            print(f'{roi}: no fit, skipped')
            continue
        model = ffg.build_full_model(sub, data, roi)
        a, b, arr = run_roi(sub, roi, data, model, phases, n_draws)
        lo.append(a)
        si.append(b)
        npz.update({f'{roi}__{k}': v for k, v in arr.items()})
        print(f'{roi} done at {time.time() - t0:.0f}s', flush=True)
    out.mkdir(parents=True, exist_ok=True)
    pd.concat(lo).assign(subject=subject).to_csv(f_loro, sep='\t', index=False)
    pd.concat(si).assign(subject=subject).to_csv(f_sim, sep='\t', index=False)
    np.savez_compressed(out / f'sub-{subject:02d}_linear.npz', subject=subject, **npz)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('subject', type=int)
    ap.add_argument('--bids-folder', default='/shares/zne.uzh/gdehol/ds-retsupp')
    ap.add_argument('--output-subdir', default='hp_terms_heldout_power_2026-11-13')
    ap.add_argument('--n-draws', type=int, default=200)
    ap.add_argument('--rois', nargs='+', default=ROIS)
    a = ap.parse_args()
    main(a.subject, a.bids_folder, a.output_subdir, a.n_draws, a.rois)
