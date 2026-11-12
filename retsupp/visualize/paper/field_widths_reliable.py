"""Field widths conditional on a reliable gain (2026-11-12): offline analysis of ``fit_width_profiles``.

From the per-run sufficient statistics (A_r, b_r, c_r) of each subject x ROI (final joint fit; the gains of one field
re-solved by least squares, everything else fixed at the joint fit):

1. Gains at the fitted widths and their leave-one-run-out jackknife SE (sustained at HP and at LP, target, distractor =
   mean of the distractor gains at HP and at LP). Calibration: half the odd − even-run difference over the jackknife
   SE (expected median 0.67 when the SE is right). Reliable: |gain| / SE > 2 (and > 3).
2. Widths (FWHM = 2.355 σ of the joint fit) among fits with a reliable gain, next to all fits; share at a bound of the
   fit (σ ≤ 0.51° or ≥ 7.9°).
3. Profile over the width: SSE(σ) = Σ c − b'A⁻¹b on a log grid (gains re-solved, all else fixed); the profile minimum
   (parabola in log σ through the three grid points around the minimum) and a 95% interval from the leave-one-run-out
   jackknife of the log profile minimum (exp(log σ̂ ± 1.96 SE)). The jackknife is used because the residuals are
   correlated over time and voxels, so a χ² threshold on ΔSSE with the nominal residual variance would be far too
   narrow. A minimum at the grid edge (0.5° or 8°) is flagged: the interval is then open on that side.
4. Gain–width trade-off (Spearman across fits; repeated-measures correlation within participants) and gain × field
   volume (g · 2πσ², the integral of the gain field) per area.
5. The two claims: sustained FWHM < 5.7° (the distance between neighbouring possible item locations) and early
   (V1–V3) wider than higher areas, among fits with a reliable sustained gain.

    python -m retsupp.visualize.paper.field_widths_reliable --root /data/ds-retsupp/derivatives/field_widths_reliable_2026-11-12
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROIS = ['V1', 'V2', 'V3', 'V3AB', 'IPS', 'hV4', 'VO', 'LO', 'TO']
EARLY = ['V1', 'V2', 'V3']
F = 2 * np.sqrt(2 * np.log(2))
SPACING = 4 * np.sqrt(2)          # 5.66°: between neighbouring possible target/distractor locations
LO_B, HI_B = .51, 7.9


def solve(A, b):
    return np.linalg.solve(A, b)


def jackknife(A, b, f=lambda g: g):
    """Full-data value, leave-one-run-out values and jackknife SE of f(gains)."""
    R = len(A)
    full = f(solve(A.sum(0), b.sum(0)))
    loo = np.stack([f(solve(A.sum(0) - A[r], b.sum(0) - b[r])) for r in range(R)])
    se = np.sqrt((R - 1) / R * ((loo - loo.mean(0)) ** 2).sum(0))
    return full, loo, se


def profile_min(grid, sse):
    """Minimum of SSE over log σ: parabola through the grid point at the minimum and its neighbours."""
    lg = np.log(grid)
    i = int(np.argmin(sse))
    if i in (0, len(grid) - 1):
        return float(grid[i]), True
    x, y = lg[i - 1:i + 2], sse[i - 1:i + 2]
    a, b, _ = np.polyfit(x, y, 2)
    xm = -b / (2 * a) if a > 0 else lg[i]
    return float(np.exp(np.clip(xm, x[0], x[-1]))), False


def sse_curve(A, b, c):
    """SSE(σ) for one subset of runs: A (S, k, k), b (S, k), c (S,) summed over the runs already."""
    return np.array([c[s] - b[s] @ np.linalg.solve(A[s], b[s]) for s in range(len(c))])


def profile(Ag, bg, cg, grid):
    """Profile minimum on all runs and its leave-one-run-out jackknife interval. Ag: (S, R, k, k)."""
    R = Ag.shape[1]
    full, edge = profile_min(grid, sse_curve(Ag.sum(1), bg.sum(1), cg.sum(1)))
    loo = np.array([profile_min(grid, sse_curve(Ag.sum(1) - Ag[:, r], bg.sum(1) - bg[:, r], cg.sum(1) - cg[:, r]))[0]
                    for r in range(R)])
    se = np.sqrt((R - 1) / R * ((np.log(loo) - np.log(loo).mean()) ** 2).sum())
    lo, hi = np.exp(np.log(full) - 1.96 * se), np.exp(np.log(full) + 1.96 * se)
    if edge:   # minimum at a grid edge: the interval is open on that side and reaches at least the furthest LOO minimum
        if full <= grid[0]:
            lo, hi = grid[0], max(hi, loo.max())
        else:
            lo, hi = min(lo, loo.min()), grid[-1]
    return full, edge, se, max(lo, grid[0]), min(hi, grid[-1])


def collect(root, exclude=(1, 2)):
    rows = []
    for f in sorted(glob.glob(f'{root}/sub-*/*_widthprof.npz')):
        z = np.load(f, allow_pickle=True)
        s, roi = int(z['subject']), str(z['roi'])
        if s in exclude:
            continue
        names = list(z['gain_names'])
        A, b = z['A'], z['b']
        full, loo, se = jackknife(A, b)
        dfun = lambda g: np.array([(g[names.index('g_D_HP')] + g[names.index('g_D_LP')]) / 2])
        dfull, _dl, dse = jackknife(A, b, dfun)
        runs = z['runs']
        odd = runs[:, 1] % 2 == 1
        g_odd, g_even = solve(A[odd].sum(0), b[odd].sum(0)), solve(A[~odd].sum(0), b[~odd].sum(0))
        r = dict(subject=s, roi=roi, n_voxels=int(z['n_voxels']), n_runs=len(A),
                 sigma_AF=float(z['sigma_AF']), sigma_dyn=float(z['sigma_dyn']), lin_err=float(z['lin_err'].max()))
        for k, n in enumerate(names):
            r[n], r[f'{n}_se'], r[f'{n}_fit'] = full[k], se[k], float(z['fitted_gains'][k])
            r[f'{n}_halfdiff'] = abs(g_odd[k] - g_even[k]) / 2
        r['g_D'], r['g_D_se'] = float(dfull[0]), float(dse[0])
        grid = z['grid']
        for fld in ('sus', 'dyn'):
            m, edge, sel, lo, hi = profile(z[f'{fld}_A'], z[f'{fld}_b'], z[f'{fld}_c'], grid)
            r.update({f'{fld}_prof_sigma': m, f'{fld}_prof_edge': edge, f'{fld}_prof_selog': sel,
                      f'{fld}_prof_lo': lo, f'{fld}_prof_hi': hi})
        rows.append(r)
    d = pd.DataFrame(rows)
    d['fwhm_sus'], d['fwhm_dyn'] = F * d.sigma_AF, F * d.sigma_dyn
    d['bound_sus'] = (d.sigma_AF <= LO_B) | (d.sigma_AF >= HI_B)
    d['bound_dyn'] = (d.sigma_dyn <= LO_B) | (d.sigma_dyn >= HI_B)
    for n in ('g_HP', 'g_LP', 'g_T', 'g_D'):
        d[f'{n}_z'] = d[n] / d[f'{n}_se']
    d['sus_z'] = np.maximum(d.g_HP_z.abs(), d.g_LP_z.abs())
    d['prof_width_sus'] = F * (d.sus_prof_hi - d.sus_prof_lo)       # width of the 95% interval, in FWHM degrees
    d['prof_width_dyn'] = F * (d.dyn_prof_hi - d.dyn_prof_lo)
    # a minimum at the grid edge leaves the width unconstrained on that side: an open (infinite) interval
    d.loc[d.sus_prof_edge.astype(bool), 'prof_width_sus'] = np.inf
    d.loc[d.dyn_prof_edge.astype(bool), 'prof_width_dyn'] = np.inf
    return d


def boot_ci(x, n=10000, seed=0):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    m = np.median(rng.choice(x, (n, len(x))), 1)
    return tuple(np.percentile(m, [2.5, 97.5]))


def summarise(d, thr):
    out = []
    for fld, col, rel, bnd in (('sustained', 'fwhm_sus', d.sus_z > thr, 'bound_sus'),
                               ('transient', 'fwhm_dyn', d.g_T_z.abs() > thr, 'bound_dyn')):
        for roi in ROIS + ['all']:
            m = (d.roi == roi) if roi != 'all' else np.ones(len(d), bool)
            for which, keep in (('all fits', m), (f'reliable (|z| > {thr})', m & rel)):
                x = d.loc[keep, col]
                if roi == 'all':      # pooled: per-participant median over areas
                    x = d.loc[keep].groupby('subject')[col].median()
                lo, hi = boot_ci(x)
                out.append(dict(field=fld, roi=roi, subset=which, n=len(x), median=x.median(), ci_lo=lo, ci_hi=hi,
                                share_bound=d.loc[keep, bnd].mean() if keep.any() else np.nan))
    return pd.DataFrame(out)


def wil(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x) & (x != 0)]
    return stats.wilcoxon(x).pvalue if len(x) > 5 else np.nan


def bh(p):
    p = np.asarray(p, float)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if ok.any():
        q[ok] = stats.false_discovery_control(p[ok])
    return q


def claims(d, thr):
    rel = d[d.sus_z > thr]
    rows = []
    for roi in ROIS:
        x = rel[rel.roi == roi].fwhm_sus
        rows.append(dict(roi=roi, n=len(x), share_below=(x < SPACING).mean() if len(x) else np.nan,
                         median=x.median(), p=wil(x - SPACING)))
    t = pd.DataFrame(rows)
    t['q'] = bh(t.p)
    pooled = rel.groupby('subject').fwhm_sus.median()
    t = pd.concat([t, pd.DataFrame([dict(roi='all (participant medians)', n=len(pooled),
                                         share_below=(pooled < SPACING).mean(), median=pooled.median(),
                                         p=wil(pooled - SPACING), q=np.nan)])])
    e = rel[rel.roi.isin(EARLY)].groupby('subject').fwhm_sus.median()
    h = rel[~rel.roi.isin(EARLY)].groupby('subject').fwhm_sus.median()
    both = pd.concat([e.rename('early'), h.rename('higher')], axis=1).dropna()
    diff = both.early - both.higher
    early_fits = {r: int((rel.roi == r).sum()) for r in EARLY}
    return t, dict(n_pairs=len(both), early_median=both.early.median(), higher_median=both.higher.median(),
                   diff_median=diff.median(), k_pos=int((diff > 0).sum()), p=wil(diff), early_fits=early_fits,
                   n_early_reliable_participants=int(e.notna().sum()))


def tradeoff(d):
    import pingouin as pg
    rows = []
    for fld, g, s in (('sustained (mean of 4 locations)', None, 'sigma_AF'), ('target', 'g_T', 'sigma_dyn'),
                      ('distractor', 'g_D', 'sigma_dyn')):
        x = d.copy()
        x['gain'] = (x.g_HP + 3 * x.g_LP) / 4 if g is None else x[g]
        rho, p = stats.spearmanr(x.gain, x[s])
        rm = pg.rm_corr(data=x, x='gain', y=s, subject='subject')
        rows.append(dict(field=fld, spearman_rho=rho, spearman_p=p, rm_r=float(rm.r.iloc[0]), rm_p=float(rm.pval.iloc[0]),
                         n_fits=len(x)))
    return pd.DataFrame(rows)


def volume(d):
    rows = []
    for fld, g, s in (('sustained at HP', 'g_HP', 'sigma_AF'), ('sustained at LP', 'g_LP', 'sigma_AF'),
                      ('target', 'g_T', 'sigma_dyn'), ('distractor', 'g_D', 'sigma_dyn')):
        v = d[g] * 2 * np.pi * d[s] ** 2
        sub = []
        for roi in ROIS:
            x = v[d.roi == roi]
            lo, hi = boot_ci(x)
            sub.append(dict(field=fld, roi=roi, n=len(x), median=x.median(), ci_lo=lo, ci_hi=hi,
                            k_pos=int((x > 0).sum()), p=wil(x)))
        t = pd.DataFrame(sub)
        t['q'] = bh(t.p)
        pooled = v.groupby(d.subject).median()
        lo, hi = boot_ci(pooled)
        rows.append(pd.concat([t, pd.DataFrame([dict(field=fld, roi='all (participant medians)', n=len(pooled),
                                                     median=pooled.median(), ci_lo=lo, ci_hi=hi,
                                                     k_pos=int((pooled > 0).sum()), p=wil(pooled), q=np.nan)])]))
    return pd.concat(rows)


def main(root, out_prefix):
    d = collect(root)
    d.to_csv(f'{out_prefix}_fits.tsv', sep='\t', index=False)
    calib = {n: float(np.median(d[f'{n}_halfdiff'] / d[f'{n}_se'])) for n in ('g_HP', 'g_LP', 'g_T')}
    print('Fits:', len(d), 'participants:', d.subject.nunique(), '| max lin_err', d.lin_err.max())
    print('Calibration, median |odd − even|/2 / SE_jk (expect ≈ 0.67):', {k: round(v, 2) for k, v in calib.items()})
    for n in ('g_HP', 'g_LP', 'g_T'):
        print(f'{n}: re-solved vs fitted, median |diff| {np.median(np.abs(d[n] - d[n + "_fit"])):.4f}')
    res = {}
    for thr in (2, 3):
        s = summarise(d, thr)
        s.to_csv(f'{out_prefix}_widths_thr{thr}.tsv', sep='\t', index=False)
        t, pair = claims(d, thr)
        t.to_csv(f'{out_prefix}_claims_thr{thr}.tsv', sep='\t', index=False)
        res[thr] = (s, t, pair)
        print(f'\n== threshold {thr}')
        print(s.round(3).to_string(index=False))
        print(t.round(4).to_string(index=False))
        print('early vs higher (participant medians):', {k: (round(v, 4) if isinstance(v, float) else v) for k, v in pair.items()})
    tr = tradeoff(d)
    tr.to_csv(f'{out_prefix}_tradeoff.tsv', sep='\t', index=False)
    print(tr.round(4).to_string(index=False))
    vo = volume(d)
    vo.to_csv(f'{out_prefix}_volume.tsv', sep='\t', index=False)
    print(vo.round(4).to_string(index=False))
    pd.DataFrame([calib]).to_csv(f'{out_prefix}_calibration.tsv', sep='\t', index=False)
    return d, res


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', default='/data/ds-retsupp/derivatives/field_widths_reliable_2026-11-12')
    ap.add_argument('--out-prefix', default='notes/data/field_widths_reliable_n28_2026-11-12')
    a = ap.parse_args()
    main(a.root, a.out_prefix)
