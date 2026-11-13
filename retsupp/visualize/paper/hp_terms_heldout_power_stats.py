"""Statistics and tables for items 1a (held-out prediction) and 1c (power) of writing-package update 11.

Inputs: the per-subject outputs of ``retsupp/modeling/hp_terms_heldout_power.py``, concatenated by ``collect`` (on
the cluster) into ``notes/data/hp_terms_loro_2026-11-13.tsv`` and ``notes/data/hp_terms_sim_2026-11-13.tsv.gz``.

Statistics as key_numbers.md: median over participants, bootstrapped 95% CI (``fig45_common.median_ci``), k of n,
two-sided Wilcoxon; per area BH across the nine areas; pooled = per participant the median over the nine areas; per
pathway = per participant the median over the pathway's areas, BH across the three. sub-01 and sub-02 are not used.

Usage::

    python -m retsupp.visualize.paper.hp_terms_heldout_power_stats collect --bids-folder /shares/... (cluster)
    python -m retsupp.visualize.paper.hp_terms_heldout_power_stats stats (laptop)
"""
from __future__ import annotations

import argparse
import glob

import numpy as np
import pandas as pd
from scipy import stats

from retsupp.visualize.paper.fig45_common import median_ci
from retsupp.visualize.paper.paper_style import ROI_GROUPS, ROI_ORDER

D = 'notes/data'
LORO = f'{D}/hp_terms_loro_2026-11-13.tsv'
SIM = f'{D}/hp_terms_sim_2026-11-13.tsv.gz'
EXCL = (1, 2)
CONTRASTS = [('dS', 'none', 'Sustained HP − LP (dS) vs no HP terms'),
             ('dS_dD', 'dS', 'Adding the distractor transient HP − LP (dD) to dS'),
             ('dS_dD_dT', 'dS_dD', 'Adding the target transient HP − LP (dT) to dS + dD'),
             ('dS_dD_dT', 'dS', 'Adding both transient terms (dD + dT) to dS'),
             ('dS_dD_dT', 'none', 'All three HP − LP terms vs no HP terms')]


def collect(bids, folder):
    for kind, out in (('loro.tsv', LORO), ('sim.tsv.gz', SIM)):
        files = sorted(glob.glob(f'{bids}/derivatives/{folder}/sub-*/sub-*_{kind}'))
        df = pd.concat([pd.read_csv(f, sep='\t') for f in files])
        df.to_csv(out, sep='\t', index=False)
        print(out, len(files), 'subjects', len(df), 'rows')


def sci(p):
    if not np.isfinite(p):
        return 'n/a'
    if p >= .001:
        return (f'{p:.2g}' if p < .01 else f'{p:.3f}' if p < .1 else f'{p:.2f}').replace('0.', '.')
    m, e = f'{p:.1e}'.split('e-')
    return f'{m} × 10' + '⁻' + ''.join('⁰¹²³⁴⁵⁶⁷⁸⁹'[int(c)] for c in str(int(e)))


def summarise(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    m, lo, hi = median_ci(v)
    return dict(median=m, ci_lo=lo, ci_hi=hi, k_pos=int((v > 0).sum()), k_neg=int((v < 0).sum()), n=len(v),
                p=stats.wilcoxon(v).pvalue)


def bh(rows):
    for r, q in zip(rows, stats.false_discovery_control([r['p'] for r in rows])):
        r['q'] = q
    return rows


def group_rows(df, col):
    """Per area (BH over nine), pooled, per pathway (BH over three) for one per (subject, roi) column."""
    rows = bh([dict(level='area', area=r, **summarise(df[df.roi == r][col])) for r in ROI_ORDER])
    rows.append(dict(level='pooled', area='pooled', **summarise(df.groupby('subject')[col].median()), q=np.nan))
    rows += bh([dict(level='pathway', area=n, **summarise(df[df.roi.isin(rs)].groupby('subject')[col].median()))
                for n, rs in ROI_GROUPS])
    return rows


def heldout(md, scale=1e4):
    lo = pd.read_csv(LORO, sep='\t')
    lo = lo[~lo.subject.isin(EXCL)]
    agg = lo.groupby(['subject', 'roi', 'model'])[['sse', 'sst']].sum().reset_index()
    agg['r2'] = 1 - agg.sse / agg.sst
    w = agg.pivot_table(index=['subject', 'roi'], columns='model', values='r2').reset_index()
    out = []
    md += ['# Item 1a: do the transient HP − LP terms improve held-out prediction?', '',
           'Leave-one-run-out cross-validation (12 folds; 11 for sub-20 and sub-24): in each fold the base terms '
           '(item weight, target gain, distractor gain, sustained gain at all four locations) and the HP − LP terms '
           'of each model were fitted on the other runs and the held-out run was predicted. pRFs and field widths: '
           'the all-runs fit of the model with item weight, target, distractor and sustained gain (m4_TDS) in the '
           'reference root (pRFs refitted together with the gain fields, update 8, '
           '`af_prf_cv_ladder_nested_full_bounded_maxsd20_reliable_roihrf_polished`; that polish was not run to full '
           'convergence), identical in every model, so their leak into the held-out run is shared. The HP − LP terms '
           'enter linearly (least squares around that fit). Held-out R² per participant × area = 1 − ΣSSE / ΣSST over '
           'the held-out runs and voxels; ΔR² in units of 10⁻⁴ (0.01 percentage points of variance). Per area: median '
           'over participants [bootstrapped 95% CI], participants with ΔR² > 0, Wilcoxon p, BH q across nine areas; '
           'pooled: per participant the median over the nine areas; pathway: median over its areas, BH across three.',
           '', f'Held-out R² of the model without HP terms (base): median over participants × areas '
           f'{np.median(w["none"]):.3f}.', '']
    for a, b, label in CONTRASTS:
        w['d'] = (w[a] - w[b]) * scale
        rows = group_rows(w, 'd')
        for r in rows:
            r.update(contrast=f'{a} − {b}', label=label)
        out += rows
        md += [f'## {label} (ΔR² = R²[{a.replace("_", " + ")}] − R²[{b.replace("_", " + ")}], × 10⁻⁴)', '',
               '| Area | Median ΔR² [95% CI] | k > 0 / n | p | q |', '|---|---|---|---|---|']
        for r in rows:
            name = r['area'] if r['level'] != 'pathway' else f"{r['area']} (pathway)"
            md.append(f"| {name if r['level'] != 'pooled' else '**Pooled**'} | {r['median']:+.2f} "
                      f"[{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}] | {r['k_pos']}/{r['n']} | {sci(r['p'])} | "
                      f"{sci(r['q'])} |")
        pl = next(r for r in rows if r['level'] == 'pooled')
        md += ['', f"*Summary:* {plain_heldout(a, b, pl, rows)}", '']
    # which model each participant prefers (pooled over areas: median R² gain over the base)
    piv = w.groupby('subject')[['none', 'dS', 'dS_dD', 'dS_dD_dT']].median()
    best = piv.sub(piv['none'], axis=0).idxmax(axis=1).value_counts()
    md += ['## Model with the best held-out prediction per participant (median over areas of R² − R²[base])', '',
           '| Model | Participants |', '|---|---|']
    for m in ('none', 'dS', 'dS_dD', 'dS_dD_dT'):
        md.append(f'| {MODEL_LABEL[m]} | {int(best.get(m, 0))} |')
    md += ['', '*Summary:* how the participants split over the four models; a model counts for a participant when '
           'its median held-out gain over the nine areas is the largest.', '']
    return pd.DataFrame(out), w


MODEL_LABEL = {'none': 'No HP terms (item weight, target, distractor, sustained gain)', 'dS': 'Sustained HP − LP only (dS)',
               'dS_dD': 'dS + distractor transient HP − LP (dD) (the HP terms of the joint fit)',
               'dS_dD_dT': 'dS + dD + target transient HP − LP (dT)'}


def plain_heldout(a, b, pl, rows):
    sig = [r['area'] for r in rows if r['level'] == 'area' and r['q'] < .05]
    sigp = [r['area'] for r in rows if r['level'] == 'pathway' and r['q'] < .05]
    better = pl['median'] > 0
    s = (f"pooled over areas, the richer model predicts the held-out run {'better' if better else 'worse'} in "
         f"{pl['k_pos']} of {pl['n']} participants (median ΔR² {pl['median']:+.2f} × 10⁻⁴, p = {sci(pl['p'])})")
    s += f"; areas with q < .05: {', '.join(sig) if sig else 'none'}"
    s += f"; pathways with q < .05: {', '.join(sigp) if sigp else 'none'}."
    return s[0].upper() + s[1:]


def power(md, n_boot_ci=False):
    sim = pd.read_csv(SIM, sep='\t')
    sim = sim[~sim.subject.isin(EXCL)]
    est = [c for c in sim.columns if c.startswith(('dS_', 'dD_', 'dT_'))]
    sim[est] = sim[est] * 100
    sim['dDmT_alone'] = sim.dD_alone - sim.dT_alone
    sim['dDmT_alone_fixedbase'] = sim.dD_alone_fixedbase - sim.dT_alone_fixedbase
    sim['dDmT_joint'] = sim.dD_joint_dS_dD_dT - sim.dT_joint_dS_dD_dT
    real = sim[sim.draw == -1]
    s = sim[sim.draw >= 0]
    tests = [('dD_alone', 'dD estimated alone (base refitted to each data set) < 0', -1),
             ('dD_alone_fixedbase', 'dD estimated alone, base held at the real-data fit < 0', -1),
             ('dDmT_alone', 'dD − dT estimated alone (base refitted) ≠ 0', 0),
             ('dDmT_alone_fixedbase', 'dD − dT estimated alone, base held at the real-data fit ≠ 0', 0),
             ('dD_joint_dS_dD', 'dD in the joint fit (dS + dD) < 0', -1),
             ('dD_joint_dS_dD_dT', 'dD in the joint fit with dT (dS + dD + dT) < 0', -1),
             ('dDmT_joint', 'dD − dT in the joint fit with dT ≠ 0', 0)]
    rows = []
    for (d, k), g in s.groupby(['delta', 'draw']):
        r = dict(delta=d, draw=k)
        for col, _l, sign in tests:
            per = g.groupby('subject')[col].median()
            p = stats.wilcoxon(per).pvalue
            m = per.median()
            r[f'{col}_median'] = m
            r[f'{col}_p'] = p
            r[f'{col}_det'] = (p < .05) and (sign == 0 or np.sign(m) == sign)
            r[f'{col}_det_neg'] = (p < .05) and m < 0
            # per area, BH across nine
            ps, ms = [], []
            for roi in ROI_ORDER:
                v = g[g.roi == roi][col]
                ps.append(stats.wilcoxon(v).pvalue)
                ms.append(v.median())
            q = stats.false_discovery_control(ps)
            r[f'{col}_any_area'] = bool(np.any((q < .05) & ((np.sign(ms) == sign) if sign else True)))
            pw = []
            for _n, rs in ROI_GROUPS:
                v = g[g.roi.isin(rs)].groupby('subject')[col].median()
                pw.append((stats.wilcoxon(v).pvalue, v.median()))
            qp = stats.false_discovery_control([x[0] for x in pw])
            r[f'{col}_any_pathway'] = bool(np.any((qp < .05) & np.array(
                [(np.sign(x[1]) == sign) if sign else True for x in pw])))
        rows.append(r)
    pr = pd.DataFrame(rows)
    n_draws = pr.draw.nunique()
    # distractor gain (joint fit) for the relative scale
    gd = real.groupby('subject').g_LP_dyn_m7.median() * 100
    obs = {}
    for col, _l, _s in tests:
        per = real.groupby('subject')[col].median()
        obs[col] = (per.median(), stats.wilcoxon(per).pvalue)
    md += ['# Item 1c: which distractor-specific HP reduction could we have detected?', '',
           f'Simulation per participant (n = {s.subject.nunique()}) and area (all nine), {n_draws} simulated data sets '
           'per level. Generator: the joint fit (all terms fitted together, m7_full, reference root of update 8: pRFs '
           'refitted together with the gain fields, `af_prf_cv_ladder_nested_full_bounded_maxsd20_reliable_roihrf_'
           'polished`; not run to full convergence), with the distractor gain at HP set to the fitted distractor gain '
           'at LP plus δ, so the planted distractor-specific HP − LP difference is dD = δ; no target-specific term '
           '(dT = 0); the fitted sustained HP − LP term (dS) kept. **δ is in the unit of the reported dD: percentage '
           'points of the transient gain** (gain 1 = 100%, so δ = −5% means the distractor at HP is enhanced 5 points '
           'less than at LP). For scale: the joint fit\'s distractor gain at LP, per participant the median over areas, '
           f'has a group median of {gd.median():+.1f}%, so δ = −2 / −5 / −10% is {abs(2 / gd.median()):.0%} / '
           f'{abs(5 / gd.median()):.0%} / {abs(10 / gd.median()):.0%} of a typical distractor gain. Noise: the joint '
           'fit\'s residuals, phase-randomised per run with one set of random phases for all voxels and areas of a '
           'participant (keeps each voxel\'s spectrum and the covariance across voxels and areas; as '
           '`retsupp/decode/simulate_decode.py`).', '',
           'Refit: (1) as the reported dD and dT: the model with item weight, target, distractor and sustained gain '
           '(m4_TDS) refitted to the simulated data (its four gain terms; pRFs and field widths held at the real-data '
           'fit), then dD, and separately dT, added alone ("estimated alone"); (2) all HP − LP terms fitted together '
           'with the base terms (dS + dD, the HP terms of the joint fit; and dS + dD + dT). The terms enter linearly, '
           'so these are least-squares fits around the real-data m4_TDS fit; the pRFs and widths were not refitted per '
           'simulated data set (the full alternating polish takes ~45 min per participant × area). Detection: the '
           'paper\'s pooled test, per participant the median over the nine areas, two-sided Wilcoxon across '
           'participants, p < .05 in the planted direction (for dD − dT: p < .05 either direction). Also: any area '
           'with BH q < .05 (nine areas) and any pathway with BH q < .05 (three). Power = share of simulated data '
           'sets; ± is the binomial 95% CI half-width.', '']
    out_rows = []
    for col, label, sign in tests:
        md += [f'## {label}', '',
               '| Planted dD | Pooled test detects | Any area (BH) | Any pathway (BH) | Group median of the estimate '
               '[2.5–97.5% over data sets] | Bias (median − planted) |', '|---|---|---|---|---|---|']
        for d, g in pr.groupby('delta', sort=False):
            n = len(g)
            pw_ = g[f'{col}_det'].mean()
            hw = 1.96 * np.sqrt(pw_ * (1 - pw_) / n)
            med = g[f'{col}_median']
            truth = d * 100   # planted dD; dT = 0 in the generator, so also the planted dD − dT
            bias = med.median() - truth
            out_rows.append(dict(test=label, column=col, delta_pct=d * 100, power=pw_, power_ci_halfwidth=hw,
                                 any_area=g[f'{col}_any_area'].mean(), any_pathway=g[f'{col}_any_pathway'].mean(),
                                 det_neg=g[f'{col}_det_neg'].mean(), median_estimate=med.median(),
                                 est_lo=med.quantile(.025), est_hi=med.quantile(.975), bias=bias, n_draws=n,
                                 observed=obs[col][0], observed_p=obs[col][1]))
            md.append(f'| {d * 100:+.0f}% | {pw_:.0%} ± {hw:.0%} | {g[f"{col}_any_area"].mean():.0%} | '
                      f'{g[f"{col}_any_pathway"].mean():.0%} | {med.median():+.2f}% [{med.quantile(.025):+.2f}, '
                      f'{med.quantile(.975):+.2f}] | {bias:+.2f} |')
        o = obs[col]
        md += ['', f'Real data, same estimator (pooled): group median {o[0]:+.2f}%, p = {sci(o[1])}.', '']
        md += [f'*Summary:* {plain_power(label, [r for r in out_rows if r["column"] == col])}', '']
    return pd.DataFrame(out_rows), pr


def plain_power(label, rows):
    by = {round(r['delta_pct']): r for r in rows}
    s = (f"with no planted effect the test fires in {by[0]['power']:.0%} of data sets; it detects a planted "
         f"−2 / −5 / −10% in {by[-2]['power']:.0%} / {by[-5]['power']:.0%} / {by[-10]['power']:.0%}")
    ok = [d for d in (-2, -5, -10) if by[d]['power'] >= .8]
    s += (f"; 80% power is reached from {ok[0]:+d}%" if ok else '; 80% power is not reached at −10%') + '.'
    return s[0].upper() + s[1:]


def main_stats(out_dir):
    md = []
    ho, w = heldout(md)
    ho.to_csv(f'{D}/hp_terms_heldout_stats_2026-11-13.tsv', sep='\t', index=False)
    w.drop(columns='d').to_csv(f'{D}/hp_terms_heldout_r2_2026-11-13.tsv', sep='\t', index=False)
    md += ['Sources: `notes/data/hp_terms_loro_2026-11-13.tsv` (per participant × area × held-out run × model: '
           'held-out SSE and SST), `notes/data/hp_terms_heldout_r2_2026-11-13.tsv` (held-out R² per participant × '
           'area × model), `notes/data/hp_terms_heldout_stats_2026-11-13.tsv` (every number above). Code: '
           '`retsupp/modeling/hp_terms_heldout_power.py` (cluster, `retsupp/modeling/slurm_jobs/hp_terms_heldout_power.sh`, '
           'output `derivatives/hp_terms_heldout_power_2026-11-13`), '
           '`retsupp/visualize/paper/hp_terms_heldout_power_stats.py` (tables).']
    open(f'{out_dir}/item1a_heldout.md', 'w').write('\n'.join(md) + '\n')
    md = []
    pw, pr = power(md)
    pw.to_csv(f'{D}/hp_terms_power_2026-11-13.tsv', sep='\t', index=False)
    pr.to_csv(f'{D}/hp_terms_power_per_draw_2026-11-13.tsv', sep='\t', index=False)
    md += ['Sources: `notes/data/hp_terms_sim_2026-11-13.tsv.gz` (per participant × area × planted δ × simulated data '
           'set: every estimate; draw −1 = real data), `notes/data/hp_terms_power_per_draw_2026-11-13.tsv` (group '
           'tests per data set), `notes/data/hp_terms_power_2026-11-13.tsv` (the tables above). Code: '
           '`retsupp/modeling/hp_terms_heldout_power.py`, `retsupp/visualize/paper/hp_terms_heldout_power_stats.py`.']
    open(f'{out_dir}/item1c_sensitivity.md', 'w').write('\n'.join(md) + '\n')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('what', choices=['collect', 'stats'])
    ap.add_argument('--bids-folder', default='/shares/zne.uzh/gdehol/ds-retsupp')
    ap.add_argument('--folder', default='hp_terms_heldout_power_2026-11-13')
    ap.add_argument('--out-dir', default='notes/writing_package_2026-11-13_update11/parts')
    a = ap.parse_args()
    if a.what == 'collect':
        collect(a.bids_folder, a.folder)
    else:
        main_stats(a.out_dir)
