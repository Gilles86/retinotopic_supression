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
LINEAR = f'{D}/hp_terms_linear_2026-11-13.npz'
FOLDER = 'hp_terms_heldout_power_v2_2026-11-13'
EXCL = (1, 2)
CONTRASTS = [('dS', 'none', 'Sustained HP − LP (dS) vs no HP terms'),
             ('dS_dD', 'dS', 'Adding the distractor transient HP − LP (dD) to dS'),
             ('dS_dD_dT', 'dS_dD', 'Adding the target transient HP − LP (dT) to dS + dD'),
             ('dS_dD_dT', 'dS', 'Adding both transient terms (dD + dT) to dS'),
             ('dS_dD_dT', 'none', 'All three HP − LP terms vs no HP terms')]


def collect(bids, folder):
    files = sorted(glob.glob(f'{bids}/derivatives/{folder}/sub-*/sub-*_loro.tsv'))
    df = pd.concat([pd.read_csv(f, sep='\t') for f in files])
    df.to_csv(LORO, sep='\t', index=False)
    print(LORO, len(files), 'subjects', len(df), 'rows')
    arrs = {}
    for f in sorted(glob.glob(f'{bids}/derivatives/{folder}/sub-*/sub-*_linear.npz')):
        z = np.load(f)
        arrs.update({f"{int(z['subject'])}__{k}": z[k] for k in z.files if k != 'subject'})
    np.savez_compressed(LINEAR, **arrs)
    print(LINEAR, len(arrs), 'arrays')


def load_linear_table(path=LINEAR):
    z = np.load(path)
    out = {}
    for k in z.files:
        subj, roi, name = k.split('__')
        if int(subj) in EXCL:
            continue
        out.setdefault((int(subj), roi), {})[name] = z[k]
    return out


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
    gain = w[['subject']].copy()
    for m in ('none', 'dS', 'dS_dD', 'dS_dD_dT'):
        gain[m] = w[m] - w['none']
    best = gain.groupby('subject').median().idxmax(axis=1).value_counts()
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
    s = (f"pooled over areas, the model with the added term(s) predicts the held-out run better in {pl['k_pos']} of "
         f"{pl['n']} participants (median ΔR² {pl['median']:+.2f} × 10⁻⁴, p = {sci(pl['p'])})")
    s += f"; areas with q < .05: {', '.join(sig) if sig else 'none'}"
    s += f"; pathways with q < .05: {', '.join(sigp) if sigp else 'none'}."
    return s[0].upper() + s[1:]


NAMES = ['S0', 'D0', 'T0', 'I0', 'dS', 'dD', 'dT']
BASE = [0, 1, 2, 3]
JOINT = {'dS_dD': [0, 1, 2, 3, 4, 5], 'dS_dD_dT': [0, 1, 2, 3, 4, 5, 6]}


def estimates_batch(G, c):
    """G (K, K), c (..., K) -> dict of estimates (...,): alone (base refit / base fixed) and joint."""
    b = BASE
    beta_b = np.linalg.solve(G[np.ix_(b, b)], c[..., b].T).T
    out = {}
    for k, j in (('dS', 4), ('dD', 5), ('dT', 6)):
        out[f'{k}_alone'] = (c[..., j] - beta_b @ G[b, j]) / G[j, j]
        out[f'{k}_alone_fixedbase'] = c[..., j] / G[j, j]
    for m, sel in JOINT.items():
        beta = np.linalg.solve(G[np.ix_(sel, sel)], c[..., sel].T).T
        for k, j in (('dS', 4), ('dD', 5), ('dT', 6)):
            if j in sel:
                out[f'{k}_joint_{m}'] = beta[..., sel.index(j)]
    out['dDmT_alone'] = out['dD_alone'] - out['dT_alone']
    out['dDmT_alone_fixedbase'] = out['dD_alone_fixedbase'] - out['dT_alone_fixedbase']
    out['dDmT_joint'] = out['dD_joint_dS_dD_dT'] - out['dT_joint_dS_dD_dT']
    return out


def simulate(lin, deltas=(0.0, -0.02, -0.05, -0.10)):
    """Estimates per (generator, subject, roi, δ, draw), in % (gain × 100).

    Generators: 'alone' = the model on which dD is estimated alone (m4_TDS, real-data fit) + the participant's
    estimated-alone dS + planted dD = δ (dT = 0), noise from the residuals of the linearised model with all seven
    terms; 'joint' = the joint fit (m7_full) with its distractor gain at HP set to g_LP_dyn + δ (dT = 0), noise from
    its residuals. Both use the same random phases."""
    rows, real = [], []
    for (subj, roi), z in lin.items():
        G, c = z['Gr'].sum(0), z['cr'].sum(0)
        r = estimates_batch(G, c[None])
        real.append(dict(subject=subj, roi=roi, **{k: v[0] * 100 for k, v in r.items()}))
        dS_alone = r['dS_alone_fixedbase'][0]
        for gen in ('alone', 'joint'):
            for i, d in enumerate(z['deltas']):
                if gen == 'alone':
                    cc = G[:, 4] * dS_alone + G[:, 5] * d + z['cn_lin']
                else:
                    cc = z['det_m7'][i] + z['cn_m7']
                e = estimates_batch(G, cc)
                df = pd.DataFrame({k: v * 100 for k, v in e.items()})
                df['draw'] = np.arange(len(df))
                df['generator'], df['subject'], df['roi'], df['delta'] = gen, subj, roi, d * 100
                rows.append(df)
    return pd.concat(rows, ignore_index=True), pd.DataFrame(real)


TESTS = [('dD_alone_fixedbase', 'dD estimated alone, as reported (base held at its fit) < 0', -1),
         ('dD_alone', 'dD estimated alone, base terms refitted to each data set < 0', -1),
         ('dDmT_alone_fixedbase', 'dD − dT estimated alone, as reported ≠ 0', 0),
         ('dDmT_alone', 'dD − dT estimated alone, base terms refitted ≠ 0', 0),
         ('dD_joint_dS_dD', 'dD in the joint fit (with dS) < 0', -1),
         ('dD_joint_dS_dD_dT', 'dD in the joint fit with dS and dT < 0', -1),
         ('dDmT_joint', 'dD − dT in the joint fit with dS and dT ≠ 0', 0)]


def group_tests(g, col, sign):
    per = g.groupby('subject')[col].median()
    p = stats.wilcoxon(per).pvalue
    m = per.median()
    det = (p < .05) and (sign == 0 or np.sign(m) == sign)
    ps, ms = [], []
    for roi in ROI_ORDER:
        v = g.loc[g.roi == roi, col]
        ps.append(stats.wilcoxon(v).pvalue)
        ms.append(v.median())
    q = stats.false_discovery_control(ps)
    any_area = bool(np.any((q < .05) & ((np.sign(ms) == sign) if sign else True)))
    pw = []
    for _n, rs in ROI_GROUPS:
        v = g[g.roi.isin(rs)].groupby('subject')[col].median()
        pw.append((stats.wilcoxon(v).pvalue, v.median()))
    qp = stats.false_discovery_control([x[0] for x in pw])
    any_pw = bool(np.any((qp < .05) & np.array([(np.sign(x[1]) == sign) if sign else True for x in pw])))
    return dict(median=m, p=p, det=det, any_area=any_area, any_pathway=any_pw)


def power(md, sim, real):
    rows = []
    for (gen, d, k), g in sim.groupby(['generator', 'delta', 'draw']):
        for col, _l, sign in TESTS:
            rows.append(dict(generator=gen, delta=d, draw=k, test=col, **group_tests(g, col, sign)))
    pr = pd.DataFrame(rows)
    summ = []
    for (gen, col, d), g in pr.groupby(['generator', 'test', 'delta']):
        n = len(g)
        pw_ = g.det.mean()
        summ.append(dict(generator=gen, test=col, delta_pct=d, power=pw_,
                         power_ci_halfwidth=1.96 * np.sqrt(pw_ * (1 - pw_) / n), any_area=g.any_area.mean(),
                         any_pathway=g.any_pathway.mean(), median_estimate=g['median'].median(),
                         est_lo=g['median'].quantile(.025), est_hi=g['median'].quantile(.975),
                         bias=g['median'].median() - d, n_draws=n))
    summ = pd.DataFrame(summ)
    obs = {}
    for col, _l, _s in TESTS:
        per = real.groupby('subject')[col].median()
        obs[col] = (per.median(), stats.wilcoxon(per).pvalue)
    return summ, pr, obs


def power_tables(md, summ, obs, gen_label):
    for col, label, sign in TESTS:
        md += [f'### {label}', '',
               '| Planted dD | Pooled test detects | Any area (BH) | Any pathway (BH) | Group median of the estimate '
               '[2.5–97.5% over data sets] | Bias (median − planted) |', '|---|---|---|---|---|---|']
        for gen, glab in gen_label:
            g = summ[(summ.test == col) & (summ.generator == gen)].sort_values('delta_pct', ascending=False)
            for _, r in g.iterrows():
                md.append(f"| {r.delta_pct:+.0f}%{glab} | {r.power:.0%} ± {r.power_ci_halfwidth:.0%} | "
                          f"{r.any_area:.0%} | {r.any_pathway:.0%} | {r.median_estimate:+.2f}% "
                          f"[{r.est_lo:+.2f}, {r.est_hi:+.2f}] | {r.bias:+.2f} |".replace('-', '−'))
        o = obs[col]
        md += ['', f'Real data, same estimator (pooled): group median {o[0]:+.2f}%, p = {sci(o[1])}.'.replace('-', '−'),
               '']
        for gen, glab in gen_label:
            md += [f'*Summary{glab}:* ' + plain_power(summ[(summ.test == col) & (summ.generator == gen)]
                                                      .to_dict('records')), '']


def plain_power(rows):
    by = {round(r['delta_pct']): r for r in rows}
    s = (f"with no planted effect the test fires in {by[0]['power']:.0%} of data sets; it detects a planted "
         f"−2 / −5 / −10% in {by[-2]['power']:.0%} / {by[-5]['power']:.0%} / {by[-10]['power']:.0%}")
    ok = [d for d in (-2, -5, -10) if by[d]['power'] >= .8]
    s += (f"; 80% power from {ok[0]}%" if ok else '; 80% power is not reached at −10%') + '.'
    s = s.replace('-', '−')
    return s[0].upper() + s[1:]


def main_stats(out_dir):
    md = []
    ho, w = heldout(md)
    lin = load_linear_table()
    # in-sample (all runs) R² gain, for scale
    ins = []
    for (subj, roi), z in lin.items():
        G, c, sst = z['Gr'].sum(0), z['cr'].sum(0), z['sst'].sum()
        red = {}
        for m, sel in (('none', BASE), ('dS', BASE + [4]), ('dS_dD', JOINT['dS_dD']), ('dS_dD_dT', JOINT['dS_dD_dT'])):
            beta = np.linalg.solve(G[np.ix_(sel, sel)], c[sel])
            red[m] = beta @ c[sel] / sst
        ins.append(dict(subject=subj, roi=roi, **{f'{a}_minus_{b}': (red[a] - red[b]) * 1e4 for a, b, _ in CONTRASTS}))
    ins = pd.DataFrame(ins)
    ins.to_csv(f'{D}/hp_terms_insample_dr2_2026-11-13.tsv', sep='\t', index=False)
    md += ['## For scale: in-sample R² gain of the same terms (all runs; ≥ 0 by construction)', '',
           '| Contrast | Median over participants of the pooled in-sample ΔR² (× 10⁻⁴) |', '|---|---|']
    for a, b, label in CONTRASTS:
        v = ins.groupby('subject')[f'{a}_minus_{b}'].median()
        md.append(f'| {label} | {v.median():+.2f} |')
    md += ['', '*Summary:* in sample each added term explains only about 10⁻⁴ of the variance or less; held out, that '
           'gain is smaller than the cost of estimating the term, so held-out prediction cannot separate these models.',
           '', 'Sources: `notes/data/hp_terms_loro_2026-11-13.tsv` (per participant × area × held-out run × model: '
           'held-out SSE and SST, fold coefficients), `notes/data/hp_terms_heldout_r2_2026-11-13.tsv` (held-out R² per '
           'participant × area × model), `notes/data/hp_terms_heldout_stats_2026-11-13.tsv` (every number in the '
           'tables), `notes/data/hp_terms_insample_dr2_2026-11-13.tsv` (in-sample ΔR²). Code: '
           '`retsupp/modeling/hp_terms_heldout_power.py` (cluster, `retsupp/modeling/slurm_jobs/hp_terms_heldout_power.sh`, '
           f'output `derivatives/{FOLDER}`), `retsupp/visualize/paper/hp_terms_heldout_power_stats.py` (tables).']
    open(f'{out_dir}/item1a_heldout.md', 'w').write('\n'.join(md) + '\n')

    sim, real = simulate(lin)
    real.to_csv(f'{D}/hp_terms_real_estimates_2026-11-13.tsv', sep='\t', index=False)
    summ, pr, obs = power([], sim, real)
    summ.to_csv(f'{D}/hp_terms_power_2026-11-13.tsv', sep='\t', index=False)
    pr.to_csv(f'{D}/hp_terms_power_per_draw_2026-11-13.tsv.gz', sep='\t', index=False)
    md = ['<!-- tables generated by retsupp/visualize/paper/hp_terms_heldout_power_stats.py -->', '']
    power_tables(md, summ, obs, [('alone', ' (generator: base + dS alone)'), ('joint', ' (generator: joint fit)')])
    open(f'{out_dir}/item1c_tables_generated.md', 'w').write('\n'.join(md) + '\n')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('what', choices=['collect', 'stats'])
    ap.add_argument('--bids-folder', default='/shares/zne.uzh/gdehol/ds-retsupp')
    ap.add_argument('--folder', default=FOLDER)
    ap.add_argument('--out-dir', default='notes/writing_package_2026-11-13_update11/parts')
    a = ap.parse_args()
    if a.what == 'collect':
        collect(a.bids_folder, a.folder)
    else:
        main_stats(a.out_dir)
