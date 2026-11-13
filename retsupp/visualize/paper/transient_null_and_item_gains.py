"""Update 11 (request 2026-11-13), items 1b, 3b and 3c, on one reference fit (default: the refitted fits of update 8).

1b  Evidence for the null for the transient HP − LP terms, pooled over areas (per participant the median over the
    nine areas): dD, dT and dD − dT estimated alone (each term added alone to the model with item weight, target,
    distractor and sustained gain, m4_TDS) and dD in the joint fit (all terms together, m7_full; the joint fit has no
    dT). JZS Bayes factors (Rouder et al. 2009; Cauchy prior on the standardized effect, r = √2/2 and r = √2/4), the
    same on 10%-winsorized values, a Bayesian sign test (uniform prior on the share of participants below 0), and
    equivalence tests (two one-sided Wilcoxon signed-rank tests on the shifted values, and two one-sided t tests) at
    bounds of ±1.0, ±2, ±3.5 and ±5 percentage points, plus the smallest bound at which equivalence holds (α = .05).
    dS (estimated alone and joint) is run through the same tests as a calibration.
3b  Sizes side by side from the joint fit: target gain, distractor gain (averaged over its locations), dS joint, per
    area and pooled (median [bootstrapped 95% CI], k of n, Wilcoxon, BH over areas), and the ratio dS / target gain:
    ratio of medians with a bootstrap over participants; per participant only over areas where the target gain is
    reliably above 0 (jackknife |gain| / SE > 2 from the field-width fits of the same root).
3c  Does a larger target or distractor gain go with a more negative dS? Spearman ρ between participants (per
    participant the median over areas), between areas (area medians; bootstrap over participants) and within
    participants across areas (per participant ρ over its areas, Fisher z, Wilcoxon), for the joint dS and for dS
    estimated alone.

    python -m retsupp.visualize.paper.run_excluding --exclude 1 2 -- retsupp.visualize.paper.transient_null_and_item_gains
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy import integrate, special, stats

from retsupp.visualize.paper.fig45_common import conditional, joint, jzs_bf10, median_ci
from retsupp.visualize.paper.paper_style import ROI_ORDER

N_BOOT = 5000
BOUNDS = (1.0, 2.0, 3.5, 5.0)


def winsor(v, k=.1):
    lo, hi = np.quantile(v, [k, 1 - k])
    return np.clip(v, lo, hi)


def sign_bf01(v):
    """Bayesian sign test: H0 share below 0 = .5 vs H1 share ~ Uniform(0, 1). BF01 = C(n, k) .5^n (n + 1)."""
    v = v[v != 0]
    n, k = len(v), int((v < 0).sum())
    return np.exp(special.gammaln(n + 1) - special.gammaln(k + 1) - special.gammaln(n - k + 1) + n * np.log(.5)) * (n + 1)


def tost_wilcoxon(v, b):
    lo = stats.wilcoxon(v + b, alternative='greater').pvalue
    hi = stats.wilcoxon(v - b, alternative='less').pvalue
    return max(lo, hi)


def tost_t(v, b):
    lo = stats.ttest_1samp(v, -b, alternative='greater').pvalue
    hi = stats.ttest_1samp(v, b, alternative='less').pvalue
    return max(lo, hi)


def smallest_bound(v, test, grid=np.arange(.05, 60, .05)):
    for b in grid:
        if test(v, b) < .05:
            return b
    return np.nan


def boot_ci90(v, seed=0):
    rng = np.random.default_rng(seed)
    b = np.median(rng.choice(v, (N_BOOT, len(v))), axis=1)
    return np.percentile(b, [5, 95])


def null_evidence(js, cond):
    both = cond[['subject', 'roi', 'dD', 'dT']].dropna().assign(dDmT=lambda x: x.dD - x.dT)
    terms = [('dD', 'estimated alone', cond, 'dD'), ('dT', 'estimated alone', cond, 'dT'),
             ('dD − dT', 'estimated alone', both, 'dDmT'), ('dD', 'joint fit', js, 'dD'),
             ('dS (calibration)', 'estimated alone', cond, 'dS'), ('dS (calibration)', 'joint fit', js, 'dS')]
    rows = []
    for name, fit, df, col in terms:
        v = df.groupby('subject')[col].median().dropna().to_numpy() * 100
        m, lo, hi = median_ci(v)
        c90 = boot_ci90(v)
        row = dict(term=name, fit=fit, n=len(v), median=m, ci_lo=lo, ci_hi=hi, ci90_lo=c90[0], ci90_hi=c90[1],
                   mean=v.mean(), sd=v.std(ddof=1), k_neg=int((v < 0).sum()), p_wilcoxon=stats.wilcoxon(v).pvalue,
                   BF01_jzs_r0707=1 / jzs_bf10(v), BF01_jzs_r0354=1 / jzs_bf10(v, r=np.sqrt(2) / 4),
                   BF01_jzs_winsor10_r0707=1 / jzs_bf10(winsor(v)), BF01_sign=sign_bf01(v),
                   smallest_equiv_bound_wilcoxon=smallest_bound(v, tost_wilcoxon),
                   smallest_equiv_bound_t=smallest_bound(v, tost_t))
        for b in BOUNDS:
            row[f'tost_wilcoxon_p_{b:g}'] = tost_wilcoxon(v, b)
            row[f'tost_t_p_{b:g}'] = tost_t(v, b)
        rows.append(row)
    return pd.DataFrame(rows)


def term_stats(df, col, label, scale=100.):
    rows, ps = [], []
    for r in ROI_ORDER:
        v = df[df.roi == r][col].dropna().to_numpy() * scale
        m, lo, hi = median_ci(v)
        p = stats.wilcoxon(v).pvalue
        ps.append(p)
        rows.append(dict(term=label, area=r, median=m, ci_lo=lo, ci_hi=hi, k_pos=int((v > 0).sum()), n=len(v), p=p))
    for row, q in zip(rows, stats.false_discovery_control(ps)):
        row['q'] = q
    v = df.groupby('subject')[col].median().dropna().to_numpy() * scale
    m, lo, hi = median_ci(v)
    rows.insert(0, dict(term=label, area='pooled', median=m, ci_lo=lo, ci_hi=hi, k_pos=int((v > 0).sum()), n=len(v),
                        p=stats.wilcoxon(v).pvalue, q=np.nan))
    return rows


def ratio_of_medians(a, b, seed=0):
    """median(a) / median(b) over participants (paired arrays), with a bootstrap over participants."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), (N_BOOT, len(a)))
    bt = np.median(a[idx], 1) / np.median(b[idx], 1)
    return np.median(a) / np.median(b), *np.percentile(bt, [2.5, 97.5])


def sizes(js, widths_fits):
    rows = term_stats(js, 'T', 'Target gain') + term_stats(js, 'D', 'Distractor gain (averaged over its locations)') + \
        term_stats(js, 'dS', 'dS (sustained HP − LP)')
    tab = pd.DataFrame(rows)
    rat = []
    pooled = js.groupby('subject')[['T', 'dS']].median()
    m, lo, hi = ratio_of_medians(pooled.dS.to_numpy(), pooled['T'].to_numpy())
    rat.append(dict(area='pooled', kind='ratio of medians', ratio=m, ci_lo=lo, ci_hi=hi, n=len(pooled)))
    for r in ROI_ORDER:
        x = js[js.roi == r][['dS', 'T']].dropna()
        m, lo, hi = ratio_of_medians(x.dS.to_numpy(), x['T'].to_numpy())
        rat.append(dict(area=r, kind='ratio of medians', ratio=m, ci_lo=lo, ci_hi=hi, n=len(x)))
    # per participant, only where the target gain is reliably > 0 in the same root's jackknife (|z| > 2, z > 0)
    w = widths_fits[['subject', 'roi', 'g_T_z']]
    x = js.merge(w, on=['subject', 'roi'])
    rel = x[x.g_T_z > 2].assign(ratio=lambda d: d.dS / d['T'])
    per = rel.groupby('subject').ratio.median()
    m, lo, hi = median_ci(per.to_numpy())
    rat.append(dict(area='pooled', kind='per participant, reliable target cells', ratio=m, ci_lo=lo, ci_hi=hi,
                    n=len(per), n_cells=len(rel), k_neg=int((per < 0).sum()), p=stats.wilcoxon(per).pvalue))
    for r in ROI_ORDER:
        v = rel[rel.roi == r].ratio.to_numpy()
        if len(v) >= 6:
            m, lo, hi = median_ci(v)
            rat.append(dict(area=r, kind='per participant, reliable target cells', ratio=m, ci_lo=lo, ci_hi=hi,
                            n=len(v), n_cells=len(v), k_neg=int((v < 0).sum()), p=stats.wilcoxon(v).pvalue))
        else:
            rat.append(dict(area=r, kind='per participant, reliable target cells', n=len(v), n_cells=len(v)))
    return tab, pd.DataFrame(rat)


def spearman_boot(x, y, seed=0):
    rng = np.random.default_rng(seed)
    r = stats.spearmanr(x, y)
    idx = rng.integers(0, len(x), (N_BOOT, len(x)))
    bt = np.array([stats.spearmanr(x[i], y[i])[0] for i in idx])
    return r[0], *np.nanpercentile(bt, [2.5, 97.5]), r[1]


def correlations(js, cond):
    d = js[['subject', 'roi', 'T', 'D', 'dS']].merge(cond[['subject', 'roi', 'dS']].rename(columns={'dS': 'dS_alone'}),
                                                    on=['subject', 'roi'], how='left')
    rows = []
    for gx in ('T', 'D'):
        for gy, ylab in (('dS', 'dS joint'), ('dS_alone', 'dS estimated alone')):
            x = d[['subject', 'roi', gx, gy]].dropna()
            # between participants: per participant the median over its areas
            pp = x.groupby('subject')[[gx, gy]].median()
            rho, lo, hi, p = spearman_boot(pp[gx].to_numpy(), pp[gy].to_numpy())
            rows.append(dict(x=gx, y=ylab, level='between participants (medians over areas)', rho=rho, ci_lo=lo,
                             ci_hi=hi, p=p, n=len(pp)))
            # between areas: area medians; CI from a bootstrap over participants
            am = x.groupby('roi')[[gx, gy]].median()
            rho, p = stats.spearmanr(am[gx], am[gy])
            subs = x.subject.unique()
            rng = np.random.default_rng(1)
            bt = []
            for _ in range(2000):
                s = rng.choice(subs, len(subs))
                xb = pd.concat([x[x.subject == k] for k in s])
                mb = xb.groupby('roi')[[gx, gy]].median()
                bt.append(stats.spearmanr(mb[gx], mb[gy])[0])
            lo, hi = np.nanpercentile(bt, [2.5, 97.5])
            rows.append(dict(x=gx, y=ylab, level='between areas (area medians)', rho=rho, ci_lo=lo, ci_hi=hi, p=p,
                             n=len(am)))
            # within participants across areas
            per = []
            for s, g in x.groupby('subject'):
                if len(g) >= 5:
                    per.append(stats.spearmanr(g[gx], g[gy])[0])
            per = np.array(per)
            z = np.arctanh(np.clip(per, -.999, .999))
            m, lo, hi = median_ci(per)
            rows.append(dict(x=gx, y=ylab, level='within participants (ρ across areas, ≥ 5 areas)', rho=m, ci_lo=lo,
                             ci_hi=hi, p=stats.wilcoxon(z).pvalue, n=len(per), k_neg=int((per < 0).sum()),
                             mean_rho_from_z=np.tanh(z.mean())))
            # within participants AND areas: residuals after removing area medians, then participant medians (removes
            # the hierarchy trend of the target gain); CI from a bootstrap over participants
            def dc(xx):
                r = xx.copy()
                for c in (gx, gy):
                    r[c] = r[c] - r.groupby('roi')[c].transform('median')
                    r[c] = r[c] - r.groupby('subject')[c].transform('median')
                return r
            res = dc(x)
            rho, p = stats.spearmanr(res[gx], res[gy])
            rng = np.random.default_rng(2)
            bt = []
            for _ in range(2000):
                s = rng.choice(subs, len(subs))
                xb = pd.concat([x[x.subject == k].assign(subject=i) for i, k in enumerate(s)])
                rb = dc(xb)
                bt.append(stats.spearmanr(rb[gx], rb[gy])[0])
            lo, hi = np.nanpercentile(bt, [2.5, 97.5])
            rows.append(dict(x=gx, y=ylab, level='within participants and areas (area and participant medians removed)',
                             rho=rho, ci_lo=lo, ci_hi=hi, p=p, n=len(res)))
            # all cells, ignoring structure (for reference)
            rho, p = stats.spearmanr(x[gx], x[gy])
            rows.append(dict(x=gx, y=ylab, level='all participant × area cells (reference, not independent)',
                             rho=rho, p=p, n=len(x)))
    return pd.DataFrame(rows)


def split_half(root, exclude):
    """Is the gain–dS correlation shared estimation noise? Per participant × area, the five joint-fit gains (sustained HP,
    LP, target, distractor at HP and at LP) re-solved on odd and on even runs from the per-run sufficient statistics of
    ``fit_width_profiles`` (same root, everything else at the joint fit). Same-half ρ (gain and dS from the same runs)
    vs cross-half ρ (gain from one half, dS from the other): noise shared within a fit inflates only the same-half ρ."""
    import glob
    rows = []
    for f in sorted(glob.glob(f'{root}/sub-*/*_widthprof.npz')):
        z = np.load(f, allow_pickle=True)
        s = int(z['subject'])
        if s in exclude:
            continue
        nm = list(z['gain_names'])
        odd = z['runs'][:, 1] % 2 == 1
        for half, m in (('odd', odd), ('even', ~odd)):
            g = np.linalg.solve(z['A'][m].sum(0), z['b'][m].sum(0))
            gi = dict(zip(nm, g))
            rows.append(dict(subject=s, roi=str(z['roi']), half=half, T=gi['g_T'],
                             D=(gi['g_D_HP'] + 3 * gi['g_D_LP']) / 4, dS=gi['g_HP'] - gi['g_LP']))
    h = pd.DataFrame(rows).pivot_table(index=['subject', 'roi'], columns='half', values=['T', 'D', 'dS'])
    h.columns = [f'{a}_{b}' for a, b in h.columns]
    h = h.reset_index()

    def dc(d, cols):
        d = d.copy()
        for c in cols:
            d[c] = d[c] - d.groupby('roi')[c].transform('median')
            d[c] = d[c] - d.groupby('subject')[c].transform('median')
        return d

    out = []
    for gx in ('T', 'D'):
        pairs = {'same half': [(f'{gx}_odd', 'dS_odd'), (f'{gx}_even', 'dS_even')],
                 'cross half': [(f'{gx}_odd', 'dS_even'), (f'{gx}_even', 'dS_odd')]}
        r = dc(h, [c for c in h.columns if c not in ('subject', 'roi')])
        for kind, pp in pairs.items():
            rho_dc = np.mean([stats.spearmanr(r[a], r[b])[0] for a, b in pp])
            per = []
            for s, g in h.groupby('subject'):
                if len(g) >= 5:
                    per.append(np.mean([stats.spearmanr(g[a], g[b])[0] for a, b in pp]))
            per = np.array(per)
            m, lo, hi = median_ci(per)
            out.append(dict(x=gx, y='dS joint (split halves)', kind=kind, rho_double_centred=rho_dc,
                            within_participant_median_rho=m, ci_lo=lo, ci_hi=hi, n=len(per),
                            k_neg=int((per < 0).sum()), p=stats.wilcoxon(np.arctanh(np.clip(per, -.999, .999))).pvalue))
        # reliability of each term between halves (double-centred), for the attenuation
        for c in (gx, 'dS'):
            out.append(dict(x=c, y='split-half reliability', kind='odd vs even',
                            rho_double_centred=stats.spearmanr(r[f'{c}_odd'], r[f'{c}_even'])[0]))
    return pd.DataFrame(out).drop_duplicates(subset=['x', 'y', 'kind'])


def main(a):
    js = joint(a.gain_table, 'dog')
    js = js[~js.subject.isin(a.exclude)]
    cond = conditional(a.conditional_table)
    cond = cond[~cond.subject.isin(a.exclude)]
    wf = pd.read_csv(a.widths_fits, sep='\t')
    wf = wf[~wf.subject.isin(a.exclude)]
    pd.set_option('display.width', 250)
    ne = null_evidence(js, cond)
    ne.to_csv(f'{a.out_prefix}_null_evidence.tsv', sep='\t', index=False, float_format='%.5g')
    print(ne.round(3).to_string())
    tab, rat = sizes(js, wf)
    tab.to_csv(f'{a.out_prefix}_sizes.tsv', sep='\t', index=False, float_format='%.5g')
    rat.to_csv(f'{a.out_prefix}_ratio.tsv', sep='\t', index=False, float_format='%.5g')
    print(tab.round(3).to_string())
    print(rat.round(3).to_string())
    cr = correlations(js, cond)
    cr.to_csv(f'{a.out_prefix}_correlations.tsv', sep='\t', index=False, float_format='%.5g')
    print(cr.round(3).to_string())
    if a.width_root:
        sh = split_half(a.width_root, a.exclude)
        sh.to_csv(f'{a.out_prefix}_correlations_split_half.tsv', sep='\t', index=False, float_format='%.5g')
        print(sh.round(3).to_string())


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--gain-table', default='notes/data/gain_field_parameters_polished_2026-11-12.tsv')
    ap.add_argument('--conditional-table', default='notes/data/conditional_estimates_polished_2026-11-12.tsv')
    ap.add_argument('--widths-fits', default='notes/data/field_widths_reliable_polished_2026-11-12_n28_fits.tsv')
    ap.add_argument('--width-root', default='/data/ds-retsupp/derivatives/field_widths_reliable_polished_2026-11-12',
                    help='per-run sufficient statistics (fit_width_profiles) of the same root, for the split-half check; '
                         'empty to skip')
    ap.add_argument('--exclude', type=int, nargs='*', default=[1, 2])
    ap.add_argument('--out-prefix', default='notes/data/update11_gain_terms_polished_2026-11-12_n28')
    main(ap.parse_args())
