"""Group statistics of the main results with and without some subjects (default: the pilots sub-01 and sub-02).

Every per-subject estimate stays as it is (per-subject fits do not depend on other subjects); only the group level is
recomputed: Wilcoxon tests, Benjamini-Hochberg across areas, bootstrapped CIs, the Fig. 3 ANOVAs, the hierarchy
regression (its pRF-size axis is a median over subjects). The first column (all subjects) must reproduce
key_numbers.md; the second leaves the listed subjects out. LOSO (cluster) and the Fig. 7 hierarchical model
(``trialwise_meta_bayes``) are run separately on filtered inputs.

    python -m retsupp.visualize.paper.sensitivity_excl_subjects --exclude 1 2 \\
        --out notes/data/sensitivity_excl0102_2026-11.tsv
"""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from retsupp.visualize.paper import fig45_common
from retsupp.visualize.paper.paper_style import ROI_ORDER

D = Path('notes/data')
EARLY = ['V1', 'V2', 'V3']
EXCL: set[int] = set()


def keep(d, col='subject'):
    return d[~d[col].isin(EXCL)]


def median_ci(x, n_boot=10000, seed=0):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    b = np.median(np.random.default_rng(seed).choice(x, (n_boot, len(x))), axis=1)
    return np.median(x), *np.percentile(b, [2.5, 97.5])


def wil(x, alternative='two-sided'):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return stats.wilcoxon(x, alternative=alternative).pvalue


ROWS = []


def add(fig, result, value, p=None, n=None, extra=''):
    ROWS.append(dict(fig=fig, result=result, value=value, p=p, n=n, extra=extra))


def pooled(fig, label, d, col, sign=-1):
    """Per-subject median over areas, then median [CI] over subjects, share in the direction ``sign``, Wilcoxon."""
    s = d.groupby('subject')[col].median().dropna()
    m, lo, hi = median_ci(s)
    add(fig, label, m, wil(s), len(s), f'[{lo:+.4f}, {hi:+.4f}], {((s * sign) > 0).mean() * 100:.0f}% in direction')


def per_area(fig, label, d, col, rois=ROI_ORDER):
    ps, ms = [], []
    for r in rois:
        v = d[d.roi == r][col].dropna()
        ms.append(v.median())
        ps.append(wil(v) if len(v) > 5 else np.nan)
    q = stats.false_discovery_control(np.nan_to_num(ps, nan=1.))
    sig = [f'{r} {m:+.3f} (q={qq:.3f})' for r, m, qq in zip(rois, ms, q) if qq < .05]
    add(fig, label + ' per area, BH q < .05', len(sig), None, None, '; '.join(sig) or 'none')


# ---------------------------------------------------------------------------------------------------- Fig. 1
def fig1():
    from retsupp.visualize.paper.fig1_paradigm_behaviour import repeats_raw
    b = keep(pd.read_csv(D / 'behaviour_rt_2026-11.tsv', sep='\t'))
    w = b.pivot_table(index='subject', columns=['contrast', 'condition'], values='rt')
    rep = keep(repeats_raw(D / 'trialwise_af_trial_table.tsv.gz')).pivot_table(index='subject', columns=['contrast', 'condition'], values='rt')
    eff = {'Distractor HP − LP': w[('distractor', 'HP')] - w[('distractor', 'LP')],
           'Distractor LP − absent': w[('distractor', 'LP')] - w[('distractor', 'Absent')],
           'Distractor HP − absent': w[('distractor', 'HP')] - w[('distractor', 'Absent')],
           'Target HP − LP': w[('target', 'HP')] - w[('target', 'LP')],
           'Target repeat − switch': rep[('target', 'Repeat')] - rep[('target', 'Switch')],
           'Distractor repeat − switch': rep[('distractor', 'Repeat')] - rep[('distractor', 'Switch')]}
    for k, v in eff.items():
        v = v.dropna()
        m, lo, hi = median_ci(v)
        sgn = np.sign(m)
        add('1', k + ' (ms, median)', m, wil(v), len(v),
            f'[{lo:+.1f}, {hi:+.1f}]; {int((np.sign(v) == sgn).sum())}/{len(v)} in direction; mean {v.mean():+.1f}')


# ---------------------------------------------------------------------------------------------------- Fig. 2
def fig2():
    from retsupp.visualize.paper.fig2_prf_fits import gain_r2
    g, _ = gain_r2(D / 'prf_cv_roi_summary.tsv', D / 'hrf_comparison_jointhrf_final2_2026-11.tsv')
    g = keep(g)
    for k, lab in (('dog', 'DoG pRF over Gaussian'), ('hrf', 'Joint HRF on top of DoG'), ('both', 'Both')):
        s = g.groupby('subject')[k].median()
        m, lo, hi = median_ci(s)
        add('2d', f'Held-out R² gained: {lab}', m, wil(s), len(s), f'[{lo:+.4f}, {hi:+.4f}]; {(s > 0).sum()}/{len(s)} > 0')


# ---------------------------------------------------------------------------------------------------- Fig. 3
def fig3():
    from retsupp.visualize.paper import fig3_item_locations as f3
    orig = fig45_common.item_timecourses

    def filtered(*a, **k):
        return keep(orig(*a, **k))
    f3.item_timecourses = filtered
    fig45_common.item_timecourses = filtered
    groups = {r: g for g, rs in f3.PATHWAY.items() for r in rs}
    pw = filtered(str(D / 'event_deconv_final2_2026-11'), groups=groups)
    for term, lab in (('T', 'Target − neutral'), ('D', 'Distractor − neutral (LP)')):
        dt = f3.diff_tests(pw, term)
        for g, r in dt.items():
            add('3b' if term == 'T' else '3c', f'{lab}, {g} (% signal, mean)', r['mean'], r['p'], r['n'], f'q={r["q"]:.2g}')
    from retsupp.visualize.paper import fig_hp_heatmap as hh
    orig_hh = hh.item_timecourses
    hh.item_timecourses = filtered
    _, hres = hh.hp_tests(str(D / 'event_deconv_final2_2026-11'))
    hh.item_timecourses = orig_hh
    for g in ('Early', 'All areas'):
        r = hres.loc[g]
        add('3', f'{g}: HP vs all LP (%; HP − LP deconvolution)', r['main'], r.main_p, int(r.n),
            f'F = {r.main_F:.2f}; location × item p = {r.int_p:.3f}')
    f3.NONPOS.clear()
    _, _, res = f3.rel_stats(str(D / 'event_deconv_relative_final2_2026-11'))
    for g in ('Early', 'All areas', 'Dorsal', 'Occipitotemporal'):
        r = res.loc[g]
        add('3e', f'{g}: location type ANOVA', r.loc_F, r.loc_p, int(r.n), f'F({r.loc_df1},{r.loc_df2})')
        if g in ('Early', 'All areas'):
            add('3e', f'{g}: location × item', r.int_F, r.int_p, int(r.n), f'F({r.int_df1},{r.int_df2})')
            for k, lab in (('hp_next', 'HP − next'), ('next_opp', 'next − opposite'), ('hp_opp', 'HP − opposite')):
                add('3e', f'{g}: {lab} (%)', r[k], r[f'{k}_p'], int(r.n), f'± {r[f"{k}_se"]:.1f}')
    add('3e', 'Subject × group pairs left out (mean response ≤ 0)', len(f3.NONPOS), None, None, ', '.join(sorted(f3.NONPOS)))
    f3.item_timecourses = orig
    fig45_common.item_timecourses = orig


# ---------------------------------------------------------------------------------------------------- Fig. 5
def joint(family='dog', model='m7_full'):
    d = keep(pd.read_csv(D / 'gain_field_parameters_final2_2026-11.tsv', sep='\t'))
    d = d[(d.family == family) & (d.model == model) & d.roi.isin(ROI_ORDER)].copy()
    d['dS'], d['dD'] = d.g_HP - d.g_LP, d.g_HP_dyn - d.g_LP_dyn
    d['T'], d['D'] = d.g_T_dyn, (d.g_HP_dyn + 3 * d.g_LP_dyn) / 4
    return d


def fig5():
    js = joint()
    pooled('5a', 'Target gain, joint (pooled)', js, 'T', +1)
    per_area('5a', 'Target gain, joint', js, 'T')
    pooled('5a', 'Distractor gain, joint (pooled)', js, 'D', +1)
    per_area('5a', 'Distractor gain, joint', js, 'D')
    pooled('5c', 'dS joint m7 (pooled)', js, 'dS')
    per_area('5c', 'dS joint m7', js, 'dS')
    pooled('5d', 'dD joint m7 (pooled)', js, 'dD')
    c = keep(pd.read_csv(D / 'conditional_estimates_dog_final2_2026-11.tsv', sep='\t'))
    c = c[c.valid & c.roi.isin(ROI_ORDER)].pivot_table(index=['subject', 'roi'], columns='mech', values='estimate').reset_index()
    c['dDmdT'] = c.dD - c.dT
    for col, lab, sgn in (('dS', 'dS estimated alone', -1), ('S', 'Sustained gain S alone', -1), ('DonT', 'Distractor on top of target, alone', +1),
                          ('D', 'Distractor alone on the item model', -1), ('dD', 'dD alone', -1), ('dT', 'dT alone', -1),
                          ('dDmdT', 'dD − dT alone', +1), ('T', 'Target gain alone', +1)):
        pooled('5c' if col in ('dS', 'S') else '5d' if col in ('dD', 'dT', 'dDmdT') else '5a', lab + ' (pooled)', c, col, sgn)
        if col in ('dS', 'dD', 'dT', 'DonT', 'S'):
            per_area('5', lab, c, col)
    # split models
    a = joint(model='m8_full_adjopp')
    a['next_opp'] = a.g_adj_minus_opp
    # lp_split: g_LP is the mean of the three LP gains, δ = g_adj − g_opp; HP − next = dS − δ/3, HP − opposite = dS + 2δ/3
    a['hp_opp'] = a.g_HP - a.g_LP + 2 * a.g_adj_minus_opp / 3
    a['hp_next'] = a.g_HP - a.g_LP - a.g_adj_minus_opp / 3
    for col, lab in (('next_opp', 'Split: sustained next − opposite'), ('hp_next', 'Split: sustained HP − next'), ('hp_opp', 'Split: sustained HP − opposite')):
        pooled('5 split', lab + ' (pooled)', a, col)
    t = joint(model='m8_full_dT')
    t['tHL'] = t.g_T_HP_dyn - t.g_T_dyn
    pooled('5 split', 'Split: target HP − LP (pooled)', t, 'tHL')
    pooled('5 split', 'Split: distractor HP − LP (pooled)', t, 'dD')
    s = joint(model='m7_full_sigtied')
    pooled('5 split', 'Sigma tied: dS (pooled)', s, 'dS')
    # hierarchy regression, with the pRF-size axis from the kept subjects
    from retsupp.visualize.paper import hierarchy_regression as hr
    with tempfile.TemporaryDirectory() as tmp:
        paths = {}
        for k, f in (('gain', 'gain_field_parameters_final2_2026-11.tsv'), ('cond', 'conditional_estimates_dog_final2_2026-11.tsv'),
                     ('sigma', 'fig1_prf_sigma.tsv')   # the table hierarchy_regression_final2 used (model-4 mean fits)
                     ):
            x = pd.read_csv(D / f, sep='\t')
            paths[k] = f'{tmp}/{f}'
            keep(x).to_csv(paths[k], sep='\t', index=False)
        out = f'{tmp}/hierarchy_regression.tsv'
        hr.main(paths['gain'], paths['cond'], paths['sigma'], out)
        h = pd.read_csv(out, sep='\t')
    for par in ('Target gain', 'Distractor gain'):
        r = h[(h.parameter == par) & (h.rois == 'all nine')].iloc[0]
        add('5', f'{par} per doubling of pRF size', r.median_slope, r.p, int(r.n), f'[{r.ci_low:+.3f}, {r.ci_high:+.3f}], q={r.q:.3g}')
    # robustness: Gaussian and DN joint and alone
    for fam in ('gauss', 'dn'):
        j = joint(fam)
        for col, lab, sgn in (('T', 'Target gain joint', 1), ('D', 'Distractor gain joint', 1), ('dS', 'dS joint', -1), ('dD', 'dD joint', -1)):
            pooled(f'R {fam}', lab + ' (pooled)', j, col, sgn)
    e = keep(pd.read_csv(D / 'conditional_exact_final2_allfam_2026-11.tsv', sep='\t'))
    e = e[e.roi.isin(ROI_ORDER) & (e.nvox >= 20)]
    for fam in ('dog', 'gauss', 'dn'):
        x = e[e.family == fam].pivot_table(index=['subject', 'roi'], columns='mech', values='own').reset_index()
        for col, lab, sgn in (('dS', 'dS alone (exact)', -1), ('dD', 'dD alone (exact)', -1), ('dT', 'dT alone (exact)', -1),
                              ('T', 'Target gain alone (exact)', 1), ('DonT', 'Distractor on target alone (exact)', 1), ('S', 'S alone (exact)', -1)):
            if col in x:
                pooled(f'R {fam}', lab + ' (pooled)', x, col, sgn)


# ---------------------------------------------------------------------------------------------------- Fig. 6
def fig6():
    from retsupp.visualize.paper import fig6_decoding as f6
    from retsupp.visualize.paper.figS_decoded_bars import _half_contrasts
    for src in ('jointhrf', 'gauss_jointhrf', 'dn_jointhrf'):
        d = keep(pd.read_csv(D / f'decoding_readout_{src}_2026-11.tsv', sep='\t'))
        for dec in ('free', 'outline'):
            if dec not in set(d.decoder):
                continue
            s = f6.quadrant_effects(d, dec)
            st = f6.quadrant_stats(s)
            pre = f'{src} {dec}'
            add('6d', f'{pre}: HP (% of four-quadrant mean)', s.hp.mean(), st['p_adj'], len(s),
                f'next {s.adjacent.mean():+.2f}, opposite {s.opposite.mean():+.2f}; HP<next one-sided p; HP<opp p={st["p_opp"]:.3f}; '
                f'Friedman p={st["p_fried"]:.3f}; next vs opp two-sided p={wil(s.adjacent - s.opposite):.3f}; HP vs next two-sided p={wil(s.hp - s.adjacent):.3f}')
            if src == 'jointhrf' and dec == 'free':
                t = f6.test_bh(f6.effects(d), ROI_ORDER, dec)
                sig = t[t.q < .05]
                add('S6', 'Per-area HP quadrant − others, BH q < .05 (count)', len(sig), None, None,
                    '; '.join(f'{r.roi} p={r.p:.3f} q={r.q:.3f}' for r in t.itertuples()))
                bands = keep(pd.read_csv(D / f'decoding_bands_{src}_2026-11.tsv.gz', sep='\t'))
                be = f6.band_effects(bands, dec)
                for k in ['all'] + [b[0] for b in f6.BANDS]:
                    add('6e', f'Band {k}: HP − others (%)', be[k].mean(), wil(be[k], 'less'), len(be))
                add('6e', 'Bands differ (Friedman)', None, stats.friedmanchisquare(*[be[b[0]] for b in f6.BANDS]).pvalue, len(be))
                q = keep(pd.read_csv(D / f'decoding_quality_{src}_2026-11.tsv', sep='\t'))
                qs = q[q.decoder == dec].groupby(['subject', 'roi']).r.mean().reset_index()
                add('6b', 'Frame-wise r, V1 (median)', qs[qs.roi == 'V1'].r.median(), None, (qs.roi == 'V1').sum())
    seg = keep(pd.read_csv(D / 'decoded_bars_segments_jointhrf_outline_b11_2026-11.tsv', sep='\t'))
    for o in ('vertical', 'horizontal', 'pooled'):
        hp_half, nxt_half, overall = _half_contrasts(seg, EARLY, o)
        add('S6 bars', f'Decoded bars {o}: HP half − other half (%)', hp_half.mean(), wil(hp_half), len(hp_half),
            f'next − opposite half {nxt_half.mean():+.2f} (p={wil(nxt_half):.3f})')


def main(exclude, out):
    global EXCL
    allrows = []
    for tag, ex in (('all', set()), ('excl', set(exclude))):
        EXCL = ex
        ROWS.clear()
        for f in (fig1, fig2, fig3, fig5, fig6):
            f()
        allrows.append(pd.DataFrame(ROWS).assign(set=tag))
    a, b = allrows
    m = a.drop(columns='set').merge(b.drop(columns='set'), on=['fig', 'result'], suffixes=('_all', '_excl'), how='outer')
    m.to_csv(out, sep='\t', index=False)
    with pd.option_context('display.width', 250, 'display.max_colwidth', 120, 'display.max_rows', 500):
        print(m.to_string(index=False))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--exclude', nargs='+', type=int, default=[1, 2])
    ap.add_argument('--out', default='notes/data/sensitivity_excl0102_2026-11.tsv')
    a = ap.parse_args()
    main(a.exclude, a.out)
