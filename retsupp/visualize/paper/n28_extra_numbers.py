"""Numbers for key_numbers.md at n = 28 (pilots sub-01, sub-02 left out) that the sensitivity table does not hold.

Each block also prints the all-subjects value where a key_numbers.md (2026-11-11) value exists, as a check that the
recipe matches. Output: one row per number (block, label, n, value, ci_lo, ci_hi, extra) to ``--out``.

    python -m retsupp.visualize.paper.n28_extra_numbers --out notes/data/n28/key_numbers_extras_n28_2026-11-12.tsv
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy import stats

D = 'notes/data/'
R9 = ['V1', 'V2', 'V3', 'V3AB', 'IPS', 'hV4', 'VO', 'LO', 'TO']
ROWS = []


def add(block, label, n, value, lo=np.nan, hi=np.nan, extra=''):
    ROWS.append(dict(block=block, label=label, n=n, value=value, ci_lo=lo, ci_hi=hi, extra=extra))


def boot(x, f=np.median, n=10000, seed=0):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    b = f(np.random.default_rng(seed).choice(x, (n, len(x))), axis=1)
    return f(x), *np.percentile(b, [2.5, 97.5])


def contralateral(excl):
    c = pd.read_csv(D + 'fig1_prf_coverage_final2.tsv', sep='\t')
    c = c[~c.subject.isin(excl)].copy()
    mid = c.ix.max() / 2
    c['contra'] = np.where(c.hemi == 'L', c.ix > mid, c.ix < mid)
    c['ipsi'] = np.where(c.hemi == 'L', c.ix < mid, c.ix > mid)
    g = c.groupby(['subject', 'roi'])[['n', 'contra', 'ipsi']].apply(
        lambda x: 100 * x.n[x.contra].sum() / (x.n[x.contra].sum() + x.n[x.ipsi].sum())).rename('share').reset_index()
    for roi in R9:
        v = g[g.roi == roi].share
        add('2b contralateral', roi, len(v), v.median(), v.quantile(.25), v.quantile(.75), 'median [IQR]')


def behaviour(excl):
    b = pd.read_csv(D + 'behaviour_rt_2026-11.tsv', sep='\t')
    b = b[~b.subject.isin(excl)]
    w = b.pivot_table(index='subject', columns=['contrast', 'condition'], values='rt')
    for lab, a, c in (('Distractor HP vs LP', ('distractor', 'HP'), ('distractor', 'LP')),
                      ('Distractor LP vs absent', ('distractor', 'LP'), ('distractor', 'Absent')),
                      ('Distractor HP vs absent', ('distractor', 'HP'), ('distractor', 'Absent')),
                      ('Target HP vs LP', ('target', 'HP'), ('target', 'LP'))):
        d = (w[a] - w[c]).dropna()
        m, lo, hi = boot(d)
        add('1 behaviour', lab, len(d), m, lo, hi,
            f'means {w[a].mean() * 1000 if w[a].mean() < 10 else w[a].mean():.0f} vs {w[c].mean() * 1000 if w[c].mean() < 10 else w[c].mean():.0f}; '
            f'mean diff {d.mean():+.1f}; {int((np.sign(d) == np.sign(m)).sum())}/{len(d)}; p = {stats.wilcoxon(d).pvalue:.2g}')
    from retsupp.visualize.paper.fig1_paradigm_behaviour import repeats_raw
    rep = repeats_raw(D + 'trialwise_af_trial_table.tsv.gz')
    rep = rep[~rep.subject.isin(excl)].pivot_table(index='subject', columns=['contrast', 'condition'], values='rt')
    for lab, k in (('Target repeat − switch', 'target'), ('Distractor repeat − switch', 'distractor')):
        d = (rep[(k, 'Repeat')] - rep[(k, 'Switch')]).dropna()
        m, lo, hi = boot(d)
        add('1 behaviour', lab, len(d), m, lo, hi,
            f'mean diff {d.mean():+.1f}; {int((np.sign(d) == np.sign(m)).sum())}/{len(d)}; p = {stats.wilcoxon(d).pvalue:.2g}')


def fig2d(excl):
    from retsupp.visualize.paper.fig2_prf_fits import gain_r2
    g, _ = gain_r2(D + 'prf_cv_roi_summary.tsv', D + 'hrf_comparison_jointhrf_final2_2026-11.tsv')
    g = g[~g.subject.isin(excl)]
    for k, lab in (('dog', 'DoG over Gaussian'), ('hrf', 'joint HRF on top of DoG'), ('both', 'both')):
        s = g.groupby('subject')[k].median()
        m, lo, hi = boot(s)
        add('2d', lab, len(s), m, lo, hi, f'{(s > 0).sum()}/{len(s)}; p = {stats.wilcoxon(s).pvalue:.2g}')


def prf_models(excl):
    s = pd.read_csv(D + 'prf_jointhrf_summary_final2_2026-11.tsv', sep='\t')
    s = s[(s.stage == 3) & s.roi.isin(R9) & ~s.subject.isin(excl)]
    v = s.groupby(['family', 'subject', 'roi']).r2_test_reliable.mean().groupby(['family', 'subject']).median().unstack(0)
    for f in v.columns:
        add('R prf models', f'held-out R² {f}', v[f].notna().sum(), v[f].median())
    for a, b in (('dn', 'dog'), ('dog', 'gauss')):
        if a in v and b in v:
            d = (v[a] - v[b]).dropna()
            m, lo, hi = boot(d)
            add('R prf models', f'{a} − {b}', len(d), m, lo, hi, f'{(d > 0).sum()}/{len(d)}; p = {stats.wilcoxon(d).pvalue:.2g}')


def fig3_ci(excl):
    from retsupp.visualize.paper import fig3_item_locations as f3
    from retsupp.visualize.paper.fig45_common import item_timecourses
    pw = item_timecourses(D + 'event_deconv_final2_2026-11', groups={r: g for g, rs in f3.PATHWAY.items() for r in rs})
    pw = pw[~pw.subject.isin(excl)]
    for term, lab in (('T', 'target − neutral'), ('D', 'distractor − neutral')):
        item, neutral = f3.item_pair(pw, term)
        x = (item - neutral).rename('r').reset_index()
        for grp in f3.PATHWAY:
            g = x[x.grp == grp].pivot(index='subject', columns='t', values='r')
            v = g.loc[:, (g.columns >= f3.PEAK[0]) & (g.columns <= f3.PEAK[1])].mean(1)
            m, lo, hi = boot(v, np.mean)
            add('3bc', f'{lab}, {grp}', len(v), m, lo, hi, f'p = {stats.wilcoxon(v).pvalue:.2g}')


def split_per_area(excl):
    d = pd.read_csv(D + 'gain_field_parameters_final2_2026-11.tsv', sep='\t')
    a = d[(d.family == 'dog') & (d.model == 'm8_full_adjopp') & d.roi.isin(R9) & ~d.subject.isin(excl)].copy()
    a['hp_opp'] = a.g_HP - a.g_LP + 2 * a.g_adj_minus_opp / 3
    a['hp_next'] = a.g_HP - a.g_LP - a.g_adj_minus_opp / 3
    a['next_opp'] = a.g_adj_minus_opp
    add('5 split', 'fits', len(a), np.nan)
    for col in ('hp_next', 'hp_opp', 'next_opp'):
        ps = [stats.wilcoxon(a[a.roi == r][col].dropna()).pvalue for r in R9]
        q = stats.false_discovery_control(ps)
        sig = '; '.join(f'{r} {a[a.roi == r][col].median():+.3f} (q = {qq:.3f})' for r, qq in zip(R9, q) if qq < .05)
        add('5 split', f'{col} per area q < .05', None, np.nan, extra=sig or 'none')
    t = d[(d.family == 'dog') & (d.model == 'm8_full_dT') & d.roi.isin(R9) & ~d.subject.isin(excl)]
    s = d[(d.family == 'dog') & (d.model == 'm7_full_sigtied') & d.roi.isin(R9) & ~d.subject.isin(excl)].copy()
    add('5 split', 'm8_full_dT fits', len(t), np.nan)
    add('5 split', 'm7_full_sigtied fits', len(s), np.nan, extra=f'FWHM median {2.355 * s.sigma_AF.median():.1f}°')
    t = t.assign(dT=t.g_T_HP_dyn - t.g_T_dyn, dD=t.g_HP_dyn - t.g_LP_dyn)
    x = (t.dD - t.dT).groupby(t.subject).median()
    add('5 split', 'dT model: dD − dT pooled', len(x), x.median(), extra=f'p = {stats.wilcoxon(x).pvalue:.2g}')


def decoding(excl):
    for src in ('jointhrf', 'gauss_jointhrf', 'dn_jointhrf'):
        q = pd.read_csv(D + f'decoding_quality_{src}_2026-11.tsv', sep='\t')
        q = q[(q.decoder == 'free') & ~q.subject.isin(excl)]
        qs = q.groupby(['subject', 'roi']).r.mean().reset_index()
        for roi in (R9 if src == 'jointhrf' else ['V1']):
            v = qs[qs.roi == roi].r
            add('6b quality', f'{src} {roi}', len(v), v.median())
        if src == 'jointhrf':
            add('6a', 'decodes (subject × run × ROI)', len(q), np.nan)


def dn_locus(excl):
    c = pd.read_csv(D + 'dn_basefield_cv_complete_2026-11.tsv', sep='\t')
    c = c[~c.subject.isin(excl)]
    add('DN locus', 'columns', None, np.nan, extra=', '.join(c.columns[:20]))


def demographics(excl):
    p = pd.read_csv('/data/ds-retsupp/participants.tsv', sep='\t', na_values='n/a')
    p['s'] = p.participant_id.str[4:].astype(int)
    p = p[~p.s.isin(excl)]
    a = p.age.dropna()
    add('demographics', 'n', len(p), len(p))
    add('demographics', 'age', len(a), a.mean(), a.min(), a.max(), f'SD {a.std():.1f}; {len(p) - len(a)} missing')
    add('demographics', 'female', len(p), int((p.sex == 'F').sum()))
    add('demographics', 'male', len(p), int((p.sex == 'M').sum()))


def main(out):
    for tag, ex in (('n30', set()), ('n28', {1, 2})):
        start = len(ROWS)
        for f in (contralateral, behaviour, fig2d, prf_models, fig3_ci, split_per_area, decoding, dn_locus, demographics):
            f(ex)
        for r in ROWS[start:]:
            r['set'] = tag
    t = pd.DataFrame(ROWS)
    t.to_csv(out, sep='\t', index=False)
    with pd.option_context('display.width', 250, 'display.max_rows', 500, 'display.max_colwidth', 120):
        print(t.round(4).to_string(index=False))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', default='notes/data/n28/key_numbers_extras_n28_2026-11-12.tsv')
    main(ap.parse_args().out)
