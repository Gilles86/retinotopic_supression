"""Numbers for writing-package update 2 (n = 28): exact k of n for the pooled effects, missing bootstrapped CIs, and
the kappa / DN-baseline checks. Every value comes from an existing per-subject table (no fits).

    python -m retsupp.visualize.paper.update2_numbers --out notes/data/update2_numbers_n28_2026-11-12.tsv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from retsupp.visualize.paper import sensitivity_excl_subjects as sx
from retsupp.visualize.paper.paper_style import ROI_ORDER

D = Path('notes/data')
EARLY = ['V1', 'V2', 'V3']
ROWS = []


def boot(x, fn=np.median, n_boot=10000, seed=0):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    b = fn(np.random.default_rng(seed).choice(x, (n_boot, len(x))), axis=1)
    return fn(x), *np.percentile(b, [2.5, 97.5])


def wil(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return stats.wilcoxon(x).pvalue


def add(item, label, s, sign, source, stat='median', extra=''):
    """s: per-subject values. k = subjects in the direction ``sign`` (sign 0: no direction)."""
    s = pd.Series(s).dropna()
    fn = np.median if stat == 'median' else np.mean
    m, lo, hi = boot(s, fn)
    k = int(((s * sign) > 0).sum()) if sign else np.nan
    ROWS.append(dict(item=item, label=label, stat=stat, value=m, ci_lo=lo, ci_hi=hi, sem=s.sem(), k=k, n=len(s),
                     p=wil(s), source=source, extra=extra))


def per_subject(d, col):
    return d.groupby('subject')[col].median()


def counts_and_cis():
    gp = 'notes/data/gain_field_parameters_final2_2026-11.tsv'
    for fam in ('dog', 'gauss', 'dn'):
        j = sx.joint(fam)
        for col, lab, sgn in (('T', 'Target gain, joint', 1), ('D', 'Distractor gain, joint', 1), ('dS', 'dS, joint', -1),
                              ('dD', 'dD, joint', -1)):
            add('B4/B5', f'{fam}: {lab}', per_subject(j, col), sgn, gp + ' (m7_full)')
        j['gLP'], j['gHP'] = j.g_LP, j.g_HP
        if fam == 'dog':
            add('B4', 'dog: g_LP, joint', per_subject(j, 'gLP'), -1, gp + ' (m7_full)')
            add('B4', 'dog: g_HP, joint', per_subject(j, 'gHP'), -1, gp + ' (m7_full)')
    c = sx.keep(pd.read_csv(D / 'conditional_estimates_dog_final2_2026-11.tsv', sep='\t'))
    c = c[c.valid & c.roi.isin(ROI_ORDER)].pivot_table(index=['subject', 'roi'], columns='mech', values='estimate').reset_index()
    c['dDmdT'] = c.dD - c.dT
    src = 'notes/data/conditional_estimates_dog_final2_2026-11.tsv'
    for col, lab, sgn in (('dS', 'dS alone', -1), ('S', 'S alone', -1), ('DonT', 'Distractor on top of target, alone', 1),
                          ('D', 'Distractor alone on the item model', -1), ('dD', 'dD alone', -1), ('dT', 'dT alone', -1),
                          ('dDmdT', 'dD − dT alone', 1), ('T', 'Target gain alone', 1)):
        add('B4/B5', f'dog: {lab}', per_subject(c, col), sgn, src)
    e = sx.keep(pd.read_csv(D / 'conditional_exact_final2_allfam_2026-11.tsv', sep='\t'))
    e = e[e.roi.isin(ROI_ORDER) & (e.nvox >= 20)]
    for fam in ('gauss', 'dn'):
        x = e[e.family == fam].pivot_table(index=['subject', 'roi'], columns='mech', values='own').reset_index()
        for col, lab, sgn in (('dS', 'dS alone (exact)', -1), ('dD', 'dD alone (exact)', -1), ('dT', 'dT alone (exact)', -1),
                              ('S', 'S alone (exact)', -1), ('T', 'Target gain alone (exact)', 1)):
            add('B4/B5', f'{fam}: {lab}', per_subject(x, col), sgn, 'notes/data/conditional_exact_final2_allfam_2026-11.tsv')
    a = sx.joint(model='m8_full_adjopp')
    a['next_opp'] = a.g_adj_minus_opp
    a['hp_opp'] = a.g_HP - a.g_LP + 2 * a.g_adj_minus_opp / 3
    a['hp_next'] = a.g_HP - a.g_LP - a.g_adj_minus_opp / 3
    for col, lab in (('next_opp', 'next − opposite'), ('hp_next', 'HP − next'), ('hp_opp', 'HP − opposite')):
        add('B4/B5', f'dog split model: sustained {lab}', per_subject(a, col), -1, gp + ' (m8_full_adjopp)')
    s = sx.joint(model='m7_full_sigtied')
    add('B4', 'dog one-width model: dS', per_subject(s, 'dS'), -1, gp + ' (m7_full_sigtied)')
    t = sx.joint(model='m8_full_dT')
    t['tHL'] = t.g_T_HP_dyn - t.g_T_dyn
    add('B4', 'dog target split: target HP − LP', per_subject(t, 'tHL'), 1, gp + ' (m8_full_dT)')
    # hierarchy: share already stored as a float
    h = pd.read_csv(D / 'n28/hierarchy_regression_final2_2026-11.tsv', sep='\t')
    for par in ('Target gain', 'Distractor gain'):
        r = h[(h.parameter == par) & (h.rois == 'all nine')].iloc[0]
        ROWS.append(dict(item='B4', label=f'{par} per doubling of pRF size', stat='median slope', value=r.median_slope,
                         ci_lo=r.ci_low, ci_hi=r.ci_high, k=int(round(r.pct_positive * r.n / 100)), n=int(r.n), p=r.p,
                         source='notes/data/n28/hierarchy_regression_final2_2026-11.tsv', extra='k = slopes > 0'))
    # LOSO, DoG per subject
    lo = pd.read_csv(D / 'loso_exact_final2_excl0102_2026-11.tsv', sep='\t')
    for m in ('T', 'D', 'DonT', 'S', 'dS', 'dD', 'dT', 'items'):
        x = lo[lo.mech == m]
        better = x.pivot(index='subject', columns='roi', values='dr2').median(axis=1).dropna()
        sign = x.pivot(index='subject', columns='roi', values='dr2_sign').median(axis=1).dropna()
        ROWS.append(dict(item='B4', label=f'dog LOSO {m}: better', stat='share', value=(better > 0).mean(), k=int((better > 0).sum()),
                         n=len(better), p=wil(better), source='notes/data/loso_exact_final2_excl0102_2026-11.tsv'))
        if len(sign):
            ROWS.append(dict(item='B4', label=f'dog LOSO {m}: direction', stat='share', value=(sign > 0).mean(),
                             k=int((sign > 0).sum()), n=len(sign), p=wil(sign),
                             source='notes/data/loso_exact_final2_excl0102_2026-11.tsv'))
    # LOSO Gaussian / DN: stored shares (exact floats) x n
    la = pd.read_csv(D / 'n28/loso_per_step_allfam_final2_2026-11.tsv', sep='\t')
    for r in la[la.family != 'dog'].itertuples():
        for col, pcol, lab in (('pct_improved', 'p', 'better'), ('sign_pct', 'sign_p', 'direction')):
            v = getattr(r, col)
            if pd.notna(v):
                ROWS.append(dict(item='B4', label=f'{r.family} LOSO {r.term}: {lab}', stat='share', value=v / 100,
                                 k=int(round(v * r.n / 100)), n=int(r.n), p=getattr(r, pcol),
                                 source='notes/data/n28/loso_per_step_allfam_final2_2026-11.tsv'))
    # kappa and DN baseline LOSO
    for f, mechs in (('loso_kappa_2026-11_n28.tsv', None), ('loso_dnw_2026-11_n28.tsv', None)):
        x = pd.read_csv(D / 'n28' / f, sep='\t')
        for m in sorted(x.mech.unique()):
            xm = x[x.mech == m]
            better = xm.pivot(index='subject', columns='roi', values='dr2').median(axis=1).dropna()
            sign = xm.pivot(index='subject', columns='roi', values='dr2_sign').median(axis=1).dropna()
            for lab, v in (('better', better), ('direction', sign)):
                ROWS.append(dict(item='B4', label=f'LOSO {m}: {lab}', stat='share', value=(v > 0).mean(), k=int((v > 0).sum()),
                                 n=len(v), p=wil(v), source=f'notes/data/n28/{f}'))
    # kappa full fits, per participant mean over areas (as figS_dog_kappa)
    k = sx.keep(pd.read_csv(D / 'dog_cs_field_full_2026-11.tsv', sep='\t'))
    k = k[k.roi.isin(ROI_ORDER) & (k.model == 'm8_full_dogk')]
    add('B4', 'kappa, full fits (per participant mean over areas)', k.groupby('subject').kappa.mean(), -1,
        'notes/data/dog_cs_field_full_2026-11.tsv')
    r3 = pd.read_csv(D / 'figS_dog_kappa_n28_2026-11-11.tsv', sep='\t')
    for r in r3.itertuples():
        if r.roi == 'pooled' or str(r.roi).lower().startswith('pool'):
            ROWS.append(dict(item='B4', label=f'kappa figure, pooled {r.quantity}', stat='median', value=r.median,
                             ci_lo=r.ci_lo, ci_hi=r.ci_hi, k=int(round(r.pct_neg * r.n / 100)), n=int(r.n), p=r.p,
                             source='notes/data/figS_dog_kappa_n28_2026-11-11.tsv', extra='k = below 0'))
    kn = pd.read_csv(D / 'n28/kappa_neighbour_2026-11_n28.tsv', sep='\t')
    for r in kn[kn.pct_pos.notna()].itertuples():
        ROWS.append(dict(item='B4', label=f'kappa neighbour {r.readout} {r.source} {r.group} {r.contrast}', stat='mean',
                         value=r.mean, sem=r.sem, k=int(round(r.pct_pos * r.n / 100)), n=int(r.n), p=r.p,
                         source='notes/data/n28/kappa_neighbour_2026-11_n28.tsv', extra='k = above 0'))


def b5_decoding():
    from retsupp.visualize.paper import fig6_decoding as f6
    from retsupp.visualize.paper.figS_decoded_bars import _half_contrasts
    d = sx.keep(pd.read_csv(D / 'decoding_readout_jointhrf_2026-11.tsv', sep='\t'))
    s = f6.quadrant_effects(d, 'free')
    src = 'notes/data/decoding_readout_jointhrf_2026-11.tsv (Fig. 6d, free decoder, V1–V3)'
    for lab, v, sgn in (('next − opposite', s.adjacent - s.opposite, 1), ('HP − opposite', s.hp - s.opposite, -1),
                        ('HP − next', s.hp - s.adjacent, -1)):
        add('B5', f'Fig. 6d quadrants: {lab}', v, sgn, src, stat='mean')
    seg = sx.keep(pd.read_csv(D / 'decoded_bars_segments_jointhrf_outline_b11_2026-11.tsv', sep='\t'))
    for o in ('pooled', 'vertical', 'horizontal'):
        hp_half, nxt_half, _ = _half_contrasts(seg, EARLY, o)
        add('B5', f'Decoded half-bars {o}: HP half − other half', hp_half, -1,
            'notes/data/decoded_bars_segments_jointhrf_outline_b11_2026-11.tsv', stat='mean')
        add('B5', f'Decoded half-bars {o}: next − opposite half', nxt_half, 1,
            'notes/data/decoded_bars_segments_jointhrf_outline_b11_2026-11.tsv', stat='mean')


def b7():
    k = sx.keep(pd.read_csv(D / 'dog_cs_field_full_2026-11.tsv', sep='\t'))
    k = k[k.roi.isin(ROI_ORDER) & (k.model == 'm8_full_dogk')].copy()
    k['chg'] = k.dS - k.dS_m7
    add('B7', 'dS with kappa − dS without (m7), per participant median over areas', k.groupby('subject').chg.median(), 1,
        'notes/data/dog_cs_field_full_2026-11.tsv (rows m8_full_dogk; dS − dS_m7)', extra='k = participants with a larger dS')
    add('B7', 'dS with kappa (pooled)', k.groupby('subject').dS.median(), -1, 'notes/data/dog_cs_field_full_2026-11.tsv')
    add('B7', 'dS without kappa (m7, same table, pooled)', k.groupby('subject').dS_m7.median(), -1,
        'notes/data/dog_cs_field_full_2026-11.tsv')
    b = sx.keep(pd.read_csv(D / 'dn_basefield_full_2026-11.tsv', sep='\t'))
    b = b[(b.model == 'm8_full_dnbd') & b.roi.isin(ROI_ORDER)]
    c = b.corr_w_b_w_d.dropna()
    ROWS.append(dict(item='B7', label='DN both baselines: Gauss-Newton corr(w_b, w_d), median over subject × area fits',
                     stat='median', value=c.median(), ci_lo=c.quantile(.05), ci_hi=c.quantile(.95), n=len(c),
                     k=b.subject.nunique(), source='notes/data/dn_basefield_full_2026-11.tsv (model m8_full_dnbd)',
                     extra='ci_lo/ci_hi here = 5th/95th percentile over fits; k = participants'))
    ps = b.groupby('subject').corr_w_b_w_d.median()
    add('B7', 'DN both baselines: corr(w_b, w_d), per participant median', ps, 0, 'notes/data/dn_basefield_full_2026-11.tsv')


def main(out):
    sx.EXCL = {1, 2}
    counts_and_cis()
    b5_decoding()
    b7()
    t = pd.DataFrame(ROWS)
    t.to_csv(out, sep='\t', index=False)
    pd.set_option('display.width', 250)
    print(t[['item', 'label', 'value', 'ci_lo', 'ci_hi', 'sem', 'k', 'n', 'p']].round(4).to_string())


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', default='notes/data/update2_numbers_n28_2026-11-12.tsv')
    main(ap.parse_args().out)
