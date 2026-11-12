"""Key numbers of the gain-field results for one reference fit (Fig. 5, its supplements and the parameter table).

Every number the writing package quotes for the gain fields, computed from the tables of one reference fit, so that a
change of reference (e.g. pRFs refitted together with the gain fields, or another voxel selection) is one re-run:

* joint fit (all terms fitted together, m7_full) from ``--gain-table`` (``collect_gain_parameters``): per area and
  pooled target gain, distractor gain (averaged over its locations), sustained gain at LP and at HP, dS, dD, item weight;
* estimated alone (each term added alone to the model with item weight, target, distractor and sustained gain, m4_TDS)
  from ``--conditional-table`` (``collect_conditional_estimates``): dS, dD, dT, S, the distractor gain on top of the
  target term (DonT); dS pooled per pathway;
* field widths among fits with a reliable gain from ``--widths-prefix`` (``field_widths_reliable``);
* the slope of the target and distractor gain with pRF size from ``--slope-table`` (``hierarchy_regression``);
* optional (``--refit-prefix``, ``collect_polished_fits``): how much the refit of the pRFs together with the gain
  fields changed the fit (training SSE, pRF centres, size and amplitude) and the trade-off of the sustained gain with
  the item weight and the pRF amplitude (polished − clamped per participant × area).

Gains in % (× 100); differences between gains in percentage points. Per area: median over participants, bootstrapped
95% CI, participants below 0, Wilcoxon signed-rank p, Benjamini-Hochberg q across the nine areas. Pooled: per
participant the median over the nine areas, then as per area.

Writes ``<out>.tsv`` (one row per number, with its source table) and ``<out>.md`` (the key-numbers text).

    python -m retsupp.visualize.paper.reference_numbers --gain-table notes/data/gain_field_parameters_polished_2026-11-12.tsv \\
        --conditional-table notes/data/conditional_estimates_polished_2026-11-12.tsv \\
        --widths-prefix notes/data/field_widths_reliable_polished_n28_2026-11-12 \\
        --slope-table notes/data/hierarchy_regression_polished_n28_2026-11-12.tsv \\
        --refit-prefix notes/data/prf_refit_2026-11-12 --out notes/data/reference_numbers_polished_n28_2026-11-12
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from retsupp.visualize.paper.fig45_common import conditional, joint, median_ci
from retsupp.visualize.paper.paper_style import ROI_GROUPS, ROI_ORDER


def sci(p):
    """Exact p or q: two significant digits, × 10⁻ⁿ below .001, no leading zero."""
    if not np.isfinite(p):
        return 'n/a'
    if p >= .001:
        return f'{p:.2g}'.replace('0.', '.') if p < .01 else f'{p:.3f}'.replace('0.', '.') if p < .1 else \
            f'{p:.2f}'.replace('0.', '.')
    m, e = f'{p:.1e}'.split('e-')
    return f'{m} × 10' + ''.join('⁰¹²³⁴⁵⁶⁷⁸⁹'[int(c)] for c in str(int(e))).join(['⁻', ''])


def pc(x):
    """A value in % (input already × 100): one decimal below 10, none from 10 up."""
    s = f'{x:+.1f}' if abs(x) < 9.95 else f'{x:+.0f}'
    return s.replace('-', '−') + '%'


def stars(q):
    return '***' if q < .001 else '**' if q < .01 else '*' if q < .05 else ''


class Book:
    def __init__(self):
        self.rows = []

    def term(self, df, col, label, source, scale=100., areas=True, pooled=True):
        out = {}
        ps = []
        for r in ROI_ORDER:
            v = df[df.roi == r][col].dropna().to_numpy() * scale
            m, lo, hi = median_ci(v)
            p = stats.wilcoxon(v).pvalue if len(v) > 5 else np.nan
            ps.append(p)
            out[r] = dict(quantity=label, area=r, value=m, ci_lo=lo, ci_hi=hi, k_neg=int((v < 0).sum()), n=len(v),
                          p=p, source=source)
        for r, q in zip(ROI_ORDER, stats.false_discovery_control(np.nan_to_num(ps, nan=1.))):
            out[r]['q'] = q
        if areas:
            self.rows += list(out.values())
        if pooled:
            v = df.groupby('subject')[col].median().dropna().to_numpy() * scale
            m, lo, hi = median_ci(v)
            out['pooled'] = dict(quantity=label, area='pooled', value=m, ci_lo=lo, ci_hi=hi, k_neg=int((v < 0).sum()),
                                 n=len(v), p=stats.wilcoxon(v).pvalue, q=np.nan, source=source)
            self.rows.append(out['pooled'])
        return out

    def add(self, **k):
        self.rows.append(k)
        return k


def line(out, label, sig_only=False, unit=pc):
    """'pooled −1.3% [−1.8, −0.6], 22/28 < 0, p = …; V3 −2.7%* (q = .03), …'"""
    pl = out.get('pooled')
    s = ''
    if pl:
        s = (f"pooled {unit(pl['value'])} [{unit(pl['ci_lo'])[:-1]}, {unit(pl['ci_hi'])[:-1]}], "
             f"{pl['k_neg']}/{pl['n']} < 0, p = {sci(pl['p'])}; ")
    per = [f"{r} {unit(out[r]['value'])}{stars(out[r]['q'])} (q = {sci(out[r]['q'])})" for r in ROI_ORDER
           if (not sig_only) or out[r]['q'] < .05]
    return f'- **{label}**: ' + s + ('per area ' + ', '.join(per) if per else 'no area q < .05')


def main(a):
    excl = set(a.exclude)
    B = Book()
    js = joint(a.gain_table, 'dog')
    js = js[~js.subject.isin(excl)]
    js['items'] = js.item_scale
    cond = conditional(a.conditional_table)
    cond = cond[~cond.subject.isin(excl)]
    G, C = a.gain_table, a.conditional_table
    md = [f'# Key numbers, gain fields: {a.label}', '',
          f'Joint fit (all terms fitted together, m7_full): `{G}`. Estimated alone (each term added alone to the model '
          f'with item weight, target, distractor and sustained gain, m4_TDS): `{C}`. n = {js.subject.nunique()} '
          'participants. Gains in %; differences between gains in percentage points (written as %). Per area: median '
          'over participants, * BH q < .05 across the nine areas (Wilcoxon). Pooled: per participant the median over the '
          'nine areas; median [bootstrapped 95% CI], participants < 0, Wilcoxon p. Generated by '
          '`retsupp/visualize/paper/reference_numbers.py`; every number is also in the `.tsv` next to this file.', '']

    md += ['## dS, the sustained gain at HP minus at LP (Fig. 5c)', '']
    o = B.term(cond, 'dS', 'dS estimated alone', C)
    md.append(line(o, 'dS estimated alone'))
    pw = []
    for name, rois in ROI_GROUPS:
        v = cond[cond.roi.isin(rois)].groupby('subject').dS.median().dropna() * 100
        m, lo, hi = median_ci(v.to_numpy())
        pw.append(dict(quantity='dS estimated alone, pooled per pathway', area=name, value=m, ci_lo=lo, ci_hi=hi,
                       k_neg=int((v < 0).sum()), n=len(v), p=stats.wilcoxon(v).pvalue, source=C))
    for r, q in zip(pw, stats.false_discovery_control([r['p'] for r in pw])):
        r['q'] = q
        B.add(**r)
    md.append('- **dS estimated alone, per pathway** (per participant the median over the pathway\'s areas; BH across '
              'the three): ' + '; '.join(f"{r['area'].lower()} {pc(r['value'])} [{pc(r['ci_lo'])[:-1]}, "
                                         f"{pc(r['ci_hi'])[:-1]}], {r['k_neg']}/{r['n']} < 0, q = {sci(r['q'])}"
                                         for r in pw))
    o = B.term(js, 'dS', 'dS joint fit', G)
    md.append(line(o, 'dS, joint fit'))

    md += ['', '## Target and distractor gain, joint fit (Fig. 5b)', '']
    for col, lab in (('T', 'Target gain'), ('D', 'Distractor gain (averaged over its locations)')):
        o = B.term(js, col, f'{lab}, joint fit', G)
        md.append(line(o, lab))
    o = B.term(cond, 'DonT', 'Distractor gain on top of the target term, estimated alone', C)
    md.append(line(o, 'Distractor gain on top of the target term, estimated alone'))

    md += ['', '## Transient HP − LP (Fig. 5d)', '']
    for df, col, lab, src in ((cond, 'dD', 'dD estimated alone', C), (cond, 'dT', 'dT estimated alone', C),
                              (js, 'dD', 'dD joint fit', G)):
        o = B.term(df, col, lab, src)
        md.append(line(o, lab))
    both = cond[['subject', 'roi', 'dD', 'dT']].dropna().assign(dDmT=lambda x: x.dD - x.dT)
    o = B.term(both, 'dDmT', 'dD − dT estimated alone', C)
    md.append(line(o, 'dD − dT, estimated alone'))

    md += ['', '## The sustained gain shared by all four locations', '',
           'Quote its level with care: once the pRFs are refitted together with the gain fields, the LP gain trades off '
           'against the item weight and the pRF amplitude (below); its sign and the per-area significance are robust, '
           'its size is not.', '']
    for df, col, lab, src in ((js, 'g_LP', 'Sustained gain at LP, joint fit', G),
                              (js, 'g_HP', 'Sustained gain at HP, joint fit', G),
                              (cond, 'S', 'Sustained gain at all four locations, estimated alone', C)):
        o = B.term(df, col, lab, src)
        md.append(line(o, lab))
    o = B.term(js, 'items', 'Item weight (relative to the bar), joint fit', G, scale=1.)
    md.append(line(o, 'Item weight (relative to the bar, ratio, not %), joint fit',
                   unit=lambda x: f'{x:.2f}' if abs(x) < 10 else f'{x:.0f}'))

    if a.widths_prefix:
        md += ['', '## Field widths among fits with a reliable gain (Fig. 5a maps, Supplementary Fig. field widths)', '']
        w = pd.read_csv(f'{a.widths_prefix}_widths_thr2.tsv', sep='\t')
        cl = pd.read_csv(f'{a.widths_prefix}_claims_thr2.tsv', sep='\t')
        cal = pd.read_csv(f'{a.widths_prefix}_calibration.tsv', sep='\t').iloc[0]
        fits = pd.read_csv(f'{a.widths_prefix}_fits.tsv', sep='\t')
        src = f'{a.widths_prefix}_widths_thr2.tsv'
        sus = w[w.field == 'sustained']
        rel = sus[sus.subset.str.startswith('reliable')].set_index('roi')
        allf = sus[sus.subset == 'all fits'].set_index('roi')
        for r in ROI_ORDER + ['all']:
            B.add(quantity='Sustained FWHM (deg), reliable sustained gain', area=r, value=rel.loc[r, 'median'],
                  ci_lo=rel.loc[r, 'ci_lo'], ci_hi=rel.loc[r, 'ci_hi'], n=rel.loc[r, 'n'], source=src)
            B.add(quantity='Sustained FWHM (deg), all fits', area=r, value=allf.loc[r, 'median'],
                  ci_lo=allf.loc[r, 'ci_lo'], ci_hi=allf.loc[r, 'ci_hi'], n=allf.loc[r, 'n'], source=src)
        ar = cl[cl.roi.str.startswith('all')].iloc[0]
        cl = cl[~cl.roi.str.startswith('all')].set_index('roi')
        B.add(quantity='Sustained FWHM < 5.7 deg, participant medians', area='all', value=ar['median'],
              k_neg=int(round(ar.share_below * ar.n)), n=ar.n, p=ar.p, source=f'{a.widths_prefix}_claims_thr2.tsv')
        md.append(f"- Fits: {len(fits)} participant × area fits, n = {fits.subject.nunique()}; jackknife calibration "
                  f"(|odd − even|/2 / SE, 0.67 expected): sustained HP {cal.g_HP:.2f}, LP {cal.g_LP:.2f}, target "
                  f"{cal.g_T:.2f}. Reliable: |gain| / SE > 2.")
        md.append(f"- **Sustained FWHM, fits with a reliable sustained gain**: {rel.loc['all', 'median']:.1f}° "
                  f"[{rel.loc['all', 'ci_lo']:.1f}, {rel.loc['all', 'ci_hi']:.1f}] (participant medians; all fits "
                  f"{allf.loc['all', 'median']:.1f}°); "
                  + ', '.join(f"{r} {rel.loc[r, 'median']:.1f}° (n = {int(rel.loc[r, 'n'])}; all fits "
                              f"{allf.loc[r, 'median']:.1f}°)" for r in ROI_ORDER) + '.')
        md.append(f"- Narrower than the 5.7° between neighbouring item locations: {int(round(ar.share_below * ar.n))} "
                  f"of {int(ar.n)} participant medians, p = {sci(ar.p)}; per area q < .05 in "
                  + ', '.join(f"{r} (q = {sci(cl.loc[r, 'q'])})" for r in ROI_ORDER if cl.loc[r, 'q'] < .05)
                  + '; not in ' + ', '.join(f"{r} (n = {int(cl.loc[r, 'n'])})" for r in ROI_ORDER
                                            if not cl.loc[r, 'q'] < .05) + '.')
        pr = Path(f'{a.widths_prefix}_early_vs_higher_thr2.tsv')
        if pr.exists():
            e = pd.read_csv(pr, sep='\t').iloc[0]
            B.add(quantity='Sustained FWHM early (V1-V3) minus higher, paired participant medians', area='all',
                  value=e.diff_median, n=e.n_pairs, k_neg=int(e.n_pairs - e.k_pos), p=e.p, source=str(pr))
            md.append(f"- Early (V1–V3) vs higher areas (paired participant medians): {e.early_median:.1f}° vs "
                      f"{e.higher_median:.1f}°, {int(e.k_pos)} of {int(e.n_pairs)} wider in early, p = {sci(e.p)}.")
        tr = fits[fits.g_T_z.abs() > 2]
        md.append(f"- Transient width with a reliable target gain: at a bound of the fit in "
                  f"{100 * tr.bound_dyn.mean():.0f}% of {len(tr)} fits; not interpreted.")
        z = fits.sus_z.abs()
        pwid = fits.prof_width_sus.replace(np.inf, np.nan)
        md.append(f"- Profile interval of the sustained width (median 95%-interval width): "
                  f"{fits.loc[z < 1, 'prof_width_sus'].median():.1f}° at |z| < 1, "
                  f"{fits.loc[(z >= 1) & (z < 2), 'prof_width_sus'].median():.1f}° at 1–2, "
                  f"{fits.loc[(z >= 2) & (z < 3), 'prof_width_sus'].median():.1f}° at 2–3, "
                  f"{fits.loc[z >= 3, 'prof_width_sus'].median():.1f}° at > 3; Spearman ρ (closed intervals) = "
                  f"{stats.spearmanr(z, pwid, nan_policy='omit').statistic:.2f}.")

    if a.slope_table:
        sl = pd.read_csv(a.slope_table, sep='\t')
        sl = sl[sl.rois == 'all nine'].set_index('parameter')
        md += ['', '## Gain against pRF size (Supplementary Fig., gain vs pRF size)', '']
        for lab in ('Target gain', 'Distractor gain'):
            r = sl.loc[lab]
            B.add(quantity=f'{lab} slope per doubling of pRF size', area='all nine', value=100 * r.median_slope,
                  ci_lo=100 * r.ci_low, ci_hi=100 * r.ci_high, n=r.n, p=r.p, q=r.q, source=a.slope_table)
            md.append(f"- {lab}: {pc(100 * r.median_slope)} per doubling [{pc(100 * r.ci_low)[:-1]}, "
                      f"{pc(100 * r.ci_high)[:-1]}], {r.pct_positive:.0f}% of participants positive, p = {sci(r.p)}, "
                      f"q = {sci(r.q)} (BH over the eight parameters).")

    if a.refit_prefix:
        md += ['', '## The refit: pRFs refitted together with the gain fields', '']
        sh = pd.read_csv(f'{a.refit_prefix}_shared.tsv', sep='\t')
        sh = sh[~sh.subject.isin(excl)]
        pol = sh[sh.fit == 'polished']
        rel = 100 * (pol.sse_final - pol.sse_clamped) / pol.sse_clamped
        B.add(quantity='Training SSE change, polished vs clamped (%)', area='all cells', value=rel.median(),
              ci_lo=rel.min(), ci_hi=rel.max(), n=len(rel), k_neg=int((rel < 0).sum()),
              source=f'{a.refit_prefix}_shared.tsv')
        md.append(f"- Training SSE lower in {int((rel < 0).sum())} of {len(rel)} subject × area × model cells "
                  f"(median {rel.median():+.3f}%, range {rel.min():+.3f}% to {rel.max():+.4f}%).".replace('-', '−'))
        ps = pd.read_csv(f'{a.refit_prefix}_prf_shift.tsv', sep='\t')
        ps = ps[~ps.subject.isin(excl)]
        for vox, lab in (('all', 'all voxels'), ('near_items', 'voxels with a centre within 2° of an item location')):
            x = ps[ps.voxels == vox]
            B.add(quantity=f'pRF centre shift (deg), {lab}', area='all cells', value=x.shift_median.median(),
                  ci_hi=x.shift_p90.median(), n=len(x), source=f'{a.refit_prefix}_prf_shift.tsv')
            md.append(f"- pRF centre shift, {lab}: median {x.shift_median.median():.3f}° (90th percentile "
                      f"{x.shift_p90.median():.2f}°; medians over cells); pRF size change "
                      f"{100 * x.dsd_median.median():+.1f}%, amplitude {100 * x.damp_median.median():+.1f}% "
                      "(median relative change).".replace('-', '−') if vox == 'all' else
                      f"- pRF centre shift, {lab}: median {x.shift_median.median():.3f}° (90th percentile "
                      f"{x.shift_p90.median():.2f}°).")
        # trade-off: change of the LP gain against change of item weight and pRF amplitude (m7_full, per cell)
        w = sh[sh.model == 'm7_full'].pivot_table(index=['subject', 'roi'], columns='fit',
                                                  values=['g_LP', 'item_scale']).dropna()
        dg = 100 * (w['g_LP', 'polished'] - w['g_LP', 'clamped'])
        di = w['item_scale', 'polished'] - w['item_scale', 'clamped']
        amp = ps[(ps.model == 'm7_full') & (ps.voxels == 'all')].set_index(['subject', 'roi']).damp_median
        dd = pd.concat([dg.rename('dg'), di.rename('di'), amp.rename('damp')], axis=1).dropna()
        for col, lab in (('di', 'item weight'), ('damp', 'pRF amplitude')):
            rho, p = stats.spearmanr(dd.dg, dd[col])
            within = dd.reset_index().groupby('subject').apply(
                lambda g: stats.spearmanr(g.dg, g[col]).statistic if len(g) > 3 else np.nan,
                include_groups=False).dropna()
            pw_ = stats.wilcoxon(within).pvalue
            B.add(quantity=f'Spearman rho, change of LP gain vs change of {lab} (polished - clamped)', area='cells',
                  value=rho, n=len(dd), p=p, source=f'{a.refit_prefix}_shared.tsv / _prf_shift.tsv')
            B.add(quantity=f'Within-participant rho, change of LP gain vs change of {lab}', area='participants',
                  value=within.median(), n=len(within), p=pw_, source=f'{a.refit_prefix}_shared.tsv / _prf_shift.tsv')
            md.append(f"- Change of the LP gain (polished − clamped, m7_full) against the change of the {lab}: Spearman "
                      f"ρ = {rho:.2f} across {len(dd)} participant × area cells (p = {sci(p)}); within participants "
                      f"median ρ = {within.median():.2f} (Wilcoxon p = {sci(pw_)}, n = {len(within)}).")
        per = dd.reset_index().groupby('roi')[['dg', 'di', 'damp']].median().reindex(ROI_ORDER)
        md.append('- Per area, median change of the LP gain / item weight / pRF amplitude: '
                  + '; '.join(f"{r} {pc(per.loc[r, 'dg'])} / {per.loc[r, 'di']:+.2f} / {100 * per.loc[r, 'damp']:+.1f}%"
                              .replace('-', '−') for r in ROI_ORDER) + '.')

    t = pd.DataFrame(B.rows)
    t.to_csv(f'{a.out}.tsv', sep='\t', index=False)
    md = [('- ' + m[2:]) if m.startswith('− ') else m for m in md]
    md = [m.replace('ρ = -', 'ρ = −').replace('= -0.', '= −0.') for m in md]
    Path(f'{a.out}.md').write_text('\n'.join(md) + '\n')
    print('\n'.join(md))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--gain-table', required=True)
    ap.add_argument('--conditional-table', required=True)
    ap.add_argument('--widths-prefix', default=None)
    ap.add_argument('--slope-table', default=None)
    ap.add_argument('--refit-prefix', default=None, help='collect_polished_fits prefix (…_shared.tsv, …_prf_shift.tsv)')
    ap.add_argument('--exclude', type=int, nargs='*', default=[1, 2])
    ap.add_argument('--label', default='reference fit')
    ap.add_argument('--out', required=True)
    main(ap.parse_args())
