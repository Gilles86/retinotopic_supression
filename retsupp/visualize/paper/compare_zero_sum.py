"""Refitted-polished reference vs the zero-sum LM reference: the paper's numbers side by side.

Reads two ``reference_numbers`` TSVs (per area and pooled: median over participants in %, bootstrapped 95% CI,
participants below 0, Wilcoxon p, BH q across the nine areas) and writes one markdown table per quantity with both
references. For the zero-sum reference, a third column repeats the joint-fit numbers without the fits in which that
term (or the item scale, which the transient terms scale) ended on a bound of the zero-sum fit (``--zs-summary``, from
``modeling/summarize_zero_sum``), with the same statistics (``reference_numbers.Book``), and a fourth column with the
supplementary model m7_full_zs (rows of that model in the same gain table). The main model is m5_TDS_dS_zs (read as
m7_full from the view root).

    python -m retsupp.visualize.paper.compare_zero_sum \\
        --ref notes/data/reference_numbers_polished_2026-11-12_n28.tsv \\
        --zs notes/data/reference_numbers_zs_lm_2026-11-13_n28.tsv \\
        --zs-gain-table notes/data/gain_field_parameters_zs_lm_2026-11-13.tsv \\
        --zs-summary notes/data/zero_sum_fits_2026-11-13.tsv --out notes/data/compare_zero_sum_2026-11-13
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from retsupp.visualize.paper.fig45_common import joint
from retsupp.visualize.paper.paper_style import ROI_ORDER
from retsupp.visualize.paper.reference_numbers import Book, sci

QUANTITIES = ['dS estimated alone', 'dS joint fit', 'dD estimated alone', 'dT estimated alone',
              'Target gain, joint fit', 'Distractor gain (averaged over its locations), joint fit',
              'Item weight (relative to the bar), joint fit']
# joint-fit quantity: (column of fig45_common.joint, terms whose bound invalidates it, scale)
EXCL = {'dS joint fit': ('dS', ['g_HP'], 100.),
        'Target gain, joint fit': ('T', ['g_T_dyn', 'item_scale'], 100.),
        'Distractor gain (averaged over its locations), joint fit': ('D', ['g_HP_dyn', 'g_LP_dyn', 'item_scale'], 100.),
        'Item weight (relative to the bar), joint fit': ('item_scale', ['item_scale'], 1.)}


def cell(r):
    if r is None or not np.isfinite(r['value']):
        return '—'
    q = r.get('q', np.nan)
    sig = f'q = {sci(q)}' if np.isfinite(q) else f'p = {sci(r["p"])}'
    return f'{r["value"]:+.1f} [{r["ci_lo"]:+.1f}, {r["ci_hi"]:+.1f}], {int(r["k_neg"])}/{int(r["n"])} < 0, {sig}'


def main(ref, zs, zs_gain_table, zs_summary, out, exclude_subjects=(1, 2), main_model='m5_TDS_dS_zs'):
    a = pd.read_csv(ref, sep='\t')
    b = pd.read_csv(zs, sep='\t')
    excl, supp = {}, {}
    if zs_gain_table:
        # the supplementary model (with transient HP - LP terms), same statistics
        g7 = joint(zs_gain_table, 'dog', model='m7_full_zs')
        g7 = g7[~g7.subject.isin(exclude_subjects)]
        if len(g7):
            for qn, (col, _terms, scale) in EXCL.items():
                supp[qn] = Book().term(g7, col, qn, 'm7_full_zs', scale=scale)
    if zs_gain_table and zs_summary:
        g = joint(zs_gain_table, 'dog')
        g = g[~g.subject.isin(exclude_subjects)]
        s = pd.read_csv(zs_summary, sep='\t')
        s = s[s.kept & (s.model == main_model)][['subject', 'roi', 'on_bound']].fillna('')
        g = g.merge(s, on=['subject', 'roi'], how='left')
        g['on_bound'] = g.on_bound.fillna('')
        for qn, (col, terms, scale) in EXCL.items():
            bad = g.on_bound.apply(lambda x: any(t in x.split(',') for t in terms))
            book = Book()
            res = book.term(g[~bad], col, qn, 'excl. on-bound', scale=scale)
            excl[qn] = (res, int(bad.sum()))
    lines, rows = [], []
    for qn in QUANTITIES:
        ra = a[a.quantity == qn].set_index('area')
        rb = b[b.quantity == qn].set_index('area')
        if ra.empty and rb.empty:
            continue
        hdr = f'### {qn} (%)' if 'Item' not in qn else f'### {qn}'
        has_ex, has_s = qn in excl, qn in supp
        lines += [hdr, '', '| area | refitted-polished (alternating) | zero-sum LM, main model |'
                  + (f' main model, excl. on-bound fits (n excl. = {excl[qn][1]}) |' if has_ex else '')
                  + (' zero-sum LM, + transient HP − LP (m7_full_zs) |' if has_s else ''),
                  '|---|---|---|' + ('---|' if has_ex else '') + ('---|' if has_s else '')]
        for area in ROI_ORDER + ['pooled']:
            va = ra.loc[area].to_dict() if area in ra.index else None
            vb = rb.loc[area].to_dict() if area in rb.index else None
            ve = excl[qn][0].get(area) if has_ex else None
            vs = supp[qn].get(area) if has_s else None
            lines.append(f'| {area} | {cell(va)} | {cell(vb)} |' + (f' {cell(ve)} |' if has_ex else '')
                         + (f' {cell(vs)} |' if has_s else ''))
            rows.append(dict(quantity=qn, area=area,
                             **{f'ref_{k}': (va or {}).get(k, np.nan) for k in ('value', 'ci_lo', 'ci_hi', 'n', 'p', 'q')},
                             **{f'zs_{k}': (vb or {}).get(k, np.nan) for k in ('value', 'ci_lo', 'ci_hi', 'n', 'p', 'q')},
                             **{f'zs_excl_{k}': (ve or {}).get(k, np.nan) for k in ('value', 'ci_lo', 'ci_hi', 'n', 'p',
                                                                                    'q')},
                             **{f'm7zs_{k}': (vs or {}).get(k, np.nan) for k in ('value', 'ci_lo', 'ci_hi', 'n', 'p',
                                                                                 'q')}))
        lines.append('')
    open(f'{out}.md', 'w').write('\n'.join(lines))
    pd.DataFrame(rows).to_csv(f'{out}.tsv', sep='\t', index=False)
    print('\n'.join(lines))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ref', required=True)
    ap.add_argument('--zs', required=True)
    ap.add_argument('--zs-gain-table', default=None)
    ap.add_argument('--zs-summary', default=None)
    ap.add_argument('--out', required=True)
    x = ap.parse_args()
    main(x.ref, x.zs, x.zs_gain_table, x.zs_summary, x.out)
