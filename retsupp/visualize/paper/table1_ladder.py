"""Table 1: the ladder of nested gain-field models, with LOSO per step. Markdown and PDF.

Usage::

    python -m retsupp.visualize.paper.table1_ladder --loso-table notes/data/loso_per_step_2026-11.tsv
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import pandas as pd

from retsupp.visualize.paper.paper_style import DOUBLE, FG, MUTED, OUT

ROWS = [('m00', '', 'pRF on the bar', '', ''), ('m0', 'm00', 'Search items, weight fitted', 'Items', 'items'),
        ('m1', 'm0', 'Target gain', 'T', 'T'), ('m2', 'm0', 'Distractor gain, same at all locations', 'D', 'D'),
        ('m3', 'm0', 'Sustained gain, same at all locations', 'S', 'S'), ('m3b', 'm1', 'Distractor gain', 'D given T', 'DonT'),
        ('m4', 'm3b', 'Sustained gain', '', ''), ('m5', 'm4', 'Sustained gain differs at HP', 'dS', 'dS'),
        ('m6', 'm4', 'Distractor gain differs at HP', 'dD', 'dD'), ('m7', 'm5', 'Both HP-specific terms', '', '')]


def main(loso_table, name='table1_ladder', families=('dog', 'dn'), final=False):
    l = pd.read_csv(loso_table, sep='\t')
    fam_lab = {'dog': 'DoG', 'dn': 'DN', 'gauss': 'Gaussian'}

    def cell(term, fam):
        r = l[(l.term == term) & (l.family == fam)]
        if not term:
            return ''
        if not len(r):
            return 'pending'
        r = r.iloc[0]
        if not final:
            return f'{r.pct_improved:.0f}'
        txt = f'{r.pct_improved:.0f}% (p = {r.p:.2g})'
        if 'sign_pct' in r and pd.notna(r.sign_pct):
            txt += f'; {r.sign_pct:.0f}% (p = {r.sign_p:.2g})'
        return txt
    extra = [('m8', 'm4', 'Target gain differs at HP', 'dT', 'dT')] if final else []
    rows = [(m, p, add, test, *[cell(key, f) for f in families]) for m, p, add, test, key in ROWS + extra]
    head = ('Model', 'Parent', 'Adds', 'Contrast', *[f'{fam_lab[f]}: better; direction' if final else
                                                    (f'LOSO % improved, {fam_lab[f]}' if i == 0 else fam_lab[f])
                                                    for i, f in enumerate(families)])
    md = ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)] + ['| ' + ' | '.join(r) + ' |' for r in rows]
    note = ('LOSO: share of 30 subjects whose held-out prediction improves when the contrast is added with the effect size '
            'from the other subjects (median over ROIs). Provisional: current inputs.')
    if final:
        note = ('How much each term adds to out-of-subject prediction (leave-one-subject-out, final fits; each pRF model with '
                'its own jointly fitted HRF per subject × ROI). Better: share of 30 subjects whose ΔR² on held-out data is '
                'positive when the term is added at the median of the other 29 subjects\' own estimates (signal voxels; '
                'per subject the median over ROIs), Wilcoxon signed-rank p. Direction: sign-flip test, ΔR² at +δ minus at '
                '−δ, which removes the cost of adding a term (a true 50% null). Items: the item weight of pRFs fitted to the '
                'bar alone, scored against items at bar strength; not a test of items against no items.')
    OUT.mkdir(parents=True, exist_ok=True)
    title = ('**Table S. What each gain-field term adds to out-of-subject prediction.**' if final else
             '**Table 1. The nested gain-field models.**')
    (OUT / f'{name}.md').write_text(title + '\n\n' + '\n'.join(md) + '\n\n' + note + '\n')
    fig, ax = plt.subplots(figsize=(DOUBLE, 2.6))
    ax.set_axis_off()
    ncol = len(head) - 4
    t = ax.table(cellText=rows, colLabels=head, loc='upper center', cellLoc='left', colLoc='left',
                 colWidths=[.07, .07, .36, .12, .2, .08] if ncol == 2 and not final else
                 [.06, .06, .25, .09] + [.54 / ncol] * ncol)
    t.auto_set_font_size(False)
    t.set_fontsize(6 if final else 7)
    for (r, c), cl in t.get_celld().items():
        cl.set_edgecolor('#D5DAE0' if r else FG)
        cl.set_linewidth(.4)
        cl.visible_edges = 'B'
        if r == 0:
            cl.set_text_props(fontweight='bold', color=FG)
        if cl.get_text().get_text() == 'pending':
            cl.set_text_props(color=MUTED)
    ax.text(0, -.02, note, transform=ax.transAxes, fontsize=6, color=MUTED, va='top', wrap=True)
    fig.savefig(OUT / f'{name}.pdf')
    fig.savefig(OUT / f'{name}.png', dpi=200)
    print('Saved', OUT / f'{name}.pdf')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--loso-table', default='notes/data/loso_per_step_2026-11.tsv')
    ap.add_argument('--name', default='table1_ladder')
    ap.add_argument('--families', nargs='+', default=['dog', 'dn'])
    ap.add_argument('--final', action='store_true', help='supplementary version: better and direction with p per family')
    a = ap.parse_args()
    main(a.loso_table, a.name, a.families, a.final)
