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
        ('m1', 'm0', 'Target gain', 'T', 'T'), ('m2', 'm0', 'Singleton gain, same at all locations', 'D', 'D'),
        ('m3', 'm0', 'Sustained gain, same at all locations', 'S', 'S'), ('m3b', 'm1', 'Singleton gain', 'D given T', 'DonT'),
        ('m4', 'm3b', 'Sustained gain', '', ''), ('m5', 'm4', 'Sustained gain differs at HP', 'dS', 'dS'),
        ('m6', 'm4', 'Singleton gain differs at HP', 'dD', 'dD'), ('m7', 'm5', 'Both HP-specific terms', '', '')]


def main(loso_table, name='table1_ladder'):
    l = pd.read_csv(loso_table, sep='\t')
    cell = lambda term, fam: (f"{l[(l.term == term) & (l.family == fam)].pct_improved.iloc[0]:.0f}"
                             if term and ((l.term == term) & (l.family == fam)).any() else ('pending' if term else ''))
    rows = [(m, p, add, test, cell(key, 'dog'), cell(key, 'dn')) for m, p, add, test, key in ROWS]
    head = ('Model', 'Parent', 'Adds', 'Contrast', 'LOSO % improved, DoG', 'DN')
    md = ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)] + ['| ' + ' | '.join(r) + ' |' for r in rows]
    note = ('LOSO: share of 30 subjects whose held-out prediction improves when the contrast is added with the effect size '
            'from the other subjects (median over ROIs). Provisional: current inputs.')
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f'{name}.md').write_text('**Table 1. The nested gain-field models.**\n\n' + '\n'.join(md) + '\n\n' + note + '\n')
    fig, ax = plt.subplots(figsize=(DOUBLE, 2.6))
    ax.set_axis_off()
    t = ax.table(cellText=rows, colLabels=head, loc='upper center', cellLoc='left', colLoc='left',
                 colWidths=[.07, .07, .36, .12, .2, .08])
    t.auto_set_font_size(False)
    t.set_fontsize(7)
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
    a = ap.parse_args()
    main(a.loso_table)
