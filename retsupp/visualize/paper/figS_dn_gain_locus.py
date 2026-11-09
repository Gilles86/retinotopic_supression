"""Prototype: where does the HP effect act in a divisive-normalization (DN) pRF? (supplementary figure)

a  The DN pRF response with the three places a sustained field at HP can act: the input (stimulus gain), the
   neural baseline b, or the normalization (surround baseline d).
b  Predicted pattern over the three location types (HP quadrant, next to HP, opposite) for each locus, from the
   simulation in ``retsupp/modeling/sim_dn_field_signatures.py`` (median over synthetic voxels, bar on the pRF
   centre), scaled so the HP quadrant is −1.
c  Observed pattern, scaled the same way: measured item responses in early visual cortex (Fig. 3e) and the decoded
   bar (V1–V3, outline given, Fig. 6).
d  Placeholder for the model comparison of the DN gain-field variants (fits running).

Usage::

    python -m retsupp.visualize.paper.figS_dn_gain_locus
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from retsupp.visualize.paper.paper_style import DOUBLE, FG, HP, MUTED, letter, offset_spines, save

LOC = ['HP', 'next', 'opposite']
LOC_LAB = ['HP\nquadrant', 'Next\nto HP', 'Opposite']
LOC_COL = [HP, '#52606D', '#9AA5B1']
VARIANTS = [('c_input_gain', 'Input gain (DN)', '#0077B6'),
            ('c_input_gain_dog', 'Input gain (DoG)', '#7B8794'),
            ('a_neural_baseline', 'Neural baseline b (DN)', '#D62828'),
            ('b_surround_baseline', 'Normalization d (DN)', '#7B2CBF')]
LEGEND = """
**Figure S (prototype). Where could the HP effect act in a DN pRF?** a, The divisive-normalization pRF response: drive
in the pRF centre plus the neural baseline b, divided by drive in the suppressive surround plus the surround baseline
d. A sustained field at the HP location can scale the input (stimulus gain, before pooling), b, or d. b, Predicted
change per location type (HP quadrant, next to HP, opposite), median over synthetic voxels on the four diagonals at
1–3° with DN parameters in the range of the fits, bar on the pRF centre (``sim_dn_field_signatures.py``), each location type minus the mean of the four
locations as for the data, scaled and signed so that the HP quadrant is −1; the DoG pRF with the same input field for comparison. Only DN with an input field lifts
the quadrants next to HP (release from normalization); the baseline fields are local. c, Observed, scaled the same
way: responses to the search items in early visual cortex at the three location types (Fig. 3e; mean ± SEM, n = 30)
and the decoded bar intensity per quadrant (V1–V3, bar outline given, joint-HRF DoG pRFs, Fig. 6). d, Model
comparison of the DN gain-field variants (input field only; plus a field on b, on d, or both) on held-out data and
in a recovery simulation: fits running. Prototype: panels b and c are from existing analyses; d is a placeholder.
"""


def predicted():
    d = pd.read_csv('notes/data/dn_field_signatures_2026-11.tsv', sep='\t')
    d = d[d.bar_class == 'centre']
    m = d.groupby(['variant', 'quadrant']).rel_change.median().unstack()[LOC]
    # Same reference as the data: each location type minus the mean of the four locations (next counts twice),
    # then scaled and signed so that the HP quadrant is −1 (the sign of the field is free).
    m = m.sub((m.HP + 2 * m.next + m.opposite) / 4, axis=0)
    return m.div(-m.HP, axis=0)


def observed():
    from retsupp.visualize.paper.fig3_item_locations import rel_stats
    _, cells, _ = rel_stats('notes/data/event_deconv_relative_2026-11')
    e = cells[cells.grp == 'Early'].pivot_table(index='subject', columns='loc', values='r')   # mean over items
    e = pd.DataFrame({'HP': e.hp, 'next': (e.nextleft + e.nextright) / 2, 'opposite': e.opposite})
    e = e.sub(e.mean(1), axis=0)
    q = pd.read_csv('notes/data/decoding_prf_models_quadrants_2026-11.tsv', sep='\t')
    q = q[(q.source == 'jointhrf') & (q.decoder == 'outline')].iloc[0]
    dec = pd.DataFrame({'m': [q.hp, q.adjacent, q.opposite], 'se': [q.hp_sem, q.adjacent_sem, q.opposite_sem]}, index=LOC)
    meas = pd.DataFrame({'m': e.mean(), 'se': e.sem()})
    return {'Item responses, early areas': meas / abs(meas.m.HP), 'Decoded bar, V1–V3': dec / abs(dec.m.HP)}


def panel_scheme(ax):
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(.5, .78, r'$N \;=\; \dfrac{a\,\langle S_{\mathrm{eff}},\,G_c\rangle \;+\; b}'
                     r'{c\,\langle S_{\mathrm{eff}},\,G_s\rangle \;+\; d} \;-\; \dfrac{b}{d}$',
            ha='center', va='center', fontsize=9, color=FG)
    ax.text(.5, .5, r'$S_{\mathrm{eff}} = S\cdot(1 + \mathrm{field})$', ha='center', va='center', fontsize=8, color='#0077B6')
    for y, (lab, col) in zip((.3, .18, .06), (('Input gain: field scales the stimulus S', '#0077B6'),
                                               ('Baseline: field scales b', '#D62828'),
                                               ('Normalization: field scales d', '#7B2CBF'))):
        ax.plot(.06, y, 's', ms=5, color=col, transform=ax.transAxes, clip_on=False)
        ax.text(.1, y, lab, fontsize=6.5, color=col, va='center', transform=ax.transAxes)


def bars(ax, table, rows, title):
    x = np.arange(len(LOC))
    w = .8 / len(rows)
    for k, (key, lab, col) in enumerate(rows):
        v = table.loc[key] if isinstance(table, pd.DataFrame) else table[key]
        ax.bar(x - .4 + w * (k + .5), v.values, w * .9, color=col, lw=0, label=lab)
    ax.axhline(0, color='.6', lw=.7)
    ax.set_xticks(x)
    ax.set_xticklabels(LOC_LAB, fontsize=6.5)
    ax.set_ylabel('Change (HP quadrant = −1)', fontsize=7)
    ax.set_title(title, fontsize=7, color=FG, loc='left')
    ax.legend(frameon=False, fontsize=6, loc='lower right')
    offset_spines(ax)


def main(name='figS_dn_gain_locus'):
    fig = plt.figure(figsize=(DOUBLE, 4.6))
    gs = fig.add_gridspec(2, 2, left=.07, right=.98, top=.93, bottom=.1, wspace=.3, hspace=.55)
    ax = fig.add_subplot(gs[0, 0])
    panel_scheme(ax)
    letter(ax, 'a', x=-.05, y=1.05)
    ax = fig.add_subplot(gs[0, 1])
    bars(ax, predicted(), VARIANTS, 'Predicted, per locus of the HP field')
    letter(ax, 'b', x=-.18, y=1.08)
    ax = fig.add_subplot(gs[1, 0])
    obs = observed()
    x = np.arange(len(LOC))
    for k, ((lab, t), col) in enumerate(zip(obs.items(), (FG, '#E07A5F'))):
        ax.errorbar(x + (k - .5) * .2, t.m.values, yerr=t.se.values, fmt='o', color=col, ms=5, lw=1.2, capsize=0, label=lab)
    ax.axhline(0, color='.6', lw=.7)
    ax.set_xticks(x)
    ax.set_xticklabels(LOC_LAB, fontsize=6.5)
    ax.set_ylabel('Change (HP quadrant = −1)', fontsize=7)
    ax.set_title('Observed', fontsize=7, color=FG, loc='left')
    ax.legend(frameon=False, fontsize=6, loc='lower right')
    offset_spines(ax)
    letter(ax, 'c', x=-.18, y=1.08)
    ax = fig.add_subplot(gs[1, 1])
    ax.set_axis_off()
    ax.text(.5, .5, 'Model comparison of the DN gain-field variants\n(held-out fit and recovery simulation)\n— fits running —',
            ha='center', va='center', fontsize=7, color=MUTED)
    letter(ax, 'd', x=-.05, y=1.08)
    save(fig, name, LEGEND)


if __name__ == '__main__':
    main()
