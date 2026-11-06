"""Figure: responses at a pRF's item location, 2 × 3 (HP vs elsewhere × target, distractor, neutral item); data only.

Reads ``notes/data/event_average_2026-11/sub-XX_2026-11.tsv`` (``fig3_event_average.py``: exact onsets, 0.4 s bins) and,
if present, ``notes/data/bar_pass_average_2026-11/`` (``fig_bar_pass_average.py``).

Baseline: per subject, one value for all classes, the mean over all classes from -4 to 0 s; so any difference between
classes before onset stays visible (carry-over from the previous trial would show there).

Usage::

    python -m retsupp.visualize.paper.fig_event_responses_2x3
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from retsupp.visualize.paper.paper_style import (DISTRACTOR, DOUBLE, FG, HP, MUTED, ROI_ORDER, TARGET, letter, offset_spines,
                                                 save, title_icon)

NEUTRAL = '#7B8794'
LP_GREY = '#3E4C59'
PEAK = (4.0, 7.2)
ITEMS = [('target', 'Target', TARGET, ('target',)), ('distractor', 'Distractor', DISTRACTOR, ('distractor',)),
         ('neutral', 'Neutral item', NEUTRAL, ())]
PRF_TEAL = '#2A9D8F'
LEGEND = """
**Figure. The response at a pRF's item location depends on what appears there and on whether it is the HP location.**
Voxels with cross-run reliability > 0 and a pRF centre within 2° of one of the four item locations (all nine ROIs, no
fit-quality criterion, no gain-field model). a, What is measured. b-d, Response after each search onset when the target
(b), the distractor (c) or a neutral item (d) appears at the voxel's location, with that location being the HP location
(red) or an LP location (dark). The response to the bar (the pRF prediction for the bar alone) is removed. Exact onsets,
jittered against the 1.6 s TR, binned at 0.4 s; averaged over trials and voxels per subject, ROIs pooled weighted by
voxel count; one baseline per subject for all classes (−4 to 0 s). Lines and bands: mean ± SEM across subjects (n = {n});
grey strip, the search display (1.5 s); grey column, the window used in e. Trials follow each other every few seconds, so
epochs also contain neighbouring trials. e, Mean response in the window {p0:.1f}-{p1:.1f} s: points and bars, mean and
95% within-subject CI (Cousineau-Morey); lines join the HP and LP values of each item. Wilcoxon signed-rank tests HP vs
LP per item: {tests}.{bar}
"""


def load(folder):
    d = pd.concat([pd.read_csv(f, sep='\t') for f in sorted(Path(folder).glob('sub-*_2026-11.tsv'))])
    d = d[d.roi.isin(ROI_ORDER)].copy()
    d['wdata'] = d.data * d.n_voxels
    g = d.groupby(['subject', 'cls', 't'])
    p = (g.wdata.sum() / g.n_voxels.sum()).rename('data').reset_index()
    base = p[p.t < 0].groupby('subject').data.mean()
    p['data'] -= p.subject.map(base)
    p[['item', 'loc']] = p.cls.str.rsplit('_', n=1, expand=True)
    return p


def load_deconv(folder):
    """nideconv responses (deconvolve_location_responses.py), pooled over ROIs and locations weighted by voxel count."""
    d = pd.concat([pd.read_csv(f, sep='\t') for f in sorted(Path(folder).glob('sub-*_2026-11.tsv'))])
    d = d[d.roi.isin(ROI_ORDER)].copy()
    d['t'] = d.t.round(2)
    d['w'] = d.response * d.n_voxels
    g = d.groupby(['subject', 'cls', 't'])
    p = (g.w.sum() / g.n_voxels.sum()).rename('data').reset_index()
    p[['item', 'loc']] = p.cls.str.rsplit('_', n=1, expand=True)
    return p


def schematic(ax):
    from matplotlib.patches import Circle
    ring = 4. / np.sqrt(2) * np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]])
    ax.add_patch(Circle((0, 0), 3.17, fill=False, ec=MUTED, lw=.6))
    for a in np.deg2rad(np.arange(8) * 45):
        ax.add_patch(plt.Rectangle((4 * np.cos(a) - .45, 4 * np.sin(a) - .12), .9, .24, color='.6', lw=0))
    ax.add_patch(Circle(ring[1], 1.2, fill=False, ec=PRF_TEAL, lw=1.4))
    ax.add_patch(Circle(ring[3], 1.15, fill=False, ec=HP, lw=1., ls=(0, (2, 1.2))))
    ax.plot(0, 0, '+', color=FG, ms=4, mew=.6)
    ax.annotate('pRF on an\nitem location', xy=ring[1] + [.75, .95], xytext=(-.4, 7.4), fontsize=6.5, color=PRF_TEAL, ha='left',
                va='center', arrowprops=dict(arrowstyle='-|>', color=PRF_TEAL, lw=1., mutation_scale=7, shrinkA=1, shrinkB=2,
                                             relpos=(0, .5)))
    ax.text(ring[3, 0], ring[3, 1] - 1.9, 'HP', color=HP, fontsize=6.5, ha='center', va='top')
    ax.text(0, -6.6, 'Response after each\nsearch onset, by item\nand by HP location',
            ha='center', va='top', fontsize=6.5, color=FG, linespacing=1.25)
    ax.set_xlim(-5.8, 5.8)
    ax.set_ylim(-11.5, 8.8)
    ax.set_aspect('equal')
    ax.set_axis_off()


def timecourse(ax, d, item, col):
    for loc, c, lab in (('lp', LP_GREY, 'Elsewhere'), ('hp', HP, 'At HP')):
        m = d[(d.item == item) & (d['loc'] == loc)].groupby('t').data
        mu, se = m.mean(), m.sem()
        ax.fill_between(mu.index, mu - se, mu + se, color=c, alpha=.2, lw=0, zorder=1)
        ax.plot(mu.index, mu, color=c, lw=1.4, zorder=2)
    ax.axvspan(*PEAK, color='#ECEFF2', lw=0, zorder=0)
    ax.axhline(0, color='.75', lw=.6, zorder=0)
    ax.set_xlim(-4, 20)
    ax.set_xticks([0, 5, 10, 15, 20])
    offset_spines(ax)


def search_strip(ax):
    lo, hi = ax.get_ylim()
    ax.add_patch(plt.Rectangle((0, lo), 1.5, (hi - lo) * .04, color=MUTED, lw=0, zorder=3, clip_on=False))
    ax.text(1.8, lo + (hi - lo) * .02, 'Search', fontsize=6, color='.4', va='center', ha='left')


def cm_ci(w):
    """Cousineau-Morey 95% CI per column of a subject × condition table."""
    k = w.shape[1]
    norm = w.sub(w.mean(1), axis=0) + w.values.mean()
    return norm.sem() * np.sqrt(k / (k - 1)) * stats.t.ppf(.975, len(w) - 1)


def peaks(ax, d):
    pk = d[d.t.between(*PEAK)].groupby(['subject', 'item', 'loc']).data.mean().unstack(['item', 'loc'])
    ci = cm_ci(pk)
    tests = []
    for i, (item, lab, col, _) in enumerate(ITEMS):
        m = [pk[(item, loc)].mean() for loc in ('lp', 'hp')]
        e = [ci[(item, loc)] for loc in ('lp', 'hp')]
        xs = [i - .13, i + .13]
        ax.plot(xs, m, color=col, lw=1., zorder=1)
        for x, mm, ee, c in zip(xs, m, e, (LP_GREY, HP)):
            ax.errorbar(x, mm, yerr=ee, fmt='o', ms=5, color=c, mec='w', mew=.7, lw=1.3, capsize=0, zorder=3)
        diff = (pk[(item, 'hp')] - pk[(item, 'lp')]).dropna()
        p = stats.wilcoxon(diff).pvalue
        tests.append(f'{lab.lower()} {diff.mean():+.3f}% (HP − LP; {(diff < 0).mean() * 100:.0f}% of subjects lower at HP), p = {p:.2g}')
        st = '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else ''
        ax.text(i, max(m) + max(e) + .015, st, ha='center', va='bottom', fontsize=7, color=HP)
    ax.set_xticks(range(len(ITEMS)))
    ax.set_xticklabels(['Target', 'Distractor', 'Neutral'])
    for t, (_, _, col, _) in zip(ax.get_xticklabels(), ITEMS):
        t.set_color(col)
    ax.set_xlim(-.5, len(ITEMS) - .5)
    ax.set_ylabel('Peak response (% signal)')
    ax.text(.98, .97, 'At HP', transform=ax.transAxes, color=HP, fontsize=6.5, ha='right', va='top')
    ax.text(.98, .87, 'Elsewhere', transform=ax.transAxes, color=LP_GREY, fontsize=6.5, ha='right', va='top')
    offset_spines(ax)
    return '; '.join(tests)


def load_bar(folder):
    d = pd.concat([pd.read_csv(f, sep='\t') for f in sorted(Path(folder).glob('sub-*_2026-11.tsv'))])
    d = d[d.roi.isin(ROI_ORDER)].copy()
    d['cls'] = d.cls.replace({'adjacent': 'lp', 'opposite': 'lp'})
    d['w'] = d.data * d.n_voxels
    g = d.groupby(['subject', 'cls', 't'])
    p = (g.w.sum() / g.n_voxels.sum()).rename('data').reset_index()
    base = p[p.t < 0].groupby('subject').data.mean()
    p['data'] -= p.subject.map(base)
    return p


BAR_PEAK = (1.6, 6.4)


def bar_panels(ax, ax2, p):
    for cls, c in (('lp', LP_GREY), ('hp', HP)):
        m = p[p.cls == cls].groupby('t').data
        mu, se = m.mean(), m.sem()
        ax.fill_between(mu.index, mu - se, mu + se, color=c, alpha=.2, lw=0)
        ax.plot(mu.index, mu, color=c, lw=1.4)
    ax.axvspan(*BAR_PEAK, color='#ECEFF2', lw=0, zorder=0)
    ax.axhline(0, color='.75', lw=.6, zorder=0)
    ax.axvline(0, color='.75', lw=.6, zorder=0)
    ax.text(.98, .97, 'At HP', transform=ax.transAxes, color=HP, fontsize=6.5, ha='right', va='top')
    ax.text(.98, .87, 'Elsewhere', transform=ax.transAxes, color=LP_GREY, fontsize=6.5, ha='right', va='top')
    ax.set_xlabel('Time from bar crossing the pRF (s)')
    ax.set_ylabel('Response at the pRF (% signal)')
    offset_spines(ax)
    pk = p[p.t.between(*BAR_PEAK)].groupby(['subject', 'cls']).data.mean().unstack()
    rel = 100 * (pk.hp - pk.lp) / pk.lp
    rng = np.random.default_rng(1)
    ax2.scatter(rng.uniform(-.15, .15, len(rel)), rel, s=8, color=HP, alpha=.4, lw=0)
    m, h = rel.mean(), stats.t.ppf(.975, len(rel) - 1) * rel.std(ddof=1) / np.sqrt(len(rel))
    ax2.errorbar(.32, m, yerr=h, fmt='o', ms=5.5, color=HP, mec='w', mew=.7, lw=1.4, capsize=0)
    ax2.axhline(0, color='.6', lw=.7, ls='--', zorder=0)
    ax2.set_xlim(-.5, .7)
    ax2.set_xticks([])
    ax2.set_ylabel('At HP vs elsewhere (%)')
    offset_spines(ax2)
    ax2.spines['bottom'].set_visible(False)
    w = stats.wilcoxon(rel)
    return rel, w.pvalue


def main(folder, name='fig_event_responses_2x3', bar_folder='notes/data/bar_pass_average_2026-11', deconv=None):
    d = load_deconv(deconv) if deconv else load(folder)
    if deconv:
        name = f'{name}_deconv'
    n = d.subject.nunique()
    fig = plt.figure(figsize=(DOUBLE, 5.0))
    gs = fig.add_gridspec(1, 5, width_ratios=[.85, 1, 1, 1, 1.15], left=.005, right=.99, top=.86, bottom=.6, wspace=.42)
    ax = fig.add_subplot(gs[0])
    schematic(ax)
    letter(ax, 'a', x=.08, y=1.1)
    axes = []
    for j, (item, lab, col, marks) in enumerate(ITEMS):
        ax = fig.add_subplot(gs[j + 1], sharey=axes[0] if axes else None)
        timecourse(ax, d, item, col)
        title_icon(ax, lab, marks, col, y=1.1, h=.32)
        letter(ax, 'bcd'[j], x=-.3 if j == 0 else -.15, y=1.18)
        ax.set_xlabel('Time from onset (s)')
        if j == 0:
            ax.set_ylabel('Response at the pRF (% signal)')
            ax.text(.98, .97, 'At HP', transform=ax.transAxes, color=HP, fontsize=6.5, ha='right', va='top')
            ax.text(.98, .87, 'Elsewhere', transform=ax.transAxes, color=LP_GREY, fontsize=6.5, ha='right', va='top')
        else:
            plt.setp(ax.get_yticklabels(), visible=False)
        axes.append(ax)
    for ax in axes:
        search_strip(ax)
    ax = fig.add_subplot(gs[4])
    tests = peaks(ax, d)
    ax.text(0, 1.2, 'Peak response', transform=ax.transAxes, fontsize=7, color=FG, va='center')
    letter(ax, 'e', x=-.33, y=1.18)
    bar = ''
    if Path(bar_folder).exists():
        g2 = fig.add_gridspec(1, 3, width_ratios=[.85, 2.2, .7], left=.005, right=.75, top=.4, bottom=.1, wspace=.45)
        ax = fig.add_subplot(g2[1])
        ax2 = fig.add_subplot(g2[2])
        rel, pv = bar_panels(ax, ax2, load_bar(bar_folder))
        title_icon(ax, 'The bar passing the pRF: slightly less response at HP', ('hp', 'lp'), HP, y=1.08, h=.32)
        letter(ax, 'f', x=-.13, y=1.15)
        letter(ax2, 'g', x=-.55, y=1.15)
        bar = (f' f, Response locked to the moment the bar centre crosses the pRF centre (along the direction of the bar motion), for the '
               f'same voxels, when their location is the HP location (red) or an LP location (dark); search items not '
               f'removed; baseline −6.4 to 0 s. g, Response at HP relative to elsewhere in {BAR_PEAK[0]:.1f}-{BAR_PEAK[1]:.1f} s, '
               f'per subject (dots), mean and 95% CI: {rel.mean():+.1f}%, {(rel < 0).mean() * 100:.0f}% of subjects below 0, '
               f'Wilcoxon p = {pv:.2g}.')
        print(bar)
    save(fig, name, LEGEND.format(n=n, p0=PEAK[0], p1=PEAK[1], tests=tests, bar=bar))
    print(tests)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--folder', default='notes/data/event_average_2026-11')
    ap.add_argument('--deconv', default=None, help='Folder of nideconv responses; used instead of the event averages.')
    a = ap.parse_args()
    main(a.folder, deconv=a.deconv)
