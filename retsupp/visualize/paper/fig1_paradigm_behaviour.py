"""Figure 1: the paradigm and behaviour.

``--rt-table`` holds per-subject mean RTs; it is built from the events if missing.

Usage::

    python -m retsupp.visualize.paper.fig1_paradigm_behaviour
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle
from scipy import stats

from retsupp.visualize.paper.paper_style import DISTRACTOR, DOUBLE, FG, HP, MUTED, TARGET, letter, offset_spines, save
from retsupp.visualize.paper.plot_fig1 import APERTURE, RING, item_positions, panel_display

LEGEND = """
**Figure 1. Paradigm and behaviour.** a, Search display with the bar in its 3.17° aperture: 8 items on a ring at 4°, with target and distractor at the four diagonal positions; percentages are over all trials. The target has the odd orientation (participants report the dot on it) and never appears at the distractor's location; there is no distractor on 20% of trials. b, Two trials to scale, with the bar position every 2 s below; the bar sweeps the aperture in four directions (rightward, leftward, downward, upward; 42 s each with 4.8 s rest, the sequence twice per run), and the HP location across blocks of runs (example order). c, Response time with the distractor at the HP location, at an LP location, or absent. d, Response time with the target at the HP or an LP location, on trials with a distractor. e, f, Response time when the target (e) or the distractor (f) appears at the same location as on the previous trial, or at another location; distractor repeats are compared within HP and within LP trials and averaged. Icons below each condition: the four possible locations with HP ringed in red; filled, the item on this trial, hollow, on the previous trial. c-f, n = 30, correct trials. Dots are subjects, triangles are subjects beyond the axis limits; large points and bars are the mean ± within-subject SEM (Cousineau-Morey). Wilcoxon signed-rank tests: ** p < .01, *** p < .001.
"""




def build_rt(path, bids):
    from retsupp.visualize.vss2026 import plot_rt_talk as rt
    df = rt.add_condition_labels(rt.collect_trials(bids))
    a = rt.per_subject_means(df, 'dist_cond', ['HP', 'LP', 'Absent']).rename(columns={'dist_cond': 'condition'}).assign(contrast='distractor')
    b = rt.per_subject_means(df[df.has_distractor], 'target_cond', ['HP', 'LP']).rename(columns={'target_cond': 'condition'}).assign(contrast='target')
    pd.concat([a, b]).to_csv(path, sep='\t', index=False)


def panel_timeline(fig, ax):
    """Two trials to scale, the bar position every 2 s below them, then the HP location over blocks."""
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    bb = ax.get_position()
    aspect = bb.width * fig.get_figwidth() / (bb.height * fig.get_figheight())
    seq = [('Fixation', 1.), ('Search', 1.5), ('Feedback', 1.), ('Interval', 2.9), ('Fixation', 1.), ('Search', 1.5), ('Feedback', 1.), ('', 1.)]
    total = sum(d for _, d in seq)
    x0, scale = .01, .98 / total
    y, h = .78, .13
    onset = 0.
    for k, (name, dur) in enumerate(seq):
        col, tc = (FG, 'w') if name == 'Search' else ('#ECEFF2', FG) if name in ('Interval', '') else ('#D5DAE0', FG)
        ax.add_patch(plt.Rectangle((x0 + onset * scale, y), dur * scale - .003, h, facecolor=col, lw=0))
        if name:
            ax.text(x0 + (onset + dur / 2) * scale, y + h / 2, name, ha='center', va='center', fontsize=5.8, color=tc)
        if k < 3:
            ax.text(x0 + (onset + dur / 2) * scale, y + h + .015, f'{dur:g} s', ha='center', va='bottom', fontsize=6, color=FG)
        if k == 3:
            ax.text(x0 + (onset + dur / 2) * scale, y + h + .015, '1.3, 2.9 or 4.5 s', ha='center', va='bottom', fontsize=6, color=FG)
        onset += dur
    # The bar sweeps at .168°/s, independently of the trials.
    w = .058
    icon = w * aspect
    yb = y - icon - .03
    for t in np.arange(1., total, 2.):
        inset = ax.inset_axes([x0 + t * scale - w / 2, yb, w, icon])
        inset.add_patch(Circle((0, 0), APERTURE, facecolor='#808080', lw=0))
        bx = -2.4 + .168 * t
        for i in range(8):
            r = plt.Rectangle((bx - .4, -APERTURE + i * 2 * APERTURE / 8), .8, 2 * APERTURE / 8, facecolor='k' if i % 2 else 'w', lw=0)
            inset.add_patch(r)
            r.set_clip_path(Circle((0, 0), APERTURE, transform=inset.transData))
        inset.set_xlim(-3.3, 3.3)
        inset.set_ylim(-3.3, 3.3)
        inset.set_aspect('equal')
        inset.set_axis_off()
    ax.text(x0, yb - .012, 'Bar sweeps throughout, independent of the trials', ha='left', va='top', fontsize=6.5, color=FG)
    # The four sweep directions (42 s each, 4.8 s rest between, the sequence twice per run).
    ws = .055
    ics = ws * aspect
    ys = yb - .012 - ics - .06
    ax.text(x0, ys + ics / 2, 'Four sweep directions, 42 s each, twice per run', ha='left', va='center', fontsize=6.5, color=FG)
    for i, (ori, sgn) in enumerate(((0, 1), (0, -1), (90, -1), (90, 1))):   # rightward, leftward, downward, upward
        inset = ax.inset_axes([.6 + i * .08, ys, ws, ics])
        inset.add_patch(Circle((0, 0), APERTURE, facecolor='#808080', lw=0))
        bar = plt.Rectangle((-1.9 * sgn - .4, -APERTURE), .8, 2 * APERTURE, facecolor='w', lw=0) if ori == 0 else \
            plt.Rectangle((-APERTURE, -1.9 * sgn - .4), 2 * APERTURE, .8, facecolor='w', lw=0)
        inset.add_patch(bar)
        bar.set_clip_path(Circle((0, 0), APERTURE, transform=inset.transData))
        d = (sgn * 2.6, 0) if ori == 0 else (0, sgn * 2.6)
        st = (-sgn * 1.3, 0) if ori == 0 else (0, -sgn * 1.3)
        inset.annotate('', xy=(st[0] + d[0], st[1] + d[1]), xytext=st,
                       arrowprops=dict(arrowstyle='-|>', color=FG, lw=.8, mutation_scale=6, shrinkA=0, shrinkB=0))
        inset.set_xlim(-3.3, 3.3)
        inset.set_ylim(-3.3, 3.3)
        inset.set_aspect('equal')
        inset.set_axis_off()
    # The HP location per block of runs.
    w = .07
    icon = w * aspect
    yh = max(0., ys - .14 - icon)   # right under the sweep row, not at the bottom of the panel
    ax.text(x0, yh + icon / 2, 'HP location moves\nevery 3 runs (example)', ha='left', va='center', fontsize=6.5, color=FG)
    pos = item_positions()
    for i, k in enumerate([1, 5, 7, 3]):
        x = .32 + i * .17
        inset = ax.inset_axes([x, yh, w, icon])
        inset.add_patch(Circle((0, 0), RING, fill=False, ec=MUTED, lw=.5))
        for j in (1, 3, 5, 7):
            inset.add_patch(Circle(pos[j], 1.3, facecolor=HP if j == k else 'w', ec=HP if j == k else MUTED, lw=.5))
        inset.set_xlim(-5.6, 5.6)
        inset.set_ylim(-5.6, 5.6)
        inset.set_aspect('equal')
        inset.set_axis_off()
        if i < 3:
            ax.annotate('', xy=(x + w + .08, yh + icon / 2), xytext=(x + w + .02, yh + icon / 2),
                        arrowprops=dict(arrowstyle='-|>', color=MUTED, lw=.7, mutation_scale=5, shrinkA=0, shrinkB=0))


def stars(p):
    return '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else 'n.s.'


def rt_panel(ax, d, conds, colors, pairs, ylim):
    """Actual subject RTs on an axis capped around the means; subjects beyond it are triangles on
    the cap, so none is dropped. Error bars are within-subject SEM (Cousineau-Morey)."""
    w = d.pivot(index='subject', columns='condition', values='rt')[conds]
    k = len(conds)
    m = w.mean()
    se = (w.sub(w.mean(1), axis=0) + w.values.mean()).sem() * np.sqrt(k / (k - 1))
    lo, hi = ylim
    rng = np.random.default_rng(3)
    for i, c in enumerate(conds):
        xx = i + rng.uniform(-.14, .14, len(w))
        v = w[c].to_numpy()
        ok, over, under = (v >= lo) & (v <= hi), v > hi, v < lo
        ax.scatter(xx[ok], v[ok], s=9, color=colors[i], alpha=.3, lw=0, zorder=1)
        ax.scatter(xx[over], np.full(over.sum(), hi), s=10, marker='^', color=colors[i], alpha=.55, lw=0, zorder=1, clip_on=False)
        ax.scatter(xx[under], np.full(under.sum(), lo), s=10, marker='v', color=colors[i], alpha=.55, lw=0, zorder=1, clip_on=False)
        ax.errorbar(i, m[c], yerr=se[c], color=colors[i], marker='o', ms=6, mfc=colors[i], mec='w', mew=.7, lw=1.4, capsize=0, zorder=4)
    ax.plot(range(k), m.values, color=FG, lw=.8, zorder=3)
    span = hi - lo
    for n, (a, b) in enumerate(pairs):
        diff = w[a] - w[b]
        i, j = conds.index(a), conds.index(b)
        y = hi + (.2 + n * .13) * span
        ax.plot([i, j], [y, y], color=FG, lw=.7, clip_on=False)
        ax.text((i + j) / 2, y + .01 * span, f'{diff.mean():+.0f} ms {stars(stats.wilcoxon(diff).pvalue)}'.replace('-', '−'),
                ha='center', va='bottom', fontsize=6.5, color=FG, clip_on=False)
    ax.set_xticks(range(k))
    ax.set_xticklabels(conds)
    ax.set_xlim(-.45, k - .55)
    ax.set_ylim(lo, hi)
    ax.set_yticks([800, 900, 1000])
    offset_spines(ax)


def repeats_raw(trial_table):
    """Per subject, mean RT (ms) on trials where the target (distractor) location repeats or switches from the previous trial.

    Distractor repeats are compared within HP and within LP trials and averaged over the two (HP trials repeat more
    often and are faster anyway); see ``plot_trialwise_story.behaviour`` for the same contrast as a difference.
    """
    tt = pd.read_csv(trial_table, sep='\t')
    v = tt[tt['roi'] == 'V1']
    rows = []
    for subj, d in v.groupby('subject'):
        t = d.groupby(d['target_repeat'] > 0)['rt'].mean() * 1000
        ds = d[d['trial_type'] != 'none']
        dd = ds.groupby(['trial_type', ds['dist_repeat'] > 0])['rt'].mean().unstack().mean() * 1000
        rows += [dict(subject=subj, contrast='target', condition='Repeat', rt=t[True]),
                 dict(subject=subj, contrast='target', condition='Switch', rt=t[False]),
                 dict(subject=subj, contrast='distractor', condition='Repeat', rt=dd[True]),
                 dict(subject=subj, contrast='distractor', condition='Switch', rt=dd[False])]
    return pd.DataFrame(rows)


def cond_icons(ax, specs, y0=-.66, h=.5):
    """Under each condition, a small display: the four diagonal locations, the HP one ringed in red, and the item(s).

    specs: one list per x position of (location index, kind, filled) with kind 'target' or 'distractor'; a hollow
    marker is the previous trial's item (repeat / switch panels).
    """
    pos = item_positions()
    tf = ax.get_xaxis_transform()
    for x, items in enumerate(specs):
        ins = ax.inset_axes([x - .42, y0, .84, h], transform=tf)
        ins.add_patch(Circle((0, 0), RING, fill=False, ec=MUTED, lw=.5))
        for j in (1, 3, 5, 7):
            ins.add_patch(Circle(pos[j], .95, facecolor='w', ec=MUTED, lw=.4))
        ins.add_patch(Circle(pos[7], 1.45, fill=False, ec=HP, lw=.8, ls=(0, (2, 1.2))))
        for j, kind, filled in items:
            col, m = (TARGET, 'D') if kind == 'target' else (DISTRACTOR, 'o')
            ins.plot(*pos[j], m, ms=3.4 if filled else 6.4, mfc=col if filled else 'none', mec=col, mew=.8, zorder=5 if filled else 6)
        ins.plot(0, 0, '+', color=FG, ms=2.5, mew=.5)
        ins.set_xlim(-6, 6)
        ins.set_ylim(-6, 6)
        ins.set_aspect('equal')
        ins.set_axis_off()
    ax.tick_params(axis='x', pad=30, length=0)


def main(rt_table, bids, name='fig1_paradigm_behaviour', trial_table='notes/data/trialwise_af_trial_table.tsv.gz'):
    if not Path(rt_table).exists():
        build_rt(rt_table, bids)
    d = pd.read_csv(rt_table, sep='\t')
    rep = repeats_raw(trial_table)
    # Display and timeline on top; the four behaviour panels across the full width below, with room between.
    fig = plt.figure(figsize=(DOUBLE, 4.0))
    top = fig.add_gridspec(1, 2, width_ratios=[1, 1.62], left=.005, right=.995, top=.985, bottom=.535, wspace=.12)
    ax = fig.add_subplot(top[0])
    panel_display(ax, margin=7.4)
    letter(ax, 'a', x=.05, y=.97)
    ax = fig.add_subplot(top[1])
    panel_timeline(fig, ax)
    letter(ax, 'b', x=0, y=.95)
    low = fig.add_gridspec(1, 4, width_ratios=[3, 2, 2, 2], left=.08, right=.985, top=.36, bottom=.185, wspace=.32)
    ylim = (770, 1020)
    ax = fig.add_subplot(low[0])
    rt_panel(ax, d[d.contrast == 'distractor'], ['HP', 'LP', 'Absent'], [HP, FG, MUTED], [('HP', 'LP'), ('LP', 'Absent')], ylim)
    cond_icons(ax, [[(7, 'distractor', True)], [(3, 'distractor', True)], []])
    ax.set_xlabel('Distractor')
    ax.set_ylabel('Response time (ms)')
    letter(ax, 'c', x=-.2, y=1.55)
    ax.scatter([.0], [1.5], s=10, marker='^', color=MUTED, lw=0, transform=ax.transAxes, clip_on=False)
    ax.scatter([.05], [1.5], s=10, marker='v', color=MUTED, lw=0, transform=ax.transAxes, clip_on=False)
    ax.text(.09, 1.5, 'Off scale', transform=ax.transAxes, fontsize=6, color=FG, va='center')
    ax = fig.add_subplot(low[1])
    rt_panel(ax, d[d.contrast == 'target'], ['HP', 'LP'], [HP, FG], [('HP', 'LP')], ylim)
    cond_icons(ax, [[(7, 'target', True)], [(3, 'target', True)]])
    ax.set_xlabel('Target')
    ax.set_yticklabels([])
    letter(ax, 'd', x=-.15, y=1.55)
    for k, (contrast, col, lab, let) in enumerate((('target', TARGET, 'Target location', 'e'),
                                                   ('distractor', DISTRACTOR, 'Distractor location', 'f'))):
        ax = fig.add_subplot(low[2 + k])
        rt_panel(ax, rep[rep.contrast == contrast], ['Repeat', 'Switch'], [col, FG], [('Repeat', 'Switch')], ylim)
        cond_icons(ax, [[(3, contrast, False), (3, contrast, True)], [(1, contrast, False), (3, contrast, True)]])
        ax.set_xlabel(lab)
        ax.set_yticklabels([])
        letter(ax, let, x=-.15, y=1.55)
        if k == 0:
            # Key for the icons below: this trial's item filled, the previous trial's hollow.
            for n, (filled, txt) in enumerate(((True, 'This trial'), (False, 'Previous trial'))):
                xk = .1 + n * .45
                ax.plot(xk, 1.55, 'D', ms=3.4 if filled else 5.5, mfc=TARGET if filled else 'none', mec=TARGET, mew=.8,
                        transform=ax.transAxes, clip_on=False)
                ax.text(xk + .07, 1.55, txt, transform=ax.transAxes, fontsize=6, color=FG, va='center')
    for a in fig.axes[-4:]:
        bb = a.get_window_extent()
        assert bb.height <= bb.width * 1.02, (a.get_xlabel(), bb.width, bb.height)   # subject-dot panels never taller than wide
    save(fig, name, LEGEND)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--rt-table', default='notes/data/behaviour_rt_2026-11.tsv')
    ap.add_argument('--bids-folder', default='/data/ds-retsupp')
    a = ap.parse_args()
    main(a.rt_table, a.bids_folder)
