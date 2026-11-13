"""Figure 1, version 5: the content of version 4 in a shorter figure (180 mm wide, about 5 in high).

Row 1: a, the display; b, the trial timeline with the bar sweep under it, then the sweep directions and the HP schedule
side by side. Row 2, one height for all: c, response time by distractor location with its within-subject differences
beside it; d, the same for the target; e, intertrial priming (within-subject differences, the same ms scale as the
strips: e's height equals a strip's width), its glyph key above it.

    python -m retsupp.visualize.paper.fig1_paradigm_behaviour_v5 --exclude-subjects 1 2 --suffix _n28_palA
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from retsupp.visualize.paper.fig1_paradigm_behaviour import APERTURE, DIFF_LIM, cond_icons, rt_panel
from retsupp.visualize.paper.fig1_paradigm_behaviour_v4 import (N_BOOT, diff_strip, load, panel_priming, wide)
from retsupp.visualize.paper.paper_style import (DISTRACTOR, DOUBLE, FG, HP, LP, MUTED, NEUTRAL, TARGET, letter, save)
from retsupp.visualize.paper.plot_fig1 import RING, item_positions, panel_display


def timeline_compact(fig, ax):
    """b: two trials to scale with the bar position every 2 s under them; below, the four sweep directions and the HP
    location over blocks of runs side by side."""
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    bb = ax.get_position()
    aspect = bb.width * fig.get_figwidth() / (bb.height * fig.get_figheight())
    seq = [('Fixation', 1.), ('Search', 1.5), ('Feedback', 1.), ('Interval', 2.9), ('Fixation', 1.), ('Search', 1.5),
           ('Feedback', 1.), ('', 1.)]
    total = sum(d for _, d in seq)
    x0, scale = .0, 1. / total
    y, h = .80, .11
    onset = 0.
    for k, (name, dur) in enumerate(seq):
        col, tc = (FG, 'w') if name == 'Search' else ('#ECEFF2', FG) if name in ('Interval', '') else ('#D5DAE0', FG)
        ax.add_patch(plt.Rectangle((x0 + onset * scale, y), dur * scale - .003, h, facecolor=col, lw=0))
        if name:
            ax.text(x0 + (onset + dur / 2) * scale, y + h / 2, name, ha='center', va='center', fontsize=6, color=tc)
        lab = f'{dur:g} s' if k < 3 else '1.3, 2.9 or 4.5 s' if k == 3 else None
        if lab:
            ax.text(x0 + (onset + dur / 2) * scale, y + h + .02, lab, ha='center', va='bottom', fontsize=6, color=FG)
        onset += dur
    # The bar position every 2 s (it sweeps at .168°/s, independently of the trials).
    w = .05
    icon = w * aspect
    yb = y - icon - .03
    for t in np.arange(1., total, 2.):
        ins = ax.inset_axes([x0 + t * scale - w / 2, yb, w, icon])
        ins.add_patch(Circle((0, 0), APERTURE, facecolor='#808080', lw=0))
        bx = -2.4 + .168 * t
        for i in range(8):
            r = plt.Rectangle((bx - .4, -APERTURE + i * 2 * APERTURE / 8), .8, 2 * APERTURE / 8,
                              facecolor='k' if i % 2 else 'w', lw=0)
            ins.add_patch(r)
            r.set_clip_path(Circle((0, 0), APERTURE, transform=ins.transData))
        ins.set_xlim(-3.3, 3.3)
        ins.set_ylim(-3.3, 3.3)
        ins.set_aspect('equal')
        ins.set_axis_off()
    ax.text(0, yb - .03, 'The bar sweeps throughout, independent of the trials', ha='left', va='top', fontsize=6.5,
            color=FG)
    # Lower band: sweep directions (left) and the HP schedule (right), each a title over four icons.
    ws = .052
    ics = ws * aspect
    ys = .0
    yt = ys + ics + .045
    ax.text(0, yt, 'Four sweep directions, 42 s each, twice per run', ha='left', va='bottom', fontsize=6.5, color=FG)
    for i, (ori, sgn) in enumerate(((0, 1), (0, -1), (90, -1), (90, 1))):
        ins = ax.inset_axes([.02 + i * .095, ys, ws, ics])
        ins.add_patch(Circle((0, 0), APERTURE, facecolor='#808080', lw=0))
        bar = plt.Rectangle((-1.9 * sgn - .4, -APERTURE), .8, 2 * APERTURE, facecolor='w', lw=0) if ori == 0 else \
            plt.Rectangle((-APERTURE, -1.9 * sgn - .4), 2 * APERTURE, .8, facecolor='w', lw=0)
        ins.add_patch(bar)
        bar.set_clip_path(Circle((0, 0), APERTURE, transform=ins.transData))
        d = (sgn * 2.6, 0) if ori == 0 else (0, sgn * 2.6)
        st = (-sgn * 1.3, 0) if ori == 0 else (0, -sgn * 1.3)
        ins.annotate('', xy=(st[0] + d[0], st[1] + d[1]), xytext=st,
                     arrowprops=dict(arrowstyle='-|>', color=FG, lw=.8, mutation_scale=6, shrinkA=0, shrinkB=0))
        ins.set_xlim(-3.3, 3.3)
        ins.set_ylim(-3.3, 3.3)
        ins.set_aspect('equal')
        ins.set_axis_off()
    xh, dx = .55, .118
    ax.text(xh, yt, 'HP location moves every 3 runs (example)', ha='left', va='bottom', fontsize=6.5, color=FG)
    pos = item_positions()
    w = .062
    icon = w * aspect
    for i, k in enumerate([1, 5, 7, 3]):
        x = xh + i * dx
        ins = ax.inset_axes([x, ys + (ics - icon) / 2, w, icon])
        ins.add_patch(Circle((0, 0), RING, fill=False, ec=MUTED, lw=.5))
        for j in (1, 3, 5, 7):
            ins.add_patch(Circle(pos[j], 1.3, facecolor=HP if j == k else 'w', ec=HP if j == k else MUTED, lw=.5))
        ins.set_xlim(-5.6, 5.6)
        ins.set_ylim(-5.6, 5.6)
        ins.set_aspect('equal')
        ins.set_axis_off()
        if i < 3:
            ax.annotate('', xy=(x + dx - .01, ys + ics / 2), xytext=(x + w + .01, ys + ics / 2),
                        arrowprops=dict(arrowstyle='-|>', color=MUTED, lw=.7, mutation_scale=5, shrinkA=0, shrinkB=0))


def main(rt_table, bids, trial_table, exclude, suffix):
    d, rep = load(rt_table, bids, trial_table, exclude)
    n = d.subject.nunique()
    assert rep.subject.nunique() == n
    W, H = DOUBLE, 4.62
    fig = plt.figure(figsize=(W, H))
    at = lambda x, y, w, h: fig.add_axes([x / W, y / H, w / W, h / H])   # noqa: E731  (inches)

    # Row 1
    ax = at(.02, 2.62, 2.1, 1.96)
    panel_display(ax, margin=7.4)
    letter(ax, 'a', x=.04, y=.95)
    ax = at(2.45, 2.72, 4.6, 1.5)
    timeline_compact(fig, ax)
    letter(ax, 'b', x=-.02, y=1.08)

    # Row 2: raw RT panels and the e strips share top and bottom (Y0 .. Y0 + PH); strips are SW wide, and e's height
    # covers the same ms span as a strip's width.
    Y0, PH, SW = .92, 1.0, 1.0
    ylim = (770, 1020)
    ax_c = at(.5, Y0, 1.3, PH)
    rt_panel(ax_c, d[d.contrast == 'distractor'], ['HP', 'LP', 'Absent'], [HP, LP, NEUTRAL],
             [('HP', 'LP'), ('LP', 'Absent')], ylim)
    cond_icons(ax_c, [[(7, 'distractor', True)], [(3, 'distractor', True)], []], y0=-.44, h=.34)
    ax_c.set_xlabel('Distractor', labelpad=1)
    ax_c.set_ylabel('Response time (ms)')
    ax_c.tick_params(axis='x', pad=37)
    ax_d = at(3.3, Y0, 1.05, PH)
    rt_panel(ax_d, d[d.contrast == 'target'], ['HP', 'LP'], [HP, LP], [('HP', 'LP')], ylim)
    cond_icons(ax_d, [[(7, 'target', True)], [(3, 'target', True)]], y0=-.44, h=.34)
    ax_d.set_xlabel('Target', labelpad=1)
    ax_d.tick_params(axis='x', pad=37)
    ax_d.set_yticklabels([])
    for a in (ax_c, ax_d):
        a.set_ylim(*ylim)
        bb = a.get_window_extent()
        assert bb.height <= bb.width * 1.02, (a.get_xlabel(), bb.width, bb.height)
    letter(ax_c, 'c', x=-.3, y=1.42)
    # key for the triangles (subjects beyond the axis limits), left of c's brackets
    ky_ = (Y0 + PH + .42) / H
    for xk_, mk_ in ((.55, '^'), (.62, 'v')):
        fig.lines.append(plt.Line2D([xk_ / W], [ky_], marker=mk_, ms=3.2, ls='', color=MUTED, mew=0, transform=fig.transFigure))
    fig.text(.69 / W, ky_, 'Off scale', fontsize=6, color=FG, va='center')
    letter(ax_d, 'd', x=-.12, y=1.42)

    # Within-subject strips beside c (two) and d (one), on DIFF_LIM.
    D_, T_ = wide(d[d.contrast == 'distractor']), wide(d[d.contrast == 'target'])
    sh = .2
    out = []
    for x0, specs, pan in ((1.95, [('HP − LP', D_.HP - D_.LP, HP), ('LP − absent', D_.LP - D_.Absent, LP)], 'c'),
                           (4.5, [('HP − LP', T_.HP - T_.LP, HP)], 'd')):
        for k, (lab, v, col) in enumerate(specs):
            ys = Y0 + PH - .38 - k * .55
            s_ax = at(x0, ys, SW, sh)
            r = diff_strip(s_ax, v, col, lab, ticklabels=True, seed=k)
            r['panel'] = pan
            out.append(r)
            s_ax.tick_params(axis='x', labelsize=6, pad=1)
            if k == len(specs) - 1:
                s_ax.set_xlabel('Difference (ms)', fontsize=6.5, labelpad=1)
        fig.text((x0 + SW / 2) / W, (Y0 + PH + .1) / H, 'Within subject', fontsize=6.5, color='.4', ha='center',
                 va='bottom')

    # e: priming, same ms scale as the strips (its height = SW for the DIFF_LIM span)
    RT_, RD_ = wide(rep[rep.contrast == 'target']), wide(rep[rep.contrast == 'distractor'])
    ex0 = 6.12
    ax_e = at(ex0, Y0 + PH - SW, .93, SW)
    res_e = panel_priming(ax_e, [('Target\nlocation', RT_.Repeat - RT_.Switch, TARGET, 'target'),
                                 ('Distractor\nlocation', RD_.Repeat - RD_.Switch, DISTRACTOR, 'distractor')])
    for r in res_e:
        r['panel'] = 'e'
    out += res_e
    letter(ax_e, 'e', x=-.55, y=(PH + .42) / SW)
    # glyph key above e, one line
    ky = Y0 + PH + .36
    for k_, (filled, txt) in enumerate(((True, 'This trial'), (False, 'Previous trial'))):
        xk = ex0 - .25 + k_ * .6
        fig.lines.append(plt.Line2D([xk / W], [ky / H], marker='D', ms=3.4 if filled else 5.5, ls='',
                                    mfc=TARGET if filled else 'none', mec=TARGET, mew=.8, transform=fig.transFigure))
        fig.text((xk + .07) / W, ky / H, txt, fontsize=6, color=FG, va='center')

    # report aligned edges (inches)
    fig.canvas.draw()
    for nm, a in (('c', ax_c), ('d', ax_d), ('e', ax_e)):
        bb = a.get_window_extent()
        print(nm, 'box top/bottom', round(bb.y1 / fig.dpi, 3), round(bb.y0 / fig.dpi, 3), 'w', round(bb.width / fig.dpi, 2))

    def fmt(r):
        return (f"{r['median']:+.1f} ms [{r['lo']:+.1f}, {r['hi']:+.1f}], {r['n_side']}/{r['n']}, "
                f"p = {r['p']:.2g}").replace('-', '−')
    vals = {(r['panel'], r['label']): r for r in out}
    pilots = (f' (the two pilot participants, sub-0{exclude[0]} and sub-0{exclude[1]}, with a faster bar and a different '
              'HP schedule, are left out)') if len(exclude) == 2 else ''
    legend = f"""**Figure 1. Paradigm and behaviour.** a, Search display with the bar in its 3.17° aperture: 8 items on a ring at 4°, with target and distractor at the four diagonal positions; percentages are over all trials. The target has the odd orientation (participants report the dot on it) and never appears at the distractor's location; there is no distractor on 20% of trials. b, Two trials to scale, with the bar position every 2 s below; the bar sweeps the aperture in four directions (rightward, leftward, downward, upward; 42 s each with 4.8 s rest, the sequence twice per run), and the HP location across blocks of runs (example order). c, Response time with the distractor at the HP location (purple), at an LP location (mid purple), or absent (grey); right, the within-subject differences HP − LP and LP − absent. d, Response time with the target at the HP or an LP location, on trials with a distractor; right, the within-subject difference HP − LP. e, Intertrial priming as within-subject differences: response time when the target (orange) or the distractor (green) appears at the same location as on the previous trial minus when it appears at another location; distractor repeats are compared within HP and within LP trials and averaged; e is drawn on the same ms scale as the difference strips of c and d. Target repeat − switch {fmt(vals[('e', 'Target' + chr(10) + 'location')])}; distractor repeat − switch {fmt(vals[('e', 'Distractor' + chr(10) + 'location')])} (median [bootstrapped 95% CI], subjects on the side of the median, Wilcoxon signed-rank). The raw response times of these conditions are in Supplementary Fig. figS_priming_rt_v4{suffix}. Icons below each condition: the four possible locations, with HP ringed in purple where the condition is defined by it (c, d); filled, the item on this trial, hollow, on the previous trial (e: repeat minus switch display). c-e, n = {n}{pilots}, correct trials. Above the brackets in c, d and above each strip in e: the median within-subject difference and the Wilcoxon signed-rank test (** p < .01, *** p < .001). Differences beside c, d: distractor HP − LP {fmt(vals[('c', 'HP − LP')])}; LP − absent {fmt(vals[('c', 'LP − absent')])}; target HP − LP {fmt(vals[('d', 'HP − LP')])}. In c and d, dots are subjects, triangles subjects beyond the axis limits, and large points and bars the mean ± within-subject SEM (Cousineau-Morey). In the difference strips and e, dots are subjects (beyond the axis limits drawn at the edge), diamonds the median and bars its bootstrapped 95% CI ({N_BOOT} resamples)."""
    save(fig, f'fig1_paradigm_behaviour_v5{suffix}', legend)
    for r in out:
        print(r['panel'], r['label'].replace('\n', ' '), fmt(r))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--rt-table', default='notes/data/behaviour_rt_2026-11.tsv')
    ap.add_argument('--bids-folder', default='/data/ds-retsupp')
    ap.add_argument('--trial-table', default='notes/data/trialwise_af_trial_table.tsv.gz')
    ap.add_argument('--exclude-subjects', type=int, nargs='*', default=[])
    ap.add_argument('--suffix', default='')
    a = ap.parse_args()
    main(a.rt_table, a.bids_folder, a.trial_table, a.exclude_subjects, a.suffix)
