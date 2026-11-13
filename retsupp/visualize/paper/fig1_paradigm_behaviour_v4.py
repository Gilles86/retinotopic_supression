"""Figure 1, version 4: the paradigm and behaviour, with the intertrial priming as one compact panel of differences.

a, b and c, d as in version 3 (``fig1_paradigm_behaviour.py --strips``: raw RTs with the within-subject difference strips
below). e replaces the old e and f: target repeat − switch and distractor repeat − switch as two strips of within-subject
differences side by side on one shared ms axis, drawn to the same ms scale as the strips under c and d. The raw RTs of the
old e and f go to a supplementary figure (``figS_priming_rt_v4``).

``--exclude-subjects 1 2`` leaves out the two pilot participants (the paper's n = 28).

Usage::

    python -m retsupp.visualize.paper.fig1_paradigm_behaviour_v4 --exclude-subjects 1 2 --suffix _n28
    python -m retsupp.visualize.paper.fig1_paradigm_behaviour_v4
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle
from scipy import stats

from retsupp.visualize.paper.fig1_paradigm_behaviour import (DIFF_LIM, LEGEND as LEGEND_V3, build_rt, cond_icons,
                                                             panel_timeline, repeats_raw, rt_panel, stars)
from retsupp.visualize.paper.paper_style import (DISTRACTOR, DOUBLE, FG, HP, LP, MUTED, NEUTRAL, SINGLE, TARGET, letter, offset_spines,
                                                 save)
from retsupp.visualize.paper.plot_fig1 import RING, item_positions, panel_display

N_BOOT = 10000


def median_ci(v, n_boot=N_BOOT, seed=0):
    """Median and its percentile bootstrap 95% CI over subjects."""
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    boots = np.median(np.random.default_rng(seed).choice(v, (n_boot, len(v))), axis=1)
    return np.median(v), *np.percentile(boots, [2.5, 97.5])


def diff_strip(ax, diff, color, label, ticklabels=True, seed=0):
    """As version 3: one horizontal strip of within-subject differences (subject dots, median and bootstrapped 95% CI)."""
    v = np.asarray(diff, dtype=float)
    m, lo, hi = median_ci(v)
    yy = np.random.default_rng(seed).uniform(-.32, .32, len(v))
    ax.axvline(0, color='.75', lw=.6, zorder=0)
    ax.scatter(np.clip(v, *DIFF_LIM), yy, s=7, color=color, alpha=.4, lw=0, zorder=2, clip_on=False)
    ax.plot([lo, hi], [0, 0], color=FG, lw=1.4, solid_capstyle='butt', zorder=3)
    ax.plot(m, 0, 'D', ms=4.2, color=color, mec=FG, mew=.8, zorder=4)
    n_side = int((np.sign(v) == np.sign(m)).sum())
    ax.text(0, 1.18, label, transform=ax.transAxes, fontsize=6, color=FG, ha='left', va='bottom')
    ax.text(1, 1.18, f'{n_side}/{len(v)} {"faster" if m < 0 else "slower"}', transform=ax.transAxes, fontsize=6,
            color=FG, ha='right', va='bottom')
    ax.set_xlim(*DIFF_LIM)
    ax.set_ylim(-.5, .5)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_xticks([-100, 0, 100])
    if not ticklabels:
        ax.set_xticklabels([])
    ax.spines['bottom'].set_position(('outward', 2))
    return dict(label=label, median=m, lo=lo, hi=hi, n_side=n_side, n=len(v), p=stats.wilcoxon(v).pvalue)


def icon(ins, items):
    """One small display: the four diagonal locations and the items, (location, kind, filled); hollow = previous trial."""
    pos = item_positions()
    ins.add_patch(Circle((0, 0), RING, fill=False, ec=MUTED, lw=.5))
    for j in (1, 3, 5, 7):
        ins.add_patch(Circle(pos[j], .95, facecolor='w', ec=MUTED, lw=.4))
    for j, kind, filled in items:
        col, mk = (TARGET, 'D') if kind == 'target' else (DISTRACTOR, 'o')
        ins.plot(*pos[j], mk, ms=3.4 if filled else 6.4, mfc=col if filled else 'none', mec=col, mew=.8,
                 zorder=5 if filled else 6)
    ins.plot(0, 0, '+', color=FG, ms=2.5, mew=.5)
    ins.set_xlim(-6, 6)
    ins.set_ylim(-6, 6)
    ins.set_aspect('equal')
    ins.set_axis_off()


def panel_priming(ax, series):
    """e: per series (label, differences, colour, kind) one vertical strip: subject dots, the median (diamond) and its
    bootstrapped 95% CI (bar), the count of subjects on the median's side next to it, the median and the Wilcoxon stars
    above; under each strip the repeat and the switch display (filled: this trial, hollow: previous trial)."""
    lo_lim, hi_lim = DIFF_LIM
    ax.axhline(0, color='.75', lw=.6, zorder=0)
    out = []
    for x, (lab, v, col, kind) in enumerate(series):
        v = np.asarray(v, dtype=float)
        m, lo, hi = median_ci(v)
        p = stats.wilcoxon(v).pvalue
        xx = x - .13 + np.random.default_rng(x).uniform(-.1, .1, len(v))
        ax.scatter(xx, np.clip(v, lo_lim, hi_lim), s=7, color=col, alpha=.4, lw=0, zorder=2, clip_on=False)
        ax.plot([x + .05, x + .05], [lo, hi], color=FG, lw=1.4, solid_capstyle='butt', zorder=3)
        ax.plot(x + .05, m, 'D', ms=4.2, color=col, mec=FG, mew=.8, zorder=4)
        n_side = int((np.sign(v) == np.sign(m)).sum())
        ax.text(x + .15, m, f'{n_side}/{len(v)}\n{"faster" if m < 0 else "slower"}', fontsize=6, color=FG, ha='left',
                va='center', linespacing=1.0)
        ax.text(x, hi_lim + .04 * (hi_lim - lo_lim), f'{m:+.0f} ms {stars(p)}'.replace('-', '−'), fontsize=6.5,
                color=FG, ha='center', va='bottom', clip_on=False)
        # Repeat − switch displays under the strip.
        tf = ax.get_xaxis_transform()
        for k, items in enumerate(([(3, kind, False), (3, kind, True)], [(1, kind, False), (3, kind, True)])):
            icon(ax.inset_axes([x - .44 + k * .52, -.27, .36, .2], transform=tf), items)
        ax.text(x - .01, -.17, '−', transform=tf, fontsize=7, color=FG, ha='center', va='center')
        ax.text(x, -.3, lab, transform=tf, fontsize=7, color=FG, ha='center', va='top')
        out.append(dict(label=lab, median=m, lo=lo, hi=hi, n_side=n_side, n=len(v), p=p))
    ax.set_xlim(-.55, len(series) - .45)
    ax.set_ylim(lo_lim, hi_lim)
    ax.set_yticks([-100, 0, 100])
    ax.set_xticks([])
    ax.set_ylabel('Repeat − switch (ms)')
    offset_spines(ax)
    ax.spines['bottom'].set_visible(False)
    return out


def icon_key(fig, x, y, W, H):
    """Key for the displays: this trial's item filled, the previous trial's hollow (figure inches)."""
    for n, (filled, txt) in enumerate(((True, 'This trial'), (False, 'Previous trial'))):
        xk = x + n * .62
        fig.lines.append(plt.Line2D([xk / W], [y / H], marker='D', ms=3.4 if filled else 5.5, ls='',
                                    mfc=TARGET if filled else 'none', mec=TARGET, mew=.8, transform=fig.transFigure))
        fig.text((xk + .08) / W, y / H, txt, fontsize=6, color=FG, va='center')


def load(rt_table, bids, trial_table, exclude):
    if not Path(rt_table).exists():
        build_rt(rt_table, bids)
    d = pd.read_csv(rt_table, sep='\t')
    rep = repeats_raw(trial_table)
    return d[~d.subject.isin(exclude)], rep[~rep.subject.isin(exclude)]


def wide(x):
    return x.pivot(index='subject', columns='condition', values='rt')


def main(rt_table, bids, trial_table, exclude, suffix):
    d, rep = load(rt_table, bids, trial_table, exclude)
    n = d.subject.nunique()
    assert rep.subject.nunique() == n, (n, rep.subject.nunique())
    H0, EXTRA = 4.0, 1.2
    H = H0 + EXTRA
    W = DOUBLE
    fy = lambda f: (f * H0 + EXTRA) / H
    fig = plt.figure(figsize=(W, H))
    top = fig.add_gridspec(1, 2, width_ratios=[1, 1.62], left=.005, right=.995, top=fy(.985), bottom=fy(.535), wspace=.12)
    ax = fig.add_subplot(top[0])
    panel_display(ax, margin=7.4)
    letter(ax, 'a', x=.05, y=.97)
    ax = fig.add_subplot(top[1])
    panel_timeline(fig, ax)
    letter(ax, 'b', x=0, y=.95)
    # The lower row of version 3 (c, d, e, f); c and d keep their positions and size, e takes the old e and f columns.
    low = fig.add_gridspec(1, 4, width_ratios=[3, 2, 2, 2], left=.08, right=.985, top=fy(.36), bottom=fy(.185), wspace=.32)
    pos = [low[i].get_position(fig) for i in range(4)]
    ylim = (770, 1020)
    ax_c = fig.add_axes(pos[0].bounds)
    rt_panel(ax_c, d[d.contrast == 'distractor'], ['HP', 'LP', 'Absent'], [HP, LP, NEUTRAL], [('HP', 'LP'), ('LP', 'Absent')], ylim)
    cond_icons(ax_c, [[(7, 'distractor', True)], [(3, 'distractor', True)], []])
    ax_c.set_xlabel('Distractor')
    ax_c.set_ylabel('Response time (ms)')
    letter(ax_c, 'c', x=-.2, y=1.55)
    ax_c.scatter([.0], [1.5], s=10, marker='^', color=MUTED, lw=0, transform=ax_c.transAxes, clip_on=False)
    ax_c.scatter([.05], [1.5], s=10, marker='v', color=MUTED, lw=0, transform=ax_c.transAxes, clip_on=False)
    ax_c.text(.09, 1.5, 'Off scale', transform=ax_c.transAxes, fontsize=6, color=FG, va='center')
    ax_d = fig.add_axes(pos[1].bounds)
    rt_panel(ax_d, d[d.contrast == 'target'], ['HP', 'LP'], [HP, LP], [('HP', 'LP')], ylim)
    cond_icons(ax_d, [[(7, 'target', True)], [(3, 'target', True)]])
    ax_d.set_xlabel('Target')
    ax_d.set_yticklabels([])
    letter(ax_d, 'd', x=-.15, y=1.55)
    for a in (ax_c, ax_d):
        bb = a.get_window_extent()
        assert bb.height <= bb.width * 1.02, (a.get_xlabel(), bb.width, bb.height)   # subject-dot panels never taller than wide
    # Difference strips under c and d, as version 3.
    D, T = wide(d[d.contrast == 'distractor']), wide(d[d.contrast == 'target'])
    RT, RD = wide(rep[rep.contrast == 'target']), wide(rep[rep.contrast == 'distractor'])
    sw, sh = pos[1].width, .17 / H
    out = []
    for a, specs in zip((ax_c, ax_d), [[('HP − LP', D.HP - D.LP, HP), ('LP − absent', D.LP - D.Absent, LP)],
                                       [('HP − LP', T.HP - T.LP, HP)]]):
        for k, (lab, v, col) in enumerate(specs):
            s_ax = fig.add_axes([a.get_position().x0, (.82 - k * .42) / H, sw, sh])
            r = diff_strip(s_ax, v, col, lab, ticklabels=k == len(specs) - 1, seed=k)
            r['panel'] = 'c' if a is ax_c else 'd'
            out.append(r)
            if k == len(specs) - 1:
                s_ax.set_xlabel('Difference (ms)', fontsize=7, labelpad=1)
    fig.text(.008, .9 / H, 'Within\nsubject', fontsize=6.5, color='.45', ha='left', va='center')
    # e: the two priming strips, vertical, on one ms axis whose length per ms equals the strips' (same scale).
    e_h = sw * W                                    # inches for the DIFF_LIM span, as the horizontal strips
    e_w = pos[3].x1 - pos[2].x0 - .45 / W           # the old e and f columns, less room for the y label
    e_x0 = pos[2].x0 + .45 / W
    e_top = pos[0].y1 * H                           # top of the raw-RT panels (inches)
    e_y0 = e_top - e_h
    ax_e = fig.add_axes([e_x0, e_y0 / H, e_w * .86, e_h / H])
    res_e = panel_priming(ax_e, [('Target\nlocation', RT.Repeat - RT.Switch, TARGET, 'target'),
                                 ('Distractor\nlocation', RD.Repeat - RD.Switch, DISTRACTOR, 'distractor')])
    for r in res_e:
        r['panel'] = 'e'
    out += res_e
    letter(ax_e, 'e', x=-.3, y=1.0 + .55 * (pos[0].height * H) / e_h)
    icon_key(fig, e_x0 * W + .25, e_y0 - .72, W, H)
    fig.text((e_x0 * W + .25) / W, (e_y0 - .97) / H, 'Within-subject differences, on the same\nms scale as the strips under c and d',
             fontsize=6, color='.45', ha='left', va='center')

    def fmt(r):
        return (f"{r['median']:+.1f} ms [{r['lo']:+.1f}, {r['hi']:+.1f}], {r['n_side']}/{r['n']}, "
                f"p = {r['p']:.2g}").replace('-', '−')
    vals = {(r['panel'], r['label']): r for r in out}
    TKEY, DKEY = ('e', 'Target\nlocation'), ('e', 'Distractor\nlocation')
    pilots = (f' (the two pilot participants, sub-0{exclude[0]} and sub-0{exclude[1]}, with a faster bar and a different HP '
              'schedule, are left out)') if len(exclude) == 2 else ''
    legend = LEGEND_V3.replace('c-f, n = 30, correct trials.', f'c-e, n = {n}{pilots}, correct trials.')
    legend = legend.replace(
        'e, f, Response time when the target (e) or the distractor (f) appears at the same location as on the previous trial, or at another location; distractor repeats are compared within HP and within LP trials and averaged.',
        'e, Intertrial priming as within-subject differences: response time when the target (orange) or the distractor '
        '(blue) appears at the same location as on the previous trial minus when it appears at another location; '
        'distractor repeats are compared within HP and within LP trials and averaged. Target repeat − switch '
        f"{fmt(vals[TKEY])}; distractor repeat − switch {fmt(vals[DKEY])} "
        '(median [bootstrapped 95% CI], subjects on the side of the median, Wilcoxon signed-rank). The raw response times '
        f'of these conditions are in Supplementary Fig. figS_priming_rt_v4{suffix}.')
    legend = legend.replace('filled, the item on this trial, hollow, on the previous trial (e, f)',
                            'filled, the item on this trial, hollow, on the previous trial (e: repeat minus switch display)')
    legend = legend.replace(
        'Dots are subjects, triangles are subjects beyond the axis limits; large points and bars are the mean ± within-subject SEM (Cousineau-Morey). Wilcoxon signed-rank tests: ** p < .01, *** p < .001.',
        'Above the brackets in c, d and above each strip in e: the median within-subject difference and the Wilcoxon '
        'signed-rank test (** p < .01, *** p < .001); lower rows (c, d) and e: the within-subject differences, every strip and e on the same '
        'ms scale. c, d, lower rows: distractor HP − LP ' + fmt(vals[('c', 'HP − LP')]) + '; LP − absent '
        + fmt(vals[('c', 'LP − absent')]) + '; target HP − LP ' + fmt(vals[('d', 'HP − LP')]) + '. '
        'In c and d (upper rows), dots are subjects, triangles subjects beyond the axis limits, and large points and bars '
        'the mean ± within-subject SEM (Cousineau-Morey). In the difference strips and e, dots are subjects (beyond the axis '
        f'limits drawn at the edge), diamonds the median and bars its bootstrapped 95% CI ({N_BOOT} resamples).')
    save(fig, f'fig1_paradigm_behaviour_v4{suffix}', legend)
    for r in out:
        print(r['panel'], r['label'].replace('\n', ' '), fmt(r))

    # Supplementary figure: the raw RTs of the priming conditions (the old e and f).
    fs = plt.figure(figsize=(SINGLE * 1.25, 2.45))
    gs = fs.add_gridspec(1, 2, left=.16, right=.98, top=.70, bottom=.415, wspace=.3)
    for k, (contrast, col, lab, let) in enumerate((('target', TARGET, 'Target location', 'a'),
                                                   ('distractor', DISTRACTOR, 'Distractor location', 'b'))):
        a = fs.add_subplot(gs[k])
        rt_panel(a, rep[rep.contrast == contrast], ['Repeat', 'Switch'], [col, FG], [('Repeat', 'Switch')], ylim)
        cond_icons(a, [[(3, contrast, False), (3, contrast, True)], [(1, contrast, False), (3, contrast, True)]], hp=False)
        a.set_xlabel(lab)
        if k:
            a.set_yticklabels([])
        else:
            a.set_ylabel('Response time (ms)')
        letter(a, let, x=-.15, y=1.55)
        bb = a.get_window_extent()
        assert bb.height <= bb.width * 1.02
    for m_, (filled, txt) in enumerate(((True, 'This trial'), (False, 'Previous trial'))):
        xk = .3 + m_ * .22
        fs.lines.append(plt.Line2D([xk], [.965], marker='D', ms=3.4 if filled else 5.5, ls='', mfc=TARGET if filled else 'none',
                                   mec=TARGET, mew=.8, transform=fs.transFigure))
        fs.text(xk + .025, .965, txt, fontsize=6, color=FG, va='center')
    for xk_, mk_ in ((.78, '^'), (.805, 'v')):
        fs.lines.append(plt.Line2D([xk_], [.965], marker=mk_, ms=3.2, ls='', color=MUTED, mew=0, transform=fs.transFigure))
    fs.text(.825, .965, 'Off scale', fontsize=6, color=FG, va='center')
    legend_s = (f'**Supplementary Figure figS_priming_rt_v4{suffix}. Response times behind the intertrial priming of Fig. 1e.** '
                'Response time when the target (a) or the distractor (b) appears at the same location as on the previous trial '
                '(repeat) or at another location (switch); distractor repeats are compared within HP and within LP trials and '
                f'averaged. Icons: the four possible locations; filled, the item on this trial, hollow, on the previous trial. n = {n}, '
                'correct trials. The brackets give the median within-subject difference and the Wilcoxon signed-rank test '
                '(** p < .01, *** p < .001); the differences themselves are in Fig. 1e. Dots are subjects, triangles subjects '
                'beyond the axis limits; large points and bars are the mean ± within-subject SEM (Cousineau-Morey).')
    if exclude and len(exclude) == 2:
        legend_s += f' The two pilot participants (sub-0{exclude[0]}, sub-0{exclude[1]}) are left out.'
    save(fs, f'figS_priming_rt_v4{suffix}', legend_s)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--rt-table', default='notes/data/behaviour_rt_2026-11.tsv')
    ap.add_argument('--bids-folder', default='/data/ds-retsupp')
    ap.add_argument('--trial-table', default='notes/data/trialwise_af_trial_table.tsv.gz')
    ap.add_argument('--exclude-subjects', type=int, nargs='*', default=[])
    ap.add_argument('--suffix', default='', help='appended to the figure names, e.g. _n28')
    a = ap.parse_args()
    main(a.rt_table, a.bids_folder, a.trial_table, a.exclude_subjects, a.suffix)
