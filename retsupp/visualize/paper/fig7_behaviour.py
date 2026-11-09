"""Figure 7 (prototype): the gains behind the behavioural effects of Fig. 1, and behaviour trial by trial.

a  Location, the gain analogue of Fig. 1c, d: distractor gain HP − LP (each term estimated alone, as Fig. 5c) next to the
   target gain HP − LP (also estimated alone), per ROI.
b  History, the gain analogue of Fig. 1e, f: target gain repeat − switch next to distractor gain repeat − switch, per ROI
   (October attention-field fits with the gains split by repeat).
c  Trial by trial: target, distractor-at-HP and global gain as predictors of log RT (one Bayesian hierarchical model per
   ROI); the global gain's sign flips along the hierarchy. Keyed by small drawings of the three gain fields.

a uses the current (provisional) gain fits; b and c the October trial-wise analysis on the earlier attention-field fits
(pSig0.5), stamped as such until they are redone on the ladder fits. Between-subject brain-behaviour correlations (null)
stay in the text.

Usage::

    python -m retsupp.visualize.paper.fig7_behaviour
"""
from __future__ import annotations

import argparse

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from retsupp.visualize.paper.fig45_common import conditional, median_ci
from retsupp.visualize.paper.paper_style import (HP, DISTRACTOR, DOUBLE, FG, MUTED, ROI_ORDER, ROI_X, TARGET, ROI_GROUPS, offset_spines,
                                                 roi_axis, save)
from retsupp.visualize.paper.plot_fig1 import RING, item_positions

LEGEND = """
**Figure 7 (prototype). The behavioural effects of Fig. 1 in the gains, and gain and behaviour trial by trial.** a, b, The gain analogues of the response-time effects in Fig. 1c-f; the display icons are those of Fig. 1 (four possible locations, HP ringed in red; filled, the item on this trial; hollow, on the previous trial), and each series is the gain in the first condition minus that in the second. a, Location (Fig. 1c, d): the gain the distractor draws at the HP location minus at an LP location (dD, estimated alone, as in Fig. 5c; current fits, provisional) and the target gain at the HP location minus at an LP location (dT, likewise estimated alone; current fits, provisional). Less gain for a distractor at HP in V3AB, IPS, hV4 and VO; the target gain shows no reliable HP − LP difference after correction (smallest q = .07, V3, and a positive trend in IPS, q = .10). Fitted jointly with all other terms, dT is null in every ROI as well (text). b, History (Fig. 1e, f): target gain on target-repeat minus target-switch trials (fits with the target gain split by repeat), and distractor gain on distractor-repeat minus distractor-switch trials (fits with separate distractor gains on repeat and switch trials, the difference taken within HP and within LP and averaged, as for the response times in Fig. 1f). Less target gain on repeats in V3AB, IPS, hV4 and VO; the distractor gain does not differ after correction. a, b: points and bars are the median over subjects (n = 30) and its bootstrapped 95% CI (subject-level spread: Fig. 5); Wilcoxon signed-rank test against 0, Benjamini-Hochberg across the nine ROIs within each series; estimates with q < .05 are drawn solid, the others pale (key at the top); stars: * q < .05, ** q < .01, *** q < .001; all values below (n per ROI in the table). c, Within subjects, trial by trial: Bayesian hierarchical linear model of log RT (z within subject) per ROI on the per-trial target gain (gain field at the target), distractor gain on HP and on LP trials (gain field at the distractor), and global gain (over the whole display) (each z-scored within run and trial type), with trial number, target repeat, distractor repeat and target-at-previous-distractor as covariates, and random intercepts and gain slopes per subject (bambi, 4 × 1000 draws, no divergences, R̂ ≤ 1.01); the icons show each gain field (illustrative); up is slower. Points are posterior means, bars 95% HDIs, solid when the HDI excludes 0 and pale otherwise; stars give the posterior probability that the effect has the sign shown: * > .975, ** > .995, *** > .9995 (not corrected across ROIs). The distractor gain on LP trials and the HP − LP difference are in the table. b, c use the earlier attention-field fits (October 2026) and will be redone on the final fits.
"""
WIDTH = 5.2   # stacked layout: 1.5 columns (132 mm); the side layout (default) is double column
STAMP = 'a: provisional, current fits · b, c: earlier fits (Oct 2026)'


def stars(q):
    return '***' if q < .001 else '**' if q < .01 else '*' if q < .05 else ''


def pd_stars(row):
    """Stars from the posterior probability of the sign: * > .975 (95% HDI-like), ** > .995, *** > .9995."""
    pdir = max(row.p_gt0, row.p_lt0)
    return '***' if pdir > .9995 else '**' if pdir > .995 else '*' if pdir > .975 else ''


# ROI positions with a wider gap between pathways than between ROIs (as ROI_GROUPS in paper_style).
GAP7 = 1.7
XP, _x = {}, 0.
for _, _rois in ROI_GROUPS:
    for _r in _rois:
        XP[_r] = _x
        _x += 1
    _x += GAP7 - 1
XMAX = max(XP.values())
PALE = .35


def estimate(ax, x, m, lo, hi, color, solid):
    """Point and interval; solid when significant, pale and thinner when not."""
    if solid:
        ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], fmt='o', ms=4, color=color, mec='w', mew=.5, lw=1.0, capsize=0, zorder=4)
    else:
        ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], fmt='o', ms=3.4, color=color, mec='none', lw=.8, capsize=0, alpha=PALE,
                    zorder=3)


def zero_line(ax):
    ax.hlines(0, -.6, XMAX + .6, color='.82', lw=.6, zorder=0)


def roi_axis7(ax, labels=True, fontsize=6.5, pad_pt=2, group_names=None):
    """ROI ticks at XP with the pathway names under them (as paper_style.roi_axis)."""
    import matplotlib.transforms as mtrans
    ax.set_xticks([XP[r] for r in ROI_ORDER])
    ax.set_xlim(-.6, XMAX + .6)
    if not labels:
        ax.set_xticklabels([])
        return
    ax.set_xticklabels(ROI_ORDER, fontsize=fontsize)
    fig = ax.figure
    fig.canvas.draw()
    inv = ax.transAxes.inverted()
    bottom = min(inv.transform((0, t.get_window_extent().y0))[1] for t in ax.get_xticklabels() if t.get_text())
    tr = mtrans.blended_transform_factory(ax.transData, ax.transAxes)
    off = mtrans.offset_copy(tr, fig=fig, y=-pad_pt, units='points')
    for (name, rois), nm in zip(ROI_GROUPS, group_names or [g for g, _ in ROI_GROUPS]):
        name = nm
        x0, x1 = XP[rois[0]] - .35, XP[rois[-1]] + .35
        ax.plot([x0, x1], [bottom] * 2, transform=off, color=MUTED, lw=.6, clip_on=False, solid_capstyle='butt')
        ax.text((x0 + x1) / 2, bottom, name, transform=mtrans.offset_copy(tr, fig=fig, y=-pad_pt - 1.5, units='points'),
                ha='center', va='top', fontsize=6, color='.45', clip_on=False)


def panel_bayes(ax, bayes, terms, ylim, ylabel):
    """Posterior mean and 95% HDI per ROI and term; solid when the HDI excludes 0."""
    d = bayes[bayes.roi.isin(ROI_ORDER)]
    off = np.linspace(-.22, .22, len(terms)) if len(terms) > 1 else [0]
    for o, (term, col) in zip(off, terms):
        r = d[d.term == term].set_index('roi').loc[ROI_ORDER]
        for roi, row in r.iterrows():
            estimate(ax, XP[roi] + o, row['mean'], row.hdi_lo, row.hdi_hi, col, row.hdi_lo > 0 or row.hdi_hi < 0)
            ax.text(XP[roi] + o, ylim[1] + .02 * (ylim[1] - ylim[0]), pd_stars(row), ha='center', va='bottom', fontsize=8.5,
                    color=col, clip_on=False)
    dd = d[d.term.isin([t for t, _ in terms])]
    assert dd.hdi_lo.min() >= ylim[0] and dd.hdi_hi.max() <= ylim[1]
    zero_line(ax)
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)
    offset_spines(ax)
    roi_axis7(ax)


def draw_source(ax, kind):
    """One display with one gain field (as the gain-term illustrations): 'target', 'distractor' or 'global'."""
    from matplotlib.patches import Circle
    ring = 4. / np.sqrt(2) * np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]])
    g = np.linspace(-6.2, 6.2, 201)
    xx, yy = np.meshgrid(g, g)
    col, ix = {'target': (TARGET, 1), 'distractor': (DISTRACTOR, 3), 'global': (FG, None)}[kind]
    if ix is None:
        f = .3 / (1 + np.exp((np.hypot(xx, yy) - 5.6) / .25))   # flat over the display, soft edge
    else:
        f = .75 * np.exp(-((xx - ring[ix, 0]) ** 2 + (yy - ring[ix, 1]) ** 2) / (2 * 1.4 ** 2))
    rgba = np.zeros(f.shape + (4,))
    rgba[..., :3] = mpl.colors.to_rgb(col)
    rgba[..., 3] = f
    ax.imshow(rgba, extent=(-6.2, 6.2, -6.2, 6.2), origin='lower', zorder=1)
    ax.add_patch(Circle((0, 0), 3.17, fill=False, ec=MUTED, lw=.5, zorder=2))
    for p in ring:
        ax.add_patch(Circle(p, .7, facecolor='w', ec=MUTED, lw=.4, zorder=2))
    ax.add_patch(Circle(ring[3], 1.3, fill=False, ec=HP, lw=.7, ls=(0, (2, 1.2)), zorder=3))
    if kind == 'target':
        ax.plot(*ring[1], 'D', ms=2.6, color=TARGET, mew=0, zorder=4)
    if kind == 'distractor':
        ax.plot(*ring[3], 'o', ms=3, color=DISTRACTOR, mew=0, zorder=4)
    ax.plot(0, 0, '+', color=FG, ms=2.5, mew=.5, zorder=4)
    ax.set_xlim(-6.2, 6.2)
    ax.set_ylim(-6.2, 6.2)
    ax.set_aspect('equal')
    ax.set_axis_off()


def display_icon(ins, items):
    """The display icon of Fig. 1 (``fig1_paradigm_behaviour.cond_icons``) drawn into ``ins``: the four diagonal
    locations, HP ringed in red, and the items, (location index, 'target' or 'distractor', filled); hollow is the previous
    trial's item."""
    from matplotlib.patches import Circle
    pos = item_positions()
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


# The conditions of Fig. 1 as icon pairs (first minus second).
ICONS = {'location': {'distractor': ([(7, 'distractor', True)], [(3, 'distractor', True)]),
                      'target': ([(7, 'target', True)], [(3, 'target', True)])},
         'history': {k: ([(3, k, False), (3, k, True)], [(1, k, False), (3, k, True)]) for k in ('target', 'distractor')}}


ICON = .3   # icon size (inches)


def row(fig, y, let, title, entries, ref=None, x_title=.2, key_y=None, x_let=.02, x_ref=None):
    """One heading line (as Fig. 6): panel letter, a title stating the finding, then the key: per series its icon(s) and a
    coloured label; the Fig. 1 panels it corresponds to at the right. title: (text, colour) segments, dark with the key words
    in colour. entries: (draw functions, colour, label)."""
    W, H = fig.get_size_inches()
    fig.text(x_let / W, y / H, let, fontsize=8, fontweight='bold', va='center', ha='left')
    x = x_title
    for seg, col in title:
        t = fig.text(x / W, y / H, seg, fontsize=7.5, color=col, va='center', ha='left')
        fig.canvas.draw()
        x = t.get_window_extent().x1 / fig.dpi
    x += .3
    if key_y is not None:          # key on its own line under the title
        x, y = x_title, key_y
    for draws, col, label in entries:
        for k, draw in enumerate(draws):
            if k:
                fig.text((x + .015) / W, y / H, '−', fontsize=8, color=FG, va='center', ha='left')
                x += .1
            draw(fig.add_axes([x / W, (y - ICON / 2) / H, ICON / W, ICON / H]))
            x += ICON
        lt = fig.text((x + .04) / W, y / H, label, fontsize=6.5, color=col, va='center', ha='left')
        fig.canvas.draw()
        x = lt.get_window_extent().x1 / fig.dpi + .22
    if ref:
        r = fig.text(((x_ref or W - .08)) / W, y / H, ref, fontsize=6, color='.4', va='center', ha='right')
        fig.canvas.draw()
        assert r.get_window_extent().x0 / fig.dpi > x - .15, 'heading line too long'


def icons(contrast, kind):
    a, b = ICONS[contrast][kind]
    return [lambda ax, it=a: display_icon(ax, it), lambda ax, it=b: display_icon(ax, it)]


def series_stats(df, col):
    rows = []
    for roi in ROI_ORDER:
        v = df[df.roi == roi][col].dropna().to_numpy()
        m, a, b = median_ci(v)
        rows.append(dict(roi=roi, n=len(v), median=m, lo=a, hi=b, p=stats.wilcoxon(v).pvalue))
    out = pd.DataFrame(rows)
    out['q'] = stats.false_discovery_control(out.p)
    return out


def panel_pair(ax, series, ylim, ylabel, xlabels=True, roi_fs=6.5, group_names=None):
    """Per ROI, two series side by side: median and bootstrapped 95% CI; solid when q < .05 (Wilcoxon against 0,
    Benjamini-Hochberg across ROIs), pale otherwise. Subject-level spread is in Fig. 5."""
    lo, hi = ylim
    off = (-.17, .17)
    out = {}
    for k, (name, df, col, color) in enumerate(series):
        st = series_stats(df, col)
        out[name] = st
        for i, roi in enumerate(ROI_ORDER):
            r = st.iloc[i]
            estimate(ax, XP[roi] + off[k], r['median'], r.lo, r.hi, color, r.q < .05)
            ax.text(XP[roi] + off[k], hi + .02 * (hi - lo), stars(r.q), ha='center', va='bottom', fontsize=8.5, color=color,
                    clip_on=False)
    assert all(st.lo.min() >= lo and st.hi.max() <= hi for st in out.values()), ylim
    zero_line(ax)
    ax.set_ylim(lo, hi)
    ax.set_ylabel(ylabel)
    offset_spines(ax)
    roi_axis7(ax, labels=xlabels, fontsize=roi_fs, group_names=group_names)
    return out


def stats_table(results):
    lines = ['', '**Per-ROI statistics (a, b).** Median [bootstrapped 95% CI], Wilcoxon signed-rank p against 0, '
             'Benjamini-Hochberg q across the nine ROIs within each series; n = 30 subjects per ROI unless given otherwise.', '']
    for name, st in results.items():
        lines += [f'*{name}*', '', '| ROI | n | Median [95% CI] | p | q |', '|---|---|---|---|---|']
        lines += [f'| {r.roi} | {r.n} | {r["median"]:+.3f} [{r.lo:+.3f}, {r.hi:+.3f}] | {r.p:.3g} | {r.q:.3g} |'.replace('-', '−')
                  for _, r in st.iterrows()]
        lines.append('')
    return '\n'.join(lines)


def ax_in(fig, x, y, w, h):
    """Axes from a box in inches (left, bottom, width, height)."""
    W, H = fig.get_size_inches()
    return fig.add_axes([x / W, y / H, w / W, h / H])


def sig_key(fig, y):
    """Glyph key for the solid / pale encoding, drawn with the panels' own marks."""
    W, H = fig.get_size_inches()
    ax = fig.add_axes([.02 / W, (y - .09) / H, 4.2 / W, .18 / H])
    ax.set_xlim(0, 4.2)
    ax.set_ylim(-1, 1)
    ax.set_axis_off()
    estimate(ax, .05, 0, -.8, .8, FG, True)
    ax.text(.13, 0, 'q < .05 (a, b) or 95% HDI excludes 0 (c)', fontsize=6, color=FG, va='center')
    estimate(ax, 2.35, 0, -.8, .8, FG, False)
    ax.text(2.43, 0, 'Not significant', fontsize=6, color='.45', va='center')


def main(repeat_table, bayes_table, conditional_table, dt_table, name='fig7_behaviour', W=WIDTH, layout='side'):
    bayes = pd.read_csv(bayes_table, sep='\t')
    rep = pd.read_csv(repeat_table, sep='\t')
    cond = conditional(conditional_table)
    dt = pd.read_csv(dt_table, sep='\t')
    t_rep = rep[rep.model == 'targetRepeat'].assign(d=lambda d: d.g_T_dyn_repeat - d.g_T_dyn_switch)
    # The repeat model has separate distractor gains on repeat and on switch trials (g_HP_dyn, g_LP_dyn are the switch
    # gains); repeat − switch is taken within HP and within LP trials and averaged, as for the response times (Fig. 1f).
    d_rep = rep[rep.model == 'repeat'].assign(d=lambda d: ((d.g_HP_dyn_repeat - d.g_HP_dyn) + (d.g_LP_dyn_repeat - d.g_LP_dyn)) / 2)

    A = [('Distractor gain, HP − LP (estimated alone)', cond, 'dD', DISTRACTOR), ('Target gain, HP − LP (estimated alone)', dt, 'dT', TARGET)]
    B = [('Target gain, repeat − switch', t_rep, 'd', TARGET), ('Distractor gain, repeat − switch', d_rep, 'd', DISTRACTOR)]
    title_a = [('Location: less gain for a ', FG), ('distractor', DISTRACTOR), (' at HP', FG)]
    title_b = [('History: less ', FG), ('target', TARGET), (' gain when its location repeats', FG)]
    key_a = [(icons('location', 'distractor'), DISTRACTOR, 'Distractor gain'), (icons('location', 'target'), TARGET, 'Target gain')]
    key_b = [(icons('history', 'target'), TARGET, 'Target gain'), (icons('history', 'distractor'), DISTRACTOR, 'Distractor gain')]
    key_c = [([lambda a: draw_source(a, 'target')], TARGET, 'Target gain'),
             ([lambda a: draw_source(a, 'distractor')], DISTRACTOR, 'Distractor gain, HP trials'),
             ([lambda a: draw_source(a, 'global')], FG, 'Global gain: the sign flips')]
    title_c = [('Trial by trial: gain on a trial predicts its response time', FG)]
    results = {}
    gap = .27                      # heading to axes: room for the stars
    if layout == 'side':
        # a and b side by side, c full width below (double column).
        W = DOUBLE
        head, ah, below, ch = .44, .95, .5, 1.0
        H = .26 + head + gap + ah + below + .22 + gap + ch + .4
        fig = plt.figure(figsize=(W, H))
        Y = lambda d: H - d
        sig_key(fig, Y(.1))
        fig.text(.995, Y(.1) / H, STAMP, fontsize=6, color=MUTED, ha='right', va='center')
        pwh, xa, xb = 2.85, .62, 4.2
        short = ['Early', 'Dorsal', 'Occipitotemporal']
        for (xl, let, title, ref, key, ser, ylim, ylab, yt) in (
                (xa, 'a', title_a, 'As Fig. 1c, d', key_a, A, (-.15, .12), 'Gain, HP − LP', [-.1, 0, .1]),
                (xb, 'b', title_b, 'As Fig. 1e, f', key_b, B, (-1.15, .55), 'Gain, repeat − switch', [-1, 0])):
            y = Y(.36)
            row(fig, y, let, title, key, x_let=xl - .6, x_title=xl - .45, key_y=y - .27)
            fig.text((xl + pwh) / W, y / H, ref, fontsize=6, color='.45', ha='right', va='center')
            ax = ax_in(fig, xl, Y(.26 + head + gap + ah), pwh, ah)
            results.update(panel_pair(ax, ser, ylim, ylab, roi_fs=6, group_names=short))
            ax.set_yticks(yt)
        yc = .26 + head + gap + ah + below
        row(fig, Y(yc + .1), 'c', title_c, key_c)
        ax = ax_in(fig, .62, Y(yc + .22 + gap + ch), W - .82, ch)
    else:
        # All three stacked; the 1.5-column width (or double with the keys on the title line).
        narrow = W < 6
        pw, x0 = W - .95, .75
        head = .44 if narrow else .2
        ah, below = (.72, .72, 1.0), (.24, .56, 0)
        tops, t = [], .48 if narrow else .26
        for k in range(3):
            tops.append(t)
            t += head + gap + ah[k] + below[k]
        H = t + .4
        fig = plt.figure(figsize=(W, H))
        Y = lambda d: H - d
        if narrow:
            sig_key(fig, Y(.32))
        else:
            sig_key(fig, Y(.1))
        fig.text(.995, Y(.1) / H, STAMP, fontsize=6, color=MUTED, ha='right', va='center')
        for k, (let, title, ref, key, ser, ylim, ylab, yt) in enumerate((
                ('a', title_a, 'As Fig. 1c, d', key_a, A, (-.15, .12), 'Gain,\nHP − LP', [-.1, 0, .1]),
                ('b', title_b, 'As Fig. 1e, f', key_b, B, (-1.15, .55), 'Gain,\nrepeat − switch', [-1, 0]))):
            y = Y(tops[k] + .1)
            if narrow:
                row(fig, y, let, title + [(f'  ({ref})', '.45')], key, key_y=y - .27)
            else:
                row(fig, y, let, title, key, ref)
            ax = ax_in(fig, x0, Y(tops[k] + head + gap + ah[k]), pw, ah[k])
            results.update(panel_pair(ax, ser, ylim, ylab, xlabels=k == 1))
            ax.set_yticks(yt)
        y = Y(tops[2] + .1)
        row(fig, y, 'c', title_c, key_c, key_y=y - .27 if narrow else None)
        ax = ax_in(fig, x0, Y(tops[2] + head + gap + ah[2]), pw, ah[2])
    panel_bayes(ax, bayes, [('T', TARGET), ('D_HP', DISTRACTOR), ('G', FG)], (-.07, .125), 'Log RT per\nunit gain (z)')
    ax.set_yticks([0, .1])
    ax.text(XMAX + .25, .112, 'Slower', ha='right', va='center', fontsize=6, color='.45')
    ax.annotate('', xy=(XMAX + .45, .124), xytext=(XMAX + .45, .1),
                arrowprops=dict(arrowstyle='-|>', color='.55', lw=.6, mutation_scale=5, shrinkA=0, shrinkB=0))
    for a in fig.axes:
        if a.get_ylabel().startswith('Gain'):
            bb = a.get_window_extent()
            assert bb.height <= bb.width, (a.get_ylabel(), bb.width, bb.height)   # subject-dot panels never taller than wide
    for nm, st in results.items():
        print(nm, ', '.join(f'{r.roi} {r["median"]:+.3f} (p={r.p:.3g}, q={r.q:.3g})' for _, r in st.iterrows()))
    save(fig, name, LEGEND.strip() + '\n' + stats_table(results))

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repeat-table', default='notes/data/repeat_af_fits_params.tsv')
    ap.add_argument('--bayes-table', default='notes/data/trialwise_rt_bayes_summary.tsv')
    ap.add_argument('--conditional-table', default='notes/data/conditional_estimates_dog_2026-11.tsv')
    ap.add_argument('--dt-table', default='notes/data/conditional_dT_dog_2026-11.tsv')
    ap.add_argument('--name', default='fig7_behaviour')
    ap.add_argument('--layout', default='side', choices=['stacked', 'side'], help='side: a, b next to each other, double column')
    ap.add_argument('--width', type=float, default=WIDTH, help='inches; below 6 the keys go on their own line')
    a = ap.parse_args()
    main(a.repeat_table, a.bayes_table, a.conditional_table, a.dt_table, a.name, a.width, a.layout)
