"""Figure 7 (prototype): the gains behind the behavioural effects of Fig. 1, and behaviour trial by trial.

a  Illustration: gain at the target (each trial), at the distractor (each trial), at the learned HP location (whole block;
   Fig. 5) and over the whole display (global).
b  Location, the gain analogue of Fig. 1c, d: distractor gain HP − LP (each term estimated alone, as Fig. 5f) next to the
   target gain HP − LP (also estimated alone), per ROI.
c  History, the gain analogue of Fig. 1e, f: target gain repeat − switch next to distractor gain repeat − switch, per ROI
   (October attention-field fits with the gains split by repeat).
d  Trial by trial: target, distractor-at-HP and global gain as predictors of log RT (one Bayesian hierarchical model per
   ROI); the global gain's sign flips along the hierarchy.

b uses the current (provisional) gain fits; c and d the October trial-wise analysis on the earlier attention-field fits
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
from retsupp.visualize.paper.paper_style import (HP, DISTRACTOR, DOUBLE, FG, MUTED, ROI_ORDER, ROI_X, TARGET, letter, offset_spines,
                                                 roi_axis, save)
from retsupp.visualize.paper.plot_fig1 import RING, item_positions

LEGEND = """
**Figure 7 (prototype). The behavioural effects of Fig. 1 in the gains, and gain and behaviour trial by trial.** a, The gain terms (illustrative): at the target and at the distractor on each trial, at the HP location over a whole block (Fig. 5), and a global gain over the whole display on each trial. b, c, The gain analogues of the response-time effects in Fig. 1c-f; the display icons are those of Fig. 1 (four possible locations, HP ringed in red; filled, the item on this trial; hollow, on the previous trial), and each series is the gain in the first condition minus that in the second. b, Location (Fig. 1c, d): the gain the distractor draws at the HP location minus at an LP location (dD, estimated alone, as in Fig. 5f; current fits, provisional) and the target gain at the HP location minus at an LP location (dT, likewise estimated alone; current fits, provisional). Less gain for a distractor at HP in V3AB, IPS, hV4 and VO; the target gain shows no reliable HP − LP difference after correction (smallest q = .07, V3, and a positive trend in IPS, q = .10). Fitted jointly with all other terms, dT is null in every ROI as well (text). c, History (Fig. 1e, f): target gain on target-repeat minus target-switch trials (fits with the target gain split by repeat), and distractor gain on distractor-repeat minus distractor-switch trials (fits with separate distractor gains on repeat and switch trials, the difference taken within HP and within LP and averaged, as for the response times in Fig. 1f). Less target gain on repeats in V3AB, IPS, hV4 and VO; the distractor gain does not differ after correction. b, c: dots are subjects (n = 30; triangles, beyond the axis), points and bars the median and bootstrapped 95% CI; stars, Wilcoxon signed-rank test against 0, Benjamini-Hochberg across the nine ROIs within each series: * q < .05, ** q < .01, *** q < .001; all values below (n per ROI in the table). d, Within subjects, trial by trial: Bayesian hierarchical linear model of log RT (z within subject) per ROI on the per-trial target gain, distractor gain on HP and on LP trials, and global gain (each z-scored within run and trial type), with trial number, target repeat, distractor repeat and target-at-previous-distractor as covariates, and random intercepts and gain slopes per subject (bambi, 4 × 1000 draws, no divergences, R̂ ≤ 1.01). Points are posterior means, bars 95% HDIs; stars give the posterior probability that the effect has the sign shown: * > .975, ** > .995, *** > .9995 (not corrected across ROIs). The distractor gain on LP trials and the HP − LP difference are in the table. c, d use the earlier attention-field fits (October 2026) and will be redone on the final fits.
"""
STAMP = 'b: provisional, current fits · c, d: earlier fits (Oct 2026)'


def stars(q):
    return '***' if q < .001 else '**' if q < .01 else '*' if q < .05 else ''


def pd_stars(row):
    """Stars from the posterior probability of the sign: * > .975 (95% HDI-like), ** > .995, *** > .9995."""
    pdir = max(row.p_gt0, row.p_lt0)
    return '***' if pdir > .9995 else '**' if pdir > .995 else '*' if pdir > .975 else ''


def panel_bayes(ax, bayes, terms, ylim, ylabel, notes, stack=True):
    d = bayes[bayes.roi.isin(ROI_ORDER)]
    off = np.linspace(-.22, .22, len(terms)) if len(terms) > 1 else [0]
    for k, (o, (term, col)) in enumerate(zip(off, terms)):
        r = d[d.term == term].set_index('roi').loc[ROI_ORDER]
        for roi, row in r.iterrows():
            ax.errorbar(ROI_X[roi] + o, row['mean'], yerr=[[row['mean'] - row.hdi_lo], [row.hdi_hi - row['mean']]], fmt='o', ms=5,
                        color=col, mec='w', mew=.7, lw=1.4, capsize=0)
            ax.text(ROI_X[roi] + o, ylim[1] + (.02 + .08 * k * stack) * (ylim[1] - ylim[0]), pd_stars(row), ha='center', va='bottom',
                    fontsize=8.5, color=col, clip_on=False)
    ax.hlines(0, -.6, ROI_X[ROI_ORDER[-1]] + .6, color='.7', lw=.7, ls='--', zorder=0)
    x1 = ROI_X[ROI_ORDER[-1]] + .6
    ax.text(x1, ylim[1], notes[0], ha='right', va='top', fontsize=6, color='.4')
    ax.text(x1, ylim[0], notes[1], ha='right', va='bottom', fontsize=6, color='.4')
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)
    offset_spines(ax)
    roi_axis(ax)


def panel_sources(ax):
    """Four displays in a column, each with one gain field: target, distractor, HP location, global (whole display)."""
    from matplotlib.patches import Circle
    ring = 4. / np.sqrt(2) * np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]])
    g = np.linspace(-8.5, 8.5, 281)
    xx, yy = np.meshgrid(g, g)
    step = 19.
    # title, where and when, colour, centre (None: everywhere), width
    src = [('Target', 'At the target, each trial', TARGET, 1, 1.4),
           ('Distractor', 'At the distractor, each trial', DISTRACTOR, 3, 1.4),
           ('HP location', 'At HP, whole block (Fig. 5)', HP, 3, 2.),
           ('Global', 'Everywhere, each trial', FG, None, None)]
    for k, (title, sub, col, ix, sig) in enumerate(src):
        x0, y0 = 0., -k * step
        if ix is None:
            f = .3 / (1 + np.exp((np.hypot(xx, yy) - 5.6) / .25))   # flat over the display, soft edge
        else:
            f = .75 * np.exp(-((xx - ring[ix, 0]) ** 2 + (yy - ring[ix, 1]) ** 2) / (2 * sig ** 2))
        rgba = np.zeros(f.shape + (4,))
        rgba[..., :3] = mpl.colors.to_rgb(col)
        rgba[..., 3] = f
        ax.imshow(rgba, extent=(x0 - 6.2, x0 + 6.2, y0 - 6.2, y0 + 6.2), origin='lower', zorder=1)
        ax.add_patch(Circle((x0, y0), 3.17, fill=False, ec=MUTED, lw=.6, zorder=2))
        for p in ring:
            ax.add_patch(Circle((x0 + p[0], y0 + p[1]), .55, facecolor='w', ec=MUTED, lw=.6, zorder=2))
        ax.add_patch(Circle((x0 + ring[3, 0], y0 + ring[3, 1]), 1.15, fill=False, ec=HP, lw=.9, ls=(0, (2, 1.2)), zorder=3))
        if title == 'Target':
            ax.scatter(x0 + ring[1, 0], y0 + ring[1, 1], marker='D', s=14, color=TARGET, lw=0, zorder=4)
        if title == 'Distractor':
            ax.scatter(x0 + ring[3, 0], y0 + ring[3, 1], marker='o', s=18, color=DISTRACTOR, lw=0, zorder=4)
        ax.plot(x0, y0, '+', color=FG, ms=3.5, mew=.6, zorder=4)
        ax.text(x0, y0 + 6.3, title, ha='center', va='bottom', fontsize=7, color=col)
        ax.text(x0, y0 - 6.3, sub, ha='center', va='top', fontsize=5.8, color='.35')
    ax.set_xlim(-13, 13)
    ax.set_ylim(-3 * step - 9.5, 9.6)
    ax.set_aspect('equal')
    ax.set_axis_off()


def display_icon(ax, bounds, items):
    """The display icon of Fig. 1 (``fig1_paradigm_behaviour.cond_icons``): the four diagonal locations, HP ringed in red,
    and the items, (location index, 'target' or 'distractor', filled); hollow is the previous trial's item."""
    from matplotlib.patches import Circle
    pos = item_positions()
    ins = ax.inset_axes(bounds, transform=ax.transAxes)
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


def icon_key(ax, contrast, entries, ref, y=1.17, h=.27, step=.47):
    """Above the panel, per series: the Fig. 1 icon pair 'A − B' and what the series shows; the Fig. 1 panels at the right."""
    fig = ax.figure
    fig.canvas.draw()
    bb = ax.get_window_extent()
    w = h * bb.height / bb.width
    x = 0.
    for kind, col, label in entries:
        a, b = ICONS[contrast][kind]
        display_icon(ax, [x, y, w, h], a)
        ax.text(x + w + .004, y + h / 2, '−', transform=ax.transAxes, fontsize=8, color=FG, va='center', ha='left')
        display_icon(ax, [x + w + .02, y, w, h], b)
        ax.text(x + 2 * w + .03, y + h / 2, label, transform=ax.transAxes, fontsize=6.5, color=col, va='center', ha='left')
        x += step
    ax.text(1., y + h / 2, ref, transform=ax.transAxes, fontsize=6, color='.4', va='center', ha='right')


def series_stats(df, col):
    rows = []
    for roi in ROI_ORDER:
        v = df[df.roi == roi][col].dropna().to_numpy()
        m, a, b = median_ci(v)
        rows.append(dict(roi=roi, n=len(v), median=m, lo=a, hi=b, p=stats.wilcoxon(v).pvalue))
    out = pd.DataFrame(rows)
    out['q'] = stats.false_discovery_control(out.p)
    return out


def panel_pair(ax, series, ylim, ylabel, xlabels=True):
    """Per ROI, two series side by side (subject dots, median and bootstrapped 95% CI); stars per series, BH across ROIs."""
    lo, hi = ylim
    off = (-.2, .2)
    out = {}
    for k, (name, df, col, color) in enumerate(series):
        st = series_stats(df, col)
        out[name] = st
        for i, roi in enumerate(ROI_ORDER):
            v = df[df.roi == roi][col].dropna().to_numpy()
            rng = np.random.default_rng(10 * k + i)
            x = ROI_X[roi] + off[k]
            xx = x + rng.uniform(-.09, .09, len(v))
            ok = (v >= lo) & (v <= hi)
            ax.scatter(xx[ok], v[ok], s=5, color=color, alpha=.35, lw=0, zorder=1)
            ax.scatter(xx[v > hi], np.full((v > hi).sum(), hi), s=7, marker='^', color=color, alpha=.6, lw=0, clip_on=False)
            ax.scatter(xx[v < lo], np.full((v < lo).sum(), lo), s=7, marker='v', color=color, alpha=.6, lw=0, clip_on=False)
            r = st.iloc[i]
            ax.errorbar(x, r['median'], yerr=[[r['median'] - r.lo], [r.hi - r['median']]], fmt='o', ms=4.5, color=color, mec='w',
                        mew=.7, lw=1.4, capsize=0, zorder=4)
            ax.text(x, hi + .03 * (hi - lo), stars(r.q), ha='center', va='bottom', fontsize=8, color=color, clip_on=False)
    ax.hlines(0, -.6, ROI_X[ROI_ORDER[-1]] + .6, color='.7', lw=.7, ls='--', zorder=0)
    ax.set_ylim(lo, hi)
    ax.set_ylabel(ylabel)
    offset_spines(ax)
    if xlabels:
        roi_axis(ax)
    else:
        ax.set_xticks([ROI_X[r] for r in ROI_ORDER])
        ax.set_xticklabels([])
        ax.set_xlim(ROI_X[ROI_ORDER[0]] - .6, ROI_X[ROI_ORDER[-1]] + .6)
    return out


def header(fig, y, text, x0=.02, x1=.99):
    """Row header across the figure: what the panels below have in common."""
    fig.text(x0, y, text, fontsize=7.5, fontweight='bold', color=FG, va='bottom', ha='left')
    fig.add_artist(plt.Line2D([x0, x1], [y - .005] * 2, color=MUTED, lw=.6, transform=fig.transFigure))


def stats_table(results):
    lines = ['', '**Per-ROI statistics (b, c).** Median [bootstrapped 95% CI], Wilcoxon signed-rank p against 0, '
             'Benjamini-Hochberg q across the nine ROIs within each series; n = 30 subjects per ROI unless given otherwise.', '']
    for name, st in results.items():
        lines += [f'*{name}*', '', '| ROI | n | Median [95% CI] | p | q |', '|---|---|---|---|---|']
        lines += [f'| {r.roi} | {r.n} | {r["median"]:+.3f} [{r.lo:+.3f}, {r.hi:+.3f}] | {r.p:.3g} | {r.q:.3g} |'
                  for _, r in st.iterrows()]
        lines.append('')
    return '\n'.join(lines).replace('-0.', '−0.').replace('+0.', '+0.')


def ax_in(fig, x, y, w, h):
    """Axes from a box in inches (left, bottom, width, height)."""
    W, H = fig.get_size_inches()
    return fig.add_axes([x / W, y / H, w / W, h / H])


def main(repeat_table, bayes_table, conditional_table, dt_table, name='fig7_behaviour'):
    bayes = pd.read_csv(bayes_table, sep='\t')
    rep = pd.read_csv(repeat_table, sep='\t')
    cond = conditional(conditional_table)
    dt = pd.read_csv(dt_table, sep='\t')
    t_rep = rep[rep.model == 'targetRepeat'].assign(d=lambda d: d.g_T_dyn_repeat - d.g_T_dyn_switch)
    # The repeat model has separate distractor gains on repeat and on switch trials (g_HP_dyn, g_LP_dyn are the switch
    # gains); repeat − switch is taken within HP and within LP trials and averaged, as for the response times (Fig. 1f).
    d_rep = rep[rep.model == 'repeat'].assign(d=lambda d: ((d.g_HP_dyn_repeat - d.g_HP_dyn) + (d.g_LP_dyn_repeat - d.g_LP_dyn)) / 2)

    W, H = DOUBLE, 6.55
    fig = plt.figure(figsize=(W, H))
    x0, pw = 1.85, 5.1             # left edge and width of the per-ROI panels (inches)
    # a: the gain terms, a column at the left.
    ax = ax_in(fig, 0., 2.6, 1.45, 3.75)
    panel_sources(ax)
    letter(ax, 'a', x=.12, y=1.0)
    header(fig, (H - .2) / H, 'The behavioural effects of Fig. 1, in the gains', x0=1.42 / W)
    results = {}
    # b: location, HP − LP (Fig. 1c, d).
    ax = ax_in(fig, x0, 4.6, pw, 1.1)
    results.update(panel_pair(ax, [('Distractor gain, HP − LP (estimated alone)', cond, 'dD', DISTRACTOR),
                                   ('Target gain, HP − LP (estimated alone)', dt, 'dT', TARGET)],
                              (-.4, .4), 'Gain, HP − LP', xlabels=False))
    ax.set_yticks([-.4, -.2, 0, .2, .4])
    icon_key(ax, 'location', [('distractor', DISTRACTOR, 'Distractor: less gain at HP'), ('target', TARGET, 'Target: no reliable difference')],
             'As Fig. 1c, d')
    letter(ax, 'b', x=-.08, y=1.2)
    # c: history, repeat − switch (Fig. 1e, f).
    ax = ax_in(fig, x0, 2.8, pw, 1.1)
    results.update(panel_pair(ax, [('Target gain, repeat − switch', t_rep, 'd', TARGET),
                                   ('Distractor gain, repeat − switch', d_rep, 'd', DISTRACTOR)],
                              (-1.5, 1.), 'Gain, repeat − switch'))
    icon_key(ax, 'history', [('target', TARGET, 'Target: less gain on repeats'), ('distractor', DISTRACTOR, 'Distractor: no reliable difference')],
             'As Fig. 1e, f')
    letter(ax, 'c', x=-.08, y=1.2)
    # d: trial by trial.
    header(fig, 1.98 / H, 'Trial by trial: gain on a trial predicts its response time')
    ax = ax_in(fig, .75, .62, 6.2, 1.05)
    panel_bayes(ax, bayes, [('T', TARGET), ('D_HP', DISTRACTOR), ('G', FG)], (-.08, .13), 'Log RT per\nunit gain (z)',
                ('Slower', 'Faster'), stack=False)
    ax.set_yticks([-.05, 0, .05, .1])
    for k, (lab, col) in enumerate((('Target gain', TARGET), ('Distractor gain, HP trials', DISTRACTOR),
                                    ('Global gain: the sign flips', FG))):
        ax.text(-.4, .125 - .026 * k, lab, color=col, fontsize=6.5, ha='left', va='center')
    letter(ax, 'd', x=-.09, y=1.12)
    fig.text(.995, .998, STAMP, fontsize=5.5, color=MUTED, ha='right', va='top')
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
    a = ap.parse_args()
    main(a.repeat_table, a.bayes_table, a.conditional_table, a.dt_table)
