"""Figure 7 (prototype): the history effects of Fig. 1 in the gains, and gain and behaviour trial by trial.

a  History, the gain analogue of Fig. 1e, f: target gain repeat − switch next to distractor gain repeat − switch, per ROI
   (October attention-field fits with the gains split by repeat). The location analogue (Fig. 1c, d) is Fig. 5d.
b  Trial by trial: target, distractor (HP and LP trials averaged) and global gain as predictors of log RT (one Bayesian
   hierarchical model per ROI); the global gain's sign flips along the hierarchy. Keyed by small drawings of the gain fields.

Both use the October trial-wise analysis on the earlier attention-field fits (pSig0.5), stamped as such until they are redone
on the ladder fits. Between-subject brain-behaviour correlations (null) stay in the text. Estimates are solid when
significant, pale otherwise; stars as usual.

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

from retsupp.visualize.paper.fig45_common import median_ci
from retsupp.visualize.paper.paper_style import (HP, DISTRACTOR, DOUBLE, FG, MUTED, ROI_ORDER, ROI_X, TARGET, ROI_GROUPS, offset_spines,
                                                 roi_axis, save)
from retsupp.visualize.paper.plot_fig1 import RING, item_positions

LEGEND = """
**Figure 7 (prototype). The history effects of Fig. 1 in the gains, and gain and behaviour trial by trial.** a, History (Fig. 1e, f; the location effects of Fig. 1c, d are in Fig. 5d): target gain on target-repeat minus target-switch trials (fits with the target gain split by repeat), and distractor gain on distractor-repeat minus distractor-switch trials (fits with separate distractor gains on repeat and switch trials, the difference taken within HP and within LP and averaged, as for the response times in Fig. 1f); the display icons are those of Fig. 1e, f (four possible locations; filled, the item on this trial; hollow, on the previous trial; no HP ring, since HP and LP trials are pooled), each series the gain in the first condition minus that in the second. Less target gain on repeats in V3AB, IPS, hV4 and VO; the distractor gain does not differ after correction. Points and bars are the median over subjects (n = 30) and its bootstrapped 95% CI; Wilcoxon signed-rank test against 0, Benjamini-Hochberg across the nine ROIs within each series; estimates with q < .05 are drawn solid, the others pale (key at the top); stars: * q < .05, ** q < .01, *** q < .001; all values below. b, Within subjects, trial by trial: Bayesian hierarchical linear model of log RT (z within subject) per ROI on the per-trial target gain (gain field at the target), distractor gain (gain field at the distractor) on HP and on LP trials, and global gain (over the whole display), each z-scored within run and trial type, with trial number, target repeat, distractor repeat and target-at-previous-distractor as covariates, and random intercepts and gain slopes per subject (bambi, 4 × 1000 draws, no divergences, R̂ ≤ 1.01); the icons show each gain field (illustrative); up is slower. The distractor gain is shown as the slopes on HP and LP trials (entered separately) averaged per posterior draw; per trial type and their difference in the text and table (the HP − LP difference is not reliable: only IPS, uncorrected). Points are posterior means, bars 95% HDIs, solid when the HDI excludes 0 and pale otherwise; stars give the posterior probability that the effect has the sign shown: * > .975, ** > .995, *** > .9995 (not corrected across ROIs). Both panels use the earlier attention-field fits (October 2026) and will be redone on the final fits.
"""
STAMP = 'Earlier attention-field fits (Oct 2026)'


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
    # No HP ring: the trial-by-trial gains are not split by HP (the distractor term is averaged over HP and LP trials).
    if kind == 'target':
        ax.plot(*ring[1], 'D', ms=2.6, color=TARGET, mew=0, zorder=4)
    if kind == 'distractor':
        ax.plot(*ring[3], 'o', ms=3, color=DISTRACTOR, mew=0, zorder=4)
    ax.plot(0, 0, '+', color=FG, ms=2.5, mew=.5, zorder=4)
    ax.set_xlim(-6.2, 6.2)
    ax.set_ylim(-6.2, 6.2)
    ax.set_aspect('equal')
    ax.set_axis_off()


def display_icon(ins, items, hp=True):
    """The display icon of Fig. 1 (``fig1_paradigm_behaviour.cond_icons``) drawn into ``ins``: the four diagonal
    locations, HP ringed in red, and the items, (location index, 'target' or 'distractor', filled); hollow is the previous
    trial's item."""
    from matplotlib.patches import Circle
    pos = item_positions()
    ins.add_patch(Circle((0, 0), RING, fill=False, ec=MUTED, lw=.5))
    for j in (1, 3, 5, 7):
        ins.add_patch(Circle(pos[j], .95, facecolor='w', ec=MUTED, lw=.4))
    if hp:   # only where the contrast involves the HP location
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
    hp = contrast == 'location'   # the history contrast pools HP and LP trials, so no HP ring
    return [lambda ax, it=a: display_icon(ax, it, hp), lambda ax, it=b: display_icon(ax, it, hp)]


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
    lines = ['', '**Per-ROI statistics (a).** Median [bootstrapped 95% CI], Wilcoxon signed-rank p against 0, '
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
    ax.text(.13, 0, 'q < .05 (a) or 95% HDI excludes 0 (b)', fontsize=6, color=FG, va='center')
    estimate(ax, 2.35, 0, -.8, .8, FG, False)
    ax.text(2.43, 0, 'Not significant', fontsize=6, color='.45', va='center')


def repeat_from_bins(bins_table):
    """Repeat − switch gains from the bin-wise trial-wise fits on the final m7 fits (scheme tprime: target gain on
    target-repeat vs other trials; scheme dprime: distractor gain on repeat vs switch trials, within HP and within LP,
    averaged). Same fits as panel b."""
    b = pd.read_csv(bins_table, sep='\t')
    b = b[~b.with_global.astype(str).str.lower().eq('true')]
    w = b.pivot_table(index=['subject', 'roi', 'scheme', 'kind'], columns='bin', values='g').reset_index()
    t = w[(w.scheme == 'tprime') & (w.kind == 'T')].assign(d=lambda d: d['repeat'] - d['other'])
    dd = w[(w.scheme == 'dprime') & w.kind.isin(['D_HP', 'D_LP'])].assign(d=lambda d: d['repeat'] - d['switch'])
    dd = dd.groupby(['subject', 'roi'], as_index=False).d.mean()
    return t[['subject', 'roi', 'd']], dd


def main(repeat_table, bayes_table, name='fig7_behaviour', bins_table=None):
    global STAMP, LEGEND
    bayes = pd.read_csv(bayes_table, sep='\t')
    if bins_table:
        t_rep, d_rep = repeat_from_bins(bins_table)
        STAMP = ''
        LEGEND = LEGEND.replace('target gain on target-repeat minus target-switch trials (fits with the target gain split by repeat)',
                                'target gain on target-repeat minus other trials')
        LEGEND = LEGEND.replace('(fits with separate distractor gains on repeat and switch trials, the difference taken within HP and within LP and averaged',
                                '(the difference taken within HP and within LP and averaged')
        LEGEND = LEGEND.replace(' Both panels use the earlier attention-field fits (October 2026) and will be redone on the final fits.',
                                ' Both panels use the same fits: trial-wise and bin-wise gains estimated on top of each subject\'s final joint gain-field fit (m7, DoG pRFs, one HRF per subject × ROI).')
    else:
        rep = pd.read_csv(repeat_table, sep='\t')
        t_rep = rep[rep.model == 'targetRepeat'].assign(d=lambda d: d.g_T_dyn_repeat - d.g_T_dyn_switch)
        # The repeat model has separate distractor gains on repeat and on switch trials (g_HP_dyn, g_LP_dyn are the
        # switch gains); repeat − switch within HP and within LP trials, averaged, as for the response times (Fig. 1f).
        d_rep = rep[rep.model == 'repeat'].assign(d=lambda d: ((d.g_HP_dyn_repeat - d.g_HP_dyn) + (d.g_LP_dyn_repeat - d.g_LP_dyn)) / 2)
    results = {}
    W, x0 = DOUBLE, .75
    pw = W - .95
    gap = .27                      # heading to axes: room for the stars
    ah, ch, below = .9, 1.0, .56
    H = .26 + .2 + gap + ah + below + .2 + gap + ch + .4
    fig = plt.figure(figsize=(W, H))
    Y = lambda d: H - d            # inches from the top -> from the bottom
    sig_key(fig, Y(.1))
    fig.text(.995, Y(.1) / H, STAMP, fontsize=6, color=MUTED, ha='right', va='center')
    # a: history, repeat − switch (Fig. 1e, f); the location effect (Fig. 1c, d) is Fig. 5d.
    row(fig, Y(.36), 'a', [('History: less ', FG), ('target', TARGET), (' gain when its location repeats', FG)],
        [(icons('history', 'target'), TARGET, 'Target gain'), (icons('history', 'distractor'), DISTRACTOR, 'Distractor gain')],
        'As Fig. 1e, f')
    ax = ax_in(fig, x0, Y(.26 + .2 + gap + ah), pw, ah)
    results.update(panel_pair(ax, [('Target gain, repeat − switch', t_rep, 'd', TARGET),
                                   ('Distractor gain, repeat − switch', d_rep, 'd', DISTRACTOR)],
                              (-1.15, .55), 'Gain,\nrepeat − switch'))
    ax.set_yticks([-1, 0])
    # b: trial by trial, keyed by the gain fields themselves.
    yc = .26 + .2 + gap + ah + below
    row(fig, Y(yc + .1), 'b', [('Trial by trial: gain on a trial predicts its response time', FG)],
        [([lambda a: draw_source(a, 'target')], TARGET, 'Target gain'),
         ([lambda a: draw_source(a, 'distractor')], DISTRACTOR, 'Distractor gain'),
         ([lambda a: draw_source(a, 'global')], FG, 'Global gain: the sign flips')])
    ax = ax_in(fig, x0, Y(yc + .2 + gap + ch), pw, ch)
    panel_bayes(ax, bayes, [('T', TARGET), ('D_avg', DISTRACTOR), ('G', FG)], (-.07, .125), 'Log RT per\nunit gain (z)')
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
    ap.add_argument('--bayes-table', default='notes/data/trialwise_rt_bayes_summary_davg_2026-11.tsv')
    ap.add_argument('--bins-table', default=None,
                    help='Bin-wise trial-wise gains (analyze_trialwise_af_gains collect, *_bins.tsv) on the final fits; '
                         'replaces --repeat-table so both panels use the same fits')
    ap.add_argument('--name', default='fig7_behaviour')
    a = ap.parse_args()
    main(a.repeat_table, a.bayes_table, a.name, a.bins_table)
