"""Figure 7 (compact): the gain fields fluctuate with trial history and with response time.

a  History: the target gain on target-repeat trials and the distractor gain on distractor-repeat trials, as a %
   change of the area's typical gain.
b  Response time: the change in target, distractor and global gain per SD of log RT.

Both are BOLD-level modulators of the trial-wise gains (``fit_trialwise_af_gains``, parametric model, on the final DoG
fits) combined over subjects and areas by a hierarchical Bayesian model that weighs every subject × area estimate by
its leave-one-run-out jackknife SE (``trialwise_meta_bayes``). Large points: posterior mean and 95% HDI of each ROI
group's mean and of the mean over all nine areas; small hollow points: the nine areas' posterior means.

    python -m retsupp.visualize.paper.fig7_compact --meta notes/data/trialwise_meta_bayes_2026-11
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from retsupp.visualize.paper.fig7_behaviour import draw_source, icons
from retsupp.visualize.paper.paper_style import DISTRACTOR, DOUBLE, FG, TARGET, offset_spines, save

GROUPS = {'early': ['V1', 'V2', 'V3'], 'dorsal': ['V3AB', 'IPS'], 'occipitotemporal': ['hV4', 'VO', 'LO', 'TO']}   # ROI_ORDER
UNITS = ['all', 'early', 'dorsal', 'occipitotemporal']
UNIT_LABELS = {'early': 'Early\n(V1–V3)', 'dorsal': 'Dorsal\n(V3AB, IPS)',
               'occipitotemporal': 'Occipito-\ntemporal', 'all': 'All nine\nareas'}
UNIT_X = {'all': 0., 'early': 1.35, 'dorsal': 2.35, 'occipitotemporal': 3.35}
PALE = .4


def pd_stars(p):
    return '***' if p > .9995 else '**' if p > .995 else '*' if p > .975 else ''


def panel(ax, summary, series, scale, ylim, ylabel, width=.44):
    """series: (question, variant, colour). Per unit: the areas' posterior means (small hollow points) and the
    posterior mean and 95% HDI of the group mean / overall mean (solid when P(sign) > .975, pale otherwise)."""
    k = len(series)
    off = np.linspace(-width / 2, width / 2, k) if k > 1 else [0.]
    lo, hi = ylim
    rows = []
    for o, (q, v, col) in zip(off, series):
        s = summary[(summary.question == q) & (summary.variant == v)].set_index(['level', 'unit'])
        for u in UNITS:
            x0 = UNIT_X[u] + o
            areas = [a for aa in GROUPS.values() for a in aa] if u == 'all' else GROUPS[u]
            am = [s.loc[('area', a), 'mean'] * scale for a in areas if ('area', a) in s.index]
            r = s.loc[('overall', 'all')] if u == 'all' else s.loc[('group', u)]
            m, a, b = r['mean'] * scale, r.hdi_lo * scale, r.hdi_hi * scale
            assert lo <= a and b <= hi, (q, u, a, b, ylim)
            pdir = max(r.p_gt0, r.p_lt0)
            solid = pdir > .975          # the same criterion as the stars
            xm = x0
            big = u == 'all'             # the headline estimate: larger marks
            ax.plot([xm, xm], [a, b], color=col, lw=(2.0 if big else 1.3) if solid else 1.1, alpha=1 if solid else PALE,
                    solid_capstyle='butt', zorder=4)
            ax.plot(xm, m, 'o', ms=(6 if big else 4.4), color=col, mec='w', mew=.6,
                    alpha=1 if solid else PALE + .15, zorder=5)
            ax.text(xm, hi, pd_stars(pdir), ha='center', va='bottom', fontsize=8, color=col, clip_on=False)
            rows.append(dict(question=q, variant=v, unit=u, mean=m, hdi_lo=a, hdi_hi=b, p_gt0=r.p_gt0, p_lt0=r.p_lt0))
    ax.axhline(0, color='.8', lw=.6, zorder=0)
    ax.axvline((UNIT_X['all'] + UNIT_X['early']) / 2, color='.85', lw=.6, zorder=0)
    ax.set_xticks([UNIT_X[u] for u in UNITS])
    ax.set_xticklabels([UNIT_LABELS[u] for u in UNITS], fontsize=6.5)
    ax.tick_params(axis='x', length=0)
    ax.set_xlim(-.45, UNIT_X['occipitotemporal'] + .4)
    ax.set_ylim(lo, hi)
    ax.set_ylabel(ylabel)
    offset_spines(ax)
    ax.spines['bottom'].set_visible(False)
    return pd.DataFrame(rows)


def heading(fig, x, y, let, title, W, H):
    fig.text((x - .3) / W, y / H, let, fontsize=8, fontweight='bold', va='center', ha='left')
    for k, (seg, col) in enumerate(title):
        t = fig.text(x / W, y / H, seg, fontsize=7.5, color=col, va='center', ha='left')
        fig.canvas.draw()
        x = t.get_window_extent().x1 / fig.dpi


def key(fig, x, y, entries, W, H, icon=.27, gap=.14):
    """Icon(s) and a coloured label per series."""
    for draws, col, label in entries:
        for k, draw in enumerate(draws):
            if k:
                fig.text((x + .008) / W, y / H, '−', fontsize=7, color=FG, va='center', ha='left')
                x += .08
            draw(fig.add_axes([x / W, (y - icon / 2) / H, icon / W, icon / H]))
            x += icon
        t = fig.text((x + .03) / W, y / H, label, fontsize=6.5, color=col, va='center', ha='left')
        fig.canvas.draw()
        x = t.get_window_extent().x1 / fig.dpi + gap
    return x


def sig_key(fig, x, y, W, H):
    """The solid / pale encoding, drawn with the panels' own marks."""
    ax = fig.add_axes([x / W, (y - .07) / H, 2.6 / W, .14 / H])
    ax.set_xlim(0, 2.6)
    ax.set_ylim(-1, 1)
    ax.set_axis_off()
    ax.plot([.04, .04], [-.8, .8], color=FG, lw=1.5, solid_capstyle='butt')
    ax.plot(.04, 0, 'o', ms=4.4, color=FG, mec='w', mew=.6)
    ax.text(.12, 0, 'P(sign) > .975', fontsize=6, color=FG, va='center')
    ax.plot([1.18, 1.18], [-.8, .8], color=FG, lw=1.1, alpha=PALE, solid_capstyle='butt')
    ax.plot(1.18, 0, 'o', ms=4.4, color=FG, mec='w', mew=.6, alpha=PALE + .15)
    ax.text(1.26, 0, 'P(sign) ≤ .975', fontsize=6, color='.45', va='center')


LEGEND = """
**Figure 7. The gain fields fluctuate with trial history and with response time.** Trial-by-trial modulation of the gain fields, estimated on top of each subject's final joint gain-field fit (m7: DoG pRFs, one HRF per subject × area, each area's signal voxels). Per subject and area, the target, distractor and global gains of every trial were written as a gain plus modulators (z-scored log RT; target at the previous target's location, distractor at the previous distractor's location; target at the previous distractor's location), all centred within gain, and estimated directly on the BOLD time series of all the area's voxels (weighted least squares on the linear phasic term, other parameters fixed). The uncertainty of every subject × area estimate is its leave-one-run-out jackknife SE (12 runs; it agrees with the odd-versus-even-run difference, and includes noise shared across voxels, which a split over voxels misses by a factor of about six). Hierarchical Bayesian model per effect: estimate ~ Normal(θ, √(SE² + τ²)), θ = group mean + area deviation + subject deviation, the three group means (early V1–V3; dorsal V3AB, IPS; occipitotemporal hV4, LO, TO, VO) as fixed effects with Normal(0, 2·SD) priors, HalfNormal SDs (PyMC with nutpie, 4 × 2000 draws; {conv}). Points and bars: posterior mean and 95% HDI of the mean over all nine areas (left, larger) and of each group's mean; solid when the posterior probability of the sign shown exceeds .975, pale otherwise. Per-area estimates are in the table below. Stars give the posterior probability of the sign shown: * > .975, ** > .995, *** > .9995. n = 30 subjects (247 subject × area estimates; pairs with fewer than 10 signal voxels are left out). a, History: the gain at the target on target-repeat trials and at the distractor on distractor-repeat trials, relative to the area's typical gain (the median over subjects of |g_T| and of |(g_HP + g_LP)/2| in the m7 fits; the median of the absolute value because the signed median is close to 0 in some areas), × 100%. Repeating the target's location lowers the target gain by {T_all} on average (posterior mean and 95% HDI; P = {T_all_p}); per group early {T_early}, dorsal {T_dorsal}, occipitotemporal {T_ot}. Repeating the distractor's location does not change the distractor gain credibly ({D_all}, P = {D_all_p}). b, Response time: change in each gain per SD of log RT (z within run and trial type; positive: higher gain on slower trials). {rt_sentence} Robustness (table below): a Student-t likelihood gives the same conclusions; without the SE weighting (all subject × area estimates equally weighted) {T_nose_phrase} but the RT effects of panel b do not (distractor P = {Drt_nose_p}, global P = {Grt_nose_p}); in absolute gain units the target-repeat effect is {T_abs} (P = {T_abs_p}). Check with the gains pooled over each group's areas in one fit (absolute units, random-effects meta-analysis over subjects with the jackknife SEs, mean ± SE): target repeat {chk_T}; distractor repeat {chk_D}; distractor gain per SD log RT {chk_Drt}; global gain per SD log RT {chk_Grt}.
"""


def fmt(r, scale=1., unit=''):
    d = 3 if scale == 1 else 0

    def f(x):
        x = round(x * scale, d) + 0.0      # no "−0"
        return f'{x:+.{d}f}'
    return f'{f(r["mean"])}{unit} [{f(r.hdi_lo)}, {f(r.hdi_hi)}]'.replace('-', '−')


def pooled_check(param):
    """Check: the estimates of the gains pooled over each group's areas in one fit, combined over subjects by a
    random-effects meta-analysis (DerSimonian-Laird) with the jackknife SEs. Returns text per effect."""
    from scipy import stats
    p = pd.read_csv(param, sep='\t')
    out = {}
    for key, (wg, kinds, mod) in {'T_target_repeat': (False, ['T'], 'target_repeat'),
                                  'D_dist_repeat': (False, ['D_HP', 'D_LP'], 'dist_repeat'),
                                  'D_rt': (True, ['D_HP', 'D_LP'], 'rt'), 'G_rt': (True, ['G'], 'rt')}.items():
        d = p[(p.with_global.astype(str).str.lower() == str(wg).lower()) & p.kind.isin(kinds) & (p.modulator == mod)]
        t = d.groupby(['subject', 'roi']).agg(y=('g', 'mean'), se=('se_jk', lambda x: np.sqrt((x ** 2).sum()) / len(x)),
                                              n=('g', 'size')).reset_index()
        t = t[t.n == len(kinds)]
        parts = []
        for g in GROUPS:
            x = t[t.roi == g]
            w = 1 / x.se ** 2
            m = (w * x.y).sum() / w.sum()
            q = (w * (x.y - m) ** 2).sum()
            t2 = max(0., (q - (len(x) - 1)) / (w.sum() - (w ** 2).sum() / w.sum()))
            ws = 1 / (x.se ** 2 + t2)
            mm, se = (ws * x.y).sum() / ws.sum(), np.sqrt(1 / ws.sum())
            pv = 2 * stats.norm.sf(abs(mm / se))
            parts.append(f'{g} {mm:+.3f} ± {se:.3f} (p = {pv:.3f})'.replace('-', '−'))
        out[key] = '; '.join(parts)
    return out


FIT_LABEL = "each subject's final joint gain-field fit (m7: DoG pRFs, one HRF per subject × area, each area's signal voxels)"


def main(meta, name, param, compact=False, fit_label=FIT_LABEL):
    """compact: narrower panels (5.9 in wide), shorter titles and keys; same content."""
    meta = Path(meta)
    summary = pd.read_csv(meta / 'summary.tsv', sep='\t')
    chk = pooled_check(param)
    diag = pd.read_csv(meta / 'diagnostics.tsv', sep='\t')
    if compact:
        pw, ph = 2.2, 1.45
        xa, xb = .72, .72 + pw + .72
        W = xb + pw + .16
    else:
        W = DOUBLE
        pw, ph = 2.75, 1.45            # axes size (inches); wider than high
        xa, xb = .7, .7 + pw + .6
    ik = .24 if compact else .27
    y0 = .42                      # axes bottom (room for the unit labels)
    yk = y0 + ph + .3             # key row (room for the stars between it and the axes)
    yh = yk + .28                 # heading row
    ys = yh + .22                 # solid / pale key
    H = ys + .1
    fig = plt.figure(figsize=(W, H))
    # a
    heading(fig, xa, yh, 'a', [('Repeating the ', FG), ('target', TARGET), ('’s location lowers its gain', FG)], W, H)
    key(fig, xa, yk, [(icons('history', 'target'), TARGET, 'Target' if compact else 'Target gain'),
                      (icons('history', 'distractor'), DISTRACTOR, 'Distractor' if compact else 'Distractor gain')], W, H,
        icon=ik)
    ax = fig.add_axes([xa / W, y0 / H, pw / W, ph / H])
    sa = summary[summary.question.isin(['T_target_repeat', 'D_dist_repeat']) & (summary.variant == 'rel')
                 & summary.level.isin(['group', 'overall'])]
    ylo = min(-105., np.floor(sa.hdi_lo.min() * 100 / 10) * 10 - 5)
    yhi = max(70., np.ceil(sa.hdi_hi.max() * 100 / 10) * 10 + 5)
    ra = panel(ax, summary, [('T_target_repeat', 'rel', TARGET), ('D_dist_repeat', 'rel', DISTRACTOR)], 100,
               (ylo, yhi), 'Gain on repeat trials\n(% change)')
    ax.set_yticks([-100, -50, 0, 50])
    # b
    ov = summary[(summary.level == 'overall') & (summary.variant == 'abs')].set_index('question')
    credible = {q: max(ov.loc[q, 'p_gt0'], ov.loc[q, 'p_lt0']) > .975 for q in ('D_rt', 'G_rt')}
    if credible['G_rt']:
        title_b = ([('Lower ', FG), ('distractor', DISTRACTOR), (' and global gains on slower trials', FG)] if compact else
                   [('On slower trials the ', FG), ('distractor', DISTRACTOR), (' and global gains are lower', FG)])
    elif credible['D_rt']:
        title_b = ([('Lower ', FG), ('distractor', DISTRACTOR), (' gain on slower trials', FG)] if compact else
                   [('On slower trials the ', FG), ('distractor', DISTRACTOR), (' gain is lower', FG)])
    else:   # no credible RT effect over all nine areas
        title_b = [('Gains on slower trials', FG)]
    heading(fig, xb, yh, 'b', title_b, W, H)
    key(fig, xb, yk, [([lambda a: draw_source(a, 'target')], TARGET, 'Target'),
                      ([lambda a: draw_source(a, 'distractor')], DISTRACTOR, 'Distractor'),
                      ([lambda a: draw_source(a, 'global')], FG, 'Global' if compact else 'Global gain')], W, H, icon=ik)
    axb = fig.add_axes([xb / W, y0 / H, pw / W, ph / H])
    rb = panel(axb, summary, [('T_rt', 'abs', TARGET), ('D_rt', 'abs', DISTRACTOR), ('G_rt', 'abs', FG)], 1,
               (-.085, .065), 'Gain change per SD\nslower response', width=.5)
    axb.set_yticks([-.08, -.04, 0, .04])
    axb.text(UNIT_X['occipitotemporal'] + .38, .06, 'Higher on\nslow trials', ha='right', va='top', fontsize=6.5, color='.45')
    ax.text(UNIT_X['occipitotemporal'] + .38, 66, 'Higher on\nrepeats', ha='right', va='top', fontsize=6.5, color='.45')
    sig_key(fig, xa - .3, ys, W, H)
    for a in (ax, axb):
        bb = a.get_window_extent()
        assert bb.height <= bb.width

    # legend numbers
    s = summary.set_index(['question', 'variant', 'level', 'unit'])

    def g(q, v, u='all'):
        return s.loc[(q, v, 'overall' if u == 'all' else 'group', u)]

    def pdir(r):
        return f'{max(r.p_gt0, r.p_lt0):.3f}'.lstrip('0')
    conv = (f'{int(diag.divergences.sum())} divergences over all {len(diag)} fits, max R̂ {diag.rhat_max.max():.2f}, '
            f'min bulk ESS {int(diag.ess_bulk_min.min())}; 95% posterior predictive intervals cover '
            f'{diag.ppc_obs_coverage95.min() * 100:.0f}–{diag.ppc_obs_coverage95.max() * 100:.0f}% of the estimates and every area mean')
    gp = max(g('G_rt', 'abs').p_gt0, g('G_rt', 'abs').p_lt0)
    if gp > .975:
        rt_sentence = ('On slower trials the distractor gain ({Drt_all}, P = {Drt_all_p}) and the global gain over the whole '
                       'display ({Grt_all}, P = {Grt_all_p}) are lower; the target gain does not change credibly '
                       '({Trt_all}, P = {Trt_all_p}).')
    elif max(g('D_rt', 'abs').p_gt0, g('D_rt', 'abs').p_lt0) <= .975:
        rt_sentence = ('Over all nine areas no gain changes credibly with response time: distractor gain {Drt_all} '
                       '(P = {Drt_all_p}), global gain over the whole display {Grt_all} (P = {Grt_all_p}), target gain '
                       '{Trt_all} (P = {Trt_all_p}); group estimates in the table below.')
    else:
        rt_sentence = ('On slower trials the distractor gain is lower ({Drt_all}, P = {Drt_all_p}); the global gain over the '
                       'whole display points the same way but not credibly ({Grt_all}, P = {Grt_all_p}), and the target gain '
                       'does not change credibly ({Trt_all}, P = {Trt_all_p}).')
    n_sub, n_obs = int(diag.n_subjects.max()), int(diag.n_obs.max())
    tn = g('T_target_repeat', 'rel_nose')
    tn_p = max(tn.p_gt0, tn.p_lt0)
    T_nose_phrase = ('the target-repeat effect remains (P = {T_nose_p})' if tn_p > .975 else
                     'the target-repeat effect keeps its direction but falls just short of the threshold (P = {T_nose_p}), and'
                     if tn_p > .95 else
                     'the target-repeat effect keeps its direction but is not credible (P = {T_nose_p}), and')
    legend = LEGEND.replace('{T_nose_phrase} but the RT', '{T_nose_phrase} the RT' if T_nose_phrase.endswith(', and') else '{T_nose_phrase} but the RT').replace('{rt_sentence}', rt_sentence).replace('{T_nose_phrase}', T_nose_phrase).replace(
        'n = 30 subjects (247 subject × area estimates', f'n = {n_sub} subjects ({n_obs} subject × area estimates').format(
        conv=conv,
        T_all=fmt(g('T_target_repeat', 'rel'), 100, '%'), T_all_p=pdir(g('T_target_repeat', 'rel')),
        T_early=fmt(g('T_target_repeat', 'rel', 'early'), 100, '%'),
        T_dorsal=fmt(g('T_target_repeat', 'rel', 'dorsal'), 100, '%'),
        T_ot=fmt(g('T_target_repeat', 'rel', 'occipitotemporal'), 100, '%'),
        D_all=fmt(g('D_dist_repeat', 'rel'), 100, '%'), D_all_p=pdir(g('D_dist_repeat', 'rel')),
        Drt_all=fmt(g('D_rt', 'abs')), Drt_all_p=pdir(g('D_rt', 'abs')),
        Grt_all=fmt(g('G_rt', 'abs')), Grt_all_p=pdir(g('G_rt', 'abs')),
        Trt_all=fmt(g('T_rt', 'abs')), Trt_all_p=pdir(g('T_rt', 'abs')),
        T_nose_p=pdir(g('T_target_repeat', 'rel_nose')), Drt_nose_p=pdir(g('D_rt', 'abs_nose')),
        Grt_nose_p=pdir(g('G_rt', 'abs_nose')),
        T_abs=fmt(g('T_target_repeat', 'abs')), T_abs_p=pdir(g('T_target_repeat', 'abs')),
        chk_T=chk['T_target_repeat'], chk_D=chk['D_dist_repeat'], chk_Drt=chk['D_rt'], chk_Grt=chk['G_rt'])
    tab = ['', '**Posterior summaries** (mean [95% HDI], P of the sign shown; group means and the mean over the nine '
           'areas; a: % of the typical gain, b: gain per SD of log RT; variants: primary, Student-t likelihood, no SE weighting).',
           '', '| Effect | Unit | Primary | Student-t | No SE weighting |', '|---|---|---|---|---|']
    names = {'T_target_repeat': ('Target gain, target repeat', 'rel', 100, '%'),
             'D_dist_repeat': ('Distractor gain, distractor repeat', 'rel', 100, '%'),
             'T_rt': ('Target gain per SD log RT', 'abs', 1, ''), 'D_rt': ('Distractor gain per SD log RT', 'abs', 1, ''),
             'G_rt': ('Global gain per SD log RT', 'abs', 1, '')}
    for q, (lab, v, sc, un) in names.items():
        for u in UNITS:
            cells = []
            for vv in (v, f'{v}_t', f'{v}_nose'):
                r = g(q, vv, u)
                cells.append(f'{fmt(r, sc, un)}, P = {pdir(r)}')
            ul = UNIT_LABELS[u].replace('-\n', '').replace('\n', ' ')
            tab.append(f'| {lab} | {ul} | ' + ' | '.join(cells) + ' |')
    tab += ['', '**Per-area posterior means** (primary model; mean [95% HDI], P of the sign shown).', '',
            '| Effect | ' + ' | '.join(a for aa in GROUPS.values() for a in aa) + ' |', '|---|' + '---|' * 9]
    for q, (lab, v, sc, un) in names.items():
        cells = []
        for ar in [a for aa in GROUPS.values() for a in aa]:
            r = s.loc[(q, v, 'area', ar)]
            cells.append(f'{fmt(r, sc, un)}, P = {pdir(r)}')
        tab.append(f'| {lab} | ' + ' | '.join(cells) + ' |')
    legend = legend.replace(FIT_LABEL, fit_label)
    if 'own voxels' in fit_label:
        legend = legend.replace('fewer than 10 signal voxels', 'fewer than 10 voxels')
    save(fig, name, legend.strip() + '\n' + '\n'.join(tab))
    print(pd.concat([ra, rb]).round(4).to_string(index=False))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--meta', default='notes/data/trialwise_meta_bayes_2026-11')
    ap.add_argument('--name', default='fig7_compact')
    ap.add_argument('--param', default='notes/data/trialwise_signal_se_2026-11/trialwise_af_signal_se_2026-11_parametric.tsv',
                    help='Collected parametric table (for the pooled-group check in the legend)')
    ap.add_argument('--compact', action='store_true', help='narrower panels, shorter titles and keys (same content)')
    ap.add_argument('--fit-label', default=FIT_LABEL, help='how the legend names the base fits')
    a = ap.parse_args()
    main(a.meta, a.name, a.param, a.compact, a.fit_label)
