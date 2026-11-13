"""Supplementary figure: response time by location relative to HP, over all blocks and in session 1 block 1 alone.

Rows: all blocks; session 1 block 1 (before any location had been HP). Per row: the distractor at HP, next to HP,
opposite HP or absent (raw response times, then the within-participant differences) and the target at HP, next to HP
or opposite HP on distractor-present trials (same). Correct trials, n = 28 (sub-01, sub-02 left out by
``richter_followup.load_trials``). Per-participant condition means and the contrasts are those of
``discussion_checks.behaviour`` (notes/data/discussion_behaviour_*_n28_2026-11-12.tsv); the medians, bootstrapped 95%
CIs and Wilcoxon p come from ``richter_followup.describe`` so they match that table exactly.

    python -m retsupp.visualize.paper.figS_rt_by_location
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from retsupp.visualize.paper import richter_followup as rf
from retsupp.visualize.paper.discussion_checks import CONTRASTS, TRIALS
from retsupp.visualize.paper.fig1_paradigm_behaviour import cond_icons, rt_panel, stars
from retsupp.visualize.paper.paper_style import (DISTRACTOR, DOUBLE, FG, HP, LP_NEXT, LP_OPP, MUTED, TARGET, letter,
                                                 offset_spines, save)

NAME = 'figS_rt_by_location_n28'
RT_LIM = (680, 1160)
DIFF_LIM = (-140, 140)
LABELS = {'hp': 'HP', 'next': 'Next', 'opposite': 'Opposite', 'absent': 'Absent'}
LOC = {'hp': 7, 'next': 1, 'opposite': 3}        # icon positions: HP is location 7 (ringed), next 1, opposite 3
CLABEL = {('next', 'opposite'): 'Next\n− opp.', ('hp', 'next'): 'HP\n− next', ('hp', 'opposite'): 'HP\n− opp.',
          ('hp', 'absent'): 'HP\n− absent'}
ANOVA = 'notes/data/discussion_behaviour_anova_n28_2026-11-12.tsv'
COUNTS = 'notes/data/discussion_behaviour_trial_counts_n28_2026-11-12.tsv'


def cell_means(t, item):
    d = t if item == 'distractor' else t[t.distractor_location != 10]
    col = 'dist_rel' if item == 'distractor' else 'target_rel'
    return d[d.ok].groupby(['subject', col]).rt_ms.mean().unstack()


def strips(ax, w, pairs, color):
    """Vertical strips of within-participant differences: dots, the median (diamond) and its bootstrapped 95% CI
    (bar), the median and Wilcoxon stars above, the count of participants on the median's side beside it."""
    lo_l, hi_l = DIFF_LIM
    ax.axhline(0, color='.75', lw=.6, zorder=0)
    out = []
    for x, (a, b) in enumerate(pairs):
        v = (w[a] - w[b]).dropna()
        r = rf.describe(v, f'{a} − {b}')
        vv = v.to_numpy()
        xx = x - .14 + np.random.default_rng(x).uniform(-.09, .09, len(vv))
        inside = (vv >= lo_l) & (vv <= hi_l)
        ax.scatter(xx[inside], vv[inside], s=6, color=color, alpha=.4, lw=0, zorder=2)
        for sel, edge, mk in ((vv > hi_l, hi_l, '^'), (vv < lo_l, lo_l, 'v')):
            ax.scatter(xx[sel], np.full(sel.sum(), edge), s=9, marker=mk, color=color, alpha=.6, lw=0, zorder=2,
                       clip_on=False)
        ax.plot([x + .06] * 2, [r['ci_lo'], r['ci_hi']], color=FG, lw=1.3, solid_capstyle='butt', zorder=3)
        ax.plot(x + .06, r['median'], 'D', ms=3.8, color=color, mec=FG, mew=.7, zorder=4)
        n_side = int((np.sign(vv) == np.sign(r['median'])).sum())
        ax.text(x + .14, r['median'], f'{n_side}', fontsize=6, color=FG, ha='left', va='center')
        ax.text(x, hi_l + .05 * (hi_l - lo_l), f"{r['median']:+.0f}\n{stars(r['p'])}".replace('-', '−'), fontsize=6,
                color=FG, ha='center', va='bottom', clip_on=False, linespacing=1.0)
        r['n_side'] = n_side
        out.append(r)
    ax.set_xlim(-.55, len(pairs) - .4)
    ax.set_ylim(lo_l, hi_l)
    ax.set_xticks(range(len(pairs)))
    ax.set_xticklabels([CLABEL[p] for p in pairs], fontsize=6)
    ax.tick_params(axis='x', length=0, pad=2)
    ax.set_yticks([-100, 0, 100])
    offset_spines(ax)
    ax.spines['bottom'].set_visible(False)
    return out


def main():
    t = rf.add_history(rf.load_trials(TRIALS))
    t['sb'] = 's' + t.session.astype(str) + 'b' + t.block.astype(str)
    assert t.subject.nunique() == 28 and t.subject.min() == 3
    W, H = DOUBLE, 5.6
    fig = plt.figure(figsize=(W, H))
    # Column layout (inches): raw distractor, distractor differences, raw target, target differences.
    cols = [(.5, 1.4), (2.38, 1.45), (4.27, 1.1), (5.85, 1.1)]
    rows = [('All blocks', t, 3.65), ('Session 1, block 1: before any location had been HP', t[t.sb == 's1b1'], .85)]
    results = []
    for r_i, (rlab, tt, y0) in enumerate(rows):
        h = 1.1
        fig.text(.25 / W, (y0 + h + .5) / H, rlab, fontsize=7.5, color=FG, va='bottom', ha='left')
        for item, (cx, cw), (dx, dw), color in (('distractor', cols[0], cols[1], DISTRACTOR),
                                                ('target', cols[2], cols[3], TARGET)):
            w = cell_means(tt, item)
            conds = ['hp', 'next', 'opposite'] + (['absent'] if item == 'distractor' else [])
            long = (w[conds].rename(columns=LABELS).stack().rename('rt').reset_index()
                    .rename(columns={'dist_rel': 'condition', 'target_rel': 'condition'}))
            long = long.rename(columns={long.columns[1]: 'condition'})
            ax = fig.add_axes([cx / W, y0 / H, cw / W, h / H])
            rt_panel(ax, long, [LABELS[c] for c in conds], [HP, LP_NEXT, LP_OPP, MUTED][:len(conds)], [], RT_LIM)
            cond_icons(ax, [[(LOC[c], item, True)] if c in LOC else [] for c in conds], y0=-.5, h=.4)
            ax.tick_params(axis='x', pad=36, labelsize=6.5)
            ax.set_yticks([800, 900, 1000, 1100])
            if item == 'distractor':
                ax.set_ylabel('Response time (ms)', fontsize=7)
            else:
                ax.set_yticklabels([])
            ax.set_xlabel('Distractor at' if item == 'distractor' else 'Target at (distractor present)', fontsize=7,
                          labelpad=2)
            bb = ax.get_window_extent()
            assert bb.height <= bb.width * 1.02
            if r_i == 0:
                letter(ax, 'a' if item == 'distractor' else 'b', x=-.25, y=1.45)
            else:
                letter(ax, 'c' if item == 'distractor' else 'd', x=-.25, y=1.45)
            ax_d = fig.add_axes([dx / W, y0 / H, dw / W, h / H])
            res = strips(ax_d, w, CONTRASTS[item], color)
            ax_d.set_ylabel('Difference (ms)', fontsize=6.5, labelpad=1)
            bb = ax_d.get_window_extent()
            assert bb.height <= bb.width * 1.02
            for r in res:
                r.update(item=item, subset=rlab)
            results += res
    # Key for the marks.
    for x0, (mk, kw, txt) in zip((3.6, 4.55, 5.9), (('D', dict(mfc=MUTED, mec=FG, ms=3.8, mew=.7), 'Median'),
                                                 ('|', dict(color=FG, ms=7, mew=1.3), 'Bootstrapped 95% CI'),
                                                 ('^', dict(color=MUTED, ms=3.5, mew=0), 'Beyond the axis'))):
        fig.lines.append(plt.Line2D([x0 / W], [(H - .1) / H], marker=mk, ls='', transform=fig.transFigure, **kw))
        fig.text((x0 + .09) / W, (H - .1) / H, txt, fontsize=6, color=FG, va='center')

    res = pd.DataFrame(results)
    chk = pd.read_csv('notes/data/discussion_behaviour_by_block_n28_2026-11-12.tsv', sep='\t')
    for _, r in res.iterrows():
        sub = 'all blocks' if r.subset.startswith('All') else 's1b1'
        c = chk[(chk.item == r['item']) & (chk.label == r['label']) & (chk.subset == sub)].iloc[0]
        assert np.isclose(c['median'], r['median'], atol=1e-2) and np.isclose(c.p, r.p, rtol=1e-3), (r, c)

    an = pd.read_csv(ANOVA, sep='\t')
    cnt = pd.read_csv(COUNTS, sep='\t')
    ia = {i: an[(an.item == i) & an.effect.str.startswith('sb:')].iloc[0] for i in ('distractor', 'target')}
    s1 = cnt[cnt.sb == 's1b1']
    aov_txt = '; '.join(f"{i} F({int(a['Num DF'])}, {int(a['Den DF'])}) = {a['F Value']:.2f}, p = "
                        + f"{a['Pr > F']:.2f}".lstrip('0') for i, a in ia.items())
    ctxt = '; '.join(f"{i} {LABELS[c].lower() if c != 'hp' else 'HP'} {int(m)} ({int(lo)}–{int(hi)})"
                     for i in ('distractor', 'target') for c, m, lo, hi in
                     s1[s1['item'] == i][['cell', 'median', 'min', 'max']].itertuples(index=False))

    def fmt(r):
        pv = f"{r['p']:.2g}"
        pv = pv.lstrip('0') if 'e' not in pv else pv.replace('e-0', ' × 10⁻').replace('e-', ' × 10⁻')
        return (f"{r['median']:+.1f} ms [{r['ci_lo']:+.1f}, {r['ci_hi']:+.1f}], {r['n_side']}/{r['n']}, "
                f"p = {pv}").replace('-', '−')

    lines = []
    for subset in res.subset.unique():
        for item in ('distractor', 'target'):
            rr = res[(res.subset == subset) & (res['item'] == item)]
            short = 'All blocks' if subset.startswith('All') else 'Session 1 block 1'
            lines.append(f"{short}, {item}: " + '; '.join(
                f"{r['label'].replace('hp', 'HP')} {fmt(r)}" for _, r in rr.iterrows()))
    legend = (
        '**Supplementary Figure figS_rt_by_location_n28. Response time by the item\'s location relative to the HP location, '
        'over all blocks and before any location had been HP.** a, b, all blocks; c, d, session 1, block 1 alone, where no '
        'location had yet been the HP location, so no response can reflect an earlier HP location. a, c, response time with '
        'the distractor at the HP location, at a location next to it, at the location opposite it, or absent. b, d, response '
        'time with the target at the HP location, next to it or opposite it, on trials with a distractor. Icons: the four '
        'possible locations, HP ringed in purple, and the item. Left of each pair, response times: dots are participants (mean '
        'over correct trials), large points and bars the mean ± within-participant SEM (Cousineau-Morey). Right of each pair, '
        'within-participant differences: dots are participants (triangles beyond the axis, drawn at its edge), diamonds the '
        'median and bars its bootstrapped 95% CI (10 000 resamples); above each strip the median (ms) and the Wilcoxon '
        'signed-rank test (* p < .05, ** p < .01, *** p < .001, n.s. not significant), beside it the number of participants '
        'on the side of the median. The difference axis (±140 ms) is wider than in Fig. 1 to hold session 1 block 1. '
        f'n = 28, correct trials (the two pilot participants are left out). Trials per participant in session 1 block 1 '
        f'(median, range): {ctxt}. ' + ' '.join(l + '.' for l in lines) + ' '
        'There is no evidence that the pattern depends on the history of the locations: relation to HP × session-block (four blocks, '
        'repeated-measures ANOVA on participant means, HP, next and opposite): ' + aov_txt + '. Source: '
        '`notes/data/discussion_behaviour_by_block_n28_2026-11-12.tsv`, `..._anova_n28_2026-11-12.tsv`, '
        '`..._trial_counts_n28_2026-11-12.tsv` (`retsupp/visualize/paper/discussion_checks.py`).')
    save(fig, NAME, legend)
    print(res[['subset', 'item', 'label', 'median', 'ci_lo', 'ci_hi', 'n_side', 'p']].round(3).to_string(index=False))


if __name__ == '__main__':
    main()
