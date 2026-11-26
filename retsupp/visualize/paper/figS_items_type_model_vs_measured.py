"""Supplementary figure: the item-type contrasts of Fig. 3b, c, measured and predicted by the main model.

Input: ``items_model_vs_measured.py --part items`` on the zero-sum main model
(``--items-variant S``: ``notes/data/items_type_model_vs_measured_n28_zs_lm_main_2026-11-26.tsv``, the main model as
fitted, not the variant with dD): per pathway and for V1, V2, V3, the target −
neutral, distractor − neutral and target − distractor contrasts (% signal, window 3.2 to 6.4 s), measured, predicted by
each participant's joint zero-sum fit (noise-free prediction through the same voxels and deconvolution), and measured −
predicted; median over participants with bootstrapped 95% CI, Wilcoxon signed-rank p and BH q.

    PYTHONPATH=. python -m retsupp.visualize.paper.figS_items_type_model_vs_measured
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from retsupp.visualize.paper.paper_style import DISTRACTOR, DOUBLE, FG, TARGET, letter, offset_spines, save

CONTRASTS = [('target − neutral', 'Target − neutral', TARGET), ('distractor − neutral', 'Distractor − neutral', DISTRACTOR),
             ('target − distractor', 'Target − distractor', FG)]
UNITS = [('Early', 'Early'), ('Dorsal', 'Dorsal'), ('Occipitotemporal', 'Occ.-\ntemp.'), ('V1', 'V1'), ('V2', 'V2'),
         ('V3', 'V3')]
XS = [0, 1, 2, 3.4, 4.4, 5.4]


def stars(q):
    return '***' if q < .001 else '**' if q < .01 else '*' if q < .05 else ''


def fmt(r):
    return (f"{r['median']:+.3f} [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}]".replace('-', '−'))


def p_txt(p):
    return f'p = {p:.2g}' if p >= .001 else f'p = {p:.1e}'.replace('e-0', ' × 10⁻').replace('e-', ' × 10⁻')


def main(tsv, name):
    d = pd.read_csv(tsv, sep='\t')
    W, H = DOUBLE, 3.6
    fig = plt.figure(figsize=(W, H))
    at = lambda x, y, w, h: fig.add_axes([x / W, y / H, w / W, h / H])
    pw, gap, x0 = 1.9, .3, .62
    leg_parts = []
    for j, (c, lab, col) in enumerate(CONTRASTS):
        top = at(x0 + j * (pw + gap), 1.95, pw, 1.2)
        bot = at(x0 + j * (pw + gap), .42, pw, 1.0)
        for x, (u, _) in zip(XS, UNITS):
            g = d[(d.contrast == c) & (d.grp == u)].set_index('source')
            m, p, diff = g.loc['measured'], g.loc['joint fit predicted'], g.loc['measured − joint fit']
            top.errorbar(x - .14, m['median'], yerr=[[m['median'] - m.ci_lo], [m.ci_hi - m['median']]], fmt='o', ms=4,
                         color=col, mec='w', mew=.5, lw=1.1)
            top.errorbar(x + .14, p['median'], yerr=[[p['median'] - p.ci_lo], [p.ci_hi - p['median']]], fmt='o', ms=4,
                         color=col, mfc='w', mew=1, lw=1.1, alpha=.85)
            bot.errorbar(x, diff['median'], yerr=[[diff['median'] - diff.ci_lo], [diff.ci_hi - diff['median']]], fmt='D',
                         ms=3.5, color=col, mec='w', mew=.4, lw=1.1)
            bot.text(x, .245, stars(diff.q), ha='center', va='bottom', fontsize=7, color=col)
            leg_parts.append((lab, u, m, p, diff))
        for ax, ylim, yl in ((top, (-.1, .36), 'Contrast (% signal)'), (bot, (-.1, .25), 'Measured −\npredicted')):
            ax.axhline(0, color='.6', lw=.6, ls='--', zorder=0)
            ax.axvline(2.7, color='.85', lw=.6, zorder=0)
            ax.set_xlim(-.5, 5.9)
            ax.set_ylim(*ylim)
            ax.set_xticks(XS)
            ax.set_xticklabels([u[1] for u in UNITS] if ax is bot else [], fontsize=6.5)
            if j == 0:
                ax.set_ylabel(yl)
            else:
                ax.set_yticklabels([])
            offset_spines(ax)
        top.text(0, 1.1, lab, transform=top.transAxes, fontsize=7.5, color=col, va='bottom')
        if j == 2:
            top.plot([.05], [.93], 'o', ms=4, color=FG, mec='w', mew=.5, transform=top.transAxes, clip_on=False)
            top.plot([.05], [.81], 'o', ms=4, color=FG, mfc='w', mew=1, transform=top.transAxes, clip_on=False)
            top.text(.1, .93, 'Measured', transform=top.transAxes, fontsize=6.5, color=FG, va='center')
            top.text(.1, .81, 'Predicted (main model)', transform=top.transAxes, fontsize=6.5, color=FG, va='center')
        if j == 0:
            letter(top, 'a', x=-.26, y=1.1)
            letter(bot, 'b', x=-.26, y=1.08)

    def line(u):
        out = []
        for lab, uu, m, p, diff in leg_parts:
            if uu == u:
                out.append(f"{lab.lower()} measured {fmt(m)}, predicted {fmt(p)}, measured − predicted {fmt(diff)} "
                           f"({int(diff.pct_pos)}% of participants > 0, {p_txt(diff.p)}, q = {diff.q:.2g})")
        return f"{u}: " + '; '.join(out)
    legend = (
        "**Supplementary Figure. The item-type contrasts of Fig. 3b, c, measured and predicted by the main model: the model "
        "predicts the target − distractor difference but smaller differences from the neutral item than measured.** "
        "Voxels and deconvolution as in Fig. 3 on the main model's voxels (pRF centre within 2° of an item location; "
        "BOLD minus the bar-only prediction, nideconv, Fourier basis; window 3.2 to 6.4 s after search onset; pre-onset "
        "mean subtracted; areas pooled weighted by voxel count). Contrasts: target − neutral (HP and LP averaged), "
        "distractor − neutral and target − distractor (both at LP locations), in % signal. Predicted: the noise-free "
        "prediction of each participant's joint zero-sum gain-field fit (the main model as fitted), passed through the same voxels and "
        "deconvolution. a, Measured (filled) and predicted (open), median over participants and bootstrapped 95% CI. "
        "b, Measured minus predicted per participant; stars, Wilcoxon signed-rank test, Benjamini-Hochberg q < .05 (*), "
        "< .01 (**), < .001 (***). n = 28 (dorsal 26). Values: " + '. '.join(line(u) for u, _ in UNITS) + '.')
    save(fig, name, legend)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tsv', default='notes/data/items_type_model_vs_measured_n28_zs_lm_main_2026-11-26.tsv')
    ap.add_argument('--name', default='figS_items_type_model_vs_measured_zs_n28')
    a = ap.parse_args()
    main(a.tsv, a.name)
