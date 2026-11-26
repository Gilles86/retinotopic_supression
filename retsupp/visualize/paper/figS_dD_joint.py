"""Supplementary figure: the distractor-specific HP term (dD) in the joint zero-sum fit, and held-out prediction.

a  dD = distractor gain at HP minus at LP (percentage points), per participant and area (jackknife SE, ``dD_meta_zs``),
   through the Student-t hierarchical model (posterior mean, 95% HDI; all areas and pathways left of the divider).
b  Held-out prediction (leave one run out): change in held-out R² (× 10⁻⁴) when dD is added to the main model,
   per participant and area; all areas = per participant the median over its areas (Wilcoxon; areas BH over nine).

    PYTHONPATH=. ~/mambaforge/envs/bauer/bin/python -m retsupp.visualize.paper.figS_dD_joint
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import retsupp.visualize.paper.fig5_gain_fields as F
from retsupp.visualize.paper.fig45_common import median_ci
from retsupp.visualize.paper.fig5_gain_fields_v6 import pooled_bayes
from retsupp.visualize.paper.paper_style import DISTRACTOR, DOUBLE, FG, ROI_ORDER, save


def main(a):
    est = pd.read_csv(f'{a.dd_dir}/estimates.tsv', sep='\t')
    post = pd.read_csv(f'{a.dd_dir}/bayes.tsv', sep='\t')
    ar = post[post.level == 'area'].set_index('unit').loc[ROI_ORDER]
    sm = pd.DataFrame(dict(median=ar['mean'], ci_low=ar.hdi_lo, ci_high=ar.hdi_hi,
                           q=2 * (1 - np.maximum(ar.p_gt0, ar.p_lt0))))
    B = dict(post=post)
    h = pd.read_csv(a.heldout, sep='\t')
    h = h[~h.subject.isin([1, 2]) & h.roi.isin(ROI_ORDER)].copy()
    h['d_dD'] = (h.dS_dD - h.dS) * 1e4
    pooled = h.groupby('subject').d_dD.median()
    m, lo, hi = median_ci(pooled.to_numpy())
    p_pool = stats.wilcoxon(pooled).pvalue
    k_pos = int((pooled > 0).sum())

    fig = plt.figure(figsize=(DOUBLE, 4.6))
    ax1 = fig.add_axes([.09, .58, .88, .34])
    F.dot_panel(ax1, [(est[est.term == 'dD'], 'y', DISTRACTOR, sm)], (-30, 30), 'dD, HP − LP (%)')
    pooled_bayes(ax1, B, (-30, 30), series=(('dD', DISTRACTOR),))
    ax1.set_title('a   Distractor gain at HP minus at LP, joint zero-sum fit', loc='left', fontsize=8, color=FG,
                  pad=16)
    ax2 = fig.add_axes([.09, .09, .88, .34])
    F.dot_panel(ax2, [(h, 'd_dD', DISTRACTOR, None)], (-6, 3), 'Δ held-out R² (× 10⁻⁴)')
    from retsupp.visualize.paper.fig7_behaviour import XMAX
    x0 = -2.5
    ax2.scatter(x0 + np.random.default_rng(1).uniform(-.25, .25, len(pooled)), pooled.clip(-6, 3), s=5,
                color=DISTRACTOR, alpha=.35, lw=0)
    ax2.errorbar(x0, m, yerr=[[m - lo], [hi - m]], fmt='o', ms=6, color=DISTRACTOR, mec='w', mew=.8, lw=1.8)
    ax2.text(x0, 3 + .02 * 9, F.stars(p_pool), ha='center', va='bottom', fontsize=8.5, color=DISTRACTOR)
    ax2.axvline(-.95, color='.85', lw=.6, zorder=0)
    ax2.hlines(0, x0 - .6, -.6, color='.75', lw=.6, ls='--', zorder=0)
    ax2.set_xlim(x0 - .8, XMAX + .6)
    ticks = [x0] + list(ax2.get_xticks()[ax2.get_xticks() > -.5])
    labs = ['All\nareas'] + [t.get_text() for t in ax2.get_xticklabels() if t.get_position()[0] > -.5]
    ax2.set_xticks(ticks)
    ax2.set_xticklabels(labs, fontsize=6)
    ax2.set_title('b   Adding dD to the main model: prediction of held-out runs', loc='left', fontsize=8, color=FG,
                  pad=16)

    g = lambda u: post[(post.unit == u) & post.level.isin(['overall', 'group'])].iloc[0]
    f = lambda r: (f"{r['mean']:+.1f} [{r.hdi_lo:+.1f}, {r.hdi_hi:+.1f}], P = {max(r.p_gt0, r.p_lt0):.3f}"
                   .replace('-', '−'))
    areas = ', '.join(f"{u} {f(ar.loc[u])}" for u in ROI_ORDER)
    cal = float(np.median(est[est.term == 'dD'].halfdiff / est[est.term == 'dD'].se))
    legend = (
        "**Supplementary Figure. No distractor-specific reduction at the HP location.** The main model (DoG pRFs, "
        "zero-sum sustained field, pRFs and gain fields fitted jointly) with the distractor gain free to differ between "
        "the HP and the LP locations (dD = distractor gain at HP minus at LP, percentage points); n = 28. "
        "a, dD per participant and area (dots; triangles, beyond the axis), re-solved by least squares around the joint "
        "fit with all other parameters held, each with a leave-one-run-out jackknife SE (half the odd − even-run "
        f"difference over the SE: median {cal:.2f}, 0.67 expected). Points and bars, posterior mean and 95% HDI of the "
        "Student-t hierarchical model of Fig. 5; stars, posterior probability of the sign: * > .975, ** > .995. "
        f"All areas {f(g('all'))}; early {f(g('early'))}; dorsal {f(g('dorsal'))}; occipitotemporal "
        f"{f(g('occipitotemporal'))}. Per area: {areas}. "
        "b, Change in the R² of held-out runs (leave one run out, × 10⁻⁴) when dD is added to the main model, per "
        "participant and area; points and bars, median and bootstrapped 95% CI; stars, Wilcoxon signed-rank test, "
        "Benjamini-Hochberg across the nine areas. All areas, per participant the median over its areas: "
        f"{m:+.2f} [{lo:+.2f}, {hi:+.2f}] × 10⁻⁴, {k_pos} of {len(pooled)} participants better, "
        f"p = {p_pool:.1e}. ".replace('-', '−').replace('e−0', ' × 10⁻').replace('⁻9', '⁻⁹').replace('⁻8', '⁻⁸') +
        "A positive dD in early visual cortex (a distractor at HP lost less gain than at LP) is not supported by "
        "held-out prediction.")
    save(fig, a.name, legend)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dd-dir', default='notes/data/dD_meta_zs_2026-11-25')
    ap.add_argument('--heldout', default='notes/data/hp_terms_heldout_r2_zs_lm_2026-11-16.tsv')
    ap.add_argument('--name', default='figS_dD_joint_zs_n28_palA')
    main(ap.parse_args())
