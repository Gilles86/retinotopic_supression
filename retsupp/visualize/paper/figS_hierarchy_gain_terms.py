"""Supplementary figure: the gain terms along the visual hierarchy (outputs of ``hierarchy_gain_terms``).

a  Per area, each term (median over participants) against the area's pRF size, as a proportion of the term's
   pooled median, with the median per-participant slope drawn through the area medians.
b  Slope per doubling of pRF size per term (normalised units), median and bootstrapped 95% CI.
c  Within areas: each term per tercile of pRF size (normalised), median over participants of the median over areas.
d  Sustained field FWHM (fits with a reliable sustained gain) against pRF size, with the median slope.

    python -m retsupp.visualize.paper.figS_hierarchy_gain_terms --prefix notes/data/hierarchy_gain_terms_n28_2026-11-12
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from retsupp.visualize.paper.hierarchy_gain_terms import ROIS, median_ci, pooled_median, subject_slopes
from retsupp.visualize.paper.paper_style import DISTRACTOR, DOUBLE, FG, HP, MUTED, ROI_COLORS, TARGET, offset_spines, save

SHOW = [('T', 'Target gain', TARGET), ('D', 'Distractor gain', DISTRACTOR),
        ('S_alone', 'Shared sustained gain', FG), ('dS_alone', 'Sustained HP − LP (dS)', HP)]


def normalised(d):
    out = d.copy()
    for col, *_ in SHOW:
        out[col] = d[col] / pooled_median(d, col)
    return out


def main(prefix, name, fits_note):
    d = pd.read_csv(f'{prefix}_terms_dog.tsv', sep='\t')
    size = pd.read_csv(f'{prefix}_prf_size_axis.tsv', sep='\t', index_col=0).iloc[:, 0]
    logsize = np.log2(size)
    dn = normalised(d)
    reg = pd.read_csv(f'{prefix}_regression.tsv', sep='\t')
    w = pd.read_csv(f'{prefix}_width_slope.tsv', sep='\t').iloc[0]
    wf = pd.read_csv(f'{prefix}_width_fits.tsv', sep='\t')
    bins_path = Path(f'{prefix}_bins.tsv')

    fig = plt.figure(figsize=(DOUBLE, 4.6))
    gs = fig.add_gridspec(2, 2, left=.08, right=.985, top=.93, bottom=.1, hspace=.62, wspace=.3)
    axa, axb = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    axc, axd = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])

    # a: area medians, normalised, against pRF size; line = median per-participant slope through the area medians
    xs = logsize.reindex(ROIS)
    slopes_n, ends = {}, []
    for k, (col, lab, c) in enumerate(SHOW):
        med = dn.groupby('roi')[col].median().reindex(ROIS)
        b = subject_slopes(dn, col, logsize, ROIS)
        m, lo, hi, _ = median_ci(b)
        slopes_n[col] = (m, lo, hi, len(b), (b > 0).mean())
        axa.plot(xs, med, 'o', ms=3.2, color=c, mec='w', mew=.4, zorder=3)
        icpt = np.median(med - m * xs)
        xx = np.array([xs.min() - .05, xs.max() + .05])
        axa.plot(xx, icpt + m * xx, color=c, lw=1.1, zorder=2)
        ends.append([icpt + m * xx[1], lab, c, xx[1]])
    ends.sort(key=lambda e: e[0])
    for i in range(1, len(ends)):             # keep the end labels at least 0.28 apart
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + .28)
    for y, lab, c, x in ends:
        axa.text(x + .04, y, lab, color=c, fontsize=6, va='center', ha='left')
    axa.axhline(0, color='.85', lw=.6, zorder=0)
    axa.set_xticks(np.log2([.6, .8, 1, 1.5]))
    axa.set_xticklabels(['0.6', '0.8', '1', '1.5'])
    axa.set_xlim(xs.min() - .15, xs.max() + .95)
    axa.set_xlabel('Area pRF size, σ (°, log scale)')
    axa.set_ylabel('Term / its pooled median')
    axa.set_title('Target gain grows along the hierarchy; dS does not detectably', fontsize=7, loc='left')
    offset_spines(axa)

    # b: normalised slope per term, median and 95% CI
    order = [('T', 'Target gain', TARGET), ('D', 'Distractor gain', DISTRACTOR), ('S_alone', 'Shared sustained', FG),
             ('dS_alone', 'dS', HP)]
    for i, (col, lab, c) in enumerate(order):
        m, lo, hi, n, kp = slopes_n[col]
        axb.plot([lo, hi], [i, i], color=c, lw=1.4, solid_capstyle='butt')
        axb.plot(m, i, 'o', ms=4, color=c, mec='w', mew=.5)
        axb.text(m, i - .22, f'{m:+.2f} [{lo:+.2f}, {hi:+.2f}]', va='bottom', ha='center', fontsize=6, color=c)
    axb.set_ylim(len(order) - .5, -.7)
    axb.axvline(0, color='.75', lw=.6)
    axb.set_yticks(range(len(order)))
    axb.set_yticklabels([o[1] for o in order], fontsize=6.5)
    axb.set_xlabel('Slope per doubling of pRF size\n(proportion of the term\'s pooled median)')
    axb.set_title('Slopes, normalised (n = 28)', fontsize=7, loc='left')
    offset_spines(axb)

    # c: within-area bins of pRF size
    if bins_path.exists():
        bs = pd.read_csv(bins_path, sep='\t')
        bs = bs[bs.by == 'sd']
        cols = [('T', 'Target gain', TARGET), ('D', 'Distractor gain', DISTRACTOR), ('S', 'Shared sustained', FG),
                ('dS', 'dS', HP)]
        for col, lab, c in cols:
            pm = bs.groupby('subject')[col].median().median()
            per = bs.assign(v=bs[col] / pm).groupby(['subject', 'bin']).v.median().unstack()
            med = per.median()
            ci = np.array([median_ci(per[b].dropna())[1:3] for b in per.columns])
            axc.errorbar(per.columns + {'T': -.12, 'D': -.04, 'S': .04, 'dS': .12}[col], med,
                         yerr=[med - ci[:, 0], ci[:, 1] - med], fmt='o-', ms=3, lw=1, color=c, capsize=0,
                         label=lab)
        axc.axhline(0, color='.85', lw=.6, zorder=0)
        axc.set_xticks([0, 1, 2])
        axc.set_xticklabels(['Smallest third', 'Middle', 'Largest third'])
        axc.set_xlabel('pRF size within an area')
        axc.set_ylabel('Term / its pooled median')
        axc.legend(frameon=False, fontsize=6, ncol=2, loc='upper left', handlelength=1.2)
        axc.set_title('Within areas, by pRF size', fontsize=7, loc='left')
    else:
        axc.text(.5, .5, 'Within-area bins: pending', ha='center', va='center', transform=axc.transAxes, fontsize=7)
        axc.set_axis_off()
    offset_spines(axc)

    # d: sustained FWHM among reliable fits against pRF size
    rng = np.random.default_rng(1)
    for r in ROIS:
        v = wf[wf.roi == r].fwhm
        x = logsize[r]
        axd.scatter(x + rng.uniform(-.03, .03, len(v)), v, s=5, color=ROI_COLORS[r], alpha=.45, lw=0, zorder=2)
        if len(v):
            axd.plot(x, v.median(), 'D', ms=3.4, color=ROI_COLORS[r], mec=FG, mew=.4, zorder=4)
    xx = np.array([xs.min() - .05, xs.max() + .05])
    med_all = wf.groupby('roi').fwhm.median().reindex(ROIS)
    icpt = np.nanmedian(med_all - w.slope * xs)
    axd.plot(xx, icpt + w.slope * xx, color=FG, lw=1.1)
    axd.axhline(5.66, color=MUTED, lw=.7, ls=(0, (3, 2)))
    axd.text(xx[1] + .05, 5.66, 'Item spacing\n5.7°', fontsize=6, color='.4', va='center')
    axd.set_ylim(0, 9)
    axd.set_xticks(np.log2([.6, .8, 1, 1.5]))
    axd.set_xticklabels(['0.6', '0.8', '1', '1.5'])
    axd.set_xlim(xs.min() - .15, xs.max() + .45)
    axd.set_xlabel('Area pRF size, σ (°, log scale)')
    axd.set_ylabel('Sustained field FWHM (°)')
    axd.set_title(f'Sustained width: {w.slope:+.2f}° per doubling [{w.ci_lo:+.2f}, {w.ci_hi:+.2f}]', fontsize=7, loc='left',
                  pad=4)
    offset_spines(axd)
    for ax, let in zip((axa, axb, axc, axd), 'abcd'):
        ax.text(-.13, 1.08, let, transform=ax.transAxes, fontsize=8, fontweight='bold', va='bottom')

    rg = reg[(reg.family == 'dog') & (reg.predictor == 'log2 pRF size') & (reg.rois == 'all nine')].set_index('term')

    def r_(term):
        r = rg.loc[term]
        return f'{r.slope:+.1f} percentage points per doubling [{r.ci_lo:+.1f}, {r.ci_hi:+.1f}], {int(r.k_pos)}/{int(r.n)} positive, p = {r.p:.2g}, q = {r.q:.2g}'

    nfit = ', '.join(f'{r} {int((wf.roi == r).sum())}' for r in ROIS)
    legend = f"""
**Figure S. The gain terms along the visual hierarchy.** Each area's gain terms from the joint gain-field fit and from the terms estimated alone ({fits_note}; DoG pRFs; n = 28 participants). The hierarchy axis is the area's pRF size: the median over participants of the voxel-weighted mean pRF σ at 1–2.5° eccentricity (the axis of the target-gain regression in the main text). Each participant's per-area estimates were regressed on log2 of the area's pRF size; the slope is the change per doubling of pRF size. Because each area is fitted with the stimulus as its input, an area's gain is the gain accumulated up to that area, not the gain added at that stage. a, Area medians of the target gain, the distractor gain (averaged over its locations), the sustained gain shared by the four locations (estimated alone) and the sustained gain at HP minus at the other locations (dS, estimated alone), each divided by its pooled median (median over participants of the median over areas), so that the slopes compare; lines, the median per-participant slope. b, Slopes per term in these normalised units, median and bootstrapped 95% CI. In % gain: target gain {r_('Target gain')}; distractor gain {r_('Distractor gain')}; shared sustained gain {r_('Shared sustained gain (alone)')}; dS {r_('dS, alone')} (Wilcoxon signed-rank; Benjamini-Hochberg across the nine terms of the table). c, Within each area, the five gains re-estimated for the smallest, middle and largest third of voxels by pRF size (least squares, all other parameters fixed at the fit; bins with fewer than 20 voxels dropped), normalised as in a; median over participants of the median over areas, bootstrapped 95% CI. d, FWHM of the sustained field in fits with a reliable sustained gain (|gain|/SE > 2 at HP or at the other locations; leave-one-run-out jackknife SE), per area (dots, fits; diamonds, area medians, coloured by area as in the main figures; fits kept: {nfit}) against pRF size; line, median per-participant slope ({w.slope:+.2f}° per doubling [{w.ci_lo:+.2f}, {w.ci_hi:+.2f}], {int(w.n)} participants with at least three areas, p = {w.p:.2g}); dashed, the 5.7° spacing of neighbouring possible target and distractor locations.
"""
    save(fig, name, legend)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--prefix', default='notes/data/hierarchy_gain_terms_n28_2026-11-12')
    ap.add_argument('--name', default='figS_hierarchy_gain_terms_n28')
    ap.add_argument('--fits-note', default='current joint fits, pRFs held at the fit without gain fields')
    a = ap.parse_args()
    main(a.prefix, a.name, a.fits_note)
