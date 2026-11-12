"""Supplementary figure: field widths among fits with a reliable gain, and how well each width is constrained.

a  Sustained field FWHM per area: all fits (pale) and fits with a reliable sustained gain (|gain| / SE > 2 at HP or at
   the LP locations; solid), median and bootstrapped 95% CI; dotted, the 5.7° spacing of neighbouring item locations.
b  The same for the transient field, kept when the target gain is reliable.
c, d  Width of each fit's 95% profile interval (FWHM degrees, leave-one-run-out jackknife of the profile minimum) against
   |gain| / SE: the width is pinned down only when the gain is clearly non-zero.

    python -m retsupp.visualize.paper.figS_field_widths_v2
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from retsupp.visualize.paper.paper_style import DOUBLE, FG, HP, MUTED, ROI_COLORS, ROI_ORDER, ROI_X, offset_spines, roi_axis, save
from retsupp.visualize.paper.field_widths_reliable import SPACING, boot_ci

TRANS = '#5B6770'
plt.rcParams.update({'font.family': 'Helvetica', 'font.size': 7, 'axes.labelsize': 7.5, 'xtick.labelsize': 6.5,
                     'ytick.labelsize': 6.5, 'axes.spines.top': False, 'axes.spines.right': False,
                     'pdf.fonttype': 42, 'axes.linewidth': .8})


def width_panel(ax, d, col, rel, color, ylab, thr):
    rng = np.random.default_rng(1)
    for roi in ROI_ORDER:
        x0 = ROI_X[roi]
        for k, (keep, off, alpha, ms) in enumerate(((d.roi == roi, -.17, .35, 3.4), ((d.roi == roi) & rel, .17, 1., 4.2))):
            v = d.loc[keep, col].to_numpy()
            if not len(v):
                continue
            jit = rng.uniform(-.06, .06, len(v))
            ax.scatter(x0 + off + jit, np.clip(v, 1.0, 20), s=4, color=color, alpha=.18 if k == 0 else .45, lw=0, zorder=2)
            lo, hi = boot_ci(v)
            m = np.median(v)
            ax.plot([x0 + off] * 2, [lo, hi], color=color, alpha=alpha, lw=1.1, solid_capstyle='butt', zorder=3)
            ax.plot(x0 + off, m, 'o', ms=ms, color=color, alpha=alpha, mec='w', mew=.5, zorder=4)
            ax.text(x0 + off, 21.5, f'{len(v)}', ha='center', va='bottom', fontsize=6, color=color, alpha=max(alpha, .6))
    ax.axhline(SPACING, color=FG, lw=.6, ls=(0, (1, 2)), zorder=1)
    ax.set_yscale('log')
    ax.set_ylim(1.0, 21)
    ax.set_yticks([1, 2, 5, 10, 20])
    ax.set_yticklabels(['1', '2', '5', '10', '20'])
    ax.set_ylabel(ylab)
    offset_spines(ax)
    roi_axis(ax, rotate=False)


def profile_panel(ax, d, z, width, color, title):
    x = np.clip(d[z].abs().to_numpy(), .05, 60)
    y = d[width].to_numpy()
    open_ = ~np.isfinite(y)
    yc = np.clip(np.where(open_, 20, y), .1, 20)
    ax.scatter(x[~open_], yc[~open_], s=5, color=color, alpha=.45, lw=0)
    ax.scatter(x[open_], yc[open_], s=9, marker='^', facecolor='none', edgecolor=color, lw=.5, alpha=.7)
    ax.scatter([14], [.25], s=9, marker='^', facecolor='none', edgecolor=color, lw=.5)
    ax.text(17, .25, 'Open interval', fontsize=6, color='.4', va='center')
    ax.axvline(2, color=FG, lw=.6, ls=(0, (2, 2)))
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(.05, 60)
    ax.set_ylim(.1, 25)
    ax.set_xticks([.1, 1, 2, 10, 50])
    ax.set_xticklabels(['0.1', '1', '2', '10', '50'])
    ax.set_yticks([.1, 1, 10])
    ax.set_yticklabels(['0.1', '1', '10'])
    ax.set_xlabel('|Gain| / SE')
    ax.set_ylabel('95% interval of the\nwidth (FWHM, °)')
    ax.set_title(title, fontsize=7.5, loc='left', color=color, pad=6)
    offset_spines(ax)


def main(fits, name, thr=2.):
    d = pd.read_csv(fits, sep='\t')
    d = d[d.roi.isin(ROI_ORDER)]
    W, H = DOUBLE, 5.4
    fig = plt.figure(figsize=(W, H))
    axa = fig.add_axes([.08, .715, .90, .225])
    axb = fig.add_axes([.08, .375, .90, .225])
    axc = fig.add_axes([.10, .07, .36, .18])
    axd = fig.add_axes([.60, .07, .36, .18])
    width_panel(axa, d, 'fwhm_sus', d.sus_z > thr, HP, 'Sustained field\nFWHM (°)', thr)
    width_panel(axb, d, 'fwhm_dyn', d.g_T_z.abs() > thr, TRANS, 'Transient field\nFWHM (°)', thr)
    profile_panel(axc, d, 'sus_z', 'prof_width_sus', HP, 'Sustained field')
    profile_panel(axd, d, 'g_T_z', 'prof_width_dyn', TRANS, 'Transient field (target gain)')
    for ax, let in ((axa, 'a'), (axb, 'b'), (axc, 'c'), (axd, 'd')):
        ax.text(-.06 if ax in (axa, axb) else -.2, 1.08, let, transform=ax.transAxes, fontsize=8, weight='bold', va='bottom')
    # key (glyphs)
    k = fig.add_axes([.08, .955, .9, .04])
    k.set_axis_off()
    k.set_xlim(0, 10)
    k.set_ylim(-1, 1)
    k.plot(.1, 0, 'o', ms=3.4, color=FG, alpha=.35, mec='w', mew=.5)
    k.text(.25, 0, 'All fits', va='center', fontsize=6.5, color='.4')
    k.plot(1.3, 0, 'o', ms=4.2, color=FG, mec='w', mew=.5)
    k.text(1.45, 0, f'Fits with a reliable gain (|gain| / SE > {thr:g})', va='center', fontsize=6.5, color=FG)
    k.plot([5.2, 5.6], [0, 0], color=FG, lw=.6, ls=(0, (1, 2)))
    k.text(5.7, 0, 'Spacing of neighbouring item locations, 5.7°', va='center', fontsize=6.5, color='.4')
    k.text(9.95, 0, 'Numbers: fits per series', va='center', ha='right', fontsize=6, color='.4')
    return fig, d


def legend_from_tables(prefix, fit_label=None, thr=2):
    """The legend with every number read from the ``field_widths_reliable`` tables of ``prefix`` (any reference fit)."""
    from scipy import stats
    d = pd.read_csv(f'{prefix}_fits.tsv', sep='\t')
    d = d[d.roi.isin(ROI_ORDER)]
    cal = pd.read_csv(f'{prefix}_calibration.tsv', sep='\t').iloc[0]
    cl = pd.read_csv(f'{prefix}_claims_thr{thr}.tsv', sep='\t')
    w = pd.read_csv(f'{prefix}_widths_thr{thr}.tsv', sep='\t')
    allrow = cl[cl.roi.str.startswith('all')].iloc[0]
    cl = cl[~cl.roi.str.startswith('all')].set_index('roi').loc[ROI_ORDER]
    sw = w[(w.field == 'sustained') & (w.roi == 'all') & w.subset.str.startswith('reliable')].iloc[0]
    tw = w[(w.field == 'transient') & (w.roi == 'all') & w.subset.str.startswith('reliable')]
    rel_t = d[d.g_T_z.abs() > thr]
    tr_bound = 100 * rel_t.bound_dyn.mean()
    k_below = int(round(allrow.share_below * allrow.n))
    def fp(p):     # exact, × 10⁻ⁿ below .001
        if p >= .001:
            return '= ' + f'{p:.2g}'.replace('0.', '.')
        m_, e_ = f'{p:.1e}'.split('e-')
        return f'= {m_} × 10' + ''.join('⁰¹²³⁴⁵⁶⁷⁸⁹'[int(c)] for c in str(int(e_))).join(['⁻', ''])
    fq = lambda q: fp(q) if q < .01 else '= ' + f'{q:.3f}'.replace('0.', '.')
    yes = [r for r in ROI_ORDER if cl.loc[r, 'q'] < .05]
    no = [f'{r} (n = {int(cl.loc[r, "n"])})' for r in ROI_ORDER if not cl.loc[r, 'q'] < .05]
    z = d.sus_z.abs()
    pw = d.prof_width_sus.replace(np.inf, np.nan)
    lo_bin = d.loc[z < 1, 'prof_width_sus'].median()
    hi_bin = d.loc[z > 3, 'prof_width_sus'].median()
    rho = stats.spearmanr(z, pw, nan_policy='omit').statistic   # closed intervals
    pairs = Path(f'{prefix}_early_vs_higher_thr{thr}.tsv')
    pair_txt = ''
    if pairs.exists():
        pr = pd.read_csv(pairs, sep='\t').iloc[0]
        pair_txt = (f' Early (V1–V3) vs higher areas, paired participant medians: {pr.early_median:.1f}° vs '
                    f'{pr.higher_median:.1f}°, {int(pr.k_pos)} of {int(pr.n_pairs)} wider in early, p {fp(pr.p)}.')
    fits = fit_label or 'joint gain-field fits with DoG pRFs and one HRF per participant and area'
    return (
        f"**Figure S. Width of the gain fields among fits with a reliable gain, and how well each width is constrained.** "
        f"{fits[0].upper() + fits[1:]} (n = {d.subject.nunique()} participants, {len(d)} participant × area fits). For "
        "each fit, the gains of one field were re-estimated by least squares with every other parameter (pRFs, HRF, item "
        "scaling, the other field) held at the joint fit, run by run, giving a leave-one-run-out jackknife SE for each "
        f"gain (calibrated: half the odd − even-run difference over the SE has median {cal.g_HP:.2f}, {cal.g_LP:.2f} and "
        f"{cal.g_T:.2f} for the sustained gain at HP, at the other locations and the target gain; 0.67 expected). A gain "
        f"counts as reliable when |gain| / SE > {thr}. a, Full width at half maximum (FWHM = 2.355 σ) of the sustained "
        "field per area: all fits (pale) and fits whose sustained gain is reliable at HP or at the other three locations "
        "(solid). Points and bars, median and bootstrapped 95% CI; small dots, fits (values beyond 1–20° drawn at the "
        "edge); numbers, fits per series. Dotted line, the 5.7° between neighbouring possible item locations. Among fits "
        f"with a reliable sustained gain, the median FWHM is {sw['median']:.1f}° [{sw.ci_lo:.1f}, {sw.ci_hi:.1f}] "
        f"(participant medians over areas; {k_below} of {int(allrow.n)} below 5.7°, Wilcoxon against 5.7°, "
        f"p {fp(allrow.p)}), below 5.7° in {', '.join(f'{r} (n = {int(cl.loc[r, chr(110)])}, q {fq(cl.loc[r, chr(113)])})' for r in yes)}"
        f"{'; not in ' + ', '.join(no) if no else ''} (Benjamini-Hochberg across nine areas).{pair_txt} b, The same for "
        "the transient field, kept when the target gain is reliable. Even then the transient width sits at a bound of "
        f"the fit (0.5° or 8°) in {tr_bound:.0f}% of fits, so it is not interpreted. c, d, Width of each fit's 95% "
        "interval for the field width (FWHM degrees) against |gain| / SE. The interval comes from the profile of the fit "
        "error over the width (σ on a log grid from 0.5° to 8°; at each width the field's gains re-solved, all other "
        "parameters fixed at the joint fit) and the leave-one-run-out jackknife of the profile minimum, because the "
        "residuals are correlated over time and voxels and a χ² threshold with the nominal residual variance would be far "
        "too narrow. Triangles: profile minimum at the grid edge, so the interval is open. The sustained width is pinned "
        f"down only when its gain is clearly non-zero (median interval {lo_bin:.1f}° at |gain| / SE < 1, {hi_bin:.1f}° at "
        f"> 3; Spearman ρ = {f'{rho:.2f}'.replace('0.', '.')} over the closed intervals); the transient width stays poorly constrained even for reliable target gains. "
        f"Source: `{prefix}_*.tsv`; code `retsupp/modeling/fit_width_profiles.py`, "
        "`retsupp/visualize/paper/field_widths_reliable.py`.").replace('ρ = -', 'ρ = −').replace('= -.', '= −.')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--fits', default='notes/data/field_widths_reliable_n28_2026-11-12_fits.tsv')
    ap.add_argument('--name', default='figS_field_widths_v2_n28')
    ap.add_argument('--legend', default='notes/data/field_widths_reliable_n28_2026-11-12_legend.md',
                    help="a legend file, or 'auto': computed from the tables next to --fits")
    ap.add_argument('--fit-label', default=None, help="--legend auto: how the legend names the fits")
    a = ap.parse_args()
    fig, _ = main(a.fits, a.name)
    if a.legend == 'auto':
        text = legend_from_tables(a.fits.removesuffix('_fits.tsv'), a.fit_label)
    else:
        text = Path(a.legend).read_text() if Path(a.legend).exists() else 'Legend pending.'
    save(fig, a.name, text)
