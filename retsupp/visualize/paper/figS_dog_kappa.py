"""Supplementary figure: the DoG centre/surround variant of the sustained field (κ, model m8_full_dogk).

a  Schematic: a DoG pRF at the HP location; the sustained field multiplies the centre by 1 + (1 + κ)F and the
   surround by 1 + (1 − κ)F (κ = 0: the joint fit m7; κ < 0: more of the field on the suppressive surround).
b  κ per area from the full-data fits (subject dots, median and bootstrapped 95% CI; Wilcoxon against 0, BH over the
   nine areas).
c  Held-out ΔR² of the model with κ against the joint fit without it (pRFs frozen, run-3 cross-validation, mean over
   the three folds), same style.

Also prints the pooled κ (per subject, the mean over areas; median over subjects) with its bootstrapped 95% CI.

    python -m retsupp.visualize.paper.figS_dog_kappa --exclude-subjects 1 2 --name figS_dog_kappa_n28
"""
from __future__ import annotations

import argparse
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from retsupp.visualize.paper.paper_style import (DOUBLE, FG, HP, MUTED, PRF, ROI_COLORS, ROI_ORDER, ROI_X, offset_spines,
                                                 roi_axis, save)

FULL = 'notes/data/dog_cs_field_full_2026-11.tsv'
RUN3 = 'notes/data/dog_cs_field_2026-11.tsv'
SUR = PRF                # surround (suppressive pooling) in the schematic: the pRF colour
N_BOOT = 10_000


def median_ci(x, seed=0):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    rng = np.random.default_rng(seed)
    boots = np.median(rng.choice(x, (N_BOOT, len(x))), axis=1)
    return np.median(x), *np.percentile(boots, [2.5, 97.5])


def stars(q):
    return '***' if q < .001 else '**' if q < .01 else '*' if q < .05 else ''


def area_stats(d, col):
    rows = []
    for r in ROI_ORDER:
        v = d.loc[d.roi == r, col].dropna().to_numpy()
        m, lo, hi = median_ci(v)
        rows.append(dict(roi=r, n=len(v), median=m, ci_lo=lo, ci_hi=hi, pct_neg=(v < 0).mean() * 100,
                         p=stats.wilcoxon(v).pvalue))
    out = pd.DataFrame(rows)
    out['q'] = stats.false_discovery_control(out.p)
    return out


def pooled(d, col):
    s = d.groupby('subject')[col].mean()
    m, lo, hi = median_ci(s.to_numpy())
    return dict(n=len(s), median=m, ci_lo=lo, ci_hi=hi, pct_neg=(s < 0).mean() * 100, p=stats.wilcoxon(s).pvalue)


def dots_panel(ax, d, col, st, ylim, scale=1., seed=1):
    """Subject dots per area (off-scale values as triangles at the edge), median and bootstrapped 95% CI, stars."""
    rng = np.random.default_rng(seed)
    lo_lim, hi_lim = ylim
    for _, r in st.iterrows():
        x0 = ROI_X[r.roi]
        v = d.loc[d.roi == r.roi, col].dropna().to_numpy() * scale
        jit = rng.uniform(-.2, .2, len(v))
        inside = (v >= lo_lim) & (v <= hi_lim)
        ax.scatter(x0 - .08 + jit[inside], v[inside], s=5, color=PRF, alpha=.55, lw=0, zorder=2)
        for y, m in ((lo_lim, 'v'), (hi_lim, '^')):
            sel = (v < lo_lim) if m == 'v' else (v > hi_lim)
            if sel.any():
                ax.scatter(x0 - .08 + jit[sel], np.full(sel.sum(), y), s=7, marker=m, color=PRF,
                           alpha=.7, lw=0, zorder=2, clip_on=False)
        ax.plot([x0 + .28] * 2, [r.ci_lo * scale, r.ci_hi * scale], color=FG, lw=1.1, solid_capstyle='butt', zorder=4)
        ax.plot(x0 + .28, r['median'] * scale, 'D', ms=3.6, color=PRF, mec=FG, mew=.6, zorder=5)
        ax.text(x0 + .1, hi_lim + .04 * (hi_lim - lo_lim), stars(r.q), ha='center', va='bottom', fontsize=8,
                color=FG, clip_on=False)
    ax.axhline(0, color='.8', lw=.6, zorder=0)
    ax.set_ylim(lo_lim, hi_lim)
    offset_spines(ax)
    roi_axis(ax, rotate=False, fontsize=6)


def panel_schematic(ax):
    """1-D cut through a DoG pRF at the HP location; centre and surround components with and without the field."""
    x = np.linspace(-5, 5, 801)
    sc, ss, a_s = 1.0, 1.8, .3            # illustrative: centre σ 1°, surround 1.8× wider at 30% amplitude
    g, s_af, kappa = -.35, 2.0, -.6        # field at HP: suppression, σ_AF 2°; κ exaggerated for display
    cen = np.exp(-x ** 2 / (2 * sc ** 2))
    sur = -a_s * np.exp(-x ** 2 / (2 * (ss * sc) ** 2)) / 1.0
    F = g * np.exp(-x ** 2 / (2 * s_af ** 2))
    m_cen = np.maximum(1 + (1 + kappa) * F, 0)
    m_sur = np.maximum(1 + (1 - kappa) * F, 0)
    ax.plot(x, cen, color=FG, lw=.8, ls=(0, (2, 1.5)))
    ax.plot(x, cen * m_cen, color=FG, lw=1.4)
    ax.plot(x, sur, color=SUR, lw=.8, ls=(0, (2, 1.5)))
    ax.plot(x, sur * m_sur, color=SUR, lw=1.4)
    # The field itself, drawn as a band below the profile, FWHM marked.
    yb = -.62
    ax.fill_between(x, yb, yb + F / abs(g) * -.16, color=HP, alpha=.25, lw=0)
    fwhm = 2 * np.sqrt(2 * np.log(2)) * s_af
    ax.plot([-fwhm / 2, fwhm / 2], [yb - .05] * 2, color=HP, lw=.8)
    ax.text(0, yb - .08, f'Sustained field at HP (FWHM {fwhm:.1f}°)', color=HP, fontsize=6, ha='center', va='top')
    ax.axhline(0, color='.85', lw=.5, zorder=0)
    ax.plot(0, 1.13, 'o', ms=4.5, mfc='w', mec=HP, mew=1.0, clip_on=False)
    ax.text(.3, 1.13, 'HP location', fontsize=6, color=HP, va='center')
    ax.text(1.35, .72, 'Centre × (1 + (1 + κ)F)', fontsize=6, color=FG, va='center')
    ax.text(1.6, -.34, 'Surround × (1 + (1 − κ)F)', fontsize=6, color=SUR, va='center')
    # glyph key: dashed = without the field, solid = with it (κ < 0)
    kx, ky = -4.9, .95
    ax.plot([kx, kx + .8], [ky, ky], color=FG, lw=.8, ls=(0, (2, 1.5)))
    ax.text(kx + 1.0, ky, 'No field', fontsize=6, va='center', color=FG)
    ax.plot([kx, kx + .8], [ky - .16, ky - .16], color=FG, lw=1.4)
    ax.text(kx + 1.0, ky - .16, 'With field', fontsize=6, va='center', color=FG)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-.82, 1.2)
    ax.set_xticks([-4, -2, 0, 2, 4])
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_xlabel('Position (°)')
    offset_spines(ax)


def loso_summary(path):
    """Leave-one-subject-out test of κ (loso_fixed_group_exact): per subject the median over areas of ΔR² at the
    other subjects' median κ, and of the sign-flip contrast."""
    a = pd.read_csv(path, sep='\t')
    a = a[a.mech == 'kappa']
    x = a.pivot(index='subject', columns='roi', values='dr2').median(axis=1).dropna()
    sg = a.pivot(index='subject', columns='roi', values='dr2_sign').median(axis=1).dropna()
    return dict(n=len(x), better=(x > 0).mean() * 100, p=stats.wilcoxon(x).pvalue, med=x.median(),
                sign=(sg > 0).mean() * 100, p_sign=stats.wilcoxon(sg).pvalue, delta=a.delta.median())


def main(name, exclude, loso):
    full = pd.read_csv(FULL, sep='\t')
    full = full[(full.model == 'm8_full_dogk') & ~full.subject.isin(exclude)]
    run3 = pd.read_csv(RUN3, sep='\t')
    run3 = run3[(run3.model == 'm8_full_dogk') & ~run3.subject.isin(exclude)]
    run3 = run3.groupby(['subject', 'roi'], as_index=False).dcv_r2.mean()
    n_sub = full.subject.nunique()

    st_k, st_r = area_stats(full, 'kappa'), area_stats(run3, 'dcv_r2')
    pk, pr = pooled(full, 'kappa'), pooled(run3, 'dcv_r2')
    n_bound = int((full.kappa.abs() >= 1.999).sum())
    L = loso_summary(loso)
    loso_txt = re.sub(r'(?<=[\de ])-(?=\d)', '−', f"Leave-one-subject-out (each participant" + f""" scored at the median κ of the other participants, all voxels of the area, ΔR² on signal voxels): ΔR² higher in {L['better']:.0f}% of {L['n']} participants (Wilcoxon p = {L['p']:.2g}; median ΔR² {L['med'] * 1e-5:+.1e}, median δ {L['delta']:+.3f}), and the participants' data move the way the others' κ predicts in {L['sign']:.0f}% (sign-flip contrast, p = {L['p_sign']:.2g}). """)
    run3_missing = sorted(set(full.subject) - set(run3.subject))

    W = DOUBLE
    fig = plt.figure(figsize=(W, 2.35))
    axa = fig.add_axes([.035, .2, .25, .66])
    axb = fig.add_axes([.375, .3, .28, .54])
    axc = fig.add_axes([.715, .3, .28, .54])
    panel_schematic(axa)
    dots_panel(axb, full, 'kappa', st_k, (-.8, .5))
    axb.set_ylabel('κ')
    axb.set_yticks([-.5, 0, .5])
    dots_panel(axc, run3, 'dcv_r2', st_r, (-4.5, 2.5), scale=1e4)
    axc.set_ylabel('Held-out ΔR² (× 10⁻⁴)')
    axc.set_yticks([-4, -2, 0, 2])
    for ax, let, title in ((axa, 'a', 'The field split between centre and surround'),
                           (axb, 'b', 'κ < 0: more of the field on the surround'),
                           (axc, 'c', 'κ does not help held-out prediction')):
        fig.text(ax.get_position().x0 - .03, .965, let, fontsize=8, fontweight='bold', va='top')
        fig.text(ax.get_position().x0 - .005, .965, title, fontsize=7, va='top', color=FG)
    for ax in (axb, axc):
        bb = ax.get_window_extent()
        assert bb.height <= bb.width

    def row(r, scale=1., fmt='{:+.3f}'):
        return (f"{r.roi} {fmt.format(r['median'] * scale)} [{fmt.format(r.ci_lo * scale)}, {fmt.format(r.ci_hi * scale)}], "
                f"n = {r.n}, p = {r.p:.2g}, q = {r.q:.2g}").replace('-', '−')

    def prow(p, scale=1., fmt='{:+.3f}'):
        return (f"{fmt.format(p['median'] * scale)} [{fmt.format(p['ci_lo'] * scale)}, {fmt.format(p['ci_hi'] * scale)}], "
                f"{p['pct_neg']:.0f}% of {p['n']} subjects below 0, p = {p['p']:.2g}").replace('-', '−')

    legend = f"""
**Supplementary Figure. A centre/surround split of the sustained field is consistent in direction but too small to help prediction.** a, Schematic of the variant (model m8_full_dogk): a one-dimensional cut through a DoG pRF at the HP location. The sustained gain field F (at the four item locations; drawn here at HP, with its FWHM) multiplies the pRF's centre (driving) pooling by 1 + (1 + κ)F and its surround (suppressive) pooling by 1 + (1 − κ)F, one κ per subject × area shared by all voxels. κ = 0 is the joint fit (m7) exactly; κ < 0 puts more of the field on the surround. Dashed: centre and surround without the field; solid: with a suppressive field at HP (illustrative values; κ = −0.6 exaggerated for display). b, κ per area from the full-data fits (n = {n_sub} participants; per area the subject × area pairs with a fit). c, Held-out ΔR² of the model with κ minus the joint fit without it, with the pRFs frozen and run-3 cross-validation (mean over the three folds), × 10⁻⁴ ({run3.subject.nunique()} participants; no run-3 fits for sub-{', sub-'.join(f'{x:02d}' for x in run3_missing)}). b, c: dots are participants (triangles: beyond the axis), diamonds and bars the median and its bootstrapped 95% CI (10 000 resamples); Wilcoxon signed-rank test against 0, Benjamini-Hochberg across the nine areas, * q < .05, ** q < .01, *** q < .001.

κ is negative: pooled (per participant, the mean over areas) {prow(pk)}. Per area (median [95% CI]): {'; '.join(row(r) for _, r in st_k.iterrows())}. {n_bound} subject × area fits sit at the bound |κ| = 2. Adding κ makes held-out prediction worse: pooled ΔR² {prow(pr, 1e4, '{:+.2f}')} (× 10⁻⁴); per area {'; '.join(row(r, 1e4, '{:+.2f}') for _, r in st_r.iterrows())} (× 10⁻⁴). {loso_txt}The tilt is consistent in direction (more of the HP effect on the suppressive surround) but too small to help prediction: on held-out runs it lowers R² in every area, and across participants its gain is of the order of 10⁻⁷.
"""
    save(fig, name, legend)
    out = pd.concat([st_k.assign(quantity='kappa'), st_r.assign(quantity='dcv_r2_run3')])
    out.to_csv(f'notes/data/{name}_2026-11-11.tsv', sep='\t', index=False)
    print(out.round(5).to_string(index=False))
    print('pooled kappa', pk)
    print('pooled run3 dR2', pr)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--exclude-subjects', nargs='*', type=int, default=[])
    ap.add_argument('--name', default='figS_dog_kappa')
    ap.add_argument('--loso', default='notes/data/loso_kappa_dnw_2026-11.tsv',
                    help='LOSO table of kappa (n = 28: notes/data/loso_kappa_n28_2026-11-11.tsv)')
    a = ap.parse_args()
    main(a.name, a.exclude_subjects, a.loso)
