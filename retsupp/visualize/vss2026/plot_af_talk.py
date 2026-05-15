"""AF result figures for Docky's VSS2026 talk.

ONE figure per story (per project memory: separate figures for
separate analyses). Each figure is a single panel sized for a 16:9
slide, with subjects shown as filled colored dots (the data IS the
message — not desaturated to gray), group mean as a large dark
diamond, ±1 SEM as a vertical line, Wilcoxon stars above, per-ROI
n under each x-tick. Despined + offset per scientific-figures
house style.

Three output files (PDF + SVG each):
  - af_talk_static_HP_suppression.{pdf,svg}
        g_HP - g_LP per ROI (sustained suppression, one-sided <0)
  - af_talk_phasic_distractor.{pdf,svg}
        g_HP_dyn - g_LP_dyn per ROI (phasic distractor, two-sided)
  - af_talk_phasic_target_capture.{pdf,svg}
        g_T_dyn per ROI (phasic target capture, one-sided >0)

Source: ``notes/data/af_dog_v3_target_sharedSigma_parameters.tsv``
(canonical AF — refresh via
``python -m retsupp.visualize.paper.plot_af_results
--analysis af_dynamic_sharedSigma --rois all``).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


REPO = Path(__file__).resolve().parents[3]
DEFAULT_TSV = REPO / 'notes' / 'data' / 'af_dog_v3_target_sharedSigma_parameters.tsv'
OUT_DIR = REPO / 'notes' / 'figures'

ROI_ORDER = ['V1', 'V2', 'V3', 'V3AB', 'hV4', 'LO', 'TO', 'VO',
             'IPS', 'SPL1', 'FEF']

# Outlier cutoff for AF gain parameters. AF fits in low-SNR ROIs
# (FEF, SPL1) occasionally produce |g| > 5 — those are degenerate
# fits, not real biology. Drop them per (ROI × column) before
# computing mean + SEM so the central tendency reflects the
# well-behaved subjects.
OUTLIER_ABS_CUTOFF = 4.0

# Colors — match the talk illustrations' palette (notes/figures/talk/
# make_figures.py BUILDUP_PALETTE). Each panel's color encodes the
# *thing being measured*: HP-static → red, dynamic distractor → blue,
# target → orange.
TALK_HP     = '#D62828'   # red: sustained HP / static suppression
TALK_AF     = '#0077B6'   # blue: dynamic distractor / phasic differential
TALK_TARGET = '#F77F00'   # orange: target capture
TALK_STIM   = '#3D5A80'   # deep blue: location-marginalized distractor (stim)
TALK_FG     = '#1F2933'   # near-black: axes, text
TALK_MUTED  = '#9AA5B1'   # light gray: zero-line, secondary marks

C_DOT_EDGE  = 'white'
C_MEAN_EDGE = TALK_FG     # near-black edge so mean pops above the swarm
C_SEM       = TALK_FG
C_STAR_SIG  = TALK_FG     # stars in near-black; color already encoded by panel theme


def _apply_rcparams_talk():
    mpl.rcParams.update({
        'font.family': 'Helvetica',
        'font.sans-serif': ['Helvetica', 'Helvetica Neue',
                            'TeX Gyre Heros', 'Arial'],
        'font.size': 13,
        'axes.labelsize': 16,
        'axes.titlesize': 17,
        'xtick.labelsize': 13,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'mathtext.fontset': 'stixsans',
        'axes.linewidth': 1.1,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.labelpad': 8,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.major.size': 4,
        'ytick.major.size': 4,
        'xtick.major.width': 1.1,
        'ytick.major.width': 1.1,
        'lines.linewidth': 1.6,
        'patch.linewidth': 0.6,
        'legend.frameon': False,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'svg.fonttype': 'none',
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
    })


def _stars(p):
    if not np.isfinite(p):
        return ''
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return ''


def _p_text(p):
    """Format p-value for in-panel annotation. None if not significant."""
    if not np.isfinite(p) or p >= 0.05:
        return None
    if p < 0.001:
        return 'p < .001'
    # APA: drop leading zero, 3 decimals
    return f'p = .{int(round(p * 1000)):03d}'


def _plot_one(df, *, col, ylim, ylabel, title, alternative,
              out_stem, theme_color, variant=''):
    """Render ONE single-panel talk figure for one quantity × ROIs.

    ``theme_color`` is the panel's identity color (HP-red / AF-blue /
    target-orange from the talk palette) — used for subject dots AND
    the mean fill so the figure reads as "this is the HP panel" at a
    glance. Mean has a dark edge so it pops above the dot swarm.
    """
    _apply_rcparams_talk()
    fig, ax = plt.subplots(figsize=(9.0, 5.5), constrained_layout=True)

    rois = [r for r in ROI_ORDER if r in df['roi'].unique()]
    rng = np.random.default_rng(seed=42)

    n_kept_by_roi = []
    n_dropped_by_roi = []
    for i, roi in enumerate(rois):
        raw = df.loc[df['roi'] == roi, col].dropna().values
        if len(raw) < 2:
            n_kept_by_roi.append(0)
            n_dropped_by_roi.append(0)
            continue
        # Outlier filter: drop subjects with |g| > cutoff (per-ROI,
        # per-column). FEF / SPL1 AF gains can blow up to |g|>5 on
        # subjects with bad fits — those distort mean and SEM.
        mask_kept = np.abs(raw) <= OUTLIER_ABS_CUTOFF
        sub = raw[mask_kept]
        n_kept_by_roi.append(int(mask_kept.sum()))
        n_dropped_by_roi.append(int((~mask_kept).sum()))
        if len(sub) < 2:
            continue
        # Subject dots — panel's theme color, transparent. Show all
        # kept subjects; clip to ylim so the panel stays compact.
        v_plot = np.clip(sub, *ylim)
        xs = i + rng.uniform(-0.16, 0.16, size=len(v_plot))
        ax.scatter(xs, v_plot,
                   s=46, color=theme_color, alpha=0.35,
                   edgecolor=C_DOT_EDGE, linewidth=0.5, zorder=2)
        # Mean ± SEM on trimmed values — VSS audience default.
        m = float(np.mean(sub))
        sem = float(stats.sem(sub))
        m_plot = float(np.clip(m, *ylim))
        # Smaller marker so SEM error bars are visible above tiny
        # within-subject variance (n=18, σ~0.1 → SEM~0.024; a 13pt
        # marker swallowed the bar entirely).
        ax.errorbar(i, m_plot, yerr=sem,
                    fmt='D',
                    mfc=theme_color, mec=C_MEAN_EDGE, mew=1.2,
                    markersize=9,
                    ecolor=C_SEM, elinewidth=2.2,
                    capsize=0, zorder=10)
        # Wilcoxon stars (only printed if significant).
        try:
            _, p = stats.wilcoxon(sub, alternative=alternative,
                                  zero_method='pratt')
        except ValueError:
            p = np.nan
        star = _stars(p)
        ptxt = _p_text(p)
        if star:
            ax.text(i, ylim[1] + 0.02 * (ylim[1] - ylim[0]),
                    star, ha='center', va='bottom',
                    fontsize=18, color=C_STAR_SIG, fontweight='bold')
        if ptxt:
            ax.text(i, ylim[1] + 0.13 * (ylim[1] - ylim[0]),
                    ptxt, ha='center', va='bottom',
                    fontsize=10, color=C_STAR_SIG)

    ax.axhline(0, color='0.6', lw=0.9, ls='--', zorder=0)
    ax.set_xticks(range(len(rois)))
    # Per-ROI label shows n_used; if any subjects were trimmed (|g|
    # > cutoff), append `−k` to be honest about how many were
    # excluded for that ROI.
    # Just show ROI + kept-n. The outlier-drop count (−k) is honest
    # but the audience reads Wilcoxon p-values; the trim only protects
    # the mean+SEM marker from a single bad fit and Wilcoxon doesn't
    # care about it. Mentioning it in the figure footer instead of
    # cluttering each x-tick.
    ax.set_xticklabels([f'{r}\nn={k}'
                        for r, k in zip(rois, n_kept_by_roi)])
    ax.set_ylabel(ylabel)
    ax.set_ylim(ylim[0], ylim[1] + 0.20 * (ylim[1] - ylim[0]))
    ax.set_title(title, pad=14, fontweight='bold')

    sns.despine(ax=ax, offset=6, trim=False)

    # Small in-figure note for the audience: this is mean ± SEM AND
    # we're trimming. VSS visitors expect SEM by default, but they
    # should know about the trim.
    if any(d > 0 for d in n_dropped_by_roi):
        note = (f'Mean ± SEM (markers); Wilcoxon stats. '
                f'Outliers |g| > {OUTLIER_ABS_CUTOFF:g} excluded.')
    else:
        note = 'Mean ± SEM (markers); Wilcoxon stats.'
    ax.text(0.02, -0.30, note,
            transform=ax.transAxes,
            fontsize=10, color='0.35', ha='left', va='top',
            style='italic')

    suffix = f'_{variant}' if variant else ''
    pdf = OUT_DIR / f'{out_stem}{suffix}.pdf'
    svg = OUT_DIR / f'{out_stem}{suffix}.svg'
    pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf, dpi=300, bbox_inches='tight', pad_inches=0.05)
    fig.savefig(svg, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print(f'wrote {pdf}')
    print(f'wrote {svg}')


def main(tsv_path: Path, variant: str = ''):
    df = pd.read_csv(tsv_path, sep='\t')
    # Derived: average distractor gain across HP & LP locations
    # (location-marginalized distractor effect; tests "does a
    # distractor — anywhere — suppress or enhance the PRF, on average?").
    df['g_distractor_avg_static'] = (df['g_HP'] + df['g_LP']) / 2.0
    df['g_distractor_avg_dyn'] = (df['g_HP_dyn'] + df['g_LP_dyn']) / 2.0
    n_sub = df['subject'].nunique()
    print(f'n subjects: {n_sub}, n rows: {len(df)}')

    _plot_one(
        df,
        col='g_HP_minus_g_LP_static',
        ylim=(-0.7, 0.5),
        ylabel=r'g$_{HP}$ − g$_{LP}$  (sustained)',
        title=f'Sustained HP suppression  (n={n_sub})',
        alternative='less',
        out_stem='af_talk_static_HP_suppression',
        theme_color=TALK_HP,
        variant=variant,
    )

    _plot_one(
        df,
        col='g_HP_minus_g_LP_dyn',
        ylim=(-1.4, 1.4),
        ylabel=r'g$_{HP,dyn}$ − g$_{LP,dyn}$  (phasic)',
        title=f'Phasic distractor differential  (n={n_sub})',
        alternative='two-sided',
        out_stem='af_talk_phasic_distractor',
        theme_color=TALK_AF,
        variant=variant,
    )

    _plot_one(
        df,
        col='g_T_dyn',
        ylim=(-1.5, 2.0),
        ylabel=r'g$_{T,dyn}$  (target capture)',
        title=f'Phasic target capture  (n={n_sub})',
        alternative='greater',
        out_stem='af_talk_phasic_target_capture',
        theme_color=TALK_TARGET,
        variant=variant,
    )

    # Location-marginalized "average distractor effect":
    # what does a distractor — at HP or LP — do to the PRF on average?
    # If both HP and LP suppress relative to baseline, the average is
    # negative. Color: stim-blue (the BUILDUP_PALETTE "stim" entry),
    # since this is the location-marginal stimulus-driven effect.
    _plot_one(
        df,
        col='g_distractor_avg_static',
        ylim=(-0.7, 0.5),
        ylabel=r'(g$_{HP}$ + g$_{LP}$) / 2  (sustained)',
        title=f'Sustained distractor effect (HP & LP avg)  (n={n_sub})',
        alternative='less',
        out_stem='af_talk_static_distractor_avg',
        theme_color=TALK_STIM,
        variant=variant,
    )

    _plot_one(
        df,
        col='g_distractor_avg_dyn',
        ylim=(-1.4, 1.4),
        ylabel=r'(g$_{HP,dyn}$ + g$_{LP,dyn}$) / 2  (phasic)',
        title=f'Phasic distractor effect (HP & LP avg)  (n={n_sub})',
        alternative='two-sided',
        out_stem='af_talk_phasic_distractor_avg',
        theme_color=TALK_STIM,
        variant=variant,
    )


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tsv', type=Path, default=DEFAULT_TSV)
    p.add_argument('--variant', default='',
                   help='Suffix appended to all output filenames. '
                        'Use to keep figures from different model variants '
                        '(e.g. sharedSigma vs allSharedSigma) from '
                        'overwriting each other.')
    a = p.parse_args()
    main(a.tsv, variant=a.variant)
