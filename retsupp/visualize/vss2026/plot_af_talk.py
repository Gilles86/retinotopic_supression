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

# Colors — match the talk illustrations' palette (notes/figures/talk/
# make_figures.py BUILDUP_PALETTE). Each panel's color encodes the
# *thing being measured*: HP-static → red, dynamic distractor → blue,
# target → orange.
TALK_HP     = '#D62828'   # red: sustained HP / static suppression
TALK_AF     = '#0077B6'   # blue: dynamic distractor / phasic differential
TALK_TARGET = '#F77F00'   # orange: target capture
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


def _plot_one(df, *, col, ylim, ylabel, title, alternative,
              annotation_text, annotation_xy, annotation_xytext,
              out_stem, theme_color):
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

    for i, roi in enumerate(rois):
        sub = df.loc[df['roi'] == roi, col].dropna().values
        if len(sub) < 2:
            continue
        # Subject dots — panel's theme color, transparent so the swarm
        # reads as a cloud and the mean marker on top stays legible.
        v_plot = np.clip(sub, *ylim)
        xs = i + rng.uniform(-0.16, 0.16, size=len(v_plot))
        ax.scatter(xs, v_plot,
                   s=46, color=theme_color, alpha=0.35,
                   edgecolor=C_DOT_EDGE, linewidth=0.5, zorder=2)
        # Mean ± SEM — large filled diamond in the same theme color, but
        # with a thick dark edge so it pops above the translucent swarm.
        m = float(np.mean(sub))
        sem = float(stats.sem(sub))
        m_plot = float(np.clip(m, *ylim))
        ax.errorbar(i, m_plot, yerr=sem,
                    fmt='D',
                    mfc=theme_color, mec=C_MEAN_EDGE, mew=1.6,
                    markersize=13,
                    ecolor=C_SEM, elinewidth=2.2,
                    capsize=0, zorder=10)
        # Wilcoxon stars (only printed if significant).
        try:
            _, p = stats.wilcoxon(sub, alternative=alternative,
                                  zero_method='pratt')
        except ValueError:
            p = np.nan
        star = _stars(p)
        if star:
            ax.text(i, ylim[1] + 0.03 * (ylim[1] - ylim[0]),
                    star, ha='center', va='bottom',
                    fontsize=18, color=C_STAR_SIG, fontweight='bold')

    ax.axhline(0, color='0.6', lw=0.9, ls='--', zorder=0)
    ax.set_xticks(range(len(rois)))
    ns = [int((df['roi'] == r).sum()) for r in rois]
    ax.set_xticklabels([f'{r}\nn={n}' for r, n in zip(rois, ns)])
    ax.set_ylabel(ylabel)
    ax.set_ylim(ylim[0], ylim[1] + 0.20 * (ylim[1] - ylim[0]))
    ax.set_title(title, pad=14, fontweight='bold')

    if annotation_text:
        ax.annotate(annotation_text,
                    xy=annotation_xy, xytext=annotation_xytext,
                    fontsize=13, ha='left', va='center', color='0.25',
                    arrowprops=dict(arrowstyle='-',
                                    connectionstyle='arc3,rad=0.2',
                                    color='0.4', lw=0.9))

    sns.despine(ax=ax, offset=6, trim=False)

    pdf = OUT_DIR / f'{out_stem}.pdf'
    svg = OUT_DIR / f'{out_stem}.svg'
    pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf, dpi=300, bbox_inches='tight', pad_inches=0.05)
    fig.savefig(svg, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print(f'wrote {pdf}')
    print(f'wrote {svg}')


def main(tsv_path: Path):
    df = pd.read_csv(tsv_path, sep='\t')
    n_sub = df['subject'].nunique()
    print(f'n subjects: {n_sub}, n rows: {len(df)}')

    _plot_one(
        df,
        col='g_HP_minus_g_LP_static',
        ylim=(-0.7, 0.5),
        ylabel=r'g$_{HP}$ − g$_{LP}$  (sustained)',
        title=f'Sustained HP suppression  (n={n_sub})',
        alternative='less',
        annotation_text='HP suppressed\n(< 0)',
        annotation_xy=(2.5, -0.20),
        annotation_xytext=(6.0, -0.50),
        out_stem='af_talk_static_HP_suppression',
        theme_color=TALK_HP,  # red — matches sustained-HP in talk illustrations
    )

    _plot_one(
        df,
        col='g_HP_minus_g_LP_dyn',
        ylim=(-1.4, 1.4),
        ylabel=r'g$_{HP,dyn}$ − g$_{LP,dyn}$  (phasic)',
        title=f'Phasic distractor differential  (n={n_sub})',
        alternative='two-sided',
        annotation_text='Mostly flat\nin early visual',
        annotation_xy=(1.5, 0.1),
        annotation_xytext=(5.5, 1.0),
        out_stem='af_talk_phasic_distractor',
        theme_color=TALK_AF,  # blue — matches dynamic-distractor in illustrations
    )

    _plot_one(
        df,
        col='g_T_dyn',
        ylim=(-1.5, 2.0),
        ylabel=r'g$_{T,dyn}$  (target capture)',
        title=f'Phasic target capture  (n={n_sub})',
        alternative='greater',
        annotation_text='Capture\n(> 0)',
        annotation_xy=(5.5, 0.45),
        annotation_xytext=(2.0, 1.6),
        out_stem='af_talk_phasic_target_capture',
        theme_color=TALK_TARGET,  # orange — matches target in talk illustrations
    )


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tsv', type=Path, default=DEFAULT_TSV)
    a = p.parse_args()
    main(a.tsv)
