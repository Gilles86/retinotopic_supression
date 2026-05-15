"""AF result figure for Docky's VSS2026 talk.

Three-panel layout, talk-grade (16:9 slide, Helvetica, despined,
direct annotations). Each panel: one quantity per ROI; subjects
shown as small jittered dots, group median as a horizontal bar
colored by Wilcoxon-against-zero significance and effect direction.
Per-ROI n annotated under each x-tick (so an n=12 FEF and n=18 V1
are visually distinguishable). Significance stars above the median
bar.

Panels
------
A. Sustained HP suppression — `g_HP - g_LP` per ROI (one-sided <0)
B. Phasic distractor differential — `g_HP_dyn - g_LP_dyn` (two-sided)
C. Phasic target capture — `g_T_dyn` per ROI (one-sided >0)

Source: ``notes/data/af_dog_v3_target_sharedSigma_parameters.tsv``
(canonical AF, pulled via the model-registry resolver — pSig>0.5,
HRF-fixed commit 488e0a9). Re-aggregate the TSV after new AF lands
via:

    python -m retsupp.visualize.paper.plot_af_results \\
        --analysis af_dynamic_sharedSigma \\
        --bids-folder /shares/zne.uzh/gdehol/ds-retsupp \\
        --summary-tsv notes/data/af_dog_v3_target_sharedSigma_parameters.tsv \\
        --rois all
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
DEFAULT_OUT = REPO / 'notes' / 'figures' / 'af_talk_dog_dyn_v3_sharedSigma.pdf'

ROI_ORDER = ['V1', 'V2', 'V3', 'V3AB', 'hV4', 'LO', 'TO', 'VO',
             'IPS', 'SPL1', 'FEF']

# Muted palette (per scientific-figures skill — no tab10 default).
C_SUPPRESSION = '#5D8C3F'   # green: significant suppression direction
C_CAPTURE     = '#C44E52'   # red: significant capture / wrong-direction
C_NS          = '#9C9C9C'   # gray: not significant
C_DOT         = '#4D4D4D'   # dark gray: individual subjects (the data)


def _apply_rcparams_talk():
    """Talk-scaled house style. Bigger fonts than paper figure."""
    mpl.rcParams.update({
        'font.family': 'Helvetica',
        'font.sans-serif': ['Helvetica', 'Helvetica Neue',
                            'TeX Gyre Heros', 'Arial'],
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        'mathtext.fontset': 'stixsans',
        'axes.linewidth': 1.0,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.labelpad': 6,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.major.size': 3.5,
        'ytick.major.size': 3.5,
        'xtick.major.width': 1.0,
        'ytick.major.width': 1.0,
        'lines.linewidth': 1.5,
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
    sns.set_context('talk')


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


def _panel(ax, df, *, col, ylim, ylabel, title, alternative,
           annotation_text=None, annotation_xy=None, annotation_xytext=None):
    """One quantity × ROIs: subjects as dots, median as colored bar.

    Median-bar color encodes (significant ↔ NS) × (negative ↔ positive)
    direction. `alternative` is the Wilcoxon alternative used to compute
    p-values for the in-panel stars.
    """
    rois = [r for r in ROI_ORDER if r in df['roi'].unique()]
    rng = np.random.default_rng(seed=42)
    for i, roi in enumerate(rois):
        sub = df.loc[df['roi'] == roi, col].dropna().values
        if len(sub) < 2:
            continue
        v = np.clip(sub, *ylim)
        xs = i + rng.uniform(-0.18, 0.18, size=len(v))
        ax.scatter(xs, v, s=22, color=C_DOT, alpha=0.55,
                   edgecolor='white', linewidth=0.4, zorder=2)
        med = float(np.median(sub))   # median on raw values, not clipped
        med_plot = float(np.median(v))  # clipped median for placement
        try:
            _, p = stats.wilcoxon(sub, alternative=alternative,
                                  zero_method='pratt')
        except ValueError:
            p = np.nan
        if np.isfinite(p) and p < 0.05:
            c = C_SUPPRESSION if med < 0 else C_CAPTURE
        else:
            c = C_NS
        ax.hlines(med_plot, i - 0.32, i + 0.32, color=c, lw=3.5, zorder=5)
        star = _stars(p)
        if star:
            ax.text(i, ylim[1] + 0.04 * (ylim[1] - ylim[0]),
                    star, ha='center', va='bottom',
                    fontsize=14, color=c, fontweight='bold')

    ax.axhline(0, color='0.7', lw=0.7, ls='--', zorder=0)
    ax.set_xticks(range(len(rois)))
    ns = [int((df['roi'] == r).sum()) for r in rois]
    ax.set_xticklabels([f'{r}\nn={n}' for r, n in zip(rois, ns)])
    ax.set_ylabel(ylabel)
    ax.set_ylim(ylim[0], ylim[1] + 0.18 * (ylim[1] - ylim[0]))
    ax.set_title(title, pad=14)
    if annotation_text and annotation_xy and annotation_xytext:
        ax.annotate(annotation_text,
                    xy=annotation_xy, xytext=annotation_xytext,
                    fontsize=11, ha='left', va='center', color='0.3',
                    arrowprops=dict(arrowstyle='-',
                                    connectionstyle='arc3,rad=0.2',
                                    color='0.3', lw=0.7))


def main(tsv_path: Path, out_path: Path):
    _apply_rcparams_talk()
    df = pd.read_csv(tsv_path, sep='\t')
    n_sub = df['subject'].nunique()
    print(f'n subjects: {n_sub}, n rows: {len(df)}')

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 5.0),
                              constrained_layout=True)

    # Panel A — sustained HP suppression (the headline).
    _panel(
        axes[0], df,
        col='g_HP_minus_g_LP_static',
        ylim=(-0.7, 0.5),
        ylabel=r'g$_{HP}$ − g$_{LP}$  (sustained)',
        title='Sustained HP suppression',
        alternative='less',
        annotation_text='HP suppressed\n(< 0)',
        annotation_xy=(2.5, -0.18),
        annotation_xytext=(5.5, -0.5),
    )

    # Panel B — phasic HP-vs-LP differential (the null story in early visual).
    _panel(
        axes[1], df,
        col='g_HP_minus_g_LP_dyn',
        ylim=(-1.4, 1.4),
        ylabel=r'g$_{HP,dyn}$ − g$_{LP,dyn}$  (phasic)',
        title='Phasic distractor differential',
        alternative='two-sided',
        annotation_text='Flat in\nearly visual',
        annotation_xy=(1.5, 0.1),
        annotation_xytext=(5.0, 1.1),
    )

    # Panel C — phasic target capture (the dynamic positive effect).
    _panel(
        axes[2], df,
        col='g_T_dyn',
        ylim=(-1.5, 2.0),
        ylabel=r'g$_{T,dyn}$  (target capture)',
        title='Phasic target capture',
        alternative='greater',
        annotation_text='Capture\n(> 0)',
        annotation_xy=(5.5, 0.4),
        annotation_xytext=(2.0, 1.5),
    )

    # Despine each panel: offset + trim per skill guidance.
    for ax in axes:
        sns.despine(ax=ax, offset=5, trim=False)
        # trim=False because trim=True interacts badly with axhline ylim;
        # we set ylim explicitly above.

    # Panel letters (skill convention: top-left, outside axes, bold sans).
    for i, ax in enumerate(axes):
        ax.text(-0.16, 1.04, 'ABC'[i],
                transform=ax.transAxes,
                fontsize=18, fontweight='bold',
                va='bottom', ha='right')

    # Single short suptitle — talk gets one (paper figures don't).
    fig.suptitle(
        f'Attention-field modulation across the visual hierarchy '
        f'(n={n_sub})',
        fontsize=15, weight='bold', y=1.02,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
    fig.savefig(out_path.with_suffix('.svg'),
                bbox_inches='tight', pad_inches=0.05)
    print(f'wrote {out_path}')
    print(f'wrote {out_path.with_suffix(".svg")}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tsv', type=Path, default=DEFAULT_TSV)
    p.add_argument('--out', type=Path, default=DEFAULT_OUT)
    a = p.parse_args()
    main(a.tsv, a.out)
