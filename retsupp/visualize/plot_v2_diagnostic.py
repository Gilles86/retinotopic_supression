"""Diagnostics for the v2 dynamic-AF model + comparison with v1 (static-AF only).

v2 forward model:
    M(g, t) = 1 + sign · [
        g_HP · A_{HP_run(t)}(g) + g_LP · Σ_{ℓ ≠ HP} A_ℓ(g)            # sustained
      + g_HP_dyn · d_{HP_run(t)}(t) · A_{HP_run(t)}(g)
      + g_LP_dyn · Σ_{ℓ ≠ HP} d_ℓ(t) · A_ℓ(g)                          # dynamic
    ]

5 shared params: σ_AF, g_HP, g_LP, g_HP_dyn, g_LP_dyn.

Pages
-----
1. Cover.
2. Per-ROI distributions of all 5 v2 shared params.
3. HP-vs-LP tests (sustained AND dynamic) per ROI — Wilcoxon, median ±
   95% bootstrap CI, outlier counts.
4. Sustained vs dynamic differential per fit — does
   (g_HP − g_LP) covary with (g_HP_dyn − g_LP_dyn)?
5. v1 (static AF, fits in af_prf_joint_full/) vs v2 sustained gains:
   per-(subject, ROI) scatter for σ_AF, g_HP, g_LP. Are sustained
   estimates stable across the two models?

Inputs:
  /data/ds-retsupp/derivatives/af_prf_joint_dynamic_v2/  (v2 fits)
  /data/ds-retsupp/derivatives/af_prf_joint_full/        (v1 static fits)
"""
from __future__ import annotations

import argparse
import glob
import pickle
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats

ROI_ORDER = ['V1', 'V2', 'V3', 'V3AB', 'hV4', 'LO', 'TO', 'VO']

V1_RE = re.compile(r"sub-(\d+)_roi-([^_]+)_mode-(\w+?)_af-prf-fit\.pkl$")
V2_RE = re.compile(r"sub-(\d+)_roi-([^_]+)_mode-(\w+?)_dyn-v2-af-prf-fit\.pkl$")


def collect_v1(fits_dir: Path) -> pd.DataFrame:
    paths = sorted(glob.glob(str(fits_dir / 'sub-*/sub-*_roi-*_af-prf-fit.pkl')))
    rows = []
    for p in paths:
        m = V1_RE.search(Path(p).name)
        if not m:
            continue
        with open(p, 'rb') as f:
            d = pickle.load(f)
        sh = d.get('shared_pars', {})
        rows.append(dict(
            subject=int(m.group(1)), roi=m.group(2),
            sigma_AF_v1=float(sh.get('sigma_AF', np.nan)),
            g_HP_v1=float(sh.get('g_HP', np.nan)),
            g_LP_v1=float(sh.get('g_LP', np.nan)),
        ))
    return pd.DataFrame(rows)


def collect_v2(fits_dir: Path) -> pd.DataFrame:
    paths = sorted(glob.glob(str(fits_dir / 'sub-*/sub-*_roi-*_dyn-v2-af-prf-fit.pkl')))
    rows = []
    for p in paths:
        m = V2_RE.search(Path(p).name)
        if not m:
            continue
        with open(p, 'rb') as f:
            d = pickle.load(f)
        sh = d.get('shared_pars', {})
        rows.append(dict(
            subject=int(m.group(1)), roi=m.group(2),
            sigma_AF=float(sh.get('sigma_AF', np.nan)),
            g_HP=float(sh.get('g_HP', np.nan)),
            g_LP=float(sh.get('g_LP', np.nan)),
            g_HP_dyn=float(sh.get('g_HP_dyn', np.nan)),
            g_LP_dyn=float(sh.get('g_LP_dyn', np.nan)),
        ))
    df = pd.DataFrame(rows)
    if not df.empty:
        df['g_diff_sus'] = df['g_HP'] - df['g_LP']
        df['g_diff_dyn'] = df['g_HP_dyn'] - df['g_LP_dyn']
    return df


def cover(pdf, v2: pd.DataFrame, v1: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis('off')
    ax.text(0.5, 0.95, 'v2 dynamic AF — diagnostics + v1 comparison',
            ha='center', va='top', fontsize=16, weight='bold',
            transform=ax.transAxes)
    body = (
        f'  v2 fits     : {len(v2)}    '
        f'(subjects: {v2["subject"].nunique()}, ROIs: {v2["roi"].nunique()})\n'
        f'  v1 fits     : {len(v1)}    '
        f'(subjects: {v1["subject"].nunique()}, ROIs: {v1["roi"].nunique()})\n'
        f'\n'
        f'  v2 shared parameters:  σ_AF, g_HP, g_LP, g_HP_dyn, g_LP_dyn\n'
        f'  v1 shared parameters:  σ_AF, g_HP, g_LP\n'
        f'\n'
        f'Sections:\n'
        f'  1. v2 parameter distributions per ROI.\n'
        f'  2. v2 HP-vs-LP tests (sustained AND dynamic differential).\n'
        f'  3. Within-fit  (g_diff_sus vs g_diff_dyn)  scatter — does\n'
        f'     dynamic suppression covary with sustained suppression?\n'
        f'  4. v1 vs v2 per-fit comparison: σ_AF, g_HP, g_LP. The v2\n'
        f'     sustained estimates SHOULD be similar to v1 if the\n'
        f'     dynamic component is doing something independent.\n'
    )
    ax.text(0.04, 0.86, body, ha='left', va='top',
            family='monospace', fontsize=10, transform=ax.transAxes)
    pdf.savefig(fig); plt.close(fig)


def page_distributions(pdf, df: pd.DataFrame):
    metrics = [
        ('sigma_AF',  'σ_AF (deg)',                      0,    5.0,  False),
        ('g_HP',      'g_HP (sustained)',                -2.0, +2.0, True),
        ('g_LP',      'g_LP (sustained)',                -2.0, +2.0, True),
        ('g_HP_dyn',  'g_HP_dyn (per-TR distractor)',    -2.0, +2.0, True),
        ('g_LP_dyn',  'g_LP_dyn (per-TR distractor)',    -2.0, +2.0, True),
        ('g_diff_sus','g_HP − g_LP (sustained diff)',    -1.5, +1.5, True),
        ('g_diff_dyn','g_HP_dyn − g_LP_dyn (dyn diff)',  -1.5, +1.5, True),
    ]
    df = df.copy()
    df['roi'] = pd.Categorical(
        df['roi'], categories=[r for r in ROI_ORDER if r in df['roi'].unique()],
        ordered=True,
    )
    rois_present = list(df['roi'].cat.categories)

    fig, axes = plt.subplots(3, 3, figsize=(15, 14))
    axes = axes.ravel()
    for ax, (col, label, ylo, yhi, sym) in zip(axes, metrics):
        d_plot = df.assign(**{f'{col}_clip': df[col].clip(ylo, yhi)})
        sns.stripplot(data=d_plot, x='roi', y=f'{col}_clip', ax=ax,
                       hue='roi', palette='Set2', legend=False,
                       jitter=0.2, alpha=0.75, size=4)
        sns.boxplot(data=d_plot, x='roi', y=f'{col}_clip', ax=ax,
                     width=0.4, fill=False, color='k', linewidth=1,
                     showfliers=False)
        for i, roi in enumerate(rois_present):
            x = df.loc[df['roi'] == roi, col].dropna().values
            n_lo = int((x < ylo).sum())
            n_hi = int((x > yhi).sum())
            if n_lo:
                ax.annotate(f'↓{n_lo}', xy=(i, ylo + 0.02 * (yhi - ylo)),
                             ha='center', va='bottom', fontsize=7,
                             color='C3', weight='bold')
            if n_hi:
                ax.annotate(f'↑{n_hi}', xy=(i, yhi - 0.02 * (yhi - ylo)),
                             ha='center', va='top', fontsize=7,
                             color='C3', weight='bold')
        if sym:
            ax.axhline(0, color='gray', lw=0.4, ls='--')
        ax.set_ylim(ylo, yhi)
        ax.set_xlabel('')
        ax.set_ylabel(label, fontsize=9)
        ax.set_title(label, fontsize=10)
    for ax in axes[len(metrics):]:
        ax.axis('off')
    fig.suptitle('v2 parameter distributions per ROI '
                 '(y-clipped; ↑/↓ = #fits past limit)',
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    pdf.savefig(fig); plt.close(fig)


def page_hp_vs_lp_v2(pdf, df: pd.DataFrame):
    """Side-by-side g_diff_sus and g_diff_dyn HP-vs-LP test."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    rois = [r for r in ROI_ORDER if r in df['roi'].unique()]
    rng = np.random.default_rng(0)
    n_boot = 2000
    YLIM = 1.5

    for ax, metric, label in zip(
            axes,
            ['g_diff_sus', 'g_diff_dyn'],
            ['SUSTAINED:  g_HP − g_LP',
             'DYNAMIC:    g_HP_dyn − g_LP_dyn']):
        d_plot = df.assign(_clip=df[metric].clip(-YLIM, YLIM))
        sns.stripplot(data=d_plot, x='roi', y='_clip', ax=ax,
                       order=rois,
                       hue='roi', palette='Set2', legend=False,
                       jitter=0.18, alpha=0.7, size=5)
        ax.axhline(0, color='gray', lw=0.7, ls='--')
        for i, roi in enumerate(rois):
            x = df.loc[df['roi'] == roi, metric].dropna().values
            if len(x) < 3:
                continue
            med = float(np.median(x))
            boots = rng.choice(x, size=(n_boot, len(x)), replace=True)
            boot_meds = np.median(boots, axis=1)
            lo, hi = np.quantile(boot_meds, [0.025, 0.975])
            ax.errorbar([i], [med], yerr=[[med - lo], [hi - med]],
                         fmt='s', color='k', markersize=10, ecolor='k',
                         elinewidth=2, capsize=6, zorder=10)
            try:
                _, p_w = stats.wilcoxon(x, alternative='less',
                                          zero_method='pratt')
            except ValueError:
                p_w = np.nan
            sig = ('***' if p_w < 0.001 else
                    '**' if p_w < 0.01 else
                    '*' if p_w < 0.05 else '')
            col = 'C3' if (np.isfinite(p_w) and p_w < 0.05) else '0.4'
            ax.text(i, YLIM + 0.05,
                     f'p_w={p_w:.3f}{sig}\n<0: {(x<0).mean()*100:.0f}%',
                     ha='center', va='bottom', fontsize=8, color=col)
        ax.set_ylim(-YLIM - 0.05, YLIM + 0.40)
        ax.set_xlabel('')
        ax.set_ylabel(f'{label.split(":")[1].strip()}')
        ax.set_title(label, fontsize=11)
    fig.suptitle('v2 HP-vs-LP tests: sustained vs dynamic differential per ROI',
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    pdf.savefig(fig); plt.close(fig)


def page_sustained_vs_dynamic_scatter(pdf, df: pd.DataFrame):
    """Per-ROI scatter of g_diff_sus vs g_diff_dyn."""
    rois = [r for r in ROI_ORDER if r in df['roi'].unique()]
    n = len(rois)
    ncol = 4; nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3.5 * nrow),
                              sharex=True, sharey=True)
    axes = np.atleast_2d(axes)
    LIM = 2.0
    for i, roi in enumerate(rois):
        ax = axes[i // ncol, i % ncol]
        sub = df[df['roi'] == roi]
        x = sub['g_diff_sus'].clip(-LIM, LIM).values
        y = sub['g_diff_dyn'].clip(-LIM, LIM).values
        m = np.isfinite(x) & np.isfinite(y)
        ax.scatter(x[m], y[m], s=40, alpha=0.85, color='C0',
                    edgecolor='k', linewidth=0.4)
        if m.sum() >= 4:
            r, p = stats.spearmanr(sub['g_diff_sus'].dropna(),
                                     sub['g_diff_dyn'].dropna(),
                                     nan_policy='omit')
        else:
            r = p = np.nan
        ax.axhline(0, color='gray', lw=0.4, ls=':')
        ax.axvline(0, color='gray', lw=0.4, ls=':')
        ax.set_xlim(-LIM, LIM); ax.set_ylim(-LIM, LIM)
        ax.set_xlabel('g_diff_sus', fontsize=8)
        ax.set_ylabel('g_diff_dyn', fontsize=8)
        sig = '*' if (np.isfinite(p) and p < 0.05) else ''
        ax.set_title(f'{roi}  r_s={r:+.2f}, p={p:.3f}{sig}',
                      fontsize=9, color='C3' if sig else 'k')
    for j in range(n, nrow * ncol):
        axes[j // ncol, j % ncol].axis('off')
    fig.suptitle('Sustained vs dynamic HP-LP differential per fit\n'
                 '(do they covary? both should track HP-suppression if\n'
                 'they measure the same underlying attentional state.)',
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    pdf.savefig(fig); plt.close(fig)


def page_total_hp_vs_lp(pdf, df: pd.DataFrame):
    """Per-ROI Wilcoxon test on TOTAL HP-LP differential.

    metric = (g_HP + g_HP_dyn) − (g_LP + g_LP_dyn)
           = g_diff_sus + g_diff_dyn

    Tests whether the COMBINED (sustained + dynamic) gain at HP is
    smaller than at LP. Crucially, in early visual cortex (V2/V3) the
    sustained suppression and dynamic response go OPPOSITE directions
    and the total cancels; in V3AB/VO they add and the total reaches
    high significance.
    """
    df = df.copy()
    df['g_diff_total'] = df['g_diff_sus'] + df['g_diff_dyn']
    rois = [r for r in ROI_ORDER if r in df['roi'].unique()]
    YLIM = 2.0

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    rng = np.random.default_rng(0)
    n_boot = 2000

    panels = [
        ('g_diff_sus',   'SUSTAINED:  g_HP − g_LP',                df),
        ('g_diff_dyn',   'DYNAMIC:    g_HP_dyn − g_LP_dyn',        df),
        ('g_diff_total', 'TOTAL:      sustained + dynamic',         df),
    ]
    for ax, (metric, label, sub_df) in zip(axes, panels):
        d_plot = sub_df.assign(_clip=sub_df[metric].clip(-YLIM, YLIM))
        sns.stripplot(data=d_plot, x='roi', y='_clip', ax=ax,
                       order=rois,
                       hue='roi', palette='Set2', legend=False,
                       jitter=0.18, alpha=0.7, size=5)
        ax.axhline(0, color='gray', lw=0.7, ls='--')
        for i, roi in enumerate(rois):
            x = sub_df.loc[sub_df['roi'] == roi, metric].dropna().values
            if len(x) < 3:
                continue
            med = float(np.median(x))
            boots = rng.choice(x, size=(n_boot, len(x)), replace=True)
            boot_meds = np.median(boots, axis=1)
            lo, hi = np.quantile(boot_meds, [0.025, 0.975])
            ax.errorbar([i], [med], yerr=[[med - lo], [hi - med]],
                         fmt='s', color='k', markersize=10, ecolor='k',
                         elinewidth=2, capsize=6, zorder=10)
            try:
                _, p_w = stats.wilcoxon(x, alternative='less',
                                          zero_method='pratt')
            except ValueError:
                p_w = np.nan
            sig = ('***' if p_w < 0.001 else
                    '**' if p_w < 0.01 else
                    '*' if p_w < 0.05 else '')
            col = 'C3' if (np.isfinite(p_w) and p_w < 0.05) else '0.4'
            ax.text(i, YLIM + 0.05,
                     f'p={p_w:.3f}{sig}\n<0: {(x<0).mean()*100:.0f}%',
                     ha='center', va='bottom', fontsize=8, color=col)
        ax.set_ylim(-YLIM - 0.05, YLIM + 0.45)
        ax.set_xlabel('')
        ax.set_ylabel(label.split(':')[1].strip())
        ax.set_title(label, fontsize=11)
    fig.suptitle(
        'HP−LP differential — sustained, dynamic, and total per ROI.\n'
        'In V2/V3, sustained (negative) and dynamic (positive) cancel '
        '→ total ~0.   In V3AB/VO, they add → total strongly negative.',
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    pdf.savefig(fig); plt.close(fig)


def page_dynamic_capture_test(pdf, df: pd.DataFrame):
    """Per-ROI test: is the dynamic gain ≠ 0?

    Tests g_dyn_avg = ½(g_HP_dyn + g_LP_dyn) against zero (Wilcoxon).
    Significant POSITIVE = attentional CAPTURE.
    Significant NEGATIVE = dynamic SUPPRESSION of distractor.
    Reports both for ALL fits and for σ-filtered (σ_AF < 5°) fits.
    """
    df = df.copy()
    df['g_dyn_avg'] = 0.5 * (df['g_HP_dyn'] + df['g_LP_dyn'])
    rois = [r for r in ROI_ORDER if r in df['roi'].unique()]
    YLIM = 2.5

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    rng = np.random.default_rng(0)
    n_boot = 2000
    for ax, sub_df, label in zip(
            axes,
            [df, df[df['sigma_AF'] < 5.0]],
            ['ALL fits', 'σ_AF < 5°  (drops degenerate)']):
        d_plot = sub_df.assign(_clip=sub_df['g_dyn_avg'].clip(-YLIM, YLIM))
        sns.stripplot(data=d_plot, x='roi', y='_clip', ax=ax,
                       order=rois,
                       hue='roi', palette='Set2', legend=False,
                       jitter=0.18, alpha=0.7, size=5)
        ax.axhline(0, color='gray', lw=0.7, ls='--')
        for i, roi in enumerate(rois):
            x = sub_df.loc[sub_df['roi'] == roi, 'g_dyn_avg'].dropna().values
            if len(x) < 4:
                continue
            med = float(np.median(x))
            boots = rng.choice(x, size=(n_boot, len(x)), replace=True)
            boot_meds = np.median(boots, axis=1)
            lo, hi = np.quantile(boot_meds, [0.025, 0.975])
            ax.errorbar([i], [med], yerr=[[med - lo], [hi - med]],
                         fmt='s', color='k', markersize=10, ecolor='k',
                         elinewidth=2, capsize=6, zorder=10)
            try:
                _, p_two = stats.wilcoxon(x, alternative='two-sided',
                                            zero_method='pratt')
            except ValueError:
                p_two = np.nan
            sig = ('***' if p_two < 0.001 else
                    '**' if p_two < 0.01 else
                    '*' if p_two < 0.05 else '')
            col = ('C2' if (np.isfinite(p_two) and p_two < 0.05 and med > 0)
                   else 'C3' if (np.isfinite(p_two) and p_two < 0.05 and med < 0)
                   else '0.4')
            direction = ('+' if med > 0 else '−')
            ax.text(i, YLIM + 0.10,
                     f'p={p_two:.3f}{sig}\n{direction} n={len(x)}',
                     ha='center', va='bottom', fontsize=8, color=col)
        ax.set_ylim(-YLIM - 0.05, YLIM + 0.50)
        ax.set_xlabel('')
        ax.set_ylabel('g_dyn_avg = ½(g_HP_dyn + g_LP_dyn)')
        ax.set_title(label, fontsize=11)
    fig.suptitle(
        'Is there a dynamic distractor effect?  '
        'Wilcoxon test of g_dyn_avg ≠ 0 per ROI.\n'
        'Green = significant CAPTURE (gain > 0), '
        'Red = significant SUPPRESSION (gain < 0).',
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    pdf.savefig(fig); plt.close(fig)


def page_v1_vs_v2(pdf, v1: pd.DataFrame, v2: pd.DataFrame):
    """Per-ROI scatter of v1 vs v2 sustained-only params."""
    merged = v2.merge(v1, on=['subject', 'roi'], how='inner')
    print(f'v1 ∩ v2: {len(merged)} (subject, ROI) pairs')
    rois = [r for r in ROI_ORDER if r in merged['roi'].unique()]
    metrics = [
        ('sigma_AF', 'sigma_AF_v1', 'σ_AF',  0, 10),
        ('g_HP',     'g_HP_v1',     'g_HP',  -2, 2),
        ('g_LP',     'g_LP_v1',     'g_LP',  -2, 2),
    ]
    fig, axes = plt.subplots(len(metrics), len(rois),
                              figsize=(2.5 * len(rois) + 1, 2.8 * len(metrics)),
                              sharex='row', sharey='row')
    if len(metrics) == 1:
        axes = axes[None, :]
    for r_idx, (v2_col, v1_col, label, lo, hi) in enumerate(metrics):
        for c_idx, roi in enumerate(rois):
            ax = axes[r_idx, c_idx]
            sub = merged[merged['roi'] == roi]
            x = sub[v1_col].clip(lo, hi).values
            y = sub[v2_col].clip(lo, hi).values
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() < 2:
                ax.text(0.5, 0.5, 'no data',
                         ha='center', va='center', transform=ax.transAxes)
                ax.set_xticks([]); ax.set_yticks([])
                continue
            ax.scatter(x[m], y[m], s=25, alpha=0.85, color='C0',
                        edgecolor='k', linewidth=0.3)
            ax.plot([lo, hi], [lo, hi], 'k:', lw=0.7)
            r, p = stats.spearmanr(sub[v1_col], sub[v2_col],
                                      nan_policy='omit')
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
            ax.set_aspect('equal')
            ax.set_title(f'{roi}\nr_s={r:+.2f}', fontsize=8)
            if r_idx == len(metrics) - 1:
                ax.set_xlabel(f'v1 {label}', fontsize=8)
            if c_idx == 0:
                ax.set_ylabel(f'v2 {label}', fontsize=8)
            ax.tick_params(labelsize=7)
    fig.suptitle('v1 (static-AF only) vs v2 (with dynamic) sustained params  '
                 '— diagonal = identical fit',
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--v1-dir', type=Path,
                        default=Path('/data/ds-retsupp/derivatives/af_prf_joint_full'))
    parser.add_argument('--v2-dir', type=Path,
                        default=Path('/data/ds-retsupp/derivatives/af_prf_joint_dynamic_v2'))
    parser.add_argument('--out', type=Path,
                        default=Path('notes/af_v2_diagnostic.pdf'))
    args = parser.parse_args()

    v2 = collect_v2(args.v2_dir)
    print(f'Loaded {len(v2)} v2 fits ({v2["subject"].nunique()} subjects).')
    v1 = collect_v1(args.v1_dir)
    print(f'Loaded {len(v1)} v1 fits ({v1["subject"].nunique()} subjects).')

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(args.out) as pdf:
        cover(pdf, v2, v1)
        page_distributions(pdf, v2)
        page_hp_vs_lp_v2(pdf, v2)
        page_total_hp_vs_lp(pdf, v2)
        page_dynamic_capture_test(pdf, v2)
        page_sustained_vs_dynamic_scatter(pdf, v2)
        if not v1.empty:
            page_v1_vs_v2(pdf, v1, v2)
    print(f'Wrote {args.out}')


if __name__ == '__main__':
    main()
