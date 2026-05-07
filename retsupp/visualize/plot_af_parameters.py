"""Distributions, correlations, and joint plots of joint-AF + PRF fit parameters.

Loads all per-(subject, ROI) ``af-prf-fit.pkl`` pickles produced by
:mod:`retsupp.modeling.fit_af_prf_braincoder`, extracts the SHARED-AF
parameters (``sigma_AF``, ``g_HP``, ``g_LP``) and the per-fit summary
statistics (mean R², n voxels), and produces a reusable diagnostic PDF:

  • Per-ROI marginals of σ_AF, g_HP, g_LP, and HP-LP differential.
  • Pairwise scatter / Pearson r matrix of the three shared AF
    parameters across fits — flags identifiability problems
    (e.g. σ_AF anticorrelated with gains, gains correlated with each
    other).
  • Per-fit dot plot of (σ_AF, g_HP, g_LP) sorted by ROI then subject.
  • g_HP vs g_LP scatter coloured by ROI — diagonal would mean "no
    HP-specific effect"; deviations from y=x measure HP differential.

Usage
-----
    python -m retsupp.visualize.plot_af_parameters \\
        --fits-dir /data/ds-retsupp/derivatives/af_prf_joint_full \\
        --out notes/af_parameters.pdf
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

ROI_ORDER = ['V1', 'V2', 'V3', 'V3AB', 'hV4', 'LO', 'TO', 'VO']

PARSE_RE = re.compile(r"sub-(\d+)_roi-([^_]+)_mode-(\w+?)_af-prf-fit\.pkl$")


def collect_records(fits_dir: Path) -> pd.DataFrame:
    """Walk a `derivatives/af_prf_joint*` tree and return one row per fit."""
    paths = sorted(glob.glob(str(fits_dir / 'sub-*/sub-*_roi-*_af-prf-fit.pkl')))
    rows = []
    for p in paths:
        m = PARSE_RE.search(Path(p).name)
        if not m:
            continue
        sub = int(m.group(1))
        roi = m.group(2)
        mode = m.group(3)
        try:
            with open(p, 'rb') as f:
                d = pickle.load(f)
        except Exception as e:
            print(f"  WARN: cannot load {p}: {e}")
            continue
        sh = d.get('shared_pars', {})
        r2 = d.get('r2', None)
        r2_arr = np.asarray(r2.values if hasattr(r2, 'values') else r2)
        g_hp = float(sh.get('g_HP', np.nan))
        g_lp = float(sh.get('g_LP', np.nan))
        # Log-ratio of effective modulation factors at the AF centers
        # (M = 1 + g, clamped at a small positive eps so negative-gain
        # outliers don't blow up the metric). Scale-invariant in g, so
        # comparable across ROIs with different overall AF strength.
        eps = 0.05
        m_hp = max(1.0 + g_hp, eps)
        m_lp = max(1.0 + g_lp, eps)
        log_ratio = float(np.log(m_hp / m_lp))
        rows.append(dict(
            subject=sub, roi=roi, mode=mode,
            n_voxels=len(d.get('fit_pars', [])),
            sigma_AF=float(sh.get('sigma_AF', np.nan)),
            g_HP=g_hp,
            g_LP=g_lp,
            g_diff=g_hp - g_lp,
            g_avg=0.5 * (g_hp + g_lp),
            log_ratio=log_ratio,
            r2_mean=float(np.nanmean(r2_arr)) if r2_arr.size else np.nan,
            r2_med=float(np.nanmedian(r2_arr)) if r2_arr.size else np.nan,
        ))
    df = pd.DataFrame(rows)
    if not df.empty:
        df['roi'] = pd.Categorical(
            df['roi'],
            categories=[r for r in ROI_ORDER if r in df['roi'].unique()] +
                       sorted(set(df['roi']) - set(ROI_ORDER)),
            ordered=True,
        )
        df = df.sort_values(['roi', 'subject']).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

def cover_page(pdf, df, args):
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis('off')
    ax.text(0.5, 0.95, 'Joint AF + PRF — parameter diagnostics',
            ha='center', va='top', fontsize=18, weight='bold',
            transform=ax.transAxes)
    ax.text(0.5, 0.91, f'fits-dir = {args.fits_dir}',
            ha='center', va='top', fontsize=8, style='italic',
            transform=ax.transAxes)
    n_total = len(df)
    by_roi = df.groupby('roi', observed=True).size().to_dict()
    body = [
        f'  total fits             : {n_total}',
        f'  unique subjects        : {df["subject"].nunique() if n_total else 0}',
        f'  modes                  : {", ".join(sorted(df["mode"].unique())) if n_total else "-"}',
        '',
        '  fits per ROI:',
    ]
    for roi in [r for r in ROI_ORDER if r in by_roi] + \
               sorted(set(by_roi) - set(ROI_ORDER)):
        body.append(f'    {roi:<6}: {by_roi[roi]}')
    ax.text(0.05, 0.84, '\n'.join(body), ha='left', va='top',
            family='monospace', fontsize=10, transform=ax.transAxes)
    note = (
        '\nDiagnostics in this report:\n'
        '  • Per-ROI distributions of σ_AF, g_HP, g_LP, g_HP - g_LP, R².\n'
        '  • Pairwise scatter + Pearson r of the 3 shared AF params.\n'
        '  • Per-fit dot plot (one row per (subject, ROI)).\n'
        '  • g_HP vs g_LP coloured by ROI — diagonal = no HP differential.\n'
    )
    ax.text(0.05, 0.40, note, ha='left', va='top',
            family='monospace', fontsize=9, transform=ax.transAxes)
    pdf.savefig(fig); plt.close(fig)


def page_distributions(pdf, df):
    """Per-ROI strip+box for each of σ_AF, g_HP, g_LP, g_diff, R².

    Y-axis clipped per metric to focus on the bulk of the distribution
    (a handful of fits with σ_AF≈9° / |g|≈5 would otherwise compress
    everyone else into a thin band). Out-of-range counts shown as red
    ↑/↓ at the panel edges.
    """
    metrics = [
        ('sigma_AF', 'σ_AF (deg)',                       0,    5.0,   False),
        ('g_HP',    'g_HP',                              -2.0, +2.0,  True),
        ('g_LP',    'g_LP',                              -2.0, +2.0,  True),
        ('g_diff',  'g_HP − g_LP  (HP differential)',    -1.5, +1.5,  True),
        ('g_avg',   '½(g_HP + g_LP)  (overall AF strength)', -2.0, +2.0, True),
        ('r2_mean', 'mean voxel R²',                      0,    0.30, False),
    ]
    rois_with_data = [r for r in df['roi'].cat.categories
                      if (df['roi'] == r).any()]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, (col, label, ylo, yhi, symmetric) in zip(axes.ravel(), metrics):
        d = df.copy()
        d[col + '_clip'] = d[col].clip(ylo, yhi)
        sns.stripplot(data=d, x='roi', y=col + '_clip', ax=ax,
                      hue='roi', palette='Set2', legend=False,
                      jitter=0.2, alpha=0.8, size=5)
        sns.boxplot(data=d, x='roi', y=col + '_clip', ax=ax,
                    width=0.4, fill=False, color='k', linewidth=1,
                    showfliers=False)
        # Out-of-range outlier annotations.
        for i, roi in enumerate(rois_with_data):
            x = df.loc[df['roi'] == roi, col].dropna().values
            n_lo = int((x < ylo).sum())
            n_hi = int((x > yhi).sum())
            if n_lo:
                ax.annotate(f'↓ {n_lo}', xy=(i, ylo + 0.02 * (yhi - ylo)),
                            ha='center', va='bottom', fontsize=8,
                            color='C3', weight='bold')
            if n_hi:
                ax.annotate(f'↑ {n_hi}', xy=(i, yhi - 0.02 * (yhi - ylo)),
                            ha='center', va='top', fontsize=8,
                            color='C3', weight='bold')
        if symmetric:
            ax.axhline(0, color='gray', lw=0.5, ls='--')
        ax.set_ylim(ylo, yhi)
        ax.set_xlabel('')
        ax.set_ylabel(label)
        ax.set_title(label, fontsize=10)
    fig.suptitle('Per-ROI parameter distributions across (subject) fits  '
                 '(y-axis clipped; red ↑/↓ = #fits past the limit)',
                 fontsize=11)
    fig.tight_layout()
    pdf.savefig(fig); plt.close(fig)


def page_sigma_gain_correlation(pdf, df):
    """Per-ROI σ_AF vs gain scatter — visualizes the identifiability problem.

    If gains and σ_AF are POSITIVELY correlated within a ROI across
    subjects, that's the classic "uniform field with big gain" trade-off:
    when σ_AF is forced larger, gains compensate to maintain modulation
    magnitude.  Negative correlation could also appear via the same path.
    A clean ROI-coherent attention-field fit would have σ_AF and gains
    each clustering at modal values across subjects.
    """
    if df.empty:
        return

    rois = list(df['roi'].cat.categories)
    n_rois = len(rois)
    if n_rois == 0:
        return

    metrics = [
        ('g_HP',    '|g_HP|'),
        ('g_LP',    '|g_LP|'),
        ('g_avg',   '|g_avg|  =  ½|g_HP + g_LP|'),
    ]
    fig, axes = plt.subplots(
        len(metrics), 1, figsize=(11, 4 * len(metrics)),
        sharex=True,
    )
    palette = dict(zip(rois, sns.color_palette('Set2', n_rois)))

    for ax, (col, label) in zip(axes, metrics):
        # Plot per ROI.
        for roi in rois:
            sub = df[df['roi'] == roi]
            if len(sub) < 2:
                continue
            ax.scatter(sub['sigma_AF'], np.abs(sub[col]),
                       color=palette[roi], s=70, edgecolor='k',
                       linewidth=0.4, alpha=0.85, label=f'{roi}')
        # Annotate per-ROI Spearman r at the top of each panel.
        ann_lines = []
        from scipy import stats
        for roi in rois:
            sub = df[df['roi'] == roi].dropna(subset=['sigma_AF', col])
            if len(sub) < 4:
                continue
            r, p = stats.spearmanr(sub['sigma_AF'], np.abs(sub[col]))
            ann_lines.append(
                f'{roi}: r_s={r:+.2f}, p={p:.3f}, n={len(sub)}')
        ax.text(
            0.99, 0.99, '\n'.join(ann_lines),
            ha='right', va='top', fontsize=7,
            family='monospace', transform=ax.transAxes,
            bbox=dict(boxstyle='round,pad=0.3', fc='white',
                       ec='0.7', alpha=0.85),
        )
        ax.set_ylabel(label)
        ax.axhline(0, color='gray', lw=0.4, ls='--')
        ax.set_yscale('symlog', linthresh=0.5)
    axes[-1].set_xlabel('σ_AF  (deg)')
    axes[0].legend(loc='upper left', fontsize=8, ncol=4,
                    bbox_to_anchor=(0, 1.18), frameon=False)
    axes[0].set_title(
        'Identifiability check: σ_AF vs |gain| per ROI across subjects.\n'
        'Strong σ_AF↔|gain| correlation = trade-off; '
        'flat clouds = ROI-coherent fit.',
        fontsize=11,
    )
    fig.tight_layout()
    pdf.savefig(fig); plt.close(fig)


def page_pairwise(pdf, df):
    """Pairwise scatter + Pearson r of the 3 shared AF params."""
    cols = ['sigma_AF', 'g_HP', 'g_LP']
    g = sns.pairplot(
        df, vars=cols, hue='roi', palette='Set2',
        plot_kws=dict(alpha=0.8, s=35), diag_kws=dict(common_norm=False),
        height=2.5, aspect=1.1,
    )
    # Annotate Pearson r in upper triangle.
    for i, ci in enumerate(cols):
        for j, cj in enumerate(cols):
            if j > i:
                ax = g.axes[i, j]
                ax.cla(); ax.set_xticks([]); ax.set_yticks([])
                m = df[[ci, cj]].dropna()
                if len(m) > 2:
                    r = np.corrcoef(m[ci], m[cj])[0, 1]
                    ax.text(0.5, 0.5, f'r = {r:.3f}\nN = {len(m)}',
                            ha='center', va='center', fontsize=12,
                            transform=ax.transAxes,
                            color='C3' if abs(r) > 0.5 else 'k')
                ax.set_xlabel(''); ax.set_ylabel('')
    g.fig.suptitle(
        'Pairwise scatter + Pearson r — fits as units, coloured by ROI',
        fontsize=11, y=1.01,
    )
    pdf.savefig(g.fig); plt.close(g.fig)


def page_per_fit_dotplot(pdf, df):
    """One row per fit; columns σ_AF, g_HP, g_LP, g_diff."""
    if df.empty:
        return
    cols = ['sigma_AF', 'g_HP', 'g_LP', 'g_diff']
    df_plot = df.copy()
    df_plot['label'] = df_plot.apply(
        lambda r: f"{r['roi']:>6} sub-{int(r['subject']):02d}", axis=1)
    df_plot = df_plot.sort_values(['roi', 'subject']).reset_index(drop=True)
    fig, axes = plt.subplots(
        1, len(cols), figsize=(2.5 * len(cols) + 1, 0.16 * len(df_plot) + 2),
        sharey=True,
    )
    y = np.arange(len(df_plot))
    for ax, col in zip(axes, cols):
        # Color by ROI.
        roi_to_color = {r: c for r, c in zip(
            df_plot['roi'].cat.categories, sns.color_palette('Set2', len(df_plot['roi'].cat.categories))
        )}
        colors = [roi_to_color[r] for r in df_plot['roi']]
        ax.scatter(df_plot[col], y, c=colors, s=28, edgecolor='k',
                   linewidth=0.3)
        ax.axvline(0, color='gray', lw=0.5, ls='--')
        ax.set_xlabel(col)
        ax.set_yticks(y); ax.set_yticklabels(df_plot['label'], fontsize=6)
    fig.suptitle('Per-fit shared parameters', fontsize=11)
    fig.tight_layout()
    pdf.savefig(fig); plt.close(fig)


def page_g_hp_vs_g_lp(pdf, df):
    """Scatter g_HP vs g_LP coloured by ROI; diagonal = 'no HP effect'."""
    fig, ax = plt.subplots(figsize=(8.5, 7))
    rois = list(df['roi'].cat.categories)
    palette = dict(zip(rois, sns.color_palette('Set2', len(rois))))
    for roi, sub in df.groupby('roi', observed=True):
        ax.scatter(sub['g_LP'], sub['g_HP'],
                   color=palette[roi], s=70, edgecolor='k',
                   linewidth=0.4, alpha=0.85, label=roi)
    lim = max(df[['g_HP', 'g_LP']].abs().max().max() * 1.05, 1.0)
    ax.plot([-lim, lim], [-lim, lim], 'k:', lw=0.8, label='y = x')
    ax.axhline(0, color='gray', lw=0.4, ls='--')
    ax.axvline(0, color='gray', lw=0.4, ls='--')
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel('g_LP  (modulation strength at LP locations)')
    ax.set_ylabel('g_HP  (modulation strength at HP location)')
    ax.set_title("g_HP vs g_LP per fit  —  off-diagonal = HP-specific effect")
    ax.legend(title='ROI', loc='upper left', fontsize=8)
    ax.set_aspect('equal')
    fig.tight_layout()
    pdf.savefig(fig); plt.close(fig)


def page_summary_table(pdf, df):
    """Per-ROI median ± MAD of each parameter, count, mean R²."""
    if df.empty:
        return
    summary = df.groupby('roi', observed=True).agg(
        n_fits=('subject', 'count'),
        sigma_AF_med=('sigma_AF', 'median'),
        g_HP_med=('g_HP', 'median'),
        g_LP_med=('g_LP', 'median'),
        g_diff_med=('g_diff', 'median'),
        r2_mean=('r2_mean', 'mean'),
    ).round(3).reset_index()
    fig, ax = plt.subplots(figsize=(11, 0.4 * len(summary) + 2))
    ax.axis('off')
    ax.text(0.5, 0.95, 'Per-ROI summary (median across fits)',
            ha='center', va='top', fontsize=12, weight='bold',
            transform=ax.transAxes)
    table_str = summary.to_string(index=False)
    ax.text(0.04, 0.85, table_str, ha='left', va='top',
            family='monospace', fontsize=10, transform=ax.transAxes)
    pdf.savefig(fig); plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def page_hp_vs_lp_test(pdf, df: pd.DataFrame,
                         metric: str = 'g_diff',
                         metric_label: str = 'g_HP − g_LP',
                         h0_text: str = 'g_HP < g_LP',
                         ylim_clip: float = 1.0):
    """Per-ROI test that `metric < 0` (HP-specific suppression).

    Convention reminder for 'signed' mode (M = 1 + g_HP·A_HP + g_LP·A_LP):
      • g_HP negative = HP location is suppressed
      • g_HP < g_LP  = HP suppressed *more* than the three LP locations
    Per-ROI we run:
      1. one-sample t-test of (metric) < 0                [parametric]
      2. Wilcoxon signed-rank test of (metric) < 0        [robust]
      3. sign test (proportion of subjects with metric < 0)
    The strip-plot panel shows per-fit metric values with median + 95%
    bootstrap CI per ROI.

    `metric` is the column name in `df`.  `ylim_clip` clamps the
    y-axis so a few extreme outliers don't flatten the central tendency
    — outliers are shown as red ↑/↓ arrows with the count.
    """
    from scipy import stats
    if df.empty:
        return
    rec = []
    for roi, sub in df.groupby('roi', observed=True):
        x = sub[metric].dropna().values
        if len(x) < 3:
            rec.append(dict(
                roi=roi, n=len(x),
                mean_metric=float(np.nan), sem_metric=float(np.nan),
                t=float(np.nan), p_t_lt0=float(np.nan),
                w=float(np.nan), p_w_lt0=float(np.nan),
                frac_neg=float(np.nan),
            ))
            continue
        t_stat, p_t_two = stats.ttest_1samp(x, 0.0)
        p_t_lt = (p_t_two / 2.0) if t_stat < 0 else 1.0 - p_t_two / 2.0
        try:
            w_stat, p_w_lt = stats.wilcoxon(x, alternative='less',
                                             zero_method='pratt')
        except ValueError:
            w_stat, p_w_lt = float('nan'), float('nan')
        frac_neg = float((x < 0).mean())
        rec.append(dict(
            roi=roi, n=len(x),
            mean_metric=float(x.mean()),
            sem_metric=float(x.std(ddof=1) / np.sqrt(len(x))),
            t=float(t_stat), p_t_lt0=float(p_t_lt),
            w=float(w_stat), p_w_lt0=float(p_w_lt),
            frac_neg=frac_neg,
        ))
    summary = pd.DataFrame(rec)

    # Clip y-axis to a robust range (mostly hides extreme outliers so the
    # central tendency is visible).  A handful of subjects have σ_AF→9°
    # which lets g_HP/g_LP run to ±5 — those would dominate the plot.
    YLIM = ylim_clip
    n_outliers = (df[metric].abs() > YLIM).sum()
    n_total_for_outliers = df[metric].notna().sum()

    # ---- (a) plot: per-ROI strip + MEDIAN with 95% bootstrap CI
    #              (outlier-resistant) + p-values annotated.
    rng = np.random.default_rng(0)
    n_boot = 2000

    fig, ax = plt.subplots(figsize=(11, 6))
    df_plot = df.assign(metric_clip=df[metric].clip(-YLIM, YLIM))
    sns.stripplot(data=df_plot, x='roi', y='metric_clip', ax=ax,
                   hue='roi', palette='Set2', legend=False,
                   jitter=0.18, alpha=0.7, size=5)
    rois_with_data = [r for r in df['roi'].cat.categories
                       if (df['roi'] == r).any()]
    for i, roi in enumerate(rois_with_data):
        x = df.loc[df['roi'] == roi, metric].dropna().values
        if len(x) < 2:
            continue
        med = float(np.median(x))
        boots = rng.choice(x, size=(n_boot, len(x)), replace=True)
        boot_meds = np.median(boots, axis=1)
        lo, hi = np.quantile(boot_meds, [0.025, 0.975])
        ax.errorbar(
            [i], [med], yerr=[[med - lo], [hi - med]],
            fmt='s', color='k', markersize=10, ecolor='k',
            elinewidth=2.0, capsize=6, zorder=10,
        )
        # Annotate count of off-axis outliers near top/bottom of panel.
        n_lo = int((x < -YLIM).sum())
        n_hi = int((x > +YLIM).sum())
        if n_lo:
            ax.annotate(
                f'↓ {n_lo}', xy=(i, -YLIM * 0.97),
                ha='center', va='top', fontsize=8,
                color='C3', weight='bold',
            )
        if n_hi:
            ax.annotate(
                f'↑ {n_hi}', xy=(i, +YLIM * 0.97),
                ha='center', va='bottom', fontsize=8,
                color='C3', weight='bold',
            )

    ax.axhline(0, color='gray', lw=0.7, ls='--')
    ax.set_xlabel('')
    ax.set_ylabel(f'{metric_label}   (negative ⇒ HP suppressed more than LP)')
    ax.set_title(
        f'HP-vs-LP test ({h0_text}):  per-fit median ± 95 % bootstrap CI  '
        f'(outlier-resistant; {n_outliers}/{n_total_for_outliers} fits '
        f'with |{metric_label}|>{YLIM} clipped, count shown red ↑/↓)',
        fontsize=10,
    )
    # Annotate Wilcoxon p-values just above the YLIM line so they stay in view.
    ytxt = YLIM + 0.05
    ax.set_ylim(-YLIM - 0.05, YLIM + 0.35)
    for i, row in summary.reset_index(drop=True).iterrows():
        if not np.isfinite(row['p_w_lt0']):
            continue
        # Mark significance.
        sig = (
            '***' if row['p_w_lt0'] < 0.001 else
            '**'  if row['p_w_lt0'] < 0.01 else
            '*'   if row['p_w_lt0'] < 0.05 else
            ''
        )
        col = 'C3' if row['p_w_lt0'] < 0.05 else '0.3'
        ax.text(
            i, ytxt,
            f"Wilcoxon\np={row['p_w_lt0']:.3f}{sig}\n"
            f"<0: {row['frac_neg']*100:.0f}%",
            ha='center', va='bottom', fontsize=8, color=col,
        )
    fig.tight_layout()
    pdf.savefig(fig); plt.close(fig)

    # ---- (b) tidy table page.
    fig, ax = plt.subplots(figsize=(11, 0.45 * len(summary) + 2.5))
    ax.axis('off')
    ax.text(0.5, 0.96, f'Per-ROI HP-vs-LP test  —  H0: {metric_label} ≥ 0',
            ha='center', va='top', fontsize=13, weight='bold',
            transform=ax.transAxes)
    ax.text(0.5, 0.92, f'Alternative: {h0_text}  (HP-specific suppression)',
            ha='center', va='top', fontsize=10, style='italic',
            transform=ax.transAxes)
    table = summary.copy()
    table['mean ± SEM'] = table.apply(
        lambda r: f"{r['mean_metric']:+.3f} ± {r['sem_metric']:.3f}", axis=1)
    table = table[['roi', 'n', 'mean ± SEM',
                    't', 'p_t_lt0', 'w', 'p_w_lt0', 'frac_neg']]
    table.columns = ['ROI', 'n', f'{metric_label} mean ± SEM',
                      't', 'p (t, lt 0)', 'W', 'p (Wilcoxon, lt 0)',
                      f'frac {metric_label}<0']
    for c in ['t', 'p (t, lt 0)', 'W', 'p (Wilcoxon, lt 0)',
              f'frac {metric_label}<0']:
        table[c] = table[c].apply(
            lambda v: '—' if not np.isfinite(v) else f'{v:.3f}')
    ax.text(0.04, 0.85, table.to_string(index=False),
            ha='left', va='top',
            family='monospace', fontsize=10, transform=ax.transAxes)
    pdf.savefig(fig); plt.close(fig)


def page_predicted_vs_observed(pdf, shifts: pd.DataFrame):
    """Per-(subject, ROI) predicted-vs-observed scatter + r summary.

    Expects the long-format TSV produced by
    :mod:`retsupp.visualize.predict_shifts_from_af` with columns
    ``subject``, ``roi``, ``proj_pred``, ``proj_obs``.
    """
    if shifts.empty:
        return

    # ---- Front matter page: precisely what is being predicted vs observed.
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis('off')
    ax.text(0.5, 0.96, 'Predicted vs observed shifts — what each side is',
            ha='center', va='top', fontsize=15, weight='bold',
            transform=ax.transAxes)
    body = (
        'PER VOXEL × HP-CONDITION row, we have:\n'
        '\n'
        '  base PRF center  (x_v, y_v)\n'
        '    = the voxel\'s position parameters from the JOINT AF + Gaussian-PRF\n'
        '      fit to the BOLD time-series (the same fit that produced σ_AF,\n'
        '      g_HP, g_LP). NOT the standard mean-PRF fit.\n'
        '\n'
        '  predicted AD-pRF center  (μ̃_v_C)\n'
        '    = closed-form center-of-mass of  ρ_v_C(g) = M_C(g) · S_v(g)\n'
        '      computed on a discretized 2D grid, where M_C(g) is built from\n'
        '      the SHARED-AF parameters (σ_AF, g_HP, g_LP, mode) for the\n'
        '      condition C, and S_v(g) is the voxel\'s Gaussian PRF.\n'
        '      Negative modulation is clamped to 0 before the COM (matching the\n'
        '      braincoder forward model).\n'
        '\n'
        '  observed conditionwise PRF center  (x^obs_v_C, y^obs_v_C)\n'
        '    = the (x, y) parameters from prf_conditionfit/model4 — INDEPENDENT\n'
        '      per-condition Gaussian-PRF fits, one per HP block, run on\n'
        '      subsets of runs. These are NOT used by the joint AF + PRF fit.\n'
        '\n'
        'The two correlation axes:\n'
        '\n'
        '  proj  (1D)  : project (predicted − base) and (observed − base) shifts\n'
        '                onto the unit vector pointing AWAY from the HP location\n'
        '                of that condition C. Negative = toward HP, positive =\n'
        '                away from HP.\n'
        '\n'
        '  (Δx, Δy) (2D): full per-component shift vectors\n'
        '                Δx = x_pred − base_x;   Δy = y_pred − base_y\n'
        '                (and similarly for observed).\n'
        '\n'
        'High Pearson r between predicted and observed at the voxel level means\n'
        'the joint BOLD-level AF model genuinely recovers the voxel-wise shift\n'
        'pattern that the conditionwise PRF fits independently report.\n'
    )
    ax.text(0.04, 0.90, body, ha='left', va='top',
            family='monospace', fontsize=9, transform=ax.transAxes)
    pdf.savefig(fig); plt.close(fig)

    # Compute per-(sub, roi) Pearson r.
    rec = []
    for (s, roi), g in shifts.groupby(['subject', 'roi']):
        m = g[['proj_pred', 'proj_obs']].dropna()
        if len(m) < 10:
            continue
        r = float(np.corrcoef(m['proj_pred'], m['proj_obs'])[0, 1])
        rec.append(dict(
            subject=int(s), roi=roi, n=len(m),
            r=r,
            mean_pred=float(m['proj_pred'].mean()),
            mean_obs=float(m['proj_obs'].mean()),
        ))
    summary = pd.DataFrame(rec)
    if summary.empty:
        return
    summary['roi'] = pd.Categorical(
        summary['roi'],
        categories=[r for r in ROI_ORDER if r in summary['roi'].unique()] +
                   sorted(set(summary['roi']) - set(ROI_ORDER)),
        ordered=True,
    )

    # ---- (a) per-ROI box+strip of r values.
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.stripplot(data=summary, x='roi', y='r', ax=ax,
                   hue='roi', palette='Set2', legend=False,
                   jitter=0.2, alpha=0.85, size=6)
    sns.boxplot(data=summary, x='roi', y='r', ax=ax,
                width=0.45, fill=False, color='k', linewidth=1,
                showfliers=False)
    ax.axhline(0, color='gray', lw=0.5, ls='--')
    ax.set_ylim(-0.1, 1.0)
    ax.set_xlabel('')
    ax.set_ylabel('Pearson r — voxel-wise predicted vs observed projection')
    ax.set_title(
        'Per-fit predicted-vs-observed shift correlation\n'
        '(higher r = AF model captures real per-voxel shift structure)',
        fontsize=11,
    )
    fig.tight_layout()
    pdf.savefig(fig); plt.close(fig)

    # ---- (b) summary table.
    table = (summary.groupby('roi', observed=True)['r']
              .agg(['count', 'mean', 'median', 'min', 'max'])
              .round(3).reset_index())
    fig, ax = plt.subplots(figsize=(9, 0.4 * len(table) + 2))
    ax.axis('off')
    ax.text(0.5, 0.95, 'Per-ROI Pearson r (predicted vs observed)',
            ha='center', va='top', fontsize=12, weight='bold',
            transform=ax.transAxes)
    ax.text(0.04, 0.85, table.to_string(index=False),
            ha='left', va='top',
            family='monospace', fontsize=10, transform=ax.transAxes)
    pdf.savefig(fig); plt.close(fig)

    # ---- (c) per-(subject, ROI) BIN-AVERAGED predicted-vs-observed.
    # Bin by predicted projection (8 quantile bins per fit), then plot
    # mean observed per bin against mean predicted per bin. One marker
    # per (fit, bin), connected by a line per fit. Same x→y diagonal
    # would mean perfect calibration; deviations show systematic errors.
    rec_lines = []
    for (s_id, roi), g in shifts.groupby(['subject', 'roi']):
        m = g[['proj_pred', 'proj_obs']].dropna()
        if len(m) < 50:
            continue
        try:
            m['bin'] = pd.qcut(m['proj_pred'], q=8, duplicates='drop',
                                labels=False)
        except ValueError:
            continue
        agg = (m.groupby('bin')
                .agg(p=('proj_pred', 'mean'),
                     o=('proj_obs', 'mean'),
                     n=('proj_obs', 'size'))
                .reset_index())
        agg['subject'] = int(s_id); agg['roi'] = roi
        rec_lines.append(agg)
    if rec_lines:
        binned = pd.concat(rec_lines, ignore_index=True)
        binned['roi'] = pd.Categorical(
            binned['roi'],
            categories=[r for r in ROI_ORDER if r in binned['roi'].unique()],
            ordered=True,
        )
        rois = list(binned['roi'].cat.categories)
        n_rois = len(rois)
        ncol = 4
        nrow = int(np.ceil(n_rois / ncol))
        fig, axes = plt.subplots(
            nrow, ncol, figsize=(3.0 * ncol, 2.6 * nrow),
            sharex=True, sharey=True,
        )
        axes = np.atleast_2d(axes)
        lim = float(np.nanpercentile(
            np.abs(np.r_[binned['p'].values, binned['o'].values]), 99))
        lim = max(lim, 0.05)
        for i, roi in enumerate(rois):
            ax = axes[i // ncol, i % ncol]
            for s_id, sub in binned[binned['roi'] == roi].groupby('subject'):
                sub = sub.sort_values('p')
                ax.plot(sub['p'], sub['o'],
                        '-o', color='C0', alpha=0.5,
                        markersize=3, lw=0.7)
            # ROI-mean curve: average across fits per bin index.
            ax.plot([-lim, lim], [-lim, lim], 'k:', lw=0.8)
            ax.axhline(0, color='gray', lw=0.3, ls='--')
            ax.axvline(0, color='gray', lw=0.3, ls='--')
            ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
            ax.set_aspect('equal')
            ax.set_title(f'{roi}  (n_fits={binned[binned["roi"]==roi].subject.nunique()})',
                         fontsize=9)
            if i // ncol == nrow - 1:
                ax.set_xlabel('mean predicted (°)', fontsize=8)
            if i % ncol == 0:
                ax.set_ylabel('mean observed (°)', fontsize=8)
        for j in range(n_rois, nrow * ncol):
            axes[j // ncol, j % ncol].axis('off')
        fig.suptitle(
            'Calibration: bin-averaged predicted vs observed (one line per fit)\n'
            'On-diagonal = AF prediction matches observed shift on average',
            fontsize=11,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig); plt.close(fig)


def page_predicted_vs_observed_2d(pdf, shifts: pd.DataFrame):
    """Full 2D shift correlations (dx, dy separately) — not just the
    away-from-HP projection.

    For each voxel × condition row we have:
      dx_pred = pred_x - base_x      dx_obs = obs_x - base_x
      dy_pred = pred_y - base_y      dy_obs = obs_y - base_y
    """
    if shifts.empty:
        return
    s = shifts.copy()
    for c in ['pred_x', 'pred_y', 'obs_x', 'obs_y', 'base_x', 'base_y']:
        if c not in s.columns:
            return
    s['dx_pred'] = s['pred_x'] - s['base_x']
    s['dy_pred'] = s['pred_y'] - s['base_y']
    s['dx_obs'] = s['obs_x'] - s['base_x']
    s['dy_obs'] = s['obs_y'] - s['base_y']
    s = s.dropna(subset=['dx_pred', 'dy_pred', 'dx_obs', 'dy_obs'])
    if s.empty:
        return

    # Per-fit r_x and r_y.
    rec = []
    for (sub, roi), g in s.groupby(['subject', 'roi']):
        if len(g) < 10:
            continue
        rx = float(np.corrcoef(g['dx_pred'], g['dx_obs'])[0, 1])
        ry = float(np.corrcoef(g['dy_pred'], g['dy_obs'])[0, 1])
        # Cosine similarity of vector pairs (per-voxel), then average.
        # cos = (dx_pred·dx_obs + dy_pred·dy_obs) / (|pred| · |obs|).
        ndp = np.sqrt(g['dx_pred']**2 + g['dy_pred']**2) + 1e-12
        ndo = np.sqrt(g['dx_obs']**2 + g['dy_obs']**2) + 1e-12
        cos = ((g['dx_pred'] * g['dx_obs']
                 + g['dy_pred'] * g['dy_obs']) / (ndp * ndo)).mean()
        rec.append(dict(subject=int(sub), roi=roi, n=len(g),
                        r_x=rx, r_y=ry, cos_mean=float(cos)))
    summary = pd.DataFrame(rec)
    if summary.empty:
        return
    summary['roi'] = pd.Categorical(
        summary['roi'],
        categories=[r for r in ROI_ORDER if r in summary['roi'].unique()] +
                   sorted(set(summary['roi']) - set(ROI_ORDER)),
        ordered=True,
    )

    # ---- (a) per-ROI box+strip of r_x and r_y side by side.
    long = pd.concat([
        summary[['roi', 'r_x']].rename(columns={'r_x': 'r'}).assign(
            component='x'),
        summary[['roi', 'r_y']].rename(columns={'r_y': 'r'}).assign(
            component='y'),
    ])
    fig, ax = plt.subplots(figsize=(11, 5.5))
    sns.boxplot(data=long, x='roi', y='r', hue='component',
                ax=ax, fill=False, gap=0.1, showfliers=False)
    sns.stripplot(data=long, x='roi', y='r', hue='component',
                   ax=ax, dodge=True, jitter=0.15, size=5,
                   alpha=0.85, palette='dark', legend=False)
    ax.axhline(0, color='gray', lw=0.5, ls='--')
    ax.set_ylim(-0.2, 1.0)
    ax.set_xlabel('')
    ax.set_ylabel('Pearson r — predicted vs observed shift component')
    ax.set_title(
        'Per-fit 2D shift correlations: dx and dy components separately',
        fontsize=11,
    )
    fig.tight_layout()
    pdf.savefig(fig); plt.close(fig)

    # ---- (b) cosine similarity per ROI.
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.stripplot(data=summary, x='roi', y='cos_mean', ax=ax,
                   hue='roi', palette='Set2', legend=False,
                   jitter=0.2, alpha=0.85, size=6)
    sns.boxplot(data=summary, x='roi', y='cos_mean', ax=ax,
                width=0.45, fill=False, color='k', linewidth=1,
                showfliers=False)
    ax.axhline(0, color='gray', lw=0.5, ls='--')
    ax.set_ylim(-0.5, 1.0)
    ax.set_xlabel('')
    ax.set_ylabel('mean cosine(predicted shift, observed shift)')
    ax.set_title(
        'Per-fit mean cosine similarity of 2D shift vectors  '
        '(1 = perfect direction agreement, 0 = orthogonal)',
        fontsize=10,
    )
    fig.tight_layout()
    pdf.savefig(fig); plt.close(fig)

    # ---- (c) summary table.
    table = (summary.groupby('roi', observed=True)
              .agg(n_fits=('subject', 'count'),
                   r_x_med=('r_x', 'median'),
                   r_y_med=('r_y', 'median'),
                   cos_med=('cos_mean', 'median'))
              .round(3).reset_index())
    fig, ax = plt.subplots(figsize=(9, 0.4 * len(table) + 2))
    ax.axis('off')
    ax.text(0.5, 0.95, 'Per-ROI 2D shift agreement (medians across fits)',
            ha='center', va='top', fontsize=12, weight='bold',
            transform=ax.transAxes)
    ax.text(0.04, 0.85, table.to_string(index=False),
            ha='left', va='top',
            family='monospace', fontsize=10, transform=ax.transAxes)
    pdf.savefig(fig); plt.close(fig)

    # ---- (d) pooled per-voxel 2D hexbin of (dx_pred, dx_obs) and
    # (dy_pred, dy_obs). Hexbin instead of a 200K-point scatter so the
    # PDF stays light AND the eye sees density structure rather than a
    # cloud of dots.
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, (xc, yc, title) in zip(axes, [
            ('dx_pred', 'dx_obs', 'Δx component  (deg)'),
            ('dy_pred', 'dy_obs', 'Δy component  (deg)')]):
        lim = float(np.nanpercentile(
            np.abs(np.r_[s[xc].values, s[yc].values]), 99.5))
        lim = max(lim, 0.3)
        hb = ax.hexbin(s[xc], s[yc],
                        gridsize=60, mincnt=2,
                        cmap='Greys', bins='log',
                        extent=(-lim, lim, -lim, lim))
        ax.plot([-lim, lim], [-lim, lim], 'C3:', lw=1.0)
        ax.axhline(0, color='gray', lw=0.4)
        ax.axvline(0, color='gray', lw=0.4)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_xlabel(f'predicted {title}')
        ax.set_ylabel(f'observed {title}')
        ax.set_aspect('equal')
        ax.set_title(title)
        cb = plt.colorbar(hb, ax=ax, label='log10(n voxels)')
        cb.ax.tick_params(labelsize=8)
    fig.suptitle(f'Pooled per-voxel 2D shift density  '
                 f'(N = {len(s):,} voxel-condition rows; hexbin, log scale)',
                 fontsize=11)
    fig.tight_layout()
    pdf.savefig(fig); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fits-dir', type=Path,
                        default=Path('/data/ds-retsupp/derivatives/af_prf_joint_full'),
                        help='Root of the af_prf_joint*/ derivatives tree.')
    parser.add_argument('--out', type=Path,
                        default=Path('notes/af_parameters.pdf'))
    parser.add_argument('--tsv', type=Path,
                        default=Path('notes/af_parameters.tsv'),
                        help='Tabular dump of the per-fit records.')
    parser.add_argument('--predict-shifts-tsv', type=Path,
                        default=None,
                        help='Optional: long-format TSV from '
                             'predict_shifts_from_af.py to add '
                             'predicted-vs-observed summary pages.')
    args = parser.parse_args()

    df = collect_records(args.fits_dir)
    if df.empty:
        raise SystemExit(f'No fits found under {args.fits_dir}')

    args.tsv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.tsv, sep='\t', index=False)
    print(f'Wrote {args.tsv} ({len(df)} fits)')

    shifts = pd.DataFrame()
    if args.predict_shifts_tsv and args.predict_shifts_tsv.exists():
        shifts = pd.read_csv(args.predict_shifts_tsv, sep='\t')
        print(f'Loaded shifts: {len(shifts):,} rows from '
              f'{args.predict_shifts_tsv}')

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(args.out) as pdf:
        cover_page(pdf, df, args)
        page_summary_table(pdf, df)
        # Two complementary HP-vs-LP test pages:
        #  - g_diff (raw difference, scale-dependent)
        #  - log_ratio (log ratio of effective modulation factors,
        #    scale-invariant — better for comparing across ROIs).
        page_hp_vs_lp_test(pdf, df,
                            metric='g_diff',
                            metric_label='g_HP − g_LP',
                            h0_text='g_HP < g_LP',
                            ylim_clip=1.0)
        page_hp_vs_lp_test(pdf, df,
                            metric='log_ratio',
                            metric_label='log((1+g_HP)/(1+g_LP))',
                            h0_text='M_HP < M_LP  (scale-invariant)',
                            ylim_clip=1.5)
        page_distributions(pdf, df)
        page_sigma_gain_correlation(pdf, df)
        page_pairwise(pdf, df)
        page_g_hp_vs_g_lp(pdf, df)
        page_per_fit_dotplot(pdf, df)
        if not shifts.empty:
            page_predicted_vs_observed(pdf, shifts)
            page_predicted_vs_observed_2d(pdf, shifts)

    print(f'Wrote {args.out}')


if __name__ == '__main__':
    main()
