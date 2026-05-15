"""Three-question summary figure of DoG dynamic AF (v3 + target + sharedSigma).

Loads per-(subject, ROI) ``*-af-prf-fit.pkl`` files produced by
``retsupp.modeling.fit_dog_dynamic_af_braincoder`` and answers the three
talk-relevant questions:

  1. Does the target attract attention?         ->  g_T_dyn > 0
  2. Is the HP distractor location suppressed
     (static AND dynamic)?                       ->  g_HP - g_LP < 0
                                                     g_HP_dyn - g_LP_dyn < 0
  3. Is the static AF wider than the dynamic
     attentional pulses?                         ->  sigma_AF > sigma_dyn

Each row pairs a minimalist icon (PRF + AF bump, calling back to
``talk_buildup_01_af_basics.pdf``) with a per-ROI point plot of the
quantity across subjects (mean ± 95% CI).

Usage
-----
    python -m retsupp.visualize.paper.plot_af_results \\
        --fits-dir /data/ds-retsupp/derivatives/af_prf_joint_dynamic_v3_dog_with_target_sharedSigma \\
        --out notes/figures/af_results.pdf
"""
from __future__ import annotations

import argparse
import glob
import os
import pickle
import re
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Visual-cortex ROIs we draw on the talk figure by default.
ROI_VISUAL = ['V1', 'V2', 'V3', 'V3AB', 'hV4', 'LO', 'TO', 'VO']
ROI_FRONTOPARIETAL = ['IPS', 'SPL1', 'FEF']
ROI_ORDER = ROI_VISUAL + ROI_FRONTOPARIETAL
PARSE_RE = re.compile(r"sub-(\d+)_roi-([^_]+)_mode-[^_]+_(?:.*)-af-prf-fit\.pkl$")
# Subjects to exclude from group analyses. Sub-01/02 were pilot subjects
# with design differences and should not be pooled with sub-03+.
EXCLUDE_SUBJECTS = {1, 2}

plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 13,
    "figure.titlesize": 19,
})


def collect_records(fits_dir: Path,
                    min_mtime: float = 0.0,
                    rois: list[str] | None = None) -> pd.DataFrame:
    """Walk an AF derivative tree; return one row per (subject, ROI) fit.

    Filters by file mtime (default = no filter — include all) and by ROI
    membership (default = visual-cortex set; pass ``rois='all'`` to keep
    frontoparietal ROIs too).
    """
    paths = sorted(glob.glob(str(fits_dir / 'sub-*/sub-*-af-prf-fit.pkl')))
    if rois is None:
        rois = ROI_VISUAL
    elif rois == 'all':
        rois = ROI_ORDER
    rows = []
    for p in paths:
        m = PARSE_RE.search(Path(p).name)
        if not m:
            continue
        if os.path.getmtime(p) < min_mtime:
            continue
        sub = int(m.group(1))
        if sub in EXCLUDE_SUBJECTS:
            continue
        roi = m.group(2)
        if roi not in rois:
            continue
        try:
            with open(p, 'rb') as f:
                d = pickle.load(f)
        except Exception as e:
            print(f"  WARN: cannot load {p}: {e}")
            continue
        sh = d.get('shared_pars', {})
        # Skip fits that never produced finite shared params (older broken
        # runs wrote NaN here even though the file existed).
        if not np.isfinite(sh.get('sigma_AF', np.nan)):
            continue
        r2 = np.asarray(d.get('r2', []))
        rows.append(dict(
            subject=sub, roi=roi,
            n_voxels=len(d.get('fit_pars', [])),
            r2_mean=float(np.nanmean(r2)) if r2.size else np.nan,
            r2_median=float(np.nanmedian(r2)) if r2.size else np.nan,
            sigma_AF=float(sh.get('sigma_AF', np.nan)),
            g_HP=float(sh.get('g_HP', np.nan)),
            g_LP=float(sh.get('g_LP', np.nan)),
            sigma_dyn=float(sh.get('sigma_dyn', np.nan)),
            g_HP_dyn=float(sh.get('g_HP_dyn', np.nan)),
            g_LP_dyn=float(sh.get('g_LP_dyn', np.nan)),
            g_T_dyn=float(sh.get('g_T_dyn', np.nan)),
            sigma_T_dyn=float(sh.get('sigma_T_dyn', np.nan)),
            # Repeat-trial dynamic gain modulators (additive on top of
            # g_HP_dyn / g_LP_dyn). Only present in the _repeat variant.
            g_HP_dyn_repeat=float(sh.get('g_HP_dyn_repeat', np.nan)),
            g_LP_dyn_repeat=float(sh.get('g_LP_dyn_repeat', np.nan)),
        ))
    df = pd.DataFrame(rows)
    if not df.empty:
        df['g_HP_minus_g_LP_static'] = df['g_HP'] - df['g_LP']
        df['g_HP_minus_g_LP_dyn'] = df['g_HP_dyn'] - df['g_LP_dyn']
        df['sigma_static_minus_dyn'] = df['sigma_AF'] - df['sigma_dyn']
        # In the _repeat variant, g_HP_dyn is the switch-trial gain only.
        # Average over (switch, repeat) to get a "no-repeat-aware" effective
        # dynamic gain: g_HP_dyn_avg = g_HP_dyn + 0.5 * g_HP_dyn_repeat (equal
        # weight switch & repeat trials). Falls back to g_HP_dyn when the
        # repeat columns are all NaN (non-repeat variants).
        if df['g_HP_dyn_repeat'].notna().any():
            df['g_HP_dyn_avg'] = df['g_HP_dyn'] + 0.5 * df['g_HP_dyn_repeat']
            df['g_LP_dyn_avg'] = df['g_LP_dyn'] + 0.5 * df['g_LP_dyn_repeat']
        else:
            df['g_HP_dyn_avg'] = df['g_HP_dyn']
            df['g_LP_dyn_avg'] = df['g_LP_dyn']
        df['g_HP_minus_g_LP_dyn_avg'] = df['g_HP_dyn_avg'] - df['g_LP_dyn_avg']
        df['roi'] = pd.Categorical(
            df['roi'],
            categories=[r for r in ROI_ORDER if r in df['roi'].unique()] +
                       sorted(set(df['roi']) - set(ROI_ORDER)),
            ordered=True,
        )
        df = df.sort_values(['roi', 'subject']).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Icons — minimalist PRF + AF callbacks to talk_buildup_01_af_basics.pdf
# ---------------------------------------------------------------------------

def _icon_axis(ax, ring_r=4.0, ring_color="0.7"):
    ax.set_aspect("equal")
    ax.set_xlim(-5.2, 5.2)
    ax.set_ylim(-5.2, 5.2)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    # The 4° eccentricity ring (faint, dashed) — the search-array circle.
    circ = plt.Circle((0, 0), ring_r, fill=False, lw=1.0,
                      ls=(0, (3, 3)), color=ring_color, zorder=1)
    ax.add_patch(circ)
    ax.plot(0, 0, "+", color="0.5", ms=8, mew=1.0, zorder=2)


def icon_target_attracts(ax):
    """Icon for Q1: target draws attention -> AF bump at target location."""
    _icon_axis(ax)
    # Place a "target" + AF bump in the upper-right at ecc 4.
    tx, ty = 4 / np.sqrt(2), 4 / np.sqrt(2)
    # AF bump (red, dashed circle = FWHM).
    sigma_af = 1.4
    fwhm = 2 * np.sqrt(2 * np.log(2)) * sigma_af
    af = plt.Circle((tx, ty), fwhm / 2, fill=False, ls=(0, (4, 3)),
                    lw=1.8, color="#c0392b", zorder=3)
    ax.add_patch(af)
    ax.plot(tx, ty, marker="X", ms=12, color="#c0392b", mew=1.5, zorder=4)
    # Small PRF (teal) anywhere on the ring (lower-right say).
    px, py = 4 / np.sqrt(2), -4 / np.sqrt(2)
    prf = plt.Circle((px, py), 0.55, fill=True, color="#1abc9c",
                     alpha=0.5, zorder=3)
    ax.add_patch(prf)


def icon_hpd_suppression(ax):
    """Icon for Q2: HPD location suppressed -> NEGATIVE AF bump at HPD."""
    _icon_axis(ax)
    # HPD location upper-right; suppressive (blue) AF.
    hx, hy = 4 / np.sqrt(2), 4 / np.sqrt(2)
    sigma_af = 1.6
    fwhm = 2 * np.sqrt(2 * np.log(2)) * sigma_af
    sup = plt.Circle((hx, hy), fwhm / 2, fill=False, ls=(0, (4, 3)),
                     lw=1.8, color="#2980b9", zorder=3)
    ax.add_patch(sup)
    ax.plot(hx, hy, marker="o", ms=11, mfc="none", mec="#2980b9",
            mew=2.0, zorder=4)
    ax.plot(hx, hy, marker="_", ms=14, color="#2980b9", mew=2.5, zorder=5)
    # A "regular" (LP) location at lower-left for contrast.
    lx, ly = -4 / np.sqrt(2), -4 / np.sqrt(2)
    ax.plot(lx, ly, marker="o", ms=8, mfc="0.85", mec="0.5",
            mew=1.0, zorder=3)


def icon_repeat_priming(ax):
    """Icon for Q4: priming/repetition effect — sharper suppression on repeat."""
    _icon_axis(ax)
    hx, hy = 4 / np.sqrt(2), 4 / np.sqrt(2)
    # Wide dynamic AF (novel trial: g_HP_dyn baseline).
    sigma_novel = 1.6
    fwhm_n = 2 * np.sqrt(2 * np.log(2)) * sigma_novel
    nov = plt.Circle((hx, hy), fwhm_n / 2, fill=True,
                     color="#9b59b6", alpha=0.16, zorder=2)
    ax.add_patch(nov)
    nov2 = plt.Circle((hx, hy), fwhm_n / 2, fill=False, ls=(0, (4, 3)),
                      lw=1.6, color="#9b59b6", zorder=3)
    ax.add_patch(nov2)
    # Narrower / deeper repeat (g_HP_dyn + g_HP_dyn_repeat, the extra bit).
    sigma_rep = 1.0
    fwhm_r = 2 * np.sqrt(2 * np.log(2)) * sigma_rep
    rep = plt.Circle((hx, hy), fwhm_r / 2, fill=True,
                     color="#34495e", alpha=0.40, zorder=4)
    ax.add_patch(rep)
    rep2 = plt.Circle((hx, hy), fwhm_r / 2, fill=False, lw=1.6,
                      color="#34495e", zorder=5)
    ax.add_patch(rep2)


def icon_static_vs_dyn_sigma(ax):
    """Icon for Q3: wide static AF + narrow dynamic pulses."""
    _icon_axis(ax)
    # Wide static AF, light orange, around upper-right HPD.
    hx, hy = 4 / np.sqrt(2), 4 / np.sqrt(2)
    sigma_static = 2.2
    sigma_dyn = 0.9
    fwhm_static = 2 * np.sqrt(2 * np.log(2)) * sigma_static
    fwhm_dyn = 2 * np.sqrt(2 * np.log(2)) * sigma_dyn
    af_st = plt.Circle((hx, hy), fwhm_static / 2, fill=True,
                       color="#e67e22", alpha=0.18, zorder=2)
    ax.add_patch(af_st)
    af_st2 = plt.Circle((hx, hy), fwhm_static / 2, fill=False,
                        ls=(0, (4, 3)), lw=1.6,
                        color="#e67e22", zorder=3)
    ax.add_patch(af_st2)
    # Narrow dynamic pulse on top.
    af_dy = plt.Circle((hx, hy), fwhm_dyn / 2, fill=True,
                       color="#c0392b", alpha=0.35, zorder=4)
    ax.add_patch(af_dy)
    af_dy2 = plt.Circle((hx, hy), fwhm_dyn / 2, fill=False,
                        lw=1.6, color="#c0392b", zorder=5)
    ax.add_patch(af_dy2)


# ---------------------------------------------------------------------------
# Per-row plot helpers (point + 95% CI across subjects, one column per ROI)
# ---------------------------------------------------------------------------

def _ci95(arr):
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return np.nan, np.nan, np.nan
    m = arr.mean()
    se = arr.std(ddof=1) / np.sqrt(arr.size)
    return m, m - 1.96 * se, m + 1.96 * se


def _per_roi_plot(ax, df, value_col, title, ylabel, zero_line=True,
                  point_jitter=0.18, color="#34495e",
                  roi_order=None, annotate_n=True, min_n=3,
                  add_wilcoxon_stars=False):
    rois = roi_order if roi_order is not None else \
        [r for r in ROI_ORDER if r in df['roi'].unique()]
    x_pos = np.arange(len(rois))
    if zero_line:
        ax.axhline(0, color="0.6", lw=0.8, ls=(0, (3, 3)), zorder=1)
    # Pre-compute ylim from the per-ROI mean ± 95% CI envelope so the
    # error bars are clearly visible (individual-subject dots may clip).
    ci_lo_all, ci_hi_all = [], []
    for roi in rois:
        v = df.loc[df['roi'] == roi, value_col].dropna().to_numpy()
        if len(v) < min_n:
            continue
        m, lo, hi = _ci95(v)
        if np.isfinite(m):
            ci_lo_all.append(lo); ci_hi_all.append(hi)
    if ci_lo_all:
        ylo, yhi = min(ci_lo_all), max(ci_hi_all)
        pad = 0.25 * max(yhi - ylo, 0.5)
        ax.set_ylim(ylo - pad, yhi + pad)
    rng = np.random.default_rng(0)
    n_per_roi = []
    for i, roi in enumerate(rois):
        sub_vals = df.loc[df['roi'] == roi, value_col].dropna().to_numpy()
        n_per_roi.append(len(sub_vals))
        if len(sub_vals) == 0:
            continue
        jitter = rng.uniform(-point_jitter, point_jitter, size=sub_vals.size)
        ax.scatter(np.full_like(sub_vals, i, dtype=float) + jitter, sub_vals,
                   color=color, alpha=0.35, s=22, zorder=2,
                   edgecolor="none")
        if len(sub_vals) < min_n:
            continue
        m, lo, hi = _ci95(sub_vals)
        if np.isfinite(m):
            ax.errorbar(i, m, yerr=[[m - lo], [hi - m]],
                        fmt='o', color=color, ms=10, lw=2.5,
                        capsize=4, capthick=2.0, zorder=3)
        if add_wilcoxon_stars and len(sub_vals) >= 5:
            from scipy.stats import wilcoxon
            try:
                pw = float(wilcoxon(sub_vals).pvalue)
            except Exception:
                continue
            stars = ('***' if pw < .001 else '**' if pw < .01
                     else '*' if pw < .05 else '·' if pw < .1 else '')
            if stars:
                ax.annotate(stars, xy=(i, max(m, hi)),
                            xytext=(0, 4), textcoords='offset points',
                            ha='center', va='bottom',
                            fontsize=13, fontweight='bold', color=color)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(rois)
    ax.set_xlim(-0.6, len(rois) - 0.4)
    ax.set_title(title, loc='left', pad=10)
    ax.set_ylabel(ylabel)
    ax.grid(axis='y', alpha=0.25)
    if annotate_n:
        for i, n in enumerate(n_per_roi):
            label = f"n={n}" if n > 0 else "—"
            ax.annotate(label,
                        xy=(i, 1.0), xycoords=('data', 'axes fraction'),
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', va='bottom',
                        fontsize=10, color="0.4")


# ---------------------------------------------------------------------------
# Main figure
# ---------------------------------------------------------------------------

def make_figure(df, out_path, roi_order=None):
    rois = roi_order if roi_order is not None else \
        [r for r in ROI_ORDER if r in df['roi'].unique()]

    # Row layout:
    #   0: Q1  target attraction      (g_T_dyn)
    #   1: Q2a static HPD suppression (g_HP − g_LP, static)
    #   2: Q2b dynamic HPD suppression (g_HP_dyn_avg − g_LP_dyn_avg)
    #   3: Q3  σ_dyn vs σ_static
    #   4: Q4  repeat-priming additive (optional, _repeat variant only)
    has_repeat = ('g_HP_dyn_repeat' in df.columns
                  and df['g_HP_dyn_repeat'].notna().any())
    n_rows = 5 if has_repeat else 4

    fig = plt.figure(figsize=(max(14, 1.55 * len(rois) + 4), 3.4 * n_rows))
    gs = fig.add_gridspec(
        nrows=n_rows, ncols=2,
        width_ratios=[0.6, 8.0], height_ratios=[1] * n_rows,
        hspace=0.70, wspace=0.20,
        left=0.03, right=0.99, top=0.93, bottom=0.05,
    )

    def _despine(ax):
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)

    # Row 1: target attracts.
    ax_i1 = fig.add_subplot(gs[0, 0])
    icon_target_attracts(ax_i1)
    ax_1 = fig.add_subplot(gs[0, 1])
    _per_roi_plot(ax_1, df, 'g_T_dyn',
                  title="Q1: Does the target attract attention?",
                  ylabel="g_T,dyn",
                  color="#c0392b", roi_order=rois,
                  add_wilcoxon_stars=True)
    _despine(ax_1)

    width = 0.36
    from scipy.stats import wilcoxon

    def _paired_strip(ax, df, rois, left_col, right_col,
                       left_label, right_label,
                       left_color, right_color, title, ylabel):
        """Two side-by-side strips per ROI: shows main effect + differential.

        Y-axis is scaled to the per-ROI mean ± 95% CI envelope (so the
        SEMs are clearly visible). Individual-subject dots that fall
        outside that range are still drawn, but may be clipped.
        """
        rng = np.random.default_rng(1)
        ax.axhline(0, color="0.6", lw=0.8, ls=(0, (3, 3)), zorder=1)
        ci_lo_all, ci_hi_all = [], []
        for col in (left_col, right_col):
            for roi in rois:
                v = df.loc[df['roi'] == roi, col].dropna().to_numpy()
                if len(v) < 3:
                    continue
                m, lo, hi = _ci95(v)
                if np.isfinite(m):
                    ci_lo_all.append(lo); ci_hi_all.append(hi)
        if ci_lo_all:
            lo, hi = min(ci_lo_all), max(ci_hi_all)
            pad = 0.25 * max(hi - lo, 0.5)
            ax.set_ylim(lo - pad, hi + pad)
        for i, roi in enumerate(rois):
            top_for_diff = -np.inf
            paired_subs = df.loc[df['roi'] == roi, [left_col, right_col]].dropna()
            for ofs, col, name, color in [
                (-width / 2, left_col, left_label, left_color),
                (+width / 2, right_col, right_label, right_color),
            ]:
                v = df.loc[df['roi'] == roi, col].dropna().to_numpy()
                jitter = rng.uniform(-width * 0.4, width * 0.4, size=v.size)
                ax.scatter(np.full_like(v, i + ofs, dtype=float) + jitter, v,
                            color=color, alpha=0.30, s=20, zorder=2,
                            edgecolor='none')
                if len(v) < 3:
                    continue
                m, l, h = _ci95(v)
                if np.isfinite(m):
                    ax.errorbar(i + ofs, m, yerr=[[m - l], [h - m]],
                                fmt='o', color=color, ms=10, lw=2.5,
                                capsize=4, capthick=2.0, zorder=3,
                                label=name if i == 0 else None)
                    if np.isfinite(h):
                        top_for_diff = max(top_for_diff, h)
                if len(v) >= 5:
                    try:
                        pw = float(wilcoxon(v).pvalue)
                    except Exception:
                        continue
                    stars = ('***' if pw < .001 else '**' if pw < .01
                             else '*' if pw < .05 else '·' if pw < .1
                             else '')
                    if stars:
                        ax.annotate(stars, xy=(i + ofs, h),
                                     xytext=(0, 4),
                                     textcoords='offset points',
                                     ha='center', va='bottom',
                                     fontsize=12, fontweight='bold',
                                     color=color)
            # Within-subject paired Wilcoxon on (LEFT − RIGHT). This is the
            # central HP-vs-LP question: does each subject's HP gain differ
            # from their own LP gain at this ROI?
            if len(paired_subs) >= 5:
                diff = (paired_subs[left_col] - paired_subs[right_col]).to_numpy()
                try:
                    p_paired = float(wilcoxon(diff).pvalue)
                except Exception:
                    p_paired = float('nan')
                m_d = float(np.nanmean(diff))
                stars_p = ('***' if p_paired < .001 else
                            '**' if p_paired < .01 else
                            '*' if p_paired < .05 else
                            '·' if p_paired < .1 else '')
                label_str = f"Δ={m_d:+.2f}{stars_p}" if stars_p else ""
                if label_str:
                    ax.annotate(label_str, xy=(i, top_for_diff),
                                 xytext=(0, 18),
                                 textcoords='offset points',
                                 ha='center', va='bottom',
                                 fontsize=10, fontweight='bold',
                                 color='#2c3e50')
        ax.set_xticks(np.arange(len(rois)))
        ax.set_xticklabels(rois)
        ax.set_xlim(-0.6, len(rois) - 0.4)
        ax.set_title(title, loc='left', pad=10)
        ax.set_ylabel(ylabel)
        ax.grid(axis='y', alpha=0.25)
        ax.legend(loc='best', frameon=False)

    # Row 2a: static gains, HP vs LP side by side.
    ax_i2a = fig.add_subplot(gs[1, 0])
    icon_hpd_suppression(ax_i2a)
    ax_2a = fig.add_subplot(gs[1, 1])
    _paired_strip(ax_2a, df, rois, 'g_HP', 'g_LP',
                   'g_HP', 'g_LP',
                   '#2980b9', '#7f8c8d',
                   "Q2a: Static gains  (HP vs LP location)",
                   "g (static)")
    _despine(ax_2a)

    # Row 2b: dynamic gains, HP vs LP side by side.
    ax_i2b = fig.add_subplot(gs[2, 0])
    icon_hpd_suppression(ax_i2b)
    dyn_title = ("Q2b: Dynamic gains  (HP vs LP location"
                 + ", switch+repeat avg)" if has_repeat
                 else "Q2b: Dynamic gains  (HP vs LP location)")
    ax_2b = fig.add_subplot(gs[2, 1])
    _paired_strip(ax_2b, df, rois, 'g_HP_dyn_avg', 'g_LP_dyn_avg',
                   'g_HP_dyn', 'g_LP_dyn',
                   '#8e44ad', '#7f8c8d',
                   dyn_title,
                   "g (dynamic)")
    _despine(ax_2b)

    # Row 3: static σ vs dynamic σ.
    ax_i3 = fig.add_subplot(gs[3, 0])
    icon_static_vs_dyn_sigma(ax_i3)
    ax_3 = fig.add_subplot(gs[3, 1])
    # Robust σ ylim: cap at ~98th percentile of the pooled data.
    q3_vals = pd.concat([df['sigma_AF'], df['sigma_dyn']]).dropna()
    sigma_top = float(q3_vals.quantile(0.97)) * 1.05
    rng = np.random.default_rng(2)
    for i, roi in enumerate(rois):
        sub_v = []
        for ofs, col, label_color, name in [
            (-width / 2, 'sigma_AF', '#e67e22', 'σ_static'),
            (+width / 2, 'sigma_dyn', '#c0392b', 'σ_dyn'),
        ]:
            v = df.loc[df['roi'] == roi, col].dropna().to_numpy()
            sub_v.append(v)
            jitter = rng.uniform(-width * 0.4, width * 0.4, size=v.size)
            ax_3.scatter(np.full_like(v, i + ofs, dtype=float) + jitter,
                         np.clip(v, 0, sigma_top),
                         color=label_color, alpha=0.30, s=20, zorder=2,
                         edgecolor="none")
            if len(v) < 3:
                continue
            m, lo, hi = _ci95(v)
            if np.isfinite(m):
                ax_3.errorbar(i + ofs, m, yerr=[[m - lo], [hi - m]],
                              fmt='o', color=label_color, ms=10, lw=2.5,
                              capsize=4, capthick=2.0, zorder=3,
                              label=name if i == 0 else None)
        # Wilcoxon paired test on σ_dyn − σ_static (n ≥ 5 needed).
        if len(sub_v[0]) >= 5 and len(sub_v[1]) >= 5:
            try:
                pw = float(wilcoxon(sub_v[1] - sub_v[0]).pvalue)
            except Exception:
                pw = float('nan')
            stars = ('***' if pw < .001 else '**' if pw < .01
                     else '*' if pw < .05 else '·' if pw < .1 else '')
            if stars:
                top = max(np.nanmax(sub_v[0]), np.nanmax(sub_v[1]))
                ax_3.annotate(stars, xy=(i, top),
                              xytext=(0, 4), textcoords='offset points',
                              ha='center', va='bottom',
                              fontsize=13, fontweight='bold', color="0.2")
    ax_3.set_xticks(np.arange(len(rois)))
    ax_3.set_xticklabels(rois)
    ax_3.set_xlim(-0.6, len(rois) - 0.4)
    ax_3.set_ylim(0, sigma_top)
    ax_3.set_title("Q3: σ_dyn vs σ_static  (Gaussian widths of the AFs)",
                   loc='left', pad=10)
    ax_3.set_ylabel("σ  (deg)")
    ax_3.grid(axis='y', alpha=0.25)
    ax_3.legend(loc='best', frameon=False)
    _despine(ax_3)

    # Row 4 (optional): repeat-priming modulator g_HP_dyn_repeat /
    # g_LP_dyn_repeat. These are additive modulators on top of
    # g_HP_dyn / g_LP_dyn for repeat trials (same distractor location
    # as the previous trial). Negative = stronger suppression on repeat.
    if has_repeat:
        ax_i4 = fig.add_subplot(gs[4, 0])
        icon_repeat_priming(ax_i4)
        ax_4 = fig.add_subplot(gs[4, 1])
        rng4 = np.random.default_rng(4)
        ax_4.axhline(0, color="0.6", lw=0.8, ls=(0, (3, 3)), zorder=1)
        # Robust ylim from data percentile.
        q4_vals = pd.concat([df['g_HP_dyn_repeat'],
                             df['g_LP_dyn_repeat']]).dropna()
        q4_lim = max(abs(q4_vals.quantile(0.02)),
                      abs(q4_vals.quantile(0.98)), 1.0) * 1.2
        ax_4.set_ylim(-q4_lim, q4_lim)
        for i, roi in enumerate(rois):
            for ofs, col, name, label_color in [
                (-width / 2, 'g_HP_dyn_repeat', 'HP repeat', '#34495e'),
                (+width / 2, 'g_LP_dyn_repeat', 'LP repeat', '#9b59b6'),
            ]:
                v = df.loc[df['roi'] == roi, col].dropna().to_numpy()
                jitter = rng4.uniform(-width * 0.4, width * 0.4, size=v.size)
                ax_4.scatter(np.full_like(v, i + ofs, dtype=float) + jitter,
                             np.clip(v, -q4_lim, q4_lim),
                             color=label_color, alpha=0.30, s=16, zorder=2,
                             edgecolor='none')
                m, lo, hi = _ci95(v)
                if np.isfinite(m):
                    ax_4.errorbar(i + ofs, m, yerr=[[m - lo], [hi - m]],
                                  fmt='o', color=label_color, ms=8, lw=2.5,
                                  capsize=4, capthick=2.0, zorder=3,
                                  label=name if i == 0 else None)
                if len(v) >= 5:
                    pw = float(wilcoxon(v).pvalue)
                    stars = ('***' if pw < .001 else '**' if pw < .01
                             else '*' if pw < .05 else '·' if pw < .1 else '')
                    if stars:
                        ax_4.annotate(stars,
                                      xy=(i + ofs, m + (hi - m)),
                                      xytext=(0, 4),
                                      textcoords='offset points',
                                      ha='center', va='bottom',
                                      fontsize=11, fontweight='bold',
                                      color=label_color)
        ax_4.set_xticks(np.arange(len(rois)))
        ax_4.set_xticklabels(rois)
        ax_4.set_xlim(-0.6, len(rois) - 0.4)
        ax_4.set_title("Q4: Does the dynamic gain change on REPEAT trials? "
                       "(additive Δ to switch-trial dyn gain)",
                       loc='left', pad=8)
        ax_4.set_ylabel("Δ gain (repeat additive)")
        ax_4.grid(axis='y', alpha=0.25)
        ax_4.legend(loc='best', frameon=False)
        _despine(ax_4)

    n_subs = df['subject'].nunique()
    # Derive a friendly title from the fits-dir name.
    fig.suptitle(f"AF model: group means across {n_subs} subjects",
                 fontsize=18)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches='tight')
    print(f"Wrote {out_path}  ({n_subs} subjects, "
          f"{df['roi'].nunique()} ROIs, {len(df)} fits)")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--analysis', default=None,
                   help='Canonical analysis name in '
                        'notes/data/canonical_models.yml (preferred over '
                        '--fits-dir; mutually exclusive). Auto-prints the '
                        'resolved dir + commit pinned in the registry.')
    p.add_argument('--bids-folder', default='/data/ds-retsupp',
                   help='BIDS root, used with --analysis to resolve the '
                        'fits-dir.')
    p.add_argument(
        '--fits-dir',
        default=None,
        help='Explicit fits-dir (overrides --analysis). For ad-hoc '
             'exploration of non-registered variants only — for '
             'canonical figures use --analysis.')
    p.add_argument('--out',
                   default='notes/figures/af_results.pdf')
    p.add_argument('--summary-tsv',
                   default='notes/data/af_results.tsv')
    p.add_argument('--rois', default='visual',
                   help='"visual" (default), "all", or comma-separated list')
    p.add_argument('--min-mtime', type=float, default=0.0,
                   help='Drop fits with mtime older than this epoch')
    a = p.parse_args()
    if a.analysis and a.fits_dir:
        raise SystemExit("--analysis and --fits-dir are mutually exclusive")
    if a.analysis:
        from retsupp.utils.canonical_models import resolve_analysis, load_registry
        fits_dir = resolve_analysis(a.analysis, bids_folder=a.bids_folder)
        entry = load_registry()[a.analysis]
        pinned = entry.get('pinned_commit', '(not pinned)')
        print(f"Analysis {a.analysis!r} → {fits_dir}")
        print(f"  registry pinned_commit: {pinned}")
        # Also print the dir's own dataset_description.json commit if present.
        desc = fits_dir / 'dataset_description.json'
        if desc.exists():
            import json
            with open(desc) as f:
                d = json.load(f)
            actual = d.get('GeneratedBy', [{}])[0].get('Version', '(no version)')
            print(f"  dir's dataset_description Version: {actual}")
            if pinned != '(not pinned)' and actual != pinned:
                print(f"  WARNING: registry pin {pinned} ≠ actual {actual}")
        else:
            print(f"  (no dataset_description.json in {fits_dir.name} — "
                  f"can't verify commit)")
    elif a.fits_dir:
        fits_dir = Path(a.fits_dir)
    else:
        raise SystemExit("Provide either --analysis <name> or --fits-dir <path>")
    if a.rois == 'visual':
        rois_arg = None  # use default ROI_VISUAL
    elif a.rois == 'all':
        rois_arg = 'all'
    else:
        rois_arg = [r.strip() for r in a.rois.split(',') if r.strip()]
    df = collect_records(fits_dir, min_mtime=a.min_mtime,
                          rois=rois_arg)
    if df.empty:
        raise SystemExit(f"No valid AF fits found under {fits_dir}")
    out_tsv = Path(a.summary_tsv)
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_tsv, sep='\t', index=False)
    print(f"Wrote {out_tsv}  ({len(df)} rows)")
    # Print per-ROI subject counts so the user can see coverage at a glance.
    print("\nPer-ROI subject counts:")
    counts = df.groupby('roi').size()
    for roi in (rois_arg if isinstance(rois_arg, list) else ROI_ORDER):
        n = int(counts.get(roi, 0))
        flag = " ←" if n < 3 else ""
        print(f"  {roi:>6}  n={n}{flag}")
    make_figure(df, a.out, roi_order=ROI_ORDER if rois_arg == 'all'
                else (rois_arg if isinstance(rois_arg, list) else ROI_VISUAL))


if __name__ == '__main__':
    main()
