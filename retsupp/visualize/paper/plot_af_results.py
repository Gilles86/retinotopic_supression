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
import pickle
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROI_ORDER = ['V1', 'V2', 'V3', 'V3AB', 'hV4', 'LO', 'TO', 'VO']
PARSE_RE = re.compile(r"sub-(\d+)_roi-([^_]+)_mode-[^_]+_(?:.*)-af-prf-fit\.pkl$")

plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "figure.titlesize": 17,
})


def collect_records(fits_dir: Path) -> pd.DataFrame:
    """Walk an AF derivative tree; return one row per (subject, ROI) fit."""
    paths = sorted(glob.glob(str(fits_dir / 'sub-*/sub-*-af-prf-fit.pkl')))
    rows = []
    for p in paths:
        m = PARSE_RE.search(Path(p).name)
        if not m:
            continue
        sub = int(m.group(1))
        roi = m.group(2)
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
        ))
    df = pd.DataFrame(rows)
    if not df.empty:
        df['g_HP_minus_g_LP_static'] = df['g_HP'] - df['g_LP']
        df['g_HP_minus_g_LP_dyn'] = df['g_HP_dyn'] - df['g_LP_dyn']
        df['sigma_static_minus_dyn'] = df['sigma_AF'] - df['sigma_dyn']
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
    ax.set_xlim(-5.0, 5.0)
    ax.set_ylim(-5.0, 5.0)
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
    ax.text(tx + 0.2, ty + 0.7, "g_T,dyn > 0",
            fontsize=10, color="#c0392b", ha="left", va="bottom")


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
    ax.text(hx - 0.4, hy + 0.7, "g_HP < g_LP",
            fontsize=10, color="#2980b9", ha="left", va="bottom")


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
    ax.text(0.02, 0.96, "σ_static (orange)",
            color="#e67e22", fontsize=10,
            transform=ax.transAxes, ha="left", va="top")
    ax.text(0.02, 0.86, "σ_dyn  (red)",
            color="#c0392b", fontsize=10,
            transform=ax.transAxes, ha="left", va="top")


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
                  point_jitter=0.18, color="#34495e"):
    rois = [r for r in ROI_ORDER if r in df['roi'].unique()]
    x_pos = np.arange(len(rois))
    if zero_line:
        ax.axhline(0, color="0.6", lw=0.8, ls=(0, (3, 3)), zorder=1)
    rng = np.random.default_rng(0)
    for i, roi in enumerate(rois):
        sub_vals = df.loc[df['roi'] == roi, value_col].dropna().to_numpy()
        # Subject-level points with horizontal jitter.
        jitter = rng.uniform(-point_jitter, point_jitter, size=sub_vals.size)
        ax.scatter(np.full_like(sub_vals, i, dtype=float) + jitter, sub_vals,
                   color=color, alpha=0.35, s=18, zorder=2,
                   edgecolor="none")
        # Mean + 95% CI bar.
        m, lo, hi = _ci95(sub_vals)
        if np.isfinite(m):
            ax.errorbar(i, m, yerr=[[m - lo], [hi - m]],
                        fmt='o', color=color, ms=8, lw=2.5,
                        capsize=4, capthick=2.0, zorder=3)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(rois)
    ax.set_xlim(-0.6, len(rois) - 0.4)
    ax.set_title(title, loc='left', pad=8)
    ax.set_ylabel(ylabel)
    ax.grid(axis='y', alpha=0.25)


# ---------------------------------------------------------------------------
# Main figure
# ---------------------------------------------------------------------------

def make_figure(df, out_path):
    fig = plt.figure(figsize=(13, 11))
    gs = fig.add_gridspec(
        nrows=3, ncols=2,
        width_ratios=[1.2, 5.5], height_ratios=[1, 1, 1],
        hspace=0.45, wspace=0.05,
        left=0.07, right=0.98, top=0.94, bottom=0.07,
    )

    # Row 1: target attracts.
    ax_i1 = fig.add_subplot(gs[0, 0])
    icon_target_attracts(ax_i1)
    ax_1 = fig.add_subplot(gs[0, 1])
    _per_roi_plot(ax_1, df, 'g_T_dyn',
                  title="Q1: Does the target attract attention?",
                  ylabel="g_T,dyn  (dynamic gain at target)",
                  color="#c0392b")

    # Row 2: HPD suppression -- show both static and dynamic differentials.
    ax_i2 = fig.add_subplot(gs[1, 0])
    icon_hpd_suppression(ax_i2)
    ax_2 = fig.add_subplot(gs[1, 1])
    # Stack both: static and dyn as two side-by-side strips.
    rois = [r for r in ROI_ORDER if r in df['roi'].unique()]
    width = 0.36
    rng = np.random.default_rng(1)
    ax_2.axhline(0, color="0.6", lw=0.8, ls=(0, (3, 3)), zorder=1)
    for i, roi in enumerate(rois):
        for ofs, col, name, label_color in [
            (-width / 2, 'g_HP_minus_g_LP_static', 'static', '#2980b9'),
            (+width / 2, 'g_HP_minus_g_LP_dyn', 'dyn', '#8e44ad'),
        ]:
            v = df.loc[df['roi'] == roi, col].dropna().to_numpy()
            jitter = rng.uniform(-width * 0.4, width * 0.4, size=v.size)
            ax_2.scatter(np.full_like(v, i + ofs, dtype=float) + jitter, v,
                         color=label_color, alpha=0.30, s=16, zorder=2,
                         edgecolor="none")
            m, lo, hi = _ci95(v)
            if np.isfinite(m):
                ax_2.errorbar(i + ofs, m, yerr=[[m - lo], [hi - m]],
                              fmt='o', color=label_color, ms=8, lw=2.5,
                              capsize=4, capthick=2.0, zorder=3,
                              label=name if i == 0 else None)
    ax_2.set_xticks(np.arange(len(rois)))
    ax_2.set_xticklabels(rois)
    ax_2.set_xlim(-0.6, len(rois) - 0.4)
    ax_2.set_title("Q2: Is the HP-distractor location suppressed? "
                   "(g_HP − g_LP)",
                   loc='left', pad=8)
    ax_2.set_ylabel("Δ gain (HP − LP)")
    ax_2.grid(axis='y', alpha=0.25)
    ax_2.legend(loc='best', frameon=False)

    # Row 3: static σ vs dynamic σ.
    ax_i3 = fig.add_subplot(gs[2, 0])
    icon_static_vs_dyn_sigma(ax_i3)
    ax_3 = fig.add_subplot(gs[2, 1])
    rois = [r for r in ROI_ORDER if r in df['roi'].unique()]
    rng = np.random.default_rng(2)
    for i, roi in enumerate(rois):
        for ofs, col, label_color, name in [
            (-width / 2, 'sigma_AF', '#e67e22', 'σ_static'),
            (+width / 2, 'sigma_dyn', '#c0392b', 'σ_dyn'),
        ]:
            v = df.loc[df['roi'] == roi, col].dropna().to_numpy()
            jitter = rng.uniform(-width * 0.4, width * 0.4, size=v.size)
            ax_3.scatter(np.full_like(v, i + ofs, dtype=float) + jitter, v,
                         color=label_color, alpha=0.30, s=16, zorder=2,
                         edgecolor="none")
            m, lo, hi = _ci95(v)
            if np.isfinite(m):
                ax_3.errorbar(i + ofs, m, yerr=[[m - lo], [hi - m]],
                              fmt='o', color=label_color, ms=8, lw=2.5,
                              capsize=4, capthick=2.0, zorder=3,
                              label=name if i == 0 else None)
    ax_3.set_xticks(np.arange(len(rois)))
    ax_3.set_xticklabels(rois)
    ax_3.set_xlim(-0.6, len(rois) - 0.4)
    ax_3.set_ylim(bottom=0)
    ax_3.set_title("Q3: Is the static AF wider than the dynamic AF?",
                   loc='left', pad=8)
    ax_3.set_ylabel("σ  (deg)")
    ax_3.grid(axis='y', alpha=0.25)
    ax_3.legend(loc='best', frameon=False)

    n_subs = df['subject'].nunique()
    fig.suptitle(f"DoG dynamic AF (v3 + target + shared σ): "
                 f"group means across {n_subs} subjects",
                 fontsize=16)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches='tight')
    print(f"Wrote {out_path}  ({n_subs} subjects, "
          f"{df['roi'].nunique()} ROIs, {len(df)} fits)")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        '--fits-dir',
        default='/data/ds-retsupp/derivatives/'
                'af_prf_joint_dynamic_v3_dog_with_target_sharedSigma')
    p.add_argument('--out',
                   default='notes/figures/af_results.pdf')
    p.add_argument('--summary-tsv',
                   default='notes/data/af_results.tsv')
    a = p.parse_args()
    df = collect_records(Path(a.fits_dir))
    if df.empty:
        raise SystemExit(f"No valid AF fits found under {a.fits_dir}")
    out_tsv = Path(a.summary_tsv)
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_tsv, sep='\t', index=False)
    print(f"Wrote {out_tsv}  ({len(df)} rows)")
    make_figure(df, a.out)


if __name__ == '__main__':
    main()
