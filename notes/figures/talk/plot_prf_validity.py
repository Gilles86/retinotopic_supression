"""Slide-ready 'PRFs make sense' figure for the talk.

One figure, three panels:
  A. σ vs eccentricity per ROI — across-subject mean ± SEM curve per
     ROI, colored by visual hierarchy.
  B. Per-subject median R² per ROI — strip plot, one point per
     subject.
  C. Visual-field PRF center density (KDE) across all subjects in
     V1, V2, V3.

Voxel selection criteria (applied to all panels):
  * R² above per-voxel F-test p-value's BH-FDR threshold (α = 0.05)
  * ≥ 50% of PRF mass inside the bar aperture
  * σ ∈ [0.3°, 4°]

Loads cached parameter-summary TSV produced by
`retsupp/visualize/paper/plot_prf_coverage_per_roi.py`. Operates on
canonical model 4 (DoG with flexible HRF — the mean-fit baseline).

Output: notes/figures/talk/talk_prf_validity.pdf
"""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from matplotlib.patches import Circle

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS))
from make_figures import APERTURE_RADIUS, BUILDUP_PALETTE, base_rc  # noqa

REPO = THIS.parents[2]
TSV = REPO / "notes" / "data" / "prf_coverage_summary.tsv"
OUT = THIS / "talk_prf_validity.pdf"

ROI_ORDER = ["V1", "V2", "V3", "V3AB", "hV4", "LO", "TO", "VO",
             "IPS", "SPL1", "FEF"]

# Color hierarchy: cool early visual → warm parietal/frontal.
ROI_CMAP = LinearSegmentedColormap.from_list(
    "hierarchy",
    ["#1B4965", "#2A9D8F", "#7DB46C", "#F4A261", "#E76F51", "#9D2D55"])
ROI_COLORS = {roi: ROI_CMAP(i / (len(ROI_ORDER) - 1))
              for i, roi in enumerate(ROI_ORDER)}

# Eccentricity bins for the σ-vs-eccentricity curves.
ECC_BINS = np.arange(0, 4.5, 0.5)
ECC_CTRS = 0.5 * (ECC_BINS[:-1] + ECC_BINS[1:])

# Voxel-selection criteria (applied to all panels).
N_TIMEPOINTS = 258               # mean-BOLD timepoints per voxel (model 4)
N_PARAMS_M4 = 9                  # x, y, sd, amp, baseline, srf_size,
                                 # srf_amp, hrf_delay, hrf_dispersion
FDR_ALPHA = 0.05                 # BH-FDR significance level on R² F-test
MASS_IN_APERTURE_THR = 0.50      # ≥ 50% of PRF mass inside aperture
SIGMA_FLOOR = 0.30               # exclude pathologically small PRFs
SIGMA_CEIL = 4.00                # exclude pathologically large PRFs

# ROIs shown in the cross-subject coverage histograms (panel C).
COVERAGE_ROIS = ["V1", "V3", "LO"]


def _per_subject_sigma_curve(df_roi, ecc_bins=ECC_BINS):
    """Return DataFrame indexed by subject, columns = bin centers,
    values = median PRF σ in each bin.
    """
    out = {}
    for sub, g in df_roi.groupby("subject"):
        binned = pd.cut(g["eccen"], bins=ecc_bins, include_lowest=True)
        med = g.groupby(binned, observed=False)["sd"].median()
        out[sub] = med.values
    res = pd.DataFrame(out, index=ECC_CTRS).T
    return res


def _draw_field_density(ax, df_roi, palette, lim=4.0, grid_n=160):
    """Cross-subject Gaussian-KDE density of PRF centers within an ROI.
    Smoother than a 2D histogram, masked outside the bar aperture so
    the panel reads as 'where PRFs put their centers within the
    stimulated field'.
    """
    from scipy.stats import gaussian_kde
    if len(df_roi) < 5:
        ax.text(0.5, 0.5, "no voxels", transform=ax.transAxes,
                ha="center", va="center", color=palette["muted"])
        return 0

    # Sub-sample to keep KDE fast; ~20k points is plenty.
    n_max = 20000
    if len(df_roi) > n_max:
        df_roi = df_roi.sample(n_max, random_state=0)

    xy = np.vstack([df_roi["x"].to_numpy(), df_roi["y"].to_numpy()])
    kde = gaussian_kde(xy, bw_method=0.18)
    g = np.linspace(-lim, lim, grid_n)
    GX, GY = np.meshgrid(g, g)
    Z = kde(np.vstack([GX.ravel(), GY.ravel()])).reshape(grid_n, grid_n)
    # Mask outside aperture.
    mask = (GX ** 2 + GY ** 2) > APERTURE_RADIUS ** 2
    Z = np.where(mask, np.nan, Z)

    ax.imshow(Z, extent=(-lim, lim, -lim, lim), origin="lower",
              cmap="viridis", interpolation="bilinear", zorder=0)
    # Eccentricity isolines at 1°, 2°, 3° — pale white so the audience
    # can read off the radial scale.
    for r_iso in (1.0, 2.0, 3.0):
        ax.add_patch(Circle((0, 0), r_iso, facecolor="none",
                            edgecolor="white", lw=1.0,
                            ls=(0, (2, 4)), alpha=0.55, zorder=4))
        # tiny eccentricity labels just above each ring
        ax.text(0.06, r_iso + 0.05, f"{r_iso:.0f}°",
                color="white", fontsize=17, ha="left", va="bottom",
                alpha=0.85, zorder=4)
    # Aperture circle outline (thick dashed).
    ax.add_patch(Circle((0, 0), APERTURE_RADIUS, facecolor="none",
                        edgecolor="white", lw=2.4,
                        ls=(0, (5, 3)), zorder=5))
    ax.plot(0, 0, "+", color="white", ms=16, mew=2.2, zorder=6)
    return len(df_roi)


def _r2_pvalues(r2, n=N_TIMEPOINTS, k=N_PARAMS_M4):
    """Per-voxel one-sided F-test p-values from R²:
        F = (R²/k) / ((1−R²)/(n−k−1))  ~  F(k, n−k−1)  under H0.
    """
    from scipy.stats import f as f_dist
    r2c = np.clip(r2, 0.0, 0.999999)
    df1, df2 = k, n - k - 1
    F = (r2c / df1) / ((1.0 - r2c) / df2)
    return 1.0 - f_dist.cdf(F, df1, df2)


def _bh_fdr_r2_threshold(r2, alpha=FDR_ALPHA, n=N_TIMEPOINTS, k=N_PARAMS_M4):
    """Return the smallest R² value that survives BH-FDR at level α
    across the supplied r² array. Implementation: BH on per-voxel
    p-values, then map back to the corresponding R² cutoff.
    """
    from statsmodels.stats.multitest import multipletests
    p = _r2_pvalues(r2, n=n, k=k)
    rejected, p_adj, *_ = multipletests(p, alpha=alpha, method="fdr_bh")
    if not rejected.any():
        return np.inf
    return float(np.min(r2[rejected]))


def main():
    from scipy.stats import norm
    print(f"Loading {TSV}...")
    df = pd.read_csv(TSV, sep="\t")
    df["eccen"] = np.sqrt(df["x"] ** 2 + df["y"] ** 2)
    # Aperture-mass fraction: same radial-CDF approximation as
    # `Subject.get_aperture_mass_fraction` and `filter_prf_inside_aperture`.
    sd_safe = df["sd"].clip(lower=0.05)
    df["mass_in"] = 1.0 - norm.cdf(
        (df["eccen"] - APERTURE_RADIUS) / sd_safe)

    # FDR-corrected R² threshold across all voxels.
    fdr_thr = _bh_fdr_r2_threshold(
        df["r2"].to_numpy(), alpha=FDR_ALPHA,
        n=N_TIMEPOINTS, k=N_PARAMS_M4)
    print(f"  BH-FDR R² threshold (α={FDR_ALPHA}, "
          f"n={N_TIMEPOINTS}, k={N_PARAMS_M4}): {fdr_thr:.4f}")

    sel = ((df["r2"] >= fdr_thr)
           & (df["mass_in"] >= MASS_IN_APERTURE_THR)
           & (df["sd"] >= SIGMA_FLOOR)
           & (df["sd"] <= SIGMA_CEIL))
    good = df[sel].copy()
    n_pre = len(df); n_post = len(good)
    print(f"  Pre-selection:  {n_pre:>7,} voxels")
    print(f"  Post-selection: {n_post:>7,} voxels  "
          f"({100 * n_post / n_pre:.1f}%)")

    p = BUILDUP_PALETTE
    rc = base_rc(p)
    rc.update({
        "font.size": 20, "axes.titlesize": 24, "axes.labelsize": 22,
        "xtick.labelsize": 19, "ytick.labelsize": 19,
        "legend.fontsize": 17, "figure.titlesize": 28,
    })
    with plt.rc_context(rc):
        fig = plt.figure(figsize=(24.0, 8.6))
        gs = fig.add_gridspec(
            2, 4, height_ratios=[1.0, 1.0],
            width_ratios=[1.55, 0.95, 0.95, 0.95],
            hspace=0.45, wspace=0.28,
            left=0.045, right=0.99, top=0.84, bottom=0.10)

        # =================================================================
        # Panel A: σ vs eccentricity — mean ± SEM per ROI
        ax_A = fig.add_subplot(gs[:, 0])
        for roi in ROI_ORDER:
            roi_df = good[good.roi == roi]
            if len(roi_df) == 0:
                continue
            curves = _per_subject_sigma_curve(roi_df)
            mean_c = curves.mean()
            sem_c = curves.sem()
            valid = mean_c.notna()
            col = ROI_COLORS[roi]
            ax_A.fill_between(ECC_CTRS[valid],
                              (mean_c - sem_c)[valid],
                              (mean_c + sem_c)[valid],
                              alpha=0.18, color=col, linewidth=0)
            ax_A.plot(ECC_CTRS[valid], mean_c[valid],
                      color=col, lw=3.0, marker="o", ms=7.0,
                      label=roi)
        ax_A.set_xlabel("PRF eccentricity (°)")
        ax_A.set_ylabel("PRF σ (°)")
        ax_A.set_title("σ grows with eccentricity & hierarchy",
                       pad=10, loc="left", weight="bold")
        ax_A.set_xlim(0, max(ECC_BINS))
        ax_A.grid(alpha=0.20)
        ax_A.legend(ncol=2, frameon=False, loc="upper left",
                    handlelength=1.6, columnspacing=1.0)
        # Aperture marker on the x-axis
        ax_A.axvline(APERTURE_RADIUS, color=p["muted"], lw=1.0,
                     ls=(0, (4, 3)), alpha=0.7)
        ax_A.text(APERTURE_RADIUS - 0.05, ax_A.get_ylim()[1] * 0.95,
                  "Aperture", color=p["muted"], fontsize=17,
                  ha="right", va="top", style="italic")

        # =================================================================
        # Panel B: per-subject median R² per ROI — strip. Computed on the
        # SELECTED voxel set, so unphysical / leaky PRFs that pollute V1
        # in the unfiltered data are excluded.
        ax_B = fig.add_subplot(gs[:, 1])
        per_sub_r2 = (good.groupby(["subject", "roi"])["r2"].median()
                      .unstack().reindex(columns=ROI_ORDER))
        np.random.seed(0)
        for i, roi in enumerate(ROI_ORDER):
            vals = per_sub_r2[roi].dropna().values
            if len(vals) == 0:
                continue
            jitter = np.random.uniform(-0.20, 0.20, size=len(vals))
            ax_B.scatter(np.full(len(vals), i) + jitter, vals,
                         color=ROI_COLORS[roi], alpha=0.75, s=55,
                         edgecolor="white", linewidth=0.7)
            ax_B.scatter([i], [np.median(vals)], marker="_",
                         color=p["fg"], s=600, lw=3.0, zorder=10)
        ax_B.set_xticks(range(len(ROI_ORDER)))
        ax_B.set_xticklabels(ROI_ORDER, rotation=35, ha="right")
        ax_B.set_ylabel("Median PRF R²")
        ax_B.set_title("Fit quality decays with hierarchy",
                       pad=10, loc="left", weight="bold")
        ax_B.grid(alpha=0.20, axis="y")
        ax_B.set_ylim(bottom=0)

        # =================================================================
        # Panels C: 2D density histogram of PRF centers ACROSS subjects,
        # one panel per ROI (V1, V3AB, IPS). Shows where in the visual
        # field each ROI's PRF population concentrates.
        coverage_cells = [(0, 2), (0, 3), (1, 2)]
        for (row, col), roi in zip(coverage_cells, COVERAGE_ROIS):
            ax = fig.add_subplot(gs[row, col])
            roi_df = good[good.roi == roi]
            n = _draw_field_density(ax, roi_df, p)
            ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)
            ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            n_subs = roi_df.subject.nunique() if len(roi_df) else 0
            ax.set_title(f"{roi}", pad=4, weight="bold")

        # 4th cell: caption + selection criteria.
        ax_leg = fig.add_subplot(gs[1, 3])
        ax_leg.axis("off")
        ax_leg.text(0.0, 0.95,
                    "PRF center density",
                    fontsize=24, ha="left", va="top", weight="bold",
                    color=p["fg"], transform=ax_leg.transAxes)
        ax_leg.text(0.0, 0.76,
                    f"Pooled over subjects.\n"
                    f"Rings: 1, 2, 3°.\n"
                    f"Aperture = {APERTURE_RADIUS:.2f}°.",
                    fontsize=21, ha="left", va="top", color=p["fg"],
                    transform=ax_leg.transAxes)
        ax_leg.text(0.0, 0.36,
                    "Voxel selection",
                    fontsize=23, ha="left", va="top", weight="bold",
                    color=p["fg"], transform=ax_leg.transAxes)
        ax_leg.text(0.0, 0.24,
                    f"R² > {fdr_thr:.3f}  (BH-FDR α={FDR_ALPHA})\n"
                    f"≥ {int(MASS_IN_APERTURE_THR*100)}% mass in aperture\n"
                    f"σ ∈ [{SIGMA_FLOOR}, {SIGMA_CEIL}]°",
                    fontsize=20, ha="left", va="top", color=p["fg"],
                    transform=ax_leg.transAxes)

        fig.suptitle(
            "Canonical PRF fit (model 4) — parameters look as expected",
            x=0.045, y=0.96, ha="left", fontsize=28,
            weight="bold", color=p["fg"])
        fig.text(0.045, 0.905,
                 f"{good.subject.nunique()} subjects · "
                 f"{n_post:,} voxels (of {n_pre:,}) pass selection",
                 fontsize=21, color=p["muted"], style="italic")

        OUT.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUT, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
