"""Slide-ready 'PRFs make sense' figure for the talk.

Three rows, stacked vertically, full width — leaves room on the slide
for a brain map + parameter glossary on the left:
  A. σ vs eccentricity per ROI — across-subject mean ± SEM curve per
     ROI, colored by visual hierarchy.
  B. Per-subject median R² per ROI — strip plot, one point per
     subject.
  C. Visual-field PRF center density (KDE) across all subjects, three
     ROIs side by side (V1, V3, LO).

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
import seaborn as sns  # registers 'rocket'/'mako' colormaps
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from matplotlib.patches import Circle

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS))
from make_figures import APERTURE_RADIUS, BUILDUP_PALETTE, base_rc  # noqa
from retsupp.utils.data import select_well_fit_voxels  # noqa: E402

REPO = THIS.parents[2]
# Extended cache with hemi + HRF columns; built by
# `extract_prf_validity_data.py`. Falls back to the basic coverage
# TSV (no hemi/HRF) if the extended one is missing.
TSV_EXT = REPO / "notes" / "data" / "prf_validity_summary.tsv"
TSV_BASIC = REPO / "notes" / "data" / "prf_coverage_summary.tsv"
OUT = THIS / "talk_prf_validity.pdf"


def _resolve_paths(model=None):
    """Resolve TSV + output paths + free-param count for a given model.
    Every model writes a suffixed output (e.g. `_m4.pdf`); model 4
    additionally keeps the bare `prf_validity_summary.tsv` cache for
    backwards-compat with the extractor's default.
    """
    global TSV_EXT, OUT, ACTIVE_MODEL, ACTIVE_N_PARAMS
    if model is None:
        model = 4
    ACTIVE_MODEL = model
    ACTIVE_N_PARAMS = N_PARAMS_BY_MODEL.get(model, 5)
    if model == 4:
        # Default cache name (no suffix) for back-compat with extractor.
        TSV_EXT = REPO / "notes" / "data" / "prf_validity_summary.tsv"
    else:
        TSV_EXT = REPO / "notes" / "data" / f"prf_validity_summary_m{model}.tsv"
    OUT = THIS / f"talk_prf_validity_m{model}.pdf"

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
# Selection itself lives in retsupp.utils.data.select_well_fit_voxels;
# these constants are only used here for F-test DoF + display.
N_TIMEPOINTS = 258               # mean-BOLD timepoints per voxel
# Free params by model. Used for F-test DoF in BH-FDR.
N_PARAMS_BY_MODEL = {
    1: 5,   # x, y, sd, amplitude, baseline
    2: 7,   # + srf_size, srf_amplitude (DoG, no flex HRF)
    3: 7,   # x, y, sd, amp, baseline, hrf_delay, hrf_dispersion
    4: 9,   # DoG + flex HRF
    5: 9,   # DN parameters (no flex HRF)
    6: 11,  # DN + flex HRF
    8: 9,   # DoG variant w/ flex HRF
}
N_PARAMS_M4 = N_PARAMS_BY_MODEL[4]
FDR_ALPHA = 0.05                 # BH-FDR significance level on R² F-test
MASS_IN_APERTURE_THR = 0.50      # ≥ 50% of PRF mass inside aperture
SIGMA_FLOOR = 0.30               # exclude pathologically small PRFs
SIGMA_CEIL = 4.00                # exclude pathologically large PRFs
# Set by _resolve_paths(model) — currently selected model's free-param
# count for the F-test.
ACTIVE_MODEL = 4
ACTIVE_N_PARAMS = N_PARAMS_M4

# ROIs shown in the cross-subject coverage histograms (panel C).
# Sampled across the visual hierarchy so the audience sees the
# progression: early-visual (V1) → mid (V3, V3AB) → lateral occipital
# (LO) → parietal (IPS). Pick 5 — wider than tall enough to read.
COVERAGE_ROIS = ["V1", "V3", "V3AB", "LO", "IPS"]
COVERAGE_CMAP = "viridis"  # classic density-histogram colormap


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


def _kde_grid(df_roi, lim, grid_n, bw=0.18, n_max=15000):
    """Helper: compute a KDE on a square grid; return Z masked outside
    the bar aperture, or None if fewer than 5 voxels."""
    from scipy.stats import gaussian_kde
    if len(df_roi) < 5:
        return None
    if len(df_roi) > n_max:
        df_roi = df_roi.sample(n_max, random_state=0)
    xy = np.vstack([df_roi["x"].to_numpy(), df_roi["y"].to_numpy()])
    kde = gaussian_kde(xy, bw_method=bw)
    g = np.linspace(-lim, lim, grid_n)
    GX, GY = np.meshgrid(g, g)
    Z = kde(np.vstack([GX.ravel(), GY.ravel()])).reshape(grid_n, grid_n)
    mask = (GX ** 2 + GY ** 2) > APERTURE_RADIUS ** 2
    Z = np.where(mask, np.nan, Z)
    return Z


def _draw_apertures_and_axes(ax, palette, edge_color="black", lim=4.0,
                              ring_color="0.25", label_color="0.15"):
    """Aperture outline + 1/2/3° eccentricity rings + center cross."""
    for r_iso in (1.0, 2.0, 3.0):
        ax.add_patch(Circle((0, 0), r_iso, facecolor="none",
                            edgecolor=ring_color, lw=1.0,
                            ls=(0, (2, 4)), alpha=0.55, zorder=4))
        ax.text(0.06, r_iso + 0.05, f"{r_iso:.0f}°",
                color=label_color, fontsize=14, ha="left", va="bottom",
                alpha=0.85, zorder=4)
    ax.add_patch(Circle((0, 0), APERTURE_RADIUS, facecolor="none",
                        edgecolor=edge_color, lw=2.4,
                        ls=(0, (5, 3)), zorder=5))
    ax.plot(0, 0, "+", color=edge_color, ms=14, mew=2.0, zorder=6)


def _draw_field_density(ax, df_roi, palette, lim=4.0, grid_n=160):
    """Cross-subject Gaussian-KDE density of PRF centers within an ROI.
    """
    if len(df_roi) < 5:
        ax.text(0.5, 0.5, "no voxels", transform=ax.transAxes,
                ha="center", va="center", color=palette["muted"])
        return 0

    Z = _kde_grid(df_roi, lim, grid_n)
    if Z is None:
        return 0

    from matplotlib.colors import PowerNorm
    ax.imshow(Z, extent=(-lim, lim, -lim, lim), origin="lower",
              cmap=COVERAGE_CMAP, interpolation="bilinear",
              norm=PowerNorm(gamma=0.45), zorder=0)
    _draw_apertures_and_axes(ax, palette, edge_color="white",
                              ring_color="white", label_color="white")
    return len(df_roi)


def _draw_field_density_split(ax, df_roi, palette, lim=4.0, grid_n=160):
    """Same as _draw_field_density but renders LEFT and RIGHT hemispheres
    separately as semi-transparent contour fills, so the audience sees
    the contralateral organization at a glance: LH PRFs (orange) sit
    in the right field, RH PRFs (teal) in the left field.
    """
    from matplotlib.colors import LinearSegmentedColormap as LSC, to_rgba
    if "hemi" not in df_roi.columns or len(df_roi) < 5:
        return _draw_field_density(ax, df_roi, palette, lim, grid_n)

    # White background inside the aperture, soft grey outside.
    ax.set_facecolor("white")

    # Compute per-hemi KDEs.
    Z_L = _kde_grid(df_roi[df_roi.hemi == "L"], lim, grid_n)
    Z_R = _kde_grid(df_roi[df_roi.hemi == "R"], lim, grid_n)

    hemi_colors = {"L": "#E76F51", "R": "#1B4965"}  # warm / cool
    for hemi, Z in (("L", Z_L), ("R", Z_R)):
        if Z is None:
            continue
        col = hemi_colors[hemi]
        cmap = LSC.from_list(
            f"hemi_{hemi}", [to_rgba(col, 0), to_rgba(col, 0.95)])
        # Normalize Z to [0, 1] for level placement
        zmax = np.nanmax(Z)
        if not np.isfinite(zmax) or zmax <= 0:
            continue
        Zn = Z / zmax
        # Filled contours at progressive density levels.
        ax.contourf(
            np.linspace(-lim, lim, grid_n),
            np.linspace(-lim, lim, grid_n),
            Zn,
            levels=[0.10, 0.25, 0.45, 0.70, 1.0001],
            cmap=cmap, zorder=1)
    _draw_apertures_and_axes(ax, palette, edge_color="black",
                              ring_color="0.35", label_color="0.15")
    return len(df_roi)


def main():
    if TSV_EXT.exists():
        print(f"Loading {TSV_EXT}...")
        df = pd.read_csv(TSV_EXT, sep="\t")
        has_hemi = "hemi" in df.columns
        has_hrf = ("hrf_delay" in df.columns
                   and "hrf_dispersion" in df.columns)
    else:
        print(f"Loading {TSV_BASIC} (extended cache not found)...")
        df = pd.read_csv(TSV_BASIC, sep="\t")
        has_hemi = False
        has_hrf = False
    print(f"  hemi available: {has_hemi};  HRF available: {has_hrf}")

    good, fdr_thr = select_well_fit_voxels(
        df, n_params=ACTIVE_N_PARAMS, n_timepoints=N_TIMEPOINTS,
        fdr_alpha=FDR_ALPHA, mass_threshold=MASS_IN_APERTURE_THR,
        sigma_floor=SIGMA_FLOOR, sigma_ceil=SIGMA_CEIL,
        aperture_radius=APERTURE_RADIUS)
    print(f"  BH-FDR R² threshold "
          f"(model {ACTIVE_MODEL}, k={ACTIVE_N_PARAMS}): {fdr_thr:.4f}")
    n_pre = len(df); n_post = len(good)
    print(f"  Pre-selection:  {n_pre:>7,} voxels")
    print(f"  Post-selection: {n_post:>7,} voxels  "
          f"({100 * n_post / n_pre:.1f}%)")

    p = BUILDUP_PALETTE
    rc = base_rc(p)
    rc.update({
        "font.size": 22, "axes.titlesize": 26, "axes.labelsize": 24,
        "xtick.labelsize": 20, "ytick.labelsize": 20,
        "legend.fontsize": 19, "figure.titlesize": 30,
    })
    with plt.rc_context(rc):
        # 16:9 slide assumption — flatmap occupies the LEFT ~35% of
        # the slide, this figure fills the right ~65% (≈ 10.4:9 ≈
        # 1.16 aspect). 3 wide stacked rows; no left padding wasted
        # since the brain sits outside the figure on the slide.
        fig = plt.figure(figsize=(15.0, 11.6))
        gs = fig.add_gridspec(
            3, 1, height_ratios=[1.05, 0.85, 1.20],
            hspace=0.65,
            left=0.075, right=0.98, top=0.95, bottom=0.05)
        # Panel A row is split: plot (left ~75%) + legend column (right ~25%).
        # This shrinks the σ-vs-ecc axes width as requested.
        gs_A = gs[0, 0].subgridspec(1, 2, width_ratios=[3.1, 1.0],
                                    wspace=0.05)

        # =================================================================
        # Row A: σ vs eccentricity — mean ± SEM per ROI.
        # Plot in the left sub-cell; legend in the right sub-cell so
        # the curves can use the full plot width without colliding.
        ax_A = fig.add_subplot(gs_A[0, 0])
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
        ax_A.set_title("PRF size vs. eccentricity",
                       pad=8, loc="left", weight="bold",
                       fontsize=22)
        ax_A.set_xlim(0, max(ECC_BINS))
        ax_A.grid(alpha=0.20)
        # Legend lives in its own sub-cell (gs_A[0, 1]) so the plot
        # stays narrower and the legend never overlaps the curves.
        ax_A_leg = fig.add_subplot(gs_A[0, 1])
        ax_A_leg.axis("off")
        handles, labels = ax_A.get_legend_handles_labels()
        ax_A_leg.legend(handles, labels, ncol=2, frameon=False,
                        loc="center left",
                        handlelength=1.4, labelspacing=0.40,
                        columnspacing=1.0,
                        borderaxespad=0.0)
        # Aperture marker on the x-axis. Label placed at the BOTTOM
        # so it doesn't compete with the title.
        ax_A.axvline(APERTURE_RADIUS, color=p["muted"], lw=1.2,
                     ls=(0, (4, 3)), alpha=0.75)
        y_lo, y_hi = ax_A.get_ylim()
        ax_A.text(APERTURE_RADIUS - 0.06, y_lo + 0.04 * (y_hi - y_lo),
                  "Aperture", color=p["muted"], fontsize=15,
                  ha="right", va="bottom", style="italic")

        # =================================================================
        # Row B: per-subject medians per ROI — R² strip on the left,
        # two narrow HRF strips on the right (delay + dispersion).
        # Title-less so the audience reads the axes.
        if has_hrf:
            gs_B = gs[1, 0].subgridspec(
                1, 3, width_ratios=[3.0, 1.0, 1.0], wspace=0.32)
        else:
            gs_B = gs[1, 0].subgridspec(1, 1)

        def _per_roi_strip(ax, col, ylabel, title=None,
                           ylim_bottom=None, ylim_top=None,
                           ref_line=None):
            per_sub = (good.groupby(["subject", "roi"])[col].median()
                       .unstack().reindex(columns=ROI_ORDER))
            np.random.seed(0)
            for i, roi in enumerate(ROI_ORDER):
                vals = per_sub[roi].dropna().values
                if len(vals) == 0:
                    continue
                jitter = np.random.uniform(-0.20, 0.20, size=len(vals))
                ax.scatter(np.full(len(vals), i) + jitter, vals,
                           color=ROI_COLORS[roi], alpha=0.78, s=55,
                           edgecolor="white", linewidth=0.7)
                ax.scatter([i], [np.median(vals)], marker="_",
                           color=p["fg"], s=720, lw=3.2, zorder=10)
            ax.set_xticks(range(len(ROI_ORDER)))
            ax.set_xticklabels(ROI_ORDER, rotation=0, ha="center")
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.20, axis="y")
            if ylim_bottom is not None:
                ax.set_ylim(bottom=ylim_bottom)
            if ylim_top is not None:
                ax.set_ylim(top=ylim_top)
            ax.set_xlim(-0.6, len(ROI_ORDER) - 0.4)
            if title is not None:
                ax.set_title(title, pad=6, loc="left", weight="bold",
                             fontsize=22)
            if ref_line is not None:
                ax.axhline(ref_line, color=p["muted"], lw=1.2,
                           ls=(0, (4, 3)), alpha=0.7, zorder=1)

        ax_B = fig.add_subplot(gs_B[0, 0])
        _per_roi_strip(ax_B, "r2", "Median PRF R²",
                       title="Per-voxel R² on concatenated runs "
                             "(no averaging)",
                       ylim_bottom=0)
        # Tick labels small enough to read on the narrow strips.
        for label in ax_B.get_xticklabels():
            label.set_fontsize(18)

        if has_hrf:
            ax_Bd = fig.add_subplot(gs_B[0, 1])
            _per_roi_strip(ax_Bd, "hrf_delay", "HRF delay (s)",
                           title="HRF delay")
            for label in ax_Bd.get_xticklabels():
                label.set_fontsize(13)
                label.set_rotation(45)
                label.set_ha("right")

            ax_Bx = fig.add_subplot(gs_B[0, 2])
            _per_roi_strip(ax_Bx, "hrf_dispersion", "Dispersion",
                           title="HRF dispersion")
            for label in ax_Bx.get_xticklabels():
                label.set_fontsize(13)
                label.set_rotation(45)
                label.set_ha("right")

        # =================================================================
        # Row C: 2D density histogram of PRF centers ACROSS subjects,
        # one panel per ROI in COVERAGE_ROIS, all side by side along
        # the full width of the row.
        sub_gs = gs[2, 0].subgridspec(
            1, len(COVERAGE_ROIS),
            wspace=0.10)
        for j, roi in enumerate(COVERAGE_ROIS):
            ax = fig.add_subplot(sub_gs[0, j])
            roi_df = good[good.roi == roi]
            _draw_field_density(ax, roi_df, p)
            ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)
            ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            ax.set_title(f"{roi}", pad=6, weight="bold",
                         fontsize=22, color=ROI_COLORS[roi])

        # Row-level caption above the coverage row.
        bb = sub_gs[0, 0].get_position(fig)
        fig.text(0.075, bb.y1 + 0.052,
                 "PRF center density across subjects "
                 "(rings = 1°, 2°, 3°)",
                 fontsize=22, ha="left", va="bottom",
                 weight="bold", color=p["fg"])

        OUT.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUT, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {OUT}")


if __name__ == "__main__":
    import sys
    model = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    _resolve_paths(model)
    main()
