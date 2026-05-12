"""Companion to `plot_prf_validity.py` — show what the DoG-specific
parameters (surround amplitude and surround σ) look like across ROIs
for canonical model 4 (DoG with flexible HRF).

Three panels:
  A. Center σ vs surround σ per ROI (across-subject mean ± SEM), as
     a function of ROI hierarchy. Surround σ = sd · srf_size.
  B. Surround / center σ ratio (= srf_size) per ROI — strip plot.
  C. Surround / center amplitude fraction (= srf_amplitude) per ROI —
     strip plot.
  D. Canonical 1-D DoG cross-section using median params for a few
     ROIs (V1, V3, LO, IPS) — illustrates what 'DoG' looks like.

Voxel selection mirrors `plot_prf_validity.py`:
  R² > FDR threshold, ≥ 50% mass in aperture, σ ∈ [0.3°, 4°]. Plus an
  additional filter that srf_amplitude ∈ [0.01, 1.0] and srf_size > 1
  (surround should be wider than center for the fit to be a meaningful
  DoG; otherwise the surround Gaussian collapses into the center).

Output:
  notes/data/prf_dog_params_summary.tsv  (cached per-voxel TSV;
       re-used across runs)
  notes/figures/talk/talk_prf_dog_params.pdf  (the figure)
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS))
from make_figures import APERTURE_RADIUS, BUILDUP_PALETTE, base_rc  # noqa
from plot_prf_validity import (  # noqa: E402
    FDR_ALPHA, MASS_IN_APERTURE_THR, SIGMA_FLOOR, SIGMA_CEIL,
    N_TIMEPOINTS, N_PARAMS_M4,
    ROI_ORDER, ROI_COLORS, ECC_BINS, ECC_CTRS,
)
from retsupp.utils.data import select_well_fit_voxels  # noqa: E402

REPO = THIS.parents[2]
TSV = REPO / "notes" / "data" / "prf_dog_params_summary.tsv"
OUT = THIS / "talk_prf_dog_params.pdf"

# Additional DoG-specific filters.
SRF_SIZE_FLOOR = 1.0     # surround must be at least as wide as center
SRF_AMP_FLOOR = 0.01     # exclude voxels where surround amp ~ 0
SRF_AMP_CEIL = 1.0       # srf_amp ∈ [0, 1] by construction

# Example ROIs to show as 1D DoG cross-sections.
PROFILE_ROIS = ["V1", "V3", "LO", "IPS"]


# ---------------------------------------------------------------------------
# Extraction (slow — reads NIfTIs and resamples; cached to a TSV)
def extract_dog_params(bids_folder="/data/ds-retsupp", model=4):
    """Per-voxel DoG parameters for every (subject, ROI). Returns a
    long-format DataFrame and also writes the TSV cache.
    """
    from retsupp.utils.data import Subject
    from nilearn import image

    rois = ROI_ORDER
    rows = []
    for s in range(1, 31):
        try:
            sub = Subject(s, bids_folder=bids_folder)
            prf = sub.get_prf_parameters_volume(model=model,
                                                return_images=True)
        except Exception as e:
            print(f"  sub-{s:02d}: PRF unavailable ({e})")
            continue
        for roi in rois:
            try:
                mask_img = sub.get_retinotopic_roi(roi=roi, bold_space=True)
            except Exception:
                continue
            m = mask_img.get_fdata().astype(bool)
            cols = {}
            for par in ["x", "y", "sd", "amplitude",
                        "srf_size", "srf_amplitude", "r2"]:
                if par not in prf.index:
                    continue
                par_img = image.resample_to_img(
                    prf[par], target_img=mask_img,
                    interpolation="linear",
                    force_resample=True, copy_header=True,
                )
                cols[par] = par_img.get_fdata()[m]
            n = len(cols.get("r2", []))
            if n == 0:
                continue
            for i in range(n):
                rows.append(dict(
                    subject=s, roi=roi,
                    x=float(cols["x"][i]), y=float(cols["y"][i]),
                    sd=float(cols["sd"][i]),
                    amplitude=float(cols.get("amplitude",
                                              np.full(n, np.nan))[i]),
                    srf_size=float(cols.get("srf_size",
                                             np.full(n, np.nan))[i]),
                    srf_amplitude=float(cols.get("srf_amplitude",
                                                  np.full(n, np.nan))[i]),
                    r2=float(cols["r2"][i]),
                ))
            print(f"  sub-{s:02d} {roi}: {n} voxels")
    df = pd.DataFrame(rows)
    TSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(TSV, sep="\t", index=False)
    print(f"wrote {TSV}  ({len(df):,} rows)")
    return df


def load_or_extract(force=False, bids_folder="/data/ds-retsupp"):
    if TSV.exists() and not force:
        print(f"Loading cached {TSV}")
        return pd.read_csv(TSV, sep="\t")
    print(f"{TSV} missing — extracting (this is slow, ~10 min)")
    return extract_dog_params(bids_folder=bids_folder)


# ---------------------------------------------------------------------------
# Figure helpers
def _per_subject_median(df_roi, col, ecc_bins=None):
    """Per-subject median of `col` within each ROI."""
    return df_roi.groupby("subject")[col].median()


def _strip_panel(ax, df, col, title, ylabel, palette,
                 hline=None, hline_label=None,
                 ylim=None, log=False):
    np.random.seed(0)
    per_sub = (df.groupby(["subject", "roi"])[col].median()
               .unstack().reindex(columns=ROI_ORDER))
    for i, roi in enumerate(ROI_ORDER):
        vals = per_sub[roi].dropna().values
        if len(vals) == 0:
            continue
        jitter = np.random.uniform(-0.20, 0.20, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals,
                   color=ROI_COLORS[roi], alpha=0.75, s=55,
                   edgecolor="white", linewidth=0.7)
        ax.scatter([i], [np.median(vals)], marker="_",
                   color=palette["fg"], s=600, lw=3.0, zorder=10)
    if hline is not None:
        ax.axhline(hline, color=palette["muted"], lw=1.4,
                   ls=(0, (4, 3)), alpha=0.9)
        if hline_label:
            ax.text(len(ROI_ORDER) - 0.5, hline, "  " + hline_label,
                    color=palette["muted"], fontsize=15, va="center",
                    ha="left", style="italic")
    ax.set_xticks(range(len(ROI_ORDER)))
    ax.set_xticklabels(ROI_ORDER, rotation=35, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=10, loc="left", weight="bold")
    ax.grid(alpha=0.20, axis="y")
    if log:
        ax.set_yscale("log")
    if ylim is not None:
        ax.set_ylim(*ylim)


def _dog_profile(ax, df, palette):
    """1-D cross-section of the median DoG (center − surround_amp · surround)
    for a few representative ROIs.
    """
    r = np.linspace(0, 5, 240)
    for roi in PROFILE_ROIS:
        sub = df[df.roi == roi]
        if len(sub) == 0:
            continue
        sd_med = sub["sd"].median()
        srf_size = sub["srf_size"].median()
        srf_amp = sub["srf_amplitude"].median()
        # Normalize so center peak = 1.
        center = np.exp(-r ** 2 / (2 * sd_med ** 2))
        surround = np.exp(-r ** 2 / (2 * (sd_med * srf_size) ** 2))
        dog = center - srf_amp * surround
        ax.plot(r, dog, color=ROI_COLORS[roi], lw=3.0,
                label=(f"{roi}   σ={sd_med:.2f}°,  "
                       f"σ_s={sd_med * srf_size:.2f}°,  "
                       f"a_s={srf_amp:.2f}"))
    ax.axhline(0, color=palette["muted"], lw=0.8, alpha=0.7)
    ax.set_xlabel("Distance from PRF center (°)")
    ax.set_ylabel("DoG response  (center − a$_s$·surround)")
    ax.set_title("Canonical DoG profile per ROI",
                 pad=10, loc="left", weight="bold")
    ax.legend(frameon=False, fontsize=14, loc="upper right",
              handlelength=1.6)
    ax.set_xlim(0, 5); ax.set_ylim(-0.35, 1.05)
    ax.grid(alpha=0.20)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force-extract", action="store_true",
                    help="Re-extract from NIfTIs even if TSV exists")
    ap.add_argument("--bids-folder", default="/data/ds-retsupp")
    args = ap.parse_args()

    df = load_or_extract(force=args.force_extract,
                         bids_folder=args.bids_folder)

    # Baseline filter (same criteria as plot_prf_validity.py — well-fit
    # voxel with a sensibly placed Gaussian center). Adding the DoG
    # filters on top tells us, per ROI, what fraction of well-fit
    # voxels actually use a center-surround structure.
    base_good, fdr_thr = select_well_fit_voxels(
        df, n_params=N_PARAMS_M4, n_timepoints=N_TIMEPOINTS,
        fdr_alpha=FDR_ALPHA, mass_threshold=MASS_IN_APERTURE_THR,
        sigma_floor=SIGMA_FLOOR, sigma_ceil=SIGMA_CEIL,
        aperture_radius=APERTURE_RADIUS)
    print(f"  BH-FDR R² threshold: {fdr_thr:.4f}")
    df["_base"] = df.index.isin(base_good.index)
    df["_dog"] = (df["_base"]
                  & (df["srf_size"] >= SRF_SIZE_FLOOR)
                  & (df["srf_amplitude"] >= SRF_AMP_FLOOR)
                  & (df["srf_amplitude"] <= SRF_AMP_CEIL))
    good = df[df["_dog"]].copy()
    good["surround_sd"] = good["sd"] * good["srf_size"]
    n_pre = len(df); n_post = len(good)
    print(f"  selected {n_post:,} / {n_pre:,}  "
          f"({100 * n_post / n_pre:.1f}%) voxels")

    # Per-ROI per-subject 'fraction of well-fit voxels with a real
    # surround' (= fraction of baseline voxels that also pass the DoG
    # filters). Computed per subject so we can show across-subject
    # variability.
    frac_dog = (df.groupby(["subject", "roi"])
                  .apply(lambda d: d["_dog"].sum() / max(1, d["_base"].sum()))
                  .unstack().reindex(columns=ROI_ORDER))

    p = BUILDUP_PALETTE
    rc = base_rc(p)
    rc.update({
        "font.size": 20, "axes.titlesize": 24, "axes.labelsize": 22,
        "xtick.labelsize": 19, "ytick.labelsize": 19,
        "legend.fontsize": 16, "figure.titlesize": 28,
    })
    with plt.rc_context(rc):
        fig = plt.figure(figsize=(26.0, 9.0))
        gs = fig.add_gridspec(
            2, 4, height_ratios=[1.0, 1.0],
            hspace=0.55, wspace=0.32,
            left=0.04, right=0.99, top=0.83, bottom=0.10)

        # =================================================================
        # Panel 0 (top-left): per-ROI fraction of well-fit voxels that
        # actually use a real DoG surround.
        ax_0 = fig.add_subplot(gs[0, 0])
        xs = np.arange(len(ROI_ORDER))
        np.random.seed(0)
        for i, roi in enumerate(ROI_ORDER):
            vals = frac_dog[roi].dropna().values * 100.0
            if len(vals) == 0:
                continue
            jitter = np.random.uniform(-0.20, 0.20, size=len(vals))
            ax_0.scatter(np.full(len(vals), i) + jitter, vals,
                         color=ROI_COLORS[roi], alpha=0.75, s=55,
                         edgecolor="white", linewidth=0.7)
            ax_0.scatter([i], [np.median(vals)], marker="_",
                         color=p["fg"], s=600, lw=3.0, zorder=10)
        ax_0.set_xticks(xs); ax_0.set_xticklabels(ROI_ORDER, rotation=35,
                                                    ha="right")
        ax_0.set_ylabel("% voxels with real surround")
        ax_0.set_title("Fraction with DoG surround",
                       pad=10, loc="left", weight="bold")
        ax_0.set_ylim(0, 105)
        ax_0.grid(alpha=0.20, axis="y")

        # =================================================================
        # Panel A (top, col 1): center vs surround σ across ROI hierarchy
        ax_A = fig.add_subplot(gs[0, 1])
        per_sub_center = (good.groupby(["subject", "roi"])["sd"].median()
                          .unstack().reindex(columns=ROI_ORDER))
        per_sub_surr = (good.groupby(["subject", "roi"])["surround_sd"]
                        .median().unstack().reindex(columns=ROI_ORDER))
        xs = np.arange(len(ROI_ORDER))
        for cur, label, marker, alpha in [
            (per_sub_center, "Center σ", "o", 1.0),
            (per_sub_surr, "Surround σ  (= sd · srf_size)", "s", 0.8),
        ]:
            mean_c = cur.mean(axis=0).values
            sem_c = cur.sem(axis=0).values
            valid = ~np.isnan(mean_c)
            ax_A.fill_between(xs[valid], (mean_c - sem_c)[valid],
                              (mean_c + sem_c)[valid],
                              alpha=0.18, color=p["fg"], linewidth=0)
            ax_A.plot(xs[valid], mean_c[valid],
                      color=p["fg"], lw=2.8,
                      marker=marker, ms=10, alpha=alpha, label=label)
        ax_A.set_xticks(xs); ax_A.set_xticklabels(ROI_ORDER, rotation=35,
                                                    ha="right")
        ax_A.set_ylabel("σ (°)")
        ax_A.set_title("Center & surround σ per ROI",
                       pad=10, loc="left", weight="bold")
        ax_A.legend(frameon=False, loc="upper left",
                    handlelength=1.6)
        ax_A.grid(alpha=0.20, axis="y")
        ax_A.set_ylim(bottom=0)

        # =================================================================
        # Panel B (top, col 2): srf_size ratio per ROI
        ax_B = fig.add_subplot(gs[0, 2])
        _strip_panel(ax_B, good, "srf_size",
                     "Surround/center σ ratio  (srf_size)",
                     "srf_size  (σ_surround / σ_center)",
                     p, hline=1.0, hline_label="= same width")

        # =================================================================
        # Panel C (top, col 3): srf_amplitude per ROI
        ax_C = fig.add_subplot(gs[0, 3])
        _strip_panel(ax_C, good, "srf_amplitude",
                     "Surround amplitude  (srf_amplitude)",
                     "srf_amplitude  (fraction of center)",
                     p, ylim=(0, 1.05))

        # =================================================================
        # Panel D (bottom row, full width): 1D DoG profiles for example ROIs
        ax_D = fig.add_subplot(gs[1, :])
        _dog_profile(ax_D, good, p)

        fig.suptitle(
            "DoG (center − surround) parameters across the visual hierarchy",
            x=0.05, y=0.96, ha="left", fontsize=28,
            weight="bold", color=p["fg"])
        fig.text(0.05, 0.91,
                 f"Model 4 (DoG + flex HRF) · {good.subject.nunique()} "
                 f"subjects · {n_post:,} voxels (of {n_pre:,}) pass selection",
                 fontsize=17, color=p["muted"], style="italic")

        OUT.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUT, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
