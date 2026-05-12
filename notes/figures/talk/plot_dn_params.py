"""Sibling of `plot_dog_params.py` — show what the Divisive Normalization
(DN) model parameters look like across ROIs.

Model 6 (DivisiveNormalizationGaussianPRF2DWithHRF, flex HRF) has:
    R(x) = (rf_amplitude · G_c(x; sd)  +  neural_baseline)
           ─────────────────────────────────────────────────
           (srf_amplitude · G_s(x; sd · srf_size)  +  surround_baseline)
         + bold_baseline      (after HRF convolution)

So the DN-specific shape is set by:
  * center σ                   (`sd`)
  * surround σ                  (`sd · srf_size`)
  * surround/center amp ratio   (`srf_amplitude / rf_amplitude`)
  * neural baseline             (semi-saturation, lifts the numerator)
  * surround baseline           (lifts the denominator — limits suppression)

Panels:
  0. Fraction of voxels with 'real' DN structure (srf_amp > 0 AND
     surround actually contributes) per ROI.
  A. Center σ and surround σ per ROI (mean ± SEM).
  B. Surround / center amplitude ratio (srf_amplitude / rf_amplitude).
  C. neural_baseline per ROI (strip).
  D. 1-D DN response profile using median params per example ROI.

Output:
  notes/data/prf_dn_params_summary.tsv  (cached per-voxel TSV)
  notes/figures/talk/talk_prf_dn_params.pdf
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS))
from make_figures import APERTURE_RADIUS, BUILDUP_PALETTE, base_rc  # noqa
from plot_prf_validity import (  # noqa: E402
    FDR_ALPHA, MASS_IN_APERTURE_THR, SIGMA_FLOOR, SIGMA_CEIL,
    N_TIMEPOINTS,
    ROI_ORDER, ROI_COLORS,
)
from plot_dog_params import _strip_panel  # noqa: E402
from retsupp.utils.data import select_well_fit_voxels  # noqa: E402

REPO = THIS.parents[2]
TSV = REPO / "notes" / "data" / "prf_dn_params_summary.tsv"
OUT = THIS / "talk_prf_dn_params.pdf"

# Model 6 free parameters: x, y, sd, rf_amplitude, srf_amplitude,
# srf_size, neural_baseline, surround_baseline, bold_baseline,
# hrf_delay, hrf_dispersion → 11.
N_PARAMS_M6 = 11

# DN-specific filters.
SRF_AMP_FLOOR = 1e-3        # exclude voxels where surround amplitude is 0
RF_AMP_FLOOR = 1e-3         # exclude voxels where center amplitude is 0/neg
SRF_SIZE_FLOOR = 1.0
NEURAL_BASELINE_FLOOR = 1e-3

PROFILE_ROIS = ["V1", "V3", "LO", "IPS"]


# ---------------------------------------------------------------------------
def extract_dn_params(bids_folder="/data/ds-retsupp", model=6):
    from retsupp.utils.data import Subject
    from nilearn import image

    rows = []
    for s in range(1, 31):
        try:
            sub = Subject(s, bids_folder=bids_folder)
            prf = sub.get_prf_parameters_volume(model=model,
                                                return_images=True)
        except Exception as e:
            print(f"  sub-{s:02d}: PRF unavailable ({e})")
            continue
        for roi in ROI_ORDER:
            try:
                mask_img = sub.get_retinotopic_roi(roi=roi, bold_space=True)
            except Exception:
                continue
            m = mask_img.get_fdata().astype(bool)
            cols = {}
            for par in ["x", "y", "sd",
                        "rf_amplitude", "srf_amplitude", "srf_size",
                        "neural_baseline", "surround_baseline",
                        "bold_baseline", "r2"]:
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
                    rf_amplitude=float(cols["rf_amplitude"][i]),
                    srf_amplitude=float(cols["srf_amplitude"][i]),
                    srf_size=float(cols["srf_size"][i]),
                    neural_baseline=float(cols["neural_baseline"][i]),
                    surround_baseline=float(cols["surround_baseline"][i]),
                    bold_baseline=float(cols["bold_baseline"][i]),
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
    return extract_dn_params(bids_folder=bids_folder)


# ---------------------------------------------------------------------------
def _dn_profile(ax, df, palette):
    """1-D DN response cross-section using median params per example ROI.
    Normalised so the center peak = 1 for visual comparison.
    """
    r = np.linspace(0, 5, 300)
    for roi in PROFILE_ROIS:
        sub = df[df.roi == roi]
        if len(sub) == 0:
            continue
        sd = sub["sd"].median()
        rf_amp = sub["rf_amplitude"].median()
        srf_amp = sub["srf_amplitude"].median()
        srf_size = sub["srf_size"].median()
        nb = sub["neural_baseline"].median()
        sb = sub["surround_baseline"].median()
        Gc = np.exp(-r ** 2 / (2 * sd ** 2))
        Gs = np.exp(-r ** 2 / (2 * (sd * srf_size) ** 2))
        num = rf_amp * Gc + nb
        den = srf_amp * Gs + sb
        R = num / np.maximum(den, 1e-9)
        # Normalise so peak = 1 for visual comparison across ROIs.
        if R.max() != R.min():
            R = (R - R[-1]) / (R[0] - R[-1])
        ax.plot(r, R, color=ROI_COLORS[roi], lw=3.0,
                label=(f"{roi}   σ={sd:.2f}°  σ_s={sd*srf_size:.2f}°  "
                       f"a_s/a_c={srf_amp/max(rf_amp,1e-9):.2f}  "
                       f"nb={nb:.2g}"))
    ax.axhline(0, color=palette["muted"], lw=0.8, alpha=0.7)
    ax.set_xlabel("Distance from PRF center (°)")
    ax.set_ylabel("DN response  (peak-normalised)")
    ax.set_title("Canonical DN profile per ROI",
                 pad=10, loc="left", weight="bold")
    ax.legend(frameon=False, fontsize=14, loc="upper right",
              handlelength=1.6)
    ax.set_xlim(0, 5); ax.set_ylim(-0.1, 1.1)
    ax.grid(alpha=0.20)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force-extract", action="store_true")
    ap.add_argument("--bids-folder", default="/data/ds-retsupp")
    args = ap.parse_args()

    df = load_or_extract(force=args.force_extract,
                         bids_folder=args.bids_folder)

    # Baseline filter = same well-fit voxel criteria as the validity figure.
    base_good, fdr_thr = select_well_fit_voxels(
        df, n_params=N_PARAMS_M6, n_timepoints=N_TIMEPOINTS,
        fdr_alpha=FDR_ALPHA, mass_threshold=MASS_IN_APERTURE_THR,
        sigma_floor=SIGMA_FLOOR, sigma_ceil=SIGMA_CEIL,
        aperture_radius=APERTURE_RADIUS)
    print(f"  BH-FDR R² threshold (DN, k={N_PARAMS_M6}): {fdr_thr:.4f}")
    df["_base"] = df.index.isin(base_good.index)
    df["_dn"] = (df["_base"]
                 & (df["rf_amplitude"] >= RF_AMP_FLOOR)
                 & (df["srf_amplitude"] >= SRF_AMP_FLOOR)
                 & (df["srf_size"] >= SRF_SIZE_FLOOR)
                 & (df["neural_baseline"] >= NEURAL_BASELINE_FLOOR))
    good = df[df["_dn"]].copy()
    good["surround_sd"] = good["sd"] * good["srf_size"]
    good["amp_ratio"] = good["srf_amplitude"] / good["rf_amplitude"]
    n_pre = len(df); n_post = len(good)
    print(f"  selected {n_post:,} / {n_pre:,}  "
          f"({100 * n_post / n_pre:.1f}%) voxels")

    # Per-subject per-ROI 'fraction of well-fit voxels with real DN'.
    frac_dn = (df.groupby(["subject", "roi"])
                 .apply(lambda d: d["_dn"].sum() / max(1, d["_base"].sum()))
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

        # Panel 0 — fraction with real DN surround per ROI
        ax_0 = fig.add_subplot(gs[0, 0])
        xs = np.arange(len(ROI_ORDER))
        np.random.seed(0)
        for i, roi in enumerate(ROI_ORDER):
            vals = frac_dn[roi].dropna().values * 100.0
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
        ax_0.set_title("Fraction with DN surround",
                       pad=10, loc="left", weight="bold")
        ax_0.set_ylim(0, 105)
        ax_0.grid(alpha=0.20, axis="y")

        # Panel A — center vs surround σ per ROI
        ax_A = fig.add_subplot(gs[0, 1])
        per_sub_center = (good.groupby(["subject", "roi"])["sd"].median()
                          .unstack().reindex(columns=ROI_ORDER))
        per_sub_surr = (good.groupby(["subject", "roi"])["surround_sd"]
                        .median().unstack().reindex(columns=ROI_ORDER))
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
                      color=p["fg"], lw=2.8, marker=marker, ms=10,
                      alpha=alpha, label=label)
        ax_A.set_xticks(xs); ax_A.set_xticklabels(ROI_ORDER, rotation=35,
                                                    ha="right")
        ax_A.set_ylabel("σ (°)")
        ax_A.set_title("Center & surround σ per ROI",
                       pad=10, loc="left", weight="bold")
        ax_A.legend(frameon=False, loc="upper left", handlelength=1.6)
        ax_A.grid(alpha=0.20, axis="y")
        ax_A.set_ylim(bottom=0)

        # Panel B — surround / center amp ratio per ROI (log-y).
        ax_B = fig.add_subplot(gs[0, 2])
        _strip_panel(ax_B, good, "amp_ratio",
                     "Surround / center amp  (a$_s$ / a$_c$)",
                     "srf_amplitude / rf_amplitude",
                     p, log=True)

        # Panel C — neural_baseline per ROI (log-y; spans many orders).
        ax_C = fig.add_subplot(gs[0, 3])
        _strip_panel(ax_C, good, "neural_baseline",
                     "Neural baseline  (semi-saturation)",
                     "neural_baseline",
                     p, log=True)

        # Panel D — 1-D DN response profile per example ROI
        ax_D = fig.add_subplot(gs[1, :])
        _dn_profile(ax_D, good, p)

        fig.suptitle(
            "Divisive Normalization (DN) parameters across the visual hierarchy",
            x=0.04, y=0.96, ha="left", fontsize=28,
            weight="bold", color=p["fg"])
        fig.text(0.04, 0.91,
                 f"Model 6 (DN + flex HRF) · {good.subject.nunique()} "
                 f"subjects · {n_post:,} voxels (of {n_pre:,}) pass selection",
                 fontsize=17, color=p["muted"], style="italic")

        OUT.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUT, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
