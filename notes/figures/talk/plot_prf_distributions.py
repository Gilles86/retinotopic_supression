"""Per-subject PRF parameter + spatial-coverage distributions.

One big multi-page PDF with:
  - page 1: spatial coverage — 2D KDE of (x, y) PRF centers within
    the bar aperture, FacetGrid rows = subjects × cols = ROIs.
  - one page per scalar parameter (sd, amplitude, baseline, r2, plus
    eccentricity and polar angle derived from x, y) — histogram
    FacetGrid rows = subjects × cols = ROIs, shared x-axis per
    parameter across subjects so distributions are directly
    comparable.

Reads the warm-start TSVs at ``notes/data/prf_warmstart_m{M}_V1_sub-*.tsv``
for the requested model. Today V1 is the only ROI in the warm-start
output; when whole-brain fits land we can extend by loading per-ROI
slices from the canonical NIfTIs.

Output: ``notes/figures/talk/prf_distributions_m{M}.pdf``.

CLI::

    python plot_prf_distributions.py --model 1 [--data-dir DIR]
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import PowerNorm
from matplotlib.patches import Circle

THIS = Path(__file__).resolve().parent
REPO = THIS.parents[2]
DATA_DIR = REPO / "notes" / "data"

# Bar aperture radius (degrees) — geometry constant from retsupp
# `expsettings.yml`: aperture = ecc - size_stimuli / 1.8 = 4 - 1.5/1.8.
APERTURE_RADIUS = 3.17

# Per-parameter axis ranges (None = auto from data quantile-trimming).
# Tuned to expose the bulk of the distribution rather than outliers.
PARAM_RANGES = {
    "x": (-APERTURE_RADIUS - 0.5, APERTURE_RADIUS + 0.5),
    "y": (-APERTURE_RADIUS - 0.5, APERTURE_RADIUS + 0.5),
    "sd": (0.0, 5.0),
    "eccen": (0.0, APERTURE_RADIUS + 1.0),
    "theta": (-np.pi, np.pi),
    "amplitude": None,
    "baseline": None,
    # R² mass is heavily concentrated near 0; cap at 0.6 so the body
    # of the distribution is visible. Long right tail picked up by
    # log-y scaling (see PARAM_LOG_Y).
    "r2": (0.0, 0.6),
    # Surround/DN params (only present for m2, m4-6)
    "srf_size": (0.0, 8.0),
    "srf_amplitude": None,
    "hrf_delay": (0.0, 10.0),
    "hrf_dispersion": (0.0, 3.0),
    "rf_amplitude": None,
    "neural_baseline": None,
    "surround_baseline": None,
    "bold_baseline": None,
}

# Parameters whose histogram should use a log y-axis (long-tailed
# distributions where the small-bin counts matter for QC).
PARAM_LOG_Y = {"r2", "sd", "srf_size", "amplitude",
                "rf_amplitude", "srf_amplitude"}


def load_warmstart(model: int, data_dir: Path) -> pd.DataFrame:
    """Concat per-subject warm-start TSVs for one model into one frame
    tagged with a ``roi`` column. Today every voxel is V1 (sandbox
    only fits V1); the column is here so callers can extend with
    additional ROIs later without changing the plot code.
    """
    files = sorted(data_dir.glob(f"prf_warmstart_m{model}_V1_sub-*.tsv"))
    if not files:
        raise FileNotFoundError(
            f"No warmstart TSVs found at {data_dir}/prf_warmstart_m{model}_V1_sub-*.tsv")
    frames = []
    for f in files:
        d = pd.read_csv(f, sep="\t")
        d["roi"] = "V1"
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    # Derived geometric params.
    df["eccen"] = np.sqrt(df["x"] ** 2 + df["y"] ** 2)
    df["theta"] = np.arctan2(df["y"], df["x"])
    return df


def plot_spatial_coverage_page(pdf: PdfPages, df: pd.DataFrame,
                                 model: int) -> None:
    """One page: 2D PRF-center density per (subject, ROI). Real voxels
    only (r² > 0). KDE inside the bar aperture; subjects on rows."""
    real = df[df.r2 > 0].copy()
    subjects = sorted(real.subject.unique())
    rois = sorted(real.roi.unique())

    n_rows = len(subjects)
    n_cols = max(len(rois), 1)
    # Floor the width so the long suptitle has room when there are few
    # ROI columns. Leave a 0.3" left gutter for the row labels.
    fig_w = max(3.4 * n_cols + 1.3, 8.0)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(fig_w, 3.4 * n_rows),
        squeeze=False)
    fig.suptitle(f"Model {model} — PRF center density per subject × ROI",
                 weight="bold", fontsize=16, y=0.995)

    grid_n = 120
    g = np.linspace(-APERTURE_RADIUS - 0.5,
                     APERTURE_RADIUS + 0.5, grid_n)
    GX, GY = np.meshgrid(g, g)
    outside = (GX ** 2 + GY ** 2) > APERTURE_RADIUS ** 2

    for i, sub in enumerate(subjects):
        for j, roi in enumerate(rois):
            ax = axes[i, j]
            sel = real[(real.subject == sub) & (real.roi == roi)]
            if len(sel) < 5:
                ax.text(0.5, 0.5, "n<5", transform=ax.transAxes,
                        ha="center", va="center", color="grey")
            else:
                # Subsample for speed.
                if len(sel) > 8000:
                    sel = sel.sample(8000, random_state=0)
                from scipy.stats import gaussian_kde
                xy = np.vstack([sel.x.to_numpy(), sel.y.to_numpy()])
                try:
                    kde = gaussian_kde(xy, bw_method=0.20)
                    Z = kde(np.vstack([GX.ravel(), GY.ravel()])
                            ).reshape(grid_n, grid_n)
                    Z = np.where(outside, np.nan, Z)
                    ax.imshow(
                        Z, extent=(g[0], g[-1], g[0], g[-1]),
                        origin="lower", cmap="viridis",
                        norm=PowerNorm(gamma=0.45),
                        interpolation="bilinear")
                except Exception as e:
                    ax.text(0.5, 0.5, f"KDE fail\n{type(e).__name__}",
                             transform=ax.transAxes, ha="center",
                             va="center", color="red", fontsize=9)

            # Aperture + eccentricity rings.
            for r_iso in (1.0, 2.0, 3.0):
                ax.add_patch(Circle((0, 0), r_iso, facecolor="none",
                                     edgecolor="white", lw=0.8,
                                     ls=(0, (2, 4)), alpha=0.6))
            ax.add_patch(Circle((0, 0), APERTURE_RADIUS,
                                 facecolor="none", edgecolor="white",
                                 lw=1.8, ls=(0, (5, 3))))
            ax.plot(0, 0, "+", color="white", ms=10, mew=1.6)
            ax.set_xlim(g[0], g[-1])
            ax.set_ylim(g[0], g[-1])
            ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)

            if i == 0:
                ax.set_title(roi, weight="bold", fontsize=14)
            if j == 0:
                ax.set_ylabel(f"sub-{sub:02d}\nn={len(sel)}",
                              fontsize=11, rotation=0, ha="right",
                              va="center", labelpad=18)

    # Leave headroom for the suptitle.
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def plot_parameter_page(pdf: PdfPages, df: pd.DataFrame, param: str,
                          model: int) -> None:
    """One page per scalar parameter: histogram FacetGrid
    rows = subjects, cols = ROIs."""
    if param not in df.columns:
        return
    real = df[df.r2 > 0].copy()
    if not len(real):
        return
    subjects = sorted(real.subject.unique())
    rois = sorted(real.roi.unique())

    # Quantile-clip extreme outliers so the bulk is visible.
    rng = PARAM_RANGES.get(param)
    if rng is None:
        lo = float(real[param].quantile(0.001))
        hi = float(real[param].quantile(0.999))
        rng = (lo, hi)
    real = real[(real[param] >= rng[0]) & (real[param] <= rng[1])]

    # Wider aspect so single-column FacetGrid still leaves room for
    # the suptitle on the page.
    n_cols = max(len(rois), 1)
    aspect = 2.6 if n_cols == 1 else 1.6
    g = sns.FacetGrid(
        real, row="subject", col="roi",
        row_order=subjects, col_order=rois,
        height=1.7, aspect=aspect, sharex=True, sharey=False,
        margin_titles=True, despine=True)
    g.map_dataframe(
        sns.histplot, x=param, stat="density",
        bins=40, color="#1B4965", alpha=0.85,
        edgecolor="white", linewidth=0.3)
    g.set_titles(row_template="sub-{row_name:02d}", col_template="{col_name}")
    g.set_axis_labels(param, "density")
    for ax in g.axes.flat:
        ax.set_xlim(rng)
        if param in PARAM_LOG_Y:
            ax.set_yscale("log")
        ax.grid(alpha=0.18, axis="y")
        ax.tick_params(labelsize=9)
    # Add headroom above the FacetGrid for the suptitle.
    g.fig.subplots_adjust(top=0.92)
    g.fig.suptitle(f"Model {model} — distribution of {param}",
                    weight="bold", fontsize=15)
    pdf.savefig(g.fig, bbox_inches="tight")
    plt.close(g.fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=int, default=1)
    ap.add_argument("--data-dir", type=Path, default=None,
                     help="Where to find prf_warmstart_m{M}_V1_sub-*.tsv "
                          "(default: notes/data/, falls back to /tmp/v1_results/)")
    ap.add_argument("--out", type=Path, default=None,
                     help="Output PDF path (default: notes/figures/talk/"
                          "prf_distributions_m{M}.pdf)")
    args = ap.parse_args()

    # Resolve data dir — prefer repo notes/data; fall back to /tmp.
    if args.data_dir is None:
        for cand in (DATA_DIR, Path("/tmp/v1_results")):
            if any(cand.glob(f"prf_warmstart_m{args.model}_V1_sub-*.tsv")):
                args.data_dir = cand
                break
        if args.data_dir is None:
            args.data_dir = DATA_DIR

    out_path = args.out or (THIS / f"prf_distributions_m{args.model}.pdf")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = load_warmstart(args.model, args.data_dir)
    print(f"Loaded {len(df):,} voxel-rows from {args.data_dir} "
          f"({df.subject.nunique()} subjects)")

    sns.set_theme(context="talk", style="ticks", font_scale=0.85)

    # Parameters to plot — only those actually present.
    present = [p for p in ("x", "y", "eccen", "theta", "sd",
                            "amplitude", "baseline", "r2",
                            "srf_size", "srf_amplitude",
                            "hrf_delay", "hrf_dispersion",
                            "rf_amplitude", "neural_baseline",
                            "surround_baseline", "bold_baseline")
               if p in df.columns]

    with PdfPages(out_path) as pdf:
        plot_spatial_coverage_page(pdf, df, args.model)
        for param in present:
            plot_parameter_page(pdf, df, param, args.model)

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
