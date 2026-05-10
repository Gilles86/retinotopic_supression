"""Per-subject projection-vs-distance curves, one panel per ROI.

Reuses the data pipeline from ``plot_gam_clusters`` (load conditionwise
PRF parameters, aperture filter, per-(subject, distance-bin) aggregation),
then plots one line per subject per ROI plus the across-subject mean ±
SEM. Useful for sanity-checking whether the population GAM result is
driven by a few subjects or is truly common across the sample.

Outputs a multi-page PDF: one page per ROI showing all subjects'
per-bin means.

Usage
-----
    python -m retsupp.visualize.plot_per_subject_curves \\
        --out notes/per_subject_curves.pdf
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

from retsupp.visualize.plot_gam_clusters import (
    DEFAULT_ROIS,
    build_subject_bin_table,
    filter_prf_inside_aperture,
    load_all_conditionwise,
)


def plot_per_subject_panel(df_roi: pd.DataFrame, roi: str, ax,
                            ylim: tuple[float, float] | None = None):
    """Plot per-subject bin-mean lines + group mean ± SEM on `ax`."""
    subs = sorted(df_roi["subject"].unique())
    cmap = plt.get_cmap("tab20", max(len(subs), 1))

    # Line per subject — connect bin means with no smoothing.
    for i, s in enumerate(subs):
        d = df_roi[df_roi["subject"] == s].sort_values("distance")
        ax.plot(d["distance"], d["proj"],
                color=cmap(i), alpha=0.55, lw=0.9,
                marker="o", markersize=2.5,
                label=f"sub-{int(s):02d}")
    # Group mean ± SEM across subjects per bin.
    grp = (df_roi.groupby("distance")
                .agg(mean=("proj", "mean"),
                     sem=("proj",
                          lambda v: v.std(ddof=1) / np.sqrt(max(len(v), 1))),
                     n=("proj", "size"))
                .reset_index())
    ax.errorbar(grp["distance"], grp["mean"], yerr=grp["sem"],
                color="k", lw=2.0, marker="s", markersize=4,
                capsize=3, zorder=10, label="group mean ± SEM")
    ax.axhline(0, color="0.5", lw=0.5, ls="--")
    ax.set_xlabel("distance of mean PRF from HP (°)")
    ax.set_ylabel("proj. away from HP (°)")
    ax.set_title(f"{roi} — n_subj = {len(subs)}")
    if ylim is not None:
        ax.set_ylim(*ylim)


def cover_page(pdf, args):
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")
    ax.text(0.5, 0.92, "Per-subject projection-vs-distance curves",
            ha="center", va="top", fontsize=18, weight="bold",
            transform=ax.transAxes)
    body = (
        "  • One line per subject: per-(distance-bin) mean projection along\n"
        "    the away-from-HP axis (after rotating each condition so HP is\n"
        "    at the canonical (0, +4°) location).\n"
        "  • Black thick line: across-subject mean ± SEM per bin.\n"
        "  • No smoothing applied here — these are raw bin means, in\n"
        "    contrast to the bambi/HSGP smooth in plot_gam_clusters.\n\n"
        f"  bin_step           = {args.bin_step:.2f}°\n"
        f"  clip |proj|        = {args.clip_projection:.2f}°\n"
        f"  r²-threshold       = {args.r2_thr:.2f}\n"
        f"  aperture filter    = ≤{args.aperture_fraction_outside*100:.0f}% PRF outside aperture\n"
        f"  shared y-lim       = {args.shared_ylim if args.shared_ylim else 'per-panel auto'}\n"
    )
    ax.text(0.04, 0.83, body, ha="left", va="top",
            family="monospace", fontsize=10, transform=ax.transAxes)
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bids-folder", default="/data/ds-retsupp")
    parser.add_argument("--out", type=Path,
                        default=Path("notes/per_subject_curves.pdf"))
    parser.add_argument("--rois", nargs="+", default=DEFAULT_ROIS)
    parser.add_argument("--r2-thr", type=float, default=0.20)
    parser.add_argument("--aperture-fraction-outside", type=float, default=0.5)
    parser.add_argument("--bin-step", type=float, default=0.5)
    parser.add_argument("--bin-min", type=float, default=0.0)
    parser.add_argument("--bin-max", type=float, default=7.0)
    parser.add_argument("--clip-projection", type=float, default=1.5)
    parser.add_argument("--shared-ylim", type=float, default=0.4,
                        help="Hard y-limits ±this for all ROI panels "
                             "(for cross-ROI comparison). 0 → per-panel auto.")
    args = parser.parse_args()

    bin_edges = np.arange(args.bin_min, args.bin_max + args.bin_step / 2,
                          args.bin_step)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    print("Loading conditionwise data ...")
    pars = load_all_conditionwise(
        bids_folder=args.bids_folder, r2_thr=args.r2_thr,
    )
    print(f"  {len(pars):,} rows before aperture filter.")
    pars = filter_prf_inside_aperture(
        pars, aperture_radius=3.17,
        max_fraction_outside=args.aperture_fraction_outside,
        use_mean_only=False,
    )
    print(f"  {len(pars):,} rows after aperture filter.")

    agg = build_subject_bin_table(
        pars, args.rois, bin_edges, args.clip_projection,
    )
    print(f"  Total aggregated rows: {len(agg):,}")

    ylim = ((-args.shared_ylim, +args.shared_ylim)
             if args.shared_ylim and args.shared_ylim > 0 else None)

    with PdfPages(args.out) as pdf:
        cover_page(pdf, args)

        # ---- Per-ROI overlay (1 page per ROI).
        for roi in args.rois:
            df_roi = (agg[agg["roi_base"] == roi]
                       [["subject", "distance", "proj"]]
                       .dropna(subset=["proj", "distance"]))
            if len(df_roi) == 0:
                print(f"  {roi}: no data — skipping")
                continue
            fig, ax = plt.subplots(figsize=(10.5, 6.5))
            plot_per_subject_panel(df_roi, roi, ax, ylim=ylim)
            ax.legend(loc="upper right", fontsize=6, ncol=2,
                      frameon=False, bbox_to_anchor=(1.18, 1.0))
            fig.tight_layout()
            pdf.savefig(fig); plt.close(fig)

        # ---- Faceted small-multiples: per-subject panels per ROI (sub_grid).
        for roi in args.rois:
            df_roi = (agg[agg["roi_base"] == roi]
                       [["subject", "distance", "proj"]]
                       .dropna(subset=["proj", "distance"]))
            if len(df_roi) == 0:
                continue
            subs = sorted(df_roi["subject"].unique())
            ncol = 6
            nrow = int(np.ceil(len(subs) / ncol))
            fig, axes = plt.subplots(
                nrow, ncol, figsize=(2.0 * ncol, 1.6 * nrow),
                sharex=True, sharey=True,
            )
            axes = np.atleast_2d(axes)
            grp = (df_roi.groupby("distance")["proj"].mean().reset_index())
            for i, s in enumerate(subs):
                ax = axes[i // ncol, i % ncol]
                d = df_roi[df_roi["subject"] == s].sort_values("distance")
                ax.plot(d["distance"], d["proj"], "-o", markersize=2.5,
                        color="C0", lw=1)
                ax.plot(grp["distance"], grp["mean"], color="k",
                        lw=1.0, ls="--", alpha=0.6)
                ax.axhline(0, color="0.6", lw=0.4, ls=":")
                ax.set_title(f"sub-{int(s):02d}", fontsize=8)
                if ylim is not None:
                    ax.set_ylim(*ylim)
            for j in range(len(subs), nrow * ncol):
                axes[j // ncol, j % ncol].axis("off")
            fig.suptitle(f"{roi} — per-subject (group mean dashed)",
                         fontsize=11)
            fig.text(0.5, 0.005, "distance of mean PRF from HP (°)",
                     ha="center", fontsize=9)
            fig.text(0.005, 0.5, "proj. away from HP (°)",
                     va="center", rotation="vertical", fontsize=9)
            fig.tight_layout(rect=[0.02, 0.03, 1, 0.97])
            pdf.savefig(fig); plt.close(fig)

    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
