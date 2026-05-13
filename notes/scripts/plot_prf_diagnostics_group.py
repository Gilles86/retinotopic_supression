"""Group-level PRF diagnostics: ROI × model, pooled across subjects.

Same 4 pages as ``plot_prf_diagnostics.py`` but each cell pools
well-fit voxels (FDR α=0.05, per-subject) across all subjects with
data for that (ROI, model) combination. Layout is transposed:
rows = ROIs, cols = models. One PDF total.

Output: ``notes/figures/prf_diagnostics/_group.pdf``.

Usage:
    python notes/scripts/plot_prf_diagnostics_group.py
"""
from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Circle
from scipy.stats import pearsonr

# Reuse loaders + constants from per-subject diagnostics.
import sys
sys.path.insert(0, str(Path(__file__).parent))
from plot_prf_diagnostics import (
    ROIS, MODELS, MODEL_LABELS, APERTURE_R, R2_PAIRS,
    load_roi_voxels,
)
from retsupp.utils.data import Subject

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "figure.titlesize": 14,
})


def load_group(subjects, roi, model, bids_folder):
    """Concatenate (all_df, well_df) across subjects for (ROI, model)."""
    all_chunks, well_chunks = [], []
    for sid in subjects:
        sub = Subject(sid, bids_folder=bids_folder)
        all_df, well_df = load_roi_voxels(sub, roi, model)
        if all_df is None or len(all_df) == 0:
            continue
        all_chunks.append(all_df.assign(subject=sid))
        well_chunks.append(well_df.assign(subject=sid))
    if not all_chunks:
        return None, None
    return (pd.concat(all_chunks, ignore_index=True),
            pd.concat(well_chunks, ignore_index=True))


def cell_coverage(ax, dfs):
    if dfs is None or dfs[1] is None or len(dfs[1]) == 0:
        ax.set_axis_off(); return
    well = dfs[1]
    if len(well) >= 20:
        ax.hexbin(well["x"], well["y"], gridsize=30, cmap="viridis",
                  bins="log", extent=(-5, 5, -5, 5), mincnt=1)
    else:
        ax.scatter(well["x"], well["y"], s=3, alpha=0.7, color="#1B4965")
    ax.add_patch(Circle((0, 0), APERTURE_R, facecolor="none",
                         edgecolor="white", lw=1.0, ls="--", alpha=0.8))
    ax.set_xlim(-5, 5); ax.set_ylim(-5, 5); ax.set_aspect("equal")
    ax.set_xticks([-3, 0, 3]); ax.set_yticks([-3, 0, 3])
    n_subs = well["subject"].nunique()
    ax.text(0.02, 0.98, f"n={len(well)}\n{n_subs} subs",
            transform=ax.transAxes, color="white", fontsize=7, va="top")


def cell_r2_hist(ax, dfs):
    if dfs is None or dfs[0] is None or len(dfs[0]) == 0:
        ax.set_axis_off(); return
    df, well = dfs
    bins = np.linspace(0, 0.5, 50)
    ax.hist(df["r2"], bins=bins, color="#E76F51", alpha=0.85, edgecolor="none")
    ax.set_yscale("log")
    ax.axvline(np.median(df["r2"]), color="black", lw=0.8)
    n_subs = df["subject"].nunique()
    ax.text(0.98, 0.95,
            f"med={np.median(df['r2']):.3f}\nFDR n={len(well)}\n{n_subs} subs",
            transform=ax.transAxes, ha="right", va="top", fontsize=7)
    ax.set_xlim(0, 0.5); ax.set_xticks([0, 0.25, 0.5])


def cell_sd_vs_ecc(ax, dfs):
    if dfs is None or dfs[1] is None: ax.set_axis_off(); return
    well = dfs[1]
    if len(well) < 5:
        ax.text(0.5, 0.5, f"n={len(well)}", ha="center", va="center",
                transform=ax.transAxes); return
    sample = well.sample(n=4000, random_state=0) if len(well) > 4000 else well
    ax.scatter(sample["eccen"], sample["sd"], s=2, alpha=0.2,
                color="#1B4965", edgecolor="none", rasterized=True)
    try:
        r, _ = pearsonr(well["eccen"], well["sd"])
        slope, intercept = np.polyfit(well["eccen"], well["sd"], 1)
        xs = np.array([0, 5])
        ax.plot(xs, slope * xs + intercept, color="#E76F51", lw=1.5)
        n_subs = well["subject"].nunique()
        ax.text(0.02, 0.95,
                f"r={r:.2f}\nslope={slope:.2f}\n{n_subs} subs",
                transform=ax.transAxes, va="top", fontsize=7)
    except (ValueError, np.linalg.LinAlgError):
        pass
    ax.axvline(APERTURE_R, color="black", lw=0.6, ls="--", alpha=0.5)
    ax.set_xlim(0, 5); ax.set_ylim(0, 3)
    ax.set_xticks([0, 2, 4]); ax.set_yticks([0, 1, 2, 3])


def cell_r2_pair(ax, dfs_x, dfs_y):
    if dfs_x is None or dfs_y is None or dfs_x[0] is None or dfs_y[0] is None:
        ax.set_axis_off(); return
    df_x, df_y = dfs_x[0], dfs_y[0]
    # Join on (subject, voxel index within subject) — both dfs share the
    # same row order per subject because load_roi_voxels reads the same
    # ROI mask in the same order. We can pair by index within each
    # subject's slice.
    merged = pd.merge(
        df_x[["subject", "r2"]].assign(vi=df_x.groupby("subject").cumcount()),
        df_y[["subject", "r2"]].assign(vi=df_y.groupby("subject").cumcount()),
        on=["subject", "vi"], suffixes=("_x", "_y"))
    # Restrict to voxels FDR-surviving in either model.
    wxs = set(zip(dfs_x[1]["subject"],
                  dfs_x[1].groupby("subject").cumcount()))
    wys = set(zip(dfs_y[1]["subject"],
                  dfs_y[1].groupby("subject").cumcount()))
    keep_keys = wxs | wys
    keys = list(zip(merged["subject"], merged["vi"]))
    keep = np.array([k in keep_keys for k in keys])
    if keep.sum() < 5:
        ax.set_axis_off(); return
    rx = merged.loc[keep, "r2_x"].values
    ry = merged.loc[keep, "r2_y"].values
    sample = np.random.default_rng(0).choice(
        len(rx), min(5000, len(rx)), replace=False)
    ax.scatter(rx[sample], ry[sample], s=2, alpha=0.2,
                color="#1B4965", edgecolor="none", rasterized=True)
    lim = max(rx.max(), ry.max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=0.6, alpha=0.6)
    above = (ry > rx).mean()
    delta_med = float(np.median(ry - rx))
    n_subs = merged.loc[keep, "subject"].nunique()
    ax.text(0.02, 0.98,
             f"n={len(rx)}\n{above:.0%} > diag\nΔmed={delta_med:+.3f}\n"
             f"{n_subs} subs",
             transform=ax.transAxes, va="top", fontsize=7)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_aspect("equal")


PER_MODEL_PAGES = [
    ("PRF coverage (FDR α=0.05, pooled across subjects)", cell_coverage),
    ("R² distribution (log-y; pooled all voxels)", cell_r2_hist),
    ("σ vs eccentricity (FDR-surviving, pooled)", cell_sd_vs_ecc),
]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bids-folder", default="/data/ds-retsupp")
    p.add_argument("--subjects", nargs="+", type=int,
                    default=list(range(1, 31)))
    p.add_argument("--rois", nargs="+", default=ROIS)
    p.add_argument("--out",
                    default="/Users/gdehol/git/retsupp/notes/figures/prf_diagnostics/_group.pdf")
    args = p.parse_args()

    print(f"Loading group data ({len(args.subjects)} subjects × "
          f"{len(args.rois)} ROIs × {len(MODELS)} models)...")
    cache = {}  # (roi, model) -> (all_df, well_df)
    for roi in args.rois:
        for m in MODELS:
            cache[(roi, m)] = load_group(args.subjects, roi, m,
                                          args.bids_folder)
        n_filled = sum(1 for m in MODELS if cache[(roi, m)][0] is not None)
        print(f"  {roi}: {n_filled} / {len(MODELS)} models have data")

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    n_rows = len(args.rois); n_cols = len(MODELS)
    with PdfPages(out) as pdf:
        for page_title, cell_fn in PER_MODEL_PAGES:
            fig, axes = plt.subplots(n_rows, n_cols,
                                      figsize=(1.6 * n_cols, 1.6 * n_rows),
                                      squeeze=False)
            for i, roi in enumerate(args.rois):
                for j, m in enumerate(MODELS):
                    ax = axes[i, j]
                    cell_fn(ax, cache.get((roi, m)))
                    if i == 0:
                        ax.set_title(MODEL_LABELS[m], fontsize=9)
                    if j == 0:
                        ax.set_ylabel(roi, fontsize=11, weight="bold",
                                       rotation=0, ha="right", va="center")
            fig.suptitle(f"GROUP — {page_title}", weight="bold")
            fig.tight_layout(rect=(0, 0, 1, 0.97))
            pdf.savefig(fig); plt.close(fig)

        # Pair page
        n_pair_cols = len(R2_PAIRS)
        fig, axes = plt.subplots(n_rows, n_pair_cols,
                                  figsize=(1.6 * n_pair_cols, 1.6 * n_rows),
                                  squeeze=False)
        for i, roi in enumerate(args.rois):
            for j, (m_y, m_x) in enumerate(R2_PAIRS):
                ax = axes[i, j]
                cell_r2_pair(ax, cache.get((roi, m_x)), cache.get((roi, m_y)))
                if i == 0:
                    ax.set_title(f"m{m_y} vs m{m_x}", fontsize=9)
                if j == 0:
                    ax.set_ylabel(roi, fontsize=11, weight="bold",
                                   rotation=0, ha="right", va="center")
        fig.suptitle("GROUP — R² pairwise (FDR-surviving in either model, pooled)",
                     weight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        pdf.savefig(fig); plt.close(fig)

    print(f"wrote {out}")


if __name__ == "__main__":
    main()
