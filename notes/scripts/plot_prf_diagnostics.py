"""PRF diagnostic plots: ROI × subject × model across the visual hierarchy.

For each ROI, makes a multi-page PDF where each page is a different
diagnostic (PRF center coverage, R² histogram, σ vs eccentricity).
Each page is a grid of rows=subjects × cols=models. Skips
(subject, model) cells where the NIfTI is missing — re-run as data
lands.

Outputs go to ``notes/figures/prf_diagnostics/{ROI}.pdf``.

Usage:
    python notes/scripts/plot_prf_diagnostics.py
"""
from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Circle
from scipy.stats import pearsonr

from retsupp.utils.data import Subject, select_well_fit_voxels

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "figure.titlesize": 14,
})

ROIS = ["V1", "V2", "V3", "V3AB", "hV4", "LO1", "TO1", "IPS"]
MODELS = [1, 2, 3, 4, 5, 6]
MODEL_LABELS = {
    1: "m1 Gauss",
    2: "m2 DoG",
    3: "m3 Gauss+HRF",
    4: "m4 DoG+HRF",
    5: "m5 DN",
    6: "m6 DN+HRF",
}
# Free parameter counts per model (for the F-test in select_well_fit_voxels)
N_PARAMS = {1: 5, 2: 7, 3: 7, 4: 9, 5: 9, 6: 11}
N_TIMEPOINTS = 3096  # 12 concatenated runs × 258 TR
APERTURE_R = 3.17  # bar PRF mapping aperture, degrees


def load_roi_voxels(sub: Subject, roi: str, model: int):
    """Return (all_df, well_df) for one (subject, model, ROI).

    ``all_df`` has every voxel in the ROI with sd > 0 (active);
    ``well_df`` is the subset passing FDR + aperture + σ filters via
    :func:`select_well_fit_voxels`. Returns ``(None, None)`` if the
    NIfTI is missing.
    """
    bids = sub.bids_folder
    base = bids / "derivatives" / "prf" / f"model{model}" / f"sub-{sub.subject_id:02d}"
    r2_path = base / f"sub-{sub.subject_id:02d}_desc-r2.nii.gz"
    if not r2_path.exists():
        return None, None
    try:
        roi_img = sub.get_retinotopic_roi(roi, bold_space=True)
    except Exception:
        return None, None
    roi_mask = np.asarray(roi_img.get_fdata(), dtype=bool).ravel()
    if roi_mask.sum() == 0:
        return None, None
    pars = {}
    for p in ("x", "y", "sd", "r2"):
        f = base / f"sub-{sub.subject_id:02d}_desc-{p}.nii.gz"
        if not f.exists():
            return None, None
        pars[p] = nib.load(f).get_fdata().ravel()[roi_mask]
    df = pd.DataFrame(pars)
    df = df[df["sd"] > 1e-3].reset_index(drop=True)
    if len(df) == 0:
        return df, df
    try:
        well, _ = select_well_fit_voxels(
            df, n_params=N_PARAMS[model], n_timepoints=N_TIMEPOINTS)
    except Exception:
        well = df.iloc[0:0]
    return df, well


def cell_coverage(ax, dfs):
    """dfs: (all_df, well_df) for this (sub, model). Plot well_df coverage."""
    if dfs is None or dfs[1] is None or len(dfs[1]) == 0:
        ax.set_axis_off()
        return
    well = dfs[1]
    if len(well) >= 30:
        # Subsample for KDE speed; ROIs can have 30k+ voxels and
        # gaussian_kde scales O(N²) in evaluation.
        sample = (well.sample(n=4000, random_state=0)
                   if len(well) > 4000 else well)
        try:
            sns.kdeplot(x=sample["x"], y=sample["y"], ax=ax,
                         fill=True, cmap="viridis", levels=8,
                         thresh=0.05, bw_adjust=0.7)
        except Exception:
            ax.scatter(well["x"], well["y"], s=2, alpha=0.3,
                        color="#1B4965")
    else:
        ax.scatter(well["x"], well["y"], s=3, alpha=0.7,
                    color="#1B4965")
    ax.add_patch(Circle((0, 0), APERTURE_R, facecolor="none",
                         edgecolor="black", lw=1.0, ls="--", alpha=0.7))
    ax.set_xlim(-5, 5); ax.set_ylim(-5, 5); ax.set_aspect("equal")
    ax.set_xticks([-3, 0, 3]); ax.set_yticks([-3, 0, 3])
    ax.text(0.02, 0.98, f"n={len(well)}", transform=ax.transAxes,
            color="black", fontsize=6, va="top")


def cell_r2_hist(ax, dfs):
    """dfs: (all_df, well_df). Histogram of all-voxel R²; mark FDR-survivors."""
    if dfs is None or dfs[0] is None or len(dfs[0]) == 0:
        ax.set_axis_off()
        return
    df, well = dfs
    bins = np.linspace(0, 0.5, 40)
    ax.hist(df["r2"], bins=bins, color="#E76F51", alpha=0.85,
            edgecolor="none")
    ax.set_yscale("log")
    ax.axvline(np.median(df["r2"]), color="black", lw=0.8)
    n_well = len(well)
    ax.text(0.98, 0.95,
            f"med={np.median(df['r2']):.3f}\nFDR n={n_well}",
            transform=ax.transAxes, ha="right", va="top", fontsize=6)
    ax.set_xlim(0, 0.5)
    ax.set_xticks([0, 0.25, 0.5])


def cell_sd_vs_ecc(ax, dfs):
    """dfs: (all_df, well_df). Scatter well-fit voxels + regression."""
    if dfs is None or dfs[1] is None:
        ax.set_axis_off()
        return
    well = dfs[1]
    if len(well) < 5:
        ax.text(0.5, 0.5, f"n={len(well)}", ha="center", va="center",
                transform=ax.transAxes)
    else:
        sample = (well.sample(n=2000, random_state=0)
                   if len(well) > 2000 else well)
        ax.scatter(sample["eccen"], sample["sd"], s=2, alpha=0.25,
                    color="#1B4965", edgecolor="none", rasterized=True)
        # OLS regression line + Pearson r on the well-fit set
        try:
            r, _ = pearsonr(well["eccen"], well["sd"])
            slope, intercept = np.polyfit(well["eccen"], well["sd"], 1)
            xs = np.array([0, 5])
            ax.plot(xs, slope * xs + intercept, color="#E76F51", lw=1.2)
            ax.text(0.02, 0.95, f"r={r:.2f}\nslope={slope:.2f}",
                    transform=ax.transAxes, va="top", fontsize=6)
        except (ValueError, np.linalg.LinAlgError):
            pass
    ax.axvline(APERTURE_R, color="black", lw=0.6, ls="--", alpha=0.5)
    ax.set_xlim(0, 5); ax.set_ylim(0, 3)
    ax.set_xticks([0, 2, 4]); ax.set_yticks([0, 1, 2, 3])


# Per-model pages: each cell = (subject, model)
PER_MODEL_PAGES = [
    ("PRF center coverage (well-fit, r²>0.05)", cell_coverage),
    ("R² distribution (log-y)", cell_r2_hist),
    ("σ vs eccentricity (well-fit)", cell_sd_vs_ecc),
]

# R² pair pages: each cell = (subject, (model_y, model_x))
# Compare model_y vs model_x R² per voxel; expect points above identity
# when model_y is the more flexible model.
R2_PAIRS = [(2, 1), (3, 1), (4, 1), (4, 2), (5, 2), (6, 5)]


def cell_r2_pair(ax, sid, m_y, m_x, cache):
    dfs_x = cache.get((sid, m_x)); dfs_y = cache.get((sid, m_y))
    if dfs_x is None or dfs_y is None:
        ax.set_axis_off(); return
    df_x = dfs_x[0]; df_y = dfs_y[0]   # use FULL ROI voxels, not just well-fit
    if df_x is None or df_y is None or len(df_x) != len(df_y):
        ax.set_axis_off()
        return
    # Voxels FDR-surviving in EITHER model — well-fit indices
    well_idx = set(dfs_x[1].index.tolist()) | set(dfs_y[1].index.tolist())
    if len(well_idx) < 5:
        ax.set_axis_off()
        return
    idx = sorted(well_idx)
    rx = df_x["r2"].iloc[idx].values
    ry = df_y["r2"].iloc[idx].values
    sample_idx = (np.random.default_rng(0).choice(len(rx),
                  min(3000, len(rx)), replace=False))
    ax.scatter(rx[sample_idx], ry[sample_idx], s=2, alpha=0.25,
                color="#1B4965", edgecolor="none", rasterized=True)
    lim = max(rx.max(), ry.max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=0.6, alpha=0.6)
    above = (ry > rx).mean()
    delta_med = float(np.median(ry - rx))
    ax.text(0.02, 0.98,
             f"n={len(rx)}\n{above:.0%} > diag\nΔmed={delta_med:+.3f}",
             transform=ax.transAxes, va="top", fontsize=6)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_aspect("equal")


def make_pdf(roi: str, subjects: list[int], bids_folder: str, out_dir: Path):
    """One ROI → one PDF. Per-model pages first, then R² pair page."""
    print(f"  {roi}: loading data...")
    cache = {}  # (sub, model) -> (all_df, well_df)
    for sid in subjects:
        sub = Subject(sid, bids_folder=bids_folder)
        for m in MODELS:
            cache[(sid, m)] = load_roi_voxels(sub, roi, m)

    pdf_path = out_dir / f"{roi}.pdf"
    with PdfPages(pdf_path) as pdf:
        # --- Per-model pages (subject × model grid) ---
        for page_title, cell_fn in PER_MODEL_PAGES:
            n_rows = len(subjects); n_cols = len(MODELS)
            fig, axes = plt.subplots(n_rows, n_cols,
                                      figsize=(1.4 * n_cols, 1.4 * n_rows),
                                      squeeze=False)
            for i, sid in enumerate(subjects):
                for j, m in enumerate(MODELS):
                    ax = axes[i, j]
                    cell_fn(ax, cache.get((sid, m)))
                    if i == 0:
                        ax.set_title(MODEL_LABELS[m], fontsize=8)
                    if j == 0:
                        ax.set_ylabel(f"sub-{sid:02d}", fontsize=8,
                                       rotation=0, ha="right", va="center")
            fig.suptitle(f"{roi} — {page_title}", weight="bold")
            fig.tight_layout(rect=(0, 0, 1, 0.97))
            pdf.savefig(fig)
            plt.close(fig)

        # --- R² pair page (subject × model-pair grid) ---
        n_rows = len(subjects); n_cols = len(R2_PAIRS)
        fig, axes = plt.subplots(n_rows, n_cols,
                                  figsize=(1.4 * n_cols, 1.4 * n_rows),
                                  squeeze=False)
        for i, sid in enumerate(subjects):
            for j, (m_y, m_x) in enumerate(R2_PAIRS):
                ax = axes[i, j]
                cell_r2_pair(ax, sid, m_y, m_x, cache)
                if i == 0:
                    ax.set_title(f"m{m_y} vs m{m_x}", fontsize=8)
                if j == 0:
                    ax.set_ylabel(f"sub-{sid:02d}", fontsize=8,
                                   rotation=0, ha="right", va="center")
        fig.suptitle(f"{roi} — R² pairwise (FDR-surviving voxels)",
                     weight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        pdf.savefig(fig)
        plt.close(fig)

    print(f"  wrote {pdf_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bids-folder", default="/data/ds-retsupp")
    p.add_argument("--subjects", nargs="+", type=int,
                    default=list(range(1, 31)),
                    help="Subject IDs to include (default 1..30)")
    p.add_argument("--rois", nargs="+", default=ROIS,
                    help=f"ROIs to plot (default {ROIS})")
    p.add_argument("--out-dir",
                    default="/Users/gdehol/git/retsupp/notes/figures/prf_diagnostics")
    args = p.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    print(f"Writing to {out}")
    for roi in args.rois:
        make_pdf(roi, args.subjects, args.bids_folder, out)
    print("Done.")


if __name__ == "__main__":
    main()
