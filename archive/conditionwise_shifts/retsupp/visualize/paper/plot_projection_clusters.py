#!/usr/bin/env python3
"""
Standalone plot: projection-onto-HP-opposite-axis vs distance from HP,
per ROI, smoothed per subject + 1D cluster-based permutation test.

This is the focused-version of Figure 3b from the meeting report — split
out so we can iterate on it without rebuilding the rest of the report.

The plot:
    Per (subject, ROI): voxels' projection along HP→opposite axis,
    binned by distance-from-HP into fine bins. Per-subject Gaussian
    smoothing across bins. Group-level: mean across subjects (bold line)
    ± 1 SEM (shaded band).

The test:
    At each bin, one-sample t against zero. Find clusters of consecutive
    bins with |t| > threshold. Cluster statistic = sum of |t| inside.
    Sign-flip permutation: each subject's smoothed curve gets its sign
    flipped with p=0.5; recompute t-curve, find max cluster stat. Repeat
    n_perm times → null distribution of max cluster statistic.

Multi-R² panels: by default produces panels at r²>{0.1, 0.2, 0.3, 0.4}
in a 2×2 layout per ROI (or wide horizontal if more rois) so you can
see how the effect changes with the inclusion threshold.

Usage:
    python -m retsupp.visualize.plot_projection_clusters \\
        --rois V3AB hV4 \\
        --r2-thresholds 0.1 0.2 0.3 0.4 \\
        --out notes/projection_clusters.pdf
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats
from scipy.ndimage import gaussian_filter1d

from retsupp.utils.data import roi_order
from retsupp.visualize.vss2026_arrows import load_all_conditionwise
from retsupp.visualize.meeting_report import filter_prf_inside_aperture


def smooth_per_subject_curves(
    pars: pd.DataFrame,
    rois: list[str],
    bin_edges: np.ndarray,
    smooth_sigma_bins: float,
    clip_projection: float,
    smoother: str = "lowess",
    lowess_frac: float = 0.4,
) -> dict[str, np.ndarray]:
    """For each ROI: per-subject smoothed curve of projection-away-from-HP
    vs distance, evaluated at the bin centers.

    `smoother`:
      "kernel"    — Nadaraya-Watson kernel regression with FIXED
                    Gaussian bandwidth (in degrees). Same effective
                    smoothing across ROIs regardless of voxel count.
                    Default — gives comparable curves.
      "gaussian"  — bin per subject, then Gaussian-smooth across bins
                    (low statistical power if few voxels per bin).
      "lowess"    — fit LOWESS directly to all per-voxel (distance,
                    projection) pairs per subject; evaluate at bin
                    centers.  Uses all voxel-level data → more power
                    but adaptive smoothness per ROI.

    Returns {roi: (n_subjects, n_bins)}, bin centers.
    """
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    work = pars.reset_index().copy()
    work = work[work["roi_base"].isin(rois)]
    work["proj_away_from_HP"] = -work["dy_rotated"]
    work = work[work["proj_away_from_HP"].abs() <= clip_projection]

    out = {}
    if smoother == "kernel":
        # Nadaraya-Watson with fixed Gaussian bandwidth in degrees.
        # Bandwidth defaults to lowess_frac if no --kernel-bandwidth was given.
        bw = lowess_frac  # repurpose the arg slot; CLI exposes --kernel-bandwidth.
        for roi in rois:
            roi_df = work[work["roi_base"] == roi]
            smoothed_rows = []
            for sub, sub_df in roi_df.groupby("subject", observed=True):
                d = sub_df["distance_from_distractor_mean"].values
                y = sub_df["proj_away_from_HP"].values
                if len(d) < 50:
                    continue
                # Gaussian kernel weights at each evaluation point.
                # weight_ij = exp(-(centers_i - d_j)^2 / (2*bw^2))
                # Estimate_i = sum_j weight_ij * y_j / sum_j weight_ij.
                diff = centers[:, None] - d[None, :]            # (n_bins, n_voxels)
                weights = np.exp(-(diff ** 2) / (2 * bw ** 2))
                w_sum = weights.sum(axis=1)
                # Don't predict where there's almost no data (effective N < 5)
                # to avoid wild extrapolation tails.
                eff_n = (weights / weights.max(axis=1, keepdims=True)).sum(axis=1)
                vals = (weights @ y) / np.maximum(w_sum, 1e-9)
                vals = np.where(eff_n >= 5, vals, np.nan)
                smoothed_rows.append(vals)
            out[roi] = (
                np.stack(smoothed_rows) if smoothed_rows
                else np.empty((0, len(centers)))
            )
    elif smoother == "lowess":
        from statsmodels.nonparametric.smoothers_lowess import lowess
        for roi in rois:
            roi_df = work[work["roi_base"] == roi]
            smoothed_rows = []
            for sub, sub_df in roi_df.groupby("subject", observed=True):
                d = sub_df["distance_from_distractor_mean"].values
                y = sub_df["proj_away_from_HP"].values
                if len(d) < 50:
                    continue
                # LOWESS expects sorted x; statsmodels handles that.
                # Evaluate at our bin centers.
                sm = lowess(
                    y, d, frac=lowess_frac,
                    it=1, return_sorted=True, missing="drop",
                )
                # Linearly interpolate the LOWESS output to bin centers.
                xs, ys = sm[:, 0], sm[:, 1]
                vals = np.interp(centers, xs, ys, left=np.nan, right=np.nan)
                smoothed_rows.append(vals)
            out[roi] = (
                np.stack(smoothed_rows) if smoothed_rows
                else np.empty((0, len(centers)))
            )
    else:
        # Original: bin then Gaussian-smooth.
        work["d_bin"] = pd.cut(
            work["distance_from_distractor_mean"], bin_edges,
            labels=centers, include_lowest=True,
        )
        work = work.dropna(subset=["d_bin"])
        work["d_bin"] = work["d_bin"].astype(float)
        per_sub_bin = (
            work.groupby(["subject", "roi_base", "d_bin"], observed=True)
            ["proj_away_from_HP"].mean().reset_index()
        )
        for roi in rois:
            roi_df = per_sub_bin[per_sub_bin["roi_base"] == roi]
            pivot = roi_df.pivot_table(
                index="subject", columns="d_bin", values="proj_away_from_HP",
            ).reindex(columns=centers)
            smoothed_rows = []
            for sub in pivot.index:
                row = pivot.loc[sub].values.astype(float)
                mask = ~np.isnan(row)
                if mask.sum() < 3:
                    continue
                row_filled = row.copy()
                row_filled[~mask] = np.nanmean(row)
                smoothed_rows.append(gaussian_filter1d(row_filled, smooth_sigma_bins))
            out[roi] = (
                np.stack(smoothed_rows) if smoothed_rows
                else np.empty((0, len(centers)))
            )
    return out, centers


def cluster_perm_test(
    smoothed: np.ndarray,
    cluster_threshold_t: float = 2.0,
    min_subj_per_bin: int | None = None,
    n_perm: int = 500,
    rng: np.random.Generator | None = None,
):
    """Run 1D cluster-based permutation test on a (n_subjects, n_bins)
    matrix of per-subject smoothed values.

    Returns: t_obs (n_bins,), clusters (list of (i_start, i_end, stat)),
             cluster_ps (list of float), null_max_stats (n_perm,).
    """
    if rng is None:
        rng = np.random.default_rng(42)
    if min_subj_per_bin is None:
        min_subj_per_bin = max(3, int(0.6 * smoothed.shape[0]))

    n_bins = smoothed.shape[1]

    def _t_curve(arr):
        t = np.full(n_bins, np.nan)
        for j in range(n_bins):
            col = arr[:, j]
            col = col[~np.isnan(col)]
            if len(col) >= min_subj_per_bin:
                t[j], _ = stats.ttest_1samp(col, 0.0)
        return t

    def _find_clusters(t_array, thr):
        sig = np.abs(t_array) > thr
        sig = np.nan_to_num(sig, nan=False).astype(bool)
        clusters = []
        i = 0
        while i < len(sig):
            if sig[i]:
                j = i
                while j < len(sig) and sig[j]:
                    j += 1
                cs = float(np.nansum(t_array[i:j]))
                clusters.append((i, j - 1, cs))
                i = j
            else:
                i += 1
        return clusters

    t_obs = _t_curve(smoothed)
    obs_clusters = _find_clusters(t_obs, cluster_threshold_t)

    null_max = np.zeros(n_perm)
    for k in range(n_perm):
        signs = rng.choice([-1, 1], size=smoothed.shape[0])
        t_perm = _t_curve(smoothed * signs[:, None])
        perm_clusters = _find_clusters(t_perm, cluster_threshold_t)
        null_max[k] = max((abs(c[2]) for c in perm_clusters), default=0.0)

    cluster_ps = []
    for _, _, cs in obs_clusters:
        p = (1 + np.sum(null_max >= abs(cs))) / (1 + n_perm)
        cluster_ps.append(float(p))
    return t_obs, obs_clusters, cluster_ps, null_max


def plot_panel(
    ax,
    smoothed: np.ndarray,
    centers: np.ndarray,
    bin_edges: np.ndarray,
    clusters: list,
    cluster_ps: list,
    title: str,
    y_lim: float,
):
    """Single panel: subject-level mean ± SEM with cluster shading."""
    if smoothed.size == 0:
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title(title)
        return
    mean = np.nanmean(smoothed, axis=0)
    n_per_bin = (~np.isnan(smoothed)).sum(axis=0).clip(min=1)
    sem = np.nanstd(smoothed, axis=0, ddof=1) / np.sqrt(n_per_bin)
    ax.fill_between(centers, mean - sem, mean + sem,
                    color="#444", alpha=0.25, label="±1 SEM across subjects")
    ax.plot(centers, mean, color="#222", lw=2.0,
            label=f"mean smoothed (n={smoothed.shape[0]})")
    ax.axhline(0, color="k", lw=0.5, ls="--")

    for (i_start, i_end, cs), p in zip(clusters, cluster_ps):
        color = "#d62728" if p < 0.05 else "#f0a0a0"
        ax.axvspan(bin_edges[i_start], bin_edges[i_end + 1],
                   color=color, alpha=0.25, zorder=0)
        x_mid = 0.5 * (bin_edges[i_start] + bin_edges[i_end + 1])
        ax.text(x_mid, ax.get_ylim()[1] * 0.92,
                f"p={p:.3f}", ha="center", va="top", fontsize=8,
                color="black" if p < 0.05 else "0.4",
                weight="bold" if p < 0.05 else "normal")

    ax.set_ylim(-y_lim, y_lim)
    ax.set_title(title, fontsize=10)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--bids-folder", default="/data/ds-retsupp")
    parser.add_argument("--out", type=Path,
                        default=Path("notes/projection_clusters.pdf"))
    parser.add_argument(
        "--rois", nargs="+",
        default=["V1", "V2", "V3", "V3AB", "hV4", "LO", "TO", "VO"],
    )
    parser.add_argument(
        "--r2-thresholds", type=float, nargs="+",
        default=[0.10, 0.20, 0.30, 0.40],
        help="R² thresholds to compare (one panel column per threshold)",
    )
    parser.add_argument("--bin-step", type=float, default=0.25,
                        help="Distance bin width in degrees")
    parser.add_argument("--smoother", choices=["kernel", "lowess", "gaussian"],
                        default="kernel",
                        help="kernel (default): Nadaraya-Watson with FIXED Gaussian "
                             "bandwidth in degrees — uniform smoothness across ROIs. "
                             "lowess: adaptive bandwidth (smoother for sparse ROIs). "
                             "gaussian: bin first, then smooth.")
    parser.add_argument("--kernel-bandwidth", type=float, default=0.5,
                        help="Kernel regression Gaussian bandwidth in degrees (kernel only)")
    parser.add_argument("--lowess-frac", type=float, default=0.4,
                        help="LOWESS frac (lowess only)")
    parser.add_argument("--smooth-sigma-bins", type=float, default=1.0,
                        help="Per-subject Gaussian smoothing kernel σ in bins (gaussian only)")
    parser.add_argument("--clip-projection", type=float, default=1.5,
                        help="Drop voxel-condition rows with |proj| > this")
    parser.add_argument("--cluster-threshold-t", type=float, default=2.0,
                        help="Per-bin |t| threshold for cluster definition")
    parser.add_argument("--n-perm", type=int, default=500,
                        help="Number of sign-flip permutations")
    parser.add_argument("--y-lim", type=float, default=0.10,
                        help="Y-axis limit (±) in degrees")
    parser.add_argument("--aperture-fraction-outside", type=float, default=0.5,
                        help="Drop voxels with > this fraction of any PRF outside aperture")
    args = parser.parse_args()

    bin_edges = np.arange(0.0, 7.25, args.bin_step)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    with PdfPages(args.out) as pdf:
        # Cover.
        fig_cover = plt.figure(figsize=(11, 8.5))
        ax = fig_cover.add_subplot(111); ax.axis("off")
        ax.text(0.5, 0.93,
                "Projection-onto-HP-opposite axis vs distance from HP\n"
                "Per-subject smoothed curves + 1D cluster-based permutation",
                ha="center", va="top", fontsize=16, weight="bold",
                transform=ax.transAxes)
        cover_text = (
            f"Inclusion: aperture filter (drop voxels with > "
            f"{args.aperture_fraction_outside:.0%} of any conditionwise PRF outside aperture).\n"
            f"  Plus per-panel r²_mean_model > threshold (varied across columns).\n\n"
            f"Pipeline:\n"
            f"  • Per (subject, ROI, distance bin): mean over voxels of projection_away_from_HP\n"
            f"    = −dy_rotated, where rotated frame puts HP at canonical (0, +4°).\n"
            f"  • Per subject: Gaussian-smooth across bins (σ = {args.smooth_sigma_bins} bins).\n"
            f"  • Group-level: BOLD LINE = mean across subjects of smoothed curves\n"
            f"                 SHADED BAND = ±1 SEM across subjects.\n"
            f"  • At each bin: one-sample t against 0. Bins with <60% subjects are skipped.\n"
            f"  • Cluster: consecutive bins with |t| > {args.cluster_threshold_t}.\n"
            f"    Cluster stat = Σ t inside.\n"
            f"  • Permutation null: each subject's smoothed curve sign-flipped with p=0.5,\n"
            f"    {args.n_perm} permutations, max-cluster-stat distribution.\n"
            f"  • Cluster significant if observed |stat| > 95th percentile of null.\n\n"
            f"Bin width: {args.bin_step}°.  |proj| clip: ±{args.clip_projection}°.  "
            f"y-axis: ±{args.y_lim}°.\n\n"
            f"R² thresholds compared: {args.r2_thresholds}\n"
            f"ROIs: {args.rois}"
        )
        ax.text(0.04, 0.83, cover_text, ha="left", va="top",
                fontsize=10, family="monospace", transform=ax.transAxes)
        pdf.savefig(fig_cover, bbox_inches="tight"); plt.close(fig_cover)

        # Load each R² threshold once (not once per ROI × threshold).
        print("Loading conditionwise data per R² threshold ...")
        pars_per_thr = {}
        for r2_thr in args.r2_thresholds:
            pars = load_all_conditionwise(
                bids_folder=args.bids_folder, r2_thr=r2_thr,
            )
            pars = filter_prf_inside_aperture(
                pars, aperture_radius=3.17,
                max_fraction_outside=args.aperture_fraction_outside,
                use_mean_only=False,
            )
            pars_per_thr[r2_thr] = pars
            print(f"  r2>{r2_thr:.2f}: {len(pars):,} rows")

        # ---------- HIERARCHY OVERVIEW PAGE ----------
        # For ONE chosen R² threshold (use the middle one), per ROI compute
        # two summary statistics from the smoothed curves:
        #   near suppression = mean of curve over d ∈ [0.5, 2.0°]
        #   far attraction   = mean of curve over d ∈ [4.0, 6.0°]
        # Per subject: same averages from per-subject smoothed curve.
        # Group: mean ± SEM across subjects, t-test against 0.
        from scipy import stats as _stats
        overview_thr = args.r2_thresholds[
            min(2, len(args.r2_thresholds) - 1)
        ]  # default index 2 (e.g., 0.30)
        overview_pars = pars_per_thr[overview_thr]
        overview_smoothed, centers = smooth_per_subject_curves(
            overview_pars, args.rois, bin_edges,
            args.smooth_sigma_bins, args.clip_projection,
            smoother=args.smoother,
            lowess_frac=(args.kernel_bandwidth
                         if args.smoother == "kernel" else args.lowess_frac),
        )
        near_mask = (centers >= 0.5) & (centers <= 2.0)
        far_mask = (centers >= 4.0) & (centers <= 6.0)

        overview_rows = []
        for roi in args.rois:
            sm = overview_smoothed[roi]                     # (n_subj, n_bins)
            if sm.size == 0:
                continue
            near_per_subj = np.nanmean(sm[:, near_mask], axis=1)
            far_per_subj = np.nanmean(sm[:, far_mask], axis=1)
            near_per_subj = near_per_subj[~np.isnan(near_per_subj)]
            far_per_subj = far_per_subj[~np.isnan(far_per_subj)]
            t_n, p_n = (_stats.ttest_1samp(near_per_subj, 0.0)
                        if len(near_per_subj) >= 3 else (np.nan, np.nan))
            t_f, p_f = (_stats.ttest_1samp(far_per_subj, 0.0)
                        if len(far_per_subj) >= 3 else (np.nan, np.nan))
            overview_rows.append({
                "roi": roi,
                "near_mean": float(np.mean(near_per_subj)) if len(near_per_subj) else np.nan,
                "near_sem": float(_stats.sem(near_per_subj)) if len(near_per_subj) else np.nan,
                "near_p": p_n,
                "far_mean": float(np.mean(far_per_subj)) if len(far_per_subj) else np.nan,
                "far_sem": float(_stats.sem(far_per_subj)) if len(far_per_subj) else np.nan,
                "far_p": p_f,
                "n_subj": int(len(near_per_subj)),
            })
        overview_df = pd.DataFrame(overview_rows)

        fig_ov, axes_ov = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
        x_pos = np.arange(len(overview_df))
        for ax, col_mean, col_sem, col_p, label, color in [
            (axes_ov[0], "near_mean", "near_sem", "near_p",
             f"NEAR HP (d∈[0.5,2.0]°) — suppression should be > 0", "#d62728"),
            (axes_ov[1], "far_mean",  "far_sem",  "far_p",
             f"FAR FROM HP (d∈[4.0,6.0]°) — attraction would be < 0", "#1f77b4"),
        ]:
            ax.errorbar(x_pos, overview_df[col_mean],
                        yerr=overview_df[col_sem], fmt="o",
                        color=color, markersize=8, capsize=5, lw=1.5)
            ax.axhline(0, color="k", lw=0.5, ls="--")
            ax.set_xticks(x_pos)
            ax.set_xticklabels(overview_df["roi"])
            ax.set_xlabel("ROI (visual hierarchy →)")
            ax.set_title(label, fontsize=10)
            ax.set_ylabel("smoothed proj. away from HP (deg)")
            # Significance asterisks.
            for i, p in enumerate(overview_df[col_p]):
                if pd.isna(p):
                    continue
                marker = ("***" if p < 0.001 else "**" if p < 0.01
                          else "*" if p < 0.05 else "")
                if marker:
                    y = overview_df[col_mean].iloc[i] + np.sign(
                        overview_df[col_mean].iloc[i] or 1.0
                    ) * (overview_df[col_sem].iloc[i] + 0.005)
                    ax.text(i, y, marker, ha="center",
                            va="bottom" if (overview_df[col_mean].iloc[i] or 0) >= 0 else "top",
                            fontsize=14, color="black")
        fig_ov.suptitle(
            f"Hierarchy overview — per-ROI summary at r²>{overview_thr}\n"
            f"Each dot = mean across subjects of (subject's mean smoothed curve in band) ± SEM. "
            f"* p<0.05, ** p<0.01, *** p<0.001 (one-sample t against 0, no FDR).",
            y=1.02,
        )
        fig_ov.tight_layout()
        pdf.savefig(fig_ov, bbox_inches="tight"); plt.close(fig_ov)

        # One page per ROI: columns = R² thresholds.
        for roi in args.rois:
            n_thr = len(args.r2_thresholds)
            fig, axes = plt.subplots(
                1, n_thr,
                figsize=(3.5 * n_thr, 3.2),
                sharey=True, sharex=True,
            )
            axes = np.atleast_1d(axes)

            for ax, r2_thr in zip(axes, args.r2_thresholds):
                pars = pars_per_thr[r2_thr]
                # For kernel smoother, pass the bandwidth via lowess_frac slot.
                bw_or_frac = (
                    args.kernel_bandwidth if args.smoother == "kernel"
                    else args.lowess_frac
                )
                smoothed_per_roi, centers = smooth_per_subject_curves(
                    pars, [roi], bin_edges,
                    args.smooth_sigma_bins, args.clip_projection,
                    smoother=args.smoother, lowess_frac=bw_or_frac,
                )
                smoothed = smoothed_per_roi[roi]

                _, clusters, cluster_ps, _ = cluster_perm_test(
                    smoothed, cluster_threshold_t=args.cluster_threshold_t,
                    n_perm=args.n_perm, rng=rng,
                )

                plot_panel(
                    ax, smoothed, centers, bin_edges,
                    clusters, cluster_ps,
                    title=f"r²>{r2_thr:.2f}  (n_subj={smoothed.shape[0]})",
                    y_lim=args.y_lim,
                )
                ax.set_xlabel("distance of mean PRF from HP (deg)")
            axes[0].set_ylabel("smoothed proj. away from HP (deg)")

            fig.suptitle(
                f"{roi} — projection-vs-distance, per R² threshold "
                f"(bold = mean across subjects, shaded = ±1 SEM, "
                f"red = p<0.05 cluster)",
                y=1.04,
            )
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)
            print(f"  {roi} done.")

    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
