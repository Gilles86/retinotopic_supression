"""VSS 2026 Figure B: PRF shifts as a function of HP-distractor location.

Two panels:
  B1 — arrow field: per spatial bin of mean-PRF position, the mean shift
       (x - x_mean, y - y_mean) across voxels in one ROI, one panel per
       HP-distractor condition, with the HP location overlaid as a star.
  B2 — rotated hexbin: pooling all four conditions after rotating each
       voxel so the current HP location is at the top, color = the radial
       shift in distance-from-HP-distractor (negative = pulled toward,
       positive = pushed away). One panel per ROI.

Run as a script to dump both panels to a PDF; or import the plotting
functions into a notebook.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import TwoSlopeNorm

from retsupp.utils.data import (
    Subject,
    distractor_locations,
    get_subject_ids,
    location_angles,
    roi_order,
)


CONDITIONS = ["upper_right", "upper_left", "lower_left", "lower_right"]
CONDITION_COLORS = {
    "upper_right": "#d62728",
    "upper_left": "#1f77b4",
    "lower_left": "#2ca02c",
    "lower_right": "#9467bd",
}

# For each condition (underscore form), the diametrically opposite ring position
# (space form, matches the `distance_from_<loc>` columns).
OPPOSITE_LOCATION = {
    "upper_right": "lower left",
    "upper_left": "lower right",
    "lower_left": "upper right",
    "lower_right": "upper left",
}


def load_all_conditionwise(
    bids_folder: str | Path = "/data/ds-retsupp",
    model: int = 8,
    r2_thr: float = 0.2,
    ecc_thr: float = 3.0,
    sd_thr: float = 0.3,
    sd_max: float = 4.0,
) -> pd.DataFrame:
    """Load conditionwise PRF summaries for all subjects with data, concat, filter.

    Inclusion criteria (all on **mean-model** = standalone model-4 fit, so
    voxel set is identical across the four HP conditions — no condition-
    dependent selection bias):
        r2_mean_model > r2_thr      (default 0.2)   — well-fit voxels.
        ecc_mean_model < ecc_thr    (default 3.0)   — within stimulus FOV.
        sd_mean_model > sd_thr      (default 0.3)   — meaningful PRF size.
                                                      Stimulus grid pixel ≈ 0.16°,
                                                      so 0.3° ≈ 2 px lower bound.
        sd_mean_model < sd_max      (default 4.0)   — drop runaway fits that
                                                      exceed the aperture.
    """
    bids_folder = Path(bids_folder)
    frames = []
    for subject_id in get_subject_ids():
        sub = Subject(subject_id, bids_folder=bids_folder)
        try:
            df = sub.get_conditionwise_summary_prf_pars(model=model)
        except FileNotFoundError:
            continue
        df["subject"] = subject_id
        frames.append(df)

    if not frames:
        raise RuntimeError(f"No conditionwise data found under {bids_folder}")

    pars = pd.concat(frames)
    pars = pars[
        (pars["r2_mean_model"] > r2_thr)
        & (pars["ecc_mean_model"] < ecc_thr)
        & (pars["sd_mean_model"] > sd_thr)
        & (pars["sd_mean_model"] < sd_max)
    ]

    pars["dx"] = pars["x"] - pars["x_mean"]
    pars["dy"] = pars["y"] - pars["y_mean"]
    pars["dx_rotated"] = pars["x_rotated"] - pars["x_mean_rotated"]
    pars["dy_rotated"] = pars["y_rotated"] - pars["y_mean_rotated"]
    pars["shift_from_distractor"] = (
        pars["distance_from_distractor"] - pars["distance_from_distractor_mean"]
    )
    pars["shift_from_distractor_rotated"] = (
        pars["distance_from_distractor_rotated"]
        - pars["distance_from_distractor_mean_rotated"]
    )

    # Paired "opposite anchor" baseline: distance from the diametrically
    # opposite ring location (in the same condition row, voxel position
    # comes from this condition; mean comes from across conditions).
    # Under isotropic fitting noise (Jensen bias), HP_shift and
    # opposite_shift have equal expectation. Under real HP-specific
    # suppression, HP_shift > 0 and opposite_shift < 0 because pushing
    # away from HP brings the voxel closer to the opposite ring point.
    cond_idx = pars.index.get_level_values("condition").to_numpy()
    opp_d = np.empty(len(pars))
    opp_dm = np.empty(len(pars))
    for cond in CONDITIONS:
        mask = cond_idx == cond
        opp_loc = OPPOSITE_LOCATION[cond]
        opp_d[mask] = pars.loc[mask, f"distance_from_{opp_loc}"].to_numpy()
        opp_dm[mask] = pars.loc[mask, f"distance_from_{opp_loc}_mean"].to_numpy()
    pars["distance_from_opposite"] = opp_d
    pars["distance_from_opposite_mean"] = opp_dm
    pars["shift_from_opposite"] = opp_d - opp_dm
    # The paired metric: HP shift minus opposite shift.
    # Centers at 0 under Jensen-only null; equals 2x the suppression
    # effect under real HP-specific push-away.
    pars["paired_hp_minus_opposite"] = (
        pars["shift_from_distractor"] - pars["shift_from_opposite"]
    )
    return pars


def _bin_arrows(
    df: pd.DataFrame,
    n_bins: int = 8,
    extent: float = 3.0,
    min_subjects: int = 5,
    clip_magnitude: float | None = 1.0,
    x_col: str = "x_mean",
    y_col: str = "y_mean",
    dx_col: str = "dx",
    dy_col: str = "dy",
) -> pd.DataFrame:
    """Bin voxels by (x_col, y_col), return per-bin median (dx_col, dy_col).

    Two-stage aggregation: within-subject median per bin, then median across
    subjects. Median (rather than mean) makes the field robust to bins with
    a handful of extreme voxels. Bins with fewer than `min_subjects`
    contributing subjects are dropped. If `clip_magnitude` is set, arrow
    lengths are clipped to that radius (in degrees) for display.

    Pass the rotated-frame columns (e.g. `x_col='x_mean_rotated'`,
    `dx_col='dx_rotated'`) to bin in the HP-aligned frame.
    """
    edges = np.linspace(-extent, extent, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    work = df.copy()
    work["xb"] = pd.cut(work[x_col], edges, labels=centers, include_lowest=True)
    work["yb"] = pd.cut(work[y_col], edges, labels=centers, include_lowest=True)
    work = work.dropna(subset=["xb", "yb"])

    per_subject = (
        work.groupby(["subject", "xb", "yb"], observed=True)
        .agg(dx=(dx_col, "median"), dy=(dy_col, "median"))
        .reset_index()
    )
    pooled = (
        per_subject.groupby(["xb", "yb"], observed=True)
        .agg(dx=("dx", "median"), dy=("dy", "median"), n_subj=("dx", "size"))
        .reset_index()
    )
    pooled["xb"] = pooled["xb"].astype(float)
    pooled["yb"] = pooled["yb"].astype(float)
    pooled = pooled[pooled["n_subj"] >= min_subjects]

    if clip_magnitude is not None:
        mag = np.hypot(pooled["dx"], pooled["dy"])
        scale = np.where(mag > clip_magnitude, clip_magnitude / np.maximum(mag, 1e-9), 1.0)
        pooled["dx"] = pooled["dx"] * scale
        pooled["dy"] = pooled["dy"] * scale
    return pooled


def plot_arrow_field(
    pars: pd.DataFrame,
    roi: str = "V1",
    n_bins: int = 5,
    extent: float = 3.0,
    arrow_scale: float = 4.0,
    fig: plt.Figure | None = None,
):
    """Figure B1 — 2x2 arrow plot of binned PRF shifts, per HP condition."""
    sub_df = pars.xs(roi, level="roi_base")
    if fig is None:
        fig, axes = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)
    else:
        axes = np.array(fig.axes).reshape(2, 2)

    fig.suptitle(f"PRF shifts by HP-distractor location — {roi}", y=0.99)

    for ax, condition in zip(axes.ravel(), CONDITIONS):
        cond_df = sub_df.xs(condition, level="condition")
        binned = _bin_arrows(cond_df, n_bins=n_bins, extent=extent)

        # Distractor ring at 4° eccentricity.
        theta = np.linspace(0, 2 * np.pi, 200)
        ax.plot(4 * np.cos(theta), 4 * np.sin(theta), "k--", lw=0.5, alpha=0.3)

        # All four HP positions, faded.
        for c, (hx, hy) in distractor_locations.items():
            ax.scatter(hx, hy, s=70, color="lightgray", marker="*", zorder=2)

        # Current HP, highlighted.
        cond_label = condition.replace("_", " ")
        hx, hy = distractor_locations[cond_label]
        ax.scatter(
            hx, hy, s=300, marker="*",
            color=CONDITION_COLORS[condition], edgecolor="k", zorder=3,
        )

        ax.quiver(
            binned["xb"], binned["yb"], binned["dx"], binned["dy"],
            angles="xy", scale_units="xy", scale=1.0 / arrow_scale,
            color=CONDITION_COLORS[condition], width=0.005, alpha=0.85,
        )

        ax.set_xlim(-extent - 1, extent + 1)
        ax.set_ylim(-extent - 1, extent + 1)
        ax.set_aspect("equal")
        ax.axhline(0, color="k", lw=0.3, alpha=0.3)
        ax.axvline(0, color="k", lw=0.3, alpha=0.3)
        ax.set_title(f"HP = {cond_label}")
        ax.set_xlabel("x (deg)")
        ax.set_ylabel("y (deg)")

    fig.tight_layout()
    return fig


def plot_arrow_field_rotated(
    pars: pd.DataFrame,
    rois: list[str] | None = None,
    n_bins: int = 5,
    extent: float = 4.0,
    arrow_scale: float = 6.0,
    fig: plt.Figure | None = None,
):
    """Combined-rotated arrow plot — one panel per ROI, all 4 conditions pooled.

    Each voxel × condition is rotated so its current HP location sits at
    (0, 4°). Arrows show the per-spatial-bin median shift in HP-aligned
    coordinates. Pulls toward HP point upward; pushes away point downward.
    """
    if rois is None:
        rois = [r for r in roi_order if r in pars.index.get_level_values("roi_base").unique()]
    n = len(rois)
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))
    if fig is None:
        fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows + 0.5),
                                 sharex=True, sharey=True)
    axes = np.atleast_2d(axes).ravel()

    fig.suptitle("PRF shifts in HP-aligned frame (HP at top, all conditions pooled)", y=1.0)

    for ax, roi in zip(axes, rois):
        roi_df = pars.xs(roi, level="roi_base")
        binned = _bin_arrows(
            roi_df,
            n_bins=n_bins,
            extent=extent - 1,
            x_col="x_mean_rotated",
            y_col="y_mean_rotated",
            dx_col="dx_rotated",
            dy_col="dy_rotated",
        )

        theta = np.linspace(0, 2 * np.pi, 200)
        ax.plot(4 * np.cos(theta), 4 * np.sin(theta), "k--", lw=0.5, alpha=0.3)
        ax.scatter(0, 4, s=300, marker="*", color="k", zorder=3)

        ax.quiver(
            binned["xb"], binned["yb"], binned["dx"], binned["dy"],
            angles="xy", scale_units="xy", scale=1.0 / arrow_scale,
            color="C0", width=0.006, alpha=0.85,
        )
        ax.set_xlim(-extent, extent)
        ax.set_ylim(-extent, extent)
        ax.set_aspect("equal")
        ax.axhline(0, color="k", lw=0.3, alpha=0.3)
        ax.axvline(0, color="k", lw=0.3, alpha=0.3)
        ax.set_title(roi)

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.tight_layout()
    return fig


def plot_rotated_hexbin(
    pars: pd.DataFrame,
    rois: list[str] | None = None,
    extent: float = 4.0,
    gridsize: int = 12,
    vmax: float = 0.3,
    fig: plt.Figure | None = None,
):
    """Figure B2 — rotated-coordinate hexbin of radial Δ-distance per ROI.

    All conditions pooled after rotating each voxel so its current HP
    is at the top (y > 0). Color = radial shift from the HP location:
    blue = pulled toward HP, red = pushed away.
    """
    if rois is None:
        rois = [r for r in roi_order if r in pars.index.get_level_values("roi_base").unique()]
    n = len(rois)
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))
    if fig is None:
        fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows + 0.5), sharex=True, sharey=True)
    axes = np.atleast_2d(axes).ravel()

    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    last_hb = None

    for ax, roi in zip(axes, rois):
        roi_df = pars.xs(roi, level="roi_base")
        last_hb = ax.hexbin(
            roi_df["x_mean_rotated"],
            roi_df["y_mean_rotated"],
            C=roi_df["shift_from_distractor_rotated"],
            gridsize=gridsize, cmap="coolwarm", norm=norm,
            extent=(-extent, extent, -extent, extent), mincnt=20,
        )
        # HP location (always at top after rotation).
        ax.scatter(0, 4, s=300, marker="*", color="k", zorder=3)
        # Distractor ring.
        theta = np.linspace(0, 2 * np.pi, 200)
        ax.plot(4 * np.cos(theta), 4 * np.sin(theta), "k--", lw=0.5, alpha=0.3)
        ax.axhline(0, color="k", lw=0.3, alpha=0.3)
        ax.axvline(0, color="k", lw=0.3, alpha=0.3)
        ax.set_xlim(-extent, extent)
        ax.set_ylim(-extent, extent)
        ax.set_aspect("equal")
        ax.set_title(roi)

    for ax in axes[n:]:
        ax.set_visible(False)

    if last_hb is not None:
        cbar = fig.colorbar(last_hb, ax=axes[:n].tolist(), shrink=0.8, pad=0.02)
        cbar.set_label("Δ distance from HP (deg)\n(blue = pulled toward, red = pushed away)")

    fig.suptitle("Rotated-coordinate radial shifts by ROI", y=1.01)
    return fig


def shift_vs_distance_table(
    pars: pd.DataFrame,
    rois: list[str] | None = None,
    bin_edges: np.ndarray | None = None,
) -> pd.DataFrame:
    """Per-(subject, ROI, distance bin) mean radial shift from HP.

    `distance_from_distractor_mean` is the voxel's *mean-condition* PRF
    distance from the HP locus — this is the natural x-axis because we
    expect the AF+ prediction (effect only when the pRF overlaps the HP)
    to manifest as a curve that decays with distance.

    Sign convention: shift_from_distractor > 0 means the conditionwise
    PRF is **further** from HP than the mean PRF (pushed away);
    shift < 0 means pulled toward.
    """
    if bin_edges is None:
        bin_edges = np.arange(0.0, 7.5, 0.5)
    if rois is None:
        rois = [r for r in roi_order if r in pars.index.get_level_values("roi_base").unique()]

    work = pars.reset_index()
    work = work[work["roi_base"].isin(rois)].copy()
    work["d_bin"] = pd.cut(
        work["distance_from_distractor_mean"], bin_edges,
        labels=0.5 * (bin_edges[:-1] + bin_edges[1:]),
        include_lowest=True,
    )
    work = work.dropna(subset=["d_bin"])
    work["d_bin"] = work["d_bin"].astype(float)

    per_subject = (
        work.groupby(["subject", "roi_base", "d_bin"], observed=True)["shift_from_distractor"]
        .mean()
        .reset_index()
    )
    per_subject.rename(columns={"shift_from_distractor": "mean_shift"}, inplace=True)
    return per_subject


def plot_shift_vs_distance(
    pars: pd.DataFrame,
    rois: list[str] | None = None,
    bin_edges: np.ndarray | None = None,
    smooth: bool = True,
    fig: plt.Figure | None = None,
):
    """Lineplot of mean radial shift vs distance from HP, one line per ROI.

    Wider bins (1° default) + an overlaid LOWESS smooth (when `smooth`)
    to suppress per-bin noise without hiding the underlying shape.
    """
    import seaborn as sns

    if rois is None:
        rois = [r for r in roi_order if r in pars.index.get_level_values("roi_base").unique()]
    if bin_edges is None:
        bin_edges = np.arange(0.0, 7.5, 1.0)  # 1.0° bins

    table = shift_vs_distance_table(pars, rois=rois, bin_edges=bin_edges)

    if fig is None:
        fig, ax = plt.subplots(figsize=(9, 5))
    else:
        ax = fig.gca()

    palette = sns.color_palette("viridis", n_colors=len(rois))

    if smooth:
        # Discrete bin points with light alpha + a LOWESS smooth overlay per ROI.
        sns.lineplot(
            data=table, x="d_bin", y="mean_shift", hue="roi_base", hue_order=rois,
            palette=palette, errorbar=("ci", 95), ax=ax,
            estimator="mean", alpha=0.25, legend=False,
        )
        from statsmodels.nonparametric.smoothers_lowess import lowess
        for color, roi in zip(palette, rois):
            sub = table[table["roi_base"] == roi].dropna(subset=["d_bin", "mean_shift"])
            if len(sub) < 5:
                continue
            sm = lowess(sub["mean_shift"].values, sub["d_bin"].astype(float).values,
                        frac=0.6, return_sorted=True)
            ax.plot(sm[:, 0], sm[:, 1], color=color, lw=2.0, label=roi)
        ax.legend(title="ROI", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    else:
        sns.lineplot(
            data=table, x="d_bin", y="mean_shift", hue="roi_base", hue_order=rois,
            palette=palette, errorbar=("ci", 95), ax=ax,
        )
        ax.legend(title="ROI", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)

    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.set_xlabel("Distance of mean PRF from HP distractor (deg)")
    ax.set_ylabel("Δ distance from HP (deg)\n(+ = pushed away, − = pulled toward)")
    ax.set_title("Radial PRF shift as a function of distance from HP distractor"
                 + ("\n(faded = 1° bins ± 95% CI; bold = LOWESS smooth)" if smooth else ""))
    fig.tight_layout()
    return fig


def plot_shift_by_roi_and_distance(
    pars: pd.DataFrame,
    rois: list[str] | None = None,
    bin_edges: np.ndarray | None = None,
    fig: plt.Figure | None = None,
):
    """Per-ROI mean shift across several distance-from-HP bins, one panel.

    x = ROI, y = mean shift, hue = distance bin (non-overlapping). Shows
    how the suppression effect attenuates with distance per ROI. The
    near-distractor bin is the dominant signal; far bins should sit at
    zero across all ROIs.
    """
    import seaborn as sns

    if rois is None:
        rois = [r for r in roi_order if r in pars.index.get_level_values("roi_base").unique()]
    if bin_edges is None:
        bin_edges = np.array([0.0, 1.0, 2.0, 3.0, 5.0])

    work = pars.reset_index()
    work = work[work["roi_base"].isin(rois)].copy()
    bin_labels = [f"{bin_edges[i]:.1f}–{bin_edges[i+1]:.1f}°" for i in range(len(bin_edges) - 1)]
    work["d_bin"] = pd.cut(
        work["distance_from_distractor_mean"], bin_edges,
        labels=bin_labels, include_lowest=True,
    )
    work = work.dropna(subset=["d_bin"])

    per_subject = (
        work.groupby(["subject", "roi_base", "d_bin"], observed=True)["shift_from_distractor"]
        .mean()
        .reset_index()
    )

    if fig is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        ax = fig.gca()

    palette = sns.color_palette("rocket_r", n_colors=len(bin_labels))
    sns.pointplot(
        data=per_subject,
        x="roi_base", y="shift_from_distractor",
        hue="d_bin", order=rois, hue_order=bin_labels,
        palette=palette, errorbar=("ci", 95),
        dodge=0.4, markers="o", linestyles="-", ax=ax,
    )
    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.set_xlabel("ROI")
    ax.set_ylabel("Δ distance from HP (deg)\n(+ = pushed away)")
    ax.set_title("Mean PRF shift by ROI, stratified by mean PRF distance from HP distractor"
                 "\n(per-subject means; error = 95% CI)")
    ax.legend(title="Distance from HP", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    fig.tight_layout()
    return fig


def near_distractor_stats(
    pars: pd.DataFrame,
    rois: list[str] | None = None,
    near_threshold: float = 1.5,
) -> pd.DataFrame:
    """Per-ROI one-sample t-test of mean shift for voxels with mean PRF within
    `near_threshold` deg of the HP locus.

    Returns a DataFrame with: roi, n_subj, mean, sem, t, p, p_fdr.
    """
    from scipy import stats
    from statsmodels.stats.multitest import multipletests

    if rois is None:
        rois = [r for r in roi_order if r in pars.index.get_level_values("roi_base").unique()]

    work = pars.reset_index()
    near = work[work["distance_from_distractor_mean"] <= near_threshold]
    per_subject = (
        near.groupby(["subject", "roi_base"], observed=True)["shift_from_distractor"]
        .mean()
        .reset_index()
    )

    rows = []
    for roi in rois:
        vals = per_subject.loc[per_subject["roi_base"] == roi, "shift_from_distractor"].dropna().values
        if len(vals) < 3:
            rows.append({"roi": roi, "n_subj": len(vals), "mean": np.nan,
                         "sem": np.nan, "t": np.nan, "p": np.nan})
            continue
        t, p = stats.ttest_1samp(vals, 0.0)
        rows.append({
            "roi": roi, "n_subj": len(vals),
            "mean": float(np.mean(vals)),
            "sem": float(stats.sem(vals)),
            "t": float(t), "p": float(p),
        })

    out = pd.DataFrame(rows)
    valid = out["p"].notna()
    if valid.any():
        _, p_fdr, _, _ = multipletests(out.loc[valid, "p"].values, method="fdr_bh")
        out.loc[valid, "p_fdr"] = p_fdr
    out.attrs["near_threshold"] = near_threshold
    out.attrs["per_subject"] = per_subject
    return out


def plot_near_distractor_summary(
    stats_df: pd.DataFrame,
    fig: plt.Figure | None = None,
):
    """Per-ROI dot-and-error plot of the mean radial shift in the
    near-HP bin (output of `near_distractor_stats`)."""
    import seaborn as sns

    per_subject = stats_df.attrs.get("per_subject")
    near_threshold = stats_df.attrs.get("near_threshold")

    if fig is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        ax = fig.gca()

    rois = stats_df["roi"].tolist()
    if per_subject is not None:
        sns.stripplot(
            data=per_subject, x="roi_base", y="shift_from_distractor",
            order=rois, color="0.6", alpha=0.6, jitter=0.15, ax=ax,
        )
    ax.errorbar(
        x=np.arange(len(rois)),
        y=stats_df["mean"], yerr=stats_df["sem"],
        fmt="o", color="black", capsize=4, lw=1.5, markersize=7, zorder=3,
    )
    ax.axhline(0, color="k", lw=0.5, ls="--")

    # FDR significance markers above each ROI.
    p_col = "p_fdr" if "p_fdr" in stats_df.columns else "p"
    ymax = stats_df["mean"].add(stats_df["sem"]).max()
    for i, (roi, p) in enumerate(zip(rois, stats_df[p_col])):
        if pd.isna(p):
            continue
        marker = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        ax.text(i, ymax * 1.15, marker, ha="center", va="bottom",
                fontsize=11 if marker != "n.s." else 9,
                color="black" if marker != "n.s." else "0.5")

    ax.set_xlabel("ROI")
    ax.set_ylabel("Δ distance from HP (deg)\n(+ = pushed away)")
    ax.set_title(f"Mean PRF shift for voxels within {near_threshold:.1f}° of HP distractor"
                 f"\n(per-subject means; error = SEM; FDR-corrected)")
    fig.tight_layout()
    return fig


def add_anchor_shifts(pars: pd.DataFrame) -> pd.DataFrame:
    """Long-format table with one row per (voxel, condition, anchor).

    For each voxel × condition, computes:
        - radial shift relative to each of the four ring anchors (Jensen-prone)
        - signed projection of the shift vector onto the voxel→anchor
          direction (no Jensen — purely linear)

    `is_hp` flags the anchor that matches the condition's actual HP, so
    rows can be grouped per-anchor into 1 HP row vs 3 LP rows per voxel
    (same anchor, different HP-status).
    """
    work = pars.reset_index().copy()
    out = []
    for loc in distractor_locations:
        col_d = f"distance_from_{loc}"
        col_dm = f"distance_from_{loc}_mean"
        if col_d not in work.columns or col_dm not in work.columns:
            continue
        sub = work[[
            "subject", "roi_base", "voxel", "condition",
            "x", "y", "x_mean", "y_mean",
            col_d, col_dm,
        ]].copy()
        sub.rename(columns={col_d: "anchor_distance",
                            col_dm: "anchor_distance_mean"}, inplace=True)
        sub["anchor"] = loc
        sub["shift_from_anchor"] = sub["anchor_distance"] - sub["anchor_distance_mean"]

        # Projection of shift onto the voxel-mean → anchor direction:
        #   Δ = (x − x_mean, y − y_mean)
        #   u = (anchor − mean) / ||anchor − mean||
        #   proj = Δ · u  > 0 if voxel moved toward anchor.
        anchor_xy = np.array(distractor_locations[loc])
        ux = anchor_xy[0] - sub["x_mean"]
        uy = anchor_xy[1] - sub["y_mean"]
        norm = np.sqrt(ux ** 2 + uy ** 2)
        ux = np.where(norm > 1e-9, ux / norm, 0.0)
        uy = np.where(norm > 1e-9, uy / norm, 0.0)
        sub["projection_toward_anchor"] = (sub["x"] - sub["x_mean"]) * ux + (sub["y"] - sub["y_mean"]) * uy
        # Sign convention swap so + means moved AWAY from anchor (suppression-like).
        sub["projection_away_from_anchor"] = -sub["projection_toward_anchor"]

        sub["is_hp"] = (sub["condition"].str.replace("_", " ") == loc)
        out.append(sub)
    return pd.concat(out, ignore_index=True)


def plot_hp_vs_lp_per_anchor(
    pars: pd.DataFrame,
    rois: list[str] | None = None,
    near_threshold: float = 1.5,
    metric: str = "projection_away_from_anchor",
    fig: plt.Figure | None = None,
):
    """Per-anchor HP-vs-LP contrast, transparent design view.

    For each ring anchor, compare voxels' shift in the condition where
    that anchor IS the HP (1 condition) vs the three conditions where
    the same anchor is LP (low-probability). Same anchor throughout —
    this is the comparison Gilles described.

    Default metric is the linear projection (no Jensen bias).
    """
    import seaborn as sns
    from scipy import stats

    if rois is None:
        rois = [r for r in roi_order if r in pars.index.get_level_values("roi_base").unique()]

    tbl = add_anchor_shifts(pars)
    tbl = tbl[
        (tbl["roi_base"].isin(rois))
        & (tbl["anchor_distance_mean"] <= near_threshold)
    ].copy()
    tbl["status"] = tbl["is_hp"].map({True: "HP", False: "LP"})

    per_subj = (
        tbl.groupby(["subject", "roi_base", "anchor", "status"], observed=True)
        [metric].mean().reset_index()
    )

    if fig is None:
        fig, axes = plt.subplots(1, len(rois), figsize=(4.5 * len(rois), 5),
                                 sharey=True)
    axes = np.atleast_1d(axes)

    anchor_order = ["upper right", "upper left", "lower left", "lower right"]
    for ax, roi in zip(axes, rois):
        roi_df = per_subj[per_subj["roi_base"] == roi]
        sns.pointplot(
            data=roi_df, x="anchor", y=metric,
            hue="status", order=anchor_order, hue_order=["HP", "LP"],
            palette={"HP": "#d62728", "LP": "#1f77b4"},
            errorbar=("ci", 95), dodge=0.4,
            markers=["o", "s"], linestyles=["-", "--"], ax=ax,
        )
        ax.axhline(0, color="k", lw=0.5, ls="--")
        ax.set_xticklabels([a.replace(" ", "\n") for a in anchor_order], fontsize=9)
        ax.set_xlabel("anchor (distractor location)")
        ax.set_ylabel(f"{metric} (deg)")
        ax.set_title(f"{roi}")

        # Annotate paired t per anchor (HP - LP).
        ymax = 0
        for i, anchor in enumerate(anchor_order):
            piv = (
                roi_df[roi_df["anchor"] == anchor]
                .pivot_table(index="subject", columns="status", values=metric)
            )
            if "HP" in piv.columns and "LP" in piv.columns:
                diff = piv["HP"] - piv["LP"]
                diff = diff.dropna()
                if len(diff) >= 3:
                    t, p = stats.ttest_1samp(diff, 0.0)
                    marker = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                    if marker:
                        ax.text(i, ax.get_ylim()[1] * 0.95, marker, ha="center",
                                va="top", fontsize=12, color="black")

    fig.suptitle(
        f"Per-anchor HP-vs-LP comparison ({metric})\n"
        f"For each ring location, contrast voxel shift when that location is "
        f"HP (red) vs LP (blue, mean of the three conditions where it's LP).",
        y=1.02,
    )
    fig.tight_layout()
    return fig


def plot_hp_vs_nonhp_summary(
    pars: pd.DataFrame,
    rois: list[str] | None = None,
    near_threshold: float = 1.5,
    fig: plt.Figure | None = None,
):
    """Per-ROI HP vs non-HP shift in the near-anchor bin.

    For each voxel × condition × anchor with mean PRF distance to anchor
    < `near_threshold`, computes the radial shift. Compares HP-anchored
    voxels to non-HP-anchored voxels per ROI. Paired t-test (subject).
    """
    import seaborn as sns
    from scipy import stats
    from statsmodels.stats.multitest import multipletests

    if rois is None:
        rois = [r for r in roi_order if r in pars.index.get_level_values("roi_base").unique()]

    tbl = add_anchor_shifts(pars)
    tbl = tbl[
        (tbl["roi_base"].isin(rois))
        & (tbl["anchor_distance_mean"] <= near_threshold)
    ]
    per_subject = (
        tbl.groupby(["subject", "roi_base", "is_hp"], observed=True)["shift_from_anchor"]
        .mean()
        .reset_index()
    )
    pivot = per_subject.pivot_table(
        index=["subject", "roi_base"], columns="is_hp", values="shift_from_anchor"
    ).reset_index()
    pivot.columns.name = None
    pivot.rename(columns={True: "hp", False: "nonhp"}, inplace=True)
    pivot["diff"] = pivot["hp"] - pivot["nonhp"]

    rows = []
    for roi in rois:
        sub = pivot[pivot["roi_base"] == roi].dropna(subset=["hp", "nonhp"])
        if len(sub) < 3:
            rows.append({"roi": roi, "n": len(sub), "hp_mean": np.nan,
                         "nonhp_mean": np.nan, "diff_mean": np.nan,
                         "t": np.nan, "p": np.nan})
            continue
        t, p = stats.ttest_rel(sub["hp"], sub["nonhp"])
        rows.append({
            "roi": roi, "n": len(sub),
            "hp_mean": float(sub["hp"].mean()),
            "nonhp_mean": float(sub["nonhp"].mean()),
            "diff_mean": float(sub["diff"].mean()),
            "diff_sem": float(sub["diff"].sem()),
            "t": float(t), "p": float(p),
        })
    stats_df = pd.DataFrame(rows)
    valid = stats_df["p"].notna()
    if valid.any():
        _, p_fdr, _, _ = multipletests(stats_df.loc[valid, "p"], method="fdr_bh")
        stats_df.loc[valid, "p_fdr"] = p_fdr

    if fig is None:
        fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13, 5))
    else:
        ax_l, ax_r = fig.axes[:2]

    melt = per_subject.copy()
    melt["anchor_type"] = melt["is_hp"].map({True: "HP", False: "non-HP"})
    sns.pointplot(
        data=melt, x="roi_base", y="shift_from_anchor",
        hue="anchor_type", order=rois, hue_order=["HP", "non-HP"],
        palette={"HP": "#d62728", "non-HP": "#888888"},
        errorbar=("ci", 95), dodge=0.3, markers="o",
        linestyles=["-", "--"], ax=ax_l,
    )
    ax_l.axhline(0, color="k", lw=0.5, ls="--")
    ax_l.set_xlabel("ROI")
    ax_l.set_ylabel(f"Δ distance from anchor (deg)\n(voxels with mean PRF within {near_threshold}° of anchor)")
    ax_l.set_title("HP-anchored vs non-HP-anchored radial shift\n"
                   "(non-HP = three other ring quadrants; should be ~0 if effect is HP-specific)")
    ax_l.legend(title="Anchor type", loc="best")

    p_col = "p_fdr" if "p_fdr" in stats_df.columns else "p"
    ax_r.errorbar(
        x=np.arange(len(stats_df)), y=stats_df["diff_mean"],
        yerr=stats_df["diff_sem"], fmt="o", color="black", capsize=4,
    )
    ax_r.set_xticks(np.arange(len(stats_df)))
    ax_r.set_xticklabels(stats_df["roi"])
    ax_r.axhline(0, color="k", lw=0.5, ls="--")
    ax_r.set_xlabel("ROI")
    ax_r.set_ylabel("HP shift − non-HP shift (deg)")
    ax_r.set_title("Paired HP-vs-non-HP difference per ROI\n(per-subject paired t; FDR-BH)")
    ymax = (stats_df["diff_mean"] + stats_df["diff_sem"]).max()
    for i, (roi, p) in enumerate(zip(stats_df["roi"], stats_df[p_col])):
        if pd.isna(p):
            continue
        marker = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        ax_r.text(i, ymax * 1.15, marker, ha="center", va="bottom",
                  fontsize=11 if marker != "n.s." else 9,
                  color="black" if marker != "n.s." else "0.5")

    fig.tight_layout()
    fig._stats_df = stats_df
    return fig


def plot_hp_vs_nonhp_curve(
    pars: pd.DataFrame,
    rois: list[str] | None = None,
    bin_edges: np.ndarray | None = None,
    fig: plt.Figure | None = None,
):
    """Shift-vs-distance, separated by HP vs non-HP anchor.

    If the shift is a Jensen's-inequality artefact, HP and non-HP curves
    overlap. If it's real HP-specific suppression, the HP curve sits
    above the non-HP curve, especially at small distances.
    """
    import seaborn as sns

    if rois is None:
        rois = [r for r in roi_order if r in pars.index.get_level_values("roi_base").unique()]
    if bin_edges is None:
        bin_edges = np.arange(0.0, 7.5, 1.0)

    tbl = add_anchor_shifts(pars)
    tbl = tbl[tbl["roi_base"].isin(rois)].copy()
    tbl["d_bin"] = pd.cut(
        tbl["anchor_distance_mean"], bin_edges,
        labels=0.5 * (bin_edges[:-1] + bin_edges[1:]),
        include_lowest=True,
    )
    tbl = tbl.dropna(subset=["d_bin"])
    tbl["d_bin"] = tbl["d_bin"].astype(float)
    tbl["anchor_type"] = tbl["is_hp"].map({True: "HP", False: "non-HP"})

    per_subject = (
        tbl.groupby(["subject", "roi_base", "anchor_type", "d_bin"], observed=True)
        ["shift_from_anchor"].mean().reset_index()
    )

    g = sns.FacetGrid(
        per_subject, col="roi_base", col_wrap=4, col_order=rois,
        height=2.6, aspect=1.2, sharey=True,
    )
    g.map_dataframe(
        sns.lineplot, x="d_bin", y="shift_from_anchor",
        hue="anchor_type", hue_order=["HP", "non-HP"],
        palette={"HP": "#d62728", "non-HP": "#888888"},
        errorbar=("ci", 95),
    )
    for ax in g.axes.flat:
        ax.axhline(0, color="k", lw=0.5, ls="--")
    g.set_axis_labels("Distance of mean PRF from anchor (deg)",
                      "Δ distance from anchor (deg)")
    g.add_legend(title="Anchor type")
    g.fig.suptitle("HP vs non-HP shift as a function of distance from anchor"
                   "\n(non-HP = three other ring quadrants per condition)", y=1.03)
    g.fig.tight_layout()
    return g.fig


def plot_paired_vs_distance_combined(
    pars: pd.DataFrame,
    rois_combined: tuple[str, ...] = ("V3AB", "hV4"),
    bin_edges: np.ndarray | None = None,
    near_threshold: float = 1.5,
    far_threshold: float = 4.0,
    fig: plt.Figure | None = None,
):
    """Combined V3AB+hV4 plot of paired (HP − Opposite) metric vs distance.

    Adds explicit shading for the 'near-HP' (suppression-expected) and
    'far-from-HP' (attraction-hypothesis) bands, plus a one-sample t-test
    of mean paired metric in the far-from-HP band against zero.
    """
    import seaborn as sns
    from scipy import stats
    if bin_edges is None:
        bin_edges = np.arange(0.0, 7.5, 0.5)

    work = pars.reset_index()
    work = work[work["roi_base"].isin(rois_combined)].copy()
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    work["d_bin"] = pd.cut(
        work["distance_from_distractor_mean"], bin_edges,
        labels=centers, include_lowest=True,
    )
    work = work.dropna(subset=["d_bin"])
    work["d_bin"] = work["d_bin"].astype(float)

    per_subject = (
        work.groupby(["subject", "d_bin"], observed=True)
        ["paired_hp_minus_opposite"].mean().reset_index()
    )

    # Far-from-HP test: per-subject mean paired metric for voxels with
    # distance_from_distractor_mean > far_threshold; t-test vs 0.
    far_per_subject = (
        work[work["distance_from_distractor_mean"] > far_threshold]
        .groupby("subject", observed=True)["paired_hp_minus_opposite"]
        .mean().dropna()
    )
    near_per_subject = (
        work[work["distance_from_distractor_mean"] <= near_threshold]
        .groupby("subject", observed=True)["paired_hp_minus_opposite"]
        .mean().dropna()
    )
    t_far, p_far = stats.ttest_1samp(far_per_subject, 0.0)
    t_near, p_near = stats.ttest_1samp(near_per_subject, 0.0)

    if fig is None:
        fig, ax = plt.subplots(figsize=(9, 5))
    else:
        ax = fig.gca()

    # Shaded bands.
    ax.axvspan(0, near_threshold, color="#d62728", alpha=0.08,
               label=f"near (d < {near_threshold}°)")
    ax.axvspan(far_threshold, bin_edges[-1], color="#1f77b4", alpha=0.08,
               label=f"far (d > {far_threshold}°)")

    sns.lineplot(
        data=per_subject, x="d_bin", y="paired_hp_minus_opposite",
        errorbar=("ci", 95), color="#444444", alpha=0.4, ax=ax,
    )
    from statsmodels.nonparametric.smoothers_lowess import lowess
    sub = per_subject.dropna(subset=["d_bin", "paired_hp_minus_opposite"])
    if len(sub) >= 5:
        sm = lowess(sub["paired_hp_minus_opposite"].values,
                    sub["d_bin"].astype(float).values,
                    frac=0.5, return_sorted=True)
        ax.plot(sm[:, 0], sm[:, 1], color="#444444", lw=2.5, label="LOWESS")

    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.set_xlabel("Distance of mean PRF from HP (deg)")
    ax.set_ylabel("HP shift − Opposite shift (deg)\n(>0: suppression; <0: attraction)")

    far_marker = ("***" if p_far < 0.001 else "**" if p_far < 0.01
                  else "*" if p_far < 0.05 else "n.s.")
    near_marker = ("***" if p_near < 0.001 else "**" if p_near < 0.01
                   else "*" if p_near < 0.05 else "n.s.")
    ax.set_title(
        f"Paired metric vs distance from HP in {' + '.join(rois_combined)} combined\n"
        f"near (d < {near_threshold}°): mean = {near_per_subject.mean():+.3f}° "
        f"(t = {t_near:+.2f}, p = {p_near:.4f}, {near_marker} suppression)   ·   "
        f"far (d > {far_threshold}°): mean = {far_per_subject.mean():+.3f}° "
        f"(t = {t_far:+.2f}, p = {p_far:.4f}, {far_marker} attraction)"
    )
    ax.legend(loc="best")
    fig.tight_layout()
    fig._near_p = p_near
    fig._far_p = p_far
    fig._near_mean = float(near_per_subject.mean())
    fig._far_mean = float(far_per_subject.mean())
    return fig


def fit_af_models_per_subject(
    pars: pd.DataFrame,
    rois_combined: tuple[str, ...] = ("V3AB", "hV4"),
):
    """Fit both AF and AF+ per subject in V3AB+hV4 combined.

    Returns a dict with per-subject parameter arrays plus a summary DataFrame.
    """
    from retsupp.modeling.af_model import fit_per_subject

    af_results = fit_per_subject(pars, list(rois_combined), model="AF")
    afp_results = fit_per_subject(pars, list(rois_combined), model="AF+")

    rows = []
    af_mu, afp_mu = [], []
    af_sigma, afp_sigma = [], []
    afp_log_ba = []
    af_r2, afp_r2 = [], []
    af_aic, afp_aic = [], []
    subjects = []

    afp_by_subject = {sub: (fit, idx) for sub, fit, idx in afp_results}

    for sub, af_fit, _ in af_results:
        if sub not in afp_by_subject:
            continue
        afp_fit, _ = afp_by_subject[sub]
        subjects.append(sub)
        af_mu.append(af_fit.mu_AF)
        afp_mu.append(afp_fit.mu_AF)
        af_sigma.append(af_fit.sigma_AF)
        afp_sigma.append(afp_fit.sigma_AF)
        afp_log_ba.append(afp_fit.log_b_over_a)
        af_r2.append(af_fit.r2)
        afp_r2.append(afp_fit.r2)
        af_aic.append(af_fit.aic)
        afp_aic.append(afp_fit.aic)
        rows.append({
            "subject": sub,
            "AF_sigma_AF": af_fit.sigma_AF,
            "AF_R2": af_fit.r2,
            "AF_AIC": af_fit.aic,
            "AFplus_sigma_AF": afp_fit.sigma_AF,
            "AFplus_log_b_over_a": afp_fit.log_b_over_a,
            "AFplus_b_over_a": np.exp(afp_fit.log_b_over_a),
            "AFplus_R2": afp_fit.r2,
            "AFplus_AIC": afp_fit.aic,
            "AIC_AF_minus_AFplus": af_fit.aic - afp_fit.aic,
        })

    return {
        "summary": pd.DataFrame(rows),
        "subjects": subjects,
        "af_mu": np.stack(af_mu) if af_mu else np.empty((0, 4, 2)),
        "afp_mu": np.stack(afp_mu) if afp_mu else np.empty((0, 4, 2)),
        "af_sigma": np.array(af_sigma),
        "afp_sigma": np.array(afp_sigma),
        "afp_log_ba": np.array(afp_log_ba),
    }


def plot_af_fit_results(
    fit_dict: dict,
    rois_combined: tuple[str, ...] = ("V3AB", "hV4"),
    r2_threshold: float = 0.01,
    fig=None,
):
    """Visualize fit AF / AF+ parameters and model comparison.

    Subjects are split by within-voxel R²:
        R² > r2_threshold → "responder" (real conditionwise variation explained)
        R² ≤ r2_threshold → "non-responder" (model fits no AF effect)
    Only responders are shown in the AF-center scatter plots.
    """
    import seaborn as sns
    summary = fit_dict["summary"].copy()
    af_mu = fit_dict["af_mu"]      # (n_sub, 4, 2)
    afp_mu = fit_dict["afp_mu"]

    af_responder = summary["AF_R2"].values > r2_threshold
    afp_responder = summary["AFplus_R2"].values > r2_threshold
    n_total = len(summary)
    n_af_responders = int(af_responder.sum())
    n_afp_responders = int(afp_responder.sum())

    if fig is None:
        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(2, 3, height_ratios=[1.2, 1])
        ax_af_centers = fig.add_subplot(gs[0, 0])
        ax_afp_centers = fig.add_subplot(gs[0, 1])
        ax_legend = fig.add_subplot(gs[0, 2])
        ax_sigma = fig.add_subplot(gs[1, 0])
        ax_log_ba = fig.add_subplot(gs[1, 1])
        ax_r2 = fig.add_subplot(gs[1, 2])

    # Reference HP locations and their antipodes.
    from retsupp.utils.data import distractor_locations
    hp_order = ["upper right", "upper left", "lower left", "lower right"]
    cond_colors = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd"]
    hp_xy = np.array([list(distractor_locations[c]) for c in hp_order])
    opp_xy = -hp_xy

    for ax, mu_array, mask, model_label, n_resp in [
        (ax_af_centers, af_mu, af_responder, "AF", n_af_responders),
        (ax_afp_centers, afp_mu, afp_responder, "AF+", n_afp_responders),
    ]:
        theta = np.linspace(0, 2 * np.pi, 200)
        ax.plot(4 * np.cos(theta), 4 * np.sin(theta), "k--", lw=0.5, alpha=0.3)
        for i, c in enumerate(cond_colors):
            ax.scatter(*hp_xy[i], marker="*", s=250, color=c, edgecolor="k", zorder=4)
            ax.scatter(*opp_xy[i], marker="x", s=120, color=c, lw=2, zorder=4)
        if mask.sum() > 0:
            mu_resp = mu_array[mask]
            for c in range(4):
                ax.scatter(mu_resp[:, c, 0], mu_resp[:, c, 1],
                           s=22, color=cond_colors[c], alpha=0.7,
                           edgecolor="white", lw=0.4)
                mean_xy = mu_resp[:, c, :].mean(axis=0)
                ax.scatter(*mean_xy, marker="D", s=120, color=cond_colors[c],
                           edgecolor="black", lw=1.5, zorder=5)

        ax.axhline(0, color="k", lw=0.3, alpha=0.3)
        ax.axvline(0, color="k", lw=0.3, alpha=0.3)
        ax.set_aspect("equal")
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_xlabel("x (deg)")
        ax.set_ylabel("y (deg)")
        ax.set_title(f"{model_label} fit AF centers — responders only "
                     f"({n_resp}/{n_total})\n"
                     f"★ = HP ;  × = opposite of HP ;  ◇ = group mean")

    ax_legend.axis("off")
    ax_legend.text(0.0, 0.97,
                   f"Model comparison (V3AB + hV4 combined, n = {n_total} subjects)",
                   transform=ax_legend.transAxes, va="top", fontsize=11, weight="bold")
    afp_better = (summary["AIC_AF_minus_AFplus"] > 0).sum()
    summary_text = (
        f"Within-voxel R² (cond variation only)\n"
        f"  AF responders   (R² > {r2_threshold}): {n_af_responders}/{n_total}\n"
        f"  AF+ responders  (R² > {r2_threshold}): {n_afp_responders}/{n_total}\n"
        f"\n"
        f"AF responder σ_AF:   "
        f"{summary.loc[af_responder, 'AF_sigma_AF'].median():.2f}° (median)\n"
        f"AF+ responder σ_AF:  "
        f"{summary.loc[afp_responder, 'AFplus_sigma_AF'].median():.2f}° (median)\n"
        f"\n"
        f"AF+ log(b/a) responder: "
        f"{summary.loc[afp_responder, 'AFplus_log_b_over_a'].median():+.2f}\n"
        f"  (smaller → closer to simple AF;\n"
        f"   larger → more local effect)\n"
        f"\n"
        f"AIC: AF+ wins in {afp_better}/{n_total} subjects.\n"
        f"  Δ_AIC (AF − AF+) responders only:\n"
        f"  mean = {summary.loc[afp_responder | af_responder, 'AIC_AF_minus_AFplus'].mean():+.2f}"
    )
    ax_legend.text(0.0, 0.88, summary_text, transform=ax_legend.transAxes,
                   va="top", fontsize=9, family="monospace")

    if af_responder.sum() > 0:
        sns.histplot(summary.loc[af_responder, "AF_sigma_AF"], ax=ax_sigma, bins=15,
                     color="#444444", alpha=0.5, label="AF (responders)")
    if afp_responder.sum() > 0:
        sns.histplot(summary.loc[afp_responder, "AFplus_sigma_AF"], ax=ax_sigma, bins=15,
                     color="#1f77b4", alpha=0.5, label="AF+ (responders)")
    ax_sigma.set_xlabel("σ_AF (deg)")
    ax_sigma.set_title(f"σ_AF in responders ({' + '.join(rois_combined)})")
    ax_sigma.legend()

    if afp_responder.sum() > 0:
        sns.histplot(summary.loc[afp_responder, "AFplus_log_b_over_a"],
                     ax=ax_log_ba, bins=15, color="#1f77b4")
    ax_log_ba.axvline(0, color="k", ls="--", lw=0.5)
    ax_log_ba.set_xlabel("log(b/a)  (AF+ responders only)")
    ax_log_ba.set_title("Offset ratio per subject\nb/a > 1 → more local effect")

    ax_r2.scatter(summary["AF_R2"], summary["AFplus_R2"], alpha=0.7)
    lo = min(summary["AF_R2"].min(), summary["AFplus_R2"].min(), 0)
    hi = max(summary["AF_R2"].max(), summary["AFplus_R2"].max(), 0.05)
    ax_r2.plot([lo, hi], [lo, hi], "k--", lw=0.5)
    ax_r2.axvline(r2_threshold, color="r", ls=":", lw=0.6)
    ax_r2.axhline(r2_threshold, color="r", ls=":", lw=0.6)
    ax_r2.set_xlabel("Within-voxel R² (AF)")
    ax_r2.set_ylabel("Within-voxel R² (AF+)")
    ax_r2.set_title(f"Per-subject R²; red lines mark responder cutoff ({r2_threshold})")

    fig.tight_layout()
    return fig


def plot_af_rotated_canonical(
    rotated_fits: dict,
    fig=None,
    show_saturated=True,
    title_suffix="",
):
    """Visualize per-subject μ_AF_canonical in HP-aligned frame."""
    from scipy import stats
    rois = list(rotated_fits.keys())
    n = len(rois)
    if fig is None:
        fig, axes = plt.subplots(1, n, figsize=(5.4 * n, 5.5), squeeze=False)
    axes = axes.ravel() if hasattr(axes, "ravel") else np.array(axes).ravel()

    for ax, roi in zip(axes, rois):
        df = rotated_fits[roi]
        # Distractor ring.
        theta = np.linspace(0, 2 * np.pi, 200)
        ax.plot(4 * np.cos(theta), 4 * np.sin(theta), "k--", lw=0.5, alpha=0.3)
        # HP and antipode markers.
        ax.scatter(0, 4, marker="*", s=300, color="#d62728", edgecolor="k", zorder=5)
        ax.scatter(0, -4, marker="x", s=180, color="#1f77b4", lw=3, zorder=5)
        ax.text(0, 4.5, "HP", color="#d62728", ha="center", fontsize=11, weight="bold")
        ax.text(0, -4.6, "−HP (suppression target)", color="#1f77b4",
                ha="center", fontsize=10)

        # Distinguish saturated subjects.
        sat_mask = df["saturated"].values if "saturated" in df.columns else np.zeros(len(df), bool)
        plot_df = df.loc[~sat_mask] if not show_saturated else df

        # σ_AF circles per subject (translucent).
        for _, row in plot_df.iterrows():
            sat = row.get("saturated", False)
            circle = plt.Circle(
                (row["mu_AF_x"], row["mu_AF_y"]),
                row["sigma_AF"],
                color="0.5" if not sat else "#aa4444",
                fill=True, alpha=0.05 if not sat else 0.03, lw=0,
            )
            ax.add_patch(circle)

        if len(plot_df) > 0:
            non_sat = plot_df.loc[~plot_df.get("saturated", pd.Series([False] * len(plot_df))).values]
            if len(non_sat) > 0:
                sc = ax.scatter(
                    non_sat["mu_AF_x"], non_sat["mu_AF_y"],
                    c=non_sat["R2"], cmap="viridis", s=55,
                    edgecolor="white", lw=0.5, zorder=4,
                    vmin=0, vmax=max(0.05, non_sat["R2"].max()),
                )
                cbar = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
                cbar.set_label("Within-voxel R²")
            sat = plot_df.loc[plot_df.get("saturated", pd.Series([False] * len(plot_df))).values]
            if len(sat) > 0:
                ax.scatter(sat["mu_AF_x"], sat["mu_AF_y"],
                           marker="x", s=40, color="#aa4444",
                           lw=1.5, alpha=0.6, zorder=4)

            # Group mean AF center (over non-saturated only).
            ref = non_sat if len(non_sat) > 0 else plot_df
            mean_x, mean_y = ref["mu_AF_x"].mean(), ref["mu_AF_y"].mean()
            sem_x, sem_y = ref["mu_AF_x"].sem(), ref["mu_AF_y"].sem()
            ax.errorbar(mean_x, mean_y, xerr=sem_x, yerr=sem_y,
                        fmt="D", color="black", markersize=11,
                        ecolor="black", elinewidth=1.5,
                        markeredgecolor="white", markeredgewidth=2, zorder=6)

        ax.axhline(0, color="k", lw=0.3, alpha=0.4)
        ax.axvline(0, color="k", lw=0.3, alpha=0.4)
        ax.set_aspect("equal")
        lim = 8
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel("x in canonical frame (deg)")
        ax.set_ylabel("y in canonical frame (deg)")

        # One-sample t-test of mu_AF_y against 0 (over non-saturated).
        ref_y = (df.loc[~sat_mask, "mu_AF_y"]
                 if not show_saturated or sat_mask.sum() > 0 else df["mu_AF_y"])
        if len(ref_y) >= 3:
            t, p = stats.ttest_1samp(ref_y, 0.0)
            t_str = f"  t(y)={t:+.2f}, p={p:.3f}"
        else:
            t_str = ""

        n_neg = int((df["mu_AF_y"] < 0).sum())
        n_sat = int(sat_mask.sum())
        title = (
            f"{roi}{title_suffix} — single AF center per subject\n"
            f"Group mean: ({mean_x:+.2f}, {mean_y:+.2f})°  "
            f"({n_neg}/{len(df)} subj y<0; {n_sat} saturated){t_str}"
        )
        ax.set_title(title, fontsize=10)

    fig.suptitle(
        f"Rotated-frame AF fit{title_suffix} — HP at (0, +4°), antipode at (0, −4°)\n"
        "Suppression: μ_AF should land in lower half (y < 0).  Red × = σ_AF saturated at bound.",
        y=1.02,
    )
    fig.tight_layout()
    return fig


def plot_af_condition_deviations(
    fit_dict: dict,
    rois_combined: tuple[str, ...] = ("V3AB", "hV4"),
    r2_threshold: float = 0.01,
    fig=None,
):
    """Per-condition AF deviation from the across-condition per-subject mean.

    Subtracting out the per-subject mean AF removes any common attentional
    state (e.g. a default rightward bias) and isolates condition-specific
    AF structure. If the deviations point *toward* the HP for that
    condition, that's attraction; *away* from HP is suppression-like;
    null clusters at origin if there's no condition signal.

    Two panels (AF and AF+), responders only.
    """
    summary = fit_dict["summary"]
    af_mu = fit_dict["af_mu"]      # (n_sub, 4, 2)
    afp_mu = fit_dict["afp_mu"]
    af_responder = summary["AF_R2"].values > r2_threshold
    afp_responder = summary["AFplus_R2"].values > r2_threshold

    af_mean = af_mu.mean(axis=1, keepdims=True)
    afp_mean = afp_mu.mean(axis=1, keepdims=True)
    af_dev = af_mu - af_mean
    afp_dev = afp_mu - afp_mean

    if fig is None:
        fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14, 6))
    else:
        ax_l, ax_r = fig.axes[:2]

    from retsupp.utils.data import distractor_locations
    hp_order = ["upper right", "upper left", "lower left", "lower right"]
    cond_colors = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd"]
    hp_xy = np.array([list(distractor_locations[c]) for c in hp_order])

    extent = 1.5  # tight around origin since deviations are small
    for ax, dev, mask, label in [
        (ax_l, af_dev, af_responder, "AF"),
        (ax_r, afp_dev, afp_responder, "AF+"),
    ]:
        # Reference: small markers at HP and -HP directions, scaled down
        # to fit in the deviation plot (just to show direction).
        ref_scale = 0.8
        for c in range(4):
            ax.scatter(*(hp_xy[c] * ref_scale / np.linalg.norm(hp_xy[c]) * extent * 0.85),
                       marker="*", s=200, color=cond_colors[c], alpha=0.25,
                       edgecolor="k", zorder=2)

        # Per-subject deviations.
        if mask.sum() > 0:
            dev_resp = dev[mask]
            for c in range(4):
                ax.scatter(dev_resp[:, c, 0], dev_resp[:, c, 1],
                           s=25, color=cond_colors[c], alpha=0.55,
                           edgecolor="white", lw=0.4)
                # Group mean.
                mean_dev = dev_resp[:, c, :].mean(axis=0)
                sem_dev = dev_resp[:, c, :].std(axis=0, ddof=1) / np.sqrt(len(dev_resp))
                ax.errorbar(mean_dev[0], mean_dev[1],
                            xerr=sem_dev[0], yerr=sem_dev[1],
                            fmt="D", color=cond_colors[c], markersize=11,
                            ecolor="k", elinewidth=1.5,
                            markeredgecolor="black", markeredgewidth=1.5, zorder=5,
                            label=f"{hp_order[c]}")
                # Arrow from origin to mean (for clarity).
                ax.annotate("", xy=mean_dev, xytext=(0, 0),
                            arrowprops=dict(arrowstyle="->", color=cond_colors[c],
                                            lw=2, alpha=0.7))

        ax.axhline(0, color="k", lw=0.4, ls="--", alpha=0.5)
        ax.axvline(0, color="k", lw=0.4, ls="--", alpha=0.5)
        ax.set_aspect("equal")
        ax.set_xlim(-extent, extent)
        ax.set_ylim(-extent, extent)
        ax.set_xlabel("x deviation (deg)")
        ax.set_ylabel("y deviation (deg)")
        ax.set_title(
            f"{label} — condition-specific AF deviation\n"
            f"(faded ★ = direction of HP per condition; arrow = group-mean dev ± SEM)"
        )
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8)

    fig.suptitle(
        f"Per-condition AF center deviation from per-subject across-condition mean"
        f"\nResponders only (within-voxel R² > {r2_threshold})",
        y=1.02,
    )
    fig.tight_layout()
    return fig


def plot_size_and_amplitude_vs_distance(
    pars: pd.DataFrame,
    rois: list[str] | None = None,
    bin_edges: np.ndarray | None = None,
    near_threshold: float = 1.5,
    far_threshold: float = 4.0,
    fig: plt.Figure | None = None,
):
    """Per-ROI plots of conditionwise sd_diff and amplitude_diff vs distance.

    Tests AF model predictions:
        - Simple AF predicts NO per-condition size change (sd_diff = 0 across
          all distances). A non-flat sd-vs-distance pattern argues against
          simple AF and toward AF+.
        - AF (suppression interpretation, AF at -HP) predicts amplitude
          REDUCTION (amplitude_diff < 0) for voxels near HP, because the
          overlap between SD-pRF and AF is smallest there.

    Per-ROI t-tests: mean sd_diff and amplitude_diff in voxels with
    distance_from_distractor_mean ≤ near_threshold, against zero.
    """
    import seaborn as sns
    from scipy import stats

    if rois is None:
        rois = [r for r in roi_order if r in pars.index.get_level_values("roi_base").unique()]
    if bin_edges is None:
        bin_edges = np.arange(0.0, 7.5, 1.0)

    work = pars.reset_index()
    work = work[work["roi_base"].isin(rois)].copy()
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    work["d_bin"] = pd.cut(
        work["distance_from_distractor_mean"], bin_edges,
        labels=centers, include_lowest=True,
    )
    work = work.dropna(subset=["d_bin"])
    work["d_bin"] = work["d_bin"].astype(float)

    # Two facet grids stacked.
    if fig is None:
        fig, axes = plt.subplots(2, len(rois), figsize=(2.6 * len(rois), 6),
                                 sharex=True)
    axes = np.atleast_2d(axes)

    for col_metric, metric_name, row_axes, ylabel in [
        ("sd_diff", "sd_diff (deg)", axes[0, :],
         "sd_diff (deg)\nAF predicts: 0 across d"),
        ("amplitude_diff", "amplitude_diff (a.u.)", axes[1, :],
         "amplitude_diff (a.u.)\nAF predicts: <0 near HP"),
    ]:
        per_subject = (
            work.groupby(["subject", "roi_base", "d_bin"], observed=True)
            [col_metric].mean().reset_index()
        )
        for ax, roi in zip(row_axes, rois):
            sub = per_subject[per_subject["roi_base"] == roi]
            sns.lineplot(data=sub, x="d_bin", y=col_metric,
                         errorbar=("ci", 95), color="#444444", ax=ax)
            ax.axhline(0, color="k", lw=0.5, ls="--")
            ax.set_title(roi, fontsize=10)
            ax.set_xlabel("d from HP (deg)" if row_axes is axes[1, :] else "")
            ax.set_ylabel(ylabel if ax is row_axes[0] else "")

    # Stats summary printed.
    rows_summary = []
    for roi in rois:
        roi_df = work[work["roi_base"] == roi]
        near = roi_df[roi_df["distance_from_distractor_mean"] <= near_threshold]
        for col, label in [("sd_diff", "sd"), ("amplitude_diff", "amp")]:
            per_sub = (near.groupby("subject", observed=True)[col].mean().dropna())
            if len(per_sub) >= 3:
                t, p = stats.ttest_1samp(per_sub, 0.0)
            else:
                t, p = np.nan, np.nan
            rows_summary.append({
                "roi": roi, "metric": label,
                "near_mean": float(per_sub.mean()) if len(per_sub) else np.nan,
                "near_sem": float(per_sub.sem()) if len(per_sub) else np.nan,
                "t": float(t) if not np.isnan(t) else np.nan,
                "p": float(p) if not np.isnan(p) else np.nan,
                "n_subj": int(len(per_sub)),
            })
    fig._summary = pd.DataFrame(rows_summary)
    fig.suptitle(
        "Conditionwise sd and amplitude vs distance from HP, per ROI"
        "\nTop: AF predicts flat at 0;  Bottom: AF (suppression) predicts < 0 near HP",
        y=1.03,
    )
    fig.tight_layout()
    return fig


def r2_threshold_sweep(
    pars_loose: pd.DataFrame,
    thresholds: list[float] | None = None,
    near_threshold: float = 1.5,
    far_threshold: float = 4.0,
    rois_combined: tuple[str, ...] = ("V3AB", "hV4"),
) -> pd.DataFrame:
    """Sweep r2_mean_model threshold and recompute key stats.

    Tests robustness of the HP-shift / paired-metric / far-attraction
    conclusions to the choice of inclusion R² cutoff. `pars_loose` should
    be loaded with the *lowest* R² threshold of interest (e.g. 0.05);
    higher thresholds are applied in memory.

    Returns a long-format DataFrame: r2_thr × roi-group × metric.
    """
    from scipy import stats

    if thresholds is None:
        thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]

    work = pars_loose.reset_index().copy()

    groups = {}
    for roi in ["V3AB", "hV4"]:
        groups[roi] = work["roi_base"] == roi
    groups[" + ".join(rois_combined)] = work["roi_base"].isin(rois_combined)

    rows = []
    for thr in thresholds:
        thr_mask = work["r2_mean_model"] > thr
        for roi_label, roi_mask in groups.items():
            df = work[thr_mask & roi_mask]
            if df.empty:
                continue
            near = df[df["distance_from_distractor_mean"] <= near_threshold]
            far = df[df["distance_from_distractor_mean"] > far_threshold]

            hp_near_sub = (
                near.groupby("subject", observed=True)["shift_from_distractor"]
                .mean().dropna()
            )
            paired_near_sub = (
                near.groupby("subject", observed=True)["paired_hp_minus_opposite"]
                .mean().dropna()
            )
            paired_far_sub = (
                far.groupby("subject", observed=True)["paired_hp_minus_opposite"]
                .mean().dropna()
            )

            def _t1(vals):
                if len(vals) < 3:
                    return np.nan, np.nan
                t, p = stats.ttest_1samp(vals, 0.0)
                return float(t), float(p)

            t_hp, p_hp = _t1(hp_near_sub)
            t_pn, p_pn = _t1(paired_near_sub)
            t_pf, p_pf = _t1(paired_far_sub)

            rows.append({
                "r2_thr": thr,
                "roi": roi_label,
                "n_voxels": len(df),
                "n_near": len(near),
                "n_far": len(far),
                "n_subj": int(df["subject"].nunique()),
                "hp_near_mean": float(hp_near_sub.mean()) if len(hp_near_sub) else np.nan,
                "hp_near_p": p_hp,
                "paired_near_mean": float(paired_near_sub.mean()) if len(paired_near_sub) else np.nan,
                "paired_near_p": p_pn,
                "paired_far_mean": float(paired_far_sub.mean()) if len(paired_far_sub) else np.nan,
                "paired_far_p": p_pf,
            })
    return pd.DataFrame(rows)


def plot_r2_sweep(
    sweep: pd.DataFrame,
    default_thr: float = 0.20,
    fig: plt.Figure | None = None,
):
    """Visualize the R² sweep: metric values + p-values across thresholds."""
    import seaborn as sns

    if fig is None:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    else:
        axes = np.array(fig.axes).reshape(2, 2)

    palette = {"V3AB": "#1f77b4", "hV4": "#d62728", "V3AB + hV4": "#444444"}

    metrics = [
        ("hp_near_mean", "HP shift, near (d ≤ 1.5°), deg", axes[0, 0]),
        ("paired_near_mean", "Paired HP − Opp, near (d ≤ 1.5°), deg", axes[0, 1]),
        ("paired_far_mean", "Paired HP − Opp, far (d > 4°), deg", axes[1, 0]),
        ("n_voxels", "Total voxel-condition rows", axes[1, 1]),
    ]
    for col, label, ax in metrics:
        for roi, sub in sweep.groupby("roi"):
            ax.plot(sub["r2_thr"], sub[col], "o-", label=roi,
                    color=palette.get(roi, "gray"))
        ax.axvline(default_thr, color="k", lw=0.5, ls=":")
        ax.set_ylabel(label)
        if col != "n_voxels":
            ax.axhline(0, color="k", lw=0.5, ls="--")
        if col == "n_voxels":
            ax.set_yscale("log")

    for ax in axes[1, :]:
        ax.set_xlabel("r2_mean_model threshold")
    axes[0, 0].legend(title="ROI", loc="best")

    fig.suptitle(
        f"Robustness to R² threshold (default = {default_thr})\n"
        "If V3AB/hV4 effects shift sign or vanish across thresholds → fragile",
        y=1.02,
    )
    fig.tight_layout()
    return fig


def plot_paired_vs_distance_per_roi(
    pars: pd.DataFrame,
    rois: list[str] | None = None,
    bin_edges: np.ndarray | None = None,
    near_threshold: float = 1.5,
    far_threshold: float = 4.0,
    fig: plt.Figure | None = None,
):
    """One panel per ROI of paired (HP − Opposite) metric vs distance,
    with explicit shading for near/far bands and per-ROI t-test stats
    in each panel title. Same formula as the V3AB+hV4 combined plot.
    """
    import seaborn as sns
    from scipy import stats
    from statsmodels.nonparametric.smoothers_lowess import lowess

    if rois is None:
        rois = [r for r in roi_order if r in pars.index.get_level_values("roi_base").unique()]
    if bin_edges is None:
        bin_edges = np.arange(0.0, 7.5, 0.5)

    n = len(rois)
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))
    if fig is None:
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows + 1),
                                 sharex=True, sharey=True)
    axes = np.atleast_2d(axes).ravel()

    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    rows_summary = []
    for ax, roi in zip(axes, rois):
        roi_df = pars.reset_index()
        roi_df = roi_df[roi_df["roi_base"] == roi].copy()
        roi_df["d_bin"] = pd.cut(
            roi_df["distance_from_distractor_mean"], bin_edges,
            labels=centers, include_lowest=True,
        )
        roi_df = roi_df.dropna(subset=["d_bin"])
        roi_df["d_bin"] = roi_df["d_bin"].astype(float)

        per_subject = (
            roi_df.groupby(["subject", "d_bin"], observed=True)
            ["paired_hp_minus_opposite"].mean().reset_index()
        )

        near = (
            roi_df[roi_df["distance_from_distractor_mean"] <= near_threshold]
            .groupby("subject", observed=True)["paired_hp_minus_opposite"]
            .mean().dropna()
        )
        far = (
            roi_df[roi_df["distance_from_distractor_mean"] > far_threshold]
            .groupby("subject", observed=True)["paired_hp_minus_opposite"]
            .mean().dropna()
        )
        if len(near) >= 3:
            t_near, p_near = stats.ttest_1samp(near, 0.0)
        else:
            t_near, p_near = np.nan, np.nan
        if len(far) >= 3:
            t_far, p_far = stats.ttest_1samp(far, 0.0)
        else:
            t_far, p_far = np.nan, np.nan
        rows_summary.append({
            "roi": roi,
            "near_mean": float(near.mean()) if len(near) > 0 else np.nan,
            "near_p": float(p_near) if not np.isnan(p_near) else np.nan,
            "far_mean": float(far.mean()) if len(far) > 0 else np.nan,
            "far_p": float(p_far) if not np.isnan(p_far) else np.nan,
            "n_subj": len(per_subject["subject"].unique()),
        })

        ax.axvspan(0, near_threshold, color="#d62728", alpha=0.08)
        ax.axvspan(far_threshold, bin_edges[-1], color="#1f77b4", alpha=0.08)
        sns.lineplot(data=per_subject, x="d_bin", y="paired_hp_minus_opposite",
                     errorbar=("ci", 95), color="#444444", alpha=0.4, ax=ax)
        sub = per_subject.dropna(subset=["d_bin", "paired_hp_minus_opposite"])
        if len(sub) >= 5:
            sm = lowess(sub["paired_hp_minus_opposite"].values,
                        sub["d_bin"].astype(float).values,
                        frac=0.5, return_sorted=True)
            ax.plot(sm[:, 0], sm[:, 1], color="#444444", lw=2.0)

        ax.axhline(0, color="k", lw=0.5, ls="--")
        ax.set_title(
            f"{roi}\n"
            f"near: {near.mean():+.3f}° (p={p_near:.3f})  ·  "
            f"far: {far.mean():+.3f}° (p={p_far:.3f})",
            fontsize=10,
        )

    for ax in axes[n:]:
        ax.set_visible(False)
    for ax in axes[:n]:
        ax.set_xlabel("Distance of mean PRF from HP (deg)")
        ax.set_ylabel("HP − Opp shift (deg)")

    fig.suptitle(
        "Paired metric vs distance from HP, per ROI"
        f"\nRed band = near (d < {near_threshold}°);  blue band = far (d > {far_threshold}°);"
        " p-values from one-sample t against zero",
        y=1.02,
    )
    fig.tight_layout()
    fig._summary = pd.DataFrame(rows_summary)
    return fig


def plot_paired_vs_distance(
    pars: pd.DataFrame,
    rois: list[str] | None = None,
    bin_edges: np.ndarray | None = None,
    fig: plt.Figure | None = None,
):
    """Paired (HP − Opposite) shift vs distance from HP.

    The paired metric centers at zero under any isotropic-noise / Jensen-
    bias null. It's positive when voxels are pushed away from HP
    (suppression), negative when voxels are pulled toward HP (attraction).

    Plotted vs distance from HP (positive only): a sign flip from positive
    near HP to negative far from HP would indicate local suppression
    coexisting with global attraction.
    """
    import seaborn as sns

    if rois is None:
        rois = [r for r in roi_order if r in pars.index.get_level_values("roi_base").unique()]
    if bin_edges is None:
        bin_edges = np.arange(0.0, 7.5, 1.0)

    work = pars.reset_index()
    work = work[work["roi_base"].isin(rois)].copy()
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    work["d_bin"] = pd.cut(
        work["distance_from_distractor_mean"], bin_edges,
        labels=centers, include_lowest=True,
    )
    work = work.dropna(subset=["d_bin"])
    work["d_bin"] = work["d_bin"].astype(float)

    per_subject = (
        work.groupby(["subject", "roi_base", "d_bin"], observed=True)
        ["paired_hp_minus_opposite"].mean().reset_index()
    )

    g = sns.FacetGrid(
        per_subject, col="roi_base", col_wrap=4, col_order=rois,
        height=2.6, aspect=1.2, sharey=True,
    )
    g.map_dataframe(
        sns.lineplot, x="d_bin", y="paired_hp_minus_opposite",
        errorbar=("ci", 95), color="#444444",
    )
    for ax in g.axes.flat:
        ax.axhline(0, color="k", lw=0.5, ls="--")
    g.set_axis_labels("Distance of mean PRF from HP (deg)",
                      "HP shift − Opposite shift (deg)\n"
                      "(>0: suppression; <0: attraction)")
    g.fig.suptitle("Paired HP-vs-Opposite metric as a function of distance from HP"
                   "\nbias-free; sign-flip at large d would mean local suppression + global attraction",
                   y=1.04)
    g.fig.tight_layout()
    return g.fig


def plot_axis_shift_on_axis_per_roi(
    pars: pd.DataFrame,
    rois: list[str] | None = None,
    x_threshold: float = 1.5,
    y_range: float = 3.0,
    bin_edges: np.ndarray | None = None,
    y_lim: float = 0.4,
    fig: plt.Figure | None = None,
):
    """Per-ROI version of plot_axis_shift_on_axis.

    For each ROI, plots dy_rotated vs y_mean_rotated for voxels close
    to the HP-opposite line (|x_mean_rotated| < x_threshold). This is a
    position-based (rather than distance-based) view of the suppression
    pattern — preserves the sign of the shift in the HP→opposite direction.
    """
    import seaborn as sns
    if rois is None:
        rois = [r for r in roi_order if r in pars.index.get_level_values("roi_base").unique()]
    if bin_edges is None:
        bin_edges = np.arange(-y_range, y_range + 0.5, 0.5)

    work = pars.reset_index()
    work = work[
        (work["roi_base"].isin(rois))
        & (work["x_mean_rotated"].abs() < x_threshold)
        & (work["y_mean_rotated"].abs() < y_range)
    ].copy()
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    work["y_bin"] = pd.cut(
        work["y_mean_rotated"], bin_edges,
        labels=centers, include_lowest=True,
    )
    work = work.dropna(subset=["y_bin"])
    work["y_bin"] = work["y_bin"].astype(float)

    per_subject = (
        work.groupby(["subject", "roi_base", "y_bin"], observed=True)["dy_rotated"]
        .mean().reset_index()
    )

    g = sns.FacetGrid(
        per_subject, col="roi_base", col_wrap=4, col_order=rois,
        height=2.6, aspect=1.2, sharey=True,
        ylim=(-y_lim, y_lim), xlim=(-y_range - 0.2, y_range + 0.2),
    )
    g.map_dataframe(
        sns.lineplot, x="y_bin", y="dy_rotated",
        errorbar=("ci", 95), color="#1f77b4",
    )
    for ax in g.axes.flat:
        ax.axhline(0, color="k", lw=0.5, ls="--")
        ax.axvline(0, color="k", lw=0.3, alpha=0.4)
        ax.text(y_range, y_lim * 0.85, "← HP", color="red", ha="right", va="top", fontsize=8)
        ax.text(-y_range, y_lim * 0.85, "opp →", color="blue", ha="left", va="top", fontsize=8)
    g.set_axis_labels(
        "y_mean_rotated (deg)\n(signed position along HP-opposite axis)",
        "dy_rotated (deg)\n(<0: toward opposite; >0: toward HP)",
    )
    g.fig.suptitle(
        f"Position-based shift along HP-opposite axis per ROI"
        f"\n(near-axis voxels: |x_mean_rotated| < {x_threshold}°, |y_mean_rotated| < {y_range}°)"
        f"\nNegative across whole range = pure suppression  ·  "
        "sign flip across x = mixed local supp + global attr",
        y=1.06,
    )
    g.fig.tight_layout()
    return g.fig


def plot_axis_shift_on_axis(
    pars: pd.DataFrame,
    rois_combined: tuple[str, ...] = ("V3AB", "hV4"),
    x_threshold: float = 1.5,
    y_range: float = 3.0,
    bin_edges: np.ndarray | None = None,
    y_lim: float = 0.4,
    fig: plt.Figure | None = None,
):
    """dy_rotated vs y_mean_rotated, restricted to near-axis voxels.

    Restricting to |x_mean_rotated| < `x_threshold` keeps voxels close
    to the HP-opposite line; restricting to |y_mean_rotated| < `y_range`
    drops the very sparse extreme bins (where only a handful of voxels
    sit and CIs are uninterpretable). Explicit `y_lim` keeps the y-axis
    focused on the central pattern.

    dy_rotated < 0 = pushed toward opposite (away from HP);
    dy_rotated > 0 = pulled toward HP.
    """
    import seaborn as sns
    if bin_edges is None:
        bin_edges = np.arange(-y_range, y_range + 0.5, 0.5)

    work = pars.reset_index()
    work = work[
        (work["roi_base"].isin(rois_combined))
        & (work["x_mean_rotated"].abs() < x_threshold)
        & (work["y_mean_rotated"].abs() < y_range)
    ].copy()
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    work["y_bin"] = pd.cut(
        work["y_mean_rotated"], bin_edges,
        labels=centers, include_lowest=True,
    )
    work = work.dropna(subset=["y_bin"])
    work["y_bin"] = work["y_bin"].astype(float)

    per_subject = (
        work.groupby(["subject", "y_bin"], observed=True)["dy_rotated"]
        .mean().reset_index()
    )

    if fig is None:
        fig, ax = plt.subplots(figsize=(9, 5))
    else:
        ax = fig.gca()

    sns.lineplot(
        data=per_subject, x="y_bin", y="dy_rotated",
        errorbar=("ci", 95), color="#1f77b4", ax=ax,
    )
    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.axvline(0, color="k", lw=0.3, alpha=0.4)
    ax.set_xlim(-y_range - 0.5, y_range + 0.5)
    ax.set_ylim(-y_lim, y_lim)
    ax.text(y_range, y_lim * 0.85, " ← toward HP", color="red",
            ha="right", va="top", fontsize=10)
    ax.text(-y_range, y_lim * 0.85, "toward opp →  ", color="blue",
            ha="left", va="top", fontsize=10)

    n_voxels = len(work)
    ax.set_xlabel("y_mean_rotated (deg)\nsigned position along HP-opposite axis (HP at +4°, opp at −4°)")
    ax.set_ylabel("dy_rotated (deg)\n(<0: shifted toward opposite; >0: shifted toward HP)")
    ax.set_title(
        f"Axial shift along HP-opposite axis in {' + '.join(rois_combined)}\n"
        f"(near-axis voxels: |x_mean_rotated| < {x_threshold}°, |y_mean_rotated| < {y_range}°; "
        f"n = {n_voxels:,} rows)\n"
        "Pure suppression: y < 0 everywhere   ·   "
        "Mixed local-supp + global-attr: sign flip across x"
    )
    fig.tight_layout()
    return fig


def plot_shift_along_hp_axis(
    pars: pd.DataFrame,
    rois: list[str] | None = None,
    bin_edges: np.ndarray | None = None,
    smooth: bool = True,
    y_range: float = 3.0,
    y_lim: float = 0.5,
    fig: plt.Figure | None = None,
):
    """Shift along the HP-opposite axis vs signed position along that axis.

    In the rotated frame, HP is at y=+4°, opposite at y=−4°. The vertical
    axis through the visual-field center connects them.

    For each voxel × condition row:
        x = y_mean_rotated   (signed position; +4=HP side, −4=opposite side)
        y = dy_rotated       (signed shift along axis; <0 = toward opposite,
                              >0 = toward HP)

    Predictions:
        Pure local suppression  → y < 0 across the whole x range
        Pure attraction (Klein) → y > 0 across the whole x range
        Local suppression +
            global attraction   → y < 0 near HP, y > 0 near opposite
                                  (i.e. negative slope across x)

    Faceted per ROI, with a combined V3AB+hV4 panel highlighted.
    """
    import seaborn as sns

    if rois is None:
        rois = [r for r in roi_order if r in pars.index.get_level_values("roi_base").unique()]
    if bin_edges is None:
        bin_edges = np.arange(-y_range, y_range + 0.5, 0.5)

    work = pars.reset_index()
    work = work[
        (work["roi_base"].isin(rois))
        & (work["y_mean_rotated"].abs() < y_range)
    ].copy()
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    work["y_bin"] = pd.cut(
        work["y_mean_rotated"], bin_edges,
        labels=centers, include_lowest=True,
    )
    work = work.dropna(subset=["y_bin"])
    work["y_bin"] = work["y_bin"].astype(float)

    per_subject = (
        work.groupby(["subject", "roi_base", "y_bin"], observed=True)["dy_rotated"]
        .mean().reset_index()
    )

    g = sns.FacetGrid(
        per_subject, col="roi_base", col_wrap=4, col_order=rois,
        height=2.6, aspect=1.2, sharey=True,
        ylim=(-y_lim, y_lim), xlim=(-y_range - 0.2, y_range + 0.2),
    )
    g.map_dataframe(
        sns.lineplot, x="y_bin", y="dy_rotated",
        errorbar=("ci", 95), color="#1f77b4",
    )
    for ax, roi in zip(g.axes.flat, rois):
        ax.axhline(0, color="k", lw=0.5, ls="--")
        ax.axvline(0, color="k", lw=0.3, alpha=0.4)
        ax.text(y_range, y_lim * 0.85, "← HP", color="red",
                ha="right", va="top", fontsize=8)
        ax.text(-y_range, y_lim * 0.85, "opp →", color="blue",
                ha="left", va="top", fontsize=8)
    g.set_axis_labels("y_mean_rotated (deg)\n(signed position along HP-opposite axis)",
                      "dy_rotated (deg)\n(<0: toward opposite; >0: toward HP)")
    g.fig.suptitle("Shift along HP-opposite axis vs signed position along axis"
                   "\nNegative across whole range = pure suppression; "
                   "sign flip = local suppression + global attraction", y=1.04)
    g.fig.tight_layout()
    return g.fig


def plot_shift_along_hp_axis_combined(
    pars: pd.DataFrame,
    rois_combined: tuple[str, ...] = ("V3AB", "hV4"),
    bin_edges: np.ndarray | None = None,
    y_range: float = 3.0,
    y_lim: float = 0.5,
    fig: plt.Figure | None = None,
):
    """Focused combined V3AB+hV4 version of the HP-axis shift plot.

    Drops voxels with |y_mean_rotated| > y_range (extreme bins are
    sparse and dominated by noise). Y-axis fixed to ±y_lim so the
    central pattern is readable.
    """
    import seaborn as sns
    if bin_edges is None:
        bin_edges = np.arange(-y_range, y_range + 0.5, 0.5)

    work = pars.reset_index()
    work = work[
        (work["roi_base"].isin(rois_combined))
        & (work["y_mean_rotated"].abs() < y_range)
    ].copy()
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    work["y_bin"] = pd.cut(
        work["y_mean_rotated"], bin_edges,
        labels=centers, include_lowest=True,
    )
    work = work.dropna(subset=["y_bin"])
    work["y_bin"] = work["y_bin"].astype(float)

    per_subject = (
        work.groupby(["subject", "y_bin"], observed=True)["dy_rotated"]
        .mean().reset_index()
    )

    if fig is None:
        fig, ax = plt.subplots(figsize=(9, 5))
    else:
        ax = fig.gca()

    sns.lineplot(
        data=per_subject, x="y_bin", y="dy_rotated",
        errorbar=("ci", 95), color="#1f77b4", alpha=0.4, ax=ax,
    )

    # LOWESS smooth.
    from statsmodels.nonparametric.smoothers_lowess import lowess
    sub = per_subject.dropna(subset=["y_bin", "dy_rotated"])
    if len(sub) >= 5:
        sm = lowess(sub["dy_rotated"].values, sub["y_bin"].astype(float).values,
                    frac=0.5, return_sorted=True)
        ax.plot(sm[:, 0], sm[:, 1], color="#1f77b4", lw=2.5, label="LOWESS")

    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.axvline(0, color="k", lw=0.3, alpha=0.4)
    ax.set_xlim(-y_range - 0.2, y_range + 0.2)
    ax.set_ylim(-y_lim, y_lim)
    ax.text(y_range, y_lim * 0.85, "← HP", color="red",
            ha="right", va="top", fontsize=11)
    ax.text(-y_range, y_lim * 0.85, "opp →", color="blue",
            ha="left", va="top", fontsize=11)

    ax.set_xlabel("y_mean_rotated (deg)\nsigned position along HP-opposite axis")
    ax.set_ylabel("dy_rotated (deg)\n(<0: shifted toward opposite; >0: shifted toward HP)")
    ax.set_title(f"Shift along HP-opposite axis in {' + '.join(rois_combined)} combined\n"
                 f"(voxels with |y_mean_rotated| < {y_range}°; y-axis clipped to ±{y_lim}°)\n"
                 "Pure suppression: y < 0 everywhere   ·   "
                 "Mixed local-supp + global-attr: sign flip across x")
    fig.tight_layout()
    return fig


def plot_hp_vs_opposite(
    pars: pd.DataFrame,
    rois: list[str] | None = None,
    near_threshold: float = 1.5,
    fig: plt.Figure | None = None,
):
    """Per-ROI HP-anchored vs opposite-anchored radial shift.

    Shows the asymmetric directional prediction: under real HP-specific
    suppression, HP shift goes positive while opposite shift goes
    negative. Under pure isotropic Jensen noise, both are positive and
    equal. The paired difference (right panel) centers at 0 under the
    null and at 2× the true effect under real suppression.
    """
    import seaborn as sns
    from scipy import stats
    from statsmodels.stats.multitest import multipletests

    if rois is None:
        rois = [r for r in roi_order if r in pars.index.get_level_values("roi_base").unique()]

    work = pars.reset_index()
    work = work[
        (work["roi_base"].isin(rois))
        & (work["distance_from_distractor_mean"] <= near_threshold)
    ].copy()

    long = (
        work.groupby(["subject", "roi_base"], observed=True)
        .agg(hp=("shift_from_distractor", "mean"),
             opposite=("shift_from_opposite", "mean"),
             paired=("paired_hp_minus_opposite", "mean"))
        .reset_index()
    )

    # Per-ROI paired t-test on the paired metric.
    rows = []
    for roi in rois:
        sub = long[long["roi_base"] == roi].dropna(subset=["paired"])
        if len(sub) < 3:
            rows.append({"roi": roi, "n": len(sub), "hp_mean": np.nan,
                         "opp_mean": np.nan, "paired_mean": np.nan,
                         "paired_sem": np.nan, "t": np.nan, "p": np.nan})
            continue
        t, p = stats.ttest_1samp(sub["paired"], 0.0)
        rows.append({
            "roi": roi, "n": len(sub),
            "hp_mean": float(sub["hp"].mean()),
            "opp_mean": float(sub["opposite"].mean()),
            "paired_mean": float(sub["paired"].mean()),
            "paired_sem": float(sub["paired"].sem()),
            "t": float(t), "p": float(p),
        })
    stats_df = pd.DataFrame(rows)
    valid = stats_df["p"].notna()
    if valid.any():
        _, p_fdr, _, _ = multipletests(stats_df.loc[valid, "p"], method="fdr_bh")
        stats_df.loc[valid, "p_fdr"] = p_fdr

    if fig is None:
        fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13, 5))
    else:
        ax_l, ax_r = fig.axes[:2]

    melt = long.melt(
        id_vars=["subject", "roi_base"],
        value_vars=["hp", "opposite"],
        var_name="anchor", value_name="shift",
    )
    melt["anchor"] = melt["anchor"].map({"hp": "HP", "opposite": "Opposite"})
    sns.pointplot(
        data=melt, x="roi_base", y="shift",
        hue="anchor", order=rois, hue_order=["HP", "Opposite"],
        palette={"HP": "#d62728", "Opposite": "#1f77b4"},
        errorbar=("ci", 95), dodge=0.3, markers="o",
        linestyles=["-", "--"], ax=ax_l,
    )
    ax_l.axhline(0, color="k", lw=0.5, ls="--")
    ax_l.set_xlabel("ROI")
    ax_l.set_ylabel(f"Δ distance from anchor (deg)\n(voxels with mean PRF within {near_threshold}° of HP)")
    ax_l.set_title("HP-anchored vs opposite-anchored radial shift\n"
                   "(under real suppression: HP > 0, Opposite < 0)")
    ax_l.legend(title="Anchor", loc="best")

    p_col = "p_fdr" if "p_fdr" in stats_df.columns else "p"
    ax_r.errorbar(
        x=np.arange(len(stats_df)), y=stats_df["paired_mean"],
        yerr=stats_df["paired_sem"], fmt="o", color="black", capsize=4,
    )
    ax_r.set_xticks(np.arange(len(stats_df)))
    ax_r.set_xticklabels(stats_df["roi"])
    ax_r.axhline(0, color="k", lw=0.5, ls="--")
    ax_r.set_xlabel("ROI")
    ax_r.set_ylabel("HP shift − Opposite shift (deg)\n(= 2× HP-specific suppression effect)")
    ax_r.set_title("Paired HP-minus-Opposite per ROI\n(per-subject means; one-sample t; FDR-BH)")
    ymax = (stats_df["paired_mean"].fillna(0) + stats_df["paired_sem"].fillna(0)).max()
    for i, (roi, p) in enumerate(zip(stats_df["roi"], stats_df[p_col])):
        if pd.isna(p):
            continue
        marker = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        ax_r.text(i, ymax * 1.15, marker, ha="center", va="bottom",
                  fontsize=11 if marker != "n.s." else 9,
                  color="black" if marker != "n.s." else "0.5")

    fig.tight_layout()
    fig._stats_df = stats_df
    return fig


def combined_v3ab_hv4_test(
    pars: pd.DataFrame,
    rois_combined: tuple[str, ...] = ("V3AB", "hV4"),
    near_threshold: float = 1.5,
    n_perm: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    """A priori combined-ROI test on V3AB+hV4 voxels (no FDR correction).

    These are the ROIs predicted by AF+ to show the strongest HP-specific
    shift: pRFs large enough to overlap the HP locus, small enough that
    fitting noise doesn't dominate. Pooling them gives more voxels per
    subject and a single test.

    Reports two tests:
        (1) one-sample t against 0 on the raw HP shift
        (2) one-sample t against 0 on the paired HP − Opposite metric
            (controls for Jensen bias by construction)
    Plus the random-rotation permutation test on the raw HP shift.
    """
    from scipy import stats

    work = pars.reset_index()
    work = work[
        (work["roi_base"].isin(rois_combined))
        & (work["distance_from_distractor_mean"] <= near_threshold)
    ].copy()

    per_subject = (
        work.groupby("subject", observed=True)
        .agg(hp=("shift_from_distractor", "mean"),
             opposite=("shift_from_opposite", "mean"),
             paired=("paired_hp_minus_opposite", "mean"),
             n_voxel_rows=("shift_from_distractor", "size"))
        .reset_index()
    )
    n = len(per_subject)
    rows = []

    for label, col in [("HP shift (raw)", "hp"),
                       ("HP − Opposite (paired)", "paired")]:
        vals = per_subject[col].dropna()
        t, p = stats.ttest_1samp(vals, 0.0)
        rows.append({
            "test": label, "n_subj": int(len(vals)),
            "mean": float(vals.mean()), "sem": float(vals.sem()),
            "t": float(t), "p": float(p),
        })

    # Rotation permutation on the raw HP shift, restricted to V3AB+hV4.
    perm = random_rotation_permutation(
        pars[pars.index.get_level_values("roi_base").isin(rois_combined)],
        rois=list(rois_combined), near_threshold=near_threshold,
        n_perm=n_perm, seed=seed,
    )
    # Build a single combined permutation result by pooling voxels (not the
    # per-ROI averages). Easiest: compute the raw HP shift in V3AB+hV4 in
    # the real and per-permutation rotations.
    rng = np.random.default_rng(seed)
    subjects = sorted(work["subject"].unique())
    sub_idx = {s: i for i, s in enumerate(subjects)}

    cond_to_angle = {c: location_angles[c.replace("_", " ")] for c in CONDITIONS}
    work_perm = work.copy()
    work_perm["sub_idx"] = work_perm["subject"].map(sub_idx).astype(int)
    work_perm["cond_angle"] = work_perm["condition"].map(cond_to_angle).astype(float)

    x = work_perm["x"].to_numpy()
    y = work_perm["y"].to_numpy()
    xm = work_perm["x_mean"].to_numpy()
    ym = work_perm["y_mean"].to_numpy()
    cond_angle = work_perm["cond_angle"].to_numpy()
    sub_arr = work_perm["sub_idx"].to_numpy()
    sub_id_arr = work_perm["subject"].to_numpy()

    def _combined_stat(offset):
        a = cond_angle + offset[sub_arr]
        hx = 4.0 * np.cos(a)
        hy = 4.0 * np.sin(a)
        d = np.hypot(x - hx, y - hy)
        d_mean = np.hypot(xm - hx, ym - hy)
        shift = d - d_mean
        keep = d_mean <= near_threshold
        if keep.sum() == 0:
            return np.nan
        df = pd.DataFrame({"shift": shift[keep], "subject": sub_id_arr[keep]})
        return float(df.groupby("subject")["shift"].mean().mean())

    real_stat = _combined_stat(np.zeros(n))
    null = np.empty(n_perm)
    for k in range(n_perm):
        null[k] = _combined_stat(rng.uniform(0, 2 * np.pi, size=n))
    null = null[~np.isnan(null)]
    p_perm = (1 + np.sum(null >= real_stat)) / (1 + len(null))
    rows.append({
        "test": "HP shift (raw) — random-rotation perm",
        "n_subj": n, "mean": real_stat,
        "sem": float(np.std(null, ddof=1)),  # null sd as a precision proxy
        "t": np.nan, "p": float(p_perm),
    })

    out = pd.DataFrame(rows)
    out.attrs["per_subject"] = per_subject
    out.attrs["null"] = null
    out.attrs["real_stat"] = real_stat
    out.attrs["rois_combined"] = rois_combined
    return out


def plot_combined_test(combined: pd.DataFrame, fig=None):
    """Visualize the combined V3AB+hV4 a priori test."""
    import seaborn as sns
    per_subject = combined.attrs["per_subject"]
    null = combined.attrs["null"]
    real = combined.attrs["real_stat"]
    rois = combined.attrs["rois_combined"]

    if fig is None:
        fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13, 5),
                                         gridspec_kw={"width_ratios": [1, 1.2]})
    else:
        ax_l, ax_r = fig.axes[:2]

    melt = per_subject.melt(
        id_vars="subject", value_vars=["hp", "opposite", "paired"],
        var_name="metric", value_name="shift",
    )
    metric_label = {"hp": "HP shift", "opposite": "Opposite shift",
                    "paired": "HP − Opposite\n(2× true effect)"}
    melt["metric"] = melt["metric"].map(metric_label)
    sns.stripplot(data=melt, x="metric", y="shift",
                  order=list(metric_label.values()),
                  color="0.5", alpha=0.6, jitter=0.15, ax=ax_l)
    sns.pointplot(data=melt, x="metric", y="shift",
                  order=list(metric_label.values()),
                  color="black", errorbar=("ci", 95),
                  markers="D", ax=ax_l, linestyles="none")
    ax_l.axhline(0, color="k", lw=0.5, ls="--")
    ax_l.set_ylabel("Mean shift (deg)")
    ax_l.set_xlabel(None)
    ax_l.set_title(f"Per-subject means in {' + '.join(rois)}\n"
                   f"(n = {len(per_subject)}; voxels with mean PRF within 1.5° of HP)")

    ax_r.hist(null, bins=30, color="0.7", edgecolor="white")
    ax_r.axvline(real, color="red", lw=2.5, label=f"observed = {real:.3f}°")
    p_perm = (1 + np.sum(null >= real)) / (1 + len(null))
    ax_r.set_xlabel("Mean HP shift (deg)")
    ax_r.set_ylabel("Permutation count")
    ax_r.set_title(f"Random-rotation null in {' + '.join(rois)} combined\n"
                   f"observed vs null (p = {p_perm:.4f}, n_perm = {len(null)})")
    ax_r.legend(loc="best")

    fig.tight_layout()
    return fig


def random_rotation_permutation(
    pars: pd.DataFrame,
    rois: list[str] | None = None,
    near_threshold: float = 1.5,
    n_perm: int = 500,
    ecc_distractor: float = 4.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Permutation test by random-rotating HP positions per subject.

    Each iteration: pick a uniform random offset θ ∈ [0, 2π) per subject;
    rotate all four HP angles by θ (preserving the 90° spacing between
    conditions); recompute the per-ROI mean radial shift in the near-HP
    bin. The null distribution captures the expected effect under random
    placement of the HP quartet — most rotations land HPs in regions
    that never had a distractor in the actual experiment, so this is a
    cleaner null than HP-vs-non-HP.

    Returns a DataFrame: roi, real_stat, null_mean, null_sd, p,
    plus the full null array stored in `.attrs['null']`.
    """
    rng = np.random.default_rng(seed)

    if rois is None:
        rois = [r for r in roi_order if r in pars.index.get_level_values("roi_base").unique()]

    work = pars.reset_index()
    work = work[work["roi_base"].isin(rois)].copy()
    subjects = work["subject"].unique().tolist()
    sub_idx = {s: i for i, s in enumerate(subjects)}
    work["sub_idx"] = work["subject"].map(sub_idx).astype(int)

    # Per-row condition angle.
    cond_to_angle = {c: location_angles[c.replace("_", " ")] for c in CONDITIONS}
    work["cond_angle"] = work["condition"].map(cond_to_angle).astype(float)

    x = work["x"].to_numpy()
    y = work["y"].to_numpy()
    xm = work["x_mean"].to_numpy()
    ym = work["y_mean"].to_numpy()
    cond_angle = work["cond_angle"].to_numpy()
    sub_arr = work["sub_idx"].to_numpy()
    roi_arr = work["roi_base"].to_numpy()
    sub_id_arr = work["subject"].to_numpy()

    def _stat_for_offset(offset_per_subject: np.ndarray) -> dict[str, float]:
        """Compute per-ROI mean shift under a given per-subject offset."""
        offset = offset_per_subject[sub_arr]
        a = cond_angle + offset
        hx = ecc_distractor * np.cos(a)
        hy = ecc_distractor * np.sin(a)
        d = np.hypot(x - hx, y - hy)
        d_mean = np.hypot(xm - hx, ym - hy)
        shift = d - d_mean
        keep = d_mean <= near_threshold
        out = {}
        if keep.sum() == 0:
            for r in rois:
                out[r] = np.nan
            return out
        # Per-(subject, ROI) mean, then mean across subjects.
        df = pd.DataFrame({
            "shift": shift[keep],
            "subject": sub_id_arr[keep],
            "roi": roi_arr[keep],
        })
        per_sub = df.groupby(["subject", "roi"], observed=True)["shift"].mean()
        roi_means = per_sub.groupby("roi").mean()
        for r in rois:
            out[r] = float(roi_means.get(r, np.nan))
        return out

    # Real statistic (offset = 0).
    real = _stat_for_offset(np.zeros(len(subjects)))

    # Null: random per-subject offsets.
    null = {r: np.empty(n_perm) for r in rois}
    for k in range(n_perm):
        offsets = rng.uniform(0, 2 * np.pi, size=len(subjects))
        result = _stat_for_offset(offsets)
        for r in rois:
            null[r][k] = result[r]

    rows = []
    for r in rois:
        nul = null[r]
        nul_clean = nul[~np.isnan(nul)]
        real_val = real[r]
        # One-sided p (testing for HP > random rotations, i.e. real shift exceeds null).
        p = (1 + np.sum(nul_clean >= real_val)) / (1 + len(nul_clean))
        rows.append({
            "roi": r, "real_stat": real_val,
            "null_mean": float(np.mean(nul_clean)),
            "null_sd": float(np.std(nul_clean, ddof=1)),
            "null_q025": float(np.quantile(nul_clean, 0.025)),
            "null_q975": float(np.quantile(nul_clean, 0.975)),
            "p": float(p),
            "n_perm": int(len(nul_clean)),
        })
    out = pd.DataFrame(rows)
    out.attrs["null"] = null
    out.attrs["near_threshold"] = near_threshold
    return out


def plot_permutation_null(
    perm_results: pd.DataFrame,
    fig: plt.Figure | None = None,
):
    """Per-ROI null distribution + real statistic, with p-values."""
    rois = perm_results["roi"].tolist()
    null = perm_results.attrs["null"]
    near_thr = perm_results.attrs["near_threshold"]

    if fig is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        ax = fig.gca()

    for i, roi in enumerate(rois):
        nul = null[roi]
        nul = nul[~np.isnan(nul)]
        ax.scatter(np.full_like(nul, i, dtype=float) + np.random.uniform(-0.15, 0.15, size=len(nul)),
                   nul, s=3, alpha=0.25, color="#888888")
    ax.errorbar(
        x=np.arange(len(rois)),
        y=perm_results["real_stat"],
        fmt="o", color="red", markersize=10, zorder=5,
    )
    # 95% null band.
    for i, roi in enumerate(rois):
        ax.plot([i - 0.3, i + 0.3], [perm_results["null_q025"].iloc[i]] * 2, color="black", lw=1)
        ax.plot([i - 0.3, i + 0.3], [perm_results["null_q975"].iloc[i]] * 2, color="black", lw=1)

    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.set_xticks(np.arange(len(rois)))
    ax.set_xticklabels(rois)
    ax.set_xlabel("ROI")
    ax.set_ylabel(f"Mean radial shift (deg)\n(voxels with mean PRF within {near_thr:.1f}° of HP)")
    ax.set_title("Random-rotation permutation test\n"
                 "Gray dots = null (rotated HPs); red = observed; black ticks = 95% null interval")

    # p-value annotations.
    ymax = perm_results["null_q975"].max()
    real_max = perm_results["real_stat"].max()
    ymax = max(ymax, real_max) * 1.2
    for i, p in enumerate(perm_results["p"]):
        marker = (
            "p<0.001" if p < 0.001
            else f"p={p:.3f}" if p < 0.05
            else f"p={p:.2f}"
        )
        ax.text(i, ymax, marker, ha="center", va="bottom",
                fontsize=9, color="black" if p < 0.05 else "0.4")

    fig.tight_layout()
    return fig


def plot_shift_vs_size(
    pars: pd.DataFrame,
    rois: list[str] | None = None,
    near_threshold: float = 2.0,
    size_edges: np.ndarray | None = None,
    smooth: bool = True,
    fig: plt.Figure | None = None,
):
    """Mean radial shift vs mean PRF size, one line per ROI.

    Restricted to voxels whose mean PRF center is within `near_threshold`
    deg of the HP locus — i.e. the regime where AF+ predicts an effect.
    If all ROIs collapse onto the same curve, the hierarchy gradient is
    explained by PRF size alone. If higher ROIs sit above lower ROIs at
    the same PRF size, the gradient is structural over and above size.

    Wider 1° bins (default) + an overlaid LOWESS smooth (when `smooth`)
    suppress the per-bin noise that's inevitable at the high-sd end where
    only a few voxels per subject pass the inclusion criteria.
    """
    import seaborn as sns

    if rois is None:
        rois = [r for r in roi_order if r in pars.index.get_level_values("roi_base").unique()]
    if size_edges is None:
        size_edges = np.arange(0.0, 4.5, 1.0)  # 1° bins

    work = pars.reset_index()
    work = work[
        (work["roi_base"].isin(rois))
        & (work["distance_from_distractor_mean"] <= near_threshold)
    ].copy()
    work["size_bin"] = pd.cut(
        work["sd_mean_model"], size_edges,
        labels=0.5 * (size_edges[:-1] + size_edges[1:]),
        include_lowest=True,
    )
    work = work.dropna(subset=["size_bin"])
    work["size_bin"] = work["size_bin"].astype(float)

    per_subject = (
        work.groupby(["subject", "roi_base", "size_bin"], observed=True)["shift_from_distractor"]
        .mean()
        .reset_index()
    )

    if fig is None:
        fig, ax = plt.subplots(figsize=(9, 5))
    else:
        ax = fig.gca()

    palette = sns.color_palette("viridis", n_colors=len(rois))

    if smooth:
        sns.lineplot(
            data=per_subject, x="size_bin", y="shift_from_distractor",
            hue="roi_base", hue_order=rois, palette=palette,
            errorbar=("ci", 95), alpha=0.25, legend=False, ax=ax,
        )
        from statsmodels.nonparametric.smoothers_lowess import lowess
        for color, roi in zip(palette, rois):
            sub = per_subject[per_subject["roi_base"] == roi].dropna(
                subset=["size_bin", "shift_from_distractor"]
            )
            if len(sub) < 5:
                continue
            sm = lowess(sub["shift_from_distractor"].values,
                        sub["size_bin"].astype(float).values,
                        frac=0.6, return_sorted=True)
            ax.plot(sm[:, 0], sm[:, 1], color=color, lw=2.0, label=roi)
        ax.legend(title="ROI", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    else:
        sns.lineplot(
            data=per_subject, x="size_bin", y="shift_from_distractor",
            hue="roi_base", hue_order=rois, palette=palette,
            errorbar=("ci", 95), ax=ax,
        )
        ax.legend(title="ROI", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)

    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.set_xlabel("Mean-model PRF size (deg)")
    ax.set_ylabel("Δ distance from HP (deg)\n(+ = pushed away)")
    ax.set_title(
        f"Radial shift vs PRF size, voxels with mean PRF within {near_threshold:.1f}° of HP\n"
        + ("(faded = 1° bins ± 95% CI; bold = LOWESS) — collapse → RF-size driven; separation → structural"
           if smooth else "If lines collapse → RF-size driven; if separate → structural")
    )
    fig.tight_layout()
    return fig


def roi_effect_controlling_for_size(
    pars: pd.DataFrame,
    rois: list[str] | None = None,
    near_threshold: float = 2.0,
    reference_roi: str = "V1",
) -> pd.DataFrame:
    """OLS at the voxel level: shift ~ sd_mean + C(roi_base) + C(subject).

    Tests whether ROI explains shift over and above PRF size. Returns a
    DataFrame with the ROI coefficients (relative to `reference_roi`).
    Restricted to voxels near the HP (`distance_from_distractor_mean <=
    near_threshold`).

    The interpretation: if the ROI coefficients are close to zero after
    controlling for sd_mean, the hierarchy gradient is RF-size driven.
    If the coefficients still grow up the hierarchy, it's structural.
    """
    import statsmodels.formula.api as smf

    if rois is None:
        rois = [r for r in roi_order if r in pars.index.get_level_values("roi_base").unique()]

    work = pars.reset_index()
    work = work[
        (work["roi_base"].isin(rois))
        & (work["distance_from_distractor_mean"] <= near_threshold)
    ][["subject", "roi_base", "sd_mean", "shift_from_distractor"]].dropna()

    work["roi_base"] = pd.Categorical(work["roi_base"], categories=[reference_roi] +
                                       [r for r in rois if r != reference_roi])
    work["subject"] = work["subject"].astype("category")

    # Fixed-effects subject; absorb subject mean rather than fitting a random effect
    # (faster, no convergence headaches at this sample size).
    model = smf.ols(
        "shift_from_distractor ~ sd_mean + C(roi_base) + C(subject)",
        data=work,
    ).fit()

    rows = []
    rows.append({
        "term": "sd_mean (deg PRF size)", "coef": model.params["sd_mean"],
        "se": model.bse["sd_mean"], "p": model.pvalues["sd_mean"],
    })
    for roi in rois:
        if roi == reference_roi:
            rows.append({"term": f"ROI {roi} (reference)", "coef": 0.0, "se": np.nan, "p": np.nan})
            continue
        key = f"C(roi_base)[T.{roi}]"
        if key not in model.params:
            continue
        rows.append({
            "term": f"ROI {roi} - {reference_roi}",
            "coef": model.params[key], "se": model.bse[key], "p": model.pvalues[key],
        })
    out = pd.DataFrame(rows)
    out.attrs["n"] = int(model.nobs)
    out.attrs["model"] = model
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bids-folder", default="/data/ds-retsupp")
    parser.add_argument("--model", type=int, default=8)
    parser.add_argument("--out", type=Path, default=Path("notes/vss2026_figure_B.pdf"))
    parser.add_argument("--roi", default="V1", help="ROI for the arrow-field panel")
    parser.add_argument("--r2-thr", type=float, default=0.2)
    parser.add_argument("--ecc-thr", type=float, default=3.0)
    args = parser.parse_args()

    print(f"Loading conditionwise data (model={args.model}) ...")
    pars = load_all_conditionwise(
        bids_folder=args.bids_folder,
        model=args.model,
        r2_thr=args.r2_thr,
        ecc_thr=args.ecc_thr,
    )
    n_subj = pars.index.get_level_values("voxel").nunique()  # not exact, just informative
    print(
        f"Loaded {len(pars):,} voxel×condition rows from "
        f"{pars['subject'].nunique()} subjects "
        f"covering ROIs {sorted(pars.index.get_level_values('roi_base').unique())}"
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rois_present = [r for r in roi_order if r in pars.index.get_level_values("roi_base").unique()]
    with PdfPages(args.out) as pdf:
        # Per-condition arrows, one ROI per page (full detail for the deck).
        for roi in rois_present:
            fig = plot_arrow_field(pars, roi=roi)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        # Combined-rotated arrow plot — all conditions pooled, one ROI per panel.
        fig = plot_arrow_field_rotated(pars, rois=rois_present)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Rotated hexbin (the headline summary).
        fig = plot_rotated_hexbin(pars, rois=rois_present)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Shift vs distance from HP — locality of the effect, one line per ROI.
        fig = plot_shift_vs_distance(pars, rois=rois_present)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Per-ROI summary at near distractor + stats (Figure C).
        stats_df = near_distractor_stats(pars, rois=rois_present, near_threshold=1.5)
        fig = plot_near_distractor_summary(stats_df)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Per-ROI summary across multiple distance bins.
        fig = plot_shift_by_roi_and_distance(pars, rois=rois_present)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Shift vs PRF size — disentangle hierarchy from RF-size confound.
        fig = plot_shift_vs_size(pars, rois=rois_present, near_threshold=2.0)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # HP vs non-HP null test — bounds Jensen's-inequality artefact.
        fig = plot_hp_vs_nonhp_summary(pars, rois=rois_present, near_threshold=1.5)
        pdf.savefig(fig, bbox_inches="tight")
        nullstats = fig._stats_df
        plt.close(fig)

        # Per-ROI HP vs non-HP curve as a function of distance.
        fig = plot_hp_vs_nonhp_curve(pars, rois=rois_present)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Random-rotation permutation test (the cleaner null).
        print("\nRunning random-rotation permutation test (n_perm=500) ...")
        perm = random_rotation_permutation(pars, rois=rois_present,
                                           near_threshold=1.5, n_perm=500)
        fig = plot_permutation_null(perm)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Per-anchor HP-vs-LP contrast (transparent design view, projection metric).
        fig = plot_hp_vs_lp_per_anchor(
            pars, rois=rois_present, near_threshold=1.5,
            metric="projection_away_from_anchor",
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # HP vs opposite anchor — the paired-baseline test.
        fig = plot_hp_vs_opposite(pars, rois=rois_present, near_threshold=1.5)
        pdf.savefig(fig, bbox_inches="tight")
        opp_stats = fig._stats_df
        plt.close(fig)

        # A priori combined V3AB+hV4 test.
        print("\nRunning combined V3AB+hV4 a priori test (n_perm=500) ...")
        combined = combined_v3ab_hv4_test(pars, rois_combined=("V3AB", "hV4"),
                                          near_threshold=1.5, n_perm=500)
        fig = plot_combined_test(combined)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Shift along HP-opposite axis — tests for distance-dependent sign flip.
        fig = plot_shift_along_hp_axis(pars, rois=rois_present)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        fig = plot_shift_along_hp_axis_combined(pars, rois_combined=("V3AB", "hV4"))
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Near-axis-only version of the combined plot (cleaner signal).
        fig = plot_axis_shift_on_axis(pars, rois_combined=("V3AB", "hV4"),
                                      x_threshold=1.5, y_range=3.0)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Bias-free paired metric vs distance from HP — sign flip = global attraction.
        fig = plot_paired_vs_distance(pars, rois=rois_present)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Combined V3AB+hV4 paired-vs-distance with formal far-d attraction test.
        fig = plot_paired_vs_distance_combined(
            pars, rois_combined=("V3AB", "hV4"),
            near_threshold=1.5, far_threshold=4.0,
        )
        pdf.savefig(fig, bbox_inches="tight")
        far_p = fig._far_p
        far_mean = fig._far_mean
        near_p = fig._near_p
        near_mean = fig._near_mean
        plt.close(fig)

        # Per-ROI version of the paired-vs-distance test.
        fig = plot_paired_vs_distance_per_roi(
            pars, rois=rois_present,
            near_threshold=1.5, far_threshold=4.0,
        )
        pdf.savefig(fig, bbox_inches="tight")
        per_roi_paired = fig._summary
        plt.close(fig)

        # Per-ROI position-based view of axial shift along HP-opposite axis.
        fig = plot_axis_shift_on_axis_per_roi(
            pars, rois=rois_present, x_threshold=1.5, y_range=3.0,
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Size and amplitude tests — AF model predictions.
        fig = plot_size_and_amplitude_vs_distance(pars, rois=rois_present,
                                                  near_threshold=1.5)
        pdf.savefig(fig, bbox_inches="tight")
        size_amp_summary = fig._summary
        plt.close(fig)

        # AF and AF+ model fits per subject — separately for each ROI.
        per_roi_fits = {}
        for roi in ("V3AB", "hV4"):
            print(f"\nFitting AF and AF+ per subject in {roi} ...")
            fit_dict = fit_af_models_per_subject(pars, rois_combined=(roi,))
            if len(fit_dict["summary"]) > 0:
                per_roi_fits[roi] = fit_dict
                fig = plot_af_fit_results(fit_dict, rois_combined=(roi,))
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
                fig = plot_af_condition_deviations(fit_dict, rois_combined=(roi,))
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

        # Rotated-frame AF fit (single AF center per subject per ROI).
        from retsupp.modeling.af_model import fit_af_rotated_canonical
        rotated_fits = {}
        rotated_fits_fixed = {}
        for roi in ("V3AB", "hV4"):
            print(f"\nRotated-frame AF fit (free σ_AF ∈ [0.5, 8]°), {roi} ...")
            df = fit_af_rotated_canonical(pars, [roi])
            if len(df) > 0:
                rotated_fits[roi] = df
            print(f"Rotated-frame AF fit (σ_AF fixed at 4°), {roi} ...")
            df_fix = fit_af_rotated_canonical(pars, [roi], fixed_sigma_AF=4.0)
            if len(df_fix) > 0:
                rotated_fits_fixed[roi] = df_fix
        if rotated_fits:
            fig = plot_af_rotated_canonical(
                rotated_fits, title_suffix=" (free σ_AF)",
            )
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
        if rotated_fits_fixed:
            fig = plot_af_rotated_canonical(
                rotated_fits_fixed, title_suffix=" (σ_AF = 4° fixed)",
            )
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        # R² threshold sensitivity sweep — robustness check.
        print("\nRunning R² threshold sweep ...")
        pars_loose = load_all_conditionwise(
            bids_folder=args.bids_folder, model=args.model,
            r2_thr=0.05,  # loose floor
            ecc_thr=args.ecc_thr,
        )
        sweep = r2_threshold_sweep(pars_loose,
                                   thresholds=[0.05, 0.10, 0.15, 0.20, 0.25,
                                               0.30, 0.40, 0.50])
        fig = plot_r2_sweep(sweep, default_thr=args.r2_thr)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    # Print stats summaries to stdout.
    print()
    print("=" * 80)
    print("Per-ROI radial shift for voxels within 1.5 deg of HP distractor")
    print("(positive = pushed away; one-sample t-test against 0; FDR-BH)")
    print("=" * 80)
    print(stats_df.to_string(index=False, float_format=lambda v: f"{v:7.4f}"))

    print()
    print("=" * 80)
    print("ROI effect after controlling for PRF size (OLS, near voxels only)")
    print("Reference ROI = V1; coef in deg shift")
    print("=" * 80)
    roi_table = roi_effect_controlling_for_size(pars, rois=rois_present, near_threshold=2.0)
    print(roi_table.to_string(index=False, float_format=lambda v: f"{v:7.4f}"))
    print(f"\nN voxels in regression: {roi_table.attrs['n']:,}")

    print()
    print("=" * 80)
    print("HP vs non-HP null test (paired t per ROI, near-anchor voxels only)")
    print("If non-HP shift ≈ HP shift → effect is Jensen's-inequality artefact")
    print("If HP > non-HP → real HP-specific suppression")
    print("=" * 80)
    print(nullstats.to_string(index=False, float_format=lambda v: f"{v:7.4f}"))

    print()
    print("=" * 80)
    print("Random-rotation permutation test (n_perm=500, near voxels only)")
    print("Real shift compared to null distribution from random HP rotations per subject")
    print("=" * 80)
    print(perm.drop(columns=["n_perm"]).to_string(
        index=False, float_format=lambda v: f"{v:7.4f}"))

    print()
    print("=" * 80)
    print("HP vs Opposite-anchor paired test (per ROI, near voxels only)")
    print("paired = HP shift − Opposite shift; centers at 0 under Jensen-only null")
    print("=" * 80)
    print(opp_stats.to_string(index=False, float_format=lambda v: f"{v:7.4f}"))

    print()
    print("=" * 80)
    print("A priori combined V3AB + hV4 test (no FDR — single test by AF+ pre-spec)")
    print("=" * 80)
    print(combined.to_string(index=False, float_format=lambda v: f"{v:7.4f}"))

    print()
    print("=" * 80)
    print("Sign-flip test in V3AB + hV4 combined")
    print("=" * 80)
    print(f"Near HP   (d ≤ 1.5°): paired_metric = {near_mean:+.4f}°, p = {near_p:.4f}"
          f"  ({'sig suppression' if near_p < 0.05 and near_mean > 0 else 'n.s.'})")
    print(f"Far  from HP (d > 4°): paired_metric = {far_mean:+.4f}°, p = {far_p:.4f}"
          f"  ({'sig attraction' if far_p < 0.05 and far_mean < 0 else 'n.s.'})")

    print()
    print("=" * 80)
    print("Per-ROI near-vs-far paired metric tests")
    print("=" * 80)
    print(per_roi_paired.to_string(index=False, float_format=lambda v: f"{v:7.4f}"))

    print()
    print("=" * 80)
    print("Size and amplitude near HP, per ROI (one-sample t against zero)")
    print("AF predicts: sd_diff = 0 (no condition size change);  amp_diff < 0 (suppression)")
    print("=" * 80)
    print(size_amp_summary.to_string(index=False, float_format=lambda v: f"{v:7.4f}"))

    for roi, fd in per_roi_fits.items():
        print()
        print("=" * 80)
        print(f"AF and AF+ model fits per subject — {roi}")
        print("=" * 80)
        print(fd["summary"].to_string(index=False, float_format=lambda v: f"{v:7.3f}"))

    from scipy import stats
    for label, fits in [("free σ_AF", rotated_fits), ("σ_AF=4° fixed", rotated_fits_fixed)]:
        for roi, df in fits.items():
            print()
            print("=" * 80)
            print(f"Rotated-frame AF fit ({label}) — {roi}")
            print("=" * 80)
            sat = df.get("saturated", pd.Series([False] * len(df))).astype(bool)
            df_clean = df.loc[~sat]
            mean_x = df_clean["mu_AF_x"].mean()
            mean_y = df_clean["mu_AF_y"].mean()
            sem_y = df_clean["mu_AF_y"].sem()
            t, p = stats.ttest_1samp(df_clean["mu_AF_y"], 0.0) if len(df_clean) >= 3 else (np.nan, np.nan)
            n_neg = int((df_clean["mu_AF_y"] < 0).sum())
            print(f"  n subjects (non-saturated):    {len(df_clean)}/{len(df)}")
            print(f"  group mean (mu_AF_x, mu_AF_y): ({mean_x:+.3f}, {mean_y:+.3f})°")
            print(f"  mu_AF_y SEM:                   ±{sem_y:.3f}°")
            print(f"  one-sample t (y vs 0):         t={t:+.2f}, p={p:.4f}")
            print(f"  subjects with mu_AF_y < 0:     {n_neg}/{len(df_clean)}")

    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
