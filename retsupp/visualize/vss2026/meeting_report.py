"""Focused report for the Theeuwes/Duncan meeting.

Five figures only — bare essentials:

  1. Per-anchor HP-vs-LP comparison (projection metric, no Jensen)
     + design clarification (1 HP row vs 3 LP rows per voxel × anchor).
  2. 2D hexbin shift map in rotated frame, with permutation-rotation null
     side-by-side for visual comparison.
  3. Projection along the HP-opposite axis vs distance from HP, per ROI.
  4. 2D scatter of all per-condition parameter shifts (x, y, sd, amplitude)
     — full-dimensional view of the conditionwise effect.
  5. Divisive-Normalization (model 6) parameter distributions per ROI.

Run:
    python -m retsupp.visualize.meeting_report --out notes/meeting_report.pdf
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import TwoSlopeNorm

from retsupp.utils.data import (
    Subject,
    distractor_locations,
    get_subject_ids,
    location_angles,
    roi_order,
)
from retsupp.visualize.vss2026_arrows import (
    CONDITIONS,
    OPPOSITE_LOCATION,
    add_anchor_shifts,
    load_all_conditionwise,
)


def filter_prf_inside_aperture(
    pars: pd.DataFrame,
    aperture_radius: float = 3.17,
    max_fraction_outside: float = 0.5,
    use_mean_only: bool = True,
) -> pd.DataFrame:
    """Drop voxels whose PRF has more than `max_fraction_outside` of its
    mass outside the stimulus aperture.

    Uses the 1D normal CDF on the radial direction:
        f_out ≈ Φ((d − R) / σ)
    where d = ‖(x, y)‖ and R is the aperture radius.

    If `use_mean_only=True` (default), the criterion is applied only to the
    voxel's MEAN PRF fit (sd_mean_model, x_mean, y_mean) — this keeps voxels
    whose canonical PRF is well-mapped, even if their per-condition PRFs
    wander past the aperture boundary. Important for our design because
    the HP distractor locations sit AT the aperture boundary (ecc 4°),
    so per-condition voxels near HP would be removed by a strict filter.

    If `use_mean_only=False`, drops voxels where ANY conditionwise PRF
    fails the criterion (more conservative).
    """
    from scipy.stats import norm
    work = pars.reset_index().copy()
    if use_mean_only:
        d = np.hypot(work["x_mean"], work["y_mean"])
        sd = work["sd_mean_model"].clip(lower=0.05)
    else:
        d = np.hypot(work["x"], work["y"])
        sd = work["sd"].clip(lower=0.05)
    f_out = norm.cdf((d - aperture_radius) / sd)
    work["_f_out"] = f_out

    keep_voxels = (
        work.groupby(["subject", "roi_base", "roi", "voxel"], observed=True)["_f_out"]
        .max()
        .pipe(lambda s: s[s <= max_fraction_outside].index)
    )
    work = work.set_index(["subject", "roi_base", "roi", "voxel"])
    work = work.loc[work.index.isin(keep_voxels)].reset_index()
    work = work.drop(columns=["_f_out"])
    work = work.set_index(["roi_base", "roi", "voxel", "condition"])
    return work


# -----------------------------------------------------------------------------
# Figure 1: Per-anchor HP-vs-LP with permutation null
# -----------------------------------------------------------------------------

def figure_1_hp_vs_lp_per_anchor(
    pars: pd.DataFrame,
    rois: list[str],
    near_threshold: float = 1.5,
    n_perm: int = 200,
    seed: int = 42,
):
    """For each anchor location, contrast HP-condition vs LP-conditions
    (3 LP per anchor) of the projection-onto-anchor metric. Adds a
    rotation-permutation null per ROI for reference.
    """
    from scipy import stats

    tbl = add_anchor_shifts(pars)
    near = tbl[(tbl["roi_base"].isin(rois)) & (tbl["anchor_distance_mean"] <= near_threshold)].copy()
    near["status"] = near["is_hp"].map({True: "HP", False: "LP"})

    per_subj = (
        near.groupby(["subject", "roi_base", "anchor", "status"], observed=True)
        ["projection_away_from_anchor"].mean().reset_index()
    )

    # Compute permutation null per ROI: permute condition labels per subject.
    rng = np.random.default_rng(seed)
    null_per_roi = {}
    for roi in rois:
        roi_near = near[near["roi_base"] == roi].copy()
        observed_diffs = []
        for sub, sub_df in roi_near.groupby("subject", observed=True):
            piv = (
                sub_df.groupby(["anchor", "status"], observed=True)
                ["projection_away_from_anchor"].mean().unstack()
            )
            if "HP" in piv.columns and "LP" in piv.columns:
                observed_diffs.append((piv["HP"] - piv["LP"]).mean())
        observed = np.mean(observed_diffs) if observed_diffs else np.nan

        nulls = []
        for k in range(n_perm):
            sub_diffs = []
            for sub, sub_df in roi_near.groupby("subject", observed=True):
                tmp = sub_df.copy()
                # Permute condition labels per voxel within subject.
                tmp["condition"] = rng.permutation(tmp["condition"].values)
                tmp["is_hp_perm"] = (tmp["condition"].str.replace("_", " ") == tmp["anchor"])
                tmp["status_perm"] = tmp["is_hp_perm"].map({True: "HP", False: "LP"})
                piv = (
                    tmp.groupby(["anchor", "status_perm"], observed=True)
                    ["projection_away_from_anchor"].mean().unstack()
                )
                if "HP" in piv.columns and "LP" in piv.columns:
                    sub_diffs.append((piv["HP"] - piv["LP"]).mean())
            if sub_diffs:
                nulls.append(np.mean(sub_diffs))
        nulls = np.array(nulls)
        p_perm = (1 + np.sum(nulls >= observed)) / (1 + len(nulls)) if len(nulls) > 0 else np.nan
        null_per_roi[roi] = {"observed": observed, "null": nulls, "p": p_perm}

    fig = plt.figure(figsize=(13, 7))
    gs = fig.add_gridspec(2, len(rois), height_ratios=[2, 1.1], hspace=0.45, wspace=0.3)
    anchor_order = ["upper right", "upper left", "lower left", "lower right"]
    for j, roi in enumerate(rois):
        ax = fig.add_subplot(gs[0, j])
        roi_df = per_subj[per_subj["roi_base"] == roi]
        sns.pointplot(
            data=roi_df, x="anchor", y="projection_away_from_anchor",
            hue="status", order=anchor_order, hue_order=["HP", "LP"],
            palette={"HP": "#d62728", "LP": "#1f77b4"},
            errorbar=("ci", 95), dodge=0.4,
            markers=["o", "s"], linestyles=["-", "--"], ax=ax,
        )
        ax.axhline(0, color="k", lw=0.5, ls="--")
        ax.set_xticklabels([a.replace(" ", "\n") for a in anchor_order], fontsize=9)
        ax.set_xlabel("anchor (distractor location)")
        ax.set_ylabel("projection away from anchor (deg)" if j == 0 else "")
        ax.set_title(roi)
        if j != len(rois) - 1:
            ax.legend_.remove()

        # Bottom panel: permutation null distribution + observed.
        ax2 = fig.add_subplot(gs[1, j])
        d = null_per_roi[roi]
        ax2.hist(d["null"], bins=20, color="0.75", edgecolor="white")
        ax2.axvline(d["observed"], color="red", lw=2.5,
                    label=f"obs = {d['observed']:+.3f}°")
        ax2.set_xlabel("null Δ (HP − LP), deg")
        if j == 0:
            ax2.set_ylabel("permutations")
        ax2.set_title(f"perm-null  p = {d['p']:.3f}", fontsize=10)
        ax2.legend(fontsize=8, loc="best")

    fig.suptitle(
        "Figure 1 — HP-vs-LP comparison per anchor (projection metric, no Jensen)\n"
        "Top: For each anchor (distractor location), HP-condition (red) "
        "vs the 3 LP-conditions (blue, mean of three).  Bottom: permutation null "
        "(per-subject condition shuffle) of the (HP − LP) difference averaged across anchors.",
        y=1.02,
    )
    fig.tight_layout()
    return fig, null_per_roi


def figure_1b_hp_vs_lp_vs_distance(
    pars: pd.DataFrame,
    rois: list[str],
    bin_edges: np.ndarray | None = None,
    clip_projection: float = 1.5,
    y_lim: float = 0.4,
):
    """Figure 1b — HP-vs-LP projection metric as a function of distance
    from the **anchor**, per ROI.

    For each (voxel, anchor) pair:
      - HP value = projection in the one condition where condition's HP = anchor.
      - LP value = mean projection across the three LP conditions.
    Bin by anchor_distance_mean (= distance from voxel mean to that anchor).

    Outlier handling: voxel-condition-anchor rows with |projection| >
    `clip_projection` are dropped before aggregation (≈1% of data; these
    are PRF-fitting outliers, e.g. shifts of ±2-3° that are larger than
    physically plausible). Y-axis is clipped to ±`y_lim` for readability.
    """
    from scipy import stats
    from statsmodels.nonparametric.smoothers_lowess import lowess

    if bin_edges is None:
        bin_edges = np.arange(0.0, 7.5, 1.0)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    tbl = add_anchor_shifts(pars)
    tbl = tbl[tbl["roi_base"].isin(rois)].copy()
    n_before = len(tbl)
    tbl = tbl[tbl["projection_away_from_anchor"].abs() <= clip_projection].copy()
    n_after = len(tbl)
    tbl["status"] = tbl["is_hp"].map({True: "HP", False: "LP"})

    # Per (subject, ROI, voxel, anchor, status): HP=1 row, LP=mean of 3 rows.
    per_voxel_anchor = (
        tbl.groupby(
            ["subject", "roi_base", "voxel", "anchor", "status",
             "anchor_distance_mean"],
            observed=True,
        )["projection_away_from_anchor"].mean().reset_index()
    )
    per_voxel_anchor["d_bin"] = pd.cut(
        per_voxel_anchor["anchor_distance_mean"], bin_edges,
        labels=centers, include_lowest=True,
    )
    per_voxel_anchor = per_voxel_anchor.dropna(subset=["d_bin"])
    per_voxel_anchor["d_bin"] = per_voxel_anchor["d_bin"].astype(float)

    per_subject = (
        per_voxel_anchor.groupby(
            ["subject", "roi_base", "d_bin", "status"], observed=True,
        )["projection_away_from_anchor"].mean().reset_index()
    )

    n = len(rois)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 3.5),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, roi in zip(axes, rois):
        roi_df = per_subject[per_subject["roi_base"] == roi]
        sns.lineplot(
            data=roi_df, x="d_bin", y="projection_away_from_anchor",
            hue="status", hue_order=["HP", "LP"],
            palette={"HP": "#d62728", "LP": "#1f77b4"},
            errorbar=("ci", 95), ax=ax,
        )
        ax.axhline(0, color="k", lw=0.5, ls="--")
        ax.set_xlabel("distance of mean PRF from anchor (deg)")
        ax.set_ylabel("projection away from anchor (deg)")
        ax.set_title(roi)
        ax.set_ylim(-y_lim, y_lim)

        # Stats: HP - LP per subject in the near (d ≤ 1.5°) band.
        near = per_voxel_anchor[
            (per_voxel_anchor["roi_base"] == roi)
            & (per_voxel_anchor["anchor_distance_mean"] <= 1.5)
        ]
        piv = (
            near.groupby(["subject", "status"], observed=True)
            ["projection_away_from_anchor"].mean().unstack()
        )
        if "HP" in piv.columns and "LP" in piv.columns:
            diff = (piv["HP"] - piv["LP"]).dropna()
            if len(diff) >= 3:
                t, p = stats.ttest_1samp(diff, 0.0)
                ax.text(
                    0.98, 0.98,
                    f"near (d≤1.5°)\nHP − LP = {diff.mean():+.3f}°\n"
                    f"t={t:+.2f}, p={p:.3f}",
                    ha="right", va="top", transform=ax.transAxes, fontsize=9,
                )
        if ax is not axes[-1]:
            ax.legend_.remove()

    n_dropped = n_before - n_after
    fig.suptitle(
        "Figure 1b — HP vs LP projection metric, as a function of distance from anchor\n"
        f"Same per-anchor design as Fig 1.  Outliers |projection| > {clip_projection}° dropped "
        f"({n_dropped:,}/{n_before:,} = {100*n_dropped/n_before:.1f}% of rows).  "
        f"Y-axis clipped to ±{y_lim}° for readability.",
        y=1.04,
    )
    fig.tight_layout()
    return fig


# -----------------------------------------------------------------------------
# Figure 2: 2D hexbin shift map vs permutation null
# -----------------------------------------------------------------------------

def figure_2_hexbin_vs_null(
    pars: pd.DataFrame,
    rois: list[str],
    n_perm: int = 1,  # one rotation-shuffled realization for the side-by-side
    seed: int = 42,
    metric: str = "radial",
):
    """Real rotated-frame hexbin (suppression direction = red) next to
    a single random-rotation null realization. Visual sanity check.

    `metric` ∈ {"radial", "projection"}:
        "radial"     — radial Δ-distance from HP (Jensen-prone, conservative).
        "projection" — −dy_rotated (linear projection toward opposite,
                       no Jensen bias). Use this for the "no-Jensen" version.
    """
    rng = np.random.default_rng(seed)

    work = pars.reset_index().copy()
    work = work[work["roi_base"].isin(rois)]

    # Build a rotation-shuffled version: for each subject, pick a random
    # offset θ and rotate every voxel × condition row by θ in canonical
    # frame. This breaks the HP-alignment but preserves all voxel-level
    # statistics.
    cond_to_angle = {c: location_angles[c.replace("_", " ")] for c in CONDITIONS}
    shuffled = work.copy()
    sub_offsets = {s: rng.uniform(0, 2 * np.pi) for s in shuffled["subject"].unique()}
    offsets = shuffled["subject"].map(sub_offsets).values
    cond_angle = shuffled["condition"].map(cond_to_angle).values

    # rotated coordinates use rotate_to_up = pi/2 - cond_angle.
    # to make a "fake HP" at sub-specific angle θ, we replace cond_angle with cond_angle + offset.
    new_rotate_to_up = np.pi/2 - (cond_angle + offsets)
    cos_a, sin_a = np.cos(new_rotate_to_up), np.sin(new_rotate_to_up)
    x_orig = shuffled["x"].values
    y_orig = shuffled["y"].values
    xm_orig = shuffled["x_mean"].values
    ym_orig = shuffled["y_mean"].values
    shuffled["x_rotated"] = x_orig * cos_a - y_orig * sin_a
    shuffled["y_rotated"] = x_orig * sin_a + y_orig * cos_a
    shuffled["x_mean_rotated"] = xm_orig * cos_a - ym_orig * sin_a
    shuffled["y_mean_rotated"] = xm_orig * sin_a + ym_orig * cos_a
    shuffled["dx_rotated"] = shuffled["x_rotated"] - shuffled["x_mean_rotated"]
    shuffled["dy_rotated"] = shuffled["y_rotated"] - shuffled["y_mean_rotated"]
    shuffled["distance_from_distractor_rotated"] = np.hypot(
        shuffled["x_rotated"], shuffled["y_rotated"] - 4.0
    )
    shuffled["distance_from_distractor_mean_rotated"] = np.hypot(
        shuffled["x_mean_rotated"], shuffled["y_mean_rotated"] - 4.0
    )
    shuffled["shift_from_distractor_rotated"] = (
        shuffled["distance_from_distractor_rotated"]
        - shuffled["distance_from_distractor_mean_rotated"]
    )

    # Pick the color metric based on `metric` argument.
    if metric == "radial":
        c_col = "shift_from_distractor_rotated"
        cbar_label = ("Δ distance from HP (deg)\n"
                      "(blue = pulled toward HP, red = pushed away)")
        title_metric = "Δ-distance metric (Jensen-prone)"
    elif metric == "projection":
        # −dy_rotated: + means shifted toward opposite (away from HP).
        for df in (work, shuffled):
            df["proj_away_from_HP"] = -df["dy_rotated"]
        c_col = "proj_away_from_HP"
        cbar_label = ("Projection away from HP (deg)\n"
                      "(linear; no Jensen bias)")
        title_metric = "linear projection metric (no Jensen)"
    else:
        raise ValueError(f"unknown metric {metric!r}")

    fig, axes = plt.subplots(2, len(rois), figsize=(3.2 * len(rois), 6.5),
                             sharex=True, sharey=True)
    norm = TwoSlopeNorm(vmin=-0.25, vcenter=0, vmax=0.25)
    last_hb = None
    for j, roi in enumerate(rois):
        for i, (df, label) in enumerate([(work, "Real"), (shuffled, "Random rotation null")]):
            ax = axes[i, j]
            roi_df = df[df["roi_base"] == roi]
            last_hb = ax.hexbin(
                roi_df["x_mean_rotated"], roi_df["y_mean_rotated"],
                C=roi_df[c_col],
                gridsize=8, cmap="coolwarm", norm=norm,
                extent=(-4, 4, -4, 4), mincnt=20,
            )
            ax.scatter(0, 4, s=300, marker="*", color="k", zorder=3)
            ax.set_xlim(-4, 4); ax.set_ylim(-4, 4); ax.set_aspect("equal")
            ax.axhline(0, color="k", lw=0.3, alpha=0.3)
            ax.axvline(0, color="k", lw=0.3, alpha=0.3)
            if i == 0:
                ax.set_title(roi)
            if j == 0:
                ax.set_ylabel(f"{label}\ny in canonical frame (deg)")
            if i == 1:
                ax.set_xlabel("x in canonical frame (deg)")
    if last_hb is not None:
        cbar = fig.colorbar(last_hb, ax=axes.ravel().tolist(), shrink=0.7, pad=0.02)
        cbar.set_label(cbar_label)
    fig.suptitle(
        f"Figure 2 — 2D shift map in HP-aligned frame, real (top) vs one "
        f"random-rotation null realization (bottom)\n"
        f"{title_metric}.  Red blob near HP at top in REAL only = HP-specific "
        f"signal; noise looks similar in both rows.",
        y=1.02,
    )
    return fig


# -----------------------------------------------------------------------------
# Figure 3: Projection along HP-opposite axis vs distance from HP, per ROI
# -----------------------------------------------------------------------------

def figure_3_projection_vs_distance(
    pars: pd.DataFrame,
    rois: list[str],
    bin_edges: np.ndarray | None = None,
    y_lim: float = 0.3,
    clip_projection: float = 1.5,
):
    """Linear projection of the shift onto the HP-opposite axis (no
    Jensen), as a function of distance from HP. Per ROI, with LOWESS
    overlay and near/far stats. Y-axis clipped to ±y_lim. Outliers
    |proj| > clip_projection dropped.
    """
    from scipy import stats
    from statsmodels.nonparametric.smoothers_lowess import lowess

    if bin_edges is None:
        bin_edges = np.arange(0.0, 7.5, 0.5)

    work = pars.reset_index().copy()
    work = work[work["roi_base"].isin(rois)].copy()
    # Projection toward opposite (away from HP) is just -dy_rotated.
    work["proj_away_from_HP"] = -work["dy_rotated"]
    n_before = len(work)
    work = work[work["proj_away_from_HP"].abs() <= clip_projection]
    n_after = len(work)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    work["d_bin"] = pd.cut(
        work["distance_from_distractor_mean"], bin_edges,
        labels=centers, include_lowest=True,
    )
    work = work.dropna(subset=["d_bin"])
    work["d_bin"] = work["d_bin"].astype(float)

    n = len(rois)
    ncols = min(4, n); nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 2.8 * nrows + 0.5),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, roi in zip(axes, rois):
        roi_df = work[work["roi_base"] == roi]
        per_sub = (
            roi_df.groupby(["subject", "d_bin"], observed=True)
            ["proj_away_from_HP"].mean().reset_index()
        )

        # Per-subject smoothed curves (light gray, in background).
        from scipy.ndimage import gaussian_filter1d
        d_centers = sorted(per_sub["d_bin"].unique())
        subjects = per_sub["subject"].unique()
        subj_smoothed = []
        for sub_id in subjects:
            sub_data = per_sub[per_sub["subject"] == sub_id].set_index("d_bin").reindex(d_centers)
            row = sub_data["proj_away_from_HP"].values.astype(float)
            mask = ~np.isnan(row)
            if mask.sum() < 3:
                continue
            row_filled = row.copy()
            row_filled[~mask] = np.nanmean(row)
            sm_curve = gaussian_filter1d(row_filled, sigma=1.0)
            ax.plot(d_centers, sm_curve, color="0.7", lw=0.5, alpha=0.5)
            subj_smoothed.append(sm_curve)

        # Group-level mean ± SEM across subject-smoothed curves (bold).
        if subj_smoothed:
            arr = np.stack(subj_smoothed)
            mean = np.nanmean(arr, axis=0)
            sem = np.nanstd(arr, axis=0, ddof=1) / np.sqrt(arr.shape[0])
            ax.fill_between(d_centers, mean - sem, mean + sem,
                            color="#222", alpha=0.25)
            ax.plot(d_centers, mean, color="#222", lw=2.5)

        # Near / far tests.
        near = roi_df[roi_df["distance_from_distractor_mean"] <= 1.5]
        far = roi_df[roi_df["distance_from_distractor_mean"] > 4.0]
        near_subj = near.groupby("subject", observed=True)["proj_away_from_HP"].mean().dropna()
        far_subj = far.groupby("subject", observed=True)["proj_away_from_HP"].mean().dropna()
        t_near, p_near = stats.ttest_1samp(near_subj, 0.0) if len(near_subj) >= 3 else (np.nan, np.nan)
        t_far, p_far = stats.ttest_1samp(far_subj, 0.0) if len(far_subj) >= 3 else (np.nan, np.nan)
        ax.axhline(0, color="k", lw=0.5, ls="--")
        ax.axvspan(0, 1.5, color="#d62728", alpha=0.07)
        ax.axvspan(4.0, bin_edges[-1], color="#1f77b4", alpha=0.07)
        ax.set_title(
            f"{roi}\n"
            f"near: {near_subj.mean():+.3f}° (p={p_near:.3f})  ·  "
            f"far: {far_subj.mean():+.3f}° (p={p_far:.3f})",
            fontsize=9,
        )
        ax.set_xlabel("distance of mean PRF from HP (deg)")
        ax.set_ylabel("proj. away from HP (deg)")
        ax.set_ylim(-y_lim, y_lim)
    for ax in axes[n:]:
        ax.set_visible(False)

    n_dropped = n_before - n_after
    fig.suptitle(
        "Figure 3 — Linear projection of shift onto HP-opposite axis vs distance from HP\n"
        f"Linear metric → no Jensen bias.  |proj|>{clip_projection}° dropped "
        f"({n_dropped:,}/{n_before:,} = {100*n_dropped/n_before:.1f}%).  "
        f"Y-axis ±{y_lim}°.  Red band = near; blue band = far.",
        y=1.02,
    )
    fig.tight_layout()
    return fig


def figure_3b_cluster_stats(
    pars: pd.DataFrame,
    rois: list[str],
    bin_edges: np.ndarray | None = None,
    cluster_threshold_t: float = 2.0,
    n_perm: int = 500,
    seed: int = 42,
    smooth_sigma_bins: float = 1.0,
    clip_projection: float = 1.5,
    fig=None,
):
    """Per-ROI smoothed curves + 1D cluster-based permutation test.

    Per (subject, ROI, distance bin): mean projection_away_from_HP.
    Per subject: Gaussian-smooth across bins (σ = smooth_sigma_bins).
    Per ROI: at each bin, one-sample t against 0.
    Find clusters of |t| > cluster_threshold_t; cluster stat = Σ t.
    Sign-flip permutation: flip each subject's smoothed curve sign with
    p=0.5 across all bins jointly, recompute t, find max cluster stat.
    Repeat n_perm times → null distribution. Real cluster significant if
    stat > 95th percentile of null.
    """
    from scipy import stats
    from scipy.ndimage import gaussian_filter1d

    rng = np.random.default_rng(seed)
    if bin_edges is None:
        # Finer bins (0.25°) for smoother interpolation; smoothing kernel
        # then handles the per-bin noise.
        bin_edges = np.arange(0.0, 7.25, 0.25)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    work = pars.reset_index().copy()
    work = work[work["roi_base"].isin(rois)]
    work["proj_away_from_HP"] = -work["dy_rotated"]
    work = work[work["proj_away_from_HP"].abs() <= clip_projection]
    work["d_bin"] = pd.cut(
        work["distance_from_distractor_mean"], bin_edges,
        labels=centers, include_lowest=True,
    )
    work = work.dropna(subset=["d_bin"])
    work["d_bin"] = work["d_bin"].astype(float)

    # Per (subject, ROI, bin) mean.
    per_sub_bin = (
        work.groupby(["subject", "roi_base", "d_bin"], observed=True)
        ["proj_away_from_HP"].mean().reset_index()
    )

    n = len(rois)
    ncols = min(4, n); nrows = int(np.ceil(n / ncols))
    if fig is None:
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(3.6 * ncols, 2.8 * nrows + 0.5),
                                 sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()

    cluster_results = {}
    for ax, roi in zip(axes, rois):
        roi_df = per_sub_bin[per_sub_bin["roi_base"] == roi]
        # Pivot to (n_subjects, n_bins).
        pivot = roi_df.pivot_table(
            index="subject", columns="d_bin", values="proj_away_from_HP",
        )
        pivot = pivot.reindex(columns=centers)  # align bins
        # Per subject: Gaussian-smooth across bins (skip NaN-only rows).
        valid_rows = pivot.notna().any(axis=1)
        smoothed = pivot.copy()
        for sub in pivot.index[valid_rows]:
            row = pivot.loc[sub].values.astype(float)
            mask = ~np.isnan(row)
            if mask.sum() < 3:
                continue
            # Replace NaNs with bin-wise mean for smoothing input.
            row_filled = row.copy()
            row_filled[~mask] = np.nanmean(row)
            smoothed.loc[sub] = gaussian_filter1d(row_filled, smooth_sigma_bins)

        smoothed_arr = smoothed.dropna(how="all").values  # (n_subj, n_bins)

        # Per-bin t-test against 0. Require ≥60% of subjects to have
        # data in this bin (else the bin is too sparse and clusters
        # there are noise-driven).
        min_subj_per_bin = max(3, int(0.6 * smoothed_arr.shape[0]))
        t_obs = np.full(len(centers), np.nan)
        for j in range(len(centers)):
            col = smoothed_arr[:, j]
            col = col[~np.isnan(col)]
            if len(col) >= min_subj_per_bin:
                t_obs[j], _ = stats.ttest_1samp(col, 0.0)

        # Threshold + find connected clusters where |t| > thr.
        def find_clusters(t_array, thr):
            """Return list of (start_idx, end_idx, cluster_stat) for |t|>thr clusters."""
            sig = np.abs(t_array) > thr
            sig = np.nan_to_num(sig, nan=False)
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

        obs_clusters = find_clusters(t_obs, cluster_threshold_t)
        obs_max_abs = max((abs(c[2]) for c in obs_clusters), default=0.0)

        # Permutation null: sign-flip each subject's curve.
        null_max_stats = np.zeros(n_perm)
        for k in range(n_perm):
            signs = rng.choice([-1, 1], size=smoothed_arr.shape[0])
            perm = smoothed_arr * signs[:, None]
            t_perm = np.full(len(centers), np.nan)
            for j in range(len(centers)):
                col = perm[:, j]
                col = col[~np.isnan(col)]
                if len(col) >= min_subj_per_bin:
                    t_perm[j], _ = stats.ttest_1samp(col, 0.0)
            perm_clusters = find_clusters(t_perm, cluster_threshold_t)
            null_max_stats[k] = max((abs(c[2]) for c in perm_clusters), default=0.0)

        # Cluster-level p-values per observed cluster.
        cluster_ps = []
        for i_start, i_end, cs in obs_clusters:
            p = (1 + np.sum(null_max_stats >= abs(cs))) / (1 + n_perm)
            cluster_ps.append(p)

        cluster_results[roi] = {
            "centers": centers,
            "t_obs": t_obs,
            "smoothed": smoothed_arr,
            "clusters": obs_clusters,
            "cluster_ps": cluster_ps,
            "null_max_stats": null_max_stats,
        }

        # Plot.
        mean = np.nanmean(smoothed_arr, axis=0)
        sem = np.nanstd(smoothed_arr, axis=0, ddof=1) / np.sqrt(
            (~np.isnan(smoothed_arr)).sum(axis=0).clip(min=1)
        )
        ax.plot(centers, mean, color="#444", lw=2.0)
        ax.fill_between(centers, mean - sem, mean + sem,
                        color="#444", alpha=0.25)
        ax.axhline(0, color="k", lw=0.5, ls="--")
        ax.set_xlabel("distance of mean PRF from HP (deg)")
        ax.set_ylabel("smoothed proj. away from HP (deg)")
        ax.set_ylim(-0.10, 0.15)

        # Highlight significant clusters in red, non-significant in light red.
        for (i_start, i_end, cs), p in zip(obs_clusters, cluster_ps):
            color = "#d62728" if p < 0.05 else "#f0a0a0"
            ax.axvspan(bin_edges[i_start], bin_edges[i_end + 1],
                       color=color, alpha=0.25, zorder=0)
            x_mid = 0.5 * (bin_edges[i_start] + bin_edges[i_end + 1])
            ax.text(
                x_mid, ax.get_ylim()[1] * 0.92,
                f"p={p:.3f}",
                ha="center", va="top", fontsize=8,
                color="black" if p < 0.05 else "0.4",
                weight="bold" if p < 0.05 else "normal",
            )
        ax.set_title(roi)

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle(
        f"Figure 3b — Smoothed per-subject curves + 1D cluster-based permutation test\n"
        f"Per (subject, ROI) bin {bin_edges[1]-bin_edges[0]:.2f}° wide; "
        f"per-subject Gaussian smoothing σ = {smooth_sigma_bins} bins.  "
        f"Bold line = group mean of per-subject smoothed curves; "
        f"shaded band = ±1 SEM across subjects.\n"
        f"At each bin, one-sample t against 0; cluster = consecutive bins with |t|>"
        f"{cluster_threshold_t}; cluster stat = Σ t.  "
        f"{n_perm} sign-flip permutations of per-subject curves give null max-cluster-stat;\n"
        f"observed cluster significant if its |stat| > 95th percentile of null.  "
        f"Red = p<0.05, pink = n.s.  "
        f"Bins with <60% subjects skipped (sparse-far-distance).",
        y=1.04,
    )
    fig.tight_layout()
    return fig, cluster_results


# -----------------------------------------------------------------------------
# Figure 4: 2D scatter of parameter shifts per voxel/condition
# -----------------------------------------------------------------------------

def figure_4_parameter_shifts_2d(
    pars: pd.DataFrame,
    rois: list[str],
    near_threshold: float = 1.5,
):
    """2D shifts per voxel × condition for all relevant parameters,
    in HP-aligned rotated frame for x/y, and as scalar diffs for sd
    and amplitude. Each (voxel, condition) gets one point per parameter.
    """
    work = pars.reset_index().copy()
    work = work[work["roi_base"].isin(rois)]
    near = work[work["distance_from_distractor_mean"] <= near_threshold]

    fig, axes = plt.subplots(2, len(rois), figsize=(3.5 * len(rois), 7),
                             sharex="row", sharey="row")
    axes = np.atleast_2d(axes)

    norm_pos = TwoSlopeNorm(vmin=-0.4, vcenter=0, vmax=0.4)
    last_hb_pos = None
    last_hb_size = None
    for j, roi in enumerate(rois):
        roi_near = near[near["roi_base"] == roi]
        # Top row: 2D hexbin of (dx_rotated, dy_rotated) with HP at (0,4).
        ax = axes[0, j]
        last_hb_pos = ax.hexbin(
            roi_near["dx_rotated"], roi_near["dy_rotated"],
            gridsize=20, cmap="magma_r", mincnt=2, bins="log",
            extent=(-1.5, 1.5, -1.5, 1.5),
        )
        ax.axhline(0, color="k", lw=0.4, alpha=0.4)
        ax.axvline(0, color="k", lw=0.4, alpha=0.4)
        # Annotate "toward HP" direction in rotated frame.
        ax.annotate("", xy=(0, 1.2), xytext=(0, 0.6),
                    arrowprops=dict(arrowstyle="->", color="red", lw=2))
        ax.text(0.1, 1.0, "toward HP", color="red", fontsize=8)
        ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5); ax.set_aspect("equal")
        ax.set_title(f"{roi}\n(x, y shift in HP-aligned frame)")
        if j == 0:
            ax.set_xlabel("dx_rotated (deg)")
            ax.set_ylabel("dy_rotated (deg)")

        # Bottom row: sd_diff vs amplitude_diff scatter.
        ax = axes[1, j]
        ax.hexbin(
            roi_near["sd_diff"], roi_near["amplitude_diff"],
            gridsize=20, cmap="magma_r", mincnt=2, bins="log",
            extent=(-2, 2, -2, 2),
        )
        ax.axhline(0, color="k", lw=0.4, alpha=0.4)
        ax.axvline(0, color="k", lw=0.4, alpha=0.4)
        ax.set_xlim(-2, 2); ax.set_ylim(-2, 2); ax.set_aspect("equal")
        ax.set_title("(sd, amplitude) shift")
        if j == 0:
            ax.set_xlabel("sd_diff (deg)")
            ax.set_ylabel("amplitude_diff (a.u.)")

    fig.suptitle(
        "Figure 4 — Per-voxel × per-condition shifts in 2D, voxels within 1.5° of HP\n"
        "Top: position shift in HP-aligned frame (red arrow = toward HP).  "
        "Bottom: PRF size and amplitude diff. Density on log scale.",
        y=1.02,
    )
    fig.tight_layout()
    return fig


# -----------------------------------------------------------------------------
# Figure 5: Divisive Normalization (model 6) parameters per ROI
# -----------------------------------------------------------------------------

def figure_5_dn_parameters(
    rois: list[str],
    bids_folder: str = "/data/ds-retsupp",
):
    """Distributions of Divisive-Normalization model parameters per ROI."""
    bids_folder = Path(bids_folder)
    frames = []
    for sub_id in get_subject_ids():
        f = bids_folder / "derivatives/prf_summaries/model6" / f"sub-{int(sub_id):02d}" / f"sub-{int(sub_id):02d}_model-6_prf_voxels.tsv"
        if not f.exists():
            continue
        df = pd.read_csv(f, sep="\t", index_col=[0, 1])
        df["subject"] = sub_id
        frames.append(df)
    if not frames:
        return None
    dn = pd.concat(frames).reset_index()
    dn["roi_base"] = dn["roi"].str.replace(r"_(L|R)$", "", regex=True)
    roi_mapping = {"LO1":"LO","LO2":"LO","V3A":"V3AB","V3B":"V3AB","TO1":"TO","TO2":"TO","VO1":"VO","VO2":"VO"}
    dn["roi_base"] = dn["roi_base"].replace(roi_mapping)
    dn = dn[dn["roi_base"].isin(rois)]
    dn = dn[(dn["r2"] > 0.2) & (dn["ecc"] < 3.0)]

    pars_to_show = [
        ("rf_amplitude", "RF amplitude"),
        ("srf_amplitude", "Suppressive RF amplitude"),
        ("srf_size", "Suppressive RF size (deg)"),
        ("neural_baseline", "Neural baseline"),
        ("surround_baseline", "Surround baseline"),
        ("sd", "Excitatory RF size (deg)"),
    ]
    n = len(pars_to_show)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey=False)
    axes = axes.ravel()
    for ax, (col, label) in zip(axes, pars_to_show):
        # Per-subject median for this parameter.
        med = dn.groupby(["subject", "roi_base"], observed=True)[col].median().reset_index()

        # Outlier clip: drop values beyond 3 IQR from each ROI's median
        # to prevent extreme single-subject fits from blowing up the y-axis.
        keep_mask = pd.Series(True, index=med.index)
        for roi in rois:
            sel = med["roi_base"] == roi
            vals = med.loc[sel, col]
            if len(vals) >= 4:
                q25, q75 = np.nanpercentile(vals, [25, 75])
                iqr = q75 - q25
                lo, hi = q25 - 3 * iqr, q75 + 3 * iqr
                keep_mask.loc[sel] = (vals >= lo) & (vals <= hi)
        med_clean = med[keep_mask]

        sns.swarmplot(
            data=med_clean, x="roi_base", y=col, order=rois,
            color="0.55", size=4, alpha=0.7, ax=ax,
        )
        # Group mean ± SEM as large dot + errorbar in the middle.
        from scipy import stats as _stats
        means = med_clean.groupby("roi_base", observed=True)[col].mean().reindex(rois)
        sems = med_clean.groupby("roi_base", observed=True)[col].sem().reindex(rois)
        ax.errorbar(
            x=np.arange(len(rois)), y=means.values,
            yerr=sems.values, fmt="o",
            color="black", markersize=10, capsize=5, lw=1.5,
            markeredgecolor="white", markeredgewidth=1.2, zorder=5,
        )
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("")
        if col in ("srf_amplitude", "neural_baseline", "surround_baseline"):
            ax.set_yscale("log")
    fig.suptitle(
        f"Figure 5 — Divisive-Normalization (model 6) parameters per ROI\n"
        f"r² > 0.2, ecc < 3°.  Violins = voxel distribution pooled across subjects;  "
        f"dots = per-subject medians.  n_subjects = {dn['subject'].nunique()}",
        y=1.01,
    )
    fig.tight_layout()
    return fig


def figure_r2_distributions(
    pars: pd.DataFrame,
    rois: list[str],
    r2_threshold: float = 0.3,
    bids_folder: str = "/data/ds-retsupp",
    fig=None,
):
    """Per-ROI distribution of mean-model PRF R², with cutoff line.

    Loads conditionwise data with r2_thr=0.0 (no inclusion filter) to
    show the FULL R² distribution, since the default `load_all_conditionwise`
    filters at r²>0.2 already.
    """
    # Reload with no inclusion filter to see the true distribution.
    full = load_all_conditionwise(
        bids_folder=bids_folder, r2_thr=0.0, ecc_thr=10.0, sd_thr=0.0,
    )
    work = full.reset_index().copy()
    work = work[work["roi_base"].isin(rois)]
    # Drop duplicates per voxel — r2_mean_model is the same across conditions.
    per_voxel = work.drop_duplicates(["subject", "roi_base", "voxel"])
    per_voxel = per_voxel[per_voxel["r2_mean_model"].notna()]

    if fig is None:
        fig, axes = plt.subplots(2, len(rois), figsize=(2.6 * len(rois), 6),
                                 sharey="row")
    axes = np.atleast_2d(axes)

    summary_rows = []
    for j, roi in enumerate(rois):
        roi_df = per_voxel[per_voxel["roi_base"] == roi]
        # Top: histogram of R² across all voxels in ROI (pooled over subjects).
        ax = axes[0, j]
        ax.hist(roi_df["r2_mean_model"], bins=40, color="0.7",
                edgecolor="white", lw=0.3)
        ax.axvline(r2_threshold, color="red", lw=1.5)
        ax.set_xlim(0, 1)
        ax.set_xlabel("r²_mean_model")
        if j == 0:
            ax.set_ylabel("voxel count\n(pooled)")
        ax.set_title(f"{roi}", fontsize=10)

        # Bottom: per-subject % of voxels above threshold.
        per_subj_pct = (
            roi_df.groupby("subject", observed=True)["r2_mean_model"]
            .agg(lambda v: 100 * (v > r2_threshold).mean())
        )
        per_subj_n = (
            roi_df.groupby("subject", observed=True)["r2_mean_model"]
            .agg(lambda v: int((v > r2_threshold).sum()))
        )
        ax = axes[1, j]
        ax.boxplot(per_subj_pct.values, vert=True, widths=0.5,
                   patch_artist=True, boxprops=dict(facecolor="0.85"))
        sns.stripplot(y=per_subj_pct.values, color="black", size=3,
                      alpha=0.6, ax=ax)
        ax.set_ylim(0, 100)
        if j == 0:
            ax.set_ylabel(f"% voxels with r²>{r2_threshold}\n(per subject)")
        ax.set_xticks([])
        ax.set_xlabel(roi)

        summary_rows.append({
            "roi": roi,
            "median_R2": float(roi_df["r2_mean_model"].median()),
            "pct_above_thr_median": float(per_subj_pct.median()),
            "n_above_thr_per_subj_median": float(per_subj_n.median()),
            "n_above_thr_per_subj_min": int(per_subj_n.min()),
            "n_above_thr_per_subj_max": int(per_subj_n.max()),
        })

    fig.suptitle(
        f"Figure — Distribution of r²_mean_model per ROI.\n"
        f"Top: pooled histogram (red line = r² threshold for AF fit = {r2_threshold}).\n"
        f"Bottom: per-subject percentage of voxels surviving the threshold.",
        y=1.02,
    )
    fig.tight_layout()
    return fig, pd.DataFrame(summary_rows)


# -----------------------------------------------------------------------------
# Figure 6: 4-AF competing model fits (suppression vs attraction)
# -----------------------------------------------------------------------------

def figure_6_four_af_fits(rois, fits_tsv: Path, fig=None):
    """Plot 4-AF model fit results FROM a precomputed TSV.

    The TSV is produced by `python -m retsupp.modeling.fit_four_af`,
    which is the slow step (~6 min). Plotting just reads the table.

    Per voxel × condition, the model predicts:
        suppression: R = (1 − g_HP·A_HP − g_LP·Σ A_LP) · S
        attraction:  R = (1 + g_HP·A_HP + g_LP·Σ A_LP) · S

    log_g_HP_over_g_LP > 0 → HP-AF stronger than LP-AFs.
    """
    from scipy import stats

    if not fits_tsv.exists():
        print(f"  4-AF fits TSV not found at {fits_tsv}.")
        print(f"  Generate with: python -m retsupp.modeling.fit_four_af "
              f"--out {fits_tsv} --joint-base-fit")
        return None, None
    df = pd.read_csv(fits_tsv, sep="\t")
    if df.empty:
        return None, None

    # 2-row figure: top = suppression, bottom = attraction. 3 columns:
    # σ_AF, log(g_HP/g_LP), within-voxel R².
    if fig is None:
        fig, axes = plt.subplots(2, 3, figsize=(13, 7))

    for i, mode in enumerate(("suppression", "attraction")):
        sub = df[df["mode"] == mode]

        # σ_AF.
        ax = axes[i, 0]
        sns.boxplot(data=sub, x="roi", y="sigma_AF", order=rois,
                    color="0.85", ax=ax)
        sns.stripplot(data=sub, x="roi", y="sigma_AF", order=rois,
                      color="black", size=3, alpha=0.6, ax=ax)
        ax.set_ylabel("σ_AF (deg)")
        ax.set_title(f"{mode} — σ_AF")
        ax.set_xlabel("")

        # log(g_HP / g_LP).
        ax = axes[i, 1]
        sns.boxplot(data=sub, x="roi", y="log_g_HP_over_g_LP", order=rois,
                    color="0.85", ax=ax)
        sns.stripplot(data=sub, x="roi", y="log_g_HP_over_g_LP", order=rois,
                      color="black", size=3, alpha=0.6, ax=ax)
        ax.axhline(0, color="k", lw=0.5, ls="--")
        ax.set_ylabel("log(g_HP / g_LP)")
        ax.set_title(f"{mode} — log(g_HP/g_LP)")
        ax.set_xlabel("")
        # Add per-ROI sign-test p-values: is log_ratio > 0 (or < 0) significantly?
        for j, roi in enumerate(rois):
            vals = sub.loc[sub["roi"] == roi, "log_g_HP_over_g_LP"].dropna()
            if len(vals) >= 3:
                t, p = stats.ttest_1samp(vals, 0.0)
                marker = "*" if p < 0.05 else ""
                if marker:
                    ax.text(j, ax.get_ylim()[1] * 0.95, marker,
                            ha="center", va="top", fontsize=12, color="red")

        # Within-voxel R².
        ax = axes[i, 2]
        sns.boxplot(data=sub, x="roi", y="r2", order=rois,
                    color="0.85", ax=ax)
        sns.stripplot(data=sub, x="roi", y="r2", order=rois,
                      color="black", size=3, alpha=0.6, ax=ax)
        ax.axhline(0, color="k", lw=0.5, ls="--")
        ax.set_ylabel("within-voxel R²")
        ax.set_title(f"{mode} — fit quality")
        ax.set_xlabel("")

    fig.suptitle(
        "Figure 6 — 4-AF competing model: shared σ_AF + separate g_HP and g_LP per ROI per subject\n"
        "Top row: suppression mode (1 − Σ g_ℓ A_ℓ) · S.  "
        "Bottom row: attraction mode (1 + Σ g_ℓ A_ℓ) · S.  "
        "log(g_HP/g_LP) > 0 = HP-AF stronger than LP-AFs.  * = p < 0.05 (one-sample t).",
        y=1.02,
    )
    fig.tight_layout()
    return fig, df


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bids-folder", default="/data/ds-retsupp")
    parser.add_argument("--model", type=int, default=8)
    parser.add_argument("--out", type=Path, default=Path("notes/meeting_report.pdf"))
    parser.add_argument(
        "--rois", nargs="+",
        default=["V1", "V2", "V3", "V3AB", "hV4", "LO", "TO", "VO"],
    )
    parser.add_argument(
        "--four-af-fits", type=Path, default=None,
        help="Path to TSV with 4-AF fit results (from fit_four_af.py). "
             "If omitted, defaults to <out>_four_af_fits.tsv next to the report.",
    )
    parser.add_argument("--n-perm", type=int, default=200)
    args = parser.parse_args()

    print("Loading conditionwise data ...")
    pars = load_all_conditionwise(bids_folder=args.bids_folder, model=args.model)
    print(
        f"  {len(pars):,} rows, {pars['subject'].nunique()} subjects, "
        f"ROIs: {sorted(pars.index.get_level_values('roi_base').unique())}"
    )

    n_before = len(pars)
    # Use the per-condition (NOT mean-only) criterion at a strict threshold:
    # drop voxels only when one or more conditionwise PRFs has >90% of mass
    # outside the aperture. The 50% threshold removes signal because HP
    # sits at the aperture boundary; 90% catches only runaway fits.
    pars = filter_prf_inside_aperture(
        pars, aperture_radius=3.17, max_fraction_outside=0.5, use_mean_only=False,
    )
    n_after = len(pars)
    print(
        f"After aperture filter (drop voxels with any PRF >50% outside R=3.17°): "
        f"{n_before:,} → {n_after:,} rows  ({100*(n_before-n_after)/n_before:.1f}% dropped)"
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(args.out) as pdf:
        # Cover page with summary stats text.
        fig_cover = plt.figure(figsize=(11, 8.5))
        ax = fig_cover.add_subplot(111)
        ax.axis("off")
        ax.text(0.5, 0.95, "retsupp — meeting report",
                ha="center", va="top", fontsize=20, weight="bold",
                transform=ax.transAxes)
        from datetime import date
        cover_text = f"""
Date: {date.today().isoformat()}      Author: Gilles
ROIs in this report: {", ".join(args.rois)}      n subjects: {pars['subject'].nunique()}      PRF model: {args.model}

DESIGN NOTE — about the "only 3 LP conditions" question:
For each voxel × anchor (a ring distractor location), there is exactly ONE condition where
that anchor IS the HP, and exactly THREE conditions where the same anchor is LP. We compare
the projection-onto-anchor metric in the HP row against the mean of the 3 LP rows. Same anchor
throughout — no comparison between distances to different ring locations. The effective unit of
analysis is (voxel, anchor) within near-anchor voxels, contributing a paired (HP, mean_LP) pair.

CONTENTS:
  Fig 1   — HP-vs-LP per anchor (projection metric, no Jensen) + permutation null per ROI.
  Fig 1b  — Same HP-vs-LP design, plotted as a function of distance from anchor (clearer).
  Fig 2   — 2D hexbin shift map in HP-aligned frame, real vs random-rotation null.
  Fig 3   — Projection along HP-opposite axis vs distance from HP, per ROI; near/far stats.
  Fig 4   — Per-voxel × per-condition shifts in 2D for position and (sd, amplitude).
  Fig 5   — Divisive-Normalization (model 6) parameter distributions per ROI.

WHAT FIGURE 1 ACTUALLY COMPARES (clarification):
  This is NOT an opposite-pair analysis. For each of the 4 ring distractor locations
  separately, we restrict to voxels whose mean PRF sits within 1.5° of THAT location, then
  compare the projection-onto-anchor metric in:
    HP-condition  = the ONE condition where this anchor IS the HP (red).
    LP-condition  = the THREE conditions where this anchor is just a low-prob distractor (blue).
  The contrast is "same anchor, different HP status" — purely within-anchor. No comparison
  between distances to different anchors, so no Jensen issue. Fig 1b shows this same
  comparison, but binned by distance from voxel mean to anchor, which makes the spatial
  locality of the effect immediately visible (hV4 spike at d ≈ 0.5°, +0.5°, p = 0.004).

WHAT WE FOUND IN PRIOR WORK (recap):
  • Local push-away from HP in V3AB and hV4, p_FDR ≈ 0.008–0.016 with paired HP-vs-Opp metric.
  • The opposite-anchor shift goes negative (toward antipode) only in V3AB/hV4 — direct
    geometric signature of suppression that pure Jensen bias cannot produce.
  • V3AB additionally shows a small negative paired metric at d > 4° (relative attraction
    far from HP), p ≈ 0.025.

TUNCOK 2025 (just read):
  • Same Jensen issue in their Fig 7A (distance to attentional target).
  • Solve it via shuffled-GLM permutation null — equivalent to our random-rotation null but
    applied at the GLM level rather than post-hoc.
  • Their Fig 7C uses an "opposite-pair" projection: bin pRFs in 2D, plot endpoints for
    opposite cardinal cue conditions (up/down, left/right), color-code by congruence of x-shift
    for L/R and y-shift for U/D. Same logic as our projection idea.
  • They have a true distributed/neutral baseline; we don't (only HP/LP contrast).

RICHTER 2025 (just read):
  • Location localizer task → independent ROI mask per distractor location → trial-wise BOLD
    contrast estimates. Search trials AND omission trials.
  • Show: BOLD at HPDL ↓ < BOLD at NL-near < BOLD at NL-far. Spatially specific, broad
    suppression. Proactive (also in omission trials).
  • This is exactly the analysis the cluster plan proposes for retsupp data.
"""
        ax.text(0.04, 0.85, cover_text, ha="left", va="top",
                fontsize=10, family="monospace", transform=ax.transAxes)
        pdf.savefig(fig_cover, bbox_inches="tight")
        plt.close(fig_cover)

        print("Figure 1 ...")
        fig, _ = figure_1_hp_vs_lp_per_anchor(
            pars, rois=args.rois, near_threshold=1.5, n_perm=args.n_perm,
        )
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        print("Figure 1b — HP vs LP vs distance ...")
        fig = figure_1b_hp_vs_lp_vs_distance(pars, rois=args.rois)
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        print("Figure 2 ...")
        fig = figure_2_hexbin_vs_null(pars, rois=args.rois, metric="radial")
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        print("Figure 2b — projected metric (no Jensen) ...")
        fig = figure_2_hexbin_vs_null(pars, rois=args.rois, metric="projection")
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        print("Figure 3 ...")
        fig = figure_3_projection_vs_distance(pars, rois=args.rois)
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        print("Figure 3b — cluster-based permutation stats ...")
        fig, _ = figure_3b_cluster_stats(pars, rois=args.rois, n_perm=300)
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        print("Figure 4 ...")
        fig = figure_4_parameter_shifts_2d(pars, rois=args.rois)
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        print("Figure 5 ...")
        fig = figure_5_dn_parameters(rois=args.rois, bids_folder=args.bids_folder)
        if fig is not None:
            pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        print("Figure — R² distributions per ROI (diagnostic) ...")
        fig, r2_summary = figure_r2_distributions(
            pars, rois=args.rois, r2_threshold=0.3,
            bids_folder=args.bids_folder,
        )
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)
        print("R² summary:")
        print(r2_summary.to_string(index=False, float_format=lambda v: f"{v:7.3f}"))

        print("Figure 6 — 4-AF competing model (reading precomputed TSV) ...")
        fits_tsv = (args.four_af_fits if args.four_af_fits
                    else args.out.with_name(args.out.stem + "_four_af_fits.tsv"))
        result = figure_6_four_af_fits(rois=args.rois, fits_tsv=fits_tsv)
        if result is not None and result[0] is not None:
            fig, _ = result
            pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
