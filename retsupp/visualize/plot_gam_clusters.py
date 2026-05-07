#!/usr/bin/env python3
"""
Hierarchical Bayesian GAM analysis: per-ROI smooth of projection-away-from-HP
vs distance from HP.

For each ROI in {V1, V2, V3, V3AB, hV4, LO, TO, VO}, we fit a SEPARATE bambi
GAM (no joint master model). The model is:

    proj ~ hsgp(distance, m=8, c=1.5) + (1 | subject)

where `hsgp(...)` is a Hilbert-space Gaussian-process approximation that
plays the role of `s(distance, k=8)` in mgcv. Each ROI gets its own posterior
trace; we compute the posterior median + 95% HDI of the smooth at a fine
distance grid (50 points from 0.5 to 6.5°) and flag distances where
P(smooth > 0) > 0.95 or P(smooth < 0) > 0.95.

The data pipeline mirrors `plot_projection_clusters.py`:
  * `load_all_conditionwise(r2_thr=0.2)` → conditionwise PRF parameters.
  * `filter_prf_inside_aperture(max_fraction_outside=0.5)` → drop voxels
    with too much PRF mass outside the aperture.
  * Drop voxel-condition rows with |proj_away_from_HP| > 1.5 (outlier clip).
  * Aggregate per (subject, distance-bin): mean of proj_away_from_HP across
    voxel-condition rows in that cell. Bins are 0.5° wide, centers from
    0.25° to 6.75°.

Run:
    /Users/gdehol/mambaforge/envs/tms_risk/bin/python \\
        /Users/gdehol/git/retsupp/retsupp/visualize/plot_gam_clusters.py
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

# We expect to be run from the `tms_risk` env (which has bambi/pymc but does
# NOT have `retsupp` pip-installed). Add the project root to sys.path so the
# retsupp.* imports work.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

import arviz as az
import bambi as bmb

from retsupp.visualize.vss2026_arrows import load_all_conditionwise
from retsupp.visualize.meeting_report import filter_prf_inside_aperture


DEFAULT_ROIS = ["V1", "V2", "V3", "V3AB", "hV4", "LO", "TO", "VO"]


def _make_hsgp_priors(ell_mu_log: float = float(np.log(2.0)),
                       ell_sigma_log: float = 0.5):
    """Build a slightly-informative prior dict for the HSGP term.

    The length-scale `ell` (in degrees) is given a LogNormal prior centered
    around exp(ell_mu_log) ≈ 2°, the natural smoothness for the
    projection-vs-distance smooth. `sigma` (the GP marginal SD) keeps the
    bambi default Exponential(1.0).
    """
    return {
        "ell": bmb.Prior("LogNormal", mu=ell_mu_log, sigma=ell_sigma_log),
        "sigma": bmb.Prior("Exponential", lam=1.0),
    }


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------

def build_subject_bin_table(
    pars: pd.DataFrame,
    rois: list[str],
    bin_edges: np.ndarray,
    clip_projection: float,
) -> pd.DataFrame:
    """Build per-(subject, ROI, distance-bin) aggregated DataFrame.

    Columns: subject, roi_base, distance (bin center), proj (mean across
    voxel-condition rows in that cell), n (count).
    """
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    work = pars.reset_index().copy()
    work = work[work["roi_base"].isin(rois)].copy()
    work["proj_away_from_HP"] = -work["dy_rotated"]
    n_before = len(work)
    work = work[work["proj_away_from_HP"].abs() <= clip_projection]
    n_after = len(work)
    print(
        f"  Outlier clip |proj|>{clip_projection}°: "
        f"{n_before:,} → {n_after:,} rows  "
        f"({100*(n_before-n_after)/n_before:.1f}% dropped)"
    )

    work["d_bin"] = pd.cut(
        work["distance_from_distractor_mean"],
        bin_edges, labels=centers, include_lowest=True,
    )
    work = work.dropna(subset=["d_bin"])
    work["d_bin"] = work["d_bin"].astype(float)

    agg = (
        work.groupby(["subject", "roi_base", "d_bin"], observed=True)
        ["proj_away_from_HP"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "proj", "count": "n", "d_bin": "distance"})
    )
    return agg


# ---------------------------------------------------------------------------
# GAM fit + prediction
# ---------------------------------------------------------------------------

def fit_gam_per_roi(
    df_roi: pd.DataFrame,
    *,
    family: str = "gaussian",
    m: int = 8,
    c: float = 1.5,
    draws: int = 2000,
    tune: int = 2000,
    chains: int = 4,
    target_accept: float = 0.99,
    seed: int = 42,
    ell_mu_log: float = float(np.log(2.0)),
    ell_sigma_log: float = 0.5,
    nuts_sampler: str = "pymc",
    init: str = "auto",
    model_form: str = "population",
):
    """Fit GAM `proj ~ hsgp(distance, m=m, c=c) + (1 | subject)` for one ROI.

    The HSGP length-scale `ell` is given a slightly-informative LogNormal
    prior centered around ~2° (the natural smoothness for projection vs
    distance). NUTS settings default to draws=2000, tune=2000, target_accept=
    0.99 to keep divergences low (notably for LO).

    Returns (model, idata, fit_seconds, n_divergences).
    """
    if model_form == "population":
        # Population smooth + per-subject random intercept only.
        formula = f"proj ~ hsgp(distance, m={m}, c={c}) + (1|subject)"
        hsgp_term_name = f"hsgp(distance, m={m}, c={c})"
        priors = {
            hsgp_term_name: _make_hsgp_priors(
                ell_mu_log=ell_mu_log, ell_sigma_log=ell_sigma_log,
            ),
        }
    elif model_form == "subjectwise":
        # Population smooth + per-subject smooth deviations + per-subj
        # random intercept.  Both HSGP terms get the same length-scale
        # prior; the by=subject term uses share_cov so per-subject
        # smooths share GP hyperparameters (partial pooling on smoothness).
        formula = (
            f"proj ~ hsgp(distance, m={m}, c={c}) "
            f"+ hsgp(distance, m={m}, c={c}, by=subject, share_cov=True) "
            f"+ (1|subject)"
        )
        priors = {
            f"hsgp(distance, m={m}, c={c})": _make_hsgp_priors(
                ell_mu_log=ell_mu_log, ell_sigma_log=ell_sigma_log,
            ),
            f"hsgp(distance, m={m}, c={c}, by=subject, share_cov=True)":
                _make_hsgp_priors(
                    ell_mu_log=ell_mu_log,
                    ell_sigma_log=ell_sigma_log,
                ),
        }
    else:
        raise ValueError(f"Unknown model_form={model_form!r}")
    model = bmb.Model(formula, df_roi, family=family, priors=priors)
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        fit_kwargs = dict(
            draws=draws, tune=tune, chains=chains,
            target_accept=target_accept, random_seed=seed,
            progressbar=True,
        )
        # nutpie / numpyro / blackjax handle initialization themselves.
        # Only pass `init` when using PyMC's NUTS, where it controls the
        # mass-matrix adaptation strategy ("jitter+adapt_full" estimates a
        # full off-diagonal mass matrix during tuning, far better than the
        # default diagonal one for HSGP-style correlated posteriors).
        if nuts_sampler != "pymc":
            fit_kwargs["nuts_sampler"] = nuts_sampler
        else:
            fit_kwargs["init"] = init
        idata = model.fit(**fit_kwargs)
    fit_secs = time.perf_counter() - t0
    n_div = int(idata.sample_stats["diverging"].sum().values) if "diverging" in idata.sample_stats else 0
    return model, idata, fit_secs, n_div


def evaluate_smooth(
    model: bmb.Model,
    idata,
    df_roi: pd.DataFrame,
    distances: np.ndarray,
):
    """Return posterior of the marginal smooth f(distance) at the supplied grid.

    We construct a synthetic "new data" frame at each distance, with subject
    set to the first level (any value will do; the group-specific term is
    excluded from the prediction) and call `model.predict(..., kind="mean",
    include_group_specific=False)`. The returned posterior contains a
    "<response>_mean" array of shape (chain, draw, n_obs) which is the
    population-level posterior of the mean response — exactly the smooth
    we want, since the model is `proj ~ s(distance) + (1|subj)` and the
    fixed intercept is included.
    """
    # Subject is required for the formula to evaluate; pick any existing one.
    sub_dummy = df_roi["subject"].iloc[0]
    new_df = pd.DataFrame({
        "distance": distances,
        "subject": sub_dummy,
        # Response column may be required by formulae/predict for new-data
        # design-matrix evaluation. Use a finite dummy; it's not used in
        # the linear predictor for kind="mean".
        "proj": 0.0,
    })
    pred_idata = model.predict(
        idata, data=new_df, kind="mean",
        include_group_specific=False, inplace=False,
    )
    # bambi names the predicted mean "<response>_mean".
    mean_var = "proj_mean"
    if mean_var not in pred_idata.posterior.data_vars:
        # Fallback: take the only data_var that ends in "_mean".
        candidates = [v for v in pred_idata.posterior.data_vars
                      if v.endswith("_mean")]
        if not candidates:
            raise RuntimeError(
                f"Couldn't find smooth-mean variable; data_vars = "
                f"{list(pred_idata.posterior.data_vars)}"
            )
        mean_var = candidates[0]
    arr = pred_idata.posterior[mean_var].values  # (chain, draw, n_obs)
    flat = arr.reshape(-1, arr.shape[-1])  # (samples, n_obs)
    return flat


def summarize_smooth(
    samples: np.ndarray,
    distances: np.ndarray,
    hdi_prob: float = 0.95,
):
    """Per-distance posterior median, HDI, P(>0), P(<0)."""
    median = np.median(samples, axis=0)
    # arviz interprets 2D ndarray as (draw, n_obs); returns (n_obs, 2).
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        hdi = az.hdi(samples, hdi_prob=hdi_prob)
    hdi = np.asarray(hdi)
    lower = hdi[:, 0]
    upper = hdi[:, 1]
    p_pos = (samples > 0).mean(axis=0)
    p_neg = (samples < 0).mean(axis=0)
    return pd.DataFrame({
        "distance": distances,
        "median": median,
        "lower": lower,
        "upper": upper,
        "p_pos": p_pos,
        "p_neg": p_neg,
    })


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_roi_page(
    pdf: PdfPages,
    roi: str,
    df_roi: pd.DataFrame,
    summary: pd.DataFrame,
    fit_secs: float,
    n_div: int,
    n_subj: int,
    n_rows: int,
    threshold: float = 0.95,
    y_lim: float | None = 0.10,
):
    fig, ax = plt.subplots(1, 1, figsize=(8.5, 5.5))

    # Faint subject means as background dots.
    ax.scatter(
        df_roi["distance"], df_roi["proj"],
        s=8, color="0.7", alpha=0.4, zorder=0, label="per-(subj, bin) mean",
    )

    # Posterior median + 95% HDI band.
    ax.plot(summary["distance"], summary["median"],
            color="#222", lw=2.2, label="posterior median")
    ax.fill_between(summary["distance"], summary["lower"], summary["upper"],
                    color="#222", alpha=0.25, label="95% HDI")
    ax.axhline(0, color="k", lw=0.5, ls="--")

    # Highlight where P(>0) > thr (red) or P(<0) > thr (blue).
    sig_pos = summary["p_pos"] > threshold
    sig_neg = summary["p_neg"] > threshold

    def _shade(mask, color, label_prefix):
        if not mask.any():
            return
        idx = np.where(mask.values)[0]
        # Find consecutive runs.
        groups = np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)
        for k, g in enumerate(groups):
            x0 = summary["distance"].iloc[g[0]]
            x1 = summary["distance"].iloc[g[-1]]
            ax.axvspan(x0, x1, color=color, alpha=0.18, zorder=0)
            ymin = ax.get_ylim()[0]
            ax.text(
                0.5 * (x0 + x1), ymin * 0.95 if ymin < 0 else 0.0,
                f"{label_prefix} [{x0:.2f}–{x1:.2f}°]",
                ha="center", va="bottom", fontsize=8,
                color=color, weight="bold",
            )

    _shade(sig_pos, "#d62728", f"P(>0)>{threshold:.2f}")
    _shade(sig_neg, "#1f77b4", f"P(<0)>{threshold:.2f}")

    if y_lim is not None:
        # Extend a bit if HDI exceeds y_lim.
        actual_max = float(np.nanmax(np.abs(np.r_[summary["lower"], summary["upper"]])))
        eff = max(y_lim, 1.1 * actual_max)
        ax.set_ylim(-eff, eff)
    ax.set_xlabel("distance of mean PRF from HP (deg)")
    ax.set_ylabel("proj. away from HP (deg)")
    ax.set_title(
        f"{roi}  —  hierarchical Bayesian GAM on subject×bin means\n"
        f"n_subjects = {n_subj}, n_rows = {n_rows}, "
        f"fit time = {fit_secs:.1f}s, divergences = {n_div}"
    )
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def cover_page(
    pdf: PdfPages,
    args,
    bin_edges: np.ndarray,
    grid: np.ndarray,
):
    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_subplot(111); ax.axis("off")
    ax.text(
        0.5, 0.95,
        "Per-ROI hierarchical Bayesian GAM\n"
        "(projection away from HP vs distance from HP)",
        ha="center", va="top", fontsize=16, weight="bold",
        transform=ax.transAxes,
    )
    text = f"""
WHAT IS FIT (one model PER ROI — no joint master model):
  proj ~ hsgp(distance, m={args.m}, c={args.c}) + (1 | subject)

  • `hsgp(...)` is bambi's Hilbert-space Gaussian-process approximation,
    the equivalent of `s(distance, k={args.m})` in mgcv.
  • `(1 | subject)` = per-subject random intercept.
  • family = "{args.family}".
  • HSGP length-scale prior:
      ell ~ LogNormal(mu={args.ell_mu_log:.3f} = log({np.exp(args.ell_mu_log):.2f}°),
                       sigma={args.ell_sigma_log:.2f})
    (slightly informative, centered around ~{np.exp(args.ell_mu_log):.1f}°).
    HSGP marginal SD: sigma ~ Exponential(1.0).
  • Sampler: NUTS via PyMC, draws = {args.draws}, tune = {args.tune},
    chains = {args.chains}, target_accept = {args.target_accept}.

DATA PIPELINE (same as plot_projection_clusters.py):
  • load_all_conditionwise(bids_folder={args.bids_folder!r}, r2_thr={args.r2_thr}).
  • filter_prf_inside_aperture(aperture_radius=3.17,
      max_fraction_outside={args.aperture_fraction_outside},
      use_mean_only=False).
  • Outlier clip: drop voxel-condition rows with
      |proj_away_from_HP| > {args.clip_projection}°  (proj = -dy_rotated).
  • Per (subject, ROI, distance-bin): mean of proj across voxel-condition
    rows in the cell. Distance bin width = {args.bin_step}°,
    centers from {bin_edges[0] + args.bin_step/2:.2f}° to
    {bin_edges[-1] - args.bin_step/2:.2f}°.

POSTERIOR SUMMARY:
  • Smooth f(distance) is evaluated at {len(grid)} points from
    {grid[0]:.2f}° to {grid[-1]:.2f}°. Per distance:
      - posterior median + 95% HDI of the population-level mean.
      - P(f > 0): probability the smooth is > 0 at that distance.
      - P(f < 0): probability the smooth is < 0 at that distance.
  • A distance is flagged ("Bayesian cluster") when
      P(>0) > {args.threshold:.2f}  →  red shading
      P(<0) > {args.threshold:.2f}  →  blue shading

NOTES:
  • Each ROI = a separate bambi model + separate posterior trace.
  • Sampler warnings (divergences, etc.) are printed per-ROI.
  • Default seed = {args.seed}; use --seed to vary.

ROIs fit: {args.rois}
"""
    ax.text(0.04, 0.85, text, ha="left", va="top",
            fontsize=10, family="monospace", transform=ax.transAxes)
    pdf.savefig(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--bids-folder", default="/data/ds-retsupp")
    parser.add_argument(
        "--out", type=Path,
        default=Path("/Users/gdehol/git/retsupp/notes/gam_clusters.pdf"),
    )
    parser.add_argument("--rois", nargs="+", default=DEFAULT_ROIS)
    parser.add_argument("--r2-thr", type=float, default=0.20)
    parser.add_argument(
        "--aperture-fraction-outside", type=float, default=0.5,
    )
    parser.add_argument("--bin-step", type=float, default=0.5,
                        help="Distance bin width in degrees")
    parser.add_argument("--bin-min", type=float, default=0.0)
    parser.add_argument("--bin-max", type=float, default=7.0)
    parser.add_argument("--clip-projection", type=float, default=1.5)
    parser.add_argument("--m", type=int, default=8,
                        help="Number of HSGP basis functions (mgcv `k`)")
    parser.add_argument("--c", type=float, default=1.5,
                        help="HSGP boundary extension factor")
    parser.add_argument("--family", default="gaussian",
                        choices=["gaussian", "t"])
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--tune", type=int, default=2000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--target-accept", type=float, default=0.99)
    parser.add_argument("--ell-mu-log", type=float, default=float(np.log(2.0)),
                        help="LogNormal `mu` for HSGP length-scale prior "
                             "(default log(2°) — i.e. centered at ~2°).")
    parser.add_argument("--ell-sigma-log", type=float, default=0.5,
                        help="LogNormal `sigma` for HSGP length-scale prior")
    parser.add_argument("--nuts-sampler", default="pymc",
                        choices=["pymc", "nutpie", "numpyro", "blackjax"],
                        help="NUTS backend. 'nutpie' is typically much "
                             "faster and uses a full mass matrix by default.")
    parser.add_argument("--init", default="jitter+adapt_full",
                        help="PyMC NUTS init strategy (only used when "
                             "--nuts-sampler=pymc). 'jitter+adapt_full' "
                             "tunes a full off-diagonal mass matrix.")
    parser.add_argument("--model-form",
                        choices=["population", "subjectwise"],
                        default="population",
                        help="Model formula. 'population': proj ~ "
                             "hsgp(distance) + (1|subject) (default). "
                             "'subjectwise': add per-subject HSGP smooth "
                             "deviations from the population smooth.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.95,
                        help="P(>0)/P(<0) threshold for cluster flagging")
    parser.add_argument("--grid-min", type=float, default=0.5)
    parser.add_argument("--grid-max", type=float, default=6.5)
    parser.add_argument("--grid-n", type=int, default=50)
    parser.add_argument("--save-traces", action="store_true",
                        help="Save per-ROI InferenceData (.nc) next to the PDF")
    args = parser.parse_args()

    bin_edges = np.arange(args.bin_min, args.bin_max + args.bin_step / 2,
                          args.bin_step)
    grid = np.linspace(args.grid_min, args.grid_max, args.grid_n)

    args.out.parent.mkdir(parents=True, exist_ok=True)

    # Load data once.
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

    print("Building per-(subject, bin) aggregated table ...")
    agg = build_subject_bin_table(
        pars, args.rois, bin_edges, args.clip_projection,
    )
    print(f"  Total aggregated rows: {len(agg):,}")

    cluster_summary = []  # rows for end-of-run report

    with PdfPages(args.out) as pdf:
        cover_page(pdf, args, bin_edges, grid)

        for roi in args.rois:
            df_roi = (
                agg[agg["roi_base"] == roi]
                [["subject", "distance", "proj"]]
                .dropna(subset=["proj", "distance"])
                .copy()
            )
            df_roi["subject"] = df_roi["subject"].astype(str)
            n_subj = df_roi["subject"].nunique()
            n_rows = len(df_roi)
            print(f"\n=== ROI {roi}: {n_rows} rows, {n_subj} subjects ===")
            if n_rows < 20 or n_subj < 3:
                print(f"  Too few rows / subjects — skipping {roi}.")
                continue

            model, idata, fit_secs, n_div = fit_gam_per_roi(
                df_roi,
                family=args.family,
                m=args.m, c=args.c,
                draws=args.draws, tune=args.tune, chains=args.chains,
                target_accept=args.target_accept, seed=args.seed,
                ell_mu_log=args.ell_mu_log,
                ell_sigma_log=args.ell_sigma_log,
                nuts_sampler=args.nuts_sampler,
                init=args.init,
                model_form=args.model_form,
            )
            print(f"  fit took {fit_secs:.1f}s, divergences = {n_div}")

            samples = evaluate_smooth(model, idata, df_roi, grid)
            summary = summarize_smooth(samples, grid, hdi_prob=0.95)

            plot_roi_page(
                pdf, roi, df_roi, summary,
                fit_secs=fit_secs, n_div=n_div,
                n_subj=n_subj, n_rows=n_rows,
                threshold=args.threshold,
            )

            # Build summary entry for stdout report.
            sig_pos_d = summary.loc[summary["p_pos"] > args.threshold, "distance"].values
            sig_neg_d = summary.loc[summary["p_neg"] > args.threshold, "distance"].values
            cluster_summary.append({
                "roi": roi,
                "n_subj": n_subj,
                "n_rows": n_rows,
                "fit_secs": fit_secs,
                "n_div": n_div,
                "p_pos_distances": sig_pos_d,
                "p_neg_distances": sig_neg_d,
            })

            if args.save_traces:
                trace_path = args.out.with_name(
                    args.out.stem + f"_{roi}_idata.nc"
                )
                idata.to_netcdf(str(trace_path))
                print(f"  saved trace → {trace_path}")

    print(f"\nWrote {args.out}")

    print("\n========== DIVERGENCES PER ROI ==========")
    print(f"  {'ROI':<6} {'n_subj':>6} {'n_rows':>7} {'fit_s':>7} {'divs':>6}")
    print(f"  {'-'*6} {'-'*6} {'-'*7} {'-'*7} {'-'*6}")
    total_div = 0
    for r in cluster_summary:
        print(
            f"  {r['roi']:<6} {r['n_subj']:>6d} {r['n_rows']:>7d} "
            f"{r['fit_secs']:>7.1f} {r['n_div']:>6d}"
        )
        total_div += r["n_div"]
    print(f"  {'-'*6} {'-'*6} {'-'*7} {'-'*7} {'-'*6}")
    print(f"  {'TOTAL':<6} {'':>6} {'':>7} {'':>7} {total_div:>6d}")

    print("\n========== POSTERIOR-CLUSTER SUMMARY ==========")
    for r in cluster_summary:
        pos = r["p_pos_distances"]
        neg = r["p_neg_distances"]
        pos_str = (f"{pos.min():.2f}–{pos.max():.2f}°" if len(pos)
                   else "none")
        neg_str = (f"{neg.min():.2f}–{neg.max():.2f}°" if len(neg)
                   else "none")
        print(
            f"  {r['roi']:<5}  fit={r['fit_secs']:5.1f}s  div={r['n_div']:<3}  "
            f"P(>0)>{0.95}: {pos_str:<22}  P(<0)>{0.95}: {neg_str}"
        )

    print(f"\nPDF written to: {args.out}")


if __name__ == "__main__":
    main()
