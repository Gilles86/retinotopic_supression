"""PRF parameter + spatial-coverage distributions across models.

Single multi-page PDF, one page per "plot type". Every page is a
FacetGrid with rows = subjects, cols = models. So the chain
m1 → m2 → m3 → m4 (etc.) is laid out left-to-right per subject row,
making the per-voxel effect of each schedule step directly visible
at the distribution level.

Pages produced:
  - spatial coverage — 2D KDE of (x, y) PRF centers within the bar
    aperture.
  - one page per scalar parameter (x, y, eccen, theta, sd,
    amplitude, baseline, r2, plus surround / DN / HRF params when
    those columns exist) — histogram per (subject, model).

Reads the warm-start TSVs at
``notes/data/prf_warmstart_m{M}_V1_sub-*.tsv``. Today V1 is the only
ROI in the warm-start output.

Output: ``notes/figures/talk/prf_distributions.pdf``.

CLI::

    python plot_prf_distributions.py [--models all|1,2,3,4|1-4]
                                      [--data-dir DIR]
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

# Hard per-parameter axis ranges. None means: auto-compute from data
# percentiles (p1/p99 by default, see XLIM_QUANTILES). Hard ranges are
# only for parameters with natural bounds (e.g. ``theta`` ∈ [-π, π]).
PARAM_RANGES = {
    "theta": (-np.pi, np.pi),
}

# Percentile bounds used for auto xlims (param not in PARAM_RANGES).
# Default p1/p99 keeps 98% of the distribution. Heavy-tailed params
# (r², srf_amplitude, …) use tighter upper bounds so the bulk near 0
# is visible rather than crushed by the tail.
XLIM_QUANTILES_DEFAULT = (0.01, 0.99)
XLIM_QUANTILES_BY_PARAM = {
    "r2": (0.0, 0.95),
    "srf_amplitude": (0.0, 0.95),
    "amplitude": (0.02, 0.98),
}

# Parameters whose density distribution is heavy-tailed enough that
# linear y crushes the bulk into one bar. Combine percentile xlim
# with log-y so we see BOTH the bulk and the tail.
PARAM_LOG_Y = {"r2", "srf_amplitude"}

# Short semantic label per model number — used in column titles +
# footer legend so reader doesn't have to memorise "m4 = DoG + flex HRF".
MODEL_LABELS = {
    1: "Gauss",
    2: "DoG",
    3: "Gauss+HRF",
    4: "DoG+HRF",
    5: "DN",
    6: "DN+HRF",
}


def _model_legend_text(models):
    """Return a short 'm1: Gauss · m2: DoG · ...' string for the
    figure footer."""
    return "  ·  ".join(f"m{m}: {MODEL_LABELS.get(m, '?')}"
                         for m in models)


def load_warmstart_models(models, data_dir: Path) -> pd.DataFrame:
    """Long-format frame with one row per (subject, voxel_idx, model).

    Stacks every model's per-subject TSVs and tags rows with a
    ``model`` column. ``roi='V1'`` for now (sandbox only fits V1).
    Derived columns ``eccen`` and ``theta`` are added.
    """
    frames = []
    for m in models:
        files = sorted(data_dir.glob(
            f"prf_warmstart_m{m}_V1_sub-*.tsv"))
        if not files:
            print(f"  m{m}: no TSVs found, skipping")
            continue
        for f in files:
            d = pd.read_csv(f, sep="\t")
            d["model"] = m
            frames.append(d)
    if not frames:
        raise FileNotFoundError(
            f"No warmstart TSVs at {data_dir}/prf_warmstart_m*_V1_sub-*.tsv")
    df = pd.concat(frames, ignore_index=True)
    df["roi"] = "V1"
    df["eccen"] = np.sqrt(df["x"] ** 2 + df["y"] ** 2)
    df["theta"] = np.arctan2(df["y"], df["x"])
    return df


def plot_spatial_coverage_page(pdf: PdfPages, df: pd.DataFrame) -> None:
    """One page: 2D PRF-center KDE per (subject × model). Real voxels
    only (r² > 0)."""
    real = df[df.r2 > 0].copy()
    subjects = sorted(real.subject.unique())
    models = sorted(real.model.unique())

    n_rows, n_cols = len(subjects), len(models)
    fig_w = max(2.6 * n_cols + 1.3, 8.0)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(fig_w, 2.6 * n_rows),
        squeeze=False)
    fig.suptitle("PRF center density per subject × model",
                 weight="bold", fontsize=16, y=0.995)

    grid_n = 120
    g = np.linspace(-APERTURE_RADIUS - 0.5,
                     APERTURE_RADIUS + 0.5, grid_n)
    GX, GY = np.meshgrid(g, g)
    outside = (GX ** 2 + GY ** 2) > APERTURE_RADIUS ** 2
    from scipy.stats import gaussian_kde

    for i, sub in enumerate(subjects):
        for j, m in enumerate(models):
            ax = axes[i, j]
            sel = real[(real.subject == sub) & (real.model == m)]
            if len(sel) < 5:
                ax.text(0.5, 0.5, "n<5", transform=ax.transAxes,
                        ha="center", va="center", color="grey")
            else:
                if len(sel) > 8000:
                    sel = sel.sample(8000, random_state=0)
                xy = np.vstack([sel.x.to_numpy(), sel.y.to_numpy()])
                try:
                    kde = gaussian_kde(xy, bw_method=0.20)
                    Z = kde(np.vstack([GX.ravel(), GY.ravel()])
                            ).reshape(grid_n, grid_n)
                    Z = np.where(outside, np.nan, Z)
                    ax.imshow(Z, extent=(g[0], g[-1], g[0], g[-1]),
                              origin="lower", cmap="viridis",
                              norm=PowerNorm(gamma=0.45),
                              interpolation="bilinear")
                except Exception as e:
                    ax.text(0.5, 0.5, f"KDE fail\n{type(e).__name__}",
                             transform=ax.transAxes, ha="center",
                             va="center", color="red", fontsize=9)

            for r_iso in (1.0, 2.0, 3.0):
                ax.add_patch(Circle((0, 0), r_iso, facecolor="none",
                                     edgecolor="white", lw=0.8,
                                     ls=(0, (2, 4)), alpha=0.6))
            ax.add_patch(Circle((0, 0), APERTURE_RADIUS,
                                 facecolor="none", edgecolor="white",
                                 lw=1.8, ls=(0, (5, 3))))
            ax.plot(0, 0, "+", color="white", ms=10, mew=1.6)
            ax.set_xlim(g[0], g[-1]); ax.set_ylim(g[0], g[-1])
            ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)

            if i == 0:
                ax.set_title(f"m{m}\n{MODEL_LABELS.get(m, '')}",
                              weight="bold", fontsize=12)
            if j == 0:
                ax.set_ylabel(f"sub-{int(sub):02d}",
                              fontsize=11, rotation=0, ha="right",
                              va="center", labelpad=14)

    fig.text(0.5, 0.01, _model_legend_text(models),
             ha="center", va="bottom", fontsize=10, color="0.30")
    plt.tight_layout(rect=(0, 0.03, 1, 0.96))
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def plot_parameter_page(pdf: PdfPages, df: pd.DataFrame,
                          param: str) -> None:
    """One page: histogram FacetGrid rows = subjects × cols = models
    for one scalar parameter. Only models that actually carry the
    column get a column; subjects/models missing the param show empty
    cells."""
    if param not in df.columns:
        return
    real = df[(df.r2 > 0) & df[param].notna()].copy()
    if not len(real):
        return
    subjects = sorted(real.subject.unique())
    # Restrict to models that actually have any data for this param.
    models = sorted(real.model.unique())
    if not models:
        return

    # Hard range if defined, otherwise percentile bounds shared across
    # all (subject, model) cells so the histograms are comparable.
    rng = PARAM_RANGES.get(param)
    if rng is None:
        q_lo, q_hi = XLIM_QUANTILES_BY_PARAM.get(
            param, XLIM_QUANTILES_DEFAULT)
        lo = float(real[param].quantile(q_lo))
        hi = float(real[param].quantile(q_hi))
        # Avoid a degenerate zero-width range (constant column).
        if hi <= lo:
            hi = lo + 1e-6
        rng = (lo, hi)
    real = real[(real[param] >= rng[0]) & (real[param] <= rng[1])]

    g = sns.FacetGrid(
        real, row="subject", col="model",
        row_order=subjects, col_order=models,
        height=1.7, aspect=1.6, sharex=True, sharey=False,
        margin_titles=True, despine=True)
    g.map_dataframe(
        sns.histplot, x=param, stat="density",
        bins=40, color="#1B4965", alpha=0.85,
        edgecolor="white", linewidth=0.3)
    # Column titles include the semantic model label.
    g.set_titles(row_template="sub-{row_name:02d}",
                 col_template="")
    for ax, m in zip(g.axes[0], models):
        ax.set_title(f"m{m}: {MODEL_LABELS.get(m, '')}",
                     weight="bold", fontsize=12)
    g.set_axis_labels(param, "density")
    for ax in g.axes.flat:
        ax.set_xlim(rng)
        if param in PARAM_LOG_Y:
            ax.set_yscale("log")
        ax.grid(alpha=0.18, axis="y")
        ax.tick_params(labelsize=9)
    g.fig.subplots_adjust(top=0.92, bottom=0.08)
    g.fig.suptitle(f"Distribution of {param}",
                    weight="bold", fontsize=15)
    g.fig.text(0.5, 0.01, _model_legend_text(models),
                ha="center", va="bottom", fontsize=10, color="0.30")
    pdf.savefig(g.fig, bbox_inches="tight")
    plt.close(g.fig)


def _parse_models(text):
    """Accept '1,2,3,4' or '1-4' or 'all' → list[int]."""
    if text.lower() == "all":
        return None  # auto-detect from available TSVs
    if "-" in text and "," not in text:
        lo, hi = text.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in text.split(",")]


def _autodetect_models(data_dir: Path):
    """Return sorted ints for which prf_warmstart_m{M}_V1_sub-*.tsv exists."""
    found = set()
    for f in data_dir.glob("prf_warmstart_m*_V1_sub-*.tsv"):
        try:
            m = int(f.name.split("_m")[1].split("_")[0])
            found.add(m)
        except (IndexError, ValueError):
            continue
    return sorted(found)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="all",
                     help="Comma list ('1,2,3,4'), range ('1-4'), or "
                          "'all' (default; auto-detect from TSVs).")
    ap.add_argument("--data-dir", type=Path, default=None,
                     help="Where to find prf_warmstart_m{M}_V1_sub-*.tsv "
                          "(default: notes/data/, falls back to /tmp/v1_results/)")
    ap.add_argument("--out", type=Path, default=None,
                     help="Output PDF path (default: notes/figures/talk/"
                          "prf_distributions.pdf)")
    args = ap.parse_args()

    # Resolve data dir — prefer repo notes/data; fall back to /tmp.
    if args.data_dir is None:
        for cand in (DATA_DIR, Path("/tmp/v1_results")):
            if any(cand.glob("prf_warmstart_m*_V1_sub-*.tsv")):
                args.data_dir = cand
                break
        if args.data_dir is None:
            args.data_dir = DATA_DIR

    models = _parse_models(args.models)
    if models is None:
        models = _autodetect_models(args.data_dir)
    if not models:
        raise RuntimeError(f"No warmstart TSVs at {args.data_dir}")

    out_path = args.out or (THIS / "prf_distributions.pdf")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sns.set_theme(context="talk", style="ticks", font_scale=0.85)

    print(f"Models: {models}; data: {args.data_dir}; out: {out_path}")
    df = load_warmstart_models(models, args.data_dir)
    print(f"  loaded {len(df):,} rows  "
          f"({df.subject.nunique()} subjects × "
          f"{df.model.nunique()} models)")

    # Parameters to plot — only those actually present in at least
    # one model's TSV.
    candidate_params = ("x", "y", "eccen", "theta", "sd",
                         "amplitude", "baseline", "r2",
                         "srf_size", "srf_amplitude",
                         "hrf_delay", "hrf_dispersion",
                         "rf_amplitude", "neural_baseline",
                         "surround_baseline", "bold_baseline")
    present = [p for p in candidate_params if p in df.columns]

    with PdfPages(out_path) as pdf:
        plot_spatial_coverage_page(pdf, df)
        for param in present:
            plot_parameter_page(pdf, df, param)

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
