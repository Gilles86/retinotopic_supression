"""Cross-model R² pairplots for the V1 warm-start chain.

Per subject (rows) × per model pair (cols), scatter the per-voxel R²
of model A (x) against model B (y). An identity line ``y = x`` makes
"chain improves R²" vs "chain hurts R²" readable at a glance: voxels
above the diagonal got better in the downstream model, voxels below
got worse.

Voxels are matched by ``voxel_idx`` so the plot doesn't depend on row
order. Only voxels real in BOTH models (``r² > 0`` in each) are kept.

Output: ``notes/figures/talk/prf_r2_pairs.pdf``.

CLI::

    python plot_prf_r2_pairs.py \
        --pairs 1,2 1,3 2,3 \
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

THIS = Path(__file__).resolve().parent
REPO = THIS.parents[2]
DATA_DIR = REPO / "notes" / "data"


def load_model_r2(model: int, data_dir: Path) -> pd.DataFrame:
    """Concat per-subject m{model} TSVs; keep only the keys needed
    for joining + a single R² column tagged with the model number."""
    files = sorted(data_dir.glob(f"prf_warmstart_m{model}_V1_sub-*.tsv"))
    if not files:
        return pd.DataFrame(columns=["subject", "voxel_idx", f"r2_m{model}"])
    frames = []
    for f in files:
        d = pd.read_csv(f, sep="\t",
                          usecols=["subject", "voxel_idx", "r2"])
        d = d.rename(columns={"r2": f"r2_m{model}"})
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def plot_pair(ax, sub_df, m_x, m_y, alpha=0.25, identity_color="#E76F51"):
    """One axes: r2_m{m_x} vs r2_m{m_y} for a single subject."""
    rx = sub_df[f"r2_m{m_x}"].to_numpy()
    ry = sub_df[f"r2_m{m_y}"].to_numpy()
    real = (rx > 0) & (ry > 0)
    if real.sum() < 5:
        ax.text(0.5, 0.5, "n<5", transform=ax.transAxes,
                ha="center", va="center", color="grey")
        return
    rx, ry = rx[real], ry[real]
    # Subsample for legibility.
    if len(rx) > 5000:
        idx = np.random.default_rng(0).choice(len(rx), 5000, replace=False)
        rx, ry = rx[idx], ry[idx]

    ax.scatter(rx, ry, s=4, alpha=alpha, color="#1B4965",
               edgecolor="none", rasterized=True)

    # Identity line.
    lim_lo = max(0.0, min(rx.min(), ry.min()) - 0.02)
    lim_hi = min(1.0, max(rx.max(), ry.max()) + 0.02)
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi],
            color=identity_color, lw=1.4, ls="--", zorder=10)
    ax.set_xlim(lim_lo, lim_hi)
    ax.set_ylim(lim_lo, lim_hi)
    ax.set_aspect("equal")

    # Above-diagonal fraction (downstream improvement).
    frac_up = float((ry > rx).mean())
    med_dx = float(np.median(ry - rx))
    ax.text(0.04, 0.96,
            f"n={len(rx):,}\n>diag: {frac_up:.0%}\nΔR²med: {med_dx:+.3f}",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=9, family="monospace",
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="none",
                       boxstyle="round,pad=0.25"))
    ax.grid(alpha=0.18)
    ax.tick_params(labelsize=9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", default=["1,2", "1,3", "2,3"],
                     help="Model pairs as 'A,B' (default: 1,2 1,3 2,3)")
    ap.add_argument("--data-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    pairs = [tuple(int(x) for x in p.split(",")) for p in args.pairs]
    needed_models = sorted({m for pair in pairs for m in pair})

    # Resolve data dir — prefer repo notes/data; fall back to /tmp/v1_results.
    if args.data_dir is None:
        for cand in (DATA_DIR, Path("/tmp/v1_results")):
            if any(cand.glob(f"prf_warmstart_m{needed_models[0]}_V1_sub-*.tsv")):
                args.data_dir = cand
                break
        if args.data_dir is None:
            args.data_dir = DATA_DIR

    # Outer join all needed model TSVs on (subject, voxel_idx).
    df = None
    for m in needed_models:
        chunk = load_model_r2(m, args.data_dir)
        if df is None:
            df = chunk
        else:
            df = df.merge(chunk, on=["subject", "voxel_idx"], how="outer")
    if df is None or len(df) == 0:
        raise RuntimeError(f"No warmstart TSVs found at {args.data_dir} "
                            f"for models {needed_models}")

    subjects = sorted(df["subject"].unique())
    n_rows, n_cols = len(subjects), len(pairs)
    fig_w = max(2.6 * n_cols + 0.8, 7.5)
    fig_h = 2.6 * n_rows + 0.6

    sns.set_theme(context="talk", style="ticks", font_scale=0.8)
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(fig_w, fig_h),
                              squeeze=False)
    fig.suptitle(
        "Per-voxel R² across the warm-start chain (V1, all subjects)",
        weight="bold", fontsize=14, y=0.997)

    for i, sub in enumerate(subjects):
        sub_df = df[df["subject"] == sub]
        for j, (m_x, m_y) in enumerate(pairs):
            ax = axes[i, j]
            plot_pair(ax, sub_df, m_x, m_y)
            if i == 0:
                ax.set_title(f"m{m_x} vs m{m_y}", weight="bold",
                              fontsize=12, pad=4)
            if j == 0:
                ax.set_ylabel(f"sub-{int(sub):02d}\nR² ↓",
                              rotation=0, ha="right", va="center",
                              fontsize=10, labelpad=18)
            else:
                ax.set_ylabel("")
            if i == n_rows - 1:
                ax.set_xlabel(f"R² (m{m_x})", fontsize=10)
            else:
                ax.set_xlabel("")

    plt.tight_layout(rect=(0, 0, 1, 0.97))

    out_path = args.out or (THIS / "prf_r2_pairs.pdf")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out_path) as pdf:
        pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
