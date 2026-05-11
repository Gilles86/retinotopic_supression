"""Compare warm-start vs cached PRF fits for V1 across all subjects.

Renders a 4-row × 3-column grid (one row per warm-start model):
  - Col 1: V1 coverage panel (KDE of x,y centers) — warmstart only
  - Col 2: per-subject median R² strip — warmstart (orange) vs cached (grey)
  - Col 3: σ vs eccentricity — warmstart (solid) vs cached (dashed)

Models compared:
  m3 — Gaussian, flex HRF
  m4 — DoG, flex HRF
  m5 — Divisive Normalization, fixed HRF
  m6 — Divisive Normalization, flex HRF

Input:
  notes/data/prf_warmstart_m{N}_V1_sub-{NN}.tsv   (one per subject × model)
  notes/data/prf_validity_summary[_m{N}].tsv      (cached fits)

Output: notes/figures/talk/talk_warmstart_compare.pdf
"""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import PowerNorm
from matplotlib.patches import Circle
from scipy.stats import norm, gaussian_kde

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS))
from make_figures import APERTURE_RADIUS, BUILDUP_PALETTE, base_rc  # noqa

REPO = THIS.parents[2]
WARM_DIR = REPO / "notes" / "data"
OUT = THIS / "talk_warmstart_compare.pdf"

MODELS = [3, 4, 5, 6]
MODEL_LABELS = {
    3: "m3 — Gaussian + flex HRF",
    4: "m4 — DoG + flex HRF",
    5: "m5 — DN + fixed HRF",
    6: "m6 — DN + flex HRF",
}
WARM_COLOR = "#E76F51"
CACHED_COLOR = "#1B4965"

SIGMA_FLOOR, SIGMA_CEIL = 0.30, 4.00
MASS_THR = 0.50
# Warmstart now marks invalid voxels as r²=0 + NaN params via
# `mark_invalid_fits`, so we don't need an R2_CEIL hack to defend
# against braincoder sentinels. We still cap at 0.99 as a soft guard
# for the CACHED fits which weren't run through the post-hoc helper.
R2_CEIL = 0.99
ECC_BINS = np.arange(0, 4.5, 0.5)
ECC_CTRS = 0.5 * (ECC_BINS[:-1] + ECC_BINS[1:])


def load_warmstart(model):
    """Concat all per-subject TSVs for one warmstart model."""
    files = sorted(WARM_DIR.glob(
        f"prf_warmstart_m{model}_V1_sub-*.tsv"))
    if not files:
        return None
    return pd.concat([pd.read_csv(f, sep="\t") for f in files],
                     ignore_index=True)


def load_cached(model):
    """Read cached extended TSV and restrict to V1."""
    if model == 4:
        f = WARM_DIR / "prf_validity_summary.tsv"
    else:
        f = WARM_DIR / f"prf_validity_summary_m{model}.tsv"
    if not f.exists():
        return None
    df = pd.read_csv(f, sep="\t")
    return df[df.roi == "V1"].copy()


def filt(df, n_params, n_timepoints=258):
    """Apply the same FDR + aperture + sigma filter as plot_prf_validity."""
    from scipy.stats import f as f_dist
    from statsmodels.stats.multitest import multipletests
    df = df.copy()
    df["eccen"] = np.sqrt(df.x**2 + df.y**2)
    df["mass_in"] = 1.0 - norm.cdf(
        (df.eccen - APERTURE_RADIUS) / df.sd.clip(lower=0.05))
    r2c = df.r2.clip(0.0, 0.999999).to_numpy()
    df1, df2 = n_params, n_timepoints - n_params - 1
    F = (r2c / df1) / ((1.0 - r2c) / df2)
    p = 1.0 - f_dist.cdf(F, df1, df2)
    rejected, _, *_ = multipletests(p, alpha=0.05, method="fdr_bh")
    fdr_thr = (float(np.min(df.r2.to_numpy()[rejected]))
               if rejected.any() else np.inf)
    sel = ((df.r2 >= fdr_thr) & (df.r2 <= R2_CEIL)
           & (df.mass_in >= MASS_THR)
           & (df.sd >= SIGMA_FLOOR) & (df.sd <= SIGMA_CEIL))
    return df[sel], fdr_thr


def draw_field_density(ax, df, palette):
    if len(df) < 5:
        ax.text(0.5, 0.5, f"no voxels", transform=ax.transAxes,
                ha="center", va="center", color=palette["muted"])
        for s in ax.spines.values():
            s.set_visible(False)
        return
    df_s = df.sample(min(len(df), 15000), random_state=0)
    xy = np.vstack([df_s.x.to_numpy(), df_s.y.to_numpy()])
    kde = gaussian_kde(xy, bw_method=0.18)
    g = np.linspace(-4, 4, 160)
    GX, GY = np.meshgrid(g, g)
    Z = kde(np.vstack([GX.ravel(), GY.ravel()])).reshape(160, 160)
    mask = (GX**2 + GY**2) > APERTURE_RADIUS**2
    Z = np.where(mask, np.nan, Z)
    ax.imshow(Z, extent=(-4, 4, -4, 4), origin="lower",
              cmap="viridis", norm=PowerNorm(gamma=0.45))
    for r_iso in (1.0, 2.0, 3.0):
        ax.add_patch(Circle((0, 0), r_iso, facecolor="none",
                            edgecolor="white", lw=0.8,
                            ls=(0, (2, 4)), alpha=0.55))
    ax.add_patch(Circle((0, 0), APERTURE_RADIUS, facecolor="none",
                        edgecolor="white", lw=2.0, ls=(0, (5, 3))))
    ax.plot(0, 0, "+", color="white", ms=12, mew=1.8)
    ax.set_xlim(-4, 4); ax.set_ylim(-4, 4); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def per_subject_sigma_curve(df):
    out = {}
    for sub, g in df.groupby("subject"):
        binned = pd.cut(g.eccen, bins=ECC_BINS, include_lowest=True)
        out[sub] = g.groupby(binned, observed=False).sd.median().values
    return pd.DataFrame(out, index=ECC_CTRS).T


N_PARAMS_BY_MODEL = {3: 7, 4: 9, 5: 9, 6: 11}


def main():
    p = BUILDUP_PALETTE
    # Override the dark theme — we want a printable comparison.
    rc = {"font.size": 16, "axes.titlesize": 18,
          "axes.labelsize": 15, "xtick.labelsize": 13,
          "ytick.labelsize": 13, "legend.fontsize": 13,
          "figure.titlesize": 22,
          "axes.facecolor": "white", "figure.facecolor": "white",
          "axes.edgecolor": "black", "axes.labelcolor": "black",
          "xtick.color": "black", "ytick.color": "black",
          "text.color": "black"}

    with plt.rc_context(rc):
        fig, axes = plt.subplots(
            len(MODELS), 4, figsize=(18, 4.2 * len(MODELS)),
            gridspec_kw={"width_ratios": [1.0, 1.0, 1.2, 1.2],
                          "hspace": 0.55, "wspace": 0.30})

        for i, m in enumerate(MODELS):
            ax_cov_old, ax_cov_new, ax_r2, ax_sig = axes[i]
            k = N_PARAMS_BY_MODEL[m]
            warm = load_warmstart(m)
            cached = load_cached(m)

            # Row label on the leftmost cell.
            ax_cov_old.set_ylabel(MODEL_LABELS[m], fontsize=15,
                                   weight="bold", labelpad=8)

            # CACHED coverage (left panel).
            if cached is not None and len(cached) > 0:
                cached_sel, fdr_c = filt(cached, k)
                ax_cov_old.set_title(
                    f"Cached  ({len(cached_sel)}/{len(cached)} pass, "
                    f"FDR thr={fdr_c:.3f})", fontsize=12)
                draw_field_density(ax_cov_old, cached_sel, p)
            else:
                ax_cov_old.text(0.5, 0.5, "no cached data",
                                ha="center", va="center",
                                transform=ax_cov_old.transAxes,
                                color="grey")
                for s in ax_cov_old.spines.values():
                    s.set_visible(False)
                ax_cov_old.set_xticks([]); ax_cov_old.set_yticks([])

            if warm is None:
                ax_cov_new.text(0.5, 0.5, f"no warmstart data",
                                ha="center", va="center",
                                transform=ax_cov_new.transAxes,
                                color="grey")
                for ax in (ax_cov_new, ax_r2, ax_sig):
                    for s in ax.spines.values():
                        s.set_visible(False)
                    ax.set_xticks([]); ax.set_yticks([])
                continue

            warm_sel, fdr_w = filt(warm, k)
            ax_cov_new.set_title(
                f"Warmstart  ({len(warm_sel)}/{len(warm)} pass, "
                f"FDR thr={fdr_w:.3f})", fontsize=12)
            draw_field_density(ax_cov_new, warm_sel, p)

            # Per-subject median R² strip — only voxels that survive
            # the same filter as the validity figure (r² in [FDR, 0.99],
            # mass_in≥50%, σ in [0.3, 4]). This drops degenerate r²≈1
            # zero-variance voxels from inflating the warmstart distribution.
            np.random.seed(0)
            for j, (label, df, color) in enumerate([
                    ("cached", cached, CACHED_COLOR),
                    ("warm",   warm,   WARM_COLOR)]):
                if df is None or len(df) == 0:
                    continue
                df_sel, _ = filt(df, k)
                if len(df_sel) == 0:
                    continue
                med = df_sel.groupby("subject").r2.median()
                jitter = np.random.uniform(-0.18, 0.18, size=len(med))
                ax_r2.scatter(
                    np.full(len(med), j) + jitter, med.values,
                    color=color, alpha=0.75, s=55,
                    edgecolor="white", linewidth=0.7)
                ax_r2.scatter([j], [np.median(med.values)],
                              marker="_", color="black",
                              s=900, lw=3.2, zorder=10)
            ax_r2.set_xticks([0, 1])
            ax_r2.set_xticklabels(["cached", "warm"])
            ax_r2.set_ylabel("Per-subject median R²\n(selected voxels)")
            ax_r2.set_title("R² (post-selection)", loc="left", fontsize=15)
            ax_r2.grid(alpha=0.20, axis="y")
            ax_r2.set_ylim(0, 1)
            ax_r2.set_xlim(-0.5, 1.5)

            # σ vs eccen
            for label, df, color, ls in [
                    ("cached", cached, CACHED_COLOR, (0, (4, 2))),
                    ("warm",   warm,   WARM_COLOR,   "-")]:
                if df is None or len(df) == 0:
                    continue
                df_sel, _ = filt(df, k)
                if len(df_sel) == 0:
                    continue
                curves = per_subject_sigma_curve(df_sel)
                mn = curves.mean()
                se = curves.sem()
                valid = mn.notna()
                ax_sig.fill_between(
                    ECC_CTRS[valid], (mn - se)[valid],
                    (mn + se)[valid], alpha=0.15, color=color,
                    linewidth=0)
                ax_sig.plot(ECC_CTRS[valid], mn[valid],
                            color=color, lw=2.5, ls=ls,
                            marker="o", ms=5, label=label)
            ax_sig.axvline(APERTURE_RADIUS, color=p["muted"],
                           lw=1.0, ls=(0, (4, 3)), alpha=0.7)
            ax_sig.set_xlabel("PRF eccentricity (°)")
            ax_sig.set_ylabel("PRF σ (°)")
            ax_sig.set_title("σ vs eccentricity", loc="left", fontsize=15)
            ax_sig.set_xlim(0, max(ECC_BINS))
            ax_sig.grid(alpha=0.20)
            ax_sig.legend(loc="upper left", frameon=False)

        fig.suptitle("Warm-start vs cached PRF fits — V1 across subjects",
                     weight="bold", y=0.995)
        OUT.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUT, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
