"""Render the slide-07 effective-drive panel under several color palette
choices, so we can pick the one that makes the suppression vs.
enhancement contrast most legible.

Output: notes/figures/talk/compare_drive_palettes.pdf (one page, grid
of options labeled by cmap + normalization).

The chosen frame uses the same setup as slide 07: bar across the
UL+UR row, target at UL, distractor at LL, HP at UR.
"""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import (LinearSegmentedColormap, PowerNorm,
                                TwoSlopeNorm, to_rgba)
from matplotlib.patches import Circle

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS))
from make_figures import (  # noqa: E402
    APERTURE_RADIUS, BUILDUP_PALETTE, ECC, HP_KEY, LOCATIONS,
    SD_SUSTAINED, SD_TRANSIENT, AMP_SUSTAINED_FULL, AMP_SUSTAINED_HP,
    AMP_TARGET, AMP_DYN, TARGET_KEY, DYN_DIST_KEY,
    base_rc, gaussian_2d, _build_paradigm_S, _modulation_field_std,
    _gain_components_hp_weakened,
)

OUT = THIS / "compare_drive_palettes.pdf"


def main():
    p = BUILDUP_PALETTE
    lim = 4.8
    res = 280
    xs = np.linspace(-lim, lim, res)
    xx, yy = np.meshgrid(xs, xs)

    # Same configuration as slide 07.
    bar_y_center = +ECC / np.sqrt(2)
    S = _build_paradigm_S(xx, yy, bar_y_center)

    sustained = _gain_components_hp_weakened()
    target_comps = [(*LOCATIONS[TARGET_KEY], SD_TRANSIENT, AMP_TARGET)]
    dyn_comps = [(*LOCATIONS[DYN_DIST_KEY], SD_TRANSIENT, AMP_DYN)]
    M = _modulation_field_std(sustained + target_comps + dyn_comps,
                              xx, yy)
    D = S * M
    D_minus_S = D - S          # 'modulation effect': what attention added/removed

    drive_max = float(D.max())
    diff_abs = float(np.max(np.abs(D_minus_S)))

    extent = (-lim, lim, -lim, lim)

    # Custom 'fire' cmap — black → deep red → orange → bright yellow → white,
    # higher dynamic range than inferno.
    fire = LinearSegmentedColormap.from_list(
        "fire", ["#000000", "#3A0010", "#7A0010", "#C32C00",
                 "#F08400", "#FFD500", "#FFFFFF"])

    options = [
        ("inferno  γ=1.0", lambda ax: ax.imshow(
            D, extent=extent, origin="lower", cmap="inferno",
            vmin=0.0, vmax=drive_max, interpolation="bilinear")),
        ("inferno  γ=1.5  (current)", lambda ax: ax.imshow(
            D, extent=extent, origin="lower", cmap="inferno",
            norm=PowerNorm(gamma=1.5, vmin=0.0, vmax=drive_max),
            interpolation="bilinear")),
        ("plasma  γ=1.4", lambda ax: ax.imshow(
            D, extent=extent, origin="lower", cmap="plasma",
            norm=PowerNorm(gamma=1.4, vmin=0.0, vmax=drive_max),
            interpolation="bilinear")),
        ("viridis  γ=1.4", lambda ax: ax.imshow(
            D, extent=extent, origin="lower", cmap="viridis",
            norm=PowerNorm(gamma=1.4, vmin=0.0, vmax=drive_max),
            interpolation="bilinear")),
        ("cividis  γ=1.3", lambda ax: ax.imshow(
            D, extent=extent, origin="lower", cmap="cividis",
            norm=PowerNorm(gamma=1.3, vmin=0.0, vmax=drive_max),
            interpolation="bilinear")),
        ("custom 'fire'  γ=1.4", lambda ax: ax.imshow(
            D, extent=extent, origin="lower", cmap=fire,
            norm=PowerNorm(gamma=1.4, vmin=0.0, vmax=drive_max),
            interpolation="bilinear")),
        ("hot  γ=1.3", lambda ax: ax.imshow(
            D, extent=extent, origin="lower", cmap="hot",
            norm=PowerNorm(gamma=1.3, vmin=0.0, vmax=drive_max),
            interpolation="bilinear")),
        ("twilight_shifted  γ=1.0", lambda ax: ax.imshow(
            D, extent=extent, origin="lower", cmap="twilight_shifted",
            vmin=0.0, vmax=drive_max,
            interpolation="bilinear")),
        ("S × (M − 1)  diverging\n(modulation effect, RdBu_r)",
         lambda ax: ax.imshow(
             S * (M - 1.0), extent=extent, origin="lower",
             cmap="RdBu_r",
             norm=TwoSlopeNorm(vcenter=0.0,
                               vmin=-diff_abs, vmax=+diff_abs),
             interpolation="bilinear")),
    ]

    nopt = len(options)
    ncol = 3
    nrow = (nopt + ncol - 1) // ncol

    rc = base_rc(p)
    rc.update({"font.size": 14, "axes.titlesize": 16})
    with plt.rc_context(rc):
        fig, axes = plt.subplots(
            nrow, ncol, figsize=(4.6 * ncol, 4.8 * nrow),
            squeeze=False)
        for k, (label, plot_fn) in enumerate(options):
            ax = axes[k // ncol, k % ncol]
            plot_fn(ax)
            ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
            ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            # Aperture + 4 corner outlines + flashes
            ax.add_patch(Circle((0, 0), APERTURE_RADIUS, facecolor="none",
                                edgecolor="white", lw=1.0,
                                ls=(0, (4, 3)), alpha=0.6, zorder=4))
            for k_, (x, y) in LOCATIONS.items():
                edge = "white"; lw = 0.9; ls = "-"
                if k_ == HP_KEY:
                    edge = p["hp"]; lw = 1.6; ls = (0, (2, 2))
                ax.add_patch(Circle((x, y), 0.42, facecolor="none",
                                    edgecolor=edge, lw=lw, ls=ls,
                                    zorder=5))
            tx, ty = LOCATIONS[TARGET_KEY]
            ax.add_patch(Circle((tx, ty), 0.52, facecolor="none",
                                edgecolor=p["target"], lw=1.6,
                                ls=(0, (2, 2)), zorder=5))
            dx, dy = LOCATIONS[DYN_DIST_KEY]
            ax.add_patch(Circle((dx, dy), 0.52, facecolor="none",
                                edgecolor=p["af"], lw=1.6,
                                ls=(0, (2, 2)), zorder=5))
            ax.set_title(label, pad=6, weight="bold")
        # Hide leftover axes
        for k in range(nopt, nrow * ncol):
            axes[k // ncol, k % ncol].axis("off")

        fig.suptitle("Effective-drive palette comparison "
                     "— same frame, different colormap/normalization",
                     fontsize=20, weight="bold", y=0.995)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(OUT, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
