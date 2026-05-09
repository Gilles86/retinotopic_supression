"""
Minimalist schematic illustrations of the retsupp AF PRF model.

Produces 3-4 distinct visual style variants for each of:
  - paradigm (visual stimulus + search positions)
  - prf_af   (PRF gaussian + AF gain modulation)
  - compositional (sum of gain components: 1 + sustained + dynamic + target)
  - pipeline (full graphical abstract)

All figures are saved as vector PDFs into the same directory as this script.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from matplotlib.patches import (
    Circle,
    FancyArrowPatch,
    FancyBboxPatch,
    Rectangle,
    Wedge,
)

OUT = Path(__file__).parent

ECC = 4.0  # ring eccentricity in degrees
SIZE = 1.5  # search-array item footprint
APERTURE_RADIUS = ECC - SIZE / 1.8  # ~3.17

# Four corner positions (degrees of visual angle), matching distractor_locations.
LOCATIONS = {
    "upper_right": (ECC / np.sqrt(2), +ECC / np.sqrt(2)),
    "upper_left": (-ECC / np.sqrt(2), +ECC / np.sqrt(2)),
    "lower_left": (-ECC / np.sqrt(2), -ECC / np.sqrt(2)),
    "lower_right": (ECC / np.sqrt(2), -ECC / np.sqrt(2)),
}

# Style palettes -------------------------------------------------------------

PALETTES = {
    "cool": {
        "bg": "#FBFBFD",
        "fg": "#1F2933",
        "muted": "#9AA5B1",
        "stim": "#3D5A80",
        "hp": "#EE6C4D",
        "prf": "#2A9D8F",
        "af": "#E76F51",
        "target": "#F4A261",
        "accent": "#293241",
    },
    "warm": {
        "bg": "#FDFBF7",
        "fg": "#2B2118",
        "muted": "#A89B8C",
        "stim": "#7B3F00",
        "hp": "#C1272D",
        "prf": "#0A6E6E",
        "af": "#D1495B",
        "target": "#EDAE49",
        "accent": "#3D2C1E",
    },
    "mono": {
        "bg": "#FFFFFF",
        "fg": "#111111",
        "muted": "#888888",
        "stim": "#444444",
        "hp": "#111111",
        "prf": "#222222",
        "af": "#666666",
        "target": "#000000",
        "accent": "#000000",
    },
}


def base_rc(palette):
    return {
        "figure.facecolor": palette["bg"],
        "axes.facecolor": palette["bg"],
        "axes.edgecolor": palette["fg"],
        "axes.labelcolor": palette["fg"],
        "axes.titlecolor": palette["fg"],
        "xtick.color": palette["fg"],
        "ytick.color": palette["fg"],
        "text.color": palette["fg"],
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "savefig.facecolor": palette["bg"],
        "pdf.fonttype": 42,
    }


def clean_axes(ax, lim=5.0, hide=True):
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    if hide:
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)


def draw_visual_field(ax, palette, fixation=True, aperture_circle=True):
    if aperture_circle:
        ap = Circle(
            (0, 0),
            APERTURE_RADIUS,
            facecolor="none",
            edgecolor=palette["muted"],
            lw=0.8,
            ls=(0, (4, 3)),
        )
        ax.add_patch(ap)
    if fixation:
        ax.plot(0, 0, "+", color=palette["fg"], ms=8, mew=1.2)


def draw_search_positions(ax, palette, hp_key=None, target_key=None,
                          show_ring=True, marker_size=0.55):
    if show_ring:
        ring = Circle(
            (0, 0),
            ECC,
            facecolor="none",
            edgecolor=palette["muted"],
            lw=0.6,
            ls=(0, (1, 3)),
        )
        ax.add_patch(ring)
    for key, (x, y) in LOCATIONS.items():
        if key == hp_key:
            face = palette["hp"]
            edge = palette["hp"]
            lw = 1.2
        elif key == target_key:
            face = palette["target"]
            edge = palette["target"]
            lw = 1.0
        else:
            face = "none"
            edge = palette["fg"]
            lw = 1.0
        c = Circle((x, y), marker_size, facecolor=face, edgecolor=edge, lw=lw)
        ax.add_patch(c)


def gaussian_2d(xx, yy, cx, cy, sd):
    return np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sd ** 2))


# ---------------------------------------------------------------------------
# 1. PARADIGM
# ---------------------------------------------------------------------------

def paradigm_v1():
    """Cool palette, with bar at one snapshot, HP at upper_right, annotations."""
    p = PALETTES["cool"]
    with plt.rc_context(base_rc(p)):
        fig, ax = plt.subplots(figsize=(10, 6))
        clean_axes(ax, lim=5.2)
        draw_visual_field(ax, p)

        # Bar stimulus snapshot — diagonal bar across aperture
        bar_w = 1.2
        # rotate bar by -30 degrees
        theta = np.deg2rad(-30)
        # draw bar as rotated rectangle clipped (visually) by aperture
        # (we just draw it, light-weight)
        rect = Rectangle(
            (-APERTURE_RADIUS, -bar_w / 2),
            2 * APERTURE_RADIUS,
            bar_w,
            facecolor=p["stim"],
            edgecolor="none",
            alpha=0.18,
        )
        from matplotlib.transforms import Affine2D
        rect.set_transform(Affine2D().rotate(theta) + ax.transData)
        ax.add_patch(rect)

        # arrow indicating sweep direction
        arr = FancyArrowPatch(
            (-3.6, 1.6),
            (3.6, -1.6),
            arrowstyle="->",
            color=p["stim"],
            lw=1.6,
            mutation_scale=14,
            alpha=0.9,
        )
        ax.add_patch(arr)
        ax.text(3.8, -1.9, "bar sweep", color=p["stim"], fontsize=10,
                ha="left", va="top", style="italic")

        draw_search_positions(ax, p, hp_key="upper_right")

        # Annotation for HP location
        hp_x, hp_y = LOCATIONS["upper_right"]
        ax.annotate(
            "HP distractor location",
            xy=(hp_x + 0.5, hp_y + 0.5),
            xytext=(4.2, 4.6),
            color=p["hp"],
            fontsize=10.5,
            ha="left",
            arrowprops=dict(arrowstyle="-", color=p["hp"], lw=0.8),
        )
        ax.text(0.15, -0.45, "fixation", color=p["fg"], fontsize=9, alpha=0.7)
        ax.set_title("Paradigm: bar PRF mapping + visual search ring",
                     fontsize=13, pad=12, loc="left")
        fig.savefig(OUT / "talk_paradigm_v1.pdf", bbox_inches="tight")
        plt.close(fig)


def paradigm_v2():
    """Mono / sketchy — pure geometry, no colors except black."""
    p = PALETTES["mono"]
    with plt.rc_context(base_rc(p)):
        fig, ax = plt.subplots(figsize=(10, 6))
        clean_axes(ax, lim=5.2)

        # Aperture + ring
        ax.add_patch(Circle((0, 0), APERTURE_RADIUS, facecolor="none",
                            edgecolor="black", lw=1.0, ls=(0, (4, 3))))
        ax.add_patch(Circle((0, 0), ECC, facecolor="none", edgecolor="black",
                            lw=0.6, ls=(0, (1, 3))))
        ax.plot(0, 0, "+", color="black", ms=10, mew=1.4)

        # Three bar positions to imply motion
        for off, alpha in zip([-2.0, 0.0, 2.0], [0.18, 0.32, 0.55]):
            rect = Rectangle((off - 0.4, -APERTURE_RADIUS), 0.8,
                              2 * APERTURE_RADIUS,
                              facecolor="black", edgecolor="none", alpha=alpha)
            ax.add_patch(rect)

        # Search positions — open circles, HP filled
        for key, (x, y) in LOCATIONS.items():
            face = "black" if key == "upper_right" else "white"
            ax.add_patch(Circle((x, y), 0.6, facecolor=face, edgecolor="black",
                                lw=1.4))

        # Tiny inset legend
        ax.text(-5.0, 5.0, "bar PRF stim", fontsize=10, ha="left", va="top")
        ax.text(-5.0, 4.6, "○ low-prob", fontsize=9.5, ha="left", va="top")
        ax.text(-5.0, 4.25, "● high-prob (HP)", fontsize=9.5, ha="left", va="top")

        ax.set_title("Paradigm", fontsize=13, pad=10, loc="left")
        fig.savefig(OUT / "talk_paradigm_v2.pdf", bbox_inches="tight")
        plt.close(fig)


def paradigm_v3():
    """Warm palette, two-panel: stimulus over time + spatial layout."""
    p = PALETTES["warm"]
    with plt.rc_context(base_rc(p)):
        fig, axes = plt.subplots(1, 2, figsize=(10, 6),
                                  gridspec_kw={"width_ratios": [1.2, 1]})
        # Left: spatial layout
        ax = axes[0]
        clean_axes(ax, lim=5.2)
        draw_visual_field(ax, p)
        # bar at center
        rect = Rectangle((-0.6, -APERTURE_RADIUS), 1.2, 2 * APERTURE_RADIUS,
                         facecolor=p["stim"], edgecolor="none", alpha=0.22)
        ax.add_patch(rect)
        draw_search_positions(ax, p, hp_key="lower_right",
                              target_key="upper_left")
        # tiny legend
        legend_handles = [
            mpatches.Patch(facecolor=p["hp"], edgecolor=p["hp"],
                           label="HP distractor"),
            mpatches.Patch(facecolor=p["target"], edgecolor=p["target"],
                           label="target (this trial)"),
            mpatches.Patch(facecolor="none", edgecolor=p["fg"],
                           label="other location"),
        ]
        ax.legend(handles=legend_handles, loc="upper left",
                  bbox_to_anchor=(-0.02, 1.0),
                  fontsize=9, frameon=False)
        ax.set_title("Visual field", fontsize=12, pad=8, loc="left")

        # Right: timeline schematic
        ax = axes[1]
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        # Three rows of "events" stacked vertically
        rows = [
            ("bar sweep", p["stim"], 0.78,
             [(0.05, 0.4), (0.30, 0.4), (0.55, 0.4), (0.80, 0.4)]),
            ("HP gain (sustained)", p["hp"], 0.50, [(0.05, 0.85)]),
            ("trial onsets (target+distr.)", p["target"], 0.22,
             [(0.12, 0.05), (0.34, 0.05), (0.58, 0.05), (0.78, 0.05)]),
        ]

        # Time axis
        ax.annotate("", xy=(0.97, 0.05), xytext=(0.03, 0.05),
                    arrowprops=dict(arrowstyle="->", color=p["muted"], lw=0.8))
        ax.text(0.97, 0.02, "time", color=p["muted"], fontsize=9,
                ha="right", va="top")

        for label, color, y, items in rows:
            ax.text(0.0, y + 0.05, label, color=color, fontsize=10,
                    ha="left", va="bottom")
            ax.plot([0.03, 0.97], [y, y], color=p["muted"], lw=0.5,
                    alpha=0.8)
            if "sustained" in label:
                ax.add_patch(Rectangle((items[0][0], y - 0.03),
                                       0.92, 0.06,
                                       facecolor=color, edgecolor="none",
                                       alpha=0.3))
            else:
                for x, _ in items:
                    ax.plot([x, x], [y - 0.04, y + 0.04], color=color, lw=2)

        axes[1].set_title("Time course", fontsize=12, pad=8, loc="left")
        fig.suptitle("Paradigm: bar PRF mapping + search ring",
                     fontsize=13, ha="left", x=0.07)
        fig.savefig(OUT / "talk_paradigm_v3.pdf", bbox_inches="tight")
        plt.close(fig)


# ---------------------------------------------------------------------------
# 2. PRF + AF
# ---------------------------------------------------------------------------

def _prf_af_panel(ax, palette, prf_center, prf_sd, af_center, af_sd,
                  show_modulated=True, show_arrows=True):
    p = palette
    lim = 5.2
    res = 200
    xs = np.linspace(-lim, lim, res)
    yy, xx = np.meshgrid(xs, xs)

    prf = gaussian_2d(xx, yy, prf_center[0], prf_center[1], prf_sd)
    af = gaussian_2d(xx, yy, af_center[0], af_center[1], af_sd)
    modulated = prf * (1 + 0.7 * af)

    # base PRF as contours
    levels = np.array([0.2, 0.5, 0.85]) * prf.max()
    ax.contour(xx, yy, prf, levels=levels, colors=p["prf"],
               linewidths=[0.8, 1.2, 1.6])
    # AF as a soft fill
    af_levels = np.array([0.3, 0.6, 0.9]) * af.max()
    cmap_af = LinearSegmentedColormap.from_list(
        "af", [(1, 1, 1, 0), to_rgba(p["af"], 0.55)])
    ax.contourf(xx, yy, af, levels=np.linspace(0.05 * af.max(), af.max(), 8),
                cmap=cmap_af)
    ax.contour(xx, yy, af, levels=af_levels, colors=p["af"], linewidths=0.6,
               alpha=0.7)

    if show_modulated:
        # apparent (modulated) PRF — dashed
        mlevels = np.array([0.2, 0.5, 0.85]) * modulated.max()
        ax.contour(xx, yy, modulated, levels=mlevels, colors=p["accent"],
                   linewidths=[0.7, 1.0, 1.4], linestyles="--")
        # apparent center: argmax
        idx = np.unravel_index(np.argmax(modulated), modulated.shape)
        cx, cy = xx[idx], yy[idx]
        ax.plot(cx, cy, "o", color=p["accent"], ms=5, mec=p["bg"], mew=1)

        if show_arrows:
            ax.annotate("",
                        xy=(cx, cy),
                        xytext=prf_center,
                        arrowprops=dict(arrowstyle="->",
                                        color=p["accent"],
                                        lw=1.2,
                                        shrinkA=4, shrinkB=4))

    # markers
    ax.plot(*prf_center, "o", color=p["prf"], ms=4, mec=p["bg"], mew=1)
    ax.plot(*af_center, "x", color=p["af"], ms=9, mew=1.6)


def prf_af_v1():
    """Cool palette, two panels: with vs without AF, and AF + apparent shift."""
    p = PALETTES["cool"]
    with plt.rc_context(base_rc(p)):
        fig, axes = plt.subplots(1, 2, figsize=(10, 6))
        for ax in axes:
            clean_axes(ax, lim=5.2)
            draw_visual_field(ax, p, fixation=True, aperture_circle=False)
            draw_search_positions(ax, p, hp_key="upper_right",
                                   show_ring=True, marker_size=0.32)

        prf_center = (1.6, 1.0)
        af_center = LOCATIONS["upper_right"]

        # Left: pure PRF
        _prf_af_panel(axes[0], p, prf_center, 1.0, af_center, 1.4,
                      show_modulated=False, show_arrows=False)
        axes[0].set_title("PRF (no attention)", fontsize=12, pad=10, loc="left")
        axes[0].text(prf_center[0] + 0.2, prf_center[1] - 0.7,
                     "PRF center", color=p["prf"], fontsize=9.5)

        # Right: PRF + AF -> shifted
        _prf_af_panel(axes[1], p, prf_center, 1.0, af_center, 1.4,
                      show_modulated=True, show_arrows=True)
        axes[1].set_title("PRF × (1 + AF)  →  apparent shift",
                          fontsize=12, pad=10, loc="left")
        axes[1].text(af_center[0] + 0.6, af_center[1] + 0.3,
                     "AF gain", color=p["af"], fontsize=9.5)
        axes[1].text(0.1, -4.6,
                     "dashed = apparent (modulated) PRF",
                     color=p["accent"], fontsize=9, style="italic", ha="left")

        fig.suptitle("Attention-field modulation of a PRF",
                     fontsize=13, x=0.07, ha="left")
        fig.savefig(OUT / "talk_prf_af_v1.pdf", bbox_inches="tight")
        plt.close(fig)


def prf_af_v2():
    """Single panel, mono — minimalist outline-only."""
    p = PALETTES["mono"]
    with plt.rc_context(base_rc(p)):
        fig, ax = plt.subplots(figsize=(10, 6))
        clean_axes(ax, lim=5.2)
        draw_search_positions(ax, p, hp_key="upper_right", show_ring=True,
                               marker_size=0.32)
        ax.plot(0, 0, "+", color="black", ms=10, mew=1.4)

        prf_center = (1.4, 0.8)
        af_center = LOCATIONS["upper_right"]
        prf_sd, af_sd = 1.0, 1.4

        # PRF: solid concentric outlines
        for k, lw in zip([1.0, 2.0], [1.4, 0.8]):
            ax.add_patch(Circle(prf_center, k * prf_sd, facecolor="none",
                                edgecolor="black", lw=lw))
        ax.plot(*prf_center, "o", color="black", ms=5)
        ax.text(prf_center[0] + 0.1, prf_center[1] - 1.4, "PRF",
                fontsize=10, ha="left")

        # AF: dashed concentric outlines
        for k, lw in zip([1.0, 2.0], [1.0, 0.6]):
            ax.add_patch(Circle(af_center, k * af_sd, facecolor="none",
                                edgecolor="black", lw=lw, ls="--"))
        ax.plot(*af_center, "x", color="black", ms=10, mew=1.6)
        ax.text(af_center[0] + 0.5, af_center[1] - 0.2, "AF",
                fontsize=10, ha="left", style="italic")

        # Apparent center: between PRF and AF, weighted toward AF
        ax_x = (prf_center[0] + 0.45 * (af_center[0] - prf_center[0]))
        ax_y = (prf_center[1] + 0.45 * (af_center[1] - prf_center[1]))
        ax.add_patch(Circle((ax_x, ax_y), prf_sd * 0.85, facecolor="none",
                            edgecolor="black", lw=1.2, ls=(0, (1, 1.5))))
        ax.plot(ax_x, ax_y, "o", color="black", ms=5, mec="white", mew=1.2)
        # arrow shift
        arr = FancyArrowPatch(prf_center, (ax_x, ax_y),
                              arrowstyle="->", color="black", lw=1.0,
                              mutation_scale=12, shrinkA=4, shrinkB=4)
        ax.add_patch(arr)
        ax.text(ax_x + 0.1, ax_y + 0.5, "apparent\nPRF",
                fontsize=9, ha="left", va="bottom")

        ax.set_title("PRF × (1 + AF) shifts the apparent receptive field",
                     fontsize=12, pad=10, loc="left")
        fig.savefig(OUT / "talk_prf_af_v2.pdf", bbox_inches="tight")
        plt.close(fig)


def prf_af_v3():
    """Warm palette, three small panels: PRF | AF | PRF×(1+AF)."""
    p = PALETTES["warm"]
    with plt.rc_context(base_rc(p)):
        fig, axes = plt.subplots(1, 3, figsize=(10, 4.2))

        prf_center = (1.6, 1.0)
        af_center = LOCATIONS["upper_right"]
        lim = 5.2
        res = 220
        xs = np.linspace(-lim, lim, res)
        yy, xx = np.meshgrid(xs, xs)
        prf = gaussian_2d(xx, yy, prf_center[0], prf_center[1], 1.0)
        af = gaussian_2d(xx, yy, af_center[0], af_center[1], 1.4)
        modulated = prf * (1 + 0.7 * af)

        cmap_prf = LinearSegmentedColormap.from_list(
            "prf", [(1, 1, 1, 0), to_rgba(p["prf"], 0.7)])
        cmap_af = LinearSegmentedColormap.from_list(
            "af", [(1, 1, 1, 0), to_rgba(p["af"], 0.7)])
        cmap_mod = LinearSegmentedColormap.from_list(
            "mod", [(1, 1, 1, 0), to_rgba(p["accent"], 0.9)])

        for ax, data, cmap, title, color in [
            (axes[0], prf, cmap_prf, "PRF", p["prf"]),
            (axes[1], af, cmap_af, "AF (gain field)", p["af"]),
            (axes[2], modulated, cmap_mod, "PRF × (1 + AF)", p["accent"]),
        ]:
            clean_axes(ax, lim=lim)
            ax.contourf(xx, yy, data,
                        levels=np.linspace(0.05 * data.max(), data.max(), 10),
                        cmap=cmap)
            # Search ring
            ax.add_patch(Circle((0, 0), ECC, facecolor="none",
                                edgecolor=p["muted"], lw=0.5,
                                ls=(0, (1, 3))))
            ax.plot(0, 0, "+", color=p["fg"], ms=7, mew=1.0)
            for key, (x, y) in LOCATIONS.items():
                face = p["hp"] if key == "upper_right" else "none"
                ax.add_patch(Circle((x, y), 0.25, facecolor=face,
                                    edgecolor=p["fg"], lw=0.7))
            ax.set_title(title, fontsize=11, pad=8, color=color, loc="left")

        # Operator labels between panels
        fig.text(0.355, 0.5, "×", fontsize=18, ha="center", va="center",
                 color=p["fg"])
        fig.text(0.66, 0.5, "=", fontsize=18, ha="center", va="center",
                 color=p["fg"])
        fig.text(0.36, 0.42, "(1 + ·)", fontsize=9, ha="center",
                 va="center", color=p["muted"], style="italic")

        fig.suptitle("Attention-field modulation",
                     fontsize=13, x=0.07, ha="left", y=0.98)
        fig.savefig(OUT / "talk_prf_af_v3.pdf", bbox_inches="tight")
        plt.close(fig)


# ---------------------------------------------------------------------------
# 3. COMPOSITIONAL MODEL
# ---------------------------------------------------------------------------

def compositional_v1():
    """Cool, equation + small spatial+time tiles for each gain component."""
    p = PALETTES["cool"]
    with plt.rc_context(base_rc(p)):
        fig = plt.figure(figsize=(10, 6))
        # Layout: top equation, three spatial tiles, three timing tiles
        gs = fig.add_gridspec(3, 4, height_ratios=[0.6, 1.2, 0.6],
                              hspace=0.25, wspace=0.2,
                              left=0.05, right=0.97, top=0.92, bottom=0.06)

        # Equation
        eq_ax = fig.add_subplot(gs[0, :])
        eq_ax.axis("off")
        eq_ax.set_xlim(0, 1); eq_ax.set_ylim(0, 1)
        eq_ax.text(0.5, 0.5,
                   r"$M(\mathbf{x},t) \; = \; 1 \; + \; "
                   r"g_{\mathrm{HP}}\,G_{\mathrm{HP}}(\mathbf{x}) \; + \; "
                   r"g_{\mathrm{D}}\,G_{\mathrm{D}}(\mathbf{x})\,d(t) \; + \; "
                   r"g_{\mathrm{T}}\,G_{\mathrm{T}}(\mathbf{x})\,\tau(t)$",
                   fontsize=14, ha="center", va="center", color=p["fg"])

        labels = ["sustained HP", "dynamic distractor", "target capture"]
        colors = [p["hp"], p["af"], p["target"]]
        centers = [LOCATIONS["upper_right"],
                   LOCATIONS["lower_left"],  # distractor on this trial
                   LOCATIONS["upper_left"]]  # target on this trial

        # Spatial tiles
        for i, (label, color, c) in enumerate(zip(labels, colors, centers)):
            ax = fig.add_subplot(gs[1, i])
            clean_axes(ax, lim=5.2)
            ax.add_patch(Circle((0, 0), ECC, facecolor="none",
                                edgecolor=p["muted"], lw=0.4,
                                ls=(0, (1, 3))))
            ax.plot(0, 0, "+", color=p["fg"], ms=6, mew=1.0)
            res = 150
            xs = np.linspace(-5.2, 5.2, res)
            yy, xx = np.meshgrid(xs, xs)
            af = gaussian_2d(xx, yy, c[0], c[1], 1.3)
            cmap = LinearSegmentedColormap.from_list(
                "af", [(1, 1, 1, 0), to_rgba(color, 0.75)])
            ax.contourf(xx, yy, af,
                        levels=np.linspace(0.05 * af.max(), af.max(), 8),
                        cmap=cmap)
            for key, (x, y) in LOCATIONS.items():
                ax.add_patch(Circle((x, y), 0.25, facecolor="none",
                                    edgecolor=p["fg"], lw=0.6))
            ax.set_title(label, fontsize=10.5, pad=6, color=color, loc="left")

        # Final tile: combined
        ax = fig.add_subplot(gs[1, 3])
        clean_axes(ax, lim=5.2)
        ax.add_patch(Circle((0, 0), ECC, facecolor="none",
                            edgecolor=p["muted"], lw=0.4, ls=(0, (1, 3))))
        ax.plot(0, 0, "+", color=p["fg"], ms=6, mew=1.0)
        res = 150
        xs = np.linspace(-5.2, 5.2, res)
        yy, xx = np.meshgrid(xs, xs)
        gain = np.ones_like(xx)
        for c, color, w in zip(centers, colors, [0.6, 0.5, 0.45]):
            gain = gain + w * gaussian_2d(xx, yy, c[0], c[1], 1.3)
        cmap = LinearSegmentedColormap.from_list(
            "all", [to_rgba(p["bg"], 0), to_rgba(p["accent"], 0.7)])
        ax.contourf(xx, yy, gain,
                    levels=np.linspace(gain.min(), gain.max(), 12),
                    cmap=cmap)
        for key, (x, y) in LOCATIONS.items():
            ax.add_patch(Circle((x, y), 0.25, facecolor="none",
                                edgecolor=p["fg"], lw=0.6))
        ax.set_title(r"$M(\mathbf{x},t)$", fontsize=10.5, pad=6,
                     color=p["accent"], loc="left")

        # Timing tiles below the first 3
        timing = [
            ("on entire run", "sustained"),
            ("distractor onsets", "transient"),
            ("target onsets", "transient"),
        ]
        for i, ((kind, _), color) in enumerate(zip(timing, colors)):
            ax = fig.add_subplot(gs[2, i])
            ax.axis("off")
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.plot([0.05, 0.95], [0.5, 0.5], color=p["muted"], lw=0.6)
            if "sustained" in kind:
                ax.add_patch(Rectangle((0.05, 0.43), 0.9, 0.14,
                                       facecolor=color, edgecolor="none",
                                       alpha=0.4))
            else:
                for x in [0.18, 0.36, 0.55, 0.78]:
                    ax.plot([x, x], [0.32, 0.68], color=color, lw=2)
            ax.text(0.5, 0.0, kind, fontsize=9, ha="center", va="bottom",
                    color=color)

        fig.text(0.55, 0.97, "Compositional gain model",
                 fontsize=13, ha="center", va="top")
        fig.savefig(OUT / "talk_compositional_v1.pdf", bbox_inches="tight")
        plt.close(fig)


def compositional_v2():
    """Mono, stack/sum cartoon: 1 + g1 + g2 + g3 with arrows showing summation."""
    p = PALETTES["mono"]
    with plt.rc_context(base_rc(p)):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_xlim(0, 10); ax.set_ylim(0, 6)
        ax.axis("off")

        def small_field(cx, cy, gain_centers=None, baseline=False, label="",
                        sublabel=""):
            # box
            r = 0.9
            ax.add_patch(FancyBboxPatch((cx - r, cy - r), 2 * r, 2 * r,
                                        boxstyle="round,pad=0.02",
                                        facecolor="white",
                                        edgecolor="black", lw=0.8))
            # Inside: a tiny diagram
            for key, (x, y) in LOCATIONS.items():
                ax.add_patch(Circle((cx + 0.4 * x / ECC, cy + 0.4 * y / ECC),
                                    0.06, facecolor="white",
                                    edgecolor="black", lw=0.5))
            if baseline:
                ax.add_patch(Circle((cx, cy), 0.7, facecolor="black",
                                    edgecolor="none", alpha=0.08))
                ax.text(cx, cy, "1", fontsize=14, ha="center", va="center",
                        color="black")
            if gain_centers:
                for (gx, gy), alpha in gain_centers:
                    ax.add_patch(Circle(
                        (cx + 0.55 * gx / ECC, cy + 0.55 * gy / ECC),
                        0.35, facecolor="black", edgecolor="none",
                        alpha=alpha))
            ax.text(cx, cy - r - 0.3, label, fontsize=10.5, ha="center",
                    va="top")
            if sublabel:
                ax.text(cx, cy - r - 0.7, sublabel, fontsize=8.5,
                        ha="center", va="top", style="italic", color="#666")

        # Four panels in a row + result
        positions = [1.2, 3.2, 5.2, 7.2, 9.2]
        small_field(positions[0], 3.5, baseline=True,
                    label="1", sublabel="constant")
        small_field(positions[1], 3.5,
                    gain_centers=[(LOCATIONS["upper_right"], 0.6)],
                    label=r"$g_{\mathrm{HP}}\,G_{\mathrm{HP}}$",
                    sublabel="sustained HP")
        small_field(positions[2], 3.5,
                    gain_centers=[(LOCATIONS["lower_left"], 0.55)],
                    label=r"$g_{\mathrm{D}}\,G_{\mathrm{D}}\,d(t)$",
                    sublabel="distractor onset")
        small_field(positions[3], 3.5,
                    gain_centers=[(LOCATIONS["upper_left"], 0.55)],
                    label=r"$g_{\mathrm{T}}\,G_{\mathrm{T}}\,\tau(t)$",
                    sublabel="target onset")
        small_field(positions[4], 3.5,
                    gain_centers=[(LOCATIONS["upper_right"], 0.45),
                                  (LOCATIONS["lower_left"], 0.4),
                                  (LOCATIONS["upper_left"], 0.4)],
                    baseline=False,
                    label=r"$M(\mathbf{x},t)$",
                    sublabel="full gain field")

        # Plus signs and equals
        for i, (a, b) in enumerate(zip(positions[:-2], positions[1:-1])):
            xm = (a + b) / 2
            ax.text(xm, 3.5, "+", fontsize=18, ha="center", va="center")
        xm = (positions[-2] + positions[-1]) / 2
        ax.text(xm, 3.5, "=", fontsize=18, ha="center", va="center")

        ax.text(0.2, 5.6, "Compositional gain model",
                fontsize=13, ha="left", va="top")
        ax.text(0.2, 5.2,
                r"each component contributes a Gaussian gain field, gated by its own time course",
                fontsize=10, ha="left", va="top", color="#555")

        fig.savefig(OUT / "talk_compositional_v2.pdf", bbox_inches="tight")
        plt.close(fig)


def compositional_v3():
    """Warm, vertical-stack 'building up' from baseline."""
    p = PALETTES["warm"]
    with plt.rc_context(base_rc(p)):
        fig, axes = plt.subplots(1, 4, figsize=(10, 4.5))

        configs = [
            ("baseline", [], "$M = 1$"),
            ("+ sustained HP",
             [(LOCATIONS["upper_right"], 1.3, p["hp"], 0.7)],
             "$+ g_{HP}G_{HP}(\\mathbf{x})$"),
            ("+ dynamic distractor",
             [(LOCATIONS["upper_right"], 1.3, p["hp"], 0.7),
              (LOCATIONS["lower_left"], 1.2, p["af"], 0.6)],
             "$+ g_D G_D(\\mathbf{x})d(t)$"),
            ("+ target capture",
             [(LOCATIONS["upper_right"], 1.3, p["hp"], 0.7),
              (LOCATIONS["lower_left"], 1.2, p["af"], 0.6),
              (LOCATIONS["upper_left"], 1.2, p["target"], 0.6)],
             "$+ g_T G_T(\\mathbf{x})\\tau(t)$"),
        ]

        res = 160
        xs = np.linspace(-5.2, 5.2, res)
        yy, xx = np.meshgrid(xs, xs)

        for ax, (title, gains, eqn) in zip(axes, configs):
            clean_axes(ax, lim=5.2)
            field = np.ones_like(xx)
            for c, sd, color, w in gains:
                field = field + w * gaussian_2d(xx, yy, c[0], c[1], sd)
            cmap = LinearSegmentedColormap.from_list(
                "build", [to_rgba(p["bg"], 0), to_rgba(p["accent"], 0.7)])
            fmin, fmax = float(field.min()), float(field.max())
            if fmax - fmin < 1e-6:
                # Uniform baseline — render a faint flat tile instead.
                ax.add_patch(Circle((0, 0), 5.2, facecolor=to_rgba(
                    p["accent"], 0.08), edgecolor="none"))
            else:
                ax.contourf(xx, yy, field,
                            levels=np.linspace(fmin, fmax, 14),
                            cmap=cmap)
            ax.add_patch(Circle((0, 0), ECC, facecolor="none",
                                edgecolor=p["muted"], lw=0.4,
                                ls=(0, (1, 3))))
            ax.plot(0, 0, "+", color=p["fg"], ms=6, mew=1.0)
            for key, (x, y) in LOCATIONS.items():
                ax.add_patch(Circle((x, y), 0.25, facecolor="none",
                                    edgecolor=p["fg"], lw=0.6))
            ax.set_title(title, fontsize=10.5, pad=8, loc="left",
                         color=p["accent"])
            ax.text(0.5, -0.18, eqn, fontsize=9.5, transform=ax.transAxes,
                    ha="center", va="top", color=p["fg"])

        fig.suptitle("Building up the compositional gain field",
                     fontsize=13, x=0.07, ha="left", y=1.0)
        fig.savefig(OUT / "talk_compositional_v3.pdf", bbox_inches="tight")
        plt.close(fig)


# ---------------------------------------------------------------------------
# 4. PIPELINE / GRAPHICAL ABSTRACT
# ---------------------------------------------------------------------------

def _draw_arrow(ax, x0, x1, y, color, label=None, label_y=None):
    arr = FancyArrowPatch((x0, y), (x1, y), arrowstyle="->", color=color,
                          lw=1.2, mutation_scale=14)
    ax.add_patch(arr)
    if label:
        ly = label_y if label_y is not None else y + 0.25
        ax.text((x0 + x1) / 2, ly, label, fontsize=9, ha="center",
                va="bottom", style="italic", color=color)


def pipeline_v1():
    """Cool, horizontal: stim -> PRF -> AF mod -> predicted BOLD."""
    p = PALETTES["cool"]
    with plt.rc_context(base_rc(p)):
        fig = plt.figure(figsize=(10, 6))
        gs = fig.add_gridspec(1, 4, wspace=0.35, left=0.04, right=0.98,
                              top=0.85, bottom=0.18)

        # 1: stim (bar + ring)
        ax1 = fig.add_subplot(gs[0, 0])
        clean_axes(ax1, lim=5.2)
        ax1.add_patch(Circle((0, 0), APERTURE_RADIUS, facecolor="none",
                             edgecolor=p["muted"], lw=0.6, ls=(0, (4, 3))))
        ax1.add_patch(Rectangle((-0.7, -APERTURE_RADIUS), 1.4,
                                2 * APERTURE_RADIUS,
                                facecolor=p["stim"], alpha=0.25,
                                edgecolor="none"))
        ax1.plot(0, 0, "+", color=p["fg"], ms=7, mew=1.0)
        for key, (x, y) in LOCATIONS.items():
            face = p["hp"] if key == "upper_right" else "none"
            ax1.add_patch(Circle((x, y), 0.4, facecolor=face,
                                 edgecolor=p["fg"], lw=0.8))
        ax1.set_title("stimulus", fontsize=11, pad=8, loc="left",
                      color=p["stim"])

        # 2: PRF
        ax2 = fig.add_subplot(gs[0, 1])
        clean_axes(ax2, lim=5.2)
        ax2.add_patch(Circle((0, 0), ECC, facecolor="none",
                             edgecolor=p["muted"], lw=0.5, ls=(0, (1, 3))))
        prf_c = (1.6, 1.0)
        for k, lw, alpha in zip([0.6, 1.0, 1.5], [1.6, 1.0, 0.6],
                                [1, 1, 1]):
            ax2.add_patch(Circle(prf_c, k * 1.0, facecolor="none",
                                 edgecolor=p["prf"], lw=lw, alpha=alpha))
        ax2.plot(*prf_c, "o", color=p["prf"], ms=4)
        ax2.plot(0, 0, "+", color=p["fg"], ms=7, mew=1.0)
        ax2.set_title("PRF", fontsize=11, pad=8, loc="left", color=p["prf"])

        # 3: AF-modulated PRF
        ax3 = fig.add_subplot(gs[0, 2])
        clean_axes(ax3, lim=5.2)
        ax3.add_patch(Circle((0, 0), ECC, facecolor="none",
                             edgecolor=p["muted"], lw=0.5, ls=(0, (1, 3))))
        # AF
        af_c = LOCATIONS["upper_right"]
        res = 200
        xs = np.linspace(-5.2, 5.2, res)
        yy, xx = np.meshgrid(xs, xs)
        prf = gaussian_2d(xx, yy, prf_c[0], prf_c[1], 1.0)
        af = gaussian_2d(xx, yy, af_c[0], af_c[1], 1.3)
        modulated = prf * (1 + 0.7 * af)
        cmap = LinearSegmentedColormap.from_list(
            "mod", [(1, 1, 1, 0), to_rgba(p["accent"], 0.85)])
        ax3.contourf(xx, yy, modulated,
                     levels=np.linspace(0.05 * modulated.max(),
                                        modulated.max(), 8), cmap=cmap)
        ax3.contour(xx, yy, prf,
                    levels=[0.5 * prf.max()], colors=[p["prf"]],
                    linewidths=[0.8], linestyles="--")
        ax3.plot(*af_c, "x", color=p["af"], ms=8, mew=1.4)
        ax3.plot(0, 0, "+", color=p["fg"], ms=7, mew=1.0)
        ax3.set_title("PRF × (1 + AF)", fontsize=11, pad=8, loc="left",
                      color=p["accent"])

        # 4: BOLD prediction (timeseries)
        ax4 = fig.add_subplot(gs[0, 3])
        ax4.set_facecolor(p["bg"])
        for s in ax4.spines.values():
            s.set_visible(False)
        ax4.spines["bottom"].set_visible(True)
        ax4.spines["bottom"].set_color(p["muted"])
        t = np.linspace(0, 30, 400)
        # synthesise two timecourses
        baseline = np.exp(-((t - 6) ** 2) / 4) + 0.7 * np.exp(
            -((t - 14) ** 2) / 5) + 0.55 * np.exp(-((t - 24) ** 2) / 4)
        boosted = (baseline +
                   0.35 * np.exp(-((t - 14) ** 2) / 5) +
                   0.25 * np.exp(-((t - 24) ** 2) / 4))
        ax4.plot(t, baseline, color=p["prf"], lw=1.2, label="PRF only")
        ax4.plot(t, boosted, color=p["accent"], lw=1.5, label="with AF")
        ax4.set_xticks([])
        ax4.set_yticks([])
        ax4.legend(fontsize=8.5, frameon=False, loc="upper right")
        ax4.set_title("predicted BOLD", fontsize=11, pad=8, loc="left",
                      color=p["fg"])

        # Arrows between subplots (in figure coordinates)
        def fig_arrow(x0, x1, y=0.5, label=""):
            arr = FancyArrowPatch((x0, y), (x1, y),
                                   transform=fig.transFigure,
                                   arrowstyle="->", color=p["muted"],
                                   lw=1.0, mutation_scale=14)
            fig.patches.append(arr)
            if label:
                fig.text((x0 + x1) / 2, y + 0.04, label, fontsize=9,
                         ha="center", va="bottom", color=p["muted"],
                         style="italic")

        fig_arrow(0.245, 0.275, label="encode")
        fig_arrow(0.485, 0.515, label="× (1+AF)")
        fig_arrow(0.735, 0.765, label="convolve HRF")

        fig.suptitle("Pipeline: stimulus → PRF → AF modulation → BOLD",
                     fontsize=13, x=0.04, ha="left", y=0.97)
        fig.savefig(OUT / "talk_pipeline_v1.pdf", bbox_inches="tight")
        plt.close(fig)


def pipeline_v2():
    """Mono, blocky abstract — boxes + arrows, almost flowchart."""
    p = PALETTES["mono"]
    with plt.rc_context(base_rc(p)):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")

        # Boxes
        boxes = [
            (0.5, 2.5, 1.6, 1.4, "stimulus\n(bar + ring)"),
            (3.0, 2.5, 1.6, 1.4, "PRF\n$G(\\mathbf{x};\\mu,\\sigma)$"),
            (5.5, 2.5, 1.6, 1.4, "AF gain\n$1 + \\Sigma\\, gG_i\\,t_i$"),
            (8.0, 2.5, 1.6, 1.4, "BOLD\nprediction"),
        ]
        for (x, y, w, h, label) in boxes:
            ax.add_patch(FancyBboxPatch((x, y), w, h,
                                        boxstyle="round,pad=0.04",
                                        facecolor="white", edgecolor="black",
                                        lw=1.0))
            ax.text(x + w / 2, y + h / 2, label, fontsize=10.5,
                    ha="center", va="center")

        # Arrows
        for i in range(len(boxes) - 1):
            x0 = boxes[i][0] + boxes[i][2]
            x1 = boxes[i + 1][0]
            y = boxes[i][1] + boxes[i][3] / 2
            arr = FancyArrowPatch((x0, y), (x1, y), arrowstyle="->",
                                   color="black", lw=1.2, mutation_scale=14)
            ax.add_patch(arr)

        # Sub-labels above arrows
        labels = ["sample", "× modulate", "* HRF"]
        for i, lab in enumerate(labels):
            x0 = boxes[i][0] + boxes[i][2]
            x1 = boxes[i + 1][0]
            y = boxes[i][1] + boxes[i][3] / 2 + 0.15
            ax.text((x0 + x1) / 2, y, lab, fontsize=9, ha="center",
                    va="bottom", style="italic")

        # Below boxes: tiny illustrative thumbnails
        thumb_y = 1.1
        thumb_h = 1.0

        # 1: stimulus thumbnail
        tx, tw = 0.9, 0.8
        ax.add_patch(Rectangle((tx, thumb_y), tw, thumb_h,
                               facecolor="white", edgecolor="black",
                               lw=0.5))
        ax.add_patch(Rectangle((tx + tw * 0.4, thumb_y + 0.05),
                               0.12, thumb_h - 0.1,
                               facecolor="black", alpha=0.3, edgecolor="none"))

        # 2: PRF thumbnail
        tx = 3.4
        ax.add_patch(Rectangle((tx, thumb_y), tw, thumb_h,
                               facecolor="white", edgecolor="black", lw=0.5))
        ax.add_patch(Circle((tx + tw / 2, thumb_y + thumb_h / 2),
                            0.18, facecolor="none", edgecolor="black",
                            lw=1.0))
        ax.add_patch(Circle((tx + tw / 2, thumb_y + thumb_h / 2),
                            0.32, facecolor="none", edgecolor="black",
                            lw=0.5))

        # 3: AF thumbnail
        tx = 5.9
        ax.add_patch(Rectangle((tx, thumb_y), tw, thumb_h,
                               facecolor="white", edgecolor="black", lw=0.5))
        for sx, sy in [(0.55, 0.7), (0.3, 0.4)]:
            ax.add_patch(Circle((tx + tw * sx, thumb_y + thumb_h * sy),
                                0.13, facecolor="black",
                                alpha=0.25, edgecolor="none"))

        # 4: BOLD timeseries thumbnail
        tx = 8.4
        ax.add_patch(Rectangle((tx, thumb_y), tw, thumb_h,
                               facecolor="white", edgecolor="black", lw=0.5))
        t = np.linspace(0, 1, 80)
        ts = (np.exp(-((t - 0.25) ** 2) * 90) +
              0.6 * np.exp(-((t - 0.55) ** 2) * 60) +
              0.5 * np.exp(-((t - 0.85) ** 2) * 80))
        ax.plot(tx + t * tw, thumb_y + 0.15 + 0.7 * ts / ts.max(),
                color="black", lw=1.0)

        ax.text(0.2, 5.4, "Pipeline", fontsize=14, ha="left", va="top")
        ax.text(0.2, 4.95,
                "stimulus → PRF response × AF gain → BOLD",
                fontsize=10.5, ha="left", va="top", color="#555")

        fig.savefig(OUT / "talk_pipeline_v2.pdf", bbox_inches="tight")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Run all
# ---------------------------------------------------------------------------

def main():
    paradigm_v1()
    paradigm_v2()
    paradigm_v3()
    prf_af_v1()
    prf_af_v2()
    prf_af_v3()
    compositional_v1()
    compositional_v2()
    compositional_v3()
    pipeline_v1()
    pipeline_v2()
    for f in sorted(OUT.glob("talk_*.pdf")):
        print(f)


if __name__ == "__main__":
    main()
