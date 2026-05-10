"""
Slide-by-slide build of the retsupp compositional gain model, for the talk.

Six PDFs (talk_buildup_0{1..6}_*.pdf) staged as a sequential reveal:
    01 design (outlines only)
    02 sustained gain at all 4 search positions, equal — symmetric arrows
    03 HP weakened (suppression) — asymmetric arrows, net shift away from HP
    04 + target attractor (transient)
    05 + dynamic distractor (transient)
    06 full equation with σ_X / G_X / μ_X / g_X labelled
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from matplotlib.patches import Circle, Rectangle

OUT = Path(__file__).parent

ECC = 4.0  # ring eccentricity in degrees
SIZE = 1.5  # search-array item footprint
APERTURE_RADIUS = ECC - SIZE / 1.8  # ~3.17

# Four corner positions (degrees of visual angle).
LOCATIONS = {
    "upper_right": (ECC / np.sqrt(2), +ECC / np.sqrt(2)),
    "upper_left": (-ECC / np.sqrt(2), +ECC / np.sqrt(2)),
    "lower_left": (-ECC / np.sqrt(2), -ECC / np.sqrt(2)),
    "lower_right": (ECC / np.sqrt(2), -ECC / np.sqrt(2)),
}

# Buildup-specific palette: HP red, target orange, dynamic distractor blue —
# chosen so the three components are clearly distinguishable when overlaid.
BUILDUP_PALETTE = {
    "bg": "#FBFBFD",
    "fg": "#1F2933",
    "muted": "#9AA5B1",
    "stim": "#3D5A80",
    "hp": "#D62828",       # red — sustained HP
    "target": "#F77F00",   # orange — target
    "af": "#0077B6",       # blue — dynamic distractor
    "accent": "#293241",
    "prf": "#2A9D8F",      # deep teal — example PRF outlines / CoM arrows
                            # (chosen to be distinct from HP red, target
                            # orange, and AF blue, and to read clearly on
                            # both light and tinted backgrounds)
}

# Trial-level convention used across slides 04–05.
HP_KEY = "upper_right"
TARGET_KEY = "upper_left"
DYN_DIST_KEY = "lower_left"

# Per-component parameters shared across the build. The transient gain fields
# (σ_TRANSIENT) are intentionally LARGER than the sustained ones so the
# distractor/target effects spread visibly across adjacent quadrants — this
# makes the multiplicative interaction with the bar PRF stimulus more
# compelling on slide 07 and the spatial pull on slides 02–05 more obvious.
SD_SUSTAINED = 1.3
SD_TRANSIENT = 1.8
AMP_SUSTAINED_FULL = 0.7      # gain for non-HP candidate locations
AMP_SUSTAINED_HP = 0.18       # weakened HP gain (suppression story)
AMP_TARGET = 0.65             # target capture (enhancement)
AMP_DYN = -0.45               # dynamic distractor SUPPRESSES at its location
PRF_SD = 1.0

# Example PRFs to illustrate gain-driven reshaping. Two rings:
#   - 6 peripheral PRFs (r ≈ 2°, σ = 0.7°) covering the 4 quadrants plus
#     2 cardinal directions
#   - 3 foveal PRFs (r ≈ 0.85°, σ = 0.45°) — smaller and closer to the
#     fovea to illustrate that retinotopically central PRFs are much less
#     affected by the corner gain blobs than peripheral ones
# Style inspired by `retsupp/visualize/vss2026/animate_attention_field.py`.
PRF_EXAMPLES = [
    # Peripheral ring (σ = 0.7°)
    ((+1.7, +1.7), 0.7),    # near upper_right (= HP in slides 03–05)
    ((-1.7, +1.7), 0.7),    # near upper_left (= target in slides 04–05)
    ((-1.7, -1.7), 0.7),    # near lower_left (= distractor in slide 05)
    ((+1.7, -1.7), 0.7),    # near lower_right
    ((+0.0, +2.0), 0.7),    # cardinal top
    ((+0.0, -2.0), 0.7),    # cardinal bottom
    # Foveal ring (σ = 0.45°) — smaller PRFs near fixation
    ((+0.85, +0.0), 0.45),  # right of fixation
    ((-0.42, +0.74), 0.45), # upper-left of fixation (120° from +x)
    ((-0.42, -0.74), 0.45), # lower-left of fixation (240° from +x)
]


def base_rc(palette):
    # Presentation-grade font sizes: bumped up so the figures read clearly
    # when projected. Annotations should stay legible at slide scale.
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
        "font.size": 20,
        "axes.titlesize": 24,
        "axes.labelsize": 20,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 18,
        "figure.titlesize": 26,
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


def gaussian_2d(xx, yy, cx, cy, sd):
    return np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sd ** 2))


# ---------------------------------------------------------------------------
# Helpers

def _gain_components_sustained_equal():
    return [(LOCATIONS[k][0], LOCATIONS[k][1], SD_SUSTAINED, AMP_SUSTAINED_FULL)
            for k in LOCATIONS]


def _gain_components_hp_weakened():
    comps = []
    for k, (x, y) in LOCATIONS.items():
        amp = AMP_SUSTAINED_HP if k == HP_KEY else AMP_SUSTAINED_FULL
        comps.append((x, y, SD_SUSTAINED, amp))
    return comps


def _build_modulation_field(gain_components, lim=5.2, res=240):
    """Return (xx, yy, M) where M = 1 + Σ amp_i · G_i."""
    xs = np.linspace(-lim, lim, res)
    yy, xx = np.meshgrid(xs, xs)
    M = np.ones_like(xx)
    for cx, cy, sd, amp in gain_components:
        M = M + amp * gaussian_2d(xx, yy, cx, cy, sd)
    return xx, yy, M


def _example_prf_coms(gain_components, prf_examples=PRF_EXAMPLES,
                      lim=5.2, res=240):
    """Return CoM(PRF × max(M, 0)) for each example PRF.
    Used by the net-shift arrow on slide 03 to summarise the 3-PRF mean
    displacement.
    """
    xx, yy, M = _build_modulation_field(gain_components, lim=lim, res=res)
    M_pos = np.maximum(M, 0.0)
    out = np.empty((len(prf_examples), 2))
    origins = np.empty((len(prf_examples), 2))
    for i, ((px, py), prf_sd) in enumerate(prf_examples):
        prf = gaussian_2d(xx, yy, px, py, prf_sd)
        eff = prf * M_pos
        tot = eff.sum()
        if tot > 1e-9:
            out[i] = ((xx * eff).sum() / tot, (yy * eff).sum() / tot)
        else:
            out[i] = (px, py)
        origins[i] = (px, py)
    return origins, out


def _draw_gradient_quiver(ax, gain_components, palette, lim=5.2, res=200,
                          step=18, scale=4.5, color="#5A6470", alpha=0.85,
                          width=0.0085, mask_radius=APERTURE_RADIUS):
    """∇M quiver field — visualises the spatial 'pull' of the gain field
    underneath the example PRFs. Tuned for visibility at slide scale.

    `scale` is the matplotlib quiver scale: SMALLER scale → LARGER arrows.
    Gray color (default `#5A6470`) so the field reads as 'background
    physics' and doesn't compete with the colored gain blobs or PRFs.

    `mask_radius` (default = bar aperture radius) clips the quiver to
    inside the PRF-mapping aperture. Vectors outside the aperture are
    not drawn since there's no stimulus drive there anyway.
    """
    xx, yy, M = _build_modulation_field(gain_components, lim=lim, res=res)
    gy, gx = np.gradient(M, lim * 2 / (res - 1), lim * 2 / (res - 1))
    sl = (slice(step // 2, None, step), slice(step // 2, None, step))
    xs = xx[sl]; ys = yy[sl]
    us = gx[sl]; vs = gy[sl]
    if mask_radius is not None:
        mask = (xs ** 2 + ys ** 2) <= mask_radius ** 2
        xs = xs[mask]; ys = ys[mask]
        us = us[mask]; vs = vs[mask]
    ax.quiver(xs, ys, us, vs,
              color=color, alpha=alpha, scale=scale,
              scale_units="inches", width=width,
              headwidth=4.5, headlength=5.5, headaxislength=4.5,
              zorder=1)


def _draw_three_example_prfs(ax, gain_components, palette,
                             prf_examples=PRF_EXAMPLES, lim=5.2, res=240,
                             color=None, lw_orig=1.4, lw_eff=2.6,
                             arrow_lw=3.0, com_amplify=1.4,
                             show_quiver=True):
    """Render 3 example PRFs reshaped by the gain field.

    For each PRF, draws:
      - dotted 1σ outline of the unmodulated PRF (its 'native' shape)
      - solid half-max contour of the effective RF (PRF × max(M, 0)) —
        this captures the actual asymmetric reshaped RF, not a translated circle
      - prominent arrow from the original PRF center to the CoM of the
        effective RF (optionally amplified by `com_amplify` for visibility)

    A soft ∇M quiver is drawn underneath when `show_quiver=True` so the
    gain field's spatial 'pull' is visible everywhere, not just at the
    3 example PRFs.

    Style follows `retsupp/visualize/vss2026/animate_attention_field.py`.
    """
    color = color or palette["prf"]
    if show_quiver:
        _draw_gradient_quiver(ax, gain_components, palette, lim=lim)
    xx, yy, M = _build_modulation_field(gain_components, lim=lim, res=res)
    M_pos = np.maximum(M, 0.0)

    for (px, py), prf_sd in prf_examples:
        ax.add_patch(Circle((px, py), prf_sd, facecolor="none",
                            edgecolor=color, lw=lw_orig, ls=":", alpha=0.85,
                            zorder=8))
        ax.plot(px, py, "+", color=color, ms=8, mew=1.4, alpha=0.9, zorder=9)

        prf = gaussian_2d(xx, yy, px, py, prf_sd)
        eff = prf * M_pos
        if eff.max() > 1e-6:
            ax.contour(xx, yy, eff, levels=[eff.max() * 0.5],
                       colors=[color], linewidths=lw_eff, alpha=0.95,
                       zorder=9)
            tot = eff.sum()
            com_x = (xx * eff).sum() / tot
            com_y = (yy * eff).sum() / tot
        else:
            com_x, com_y = px, py

        # CoM arrow — amplified slightly so the displacement is visible
        # at slide scale even when it's small in absolute degrees.
        dx = (com_x - px) * com_amplify
        dy = (com_y - py) * com_amplify
        if dx ** 2 + dy ** 2 > 1e-3:
            ax.annotate("", xy=(px + dx, py + dy), xytext=(px, py),
                        arrowprops=dict(arrowstyle="-|>", color=color,
                                        lw=arrow_lw, mutation_scale=22,
                                        alpha=0.98, shrinkA=0, shrinkB=0),
                        zorder=10)


def _draw_layer(ax, comps, color, lim=5.2, res=240, alpha=0.6, n_levels=10):
    """Soft contourf of the contribution of `comps` (a list of
    (cx, cy, sd, amp) tuples) on top of a transparent background.

    Plots the magnitude |amp · G| — the SIGN is conveyed by the chosen
    color (e.g. blue for distractor suppression, orange for target
    enhancement), not by the contour levels. So a single layer must use
    components of consistent sign (which is the case for our HP / target /
    distractor decomposition).
    """
    xs = np.linspace(-lim, lim, res)
    yy, xx = np.meshgrid(xs, xs)
    field = np.zeros_like(xx)
    for cx, cy, sd, amp in comps:
        field = field + amp * gaussian_2d(xx, yy, cx, cy, sd)
    field_abs = np.abs(field)
    if field_abs.max() < 1e-6:
        return
    cmap = LinearSegmentedColormap.from_list(
        "layer", [(1, 1, 1, 0), to_rgba(color, alpha)])
    ax.contourf(xx, yy, field_abs,
                levels=np.linspace(0.04 * field_abs.max(),
                                   field_abs.max(), n_levels),
                cmap=cmap)


def _frame(ax, palette, show_aperture=True, show_ring=True, show_bar=True):
    clean_axes(ax, lim=5.2)
    if show_aperture:
        ax.add_patch(Circle((0, 0), APERTURE_RADIUS, facecolor="none",
                            edgecolor=palette["muted"], lw=0.7,
                            ls=(0, (4, 3))))
    if show_ring:
        ax.add_patch(Circle((0, 0), ECC, facecolor="none",
                            edgecolor=palette["muted"], lw=0.5,
                            ls=(0, (1, 3))))
    if show_bar:
        rect = Rectangle((-0.5, -APERTURE_RADIUS), 1.0, 2 * APERTURE_RADIUS,
                         facecolor="none", edgecolor=palette["stim"], lw=0.8,
                         ls=(0, (3, 3)), alpha=0.7)
        ax.add_patch(rect)
    ax.plot(0, 0, "+", color=palette["fg"], ms=10, mew=1.3)


def _draw_search_outlines(ax, palette, hp_key=None, marker_size=0.36):
    """4 search-array circles as outlines; HP gets a dashed outline in the
    HP colour to flag it as 'special'."""
    for k, (x, y) in LOCATIONS.items():
        if k == hp_key:
            ax.add_patch(Circle((x, y), marker_size + 0.06, facecolor="none",
                                edgecolor=palette["hp"], lw=1.6,
                                ls=(0, (2, 2))))
        else:
            ax.add_patch(Circle((x, y), marker_size, facecolor="none",
                                edgecolor=palette["fg"], lw=1.0))


def _net_shift_arrow(ax, prfs, apparent, palette, color=None,
                     scale=25.0, label="Net shift", anchor=(0.0, 0.0)):
    """Net-shift arrow rooted at `anchor`. `scale` magnifies the mean
    apparent-shift vector so it is visible at the figure scale."""
    net = (apparent - prfs).mean(axis=0)
    if np.linalg.norm(net) < 1e-3:
        return
    color = color or palette["af"]
    ax_x, ax_y = anchor
    end = (ax_x + net[0] * scale, ax_y + net[1] * scale)
    ax.annotate("", xy=end, xytext=(ax_x, ax_y),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=2.4,
                                mutation_scale=22))
    nx, ny = net / (np.linalg.norm(net) + 1e-9)
    perp = np.array([-ny, nx])
    label_x = (ax_x + end[0]) / 2 + 0.45 * perp[0]
    label_y = (ax_y + end[1]) / 2 + 0.45 * perp[1]
    ax.text(label_x, label_y, label, color=color, fontsize=18,
            ha="center", va="center", style="italic",
            bbox=dict(facecolor=palette["bg"], edgecolor="none", pad=1.2))


def _trial_event_strip(ax, palette, has_target=False, has_distractor=False):
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.plot([0.04, 0.96], [0.55, 0.55], color=palette["muted"], lw=0.6)
    ax.add_patch(Rectangle((0.04, 0.50), 0.92, 0.10,
                           facecolor=palette["hp"], edgecolor="none",
                           alpha=0.18))
    ax.text(0.0, 0.55, "HP", color=palette["hp"], fontsize=17,
            ha="left", va="center")
    if has_target:
        for x in [0.18, 0.42, 0.66, 0.86]:
            ax.plot([x, x], [0.30, 0.78], color=palette["target"], lw=1.6)
        ax.text(0.0, 0.16, r"$\tau(t)$", color=palette["target"], fontsize=18,
                ha="left", va="center")
    if has_distractor:
        for x in [0.22, 0.50, 0.76]:
            ax.plot([x, x], [0.30, 0.78], color=palette["af"], lw=1.6,
                    ls="--")
        ax.text(0.0, -0.06, r"$d(t)$", color=palette["af"], fontsize=18,
                ha="left", va="center")


# ---------------------------------------------------------------------------
# Slides

def buildup_01_af_basics():
    """Slide 01a: What does an attention field do to a single PRF?

    Three panels: PRF × M = Effective RF. The middle (M) panel makes
    the '+1' baseline gain visually obvious — a diverging colormap
    centered at 1 (white = baseline, red = enhancement), plus an
    inset 1D cross-section that decomposes M(x) = 1 + g·G(x).
    """
    p = BUILDUP_PALETTE
    with plt.rc_context(base_rc(p)):
        fig = plt.figure(figsize=(15.5, 6.6))
        gs = fig.add_gridspec(1, 3, wspace=0.30,
                              left=0.04, right=0.98, top=0.83, bottom=0.16)
        ax_PRF = fig.add_subplot(gs[0])
        ax_AF = fig.add_subplot(gs[1])
        ax_EFF = fig.add_subplot(gs[2])

        lim = 4.5
        res = 260
        xs = np.linspace(-lim, lim, res)
        xx, yy = np.meshgrid(xs, xs)

        # Geometry: PRF inside the aperture, AF centered OUTSIDE the
        # aperture at a canonical 4°-eccentricity HP/distractor location.
        # This matches the experiment — distractors and HP locations sit
        # at 4° eccentricity, outside the 3.17° bar PRF aperture, but
        # their AF Gaussians' tails extend into the aperture and reshape
        # any PRF whose own tail overlaps the AF.
        prf_x, prf_y, prf_sd = -1.4, +1.4, 0.85
        af_x,  af_y,  af_sd  = (ECC / np.sqrt(2),     # +2.83
                                 ECC / np.sqrt(2),     # +2.83 → at corner
                                 1.5)
        af_g = 1.6               # AF amplitude (bigger so the in-aperture
                                  # tail still produces a visible bump)

        PRF = gaussian_2d(xx, yy, prf_x, prf_y, prf_sd)
        AF = gaussian_2d(xx, yy, af_x, af_y, af_sd)
        M = 1.0 + af_g * AF
        EFF = PRF * np.maximum(M, 0.0)

        def _frame(ax, lim_=lim):
            clean_axes(ax, lim=lim_)
            ax.add_patch(Circle((0, 0), APERTURE_RADIUS, facecolor="none",
                                edgecolor=p["muted"], lw=1.2,
                                ls=(0, (4, 3)), zorder=4))
            ax.plot(0, 0, "+", color=p["fg"], ms=11, mew=1.4, zorder=5)

        # ---- Panel 1: PRF only ------------------------------------------
        prf_cmap = LinearSegmentedColormap.from_list(
            "prfcm", [(1, 1, 1, 0), to_rgba(p["prf"], 0.85)])
        ax_PRF.contourf(xx, yy, PRF,
                        levels=np.linspace(0.04 * PRF.max(), PRF.max(), 12),
                        cmap=prf_cmap, zorder=1)
        ax_PRF.add_patch(Circle((prf_x, prf_y), prf_sd, facecolor="none",
                                edgecolor=p["prf"], lw=2.5, zorder=6))
        ax_PRF.plot(prf_x, prf_y, "+", color=p["prf"], ms=13, mew=2.0,
                    zorder=7)
        _frame(ax_PRF)
        ax_PRF.set_title("PRF", pad=12, weight="bold")
        ax_PRF.text(0.5, -0.08,
                    r"$\mathrm{PRF}(\mathbf{x};\,\mu_{\mathrm{PRF}},"
                    r"\,\sigma_{\mathrm{PRF}})$",
                    transform=ax_PRF.transAxes, ha="center", va="top",
                    fontsize=21, color=p["prf"])

        # ---- Panel 2: M(x) = 1 + g·G ----------------------------------
        # Diverging cmap centered at 1: white = baseline 1, red = enhanced.
        # This makes the '+1' baseline visually obvious.
        M_field = 1.0 + af_g * AF
        m_amp = float(M_field.max() - 1.0)
        ax_AF.imshow(M_field, extent=(-lim, lim, -lim, lim),
                     origin="lower", cmap="RdBu_r",
                     vmin=1 - m_amp, vmax=1 + m_amp,
                     interpolation="bilinear", zorder=0)
        af_fwhm = 2 * np.sqrt(2 * np.log(2)) * af_sd
        ax_AF.add_patch(Circle((af_x, af_y), af_fwhm / 2,
                               facecolor="none",
                               edgecolor=p["hp"], lw=2.5, zorder=6,
                               ls=(0, (5, 3))))
        ax_AF.plot(af_x, af_y, "x", color=p["hp"], ms=15, mew=2.5,
                   zorder=7)
        # Callouts: "+1 baseline" and "+ g·G".
        ax_AF.annotate("M = 1\n(baseline gain)",
                       xy=(-1.7, -1.7), xytext=(-4.2, -3.6),
                       color=p["fg"], fontsize=18, ha="left", va="top",
                       fontweight="bold",
                       arrowprops=dict(arrowstyle="->", color=p["fg"],
                                       lw=1.4),
                       bbox=dict(boxstyle="round,pad=0.3",
                                 facecolor="white", edgecolor=p["fg"],
                                 lw=0.8, alpha=0.93), zorder=10)
        ax_AF.annotate("+ g·G\n(local AF bump)",
                       xy=(af_x - 0.4, af_y - 0.4),
                       xytext=(af_x - 2.6, af_y - 2.4),
                       color=p["hp"], fontsize=18, ha="left", va="top",
                       fontweight="bold",
                       arrowprops=dict(arrowstyle="->", color=p["hp"],
                                       lw=1.4),
                       bbox=dict(boxstyle="round,pad=0.3",
                                 facecolor="white", edgecolor=p["hp"],
                                 lw=0.8, alpha=0.93), zorder=10)
        _frame(ax_AF)
        ax_AF.set_title("Attention field  M = 1 + g·G",
                        pad=12, weight="bold")
        ax_AF.text(0.5, -0.08,
                   r"$M(\mathbf{x}) = 1 + g \cdot G(\mathbf{x};\,"
                   r"\mu_{\mathrm{AF}},\,\sigma_{\mathrm{AF}})$",
                   transform=ax_AF.transAxes, ha="center", va="top",
                   fontsize=21, color=p["hp"])

        # ---- Panel 3: Effective RF -------------------------------------
        af_cmap = LinearSegmentedColormap.from_list(
            "afcm", [(1, 1, 1, 0), to_rgba(p["hp"], 0.78)])
        ax_EFF.contourf(xx, yy, AF,
                        levels=np.linspace(0.04 * AF.max(),
                                           AF.max(), 12),
                        cmap=af_cmap, zorder=1)
        # Original PRF outline (dotted)
        ax_EFF.add_patch(Circle((prf_x, prf_y), prf_sd, facecolor="none",
                                edgecolor=p["prf"], lw=1.6, ls=":",
                                alpha=0.85, zorder=6))
        ax_EFF.plot(prf_x, prf_y, "+", color=p["prf"], ms=11, mew=1.5,
                    alpha=0.9, zorder=7)
        # Effective-RF half-max contour (solid)
        ax_EFF.contour(xx, yy, EFF, levels=[EFF.max() * 0.5],
                       colors=[p["prf"]], linewidths=3.0, alpha=0.97,
                       zorder=8)
        # CoM arrow
        tot = EFF.sum()
        com_x = (xx * EFF).sum() / tot
        com_y = (yy * EFF).sum() / tot
        ax_EFF.annotate(
            "", xy=(com_x, com_y), xytext=(prf_x, prf_y),
            arrowprops=dict(arrowstyle="-|>", color=p["prf"], lw=3.5,
                            mutation_scale=24, alpha=0.98,
                            shrinkA=0, shrinkB=0), zorder=9)
        # Soft pull-direction quiver (∇M)
        gy, gx = np.gradient(M, lim * 2 / (res - 1), lim * 2 / (res - 1))
        step = 16
        sl = (slice(step // 2, None, step), slice(step // 2, None, step))
        sx, sy = xx[sl], yy[sl]
        u, v = gx[sl], gy[sl]
        mask = sx ** 2 + sy ** 2 <= APERTURE_RADIUS ** 2
        ax_EFF.quiver(sx[mask], sy[mask], u[mask], v[mask],
                      color="#5A6470", alpha=0.75, scale=5.0,
                      scale_units="inches", width=0.008,
                      headwidth=4.5, headlength=5.5, zorder=2)
        _frame(ax_EFF)
        ax_EFF.set_title("Effective RF", pad=12, weight="bold")
        ax_EFF.text(0.5, -0.08,
                    "PRF is reshaped; CoM shifts toward the AF",
                    transform=ax_EFF.transAxes, ha="center", va="top",
                    fontsize=19, color=p["fg"], style="italic")

        # Inter-panel × and = symbols.
        fig.text(0.355, 0.50, r"$\times$", fontsize=42,
                 ha="center", va="center", color=p["fg"], weight="bold")
        fig.text(0.665, 0.50, r"$=$", fontsize=42,
                 ha="center", va="center", color=p["fg"], weight="bold")

        fig.suptitle(
            "What does an attention field do?",
            x=0.04, y=0.95, ha="left", fontsize=26, weight="bold",
            color=p["fg"])

        fig.savefig(OUT / "talk_buildup_01_af_basics.pdf",
                    bbox_inches="tight")
        plt.close(fig)


def buildup_01_design():
    p = BUILDUP_PALETTE
    with plt.rc_context(base_rc(p)):
        fig, ax = plt.subplots(figsize=(8.5, 6.5))
        _frame(ax, p, show_aperture=True, show_ring=True, show_bar=True)
        for k, (x, y) in LOCATIONS.items():
            ax.add_patch(Circle((x, y), 0.42, facecolor="none",
                                edgecolor=p["fg"], lw=1.1))

        ax.annotate("Aperture", xy=(APERTURE_RADIUS * np.cos(np.deg2rad(125)),
                                    APERTURE_RADIUS * np.sin(np.deg2rad(125))),
                    xytext=(-5.0, 4.4), color=p["muted"], fontsize=18,
                    arrowprops=dict(arrowstyle="-", color=p["muted"], lw=0.6))
        ax.annotate("Bar PRF\nstimulus", xy=(0.5, -2.4), xytext=(-5.0, -3.6),
                    color=p["stim"], fontsize=18, ha="left",
                    arrowprops=dict(arrowstyle="-", color=p["stim"], lw=0.6))
        ax.annotate("Search\npositions (4)",
                    xy=(LOCATIONS["upper_right"][0] + 0.45,
                        LOCATIONS["upper_right"][1] + 0.45),
                    xytext=(3.4, 4.7), color=p["fg"], fontsize=18,
                    arrowprops=dict(arrowstyle="-", color=p["fg"], lw=0.6))
        ax.text(0.18, -0.45, "Fixation", color=p["fg"], fontsize=18, alpha=0.7)

        ax.set_title("Design", fontsize=21, pad=10, loc="left")
        fig.savefig(OUT / "talk_buildup_01_design.pdf", bbox_inches="tight")
        plt.close(fig)


def buildup_01b_two_models():
    """Slide 01b: AF vs AF+ — two attention-field model variants.

    Schematic comparison of Sumiya / Abdirashid (2026, Ch. 2):
        AF  model:  R = A · S            → predicts GLOBAL pull
        AF+ model:  R = (a·A + b) · S    → predicts LOCAL pull only

    Visualised as a 2D field with one AF blob in the upper-right and
    six example PRFs scattered across the visual field. For each PRF
    we draw an arrow from the original PRF center to the
    center-of-mass of R = (response under that model). Under AF,
    every PRF gets pulled toward the AF (pull magnitude depends on
    σ_AF/σ_PRF only, not on distance). Under AF+, only PRFs whose
    tails overlap the AF are pulled — far PRFs sit still.

    Empirical data favor AF+ (PRF attraction is observed only when
    the PRF overlaps the attended locus; Abdirashid et al. 2025 +
    thesis Ch. 2). This is also the model implicit in our
    `M(x) = 1 + g·G(x)` parameterisation — the '+1' IS the offset.
    """
    p = BUILDUP_PALETTE
    with plt.rc_context(base_rc(p)):
        fig = plt.figure(figsize=(15.5, 8.5))
        gs = fig.add_gridspec(1, 2, wspace=0.18,
                              left=0.04, right=0.985, top=0.82,
                              bottom=0.18)
        ax_AF = fig.add_subplot(gs[0])
        ax_AFp = fig.add_subplot(gs[1])

        lim = 4.5
        res = 280
        xs = np.linspace(-lim, lim, res)
        xx, yy = np.meshgrid(xs, xs)

        # AF setup.
        af_x, af_y, af_sd = +1.8, +1.8, 1.4
        AF = gaussian_2d(xx, yy, af_x, af_y, af_sd)
        # Normalize AF so its peak is 1 (so a, b parameters in AF+ are
        # interpretable as straight gain values).
        AF = AF / AF.max()

        # PRF set covering several distances/directions from AF.
        prf_examples = [
            ((+1.4, +1.4), 0.7),    # near AF
            ((+0.0, +1.0), 0.7),    # mid distance
            ((-1.5, +1.5), 0.7),    # other corner same-y
            ((-2.0, -1.5), 0.7),    # opposite quadrant
            ((+1.5, -1.7), 0.7),    # below
            ((-1.0, -2.2), 0.7),    # far below
        ]

        # AF+ parameters in R = (a·A + b)·S — chosen so far PRFs are
        # nearly unaffected and near PRFs feel a strong pull.
        a_AF, b_AF = 1.6, 1.0

        af_cmap = LinearSegmentedColormap.from_list(
            "afsumi", [(1, 1, 1, 0), to_rgba(p["hp"], 0.85)])

        def _frame(ax):
            clean_axes(ax, lim=lim)
            ax.add_patch(Circle((0, 0), APERTURE_RADIUS, facecolor="none",
                                edgecolor=p["muted"], lw=1.2,
                                ls=(0, (4, 3)), zorder=4))
            ax.plot(0, 0, "+", color=p["fg"], ms=11, mew=1.4, zorder=5)
            # AF blob underlay
            ax.contourf(xx, yy, AF,
                        levels=np.linspace(0.04, 1.0, 12),
                        cmap=af_cmap, zorder=1)
            af_fwhm = 2 * np.sqrt(2 * np.log(2)) * af_sd
            ax.add_patch(Circle((af_x, af_y), af_fwhm / 2,
                                facecolor="none",
                                edgecolor=p["hp"], lw=2.0,
                                ls=(0, (5, 3)), zorder=6))
            ax.plot(af_x, af_y, "x", color=p["hp"], ms=14, mew=2.2,
                    zorder=7)
            ax.text(af_x, af_y - 0.85, "AF", color=p["hp"],
                    fontsize=19, ha="center", va="top", weight="bold",
                    zorder=8,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                              edgecolor=p["hp"], lw=0.8, alpha=0.9))

        def _draw_predictions(ax, model):
            for (px, py), prf_sd in prf_examples:
                S = gaussian_2d(xx, yy, px, py, prf_sd)
                if model == "AF":
                    R = AF * S
                else:  # AF+
                    R = (a_AF * AF + b_AF) * S
                tot = R.sum()
                if tot > 1e-9:
                    com_x = (xx * R).sum() / tot
                    com_y = (yy * R).sum() / tot
                else:
                    com_x, com_y = px, py
                # Original PRF outline (dotted teal)
                ax.add_patch(Circle((px, py), prf_sd, facecolor="none",
                                    edgecolor=p["prf"], lw=1.4, ls=":",
                                    alpha=0.85, zorder=8))
                ax.plot(px, py, "+", color=p["prf"], ms=8, mew=1.2,
                        alpha=0.85, zorder=9)
                if (com_x - px) ** 2 + (com_y - py) ** 2 > 5e-3:
                    ax.annotate(
                        "", xy=(com_x, com_y), xytext=(px, py),
                        arrowprops=dict(arrowstyle="-|>",
                                        color=p["prf"], lw=2.6,
                                        mutation_scale=18,
                                        alpha=0.97,
                                        shrinkA=0, shrinkB=0),
                        zorder=10)

        _frame(ax_AF); _draw_predictions(ax_AF, "AF")
        _frame(ax_AFp); _draw_predictions(ax_AFp, "AF+")

        # Compact, equation-forward titles. The full equation already
        # signals what the model is — captions stay to one short line.
        ax_AF.set_title(r"AF :  $R = A \cdot S$",
                        pad=14, weight="bold", fontsize=28)
        ax_AFp.set_title(
            r"AF+ :  $R = (a A + b) \cdot S$",
            pad=14, weight="bold", fontsize=28)

        ax_AF.text(0.5, -0.04, "Global pull — every PRF shifts",
                   transform=ax_AF.transAxes, ha="center", va="top",
                   fontsize=21, color=p["fg"], weight="bold")
        ax_AF.text(0.5, -0.10,
                   "Reynolds & Heeger 2009  •  Klein, Harvey & "
                   "Dumoulin 2014",
                   transform=ax_AF.transAxes, ha="center", va="top",
                   fontsize=14, color=p["muted"], style="italic")
        ax_AFp.text(0.5, -0.04,
                    "Local pull — only PRFs near the AF shift",
                    transform=ax_AFp.transAxes, ha="center", va="top",
                    fontsize=21, color=p["fg"], weight="bold")
        ax_AFp.text(0.5, -0.10,
                    "Abdirashid (2026), Ch. 2",
                    transform=ax_AFp.transAxes, ha="center", va="top",
                    fontsize=14, color=p["muted"], style="italic")

        # Inset: predicted shift magnitude vs distance from AF center,
        # for a 1D analogue of each model. Makes the 'global' vs
        # 'local' qualifier quantitative without extra paragraph text.
        from mpl_toolkits.axes_grid1 import inset_locator

        d_grid = np.linspace(0.0, 5.0, 240)
        # 1D analogue: PRF at distance d from AF center, σ_PRF = 0.7,
        # σ_AF = 1.4. Compute response on a fine 1D grid and find the
        # shift = CoM - μ_PRF.
        def _shift_curve(model):
            shifts = np.empty_like(d_grid)
            xline = np.linspace(-8, 8, 600)
            S_curve_AF_x = 0.0
            for i, d in enumerate(d_grid):
                A = np.exp(-((xline - 0.0) ** 2) / (2 * 1.4 ** 2))
                A = A / A.max()
                S = np.exp(-((xline - d) ** 2) / (2 * 0.7 ** 2))
                if model == "AF":
                    R = A * S
                else:
                    R = (a_AF * A + b_AF) * S
                tot = R.sum()
                if tot > 1e-9:
                    com = (xline * R).sum() / tot
                    shifts[i] = abs(d - com)
                else:
                    shifts[i] = 0.0
            return shifts

        sh_AF = _shift_curve("AF")
        sh_AFp = _shift_curve("AF+")

        for ax_panel, sh, color in [
            (ax_AF, sh_AF, p["hp"]),
            (ax_AFp, sh_AFp, p["hp"]),
        ]:
            inset = inset_locator.inset_axes(
                ax_panel, width="42%", height="22%",
                loc="lower left",
                bbox_to_anchor=(0.04, 0.04, 1, 1),
                bbox_transform=ax_panel.transAxes, borderpad=0)
            inset.fill_between(d_grid, 0, sh, color=color, alpha=0.30)
            inset.plot(d_grid, sh, color=color, lw=2.4)
            inset.set_xlim(0, 5); inset.set_ylim(0, max(sh_AF.max(),
                                                         sh_AFp.max()) * 1.12)
            inset.set_xlabel("Distance from AF (°)", fontsize=17)
            inset.set_ylabel("|shift|", fontsize=17)
            inset.tick_params(axis="both", labelsize=13)
            for spn in ("top", "right"):
                inset.spines[spn].set_visible(False)
            inset.set_facecolor("white")
            inset.patch.set_alpha(0.85)

        # Top banner — concise. Per-panel references are in each panel
        # below the equation, so no global subtitle needed.
        fig.suptitle(
            "Two attention-field models   —   AF vs AF+",
            x=0.04, y=0.95, ha="left", fontsize=30, weight="bold",
            color=p["fg"])

        # Bottom take-home banner — short, highest font size.
        fig.text(
            0.5, 0.07,
            r"$\Rightarrow$  AF+ matches data  •  "
            r"retsupp uses  $M = 1 + g\!\cdot\!G$  (= AF+) "
            r"fit jointly, not post-hoc",
            ha="center", va="center", fontsize=21, color=p["fg"],
            weight="bold",
            bbox=dict(boxstyle="round,pad=0.55", facecolor="#FFF1A6",
                      edgecolor="#C9A227", lw=1.4))

        fig.savefig(OUT / "talk_buildup_01b_two_models.pdf",
                    bbox_inches="tight")
        plt.close(fig)


def buildup_02_sustained_equal():
    p = BUILDUP_PALETTE
    with plt.rc_context(base_rc(p)):
        fig, ax = plt.subplots(figsize=(8.5, 6.5))
        _frame(ax, p, show_aperture=True, show_ring=True, show_bar=False)

        comps = _gain_components_sustained_equal()
        _draw_layer(ax, comps, p["hp"], alpha=0.55)
        _draw_search_outlines(ax, p, hp_key=None)

        _draw_three_example_prfs(ax, comps, p)

        ax.set_title("Sustained gain at all 4 search positions (equal)",
                     fontsize=20, pad=10, loc="left")
        ax.text(0.5, -0.06,
                "Each PRF is reshaped by the nearest gain blob; CoMs "
                "pull radially outward. Symmetric → net shift = 0",
                transform=ax.transAxes, ha="center", va="top", fontsize=18.5,
                color=p["muted"], style="italic")
        fig.savefig(OUT / "talk_buildup_02_sustained_equal.pdf",
                    bbox_inches="tight")
        plt.close(fig)


def buildup_03_hp_weakened():
    p = BUILDUP_PALETTE
    with plt.rc_context(base_rc(p)):
        fig, ax = plt.subplots(figsize=(8.5, 6.5))
        _frame(ax, p, show_aperture=True, show_ring=True, show_bar=False)

        comps = _gain_components_hp_weakened()
        _draw_layer(ax, comps, p["hp"], alpha=0.55)
        _draw_search_outlines(ax, p, hp_key=HP_KEY)

        _draw_three_example_prfs(ax, comps, p)
        origins, com = _example_prf_coms(comps)
        _net_shift_arrow(ax, origins, com, p, color=p["af"], scale=30.0,
                         label="Net shift")

        hp_x, hp_y = LOCATIONS[HP_KEY]
        ax.annotate("HP — suppressed\n(weaker gain)",
                    xy=(hp_x, hp_y), xytext=(hp_x + 1.0, hp_y + 1.4),
                    color=p["hp"], fontsize=18, ha="left",
                    arrowprops=dict(arrowstyle="-", color=p["hp"], lw=0.8))

        ax.set_title("HP gain reduced  →  net shift AWAY from HP",
                     fontsize=20, pad=10, loc="left")
        fig.savefig(OUT / "talk_buildup_03_hp_weakened.pdf",
                    bbox_inches="tight")
        plt.close(fig)


def _buildup_with_transients(filename, title, has_target, has_distractor):
    p = BUILDUP_PALETTE
    with plt.rc_context(base_rc(p)):
        fig = plt.figure(figsize=(9.5, 7.0))
        gs = fig.add_gridspec(2, 1, height_ratios=[6.0, 0.8],
                              hspace=0.05, top=0.95, bottom=0.06,
                              left=0.06, right=0.96)
        ax = fig.add_subplot(gs[0])
        _frame(ax, p, show_aperture=True, show_ring=True, show_bar=False)

        sustained_comps = _gain_components_hp_weakened()
        _draw_layer(ax, sustained_comps, p["hp"], alpha=0.50)

        target_comps = []
        if has_target:
            tx, ty = LOCATIONS[TARGET_KEY]
            target_comps.append((tx, ty, SD_TRANSIENT, AMP_TARGET))
            _draw_layer(ax, target_comps, p["target"], alpha=0.55)

        dyn_comps = []
        if has_distractor:
            dx, dy = LOCATIONS[DYN_DIST_KEY]
            dyn_comps.append((dx, dy, SD_TRANSIENT, AMP_DYN))
            _draw_layer(ax, dyn_comps, p["af"], alpha=0.55)

        _draw_search_outlines(ax, p, hp_key=HP_KEY)

        if has_target:
            tx, ty = LOCATIONS[TARGET_KEY]
            ax.add_patch(Circle((tx, ty), 0.46, facecolor="none",
                                edgecolor=p["target"], lw=1.8,
                                ls=(0, (2, 2))))
            ax.annotate("Target\n(this trial)",
                        xy=(tx, ty), xytext=(tx - 1.2, ty + 1.4),
                        color=p["target"], fontsize=18, ha="right",
                        arrowprops=dict(arrowstyle="-", color=p["target"],
                                        lw=0.8))
        if has_distractor:
            dx, dy = LOCATIONS[DYN_DIST_KEY]
            ax.add_patch(Circle((dx, dy), 0.46, facecolor="none",
                                edgecolor=p["af"], lw=1.8,
                                ls=(0, (2, 2))))
            ax.annotate("Distractor\n(this trial)",
                        xy=(dx, dy), xytext=(dx - 1.2, dy - 1.4),
                        color=p["af"], fontsize=18, ha="right",
                        arrowprops=dict(arrowstyle="-", color=p["af"], lw=0.8))

        all_comps = sustained_comps + target_comps + dyn_comps
        _draw_three_example_prfs(ax, all_comps, p)

        ax.set_title(title, fontsize=20, pad=10, loc="left")

        tax = fig.add_subplot(gs[1])
        _trial_event_strip(tax, p, has_target=has_target,
                           has_distractor=has_distractor)
        tax.text(0.5, 1.05, "Trial timeline", transform=tax.transAxes,
                 ha="center", va="bottom", fontsize=18, color=p["muted"])

        fig.savefig(OUT / filename, bbox_inches="tight")
        plt.close(fig)


def buildup_04_target():
    _buildup_with_transients(
        "talk_buildup_04_target.pdf",
        "+ Target capture (transient, attractive)",
        has_target=True, has_distractor=False)


def buildup_05_dynamic_distractor():
    _buildup_with_transients(
        "talk_buildup_05_dynamic_distractor.pdf",
        "+ Dynamic distractor on this trial (transient, suppressive)",
        has_target=True, has_distractor=True)


def buildup_06_equation():
    p = BUILDUP_PALETTE
    with plt.rc_context(base_rc(p)):
        fig = plt.figure(figsize=(16.0, 9.5))
        # Slightly more room for the bottom heatmaps (more visual weight
        # for the icons; less for the equation row + glossary).
        gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1.4],
                              hspace=0.32, wspace=0.18,
                              left=0.035, right=0.985, top=0.94,
                              bottom=0.05)

        eq_ax = fig.add_subplot(gs[0, :])
        eq_ax.axis("off")
        eq_ax.set_xlim(0, 1)
        eq_ax.set_ylim(0, 1)

        y_eq = 0.78

        def piece(x, s, color, fs=18, ha="left", va="center"):
            eq_ax.text(x, y_eq, s, color=color, fontsize=fs, ha=ha, va=va)

        piece(0.04, r"$M(\mathbf{x}, t) \;=\;$", p["fg"], fs=20)
        piece(0.18, r"$1$", p["muted"], fs=20)
        piece(0.22, r"$+$", p["fg"], fs=20)
        piece(0.27,
              r"$g_{\mathrm{HP}} \, G_{\mathrm{HP}}(\mathbf{x};\,"
              r"\mu_{\mathrm{HP}},\,\sigma_{\mathrm{HP}})$",
              p["hp"], fs=18)
        piece(0.50, r"$+$", p["fg"], fs=20)
        piece(0.54,
              r"$g_{\mathrm{D}} \, G_{\mathrm{D}}(\mathbf{x};\,"
              r"\mu_{\mathrm{D}},\,\sigma_{\mathrm{D}})\, d(t)$",
              p["af"], fs=18)
        piece(0.78, r"$+$", p["fg"], fs=20)
        piece(0.82,
              r"$g_{\mathrm{T}} \, G_{\mathrm{T}}(\mathbf{x};\,"
              r"\mu_{\mathrm{T}},\,\sigma_{\mathrm{T}})\, \tau(t)$",
              p["target"], fs=18)

        eq_ax.annotate("Baseline gain\n(no modulation)",
                       xy=(0.185, y_eq + 0.08), xytext=(0.18, 0.97),
                       color=p["muted"], fontsize=18, ha="center", va="top",
                       arrowprops=dict(arrowstyle="-", color=p["muted"],
                                       lw=0.6))
        eq_ax.annotate("Sustained\nHP gain field",
                       xy=(0.39, y_eq + 0.08), xytext=(0.39, 0.97),
                       color=p["hp"], fontsize=18, ha="center", va="top",
                       arrowprops=dict(arrowstyle="-", color=p["hp"], lw=0.6))
        eq_ax.annotate("Transient gain\nat distractor onset",
                       xy=(0.65, y_eq + 0.08), xytext=(0.65, 0.97),
                       color=p["af"], fontsize=18, ha="center", va="top",
                       arrowprops=dict(arrowstyle="-", color=p["af"], lw=0.6))
        eq_ax.annotate("Transient gain\nat target onset",
                       xy=(0.90, y_eq + 0.08), xytext=(0.90, 0.97),
                       color=p["target"], fontsize=18, ha="center", va="top",
                       arrowprops=dict(arrowstyle="-", color=p["target"],
                                       lw=0.6))

        # (Generic g_X · G_X · h(t) zoom row removed — its captions
        # crowded the free-parameter badges. The top equation already
        # spells out each component; the badges below say which ones
        # are free.)

        # Free-parameter rows — make the GRANULARITY explicit:
        #   Per VOXEL  → PRF params  (green badges)
        #   Per ROI × subject → gain-field params (yellow badges)
        # This is the load-bearing methodological message of the talk.
        VOXEL_BG = "#D7F2D6"; VOXEL_EDGE = "#3D8B40"
        ROI_BG = "#FFF1A6";   ROI_EDGE = "#C9A227"

        row_voxel_y = 0.42
        row_roi_y = 0.16

        # Row 1: per-VOXEL PRF params (green badges)
        eq_ax.text(0.04, row_voxel_y, "Per VOXEL",
                   color=VOXEL_EDGE, fontsize=22, weight="bold",
                   ha="left", va="center")
        eq_ax.text(0.04, row_voxel_y - 0.07, "PRF params",
                   color=VOXEL_EDGE, fontsize=17, style="italic",
                   ha="left", va="center")
        for x, sym in [
            (0.27, r"$\mu_{\mathrm{PRF}}$"),
            (0.34, r"$\sigma_{\mathrm{PRF}}$"),
            (0.45, "amp"),
            (0.55, "baseline"),
        ]:
            eq_ax.text(x, row_voxel_y, sym, color=VOXEL_EDGE, fontsize=21,
                       ha="center", va="center",
                       bbox=dict(boxstyle="round,pad=0.30",
                                 facecolor=VOXEL_BG, edgecolor=VOXEL_EDGE,
                                 lw=1.2))

        # Row 2: per-ROI gain-field params (yellow badges)
        eq_ax.text(0.04, row_roi_y, "Per ROI × subject",
                   color=ROI_EDGE, fontsize=22, weight="bold",
                   ha="left", va="center")
        eq_ax.text(0.04, row_roi_y - 0.07, "Gain-field params",
                   color=ROI_EDGE, fontsize=17, style="italic",
                   ha="left", va="center")
        for x, sym, col in [
            (0.27, r"$g_{\mathrm{HP}}$", p["hp"]),
            (0.33, r"$g_{\mathrm{D}}$", p["af"]),
            (0.39, r"$g_{\mathrm{T}}$", p["target"]),
            (0.47, r"$\sigma_{\mathrm{HP}}$", p["hp"]),
            (0.55, r"$\sigma_{\mathrm{dyn}}$", p["fg"]),
        ]:
            eq_ax.text(x, row_roi_y, sym, color=col, fontsize=21,
                       ha="center", va="center",
                       bbox=dict(boxstyle="round,pad=0.30",
                                 facecolor=ROI_BG, edgecolor=ROI_EDGE,
                                 lw=1.2))

        # Right column: fixed-by-paradigm.
        eq_ax.text(0.68, row_voxel_y, "Fixed by paradigm",
                   color=p["muted"], fontsize=22, weight="bold",
                   ha="left", va="center")
        eq_ax.text(0.68, row_voxel_y - 0.07,
                   r"$\mu_{\mathrm{HP/D/T}}$ (trial events)",
                   color=p["muted"], fontsize=17,
                   ha="left", va="center")
        eq_ax.text(0.68, row_voxel_y - 0.13,
                   r"$d(t),\,\tau(t),\,S(\mathbf{x},t)$",
                   color=p["muted"], fontsize=17,
                   ha="left", va="center")

        comp_specs = [
            dict(key="HP", title="Sustained HP",
                 color=p["hp"], center=LOCATIONS[HP_KEY],
                 sigma=SD_SUSTAINED, amp=-0.55,
                 caption=r"$g_{\mathrm{HP}}<0$ (suppression) · "
                         r"sustained $h(t)=1$"),
            dict(key="D", title="Dynamic distractor",
                 color=p["af"], center=LOCATIONS[DYN_DIST_KEY],
                 sigma=SD_TRANSIENT, amp=AMP_DYN,
                 caption=r"$g_{\mathrm{D}}<0$ · transient $d(t)$ · "
                         r"$\mu_{\mathrm{D}}$ varies per trial"),
            dict(key="T", title="Target capture",
                 color=p["target"], center=LOCATIONS[TARGET_KEY],
                 sigma=SD_TRANSIENT, amp=AMP_TARGET,
                 caption=r"$g_{\mathrm{T}}>0$ · transient $\tau(t)$ · "
                         r"$\mu_{\mathrm{T}}$ varies per trial"),
        ]

        for col, spec in enumerate(comp_specs):
            ax = fig.add_subplot(gs[1, col])
            clean_axes(ax, lim=4.4)
            # Bar PRF aperture outline — thick so it reads as a real
            # boundary, not a faint guide.
            ax.add_patch(Circle((0, 0), APERTURE_RADIUS, facecolor="none",
                                edgecolor=p["muted"], lw=1.6,
                                ls=(0, (4, 3))))
            ax.add_patch(Circle((0, 0), ECC, facecolor="none",
                                edgecolor=p["muted"], lw=0.4,
                                ls=(0, (1, 3))))
            for k, (x, y) in LOCATIONS.items():
                ax.add_patch(Circle((x, y), 0.32, facecolor="none",
                                    edgecolor=p["fg"], lw=1.0))
            ax.plot(0, 0, "+", color=p["fg"], ms=8, mew=1.2)

            cx, cy = spec["center"]
            sigma = spec["sigma"]
            amp = spec["amp"]
            res = 160
            xs = np.linspace(-4.4, 4.4, res)
            yy, xx = np.meshgrid(xs, xs)
            field = amp * gaussian_2d(xx, yy, cx, cy, sigma)
            cmap = LinearSegmentedColormap.from_list(
                "c", [(1, 1, 1, 0), to_rgba(spec["color"], 0.78)])
            if amp >= 0:
                ax.contourf(xx, yy, field,
                            levels=np.linspace(0.02 * field.max(),
                                               field.max(), 10),
                            cmap=cmap)
            else:
                ax.contourf(xx, yy, -field,
                            levels=np.linspace(0.02 * (-field).max(),
                                               (-field).max(), 10),
                            cmap=cmap, alpha=0.6)
                ax.contour(xx, yy, -field,
                           levels=[0.5 * (-field).max()],
                           colors=[spec["color"]], linewidths=0.8,
                           linestyles=":")

            fwhm = 2 * np.sqrt(2 * np.log(2)) * sigma
            ax.add_patch(Circle((cx, cy), fwhm / 2, facecolor="none",
                                edgecolor=spec["color"], lw=1.2,
                                ls="--"))
            ax.plot(cx, cy, "x", color=spec["color"], ms=8, mew=1.4)
            ax.annotate(r"$\mu_{\mathrm{%s}}$" % spec["key"],
                        xy=(cx, cy), xytext=(cx + 0.25, cy + 0.25),
                        color=spec["color"], fontsize=19, ha="left")
            ax.annotate("", xy=(cx + fwhm / 2, cy), xytext=(cx, cy),
                        arrowprops=dict(arrowstyle="<->", color=spec["color"],
                                        lw=1.0))
            ax.text(cx + fwhm / 4, cy - 0.4,
                    r"FWHM = 2.355·$\sigma_{\mathrm{%s}}$" % spec["key"],
                    color=spec["color"], fontsize=17, ha="center", va="top")

            ax.set_title(spec["title"], color=spec["color"],
                         fontsize=22, loc="left", pad=6,
                         weight="bold")
            ax.text(0.5, -0.04, spec["caption"], transform=ax.transAxes,
                    ha="center", va="top", fontsize=16, color=p["fg"])

        fig.suptitle("Compositional gain model — parameter glossary",
                     fontsize=24, x=0.04, ha="left", y=0.99, weight="bold")
        fig.savefig(OUT / "talk_buildup_06_equation.pdf",
                    bbox_inches="tight")
        plt.close(fig)


def _build_paradigm_S(xx, yy, bar_y_center, bar_thickness=1.6, item_r=0.42):
    """Stimulus drive S(x, t): horizontal bar inside the aperture +
    4 search-array items at the corner positions. Returns a [0, 1] array.

    Uses STANDARD meshgrid convention: xx varies along columns (X-axis),
    yy varies along rows (Y-axis). Caller must pass meshgrid as
    `xx, yy = np.meshgrid(xs, xs)`.
    """
    bar_soft = np.exp(-((yy - bar_y_center) / (bar_thickness * 0.30)) ** 8)
    in_aperture = (xx ** 2 + yy ** 2) <= APERTURE_RADIUS ** 2
    bar = bar_soft * in_aperture

    item_field = np.zeros_like(xx)
    for k, (sx, sy) in LOCATIONS.items():
        item_field = np.maximum(
            item_field,
            np.exp(-(((xx - sx) ** 2 + (yy - sy) ** 2)
                     / (2 * (item_r * 0.55) ** 2)) ** 1.5) * 0.85)
    return np.clip(np.maximum(bar, item_field), 0.0, 1.0)


def _modulation_field_std(gain_components, xx, yy):
    """Build M(x) = 1 + Σ amp · G on a STANDARD-meshgrid (xx, yy) pair."""
    M = np.ones_like(xx)
    for cx, cy, sd, amp in gain_components:
        M = M + amp * gaussian_2d(xx, yy, cx, cy, sd)
    return M


def buildup_07_effective_drive():
    """3-panel slide: paradigm drive S × attention field M = effective drive.

    Single representative timepoint chosen so each panel is visibly different:
    bar across the UL-UR row with target onset at UL and distractor onset at LR.
    HP is at UR (suppressed). Right panel highlights three regimes:
      - bar+target (UL)        → enhanced drive
      - bar+HP suppression (UR)→ reduced drive
      - distractor without bar (LR) → 'wasted' attention (multiplicative gating)

    The right panel also flags which symbols are FREE PARAMETERS estimated
    from data (g_HP, g_D, g_T, σ_HP, σ_D, σ_T) versus what's fixed by the
    paradigm (μ_HP / μ_D / μ_T = trial events; bar position = paradigm).
    """
    p = BUILDUP_PALETTE
    with plt.rc_context(base_rc(p)):
        fig = plt.figure(figsize=(16.0, 7.0))
        gs = fig.add_gridspec(1, 3, wspace=0.42,
                              left=0.04, right=0.985, top=0.85,
                              bottom=0.20)
        ax_S = fig.add_subplot(gs[0])
        ax_M = fig.add_subplot(gs[1])
        ax_D = fig.add_subplot(gs[2])

        lim = 4.8
        res = 280
        xs = np.linspace(-lim, lim, res)
        # Standard meshgrid convention here (unlike helpers above which
        # use a swapped convention that's compensated by contourf).
        xx, yy = np.meshgrid(xs, xs)

        bar_y_center = +ECC / np.sqrt(2)         # ~+2.83°  → row of UL & UR
        S = _build_paradigm_S(xx, yy, bar_y_center)

        sustained_comps = _gain_components_hp_weakened()
        target_comps = [(*LOCATIONS[TARGET_KEY], SD_TRANSIENT, AMP_TARGET)]
        dyn_comps = [(*LOCATIONS[DYN_DIST_KEY], SD_TRANSIENT, AMP_DYN)]
        all_comps = sustained_comps + target_comps + dyn_comps
        M = _modulation_field_std(all_comps, xx, yy)

        D = S * M

        def _frame_panel(ax, hp_marker=False, target_flash=False,
                         dist_flash=False):
            clean_axes(ax, lim=lim)
            ax.add_patch(Circle((0, 0), APERTURE_RADIUS, facecolor="none",
                                edgecolor=p["muted"], lw=0.9,
                                ls=(0, (4, 3))))
            ax.plot(0, 0, "+", color=p["fg"], ms=9, mew=1.2)
            for k, (x, y) in LOCATIONS.items():
                edge = p["fg"]
                lw = 1.0
                ls = "-"
                if k == HP_KEY and hp_marker:
                    edge = p["hp"]; lw = 1.8; ls = (0, (2, 2))
                ax.add_patch(Circle((x, y), 0.42, facecolor="none",
                                    edgecolor=edge, lw=lw, ls=ls))
            if target_flash:
                tx, ty = LOCATIONS[TARGET_KEY]
                ax.add_patch(Circle((tx, ty), 0.52, facecolor="none",
                                    edgecolor=p["target"], lw=1.8,
                                    ls=(0, (2, 2))))
            if dist_flash:
                dx, dy = LOCATIONS[DYN_DIST_KEY]
                ax.add_patch(Circle((dx, dy), 0.52, facecolor="none",
                                    edgecolor=p["af"], lw=1.8,
                                    ls=(0, (2, 2))))

        extent = (-lim, lim, -lim, lim)

        # Panel 1: paradigm drive
        ax_S.imshow(S, extent=extent, origin="lower", cmap="magma",
                    vmin=0.0, vmax=1.0, interpolation="bilinear", zorder=0)
        _frame_panel(ax_S, hp_marker=False,
                     target_flash=True, dist_flash=True)
        ax_S.set_title(r"$S(\mathbf{x}, t)$   Paradigm drive",
                       fontsize=22, pad=10)
        ax_S.text(0.5, -0.04,
                  "Bar + search array",
                  transform=ax_S.transAxes, ha="center", va="top",
                  fontsize=18, color=p["muted"], style="italic")

        # Panel 2: attention field
        abs_max = float(np.max(np.abs(M - 1.0)))
        im_M = ax_M.imshow(M, extent=extent, origin="lower",
                           cmap="RdBu_r", vmin=1 - abs_max, vmax=1 + abs_max,
                           interpolation="bilinear", zorder=0)
        _frame_panel(ax_M, hp_marker=True,
                     target_flash=True, dist_flash=True)
        ax_M.set_title(r"$M(\mathbf{x}, t)$   Attention field",
                       fontsize=22, pad=10)
        ax_M.text(0.5, -0.04,
                  "Blue = suppress, red = enhance",
                  transform=ax_M.transAxes, ha="center", va="top",
                  fontsize=18, color=p["muted"], style="italic")
        # M panel doesn't need a colorbar — the diverging red/blue
        # mapping (centered at 1) is self-explanatory given the title,
        # and skipping the cbar avoids overlap with the '=' between
        # panels.

        # Panel 3: effective stimulus drive — distinct cmap from the
        # paradigm panel (left = magma, right = plasma) so the drive
        # panel reads as 'a different quantity', plus a PowerNorm γ=1.4
        # to make the enhancement vs. suppression contrast pop.
        # Plasma chosen for its hue+lightness variation (purple→yellow)
        # so dim regions read clearly different from bright ones.
        from matplotlib.colors import PowerNorm
        drive_max = float(D.max())
        ax_D.imshow(D, extent=extent, origin="lower", cmap="plasma",
                    norm=PowerNorm(gamma=1.4, vmin=0.0, vmax=drive_max),
                    interpolation="bilinear", zorder=0)
        _frame_panel(ax_D, hp_marker=True,
                     target_flash=True, dist_flash=True)
        ax_D.set_title(
            r"$S \cdot M$   Effective stimulus drive",
            fontsize=22, pad=10)
        ax_D.text(0.5, -0.04,
                  "What the cortex 'sees'",
                  transform=ax_D.transAxes, ha="center", va="top",
                  fontsize=18, color=p["muted"], style="italic")

        fig.text(0.355, 0.55, r"$\times$", fontsize=44,
                 ha="center", va="center", color=p["fg"], weight="bold")
        fig.text(0.670, 0.55, r"$=$", fontsize=44,
                 ha="center", va="center", color=p["fg"], weight="bold")

        # (Regime labels removed — at presentation font size they crowd
        # the panel; the panel title + caption + heatmap convey the
        # story without per-corner labels.)

        fig.suptitle("Effective drive = paradigm × attention field",
                     fontsize=24, x=0.04, ha="left", y=0.97, weight="bold")

        # Joint-fit callout: PRF parameters (per VOXEL) AND gain-field
        # parameters (per ROI × subject) are estimated JOINTLY from the
        # BOLD timecourse — NOT fit post-hoc to per-condition PRF
        # estimates. This is the load-bearing methodological message.
        fig.text(
            0.5, 0.07,
            r"$\mathbf{Joint\ fit}$  (NOT post-hoc):  "
            r"PRF params  ($\mu_{\mathrm{PRF}}, \sigma_{\mathrm{PRF}}, "
            r"\mathrm{amp}, \mathrm{baseline}$, per VOXEL)  "
            r"$\;+\;$  "
            r"gain-field params  ($g_{\mathrm{HP}}, g_{\mathrm{D}}, "
            r"g_{\mathrm{T}}, \sigma_{\mathrm{HP}}, \sigma_{\mathrm{dyn}}$, "
            r"per ROI × subject)",
            ha="center", va="center", fontsize=21, color=p["fg"],
            weight="bold",
            bbox=dict(boxstyle="round,pad=0.6", facecolor="#FFF1A6",
                      edgecolor="#C9A227", lw=1.4))
        # (Redundant 'fixed by paradigm' line removed — keep the bottom
        # banner uncluttered for slide presentation.)

        fig.savefig(OUT / "talk_buildup_07_effective_drive.pdf",
                    bbox_inches="tight")
        plt.close(fig)


def main():
    buildup_01_af_basics()
    buildup_01b_two_models()
    buildup_02_sustained_equal()
    buildup_03_hp_weakened()
    buildup_04_target()
    buildup_05_dynamic_distractor()
    buildup_06_equation()
    buildup_07_effective_drive()
    for f in sorted(OUT.glob("talk_buildup_*.pdf")):
        print(f)


if __name__ == "__main__":
    main()
