"""Movie version of talk_buildup_07_effective_drive.

3 panels updating over time:
  S(x, t)       paradigm drive — bar sweeps across the aperture, plus
                4 search items, with target/distractor flashes on a few
                strategically-timed trials.
  M(x, t)       attention field — sustained HP suppression always on,
                transient enhance/suppress at target/distractor onsets.
  S(x, t)·M(x, t)   effective drive — the multiplicative product, what
                the cortex 'sees' through attention.

Output: notes/figures/talk/talk_buildup_07_effective_drive.gif

Reuses parameters from make_figures.py so the movie matches the static
slide exactly.
"""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle, Rectangle

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS))
from make_figures import (  # noqa: E402
    APERTURE_RADIUS, BUILDUP_PALETTE, ECC, HP_KEY, LOCATIONS,
    SD_SUSTAINED, SD_TRANSIENT, AMP_SUSTAINED_FULL, AMP_SUSTAINED_HP,
    AMP_TARGET, AMP_DYN, base_rc, gaussian_2d,
    _build_paradigm_S, _modulation_field_std,
)

OUT = THIS / "talk_buildup_07_effective_drive.gif"

# ---------------------------------------------------------------------------
# Time / paradigm setup
FPS = 10
DURATION = 16.0           # seconds
N_FRAMES = int(DURATION * FPS)
DT = 1.0 / FPS
TIMES = np.arange(N_FRAMES) * DT

# Bar sweeps from y = +3 (top) down to y = −3 — so the bar passes through
# the HP/target row (UL+UR at y ≈ +2.83) FIRST, and the audience sees
# HP suppression immediately as the movie starts.
BAR_Y_START = +3.0
BAR_Y_END = -3.0
BAR_THICKNESS = 1.6

# Trials: each is (onset_t, duration_t, target_loc_key, distractor_loc_key).
# Re-timed for the top→down sweep:
#   Trial 1 fires immediately, while the bar is at the UL/UR row, so the
#     viewer sees bar+target (UL → bright) AND bar+HP (UR → dim) at once.
#   Trial 2 fires mid-sweep with HP distractor at UR while bar is gone —
#     shows that suppression is local to the bar zone (no stim → no
#     visible drive change in the right panel).
#   Trial 3 fires when the bar reaches the LL/LR row, with target+
#     distractor both at the bottom corners.
TRIALS = [
    (1.0, 2.5, "upper_left", "upper_right"),    # bar+UL=bright, bar+HP=dim
    (7.0, 2.0, "lower_right", "upper_right"),   # HP distractor away from bar
    (12.5, 2.5, "lower_left", "lower_right"),   # bottom-row trial
]

# Visual-field grid. Standard meshgrid (xx varies along columns / X-axis).
LIM = 4.8
RES = 220
XS = np.linspace(-LIM, LIM, RES)
XX, YY = np.meshgrid(XS, XS)


def bar_y_at(t):
    """Linear sweep across the duration."""
    f = np.clip(t / DURATION, 0.0, 1.0)
    return BAR_Y_START + (BAR_Y_END - BAR_Y_START) * f


def active_trial(t, kind):
    """Return location key (e.g. 'upper_left') if a target/distractor of
    the given kind is active at time t, else None.
    `kind` is 'target' or 'distractor'.
    """
    for onset, dur, tgt, dst in TRIALS:
        if onset <= t < onset + dur:
            return tgt if kind == "target" else dst
    return None


def build_M_at(t):
    sustained = []
    for k, (cx, cy) in LOCATIONS.items():
        amp = AMP_SUSTAINED_HP if k == HP_KEY else AMP_SUSTAINED_FULL
        sustained.append((cx, cy, SD_SUSTAINED, amp))

    transients = []
    tgt_key = active_trial(t, "target")
    if tgt_key is not None:
        cx, cy = LOCATIONS[tgt_key]
        transients.append((cx, cy, SD_TRANSIENT, AMP_TARGET))
    dst_key = active_trial(t, "distractor")
    if dst_key is not None:
        cx, cy = LOCATIONS[dst_key]
        transients.append((cx, cy, SD_TRANSIENT, AMP_DYN))   # negative amp

    return _modulation_field_std(sustained + transients, XX, YY)


def build_S_at(t):
    return _build_paradigm_S(XX, YY, bar_y_at(t))


# ---------------------------------------------------------------------------
def main():
    p = BUILDUP_PALETTE

    # Pre-compute every frame so we can pin colorbar limits sensibly.
    print(f"Pre-computing {N_FRAMES} frames...")
    Ss = np.stack([build_S_at(t) for t in TIMES])
    Ms = np.stack([build_M_at(t) for t in TIMES])
    Ds = Ss * Ms
    abs_max_M = float(np.max(np.abs(Ms - 1.0)))
    drive_vmax = float(np.percentile(Ds, 99.5))
    print(f"M ∈ 1 ± {abs_max_M:.2f},  drive_vmax = {drive_vmax:.2f}")

    extent = (-LIM, LIM, -LIM, LIM)

    rc = base_rc(p)
    rc.update({
        "font.size": 18, "axes.titlesize": 22, "axes.labelsize": 18,
        "xtick.labelsize": 16, "ytick.labelsize": 16,
        "figure.titlesize": 26,
    })
    with plt.rc_context(rc):
        fig = plt.figure(figsize=(16.5, 7.6))
        # Wider left margin so 'Target' / 'Distractor' labels on the
        # timeline strip don't get clipped. Generous wspace so the
        # × and = symbols sit cleanly between panels (no colorbar to
        # collide with — the M panel doesn't need one for slide use).
        gs = fig.add_gridspec(
            2, 3, height_ratios=[8.5, 1.0],
            hspace=0.20, wspace=0.40,
            left=0.105, right=0.985, top=0.88, bottom=0.10)
        ax_S = fig.add_subplot(gs[0, 0])
        ax_M = fig.add_subplot(gs[0, 1])
        ax_D = fig.add_subplot(gs[0, 2])
        ax_t = fig.add_subplot(gs[1, :])

        # ---- panel framing ----------------------------------------------
        def _frame_panel(ax):
            ax.set_xlim(-LIM, LIM); ax.set_ylim(-LIM, LIM)
            ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            ax.add_patch(Circle((0, 0), APERTURE_RADIUS, facecolor="none",
                                edgecolor=p["muted"], lw=0.9,
                                ls=(0, (4, 3))))
            ax.plot(0, 0, "+", color=p["fg"], ms=10, mew=1.2)
            for k, (x, y) in LOCATIONS.items():
                edge = p["fg"]; lw = 1.0; ls = "-"
                if k == HP_KEY:
                    edge = p["hp"]; lw = 1.8; ls = (0, (2, 2))
                ax.add_patch(Circle((x, y), 0.42, facecolor="none",
                                    edgecolor=edge, lw=lw, ls=ls))

        _frame_panel(ax_S); _frame_panel(ax_M); _frame_panel(ax_D)

        # ---- imshow handles --------------------------------------------
        im_S = ax_S.imshow(Ss[0], extent=extent, origin="lower",
                           cmap="magma", vmin=0.0, vmax=1.0,
                           interpolation="bilinear", zorder=0)
        im_M = ax_M.imshow(Ms[0], extent=extent, origin="lower",
                           cmap="RdBu_r",
                           vmin=1 - abs_max_M, vmax=1 + abs_max_M,
                           interpolation="bilinear", zorder=0)
        # Plasma γ=1.4: hue+lightness variation makes dim (HP-suppressed)
        # regions read as purple/dark and bright (target-enhanced) regions
        # read as yellow, so the suppression/enhancement contrast pops
        # more clearly than a flat magma copy of the paradigm panel.
        from matplotlib.colors import PowerNorm
        im_D = ax_D.imshow(Ds[0], extent=extent, origin="lower",
                           cmap="plasma",
                           norm=PowerNorm(gamma=1.4, vmin=0.0,
                                          vmax=drive_vmax),
                           interpolation="bilinear", zorder=0)

        # transient flash markers (alpha-toggled per frame)
        target_circle = Circle((0, 0), 0.52, facecolor="none",
                                edgecolor=p["target"], lw=2.0,
                                ls=(0, (2, 2)), alpha=0.0, zorder=8)
        distr_circle = Circle((0, 0), 0.52, facecolor="none",
                               edgecolor=p["af"], lw=2.0,
                               ls=(0, (2, 2)), alpha=0.0, zorder=8)
        ax_M.add_patch(target_circle)
        target_circle_S = Circle((0, 0), 0.52, facecolor="none",
                                  edgecolor=p["target"], lw=2.0,
                                  ls=(0, (2, 2)), alpha=0.0, zorder=8)
        distr_circle_S = Circle((0, 0), 0.52, facecolor="none",
                                 edgecolor=p["af"], lw=2.0,
                                 ls=(0, (2, 2)), alpha=0.0, zorder=8)
        ax_S.add_patch(target_circle_S); ax_S.add_patch(distr_circle_S)
        target_circle_D = Circle((0, 0), 0.52, facecolor="none",
                                  edgecolor=p["target"], lw=2.0,
                                  ls=(0, (2, 2)), alpha=0.0, zorder=8)
        distr_circle_D = Circle((0, 0), 0.52, facecolor="none",
                                 edgecolor=p["af"], lw=2.0,
                                 ls=(0, (2, 2)), alpha=0.0, zorder=8)
        ax_D.add_patch(target_circle_D); ax_D.add_patch(distr_circle_D)
        ax_M.add_patch(distr_circle)

        ax_S.set_title(r"$S(\mathbf{x}, t)$   Paradigm drive",
                       pad=12, weight="bold")
        ax_M.set_title(r"$M(\mathbf{x}, t)$   Attention field",
                       pad=12, weight="bold")
        ax_D.set_title(r"$S \cdot M$   Effective stimulus drive",
                       pad=12, weight="bold")

        fig.text(0.370, 0.52, r"$\times$", fontsize=44,
                 ha="center", va="center", color=p["fg"], weight="bold")
        fig.text(0.673, 0.52, r"$=$", fontsize=44,
                 ha="center", va="center", color=p["fg"], weight="bold")

        # M panel doesn't need a colorbar for the slide context — the
        # red/blue mapping (suppression vs. enhancement, centered at 1)
        # is already labeled by the panel title and is symmetric by
        # construction. Skipping the cbar also removes the overlap with
        # the inter-panel '=' symbol.

        # ---- timeline strip --------------------------------------------
        ax_t.set_xlim(0, DURATION)
        ax_t.set_ylim(0, 1)
        ax_t.set_xlabel("Time (s)")
        ax_t.set_yticks([])
        ax_t.tick_params(axis="x", labelsize=15)
        for sp in ("top", "left", "right"):
            ax_t.spines[sp].set_visible(False)
        ax_t.spines["bottom"].set_color(p["muted"])
        for onset, dur, tgt, dst in TRIALS:
            ax_t.add_patch(Rectangle((onset, 0.55), dur, 0.35,
                                     facecolor=p["target"], edgecolor="none",
                                     alpha=0.45))
            ax_t.add_patch(Rectangle((onset, 0.10), dur, 0.35,
                                     facecolor=p["af"], edgecolor="none",
                                     alpha=0.45))
        ax_t.text(-0.6, 0.72, "Target", color=p["target"], fontsize=16,
                  ha="right", va="center", weight="bold")
        ax_t.text(-0.6, 0.27, "Distractor", color=p["af"], fontsize=16,
                  ha="right", va="center", weight="bold")
        cursor = ax_t.axvline(0, color=p["fg"], lw=2.0, alpha=0.95)

        title_text = fig.suptitle(
            "Effective drive = paradigm × attention field",
            fontsize=24, x=0.04, ha="left", y=0.97, weight="bold")
        time_text = fig.text(
            0.97, 0.97, "t = 0.0 s", ha="right", va="top",
            fontsize=20, color=p["fg"], weight="bold")

        # ---- update ----------------------------------------------------
        def update(i):
            t = TIMES[i]
            im_S.set_data(Ss[i]); im_M.set_data(Ms[i]); im_D.set_data(Ds[i])
            cursor.set_xdata([t, t])
            time_text.set_text(f"t = {t:4.1f} s")

            tgt_key = active_trial(t, "target")
            dst_key = active_trial(t, "distractor")
            for c_set, key in [
                ((target_circle_S, target_circle, target_circle_D), tgt_key),
                ((distr_circle_S, distr_circle, distr_circle_D), dst_key),
            ]:
                if key is not None:
                    cx, cy = LOCATIONS[key]
                    for c in c_set:
                        c.set_center((cx, cy)); c.set_alpha(0.95)
                else:
                    for c in c_set:
                        c.set_alpha(0.0)
            return [im_S, im_M, im_D, cursor, time_text,
                    target_circle_S, target_circle, target_circle_D,
                    distr_circle_S, distr_circle, distr_circle_D]

        anim = FuncAnimation(fig, update, frames=N_FRAMES,
                             interval=1000 / FPS, blit=False)
        OUT.parent.mkdir(parents=True, exist_ok=True)
        anim.save(OUT, writer=PillowWriter(fps=FPS))
        plt.close(fig)
        print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
