"""Build a story-driven slide deck for the meeting with Dock Duncan and
Jan Theeuwes.

Four sections:
  (a) Jensen's inequality makes naive distance-from-HP analyses tricky.
  (b) We have an attention-field × PRF model that makes complex
      predictions.
  (c) Parameter estimates per ROI — distributions + HP-vs-LP test.
  (d) Per-ROI predicted shift vector fields at empirical median params.

All slides are 11 × 8.5 in landscape, larger fonts than the working
diagnostic PDFs.

Inputs:
  notes/af_parameters.tsv       (per-fit shared params, all 30 subjects)
  notes/predict_shifts_cluster.tsv

Output:
  notes/talk_docky_jan.pdf
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Circle
from scipy import stats

from retsupp.utils.data import distractor_locations


CONDITIONS = ['upper_right', 'upper_left', 'lower_left', 'lower_right']
RING_KEYS = ['upper right', 'upper left', 'lower left', 'lower right']
APERTURE = 3.17
RING_R = 4.0
ROI_ORDER = ['V1', 'V2', 'V3', 'V3AB', 'hV4', 'LO', 'TO', 'VO']

# Larger font defaults — for slides, not diagnostics.
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18,
})


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def get_ring_positions():
    return np.array([list(distractor_locations[k]) for k in RING_KEYS],
                    dtype=np.float32)


def make_grid(resolution=121, grid_radius=5.0):
    g1d = np.linspace(-grid_radius, grid_radius, resolution).astype(np.float32)
    GX, GY = np.meshgrid(g1d, g1d)
    return GX, GY, g1d


def gaussian_2d(GX, GY, mu, sigma):
    return np.exp(-((GX - mu[0]) ** 2 + (GY - mu[1]) ** 2)
                   / (2 * sigma ** 2))


def modulation_field(GX, GY, ring, hp_idx, sigma_AF, g_HP, g_LP):
    A = np.stack([gaussian_2d(GX, GY, ring[i], sigma_AF)
                   for i in range(len(ring))], axis=0)
    a_hp = A[hp_idx]
    a_lp = A.sum(axis=0) - a_hp
    M = 1.0 + g_HP * a_hp + g_LP * a_lp
    return np.maximum(M, 0.0)


def predict_shift_field(sigma_AF, g_HP, g_LP, seeds,
                          prf_sd=1.0, integration_resolution=81,
                          integration_radius=5.0):
    GX, GY, _ = make_grid(integration_resolution, integration_radius)
    ring = get_ring_positions()
    pred = np.empty((len(CONDITIONS), seeds.shape[0], 2), dtype=np.float32)
    for ci in range(len(CONDITIONS)):
        M = modulation_field(GX, GY, ring, hp_idx=ci, sigma_AF=sigma_AF,
                              g_HP=g_HP, g_LP=g_LP)
        for vi, (vx, vy) in enumerate(seeds):
            S = gaussian_2d(GX, GY, (vx, vy), prf_sd)
            rho = np.clip(M * S, 0, None)
            Z = rho.sum() + 1e-12
            pred[ci, vi, 0] = (GX * rho).sum() / Z
            pred[ci, vi, 1] = (GY * rho).sum() / Z
    return pred


def add_aperture_and_ring(ax, ring, hp_idx=None):
    ax.add_patch(Circle((0, 0), APERTURE, fill=False, ec='0.4', ls='--', lw=0.8))
    for i in range(len(ring)):
        c = 'C3' if (hp_idx is not None and i == hp_idx) else '0.5'
        ms = 14 if (hp_idx is not None and i == hp_idx) else 8
        ax.plot(ring[i, 0], ring[i, 1], 'o' if c == '0.5' else '*',
                markersize=ms, color=c, mec='k', alpha=0.9)


# ---------------------------------------------------------------------------
# Slides
# ---------------------------------------------------------------------------

def title_slide(pdf):
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis('off')
    ax.text(0.5, 0.65, 'Joint Attention-Field × PRF model',
            ha='center', va='center', fontsize=30, weight='bold',
            transform=ax.transAxes)
    ax.text(0.5, 0.55,
             'Distractor-driven shifts of population receptive\n'
             'fields in human visual cortex',
             ha='center', va='center', fontsize=18,
             transform=ax.transAxes)
    ax.text(0.5, 0.30,
             'meeting with Dock Duncan & Jan Theeuwes',
             ha='center', va='center', fontsize=14, style='italic',
             transform=ax.transAxes, color='0.4')
    ax.text(0.5, 0.20,
             'retsupp · 7T fMRI · 30 subjects · 4 HP-distractor blocks · '
             'sweeping bar PRF mapping',
             ha='center', va='center', fontsize=11,
             transform=ax.transAxes, color='0.4')
    pdf.savefig(fig); plt.close(fig)


def section_divider(pdf, letter, title, subtitle=''):
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis('off')
    ax.text(0.5, 0.55, f'({letter})  {title}',
            ha='center', va='center', fontsize=28, weight='bold',
            transform=ax.transAxes)
    if subtitle:
        ax.text(0.5, 0.45, subtitle,
                ha='center', va='center', fontsize=16, style='italic',
                color='0.4', transform=ax.transAxes)
    pdf.savefig(fig); plt.close(fig)


# --- (a) Jensen ------------------------------------------------------------

def slide_jensen(pdf):
    """Single visual: random noise around a point biases distance UP.

    A point at distance d from HP — a noise cloud around it — show that
    on average noisy realizations sit further from HP than d. So if a
    voxel's PRF estimate is jittered by noise, the apparent shift will
    look "AWAY from HP" even with zero true attentional effect.
    """
    rng = np.random.default_rng(0)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))

    # ---- (a) cartoon: HP + a point + a noise cloud.
    ax = axes[0]
    HP = np.array([0.0, 4.0])
    P  = np.array([0.0, 1.0])    # point at distance 3.0 from HP
    sigma_n = 0.7
    n = 300
    eps = rng.normal(0, sigma_n, size=(n, 2))
    cloud = P + eps
    d0 = np.linalg.norm(P - HP)
    d_n = np.linalg.norm(cloud - HP, axis=1)

    # Draw a circle on which P lies (constant distance from HP).
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(HP[0] + d0 * np.cos(theta), HP[1] + d0 * np.sin(theta),
             '--', color='0.5', lw=1, alpha=0.7)
    # HP star.
    ax.plot(HP[0], HP[1], '*', color='C3', markersize=30, mec='k', zorder=10)
    ax.text(HP[0] + 0.2, HP[1] + 0.2, 'HP', fontsize=18, color='C3',
             weight='bold')
    # The point P.
    ax.plot(P[0], P[1], 'o', color='C0', markersize=14, mec='k', zorder=8)
    ax.text(P[0] - 0.2, P[1] + 0.4, 'true\nPRF', ha='right', fontsize=12,
             color='C0', weight='bold')
    # Distance arrow.
    ax.annotate('', xy=HP, xytext=P,
                 arrowprops=dict(arrowstyle='<->', color='0.5', lw=1.0))
    ax.text(0.3, 2.5, f'd = {d0:.0f}', fontsize=14, color='0.4',
             style='italic')
    # Noise cloud (semi-transparent dots).
    ax.scatter(cloud[:, 0], cloud[:, 1], s=10, alpha=0.30,
                color='C2', edgecolor='none')
    # Mean of cloud: mark.
    ax.plot(cloud.mean(0)[0], cloud.mean(0)[1], 'x',
             color='k', markersize=10, mew=2, zorder=12)
    ax.text(cloud.mean(0)[0] + 0.3, cloud.mean(0)[1] - 0.3,
             'noisy estimates', fontsize=10, color='C2',
             style='italic')
    ax.set_xlim(-3.5, 3.5); ax.set_ylim(-1.5, 5.5)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title('Take any voxel a distance d from HP.\n'
                  'Add isotropic noise to its PRF estimate.',
                  fontsize=14)

    # ---- (b) histogram of distances: original (vertical line) vs noisy (cloud).
    ax = axes[1]
    ax.hist(d_n, bins=25, color='C2', alpha=0.6,
             edgecolor='k', linewidth=0.4,
             label=f'noisy distance to HP\n(mean = {d_n.mean():.2f})')
    ax.axvline(d0, color='C0', lw=3,
                label=f'true distance to HP\n(d = {d0:.2f})')
    ax.axvline(d_n.mean(), color='k', lw=2, ls='--',
                label='mean of noisy distances')
    ax.set_xlabel('distance to HP', fontsize=14)
    ax.set_ylabel('count', fontsize=14)
    ax.legend(fontsize=11, loc='upper right')
    ax.set_title('On AVERAGE, noisy distances > true distance.\n'
                  '(Distance is convex — Jensen.)',
                  fontsize=14)
    ax.grid(alpha=0.2)

    fig.suptitle(
        '(a)  Random noise pushes "distance to HP" upward.\n'
        'Hence "the PRF shifted AWAY from HP" can be a noise artifact.',
        fontsize=15, weight='bold',
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    pdf.savefig(fig); plt.close(fig)


# --- (b) The model --------------------------------------------------------

def slide_model_formula(pdf):
    """Schematic + minimal text. The 4 ring AFs with one annotated."""
    fig = plt.figure(figsize=(13, 8))

    # Title.
    fig.text(0.5, 0.94, '(b)  The model',
              ha='center', va='top', fontsize=24, weight='bold')

    # ---- Left: schematic of the 4 ring AFs in visual space.
    ax = fig.add_axes([0.04, 0.10, 0.50, 0.78])
    ax.set_xlim(-5.5, 5.5); ax.set_ylim(-5.5, 5.5)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    ax.spines[:].set_visible(False)

    # Bar aperture.
    ax.add_patch(Circle((0, 0), APERTURE, fill=False,
                          ec='0.4', ls='--', lw=1.2))
    ax.text(0, -3.6, 'bar aperture (3.17°)', ha='center',
             fontsize=10, color='0.4', style='italic')

    # 4 ring AFs (circles at 1σ, plus translucent fill).
    ring = get_ring_positions()
    sigma_show = 1.2     # for the schematic
    for i, mu in enumerate(ring):
        is_hp = (i == 0)
        col = 'C3' if is_hp else 'C0'
        ax.add_patch(Circle(mu, sigma_show, alpha=0.18, color=col, zorder=2))
        ax.add_patch(Circle(mu, sigma_show, fill=False, ec=col, lw=1.5, zorder=3))
        ax.plot(mu[0], mu[1],
                '*' if is_hp else 'o',
                color=col, mec='k',
                markersize=22 if is_hp else 12, zorder=5)

    # Annotate σ_AF on the HP one.
    hp = ring[0]
    ax.annotate('', xy=hp + np.array([sigma_show, 0]), xytext=hp,
                 arrowprops=dict(arrowstyle='->', color='C3', lw=1.5))
    ax.text(hp[0] + sigma_show / 2, hp[1] + 0.35, 'σ_AF',
             color='C3', fontsize=13, weight='bold', ha='center')
    # HP label.
    ax.annotate('HP\n(gain = g_HP)',
                 xy=hp, xytext=(4.3, 4.3),
                 fontsize=12, color='C3', weight='bold', ha='center',
                 arrowprops=dict(arrowstyle='->', color='C3', lw=1))
    # LP label (point to one of the 3).
    lp = ring[1]
    ax.annotate('LP\n(gain = g_LP)',
                 xy=lp, xytext=(-4.5, 3.0),
                 fontsize=12, color='C0', weight='bold', ha='center',
                 arrowprops=dict(arrowstyle='->', color='C0', lw=1))
    # 3-LP note.
    ax.text(0, -4.8,
             '4 attention fields at fixed ring locations\n'
             '(4° eccentricity).\n'
             'HP gets gain  g_HP,  the other 3 share gain  g_LP.',
             ha='center', fontsize=11, color='0.25')

    # ---- Right: tight formula box.
    ax2 = fig.add_axes([0.58, 0.18, 0.40, 0.65])
    ax2.axis('off')
    ax2.text(0.5, 0.95,
              'Forward model',
              ha='center', va='top', fontsize=14, weight='bold',
              transform=ax2.transAxes)
    formula = (
        'For each voxel v under HP condition C:\n'
        '\n'
        '  R_v_C (g) = M_C (g)  ·  S_v (g)\n'
        '\n'
        'with the modulation field\n'
        '\n'
        '  M_C (g) = 1\n'
        '          + g_HP · A_HP_C (g)\n'
        '          + g_LP · Σ A_LP (g)\n'
    )
    ax2.text(0.04, 0.83, formula, ha='left', va='top',
              fontsize=13, family='monospace',
              transform=ax2.transAxes)

    ax2.text(0.04, 0.36,
              '3 SHARED parameters per ROI:\n\n'
              '  σ_AF      AF size\n'
              '  g_HP      gain at HP location\n'
              '  g_LP      gain at LP locations\n'
              '\n'
              'Negative gain  ⇒  suppression.',
              ha='left', va='top', fontsize=12, family='monospace',
              color='0.25', transform=ax2.transAxes)

    pdf.savefig(fig); plt.close(fig)


def slide_model_panels(pdf):
    """Schematic + several modulation-field heatmaps at different gains/sigmas."""
    GX, GY, _ = make_grid()
    ring = get_ring_positions()
    hp_idx = 0   # HP = upper-right

    # Panels: schematic + 4 parameter regimes.
    panel_specs = [
        ('schematic', None,
         'Geometry',
         '★ = HP,  ● = LP,  dashed = bar aperture'),
        ('heat',      (2.0, +0.5,  0.0),
         'HP attraction\n(pure gain at HP)',
         'σ=2°,  g_HP=+0.5,  g_LP=0'),
        ('heat',      (2.0, -0.5,  0.0),
         'HP suppression\n(pure gain at HP)',
         'σ=2°,  g_HP=−0.5,  g_LP=0'),
        ('heat',      (2.0, -0.5, +0.2),
         'HP suppress + LP attract\n(HP-specific, mixed signs)',
         'σ=2°,  g_HP=−0.5,  g_LP=+0.2'),
        ('heat',      (2.0, -0.3, -0.3),
         'Uniform 4-AF suppression\n(NO HP-specific effect)',
         'σ=2°,  g_HP=g_LP=−0.3'),
    ]

    fig, axes = plt.subplots(2, 5, figsize=(20, 9))

    # ---- Top row: modulation field heatmaps OR schematic.
    for col_i, (kind, params, title_main, title_params) in enumerate(panel_specs):
        ax = axes[0, col_i]
        if kind == 'schematic':
            ax.set_xlim(-5, 5); ax.set_ylim(-5, 5)
            ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
            ax.add_patch(Circle((0, 0), APERTURE, fill=False,
                                  ec='0.4', ls='--', lw=1.0))
            sigma_show = 1.2
            for i, mu in enumerate(ring):
                is_hp = (i == hp_idx)
                col = 'C3' if is_hp else 'C0'
                ax.add_patch(Circle(mu, sigma_show, alpha=0.20, color=col))
                ax.add_patch(Circle(mu, sigma_show, fill=False, ec=col, lw=1.2))
                ax.plot(mu[0], mu[1], '*' if is_hp else 'o',
                        color=col, mec='k',
                        markersize=20 if is_hp else 11)
            ax.annotate('σ_AF', xy=ring[hp_idx] + np.array([sigma_show, 0]),
                         xytext=ring[hp_idx],
                         arrowprops=dict(arrowstyle='->', color='C3', lw=1.0),
                         fontsize=12, color='C3', weight='bold')
        else:
            sigma_AF, g_HP, g_LP = params
            M = modulation_field(GX, GY, ring, hp_idx=hp_idx,
                                  sigma_AF=sigma_AF, g_HP=g_HP, g_LP=g_LP)
            vmax = 0.6
            ax.imshow(M, extent=(-5, 5, -5, 5), origin='lower',
                       cmap='RdBu', vmin=1 - vmax, vmax=1 + vmax)
            add_aperture_and_ring(ax, ring, hp_idx=hp_idx)
            ax.set_xlim(-5, 5); ax.set_ylim(-5, 5)
            ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title_main, fontsize=11, weight='bold')
        ax.text(0.5, -0.10, title_params, ha='center', va='top',
                 fontsize=9.5, color='0.25', transform=ax.transAxes)

    # Annotate top-row labels.
    axes[0, 0].set_ylabel('M(g)\nmodulation field', fontsize=11, weight='bold')

    # ---- Bottom row: predicted PRF-center shift VECTOR FIELDS for the
    # same parameter regimes.
    s1d = np.arange(-3, 3.1, 0.7, dtype=np.float32)
    SX, SY = np.meshgrid(s1d, s1d)
    seeds = np.stack([SX.ravel(), SY.ravel()], axis=1).astype(np.float32)
    inside = np.linalg.norm(seeds, axis=1) <= APERTURE
    seeds = seeds[inside]
    for col_i, (kind, params, _, _) in enumerate(panel_specs):
        ax = axes[1, col_i]
        if kind == 'schematic':
            # Show a "neutral" reference grid: all gains 0 → no shift.
            ax.scatter(seeds[:, 0], seeds[:, 1], s=14, color='0.5',
                        alpha=0.7)
            add_aperture_and_ring(ax, ring, hp_idx=hp_idx)
            ax.text(0, -4.4, 'PRFs at rest\n(no modulation)',
                    ha='center', fontsize=10, color='0.4', style='italic')
        else:
            sigma_AF, g_HP, g_LP = params
            pred = predict_shift_field(sigma_AF, g_HP, g_LP, seeds,
                                          prf_sd=1.0)
            dx = pred[0, :, 0] - seeds[:, 0]
            dy = pred[0, :, 1] - seeds[:, 1]
            add_aperture_and_ring(ax, ring, hp_idx=hp_idx)
            ax.quiver(seeds[:, 0], seeds[:, 1], dx, dy,
                       angles='xy', scale_units='xy',
                       scale=1.0 / 8.0, color='k',
                       width=0.008, alpha=0.95)
        ax.set_xlim(-5, 5); ax.set_ylim(-5, 5)
        ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    axes[1, 0].set_ylabel('predicted\nPRF-center shifts\n(arrows ×8)',
                            fontsize=11, weight='bold')

    fig.suptitle(
        '(b)  What the model predicts at different parameter settings.\n'
        'Top: modulation field M(g)  (RED = suppression,  BLUE = attraction).   '
        'Bottom: predicted PRF-center shifts under HP=★.',
        fontsize=15, weight='bold',
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    pdf.savefig(fig); plt.close(fig)


# --- (c) Parameter estimates ---------------------------------------------

def slide_parameter_distributions(pdf, df: pd.DataFrame):
    """Three wide panels: σ_AF, g_HP, g_LP per ROI with mean ± SEM."""
    rois = [r for r in ROI_ORDER if r in df['roi'].unique()]
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
    metrics = [
        ('sigma_AF', 'σ_AF  (deg)',  0,    8.0,  False),
        ('g_HP',    'g_HP',         -1.5, +1.5, True),
        ('g_LP',    'g_LP',         -1.5, +1.5, True),
    ]
    import seaborn as sns
    for ax, (col, label, ylo, yhi, sym) in zip(axes, metrics):
        d_plot = df.assign(_clip=df[col].clip(ylo, yhi))
        sns.stripplot(data=d_plot, x='roi', y='_clip', ax=ax,
                       order=rois, hue='roi', palette='Set2', legend=False,
                       jitter=0.20, alpha=0.55, size=8)
        # Mean ± SEM error bars + (for the gain panels) t-test against 0.
        for i, roi in enumerate(rois):
            x = df.loc[df['roi'] == roi, col].dropna().values
            if len(x) < 2:
                continue
            mean = float(np.mean(x))
            sem = float(np.std(x, ddof=1) / np.sqrt(len(x)))
            ax.errorbar([i], [mean], yerr=[sem],
                         fmt='s', color='k', markersize=11,
                         ecolor='k', elinewidth=2.5, capsize=8, zorder=10)
            if sym:    # only annotate symmetric (gain) panels
                t, p = stats.ttest_1samp(x, 0.0)
                sig = ('***' if p < 0.001 else '**' if p < 0.01
                        else '*' if p < 0.05 else 'n.s.')
                col_anno = ('C2' if (p < 0.05 and mean > 0)
                             else 'C3' if (p < 0.05 and mean < 0)
                             else '0.5')
                ax.text(i, yhi - 0.05, sig,
                         ha='center', va='top', fontsize=14,
                         color=col_anno, weight='bold')
        if sym:
            ax.axhline(0, color='gray', lw=0.7, ls='--')
        ax.set_ylim(ylo, yhi)
        ax.set_xlabel('')
        ax.set_ylabel(label, fontsize=18)
        ax.set_title(label, fontsize=20, weight='bold')
        ax.grid(alpha=0.2)
        ax.tick_params(axis='x', labelsize=14)
        ax.tick_params(axis='y', labelsize=13)
    fig.suptitle(
        '(c)  AF parameter estimates per ROI '
        f'({df.subject.nunique()} subjects, {len(df)} fits)   '
        '— dots = subjects, black square = mean ± SEM',
        fontsize=18, weight='bold',
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    pdf.savefig(fig); plt.close(fig)


def slide_hp_lp_test(pdf, df: pd.DataFrame):
    """Headline HP-vs-LP test using the SCALE-INVARIANT log-ratio metric.

    metric = log( (1 + g_HP) / (1 + g_LP) )
           = log of the ratio of effective HP-modulation to LP-modulation.

    Y-axis labelled as natural multiplicative factors (1×, 0.75×, 0.5×, …)
    so the eye reads "M_HP is half of M_LP" instead of "log = -0.69".
    """
    rois = [r for r in ROI_ORDER if r in df['roi'].unique()]
    YLIM = float(np.log(2.5))   # ~ ±2.5×
    rng = np.random.default_rng(0)
    n_boot = 2000

    fig, ax = plt.subplots(figsize=(15, 7))
    import seaborn as sns
    d_plot = df.assign(_clip=df['log_ratio'].clip(-YLIM, YLIM))
    sns.stripplot(data=d_plot, x='roi', y='_clip', ax=ax,
                   order=rois, hue='roi', palette='Set2', legend=False,
                   jitter=0.18, alpha=0.55, size=8)
    ax.axhline(0, color='gray', lw=0.7, ls='--')

    for i, roi in enumerate(rois):
        x = df.loc[df['roi'] == roi, 'log_ratio'].dropna().values
        if len(x) < 3:
            continue
        med = float(np.median(x))
        boots = rng.choice(x, size=(n_boot, len(x)), replace=True)
        boot_meds = np.median(boots, axis=1)
        lo, hi = np.quantile(boot_meds, [0.025, 0.975])
        ax.errorbar([i], [med], yerr=[[med - lo], [hi - med]],
                     fmt='s', color='k', markersize=13,
                     ecolor='k', elinewidth=2.5, capsize=8, zorder=10)
        try:
            _, p = stats.wilcoxon(x, alternative='less', zero_method='pratt')
        except ValueError:
            p = np.nan
        sig = ('***' if p < 0.001 else '**' if p < 0.01
                else '*' if p < 0.05 else 'n.s.')
        col = 'C3' if (np.isfinite(p) and p < 0.05) else '0.4'
        ax.text(i, YLIM + 0.08,
                 f'p = {p:.3f}\n{sig}',
                 ha='center', va='bottom', fontsize=12,
                 color=col, weight='bold' if p < 0.05 else 'normal')
        n_lo = int((x < -YLIM).sum())
        n_hi = int((x > +YLIM).sum())
        if n_lo:
            ax.annotate(f'↓{n_lo}', xy=(i, -YLIM + 0.02),
                         ha='center', va='bottom', fontsize=11, color='C3')
        if n_hi:
            ax.annotate(f'↑{n_hi}', xy=(i, YLIM - 0.02),
                         ha='center', va='top', fontsize=11, color='C3')

    # Multiplicative-factor ticks.
    factor_pcts = [-50, -33, -25, -10, 0, +10, +25, +50, +100]
    ticks, labels = [], []
    for pct in factor_pcts:
        v = float(np.log(1.0 + pct / 100.0))
        if -YLIM <= v <= YLIM:
            ticks.append(v)
            labels.append(f'{pct:+d}%' if pct != 0 else '0')
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=13)
    ax.set_ylim(-YLIM - 0.05, YLIM + 0.30)
    ax.tick_params(axis='x', labelsize=15)
    ax.set_xlabel('')
    ax.set_ylabel(
        'M_HP / M_LP   (negative ⇒ HP suppressed more than LP)',
        fontsize=16,
    )
    ax.set_title(
        '(c)  HP-vs-LP test  —  scale-invariant log-ratio  '
        f'(N = {df.subject.nunique()} subjects, Wilcoxon one-sided)',
        fontsize=17, weight='bold',
    )
    ax.grid(alpha=0.2)
    pdf.savefig(fig); plt.close(fig)


# --- (d) Per-ROI predictions -----------------------------------------------

def _pick_params(df: pd.DataFrame, roi: str, agg='median'):
    sub = df[df['roi'] == roi]
    f = np.median if agg == 'median' else np.mean
    return (float(f(sub['sigma_AF'])),
             float(f(sub['g_HP'])),
             float(f(sub['g_LP'])))


def _draw_af_visual(ax, ring, sigma_AF, g_HP, g_LP, hp_idx=0,
                      gain_norm: float | None = None):
    """Draw 4 ring AFs with size = σ_AF and shading = |gain|."""
    if gain_norm is None:
        gain_norm = max(abs(g_HP), abs(g_LP), 0.05)
    for i, mu in enumerate(ring):
        is_hp = (i == hp_idx)
        g = g_HP if is_hp else g_LP
        # Color reflects sign of gain.
        edgecol = 'C3' if g < 0 else 'C0'
        # Alpha reflects |g| relative to the larger of the two gains.
        alpha = min(0.85, 0.15 + 0.85 * abs(g) / gain_norm)
        ax.add_patch(Circle(mu, sigma_AF, alpha=alpha, color=edgecol,
                              zorder=2))
        ax.add_patch(Circle(mu, sigma_AF, fill=False, ec=edgecol,
                              lw=1.5, zorder=3, alpha=0.9))
        ax.plot(mu[0], mu[1], '*' if is_hp else 'o',
                color='gold' if is_hp else 'k', mec='k',
                markersize=18 if is_hp else 9, zorder=8)
    ax.add_patch(Circle((0, 0), APERTURE, fill=False,
                          ec='0.4', ls='--', lw=1.0))


def _auto_quiver_scale(dxs, dys, bin_width=1.0,
                         target_fraction=0.6, percentile=85):
    """Pick quiver scale so the percentile-th arrow occupies
    target_fraction of bin_width."""
    mags = np.sqrt(np.asarray(dxs) ** 2 + np.asarray(dys) ** 2)
    mags = mags[np.isfinite(mags)]
    if len(mags) == 0:
        return 1.0
    p = float(np.percentile(mags, percentile))
    if p <= 0:
        return 1.0
    return max(1.0, target_fraction * bin_width / p)


def _vector_field_panel(ax, sigma_AF, g_HP, g_LP, ring, prf_sd=1.0,
                         spacing=0.6, scale=None):
    """Per-condition shift relative to the mean prediction across all 4
    conditions. Arrows START at the MEAN-across-conditions prediction
    and POINT toward the per-condition (HP=UR) prediction. So the arrow
    base is "where the PRF would be if attention had no HP-specificity"
    and the arrow tip is "where HP=UR pushes it"."""
    s1d = np.arange(-3, 3.1, spacing, dtype=np.float32)
    SX, SY = np.meshgrid(s1d, s1d)
    seeds = np.stack([SX.ravel(), SY.ravel()], axis=1).astype(np.float32)
    inside = np.linalg.norm(seeds, axis=1) <= APERTURE
    seeds = seeds[inside]
    pred = predict_shift_field(sigma_AF, g_HP, g_LP, seeds, prf_sd=prf_sd)
    mean_pred = pred.mean(axis=0)             # (V, 2)
    # Condition-specific shift at HP=UR.
    dx = pred[0, :, 0] - mean_pred[:, 0]
    dy = pred[0, :, 1] - mean_pred[:, 1]
    # Auto-scale arrows to fit the panel (each panel gets its own scale).
    if scale is None:
        scale = _auto_quiver_scale(dx, dy, bin_width=spacing,
                                      target_fraction=0.7, percentile=85)
    # AF circles + aperture.
    _draw_af_visual(ax, ring, sigma_AF, g_HP, g_LP, hp_idx=0)
    # Mark the MEAN positions (small grey dots) — arrow START points.
    ax.scatter(mean_pred[:, 0], mean_pred[:, 1],
                s=4, color='0.45', alpha=0.7, zorder=4)
    # Quiver of condition-specific shift, anchored at MEAN positions.
    ax.quiver(mean_pred[:, 0], mean_pred[:, 1], dx, dy,
              angles='xy', scale_units='xy',
              scale=1.0 / scale, color='k',
              width=0.007, alpha=0.95, zorder=10)
    ax.set_xlim(-4.7, 4.7); ax.set_ylim(-4.7, 4.7)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    # Annotate scale.
    ax.text(0.02, 0.02, f'×{scale:.0f}',
             ha='left', va='bottom', fontsize=10, color='0.4',
             transform=ax.transAxes)


def _observed_field_panel(ax, df_sh_roi, ring, scale=None,
                            grid_extent=3.5, grid_n=2,
                            min_n_subj_per_bin=3):
    """Observed shift vector field (data) side of the per-ROI panel.

    Procedure (apples-to-apples with PRED panel):
      1. Per voxel (subject × voxel_idx), compute the mean of the
         observed conditionwise positions across the 4 conditions —
         this is the per-voxel "what would the PRF be without
         HP-specificity" reference, mirroring the PRED panel which
         uses mean(pred) across conditions.
      2. Rotate every (subject × condition × voxel) row to canonical
         (HP at (0, +4°)). Compute the rotated DEVIATION from this
         reference: obs_C_rot − mean_obs_rot.
      3. Bin in a 2D grid (default 2×2 = 4 quadrants).
      4. Per (subject, bin): median of the rotated deviations.
      5. Per bin: mean of those per-subject medians across subjects.
    Bins outside the bar aperture are dropped.
    """
    if df_sh_roi is None or len(df_sh_roi) == 0:
        ax.axis('off')
        return
    # Per-voxel mean position across the 4 conditions.
    mean_obs = (df_sh_roi.groupby(['subject', 'voxel_idx'], observed=True)
                  [['obs_x', 'obs_y']].transform('mean'))
    df_sh_roi = df_sh_roi.assign(
        obs_x_dev=df_sh_roi['obs_x'] - mean_obs['obs_x'],
        obs_y_dev=df_sh_roi['obs_y'] - mean_obs['obs_y'],
    )
    R = {c: rotation_to_canonical(c) for c in CONDITIONS}
    bx = df_sh_roi['base_x'].values; by = df_sh_roi['base_y'].values
    odx = df_sh_roi['obs_x_dev'].values
    ody = df_sh_roi['obs_y_dev'].values
    rb_x = np.empty_like(bx); rb_y = np.empty_like(by)
    do_x = np.empty_like(odx); do_y = np.empty_like(ody)
    for c in CONDITIONS:
        m = (df_sh_roi['condition'] == c).values
        if not m.any():
            continue
        Rc = R[c]
        rb_x[m] = Rc[0, 0] * bx[m] + Rc[0, 1] * by[m]
        rb_y[m] = Rc[1, 0] * bx[m] + Rc[1, 1] * by[m]
        do_x[m] = Rc[0, 0] * odx[m] + Rc[0, 1] * ody[m]
        do_y[m] = Rc[1, 0] * odx[m] + Rc[1, 1] * ody[m]
    # Bin.
    edges = np.linspace(-grid_extent, grid_extent, grid_n + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    ix = np.digitize(rb_x, edges) - 1
    iy = np.digitize(rb_y, edges) - 1
    valid = (ix >= 0) & (ix < grid_n) & (iy >= 0) & (iy < grid_n)
    sub = df_sh_roi['subject'].values

    # Per-(subject, bin) median.
    per_sb = {}
    for k in np.where(valid)[0]:
        key = (int(sub[k]), int(ix[k]), int(iy[k]))
        per_sb.setdefault(key, []).append((do_x[k], do_y[k]))
    # Per-bin: mean (across subjects) of (per-subject median Δx, Δy).
    grid = {}
    for (s, i, j), vec_list in per_sb.items():
        arr = np.array(vec_list)
        med = np.median(arr, axis=0)        # per-subject median
        grid.setdefault((i, j), []).append(med)

    cx, cy = [], []; vx, vy = [], []
    for (i, j), per_subj_medians in grid.items():
        if len(per_subj_medians) < min_n_subj_per_bin:
            continue
        ax_pt, ay_pt = centers[i], centers[j]
        # Aperture filter on bin CENTER.
        if np.sqrt(ax_pt ** 2 + ay_pt ** 2) > APERTURE:
            continue
        m = np.mean(per_subj_medians, axis=0)
        cx.append(ax_pt); cy.append(ay_pt)
        vx.append(m[0]); vy.append(m[1])
    cx = np.array(cx); cy = np.array(cy); vx = np.array(vx); vy = np.array(vy)

    bin_w = (2.0 * grid_extent) / grid_n
    if scale is None:
        scale = _auto_quiver_scale(vx, vy, bin_width=bin_w,
                                      target_fraction=0.7, percentile=85)

    # Aperture + ring positions (canonical orientation).
    canon_ring = np.array([
        [0.0, +RING_R], [-RING_R, 0.0],
        [0.0, -RING_R], [+RING_R, 0.0],
    ], dtype=np.float32)
    ax.add_patch(Circle((0, 0), APERTURE, fill=False,
                          ec='0.4', ls='--', lw=1.0))
    for k, mu in enumerate(canon_ring):
        is_hp = (k == 0)
        ax.plot(mu[0], mu[1],
                '*' if is_hp else 'o',
                color='gold' if is_hp else 'k', mec='k',
                markersize=18 if is_hp else 9, zorder=8)
    if len(cx):
        ax.quiver(cx, cy, vx, vy,
                   angles='xy', scale_units='xy',
                   scale=1.0 / scale, color='C0',
                   width=0.01, alpha=0.95, zorder=10)
    ax.set_xlim(-4.7, 4.7); ax.set_ylim(-4.7, 4.7)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    # Annotate scale.
    ax.text(0.02, 0.02, f'×{scale:.0f}',
             ha='left', va='bottom', fontsize=10, color='0.4',
             transform=ax.transAxes)


def _per_roi_grid(pdf, df: pd.DataFrame, label: str,
                    shifts_df: pd.DataFrame | None = None,
                    agg: str = 'median',
                    scale: float | None = None):
    """Render ROI grid: each ROI gets a (predicted | observed) pair.
    8 ROIs × 2 panels = 16 panels in a 4×4 grid."""
    rois = [r for r in ROI_ORDER if r in df['roi'].unique()]
    n = len(rois)
    ring = get_ring_positions()

    # 4x4 grid: 4 rows × 4 columns. Each ROI takes 2 horizontally
    # adjacent cells (PRED, OBS).
    fig, axes = plt.subplots(4, 4, figsize=(20, 20))
    fig.suptitle(
        f'(d)  {label}  —  per-ROI predicted vs observed PRF-center shifts.\n'
        '   For each ROI: LEFT = model prediction, RIGHT = data.\n'
        '   Arrows: condition-specific shift   (HP=★ − mean across 4 conditions for PRED;\n'
        '            mean-across-subjects of per-subject median shift for OBS).\n'
        '   Circles: each AF (size = σ_AF, shading ∝ |gain|, red = suppression, blue = attraction).\n'
        '   Arrow ×N annotation in each panel = auto-scale '
        '(85th-pctile arrow ≈ 70% of a bin width, so PRED and OBS are comparable in shape).',
        fontsize=13, weight='bold',
    )
    for i, roi in enumerate(rois):
        # Layout: 4 ROIs per row, 2 panels each. So ROI i goes to:
        #   row = i // 2  (2 ROIs per row)
        #   col_base = (i % 2) * 2  (each ROI takes 2 cols)
        row = i // 2
        col_base = (i % 2) * 2
        ax_pred = axes[row, col_base]
        ax_obs  = axes[row, col_base + 1]

        sub = df[df['roi'] == roi]
        if len(sub) < 1:
            ax_pred.axis('off'); ax_obs.axis('off'); continue
        sigma_AF, g_HP, g_LP = _pick_params(df, roi, agg=agg)
        if not np.isfinite(sigma_AF) or not np.isfinite(g_HP):
            ax_pred.axis('off'); ax_obs.axis('off'); continue
        _vector_field_panel(ax_pred, sigma_AF, g_HP, g_LP, ring, scale=scale)
        ax_pred.set_title(
            f'{roi}  PRED   σ={sigma_AF:.2f}  '
            f'g_HP={g_HP:+.2f}  g_LP={g_LP:+.2f}',
            fontsize=11, weight='bold',
        )
        # Observed.
        if shifts_df is not None:
            _observed_field_panel(ax_obs,
                                     shifts_df[shifts_df['roi'] == roi],
                                     ring, scale=scale)
            ax_obs.set_title(f'{roi}  DATA   n_subj = '
                             f'{shifts_df[shifts_df["roi"]==roi]["subject"].nunique()}',
                              fontsize=11, weight='bold')
        else:
            ax_obs.axis('off')
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    pdf.savefig(fig); plt.close(fig)


# Helper: rotation_to_canonical (also used by plot_rotated_shift_heatmaps).
def rotation_to_canonical(condition: str):
    ring_idx = CONDITIONS.index(condition)
    ring = get_ring_positions()
    hp_actual = ring[ring_idx]
    target = np.array([0.0, RING_R])
    a_actual = np.arctan2(hp_actual[1], hp_actual[0])
    a_target = np.arctan2(target[1], target[0])
    theta = a_target - a_actual
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float32)


def slide_per_roi_vector_field(pdf, df: pd.DataFrame, shifts_df=None):
    """Group page: per-ROI PRED + OBS at empirical median params."""
    _per_roi_grid(pdf, df,
                    label='Group  (median across all 30 subjects)',
                    shifts_df=shifts_df,
                    agg='median')


def slide_per_subject_vector_fields(pdf, df: pd.DataFrame,
                                       subjects: list[int],
                                       shifts_df=None):
    """One page per well-fitting subject."""
    for sub_id in subjects:
        sub_df = df[df['subject'] == sub_id]
        if len(sub_df) == 0:
            continue
        sub_shifts = (shifts_df[shifts_df['subject'] == sub_id]
                      if shifts_df is not None else None)
        _per_roi_grid(pdf, sub_df,
                       label=f'Subject sub-{int(sub_id):02d}',
                       shifts_df=sub_shifts,
                       agg='median')


def slide_dynamic_intro(pdf):
    """Visual intro to the v2 dynamic model — schematic + formula."""
    fig = plt.figure(figsize=(13, 8))
    fig.text(0.5, 0.94,
              '(e)  Dynamic AF — adds a per-TR distractor pulse',
              ha='center', va='top', fontsize=24, weight='bold')

    # Left: schematic with sustained + dynamic.
    ax = fig.add_axes([0.04, 0.10, 0.50, 0.78])
    ax.set_xlim(-5.5, 5.5); ax.set_ylim(-5.5, 5.5)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    ax.spines[:].set_visible(False)
    ax.add_patch(Circle((0, 0), APERTURE, fill=False,
                          ec='0.4', ls='--', lw=1.2))

    ring = get_ring_positions()
    sigma_show = 1.4
    # Sustained AFs (background) + dynamic (foreground, smaller alpha-rim).
    for i, mu in enumerate(ring):
        is_hp = (i == 0)
        col = 'C3' if is_hp else 'C0'
        ax.add_patch(Circle(mu, sigma_show, alpha=0.20, color=col, zorder=2))
        ax.add_patch(Circle(mu, sigma_show, fill=False, ec=col,
                              lw=1.5, zorder=3))
        ax.plot(mu[0], mu[1],
                '*' if is_hp else 'o',
                color='gold' if is_hp else col, mec='k',
                markersize=22 if is_hp else 12, zorder=5)

    # Schematic of "distractor on screen at LP location" (a small pulse).
    lp_ll = ring[2]
    ax.plot(lp_ll[0], lp_ll[1], 'o', color='magenta',
            markersize=20, mec='k', alpha=0.9, zorder=8)
    ax.annotate('distractor\non screen now',
                 xy=lp_ll, xytext=(-4.5, -4.0),
                 fontsize=12, color='magenta', weight='bold',
                 ha='center',
                 arrowprops=dict(arrowstyle='->', color='magenta', lw=1.5))
    ax.text(0, 4.7,
             'sustained AF at all 4 ring locations\n'
             '+ TRANSIENT AF at the location currently\n'
             'showing a distractor (on/off per TR)',
             ha='center', fontsize=11, color='0.25')

    # Right: parameters.
    ax2 = fig.add_axes([0.58, 0.18, 0.40, 0.65])
    ax2.axis('off')
    ax2.text(0.5, 0.95, 'Five shared parameters per ROI',
              ha='center', va='top', fontsize=14, weight='bold',
              transform=ax2.transAxes)
    body = (
        '  σ_AF        AF size (shared sustained ↔ dynamic)\n\n'
        '  g_HP        sustained gain at HP\n'
        '  g_LP        sustained gain at LP\n\n'
        '  g_HP_dyn    transient gain when distractor at HP\n'
        '  g_LP_dyn    transient gain when distractor at LP\n'
        '\n'
        'M(g, t) = 1\n'
        '       + g_HP · A_HP_C(g)\n'
        '       + g_LP · Σ A_LP(g)\n'
        '       + g_HP_dyn · d_HP(t) · A_HP_C(g)\n'
        '       + g_LP_dyn · Σ d_LP(t) · A_LP(g)\n'
        '\n'
        'd_ℓ(t) ∈ [0, 1]:  fraction of TR with\n'
        'distractor on screen at ring location ℓ.\n'
    )
    ax2.text(0.04, 0.86, body, ha='left', va='top',
              fontsize=11, family='monospace',
              transform=ax2.transAxes)
    pdf.savefig(fig); plt.close(fig)


def slide_dynamic_raw_gains(pdf, v2_tsv: Path):
    """Per-ROI test: are g_HP_dyn AND g_LP_dyn ≠ 0 across subjects?

    This is the "is there transient attentional capture / suppression
    at all?" question — independent of whether HP and LP differ. Tests
    each gain against zero with a two-sided Wilcoxon, plus g_dyn_avg
    = ½(g_HP_dyn + g_LP_dyn).
    """
    if not v2_tsv.exists():
        return
    df = pd.read_csv(v2_tsv, sep='\t')
    df['g_dyn_avg'] = 0.5 * (df['g_HP_dyn'] + df['g_LP_dyn'])
    rois = [r for r in ROI_ORDER if r in df['roi'].unique()]
    YLIM = 1.5
    rng = np.random.default_rng(0)
    n_boot = 2000
    import seaborn as sns

    fig, axes = plt.subplots(1, 3, figsize=(20, 6.5))
    panels = [
        ('g_HP_dyn',  'g_HP_dyn  (transient gain at HP)'),
        ('g_LP_dyn',  'g_LP_dyn  (transient gain at LP)'),
        ('g_dyn_avg', '½(g_HP_dyn + g_LP_dyn)'),
    ]
    for ax, (metric, label) in zip(axes, panels):
        d_plot = df.assign(_clip=df[metric].clip(-YLIM, YLIM))
        sns.stripplot(data=d_plot, x='roi', y='_clip', ax=ax,
                       order=rois, hue='roi', palette='Set2', legend=False,
                       jitter=0.18, alpha=0.55, size=8)
        ax.axhline(0, color='gray', lw=0.7, ls='--')
        for i, roi in enumerate(rois):
            x = df.loc[df['roi'] == roi, metric].dropna().values
            if len(x) < 3:
                continue
            med = float(np.median(x))
            boots = rng.choice(x, size=(n_boot, len(x)), replace=True)
            boot_meds = np.median(boots, axis=1)
            lo, hi = np.quantile(boot_meds, [0.025, 0.975])
            ax.errorbar([i], [med], yerr=[[med - lo], [hi - med]],
                         fmt='s', color='k', markersize=12, ecolor='k',
                         elinewidth=2.5, capsize=8, zorder=10)
            try:
                _, p = stats.wilcoxon(x, alternative='two-sided',
                                        zero_method='pratt')
            except ValueError:
                p = np.nan
            sig = ('***' if p < 0.001 else '**' if p < 0.01
                    else '*' if p < 0.05 else 'n.s.')
            # Color: GREEN for sig + positive (capture), RED for sig + negative.
            if np.isfinite(p) and p < 0.05:
                col = 'C2' if med > 0 else 'C3'
            else:
                col = '0.4'
            direction = '+' if med > 0 else '−'
            ax.text(i, YLIM + 0.10,
                     f'{direction} p={p:.3f}\n{sig}',
                     ha='center', va='bottom', fontsize=11,
                     color=col,
                     weight='bold' if (np.isfinite(p) and p < 0.05) else 'normal')
        ax.set_ylim(-YLIM - 0.05, YLIM + 0.45)
        ax.set_xlabel('')
        ax.set_ylabel(label, fontsize=14)
        ax.set_title(label, fontsize=14, weight='bold')
        ax.tick_params(axis='x', labelsize=13)
        ax.tick_params(axis='y', labelsize=12)
        ax.grid(alpha=0.2)
    fig.suptitle(
        '(e)  Is there transient attention at all?  '
        'g_HP_dyn and g_LP_dyn vs zero per ROI.\n'
        'GREEN = significant CAPTURE (attraction).  '
        'RED = significant SUPPRESSION.',
        fontsize=15, weight='bold',
    )
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    pdf.savefig(fig); plt.close(fig)


def slide_dynamic_results(pdf, v2_tsv: Path):
    """Headline dynamic-model findings: 3-panel HP-vs-LP test
    (sustained / dynamic / total)."""
    if not v2_tsv.exists():
        # fallback: skip if no v2 fits.
        return
    df = pd.read_csv(v2_tsv, sep='\t')
    df['g_diff_total'] = df['g_diff_sus'] + df['g_diff_dyn']
    rois = [r for r in ROI_ORDER if r in df['roi'].unique()]
    YLIM = 1.5
    rng = np.random.default_rng(0)
    n_boot = 2000
    import seaborn as sns

    fig, axes = plt.subplots(1, 3, figsize=(20, 6.5))
    panels = [
        ('g_diff_sus',   'SUSTAINED:  g_HP − g_LP'),
        ('g_diff_dyn',   'DYNAMIC:    g_HP_dyn − g_LP_dyn'),
        ('g_diff_total', 'TOTAL:      sustained + dynamic'),
    ]
    for ax, (metric, label) in zip(axes, panels):
        d_plot = df.assign(_clip=df[metric].clip(-YLIM, YLIM))
        sns.stripplot(data=d_plot, x='roi', y='_clip', ax=ax,
                       order=rois, hue='roi', palette='Set2', legend=False,
                       jitter=0.18, alpha=0.55, size=8)
        ax.axhline(0, color='gray', lw=0.7, ls='--')
        for i, roi in enumerate(rois):
            x = df.loc[df['roi'] == roi, metric].dropna().values
            if len(x) < 3:
                continue
            med = float(np.median(x))
            boots = rng.choice(x, size=(n_boot, len(x)), replace=True)
            boot_meds = np.median(boots, axis=1)
            lo, hi = np.quantile(boot_meds, [0.025, 0.975])
            ax.errorbar([i], [med], yerr=[[med - lo], [hi - med]],
                         fmt='s', color='k', markersize=12, ecolor='k',
                         elinewidth=2.5, capsize=8, zorder=10)
            try:
                _, p = stats.wilcoxon(x, alternative='less', zero_method='pratt')
            except ValueError:
                p = np.nan
            sig = ('***' if p < 0.001 else '**' if p < 0.01
                    else '*' if p < 0.05 else 'n.s.')
            col = 'C3' if (np.isfinite(p) and p < 0.05) else '0.4'
            ax.text(i, YLIM + 0.10,
                     f'p={p:.3f}\n{sig}',
                     ha='center', va='bottom', fontsize=11,
                     color=col,
                     weight='bold' if p < 0.05 else 'normal')
        ax.set_ylim(-YLIM - 0.05, YLIM + 0.45)
        ax.set_xlabel('')
        ax.set_ylabel(label.split(':')[1].strip(), fontsize=14)
        ax.set_title(label, fontsize=14, weight='bold')
        ax.tick_params(axis='x', labelsize=13)
        ax.tick_params(axis='y', labelsize=12)
        ax.grid(alpha=0.2)
    fig.suptitle(
        '(e)  HP−LP differential: sustained, dynamic, and total per ROI.\n'
        'Wilcoxon one-sided  H₁: HP < LP. '
        'V2/V3 cancel — dynamic offsets sustained. V3AB/VO add.',
        fontsize=15, weight='bold',
    )
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    pdf.savefig(fig); plt.close(fig)


def slide_hierarchy(pdf, af_tsv: Path, shifts_tsv: Path):
    """Shifts grow up the visual hierarchy — clean 3-panel.

    Left:    |g_HP − g_LP|  per ROI in hierarchy order, mean ± SEM.
    Middle:  median |predicted shift|  per ROI (model-implied).
    Right:   median |observed shift|   per ROI (data, with Jensen caveat).
    Each panel annotated with overall Spearman r vs hierarchy index.
    """
    if not (af_tsv.exists() and shifts_tsv.exists()):
        return
    af = pd.read_csv(af_tsv, sep='\t')
    sh = pd.read_csv(shifts_tsv, sep='\t')
    inside = np.sqrt(sh['base_x'].values ** 2
                      + sh['base_y'].values ** 2) <= APERTURE
    sh = sh.loc[inside].copy()
    sh['shift_pred_mag'] = np.sqrt(
        (sh['pred_x'] - sh['base_x']) ** 2
        + (sh['pred_y'] - sh['base_y']) ** 2)
    sh['shift_obs_mag'] = np.sqrt(
        (sh['obs_x'] - sh['base_x']) ** 2
        + (sh['obs_y'] - sh['base_y']) ** 2)

    rois = [r for r in ROI_ORDER if r in af['roi'].unique()]
    af['hier_idx'] = af['roi'].map({r: i for i, r in enumerate(rois)})
    sh['hier_idx'] = sh['roi'].map({r: i for i, r in enumerate(rois)})

    # Per-ROI mean ± SEM of |g_diff| (model parameter).
    af_abs = af.assign(g_diff_abs=af['g_diff'].abs())
    g_means = af_abs.groupby('roi', observed=True).agg(
        m=('g_diff_abs', 'mean'),
        sem=('g_diff_abs',
              lambda v: v.std(ddof=1) / np.sqrt(max(len(v), 1))),
    ).loc[rois]

    # Per-ROI per-subject median |predicted shift| → mean ± SEM across subjects.
    pred_subj = (sh.groupby(['subject', 'roi'], observed=True)
                    ['shift_pred_mag'].median().reset_index())
    pred_summary = pred_subj.groupby('roi', observed=True).agg(
        m=('shift_pred_mag', 'mean'),
        sem=('shift_pred_mag',
              lambda v: v.std(ddof=1) / np.sqrt(max(len(v), 1))),
    ).loc[rois]
    obs_subj = (sh.groupby(['subject', 'roi'], observed=True)
                   ['shift_obs_mag'].median().reset_index())
    obs_summary = obs_subj.groupby('roi', observed=True).agg(
        m=('shift_obs_mag', 'mean'),
        sem=('shift_obs_mag',
              lambda v: v.std(ddof=1) / np.sqrt(max(len(v), 1))),
    ).loc[rois]

    # Spearman r vs hierarchy index using per-(subject, ROI) rows.
    af_abs2 = af_abs.dropna(subset=['hier_idx', 'g_diff_abs'])
    r_g, p_g = stats.spearmanr(af_abs2['hier_idx'], af_abs2['g_diff_abs'])
    pred_subj_h = pred_subj.dropna()
    pred_subj_h['hier_idx'] = pred_subj_h['roi'].map({r: i for i, r in enumerate(rois)})
    r_p, p_p = stats.spearmanr(pred_subj_h['hier_idx'], pred_subj_h['shift_pred_mag'])
    obs_subj_h = obs_subj.dropna()
    obs_subj_h['hier_idx'] = obs_subj_h['roi'].map({r: i for i, r in enumerate(rois)})
    r_o, p_o = stats.spearmanr(obs_subj_h['hier_idx'], obs_subj_h['shift_obs_mag'])

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    panels = [
        (g_means,    '|g_HP − g_LP|   (model param)',
         f'Spearman r = {r_g:+.2f},  p = {p_g:.2e}',
         '|g_HP − g_LP|'),
        (pred_summary, 'median |predicted shift|   (deg)',
         f'r = {r_p:+.2f},  p = {p_p:.2e}',
         '|predicted shift| (deg)'),
        (obs_summary, 'median |observed shift|   (deg)',
         f'r = {r_o:+.2f},  p = {p_o:.2e}',
         '|observed shift| (deg)'),
    ]
    pal = plt.get_cmap('viridis')(np.linspace(0.15, 0.85, len(rois)))
    for ax, (summary, title, sub_title, ylabel) in zip(axes, panels):
        x = np.arange(len(rois))
        ax.bar(x, summary['m'], yerr=summary['sem'],
                color=pal, edgecolor='k', linewidth=0.7,
                error_kw=dict(lw=1.5, capsize=5, color='k'))
        ax.set_xticks(x); ax.set_xticklabels(rois, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.set_title(f'{title}\n{sub_title}', fontsize=12, weight='bold')
        ax.grid(alpha=0.2, axis='y')
    fig.suptitle(
        '(d″)  Shifts grow up the visual hierarchy.\n'
        'V1 → V2 → V3 → V3AB → hV4 → LO → TO → VO  '
        '(mean across 30 subjects ± SEM).',
        fontsize=15, weight='bold',
    )
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    pdf.savefig(fig); plt.close(fig)


def slide_actual_vs_predicted(pdf, shifts_tsv: Path,
                                 normalize_pred: bool = True):
    """Per-ROI actual-vs-predicted shift comparison: projection on
    away-from-HP axis as a function of distance from HP.

    The qualitative-pattern-match test: do model and data peak at the
    same distance? Have the same decay shape? If yes, the model
    captures the spatial structure, even if magnitudes differ
    (Jensen + kernel mismatch inflate observed). Per-ROI Spearman r
    between bin-median predicted and bin-median observed annotated
    on each panel.

    `normalize_pred=True` rescales the predicted curve to have the
    same per-ROI peak amplitude as observed — turns the comparison
    into a pure SHAPE test.
    """
    if not shifts_tsv.exists():
        return
    df = pd.read_csv(shifts_tsv, sep='\t')
    # Aperture filter so we only test the visual-field region the
    # experiment probes.
    inside = np.sqrt(df['base_x'].values ** 2 + df['base_y'].values ** 2) <= APERTURE
    df = df.loc[inside].copy()
    # Distance from HP per row.
    ring = get_ring_positions()
    cond_to_idx = {c: i for i, c in enumerate(CONDITIONS)}
    hp_x = np.zeros(len(df), dtype=np.float32)
    hp_y = np.zeros(len(df), dtype=np.float32)
    for c in CONDITIONS:
        m = (df['condition'] == c).values
        if not m.any():
            continue
        hp_x[m] = ring[cond_to_idx[c], 0]
        hp_y[m] = ring[cond_to_idx[c], 1]
    df['dist_HP'] = np.sqrt((df['base_x'].values - hp_x) ** 2
                              + (df['base_y'].values - hp_y) ** 2)

    bin_edges = np.arange(0.0, 7.5, 0.5)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    df['d_bin'] = pd.cut(df['dist_HP'], bin_edges, labels=centers,
                          include_lowest=True)
    df = df.dropna(subset=['d_bin', 'proj_obs', 'proj_pred'])
    df['d_bin'] = df['d_bin'].astype(float)

    rois = [r for r in ROI_ORDER if r in df['roi'].unique()]
    n = len(rois)
    ncol = 4; nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3.0 * nrow),
                              sharex=True, sharey=True)
    axes = np.atleast_2d(axes)
    for i, roi in enumerate(rois):
        ax = axes[i // ncol, i % ncol]
        sub = df[df['roi'] == roi]
        if len(sub) == 0:
            ax.axis('off'); continue
        # Per-(subject, bin) means → per-bin median across subjects ± SEM.
        per_subj = (sub.groupby(['subject', 'd_bin'], observed=True)
                       [['proj_obs', 'proj_pred']]
                       .mean().reset_index())
        summary = (per_subj.groupby('d_bin', observed=True)
                       .agg(obs_med=('proj_obs', 'median'),
                            obs_sem=('proj_obs',
                                      lambda v: v.std(ddof=1)
                                                 / np.sqrt(max(len(v), 1))),
                            pred_med=('proj_pred', 'median'),
                            n=('proj_obs', 'size'))
                       .reset_index().sort_values('d_bin'))
        # Optionally rescale predicted curve to match observed peak —
        # makes the comparison a pure SHAPE test (magnitude differences
        # from Jensen & kernel mismatch are expected and not the point).
        pred_plot = summary['pred_med'].values.astype(float)
        if normalize_pred:
            obs_pk = float(np.nanmax(np.abs(summary['obs_med'])))
            pred_pk = float(np.nanmax(np.abs(pred_plot))) or 1e-9
            if pred_pk > 1e-9 and obs_pk > 1e-9:
                pred_plot = pred_plot * (obs_pk / pred_pk)
        ax.fill_between(summary['d_bin'],
                         summary['obs_med'] - summary['obs_sem'],
                         summary['obs_med'] + summary['obs_sem'],
                         color='C0', alpha=0.20)
        ax.plot(summary['d_bin'], summary['obs_med'],
                 color='C0', lw=2.5, marker='o', markersize=5,
                 label='observed (data)')
        ax.plot(summary['d_bin'], pred_plot,
                 color='C3', lw=2.5, ls='--', marker='s', markersize=5,
                 label='model prediction (rescaled to obs peak)'
                       if normalize_pred else 'model prediction')
        ax.axhline(0, color='gray', lw=0.5, ls=':')
        # Spearman r of the shapes (bin-median series).
        valid = (summary['obs_med'].notna()
                  & np.isfinite(pred_plot))
        if valid.sum() >= 4:
            r_sh, p_sh = stats.spearmanr(
                summary['obs_med'][valid].values,
                pred_plot[valid],
            )
        else:
            r_sh, p_sh = np.nan, np.nan
        sig = '*' if (np.isfinite(p_sh) and p_sh < 0.05) else ''
        col = 'C2' if (np.isfinite(p_sh) and p_sh < 0.05 and r_sh > 0) else '0.3'
        ax.set_title(
            f'{roi}   n_subj = {sub["subject"].nunique()}\n'
            f'shape r_s = {r_sh:+.2f}, p = {p_sh:.3f}{sig}',
            fontsize=12, weight='bold', color=col,
        )
        ax.grid(alpha=0.2)
        ax.tick_params(labelsize=11)
        if i == 0:
            ax.legend(fontsize=10, loc='upper right')
    for j in range(n, nrow * ncol):
        axes[j // ncol, j % ncol].axis('off')
    fig.suptitle(
        '(g)  Qualitative pattern test: model SHAPE vs data SHAPE.\n'
        'Per ROI: projection on away-from-HP axis vs distance from HP.\n'
        'Predicted curve rescaled to match observed peak — '
        'magnitude is confounded by Jensen & kernel mismatch, but the '
        'SHAPE (peak location, decay) is the test.',
        fontsize=13, weight='bold',
    )
    fig.text(0.5, 0.02,
              'distance of base PRF from HP (deg)',
              ha='center', fontsize=12)
    fig.text(0.005, 0.5,
              'projection on away-from-HP axis (deg)',
              va='center', rotation='vertical', fontsize=12)
    fig.tight_layout(rect=[0.02, 0.04, 1, 0.93])
    pdf.savefig(fig); plt.close(fig)


def slide_behavior(pdf, af_tsv: Path, rt_tsv: Path):
    """Behavioral correlations.

    Panel (left): per-ROI scatter of behavioral RT(HP)−RT(LP)
    against the model log-ratio (g_HP vs g_LP), with Spearman r.
    Panel (right): PCA across ROIs of g_HP (collapses 8 weak per-ROI
    tests into one omnibus subject score), correlated with the
    "any-distractor cost" RT(dist)−RT(no-dist).
    """
    if not (af_tsv.exists() and rt_tsv.exists()):
        return
    from sklearn.decomposition import PCA

    af = pd.read_csv(af_tsv, sep='\t')
    rt = pd.read_csv(rt_tsv, sep='\t')
    rt['subject'] = rt['subject'].astype(int)
    merged = af.merge(rt, on='subject', how='inner')

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))

    # ---- (left) per-ROI Spearman of log_ratio × RT_HP-LP, summary bar.
    ax = axes[0]
    rois = [r for r in ROI_ORDER if r in merged['roi'].unique()]
    rs, ps = [], []
    for roi in rois:
        sub = merged[merged['roi'] == roi]
        x = sub['RT_HP_minus_LP'].values
        y = sub['log_ratio'].values
        m = np.isfinite(x) & np.isfinite(y)
        r_s, p_s = stats.spearmanr(x[m], y[m]) if m.sum() >= 4 else (np.nan, np.nan)
        rs.append(r_s); ps.append(p_s)
    cols = ['C2' if (np.isfinite(p) and p < 0.05 and r > 0)
             else 'C3' if (np.isfinite(p) and p < 0.05 and r < 0)
             else '0.55' for r, p in zip(rs, ps)]
    ax.bar(np.arange(len(rois)), rs, color=cols, edgecolor='k', linewidth=0.7)
    ax.axhline(0, color='gray', lw=0.5)
    for i, (r, p) in enumerate(zip(rs, ps)):
        if np.isfinite(r):
            sig = '*' if (np.isfinite(p) and p < 0.05) else ''
            ax.text(i, r + (0.04 if r >= 0 else -0.04),
                     f'{r:+.2f}{sig}',
                     ha='center', va='bottom' if r >= 0 else 'top',
                     fontsize=10)
    ax.set_xticks(np.arange(len(rois)))
    ax.set_xticklabels(rois, fontsize=12)
    ax.set_ylim(-0.6, 0.6)
    ax.set_ylabel('Spearman r', fontsize=13)
    ax.set_title(
        'log( M_HP / M_LP )  ×  RT(HP) − RT(LP)\n'
        'per-ROI Spearman across subjects (no per-ROI test reaches p<0.05)',
        fontsize=11,
    )
    ax.grid(alpha=0.2, axis='y')

    # ---- (right) two-panel mini-grid: targeted test (null) AND
    # the test that DOES come out marginal.
    ax = axes[1]

    def _pca_corr(model_metric, behavior_metric):
        wide_ = (af.pivot_table(index='subject', columns='roi',
                                  values=model_metric, aggfunc='first')
                   [rois]).dropna()
        Z = wide_.rank(axis=0).values
        Z = (Z - Z.mean(axis=0)) / Z.std(axis=0, ddof=1)
        pca = PCA(n_components=1).fit(Z)
        PC1 = pca.transform(Z).ravel()
        merged_pc = pd.DataFrame({'subject': wide_.index, 'PC1': PC1}).merge(
            rt, on='subject', how='inner')
        x = merged_pc[behavior_metric].values
        y = merged_pc['PC1'].values
        m = np.isfinite(x) & np.isfinite(y)
        r, p = stats.spearmanr(x[m], y[m])
        return x[m], y[m], r, p, float(pca.explained_variance_ratio_[0])

    ax.set_xticks([]); ax.set_yticks([]); ax.spines[:].set_visible(False)
    inset_locs = [
        (0.08, 0.55, 0.42, 0.40,
         'log_ratio', 'RT_HP_minus_LP',
         'PC1 of log(M_HP/M_LP)',
         'RT(HP) − RT(LP)  (s)',
         'targeted HP-suppression test'),
        (0.55, 0.55, 0.42, 0.40,
         'g_HP', 'RT_HP_minus_no',
         'PC1 of g_HP',
         'RT(HP) − RT(no-dist)  (s)',
         'overall HP-AF gain × HP cost'),
    ]
    for x0, y0, w, h, mm, bm, ylab, xlab, title in inset_locs:
        ax_in = ax.inset_axes([x0, y0, w, h])
        x, y, r, p, var = _pca_corr(mm, bm)
        ax_in.scatter(x, y, s=40, alpha=0.85, color='C0',
                       edgecolor='k', linewidth=0.4)
        if len(x) >= 4:
            slope, intercept = np.polyfit(x, y, 1)
            xs = np.linspace(x.min(), x.max(), 50)
            ax_in.plot(xs, slope * xs + intercept, 'k--', lw=1.0)
        ax_in.axhline(0, color='gray', lw=0.3, ls=':')
        ax_in.axvline(0, color='gray', lw=0.3, ls=':')
        ax_in.set_xlabel(xlab, fontsize=10)
        ax_in.set_ylabel(ylab, fontsize=10)
        sig = '*' if p < 0.05 else ''
        ax_in.set_title(
            f'{title}\nr_s={r:+.2f}, p={p:.3f}{sig}',
            fontsize=10, weight='bold',
            color='C2' if p < 0.05 else 'k',
        )
        ax_in.tick_params(labelsize=9)
        ax_in.grid(alpha=0.2)
    # Caption below the two insets.
    ax.text(0.5, 0.40,
             'PCA across ROIs collapses 8 weak per-ROI signals into one\n'
             'subject score. The TARGETED test (HP-vs-LP × HP-vs-LP) is\n'
             'null with N=30. The omnibus g_HP × HP-cost test reaches *.',
             ha='center', va='top', fontsize=11, color='0.25',
             transform=ax.transAxes)

    fig.suptitle(
        '(f)  Correlations with behavior  '
        f'(N = {merged.subject.nunique()} subjects)',
        fontsize=15, weight='bold',
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    pdf.savefig(fig); plt.close(fig)


def slide_summary(pdf):
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis('off')
    ax.text(0.5, 0.95, 'Summary', ha='center', va='top',
             fontsize=24, weight='bold', transform=ax.transAxes)
    body = (
        '\n'
        '1.  Naive distance-from-HP analyses are biased by Jensen / noise.\n'
        '    The pattern, not the magnitude, is what tests the model.\n'
        '\n'
        '2.  Joint AF + PRF model (4 ring AFs, AF+ formulation, single Gaussian\n'
        '    voxel kernel) fits per (subject, ROI) at the BOLD level.\n'
        '\n'
        '3.  HP-specific suppression survives in early-to-ventral cortex:\n'
        '\n'
        '       V2     Wilcoxon p = 0.0002    80% of 30 subjects g_HP < g_LP\n'
        '       V3     Wilcoxon p = 0.0002    80%\n'
        '       V3AB   Wilcoxon p = 0.05\n'
        '       VO     Wilcoxon p = 0.003     80%\n'
        '\n'
        '4.  Per-ROI predicted shift vector fields show what each area\n'
        '    is doing — see (d) for the spatial pattern at each ROI\'s\n'
        '    empirical median parameters.\n'
        '\n'
        'Caveats:\n'
        '  • Model still has identifiability issues (σ_AF × |gain| trade-off\n'
        '    in mid-tier ROIs) but the rank-based HP-vs-LP test survives.\n'
        '  • PRED side uses single Gaussian voxel kernels; OBS side uses\n'
        '    DoG (model 4). Apples-to-apples Gaussian conditionwise refit\n'
        '    is in flight on cluster.\n'
    )
    ax.text(0.04, 0.85, body, ha='left', va='top', fontsize=12,
             family='monospace', transform=ax.transAxes)
    pdf.savefig(fig); plt.close(fig)


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--af-tsv', type=Path,
                        default=Path('notes/af_parameters.tsv'))
    parser.add_argument('--shifts-tsv', type=Path,
                        default=Path('notes/predict_shifts_cluster.tsv'),
                        help='For picking the top-3 well-fitting subjects.')
    parser.add_argument('--v2-tsv', type=Path,
                        default=Path('notes/af_v2_parameters.tsv'),
                        help='v2 dynamic-model parameters (optional).')
    parser.add_argument('--out', type=Path,
                        default=Path('notes/talk_docky_jan.pdf'))
    parser.add_argument('--skip-vector-fields', action='store_true',
                        help='Drop the per-ROI predicted+observed '
                             'vector-field grids (sections d and d′). '
                             'Use for the email-ready clean deck.')
    parser.add_argument('--example-subjects', nargs='+', type=int,
                        default=None,
                        help='Subject IDs for individual example pages. '
                             'Default: top-3 by predicted-vs-observed r.')
    args = parser.parse_args()

    df = pd.read_csv(args.af_tsv, sep='\t')
    df['roi'] = pd.Categorical(
        df['roi'],
        categories=[r for r in ROI_ORDER if r in df['roi'].unique()],
        ordered=True,
    )
    print(f'Loaded {len(df)} fits, {df.subject.nunique()} subjects.')

    # Pick top-3 well-fitting subjects by predicted-vs-observed r if available.
    if args.example_subjects is None and args.shifts_tsv.exists():
        s = pd.read_csv(args.shifts_tsv, sep='\t')
        rec = []
        for (sub, roi), g in s.groupby(['subject', 'roi']):
            m = g[['proj_pred', 'proj_obs']].dropna()
            if len(m) < 50:
                continue
            r = float(np.corrcoef(m['proj_pred'], m['proj_obs'])[0, 1])
            rec.append(dict(subject=int(sub), roi=roi, r=r))
        rdf = pd.DataFrame(rec)
        if len(rdf):
            score = (rdf.groupby('subject')['r'].mean()
                       .sort_values(ascending=False))
            args.example_subjects = score.head(3).index.tolist()
            print(f'Top-3 example subjects: {args.example_subjects}')
    if not args.example_subjects:
        args.example_subjects = []

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(args.out) as pdf:
        title_slide(pdf)

        section_divider(pdf, 'a',
                          'Distance binning is biased',
                          'Random noise on PRF position pushes distance-to-HP up.')
        slide_jensen(pdf)

        section_divider(pdf, 'b',
                          'The model',
                          '4 attention fields × Gaussian voxel kernel; '
                          '3 shared parameters per ROI per subject.')
        slide_model_formula(pdf)
        slide_model_panels(pdf)

        section_divider(pdf, 'c',
                          'Parameter estimates',
                          f'Per-ROI distributions and HP-vs-LP test '
                          f'(N = {df.subject.nunique()} subjects).')
        slide_parameter_distributions(pdf, df)
        slide_hp_lp_test(pdf, df)

        # Load shifts TSV once (apples-to-apples Gaussian preferred).
        shifts_df = None
        gauss_tsv = Path('notes/predict_shifts_gauss.tsv')
        load_tsv = (gauss_tsv if gauss_tsv.exists()
                     else args.shifts_tsv)
        if load_tsv.exists():
            shifts_df = pd.read_csv(load_tsv, sep='\t')
            print(f'  observed shifts: {len(shifts_df):,} rows from {load_tsv}')

        if not args.skip_vector_fields:
            section_divider(pdf, 'd',
                              'What the model predicts per ROI',
                              'Per-voxel mean across 4 conditions vs HP=★ '
                              'condition. Model on the left, data on the right.')
            slide_per_roi_vector_field(pdf, df, shifts_df=shifts_df)

        # Hierarchy slide.
        if args.shifts_tsv.exists():
            slide_hierarchy(pdf, args.af_tsv,
                              args.shifts_tsv if args.shifts_tsv.exists()
                              else Path('notes/predict_shifts_cluster.tsv'))

        if args.example_subjects and not args.skip_vector_fields:
            section_divider(
                pdf, 'd′',
                'Individual example subjects',
                f'sub-{int(args.example_subjects[0]):02d}, '
                f'sub-{int(args.example_subjects[1]):02d}, '
                f'sub-{int(args.example_subjects[2]):02d} '
                '(top-3 by predicted-vs-observed r).',
            )
            slide_per_subject_vector_fields(
                pdf, df, args.example_subjects,
                shifts_df=shifts_df,
            )

        section_divider(pdf, 'e',
                          'Dynamic AF — distractor pulses on top of the '
                          'sustained priority map',
                          '5 shared parameters per ROI; same per-voxel '
                          'Gaussian kernel.')
        slide_dynamic_intro(pdf)
        slide_dynamic_raw_gains(pdf, args.v2_tsv)
        slide_dynamic_results(pdf, args.v2_tsv)

        # Pick whichever predict_shifts TSV exists (apples-to-apples
        # Gaussian preferred if available).
        gauss_tsv = Path('notes/predict_shifts_gauss.tsv')
        shifts_tsv_for_actual = (gauss_tsv if gauss_tsv.exists()
                                  else args.shifts_tsv)
        if shifts_tsv_for_actual.exists():
            section_divider(
                pdf, 'g',
                'Observed vs predicted shifts',
                'Per-ROI projection-vs-distance from HP. '
                'Solid = data, dashed = model.',
            )
            slide_actual_vs_predicted(pdf, shifts_tsv_for_actual)

        # Behavior correlations.
        rt_tsv = Path('notes/rt_summary.tsv')
        if rt_tsv.exists():
            section_divider(
                pdf, 'f',
                'Correlations with behavior',
                'Subject-level co-variation between AF parameters and '
                'distractor RT cost.',
            )
            slide_behavior(pdf, args.af_tsv, rt_tsv)

        slide_summary(pdf)

    print(f'Wrote {args.out}')


if __name__ == '__main__':
    main()
