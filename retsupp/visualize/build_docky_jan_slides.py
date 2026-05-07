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
                       cmap='RdBu_r', vmin=1 - vmax, vmax=1 + vmax)
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
        'Top: modulation field M(g)  (red = attraction,  blue = suppression).   '
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


def _vector_field_panel(ax, sigma_AF, g_HP, g_LP, ring, prf_sd=1.0,
                         spacing=0.6, scale=10.0):
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


def _per_roi_grid(pdf, df: pd.DataFrame, label: str,
                    agg: str = 'median',
                    scale: float = 10.0):
    """Render an 8-ROI vector-field grid for the given (sub-)dataframe."""
    rois = [r for r in ROI_ORDER if r in df['roi'].unique()]
    n = len(rois); ncol = 4; nrow = int(np.ceil(n / ncol))
    ring = get_ring_positions()
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 4 * nrow))
    axes = np.atleast_2d(axes)
    fig.suptitle(
        f'(d)  {label}  —  per-ROI predicted PRF-center shifts.\n'
        '   Arrows: condition-specific shift = '
        '(predicted at HP=★) − (average across all 4 conditions).\n'
        '   Circles: each AF (size = σ_AF, '
        'shading ∝ |gain|, red = suppression, blue = attraction). '
        f'arrows ×{int(scale)}.',
        fontsize=14, weight='bold',
    )
    for i, roi in enumerate(rois):
        ax = axes[i // ncol, i % ncol]
        sub = df[df['roi'] == roi]
        if len(sub) < 1:
            ax.axis('off'); continue
        sigma_AF, g_HP, g_LP = _pick_params(df, roi, agg=agg)
        if not np.isfinite(sigma_AF) or not np.isfinite(g_HP):
            ax.axis('off'); continue
        _vector_field_panel(ax, sigma_AF, g_HP, g_LP, ring, scale=scale)
        ax.set_title(
            f'{roi}   n = {len(sub)}\n'
            f'σ={sigma_AF:.2f},  g_HP={g_HP:+.2f},  g_LP={g_LP:+.2f}',
            fontsize=12,
        )
    for j in range(n, nrow * ncol):
        axes[j // ncol, j % ncol].axis('off')
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    pdf.savefig(fig); plt.close(fig)


def slide_per_roi_vector_field(pdf, df: pd.DataFrame):
    """Group page (8 ROIs) at empirical median params."""
    _per_roi_grid(pdf, df, label='Group  (median across all 30 subjects)',
                    agg='median')


def slide_per_subject_vector_fields(pdf, df: pd.DataFrame, subjects: list[int]):
    """One page per well-fitting subject."""
    for sub_id in subjects:
        sub_df = df[df['subject'] == sub_id]
        if len(sub_df) == 0:
            continue
        _per_roi_grid(pdf, sub_df,
                       label=f'Subject sub-{int(sub_id):02d}  '
                              '(individual fits, not average)',
                       agg='median', scale=10.0)


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

        section_divider(pdf, 'd',
                          'What the model predicts per ROI',
                          'Condition-specific PRF-center shift = '
                          '(predicted at HP) − (mean across all 4 conditions).')
        slide_per_roi_vector_field(pdf, df)

        if args.example_subjects:
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
            )

        section_divider(pdf, 'e',
                          'Dynamic AF — distractor pulses on top of the '
                          'sustained priority map',
                          '5 shared parameters per ROI; same per-voxel '
                          'Gaussian kernel.')
        slide_dynamic_intro(pdf)
        slide_dynamic_raw_gains(pdf, args.v2_tsv)
        slide_dynamic_results(pdf, args.v2_tsv)

        slide_summary(pdf)

    print(f'Wrote {args.out}')


if __name__ == '__main__':
    main()
