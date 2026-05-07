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
    """Cartoon of why distance-binning is biased."""
    fig = plt.figure(figsize=(11, 8.5))

    # Top: text panel.
    ax_text = fig.add_axes([0.05, 0.55, 0.9, 0.40])
    ax_text.axis('off')
    ax_text.text(0.5, 0.95, '(a)  The naive distance-from-HP analysis is biased',
                  ha='center', va='top', fontsize=18, weight='bold',
                  transform=ax_text.transAxes)
    body = (
        "We bin voxels by their estimated PRF distance to HP. But the\n"
        "BASE PRF position is itself a noisy estimate of the true PRF.\n"
        "\n"
        "  • Voxels that LAND in the small-distance bin are either:\n"
        "      (i) voxels with TRUE PRFs near HP, or\n"
        "      (ii) voxels with noise pushing the base estimate toward HP.\n"
        "\n"
        "  • For (ii), the conditionwise PRF (independent noise) regresses\n"
        "    back toward truth → looks like a shift AWAY from HP.\n"
        "\n"
        "  • Result: the empirical away-from-HP shift is INFLATED at small\n"
        "    distances even with NO real attentional effect.\n"
        "\n"
        "Implication: the test of the AF model isn't 'do empirical and\n"
        "predicted MAGNITUDES match' — that's confounded by Jensen.\n"
        "The test is 'does the predicted SPATIAL PATTERN (peak location,\n"
        "decay shape) recapitulate the empirical pattern?'"
    )
    ax_text.text(0.04, 0.85, body, ha='left', va='top', fontsize=12,
                  family='monospace', transform=ax_text.transAxes)

    # Bottom: cartoon — two voxels, one truly near HP, one noise-pushed near HP.
    ax_cart = fig.add_axes([0.10, 0.07, 0.80, 0.42])
    ax_cart.set_xlim(-1.0, 5.5); ax_cart.set_ylim(-2.0, 2.0)
    ax_cart.set_aspect('equal'); ax_cart.set_xticks([]); ax_cart.set_yticks([])
    ax_cart.axhline(0, color='k', lw=0.5)
    # HP star.
    ax_cart.plot(5.0, 0, '*', color='C3', markersize=22, mec='k', zorder=5)
    ax_cart.text(5.0, 0.5, 'HP', ha='center', fontsize=11, color='C3', weight='bold')
    # Truly-near voxel.
    ax_cart.plot(4.5, 0, 'o', color='C0', markersize=14, mec='k')
    ax_cart.text(4.5, -0.6, 'true PRF\n(actually near HP)',
                  ha='center', fontsize=9, color='C0')
    ax_cart.annotate('', xy=(4.0, 0), xytext=(4.5, 0),
                      arrowprops=dict(arrowstyle='->', color='C0', lw=1.5))
    # Noise-pushed voxel.
    ax_cart.plot(2.5, 0, 'o', color='gray', markersize=14, mec='k', alpha=0.5)
    ax_cart.plot(4.5, 1.0, 'o', color='C1', markersize=14, mec='k')
    ax_cart.annotate('', xy=(4.5, 1.0), xytext=(2.5, 0),
                      arrowprops=dict(arrowstyle='->', color='C1', lw=1.0,
                                       linestyle='dashed', alpha=0.6))
    ax_cart.text(2.5, -0.6, 'true PRF\n(actually FAR)',
                  ha='center', fontsize=9, color='gray')
    ax_cart.text(4.5, 1.5, 'BASE estimate\n(noise pushed it near HP)',
                  ha='center', fontsize=9, color='C1')
    # CW estimate of voxel #2 — back toward truth.
    ax_cart.annotate('', xy=(2.7, 0.1), xytext=(4.5, 1.0),
                      arrowprops=dict(arrowstyle='->', color='C2', lw=2,
                                       alpha=0.85))
    ax_cart.text(3.5, 1.4, 'cond-wise PRF (independent noise)\n'
                            'regresses back to truth = "shifts AWAY from HP"',
                  ha='center', fontsize=9, color='C2', style='italic')
    pdf.savefig(fig); plt.close(fig)


# --- (b) The model --------------------------------------------------------

def slide_model_formula(pdf):
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis('off')
    ax.text(0.5, 0.95, '(b)  The model',
             ha='center', va='top', fontsize=22, weight='bold',
             transform=ax.transAxes)

    formula = (
        '\n   Per voxel v, per HP condition C:\n\n'
        '       R_v_C(g)  =  M_C(g)  ·  S_v(g)\n\n'
        '   M_C(g)  =  1 + g_HP · A_{HP_C}(g)  +  g_LP · Σ_{ℓ ≠ HP_C} A_ℓ(g)\n\n'
        '   A_ℓ(g)  =  exp(−‖g − μ_ℓ‖² / (2 σ_AF²))     (peak amplitude 1)\n'
        '\n'
        '   S_v(g)  =  Gaussian PRF at  (x_v, y_v)  with width  sd_v\n'
    )
    ax.text(0.06, 0.78, formula, ha='left', va='top', fontsize=15,
             family='monospace', transform=ax.transAxes)

    note = (
        '\nThree shared parameters per ROI per subject:\n'
        '\n'
        '   • σ_AF  : spatial extent of each AF (single value, all 4 ring locations)\n'
        '   • g_HP  : modulation gain at the HP location  (negative = suppression)\n'
        '   • g_LP  : modulation gain at the 3 LP locations\n'
        '\n'
        'AF+ formulation (Sumiya 2026, Klein 2014): the "1 +" makes the\n'
        'effect LOCAL — voxels far from any AF feel no modulation, no\n'
        'far-field artifacts. The 4 ring positions are FIXED at 4° eccentricity.\n'
        '\n'
        'Quantity of interest:  g_HP − g_LP   (negative ⇒ HP-specific suppression).\n'
    )
    ax.text(0.06, 0.43, note, ha='left', va='top', fontsize=12,
             family='monospace', transform=ax.transAxes,
             color='0.25')
    pdf.savefig(fig); plt.close(fig)


def slide_model_panels(pdf, sigma_AF=2.0, g_HP=-0.3, g_LP=-0.1):
    """Three panels: 4 ring AFs / modulation field for one HP / shift vectors."""
    GX, GY, _ = make_grid()
    ring = get_ring_positions()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))

    # (1) Sum of 4 ring AFs.
    ax = axes[0]
    A_sum = np.stack([gaussian_2d(GX, GY, ring[i], sigma_AF)
                       for i in range(4)], axis=0).sum(axis=0)
    im = ax.imshow(A_sum, extent=(-5, 5, -5, 5), origin='lower',
                    cmap='Reds', vmin=0, vmax=1)
    add_aperture_and_ring(ax, ring)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title('Σ_ℓ A_ℓ(g)\n4 attention fields\n(σ_AF = 2°)',
                  fontsize=13)

    # (2) Modulation for one condition (HP=upper_right).
    ax = axes[1]
    M = modulation_field(GX, GY, ring, hp_idx=0,
                          sigma_AF=sigma_AF, g_HP=g_HP, g_LP=g_LP)
    d = max(abs(M.max() - 1.0), abs(1.0 - M.min()), 0.05)
    ax.imshow(M, extent=(-5, 5, -5, 5), origin='lower',
              cmap='RdBu_r', vmin=1 - d, vmax=1 + d)
    add_aperture_and_ring(ax, ring, hp_idx=0)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f'M_C(g)  for HP = ★\n'
                 f'g_HP = {g_HP:+.1f},  g_LP = {g_LP:+.1f}\n'
                 f'(blue = suppression)',
                 fontsize=13)

    # (3) Predicted shift vectors at HP=upper_right.
    ax = axes[2]
    s1d = np.arange(-3, 3.1, 0.6, dtype=np.float32)
    SX, SY = np.meshgrid(s1d, s1d)
    seeds = np.stack([SX.ravel(), SY.ravel()], axis=1).astype(np.float32)
    inside = np.linalg.norm(seeds, axis=1) <= APERTURE
    seeds = seeds[inside]
    pred = predict_shift_field(sigma_AF, g_HP, g_LP, seeds, prf_sd=1.0)
    dx = pred[0, :, 0] - seeds[:, 0]
    dy = pred[0, :, 1] - seeds[:, 1]
    add_aperture_and_ring(ax, ring, hp_idx=0)
    ax.quiver(seeds[:, 0], seeds[:, 1], dx, dy,
              angles='xy', scale_units='xy',
              scale=1.0 / 8.0, color='C0', width=0.007, alpha=0.9)
    ax.set_xlim(-4.5, 4.5); ax.set_ylim(-4.5, 4.5)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title('Predicted PRF-center shifts\n(arrows ×8 for visibility)',
                 fontsize=13)
    fig.suptitle('(b)  What the model predicts at typical parameters',
                  fontsize=16, weight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    pdf.savefig(fig); plt.close(fig)


# --- (c) Parameter estimates ---------------------------------------------

def slide_parameter_distributions(pdf, df: pd.DataFrame):
    """Three panels: σ_AF, g_HP, g_LP per ROI."""
    rois = [r for r in ROI_ORDER if r in df['roi'].unique()]
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    metrics = [
        ('sigma_AF', 'σ_AF (deg)',  0,    8.0,  False),
        ('g_HP',    'g_HP',         -1.5, +1.5, True),
        ('g_LP',    'g_LP',         -1.5, +1.5, True),
    ]
    import seaborn as sns
    for ax, (col, label, ylo, yhi, sym) in zip(axes, metrics):
        d_plot = df.assign(_clip=df[col].clip(ylo, yhi))
        sns.stripplot(data=d_plot, x='roi', y='_clip', ax=ax,
                       order=rois, hue='roi', palette='Set2', legend=False,
                       jitter=0.18, alpha=0.75, size=6)
        # Median markers.
        for i, roi in enumerate(rois):
            x = df.loc[df['roi'] == roi, col].dropna().values
            if len(x):
                ax.scatter([i], [np.median(x)], marker='_',
                            s=400, color='k', zorder=10, lw=2.5)
        if sym:
            ax.axhline(0, color='gray', lw=0.5, ls='--')
        ax.set_ylim(ylo, yhi)
        ax.set_xlabel('')
        ax.set_ylabel(label)
        ax.set_title(label, fontsize=15)
        ax.grid(alpha=0.2)
        for tick in ax.get_xticklabels():
            tick.set_rotation(20)
    fig.suptitle(
        '(c)  AF parameter estimates per ROI '
        f'({df.subject.nunique()} subjects, {len(df)} fits)\n'
        '— each dot = 1 subject — black bar = median',
        fontsize=16, weight='bold',
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    pdf.savefig(fig); plt.close(fig)


def slide_hp_lp_test(pdf, df: pd.DataFrame):
    """The headline result: HP-vs-LP differential per ROI with Wilcoxon p."""
    rois = [r for r in ROI_ORDER if r in df['roi'].unique()]
    fig, ax = plt.subplots(figsize=(13, 6.5))
    YLIM = 1.0
    rng = np.random.default_rng(0)
    n_boot = 2000
    import seaborn as sns
    d_plot = df.assign(_clip=df['g_diff'].clip(-YLIM, YLIM))
    sns.stripplot(data=d_plot, x='roi', y='_clip', ax=ax,
                   order=rois, hue='roi', palette='Set2', legend=False,
                   jitter=0.18, alpha=0.7, size=7)
    ax.axhline(0, color='gray', lw=0.7, ls='--')
    for i, roi in enumerate(rois):
        x = df.loc[df['roi'] == roi, 'g_diff'].dropna().values
        if len(x) < 3:
            continue
        med = float(np.median(x))
        boots = rng.choice(x, size=(n_boot, len(x)), replace=True)
        boot_meds = np.median(boots, axis=1)
        lo, hi = np.quantile(boot_meds, [0.025, 0.975])
        ax.errorbar([i], [med], yerr=[[med - lo], [hi - med]],
                     fmt='s', color='k', markersize=12,
                     ecolor='k', elinewidth=2.5, capsize=8, zorder=10)
        try:
            _, p = stats.wilcoxon(x, alternative='less', zero_method='pratt')
        except ValueError:
            p = np.nan
        sig = ('***' if p < 0.001 else '**' if p < 0.01
                else '*' if p < 0.05 else 'n.s.')
        col = 'C3' if (np.isfinite(p) and p < 0.05) else '0.4'
        ax.text(i, YLIM + 0.07,
                 f'p = {p:.3f}\n{sig}',
                 ha='center', va='bottom', fontsize=11, color=col,
                 weight='bold' if p < 0.05 else 'normal')
        n_lo = int((x < -YLIM).sum())
        n_hi = int((x > +YLIM).sum())
        if n_lo:
            ax.annotate(f'↓{n_lo}', xy=(i, -YLIM * 0.97),
                         ha='center', va='top', fontsize=10, color='C3')
        if n_hi:
            ax.annotate(f'↑{n_hi}', xy=(i, +YLIM * 0.97),
                         ha='center', va='bottom', fontsize=10, color='C3')
    ax.set_ylim(-YLIM - 0.05, YLIM + 0.40)
    ax.set_xlabel('')
    ax.set_ylabel('g_HP − g_LP\n(negative ⇒ HP suppressed more than LP)',
                   fontsize=14)
    ax.set_title(
        '(c)  HP-suppression test per ROI\n'
        'Wilcoxon one-sided  H₁: g_HP < g_LP   '
        f'(N = {df.subject.nunique()} subjects)',
        fontsize=15, weight='bold',
    )
    ax.grid(alpha=0.2)
    pdf.savefig(fig); plt.close(fig)


# --- (d) Per-ROI predictions -----------------------------------------------

def slide_per_roi_vector_field(pdf, df: pd.DataFrame, prf_sd=1.0,
                                  spacing=0.6, scale=10.0):
    """Per-ROI shift-prediction vector field at empirical median params."""
    rois = [r for r in ROI_ORDER if r in df['roi'].unique()]
    s1d = np.arange(-3, 3.1, spacing, dtype=np.float32)
    SX, SY = np.meshgrid(s1d, s1d)
    seeds = np.stack([SX.ravel(), SY.ravel()], axis=1).astype(np.float32)
    inside = np.linalg.norm(seeds, axis=1) <= APERTURE
    seeds = seeds[inside]
    ring = get_ring_positions()

    n = len(rois)
    ncol = 4
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 4 * nrow))
    axes = np.atleast_2d(axes)
    fig.suptitle(
        '(d)  Per-ROI predicted PRF-center shifts at empirical '
        'MEDIAN parameters\n'
        f'(arrows ×{int(scale)}, HP = upper-right)',
        fontsize=16, weight='bold',
    )
    for i, roi in enumerate(rois):
        ax = axes[i // ncol, i % ncol]
        sub = df[df['roi'] == roi]
        sigma_AF = float(np.median(sub['sigma_AF']))
        g_HP = float(np.median(sub['g_HP']))
        g_LP = float(np.median(sub['g_LP']))
        pred = predict_shift_field(sigma_AF, g_HP, g_LP, seeds, prf_sd=prf_sd)
        dx = pred[0, :, 0] - seeds[:, 0]
        dy = pred[0, :, 1] - seeds[:, 1]
        add_aperture_and_ring(ax, ring, hp_idx=0)
        ax.quiver(seeds[:, 0], seeds[:, 1], dx, dy,
                   angles='xy', scale_units='xy',
                   scale=1.0 / scale, color='C0',
                   width=0.008, alpha=0.9)
        ax.set_xlim(-4.5, 4.5); ax.set_ylim(-4.5, 4.5)
        ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(
            f'{roi}  (n={len(sub)})\n'
            f'σ={sigma_AF:.2f},  '
            f'g_HP={g_HP:+.2f},  g_LP={g_LP:+.2f}',
            fontsize=11,
        )
    for j in range(n, nrow * ncol):
        axes[j // ncol, j % ncol].axis('off')
    fig.tight_layout(rect=[0, 0, 1, 0.94])
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
    parser.add_argument('--out', type=Path,
                        default=Path('notes/talk_docky_jan.pdf'))
    args = parser.parse_args()

    df = pd.read_csv(args.af_tsv, sep='\t')
    df['roi'] = pd.Categorical(
        df['roi'],
        categories=[r for r in ROI_ORDER if r in df['roi'].unique()],
        ordered=True,
    )
    print(f'Loaded {len(df)} fits, {df.subject.nunique()} subjects.')

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(args.out) as pdf:
        title_slide(pdf)

        section_divider(pdf, 'a',
                          'Distance binning is biased',
                          "Jensen's inequality + measurement noise inflate the empirical curve.")
        slide_jensen(pdf)

        section_divider(pdf, 'b',
                          'The model',
                          '4 attention fields × Gaussian voxel kernel; '
                          '3 shared parameters per ROI per subject.')
        slide_model_formula(pdf)
        slide_model_panels(pdf)

        section_divider(pdf, 'c',
                          'Parameter estimates',
                          'Per-ROI distributions and HP-vs-LP test '
                          f'(N = {df.subject.nunique()} subjects).')
        slide_parameter_distributions(pdf, df)
        slide_hp_lp_test(pdf, df)

        section_divider(pdf, 'd',
                          'What the model predicts per ROI',
                          'Vector field of PRF-center shifts at each ROI\'s '
                          'EMPIRICAL MEDIAN parameters.')
        slide_per_roi_vector_field(pdf, df)

        slide_summary(pdf)

    print(f'Wrote {args.out}')


if __name__ == '__main__':
    main()
