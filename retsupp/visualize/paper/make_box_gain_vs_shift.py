"""Outline + draft figure for an empirical-paper Box on:

  "Attention as gain modulation, not receptive-field translation."

Idea:
- The dominant framework (post-hoc PRF refits) reports position shifts.
- These shifts are mathematically equivalent (under symmetric Gaussian
  PRFs) to spatially structured amplitude modulation.
- Most prior work doesn't directly fit the modulation; it fits PRFs
  per condition and infers shift.
- We instead fit the modulation directly at the BOLD level. This:
  - Allows DoG / asymmetric voxel kernels.
  - Naturally handles negative gains (suppression).
  - Handles "1+" baseline → far-from-AF voxels are unaffected.
  - Surfaces a per-ROI gain HIERARCHY that pure-shift analyses obscure.

Output: notes/figures/box_gain_vs_shift.pdf
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Circle
from scipy import stats

REPO = Path(__file__).resolve().parents[2]
TSV = REPO / 'notes' / 'data' / 'af_dog_v3_target_sharedSigma_parameters.tsv'
OUT = REPO / 'notes' / 'figures' / 'box_gain_vs_shift.pdf'

ROI_ORDER = ['V1', 'V2', 'V3', 'V3AB', 'hV4', 'LO', 'TO', 'VO']

plt.rcParams.update({'font.size': 10, 'axes.titlesize': 11})


def schematic_panel(ax, prf_xy=(2.5, 1.0), sigma_prf=0.8,
                    af_xy=(2.83, 2.83), sigma_af=2.0, gain=-1.5):
    """Panel A: schematic showing PRF × M(g) yields an asymmetric
    'effective response profile' whose center of mass is shifted —
    but the underlying PRF didn't move."""
    res = 200
    g1d = np.linspace(-5, 5, res)
    GX, GY = np.meshgrid(g1d, g1d)
    prf = np.exp(-((GX - prf_xy[0])**2 + (GY - prf_xy[1])**2)
                 / (2 * sigma_prf**2))
    A = np.exp(-((GX - af_xy[0])**2 + (GY - af_xy[1])**2)
               / (2 * sigma_af**2))
    M = np.maximum(1.0 + gain * A, 0.0)
    eff = prf * M
    ax.imshow(eff, extent=[-5, 5, -5, 5], origin='lower',
              cmap='magma', aspect='equal')
    # Original PRF center marker
    ax.scatter(*prf_xy, s=200, marker='+', color='cyan', lw=2.5,
               label='original PRF μ')
    # COM of PRF × M (apparent shifted center after refit)
    com_x = (GX * eff).sum() / eff.sum()
    com_y = (GY * eff).sum() / eff.sum()
    ax.scatter(com_x, com_y, s=200, marker='*', color='yellow',
               edgecolor='black', lw=1.0, label='COM(PRF × M)\n= apparent center')
    # AF center (gold star)
    ax.scatter(*af_xy, s=300, marker='o', facecolor='none',
               edgecolor='gold', lw=2.5, label='AF location')
    # arrow original → apparent
    ax.annotate('', xy=(com_x, com_y), xytext=prf_xy,
                arrowprops=dict(arrowstyle='->', color='cyan', lw=2))
    ax.set_xlim(-5, 5); ax.set_ylim(-5, 5)
    ax.set_xlabel('x (deg)'); ax.set_ylabel('y (deg)')
    ax.set_title(f'Effective response profile = PRF · M(g)\n'
                  f'AF gain = {gain:+.1f}, σ_AF = {sigma_af}°',
                  fontsize=10)
    ax.legend(loc='lower left', fontsize=8, framealpha=0.85)


def gain_hierarchy_panel(ax, df, col, ylabel, title, ylim,
                          alternative='two-sided'):
    rois = [r for r in ROI_ORDER if r in df['roi'].unique()]
    for i, roi in enumerate(rois):
        sub = df[df['roi'] == roi]
        if len(sub) < 2: continue
        v = np.clip(sub[col].values, *ylim)
        ax.scatter(np.full(len(v), i), v, color='C0', alpha=0.5, s=22,
                   edgecolor='k', linewidth=0.3)
        med = np.median(v)
        ax.scatter([i], [med], marker='_', s=350, color='k', lw=2.5,
                   zorder=10)
        try:
            _, p = stats.wilcoxon(sub[col], alternative=alternative,
                                  zero_method='pratt')
        except ValueError:
            p = float('nan')
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        c = ('C2' if (np.isfinite(p) and p < 0.05 and med < 0)
             else 'C3' if (np.isfinite(p) and p < 0.05 and med > 0)
             else '0.5')
        ax.text(i, ylim[1] + 0.04 * (ylim[1] - ylim[0]),
                f'{med:+.2f}\n{sig}',
                ha='center', va='bottom', fontsize=8, color=c,
                weight='bold' if p < 0.05 else 'normal')
    ax.set_xticks(range(len(rois)))
    ax.set_xticklabels(rois, fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_ylim(ylim[0], ylim[1] + 0.25 * (ylim[1] - ylim[0]))
    ax.axhline(0, color='gray', lw=0.7, ls='--')
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.2, axis='y')


def main():
    df = pd.read_csv(TSV, sep='\t')
    df['g_diff_sus'] = df['g_HP'] - df['g_LP']
    n_sub = df['subject'].nunique()
    print(f'n subjects: {n_sub}, n ROIs: {df.roi.nunique()}')

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(OUT) as pdf:
        # ---- Page 1: TEXT — outline of the Box ----------------------
        fig = plt.figure(figsize=(8.5, 11))
        ax = fig.add_subplot(111)
        ax.axis('off')
        text = """Box [draft]   Direct estimation of attention-field gains beyond
                receptive-field translation

Background.  Most prior work on attentional modulation of population
receptive fields (PRFs) (Womelsdorf et al. 2008; Klein et al. 2014;
Anstis et al. 2017; Sumiya 2026) infers attention's effect by fitting
symmetric Gaussian PRFs separately under different attentional
conditions and reporting position shifts (and sometimes size changes).
This implicit assumption — that attention literally translates
receptive fields toward attended locations — has been the dominant
mechanistic interpretation.

A growing literature notes that an alternative reading exists: the
same observed shift can arise from spatially structured amplitude
modulation of the underlying response, with no actual translation of
the PRF. Sumiya's chapter 2 makes this concrete by deriving an "AF+"
model where an additive offset (R = [a · A + b] · S) makes attention
LOCAL — only PRFs whose tail overlaps the attended location are
affected — in contrast to the bare Gaussian-product framework where
every PRF is pulled toward attention.

What we add.  We fit the modulation field directly to BOLD time series
rather than infer it post-hoc from per-condition PRF refits. The
fitted parameters are the gains and spatial extents of an attention
field with biologically motivated structure (sustained per-location
gains; transient distractor-onset and target-onset gains; shared σ
across phasic terms).  This:

  1.  Avoids mis-fitting symmetric Gaussians to asymmetric modulated
      responses — a step that contaminates the recovered PRF center
      with the modulation's spatial structure.
  2.  Naturally accommodates difference-of-Gaussians voxel kernels
      and surround structure in early visual cortex.
  3.  Naturally accommodates suppression (negative gains).
  4.  Preserves a "1+" baseline that makes the modulation local, like
      Sumiya's AF+, instead of attracting all PRFs to the attended
      location.

Empirically, this surfaces a HIERARCHY of attention-field gain
parameters across the visual stream that pure-shift analyses do not.
For instance (next page): target-onset capture (g_T_dyn) is null in
V1/V2/V3 but rises monotonically through V3 → hV4 → V3AB → TO → VO.
Sustained HP-vs-LP suppression survives in V2/V3/V3AB/hV4/TO/VO
(6 of 8 ROIs).

The model-comparison agenda.  Since the Sumiya-shift and gain-field
formulations are observationally close in many cases (and identical in
the limit of symmetric Gaussian PRFs and zero-offset modulation), the
practical question is which generalizes better.  We perform a head-to-
head Drive (gain) vs Analytical-Sumiya vs Numerical AF+ comparison on
held-out CV-R²; the result is reported in the main text.

Why this matters.  The "amplitudes change" view is methodologically
straightforward — fit gains, get an interpretable hierarchy.  But it
has been overshadowed by the "PRFs translate" framing because
amplitude information is conventionally normalized away in PRF
analysis.  Direct gain-field fitting preserves this information and
makes attention's spatial structure quantitatively comparable across
ROIs and across subject populations."""
        ax.text(0.05, 0.97, text, transform=ax.transAxes,
                 fontsize=9, va='top', ha='left',
                 family='serif', wrap=True)
        pdf.savefig(fig); plt.close(fig)

        # ---- Page 2: schematic ---------------------------------------
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        # Left: PRF × M with extreme negative gain (HP suppression)
        schematic_panel(axes[0], prf_xy=(2.0, 1.5), sigma_prf=0.8,
                         af_xy=(2.83, 2.83), sigma_af=2.0, gain=-1.5)
        axes[0].set_title('A.  Suppression (g < 0):\n'
                           'PRF stays put, but COM(PRF·M) shifts AWAY\n'
                           'from the attention focus.', fontsize=10)
        # Right: PRF × M with positive gain (target capture)
        schematic_panel(axes[1], prf_xy=(2.0, 1.5), sigma_prf=0.8,
                         af_xy=(2.83, 2.83), sigma_af=2.0, gain=+2.5)
        axes[1].set_title('B.  Capture (g > 0):\n'
                           'PRF stays put, but COM(PRF·M) shifts TOWARD\n'
                           'the attention focus.', fontsize=10)
        fig.suptitle('Equivalence of gain-modulation and apparent PRF shift\n'
                      'PRF center is fixed; only its EFFECTIVE response profile changes.\n'
                      'Refitting a symmetric Gaussian to that profile yields an apparent shift.',
                      fontsize=11, weight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        pdf.savefig(fig); plt.close(fig)

        # ---- Page 3: gain hierarchy (n=29) --------------------------
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))

        gain_hierarchy_panel(
            axes[0, 0], df, 'g_T_dyn',
            ylabel='g_T_dyn',
            title='Target-onset gain (signed): hierarchical capture\n'
                  '(null in V1-V3; monotone through V3 → V3AB → VO)',
            ylim=(-3, 8),
            alternative='two-sided')

        gain_hierarchy_panel(
            axes[0, 1], df, 'g_diff_sus',
            ylabel='g_HP − g_LP (sustained)',
            title='Sustained HP-suppression vs LP\n'
                  '(< 0  ⇒  HP-priority-map suppression\n'
                  ' robust in V2, V3, V3AB, hV4, TO, VO)',
            ylim=(-3, 3),
            alternative='less')

        gain_hierarchy_panel(
            axes[1, 0], df, 'g_HP_dyn',
            ylabel='g_HP_dyn',
            title='Phasic HP-distractor gain (signed)',
            ylim=(-3, 4),
            alternative='two-sided')

        gain_hierarchy_panel(
            axes[1, 1], df, 'g_LP_dyn',
            ylabel='g_LP_dyn',
            title='Phasic LP-distractor gain (signed)',
            ylim=(-3, 4),
            alternative='two-sided')

        fig.suptitle(
            f'C.  Per-ROI attention-field gain hierarchy '
            f'(DoG-dyn-v3 + target, n={n_sub})\n'
            'These multi-parameter gain fingerprints are not '
            'recoverable from pure-shift analyses.',
            fontsize=12, weight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        pdf.savefig(fig); plt.close(fig)

        # ---- Page 3b: empirical comparison ----------------------------
        # Per-ROI Pearson r values from the comparison analysis (n=29
        # subjects, observed conditionwise PRF shifts vs Sumiya analytical
        # vs amplitude-COM predictions).
        comparison_rs = {
            'V1':   (0.31, 0.52),
            'V2':   (0.03, 0.47),
            'V3':   (0.07, 0.49),
            'V3AB': (0.05, 0.52),
            'hV4':  (0.02, 0.43),
            'LO':   (0.00, 0.62),
            'TO':   (0.00, 0.57),
            'VO':   (0.04, 0.47),
        }
        rois = list(comparison_rs.keys())
        sumiya_r = [comparison_rs[r][0] for r in rois]
        ampcom_r = [comparison_rs[r][1] for r in rois]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5),
                                   gridspec_kw={'width_ratios': [1.2, 1]})
        # bar plot of r values
        ax_bar = axes[0]
        x = np.arange(len(rois))
        w = 0.35
        ax_bar.bar(x - w/2, sumiya_r, width=w, color='steelblue',
                    label='Sumiya analytical')
        ax_bar.bar(x + w/2, ampcom_r, width=w, color='C3',
                    label='Amplitude-COM (our framework)')
        ax_bar.set_xticks(x); ax_bar.set_xticklabels(rois)
        ax_bar.set_ylabel('Pearson r ( predicted vs observed |shift| )')
        ax_bar.set_ylim(-0.05, 0.7)
        ax_bar.axhline(0, color='gray', lw=0.5)
        ax_bar.set_title(
            'D.  Per-ROI prediction quality\n'
            'Amplitude-COM matches/exceeds Sumiya in every ROI;\n'
            'Sumiya collapses to r ≈ 0 in 6/8 ROIs.')
        ax_bar.legend(loc='upper right', fontsize=9)
        ax_bar.grid(alpha=0.2, axis='y')

        # narrative panel
        ax_text = axes[1]
        ax_text.axis('off')
        ax_text.text(0, 0.99,
                      'WHY SUMIYA BREAKS:\n\n'
                      'Sumiya analytical prediction =\n'
                      '   (Σ precisions × locations) / (Σ precisions)\n'
                      'where precision_ℓ = g_ℓ / σ_AF².\n\n'
                      'When g < 0 (suppression), the denominator\n'
                      'becomes small or negative — the formula\n'
                      'spits out unbounded shifts (20° to 100°).\n\n'
                      'In V3AB sub-28 (g_HP ≈ −2), Sumiya\n'
                      'predicts shifts that run off the screen\n'
                      'entirely. Amplitude-COM stays bounded.\n\n'
                      'IMPLICATION:\n'
                      'In the suppression regime that dominates\n'
                      'our retsupp data, Sumiya analytical is\n'
                      'unusable. The amplitude-modulation framework\n'
                      'with the "1+" baseline (Sumiya AF+ in spirit,\n'
                      'numerical AF+ in practice) is the only one\n'
                      'that recovers observed shift structure.\n\n'
                      'Per-ROI r values to the right.\n'
                      'Full empirical comparison:\n'
                      'notes/figures/predicted_vs_observed_shifts.pdf',
                      transform=ax_text.transAxes,
                      fontsize=10, va='top', ha='left',
                      family='monospace')

        fig.suptitle(
            'Empirical comparison — observed conditionwise shifts vs predictions  '
            '(n=29, voxels with mean-fit r²>0.10)',
            fontsize=11, weight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        pdf.savefig(fig); plt.close(fig)

        # ---- Page 4: Box paragraph drafts ----------------------------
        fig = plt.figure(figsize=(8.5, 11))
        ax = fig.add_subplot(111)
        ax.axis('off')
        text2 = """Box [draft]   Suggested text and figure plan

PARAGRAPH 1  (Background, ~120 words)
  Standard PRF/attention work fits Gaussian PRFs per condition and
  reports position shifts (cite Klein 2014; Womelsdorf 2008; Sumiya
  2026).  Implicit interpretation: attention TRANSLATES receptive
  fields.  But the same observable can be produced by spatially
  structured amplitude modulation.  Sumiya's chapter 2 distinguishes
  AF+ (additive offset, local effect) from pure AF (no offset,
  universal attraction) and shows the data favor AF+.

PARAGRAPH 2  (Our approach, ~120 words)
  We fit the modulation field DIRECTLY to BOLD: per-location gains
  (g_HP, g_LP) modulate response amplitude at each visual-field
  location, while the PRF center stays fixed.  Apparent PRF shifts
  emerge from refitting symmetric Gaussians to the modulated
  responses, but our model does not translate PRFs.  This formulation
  generalizes Sumiya AF+ to (i) DoG voxel kernels, (ii) suppression
  via negative gains, and (iii) per-event-type modulation
  (sustained/dynamic-distractor/target).

PARAGRAPH 3  (Result, ~140 words)
  In 30 subjects (DoG-dyn-v3 + target shared-σ, n=29 with completed
  fits) we find: a target-capture hierarchy that's null in V1/V2/V3
  and monotonically increasing through V3 → V3AB → TO → VO; sustained
  HP-suppression in 6 of 8 ROIs; and phasic distractor terms that
  largely absorb into the target term.  These per-ROI gain
  fingerprints are not recoverable from a pure-shift framework.

  A head-to-head CV-R² comparison of three models on the same
  Gaussian-PRF backbone — DRIVE (our gain-field) vs ANALYTICAL-SUMIYA
  (no offset) vs NUMERICAL AF+ — shows that DRIVE/AF+ outperform
  ANALYTICAL-SUMIYA in nearly all ROIs (REPORT EXACT NUMBERS WHEN
  CV LANDS).  Drive and AF+ are observationally similar; we report
  Drive in the main text for interpretability.

FIGURE PLAN
  Box-Fig.  3 panels.  (A) schematic: gain × PRF → apparent shift.
                        (B) per-ROI g_T_dyn hierarchy.
                        (C) sustained HP-vs-LP differential.
  Caption emphasizes equivalence + extra information (gain magnitudes,
  signs, hierarchies) that pure-shift cannot give.

OPEN QUESTIONS / TO RESOLVE BEFORE PRESS
  • Final CV-R² numbers for the 3-way model comparison.
  • Whether to lead with Drive or AF+ (both are gain-field; AF+ has
    Sumiya-anchored interpretation, Drive is more flexible).
  • Whether to include the rectangle-paradigm refit in main results
    (~30-70% magnitude change expected, pattern preserved).
  • Whether to include or relegate the σ_init robustness caveat to
    Methods (σ_dyn is poorly identifiable; pin σ at half-radius).
"""
        ax.text(0.05, 0.97, text2, transform=ax.transAxes,
                 fontsize=9, va='top', ha='left',
                 family='serif')
        pdf.savefig(fig); plt.close(fig)

    print(f'wrote {OUT}')


if __name__ == '__main__':
    main()
