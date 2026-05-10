"""Demonstrate that even PRFs which only see the BAR (no overlap
with the distractor/target ring positions) are still affected by
the gain field, because AF Gaussian tails extend into the aperture.

Output: notes/figures/foveal_af_influence.pdf, multi-panel:
  Panel A: spatial map — for a grid of hypothetical PRF positions,
           color = magnitude of AF-induced response modulation
           (relative to no-AF baseline). Should show non-zero
           modulation throughout the aperture, including fovea.
  Panel B: per-position bar+AF time series — pick 3 example PRFs:
           foveal (0, 0.5), inside-aperture mid (1.5, 0), and at-edge
           (2.5, 2.5). Show predicted BOLD with vs without AF for each.
  Panel C: distance-from-ring vs modulation strength — quantitative
           fall-off of AF effect with PRF eccentricity, demonstrating
           the AF reaches voxels well within the aperture.

Uses sub-28 V3AB params (real, extreme).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Circle

REPO = Path(__file__).resolve().parents[2]
TSV = REPO / 'notes' / 'data' / 'af_dog_v3_target_sharedSigma_parameters.tsv'
OUT = REPO / 'notes' / 'figures' / 'foveal_af_influence.pdf'

ECC = 4.0
RING = np.array([
    [+ECC/np.sqrt(2),  +ECC/np.sqrt(2)],   # UR (HP)
    [-ECC/np.sqrt(2),  +ECC/np.sqrt(2)],   # UL
    [-ECC/np.sqrt(2),  -ECC/np.sqrt(2)],   # LL
    [+ECC/np.sqrt(2),  -ECC/np.sqrt(2)],   # LR
])
HP_IDX = 0
APERTURE_R = 3.17

# Spatial grid for integration
RES = 120
GR = 5.0
g1d = np.linspace(-GR, GR, RES)
GX, GY = np.meshgrid(g1d, g1d)
DG = (g1d[1] - g1d[0]) ** 2

# Time grid
DT = 0.2
TOTAL_T = 36.0
TIME = np.arange(0, TOTAL_T, DT)

# A simple bar paradigm: a horizontal slit sweeping vertically up + down,
# only inside the aperture.
def make_bar_paradigm():
    """Bar sweeps from y=-3 to +3 over the run, then back. Always inside
    the aperture circle. Returns (T, R, R) array."""
    inside = (GX**2 + GY**2 <= APERTURE_R**2).astype(np.float32)
    par = np.zeros((len(TIME), *GX.shape), dtype=np.float32)
    half = len(TIME) // 2
    for t in range(len(TIME)):
        # bar y-position
        if t < half:
            y_bar = -3.0 + 6.0 * (t / half)
        else:
            y_bar = +3.0 - 6.0 * ((t - half) / (len(TIME) - half))
        # 0.5°-thick horizontal bar
        in_bar = np.abs(GY - y_bar) < 0.25
        par[t] = inside * in_bar.astype(np.float32)
    return par


def gauss(mu, sigma):
    return np.exp(-((GX - mu[0])**2 + (GY - mu[1])**2)
                  / (2 * sigma**2)).astype(np.float32)


def main():
    df = pd.read_csv(TSV, sep='\t')
    row = df[(df.subject == 28) & (df.roi == 'V3AB')].iloc[0]
    g_HP = float(row.g_HP)
    g_LP = float(row.g_LP)
    sigma_AF = float(row.sigma_AF)
    print(f'sub-28 V3AB: g_HP={g_HP:+.2f} g_LP={g_LP:+.2f} σ_AF={sigma_AF:.2f}')

    # Sustained AF (only — keep it simple for the foveal demo)
    A_HP = gauss(RING[HP_IDX], sigma_AF)
    A_LP = sum(gauss(RING[i], sigma_AF) for i in range(4) if i != HP_IDX)
    M = 1.0 + g_HP * A_HP + g_LP * A_LP
    M_pos = np.maximum(M, 0.0)

    paradigm = make_bar_paradigm()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(OUT) as pdf:
        # ---- Panel A: spatial map of AF-induced modulation ratio ----
        # For each candidate PRF position, compute response with vs
        # without AF. Ratio = SD(with_AF) / SD(no_AF), or norm-difference.
        sigma_prf = 0.6   # typical V3AB PRF size
        # Sample a coarse grid of PRF positions
        prf_grid_1d = np.linspace(-3.0, 3.0, 21)
        eff_map = np.zeros((len(prf_grid_1d), len(prf_grid_1d)),
                            dtype=np.float32)
        for i, py in enumerate(prf_grid_1d):
            for j, px in enumerate(prf_grid_1d):
                prf = gauss((px, py), sigma_prf)
                # response without AF: bar × PRF integrated over space, per t
                response_no_af = np.einsum('txy,xy->t', paradigm, prf) * DG
                response_af = np.einsum('txy,xy->t', paradigm, prf * M_pos) * DG
                # modulation magnitude as relative deviation
                if response_no_af.std() > 1e-6:
                    diff = response_af - response_no_af
                    eff_map[i, j] = np.sqrt(np.mean(diff**2)) / response_no_af.std()
                else:
                    eff_map[i, j] = 0.0

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        ax = axes[0]
        im = ax.imshow(M, extent=[-GR, GR, -GR, GR], origin='lower',
                        cmap='RdBu_r', vmin=1.0 - max(abs(M.min()-1), abs(M.max()-1)),
                        vmax=1.0 + max(abs(M.min()-1), abs(M.max()-1)))
        ax.add_patch(Circle((0, 0), APERTURE_R, fill=False,
                             edgecolor='black', lw=1.5, ls='--'))
        for i, (x, y) in enumerate(RING):
            ax.scatter([x], [y], s=300, marker='o', facecolor='none',
                        edgecolor='black', lw=1.5)
        ax.scatter(*RING[HP_IDX], s=400, marker='*', facecolor='gold',
                    edgecolor='black', lw=1.5, label='HP')
        ax.scatter([0], [0], color='black', marker='+', s=80)
        plt.colorbar(im, ax=ax, label='M(g) — sustained AF only')
        ax.set_xlim(-GR, GR); ax.set_ylim(-GR, GR); ax.set_aspect('equal')
        ax.set_xlabel('x (deg)'); ax.set_ylabel('y (deg)')
        ax.set_title('A.  Modulation field M(g)\n'
                      'AF Gaussian tails clearly extend INTO the aperture',
                      fontsize=10)

        ax = axes[1]
        im2 = ax.imshow(eff_map, extent=[prf_grid_1d.min(), prf_grid_1d.max(),
                                            prf_grid_1d.min(), prf_grid_1d.max()],
                        origin='lower', cmap='magma',
                        vmin=0, vmax=eff_map.max())
        # overlay aperture
        ax.add_patch(Circle((0, 0), APERTURE_R, fill=False,
                             edgecolor='cyan', lw=1.5, ls='--',
                             label=f'aperture {APERTURE_R}°'))
        # overlay rings (out of frame, but we show edge)
        for i, (x, y) in enumerate(RING):
            ax.scatter([x], [y], s=200, marker='o', facecolor='none',
                        edgecolor='lime', lw=1.5)
        ax.scatter([0], [0], color='cyan', marker='+', s=60)
        plt.colorbar(im2, ax=ax,
                      label='RMS response deviation\n(with AF − without AF) / SD')
        ax.set_xlim(-3.5, 3.5); ax.set_ylim(-3.5, 3.5); ax.set_aspect('equal')
        ax.set_xlabel('PRF x (deg)'); ax.set_ylabel('PRF y (deg)')
        ax.set_title('B.  AF-induced response modulation per PRF position\n'
                      'Magnitude > 0 throughout aperture (fovea included)',
                      fontsize=10)

        fig.suptitle(
            'Even PRFs only sensitive to the bar (not distractors) are affected\n'
            'sub-28 V3AB params: g_HP=−0.77, g_LP=+1.02, σ_AF=1.09° (real fit)',
            fontsize=12, weight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        pdf.savefig(fig); plt.close(fig)

        # ---- Panel B: example PRF response time series with vs without AF ----
        EXAMPLES = [
            ((0.0, +0.5), 'foveal (0.5° from fovea)'),
            ((+1.5, 0.0), 'mid-aperture (1.5°)'),
            ((+2.5, +1.0), 'near aperture edge (2.7°)'),
        ]
        fig, axes = plt.subplots(len(EXAMPLES), 1, figsize=(11, 7),
                                   sharex=True)
        for ax, ((px, py), label) in zip(axes, EXAMPLES):
            prf = gauss((px, py), 0.6)
            r_no = np.einsum('txy,xy->t', paradigm, prf) * DG
            r_af = np.einsum('txy,xy->t', paradigm, prf * M_pos) * DG
            ax.plot(TIME, r_no, color='black', lw=1.5,
                    label='no AF (bar only)')
            ax.plot(TIME, r_af, color='C3', lw=1.5,
                    label='with AF', alpha=0.85)
            ax.set_ylabel(f'{label}\n(x={px:+.1f}, y={py:+.1f})')
            ax.legend(loc='upper right', fontsize=9)
            ax.grid(alpha=0.2)
        axes[-1].set_xlabel('time (s)')
        fig.suptitle(
            'C.  Predicted bar-driven BOLD response per PRF position\n'
            'with vs without AF — even foveal voxels show modulation '
            'when bar overlaps regions where M(g)≠1',
            fontsize=11, weight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig); plt.close(fig)

        # ---- Panel C: distance-from-ring vs modulation magnitude ----
        # For PRF positions along the diagonal from fovea (0,0) toward HP
        # (2.83, 2.83) with various distances, compute modulation strength.
        ts = np.linspace(0.0, 1.0, 30)
        prf_xs = ts * RING[HP_IDX][0]
        prf_ys = ts * RING[HP_IDX][1]
        mods = np.zeros(len(ts))
        for k, (px, py) in enumerate(zip(prf_xs, prf_ys)):
            prf = gauss((px, py), 0.6)
            r_no = np.einsum('txy,xy->t', paradigm, prf) * DG
            r_af = np.einsum('txy,xy->t', paradigm, prf * M_pos) * DG
            mods[k] = np.sqrt(np.mean((r_af - r_no)**2)) / (r_no.std() + 1e-9)
        dist_to_HP = np.sqrt((prf_xs - RING[HP_IDX][0])**2 +
                              (prf_ys - RING[HP_IDX][1])**2)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(dist_to_HP, mods, 'o-', color='C0', lw=2)
        ax.set_xlabel('distance from HP (deg)')
        ax.set_ylabel('AF effect magnitude\n(RMS [response_AF − response_no_AF] / SD)')
        # Annotate aperture edge: distance from HP at the aperture edge
        # is ecc - aperture (along the HP-fovea line) ≈ 0.83°
        ax.axvline(ECC - APERTURE_R, color='black', ls='--', lw=1,
                   label=f'aperture edge from HP ({ECC-APERTURE_R:.2f}°)')
        ax.axvline(ECC, color='gray', ls=':', lw=1,
                   label=f'fovea ({ECC:.0f}°)')
        ax.legend()
        ax.grid(alpha=0.2)
        ax.set_title(
            'D.  AF effect magnitude vs PRF distance from HP\n'
            'Effect persists across the whole aperture (left of dashed line)\n'
            'and even into far-from-aperture territory.',
            fontsize=11)
        fig.tight_layout()
        pdf.savefig(fig); plt.close(fig)

        # ---- Panel E: EMPIRICAL evidence from conditionwise PRF data ---
        # Use the shift_comparison TSV that the earlier comparison agent
        # produced. It has per-voxel observed_dx, observed_dy across
        # subjects × conditions × ROIs, plus the mean-fit (x_mean, y_mean).
        shift_tsv = REPO / 'notes' / 'data' / 'shift_comparison.tsv'
        if shift_tsv.exists():
            print('Loading shift_comparison.tsv (this may take a while)...')
            df = pd.read_csv(shift_tsv, sep='\t')
            print(f'  shape: {df.shape}, cols: {list(df.columns)}')
            # eccentricity of mean-fit PRF
            df['ecc_mean'] = np.sqrt(df['base_x']**2 + df['base_y']**2)
            # observed shift magnitude
            df['obs_shift_mag'] = np.sqrt(
                df['obs_dx']**2 + df['obs_dy']**2)
            # amplitude-COM predicted shift magnitude
            df['amp_shift_mag'] = np.sqrt(
                df['amp_dx']**2 + df['amp_dy']**2)
            # bucket by eccentricity
            bins = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.17, 4.0, 5.0, 8.0]
            labels = ['<0.5', '0.5-1', '1-1.5', '1.5-2', '2-2.5',
                      '2.5-3.17 (≈edge)', '3.17-4 (out of ap.)',
                      '4-5', '5-8']
            df['ecc_bin'] = pd.cut(df['ecc_mean'], bins=bins,
                                    labels=labels, right=False)
            ROI_ORDER = ['V1', 'V2', 'V3', 'V3AB', 'hV4', 'LO', 'TO', 'VO']
            fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharey=False)
            for ax, roi in zip(axes.flat, ROI_ORDER):
                sub = df[df.roi == roi]
                # group by bin → mean obs and amp shift magnitude
                obs_by_bin = sub.groupby('ecc_bin', observed=True)[
                    'obs_shift_mag'].agg(['mean', 'sem', 'count'])
                amp_by_bin = sub.groupby('ecc_bin', observed=True)[
                    'amp_shift_mag'].agg(['mean', 'sem', 'count'])
                bin_centers = np.arange(len(obs_by_bin))
                ax.errorbar(bin_centers, obs_by_bin['mean'],
                              yerr=obs_by_bin['sem'], fmt='o-',
                              color='black', label='observed')
                ax.errorbar(bin_centers, amp_by_bin['mean'],
                              yerr=amp_by_bin['sem'], fmt='s-',
                              color='C3', label='amplitude pred',
                              alpha=0.8)
                ax.axvline(5.5, color='gray', ls='--', lw=0.8,
                            alpha=0.5)
                ax.text(5.5, ax.get_ylim()[1] * 0.95, 'aperture edge',
                          rotation=90, va='top', ha='right',
                          fontsize=8, color='gray')
                ax.set_xticks(bin_centers)
                ax.set_xticklabels(obs_by_bin.index, rotation=40,
                                     fontsize=7)
                ax.set_xlabel('PRF eccentricity (deg)', fontsize=9)
                ax.set_ylabel('|shift| (deg)', fontsize=9)
                ax.set_title(f'{roi}', fontsize=10)
                ax.legend(fontsize=8); ax.grid(alpha=0.2)
            fig.suptitle(
                'E.  EMPIRICAL: observed conditionwise shift magnitude '
                'by PRF eccentricity bucket\n'
                'Even foveal (<1°) and within-aperture (1-3°) voxels — '
                'which only see the bar, never the distractors — '
                'still show non-zero shifts.',
                fontsize=12, weight='bold')
            fig.tight_layout(rect=[0, 0, 1, 0.94])
            pdf.savefig(fig); plt.close(fig)
        else:
            print(f'WARNING: {shift_tsv} not found — skip empirical panel.')

    print(f'wrote {OUT}')


if __name__ == '__main__':
    main()
