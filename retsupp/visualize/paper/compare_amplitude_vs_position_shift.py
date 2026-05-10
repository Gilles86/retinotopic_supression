"""Compare two AF formulations on the same fitted parameters.

  AMPLITUDE-MOD (our DoG-dyn-v3):  voxel_response = ∫ PRF(g) × paradigm × M(g,t) dg
  POSITION-SHIFT (Sumiya):         voxel_response = ∫ PRF_at_shifted_center(g, t) × paradigm dg
                                   where shifted center = precision-weighted mean.

For each example voxel we compute predicted BOLD response under both
formulations using the SAME fitted parameters (sub-28 V3AB), and
compare the time series. If they're nearly identical, the two
interpretations are observationally equivalent.

Output: notes/figures/amplitude_vs_position_shift.pdf
  page 1: predicted shifted PRF center trajectories (x_t, y_t) over time
          under Sumiya formulation, for several example voxels.
  page 2: predicted BOLD per voxel under both formulations, overlaid.
  page 3: scatter of per-voxel predicted-response correlations
          between the two models, plus example mismatches.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

REPO = Path(__file__).resolve().parents[2]
TSV = REPO / 'notes' / 'data' / 'af_dog_v3_target_sharedSigma_parameters.tsv'
OUT = REPO / 'notes' / 'figures' / 'amplitude_vs_position_shift.pdf'

# Geometry — must match the fits
ECC = 4.0
RING = np.array([
    [+ECC/np.sqrt(2),  +ECC/np.sqrt(2)],   # 0: UR
    [-ECC/np.sqrt(2),  +ECC/np.sqrt(2)],   # 1: UL
    [-ECC/np.sqrt(2),  -ECC/np.sqrt(2)],   # 2: LL
    [+ECC/np.sqrt(2),  -ECC/np.sqrt(2)],   # 3: LR
])
HP_IDX = 0  # demo HP location

# Time grid
DT = 0.2
TOTAL_T = 36.0
TIME = np.arange(0, TOTAL_T, DT)

# Trials (same as movie)
EVENT_DURATION = 3.0
TRIALS = [
    {'onset': 4.0,  'target': 1, 'distractor': 0},
    {'onset': 14.0, 'target': 3, 'distractor': 2},
    {'onset': 26.0, 'target': 2, 'distractor': 3},
]

# Spatial grid for amplitude-model integration
RES = 100
GR = 5.5
g1d = np.linspace(-GR, GR, RES)
GX, GY = np.meshgrid(g1d, g1d)
DG = (g1d[1] - g1d[0]) ** 2  # area element

# Example voxels — inside aperture, varied positions
EXAMPLE_VOXELS = [
    ((+1.5, +1.5), 0.7),    # near HP
    ((-1.5, +1.5), 0.7),    # near LP
    ((+0.0, -1.5), 0.7),    # below fovea
    ((+2.5, +0.0), 1.0),    # right side
    ((-1.0, -2.0), 0.9),    # lower left quadrant
]

# Bar paradigm proxy: a moving slit. Same value at all g for simplicity
# (so the integrals reduce to PRF × M comparison; we're testing PRF-vs-M
# kernel structure, not bar dynamics).
def bar_paradigm():
    """Approximate the paradigm by a uniform stimulus 'on' inside the
    aperture. (Real paradigm sweeps a bar; for this comparison the
    important thing is the AF modulation and how it affects different
    PRF positions.)"""
    inside = (GX**2 + GY**2 <= 5.0**2).astype(np.float32)
    return inside


def gauss(mu, sigma):
    return np.exp(-((GX - mu[0])**2 + (GY - mu[1])**2)
                  / (2 * sigma**2)).astype(np.float32)


def build_indicators():
    dyn = np.zeros((len(TIME), 4), dtype=np.float32)
    tgt = np.zeros((len(TIME), 4), dtype=np.float32)
    for trial in TRIALS:
        s = int(trial['onset'] / DT)
        e = s + int(EVENT_DURATION / DT)
        if trial['target'] is not None:
            tgt[s:e, trial['target']] = 1.0
        if trial['distractor'] is not None:
            dyn[s:e, trial['distractor']] = 1.0
    return dyn, tgt


def main():
    df = pd.read_csv(TSV, sep='\t')
    row = df[(df.subject == 28) & (df.roi == 'V3AB')].iloc[0]
    g_HP = float(row.g_HP)
    g_LP = float(row.g_LP)
    g_HP_dyn = float(row.g_HP_dyn)
    g_LP_dyn = float(row.g_LP_dyn)
    g_T_dyn = float(row.g_T_dyn)
    sigma_AF = float(row.sigma_AF)
    sigma_dyn = float(row.sigma_dyn)
    print(f'sub-28 V3AB params:')
    print(f'  g_HP={g_HP:+.3f}  g_LP={g_LP:+.3f}')
    print(f'  g_HP_dyn={g_HP_dyn:+.3f}  g_LP_dyn={g_LP_dyn:+.3f}  g_T_dyn={g_T_dyn:+.3f}')
    print(f'  σ_AF={sigma_AF:.2f}  σ_dyn={sigma_dyn:.2f}')

    paradigm = bar_paradigm()
    dyn_raw, tgt_raw = build_indicators()

    # Pre-compute spatial Gaussians per ring location.
    A_AF = np.stack([gauss(RING[i], sigma_AF) for i in range(4)], axis=0)
    A_dyn = np.stack([gauss(RING[i], sigma_dyn) for i in range(4)], axis=0)

    # Sustained M (constant) — depends on HP_IDX
    sus = sum(
        (g_HP if i == HP_IDX else g_LP) * A_AF[i]
        for i in range(4)
    )

    # Per-TR full M(g, t)
    def M_at(t_idx):
        M = 1.0 + sus.copy()
        for i in range(4):
            g_i = g_HP_dyn if i == HP_IDX else g_LP_dyn
            M = M + g_i * dyn_raw[t_idx, i] * A_dyn[i]
            M = M + g_T_dyn * tgt_raw[t_idx, i] * A_dyn[i]
        return M

    # ---- Position-shift model: precision-weighted mean per TR -----------
    def shifted_center_at(t_idx, prf_xy, sigma_prf):
        """Sumiya formulation: precision-weighted mean of PRF center
        and the active AF Gaussian centers, weighted by gain × precision."""
        prec_prf = 1.0 / sigma_prf**2
        sum_prec = prec_prf
        sum_prec_x = prec_prf * prf_xy[0]
        sum_prec_y = prec_prf * prf_xy[1]
        # Sustained AF locations always active (with gain g_HP or g_LP)
        for i in range(4):
            g_sus = g_HP if i == HP_IDX else g_LP
            # Sumiya only allows positive gains. We extend to signed by
            # treating |g| as weight and using sign for direction.
            # Standard interpretation when g > 0: attractive Gaussian.
            # When g < 0: REPULSIVE (push away from this location). We
            # handle that by NEGATING the precision contribution (the
            # PRF center moves AWAY from the location).
            prec_i = g_sus * 1.0 / sigma_AF**2
            sum_prec += prec_i
            sum_prec_x += prec_i * RING[i][0]
            sum_prec_y += prec_i * RING[i][1]
        # Dynamic distractor (only when active)
        for i in range(4):
            g_i = g_HP_dyn if i == HP_IDX else g_LP_dyn
            w = dyn_raw[t_idx, i]
            if w == 0: continue
            prec_i = g_i * w * 1.0 / sigma_dyn**2
            sum_prec += prec_i
            sum_prec_x += prec_i * RING[i][0]
            sum_prec_y += prec_i * RING[i][1]
        # Target onset (only when active)
        for i in range(4):
            w = tgt_raw[t_idx, i]
            if w == 0: continue
            prec_i = g_T_dyn * w * 1.0 / sigma_dyn**2
            sum_prec += prec_i
            sum_prec_x += prec_i * RING[i][0]
            sum_prec_y += prec_i * RING[i][1]
        if abs(sum_prec) < 1e-9:
            return prf_xy[0], prf_xy[1]
        return sum_prec_x / sum_prec, sum_prec_y / sum_prec

    # ---- Predicted responses --------------------------------------------
    # Amplitude-mod: response[t] = ∫ PRF(g) × paradigm(g) × M(g, t) dg
    # Position-shift: response[t] = ∫ PRF_at_shifted_center(g, t) × paradigm(g) dg
    def predict_amplitude(prf_xy, sigma_prf):
        prf = gauss(prf_xy, sigma_prf)
        out = np.zeros(len(TIME), dtype=np.float32)
        for t in range(len(TIME)):
            M = M_at(t)
            # Rectified to handle negative M
            M_rect = np.maximum(M, 0.0)
            out[t] = (prf * paradigm * M_rect).sum() * DG
        return out

    def predict_position(prf_xy, sigma_prf):
        out = np.zeros(len(TIME), dtype=np.float32)
        for t in range(len(TIME)):
            sx, sy = shifted_center_at(t, prf_xy, sigma_prf)
            shifted_prf = gauss((sx, sy), sigma_prf)
            out[t] = (shifted_prf * paradigm).sum() * DG
        return out

    # ---- Run for example voxels and collect ----------------------------
    results = []
    for prf_xy, sigma_prf in EXAMPLE_VOXELS:
        amp = predict_amplitude(prf_xy, sigma_prf)
        pos = predict_position(prf_xy, sigma_prf)
        # Z-score for shape comparison (amplitudes differ between models)
        amp_z = (amp - amp.mean()) / (amp.std() + 1e-9)
        pos_z = (pos - pos.mean()) / (pos.std() + 1e-9)
        r = np.corrcoef(amp, pos)[0, 1]
        # Also compute the trajectory of shifted center
        traj_x = np.zeros(len(TIME))
        traj_y = np.zeros(len(TIME))
        for t in range(len(TIME)):
            sx, sy = shifted_center_at(t, prf_xy, sigma_prf)
            traj_x[t] = sx; traj_y[t] = sy
        results.append({
            'prf_xy': prf_xy, 'sigma_prf': sigma_prf,
            'amp': amp, 'pos': pos, 'amp_z': amp_z, 'pos_z': pos_z,
            'corr': r, 'traj_x': traj_x, 'traj_y': traj_y,
        })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(OUT) as pdf:
        # Page 1: shifted center trajectories
        fig, axes = plt.subplots(len(EXAMPLE_VOXELS), 2,
                                   figsize=(12, 2.5 * len(EXAMPLE_VOXELS)))
        for i, res in enumerate(results):
            ax_traj = axes[i, 0]
            ax_xy = axes[i, 1]
            ax_traj.plot(TIME, res['traj_x'], color='C0', label='shifted x')
            ax_traj.plot(TIME, res['traj_y'], color='C3', label='shifted y')
            ax_traj.axhline(res['prf_xy'][0], color='C0', ls=':', alpha=0.6)
            ax_traj.axhline(res['prf_xy'][1], color='C3', ls=':', alpha=0.6)
            for trial in TRIALS:
                ax_traj.axvspan(trial['onset'], trial['onset'] + EVENT_DURATION,
                                 color='lime', alpha=0.1)
            ax_traj.set_ylabel(f'PRF=({res["prf_xy"][0]:+.1f},{res["prf_xy"][1]:+.1f})\n'
                                f'σ={res["sigma_prf"]}')
            ax_traj.set_xlabel('time (s)')
            ax_traj.legend(fontsize=8)
            ax_traj.grid(alpha=0.2)
            # right panel: 2D trajectory in visual field
            ax_xy.plot(res['traj_x'], res['traj_y'], 'C0-', alpha=0.6)
            ax_xy.scatter(res['traj_x'][0], res['traj_y'][0],
                           color='black', s=80, marker='o', label='start (sustained)')
            for j in range(4):
                ax_xy.scatter(*RING[j], s=200, marker='o',
                               facecolor='none', edgecolor='gray', lw=1.2)
            ax_xy.scatter(*RING[HP_IDX], s=300, marker='*',
                           facecolor='gold', edgecolor='black', lw=1.2)
            ax_xy.scatter(*res['prf_xy'], color='magenta', marker='+', s=100, label='original PRF')
            ax_xy.set_xlim(-5, 5); ax_xy.set_ylim(-5, 5); ax_xy.set_aspect('equal')
            ax_xy.set_title(f'2D trajectory  (corr with amp model: {res["corr"]:.3f})')
            ax_xy.grid(alpha=0.2)
            ax_xy.legend(fontsize=8, loc='lower left')
        fig.suptitle('Position-shift model: shifted PRF center trajectory per voxel\n'
                      'sub-28 V3AB params, sustained + dynamic + target',
                      fontsize=12, weight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        pdf.savefig(fig); plt.close(fig)

        # Page 2: predicted BOLD comparison per voxel
        fig, axes = plt.subplots(len(EXAMPLE_VOXELS), 1,
                                   figsize=(11, 2 * len(EXAMPLE_VOXELS)))
        for i, res in enumerate(results):
            ax = axes[i] if len(EXAMPLE_VOXELS) > 1 else axes
            ax.plot(TIME, res['amp_z'], color='C0', lw=1.5, label='amplitude-mod (z)')
            ax.plot(TIME, res['pos_z'], color='C3', lw=1.5, ls='--',
                    label='position-shift (z)')
            for trial in TRIALS:
                ax.axvspan(trial['onset'], trial['onset'] + EVENT_DURATION,
                            color='lime', alpha=0.08)
            ax.set_ylabel(f'PRF=({res["prf_xy"][0]:+.1f},{res["prf_xy"][1]:+.1f})')
            ax.set_xlabel('time (s)')
            ax.legend(fontsize=8, loc='upper right')
            ax.set_title(f'corr = {res["corr"]:.3f}',
                          fontsize=10)
            ax.grid(alpha=0.2)
        fig.suptitle('Predicted BOLD per voxel — amplitude-mod vs position-shift '
                      '(z-scored)', fontsize=12, weight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        pdf.savefig(fig); plt.close(fig)

        # Page 3: per-voxel correlation summary
        fig, ax = plt.subplots(figsize=(8, 5))
        rs = [r['corr'] for r in results]
        labels = [f'PRF=({r["prf_xy"][0]:+.1f},{r["prf_xy"][1]:+.1f})\nσ={r["sigma_prf"]}'
                  for r in results]
        ax.barh(range(len(rs)), rs, color=['C2' if r > 0.95 else 'C1' if r > 0.7 else 'C3'
                                              for r in rs])
        ax.set_yticks(range(len(rs)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlim(-1, 1); ax.axvline(1.0, color='gray', ls=':')
        ax.set_xlabel('Pearson r between amplitude-mod and position-shift predictions')
        ax.set_title('Time-series correlation between the two AF formulations\n'
                      'Green: ≥ 0.95 (effectively equivalent). Orange: 0.7-0.95. Red: < 0.7.')
        ax.grid(alpha=0.2, axis='x')
        fig.tight_layout()
        pdf.savefig(fig); plt.close(fig)

    print()
    print('Per-voxel correlations (amplitude-mod vs position-shift):')
    for r in results:
        print(f'  PRF=({r["prf_xy"][0]:+.1f},{r["prf_xy"][1]:+.1f})  '
              f'σ={r["sigma_prf"]}  corr={r["corr"]:.4f}')
    print(f'\nwrote {OUT}')


if __name__ == '__main__':
    main()
