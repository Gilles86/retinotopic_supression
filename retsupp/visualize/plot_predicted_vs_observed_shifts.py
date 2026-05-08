"""Compare observed conditionwise PRF shifts to two model predictions.

Two predictions:
  (a) Sumiya analytical:   precision-weighted-mean shift of (PRF × Π A_i)
                           where each AF Gaussian contributes precision
                           g_i / sigma_AF^2 (signed; negative gain ⇒ repulsion).
                           No "1+" baseline — pure product-of-Gaussians.
  (b) Amplitude-COM:       rectified centre-of-mass of PRF × max(M, 0)
                           with M(g) = 1 + g_HP·A_HP + g_LP·Σ_{ell≠HP} A_ell.
                           Numerical integration on a fine 2D grid.

For each subject × ROI × voxel × condition we compute:
    observed (dx, dy):  conditionwise PRF (x, y)  −  mean-fit PRF (x_mean_model,
                        y_mean_model) from model 4.
    Sumiya  (dx, dy):   compute_net_shifts(...) but with each ring location's
                        precision weighted by (g_HP if HP else g_LP) / sigma_AF^2.
    AmpCOM  (dx, dy):   rho = PRF × max(M, 0); shift = COM(rho) − PRF centre.

The script writes:
    notes/data/shift_comparison.tsv          long-format per-voxel rows
    notes/figures/predicted_vs_observed_shifts.pdf  4-page summary

Usage
-----
    ~/mambaforge/envs/retsupp/bin/python \\
        retsupp/visualize/plot_predicted_vs_observed_shifts.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import pearsonr

from retsupp.utils.data import Subject

# ------------------------- Configuration --------------------------------- #

REPO = Path(__file__).resolve().parents[2]
AF_TSV = REPO / 'notes' / 'data' / 'af_dog_v3_target_sharedSigma_parameters.tsv'
OUT_PDF = REPO / 'notes' / 'figures' / 'predicted_vs_observed_shifts.pdf'
OUT_TSV = REPO / 'notes' / 'data' / 'shift_comparison.tsv'

CONDITIONS = ['upper_right', 'upper_left', 'lower_left', 'lower_right']
ROI_ORDER = ['V1', 'V2', 'V3', 'V3AB', 'hV4', 'LO', 'TO', 'VO']

ECC_RING = 4.0
RING_XY = {
    'upper_right': (+ECC_RING / np.sqrt(2.), +ECC_RING / np.sqrt(2.)),
    'upper_left':  (-ECC_RING / np.sqrt(2.), +ECC_RING / np.sqrt(2.)),
    'lower_left':  (-ECC_RING / np.sqrt(2.), -ECC_RING / np.sqrt(2.)),
    'lower_right': (+ECC_RING / np.sqrt(2.), -ECC_RING / np.sqrt(2.)),
}

# QC cutoffs (per task)
R2_MIN = 0.10                 # mean-fit r2 cutoff
SD_MIN = 0.10                 # avoid degenerate PRF widths
ECC_MIN = 0.5                 # avoid foveal voxels (predictions ≈ 0)
ECC_MAX = 8.0                 # avoid extreme periphery (no AF coverage)

# Amplitude-COM grid
GRID_RES = 81
GRID_R = 5.5

# Example subject for vector field demo
DEMO_SUBJECT = 28
DEMO_ROI = 'V3AB'

plt.rcParams.update({'font.size': 10, 'axes.titlesize': 11,
                     'axes.labelsize': 10, 'figure.titlesize': 12})


# ----------------------- Predictions ------------------------------------- #

def sumiya_shifts(prf_x, prf_y, prf_sd, condition, sigma_AF, g_HP, g_LP):
    """Precision-weighted-mean shift, signed-gain Sumiya extension.

    For each ring location ell we contribute to a precision-weighted
    average:  prec_ell = g_ell / sigma_AF^2  (signed). Negative gain ⇒
    the centre of mass is pulled AWAY from that location (repulsion).

    Returns (dx, dy) shift relative to PRF centre. If the resulting
    total precision is non-positive (degenerate), returns (0, 0).
    """
    var_prf = prf_sd ** 2
    prec_prf = 1.0 / var_prf
    sum_prec = prec_prf
    sum_prec_x = prec_prf * prf_x
    sum_prec_y = prec_prf * prf_y
    for name, (rx, ry) in RING_XY.items():
        g = g_HP if name == condition else g_LP
        prec = g / (sigma_AF ** 2)
        sum_prec += prec
        sum_prec_x += prec * rx
        sum_prec_y += prec * ry
    if abs(sum_prec) < 1e-9 or sum_prec <= 0:
        return 0.0, 0.0
    new_x = sum_prec_x / sum_prec
    new_y = sum_prec_y / sum_prec
    return new_x - prf_x, new_y - prf_y


def precompute_amplitude_grid(sigma_AF, g_HP, g_LP):
    """Return per-condition modulation field M_C(g) on a fixed grid.

    Output shape (len(CONDITIONS), GRID_RES**2).
    """
    g1d = np.linspace(-GRID_R, GRID_R, GRID_RES, dtype=np.float64)
    GX, GY = np.meshgrid(g1d, g1d)
    G = np.stack([GX.ravel(), GY.ravel()], axis=1)
    ring = np.array([RING_XY[c] for c in CONDITIONS], dtype=np.float64)
    A = np.exp(
        -np.sum((G[:, None, :] - ring[None, :, :]) ** 2, axis=-1)
        / (2.0 * sigma_AF ** 2)
    )                                                    # (R^2, 4)
    M = np.empty((len(CONDITIONS), G.shape[0]), dtype=np.float64)
    for ci in range(len(CONDITIONS)):
        a_hp = A[:, ci]
        a_lp = A.sum(axis=1) - a_hp
        M[ci] = 1.0 + g_HP * a_hp + g_LP * a_lp
    return G, M


def amplitude_com_shifts(prf_x, prf_y, prf_sd, condition_idx, G, M):
    """Rectified COM of PRF × max(M, 0) on the fixed grid.

    Returns (dx, dy) shift relative to PRF centre; (0, 0) on a
    degenerate (total mass ≈ 0) result.
    """
    M_c = M[condition_idx]
    M_rect = np.clip(M_c, 0.0, None)
    d2 = (G[:, 0] - prf_x) ** 2 + (G[:, 1] - prf_y) ** 2
    S = np.exp(-d2 / (2.0 * prf_sd ** 2))
    rho = S * M_rect
    Z = rho.sum()
    if Z < 1e-9:
        return 0.0, 0.0
    cx = (G[:, 0] * rho).sum() / Z
    cy = (G[:, 1] * rho).sum() / Z
    return cx - prf_x, cy - prf_y


# ----------------------- Per-subject processing -------------------------- #

def process_subject(subject_id, bids_folder, af_df):
    sub = Subject(subject_id, bids_folder)
    cond = sub.get_conditionwise_summary_prf_pars(model=4)
    cond = cond.reset_index()

    # QC filter on the *mean-fit* (model 4) parameters: per task spec.
    qc = (
        (cond['r2_mean_model'] > R2_MIN)
        & (cond['sd_mean_model'] > SD_MIN)
        & (cond['ecc_mean_model'] > ECC_MIN)
        & (cond['ecc_mean_model'] < ECC_MAX)
    )
    cond = cond.loc[qc].copy()

    if len(cond) == 0:
        return None

    out_rows = []
    for roi, rdf in cond.groupby('roi_base'):
        af_row = af_df[(af_df['subject'] == subject_id)
                        & (af_df['roi'] == roi)]
        if len(af_row) == 0:
            continue
        af_row = af_row.iloc[0]
        sigma_AF = float(af_row['sigma_AF'])
        g_HP = float(af_row['g_HP'])
        g_LP = float(af_row['g_LP'])

        # Pre-compute modulation field once per ROI.
        G, M = precompute_amplitude_grid(sigma_AF, g_HP, g_LP)

        for ci, cond_name in enumerate(CONDITIONS):
            sub_df = rdf[rdf['condition'] == cond_name]
            if len(sub_df) == 0:
                continue
            xs = sub_df['x_mean_model'].to_numpy(dtype=np.float64)
            ys = sub_df['y_mean_model'].to_numpy(dtype=np.float64)
            sds = sub_df['sd_mean_model'].to_numpy(dtype=np.float64)
            obs_x = sub_df['x'].to_numpy(dtype=np.float64)
            obs_y = sub_df['y'].to_numpy(dtype=np.float64)
            obs_dx = obs_x - xs
            obs_dy = obs_y - ys

            sumiya_dx = np.empty(len(sub_df))
            sumiya_dy = np.empty(len(sub_df))
            amp_dx = np.empty(len(sub_df))
            amp_dy = np.empty(len(sub_df))
            for i in range(len(sub_df)):
                sumiya_dx[i], sumiya_dy[i] = sumiya_shifts(
                    xs[i], ys[i], sds[i], cond_name,
                    sigma_AF, g_HP, g_LP,
                )
                amp_dx[i], amp_dy[i] = amplitude_com_shifts(
                    xs[i], ys[i], sds[i], ci, G, M,
                )

            # Distance from HP location (in the *base* PRF frame).
            hp_x, hp_y = RING_XY[cond_name]
            dist_hp = np.hypot(xs - hp_x, ys - hp_y)

            r2_arr = sub_df['r2_mean_model'].to_numpy(dtype=np.float64)
            voxel_arr = sub_df['voxel'].to_numpy(dtype=np.int32)
            for i in range(len(sub_df)):
                out_rows.append({
                    'subject': subject_id,
                    'roi': roi,
                    'voxel': int(voxel_arr[i]),
                    'condition': cond_name,
                    'base_x': xs[i], 'base_y': ys[i], 'base_sd': sds[i],
                    'obs_dx': obs_dx[i], 'obs_dy': obs_dy[i],
                    'sumiya_dx': sumiya_dx[i], 'sumiya_dy': sumiya_dy[i],
                    'amp_dx': amp_dx[i], 'amp_dy': amp_dy[i],
                    'dist_from_hp': float(dist_hp[i]),
                    'r2_mean_model': float(r2_arr[i]),
                })

    if not out_rows:
        return None
    return pd.DataFrame(out_rows)


# ----------------------- Plotting --------------------------------------- #

def _bin_and_plot_scatter(ax, x, y, label_x, label_y, title, max_pts=4000):
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]; y = y[mask]
    if len(x) < 2:
        ax.text(0.5, 0.5, 'no data', ha='center', va='center',
                transform=ax.transAxes)
        return
    if len(x) > max_pts:
        idx = np.random.RandomState(0).choice(len(x), max_pts, replace=False)
        ax.scatter(x[idx], y[idx], s=2, alpha=0.15, color='0.4', rasterized=True)
    else:
        ax.scatter(x, y, s=2, alpha=0.2, color='0.4', rasterized=True)
    # Bin x, mean y per bin.
    bins = np.linspace(0, np.nanpercentile(x, 98), 12)
    digit = np.digitize(x, bins)
    bin_x = []; bin_y = []
    for b in range(1, len(bins)):
        sel = digit == b
        if sel.sum() < 5:
            continue
        bin_x.append(np.median(x[sel]))
        bin_y.append(np.mean(y[sel]))
    if bin_x:
        ax.plot(bin_x, bin_y, '-o', color='C3', lw=2, ms=5, label='binned mean')
    r, _ = pearsonr(x, y) if len(x) > 1 else (np.nan, np.nan)
    lim = float(np.nanpercentile(np.r_[x, y], 99))
    lim = max(lim, 0.05)
    ax.plot([0, lim], [0, lim], 'k:', lw=0.6)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim * 1.5)
    ax.set_xlabel(label_x); ax.set_ylabel(label_y)
    ax.set_title(f'{title}   r={r:.3f}, n={len(x)}')


def page1_per_roi_scatter(records, pdf):
    rois = [r for r in ROI_ORDER if r in records['roi'].unique()]
    n = len(rois)
    fig, axes = plt.subplots(n, 2, figsize=(11, 2.4 * n), squeeze=False)
    fig.suptitle('Page 1: Predicted vs observed shift magnitudes (per ROI)',
                 fontsize=12, weight='bold')
    for i, roi in enumerate(rois):
        sub = records[records['roi'] == roi]
        # Drop rows where BOTH predictions are 0 (no AF coverage).
        sub = sub[(sub['sumiya_mag'] > 1e-3) | (sub['amp_mag'] > 1e-3)]
        _bin_and_plot_scatter(
            axes[i, 0],
            sub['sumiya_mag'].to_numpy(),
            sub['obs_mag'].to_numpy(),
            'Sumiya |Δ| (°)', 'observed |Δ| (°)',
            f'{roi}  Sumiya',
        )
        _bin_and_plot_scatter(
            axes[i, 1],
            sub['amp_mag'].to_numpy(),
            sub['obs_mag'].to_numpy(),
            'Amplitude-COM |Δ| (°)', 'observed |Δ| (°)',
            f'{roi}  Amplitude-COM',
        )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)


def page2_vector_field(records, pdf, demo_sub=DEMO_SUBJECT, demo_roi=DEMO_ROI):
    sub = records[(records['subject'] == demo_sub)
                   & (records['roi'] == demo_roi)]
    if len(sub) == 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, f'sub-{demo_sub} {demo_roi} has no data',
                ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')
        pdf.savefig(fig); plt.close(fig)
        return
    fig, axes = plt.subplots(3, 4, figsize=(14, 11),
                              gridspec_kw={'hspace': 0.3})
    fig.suptitle(f'Page 2: 2D shift fields — sub-{demo_sub:02d} {demo_roi}',
                 fontsize=12, weight='bold')
    # Subsample once per condition for clarity
    rng = np.random.RandomState(0)
    for ci, cond in enumerate(CONDITIONS):
        sub_c = sub[sub['condition'] == cond]
        if len(sub_c) == 0:
            for r in range(3):
                axes[r, ci].axis('off')
            continue
        idx = sub_c.index.to_numpy()
        if len(idx) > 120:
            idx = rng.choice(idx, 120, replace=False)
        rows = sub_c.loc[idx]
        bx = rows['base_x'].to_numpy(); by = rows['base_y'].to_numpy()
        hp_x, hp_y = RING_XY[cond]
        for r, (dx_col, dy_col, name, color) in enumerate([
            ('obs_dx', 'obs_dy', 'observed', 'k'),
            ('sumiya_dx', 'sumiya_dy', 'Sumiya', 'C0'),
            ('amp_dx', 'amp_dy', 'Amplitude-COM', 'C3'),
        ]):
            ax = axes[r, ci]
            dx = rows[dx_col].to_numpy(); dy = rows[dy_col].to_numpy()
            ax.quiver(bx, by, dx, dy, color=color, angles='xy',
                       scale_units='xy', scale=1.0, width=0.004,
                       alpha=0.6)
            ax.scatter(bx, by, s=2, color='0.6')
            ax.scatter(hp_x, hp_y, marker='*', s=240, color='gold',
                        edgecolor='k', zorder=10)
            for cn, (rx, ry) in RING_XY.items():
                if cn != cond:
                    ax.scatter(rx, ry, s=80, marker='o', facecolor='none',
                                edgecolor='gray', lw=0.8, zorder=9)
            circ = plt.Circle((0, 0), 4.0, fill=False, color='0.7',
                                ls='--', lw=0.5)
            ax.add_patch(circ)
            ax.set_xlim(-5, 5); ax.set_ylim(-5, 5)
            ax.set_aspect('equal')
            ax.axhline(0, color='0.85', lw=0.4)
            ax.axvline(0, color='0.85', lw=0.4)
            if r == 0:
                ax.set_title(cond.replace('_', ' '))
            if ci == 0:
                ax.set_ylabel(name)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)


def page3_distance_curves(records, pdf, demo_sub=DEMO_SUBJECT,
                            demo_roi=DEMO_ROI):
    rois = [r for r in ROI_ORDER if r in records['roi'].unique()]
    n = len(rois)
    fig, axes = plt.subplots(n, 1, figsize=(8, 1.6 * n), squeeze=False)
    fig.suptitle('Page 3a: shift magnitude vs distance from HP (per ROI)\n'
                  '(Sumiya predicts non-zero shifts at large distance; '
                  'amplitude predicts ~0)',
                  fontsize=11, weight='bold')
    bins = np.arange(0, 9, 0.5)
    bin_centres = 0.5 * (bins[:-1] + bins[1:])
    for i, roi in enumerate(rois):
        ax = axes[i, 0]
        sub = records[records['roi'] == roi]
        if len(sub) == 0:
            continue
        digit = np.digitize(sub['dist_from_hp'].to_numpy(), bins)
        for col, color, label in [
            ('obs_mag', 'k', 'observed'),
            ('sumiya_mag', 'C0', 'Sumiya'),
            ('amp_mag', 'C3', 'Amplitude'),
        ]:
            v = sub[col].to_numpy()
            curve = []
            sem = []
            for b in range(1, len(bins)):
                sel = digit == b
                if sel.sum() < 5:
                    curve.append(np.nan); sem.append(np.nan); continue
                curve.append(np.mean(v[sel]))
                sem.append(np.std(v[sel]) / np.sqrt(sel.sum()))
            curve = np.asarray(curve); sem = np.asarray(sem)
            ax.plot(bin_centres, curve, '-', color=color, lw=1.5, label=label)
            ax.fill_between(bin_centres, curve - sem, curve + sem,
                              color=color, alpha=0.15)
        ax.set_ylabel(roi)
        ax.set_xlabel('distance from HP (°)')
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(alpha=0.2)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)

    # Page 3b: negative-gain handling (where g_HP < -0.5).
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle(f'Page 3b: Shift magnitude vs |g_HP| — sub-{demo_sub:02d} '
                  f'{demo_roi} and ROI-aggregated (negative-gain regime).',
                  fontsize=11, weight='bold')
    # Aggregate per (subject, ROI) — one (g_HP, mean shift) point each.
    agg = (records.groupby(['subject', 'roi'])
            .agg(g_HP=('g_HP', 'first'),
                 obs=('obs_mag', 'mean'),
                 sumiya=('sumiya_mag', 'mean'),
                 amp=('amp_mag', 'mean'))
            .reset_index())
    for ax, ycol, name, color in [
        (axes[0], 'sumiya', 'Sumiya', 'C0'),
        (axes[1], 'amp', 'Amplitude-COM', 'C3'),
    ]:
        sns.scatterplot(data=agg, x='g_HP', y=ycol,
                          hue='roi', ax=ax, s=30, alpha=0.7)
        sns.scatterplot(data=agg, x='g_HP', y='obs',
                          ax=ax, s=18, color='k', alpha=0.5,
                          marker='x', label='observed')
        ax.axvline(0, color='0.7', lw=0.5)
        ax.set_xlabel('g_HP (sustained gain)')
        ax.set_ylabel(f'mean {name} |Δ| per (subj, ROI)')
        ax.set_title(name)
        ax.legend(fontsize=6, loc='upper left', ncol=2)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)


def page4_quantitative_summary(records, pdf):
    """Per-ROI Pearson r table and bar chart."""
    rois = [r for r in ROI_ORDER if r in records['roi'].unique()]
    rows = []
    # Per-ROI: pool voxels across subjects.
    for roi in rois:
        sub = records[records['roi'] == roi]
        # Don't bias by all-zero predictions.
        sub = sub[(sub['sumiya_mag'] > 1e-3) | (sub['amp_mag'] > 1e-3)]
        if len(sub) < 5:
            rows.append({'roi': roi, 'r_sumiya': np.nan, 'r_amp': np.nan,
                          'n': len(sub)})
            continue
        r_s, _ = pearsonr(sub['sumiya_mag'], sub['obs_mag'])
        r_a, _ = pearsonr(sub['amp_mag'], sub['obs_mag'])
        rows.append({'roi': roi, 'r_sumiya': r_s, 'r_amp': r_a,
                      'n': len(sub)})
    summary = pd.DataFrame(rows)

    fig, (ax_bar, ax_tab) = plt.subplots(2, 1, figsize=(9, 7),
                                           gridspec_kw={'height_ratios': [3, 2]})
    fig.suptitle('Page 4: Pearson r(observed |Δ|, predicted |Δ|) per ROI',
                  fontsize=12, weight='bold')
    xs = np.arange(len(summary))
    ax_bar.bar(xs - 0.18, summary['r_sumiya'], width=0.36, color='C0',
                label='Sumiya')
    ax_bar.bar(xs + 0.18, summary['r_amp'], width=0.36, color='C3',
                label='Amplitude-COM')
    ax_bar.set_xticks(xs); ax_bar.set_xticklabels(summary['roi'])
    ax_bar.axhline(0, color='k', lw=0.5)
    ax_bar.set_ylabel('Pearson r')
    ax_bar.legend(loc='upper right')
    ax_bar.grid(alpha=0.2, axis='y')
    for i, row in summary.iterrows():
        ax_bar.text(i - 0.18, row['r_sumiya'] + 0.005,
                     f'{row["r_sumiya"]:.2f}',
                     ha='center', va='bottom', fontsize=8)
        ax_bar.text(i + 0.18, row['r_amp'] + 0.005,
                     f'{row["r_amp"]:.2f}',
                     ha='center', va='bottom', fontsize=8)
    ax_tab.axis('off')
    table = summary.copy()
    table['r_sumiya'] = table['r_sumiya'].round(3)
    table['r_amp'] = table['r_amp'].round(3)
    ax_tab.text(0.0, 0.95, table.to_string(index=False),
                  family='monospace', fontsize=9, va='top')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)
    return summary


# ----------------------- Main ------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bids-folder', default='/data/ds-retsupp')
    parser.add_argument('--out-pdf', type=Path, default=OUT_PDF)
    parser.add_argument('--out-tsv', type=Path, default=OUT_TSV)
    parser.add_argument('--af-tsv', type=Path, default=AF_TSV)
    parser.add_argument('--limit-subjects', type=int, default=None,
                          help='for quick testing, process only first N subjects')
    args = parser.parse_args()

    af_df = pd.read_csv(args.af_tsv, sep='\t')
    subjects = sorted(af_df['subject'].unique().tolist())
    if args.limit_subjects:
        subjects = subjects[:args.limit_subjects]
    print(f'Processing {len(subjects)} subjects: {subjects}')

    all_dfs = []
    for sid in subjects:
        try:
            print(f'-- sub-{sid:02d}', flush=True)
            df = process_subject(sid, args.bids_folder, af_df)
            if df is not None:
                all_dfs.append(df)
                print(f'   {len(df)} rows', flush=True)
        except FileNotFoundError as e:
            print(f'   skip (missing data): {e}')
        except Exception as e:
            print(f'   ERROR: {e}')

    if not all_dfs:
        raise SystemExit('No subjects produced data — abort.')

    records = pd.concat(all_dfs, ignore_index=True)
    # Derived magnitudes used by the plotting code.
    records['obs_mag'] = np.hypot(records['obs_dx'], records['obs_dy'])
    records['sumiya_mag'] = np.hypot(records['sumiya_dx'], records['sumiya_dy'])
    records['amp_mag'] = np.hypot(records['amp_dx'], records['amp_dy'])
    # Re-attach AF parameters per (subject, roi) row.
    records = records.merge(
        af_df[['subject', 'roi', 'sigma_AF', 'g_HP', 'g_LP']],
        on=['subject', 'roi'], how='left',
    )
    args.out_tsv.parent.mkdir(parents=True, exist_ok=True)
    # The on-disk TSV is the slim version (only the per-voxel observed +
    # predicted shifts plus base-PRF metadata). Reconstructable info
    # (g_HP, g_LP, sigma_AF, magnitudes) is dropped from the file but
    # kept on the in-memory ``records`` DataFrame for plotting.
    slim_cols = ['subject', 'roi', 'voxel', 'condition',
                  'base_x', 'base_y', 'base_sd',
                  'obs_dx', 'obs_dy',
                  'sumiya_dx', 'sumiya_dy',
                  'amp_dx', 'amp_dy',
                  'dist_from_hp', 'r2_mean_model']
    slim = records[slim_cols].copy()
    float_cols = slim.select_dtypes(include=[np.floating]).columns
    slim[float_cols] = slim[float_cols].astype(np.float32)
    slim.to_csv(args.out_tsv, sep='\t', index=False, float_format='%.4f')
    print(f'Wrote {args.out_tsv}  ({len(slim)} rows)')

    args.out_pdf.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(args.out_pdf) as pdf:
        page1_per_roi_scatter(records, pdf)
        page2_vector_field(records, pdf)
        page3_distance_curves(records, pdf)
        summary = page4_quantitative_summary(records, pdf)
    print(f'Wrote {args.out_pdf}')
    print('\nPer-ROI Pearson r summary:')
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
