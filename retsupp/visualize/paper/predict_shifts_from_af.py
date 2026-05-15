"""Predict per-condition PRF center shifts from a joint AF+PRF fit.

Loads a fitted ``AttentionFieldPRF2DWithHRF`` pickle (per subject × ROI),
computes for each voxel the *predicted* center of the modulated PRF in
each of the 4 HP conditions, and compares to the *observed* conditionwise
PRF centers from ``derivatives/prf_conditionfit``.

For each voxel v and condition C the AD-pRF profile in stimulus space is

    rho_v_C(g) = M_C(g) * S_v(g)

with shared modulation field

    M_C(g) = 1 + sign * ( g_HP * A_{HP_C}(g)
                          + g_LP * sum_{ell != HP_C} A_ell(g) )

where each AF Gaussian A_ell has the same sigma_AF and unit peak. The
predicted observed PRF center is the center of mass of rho_v_C on a
discrete grid:

    mu_tilde_v_C = sum_g g * rho(g) / sum_g rho(g)

Outputs a multi-page PDF and a long-format TSV with one row per
(subject, ROI, voxel, condition) of base/predicted/observed positions.

Usage
-----
    python -m retsupp.visualize.predict_shifts_from_af \\
        --fits derivatives/af_prf_joint_poc/sub-02/sub-02_roi-V3AB_*.pkl \\
        --conditionfit-model 4 \\
        --out notes/predict_shifts.pdf
"""
from __future__ import annotations

import argparse
import glob
import pickle
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from nilearn import input_data, masking

from retsupp.utils.data import Subject, distractor_locations


CONDITIONS = ['upper_right', 'upper_left', 'lower_left', 'lower_right']
RING_KEYS = ['upper right', 'upper left', 'lower left', 'lower right']


def get_ring_positions():
    return np.array([list(distractor_locations[k]) for k in RING_KEYS],
                    dtype=np.float32)


def parse_fit_path(p: Path):
    """Recover (subject, roi, mode) from filename pattern
    ``sub-XX_roi-RR_mode-MM_af-prf-fit.pkl``."""
    m = re.search(r'sub-(\d+)_roi-([^_]+)_mode-(\w+?)_', p.name)
    if not m:
        raise ValueError(f'Cannot parse {p.name}')
    return int(m.group(1)), m.group(2), m.group(3)


def predict_centers(fit_pars: pd.DataFrame, shared: dict, mode: str,
                     resolution: int = 81, grid_radius: float = 5.0):
    """Compute predicted observed PRF centers for each (voxel, condition).

    Returns
    -------
    pred : ndarray (V, C, 2)  — predicted (x, y) per voxel × condition
    base : ndarray (V, 2)     — base PRF (x, y) from fit_pars
    """
    sign = +1 if mode == 'attraction' else -1   # +1 attraction, -1 suppression
    sigma_AF = float(shared['sigma_AF'])
    g_HP = float(shared['g_HP'])
    g_LP = float(shared['g_LP'])

    g1d = np.linspace(-grid_radius, grid_radius, resolution).astype(np.float32)
    GX, GY = np.meshgrid(g1d, g1d)
    G = np.stack([GX.ravel(), GY.ravel()], axis=1)            # (R^2, 2)

    ring = get_ring_positions()                                # (4, 2)
    # A_ell(g) = exp(-||g - mu_ell||^2 / (2 sigma_AF^2))       — peak 1
    A = np.exp(
        -np.sum((G[:, None, :] - ring[None, :, :]) ** 2, axis=-1)
        / (2.0 * sigma_AF ** 2)
    )                                                          # (R^2, 4)

    # M_C(g) = 1 + sign * ( g_HP * A_HP + g_LP * sum_{ell!=HP} A_ell )
    M = np.empty((G.shape[0], len(CONDITIONS)), dtype=np.float32)
    for ci in range(len(CONDITIONS)):
        a_hp = A[:, ci]
        a_lp = A.sum(axis=1) - a_hp
        M[:, ci] = 1.0 + sign * (g_HP * a_hp + g_LP * a_lp)

    # Voxel PRFs: S_v(g) = exp(-||g - (x, y)||^2 / (2 sd^2))
    x = fit_pars['x'].values.astype(np.float32)
    y = fit_pars['y'].values.astype(np.float32)
    sd = fit_pars['sd'].values.astype(np.float32)
    V = len(x)

    # Distance^2 from each grid point to each voxel center.  (R^2, V)
    d2 = (
        (G[:, 0:1] - x[None, :]) ** 2
        + (G[:, 1:2] - y[None, :]) ** 2
    )
    S = np.exp(-d2 / (2.0 * sd[None, :] ** 2))                  # (R^2, V)

    pred = np.empty((V, len(CONDITIONS), 2), dtype=np.float32)
    for ci in range(len(CONDITIONS)):
        rho = S * M[:, ci:ci + 1]                              # (R^2, V)
        # Clip negatives — modulation may dip below 0 at high gains.
        rho = np.clip(rho, 0.0, None)
        Z = rho.sum(axis=0) + 1e-12
        pred[:, ci, 0] = (G[:, 0:1] * rho).sum(axis=0) / Z
        pred[:, ci, 1] = (G[:, 1:2] * rho).sum(axis=0) / Z

    base = np.stack([x, y], axis=1)
    return pred, base


def project_onto_hp(shift_xy, hp_xy):
    """Project a (V, 2) shift vector onto the unit vector pointing AWAY from HP."""
    hp = np.asarray(hp_xy, dtype=np.float32)
    u = -hp / (np.linalg.norm(hp) + 1e-12)   # away-from-HP
    return shift_xy @ u


def load_observed_centers(sub: Subject, voxel_indices: np.ndarray,
                          model: int = 4):
    """Read conditionwise (x, y) for the given voxel indices.

    Returns
    -------
    obs : ndarray (V, C, 2)  — (x, y) per voxel × condition in CONDITIONS order
    obs_r2 : ndarray (V, C)  — r2 per condition
    """
    df = sub.get_prf_parameters_volume(model=model, type='conditionwise')
    masker = sub.get_bold_mask(return_masker=True)
    masker.fit()

    obs = np.full((len(voxel_indices), len(CONDITIONS), 2), np.nan,
                   dtype=np.float32)
    obs_r2 = np.full((len(voxel_indices), len(CONDITIONS)), np.nan,
                     dtype=np.float32)

    for ci, cond in enumerate(CONDITIONS):
        for pi, par in enumerate(['x', 'y']):
            # ensure_finite=False so mark_invalid_fits NaN sentinels survive.
            arr = masking.apply_mask(df.loc[cond, par], masker.mask_img_,
                                      ensure_finite=False)
            obs[:, ci, pi] = arr[voxel_indices]
        if 'r2' in df.columns:
            r2_arr = masking.apply_mask(df.loc[cond, 'r2'], masker.mask_img_,
                                         ensure_finite=False)
            obs_r2[:, ci] = r2_arr[voxel_indices]
    return obs, obs_r2


def plot_predicted_vs_observed(records: pd.DataFrame, out_pdf: Path):
    """Multi-page PDF: per-(subject, ROI) scatter and vector fields."""
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out_pdf) as pdf:
        # ---- Page 1: cover summary table.
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis('off')
        summary = records.groupby(['subject', 'roi', 'mode']).agg(
            n_vox=('voxel_idx', 'nunique'),
            sigma_AF=('sigma_AF', 'first'),
            g_HP=('g_HP', 'first'),
            g_LP=('g_LP', 'first'),
            r_pred_obs=('proj_pred', lambda s: np.corrcoef(
                s.values, records.loc[s.index, 'proj_obs'].values)[0, 1]),
        ).reset_index()
        ax.text(0.5, 0.95, 'AF-predicted vs observed PRF shifts',
                ha='center', va='top', fontsize=18, weight='bold',
                transform=ax.transAxes)
        ax.text(0.04, 0.88, summary.round(3).to_string(index=False),
                family='monospace', fontsize=9, va='top',
                transform=ax.transAxes)
        body = (
            'For each (sub, ROI):\n'
            '  proj_pred = AF-predicted shift projected onto away-from-HP unit vector\n'
            '  proj_obs  = observed (conditionwise PRF center − base) projected on same axis\n'
            'Positive proj = shift AWAY from the HP location (suppression).\n'
            'r_pred_obs    = Pearson r across all (voxel × condition) rows.\n'
        )
        ax.text(0.04, 0.55, body, family='monospace', fontsize=9, va='top',
                transform=ax.transAxes)
        pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)

        # ---- One page per (subject, ROI).
        for (sub_id, roi, mode), g in records.groupby(['subject', 'roi', 'mode']):
            fig = plt.figure(figsize=(13, 9))
            fig.suptitle(f'sub-{sub_id:02d}  ROI={roi}  mode={mode}  '
                         f'σ_AF={g["sigma_AF"].iloc[0]:.2f}  '
                         f'g_HP={g["g_HP"].iloc[0]:.3f}  '
                         f'g_LP={g["g_LP"].iloc[0]:.3f}',
                         fontsize=12)

            # (a) scatter: predicted vs observed projection.
            ax = fig.add_subplot(2, 3, 1)
            x = g['proj_pred'].values
            y = g['proj_obs'].values
            mask = np.isfinite(x) & np.isfinite(y)
            ax.scatter(x[mask], y[mask], s=4, alpha=0.3)
            lim = np.nanpercentile(np.abs(np.r_[x[mask], y[mask]]), 99)
            lim = max(lim, 0.1)
            ax.plot([-lim, lim], [-lim, lim], 'k:', lw=0.8)
            ax.axhline(0, color='gray', lw=0.5)
            ax.axvline(0, color='gray', lw=0.5)
            ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
            ax.set_xlabel('predicted projection (away-from-HP, °)')
            ax.set_ylabel('observed projection (°)')
            r = np.corrcoef(x[mask], y[mask])[0, 1] if mask.sum() > 1 else np.nan
            ax.set_title(f'predicted vs observed   r={r:.3f}')

            # (b–e) per-condition vector field.  4 small panels, base→predicted/observed arrows.
            for ci, cond in enumerate(CONDITIONS):
                ax = fig.add_subplot(2, 3, 2 + ci if ci < 4 else 6)
                gi = g[g['condition'] == cond]
                if len(gi) == 0:
                    continue
                bx = gi['base_x'].values; by = gi['base_y'].values
                px = gi['pred_x'].values; py = gi['pred_y'].values
                ox = gi['obs_x'].values; oy = gi['obs_y'].values
                # subsample to ≤200 voxels for clarity
                idx = np.arange(len(gi))
                if len(idx) > 200:
                    idx = np.random.RandomState(0).choice(idx, 200, replace=False)
                # observed shift (light gray) and predicted shift (red).
                for j in idx:
                    if np.isfinite(ox[j]) and np.isfinite(oy[j]):
                        ax.plot([bx[j], ox[j]], [by[j], oy[j]],
                                color='0.6', lw=0.4, alpha=0.4)
                for j in idx:
                    if np.isfinite(px[j]) and np.isfinite(py[j]):
                        ax.plot([bx[j], px[j]], [by[j], py[j]],
                                color='C3', lw=0.4, alpha=0.5)
                ax.scatter(bx[idx], by[idx], s=1, color='k', alpha=0.4)
                # mark HP
                hp = distractor_locations[cond.replace('_', ' ')]
                ax.plot(hp[0], hp[1], marker='*', markersize=14,
                        color='C0', mec='k')
                ax.set_xlim(-5, 5); ax.set_ylim(-5, 5)
                ax.set_aspect('equal')
                ax.axhline(0, color='gray', lw=0.3); ax.axvline(0, color='gray', lw=0.3)
                ax.set_title(f'{cond}  (gray=obs, red=pred)', fontsize=9)
                ax.set_xlabel('x (°)'); ax.set_ylabel('y (°)')

            pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)

            # Per-condition summary bar chart.
            fig, ax = plt.subplots(figsize=(7, 4))
            agg = (g.groupby('condition')[['proj_pred', 'proj_obs']]
                    .mean().reindex(CONDITIONS))
            xpos = np.arange(len(CONDITIONS))
            ax.bar(xpos - 0.2, agg['proj_pred'], width=0.4, color='C3',
                   label='predicted')
            ax.bar(xpos + 0.2, agg['proj_obs'], width=0.4, color='0.5',
                   label='observed')
            ax.axhline(0, color='k', lw=0.5)
            ax.set_xticks(xpos); ax.set_xticklabels(CONDITIONS, rotation=20)
            ax.set_ylabel('mean projection away from HP (°)')
            ax.set_title(f'sub-{sub_id:02d} {roi} — mean shift per condition')
            ax.legend()
            pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)

    print(f'Wrote {out_pdf} ({records["subject"].nunique()} subj × '
          f'{records["roi"].nunique()} ROI)')


def process_one_fit(fit_pkl: Path, sub_id: int, roi: str, mode: str,
                    bids_folder: str, conditionfit_model: int = 4):
    print(f'-- sub-{sub_id:02d} ROI={roi} mode={mode} --')
    with open(fit_pkl, 'rb') as f:
        d = pickle.load(f)
    fit_pars: pd.DataFrame = d['fit_pars']
    shared: dict = d['shared_pars']
    voxel_indices: np.ndarray = d['voxel_mask_indices']

    pred, base = predict_centers(fit_pars, shared, mode)
    sub = Subject(sub_id, bids_folder)
    obs, obs_r2 = load_observed_centers(sub, voxel_indices,
                                          model=conditionfit_model)

    rows = []
    ring = get_ring_positions()
    for ci, cond in enumerate(CONDITIONS):
        hp_xy = ring[ci]
        proj_pred = project_onto_hp(pred[:, ci, :] - base, hp_xy)
        proj_obs = project_onto_hp(obs[:, ci, :] - base, hp_xy)
        for vi in range(len(voxel_indices)):
            rows.append(dict(
                subject=sub_id, roi=roi, mode=mode, condition=cond,
                voxel_idx=int(voxel_indices[vi]),
                base_x=float(base[vi, 0]), base_y=float(base[vi, 1]),
                pred_x=float(pred[vi, ci, 0]), pred_y=float(pred[vi, ci, 1]),
                obs_x=float(obs[vi, ci, 0]), obs_y=float(obs[vi, ci, 1]),
                obs_r2=float(obs_r2[vi, ci]),
                proj_pred=float(proj_pred[vi]),
                proj_obs=float(proj_obs[vi]),
                sigma_AF=float(shared['sigma_AF']),
                g_HP=float(shared['g_HP']),
                g_LP=float(shared['g_LP']),
            ))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fits', nargs='+', required=True,
                        help='Glob/paths to AF+PRF fit pickles')
    parser.add_argument('--bids-folder', default='/data/ds-retsupp')
    parser.add_argument('--conditionfit-model', type=int, default=4)
    parser.add_argument('--out', type=Path,
                        default=Path('notes/predict_shifts.pdf'))
    parser.add_argument('--out-tsv', type=Path,
                        default=Path('notes/predict_shifts.tsv'))
    args = parser.parse_args()

    all_paths = []
    for spec in args.fits:
        all_paths += [Path(p) for p in glob.glob(spec)]
    if not all_paths:
        raise SystemExit(f'No fits matched: {args.fits}')

    all_rows = []
    for p in all_paths:
        sub_id, roi, mode = parse_fit_path(p)
        try:
            df = process_one_fit(p, sub_id, roi, mode, args.bids_folder,
                                  conditionfit_model=args.conditionfit_model)
            all_rows.append(df)
        except Exception as e:
            print(f'  ERROR {p.name}: {e}')
    if not all_rows:
        raise SystemExit('No fits processed successfully.')
    records = pd.concat(all_rows, ignore_index=True)
    args.out_tsv.parent.mkdir(parents=True, exist_ok=True)
    records.to_csv(args.out_tsv, sep='\t', index=False)
    print(f'Wrote {args.out_tsv} ({len(records)} rows)')

    plot_predicted_vs_observed(records, args.out)


if __name__ == '__main__':
    main()
