"""HP-vs-LP decoded-mass scatter (plan §5c).

For each (subject, ROI):

1. Load all per-run decoded npzs.
2. Compute the time-averaged decoded mass inside a small disk at
   each of the 4 ring locations (using
   :func:`retsupp.decode.decoder.sample_at_ring_positions`).
3. Average mass at the run's HP location (``mass_HP``) vs the mean of
   the three non-HP locations (``mass_LP``).
4. One point per (subject, ROI). Identity line + below-line = HP
   suppression.

Output: ``notes/figures/decoding/hp_vs_lp_decoded_model{M}.pdf``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from retsupp.utils.data import Subject, distractor_locations
from retsupp.decode.aggregate_decoded import _run_npz_path


def _sample_disk(decoded_mean: np.ndarray, gx: np.ndarray, gy: np.ndarray,
                  cx: float, cy: float, radius: float = 0.4) -> float:
    """Mean of ``decoded_mean`` inside a disk at (cx, cy)."""
    d2 = (gx - cx) ** 2 + (gy - cy) ** 2
    mask = d2 <= radius ** 2
    if not mask.any():
        # Fall back to the single nearest grid point.
        ii = int(np.argmin(d2.ravel()))
        return float(decoded_mean.ravel()[ii])
    return float(decoded_mean[mask].mean())


def per_subject_roi_mass(bids: Path, roi: str, model: int,
                          subjects: list,
                          drop_pre_tr: int = 4,
                          ring_disk_radius: float = 0.4) -> pd.DataFrame:
    """Per (subject, ROI) HP and LP decoded mass."""
    rows = []
    for subject in subjects:
        sub = Subject(subject, bids_folder=str(bids))
        hp_by = sub.get_hpd_locations()
        # Accumulate per-ring across all 12 runs (each run's HP differs).
        per_run = []
        for (ses, run), hp_loc in hp_by.items():
            p = _run_npz_path(bids, subject, model, roi, ses, run)
            if not p.exists():
                continue
            with np.load(p) as d:
                decoded = d['decoded'].astype(np.float32)
                grid = d['grid'].astype(np.float32)
            decoded = decoded[drop_pre_tr:]
            mean_map = decoded.mean(axis=0)
            R = mean_map.shape[0]
            gx = grid[:, 0].reshape(R, R)
            gy = grid[:, 1].reshape(R, R)
            mass_at = {}
            for name, (cx, cy) in distractor_locations.items():
                # distractor_locations uses space-form keys; normalise to
                # underscore so HP comparison works.
                key = name.replace(' ', '_')
                mass_at[key] = _sample_disk(mean_map, gx, gy, cx, cy,
                                              radius=ring_disk_radius)
            if hp_loc not in mass_at:
                continue
            lp_mean = np.mean([v for k, v in mass_at.items() if k != hp_loc])
            per_run.append({'mass_HP': mass_at[hp_loc],
                            'mass_LP': lp_mean})
        if not per_run:
            continue
        df = pd.DataFrame(per_run)
        rows.append({
            'subject': subject,
            'roi': roi,
            'n_runs': len(per_run),
            'mass_HP': df['mass_HP'].mean(),
            'mass_LP': df['mass_LP'].mean(),
        })
    return pd.DataFrame(rows)


def plot_hp_vs_lp(bids_folder: str, rois: list, model: int = 4,
                   subjects: list | None = None,
                   out_path: str | Path = 'notes/figures/decoding/hp_vs_lp_decoded_model4.pdf',
                   ring_disk_radius: float = 0.4):
    bids = Path(bids_folder)
    if subjects is None:
        from retsupp.data import load_subjects
        subjects = sorted(int(s) for s in load_subjects())

    dfs = [per_subject_roi_mass(bids, roi, model, subjects,
                                  ring_disk_radius=ring_disk_radius)
           for roi in rois]
    df = pd.concat([d for d in dfs if not d.empty], ignore_index=True)
    if df.empty:
        raise RuntimeError('No per-run decoded npzs found across any ROI.')

    g = sns.FacetGrid(df, col='roi', col_wrap=4, height=2.6,
                       sharex=True, sharey=True)
    g.map_dataframe(sns.scatterplot, x='mass_HP', y='mass_LP',
                     s=22, alpha=0.7, color='#444')
    lim_lo = float(min(df['mass_HP'].min(), df['mass_LP'].min()))
    lim_hi = float(max(df['mass_HP'].max(), df['mass_LP'].max()))
    pad = 0.05 * (lim_hi - lim_lo + 1e-9)
    lim = (lim_lo - pad, lim_hi + pad)
    for ax in g.axes.flat:
        ax.plot(lim, lim, 'k--', lw=0.7, alpha=0.6)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_aspect('equal')
    g.set_axis_labels('Decoded mass at HP', 'Mean decoded mass at LPs')
    g.fig.suptitle(
        f'Per-subject decoded mass: HP vs LPs (m{model}); '
        f'below dashed = suppression at HP',
        fontsize=10, y=1.02)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    g.savefig(out, bbox_inches='tight')
    plt.close(g.fig)
    print(f'Wrote {out}')

    # Also dump the raw TSV for downstream Wilcoxon / re-plotting.
    tsv = out.with_suffix('.tsv')
    df.to_csv(tsv, sep='\t', index=False)
    print(f'Wrote {tsv}')


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--bids-folder', default='/data/ds-retsupp')
    p.add_argument('--rois', nargs='+',
                   default=['V1', 'V2', 'V3', 'V3AB', 'hV4', 'LO', 'TO', 'VO'])
    p.add_argument('--model', type=int, default=4)
    p.add_argument('--subjects', nargs='*', type=int, default=None)
    p.add_argument('--ring-disk-radius', type=float, default=0.4)
    p.add_argument('--out',
                   default='notes/figures/decoding/hp_vs_lp_decoded_model4.pdf')
    a = p.parse_args()
    plot_hp_vs_lp(
        bids_folder=a.bids_folder, rois=a.rois, model=a.model,
        subjects=a.subjects, ring_disk_radius=a.ring_disk_radius,
        out_path=a.out,
    )


if __name__ == '__main__':
    main()
