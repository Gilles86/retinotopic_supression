"""Bar-pass-locked mean time series for voxels whose pRF sits on an item location, by where HP is.

Per subject and ROI, on the full-data nested fits (pRF positions from m7_full; no gain-field model used):

1. Voxels: cross-run reliability > 0 and a pRF centre within 2° of a ring location (nearest = the
   voxel's location). No R² criterion.
2. Bar passes: per TR the centre of the bar (centroid of the bar-only stimulus). A pass is the TR at
   which the bar centre crosses the pRF centre along the bar's direction of motion (x for a vertical bar
   moving sideways, y for a horizontal bar moving up or down).
3. Each pass is classed by where the run's HP location is relative to the voxel's location: the same
   location (hp), the opposite one, or one of the two adjacent ones.
4. Epochs of the cleaned BOLD (% signal) from -4 to +12 TRs around the pass, averaged over passes per
   voxel, then over voxels. The search items are not removed: their responses add to every class, with
   the distractor more often at the HP location.

Writes ``notes/data/fig_bar_pass_average/sub-XX_2026-11.tsv`` (one row per ROI × class × lag).

Usage::

    python -m retsupp.visualize.paper.fig_bar_pass_average 23
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from retsupp.visualize.paper.fig3_example_voxels import load_fit

ROIS = ['V1', 'V2', 'V3', 'V3AB', 'IPS', 'hV4', 'VO', 'LO', 'TO']
PRE, POST = 4, 12
OPPOSITE = {0: 2, 1: 3, 2: 0, 3: 1}


def bar_centre(par_bar, grid, run):
    bb = np.asarray(par_bar)[run.start:run.start + run.n_T].reshape(run.n_T, -1)
    w = bb.sum(1)
    with np.errstate(invalid='ignore', divide='ignore'):
        return (bb @ grid[:, 0]) / w, (bb @ grid[:, 1]) / w


def crossings(pos, other, centres):
    """TRs where ``pos`` crosses each centre while the bar moves along this axis. Returns (t, voxel) pairs."""
    d, o = np.diff(pos), np.diff(other)
    moving = np.isfinite(d) & np.isfinite(o) & (np.abs(d) > np.abs(o))
    s = np.sign(pos[:, None] - centres[None, :])
    hit = (s[1:] != s[:-1]) & moving[:, None] & np.isfinite(s[1:]) & np.isfinite(s[:-1])
    t, v = np.nonzero(hit)
    return t + 1, v


def main(subject, rois, bids, out_dir):
    from retsupp.modeling.fit_af_prf_cv_v2 import build_data_and_paradigm, get_ring_positions
    from retsupp.modeling.fit_cv_ladder_nested import bar_paradigm
    from retsupp.utils.data import Subject
    deriv = Path(bids) / 'derivatives'
    sub = Subject(subject, bids)
    bold, par, ci, di, ti, grid, _, meta = build_data_and_paradigm(sub, resolution=50, grid_radius=5.0, item_shape='rectangle')
    par_bar = bar_paradigm(sub, meta, 50, 5.0)
    grid = np.asarray(grid)
    ring = get_ring_positions()
    rel_all = pd.concat([pd.read_csv(f, sep='\t') for f in sorted(
        (deriv / 'prf_cv_roi_reliability' / f'sub-{subject:02d}').glob('*_reliability.tsv.gz'))])
    rel_all = rel_all.groupby('voxel_index').reliability.mean()
    centres = [bar_centre(par_bar, grid, r) for r in meta]
    lags = np.arange(-PRE, POST + 1)
    rows = []
    for roi in rois:
        try:
            fit = load_fit(deriv, 'm7_full', subject, roi)[0]
        except FileNotFoundError as e:
            print('skip', roi, e, flush=True)
            continue
        ids = np.asarray(fit['voxel_ids'])
        p = fit['fit_pars_per_fold'][0]
        x, y = p.x.to_numpy(), p.y.to_numpy()
        dist = np.hypot(x[:, None] - ring[:, 0], y[:, None] - ring[:, 1])
        loc_v = dist.argmin(1)
        ok = (rel_all.reindex(ids).to_numpy() > 0) & (dist.min(1) < 2)
        vox = np.flatnonzero(ok)
        print(f'{roi}: {len(vox)} of {len(ids)} voxels', flush=True)
        if not len(vox):
            continue
        y_all = bold[:, ids[vox]]
        acc = {}
        for r, (bx, by) in zip(meta, centres):
            hp = int(np.argmax(ci[r.start]))
            for pos, other, c in ((bx, by, x[vox]), (by, bx, y[vox])):
                t, v = crossings(pos, other, c)
                keep = (t >= PRE) & (t < r.n_T - POST)
                for tt, vv in zip(t[keep], v[keep]):
                    loc = loc_v[vox[vv]]
                    cls = 'hp' if loc == hp else 'opposite' if OPPOSITE[loc] == hp else 'adjacent'
                    ep = y_all[r.start + tt + lags, vv]
                    a = acc.setdefault((cls, vv), [np.zeros(len(lags)), 0])
                    a[0] += ep
                    a[1] += 1
        for cls in ('hp', 'adjacent', 'opposite'):
            per_vox = [s / n for (c, vv), (s, n) in acc.items() if c == cls]
            if not per_vox:
                continue
            m = np.mean(per_vox, 0)
            n_pass = sum(n for (c, vv), (s, n) in acc.items() if c == cls)
            for lag, val in zip(lags, m):
                rows.append(dict(subject=subject, roi=roi, cls=cls, lag=int(lag), t=float(lag) * 1.6, data=val,
                                 n_voxels=len(per_vox), n_passes=n_pass))
    out = Path(out_dir) / 'fig_bar_pass_average'
    out.mkdir(parents=True, exist_ok=True)
    path = out / f'sub-{subject:02d}_2026-11.tsv'
    pd.DataFrame(rows).to_csv(path, sep='\t', index=False)
    print('Saved', path, flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('subject', type=int)
    ap.add_argument('--rois', nargs='+', default=ROIS)
    ap.add_argument('--bids-folder', default='/shares/zne.uzh/gdehol/ds-retsupp')
    ap.add_argument('--out-dir', default='/shares/zne.uzh/gdehol/ds-retsupp/derivatives/paper_2026-11')
    a = ap.parse_args()
    main(a.subject, a.rois, a.bids_folder, a.out_dir)
