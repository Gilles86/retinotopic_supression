"""Scan voxels and windows for the Figure 3c example where m7_full visibly improves on m0_prf.

Selected for illustration; the criteria, which go in the legend, are:

1. Fitted voxels of the full-data nested fits (m0_prf, m7_full) with cross-run reliability > 0, a pRF
   centre within 2° of a ring location, and m7 R² > ``--min-r2`` over the concatenation of all runs.
2. Every window of ``--window`` s (one-TR steps, inside one run) that contains at least
   ``--min-events`` target or distractor onsets at that ring location, and where m7 explains the
   window itself with R² ≥ ``--min-local-r2`` (SST about the window mean).
3. Score per window: local ΔR² = (SSE_m0 − SSE_m7) / SST_window. The best window per voxel and the
   best voxel per (subject, ROI) are kept. Also recorded: ΔSSE, the peak |m7 − m0| in the window
   relative to the voxel's BOLD SD over all runs, and the whole-data ΔR² of m7 over m0.

Writes ``notes/data/fig3_voxel_scan/sub-XX_ROI_2026-11.tsv`` (best window of the 10 best voxels;
first row is the pick) and ``notes/data/fig3_example_voxel_scan_sub-XX_ROI_2026-11.tsv`` (the pick, in
the format of ``fig3_example_voxels.py`` that ``plot_fig3_v2.py --voxel-table`` renders).

Usage::

    python -m retsupp.visualize.paper.fig3_scan_voxels 23 V3
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from retsupp.visualize.paper.fig3_example_voxels import load_fit


def window_sums(a, n):
    """Sums of ``a`` (T, V) over every window of n rows: (T - n + 1, V)."""
    c = np.vstack([np.zeros((1, a.shape[1])), np.cumsum(a, 0)])
    return c[n:] - c[:-n]


def onsets(x):
    on = x > 0
    return on & ~np.concatenate([[False], on[:-1]])


def main(subject, roi, bids, window, out_dir, min_r2, min_events, min_local_r2, n_top):
    from braincoder.hrf import SPMHRFModel
    from retsupp.modeling.fit_af_prf_cv_v2 import build_data_and_paradigm, build_model, get_ring_positions
    from retsupp.modeling.fit_cv_ladder_nested import bar_paradigm, sign_pattern_of
    from retsupp.utils.data import Subject
    deriv = Path(bids) / 'derivatives'
    sub = Subject(subject, bids)
    bold, par, ci, di, ti, grid, _, meta = build_data_and_paradigm(sub, resolution=50, grid_radius=5.0, item_shape='rectangle')
    par_bar = bar_paradigm(sub, meta, 50, 5.0)
    ring = get_ring_positions()
    hrf = SPMHRFModel(tr=sub.get_tr(session=1, run=1), delay=4.5, dispersion=0.75)
    preds, ids, pars7, roots = {}, None, None, {}
    for name in ('m0_prf', 'm7_full'):
        fit, root = load_fit(deriv, name, subject, roi)
        p = fit['fit_pars_per_fold'][0]
        if ids is not None:
            assert np.array_equal(ids, np.asarray(fit['voxel_ids'])), 'm0 and m7 voxel sets differ'
        ids = np.asarray(fit['voxel_ids'])
        model = build_model(grid_coords=grid, paradigm=par, condition_indicator=ci, dynamic_indicator=di, target_indicator=ti,
                            ring_positions=ring, hrf_model=hrf, sign_pattern=sign_pattern_of(name), item_scale='free')
        preds[name] = np.asarray(model.predict(paradigm=par, parameters=p.reindex(columns=list(model.parameter_labels))))
        roots[name] = root
        pars7 = p  # pRF columns of m7, as in fig3_example_voxels
        print(name, root, preds[name].shape, flush=True)
    y = bold[:, ids]
    p0, p7 = preds['m0_prf'], preds['m7_full']
    sst = ((y - y.mean(0)) ** 2).sum(0)
    r2_m7 = 1 - ((y - p7) ** 2).sum(0) / sst
    r2_m0 = 1 - ((y - p0) ** 2).sum(0) / sst
    sd = y.std(0)
    rel = pd.concat([pd.read_csv(f, sep='\t') for f in sorted((deriv / 'prf_cv_roi_reliability' / f'sub-{subject:02d}').glob('*_reliability.tsv.gz'))])
    rel = rel.groupby('voxel_index').reliability.mean().reindex(ids).to_numpy()
    dist = np.hypot(pars7.x.to_numpy()[:, None] - ring[:, 0], pars7.y.to_numpy()[:, None] - ring[:, 1])
    loc_v = dist.argmin(1)
    ok = (rel > 0) & (dist.min(1) < 2) & (r2_m7 > min_r2)
    print(f'{ok.sum()} of {len(ids)} voxels pass reliability > 0, pRF < 2° from ring, m7 R² > {min_r2}', flush=True)
    if not ok.any():
        raise ValueError('no voxel passes')
    cand = np.where(ok)[0]
    n = int(window / 1.6)
    rows = []
    for r in meta:
        sl = slice(r.start, r.start + r.n_T)
        yy, a0, a7 = y[sl][:, cand], p0[sl][:, cand], p7[sl][:, cand]
        sse0 = window_sums((yy - a0) ** 2, n)
        sse7 = window_sums((yy - a7) ** 2, n)
        s1, s2 = window_sums(yy, n), window_sums(yy ** 2, n)
        sstw = s2 - s1 ** 2 / n
        peak = np.lib.stride_tricks.sliding_window_view(np.abs(a7 - a0), n, axis=0).max(-1) / sd[cand]
        ev_t = np.stack([onsets(ti[sl, k]) for k in range(len(ring))], 1).astype(float)
        ev_d = np.stack([onsets(di[sl, k]) for k in range(len(ring))], 1).astype(float)
        nt, nd = window_sums(ev_t, n)[:, loc_v[cand]], window_sums(ev_d, n)[:, loc_v[cand]]
        dr2 = (sse0 - sse7) / sstw
        lr7, lr0 = 1 - sse7 / sstw, 1 - sse0 / sstw
        valid = ((nt + nd) >= min_events) & (lr7 >= min_local_r2)
        score = np.where(valid, dr2, -np.inf)
        best = score.argmax(0)
        j = np.arange(len(cand))
        rows.append(pd.DataFrame(dict(
            vi=cand, session=r.session, run=r.run, hp=r.hp, start_tr=best, start_s=best * 1.6, run_start=r.start,
            local_dr2=score[best, j], dsse=(sse0 - sse7)[best, j], local_r2_m7=lr7[best, j], local_r2_m0=lr0[best, j],
            peak_diff_sd=peak[best, j], n_target_here=nt[best, j], n_singleton_here=nd[best, j],
            n_valid_windows=valid.sum(0), frac_valid_m7_better=np.where(valid, dr2 > 0, False).sum(0) / np.maximum(valid.sum(0), 1))))
    w = pd.concat(rows)
    w = w[np.isfinite(w.local_dr2)]
    if w.empty:
        raise ValueError('no window passes')
    w = w.sort_values('local_dr2', ascending=False).drop_duplicates('vi')
    vi = w.vi.to_numpy()
    w = w.assign(subject=subject, roi=roi, voxel_id=ids[vi], x=pars7.x.to_numpy()[vi], y=pars7.y.to_numpy()[vi], sd=pars7.sd.to_numpy()[vi],
                 r2_m7=r2_m7[vi], r2_m0=r2_m0[vi], global_dr2=r2_m7[vi] - r2_m0[vi], reliability=rel[vi],
                 ring_x=ring[loc_v[vi], 0], ring_y=ring[loc_v[vi], 1], n_candidates=len(cand), root=roots['m7_full'])
    Path(out_dir, 'fig3_voxel_scan').mkdir(parents=True, exist_ok=True)
    top = w.head(n_top).drop(columns='vi')
    path = Path(out_dir) / 'fig3_voxel_scan' / f'sub-{subject:02d}_{roi}_2026-11.tsv'
    top.to_csv(path, sep='\t', index=False)
    print('Saved', path, flush=True)
    b = w.iloc[0]
    v, loc = int(b.vi), loc_v[int(b.vi)]
    s = int(b.run_start + b.start_tr)
    e = s + n
    gx, gy = grid[:, 0], grid[:, 1]
    bb = np.asarray(par_bar)[s:e].reshape(e - s, -1)
    bw = bb.sum(1)
    with np.errstate(invalid='ignore', divide='ignore'):
        bx, by = (bb @ gx) / bw, (bb @ gy) / bw
    out = pd.DataFrame(dict(t=b.start_s + np.arange(n) * 1.6, bold=y[s:e, v], m0=p0[s:e, v], m7=p7[s:e, v],
                            target_here=ti[s:e, loc], singleton_here=di[s:e, loc], target_any=ti[s:e].sum(1), singleton_any=di[s:e].sum(1),
                            bar_x=bx, bar_y=by))
    for k, val in dict(subject=subject, roi=roi, voxel_id=ids[v], x=b.x, y=b.y, sd=b.sd, r2_m7=b.r2_m7, reliability=b.reliability,
                       ring_x=b.ring_x, ring_y=b.ring_y, hp=b.hp, session=b.session, run=b.run).items():
        out[k] = val
    path = Path(out_dir) / f'fig3_example_voxel_scan_sub-{subject:02d}_{roi}_2026-11.tsv'
    out.to_csv(path, sep='\t', index=False)
    print('Saved', path, f'voxel {ids[v]}, R² {b.r2_m7:.3f}, run {b.session}-{b.run} t={b.start_s:.1f}s, local ΔR² {b.local_dr2:.3f}, '
          f'peak {b.peak_diff_sd:.2f} SD', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('subject', type=int)
    ap.add_argument('roi')
    ap.add_argument('--bids-folder', default='/shares/zne.uzh/gdehol/ds-retsupp')
    ap.add_argument('--window', type=float, default=75.)
    ap.add_argument('--out-dir', default='notes/data')
    ap.add_argument('--min-r2', type=float, default=.2)
    ap.add_argument('--min-events', type=int, default=2)
    ap.add_argument('--min-local-r2', type=float, default=.3)
    ap.add_argument('--n-top', type=int, default=10)
    a = ap.parse_args()
    main(a.subject, a.roi, a.bids_folder, a.window, a.out_dir, a.min_r2, a.min_events, a.min_local_r2, a.n_top)
