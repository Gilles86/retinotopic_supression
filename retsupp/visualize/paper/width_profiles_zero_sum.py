"""Are the field widths of the zero-sum main model identified? Width profiles from per-run sufficient statistics.

Input: ``fit_width_profiles`` run on the zero-sum main model (view root ``..._zs_lm_ref``, where m7_full is
m5_TDS_dS_zs), i.e. per subject x ROI and per run the least-squares statistics A, b, c of the gains of one field on a
width grid (0.5-8 deg, 17 log steps), every other parameter (pRFs, HRF, item weight, the other field) held at the fit.

Here the gains are re-solved in the main model's own parametrisation:

* sustained (sus): one gain along the zero-sum direction, g_HP = 3/4 dS, g_LP = -1/4 dS (so g_HP + 3 g_LP = 0);
* transient (dyn): target gain and one distractor gain (HP = LP, tied as in m5_TDS_dS_zs).

Per cell and field: the profile dSSE(sigma) = SSE(sigma) - min, its jackknife SE over runs (leave one run out,
scaled to all runs), the width at the minimum, and whether the minimum is interior or on the grid edge; dS at every
sustained width. A width is excluded when dSSE > 2 SE; a profile is "flat" when no grid width is excluded.

    python -m retsupp.visualize.paper.width_profiles_zero_sum --in-dir <dir with sub-XX/*_widthprof.npz> --out <tsv>
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

V_SUS = np.array([[0.75], [-0.25]])                      # (g_HP, g_LP) per unit dS
M_DYN = np.array([[1., 0.], [0., 1.], [0., 1.]])         # (g_T, g_D_HP, g_D_LP) from (g_T, g_D)


def profile(A, b, c, M):
    """A (G, R, k, k), b (G, R, k), c (G, R) -> SSE (G,), gains (G, m), and leave-one-run-out SSE (R, G)."""
    A2 = np.einsum('ki,grkl,lj->grij', M, A, M)
    b2 = np.einsum('ki,grk->gri', M, b)

    def solve(As, bs, cs):
        g = np.linalg.solve(As, bs[..., None])[..., 0]
        return cs - np.einsum('gi,gi->g', bs, g), g

    sse, gains = solve(A2.sum(1), b2.sum(1), c.sum(1))
    R = A.shape[1]
    loo = np.stack([solve(A2.sum(1) - A2[:, r], b2.sum(1) - b2[:, r], c.sum(1) - c[:, r])[0] for r in range(R)])
    return sse, gains, loo


def summarise(grid, sse, loo, fitted):
    d = sse - sse.min()
    R = loo.shape[0]
    d_loo = (loo - loo.min(1, keepdims=True)) * R / (R - 1)
    se = np.sqrt((R - 1) / R * ((d_loo - d_loo.mean(0)) ** 2).sum(0))
    i = int(np.argmin(sse))
    excl = d > 2 * np.maximum(se, 1e-12)
    ok = grid[~excl]
    am_loo = grid[np.argmin(loo, axis=1)]
    return dict(argmin=grid[i], edge=('low' if i == 0 else 'high' if i == len(grid) - 1 else 'interior'),
                fitted=fitted, flat=not excl.any(), ci_lo=ok.min(), ci_hi=ok.max(),
                n_excluded=int(excl.sum()), dsse_lo=d[0], dsse_hi=d[-1], se_lo=se[0], se_hi=se[-1],
                loo_argmin_lo=am_loo.min(), loo_argmin_hi=am_loo.max())


def main(in_dir, out):
    rows, prof = [], []
    for f in sorted(glob.glob(str(Path(in_dir) / 'sub-*' / '*_widthprof.npz'))):
        z = np.load(f, allow_pickle=True)
        grid = z['grid']
        base = dict(subject=int(z['subject']), roi=str(z['roi']), n_voxels=int(z['n_voxels']))
        for field, M, wcol in (('sus', V_SUS, 'sigma_AF'), ('dyn', M_DYN, 'sigma_dyn')):
            sse, gains, loo = profile(z[f'{field}_A'], z[f'{field}_b'], z[f'{field}_c'], M)
            r = dict(base, field=field, **summarise(grid, sse, loo, float(z[wcol])))
            if field == 'sus':
                r['dS_at_min'] = gains[np.argmin(sse), 0]
                r['dS_min_over_grid'], r['dS_max_over_grid'] = gains[:, 0].min(), gains[:, 0].max()
                r['dS_fitted_width'] = gains[np.argmin(np.abs(np.log(grid / float(z[wcol])))), 0]
            else:
                r['gT_at_min'], r['gD_at_min'] = gains[np.argmin(sse)]
            rows.append(r)
            for s, e, g in zip(grid, sse - sse.min(), gains):
                prof.append(dict(base, field=field, sigma=s, dsse=e, **{f'gain{j}': v for j, v in enumerate(g)}))
    t = pd.DataFrame(rows)
    t.to_csv(out, sep='\t', index=False)
    pd.DataFrame(prof).to_csv(str(out).replace('.tsv', '_profiles.tsv'), sep='\t', index=False)
    for field, g in t.groupby('field'):
        print(f'\n== {field}: {len(g)} cells')
        print('  minimum on grid edge:', g.edge.value_counts().to_dict())
        print(f'  flat profiles (no width excluded at 2 SE): {g.flat.sum()} / {len(g)}')
        print(f'  median excluded widths: {g.n_excluded.median():.0f} of 17')
        if field == 'sus':
            spread = g.dS_max_over_grid - g.dS_min_over_grid
            print(f'  dS range over the whole width grid: median {spread.median():.3f}; '
                  f'dS sign constant over grid in {(np.sign(g.dS_min_over_grid) == np.sign(g.dS_max_over_grid)).mean():.0%}')
    print('\nflat share by area:')
    print(t.pivot_table(index='roi', columns='field', values='flat', aggfunc='mean').round(2).to_string())


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--in-dir', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    main(a.in_dir, a.out)
