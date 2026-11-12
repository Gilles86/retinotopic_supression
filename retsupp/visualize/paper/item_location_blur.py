"""How much does a response at one item location leak into the response measured at the others (Fig. 3d, e)?

Resolution check for the location-type contrasts (HP, next to HP, opposite). Per voxel of the Fig. 3 selection (pRF
centre within 2° of one of the four ring locations; ladder voxels of ``--root``), the fitted DoG pRF (m7_full, all
data) is integrated over the footprint of each of the four item locations (disk, ``--item-radius``). The pRF's
weight (amplitude / sd, the model's 'legacy' scale, so voxels enter the pooled mean as they do in the data) is kept.
Per subject and pathway, pooled over areas and voxels as Fig. 3 (mean over voxels), with each voxel's own location as
'self':

    A_self, A_next (mean of the two neighbours, 5.7° away), A_opp (8° away).

For a modulation that is confined to the HP item, the measured location values are A_self, A_next, A_opp times that
modulation, so relative to the unblurred contrast (A_self):

    retained HP − opposite = 1 − A_opp / A_self,  retained HP − next = 1 − A_next / A_self,
    spurious next − opposite = (A_next − A_opp) / A_self   (a graded pattern that blur alone creates).

The 4 × 4 mixing is circulant, so the same factors hold for a modulation at any location (the reverse direction).
Writes ``--out`` (per subject × pathway) and prints the medians over subjects.

Usage::

    python -m retsupp.visualize.paper.item_location_blur \\
        --root /data/ds-retsupp/derivatives/af_prf_cv_ladder_nested_full_bounded_maxsd20_reliable_roihrf_polished
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from retsupp.visualize.paper.fig2_prf_fits import PATHWAY

ECC = 4.0
RING = ECC / np.sqrt(2) * np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]])   # upper right, upper left, lower left, lower right


def footprint(radius, step=0.05):
    g = np.arange(-radius, radius + step / 2, step)
    xx, yy = np.meshgrid(g, g)
    inside = xx ** 2 + yy ** 2 <= radius ** 2
    return xx[inside], yy[inside], step ** 2


def item_weights(p, radius, centre_only=False):
    """Voxels × 4: the DoG pRF (braincoder 'legacy' scale) integrated over each ring location's disk; ``centre_only``
    drops the surround (its σ is mostly tens of degrees, so it is a nearly flat negative offset)."""
    fx, fy, da = footprint(radius)
    out = np.zeros((len(p), 4))
    x, y, sd = p.x.to_numpy()[:, None], p.y.to_numpy()[:, None], p.sd.to_numpy()[:, None]
    amp = p.amplitude.to_numpy()[:, None]
    sa, ss = p.srf_amplitude.to_numpy()[:, None], p.srf_size.to_numpy()[:, None]
    if centre_only:
        sa = np.zeros_like(sa)
    for j, (cx, cy) in enumerate(RING):
        r2 = (cx + fx[None] - x) ** 2 + (cy + fy[None] - y) ** 2
        rf = np.exp(-r2 / (2 * sd ** 2)) - sa * np.exp(-r2 / (2 * (sd * ss) ** 2))
        out[:, j] = (amp / sd * rf).sum(1) * da
    return out


def voxel_table(root, subjects, rois, radius, max_dist=2.0, model='m7_full', centre_only=False):
    rows = []
    for s in subjects:
        for roi in rois:
            f = Path(root) / model / f'sub-{s:02d}' / f'sub-{s:02d}_roi-{roi}_cv-params.tsv'
            if not f.exists():
                continue
            p = pd.read_csv(f, sep='\t')
            p = p[p.fold == p.fold.min()]
            d = np.hypot(p.x.to_numpy()[:, None] - RING[:, 0], p.y.to_numpy()[:, None] - RING[:, 1])
            keep = d.min(1) < max_dist
            p, d = p[keep], d[keep]
            if not len(p):
                continue
            w = item_weights(p, radius, centre_only)
            loc = d.argmin(1)
            k = np.arange(len(p))
            rows.append(pd.DataFrame(dict(
                subject=s, roi=roi, voxel_id=p.voxel_id.to_numpy(), loc=loc, sd=p.sd.to_numpy(),
                fwhm=2.3548 * p.sd.to_numpy(), dist=d.min(1),
                a_self=w[k, loc], a_next=(w[k, (loc + 1) % 4] + w[k, (loc + 3) % 4]) / 2, a_opp=w[k, (loc + 2) % 4])))
    return pd.concat(rows, ignore_index=True)


def summarize(v):
    grp = {r: g for g, rs in PATHWAY.items() for r in rs}
    v = pd.concat([v.assign(grp=v.roi.map(grp)), v.assign(grp='All areas')])
    g = v.groupby(['subject', 'grp'])
    s = g[['a_self', 'a_next', 'a_opp']].mean()
    s['n_voxels'] = g.size()
    s['fwhm_median'] = g.fwhm.median()
    s['leak_next'] = s.a_next / s.a_self
    s['leak_opp'] = s.a_opp / s.a_self
    s['retained_hp_opp'] = 1 - s.leak_opp
    s['retained_hp_next'] = 1 - s.leak_next
    s['spurious_next_opp'] = s.leak_next - s.leak_opp
    return s.reset_index()


def main(root, out, subjects, radius):
    rois = [r for rs in PATHWAY.values() for r in rs]
    res = []
    for kind, co in (('dog', False), ('centre', True)):
        v = voxel_table(root, subjects, rois, radius, centre_only=co)
        res.append(summarize(v).assign(rf=kind))
        report(root, radius, kind, v, res[-1])
    s = pd.concat(res)
    s.to_csv(out, sep='\t', index=False, float_format='%.5g')
    print('Saved', out)
    return s


def report(root, radius, kind, v, s):
    cols = ['n_voxels', 'fwhm_median', 'leak_next', 'leak_opp', 'retained_hp_opp', 'retained_hp_next',
            'spurious_next_opp']
    print(f'{root}: {s.subject.nunique()} subjects, item radius {radius}°, pRF {kind}')
    print(s.groupby('grp')[cols].median().round(3).to_string())
    per_roi = v.groupby(['subject', 'roi'])[['a_self', 'a_next', 'a_opp']].mean()
    per_roi = (per_roi.a_next / per_roi.a_self).rename('leak_next').to_frame().assign(
        leak_opp=per_roi.a_opp / per_roi.a_self, fwhm=v.groupby(['subject', 'roi']).fwhm.median())
    print(per_roi.groupby('roi').median().round(3).to_string())


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', default='/data/ds-retsupp/derivatives/'
                    'af_prf_cv_ladder_nested_full_bounded_maxsd20_reliable_roihrf_polished')
    ap.add_argument('--out', default='notes/data/fig3_item_location_blur_n28_2026-11-12.tsv')
    ap.add_argument('--subjects', type=int, nargs='+', default=list(range(3, 31)))
    ap.add_argument('--item-radius', type=float, default=0.75, help='half the item size (1.5°)')
    a = ap.parse_args()
    main(a.root, a.out, a.subjects, a.item_radius)
