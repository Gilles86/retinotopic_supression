"""Do the pRF centres and sizes make sense, per pRF model and against the R² mixture?

Per subject, ROI and model: the share of contralateral pRF centres at several R² cutoffs, at the
whole-brain tail-FDR threshold and at p_signal > .5 of the logit-R² mixture, and pRF size against
eccentricity. If unreliable centres fall on either side about equally, twice the ipsilateral
share estimates the false-positive share, to compare with the FDR the mixture promises.

Usage::

    python -m retsupp.visualize.paper.prf_model_sanity collect
    python -m retsupp.visualize.paper.prf_model_sanity plot
"""
from __future__ import annotations

import argparse
import multiprocessing as mp

import numpy as np
import pandas as pd

MODELS = {1: ('Gaussian', 'Fixed'), 2: ('DoG', 'Fixed'), 5: ('DN', 'Fixed'), 3: ('Gaussian', 'Flexible'), 4: ('DoG', 'Flexible')}
ROIS = {'V1': ['V1'], 'V2': ['V2'], 'V3': ['V3'], 'hV4': ['hV4'], 'V3AB': ['V3A', 'V3B'], 'IPS': ['IPS']}
CUTS = [.02, .05, .1, .2, .3]
APERTURE = 3.17
ECC = np.arange(0, 3.51, .5)
BIDS = '/data/ds-retsupp'
TAB, CURVE = 'notes/data/prf_model_sanity_contra.tsv', 'notes/data/prf_model_sanity_sigma.tsv'


def one(s):
    from braincoder.utils.stats import fit_and_classify, r2_fdr_threshold
    from retsupp.utils.data import Subject
    sub = Subject(s, BIDS)
    masker = sub.get_bold_mask(return_masker=True)
    masker.fit()
    hemi = {}
    for roi, comps in ROIS.items():
        for h in 'LR':
            m = None
            for c in comps:
                try:
                    x = masker.transform(sub.get_retinotopic_roi(f'{c}_{h}', bold_space=True)).ravel() > .5
                except Exception:
                    continue
                m = x if m is None else m | x
            if m is not None:
                hemi[roi, h] = m
    rows, curves = [], []
    for model in MODELS:
        try:
            p = pd.DataFrame(sub.get_prf_parameters_volume(model=model, return_images=False))
        except Exception as e:
            print('skip', s, model, type(e).__name__, flush=True)
            continue
        r2 = p.r2.to_numpy()
        real = np.isfinite(r2) & (r2 > 0) & (r2 < .99) & np.isfinite(p.sd.to_numpy())
        thr = float(r2_fdr_threshold(r2[real], alpha=.05))
        ps = np.zeros(len(p))
        ps[real] = fit_and_classify(r2[real])['p_signal']
        ecc = np.hypot(p.x, p.y).to_numpy()
        ok = real & (ecc <= APERTURE) & (p.sd.to_numpy() >= .1)
        for roi in ROIS:
            sel_all, contra_all = [], []
            for h in 'LR':
                if (roi, h) not in hemi:
                    continue
                ix = np.where(hemi[roi, h] & ok)[0]
                sel_all.append(ix)
                contra_all.append(p.x.to_numpy()[ix] > 0 if h == 'L' else p.x.to_numpy()[ix] < 0)
            if not sel_all:
                continue
            ix, con = np.concatenate(sel_all), np.concatenate(contra_all)
            for name, keep in [(f'{c:g}', r2[ix] >= c) for c in CUTS] + [('fdr', r2[ix] >= thr), ('p_signal', ps[ix] > .5)]:
                rows.append(dict(subject=s, model=model, roi=roi, cut=name, n=int(keep.sum()),
                                 contra=con[keep].mean() if keep.sum() else np.nan,
                                 mean_p_signal=ps[ix][keep].mean() if keep.sum() else np.nan,
                                 threshold=thr))
            keep = r2[ix] >= .1
            g = pd.DataFrame(dict(ecc=ecc[ix][keep], sd=p.sd.to_numpy()[ix][keep]))
            if len(g) >= 5:
                b = g.groupby(pd.cut(g.ecc, ECC, include_lowest=True), observed=True).sd.agg(['median', 'size']).reset_index()
                curves.append(pd.DataFrame(dict(subject=s, model=model, roi=roi, ecc=[i.mid for i in b.ecc],
                                                sd=b['median'], n=b['size'])))
    print(s, flush=True)
    return pd.DataFrame(rows), pd.concat(curves) if curves else pd.DataFrame()


def collect():
    with mp.Pool(6) as pool:
        out = pool.map(one, range(1, 31))
    pd.concat(o[0] for o in out).to_csv(TAB, sep='\t', index=False)
    pd.concat(o[1] for o in out).to_csv(CURVE, sep='\t', index=False)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action', choices=['collect'])
    collect()
