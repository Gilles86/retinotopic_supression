"""Fig. 6c (decoded-bar HP effect per area) through the hierarchical model of the gain-field terms (update-3 check 6).

Per participant x area, the Fig. 6c effect (decoded intensity on the bar's pixels in the quadrant facing HP minus the
mean of the other three, in % of the mean of the four; runs averaged first, as ``fig6_decoding.quadrant_effects``),
its leave-one-run-out jackknife SE and its odd-even calibration. Then the Student-t hierarchical model of
``gain_meta_zs`` (pathway = mean of its areas) and, for comparison, the Wilcoxon table.

    PYTHONPATH=. ~/mambaforge/envs/bauer/bin/python -m retsupp.visualize.paper.decode_meta
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from retsupp.visualize.paper.fig6_decoding import DECODER, QUAD_ANGLE
from retsupp.visualize.paper.gain_meta_zs import PW_OF, bayes, wilcoxon_table


def effect(g):
    """g: rows of one subject x area (some runs). HP minus mean of the others, % of the mean of the four."""
    r = g.groupby('pos').drive_bar.mean()
    r = r / r.mean() * 100 - 100
    return r['hp'] - r.drop('hp').mean()


def collect(readout, exclude=(1, 2)):
    d = pd.read_csv(readout, sep='\t')
    d = d[(d.decoder == DECODER) & ~d.subject.isin(exclude)].copy()
    rel = (d.quadrant.map(QUAD_ANGLE) - d.hp_location.map(QUAD_ANGLE)) % 360
    d['pos'] = rel.map({0: 'hp', 90: 'left', 180: 'opposite', 270: 'right'})
    d['runkey'] = d.session.astype(str) + '_' + d.run.astype(str)
    rows = []
    for (s, roi), g in d.groupby(['subject', 'roi']):
        keys = sorted(g.runkey.unique())
        R = len(keys)
        if R < 4:
            continue
        full = effect(g)
        loo = np.array([effect(g[g.runkey != k]) for k in keys])
        se = np.sqrt((R - 1) / R * ((loo - loo.mean()) ** 2).sum())
        odd = g.run % 2 == 1
        half = abs(effect(g[odd]) - effect(g[~odd])) / 2
        rows.append(dict(subject=s, roi=roi, term='decoded', y=full, se=se, halfdiff=half, n_runs=R))
    t = pd.DataFrame(rows)
    t['pathway'] = t.roi.map(PW_OF)
    return t


def main(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t = collect(a.readout)
    t.to_csv(out / 'estimates.tsv', sep='\t', index=False)
    print('calibration (0.67 expected):', round(float(np.median(t.halfdiff / t.se)), 3),
          t.groupby('pathway').apply(lambda x: np.median(x.halfdiff / x.se), include_groups=False).round(3).to_dict())
    wilcoxon_table(t).to_csv(out / 'wilcoxon.tsv', sep='\t', index=False)
    s, dg = bayes(t)
    s.assign(variant='t').to_csv(out / 'bayes.tsv', sep='\t', index=False)
    pd.DataFrame([dg]).to_csv(out / 'bayes_diagnostics.tsv', sep='\t', index=False)
    print(s.round(3).to_string(), dg)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--readout', default='notes/data/decoding_readout_jointhrf_2026-11.tsv')
    ap.add_argument('--out', default='notes/data/decode_meta_2026-11-25')
    main(ap.parse_args())
