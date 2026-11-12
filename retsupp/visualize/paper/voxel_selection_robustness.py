"""dS (HP − LP sustained gain) and LP gain per area and pooled, per voxel selection, with SEs across subjects.

Input: a ``collect_gain_parameters.py`` table with one ``family`` per voxel selection (e.g. ``current``, ``r0.2``,
``r0.1``), restricted to m4_TDS / m7_full. dS comes from m7_full (g_HP − g_LP); the LP gain is reported for m7_full
and for m4_TDS (where the sustained gain is tied, g_HP = g_LP). Pooled = mean over the nine areas within a subject,
then mean ± SE across subjects. Subjects 1 and 2 are left out; only subject × area cells present in every
selection are compared (paired).

    python -m retsupp.visualize.paper.voxel_selection_robustness --tsv <collected.tsv> --out <summary.tsv>
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

AREAS = ['V1', 'V2', 'V3', 'V3AB', 'hV4', 'LO', 'TO', 'VO', 'IPS']


def summarize(t):
    t = t[(t.subject > 2) & t.roi.isin(AREAS)].copy()
    m7 = t[t.model == 'm7_full'].assign(dS=lambda d: d.g_HP - d.g_LP)
    m4 = t[t.model == 'm4_TDS']
    long = pd.concat([
        m7[['family', 'subject', 'roi', 'n_vox', 'dS']].rename(columns={'dS': 'value'}).assign(term='dS (m7_full)'),
        m7[['family', 'subject', 'roi', 'n_vox', 'g_LP']].rename(columns={'g_LP': 'value'}).assign(
            term='g_LP (m7_full)'),
        m4[['family', 'subject', 'roi', 'n_vox', 'g_LP']].rename(columns={'g_LP': 'value'}).assign(
            term='g_LP (m4_TDS)'),
    ])
    n_fam = long.family.nunique()
    keep = long.groupby(['term', 'subject', 'roi']).family.transform('nunique') == n_fam
    long = long[keep]
    pooled = long.groupby(['family', 'term', 'subject']).agg(value=('value', 'mean'), n_vox=('n_vox', 'sum'))
    pooled = pooled.reset_index().assign(roi='pooled')
    long = pd.concat([long, pooled])

    def stats(g):
        v = g.value.to_numpy()
        return pd.Series(dict(n_sub=len(v), mean=v.mean(), se=v.std(ddof=1) / np.sqrt(len(v)),
                              median_n_vox=np.median(g.n_vox)))
    out = long.groupby(['term', 'roi', 'family']).apply(stats).reset_index()
    out['roi'] = pd.Categorical(out.roi, AREAS + ['pooled'], ordered=True)
    return out.sort_values(['term', 'roi', 'family'])


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tsv', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    s = summarize(pd.read_csv(a.tsv, sep='\t'))
    s.to_csv(a.out, sep='\t', index=False)
    with pd.option_context('display.width', 200, 'display.max_rows', 500):
        print(s.to_string(index=False, float_format='%.4f'))
