"""dS (HP − LP sustained gain) and LP gain per area and pooled, per voxel selection, with SEs across subjects.

Input: a ``collect_gain_parameters.py`` table with one ``family`` per voxel selection (e.g. ``current``, ``r0.2``,
``r0.1``), restricted to m4_TDS / m7_full, and per selection the ``collect_conditional_estimates.py`` table of the
estimated-alone terms (``--alone family=tsv``; mech dS, fits with a failed linearity check (``valid`` False) left
out, as for every paper number). Terms: dS estimated alone (on m4_TDS), dS in the joint fit (m7_full, g_HP − g_LP),
the LP gain in m7_full and the shared sustained gain of m4_TDS (g_HP = g_LP there). All in % gain (× 100). Pooled as in the paper
(``reference_numbers``): per subject the median over the nine areas. Per area and pooled: mean ± SE across subjects
(the decision statistic) and the paper's median [bootstrapped 95% CI] with its width. Subjects 1 and 2 are left out; only subject × area cells present in every
selection are compared (paired).

    python -m retsupp.visualize.paper.voxel_selection_robustness --tsv <collected.tsv> --out <summary.tsv>
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from retsupp.visualize.paper.fig45_common import median_ci

AREAS = ['V1', 'V2', 'V3', 'V3AB', 'hV4', 'LO', 'TO', 'VO', 'IPS']


def summarize(t, alone=None):
    t = t[(t.subject > 2) & t.roi.isin(AREAS)].copy()
    m7 = t[t.model == 'm7_full'].assign(dS=lambda d: d.g_HP - d.g_LP)
    m4 = t[t.model == 'm4_TDS']
    long = pd.concat([
        m7[['family', 'subject', 'roi', 'n_vox', 'dS']].rename(columns={'dS': 'value'}).assign(term='dS (m7_full)'),
        m7[['family', 'subject', 'roi', 'n_vox', 'g_LP']].rename(columns={'g_LP': 'value'}).assign(
            term='g_LP (m7_full)'),
        m4[['family', 'subject', 'roi', 'n_vox', 'g_LP']].rename(columns={'g_LP': 'value'}).assign(
            term='g_LP (m4_TDS)'),
    ] + ([alone] if alone is not None else []))
    long['value'] = long['value'] * 100
    n_fam = long.family.nunique()
    keep = long.groupby(['term', 'subject', 'roi']).family.transform('nunique') == n_fam
    long = long[keep]
    pooled = long.groupby(['family', 'term', 'subject']).agg(value=('value', 'median'), n_vox=('n_vox', 'sum'))
    pooled = pooled.reset_index().assign(roi='pooled')
    long = pd.concat([long, pooled])

    def stats(g):
        v = g.value.to_numpy()
        med, lo, hi = median_ci(v)
        return pd.Series(dict(n_sub=len(v), mean=v.mean(), se=v.std(ddof=1) / np.sqrt(len(v)), median=med,
                              ci_lo=lo, ci_hi=hi, ci_width=hi - lo, k_neg=int((v < 0).sum()),
                              median_n_vox=np.median(g.n_vox)))
    out = long.groupby(['term', 'roi', 'family'])[['value', 'n_vox']].apply(stats).reset_index()
    out['roi'] = pd.Categorical(out.roi, AREAS + ['pooled'], ordered=True)
    return out.sort_values(['term', 'roi', 'family'])


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tsv', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--alone', action='append', default=[], help='family=conditional-estimates tsv (repeatable)')
    a = ap.parse_args()
    t = pd.read_csv(a.tsv, sep='\t')
    alone = []
    for spec in a.alone:
        fam, f = spec.split('=', 1)
        c = pd.read_csv(f, sep='\t')
        c = c[(c.mech == 'dS') & c.valid & (c.subject > 2) & c.roi.isin(AREAS)]
        nv = t[(t.family == fam) & (t.model == 'm4_TDS')][['subject', 'roi', 'n_vox']]
        # inner join: a cell without this selection's m4_TDS fit has no valid estimated-alone base
        alone.append(c.merge(nv, on=['subject', 'roi'], how='inner').assign(family=fam, term='dS (alone)')
                     .rename(columns={'estimate': 'value'})[['family', 'subject', 'roi', 'n_vox', 'value', 'term']])
    s = summarize(t, pd.concat(alone) if alone else None)
    s.to_csv(a.out, sep='\t', index=False)
    with pd.option_context('display.width', 200, 'display.max_rows', 500):
        print(s.to_string(index=False, float_format='%.4f'))
