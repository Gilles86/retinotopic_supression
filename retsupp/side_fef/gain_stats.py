"""Side project (FEF / sPCS): group statistics of the gain-field estimates in the frontal ROIs.

Inputs: ``visualize/paper/collect_polished_fits`` output (``<prefix>_shared.tsv``: m4_TDS and m7_full shared terms,
clamped and polished) and ``visualize/paper/collect_conditional_estimates`` output (dS estimated alone, sum b / sum a
on the polished m4_TDS base). Terms in % (gain x 100):

- dS alone: g_HP − g_LP added alone to m4_TDS (linear least squares from the base);
- dS joint, g_HP, g_LP, target (g_T_dyn), distractor (mean over the four locations, (g_HP_dyn + 3 g_LP_dyn) / 4),
  items (item_scale): m7_full with pRFs refitted together with the gain fields (polished).

Per ROI: n, median, bootstrapped 95% CI of the median (10,000 resamples, seed 0, percentile), k of n with the sign of
the median, Wilcoxon signed-rank against 0 (two-sided), as the paper reports them.

    python -m retsupp.side_fef.gain_stats --shared notes/side_fef/data/gain_fits_shared.tsv \\
        --cond notes/side_fef/data/dS_alone.tsv --out notes/side_fef/data/gain_stats.tsv
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def median_ci(x, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    b = np.median(rng.choice(x, (n_boot, len(x))), axis=1)
    return np.median(x), np.percentile(b, 2.5), np.percentile(b, 97.5)


def summarise(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {}
    med, lo, hi = median_ci(x)
    k = int((np.sign(x) == np.sign(med)).sum()) if med != 0 else 0
    p = wilcoxon(x).pvalue if len(x) >= 5 and np.any(x != 0) else np.nan
    return dict(n=len(x), median_pct=100 * med, ci_lo_pct=100 * lo, ci_hi_pct=100 * hi, k=k, p_wilcoxon=p)


def main(shared, cond, out):
    sh = pd.read_csv(shared, sep='\t')
    sh = sh[sh.subject.between(3, 30)]
    j = sh[(sh.model == 'm7_full') & (sh.fit == 'polished')].copy()
    j['target'] = j.g_T_dyn
    j['distractor'] = (j.g_HP_dyn + 3 * j.g_LP_dyn) / 4
    c = pd.read_csv(cond, sep='\t')
    c = c[c.subject.between(3, 30) & (c.mech == 'dS')]
    rows = []
    for roi in sorted(set(j.roi) | set(c.roi)):
        terms = [('dS alone', c.loc[c.roi == roi, 'estimate'])]
        for col, lab in [('dS', 'dS joint'), ('g_HP', 'g_HP joint'), ('g_LP', 'g_LP joint'),
                         ('target', 'target joint'), ('distractor', 'distractor joint'), ('item_scale', 'items joint')]:
            terms.append((lab, j.loc[j.roi == roi, col]))
        for lab, x in terms:
            rows.append(dict(roi=roi, term=lab, **summarise(x)))
        rows.append(dict(roi=roi, term='n_voxels (median)', median_pct=np.nan,
                         n=int(j.loc[j.roi == roi, 'n_voxels'].median()) if (j.roi == roi).any() else 0))
    t = pd.DataFrame(rows)
    t.to_csv(out, sep='\t', index=False)
    pd.set_option('display.width', 200)
    print(t.round(4).to_string(index=False))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--shared', required=True)
    ap.add_argument('--cond', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    main(a.shared, a.cond, a.out)
