"""Fig. 7 links (trial-wise gains vs trial history and RT) side by side for several base fits.

Reads ``trialwise_meta_bayes`` output folders and writes, per fit, the mean over all nine areas (posterior mean, 95%
HDI, P of the sign shown) of each effect and variant, with the convergence of that fit (divergences, max R̂, min
bulk ESS). Gains are × 100 (a: % of the area's typical gain; b: percentage points of gain per SD of log RT).

    python -m retsupp.visualize.paper.rt_links_compare --meta notes/data/n28/trialwise_meta_bayes_2026-11 earlier \
        --meta notes/data/trialwise_meta_bayes_signal_se_polished_2026-11-12 refitted --out <tsv>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

EFFECTS = [('T_target_repeat', 'rel', 'Target gain, target repeat (% of typical gain)'),
           ('D_dist_repeat', 'rel', 'Distractor gain, distractor repeat (% of typical gain)'),
           ('D_rt', 'abs', 'Distractor gain per SD log RT (×100)'),
           ('G_rt', 'abs', 'Global gain per SD log RT (×100)'),
           ('T_rt', 'abs', 'Target gain per SD log RT (×100)')]


def main(metas, out):
    rows = []
    for folder, label in metas:
        s = pd.read_csv(Path(folder) / 'summary.tsv', sep='\t')
        g = pd.read_csv(Path(folder) / 'diagnostics.tsv', sep='\t')
        s = s[s.level == 'overall']
        for q, v0, name in EFFECTS:
            for suffix, vname in (('', 'primary'), ('_t', 'Student-t'), ('_nose', 'no SE weighting')):
                r = s[(s.question == q) & (s.variant == v0 + suffix)]
                d = g[(g.question == q) & (g.variant == v0 + suffix)]
                if r.empty:
                    continue
                r, d = r.iloc[0], d.iloc[0]
                rows.append(dict(fit=label, effect=name, variant=vname, mean=100 * r['mean'], hdi_lo=100 * r.hdi_lo,
                                 hdi_hi=100 * r.hdi_hi, p_sign=max(r.p_gt0, r.p_lt0), n_obs=int(d.n_obs),
                                 n_subjects=int(d.n_subjects), divergences=int(d.divergences), rhat_max=d.rhat_max,
                                 ess_bulk_min=d.ess_bulk_min, source=str(folder)))
    res = pd.DataFrame(rows)
    res.to_csv(out, sep='\t', index=False)
    p = res[res.variant == 'primary']
    with pd.option_context('display.width', 250, 'display.max_colwidth', 55):
        print(p.drop(columns=['source', 'variant']).round(3).to_string(index=False))
    for label in res.fit.unique():
        g = res[res.fit == label]
        print(f'{label}: divergences (sum over the listed fits) {g.divergences.sum()}, max R̂ {g.rhat_max.max():.3f}, '
              f'min bulk ESS {g.ess_bulk_min.min():.0f}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--meta', nargs=2, action='append', required=True, metavar=('FOLDER', 'LABEL'))
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    main(a.meta, a.out)
