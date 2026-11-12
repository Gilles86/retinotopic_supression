"""Between-subject link of the HP-specific sustained gain (dS = g_HP − g_LP) with the behavioural HP benefit
(LP − HP RT, correct trials), for several gain-field fits (earlier clamped fits, pRFs refitted with the gains,
voxel-selection robustness roots).

Same computation as ``richter_followup.d1_intervals`` (the paper's "Between subjects" sentence): per subject the
median over the nine areas of dS estimated alone (``conditional_estimates`` rows mech == 'dS' and ``valid``) or of
the joint m7 dS (DoG ``m7_full`` rows of a gain-parameter table, g_HP − g_LP); Spearman ρ with the benefit, percentile
bootstrap CI over subjects (10 000 resamples, seed 0). n = 28 (sub-03 … sub-30).

    python -m retsupp.visualize.paper.rt_links_between_subjects \
        --fit earlier notes/data/conditional_estimates_dog_final2_2026-11.tsv notes/data/gain_field_parameters_final2_2026-11.tsv \
        --fit refitted notes/data/conditional_estimates_polished_2026-11-12.tsv notes/data/gain_field_parameters_polished_2026-11-12.tsv \
        --fit r0.2 - notes/data/gain_field_parameters_relnull_r0.2_polished_2026-11-12.tsv \
        --out notes/data/rt_links_between_subjects_2026-11-12.tsv

``-`` for the conditional table: no estimated-alone dS for that fit (joint dS only).
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy import stats

from retsupp.visualize.paper.richter_followup import MIN_SUBJECT, N_BOOT, RNG_SEED, load_trials

AREAS = ['V1', 'V2', 'V3', 'V3AB', 'hV4', 'LO', 'TO', 'VO', 'IPS']


def ds_alone(path):
    c = pd.read_csv(path, sep='\t')
    c = c[(c.mech == 'dS') & c.valid & c.roi.isin(AREAS) & (c.subject >= MIN_SUBJECT)]
    return c.groupby('subject').estimate.median(), int(len(c))


def ds_joint(path):
    g = pd.read_csv(path, sep='\t')
    g = g[(g.family == 'dog') & (g.model == 'm7_full') & g.roi.isin(AREAS) & (g.subject >= MIN_SUBJECT)]
    return (g.g_HP - g.g_LP).groupby(g.subject).median(), int(len(g))


def spearman(ds, benefit):
    j = pd.concat([ds.rename('ds'), benefit.rename('b')], axis=1).dropna()
    rho, p = stats.spearmanr(j.ds, j.b)
    rng = np.random.default_rng(RNG_SEED)
    bs = [stats.spearmanr(*j.iloc[rng.integers(0, len(j), len(j))].T.values)[0] for _ in range(N_BOOT)]
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return dict(n=len(j), rho=rho, ci_lo=lo, ci_hi=hi, p=p, ds_median=j.ds.median(),
                ds_pct_neg=(j.ds < 0).mean() * 100)


def main(fits, trials, out):
    t = load_trials(trials)
    ok = t[t.ok]
    d = ok.groupby(['subject', 'dist_cond']).rt_ms.mean().unstack()
    benefit = d.LP - d.HP
    rows = []
    for label, cond, gain in fits:
        for kind, path, fn in (('dS alone', cond, ds_alone), ('dS joint m7', gain, ds_joint)):
            if path in ('-', '', None):
                continue
            ds, n_rows = fn(path)
            rows.append(dict(fit=label, estimate=kind, n_subject_area=n_rows, **spearman(ds, benefit), source=path))
    res = pd.DataFrame(rows)
    res.to_csv(out, sep='\t', index=False)
    with pd.option_context('display.width', 200, 'display.max_colwidth', 60):
        print(res.drop(columns='source').round(3).to_string(index=False))
    print(f'HP benefit (LP − HP RT): median {benefit.median():.1f} ms over {benefit.notna().sum()} subjects')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--fit', nargs=3, action='append', required=True, metavar=('LABEL', 'COND_TSV', 'GAIN_TSV'))
    ap.add_argument('--trials', default='notes/data/behaviour_trials_2026-11-11.tsv.gz')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    main(a.fit, a.trials, a.out)
