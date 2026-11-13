"""Side analysis: negative pRFs in the default network, summarised over subjects (local, small table).

Reads ``region_stats.tsv`` (one row per subject x region x hemisphere, from ``group_fsaverage.py``), pools the
hemispheres per subject (vertex-weighted), and writes a group table (mean, SEM and a one-sample t test over
subjects) to ``notes/side_prf_atlas/data/negative_prf_summary.tsv``.

Chance levels: a vertex is "cross-validated" at p < .05 against 50 shifted-stimulus fits, so about 4-5% of
null vertices pass, and half of those (or fewer, as both session fits must agree in sign) are negative; the
negative-pRF fraction is tested against 0.025. Laterality (Szinte & Knapen's index): (ipsi - contra) / (ipsi + contra),
-1 = all contralateral; tested against 0. ``session_angle_cos``: mean cos of the polar-angle difference between
the session-1 and session-2 fits (0 = no agreement).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def pool(df):
    out = []
    for (s, r), d in df.groupby(['subject', 'region']):
        nv = d.n_vertices.sum()
        row = dict(subject=s, region=r, n_vertices=nv)
        for c in ('frac_rel', 'frac_cv', 'frac_neg', 'frac_pos'):
            row[c] = (d[c] * d.n_vertices).sum() / max(nv, 1)
        for lab in ('neg', 'pos'):
            n = d[f'n_{lab}'].fillna(0)
            row[f'n_{lab}'] = n.sum()
            for c in [c for c in d.columns if c.startswith(f'{lab}_')]:
                v = d[c]
                okv = v.notna() & (n > 0)
                row[c] = np.average(v[okv], weights=n[okv]) if okv.any() else np.nan
        out.append(row)
    return pd.DataFrame(out)


def summarise(p, min_n=20):
    rows = []
    tests = dict(frac_neg=.025, frac_pos=.025, frac_cv=.05, neg_laterality=0., pos_laterality=0.,
                 neg_session_angle_cos=0., pos_session_angle_cos=0., neg_frac_beats_global=.5,
                 neg_ecc_sd_spearman=0., pos_ecc_sd_spearman=0.)
    for r, d in p.groupby('region'):
        row = dict(region=r, n_subjects=len(d))
        for c in ('frac_rel', 'frac_cv', 'frac_neg', 'frac_pos', 'neg_frac_contra', 'neg_laterality', 'pos_laterality',
                  'neg_session_angle_cos', 'pos_session_angle_cos', 'neg_session_x_sign_agree',
                  'neg_frac_beats_global', 'pos_frac_beats_global', 'neg_median_ecc', 'pos_median_ecc',
                  'neg_median_sd', 'pos_median_sd', 'neg_frac_sd_max', 'neg_ecc_sd_spearman', 'pos_ecc_sd_spearman',
                  'neg_median_cv_r', 'neg_median_glob_cv_r'):
            if c not in d:
                continue
            # per-subject quantities of negative/positive pRFs only from subjects with >= min_n such vertices
            lab = c.split('_')[0]
            dd = d[d[f'n_{lab}'] >= min_n] if lab in ('neg', 'pos') else d
            v = dd[c].dropna()
            row[f'{c}_mean'] = v.mean()
            row[f'{c}_sem'] = v.sem()
            row[f'{c}_n'] = len(v)
            if c in tests and len(v) > 2:
                alt = 'greater' if c.startswith('frac_') else 'two-sided'   # fractions: above chance
                row[f'{c}_t'], row[f'{c}_p'] = stats.ttest_1samp(v, tests[c], alternative=alt)
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--region-stats', required=True)
    ap.add_argument('--out', default='notes/side_prf_atlas/data/negative_prf_summary.tsv')
    a = ap.parse_args()
    df = pd.read_csv(a.region_stats, sep='\t')
    p = pool(df)
    s = summarise(p)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    s.to_csv(a.out, sep='\t', index=False)
    pd.set_option('display.width', 250, 'display.max_columns', 40, 'display.precision', 3)
    show = ['region', 'n_subjects', 'frac_rel_mean', 'frac_cv_mean', 'frac_neg_mean', 'frac_neg_p', 'frac_pos_mean',
            'neg_laterality_mean', 'neg_laterality_p', 'neg_laterality_n', 'pos_laterality_mean',
            'neg_session_angle_cos_mean', 'neg_session_angle_cos_p', 'pos_session_angle_cos_mean',
            'neg_frac_beats_global_mean', 'neg_median_ecc_mean', 'neg_median_sd_mean', 'neg_frac_sd_max_mean',
            'neg_ecc_sd_spearman_mean', 'neg_ecc_sd_spearman_p']
    print(s[[c for c in show if c in s]].T.to_string())
