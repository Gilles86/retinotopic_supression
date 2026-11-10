"""Bayesian hierarchical trial-wise RT ~ phasic AF gain models, one per ROI.

Bayesian counterpart of the per-subject OLS / lme4 models in
``analyze_trialwise_af_gains.py`` (bambi + nutpie; structure as
retinonumeral/analyses/uncertainty_behaviour_bayes.py). Input: the trial table
written by ``analyze_trialwise_af_gains.py collect``
(``notes/data/trialwise_af_trial_table.tsv.gz``): one row per valid-RT trial ×
ROI, with log RT and the per-trial ridge gains z-scored within
(session, run, trial type).

    logrt ~ 1 + T + D_HP + D_LP + G + log_trial
            + dist_repeat + target_repeat + target_at_prev_dist
            + (1 + T + D_HP + D_LP + G + log_trial | subject)

T: target gain at the target's ring. D_HP / D_LP: distractor gain on HP- / LP-
distractor trials (0 elsewhere). G: global per-trial gain. log_trial:
log(experiment-wide trial number), centred.

    python -m retsupp.visualize.paper.trialwise_rt_bayes \\
        --trials notes/data/trialwise_af_trial_table.tsv.gz --roi V1 \\
        --out-dir notes/data/trialwise_rt_bayes
"""
import argparse
import time
from pathlib import Path

import arviz as az
import bambi as bmb
import numpy as np
import pandas as pd

ROIS = ['V1', 'V2', 'V3', 'V3AB', 'hV4', 'LO', 'TO', 'VO', 'IPS', 'SPL1', 'FEF', 'early', 'dorsal', 'occipitotemporal']
TRIALS_PER_RUN = 60
SLOPES = ['T', 'D_HP', 'D_LP', 'G', 'log_trial']
# --hp-split: the gain at HP split by what HP holds on the trial (target, distractor, neutral item).
SLOPES_HP = ['T_HP', 'T_LP', 'D_HP', 'D_LP', 'H', 'G', 'log_trial']
COVARIATES = ['dist_repeat', 'target_repeat', 'target_at_prev_dist']


def prepare(t: pd.DataFrame) -> pd.DataFrame:
    t = t.copy()
    t['subject'] = 's' + t['subject'].astype(str).str.zfill(2)
    t['D_HP'] = t['D'] * (t['trial_type'] == 'HP')
    t['D_LP'] = t['D'] * (t['trial_type'] == 'LP')
    if 'target_at_hp' in t:
        t['T_HP'] = t['T'] * t['target_at_hp']
        t['T_LP'] = t['T'] * (1 - t['target_at_hp'])
    # Experiment-wide trial number (runs ordered by session, run; 60 trials each),
    # so the covariate can absorb learning across the whole experiment.
    runs = (t[['subject', 'session', 'run']].drop_duplicates()
            .sort_values(['subject', 'session', 'run']))
    runs['run_rank'] = runs.groupby('subject').cumcount()
    t = t.merge(runs, on=['subject', 'session', 'run'], how='left')
    lt = np.log(t['run_rank'] * TRIALS_PER_RUN + t['trial_idx'] + 1)
    t['log_trial'] = (lt - lt.mean()) / lt.std()
    return t


def build(t: pd.DataFrame, slopes=SLOPES, noncentered: bool = True):
    hn = lambda s: bmb.Prior('HalfNormal', sigma=s)  # noqa: E731
    N = lambda m, s: bmb.Prior('Normal', mu=m, sigma=s)  # noqa: E731
    fixed = slopes + COVARIATES
    f = ('logrt ~ 1 + ' + ' + '.join(fixed)
         + f" + (1 + {' + '.join(slopes)} | subject)")
    priors = {'Intercept': N(0, 0.5), '1|subject': N(0, hn(0.3)),
              'sigma': hn(1.0)}
    for c in fixed:
        priors[c] = N(0, 0.3)
    for c in slopes:
        priors[f'{c}|subject'] = N(0, hn(0.2))
    return bmb.Model(f, t, priors=priors, noncentered=noncentered)


def summarize(idata, roi, n_sub, n_trials, seconds, a, slopes=SLOPES) -> pd.DataFrame:
    post = idata.posterior
    full = az.summary(idata)
    terms = {c: post[c].values.ravel() for c in slopes + COVARIATES}
    if 'D_HP' in terms and 'D_LP' in terms:
        terms['D_HP-D_LP'] = terms['D_HP'] - terms['D_LP']
        terms['D_avg'] = 0.5 * (terms['D_HP'] + terms['D_LP'])
    if 'T_HP' in terms and 'H' in terms:
        terms['T_HP-T_LP'] = terms['T_HP'] - terms['T_LP']
        terms['D_HP-T_HP'] = terms['D_HP'] - terms['T_HP']   # crossover: what HP holds
        terms['D_HP-H'] = terms['D_HP'] - terms['H']
    rows = []
    for term, x in terms.items():
        lo, hi = az.hdi(x, hdi_prob=0.95)
        sd_name = f'{term}|subject_sigma'
        rows.append(dict(
            roi=roi, term=term, n_subjects=n_sub, n_trials=n_trials,
            mean=x.mean(), hdi_lo=lo, hdi_hi=hi,
            p_gt0=(x > 0).mean(), p_lt0=(x < 0).mean(),
            sd_between=float(post[sd_name].values.mean()) if sd_name in post else np.nan,
            divergences=int(idata.sample_stats.diverging.sum()),
            n_maxdepth=(int((idata.sample_stats.depth >= 10).sum())
                        if 'depth' in idata.sample_stats else -1),
            rhat_max=float(full.r_hat.max()),
            ess_bulk_min=float(full.ess_bulk.min()),
            ess_tail_min=float(full.ess_tail.min()),
            seconds=seconds, draws=a.draws, tune=a.tune, chains=a.chains,
            bambi=bmb.__version__))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--trials', required=True)
    ap.add_argument('--roi', required=True, choices=ROIS)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--exclude-subjects', type=int, nargs='*', default=[])
    ap.add_argument('--exclude-no-distractor', action='store_true')
    ap.add_argument('--hp-split', action='store_true',
                    help='Split the target gain by target at HP and add H (needs a table from --hp-neutral fits).')
    ap.add_argument('--drop', nargs='*', default=[], choices=sorted(set(SLOPES + SLOPES_HP)),
                    help='Gain terms to leave out of the model.')
    ap.add_argument('--draws', type=int, default=1000)
    ap.add_argument('--tune', type=int, default=1000)
    ap.add_argument('--chains', type=int, default=4)
    a = ap.parse_args()

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    allt = pd.read_csv(a.trials, sep='\t')
    t = allt[(allt['roi'] == a.roi)
             & ~allt['subject'].isin(a.exclude_subjects)]
    if a.exclude_no_distractor:
        t = t[t['trial_type'].isin(['HP', 'LP'])]
    t = prepare(t)
    n_sub = t['subject'].nunique()
    print(f'{a.roi}: {n_sub} subjects, {len(t)} trials', flush=True)

    slopes = [c for c in (SLOPES_HP if a.hp_split else SLOPES) if c not in a.drop]
    m = build(t, slopes)
    t0 = time.time()
    idata = m.fit(draws=a.draws, tune=a.tune, chains=a.chains,
                  random_seed=20261027, target_accept=0.95,
                  inference_method='nutpie')
    seconds = time.time() - t0
    idata.to_netcdf(out / f'roi-{a.roi}_idata.nc')
    res = summarize(idata, a.roi, n_sub, len(t), seconds, a, slopes)
    res.to_csv(out / f'roi-{a.roi}_summary.tsv', sep='\t', index=False,
               float_format='%.5g')
    with pd.option_context('display.width', 200):
        print(res.round(4).to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
