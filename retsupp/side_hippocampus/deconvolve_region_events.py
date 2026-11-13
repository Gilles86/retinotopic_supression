"""Side project (hippocampus): pRF-free region-level test of the HP regularity.

Per subject and anatomical region (all voxels, both hemispheres; no selection on the data, so nothing circular): the
region-mean cleaned BOLD of every run (``export_region_mean_ts.py``), z-scored within run, runs concatenated with one
intercept per run. Every search display (target onset) is one event, classified by where the distractor was (at the
run's HP location / at an LP location / absent) and where the target was (HP / LP); the target is never at the
distractor's location, so five event types:

    dHP_tLP, dLP_tHP, dLP_tLP, dabs_tHP, dabs_tLP

Deconvolution as in ``visualize/paper/deconvolve_location_responses.py`` (nideconv ResponseFitter, Fourier basis,
13 regressors, −4 to 20 s, OLS, read out at 0.4 s). ``--rt`` adds one more event on every trial whose amplitude is
the trial's RT (z-scored within run; trials without a response get 0), so differences that only follow
response time are taken out.

A sustained (block-level) HP regressor is not possible: the HP location is constant within a run, so it is collinear
with the run intercepts.

``--split``: only some runs: ses1 / ses2, or ``pos1`` / ``pos3``, the first / third run of each HP block (the block's
learning onset vs after two runs of exposure).

Writes one long table: subject, region, cls, t, response (+ split, rt flags) and a trial table with RTs.

Usage (local; nideconv is not in the cluster env)::

    python -m retsupp.side_hippocampus.deconvolve_region_events \\
        --ts /data/ds-retsupp/derivatives/side_hpc_region_mean_ts/region_mean_ts.tsv.gz \\
        --out notes/side_hippocampus/data/region_event_deconv.tsv.gz
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd

from retsupp.utils.data import Subject

INTERVAL = [-4., 20.]
N_REGRESSORS = 13
CLASSES = ['dHP_tLP', 'dLP_tHP', 'dLP_tLP', 'dabs_tHP', 'dabs_tLP']
SPLITS = [None, 'ses1', 'ses2', 'pos1', 'pos3']


def trials_of(subject, bids):
    sub = Subject(subject, bids)
    rows = []
    for ses in (1, 2):
        for k, run in enumerate(sub.get_runs(ses)):
            e = sub.get_onsets(ses, run)
            t = e[e.event_type == 'target'].copy()
            resp = e[e.event_type == 'response'].groupby('trial_nr').onset.min()
            t['rt'] = t.trial_nr.map(resp) - t.onset
            t.loc[(t.rt <= 0) | (t.rt > 3), 'rt'] = np.nan
            hp = t.HPL_distractor.mode()[0]
            d = np.where(t.distractor_location == hp, 'dHP', np.where(t.distractor_location == 10, 'dabs', 'dLP'))
            tg = np.where(t.target_location == hp, 'tHP', 'tLP')
            rows.append(pd.DataFrame(dict(subject=subject, session=ses, run=run, onset=t.onset.to_numpy(),
                                          cls=[f'{a}_{b}' for a, b in zip(d, tg)], rt=t.rt.to_numpy(),
                                          hp=hp, correct=t.correct_response.to_numpy())))
    tr = pd.concat(rows, ignore_index=True)
    # position of the run within its HP block (1, 2, 3) per session
    blk = tr.groupby(['session', 'run']).hp.first().reset_index()
    blk['pos'] = blk.groupby(['session', 'hp']).cumcount() + 1
    return tr.merge(blk[['session', 'run', 'pos']], on=['session', 'run'])


def deconvolve(ts, trials, rt=False, tr=1.6, oversample=4):
    from nideconv import ResponseFitter
    runs = ts[['session', 'run']].drop_duplicates().sort_values(['session', 'run']).to_records(index=False)
    offset, sig, conf = 0., [], []
    on = {c: [] for c in CLASSES}
    rt_on, rt_val = [], []
    for k, (s, r) in enumerate(runs):
        x = ts[(ts.session == s) & (ts.run == r)].sort_values('tr_index').signal.to_numpy()
        sig.append((x - x.mean()) / x.std())
        c = np.zeros((len(x), len(runs) - 1))
        if k > 0:
            c[:, k - 1] = 1
        conf.append(c)
        t = trials[(trials.session == s) & (trials.run == r)]
        for cls in CLASSES:
            on[cls].append(t[t.cls == cls].onset.to_numpy() + offset)
        z = (t.rt - t.rt.mean()) / t.rt.std()
        rt_on.append(t.onset.to_numpy() + offset)
        rt_val.append(z.fillna(0).to_numpy())
        offset += len(x) * tr
    y = np.concatenate(sig)
    time = np.arange(len(y)) * tr
    rf = ResponseFitter(pd.DataFrame({'signal': y}, index=pd.Index(time, name='time')), sample_rate=1 / tr)
    rf.add_confounds('run', pd.DataFrame(np.vstack(conf), index=time))
    for cls in CLASSES:
        o = np.concatenate(on[cls])
        if len(o):
            rf.add_event(cls, onsets=o, interval=INTERVAL, basis_set='fourier', n_regressors=N_REGRESSORS)
    if rt:
        rf.add_event('rt', onsets=np.concatenate(rt_on), interval=INTERVAL, basis_set='fourier',
                     n_regressors=N_REGRESSORS, covariates=pd.DataFrame({'rt': np.concatenate(rt_val)}))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        rf.fit()
        tc = rf.get_timecourses(oversample=oversample)
    tc = tc.reset_index()
    tc.columns = ['event_type', 'covariate', 'time', 'response']
    tc = tc[tc.event_type != 'rt']
    return tc[['event_type', 'time', 'response']].rename(columns={'event_type': 'cls', 'time': 't'})


def main(ts_file, out, trials_out, bids):
    ts_all = pd.read_csv(ts_file, sep='\t')
    ts_all = ts_all[ts_all.hemi == 'both']
    rows, trs = [], []
    for s, ts_s in ts_all.groupby('subject'):
        tr = trials_of(s, bids)
        trs.append(tr)
        for split in SPLITS:
            keep = tr if split is None else tr[(tr.session == int(split[-1])) if split.startswith('ses')
                                               else (tr.pos == int(split[-1]))]
            runs = keep[['session', 'run']].drop_duplicates()
            for region, x in ts_s.groupby('region'):
                x = x.merge(runs, on=['session', 'run'])
                for rt in (False, True):
                    tc = deconvolve(x, keep, rt=rt)
                    rows.append(tc.assign(subject=s, region=region, split=split or 'all', rt=rt))
        print('sub', s, flush=True)
    pd.concat(rows).to_csv(out, sep='\t', index=False, float_format='%.5g')
    pd.concat(trs).to_csv(trials_out, sep='\t', index=False)
    print('wrote', out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ts', default='/data/ds-retsupp/derivatives/side_hpc_region_mean_ts/region_mean_ts.tsv.gz')
    ap.add_argument('--out', default='notes/side_hippocampus/data/region_event_deconv.tsv.gz')
    ap.add_argument('--trials-out', default='notes/side_hippocampus/data/region_event_trials.tsv.gz')
    ap.add_argument('--bids-folder', default='/data/ds-retsupp')
    a = ap.parse_args()
    main(a.ts, a.out, a.trials_out, a.bids_folder)
