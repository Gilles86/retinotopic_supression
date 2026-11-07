"""Deconvolved responses at an item location (nideconv), 2 × 3: HP vs LP location × target, distractor, neutral item.

Per subject, ROI and ring location: the mean time series of the voxels on that location (bar response removed;
``export_location_timeseries.py``), runs concatenated with one intercept per run. Six event types, one per class of
what appears at that location on a trial; every trial is one of the six, so overlapping neighbouring trials are
separated by the regression instead of averaged into each epoch. Fourier basis over -4 to 20 s, OLS
(``nideconv.ResponseFitter``), time courses read out at 0.4 s.

Writes ``notes/data/event_deconv_2026-11.tsv``: subject, roi, loc, n_voxels, cls, t, response.

Usage::

    python -m retsupp.visualize.paper.deconvolve_location_responses --folder notes/data/location_timeseries_2026-11
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

INTERVAL = [-4., 20.]
N_REGRESSORS = 13
CLASSES = [f'{w}_{h}' for w in ('target', 'distractor', 'neutral') for h in ('hp', 'lp')]


def deconvolve(ts, trials, tr=1.6, oversample=4):
    from nideconv import ResponseFitter
    runs = ts[['session', 'run']].drop_duplicates().sort_values(['session', 'run']).to_records(index=False)
    classes = [c for c in CLASSES if c in set(trials.cls)] or sorted(trials.cls.unique())
    offset, sig, conf, onsets = 0., [], [], {c: [] for c in classes}
    for k, (s, r) in enumerate(runs):
        x = ts[(ts.session == s) & (ts.run == r)].sort_values('tr_index')
        sig.append(x.signal.to_numpy())
        c = np.zeros((len(x), len(runs) - 1))
        if k > 0:
            c[:, k - 1] = 1
        conf.append(c)
        t = trials[(trials.session == s) & (trials.run == r)]
        for cls in classes:
            onsets[cls].append(t[t.cls == cls].onset.to_numpy() + offset)
        offset += len(x) * tr
    y = np.concatenate(sig)
    time = np.arange(len(y)) * tr
    rf = ResponseFitter(pd.DataFrame({'signal': y}, index=pd.Index(time, name='time')), sample_rate=1 / tr)
    rf.add_confounds('run', pd.DataFrame(np.vstack(conf), index=time))
    for cls in classes:
        on = np.concatenate(onsets[cls])
        if len(on):
            rf.add_event(cls, onsets=on, interval=INTERVAL, basis_set='fourier', n_regressors=N_REGRESSORS)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        rf.fit()
        tc = rf.get_timecourses(oversample=oversample)
    tc = tc.reset_index()
    tc.columns = ['event_type', 'covariate', 'time', 'response']
    return tc[['event_type', 'time', 'response']].rename(columns={'event_type': 'cls', 'time': 't'})


def main(folder, out, subject=None):
    rows = []
    pattern = f'sub-{subject:02d}_timeseries.tsv.gz' if subject else 'sub-*_timeseries.tsv.gz'
    for f in sorted(Path(folder).glob(pattern)):
        subject = int(f.name[4:6])
        ts = pd.read_csv(f, sep='\t')
        trials = pd.read_csv(f.with_name(f'sub-{subject:02d}_trials.tsv'), sep='\t')
        keys = ['roi', 'loc'] + (['dbin'] if 'dbin' in ts else [])
        for key, x in ts.groupby(keys):
            tc = deconvolve(x, trials[trials['loc'] == key[1]])
            rows.append(tc.assign(subject=subject, n_voxels=int(x.n_voxels.iloc[0]), **dict(zip(keys, key))))
        print(f.name, flush=True)
    d = pd.concat(rows)
    d.to_csv(out, sep='\t', index=False, float_format='%.5g')
    print('Saved', out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--folder', default='notes/data/location_timeseries_2026-11')
    ap.add_argument('--out', default='notes/data/event_deconv_2026-11.tsv')
    ap.add_argument('--subject', type=int, default=None, help='Only this subject (cluster array task).')
    a = ap.parse_args()
    main(a.folder, a.out, a.subject)
