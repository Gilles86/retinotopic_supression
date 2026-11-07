"""Joint model of the bar and the search items at an item location (nideconv): the sustained HP effect on the bar
response, separated from the item responses, and the six item responses (2 × 3) with the bar regressed out jointly.

Per subject, ROI and ring location: the voxels' mean BOLD (``raw``, bar not removed; ``export_location_timeseries.py
--subdir location_timeseries_joint``), runs concatenated. Regressors, fitted together by OLS (``nideconv.ResponseFitter``):

- bar × HP and bar × LP: m0's prediction for the bar alone (mean over the same voxels), split by whether the location is
  the run's HP location (its own column per kind; 0 in runs of the other kind);
- six item events, one per class of what appears at the location (target, distractor, neutral × HP, LP), Fourier
  basis over -4 to 20 s, read out at 0.4 s;
- one intercept per run.

The ratio of the bar × HP to the bar × LP beta is the bar response at HP relative to LP (1 = no difference).

Writes ``<out>`` (item responses: subject, roi, loc, n_voxels, cls, t, response) and ``<out stem>_bar.tsv``
(subject, roi, loc, n_voxels, beta_bar_hp, beta_bar_lp).

Usage::

    python -m retsupp.visualize.paper.deconvolve_bar_items --folder <location_timeseries_joint> --subject 23 --out x.tsv
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


def fit_location(ts, trials, tr=1.6, oversample=4):
    from nideconv import ResponseFitter
    runs = ts[['session', 'run']].drop_duplicates().sort_values(['session', 'run']).to_records(index=False)
    offset, sig, conf, onsets = 0., [], [], {c: [] for c in CLASSES}
    for k, (s, r) in enumerate(runs):
        x = ts[(ts.session == s) & (ts.run == r)].sort_values('tr_index')
        sig.append(x.raw.to_numpy())
        c = np.zeros((len(x), len(runs) - 1 + 2))
        if k > 0:
            c[:, k - 1] = 1
        hp = int(x.hp.iloc[0])
        c[:, -2] = x.bar.to_numpy() * hp
        c[:, -1] = x.bar.to_numpy() * (1 - hp)
        conf.append(c)
        t = trials[(trials.session == s) & (trials.run == r)]
        for cls in CLASSES:
            onsets[cls].append(t[t.cls == cls].onset.to_numpy() + offset)
        offset += len(x) * tr
    y = np.concatenate(sig)
    time = np.arange(len(y)) * tr
    cols = [f'run{k}' for k in range(1, len(runs))] + ['bar_hp', 'bar_lp']
    rf = ResponseFitter(pd.DataFrame({'signal': y}, index=pd.Index(time, name='time')), sample_rate=1 / tr)
    rf.add_confounds('nuisance', pd.DataFrame(np.vstack(conf), index=time, columns=cols))
    for cls in CLASSES:
        on = np.concatenate(onsets[cls])
        if len(on):
            rf.add_event(cls, onsets=on, interval=INTERVAL, basis_set='fourier', n_regressors=N_REGRESSORS)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        rf.fit()
        tc = rf.get_timecourses(oversample=oversample)
    b = rf.betas['signal']
    bar = dict(beta_bar_hp=float(b.loc[('confounds', 'nuisance', 'bar_hp')]),
               beta_bar_lp=float(b.loc[('confounds', 'nuisance', 'bar_lp')]))
    tc = tc.reset_index()
    tc.columns = ['event_type', 'covariate', 'time', 'response']
    return tc[['event_type', 'time', 'response']].rename(columns={'event_type': 'cls', 'time': 't'}), bar


def main(folder, out, subject=None):
    rows, bars = [], []
    pattern = f'sub-{subject:02d}_timeseries.tsv.gz' if subject else 'sub-*_timeseries.tsv.gz'
    for f in sorted(Path(folder).glob(pattern)):
        s = int(f.name[4:6])
        ts = pd.read_csv(f, sep='\t')
        trials = pd.read_csv(f.with_name(f'sub-{s:02d}_trials.tsv'), sep='\t')
        for (roi, loc), x in ts.groupby(['roi', 'loc']):
            tc, bar = fit_location(x, trials[trials['loc'] == loc])
            nv = int(x.n_voxels.iloc[0])
            rows.append(tc.assign(subject=s, roi=roi, loc=loc, n_voxels=nv))
            bars.append(dict(subject=s, roi=roi, loc=loc, n_voxels=nv, **bar))
        print(f.name, flush=True)
    pd.concat(rows).to_csv(out, sep='\t', index=False, float_format='%.5g')
    pd.DataFrame(bars).to_csv(str(out).replace('.tsv', '_bar.tsv'), sep='\t', index=False, float_format='%.6g')
    print('Saved', out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--folder', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--subject', type=int, default=None)
    a = ap.parse_args()
    main(a.folder, a.out, a.subject)
