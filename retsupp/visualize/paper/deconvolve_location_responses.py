"""Deconvolved responses at an item location (nideconv), 2 × 3: HP vs LP location × target, distractor, neutral item.

Per subject, ROI and ring location: the mean time series of the voxels on that location (bar response removed;
``export_location_timeseries.py``), runs concatenated with one intercept per run. Six event types, one per class of
what appears at that location on a trial; every trial is one of the six, so overlapping neighbouring trials are
separated by the regression instead of averaged into each epoch. Fourier basis (or ``--basis dct``) over -4 to 20 s, OLS
(``nideconv.ResponseFitter``), time courses read out at 0.4 s.

Writes ``notes/data/event_deconv_2026-11.tsv``: subject, roi, loc, n_voxels, cls, t, response.

``--relative``: the location is labelled by where it lies relative to the run's HP location instead of HP vs LP, so
twelve event types (item × HP, next to HP on the left, next to HP on the right, opposite HP). Left and right are seen
from fixation looking towards HP (HP rotated to the top): left is the counterclockwise neighbour. ``loc`` keeps the
absolute location, so the neighbours can also be split into the one in the same and the other hemifield.

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
RELATIVE = {0: 'hp', 1: 'nextleft', 2: 'opposite', 3: 'nextright'}   # (loc - HP loc) mod 4; ring runs counterclockwise
CLASSES += [f'{w}_{h}' for w in ('target', 'distractor', 'neutral') for h in RELATIVE.values() if h != 'hp']


def relabel_relative(trials):
    """target/distractor/neutral × HP/LP -> × HP/next left/next right/opposite, per run's HP location."""
    hp = trials[trials.cls.str.endswith('_hp')].groupby(['session', 'run'])['loc'].agg(lambda x: x.mode()[0]).rename('hp_loc')
    t = trials.join(hp, on=['session', 'run'])
    assert t.hp_loc.notna().all(), 'run without an HP location'
    rel = ((t['loc'] - t.hp_loc) % 4).astype(int).map(RELATIVE)
    t['cls'] = t.cls.str.rsplit('_', n=1).str[0] + '_' + rel
    return t.drop(columns='hp_loc')


SPLITS = ('ses1', 'ses2', 'block1', 'block2', 's1b1', 's1b2', 's2b1', 's2b2')


def run_blocks(trials):
    """Per (session, run): the block within the session, 1 or 2 (which of the session's two HP locations)."""
    hp = trials[trials.cls.str.endswith('_hp')].groupby(['session', 'run'])['loc'].agg(lambda x: x.mode()[0])
    first = hp.groupby(level='session').transform('first')
    return pd.Series(np.where(hp == first, 1, 2), index=hp.index, name='block')


def select_runs(ts, trials, split):
    """Only the runs of one session (ses1, ses2), one block (block1, block2: both sessions) or one session x block."""
    b = run_blocks(trials).reset_index()
    ses = int(split[3]) if split.startswith('ses') else int(split[1]) if split.startswith('s') and 'b' in split else None
    blk = int(split[-1]) if 'b' in split[1:] or split.startswith('block') else None
    keep = b[((b.session == ses) if ses else True) & ((b.block == blk) if blk else True)][['session', 'run']]
    on = ['session', 'run']
    return ts.merge(keep, on=on), trials.merge(keep, on=on)


def deconvolve(ts, trials, tr=1.6, oversample=4, basis='fourier'):
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
            rf.add_event(cls, onsets=on, interval=INTERVAL, basis_set=basis, n_regressors=N_REGRESSORS)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        rf.fit()
        tc = rf.get_timecourses(oversample=oversample)
    tc = tc.reset_index()
    tc.columns = ['event_type', 'covariate', 'time', 'response']
    return tc[['event_type', 'time', 'response']].rename(columns={'event_type': 'cls', 'time': 't'})


def main(folder, out, subject=None, basis='fourier', relative=False, split=None):
    rows = []
    pattern = f'sub-{subject:02d}_timeseries.tsv.gz' if subject else 'sub-*_timeseries.tsv.gz'
    for f in sorted(Path(folder).glob(pattern)):
        subject = int(f.name[4:6])
        ts = pd.read_csv(f, sep='\t')
        trials = pd.read_csv(f.with_name(f'sub-{subject:02d}_trials.tsv'), sep='\t')
        if split:
            ts, trials = select_runs(ts, trials, split)
        if relative:
            trials = relabel_relative(trials)
        keys = ['roi', 'loc'] + (['dbin'] if 'dbin' in ts else [])
        for key, x in ts.groupby(keys):
            tc = deconvolve(x, trials[trials['loc'] == key[1]], basis=basis)
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
    ap.add_argument('--basis', default='fourier', choices=['fourier', 'dct', 'fir'],
                    help="nideconv basis set; 'dct' needs nideconv with the oversampling fix (branch dct-oversample-fix).")
    ap.add_argument('--relative', action='store_true',
                    help='Label locations HP / next left / next right / opposite instead of HP / LP.')
    ap.add_argument('--split', choices=SPLITS, default=None,
                    help='Only the runs of one session, one block within the sessions (HP location 1 or 2 of the '
                         'session), or one session x block (A1/A2 of the Richter follow-up).')
    a = ap.parse_args()
    main(a.folder, a.out, a.subject, a.basis, a.relative, a.split)
