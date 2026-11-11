"""Behavioural RT effects for notes/results_overview_2026-11.md: all 30 subjects, and without the two pilots.

The pilots' events have no HPL_distractor column, so the HP location of every run comes from
Subject.get_hpd_locations (modal singleton location), checked against that column where it exists.
"""
import numpy as np
import pandas as pd
import yaml
from scipy import stats

from retsupp.utils.data import Subject

CODE = {'upper_right': 1., 'upper_left': 3., 'lower_left': 5., 'lower_right': 7.}
rows = []
for s in yaml.safe_load(open('retsupp/data/subjects.yml')):
    sub = Subject(s, bids_folder='/data/ds-retsupp')
    hp = sub.get_hpd_locations()
    for ses in (1, 2):
        for run in sub.get_runs(ses):
            ev = sub.get_onsets(ses, run)
            t = ev[ev.event_type == 'feedback'].copy()
            t['subject'], t['session'], t['run'] = s, ses, run
            t['hp'] = CODE[hp[(ses, run)]]
            t['hpl'] = t.HPL_distractor if 'HPL_distractor' in t else np.nan
            rows.append(t[['subject', 'session', 'run', 'rt', 'correct', 'distractor_location', 'target_location', 'hp', 'hpl']])
df = pd.concat(rows, ignore_index=True)
c = df[df.hpl.notna()]
print('Trials where get_hpd_locations equals HPL_distractor:', (c.hp == c.hpl).mean(), 'subjects without the column:',
      sorted(set(df.subject) - set(c.subject)))
df['has_d'] = df.distractor_location != 10
df['dist'] = np.where(~df.has_d, 'Absent', np.where(df.distractor_location == df.hp, 'HP', 'LP'))
df['targ'] = np.where(df.target_location == df.hp, 'HP', 'LP')
print('Share of singleton trials at HP by subject group:',
      df[df.has_d].groupby(df.subject <= 2).apply(lambda g: (g.distractor_location == g.hp).mean()).round(3).to_dict())
print('Accuracy, pilots:', df[df.subject <= 2].assign(correct=lambda x: x.correct.astype(float)).groupby('subject').correct.mean().round(3).to_dict(),
      ' others (median):', round(df[df.subject > 2].assign(correct=lambda x: x.correct.astype(float)).groupby('subject').correct.mean().median(), 3))
ok = df[(df.correct == True) & df.rt.notna()]


def means(d, col):
    return d.groupby(['subject', col]).rt.mean().mul(1000).unstack()


def report(mask, label):
    print(f'\n=== {label}: {df[mask(df)].subject.nunique()} subjects, {mask(df).sum()} trials',
          df[mask(df)].groupby('dist').size().to_dict())
    w = means(ok[mask(ok)], 'dist').dropna()
    print(w.mean().round(1).to_dict())
    for a, b in [('HP', 'LP'), ('LP', 'Absent'), ('HP', 'Absent')]:
        d = w[a] - w[b]
        print(f'{a} vs {b}: {d.mean():+.1f} ms, {(d > 0).mean() * 100:.0f}% positive, p={stats.wilcoxon(d).pvalue:.2g}, n={len(d)}')
    x = ok[mask(ok) & ok.has_d]
    t = means(x, 'targ').dropna()
    d = t.HP - t.LP
    print(t.mean().round(1).to_dict())
    print(f'Target HP vs LP: {d.mean():+.1f} ms, {(d > 0).mean() * 100:.0f}% positive, p={stats.wilcoxon(d).pvalue:.2g}, n={len(d)}')
    return w, t


w, t = report(lambda d: d.subject > 0, 'All')
report(lambda d: d.subject > 2, 'Without pilots')
print('\nPilots:')
print(pd.concat([w.loc[[1, 2]], t.loc[[1, 2]].add_prefix('target_')], axis=1).round(1).to_string())
