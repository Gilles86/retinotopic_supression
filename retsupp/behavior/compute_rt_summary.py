"""Aggregate per-subject RT summaries from events.tsv files.

For each subject, walks all session-run pairs and computes:
  • RT_no_distractor    — mean RT on no-distractor trials (loc==10)
  • RT_HP               — mean RT on HP-distractor trials (loc==HPL_distractor)
  • RT_LP               — mean RT on LP-distractor trials (one of the 3 other ring locations)
  • RT_distractor       — mean RT on any distractor trial (loc in {1,3,5,7})

Outputs a long-format TSV with one row per subject:
  subject, RT_no_dist, RT_HP, RT_LP, RT_dist, n_no_dist, n_HP, n_LP

(plus derived columns for the contrasts.)

Usage:
    python -m retsupp.behavior.compute_rt_summary --out notes/rt_summary.tsv
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd


NO_DIST_LOC = 10
DIST_LOCS = {1, 3, 5, 7}


def globs_for_subject(subject: int) -> list[str]:
    """Subjects 1,2 use a flat sub-{N}/ses-{ses+1}/ layout (no run subdir,
    no zero-padding); subjects 3+ use sub-{N:02d}/ses-{ses+1}/run-{run}/.
    See retsupp/utils/data.py:get_onsets for the canonical pattern."""
    if subject < 3:
        return [
            f'/data/ds-retsupp/sourcedata/behavior/logs/sub-{subject}/'
            f'ses-*/sub-{subject}_ses-*_task-ret_sup_run-*_events.tsv'
        ]
    return [
        f'/data/ds-retsupp/sourcedata/behavior/logs/sub-{subject:02d}/'
        f'ses-*/run-*/sub-{subject:02d}_ses-*_'
        f'task-ret_sup_run-*_events.tsv'
    ]


def load_subject(subject: int) -> pd.DataFrame:
    paths = []
    for g in globs_for_subject(subject):
        paths += glob.glob(g)
    paths = sorted(paths)
    rows = []
    for p in paths:
        try:
            df = pd.read_csv(p, sep='\t')
        except Exception as e:
            print(f'  WARN sub-{subject:02d} {p}: {e}')
            continue
        # RT lives on `feedback` events, not `target` events.
        targets = df[df['event_type'] == 'feedback'].copy()
        if 'rt' not in targets.columns or len(targets) == 0:
            continue
        targets = targets.dropna(subset=['rt'])
        if 'HPL_distractor' not in targets.columns:
            # Subjects 1, 2 don't carry HPL_distractor: infer per-run from
            # the modal *non-no-distractor* distractor_location.
            ring_only = targets[
                targets['distractor_location'].isin(DIST_LOCS)
            ]
            if len(ring_only) == 0:
                hpl = np.nan
            else:
                hpl = ring_only['distractor_location'].mode().iloc[0]
            targets['HPL_distractor'] = hpl
        keep_cols = ['rt', 'distractor_location', 'HPL_distractor']
        if 'correct' in targets.columns:
            keep_cols.append('correct')
        rows.append(targets[keep_cols])
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def per_subject_summary(subject: int) -> dict:
    df = load_subject(subject)
    if df.empty:
        return {}
    # Drop incorrect trials (RT meaningful only when target was found).
    # `correct` is True/False (or NaN if no response) on feedback events.
    if 'correct' in df.columns:
        df = df[df['correct'] == True]
    # Keep only valid RTs (long upper tail for missed responses).
    df = df[(df['rt'] > 0.1) & (df['rt'] < 2.0)]
    no_dist = df[df['distractor_location'] == NO_DIST_LOC]
    dist = df[df['distractor_location'].isin(DIST_LOCS)]
    hp = df[df['distractor_location'] == df['HPL_distractor']]
    lp = df[df['distractor_location'].isin(DIST_LOCS)
             & (df['distractor_location'] != df['HPL_distractor'])]
    return dict(
        subject=subject,
        n_total=len(df),
        n_no_dist=len(no_dist), n_dist=len(dist),
        n_HP=len(hp), n_LP=len(lp),
        RT_no_dist=float(no_dist['rt'].mean()),
        RT_dist=float(dist['rt'].mean()),
        RT_HP=float(hp['rt'].mean()),
        RT_LP=float(lp['rt'].mean()),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subjects', type=int, nargs='+',
                        default=list(range(1, 31)))
    parser.add_argument('--out', type=Path,
                        default=Path('notes/rt_summary.tsv'))
    args = parser.parse_args()

    rows = []
    for s in args.subjects:
        print(f'sub-{s:02d}...', end=' ')
        r = per_subject_summary(s)
        if r:
            print(f'no_dist={r["n_no_dist"]}, HP={r["n_HP"]}, '
                  f'LP={r["n_LP"]}, RT_dist−RT_no={r["RT_dist"]-r["RT_no_dist"]:+.4f}, '
                  f'RT_HP−RT_LP={r["RT_HP"]-r["RT_LP"]:+.4f}')
            rows.append(r)
        else:
            print('NO DATA')
    df = pd.DataFrame(rows)
    df['RT_dist_minus_no'] = df['RT_dist'] - df['RT_no_dist']
    df['RT_HP_minus_LP']   = df['RT_HP'] - df['RT_LP']
    df['RT_HP_minus_no']   = df['RT_HP'] - df['RT_no_dist']
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, sep='\t', index=False)
    print(f'\nWrote {args.out}')


if __name__ == '__main__':
    main()
