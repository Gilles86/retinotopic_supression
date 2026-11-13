"""Pool the per-trial gains of ``fit_trialwise_af_gains_trialse`` over areas, with their 3 × 3 sampling covariance.

Per participant and kind (T, D = D_HP or D_LP, G): each area's deviation from its canonical gain, g − g0, averaged over
the areas of a pool with fixed weights 1 / median(SE_shift²) of that participant × area × kind. The same weighted mean
of the circular-shift placebo estimates (identical offsets in every area) gives the pooled estimate's covariance across
T, D and G of the trial, with the cross-area error correlation included. Pools: ``all9`` (main), and the pathways
``early``, ``dorsal``, ``occipitotemporal``.

Output: one TSV, one row per subject × pool × trial (all trials; the RT model selects correct ones), with gT, gD, gG,
vTT, vDD, vGG, cTD, cTG, cDG (gD and its terms NaN on trials without a distractor); and a per-area summary of SEs.

    python -m retsupp.modeling.collect_trialse --folder af_trialwise_trialse_polished_2026-11-13 --out <tsv>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from retsupp.modeling.fit_trialwise_af_gains import GROUPS

POOLS = {'all9': [a for g in GROUPS.values() for a in g], **GROUPS}
KINDS = ['T', 'D', 'G']


def load_area(f):
    z = np.load(f)
    kind = pd.Series(z['kind']).replace({'D_HP': 'D', 'D_LP': 'D'}).to_numpy()
    return dict(trial_idx=z['trial_idx'], kind=kind, dev=z['g'] - z['g0'], draws=z['draws'].astype(np.float64))


def subject_rows(sub_dir):
    s = int(sub_dir.name[4:])
    trials = pd.read_csv(sub_dir / f'sub-{s:02d}_trials.tsv', sep='\t')
    n = len(trials)
    areas, summ = {}, []
    for f in sorted(sub_dir.glob(f'sub-{s:02d}_roi-*_trialse_shifts.npz')):
        roi = f.name.split('_roi-')[1].split('_')[0]
        a = load_area(f)
        S = a['draws'].shape[0]
        est = np.full((n, 3), np.nan)
        dr = np.full((S, n, 3), np.nan)
        w = np.zeros(3)
        se = a['draws'].std(0, ddof=1)
        for j, k in enumerate(KINDS):
            sel = a['kind'] == k
            idx = a['trial_idx'][sel]
            est[idx, j] = a['dev'][sel]
            dr[:, idx, j] = a['draws'][:, sel]
            w[j] = 1.0 / np.median(se[sel] ** 2)
            summ.append(dict(subject=s, roi=roi, kind=k, n=int(sel.sum()), median_se=float(np.median(se[sel])),
                             sd_estimate=float(np.std(a['dev'][sel])),
                             reliability=float(1 - np.median(se[sel] ** 2) / np.var(a['dev'][sel]))))
        areas[roi] = (est, dr, w)
    rows = []
    for pool, members in POOLS.items():
        have = [r for r in members if r in areas]
        if not have:
            continue
        num = sum(areas[r][2] * np.nan_to_num(areas[r][0]) for r in have)
        den = sum(areas[r][2] * ~np.isnan(areas[r][0]) for r in have)
        g = np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)
        dnum = sum(areas[r][2] * np.nan_to_num(areas[r][1]) for r in have)
        dd = dnum / np.where(den > 0, den, 1)[np.newaxis]
        dd = dd - dd.mean(0, keepdims=True)
        C = np.einsum('sni,snj->nij', dd, dd) / (dd.shape[0] - 1)
        out = trials.copy()
        out.insert(0, 'pool', pool)
        out['n_areas'] = len(have)
        out['areas'] = ' '.join(have)
        out['gT'], out['gD'], out['gG'] = g[:, 0], g[:, 1], g[:, 2]
        out['vTT'], out['vDD'], out['vGG'] = C[:, 0, 0], C[:, 1, 1], C[:, 2, 2]
        out['cTD'], out['cTG'], out['cDG'] = C[:, 0, 1], C[:, 0, 2], C[:, 1, 2]
        nod = np.isnan(out['gD'])
        out.loc[nod, ['vDD', 'cTD', 'cDG']] = np.nan
        rows.append(out)
    return pd.concat(rows), pd.DataFrame(summ)


def main(bids_folder, folder, out):
    root = Path(bids_folder) / 'derivatives' / folder
    tabs, summs = [], []
    for d in sorted(root.glob('sub-*')):
        if not list(d.glob('*_trialse_shifts.npz')):
            continue
        t, s = subject_rows(d)
        tabs.append(t)
        summs.append(s)
        print(d.name, t.groupby('pool').size().to_dict(), flush=True)
    tab = pd.concat(tabs)
    tab.to_csv(out, sep='\t', index=False)
    summ = pd.concat(summs)
    summ.to_csv(str(out).replace('.tsv', '_area_summary.tsv'), sep='\t', index=False)
    print(summ.groupby(['roi', 'kind'])[['median_se', 'sd_estimate', 'reliability']].median().round(3).to_string())
    print(f'Saved {out}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--bids-folder', default='/shares/zne.uzh/gdehol/ds-retsupp')
    p.add_argument('--folder', default='af_trialwise_trialse_polished_2026-11-13')
    p.add_argument('--out', required=True)
    a = p.parse_args()
    main(a.bids_folder, a.folder, a.out)
