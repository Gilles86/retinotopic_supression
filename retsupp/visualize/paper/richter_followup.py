"""Follow-up analyses for the comparison with Richter, van Moorselaar & Theeuwes (eLife 2025), 2026-11-11.

Local, from existing tables (no fits): behaviour by location relative to HP (B), behaviour by HP history (A4), medians
without the two pilots (D2), bootstrapped CIs (D1), target − distractor at the item locations (D5), location contrasts
per early area (D6), and the m7 prediction of the Fig. 3 location contrasts (C, full model only).

    python -m retsupp.visualize.paper.richter_followup --out-dir notes/data

Writes ``richter_*_2026-11-11.tsv`` into ``--out-dir``. Medians of within-subject differences come with a percentile
bootstrap CI over subjects (10 000 resamples, seed 0); means with SEM and a bootstrap CI of the mean.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

RNG_SEED, N_BOOT = 0, 10_000
MIN_SUBJECT = 3   # sub-01, sub-02 (pilots: faster bar, irregular HP schedule) are left out of the paper; 1 = all 30
SUFFIX = '2026-11-11'
# Location codes 1, 3, 5, 7 (upper right, upper left, lower left, lower right) run counterclockwise around the ring.
NEXT = {1: (3, 7), 3: (5, 1), 5: (7, 3), 7: (1, 5)}
OPP = {1: 5, 3: 7, 5: 1, 7: 3}


def boot_ci(v, stat=np.median, n=N_BOOT, seed=RNG_SEED):
    v = np.asarray(pd.Series(v).dropna(), float)
    rng = np.random.default_rng(seed)
    b = stat(v[rng.integers(0, len(v), (n, len(v)))], axis=1)
    return np.percentile(b, [2.5, 97.5])


def describe(v, label, **extra):
    v = pd.Series(v).dropna()
    lo, hi = boot_ci(v)
    mlo, mhi = boot_ci(v, np.mean)
    p = stats.wilcoxon(v).pvalue if len(v) > 5 and (v != 0).any() else np.nan
    return dict(label=label, n=len(v), median=v.median(), ci_lo=lo, ci_hi=hi, mean=v.mean(), sem=v.sem(),
                mean_ci_lo=mlo, mean_ci_hi=mhi, pct_pos=(v > 0).mean() * 100, p=p, **extra)


# ---------------------------------------------------------------- behaviour
def load_trials(path):
    t = pd.read_csv(path, sep='\t')
    t = t[t.subject >= MIN_SUBJECT].copy()
    t['ok'] = (t.correct == True) & t.rt.notna()   # noqa: E712  (correct is object: True/False/NaN)
    t['acc'] = (t.correct == True).astype(float)    # misses (no response) count as errors  # noqa: E712
    t['rt_ms'] = t.rt * 1000
    hp = t.HPL_distractor.astype(int)
    rel = lambda loc: np.select([loc == hp, loc == hp.map(OPP)], ['hp', 'opposite'], 'next')   # noqa: E731
    t['dist_rel'] = np.where(t.distractor_location == 10, 'absent', rel(t.distractor_location.astype(int)))
    t['target_rel'] = rel(t.target_location.astype(int))
    return t


def cell_means(t, col, levels, mask=None):
    d = t if mask is None else t[mask]
    rt = d[d.ok].groupby(['subject', col]).rt_ms.mean().unstack()[levels]
    acc = d.groupby(['subject', col]).acc.mean().unstack()[levels] * 100
    return rt, acc


def behaviour_by_relation(t):
    """B: RT (correct trials) and accuracy for the distractor and the target at HP, next to HP, opposite HP."""
    rows, cells = [], []
    specs = [('distractor', 'dist_rel', ['hp', 'next', 'opposite', 'absent'], None),
             ('target', 'target_rel', ['hp', 'next', 'opposite'], t.distractor_location != 10)]
    for item, col, lev, mask in specs:
        rt, acc = cell_means(t, col, lev, mask)
        for measure, m in (('rt_ms', rt), ('accuracy_pct', acc)):
            for c in lev:
                cells.append(dict(item=item, measure=measure, cell=c, n=m[c].notna().sum(), mean=m[c].mean(),
                                  median=m[c].median()))
            for a, b in (('next', 'opposite'), ('hp', 'next'), ('hp', 'opposite')) + ((('hp', 'absent'),) if item == 'distractor' else ()):
                rows.append(describe(m[a] - m[b], f'{a} − {b}', item=item, measure=measure))
    return pd.DataFrame(rows), pd.DataFrame(cells)


def add_history(t):
    """Block (1, 2) within session = which of the session's two HP locations; history of each location."""
    t = t.copy()
    first = t.sort_values(['session', 'run']).groupby(['subject', 'session']).HPL_distractor.transform('first').reindex(t.index)
    t['block'] = np.where(t.HPL_distractor == first, 1, 2)
    hp_blocks = t.groupby(['subject', 'session', 'block']).HPL_distractor.agg(lambda x: x.mode()[0])
    ses1 = t[t.session == 1].groupby('subject').HPL_distractor.agg(lambda x: set(x.astype(int)))

    def history(row, loc):
        if loc == 10:
            return 'absent'
        if loc == row.HPL_distractor:
            return 'current'
        if row.block == 2 and loc == hp_blocks[(row.subject, row.session, 1)]:
            return 'prev_block'
        if row.session == 2 and loc in ses1[row.subject]:
            return 'prev_session'
        return 'never'
    t['dist_hist'] = [history(r, int(r.distractor_location)) for r in t.itertuples()]
    t['target_hist'] = [history(r, int(r.target_location)) for r in t.itertuples()]
    return t


def behaviour_by_history(t):
    """A4: RT of a distractor (target) at a location that was HP before vs one that never was, sub-03–30.

    Was-HP-before only occurs at fixed relations to the current HP (prev block: opposite, block 2; prev session: next,
    session 2) and later in the experiment, so each is compared with the same relation when it was never HP, minus the
    practice change over the same step at a location whose history did not change (difference in differences)."""
    t = add_history(t[t.subject >= 3])   # the regular HP schedule (also with --all-subjects)
    ok = t[t.ok]
    rows, cells = [], []
    for item, rel, hist in (('distractor', 'dist_rel', 'dist_hist'), ('target', 'target_rel', 'target_hist')):
        d = ok if item == 'distractor' else ok[ok.distractor_location != 10]
        g = d.groupby(['subject', 'session', 'block', rel, hist]).rt_ms.mean()
        cells.append(g.rename('rt_ms').reset_index().rename(columns={rel: 'relation', hist: 'history'}).assign(item=item))
        m = d.groupby(['subject', 'session', 'block', rel]).rt_ms.mean().unstack([1, 2, 3])
        # Previous block: opposite in block 2 (was HP in block 1) vs opposite in block 1 (never), both sessions pooled;
        # practice reference: next to HP, block 2 − block 1 (its history is the same in both blocks).
        opp = m.xs('opposite', axis=1, level=2)
        nxt = m.xs('next', axis=1, level=2)
        d_opp = (opp.xs(2, axis=1, level=1) - opp.xs(1, axis=1, level=1)).mean(1)
        d_nxt = (nxt.xs(2, axis=1, level=1) - nxt.xs(1, axis=1, level=1)).mean(1)
        rows.append(describe(d_opp, 'opposite: block 2 (was HP in block 1) − block 1 (never HP)', item=item))
        rows.append(describe(d_opp - d_nxt, 'previous-block HP, difference in differences '
                             '[opposite b2 − b1] − [next b2 − b1]', item=item))
        # Previous session: next to HP in session 2 (both were HP in session 1) vs session 1 (never); practice
        # reference: opposite in block 1 (never HP in both sessions).
        n2 = nxt.xs(2, axis=1, level=0).mean(1)
        n1 = nxt.xs(1, axis=1, level=0).mean(1)
        o2, o1 = opp[(2, 1)], opp[(1, 1)]
        rows.append(describe(n2 - n1, 'next: session 2 (was HP in session 1) − session 1 (never HP)', item=item))
        rows.append(describe((n2 - n1) - (o2 - o1), 'previous-session HP, difference in differences '
                             '[next s2 − s1] − [opposite (block 1) s2 − s1]', item=item))
    return pd.DataFrame(rows), pd.concat(cells)


def pilots(t):
    """D2: HP benefit and target cost without sub-01, sub-02: mean and median with CIs."""
    ok = t[t.ok & (t.subject >= 3)]   # D2 is the n = 28 number by definition
    d = ok.groupby(['subject', 'dist_cond']).rt_ms.mean().unstack()
    g = ok[ok.distractor_location != 10].groupby(['subject', 'target_cond']).rt_ms.mean().unstack()
    return pd.DataFrame([describe(d.HP - d.LP, 'distractor at HP − LP, sub-03–30'),
                         describe(g.HP - g.LP, 'target at HP − LP (distractor trials), sub-03–30')])


# ---------------------------------------------------------------- item locations (Fig. 3)
def roi_cells(folder, groups):
    """As fig3_item_locations.rel_stats, for arbitrary ROI groups: window means per item × location type in % of the
    subject's mean over the twelve cells. Returns per subject × group the location contrasts (mean over items)."""
    from retsupp.visualize.paper.fig3_item_locations import ITEMS, PEAK
    from retsupp.visualize.paper.fig45_common import item_timecourses
    p = item_timecourses(folder, groups)
    pk = p[p.t.between(*PEAK)].groupby(['subject', 'grp', 'cls']).r.mean().unstack().dropna()
    pk = pk[pk.mean(1) > 0]
    pk = pk[pk.index.get_level_values('subject') >= MIN_SUBJECT]
    pk = pk.div(pk.mean(1), axis=0) * 100
    loc = {c: pk[[f'{it}_{c}' for it, _, _ in ITEMS]].mean(1) for c in ('hp', 'nextleft', 'nextright', 'opposite')}
    nxt = (loc['nextleft'] + loc['nextright']) / 2
    lp = (loc['nextleft'] + loc['nextright'] + loc['opposite']) / 3
    return pd.DataFrame({'hp_next': loc['hp'] - nxt, 'next_opp': nxt - loc['opposite'],
                         'hp_opp': loc['hp'] - loc['opposite'], 'hp_lp': loc['hp'] - lp}).reset_index()


def location_contrasts(rel_folder, label):
    from retsupp.visualize.paper.fig2_prf_fits import PATHWAY
    groups = {r: g for g, rs in PATHWAY.items() for r in rs}
    c = pd.concat([roi_cells(rel_folder, groups), roi_cells(rel_folder, None).assign(grp='All areas')])
    rows = []
    for g in ('Early', 'All areas'):
        for k in ('hp_lp', 'hp_opp', 'hp_next', 'next_opp'):
            rows.append(describe(c[c.grp == g][k], k, group=g, source=label))
    return pd.DataFrame(rows), c


def per_area(rel_folder):
    """D6: HP − opposite and next − opposite for V1, V2, V3 separately (uncorrected, exploratory)."""
    c = roi_cells(rel_folder, {r: r for r in ('V1', 'V2', 'V3')})
    rows = [describe(c[c.grp == r][k], k, group=r) for r in ('V1', 'V2', 'V3') for k in ('hp_opp', 'next_opp', 'hp_next')]
    return pd.DataFrame(rows)


def target_vs_distractor(folder):
    """D5: target − distractor, both at LP locations, per pathway, window mean (% signal), BH over the pathways."""
    from retsupp.visualize.paper.fig2_prf_fits import PATHWAY
    from retsupp.visualize.paper.fig3_item_locations import PEAK
    from retsupp.visualize.paper.fig45_common import item_timecourses
    p = item_timecourses(folder, {r: g for g, rs in PATHWAY.items() for r in rs})
    w = p[p.t.between(*PEAK)].groupby(['subject', 'grp', 'cls']).r.mean().unstack()
    w = w[w.index.get_level_values('subject') >= MIN_SUBJECT]
    rows = []
    for g in PATHWAY:
        x = w.xs(g, level='grp')
        rows.append(describe(x.target_lp - x.distractor_lp, 'target − distractor (both at LP)', group=g))
        rows.append(describe((x.target_hp + x.target_lp) / 2 - x.distractor_lp,
                             'target (HP and LP averaged) − distractor (LP)', group=g))
    out = pd.DataFrame(rows)
    for lab in out.label.unique():
        m = out.label == lab
        out.loc[m, 'q'] = stats.false_discovery_control(out.loc[m, 'p'])
    return out


# ---------------------------------------------------------------- D1: other intervals
def d1_intervals(behav_trials):
    R = ['V1', 'V2', 'V3', 'V3AB', 'hV4', 'LO', 'TO', 'VO', 'IPS']
    rows = []
    s = pd.read_csv('notes/data/prf_jointhrf_summary_final2_2026-11.tsv', sep='\t')
    s = s[(s.stage == 3) & s.roi.isin(R) & (s.subject >= MIN_SUBJECT)]
    x = s.groupby(['family', 'subject', 'roi']).r2_test_reliable.mean().groupby(['family', 'subject']).median().unstack(0)
    rows.append(describe(x.dn - x.dog, 'held-out R², DN − DoG (stage 3, fold mean, median over nine areas)',
                         source='notes/data/prf_jointhrf_summary_final2_2026-11.tsv'))
    # dS estimated alone, as key_numbers: DoG from the linear conditional estimates (valid fits), Gaussian and DN from
    # the exact ΔSSE minima (areas with >= 20 voxels); per-subject median over areas.
    lin = pd.read_csv('notes/data/conditional_estimates_dog_final2_2026-11.tsv', sep='\t')
    lin = lin[(lin.mech == 'dS') & lin.valid & lin.roi.isin(R) & (lin.subject >= MIN_SUBJECT)]
    ds_dog = lin.groupby('subject').estimate.median()
    rows.append(describe(ds_dog, 'dS estimated alone, dog (per-subject median over areas)',
                         source='notes/data/conditional_estimates_dog_final2_2026-11.tsv'))
    c = pd.read_csv('notes/data/conditional_exact_final2_allfam_2026-11.tsv', sep='\t')
    c = c[(c.mech == 'dS') & (c.nvox >= 20) & c.roi.isin(R) & (c.subject >= MIN_SUBJECT)]
    for fam in ('gauss', 'dn'):
        v = c[c.family == fam].groupby('subject').own.median()
        rows.append(describe(v, f'dS estimated alone, {fam} (per-subject median over areas)',
                             source='notes/data/conditional_exact_final2_allfam_2026-11.tsv'))
    g = pd.read_csv('notes/data/gain_field_parameters_final2_2026-11.tsv', sep='\t')
    g = g[g.subject >= MIN_SUBJECT]
    a = g[(g.family == 'dog') & (g.model == 'm8_full_adjopp') & g.roi.isin(R)].copy()
    # m8_full_adjopp: g_LP is the mean of the three LP gains, g_adj_minus_opp = next − opposite (collect_lp_split), so
    # next = g_LP + d/3 and opposite = g_LP − 2d/3.
    dd = a.g_adj_minus_opp
    for lab, v in (('split model: next − opposite', dd), ('split model: HP − next', a.g_HP - a.g_LP - dd / 3),
                   ('split model: HP − opposite', a.g_HP - a.g_LP + 2 * dd / 3)):
        rows.append(describe(v.groupby(a.subject).median(), lab + ' (per-subject median over areas)',
                             source='notes/data/gain_field_parameters_final2_2026-11.tsv (m8_full_adjopp)'))
    # Between subjects: dS (estimated alone, DoG; joint m7) vs the behavioural HP benefit (LP − HP RT, ms).
    ok = behav_trials[behav_trials.ok]
    d = ok.groupby(['subject', 'dist_cond']).rt_ms.mean().unstack()
    benefit = d.LP - d.HP
    m7 = g[(g.family == 'dog') & (g.model == 'm7_full') & g.roi.isin(R)]
    for lab, ds in (('dS alone (DoG)', ds_dog),
                    ('dS joint m7 (DoG)', (m7.g_HP - m7.g_LP).groupby(m7.subject).median())):
        j = pd.concat([ds.rename('ds'), benefit.rename('b')], axis=1).dropna()
        rho, p = stats.spearmanr(j.ds, j.b)
        rng = np.random.default_rng(RNG_SEED)
        bs = [stats.spearmanr(*j.iloc[rng.integers(0, len(j), len(j))].T.values)[0] for _ in range(N_BOOT)]
        lo, hi = np.nanpercentile(bs, [2.5, 97.5])
        rows.append(dict(label=f'Spearman ρ, {lab} vs HP benefit (LP − HP RT)', n=len(j), median=rho, ci_lo=lo,
                         ci_hi=hi, p=p, source='conditional_exact / gain_field_parameters + behaviour trials'))
    return pd.DataFrame(rows)


def main(out_dir, trials_path, rel_folder, hplp_folder, m7_pred, all_subjects=False):
    global MIN_SUBJECT
    MIN_SUBJECT = 1 if all_subjects else 3
    out = Path(out_dir)
    tag = 'n30' if all_subjects else 'n28'
    t = load_trials(trials_path)
    w = lambda df, name: df.to_csv(out / f'richter_{name}_{tag}_{SUFFIX}.tsv', sep='\t', index=False, float_format='%.5g')  # noqa
    b, bc = behaviour_by_relation(t)
    w(b, 'B_behaviour_contrasts'), w(bc, 'B_behaviour_cells')
    h, hc = behaviour_by_history(t)
    w(h, 'A4_behaviour_history'), w(hc, 'A4_behaviour_history_cells')
    w(pilots(t), 'D2_without_pilots')
    w(d1_intervals(t), 'D1_intervals')
    data, _ = location_contrasts(rel_folder, 'data')
    pred, _ = location_contrasts(m7_pred, 'm7 prediction')
    w(pd.concat([data, pred]), 'C_D1_location_contrasts')
    w(per_area(rel_folder), 'D6_per_area')
    w(target_vs_distractor(hplp_folder), 'D5_target_vs_distractor')
    print('Saved', out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out-dir', default='notes/data')
    ap.add_argument('--trials', default='notes/data/behaviour_trials_2026-11-11.tsv.gz')
    ap.add_argument('--rel-folder', default='notes/data/event_deconv_relative_final2_2026-11')
    ap.add_argument('--hplp-folder', default='notes/data/event_deconv_final2_2026-11')
    ap.add_argument('--m7-pred', default='notes/data/kappa_neighbour_2026-11/event_deconv_m7')
    ap.add_argument('--all-subjects', action='store_true', help='n = 30 (with the pilots) instead of n = 28')
    a = ap.parse_args()
    main(a.out_dir, a.trials, a.rel_folder, a.hplp_folder, a.m7_pred, a.all_subjects)
