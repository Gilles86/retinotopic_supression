"""Supplementary figure and tables: baseline shift vs response gain at the HP location (n = 28; current final fits).

Collects ``derivatives/baseline_vs_gain_2026-11-12/sub-XX/*_{terms,sim}.tsv`` (``retsupp.modeling.fit_baseline_terms``,
rsynced to ``--in-dir``) and the step-1 slope table (``baseline_vs_gain.py step1``), writes the results tables to
notes/data and the figure figS_baseline_vs_gain_n28 (a schematic, b bar-response slope at HP, c held-out ΔR² of the
trial-locked variants against the joint fit, d recovery confusion matrix).

    python -m retsupp.visualize.paper.figS_baseline_vs_gain --in-dir /data/ds-retsupp/derivatives/baseline_vs_gain_2026-11-12
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from retsupp.visualize.paper.paper_style import DISTRACTOR, FG, HP, MUTED, TARGET, offset_spines, save

ROIS = ['V1', 'V2', 'V3', 'V3AB', 'IPS', 'hV4', 'VO', 'LO', 'TO']
GAIN_TERMS = {'dS', 'dD', 'dT'}           # fractional gain → % (×100)
RNG = np.random.default_rng(0)


def boot(x, n=10000):
    x = np.asarray(x)
    return np.percentile([np.median(RNG.choice(x, len(x))) for _ in range(n)], [2.5, 97.5])


def pooled(d, col):
    """Per subject the median over areas, then median [CI] over subjects, k < 0, k > 0, Wilcoxon."""
    s = d.groupby('subject')[col].median().dropna()
    lo, hi = boot(s)
    return dict(n=len(s), median=s.median(), ci_lo=lo, ci_hi=hi, k_neg=int((s < 0).sum()), k_pos=int((s > 0).sum()),
                p=wilcoxon(s).pvalue if len(s) > 5 and (s != 0).any() else np.nan)


def collect(in_dir, exclude=(1, 2)):
    t = pd.concat([pd.read_csv(f, sep='\t') for f in sorted(Path(in_dir).glob('sub-*/*_terms.tsv'))])
    t = t[~t.subject.isin(exclude)]
    sims = sorted(Path(in_dir).glob('sub-*/*_sim.tsv'))
    s = pd.concat([pd.read_csv(f, sep='\t') for f in sims]) if sims else None
    if s is not None:
        s = s[~s.subject.isin(exclude)]
    return t, s


def results_table(t):
    rows = []
    for (design, term), g in t.groupby(['design', 'term'], sort=False):
        scale = 100. if term in GAIN_TERMS else 1.
        r = pooled(g.assign(e=g.estimate * scale), 'e')
        dr = pooled(g.drop_duplicates(['subject', 'roi']).assign(e=g.drop_duplicates(['subject', 'roi']).heldout_dr2 * 1e4), 'e')
        rows.append(dict(design=design, term=term, unit='%' if term in GAIN_TERMS else '% signal change (w = 1)',
                         **{f'est_{k}': v for k, v in r.items()},
                         heldout_dr2_e4_median=dr['median'], heldout_dr2_e4_ci_lo=dr['ci_lo'], heldout_dr2_e4_ci_hi=dr['ci_hi'],
                         heldout_k_better=dr['k_pos'], heldout_n=dr['n'], heldout_p=dr['p']))
    return pd.DataFrame(rows)


def change_table(t):
    """Do dS, dD, dT change when the trial-locked term is added? Paired per subject (median over areas)."""
    out = []
    for term, (a, b) in {'dS': ('dS', 'dS+trial_search'), 'dD': ('dDdT', 'trial_search+dDdT'),
                         'dT': ('dDdT', 'trial_search+dDdT')}.items():
        x = t[(t.design == a) & (t.term == term)].set_index(['subject', 'roi']).estimate
        y = t[(t.design == b) & (t.term == term)].set_index(['subject', 'roi']).estimate
        d = (100 * (y - x)).rename('e').reset_index()
        r = pooled(d, 'e')
        out.append(dict(term=term, without=f'{a}', with_=f'{b}', change_pct_median=r['median'], ci_lo=r['ci_lo'],
                        ci_hi=r['ci_hi'], p=r['p'], n=r['n']))
    return pd.DataFrame(out)


def confusion(s, alpha=.05):
    """Per generator x simulated dataset: which term the group test detects in the joint design dS + trial (search):
    per subject the median over areas of each estimate, Wilcoxon over subjects; detected = p < alpha with the sign of
    a suppression (negative)."""
    rows = []
    j = s[s.design == 'dS+trial_search']
    for (gen, k), g in j.groupby(['generator', 'sim']):
        det = {}
        for term in ('dS', 'trial_spec_search'):
            x = g[g.term == term].groupby('subject').estimate.median()
            p = wilcoxon(x).pvalue if (x != 0).any() else 1.
            det[term] = (p < alpha) and (x.median() < 0)
        label = ('both' if all(det.values()) else 'gain (dS)' if det['dS'] else
                 'trial-locked baseline' if det['trial_spec_search'] else 'neither')
        rows.append(dict(generator=gen, sim=k, detected=label))
    c = pd.DataFrame(rows)
    return pd.crosstab(c.generator, c.detected, normalize='index').reindex(
        columns=['gain (dS)', 'trial-locked baseline', 'both', 'neither'], fill_value=0.), c


def figure(step1, res, t, conf, name):
    W, H = 7.09, 2.35
    fig = plt.figure(figsize=(W, H))
    # a: schematic
    ax = fig.add_axes([.04, .2, .2, .62])
    x = np.linspace(0, 1, 200)
    drive = np.exp(-((x - .5) ** 2) / (2 * .12 ** 2))
    ax.plot(x, drive, color=MUTED, lw=1.2)
    ax.plot(x, .8 * drive, color=HP, lw=1.2)
    ax.plot(x, drive - .2, color=HP, lw=1.2, ls=(0, (2.5, 1.5)))
    ax.text(.5, 1.06, 'Bar crosses the location', ha='center', fontsize=6, color='.35')
    ax.text(.86, .62, 'Other location', fontsize=6, color='.45', ha='left')
    ax.text(.86, .45, 'HP: gain', fontsize=6, color=HP, ha='left')
    ax.text(.86, .28, 'HP: offset', fontsize=6, color=HP, ha='left')
    ax.set_xlim(0, 1.35)
    ax.set_ylim(-.3, 1.15)
    ax.set_axis_off()
    ax.text(-.05, 1.2, 'a', fontsize=8, weight='bold', transform=ax.transAxes)
    ax.text(.0, -.02, 'Gain scales the response;\nan offset shifts it (and is removed\nwith each run\'s mean)',
            fontsize=6, color='.35', transform=ax.transAxes, va='top')
    # b: step-1 slope at HP vs LP
    ax = fig.add_axes([.33, .26, .2, .56])
    st = step1[step1.contrast == 'hp_vs_lp'].set_index('group')
    order = ['All areas', 'Early (V1–V3)', 'Dorsal (V3AB, IPS)', 'Occipitotemporal (hV4, VO, LO, TO)']
    lab = ['All', 'Early', 'Dorsal', 'Occ.-temp.']
    for i, gname in enumerate(order):
        r = st.loc[gname]
        big = i == 0
        ax.plot([i, i], [r.ci_lo, r.ci_hi], color=HP, lw=1.6 if big else 1.1, alpha=1 if r.p < .05 else .45)
        ax.plot(i, r.median_pct, 'o', ms=5 if big else 4, color=HP, mec='w', mew=.5, alpha=1 if r.p < .05 else .5)
        star = '**' if r.p < .01 else '*' if r.p < .05 else ''
        ax.text(i, r.ci_hi + .5, star, ha='center', fontsize=7, color=HP)
    ax.axhline(0, color='.8', lw=.6, zorder=0)
    ax.set_xticks(range(4))
    ax.set_xticklabels(lab, fontsize=6)
    ax.set_xlim(-.5, 3.5)
    ax.set_ylabel('Bar-response slope,\nHP vs LP (%)', fontsize=6.5)
    ax.tick_params(labelsize=6)
    offset_spines(ax)
    ax.text(-.32, 1.12, 'b', fontsize=8, weight='bold', transform=ax.transAxes)
    ax.set_title('Smaller response to the bar at HP', fontsize=6.5, loc='left', pad=6)
    # c: held-out ΔR² of the trial-locked variants vs the joint fit
    ax = fig.add_axes([.62, .26, .13, .56])
    designs = [('trial_search', 'Locked to\nsearch'), ('trial_fix', 'Locked to\nfixation')]
    for i, (d, lab_) in enumerate(designs):
        g = t[t.design == d].drop_duplicates(['subject', 'roi'])
        sub = g.groupby('subject').heldout_dr2.median() * 1e4
        jit = RNG.uniform(-.15, .15, len(sub))
        col = '.35'
        ax.scatter(i + jit, sub, s=4, color=col, alpha=.45, lw=0)
        lo, hi = boot(sub)
        ax.plot([i + .28] * 2, [lo, hi], color=col, lw=1.2)
        ax.plot(i + .28, sub.median(), 'o', ms=4, color=col, mec='w', mew=.5)
    ax.axhline(0, color='.8', lw=.6, zorder=0)
    ax.set_xticks(range(len(designs)))
    ax.set_xticklabels([l for _, l in designs], fontsize=6)
    ax.set_xlim(-.5, len(designs) - .3)
    ax.set_ylabel('Held-out ΔR² vs joint fit\n(× 10⁻⁴, run 3)', fontsize=6.5)
    ax.tick_params(labelsize=6)
    offset_spines(ax)
    ax.text(-.4, 1.12, 'c', fontsize=8, weight='bold', transform=ax.transAxes)
    ax.set_title('Trial-locked, held out', fontsize=6.5, loc='left', pad=6)
    # d: confusion matrix
    ax = fig.add_axes([.85, .3, .13, .5])
    if conf is not None:
        gens = [g for g in ('multiplicative', 'trial_additive', 'sustained_additive') if g in conf.index]
        m = conf.loc[gens].to_numpy()
        ax.imshow(m, cmap='Greys', vmin=0, vmax=1, aspect='auto')
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                ax.text(j, i, f'{m[i, j]:.1f}', ha='center', va='center', fontsize=6, color='w' if m[i, j] > .5 else FG)
        ax.set_yticks(range(len(gens)))
        ax.set_yticklabels({'multiplicative': 'Gain', 'trial_additive': 'Trial-locked\nbaseline',
                            'sustained_additive': 'Sustained\nbaseline'}[g] if False else
                           [{'multiplicative': 'Gain', 'trial_additive': 'Trial-locked\nbaseline',
                             'sustained_additive': 'Sustained\nbaseline'}[g] for g in gens], fontsize=6)
        ax.set_xticks(range(4))
        ax.set_xticklabels(['Gain', 'Trial', 'Both', 'Neither'], fontsize=6, rotation=45, ha='right')
        ax.set_ylabel('Simulated', fontsize=6.5)
        ax.set_xlabel('Detected', fontsize=6.5, labelpad=1)
    ax.text(-.62, 1.12, 'd', fontsize=8, weight='bold', transform=ax.transAxes)
    ax.set_title('Recovery', fontsize=6.5, loc='left', pad=6)
    return fig


def main(in_dir, step1_path, name):
    t, s = collect(in_dir)
    res = results_table(t)
    chg = change_table(t)
    res.to_csv('notes/data/baseline_vs_gain_terms_n28_2026-11-12.tsv', sep='\t', index=False)
    chg.to_csv('notes/data/baseline_vs_gain_change_n28_2026-11-12.tsv', sep='\t', index=False)
    conf = None
    if s is not None:
        conf, per = confusion(s)
        conf.to_csv('notes/data/baseline_vs_gain_confusion_n28_2026-11-12.tsv', sep='\t')
        per.to_csv('notes/data/baseline_vs_gain_confusion_per_sim_n28_2026-11-12.tsv', sep='\t', index=False)
    kept = t.add_sustained_kept.max()
    step1 = pd.read_csv(step1_path, sep='\t')
    print(res.round(4).to_string(index=False))
    print(chg.round(4).to_string(index=False))
    print(conf)
    print('additive sustained regressor kept after run-mean removal, max over subject x ROI:', kept)
    fig = figure(step1, res, t, conf, name)
    leg = Path('notes/figures/paper_2026-11') / f'{name}.md'
    save(fig, name, leg.read_text().strip() if leg.exists() else '(legend: see notes/baseline_vs_gain_2026-11-12.md)')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--in-dir', default='/data/ds-retsupp/derivatives/baseline_vs_gain_2026-11-12')
    ap.add_argument('--step1', default='notes/data/baseline_vs_gain_step1_n28_2026-11-12.tsv')
    ap.add_argument('--name', default='figS_baseline_vs_gain_n28')
    a = ap.parse_args()
    main(a.in_dir, a.step1, a.name)
