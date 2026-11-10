"""Shared loading and statistics for Figures 3 and 5 and Table 1."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import integrate, stats

from retsupp.visualize.paper.paper_style import ROI_ORDER


def joint(table, family, model='m7_full'):
    d = pd.read_csv(table, sep='\t')
    d = d[(d.family == family) & (d.model == model) & d.roi.isin(ROI_ORDER)].copy()
    d['dS'] = d.g_HP - d.g_LP
    d['dD'] = d.g_HP_dyn - d.g_LP_dyn
    d['T'] = d.g_T_dyn
    d['D'] = (d.g_HP_dyn + 3 * d.g_LP_dyn) / 4   # distractor gain averaged over locations, weighted by location count
    return d


def conditional(table):
    d = pd.read_csv(table, sep='\t')
    d = d[d.valid & d.roi.isin(ROI_ORDER)]
    return d.pivot_table(index=['subject', 'roi'], columns='mech', values='estimate').reset_index()


def median_ci(x, n_boot=5000, seed=0):
    x = np.asarray(x)[np.isfinite(x)]
    rng = np.random.default_rng(seed)
    boots = np.median(rng.choice(x, (n_boot, len(x))), axis=1)
    return np.median(x), *np.percentile(boots, [2.5, 97.5])


def jzs_bf10(x, r=np.sqrt(2) / 2):
    """One-sample JZS Bayes factor (Rouder et al. 2009) for mean != 0."""
    x = np.asarray(x)[np.isfinite(x)]
    n = len(x)
    t = stats.ttest_1samp(x, 0).statistic
    nu = n - 1
    num = (1 + t ** 2 / nu) ** (-(nu + 1) / 2)
    f = lambda g: (1 + n * g * r ** 2) ** -.5 * (1 + t ** 2 / ((1 + n * g * r ** 2) * nu)) ** (-(nu + 1) / 2) * \
        (2 * np.pi) ** -.5 * g ** -1.5 * np.exp(-1 / (2 * g))
    den, _ = integrate.quad(f, 0, np.inf)
    return den / num


LOSO_TABLE = 'notes/data/loso_per_roi_2026-11.tsv'


def generalization(term, family='dog', table=None):
    """Per ROI: % of subjects predicted better with the others' size of the term (+δ) than with -δ, and BH q.
    table: default the module's LOSO_TABLE at call time (a caller may set it)."""
    d = pd.read_csv(table or LOSO_TABLE, sep='\t')
    d = d[(d.family == family) & (d.mech == term)]
    pct, ps = [], []
    for roi in ROI_ORDER:
        v = d[d.roi == roi].dr2_sign.dropna()
        pct.append((v > 0).mean() * 100)
        # One-sided: the question is whether the others' estimate predicts better than its opposite.
        ps.append(stats.wilcoxon(v, alternative='greater').pvalue)
    return pd.DataFrame(dict(roi=ROI_ORDER, pct=pct, q=stats.false_discovery_control(ps)))


def generalization_row(ax, term, color, lo, hi, label=True):
    """A band under the data: one square per ROI, filled where the term predicts new subjects (q < .05, > 50%).

    Extends the y-limits downwards; the left spine stays on the data range."""
    from retsupp.visualize.paper.paper_style import ROI_X
    span = hi - lo
    band = .17 * span
    ax.set_ylim(lo - band, hi)
    ax.spines['left'].set_bounds(lo, hi)
    y = lo - .62 * band
    g = generalization(term)
    for _, r in g.iterrows():
        x = ROI_X[r.roi]
        if r.q < .05 and r.pct > 50:
            ax.plot(x, y, 's', ms=4.2, color=color, mec=color, clip_on=False, zorder=5)
        else:
            ax.plot(x, y, 's', ms=4.2, mfc='w', mec=color, mew=.8, clip_on=False, zorder=5)
    # The band is labelled as a y tick, so the label sits with the other tick labels and never on data.
    ticks = [t for t in ax.get_yticks() if lo - 1e-9 <= t <= hi + 1e-9]
    labels = [f'{t:g}'.replace('-', '−') for t in ticks]
    if label:
        ticks, labels = ticks + [y], labels + ['New subjects']
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)
    if label:
        tl = ax.get_yticklabels()[-1]
        tl.set_fontsize(5.5)
        tl.set_color('.4')
        ax.yaxis.get_major_ticks()[-1].tick1line.set_visible(False)
    return g


def item_timecourses(folder='notes/data/event_deconv_2026-11', groups=None):
    """Deconvolved responses at the item locations (deconvolve_location_responses.py), per subject, group and class.

    groups: {roi: group name}; None pools all ROIs. ROIs are pooled weighted by voxel count; each subject's
    pre-onset mean (-4 to 0 s) per group and class is subtracted."""
    from pathlib import Path
    d = pd.concat([pd.read_csv(f, sep='\t') for f in sorted(Path(folder).glob('sub-*_2026-11.tsv'))])
    d = d[d.roi.isin(ROI_ORDER)].copy()
    d['grp'] = d.roi.map(groups) if groups else 'All'
    d = d.dropna(subset=['grp'])
    d['t'] = d.t.round(2)
    d['w'] = d.response * d.n_voxels
    g = d.groupby(['subject', 'grp', 'cls', 't'])
    p = (g.w.sum() / g.n_voxels.sum()).rename('r').reset_index()
    base = p[p.t < 0].groupby(['subject', 'grp', 'cls']).r.mean()
    p['r'] -= p.set_index(['subject', 'grp', 'cls']).index.map(base)
    return p


def tc_line(ax, p, cls, color, label=None, ls='-'):
    """Mean over subjects ± SEM band of one class's time course."""
    x = p[p.cls == cls].pivot(index='subject', columns='t', values='r')
    m, se = x.mean(), x.sem()
    ax.fill_between(m.index, m - se, m + se, color=color, alpha=.18, lw=0)
    ax.plot(m.index, m.values, color=color, lw=1.3, ls=ls, label=label)
    return m
