"""Supplementary figure: Fig. 6 decoding under three voxel selections, plus decoders on pooled area sets.

Selections (all per cross-validation fold, on the fold's training runs only, so the decoded run never influences its
voxels; BOLD SD <= 20%; at most 2000 voxels per area, highest reliability first; area skipped for a subject if < 20):

* ``current`` (Fig. 6): leave-one-run-out reliability R² > 0 (``prf_cv_roi_reliability``; roughly r > 0.3).
* ``r0.2`` / ``r0.1``: cross-run reliability r > 0.2 / 0.1 and circular-shift p <= .01
  (``modeling/cv/voxel_reliability_null.py --folds``; the gain-field robustness check's rule, per fold).

Pooled sets (``decode_final.POOLED``): one decoder on the union of the areas' voxels, each voxel with its own fold
pRF; current and r > 0.2 selection, cap 2000 per set.

Inputs (``notes/data``, from ``fig6_decoding.py collect`` and ``fig6_collect.py`` with ``--source <root>``):
``decoding_readout_<root>_2026-11.tsv`` and ``decoding_quality_<root>_2026-11.tsv`` for the roots
``jointhrf``, ``jointhrf_relnull_r0.2``, ``jointhrf_relnull_r0.1``, ``jointhrf_pooled``, ``jointhrf_relnull_r0.2_pooled``.

Outputs: ``notes/data/decoding_voxel_selection_2026-11.tsv`` (every number of the figure, long format) and
``notes/figures/paper_2026-11/<name>.pdf/.png/.md``.

Usage::

    python -m retsupp.visualize.paper.figS_decoding_voxel_selection --exclude-subjects 1 2 \\
        --name figS_decoding_voxel_selection_n28
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from retsupp.visualize.paper.fig6_decoding import (DECODER, EARLY, PATHWAYS, ROI_ORDER_ALL, area_stats,
                                                   pathway_effects, quadrant_effects, stars)

DATA = Path('notes/data')
SELECTIONS = [('current', 'jointhrf', 'Training R² > 0 (Fig. 6)'),
              ('r0.2', 'jointhrf_relnull_r0.2', 'r > 0.2, p ≤ .01'),
              ('r0.1', 'jointhrf_relnull_r0.1', 'r > 0.1, p ≤ .01')]
POOLED_ROOTS = [('current', 'jointhrf_pooled'), ('r0.2', 'jointhrf_relnull_r0.2_pooled')]
POOLED_SETS = [('mid', 'V3AB + hV4'), ('high', 'LO + TO + VO + IPS'), ('dorsal', 'V3AB + IPS'),
               ('occtemp', 'hV4 + VO + LO + TO')]
COLORS = {'current': '#1F2933', 'r0.2': '#3D7EB8', 'r0.1': '#7DAAD6'}
CAP = 2000


def load(root, excl):
    d = pd.read_csv(DATA / f'decoding_readout_{root}_2026-11.tsv', sep='\t')
    q = pd.read_csv(DATA / f'decoding_quality_{root}_2026-11.tsv', sep='\t')
    return d[~d.subject.isin(excl)], q[~q.subject.isin(excl)]


def quality(q, rois):
    """Per subject × ROI (free decoder): mean over runs of r and r_shuffle."""
    qs = q[(q.decoder == DECODER) & q.roi.isin(rois)].groupby(['subject', 'roi'])[['r', 'r_shuffle']].mean()
    return qs.reset_index()


def set_effect(d, roi):
    """Per subject: HP − mean of the other three quadrants, % of the mean of the four (Fig. 6d normalisation)."""
    s = quadrant_effects(d, DECODER, [roi])
    return s.hp - (s.left + s.right + s.opposite) / 3


def summarise(excl):
    rows, keep = [], {}

    def add(sel, kind, unit, v=None, **kw):
        r = dict(selection=sel, kind=kind, unit=unit, **kw)
        if v is not None:
            v = pd.Series(v).dropna()
            r.update(n=len(v), mean=v.mean(), sem=v.sem())
        rows.append(r)

    for sel, root, _ in SELECTIONS:
        if not (DATA / f'decoding_readout_{root}_2026-11.tsv').exists():
            print('missing', root)
            continue
        d, q = load(root, excl)
        one = d[(d.decoder == DECODER)].drop_duplicates(['subject', 'session', 'run', 'roi'])
        for roi in ROI_ORDER_ALL:
            g = one[one.roi == roi]
            per_sub = g.groupby('subject').n_voxels.mean()
            npass = g.n_voxels_pass if 'n_voxels_pass' in g else g.n_voxels
            add(sel, 'voxels', roi, per_sub, median_run=g.n_voxels.median(), n_runs=len(g),
                cap_binds_runs=int((npass > CAP).sum()))
        qs = quality(q, ROI_ORDER_ALL)
        for roi, g in qs.groupby('roi'):
            add(sel, 'quality_r', roi, g.r.to_numpy(), r_shuffle=g.r_shuffle.mean(),
                p=stats.wilcoxon(g.r - g.r_shuffle, alternative='greater').pvalue)
        early = qs[qs.roi.isin(EARLY)].groupby('subject').r.mean()
        add(sel, 'quality_r', 'V1-V3', early)
        qe = quadrant_effects(d, DECODER)
        for name, v, alt in [('HP − next', qe.hp - qe.adjacent, 'less'), ('HP − opposite', qe.hp - qe.opposite, 'less'),
                             ('Next − opposite', qe.adjacent - qe.opposite, 'two-sided')]:
            add(sel, 'quadrant', name, v, p=stats.wilcoxon(v.dropna(), alternative=alt).pvalue, test=alt)
        per, pw = pathway_effects(d, DECODER)
        ast = area_stats(per, pw)
        for k, s in ast['pooled'].items():
            add(sel, 'pooled_hp', k, s['v'], p=s['p'])
        for r, s in ast['area'].items():
            add(sel, 'area_hp', r, per[r], p=s['p'], q=s['q'])
        keep[sel] = dict(qs=qs, qe=qe, per=per, ast=ast, early=early)
    for sel, root in POOLED_ROOTS:
        f = DATA / f'decoding_readout_{root}_2026-11.tsv'
        if not f.exists():
            print('missing', f)
            continue
        d, q = load(root, excl)
        qs = quality(q, [s for s, _ in POOLED_SETS])
        one = d[(d.decoder == DECODER)].drop_duplicates(['subject', 'session', 'run', 'roi'])
        keep[sel + '_pooled'] = {}
        for s, _ in POOLED_SETS:
            if s not in set(d.roi):
                continue
            v = set_effect(d, s)
            g = qs[qs.roi == s]
            add(sel, 'set_hp', s, v, p=stats.wilcoxon(v.dropna(), alternative='less').pvalue)
            add(sel, 'set_quality_r', s, g.r.to_numpy(), r_shuffle=g.r_shuffle.mean(),
                p=stats.wilcoxon(g.r - g.r_shuffle, alternative='greater').pvalue)
            gv = one[one.roi == s]
            npass = gv.n_voxels_pass if 'n_voxels_pass' in gv else gv.n_voxels
            add(sel, 'set_voxels', s, gv.groupby('subject').n_voxels.mean(), median_run=gv.n_voxels.median(),
                n_runs=len(gv), cap_binds_runs=int((npass > CAP).sum()))
            keep[sel + '_pooled'][s] = dict(v=v, r=g.set_index('subject').r)
    # Bar sweep (Fig. 6c, outline decoder, V1–V3): per subject the segment differences averaged over the bar positions
    # where the segment exists; HP and next-to-HP segments and their difference.
    from retsupp.visualize.paper.fig6_decoding import sweep_segments
    from retsupp.visualize.paper.figS_decoded_bars import pooled_condition
    for sel, root, _ in SELECTIONS:
        f = DATA / f'decoded_bars_condition_{root}_outline_2026-11.npz'
        if not f.exists() or sel not in keep:
            print('missing', f)
            continue
        z = np.load(f)
        ok = ~np.isin(z['subjects'], list(excl))
        hp_v, nx_v, op_v = sweep_segments(pooled_condition(z['mean'][ok]), z['grid'], opposite=True)
        subs = z['subjects'][ok]
        hp, nx, op = (pd.Series(np.nanmean(v, 1), index=subs) for v in (hp_v, nx_v, op_v))
        add(sel, 'sweep', 'HP segment', hp, p=stats.wilcoxon(hp.dropna(), alternative='less').pvalue)
        add(sel, 'sweep', 'Next segment', nx, p=stats.wilcoxon(nx.dropna()).pvalue)
        add(sel, 'sweep', 'HP − next segment', hp - nx, p=stats.wilcoxon((hp - nx).dropna(), alternative='less').pvalue)
        keep[sel]['sweep'] = dict(hp=hp_v, nx=nx_v, op=op_v, centres=(z['edges'][:-1] + z['edges'][1:]) / 2)
    # Paired comparisons with the current selection (per subject): V1–V3 quality and pooled HP effects.
    for sel in [s for s in ('r0.2', 'r0.1') if s in keep]:
        a, b = keep['current']['early'], keep[sel]['early']
        j = a.index.intersection(b.index)
        add(sel, 'quality_vs_current', 'V1-V3', b[j] - a[j], p=stats.wilcoxon(b[j] - a[j]).pvalue)
    return pd.DataFrame(rows), keep


# ------------------------------------------------------------------------------------------------------------- plot
LIM = (-10.5, 11.6)      # y range of the subject-dot panels; values beyond it are drawn as triangles at the edge


def dots(ax, x, v, color, seed, w=.09, s=4, lim=LIM):
    v = pd.Series(v).dropna().to_numpy()
    rng = np.random.default_rng(seed)
    jx = x + rng.uniform(-w, w, len(v))
    lo, hi = lim[0] + .4, 7.6
    inr = (v >= lo) & (v <= hi)
    ax.scatter(jx[inr], v[inr], s=s, color=color, alpha=.35, lw=0, zorder=2)
    for m, y in (('^', hi), ('v', lo)):
        k = (v > hi) if m == '^' else (v < lo)
        if k.any():
            ax.scatter(jx[k], np.full(k.sum(), y), s=9, marker=m, facecolor='none', edgecolor=color, lw=.6, zorder=3)
    ax.errorbar(x + w + .07, v.mean(), yerr=v.std(ddof=1) / np.sqrt(len(v)), fmt='o', ms=3.6, color=color, mec='w', mew=.5, lw=1.1,
                capsize=0, zorder=4)


def p_stack(ax, x, y, ps, colors, fs=5.5, dy=1.05):
    """p values of one group, one line per selection in its colour (stars when p < .05), stacked down from y."""
    for i, (p, c) in enumerate(zip(ps, colors)):
        t = (f'p = {p:.3f}' if p >= .001 else f'p = {p:.0e}').replace('0.', '.')
        ax.text(x, y - i * dy, (stars(p) + ' ' if p < .05 else '') + t, ha='center', va='top', fontsize=fs, color=c)


def sel_key(fig, x, y, W, sels=('current', 'r0.2', 'r0.1')):
    """Horizontal key in figure inches: a marker and the selection's name in its colour."""
    lab = {s: l for s, _, l in SELECTIONS}
    for s in sels:
        fig.lines.append(__import__('matplotlib').lines.Line2D([x / W], [y / fig.get_figheight()], marker='o', ms=3.4,
                                                              color=COLORS[s], transform=fig.transFigure))
        t = fig.text((x + .07) / W, y / fig.get_figheight(), lab[s], fontsize=6.5, color=COLORS[s], va='center')
        fig.canvas.draw()
        x += t.get_window_extent().width / fig.dpi + .3


def plot(tab, keep, name, n_note):
    import matplotlib.pyplot as plt
    from retsupp.visualize.paper.paper_style import DOUBLE, HP, ROI_X, letter, offset_spines, roi_axis, save
    sels = [s for s, _, _ in SELECTIONS if s in keep]
    off = {'current': -.25, 'r0.2': 0., 'r0.1': .25}
    W, H = DOUBLE, 8.75
    fig = plt.figure(figsize=(W, H))
    fig.text(.05 / W, 8.6 / H, 'Voxel selection:', fontsize=6.5, color='.3', va='center')
    sel_key(fig, .95, 8.6, W)

    def axes(x, y, w, h):
        return fig.add_axes([x / W, y / H, w / W, h / H])

    # a: voxels per area
    ax = axes(.55, 6.85, 3.0, 1.35)
    for s in sels:
        t = tab[(tab.selection == s) & (tab.kind == 'voxels')].set_index('unit')
        xs = [ROI_X[r] + off[s] for r in ROI_ORDER_ALL]
        ax.errorbar(xs, t.loc[ROI_ORDER_ALL, 'mean'], yerr=t.loc[ROI_ORDER_ALL, 'sem'], fmt='o', ms=3, color=COLORS[s],
                    mec='w', mew=.4, lw=.9, capsize=0)
    ax.set_yscale('log')
    ax.set_yticks([30, 100, 300, 1000, 2000])
    ax.set_yticklabels(['30', '100', '300', '1000', '2000'])
    ax.axhline(CAP, color='.7', lw=.6, ls='--', zorder=0)
    ax.text(ROI_X['TO'] + .6, CAP * .9, 'Cap', fontsize=6, color='.5', ha='right', va='top')
    ax.set_ylabel('Voxels per area')
    offset_spines(ax)
    roi_axis(ax, rotate=False, fontsize=6.5)
    letter(ax, 'a', x=-.13, y=1.08)

    # b: decoding quality per area (+ pooled sets, current and r > 0.2)
    ax = axes(4.3, 6.85, 2.6, 1.35)
    for s in sels:
        t = tab[(tab.selection == s) & (tab.kind == 'quality_r')].set_index('unit')
        xs = [ROI_X[r] + off[s] for r in ROI_ORDER_ALL]
        ax.errorbar(xs, t.loc[ROI_ORDER_ALL, 'mean'], yerr=t.loc[ROI_ORDER_ALL, 'sem'], fmt='o', ms=3, color=COLORS[s],
                    mec='w', mew=.4, lw=.9, capsize=0)
        sh = t.loc[ROI_ORDER_ALL, 'r_shuffle'].mean()
    ax.axhline(0, color='.6', lw=.6, ls='--', zorder=0)
    ax.set_ylim(0, None)
    ax.set_ylabel('Decoded vs true bar (r)')
    offset_spines(ax)
    roi_axis(ax, rotate=False, fontsize=6.5)
    letter(ax, 'b', x=-.17, y=1.08)

    # c: V1–V3 quadrant contrasts
    ax = axes(.55, 4.4, 2.25, 1.6)
    names = ['HP − next', 'HP − opposite', 'Next − opposite']
    for i, nm in enumerate(names):
        for s in sels:
            qe = keep[s]['qe']
            v = {'HP − next': qe.hp - qe.adjacent, 'HP − opposite': qe.hp - qe.opposite,
                 'Next − opposite': qe.adjacent - qe.opposite}[nm]
            x = i + off[s] * 1.3
            dots(ax, x, v, COLORS[s], i * 10 + sels.index(s), w=.06, s=3)
        ps = [tab[(tab.selection == s) & (tab.kind == 'quadrant') & (tab.unit == nm)].p.iloc[0] for s in sels]
        p_stack(ax, i + .05, LIM[1], ps, [COLORS[s] for s in sels])
    ax.axhline(0, color='.6', lw=.6, ls='--', zorder=0)
    ax.set_ylim(*LIM)
    ax.set_yticks([-8, -4, 0, 4, 8])
    ax.set_xticks(range(3))
    ax.set_xticklabels(names, fontsize=6.5)
    ax.set_xlim(-.55, 2.6)
    ax.tick_params(axis='x', length=0)
    ax.set_ylabel('Difference (% of mean)')
    ax.text(.0, 1.08, 'V1–V3 pooled', transform=ax.transAxes, fontsize=6.5, color='.3', va='bottom')
    offset_spines(ax)
    letter(ax, 'c', x=-.2, y=1.12)

    # d: pooled HP − other three
    ax = axes(3.45, 4.4, 3.5, 1.6)
    groups = ['All areas', 'Early', 'Dorsal', 'Occipitotemporal']
    for i, k in enumerate(groups):
        for s in sels:
            st = keep[s]['ast']['pooled'][k]
            x = i + off[s] * 1.3
            dots(ax, x, st['v'], COLORS[s], 40 + i * 10 + sels.index(s), w=.06, s=3)
        p_stack(ax, i + .05, LIM[1], [keep[s]['ast']['pooled'][k]['p'] for s in sels], [COLORS[s] for s in sels])
    ax.axhline(0, color='.6', lw=.6, ls='--', zorder=0)
    ax.set_ylim(*LIM)
    ax.set_yticks([-8, -4, 0, 4, 8])
    ax.set_xticks(range(4))
    ax.set_xticklabels(['All areas', 'Early', 'Dorsal', 'Occipito-\ntemporal'], fontsize=6.5)
    ax.set_xlim(-.55, 3.6)
    ax.tick_params(axis='x', length=0)
    ax.set_ylabel('HP − other (%)')
    offset_spines(ax)
    letter(ax, 'd', x=-.14, y=1.12)

    # e: per area HP − other three, BH across nine
    ax = axes(.55, 2.2, 3.6, 1.45)
    for s in sels:
        for r in ROI_ORDER_ALL:
            st = keep[s]['ast']['area'][r]
            x = ROI_X[r] + off[s]
            sig = st['q'] < .05
            ax.errorbar(x, st['m'], yerr=st['se'], fmt='o', ms=3, color=COLORS[s], mec='w', mew=.4, lw=.9, capsize=0,
                        zorder=4)
            if sig:
                ax.text(x, 5.6 + .9 * sels.index(s), stars(st['q']), ha='center', va='center', fontsize=6.5,
                        color=COLORS[s])
    ax.axhline(0, color='.6', lw=.6, ls='--', zorder=0)
    ax.set_ylim(-7, 7.5)
    ax.set_yticks([-6, -3, 0, 3, 6])
    ax.set_ylabel('HP − other (%)')
    offset_spines(ax)
    roi_axis(ax, rotate=False, fontsize=6.5)
    letter(ax, 'e', x=-.13, y=1.08)

    # f: pooled-set decoders
    ax = axes(4.75, 2.2, 2.2, 1.45)
    have = [s for s in ('current', 'r0.2') if s + '_pooled' in keep]
    for i, (k, _) in enumerate(POOLED_SETS):
        hs = [s for s in have if k in keep[s + '_pooled']]
        for s in hs:
            x = i + (-.18 if s == 'current' else .18)
            dots(ax, x, keep[s + '_pooled'][k]['v'], COLORS[s], 90 + i * 10 + have.index(s), w=.05, s=3)
        p_stack(ax, i + .03, LIM[1], [tab[(tab.selection == s) & (tab.kind == 'set_hp') & (tab.unit == k)].p.iloc[0]
                                   for s in hs], [COLORS[s] for s in hs])
    ax.axhline(0, color='.6', lw=.6, ls='--', zorder=0)
    ax.set_ylim(*LIM)
    ax.set_yticks([-8, -4, 0, 4, 8])
    ax.set_xticks(range(len(POOLED_SETS)))
    ax.set_xticklabels(['V3AB\n+ hV4', 'LO, TO,\nVO, IPS', 'V3AB\n+ IPS', 'hV4, VO,\nLO, TO'], fontsize=6)
    ax.set_xlim(-.5, len(POOLED_SETS) - .45)
    ax.tick_params(axis='x', length=0)
    ax.set_ylabel('HP − other (%)')
    ax.text(.0, 1.08, 'One decoder per pooled set', transform=ax.transAxes, fontsize=6.5, color='.3', va='bottom')
    offset_spines(ax)
    letter(ax, 'f', x=-.25, y=1.12)

    # g: bar sweep, HP − next-to-HP segment per bar position (outline decoder, V1–V3)
    if all('sweep' in keep[s] for s in sels):
        ax = axes(.55, .45, 3.0, 1.15)
        for s in sels:
            sw = keep[s]['sweep']
            v = sw['hp'] - sw['nx']
            m = np.nanmean(v, 0)
            se = np.nanstd(v, 0, ddof=1) / np.sqrt(np.isfinite(v).sum(0))
            x = sw['centres'] + {'current': -.06, 'r0.2': 0, 'r0.1': .06}[s]
            ax.errorbar(x, m, yerr=se, fmt='-o', ms=2.8, color=COLORS[s], mec='w', mew=.4, lw=.9, capsize=0)
        ax.axhline(0, color='.6', lw=.6, ls='--', zorder=0)
        ax.set_xlabel('Bar centre, toward HP (°)')
        ax.set_ylabel('HP − next\nsegment (%)')
        offset_spines(ax)
        letter(ax, 'g', x=-.13, y=1.1)

    # h: decoding quality of the pooled-set decoders
    if have:
        ax = axes(4.75, .45, 2.2, 1.15)
        for i, (k, _) in enumerate(POOLED_SETS):
            for s in have:
                t = tab[(tab.selection == s) & (tab.kind == 'set_quality_r') & (tab.unit == k)]
                if len(t):
                    ax.errorbar(i + (-.12 if s == 'current' else .12), t['mean'].iloc[0], yerr=t['sem'].iloc[0], fmt='o',
                                ms=3, color=COLORS[s], mec='w', mew=.4, lw=.9, capsize=0)
        ax.set_ylim(0, .35)
        ax.set_yticks([0, .1, .2, .3])
        ax.set_xticks(range(len(POOLED_SETS)))
        ax.set_xticklabels(['V3AB\n+ hV4', 'LO, TO,\nVO, IPS', 'V3AB\n+ IPS', 'hV4, VO,\nLO, TO'], fontsize=6)
        ax.set_xlim(-.5, len(POOLED_SETS) - .5)
        ax.tick_params(axis='x', length=0)
        ax.set_ylabel('Decoded vs\ntrue bar (r)')
        offset_spines(ax)
        letter(ax, 'h', x=-.25, y=1.1)
    save(fig, name, legend(tab, n_note))


def legend(tab, n_note):
    def g(sel, kind, unit, col='mean'):
        return tab[(tab.selection == sel) & (tab.kind == kind) & (tab.unit == unit)][col].iloc[0]

    lab = {s: l for s, _, l in SELECTIONS}
    present = [s for s, _, _ in SELECTIONS if s in set(tab.selection)]
    vox = '; '.join(f'{lab[s]}: V1 {g(s, "voxels", "V1"):.0f}, LO {g(s, "voxels", "LO"):.0f}, VO {g(s, "voxels", "VO"):.0f}'
                    for s in present)
    caps = '; '.join(f'{lab[s]} {int(tab[(tab.selection == s) & (tab.kind == "voxels")].cap_binds_runs.sum())}'
                     for s in present)
    qual = '; '.join(f'{lab[s]} {g(s, "quality_r", "V1-V3"):.3f} ± {g(s, "quality_r", "V1-V3", "sem"):.3f}'
                     for s in present)
    quad = ' '.join(f'{lab[s]}: ' + ', '.join(
        f'{u} {g(s, "quadrant", u):+.2f} ± {g(s, "quadrant", u, "sem"):.2f}% (p = {g(s, "quadrant", u, "p"):.3f})'
        for u in ['HP − next', 'HP − opposite', 'Next − opposite']) + '.' for s in present)
    pooled = ' '.join(f'{lab[s]}: ' + ', '.join(
        f'{u} {g(s, "pooled_hp", u):+.2f} ± {g(s, "pooled_hp", u, "sem"):.2f}% (p = {g(s, "pooled_hp", u, "p"):.3f})'
        for u in ['All areas', 'Early', 'Dorsal', 'Occipitotemporal']) + '.' for s in present)
    sig = ' '.join(f'{lab[s]}: ' + (', '.join(
        f'{r} q = {q:.3f}' for r, q in tab[(tab.selection == s) & (tab.kind == 'area_hp') & (tab.q < .05)][['unit', 'q']]
        .itertuples(index=False)) or 'none') + '.' for s in present)
    sets = ' '.join(
        f'{lab[s]}: ' + ', '.join(f'{u} {g(s, "set_hp", u):+.2f} ± {g(s, "set_hp", u, "sem"):.2f}% (p = {g(s, "set_hp", u, "p"):.3f}, '
                                  f'r = {g(s, "set_quality_r", u):.3f}, n = {int(g(s, "set_hp", u, "n"))})'
                                  for u, _ in POOLED_SETS if len(tab[(tab.selection == s) & (tab.kind == 'set_hp') & (tab.unit == u)]))
        + '.' for s in ('current', 'r0.2') if len(tab[(tab.selection == s) & (tab.kind == 'set_hp')]))
    sweep = ' '.join(f'{lab[s]}: {g(s, "sweep", "HP − next segment"):+.2f} ± {g(s, "sweep", "HP − next segment", "sem"):.2f}% '
                     f'(p = {g(s, "sweep", "HP − next segment", "p"):.3f}).' for s in present
                     if len(tab[(tab.selection == s) & (tab.kind == 'sweep')]))
    return f"""
**Figure S. Decoding under looser voxel selections.** The decoding of Fig. 6 repeated with two other voxel selections;
everything else is unchanged (held-out run, pRFs from the cross-validation fold without it, every pixel of every frame
decoded, at most {CAP} voxels per area, highest reliability first; area skipped for a subject with fewer than 20 voxels).
Black, the selection of Fig. 6: leave-one-run-out reliability R² > 0 on the fold's training runs (about r > 0.3). Blue
and light blue: the voxel's cross-run reliability (mean over training runs of the correlation of the run with the mean of
the other training runs) above 0.2 or 0.1 and above a null of 100 circular shifts (p ≤ .01), the rule of the gain-field
robustness check, computed per fold on the training runs only so the decoded run never influences its voxels; all with
BOLD SD ≤ 20%. {n_note}
a, Voxels per area (mean over runs per subject; points and bars, mean ± SEM over subjects): {vox}. The cap of {CAP} was
reached in {caps} subject × run × area decodes.
b, Decoding quality as in Fig. 6b (spatial correlation of decoded and true frame inside the aperture; mean ± SEM over
subjects). V1–V3 mean: {qual}.
c, The V1–V3 quadrant contrasts of Fig. 6d per selection (decoded intensity on the bar's pixels per quadrant in % of the
mean of the four, V1–V3 averaged per subject; dots are subjects, points and bars mean ± SEM). One-sided Wilcoxon
signed-rank tests for HP below the other quadrant, two-sided for next − opposite. {quad}
d, The pooled HP − other-three contrast of Fig. 6f (all areas, and the areas of each pathway a subject has); one-sided
Wilcoxon, uncorrected. {pooled}
e, Per area, mean ± SEM over subjects; one-sided Wilcoxon, Benjamini-Hochberg across the nine areas; stars, q < .05
(row per selection). {sig}
f, One decoder on the pooled voxels of several areas (each voxel with its own fold pRF, selection per voxel as in the
per-area decodes, at most {CAP} voxels per set): mid-level V3AB + hV4, higher LO + TO + VO + IPS, dorsal V3AB + IPS,
occipitotemporal hV4 + VO + LO + TO; Fig. 6 selection (black) and r > 0.2 (blue); one-sided Wilcoxon, uncorrected;
values beyond the axis are drawn as open triangles at its edge (also in c and d). {sets}
g, The bar sweep of Fig. 6c (decoder with the bar outline given, V1–V3): per bar position on the HP side of the
aperture, the decoded − mean-over-HP-locations difference on the bar's segment in the quadrant facing HP minus that in
the quadrant next to it, in % of the bar's mean (mean ± SEM over subjects). Mean over positions, one-sided Wilcoxon:
{sweep}
h, Decoding quality (as in b) of the pooled-set decoders of f.
"""


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--exclude-subjects', nargs='*', type=int, default=[1, 2])
    ap.add_argument('--name', default='figS_decoding_voxel_selection_n28')
    ap.add_argument('--table', default=str(DATA / 'decoding_voxel_selection_2026-11.tsv'))
    a = ap.parse_args()
    tab, keep = summarise(set(a.exclude_subjects))
    tab.to_csv(a.table, sep='\t', index=False)
    print(tab.drop(columns=[c for c in ['test'] if c in tab]).round(4).to_string(index=False))
    n = tab[(tab.selection == 'current') & (tab.kind == 'quadrant')].n.iloc[0]
    plot(tab, keep, a.name, f'n = {int(n)} subjects (sub-01 and sub-02 left out).')
