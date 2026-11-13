"""DN pRF parameters by where the pRF lies relative to the HP distractor location (exploratory).

Input: ``notes/data/dn_by_hp_2026-11.tsv.gz`` (``modeling.collect_dn_by_hp``; fits of ``modeling.fit_dn_by_hp``,
one DN fit per HP condition, HRF fixed per subject × ROI, no cross-validation).

Voxels: positive rf_amplitude in the all-runs fit and all four condition fits, all-runs pRF centre within the
bar aperture (3.17°) and at least ``MERIDIAN``° from both meridians, so its quadrant is unambiguous. For each
voxel and HP condition the relation of the voxel's quadrant (all-runs fit) to the HP quadrant is ``HP``,
``Next`` (sharing one side, two conditions, averaged) or ``Opposite``. Each parameter is expressed within
voxel: log ratio to the voxel's geometric mean over the four conditions for positive parameters, the
difference from its mean for bold_baseline and R². Per subject × ROI (>= ``MIN_VOXELS`` voxels): median over
voxels per relation. Tests across subjects per ROI × parameter: Friedman over the three relations and
Wilcoxon HP vs Opposite, BH over ROIs within each parameter.

Second figure (``_distance``): the same within-voxel values against the distance from the all-runs pRF centre
to the HP item (all voxels within the aperture), binned; mean ± SEM over subjects.

Usage::

    python -m retsupp.visualize.paper.figS_dn_params_by_hp --table notes/data/dn_by_hp_2026-11.tsv.gz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon
from statsmodels.stats.multitest import multipletests

from retsupp.visualize.paper.paper_style import (DIV_CMAP, DOUBLE, FG, HP, LP_NEXT, LP_OPP, MUTED, OUT, ROI_GROUPS, ROI_ORDER, ROI_X,
                                                 letter)

APERTURE, MERIDIAN, MIN_VOXELS = 3.17, .25, 10
ECC_HP = 4.0
PIXEL_AREA = (10 / 49) ** 2          # resolution 50 on ±5°
QUAD = {'upper_right': (1, 1), 'upper_left': (-1, 1), 'lower_left': (-1, -1), 'lower_right': (1, -1)}
RELATIONS = ['HP', 'Next', 'Opposite']
REL_COLORS = {'HP': HP, 'Next': LP_NEXT, 'Opposite': LP_OPP}
# (column, label, log): log -> within-voxel log ratio, else difference
PARAMS = [('sd', 'σ centre', True), ('a_peak', 'Peak gain a', True), ('srf_amplitude', 'Surround amplitude c', True),
          ('srf_size', 'Size ratio σ2/σ1', True), ('neural_baseline', 'Activation constant b', True),
          ('surround_baseline', 'Normalization constant d', True), ('b_over_d', 'b / d', True),
          ('a_over_d', 'a / d', True), ('bold_baseline', 'BOLD baseline', False), ('r2', 'R² (fitted runs)', False)]
SHORT = {'sd': 'σ centre', 'a_peak': 'Gain a', 'srf_amplitude': 'Surround c', 'srf_size': 'σ2/σ1',
         'neural_baseline': 'b', 'surround_baseline': 'd', 'b_over_d': 'b / d', 'a_over_d': 'a / d',
         'bold_baseline': 'BOLD baseline', 'r2': 'R²'}
DIST_BINS = [0, 1.5, 3, 4.5, 6, 8]
HP_ANGLE = {'upper_right': 45, 'upper_left': 135, 'lower_left': 225, 'lower_right': 315}
MAP_ECC, MAP_SIGMA, MAP_RES, MIN_WEIGHT = 5., .6, 61, 1.
FWHM_K = 2 * np.sqrt(2 * np.log(2))
GAIN_TABLE = Path('notes/data/gain_field_parameters.tsv')
RADIAL_BINS = np.arange(0, 8.5, .75)
N_BOOT = 300
PATHWAY_ROWS = [('V1–V3', ('V1', 'V2', 'V3')), ('Dorsal', ('V3AB', 'IPS')), ('Occipitotemporal', ('hV4', 'VO', 'LO', 'TO'))]


def load(table, max_ecc=APERTURE):
    d = pd.read_csv(table, sep='\t')
    d = d[d.roi.isin(ROI_ORDER)].copy()
    gauss = PIXEL_AREA / (d.sd * np.sqrt(2 * np.pi))
    d['a_peak'] = d.rf_amplitude * gauss
    d['b_over_d'] = d.neural_baseline / d.surround_baseline
    d['a_over_d'] = d.a_peak / d.surround_baseline
    ref = d[d.condition == 'all'].set_index(['subject', 'voxel_index'])[['x', 'y', 'rf_amplitude']]
    c = d[d.condition != 'all'].join(ref, on=['subject', 'voxel_index'], rsuffix='_all')
    ok = c.groupby(['subject', 'voxel_index']).rf_amplitude.transform(lambda a: (a > 0).all() and len(a) == 4)
    c = c[ok.astype(bool) & (c.rf_amplitude_all > 0) & (np.hypot(c.x_all, c.y_all) <= max_ecc)].copy()
    for col, _, log in PARAMS:
        v = np.log(c[col].where(c[col] > 0)) if log else c[col]
        c[col] = v - v.groupby([c.subject, c.voxel_index]).transform('mean')
    hx = c.condition.map(lambda h: QUAD[h][0])
    hy = c.condition.map(lambda h: QUAD[h][1])
    same = (np.sign(c.x_all) == hx).astype(int) + (np.sign(c.y_all) == hy).astype(int)
    c['relation'] = same.map({2: 'HP', 1: 'Next', 0: 'Opposite'})
    c['clear'] = (c.x_all.abs() >= MERIDIAN) & (c.y_all.abs() >= MERIDIAN)
    rot = np.deg2rad(90 - c.condition.map(HP_ANGLE))     # HP location to the top
    c['x_rot'] = c.x_all * np.cos(rot) - c.y_all * np.sin(rot)
    c['y_rot'] = c.x_all * np.sin(rot) + c.y_all * np.cos(rot)
    c['dist_hp'] = np.hypot(c.x_all - hx * ECC_HP / np.sqrt(2), c.y_all - hy * ECC_HP / np.sqrt(2))
    return c


def per_subject(c, by):
    cols = [p for p, _, _ in PARAMS]
    g = c.groupby(['subject', 'roi', 'voxel_index', by])[cols].mean().reset_index()   # Next: mean of two
    n = g.groupby(['subject', 'roi']).voxel_index.nunique().rename('n').reset_index()
    s = g.groupby(['subject', 'roi', by])[cols].median().reset_index().merge(n, on=['subject', 'roi'])
    return s[s.n >= MIN_VOXELS]


def stats(s):
    rows = []
    for col, _, _ in PARAMS:
        for roi in ROI_ORDER:
            w = s[s.roi == roi].pivot(index='subject', columns='relation', values=col)
            if not set(RELATIONS) <= set(w.columns):
                continue
            w = w[RELATIONS].dropna()
            if len(w) < 6:
                continue
            rows.append(dict(param=col, roi=roi, n=len(w), mean_hp_minus_opp=(w.HP - w.Opposite).mean(),
                             p_friedman=friedmanchisquare(*[w[r] for r in RELATIONS]).pvalue,
                             p_hp_vs_opp=wilcoxon(w.HP, w.Opposite).pvalue,
                             p_hp_vs_next=wilcoxon(w.HP, w.Next).pvalue))
    st = pd.DataFrame(rows)
    for p in ('p_friedman', 'p_hp_vs_opp'):
        st[p + '_bh'] = st.groupby('param')[p].transform(lambda x: multipletests(x, method='fdr_bh')[1])
    return st


def roi_axis(ax):
    ax.set_xticks([ROI_X[r] for r in ROI_ORDER])
    ax.set_xticklabels(ROI_ORDER, rotation=90)
    ax.axhline(0, color=FG, lw=.6, zorder=0)
    ax.set_xlim(-.7, max(ROI_X.values()) + .7)


def plot_relations(s, st, out):
    fig, axes = plt.subplots(4, 3, figsize=(DOUBLE, DOUBLE * 1.15))
    off = {'HP': -.22, 'Next': 0, 'Opposite': .22}
    for i, (ax, (col, label, log)) in enumerate(zip(axes.flat, PARAMS)):
        for rel in RELATIONS:
            m = s[s.relation == rel].groupby('roi')[col].agg(['mean', 'sem'])
            m = m.reindex([r for r in ROI_ORDER if r in m.index])
            x = [ROI_X[r] + off[rel] for r in m.index]
            ax.errorbar(x, m['mean'], m['sem'], fmt='o', ms=2.5, lw=.8, color=REL_COLORS[rel], label=rel)
        roi_axis(ax)
        ax.set_title(label, fontsize=7, color=FG)
        ax.set_ylabel('Log ratio to voxel mean' if log else 'Difference from voxel mean', fontsize=6)
        for _, r in st[(st.param == col) & (st.p_friedman_bh < .05)].iterrows():
            ax.text(ROI_X[r.roi], 1, '*', transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=8)
        letter(ax, 'abcdefghijkl'[i])
    for ax in axes.flat[len(PARAMS):]:
        ax.set_axis_off()
    axes.flat[0].legend(frameon=False, fontsize=6, loc='lower left', handletextpad=.2)
    ax = axes.flat[len(PARAMS)]
    ax.text(0, .9, 'Exploratory: one DN fit per HP condition (about 3 runs each), no cross-validation.\n'
            'Pathways: ' + '; '.join(f'{g}: {", ".join(r)}' for g, r in ROI_GROUPS) + '.\n'
            '* Friedman over HP / Next / Opposite, BH over ROIs, p < .05.\n'
            'Mean ± SEM over subjects of per-subject medians over voxels.',
            transform=ax.transAxes, fontsize=6, va='top', color=FG, wrap=True)
    fig.tight_layout(h_pad=1.2)
    fig.savefig(out)
    print(f'wrote {out}')


def plot_distance(c, out):
    c = c.copy()
    c['dist_bin'] = pd.cut(c.dist_hp, DIST_BINS, labels=[(a + b) / 2 for a, b in zip(DIST_BINS, DIST_BINS[1:])])
    cols = [p for p, _, _ in PARAMS]
    s = c.groupby(['subject', 'roi', 'dist_bin'], observed=True)[cols].median().reset_index()
    fig, axes = plt.subplots(len(PARAMS), len(ROI_ORDER), figsize=(DOUBLE, DOUBLE * 1.3), sharex=True, sharey='row')
    for i, (col, label, log) in enumerate(PARAMS):
        for j, roi in enumerate(ROI_ORDER):
            ax = axes[i, j]
            m = s[s.roi == roi].groupby('dist_bin', observed=True)[col].agg(['mean', 'sem'])
            ax.errorbar(m.index.astype(float), m['mean'], m['sem'], fmt='o-', ms=2, lw=.8, color=HP)
            ax.axhline(0, color=FG, lw=.5, zorder=0)
            if i == 0:
                ax.set_title(roi, fontsize=7)
            if j == 0:
                ax.set_ylabel(label, fontsize=5.5)
            ax.tick_params(labelsize=5)
    for ax in axes[-1]:
        ax.set_xlabel('Distance to HP (°)', fontsize=6)
    fig.tight_layout(h_pad=.3, w_pad=.2)
    fig.savefig(out)
    print(f'wrote {out}')


def smooth_map(x, y, v, g):
    """Gaussian-weighted mean of v at the grid points; NaN where the summed weight < MIN_WEIGHT."""
    w = np.exp(-((g[0].ravel()[:, None] - x) ** 2 + (g[1].ravel()[:, None] - y) ** 2) / (2 * MAP_SIGMA ** 2))
    den = w.sum(1)
    with np.errstate(invalid='ignore', divide='ignore'):
        m = np.where(den >= MIN_WEIGHT, (w * v).sum(1) / den, np.nan)
    return m.reshape(g[0].shape)


def gain_field_fwhm(path=GAIN_TABLE, family='dog', model='m7_full'):
    """Median FWHM of the sustained gain field (sigma_AF) over subjects × ROIs."""
    g = pd.read_csv(path, sep='\t')
    g = g[(g.family == family) & (g.model == model)]
    return float(g.sigma_AF.median() * FWHM_K)


def field(d, a, s, c):
    return a * np.exp(-d ** 2 / (2 * s ** 2)) + c


def fit_field(d, y):
    from scipy.optimize import curve_fit
    ok = np.isfinite(y)
    try:
        p, _ = curve_fit(field, d[ok], y[ok], p0=[y[ok][0] - y[ok][-1], 2., y[ok][-1]],
                         bounds=([-np.inf, .25, -np.inf], [np.inf, 10., np.inf]), maxfev=5000)
        return p
    except (RuntimeError, ValueError, IndexError):
        return np.full(3, np.nan)


def radial_profiles(c):
    """Per pathway × parameter × subject: mean Δ per distance-to-HP bin."""
    c = c.copy()
    mids = (RADIAL_BINS[:-1] + RADIAL_BINS[1:]) / 2
    c['rbin'] = pd.cut(c.dist_hp, RADIAL_BINS, labels=mids)
    out = []
    for name, rois in PATHWAY_ROWS:
        cc = c[c.roi.isin(rois)]
        for col, _, _ in PARAMS:
            p = cc.groupby(['subject', 'rbin'], observed=False)[col].mean().unstack('rbin').reindex(columns=mids)
            out.append((name, col, p))
    return out, mids


def plot_radial(c, out, stats_out, fwhm_gain):
    profiles, mids = radial_profiles(c)
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(len(PATHWAY_ROWS), len(PARAMS), figsize=(DOUBLE, DOUBLE * .55), sharex=True)
    rows = []
    for (name, col, p) in profiles:
        i = [r[0] for r in PATHWAY_ROWS].index(name)
        j = [q[0] for q in PARAMS].index(col)
        ax = axes[i, j]
        p = p.loc[:, p.notna().sum() >= 5]
        if p.shape[1] < 4:
            ax.set_axis_off()
            continue
        d = p.columns.astype(float).to_numpy()
        m, se = p.mean().to_numpy(), p.sem().to_numpy()
        fit = fit_field(d, m)
        boot = np.array([fit_field(d, p.iloc[rng.integers(0, len(p), len(p))].mean().to_numpy())
                         for _ in range(N_BOOT)])
        lo, hi = np.nanpercentile(boot[:, 1] * FWHM_K, [2.5, 97.5])
        alo, ahi = np.nanpercentile(boot[:, 0], [2.5, 97.5])
        rows.append(dict(pathway=name, param=col, n_subjects=len(p), amplitude=fit[0], fwhm=fit[1] * FWHM_K,
                         fwhm_lo=lo, fwhm_hi=hi, amplitude_lo=alo, amplitude_hi=ahi, offset=fit[2]))
        ax.errorbar(d, m, se, fmt='o', ms=1.8, lw=.6, color=FG)
        dd = np.linspace(0, d.max(), 100)
        if np.isfinite(fit).all():
            ax.plot(dd, field(dd, *fit), color=HP, lw=.9)
            # Gain-field shape with the fitted amplitude and offset, for comparison of widths.
            ax.plot(dd, field(dd, fit[0], fwhm_gain / FWHM_K, fit[2]), color=HP, lw=.7, ls='--')
            ax.text(.5, 1.02, f'FWHM {fit[1] * FWHM_K:.1f}° [{lo:.1f}, {hi:.1f}]', transform=ax.transAxes,
                    ha='center', va='bottom', fontsize=4.3, color=HP)
        ax.axhline(0, color=MUTED, lw=.4, zorder=0)
        ax.tick_params(labelsize=4.5, length=1.5)
        if i == 0:
            ax.set_title(SHORT[col], fontsize=6, pad=9)
        if j == 0:
            ax.set_ylabel(name, fontsize=6)
        if i == len(PATHWAY_ROWS) - 1:
            ax.set_xlabel('Distance to HP (°)', fontsize=5)
    fig.suptitle(f'Δ against distance from the pRF centre to the HP item, mean ± SEM over subjects. Solid: Gaussian fit '
                 f'(free amplitude, width, offset; FWHM with 95% bootstrap CI over subjects). Dashed: sustained gain-field '
                 f'width (FWHM {fwhm_gain:.1f}°)', fontsize=5, color=FG, wrap=True)
    fig.tight_layout(rect=(0, 0, 1, .93), h_pad=1.2, w_pad=.3)
    fig.savefig(out)
    pd.DataFrame(rows).to_csv(stats_out, sep='\t', index=False, float_format='%.4g')
    print(f'wrote {out}, {stats_out}')


def plot_maps(c, out, fwhm_gain):
    """Δ per voxel × condition at the voxel's pRF centre, rotated so HP is at the top; smoothed per subject, then averaged."""
    ax1 = np.linspace(-MAP_ECC, MAP_ECC, MAP_RES)
    g = np.meshgrid(ax1, ax1)
    outside = np.hypot(*g) > MAP_ECC
    c = c[np.isfinite(c.x_rot)]
    maps = {}
    for row, (name, rois) in enumerate(PATHWAY_ROWS):
        cc = c[c.roi.isin(rois)]
        for col, _, _ in PARAMS:
            per_sub = []
            for _, s in cc[np.isfinite(cc[col])].groupby('subject'):
                m = smooth_map(s.x_rot.to_numpy(), s.y_rot.to_numpy(), s[col].to_numpy(), g)
                per_sub.append(m)
            if per_sub:
                st = np.stack(per_sub)
                n = np.isfinite(st).sum(0)
                with np.errstate(invalid='ignore'):
                    m = np.where(n >= 5, np.nanmean(st, 0), np.nan)
                m[outside] = np.nan
                maps[name, col] = (m, len(per_sub))
    fig, axes = plt.subplots(len(PATHWAY_ROWS), len(PARAMS), figsize=(DOUBLE, DOUBLE * .42))
    ring = ECC_HP * np.array([[np.cos(np.deg2rad(a)), np.sin(np.deg2rad(a))] for a in (90, 180, 270, 0)])
    th = np.linspace(0, 2 * np.pi, 200)
    for j, (col, label, log) in enumerate(PARAMS):
        vals = np.concatenate([maps[k][0].ravel() for k in maps if k[1] == col])
        lim = np.nanpercentile(np.abs(vals), 98) if np.isfinite(vals).any() else 1
        for i, (name, _) in enumerate(PATHWAY_ROWS):
            ax = axes[i, j]
            ax.set_aspect('equal')
            ax.set_xticks([]), ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if (name, col) in maps:
                im = ax.imshow(maps[name, col][0], origin='lower', extent=[-MAP_ECC, MAP_ECC] * 2, cmap=DIV_CMAP,
                               vmin=-lim, vmax=lim, interpolation='nearest')
            ax.plot(3.17 * np.cos(th), 3.17 * np.sin(th), color=FG, lw=.4, ls=':')
            ax.plot(ECC_HP * np.cos(th), ECC_HP * np.sin(th), color=MUTED, lw=.3)
            ax.scatter(ring[1:, 0], ring[1:, 1], s=6, facecolor='none', edgecolor=FG, lw=.5)
            ax.scatter(ring[:1, 0], ring[:1, 1], s=10, color=HP, lw=0)
            ax.plot(fwhm_gain / 2 * np.cos(th), ECC_HP + fwhm_gain / 2 * np.sin(th), color=HP, lw=.5, ls='--')
            ax.set_xlim(-MAP_ECC - .3, MAP_ECC + .3), ax.set_ylim(-MAP_ECC - .3, MAP_ECC + .3)
            if i == 0:
                ax.set_title(SHORT[col], fontsize=6)
            if j == 0:
                ax.text(-.12, .5, name, transform=ax.transAxes, rotation=90, ha='right', va='center', fontsize=6)
        cb = fig.colorbar(im, ax=axes[:, j], orientation='horizontal', fraction=.05, pad=.03, aspect=12)
        cb.ax.tick_params(labelsize=4.5, length=1.5)
        cb.set_ticks([-lim, lim])
        cb.set_ticklabels([f'{-lim:.2g}', f'{lim:.2g}'])
        cb.set_label('Δ log' if log else 'Δ', fontsize=5)
    fig.suptitle('Δ = condition minus voxel mean over the four HP conditions, at the pRF centre, rotated so HP '
                 '(red) is at the top; Gaussian σ = %.1f°, per subject then averaged. Dashed: sustained gain-field FWHM (%.1f°)'
                 % (MAP_SIGMA, fwhm_gain), fontsize=5.5, color=FG)
    fig.savefig(out)
    print(f'wrote {out}')


def main(table, out, stats_out):
    fwhm_gain = gain_field_fwhm()
    print(f'sustained gain field: median FWHM {fwhm_gain:.2f}°')
    wide = load(table, max_ecc=MAP_ECC)
    plot_maps(wide, str(out).replace('.pdf', '_maps.pdf'), fwhm_gain)
    plot_radial(wide, str(out).replace('.pdf', '_radial.pdf'), str(stats_out).replace('.tsv', '_radial.tsv'), fwhm_gain)
    c = load(table)
    s = per_subject(c[c.clear], 'relation')
    st = stats(s)
    Path(stats_out).parent.mkdir(parents=True, exist_ok=True)
    st.to_csv(stats_out, sep='\t', index=False, float_format='%.4g')
    print(st[st.p_friedman_bh < .05].to_string(index=False) if (st.p_friedman_bh < .05).any()
          else 'No ROI × parameter with Friedman p_BH < .05')
    plot_relations(s, st, out)
    plot_distance(c, str(out).replace('.pdf', '_distance.pdf'))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--table', default='notes/data/dn_by_hp_2026-11.tsv.gz')
    ap.add_argument('--out', default=str(OUT / 'figS_dn_params_by_hp.pdf'))
    ap.add_argument('--stats', default='notes/data/dn_by_hp_stats_2026-11.tsv')
    a = ap.parse_args()
    main(a.table, a.out, a.stats)
