"""Do the gain terms grow along the visual hierarchy? (2026-11-12, n = 28, participants 3–30).

1. Hierarchy regression for every gain term: per participant, the term's per-area estimate (in % gain) regressed on
   log2 of the area's pRF size (the axis of ``hierarchy_regression``: median over participants of the voxel-weighted
   mean pRF σ at 1–2.5° eccentricity, ``fig1_prf_sigma.tsv``), slope per doubling; median, bootstrapped 95% CI, k of
   n, Wilcoxon, BH across terms. Also with the area's hierarchical rank as predictor, for all nine areas and for the
   dorsal and occipitotemporal branches (each with V1–V3), and for Gaussian and DN pRFs.
2. Target gain vs dS: paired slopes, raw and normalised (each term divided by its pooled median).
3. dS relative to the shared sustained gain, per area and within participants.
4. Within-area dependence on pRF size (``fit_gain_bins`` outputs): five gains re-solved per tercile of pRF size (and of
   eccentricity), all else fixed at the fit.
5. Sustained field width (restricted to fits with a reliable sustained gain, update 7) against pRF size.

    python -m retsupp.visualize.paper.hierarchy_gain_terms --bins-dir <gain_bins folder>
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROIS = ['V1', 'V2', 'V3', 'V3AB', 'IPS', 'hV4', 'VO', 'LO', 'TO']
EARLY, DORSAL, OT = ['V1', 'V2', 'V3'], ['V3AB', 'IPS'], ['hV4', 'VO', 'LO', 'TO']
SETS = {'all nine': ROIS, 'early + dorsal': EARLY + DORSAL, 'early + occipitotemporal': EARLY + OT}
# Hierarchical rank: V1 < V2 < V3 < (V3AB, hV4, LO) < (IPS, VO, TO): the second tier receives most of its input from
# V3/V3AB-level areas (Wandell et al. 2007; Felleman & Van Essen 1991; IPS above V3AB dorsally, VO above hV4
# ventrally, TO above LO laterally).
RANK = {'V1': 1, 'V2': 2, 'V3': 3, 'V3AB': 4, 'hV4': 4, 'LO': 4, 'IPS': 5, 'VO': 5, 'TO': 5}
FWHM = 2 * np.sqrt(2 * np.log(2))
EXCL = (1, 2)


def median_ci(x, n_boot=10000, seed=0):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    rng = np.random.default_rng(seed)
    b = np.median(rng.choice(x, (n_boot, len(x))), axis=1)
    return np.median(x), np.percentile(b, 2.5), np.percentile(b, 97.5), b.std()


def roi_size(table, ecc=(1., 2.5)):
    d = pd.read_csv(table, sep='\t')
    d = d[d.ecc.between(*ecc) & d.roi.isin(ROIS) & ~d.subject.isin(EXCL)]
    per = d.groupby(['roi', 'subject']).apply(lambda g: np.average(g.sd, weights=g.n), include_groups=False)
    return per.groupby('roi').median()


def terms(gain_table, cond_lin, cond_exact, family):
    """Per participant x area, every gain term in % gain."""
    g = pd.read_csv(gain_table, sep='\t')
    g = g[(g.family == family) & (g.model == 'm7_full') & g.roi.isin(ROIS) & ~g.subject.isin(EXCL)].copy()
    j = pd.DataFrame(dict(subject=g.subject, roi=g.roi,
                          T=g.g_T_dyn, D=(g.g_HP_dyn + 3 * g.g_LP_dyn) / 4, g_LP=g.g_LP, g_HP=g.g_HP,
                          dS_joint=g.g_HP - g.g_LP, dD_joint=g.g_HP_dyn - g.g_LP_dyn))
    if family == 'dog':      # the estimated-alone terms of key_numbers (linearised, all valid fits)
        c = pd.read_csv(cond_lin, sep='\t')
        c = c[c.valid & c.roi.isin(ROIS) & ~c.subject.isin(EXCL)]
        c = c.pivot_table(index=['subject', 'roi'], columns='mech', values='estimate').reset_index()
    else:                    # Gaussian and DN: the exact ΔSSE minimum (DN responses curve in the gains)
        c = pd.read_csv(cond_exact, sep='\t')
        c = c[(c.family == family) & c.roi.isin(ROIS) & ~c.subject.isin(EXCL)]
        c = c.pivot_table(index=['subject', 'roi'], columns='mech', values='own').reset_index()
    c = c.rename(columns={'S': 'S_alone', 'dS': 'dS_alone', 'dD': 'dD_alone', 'dT': 'dT_alone'})
    keep = ['subject', 'roi'] + [k for k in ('S_alone', 'dS_alone', 'dD_alone', 'dT_alone') if k in c]
    d = j.merge(c[keep], on=['subject', 'roi'], how='outer')
    cols = [k for k in d.columns if k not in ('subject', 'roi')]
    d[cols] = d[cols] * 100
    return d


TERMS = [('T', 'Target gain'), ('D', 'Distractor gain'), ('S_alone', 'Shared sustained gain (alone)'),
         ('g_LP', 'Sustained gain at LP'), ('g_HP', 'Sustained gain at HP'), ('dS_alone', 'dS, alone'),
         ('dS_joint', 'dS, joint'), ('dD_alone', 'dD, alone'), ('dT_alone', 'dT, alone')]


def subject_slopes(d, col, xmap, rois):
    out = {}
    for s, g in d[d.roi.isin(rois)].dropna(subset=[col]).groupby('subject'):
        if len(g) < 3:
            continue
        x = g.roi.map(xmap).to_numpy(float)
        if np.ptp(x) == 0:
            continue
        out[s] = np.polyfit(x, g[col].to_numpy(float), 1)[0]
    return pd.Series(out, dtype=float)


def regression_table(d, xmap, family, predictor):
    rows, per = [], []
    for name, rois in SETS.items():
        for col, lab in TERMS:
            if col not in d:
                continue
            b = subject_slopes(d, col, xmap, rois)
            if len(b) < 6:
                continue
            m, lo, hi, _ = median_ci(b)
            rows.append(dict(family=family, predictor=predictor, rois=name, term=lab, n=len(b), slope=m, ci_lo=lo,
                             ci_hi=hi, k_pos=int((b > 0).sum()), p=stats.wilcoxon(b).pvalue))
            per.append(pd.DataFrame(dict(subject=b.index, slope=b.values, term=lab, rois=name, family=family,
                                         predictor=predictor)))
    t = pd.DataFrame(rows)
    t['q'] = np.nan
    for name in SETS:
        k = t.rois == name
        if k.any():
            t.loc[k, 'q'] = stats.false_discovery_control(t.loc[k, 'p'])
    return t, pd.concat(per)


def pooled_median(d, col):
    return d.groupby('subject')[col].median().median()


def dissociation(d, logsize):
    """Target slope vs dS slope per participant: raw (% per doubling) and normalised (proportion of the term's pooled
    median per doubling). Minimum detectable slope at 80% power from the bootstrap SE of the median."""
    out = []
    for ds in ('dS_alone', 'dS_joint'):
        for norm in (False, True):
            dd = d.copy()
            if norm:
                for col in ('T', ds):
                    dd[col] = dd[col] / pooled_median(d, col)
            st = subject_slopes(dd, 'T', logsize, ROIS)
            sd = subject_slopes(dd, ds, logsize, ROIS)
            both = pd.concat([st.rename('T'), sd.rename('dS')], axis=1).dropna()
            mt, lot, hit, _ = median_ci(both['T'])
            md, lod, hid, se = median_ci(both['dS'])
            out.append(dict(dS_version=ds, normalised=norm, n=len(both), target_slope=mt, target_lo=lot, target_hi=hit,
                            dS_slope=md, dS_lo=lod, dS_hi=hid, dS_p=stats.wilcoxon(both['dS']).pvalue,
                            mde80=2.8 * se, paired_diff=np.median(both['T'] - both['dS']),
                            paired_p=stats.wilcoxon(both['T'] - both['dS']).pvalue,
                            k_target_gt_dS=int((both['T'] > both['dS']).sum())))
    return pd.DataFrame(out)


def proportion(d):
    """dS relative to the sustained gain: per area the ratio of medians; within participants dS regressed on the
    shared sustained gain across areas (slope = proportion, intercept = constant amount)."""
    rows = []
    for r in ROIS:
        x = d[d.roi == r]
        rows.append(dict(roi=r, n=x.subject.nunique(), dS_alone=x.dS_alone.median(), dS_joint=x.dS_joint.median(),
                         S_alone=x.S_alone.median(), g_LP=x.g_LP.median(), g_HP=x.g_HP.median(),
                         dS_over_gLP=x.dS_joint.median() / x.g_LP.median(),
                         dS_over_gHP=x.dS_joint.median() / x.g_HP.median(),
                         dSalone_over_Salone=x.dS_alone.median() / x.S_alone.median()))
    per_area = pd.DataFrame(rows)
    reg = []
    for ycol, xcol, lab in (('dS_alone', 'S_alone', 'dS alone on shared sustained gain alone'),
                            ('dS_joint', 'g_LP', 'dS joint on sustained gain at LP (joint)')):
        sl, ic = [], []
        for s, g in d.dropna(subset=[ycol, xcol]).groupby('subject'):
            if len(g) < 4:
                continue
            b = np.polyfit(g[xcol], g[ycol], 1)
            sl.append(b[0])
            ic.append(b[1])
        m1, l1, h1, _ = median_ci(sl)
        m0, l0, h0, _ = median_ci(ic)
        reg.append(dict(regression=lab, n=len(sl), slope=m1, slope_lo=l1, slope_hi=h1, slope_p=stats.wilcoxon(sl).pvalue,
                        intercept=m0, intercept_lo=l0, intercept_hi=h0, intercept_p=stats.wilcoxon(ic).pvalue))
    return per_area, pd.DataFrame(reg)


def widths(fits_table, logsize, thr=2.):
    f = pd.read_csv(fits_table, sep='\t')
    f = f[~f.subject.isin(EXCL)]
    rel = (np.abs(f.g_HP / f.g_HP_se) > thr) | (np.abs(f.g_LP / f.g_LP_se) > thr)
    f = f[rel].assign(fwhm=FWHM * f.sigma_AF)
    sl = subject_slopes(f, 'fwhm', logsize, ROIS)
    m, lo, hi, _ = median_ci(sl)
    med = f.groupby('subject').fwhm.median().median()
    per_area = f.groupby('roi').fwhm.agg(['median', 'count']).reindex(ROIS)
    return dict(threshold=thr, n=len(sl), slope=m, ci_lo=lo, ci_hi=hi, p=stats.wilcoxon(sl).pvalue, k_pos=int((sl > 0).sum()),
                equivalence_deg=max(abs(lo), abs(hi)), median_fwhm=med,
                equivalence_rel=max(abs(lo), abs(hi)) / med, n_fits=len(f)), per_area, f


def bins(bins_dir, nbin=3, min_vox=20, by='sd'):
    """Five gains re-solved per tercile of pRF size (``by='sd'``) or eccentricity (``by='ecc'``) per participant x
    area; slope of each gain (and S, dS) on log2 of the bin's median pRF size within area."""
    rows = []
    for fn in sorted(glob.glob(f'{bins_dir}/sub-*/*_gainbins.npz')):
        z = np.load(fn, allow_pickle=True)
        s, r = int(z['subject']), str(z['roi'])
        if s in EXCL:
            continue
        names = list(z['gain_names'])
        key = z['sd'] if by == 'sd' else np.hypot(z['x'], z['y'])
        edges = np.quantile(key, np.linspace(0, 1, nbin + 1))
        idx = np.clip(np.searchsorted(edges, key, side='right') - 1, 0, nbin - 1)
        for b in range(nbin):
            m = idx == b
            if m.sum() < min_vox:
                continue
            A, bb = z['A_v'][m].sum(0), z['b_v'][m].sum(0)
            try:
                g = np.linalg.solve(A, bb)
            except np.linalg.LinAlgError:
                continue
            gd = dict(zip(names, g))
            # tied sustained gain (S) and HP − LP in the same voxels: solve with g_HP = g_LP (one regressor) / as fitted
            i_hp, i_lp = names.index('g_HP'), names.index('g_LP')
            T = np.eye(len(names))[:, [k for k in range(len(names)) if k != i_hp]]
            T[i_hp, [k for k in range(len(names)) if k != i_hp].index(i_lp)] = 1.
            gt = np.linalg.solve(T.T @ A @ T, T.T @ bb)
            rows.append(dict(subject=s, roi=r, by=by, bin=b, n_vox=int(m.sum()), size=float(np.median(z['sd'][m])),
                             ecc=float(np.median(np.hypot(z['x'], z['y'])[m])),
                             T=100 * gd['g_T'], D=100 * (gd['g_D_HP'] + 3 * gd['g_D_LP']) / 4,
                             S=100 * gt[[k for k in range(len(names)) if k != i_hp].index(i_lp)],
                             dS=100 * (gd['g_HP'] - gd['g_LP']), g_LP=100 * gd['g_LP']))
    return pd.DataFrame(rows)


def bin_slopes(b):
    """Per participant x area: slope of each gain on log2 bin pRF size; per participant the median over areas;
    Wilcoxon across participants."""
    out = []
    for col, lab in (('T', 'Target gain'), ('D', 'Distractor gain'), ('S', 'Shared sustained gain'),
                     ('g_LP', 'Sustained gain at LP'), ('dS', 'dS')):
        per = []
        for (s, r), g in b.groupby(['subject', 'roi']):
            if len(g) < 2:
                continue
            per.append((s, r, np.polyfit(np.log2(g['size']), g[col], 1)[0]))
        p = pd.DataFrame(per, columns=['subject', 'roi', 'slope'])
        ps = p.groupby('subject').slope.median()
        m, lo, hi, _ = median_ci(ps)
        out.append(dict(term=lab, n=len(ps), n_area_fits=len(p), slope=m, ci_lo=lo, ci_hi=hi, k_pos=int((ps > 0).sum()),
                        p=stats.wilcoxon(ps).pvalue))
    t = pd.DataFrame(out)
    t['q'] = stats.false_discovery_control(t.p)
    return t


TABLE_TERMS = [('Target gain', 'Target gain'), ('Distractor gain', 'Distractor gain'),
               ('Shared sustained gain (alone)', 'Sustained gain shared by the four locations, estimated alone'),
               ('Sustained gain at LP', 'Sustained gain at the LP locations, joint fit, sign only'),
               ('dS, alone', 'Sustained HP − LP (dS), estimated alone'), ('dS, joint', 'Sustained HP − LP (dS), joint fit'),
               ('dD, alone', 'Distractor HP − LP (dD), estimated alone'), ('dT, alone', 'Target HP − LP (dT), estimated alone')]


def family_table(r, out, regression_path):
    """Markdown table: slopes per term for DoG, Gaussian and DN pRFs (bold: q < .05)."""
    fam = {'dog': 'DoG (refitted)', 'gauss': 'Gaussian', 'dn': 'DN'}

    def cell(x, sign_only):
        if sign_only:
            return f"{'−' if x.slope < 0 else '+'} ({int(x.k_pos)}/{int(x.n)} positive), p = {x.p:.2g}"
        s = f'{x.slope:+.1f} [{x.ci_lo:+.1f}, {x.ci_hi:+.1f}], p = {x.p:.2g}'
        return f'**{s}**' if x.q < .05 else s

    lines = ['# Table S. Slopes of the gain terms along the hierarchy, by pRF model', '',
             "Median per-participant slope on log2 of the area's pRF size (percentage points of gain per doubling of pRF "
             'size; for the rank predictor, per rank step), bootstrapped 95% CI, Wilcoxon signed-rank p; bold, q < .05 '
             '(Benjamini-Hochberg across the nine terms per pRF model and area set). n = 28. DoG: pRFs refitted together '
             'with the gain fields (dS, dD and dT estimated alone on the refitted model; the shared sustained gain estimated '
             'alone comes from a model that was not refitted). Gaussian and DN: pRFs held at their fit without gain fields, '
             'estimated-alone terms from the exact error minimum. The sustained gain at the LP locations is given as a sign '
             'only: its size trades off against the item weight and the pRF amplitude.', '']
    for rois, pred in [('all nine', 'log2 pRF size'), ('all nine', 'hierarchical rank'),
                       ('early + dorsal', 'log2 pRF size'), ('early + occipitotemporal', 'log2 pRF size')]:
        name = rois.replace('early', 'V1–V3')
        lines += [f'## {name[0].upper() + name[1:]}; predictor: {pred}', '',
                  '| Term | ' + ' | '.join(fam.values()) + ' |', '|---|---|---|---|']
        for t, lab in TABLE_TERMS:
            row = []
            for f in fam:
                x = r[(r.family == f) & (r.rois == rois) & (r.predictor == pred) & (r.term == t)]
                row.append(cell(x.iloc[0], t == 'Sustained gain at LP') if len(x) else '—')
            lines.append(f'| {lab} | ' + ' | '.join(row) + ' |')
        lines.append('')
    lines.append('Area sets: V1, V2 and V3 with V3AB and IPS (dorsal) or with hV4, VO, LO and TO (occipitotemporal). '
                 'Rank: V1 1, V2 2, V3 3, V3AB/hV4/LO 4, IPS/VO/TO 5. Every combination (also rank within each branch) is '
                 f'in `{regression_path}`.')
    Path(out).write_text('\n'.join(lines) + '\n')


def main(a):
    size = roi_size(a.sigma_table)
    logsize = np.log2(size)
    data = {f: terms(a.gain_table, a.cond_lin, a.cond_exact, f) for f in ('dog', 'gauss', 'dn')}
    tabs, pers = [], []
    for fam, d in data.items():
        for pred, xmap in (('log2 pRF size', logsize), ('hierarchical rank', pd.Series(RANK))):
            t, p = regression_table(d, xmap, fam, pred)
            tabs.append(t)
            pers.append(p)
    reg = pd.concat(tabs)
    reg.to_csv(f'{a.out}_regression.tsv', sep='\t', index=False)
    if a.table:
        family_table(reg, a.table, f'{a.out}_regression.tsv')
    pd.concat(pers).to_csv(f'{a.out}_slopes.tsv', sep='\t', index=False)
    dog = data['dog']
    dog.to_csv(f'{a.out}_terms_dog.tsv', sep='\t', index=False)
    dis = dissociation(dog, logsize)
    dis.to_csv(f'{a.out}_dissociation.tsv', sep='\t', index=False)
    pa, pr = proportion(dog)
    pa.to_csv(f'{a.out}_proportion_per_area.tsv', sep='\t', index=False)
    pr.to_csv(f'{a.out}_proportion_regression.tsv', sep='\t', index=False)
    w, wa, wf = widths(a.widths_fits, logsize)
    w3 = widths(a.widths_fits, logsize, thr=3.)[0]
    pd.DataFrame([w, w3]).to_csv(f'{a.out}_width_slope.tsv', sep='\t', index=False)
    wf.to_csv(f'{a.out}_width_fits.tsv', sep='\t', index=False)
    pd.Series(size).rename('prf_size_deg').to_csv(f'{a.out}_prf_size_axis.tsv', sep='\t')
    with pd.option_context('display.width', 250, 'display.max_rows', 200):
        print('pRF size axis:', size.round(3).to_dict())
        print(reg[(reg.family == 'dog') & (reg.rois == 'all nine')].round(4).to_string(index=False))
        print(dis.round(4).to_string(index=False))
        print(pa.round(3).to_string(index=False))
        print(pr.round(4).to_string(index=False))
        print(w, wa.round(2).to_string())
    if a.bins_dir and glob.glob(f'{a.bins_dir}/sub-*/*_gainbins.npz'):
        bs = pd.concat([bins(a.bins_dir, by='sd'), bins(a.bins_dir, by='ecc')])
        bs.to_csv(f'{a.out}_bins.tsv', sep='\t', index=False)
        bt = pd.concat([bin_slopes(bs[bs.by == k]).assign(by=k) for k in ('sd', 'ecc')])
        bt.to_csv(f'{a.out}_bin_slopes.tsv', sep='\t', index=False)
        nv = bs.groupby(['by', 'roi', 'bin']).n_vox.median().unstack()
        nv.to_csv(f'{a.out}_bin_nvox.tsv', sep='\t')
        print(bt.round(4).to_string(index=False))
        print(nv)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--gain-table', default='notes/data/gain_field_parameters_final2_2026-11.tsv')
    ap.add_argument('--cond-lin', default='notes/data/conditional_estimates_dog_final2_2026-11.tsv')
    ap.add_argument('--cond-exact', default='notes/data/conditional_exact_final2_allfam_2026-11.tsv')
    ap.add_argument('--sigma-table', default='notes/data/fig1_prf_sigma.tsv')
    ap.add_argument('--widths-fits', default='notes/data/field_widths_reliable_n28_2026-11-12_fits.tsv')
    ap.add_argument('--bins-dir', default=None)
    ap.add_argument('--table', default=None, help='write the DoG/Gaussian/DN markdown table here')
    ap.add_argument('--out', default='notes/data/hierarchy_gain_terms_n28_2026-11-12')
    main(ap.parse_args())
