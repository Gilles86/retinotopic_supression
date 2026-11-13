"""Fig. 3 item responses, model vs measured (update 11, items 2 and 5b), n = 28, local, small tables only.

Inputs (``predict_location_timeseries_polished.py`` + its slurm job, deconvolved per subject; copied to notes/data):
noise-free predictions of the refitted joint gain-field fit through the Fig. 3 refit selection, variants nohp (no HP
terms), S (sustained HP − LP only), SD (joint fit as fitted: sustained and distractor-transient HP − LP), SDT / SDT0
(m8 with the target-transient HP − LP, pRFs held fixed; with dT / dT equalised); the measured refit series
(``event_deconv_relative_polished_2026-11-12``, ``event_deconv_hplp_polished_2026-11-13``).

Window: mean over 3.2–6.4 s after search onset, pre-onset mean subtracted (as Fig. 3). Groups: Early (V1–V3), Dorsal
(V3AB, IPS), Occipitotemporal (hV4, VO, LO, TO), ROIs pooled weighted by voxel count; and V1, V2, V3 alone.

Item 2 (``--part hp``): per item type, HP − LP (LP = mean of next left, next right, opposite) from the relative
deconvolution, in % of the subject's mean over the twelve cells (Fig. 3e units; pairs with a mean ≤ 0 left out) and
in % signal; the interaction contrasts (HP − LP of one item type minus that of another) and repeated-measures ANOVAs
location (HP, LP) × item type on the measured cells, each model's cells and measured − model.

Item 5b (``--part items``): target − neutral (HP and LP averaged), distractor − neutral (LP), target − distractor (LP),
% signal, from the HP/LP deconvolution (Fig. 3b, c); measured, joint fit (SD), measured − model. Noise band of the model:
5000 simulated experiments, each subject's predicted contrast plus a residual (measured − predicted, centred on 0)
drawn with replacement across subjects; the 95% interval of the simulated median and the share of experiments with
Wilcoxon p < .05 in the model's direction (power). Also the minimum detectable effect, 2.8 × the measured SE.
Voxel counts per area from ``*_overlap.tsv``.

    python -m retsupp.visualize.paper.items_model_vs_measured
"""
from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from retsupp.visualize.paper import richter_followup as rf
from retsupp.visualize.paper.fig2_prf_fits import PATHWAY
from retsupp.visualize.paper.fig3_item_locations import PEAK
from retsupp.visualize.paper.fig45_common import item_timecourses

D = 'notes/data'
TAG = '2026-11-13'
REL_MEAS = f'{D}/event_deconv_relative_polished_2026-11-12'
HPLP_MEAS = f'{D}/event_deconv_hplp_polished_2026-11-13'
PRED = f'{D}/event_deconv_polished_pred_2026-11-13_{{mode}}_{{v}}'
VARIANTS = {'nohp': 'no HP terms', 'S': 'sustained HP − LP only', 'SD': 'joint fit (sustained + distractor transient)',
            'SDT': 'joint fit + target transient (pRFs fixed)', 'SDT0': 'same, target transient equalised'}
ITEMS = ('target', 'distractor', 'neutral')
AREAS = ('V1', 'V2', 'V3')
GROUPINGS = {'pathway': {r: g for g, rs in PATHWAY.items() for r in rs}, 'area': {r: r for r in AREAS}}
N_SIM = 5000


def window(folder, groups, keep=None):
    """Window means per subject × group × class; ``keep``: only these (subject, roi) pairs."""
    if keep is None:
        p = item_timecourses(folder, groups)
    else:
        import retsupp.visualize.paper.fig45_common as fc
        d = pd.concat([pd.read_csv(f, sep='\t') for f in sorted(Path(folder).glob('sub-*_2026-11.tsv'))])
        d = d[pd.MultiIndex.from_frame(d[['subject', 'roi']]).isin(keep)]
        d = d[d.roi.isin(fc.ROI_ORDER)].copy()
        d['grp'] = d.roi.map(groups)
        d = d.dropna(subset=['grp'])
        d['t'] = d.t.round(2)
        d['w'] = d.response * d.n_voxels
        g = d.groupby(['subject', 'grp', 'cls', 't'])
        p = (g.w.sum() / g.n_voxels.sum()).rename('r').reset_index()
        base = p[p.t < 0].groupby(['subject', 'grp', 'cls']).r.mean()
        p['r'] -= p.set_index(['subject', 'grp', 'cls']).index.map(base)
    pk = p[p.t.between(*PEAK)].groupby(['subject', 'grp', 'cls']).r.mean().unstack()
    return pk[pk.index.get_level_values('subject') >= rf.MIN_SUBJECT]


def pairs(folder):
    d = pd.concat([pd.read_csv(f, sep='\t', usecols=['subject', 'roi']) for f in sorted(Path(folder).glob('sub-*_2026-11.tsv'))])
    return pd.MultiIndex.from_frame(d.drop_duplicates())


def all_windows(mode, meas):
    """Measured and predicted windows; the m8 variants (SDT, SDT0) exist only for some subject × ROI pairs, so they
    are compared with the measured data and the joint fit on those pairs only (keys 'measured@SDT', 'SD@SDT')."""
    out = {}
    keep = pairs(PRED.format(mode=mode, v='SDT'))
    for name, groups in GROUPINGS.items():
        out[('measured', name)] = window(meas, groups)
        out[('measured@SDT', name)] = window(meas, groups, keep)
        for v in VARIANTS:
            out[(v, name)] = window(PRED.format(mode=mode, v=v), groups)
        out[('SD@SDT', name)] = window(PRED.format(mode=mode, v='SD'), groups, keep)
    return out


def bh(df, col='p', by=('grouping', 'contrast', 'source', 'unit')):
    df = df.copy()
    df['q'] = np.nan
    for _, i in df.groupby(list(by)).groups.items():
        ok = df.loc[i, col].notna()
        if ok.sum() > 1:
            df.loc[i[ok], 'q'] = stats.false_discovery_control(df.loc[i[ok], col])
    return df


# ---------------------------------------------------------------- item 2
def hp_cells(pk, unit):
    pk = pk.dropna()
    if unit == 'relative':
        pk = pk[pk.mean(1) > 0]
        pk = pk.div(pk.mean(1), axis=0) * 100
    c = {}
    for it in ITEMS:
        lp = pk[[f'{it}_nextleft', f'{it}_nextright', f'{it}_opposite']].mean(1)
        c[(it, 'hp')], c[(it, 'lp')] = pk[f'{it}_hp'], lp
    return pd.DataFrame(c)


def hp_contrasts(c):
    d = {it: c[(it, 'hp')] - c[(it, 'lp')] for it in ITEMS}
    return {'HP − LP, target': d['target'], 'HP − LP, distractor': d['distractor'], 'HP − LP, neutral': d['neutral'],
            'HP − LP, items pooled': (d['target'] + d['distractor'] + d['neutral']) / 3,
            'interaction: target − neutral': d['target'] - d['neutral'],
            'interaction: distractor − neutral': d['distractor'] - d['neutral'],
            'interaction: target − distractor': d['target'] - d['distractor']}


def rm_anova(c):
    import pingouin as pg
    long = c.stack([0, 1]).rename('r').reset_index()
    long.columns = ['subject', 'grp', 'item', 'loc', 'r']
    a = pg.rm_anova(data=long, dv='r', within=['loc', 'item'], subject='subject')
    row = a[a.Source == 'loc * item'].iloc[0]
    rowl = a[a.Source == 'loc'].iloc[0]
    return dict(F_loc=rowl.F, p_loc=rowl['p-unc'], F_inter=row.F, p_inter=row['p-unc'],
                df1=int(row.ddof1), df2=int(row.ddof2), n=long.subject.nunique())


def part_hp():
    w = all_windows('relative', REL_MEAS)
    rows, anov = [], []
    sets = (('measured', ('nohp', 'S', 'SD')), ('measured@SDT', ('SD@SDT', 'SDT', 'SDT0')))
    for unit, grouping, (mkey, vlist) in product(('relative', 'absolute'), GROUPINGS, sets):
        meas = hp_cells(w[(mkey, grouping)], unit)
        for g in meas.index.get_level_values('grp').unique():
            mg = meas.xs(g, level='grp')
            anov.append(dict(grouping=grouping, grp=g, unit=unit, source=mkey, **rm_anova(meas.xs(g, level='grp', drop_level=False))))
            mc = hp_contrasts(mg)
            for k, v in mc.items():
                rows.append(rf.describe(v, k, grouping=grouping, grp=g, unit=unit, source=mkey))
            for var in vlist:
                pr = hp_cells(w[(var, grouping)], unit)
                if g not in pr.index.get_level_values('grp'):
                    continue
                pg_ = pr.xs(g, level='grp')
                idx = mg.index.intersection(pg_.index)
                pc = hp_contrasts(pg_.loc[idx])
                for k in mc:
                    rows.append(rf.describe(pc[k], k, grouping=grouping, grp=g, unit=unit, source=f'{var} predicted'))
                    rows.append(rf.describe(mc[k].loc[idx] - pc[k], k, grouping=grouping, grp=g, unit=unit,
                                            source=f'measured − {var}'))
                res = (mg.loc[idx] - pg_.loc[idx]).assign(grp=g).set_index('grp', append=True)
                anov.append(dict(grouping=grouping, grp=g, unit=unit, source=f'measured − {var}', **rm_anova(res)))
                anov.append(dict(grouping=grouping, grp=g, unit=unit, source=f'{var} predicted',
                                 **rm_anova(pg_.loc[idx].assign(grp=g).set_index('grp', append=True))))
    t = pd.DataFrame(rows).rename(columns={'label': 'contrast'})
    t = bh(t[t.grouping == 'area']).pipe(lambda a: pd.concat([a, t[t.grouping == 'pathway'].pipe(bh)]))
    return t, pd.DataFrame(anov)


# ---------------------------------------------------------------- item 5b
def item_contrasts(pk):
    return {'target − neutral': (pk.target_hp + pk.target_lp) / 2 - (pk.neutral_hp + pk.neutral_lp) / 2,
            'distractor − neutral': pk.distractor_lp - pk.neutral_lp,
            'target − distractor': pk.target_lp - pk.distractor_lp}


def noise_band(pred, resid, seed=0):
    rng = np.random.default_rng(seed)
    pred, rc = np.asarray(pred, float), np.asarray(resid, float) - np.mean(resid)
    sims = pred[None, :] + rc[rng.integers(0, len(rc), (N_SIM, len(rc)))]
    med = np.median(sims, 1)
    sign = np.sign(np.median(pred)) or 1.
    p = stats.wilcoxon(sims, axis=1).pvalue
    power = np.mean((p < .05) & (np.sign(med) == sign))
    return np.percentile(med, [2.5, 97.5]), power


def part_items():
    rows = []
    for grouping, groups in GROUPINGS.items():
        meas = window(HPLP_MEAS, groups)
        pred = window(PRED.format(mode='hplp', v='SD'), groups)
        for g in meas.index.get_level_values('grp').unique():
            mg = meas.xs(g, level='grp').dropna()
            pg_ = pred.xs(g, level='grp').dropna()
            idx = mg.index.intersection(pg_.index)
            mc, pc = item_contrasts(mg.loc[idx]), item_contrasts(pg_.loc[idx])
            for k in mc:
                m, p = mc[k], pc[k]
                band, power = noise_band(p, m - p)
                d = rf.describe(m, k, grouping=grouping, grp=g, source='measured')
                dp = rf.describe(p, k, grouping=grouping, grp=g, source='joint fit predicted')
                dr = rf.describe(m - p, k, grouping=grouping, grp=g, source='measured − joint fit')
                for x in (d, dp, dr):
                    x.update(model_band_lo=band[0], model_band_hi=band[1], power=power,
                             mde=2.8 * m.std(ddof=1) / np.sqrt(len(m)),
                             measured_in_band=bool(band[0] <= m.median() <= band[1]))
                rows += [d, dp, dr]
    t = pd.DataFrame(rows).rename(columns={'label': 'contrast'})
    return pd.concat([bh(t[t.grouping == gg], by=('grouping', 'contrast', 'source')) for gg in GROUPINGS])


def overlap_counts(folder=f'{D}/polished_pred_overlap_2026-11-13'):
    o = pd.concat([pd.read_csv(f, sep='\t') for f in sorted(Path(folder).glob('sub-*_overlap.tsv'))])
    o = o[o.subject >= rf.MIN_SUBJECT]
    per = o.groupby(['roi', 'subject']).agg(fig3_mean=('n_fig3', 'mean'), fig3_min=('n_fig3', 'min'),
                                            overlap_mean=('n_overlap', 'mean'), overlap_min=('n_overlap', 'min'),
                                            overlap2_mean=('n_overlap_within2', 'mean'), reliable=('n_reliable', 'first'),
                                            fwhm=('fwhm_median_fig3', 'median'))
    per['loc_empty_fig3'] = o.groupby(['roi', 'subject']).n_fig3.apply(lambda x: (x == 0).sum())
    s = per.groupby('roi').median()
    s['n_subj_any_empty_fig3'] = per.groupby('roi').loc_empty_fig3.apply(lambda x: (x > 0).sum())
    order = ['V1', 'V2', 'V3', 'V3AB', 'IPS', 'hV4', 'VO', 'LO', 'TO']
    return s.reindex(order).reset_index()


def main(parts):
    pd.set_option('display.width', 250, 'display.max_rows', 500)
    w = lambda df, name: df.to_csv(f'{D}/{name}_n28_{TAG}.tsv', sep='\t', index=False, float_format='%.5g')  # noqa
    if 'hp' in parts:
        t, a = part_hp()
        w(t, 'items_hp_model_vs_measured'), w(a, 'items_hp_model_vs_measured_anova')
    if 'items' in parts:
        w(part_items(), 'items_type_model_vs_measured')
        w(overlap_counts(), 'items_voxel_overlap')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--part', nargs='+', default=['hp', 'items'], choices=['hp', 'items'])
    main(ap.parse_args().part)
