"""Two checks for the Discussion (2026-11-12), sub-03–30 (n = 28), local, no fits.

1. Behaviour next to vs opposite HP by history: the richter_followup B contrasts per session × block, in session 1
   block 1 alone (no location has been HP yet), and the interaction (session 1 block 1 vs the other three blocks,
   paired across subjects; repeated-measures ANOVA relation × session-block on the cell means).
2. Model vs measured item responses (Fig. 3d, e): per subject, measured − m7-predicted HP − LP, HP − opposite,
   next − opposite (early areas), and the between-subject correlation of measured and predicted.

    python -m retsupp.visualize.paper.discussion_checks
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.anova import AnovaRM

from retsupp.visualize.paper import richter_followup as rf

SUFFIX = '2026-11-12'
TRIALS = 'notes/data/behaviour_trials_2026-11-11.tsv.gz'
REL = 'notes/data/event_deconv_relative_final2_2026-11'
M7 = 'notes/data/kappa_neighbour_2026-11/event_deconv_m7'
SPECS = [('distractor', 'dist_rel', ['hp', 'next', 'opposite', 'absent']),
         ('target', 'target_rel', ['hp', 'next', 'opposite'])]
CONTRASTS = {'distractor': [('next', 'opposite'), ('hp', 'next'), ('hp', 'opposite'), ('hp', 'absent')],
             'target': [('next', 'opposite'), ('hp', 'next'), ('hp', 'opposite')]}


def behaviour():
    t = rf.add_history(rf.load_trials(TRIALS))
    t['sb'] = 's' + t.session.astype(str) + 'b' + t.block.astype(str)
    rows, counts, anova = [], [], []
    for item, col, lev in SPECS:
        d = t if item == 'distractor' else t[t.distractor_location != 10]
        ok = d[d.ok]
        cnt = ok.groupby(['sb', col, 'subject']).size().groupby(['sb', col]).agg(['median', 'min', 'max'])
        counts.append(cnt.reset_index().rename(columns={col: 'cell'}).assign(item=item))
        m = ok.groupby(['subject', 'sb', col]).rt_ms.mean().unstack([1, 2])
        allm = ok.groupby(['subject', col]).rt_ms.mean().unstack()
        for a, b in CONTRASTS[item]:
            lab = f'{a} − {b}'
            rows.append(rf.describe(allm[a] - allm[b], lab, item=item, subset='all blocks'))
            per = {sb: m[(sb, a)] - m[(sb, b)] for sb in ('s1b1', 's1b2', 's2b1', 's2b2')}
            for sb, v in per.items():
                rows.append(rf.describe(v, lab, item=item, subset=sb))
            rest = pd.concat([per[k] for k in ('s1b2', 's2b1', 's2b2')], axis=1).mean(1)
            rows.append(rf.describe(per['s1b1'] - rest, lab, item=item, subset='s1b1 − mean(other three blocks)'))
        # Repeated-measures ANOVA: relation (without 'absent') × session-block, subjects complete in every cell.
        rel = [c for c in lev if c != 'absent']
        long = (ok[ok[col].isin(rel)].groupby(['subject', 'sb', col]).rt_ms.mean().rename('rt').reset_index())
        full = long.groupby('subject').size() == len(rel) * 4
        long = long[long.subject.isin(full[full].index)]
        res = AnovaRM(long, 'rt', 'subject', within=['sb', col]).fit().anova_table
        anova.append(res.reset_index().rename(columns={'index': 'effect'}).assign(item=item, n=long.subject.nunique()))
    return pd.DataFrame(rows), pd.concat(counts), pd.concat(anova)


def model_vs_measured():
    from retsupp.visualize.paper.fig2_prf_fits import PATHWAY
    groups = {r: g for g, rs in PATHWAY.items() for r in rs}
    out, cor = [], []
    for g, grp in (('Early', groups), ('All areas', None)):
        meas = rf.roi_cells(REL, grp)
        pred = rf.roi_cells(M7, grp)
        if grp is None:
            meas, pred = meas.assign(grp='All areas'), pred.assign(grp='All areas')
        mm = meas[meas.grp == g].set_index('subject')
        pp = pred[pred.grp == g].set_index('subject')
        idx = mm.index.intersection(pp.index)
        for k in ('hp_lp', 'hp_opp', 'next_opp', 'hp_next'):
            out.append(rf.describe(mm.loc[idx, k], k, group=g, source='measured'))
            out.append(rf.describe(pp.loc[idx, k], k, group=g, source='m7 predicted'))
            out.append(rf.describe(mm.loc[idx, k] - pp.loc[idx, k], k, group=g, source='measured − predicted'))
            r, p = stats.spearmanr(mm.loc[idx, k], pp.loc[idx, k])
            rp, pp_ = stats.pearsonr(mm.loc[idx, k], pp.loc[idx, k])
            cor.append(dict(group=g, contrast=k, n=len(idx), spearman_rho=r, spearman_p=p, pearson_r=rp, pearson_p=pp_))
    return pd.DataFrame(out), pd.DataFrame(cor)


def main():
    w = lambda df, name: df.to_csv(f'notes/data/{name}_{SUFFIX}.tsv', sep='\t', index=False, float_format='%.5g')  # noqa
    b, c, a = behaviour()
    w(b, 'discussion_behaviour_by_block_n28'), w(c, 'discussion_behaviour_trial_counts_n28')
    w(a, 'discussion_behaviour_anova_n28')
    m, r = model_vs_measured()
    w(m, 'discussion_model_vs_measured_n28'), w(r, 'discussion_model_vs_measured_corr_n28')
    pd.set_option('display.width', 220)
    print(b.round(3).to_string(index=False)); print(c.to_string(index=False)); print(a.round(4).to_string(index=False))
    print(m.round(3).to_string(index=False)); print(r.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
