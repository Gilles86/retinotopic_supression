"""LOSO dS / dD binned by pRF centre distance from the condition's HP location (DoG, linear stats).
delta is unchanged from the reported test (median of other subjects' own estimate, all voxels)."""
import glob
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

D = '/data/ds-retsupp/derivatives'
ecc = 4 / np.sqrt(2)
LOC = {'upper_right': (ecc, ecc), 'upper_left': (-ecc, ecc), 'lower_left': (-ecc, -ecc), 'lower_right': (ecc, -ecc)}
ROIS = ['V1', 'V2', 'V3', 'V3AB', 'hV4', 'LO', 'TO', 'VO', 'IPS']

files = sorted(glob.glob(f'{D}/af_fixed_group_mechanisms_maxsd20/sub-*/*_stats.tsv.gz'))
st = pd.concat([pd.read_csv(f, sep='\t') for f in files])
st = st[st.mech.isin(['dS', 'dD', 'T', 'S', 'items']) & st.roi.isin(ROIS)]
print('rows per voxel x mech (conditions):', st.groupby(['subject', 'roi', 'mech', 'voxel_id']).size().value_counts().to_dict())
print('hp values', st.hp.unique())

null = pd.read_csv(f'{D}/af_prf_cv_ladder_nested/cv_ladder_per_voxel.tsv', sep='\t',
                   usecols=['subject', 'roi', 'voxel_id', 'cv_r2_null'])
null = null.groupby(['subject', 'roi', 'voxel_id']).cv_r2_null.mean().reset_index()
st = st.merge(null, on=['subject', 'roi', 'voxel_id'], how='left')

own = st.groupby(['subject', 'roi', 'mech']).agg(a=('a', 'sum'), b=('b', 'sum'), lin=('lin_err', 'first')).reset_index()
own['own'] = own.b / own.a
own.loc[own.lin > 1e-3, 'own'] = np.nan
own['delta'] = [own[(own.roi == r) & (own.mech == m) & (own.subject != s)].own.median()
                for s, r, m in zip(own.subject, own.roi, own.mech)]
st = st.merge(own[['subject', 'roi', 'mech', 'delta', 'lin']], on=['subject', 'roi', 'mech'])
st = st[st.lin <= 1e-3]

hx = st.hp.map(lambda h: LOC[h][0]).values
hy = st.hp.map(lambda h: LOC[h][1]).values
st['dist'] = np.hypot(st.x - hx, st.y - hy)
st['gain'] = -(st.delta ** 2 * st.a - 2 * st.delta * st.b)  # -dSSE
bins = [0, 1, 2, 3, 4, 6, 100]
st['dbin'] = pd.cut(st.dist, bins, right=False)
sig = st[st.cv_r2_null > 0]


def report(df, name):
    out = []
    for keys, g in df.groupby('dbin', observed=True):
        s = g.groupby(['subject', 'roi']).agg(gain=('gain', 'sum'), sst=('sst', 'sum'), a=('a', 'sum'), b=('b', 'sum'),
                                              n=('voxel_id', 'count')).reset_index()
        s = s[s.n >= 5]
        s['dr2'] = s.gain / s.sst * 1e5
        s['own'] = s.b / s.a
        x = s.groupby('subject').dr2.median().dropna()
        o = s.groupby('subject').own.median().dropna()
        out.append(dict(bin=str(keys), n_sub=len(x), vox=int(g.voxel_id.count()), share_a=g.a.sum() / df.a.sum(),
                        pct_improve=(x > 0).mean() * 100, p=wilcoxon(x).pvalue, dr2_e5=x.median(),
                        own_med=o.median(), own_pct_neg=(o < 0).mean() * 100, own_p=wilcoxon(o).pvalue))
    print(f'\n== {name}')
    print(pd.DataFrame(out).round(4).to_string(index=False))


for m in ['dS', 'dD', 'T']:
    report(sig[sig.mech == m], f'{m}, signal voxels, by distance of pRF centre from HP')

# top-sensitivity voxels: a-based (blind to sign of b), per subject x roi x condition
for m in ['dS', 'dD']:
    g = sig[sig.mech == m].copy()
    g['rk'] = g.groupby(['subject', 'roi', 'hp']).a.rank(pct=True)
    print(f'\n== {m}, by sensitivity percentile (a), signal voxels')
    for lo, hi in [(0, .5), (.5, .75), (.75, .9), (.9, 1.01)]:
        h = g[(g.rk >= lo) & (g.rk < hi)]
        s = h.groupby(['subject', 'roi']).agg(gain=('gain', 'sum'), sst=('sst', 'sum'), a=('a', 'sum'), b=('b', 'sum')).reset_index()
        s['dr2'] = s.gain / s.sst * 1e5
        s['own'] = s.b / s.a
        x = s.groupby('subject').dr2.median()
        o = s.groupby('subject').own.median()
        print(f'  a pct {lo:.2f}-{hi:.2f}: share_a {h.a.sum() / g.a.sum():.3f}  improve {(x > 0).mean() * 100:3.0f}%  p={wilcoxon(x).pvalue:.4f}  '
              f'dR2 {x.median():+.2f}e-5  own {o.median():+.4f} neg {(o < 0).mean() * 100:3.0f}% p={wilcoxon(o).pvalue:.4f}')

# per ROI within 2 deg of HP
g = sig[(sig.mech == 'dS') & (sig.dist < 2)]
s = g.groupby(['subject', 'roi']).agg(gain=('gain', 'sum'), sst=('sst', 'sum'), n=('voxel_id', 'count')).reset_index()
s = s[s.n >= 5]
s['dr2'] = s.gain / s.sst * 1e5
print('\n== dS, <2 deg from HP, by ROI')
for r in ROIS:
    x = s[s.roi == r].dr2
    if len(x) > 5:
        print(f'  {r:5s} n={len(x):2d} improve {(x > 0).mean() * 100:3.0f}%  p={wilcoxon(x).pvalue:.4f}  dR2 {x.median():+.2f}e-5')
print('\nCheck, all signal voxels, should match reported dS 77% p=.010:')
s = sig[sig.mech == 'dS'].groupby(['subject', 'roi']).agg(gain=('gain', 'sum'), sst=('sst', 'sum')).reset_index()
s['dr2'] = s.gain / s.sst * 1e5
x = s.groupby('subject').dr2.median()
print(f'  {(x > 0).mean() * 100:.0f}% p={wilcoxon(x).pvalue:.4f} (9 ROIs V1-IPS)')
