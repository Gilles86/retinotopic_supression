"""Why does m5 lose to m4 in leave-one-HP-condition-out CV?
(1) decompose actual loss with the fold's fitted g_HP. (2) linear LOCO: delta from 3 conditions, scored on the 4th.
(3) power: re-centre each subject x ROI on a planted dS, keeping the real across-condition noise."""
import glob, numpy as np, pandas as pd
from scipy.stats import wilcoxon, spearmanr
D='/data/ds-retsupp/derivatives/'; N=D+'af_prf_cv_ladder_nested/'
ROIS=['V1','V2','V3','V3AB','hV4','LO','TO','VO','IPS']
st=pd.concat([pd.read_csv(f,sep='\t',usecols=['subject','roi','mech','voxel_id','hp','sst','a','b','lin_err'])
              for f in sorted(glob.glob(D+'af_fixed_group_mechanisms_maxsd20/sub-*/*_stats.tsv.gz'))])
st=st[(st.mech=='dS')&st.roi.isin(ROIS)&(st.lin_err<=1e-3)]
null=pd.read_csv(N+'cv_ladder_per_voxel.tsv',sep='\t',usecols=['subject','roi','voxel_id','cv_r2_null']).groupby(['subject','roi','voxel_id']).cv_r2_null.mean().reset_index()
st=st.merge(null,on=['subject','roi','voxel_id'],how='left'); st['sig']=st.cv_r2_null>0

# per condition sums: all voxels (to estimate), signal voxels (to score)
K=['subject','roi','hp']
al=st.groupby(K).agg(a=('a','sum'),b=('b','sum')).reset_index()
sg=st[st.sig].groupby(K).agg(sa=('a','sum'),sb=('b','sum'),sst=('sst','sum'),n=('voxel_id','count')).reset_index()
c=al.merge(sg,on=K); c=c[c.n>=5]
c['cond_est']=c.b/c.a
print('Across-condition SD of conditional dS, subject x ROI, median:',c.groupby(['subject','roi']).cond_est.std().median().round(4),
      ' mean effect median:',c.groupby(['subject','roi']).apply(lambda g:g.b.sum()/g.a.sum()).median().round(4))

# (1) actual m5-m4 on held-out condition vs linear prediction from fold fit
p=pd.read_csv(N+'stability_20261101/shared_params.tsv',sep='\t'); p=p[(p.fit=='bounded')&p.model.isin(['m4_TDS','m5_TDS_dS'])]
w=p.pivot_table(index=['subject','roi','fold'],columns='model',values=['g_HP','g_LP']).reset_index()
w.columns=['subject','roi','fold','hp4','hp5','lp4','lp5'] if False else [''.join(x) if x[1]=='' else f'{x[0]}_{x[1]}' for x in w.columns]
w['dS_fold']=w.g_HP_m5_TDS_dS-w.g_LP_m5_TDS_dS; w['move_HP']=w.g_HP_m5_TDS_dS-w.g_HP_m4_TDS; w['move_LP']=w.g_LP_m5_TDS_dS-w.g_LP_m4_TDS
s=pd.read_csv(N+'spatial_20261101/spatial_cv_bounded.tsv.gz',sep='\t',usecols=['subject','roi','fold','voxel_id','hp','m4_TDS','m5_TDS_dS'])
s=s[s.roi.isin(ROIS)].merge(st[['subject','roi','voxel_id','hp','sst','a','b','sig']],on=['subject','roi','voxel_id','hp'])
s=s[s.sig]; s['dsse']=-(s.m5_TDS_dS-s.m4_TDS)*s.sst
f=s.groupby(['subject','roi','fold','hp']).agg(dsse=('dsse','sum'),a=('a','sum'),b=('b','sum'),sst=('sst','sum'),n=('voxel_id','count')).reset_index().merge(w,on=['subject','roi','fold'])
f=f[f.n>=5]
print('\nFold dS (m5, bounded): median',f.dS_fold.median().round(4),'SD',f.dS_fold.std().round(4),'IQR',f.dS_fold.quantile([.25,.75]).round(3).tolist(),'pct neg',round((f.dS_fold<0).mean()*100))
print('g_HP move m4->m5 median',f.move_HP.median().round(4),' g_LP move median',f.move_LP.median().round(4))
f['noise']=f.move_HP**2*f.a; f['signal']=2*f.move_HP*f.b; f['pred']=f.noise-f.signal
print('Actual dSSE vs predicted from g_HP move: spearman',round(spearmanr(f.dsse,f.pred)[0],3),' share of cells actual loss',round((f.dsse>0).mean(),3),' predicted loss',round((f.pred>0).mean(),3))
print('Sum over cells /sum sst e-5: actual loss',round(f.dsse.sum()/f.sst.sum()*1e5,2),' penalty d^2 a',round(f.noise.sum()/f.sst.sum()*1e5,2),' gain 2 d b',round(f.signal.sum()/f.sst.sum()*1e5,2))
print('median cell e-5: actual',round((f.dsse/f.sst).median()*1e5,3),' penalty',round((f.noise/f.sst).median()*1e5,3),' gain',round((f.signal/f.sst).median()*1e5,3))

# (2,3) linear LOCO
def loco(c,shrink=1.0):
    r=[]
    for (su,ro),g in c.groupby(['subject','roi']):
        if len(g)<3: continue
        for i in range(len(g)):
            te=g.iloc[i]; tr=g.drop(g.index[i])
            d=shrink*tr.b.sum()/tr.a.sum()
            r.append((su,ro,-(d**2*te.sa-2*d*te.sb),te.sst))
    r=pd.DataFrame(r,columns=['subject','roi','gain','sst']).groupby(['subject','roi']).sum().reset_index()
    r['dr2']=r.gain/r.sst*1e5
    x=r.groupby('subject').dr2.median()
    return (x>0).mean()*100,wilcoxon(x).pvalue,x.median()
print('\nLinear LOCO on real data (delta from other 3 conditions, no refit, no shared-term changes):')
for k in [1,.5,.25]:
    print(f'  shrink {k}: improved %.0f%%  p=%.4f  dR2 %+.3fe-5'%loco(c,k))
# power, re-centred on planted effect
own=c.groupby(['subject','roi']).apply(lambda g:g.b.sum()/g.a.sum()).rename('own').reset_index()
print('\nPower: every subject x ROI re-centred on planted dS, real condition noise kept (500 not needed: deterministic):')
for planted in [0,-.005,-.016,-.03,-.06,-.1,-.2]:
    t=c.merge(own,on=['subject','roi']).copy()
    t['b']=t.b+(planted-t.own)*t.a; t['sb']=t.sb+(planted-t.own)*t.sa
    print(f'  planted {planted:+.3f}: improved %.0f%%  p=%.4f  dR2 %+.3fe-5'%loco(t))
