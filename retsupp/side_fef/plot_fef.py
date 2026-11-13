"""Side project (FEF / sPCS): group tests and figure.

Map tests (``collect_prf_cv`` summary, DoG + region HRF), n = 28, Wilcoxon signed-rank one-sided in the direction a
pRF predicts, Benjamini-Hochberg across the frontal ROIs (controls uncorrected), as in the hippocampus project:
cvR² true − scramble > 0; fraction of voxels above the scramble 95th percentile > 0.05; contralateral fraction among
those voxels > 0.5.

Figure (180 mm): a-c the three map tests per ROI; d visual-field centres of the pRF voxels of the leave-one-out sPCS
(all participants, by hemisphere); e group CV R² on the inflated right hemisphere with the group ROIs; f (with ``--gain-wide``) per-participant gain-field estimates in sPCS.

    python -m retsupp.side_fef.plot_fef --summary notes/side_fef/data/prf_cv_summary.tsv \\
        --voxels notes/side_fef/data/prf_cv_voxels_prf.tsv.gz [--gain-wide notes/side_fef/data/gain_wide.tsv] \\
        --out-dir notes/side_fef
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

REGIONS = ['sPCSloso', 'iPCSloso', 'sPCSind', 'FEFatlas', 'precentral', 'lateral_occipital']
FRONTAL = ['sPCSloso', 'iPCSloso', 'sPCSind', 'FEFatlas']
EXTRA = ['sPCS', 'iPCS']   # group ROIs including the tested participant (table only; selection is circular)
LABEL = {'sPCSloso': 'sPCS', 'iPCSloso': 'iPCS', 'sPCSind': 'Frontal, per\nparticipant',
         'FEFatlas': 'FEF atlas', 'precentral': 'Precentral', 'lateral_occipital': 'Lateral occ.',
         'sPCS': 'sPCS (group, all 28)', 'iPCS': 'iPCS (group, all 28)'}
COLOR = {r: '#D62828' for r in FRONTAL} | {'precentral': '#9AA5B1', 'lateral_occipital': '#2A9D8F'}
TESTS = [('d_cvr2', 0, 'Held-out R², true − scramble'), ('frac_above_thr', .05, 'Fraction above scramble 95th pct'),
         ('contra_prf', .5, 'Contralateral fraction')]
GAIN_TERMS = [('dS alone', 'dS alone'), ('dS joint', 'dS joint'), ('target joint', 'Target'),
              ('distractor joint', 'Distractor')]
GAIN_COLOR = {'dS alone': '#D62828', 'dS joint': '#D62828', 'target joint': '#F77F00', 'distractor joint': '#0077B6'}


def style():
    mpl.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['Helvetica', 'Helvetica Neue', 'Arial'], 'font.size': 7,
        'axes.labelsize': 7, 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'axes.linewidth': .8,
        'axes.spines.top': False, 'axes.spines.right': False, 'xtick.major.size': 3, 'ytick.major.size': 3,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none', 'savefig.dpi': 300})


def run_tests(s):
    rows = []
    for col, null, lab in TESTS:
        for r in REGIONS + EXTRA:
            v = (s.loc[s.region == r, col] - null).dropna()
            p = stats.wilcoxon(v, alternative='greater', method='exact' if len(v) <= 50 else 'auto').pvalue \
                if len(v) > 0 and (v != 0).any() else np.nan
            rows.append(dict(measure=col, label=lab, region=r, n=len(v), mean=v.mean() + null, sem=v.sem(),
                             median=v.median() + null, n_above_null=int((v > 0).sum()), p_one_sided=p))
    t = pd.DataFrame(rows)
    t['p_bh_frontal'] = np.nan
    for col in t.measure.unique():
        ix = t.index[(t.measure == col) & t.region.isin(FRONTAL)]
        t.loc[ix, 'p_bh_frontal'] = multipletests(t.loc[ix, 'p_one_sided'], method='fdr_bh')[1]
    return t


def main(summary, voxels, out_dir, gain_wide=None, model='dog_roihrf', fsaverage=None, sums=None, roi_npz=None):
    s = pd.read_csv(summary, sep='\t')
    s = s[s.model == model].copy()
    s['d_cvr2'] = s.median_cvr2_true - s.median_cvr2_scramble
    out_dir = Path(out_dir)
    t = run_tests(s)
    t.to_csv(out_dir / 'data' / f'prf_cv_tests_{model}.tsv', sep='\t', index=False)
    pd.set_option('display.width', 220)
    print(t.round(4).to_string(index=False))

    style()
    fig = plt.figure(figsize=(180 / 25.4, 120 / 25.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], hspace=.75, wspace=.45)
    rng = np.random.default_rng(0)
    for k, (col, null, lab) in enumerate(TESTS):
        ax = fig.add_subplot(gs[0, k])
        for i, r in enumerate(REGIONS):
            v = s.loc[s.region == r, col].dropna().to_numpy()
            ax.scatter(i + rng.uniform(-.18, .18, len(v)), v, s=5, color=COLOR[r], alpha=.35, lw=0)
            if len(v):
                ax.errorbar(i, v.mean(), v.std(ddof=1) / np.sqrt(len(v)), fmt='D', ms=4, color=COLOR[r], mec='#1F2933',
                            mew=.8, elinewidth=1, capsize=0, zorder=3)
            p = t.loc[(t.measure == col) & (t.region == r), 'p_bh_frontal' if r in FRONTAL else 'p_one_sided'].iloc[0]
            if p < .05:
                ax.text(i, 1.0, '*', transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=8)
        ax.axhline(null, color='#9AA5B1', lw=.6, ls='--', zorder=0)
        ax.set_xticks(range(len(REGIONS)))
        ax.set_xticklabels([LABEL[r] for r in REGIONS], rotation=45, ha='right', rotation_mode='anchor', fontsize=6)
        ax.set_ylabel(lab)
        if col == 'd_cvr2':
            lo, hi = np.nanpercentile(s.loc[s.region.isin(FRONTAL + ['precentral']), col], [1, 99])
            ax.set_ylim(min(lo, -.002) * 1.3, max(hi, .002) * 1.3)
            ax.text(.98, .97, 'Lateral occipital\nmostly above axis', transform=ax.transAxes, ha='right', va='top',
                    fontsize=5.5, color='#2A9D8F')
        ax.text(-.3, 1.05, 'abc'[k], transform=ax.transAxes, fontsize=8, fontweight='bold', va='bottom')

    # d: visual-field centres of sPCS pRF voxels
    ax = fig.add_subplot(gs[1, 0])
    v = pd.read_csv(voxels, sep='\t')
    v = v[(v.region == 'sPCSloso') & (v.model == model)]
    for h, c, lab in (('L', '#3D5A80', 'Left hemisphere'), ('R', '#D62828', 'Right hemisphere')):
        d = v[v.hemi == h]
        ax.scatter(d.x, d.y, s=4, color=c, alpha=.5, lw=0, label=f'{lab} (n = {len(d)})')
    ax.add_patch(plt.Circle((0, 0), 3.17, fill=False, color='#9AA5B1', lw=.6, ls='--'))
    ax.axhline(0, color='#9AA5B1', lw=.4)
    ax.axvline(0, color='#9AA5B1', lw=.4)
    ax.set_xlim(-5.2, 5.2)
    ax.set_ylim(-5.2, 5.2)
    ax.set_aspect('equal')
    ax.set_xlabel('x (°)')
    ax.set_ylabel('y (°)')
    ax.legend(frameon=False, fontsize=5.5, loc='upper left', bbox_to_anchor=(-.05, 1.22), handletextpad=.1)
    ax.text(-.3, 1.05, 'd', transform=ax.transAxes, fontsize=8, fontweight='bold', va='bottom')

    # e: where the group ROIs are: right hemisphere, inflated fsaverage, lateral view
    ax = fig.add_subplot(gs[1, 1])
    if fsaverage is not None:
        import nibabel as nib
        fsd = Path(fsaverage)
        vi = nib.freesurfer.read_geometry(str(fsd / 'surf' / 'rh.inflated'))[0]
        sulc = nib.freesurfer.read_morph_data(str(fsd / 'surf' / 'rh.sulc'))
        q = np.load(sums)['median_cv_r2'][1]
        rois = np.load(roi_npz)
        lat = vi[:, 0] > np.percentile(vi[:, 0], 30)          # lateral-facing half of the inflated surface
        o = np.argsort(vi[lat, 0])
        yy, zz = vi[lat, 1][o], vi[lat, 2][o]
        ax.scatter(yy, zz, c=np.where(sulc[lat][o] > 0, '#C8CED6', '#E8EBEF'), s=.3, lw=0, rasterized=True)
        qq = q[lat][o]
        hot = qq > .0025
        sc = ax.scatter(yy[hot], zz[hot], c=np.clip(qq[hot], 0, .02), cmap='magma_r', vmin=0, vmax=.02, s=.6, lw=0,
                        rasterized=True)
        for r, c in (('sPCS', '#D62828'), ('iPCS', '#0077B6')):
            m = rois[r][1][lat][o]
            ax.scatter(yy[m], zz[m], s=.6, color=c, lw=0, alpha=.25, rasterized=True)
            ax.text(np.median(yy[m]) + 4, np.median(zz[m]) + 8, r, color=c, fontsize=6, ha='left')
        ax.set_aspect('equal')
        ax.set_axis_off()
        cb = fig.colorbar(sc, ax=ax, fraction=.04, pad=.01, shrink=.6)
        cb.set_label('Group median CV R²', fontsize=6)
        cb.ax.tick_params(labelsize=5.5)
        ax.set_title('Right hemisphere, lateral (inflated); anterior right', fontsize=6)
    ax.text(-.1, 1.05, 'e', transform=ax.transAxes, fontsize=8, fontweight='bold', va='bottom')

    ax = fig.add_subplot(gs[1, 2])
    if gain_wide is not None:
        g = pd.read_csv(gain_wide, sep='\t')
        g = g[g.roi == 'sPCSloso']
        for i, (term, lab) in enumerate(GAIN_TERMS):
            x = 100 * g[term].dropna().to_numpy()
            ax.scatter(i + rng.uniform(-.18, .18, len(x)), x, s=5, color=GAIN_COLOR[term], alpha=.35, lw=0)
            b = np.median(np.random.default_rng(0).choice(x, (10000, len(x))), 1)
            ax.errorbar(i, np.median(x), [[np.median(x) - np.percentile(b, 2.5)], [np.percentile(b, 97.5) - np.median(x)]],
                        fmt='o', ms=4, color=GAIN_COLOR[term], mec='#1F2933', mew=.8, elinewidth=1, capsize=0, zorder=3)
        ax.axhline(0, color='#1F2933', lw=.6)
        ax.set_xticks(range(len(GAIN_TERMS)))
        ax.set_xticklabels([lab for _, lab in GAIN_TERMS], rotation=45, ha='right', rotation_mode='anchor', fontsize=6)
        ax.set_ylabel('Gain in sPCS (%)')
        lim = np.nanpercentile(np.abs(100 * g[[t for t, _ in GAIN_TERMS]].to_numpy()), 95) * 1.2
        ax.set_ylim(-lim, lim)
    else:
        ax.axis('off')
        ax.text(.5, .5, 'Gain fields not fitted', ha='center', va='center', transform=ax.transAxes)
    ax.text(-.3, 1.05, 'f', transform=ax.transAxes, fontsize=8, fontweight='bold', va='bottom')
    for ext in ('pdf', 'png'):
        fig.savefig(out_dir / f'fig_fef.{ext}', bbox_inches='tight', pad_inches=.02)
    print('wrote', out_dir / 'fig_fef.pdf')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--summary', required=True)
    ap.add_argument('--voxels', required=True)
    ap.add_argument('--gain-wide', default=None)
    ap.add_argument('--out-dir', default='notes/side_fef')
    ap.add_argument('--model', default='dog_roihrf')
    ap.add_argument('--fsaverage', default=None)
    ap.add_argument('--sums', default=None, help='group_sums_fsaverage_v3.npz')
    ap.add_argument('--roi-npz', default='notes/side_fef/data/fsaverage_rois.npz')
    a = ap.parse_args()
    main(a.summary, a.voxels, a.out_dir, a.gain_wide, a.model, a.fsaverage, a.sums, a.roi_npz)
