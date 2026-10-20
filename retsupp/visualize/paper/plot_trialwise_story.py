"""Narrative figure: trial history and trial-by-trial behaviour in the
attention-field gains.

Story, panel by panel:

a  The task, and what a target repeat is (icons of the search display).
b  Behaviour: repeating the target location speeds responses; repeating
   the distractor location barely does.
c  Joint AF fits with repeat/switch-split gains: the target gain drops on
   target-repeat trials from V3 upward; the distractor gain doesn't.
d  The same target-repeat effect painted on cortex.
e  Trial-by-trial: the overall trial response predicts RT with opposite
   signs in early visual vs dorsal cortex (cortex map) ...
f  ... per ROI (linear mixed models).
g  Single-trial response fluctuations are shared across ROIs.
h  Within dorsal ROIs the target / global / distractor gains are almost
   perfectly collinear per trial, so only the overall response is
   interpretable there.

Inputs (all small, under notes/data):
- repeat_af_fits_params.tsv           (collect_repeat_af_fits.py)
- trialwise_af_trial_table.tsv.gz     (analyze_trialwise_af_gains.py collect)
- trialwise_af_mixed_stats.tsv        (mixed_trialwise_af_rt.R)
- trialwise_af_gain_collinearity.tsv  (mixed_trialwise_af_rt.R)
- fsaverage_roi_flatmap.npz           (make_roi_flatmap_cache.py)

``python -m retsupp.visualize.paper.plot_trialwise_story``
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
from scipy import ndimage, stats
from statsmodels.stats.multitest import multipletests

ROI_ORDER = ['V1', 'V2', 'V3', 'V3AB', 'hV4', 'LO', 'TO', 'VO', 'IPS', 'SPL1', 'FEF']
EARLY, DORSAL = ['V1', 'V2', 'V3'], ['IPS', 'SPL1', 'FEF']

PAL = {
    'fg': '#1F2933', 'muted': '#9AA5B1', 'target': '#F77F00',
    'dist': '#0077B6', 'fast': '#2A9D8F', 'slow': '#7B2CBF',
    'item': '#E09000', 'singleton': '#00A600', 'screen': '#8C8C8C',
}
RT_CMAP = LinearSegmentedColormap.from_list(
    'rt', [PAL['fast'], '#F4F4F4', PAL['slow']])
REPEAT_CMAP = LinearSegmentedColormap.from_list(
    'rep', [PAL['target'], '#F4F4F4', PAL['muted']])

mpl.rcParams.update({
    'font.family': 'Helvetica',
    'font.sans-serif': ['Helvetica', 'Helvetica Neue', 'Arial'],
    'font.size': 7, 'axes.labelsize': 8, 'axes.titlesize': 8,
    'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'axes.linewidth': 0.8, 'axes.spines.top': False,
    'axes.spines.right': False, 'axes.labelpad': 3,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
    'lines.linewidth': 1.2, 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'svg.fonttype': 'none', 'savefig.dpi': 300,
})


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def one_sample(df, col, by='roi', order=ROI_ORDER):
    rows = []
    for key in order:
        x = df.loc[df[by] == key, col].dropna()
        t, p = stats.ttest_1samp(x, 0.0)
        ci = stats.t.ppf(0.975, len(x) - 1) * x.sem()
        rows.append(dict(roi=key, mean=x.mean(), ci=ci, t=t, p=p))
    out = pd.DataFrame(rows)
    out['q'] = multipletests(out['p'], method='fdr_bh')[1]
    return out


def repeat_effects(fits):
    t = fits[fits['model'] == 'targetRepeat'].assign(
        d=lambda d: d['g_T_dyn_repeat'] - d['g_T_dyn_switch'])
    r = fits[fits['model'] == 'repeat'].assign(
        d=lambda d: ((d['g_HP_dyn_repeat'] - d['g_HP_dyn'])
                     + (d['g_LP_dyn_repeat'] - d['g_LP_dyn'])) / 2)
    return one_sample(t, 'd'), one_sample(r, 'd')


def behaviour(tt):
    """Per-subject RT (ms), repeat − switch, for target and distractor."""
    v = tt[tt['roi'] == 'V1']
    rows = []
    for subj, d in v.groupby('subject'):
        tgt = d.groupby(d['target_repeat'] > 0)['rt'].mean() * 1000
        # distractor repeats are mostly HP trials (which are faster
        # anyway), so compare repeat vs switch within HP and LP trials
        dst = [g.groupby(g['dist_repeat'] > 0)['rt'].mean().diff().iloc[-1]
               for _, g in d[d['trial_type'] != 'none'].groupby('trial_type')]
        rows += [dict(subject=subj, kind='Target', d=tgt[True] - tgt[False]),
                 dict(subject=subj, kind='Distractor',
                      d=1000 * np.nanmean(dst))]
    return pd.DataFrame(rows)


def amplitude(tt):
    """Overall single-trial response: z of the mean z-scored gain, within
    (subject, session, run, trial type) — matches the R mixed models."""
    cell = ['subject', 'roi', 'session', 'run', 'trial_type']
    n = 2 + (tt['trial_type'] != 'none')
    a = (tt['T'] + tt['G'] + tt['D']) / n
    return tt.assign(A=a.groupby([tt[c] for c in cell]).transform(
        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else x * 0))


def cross_roi_corr(tt):
    mats = []
    for _, d in tt.groupby('subject'):
        w = d.pivot_table(index='trial_idx', columns='roi', values='A')
        mats.append(np.arctanh(np.clip(w[ROI_ORDER].corr().values, -.999, .999)))
    return pd.DataFrame(np.tanh(np.nanmean(mats, axis=0)),
                        index=ROI_ORDER, columns=ROI_ORDER)


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def panel_letter(ax, letter, x=-0.12, y=1.04):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=8,
            fontweight='bold', va='bottom', ha='right')


def search_display(ax, cx, cy, r, target_pos, dist_pos, scale=1.0):
    """Mini 8-item search display: orange bars, the target is the bar with
    the odd orientation, the distractor the green colour singleton."""
    ax.add_patch(Rectangle((cx - 1.35 * r, cy - 1.35 * r), 2.7 * r, 2.7 * r,
                           facecolor=PAL['screen'], edgecolor='none',
                           zorder=0))
    L, W = 0.42 * r * scale, 0.11 * r * scale
    for i in range(8):
        ang = np.deg2rad(i * 45)
        x, y = cx + r * np.cos(ang), cy + r * np.sin(ang)
        vertical = i != target_pos
        col = PAL['singleton'] if i == dist_pos else PAL['item']
        w, h = (W, L) if vertical else (L, W)
        ax.add_patch(Rectangle((x - w / 2, y - h / 2), w, h, facecolor=col,
                               edgecolor='none', zorder=2))
    ax.add_patch(Circle((cx, cy), 0.05 * r, color='white', zorder=2))


def roi_points(ax, st, color, order=ROI_ORDER, offset=0.0, ms=4):
    """``color``: one colour, or a function of the ROI's stats row."""
    xs = np.arange(len(order)) + offset
    for x, r in zip(xs, st.set_index('roi').loc[order].itertuples()):
        c = color(r) if callable(color) else color
        ax.plot([x, x], [r.mean - r.ci, r.mean + r.ci], color=c, lw=1.1,
                solid_capstyle='butt', zorder=2)
        ax.plot(x, r.mean, 'o', ms=ms, color=c, mew=0, zorder=3)
        if r.q < 0.05:
            s = np.sign(r.mean)
            ax.text(x, r.mean + s * (r.ci * 1.08), '**' if r.q < .01 else '*',
                    ha='center', va='bottom' if s > 0 else 'top',
                    fontsize=8, fontweight='bold', color=c, zorder=4)
    ax.axhline(0, color=PAL['fg'], lw=0.6, zorder=1)
    ax.set_xticks(range(len(order)), order, rotation=90)
    ax.set_xlim(-0.6, len(order) - 0.4)


def flatmap(ax, flat, values, cmap, norm, title=None):
    """Paint per-ROI values onto the left-hemisphere fsaverage flatmap."""
    roi, curv = flat['roi'], flat['curv']
    h, w = roi.shape
    # left hemisphere occupies the left half; crop to occipito-parietal +
    # frontal eye fields
    x0, x1, y0, y1 = int(.17 * w), int(.50 * w), int(.12 * h), int(.88 * h)
    roi, curv = roi[y0:y1, x0:x1].copy(), curv[y0:y1, x0:x1]
    # keep each ROI's largest connected patch (drops atlas specks)
    for i in range(1, len(ROI_ORDER) + 1):
        comp, n = ndimage.label(roi == i)
        if n > 1:
            sizes = ndimage.sum(np.ones_like(comp), comp, range(1, n + 1))
            roi[(comp > 0) & (comp != np.argmax(sizes) + 1)] = np.nan
    ax.imshow(np.where(np.isnan(curv), np.nan, curv > 0), cmap='Greys',
              vmin=-1.5, vmax=3.5, origin='upper', interpolation='bilinear')
    rgba = np.zeros(roi.shape + (4,))
    for i, name in enumerate(ROI_ORDER, start=1):
        if name in values:
            rgba[roi == i] = cmap(norm(values[name]))
    ax.imshow(rgba, origin='upper', interpolation='nearest')
    # ROI outlines + labels
    ax.contour(np.nan_to_num(roi), levels=np.arange(0.5, 12), colors=PAL['fg'],
               linewidths=0.3)
    nudge = {'FEF': (0, -60), 'SPL1': (-70, -10), 'V1': (35, 0)}
    for i, name in enumerate(ROI_ORDER, start=1):
        yy, xx = np.nonzero(roi == i)
        if len(xx) and name not in ('V2', 'V3'):
            dx, dy = nudge.get(name, (0, 0))
            ax.text(np.median(xx) + dx, np.median(yy) + dy, name, fontsize=5.5,
                    ha='center', va='center', color=PAL['fg'])
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=7, loc='left')


def colorbar(fig, ax, cmap, norm, label, ticks):
    cax = ax.inset_axes([0.62, 0.06, 0.34, 0.035])
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                      orientation='horizontal', ticks=ticks)
    cb.outline.set_linewidth(0.5)
    cb.ax.tick_params(labelsize=6, length=2, width=0.5)
    cb.set_label(label, fontsize=6, labelpad=1)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def main(data: Path, out: Path):
    fits = pd.read_csv(data / 'repeat_af_fits_params.tsv', sep='\t')
    tt = amplitude(pd.read_csv(data / 'trialwise_af_trial_table.tsv.gz', sep='\t'))
    mixed = pd.read_csv(data / 'trialwise_af_mixed_stats.tsv', sep='\t')
    coll = pd.read_csv(data / 'trialwise_af_gain_collinearity.tsv', sep='\t')
    flat = np.load(data / 'fsaverage_roi_flatmap.npz')

    st_T, st_D = repeat_effects(fits)
    beh = behaviour(tt)
    amp = (mixed[(mixed['level'] == 'roi') & (mixed['model'] == 'overall')]
           .rename(columns={'unit': 'roi', 'beta': 'mean'}))
    amp['ci'] = stats.t.ppf(0.975, amp['df']) * amp['se']
    corr = cross_roi_corr(tt)

    fig = plt.figure(figsize=(7.25, 6.9))
    outer = fig.add_gridspec(3, 1, height_ratios=[1, 0.95, 1.12], hspace=0.5,
                             left=0.075, right=0.94, top=0.965, bottom=0.075)
    row1 = outer[0].subgridspec(1, 4, width_ratios=[1.35, 0.75, 1.75, 1.15],
                                wspace=0.55)
    row2 = outer[1].subgridspec(1, 2, width_ratios=[1.15, 2.1], wspace=0.22)
    row3 = outer[2].subgridspec(1, 2, width_ratios=[1.15, 1.6], wspace=0.35)

    # a — task + definition of a repeat -----------------------------------
    ax = fig.add_subplot(row1[0])
    ax.set_xlim(0, 10.4)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.set_axis_off()
    for x, d_pos, lab in [(2.3, 6, 'Trial n − 1'), (8.0, 1, 'Trial n')]:
        search_display(ax, x, 6.3, 1.5, target_pos=3, dist_pos=d_pos)
        tx = x + 1.5 * np.cos(np.deg2rad(135))
        ty = 6.3 + 1.5 * np.sin(np.deg2rad(135))
        ax.add_patch(Circle((tx, ty), 0.5, fill=False, ec=PAL['target'],
                            lw=1.0, zorder=3))
        ax.text(x, 4.05, lab, ha='center', va='top', fontsize=7)
    ax.add_patch(FancyArrowPatch((4.55, 6.3), (5.75, 6.3), arrowstyle='-|>',
                                 mutation_scale=7, color=PAL['fg'], lw=0.9))
    ax.text(5.15, 2.45, 'Target repeat: target at the\nsame location as before',
            ha='center', va='center', fontsize=6.5, color=PAL['target'])
    ax.add_patch(Rectangle((0.9, 0.55), 0.5, 0.13, color=PAL['item']))
    ax.text(1.65, 0.62, 'Target', fontsize=6, color=PAL['fg'], va='center')
    ax.add_patch(Rectangle((5.6, 0.37), 0.13, 0.5, color=PAL['singleton']))
    ax.text(6.0, 0.62, 'Distractor', fontsize=6, color=PAL['fg'],
            va='center')
    panel_letter(ax, 'a', x=0.0, y=0.98)

    # b — behaviour ---------------------------------------------------------
    ax = fig.add_subplot(row1[1])
    order = ['Target', 'Distractor']
    cols = {'Target': PAL['target'], 'Distractor': PAL['dist']}
    sns.stripplot(beh, x='kind', y='d', order=order, ax=ax, size=2,
                  palette=cols, hue='kind', alpha=0.45, jitter=0.2,
                  legend=False, zorder=1)
    for i, k in enumerate(order):
        x = beh.loc[beh['kind'] == k, 'd']
        ci = stats.t.ppf(.975, len(x) - 1) * x.sem()
        ax.plot([i, i], [x.mean() - ci, x.mean() + ci], color=PAL['fg'],
                lw=1.2, zorder=3)
        ax.plot(i, x.mean(), 'D', ms=5, color=cols[k], mec=PAL['fg'],
                mew=0.8, zorder=4)
    ax.axhline(0, color=PAL['fg'], lw=0.6, zorder=0)
    ax.set_xticks([0, 1], ['Target', 'Distractor'])
    ax.set(xlabel='Repeated', ylabel='RT, repeat − switch (ms)',
           ylim=(-90, 40), yticks=[-80, -40, 0, 40])
    panel_letter(ax, 'b', x=-0.55)

    # c — target / distractor gain, repeat − switch -----------------------
    ax = fig.add_subplot(row1[2])
    roi_points(ax, st_T, PAL['target'], offset=-0.17)
    roi_points(ax, st_D, PAL['dist'], offset=0.17, ms=3)
    ax.set(ylabel='Gain, repeat − switch', ylim=(-1.9, 1.0),
           yticks=[-1.5, -1, -0.5, 0, 0.5, 1])
    ax.text(0.03, 0.97, 'Target gain', transform=ax.transAxes,
            color=PAL['target'], fontsize=7, va='top')
    ax.text(0.03, 0.88, 'Distractor gain', transform=ax.transAxes,
            color=PAL['dist'], fontsize=7, va='top')
    ax.text(0.03, 0.04, 'Repeated target: less\nenhancement from V3 up',
            transform=ax.transAxes, color=PAL['target'], fontsize=6.5,
            va='bottom', ha='left')
    ax.tick_params(axis='x', labelrotation=90)
    panel_letter(ax, 'c', x=-0.2)

    # d — cortex map of the target-repeat effect ---------------------------
    ax = fig.add_subplot(row1[3])
    norm_rep = TwoSlopeNorm(vmin=-4, vcenter=0, vmax=4)
    flatmap(ax, flat, st_T.set_index('roi')['t'].to_dict(), REPEAT_CMAP,
            norm_rep)
    colorbar(fig, ax, REPEAT_CMAP, norm_rep, 'Target repeat effect (t)',
             [-4, 0, 4])
    panel_letter(ax, 'd', x=0.02, y=0.98)

    # e — cortex map of the RT link -----------------------------------------
    norm_rt = TwoSlopeNorm(vmin=-7, vcenter=0, vmax=7)
    ax = fig.add_subplot(row2[0])
    flatmap(ax, flat, amp.set_index('roi')['t'].to_dict(), RT_CMAP, norm_rt)
    colorbar(fig, ax, RT_CMAP, norm_rt, 'Response–RT link (t)', [-7, 0, 7])
    panel_letter(ax, 'e', x=0.02, y=0.98)

    # f — RT link per ROI ---------------------------------------------------
    ax = fig.add_subplot(row2[1])

    def rt_color(r):
        if r.q >= 0.05:
            return PAL['muted']
        return PAL['fast'] if r.mean < 0 else PAL['slow']
    roi_points(ax, amp, rt_color, ms=5)
    ax.set(ylabel='RT per SD of\nsingle-trial response (β)', ylim=(-0.07, 0.16),
           yticks=[-0.05, 0, 0.05, 0.1, 0.15])
    ax.set_xticks(range(len(ROI_ORDER)), ROI_ORDER, rotation=0)
    ax.axvspan(-0.5, 2.5, color=PAL['fast'], alpha=0.08, lw=0, zorder=0)
    ax.axvspan(7.5, 10.5, color=PAL['slow'], alpha=0.08, lw=0, zorder=0)
    ax.text(1, 0.155, 'Early visual:\nstronger response,\nfaster RT',
            ha='center', va='top', fontsize=6.5, color=PAL['fast'])
    ax.text(9, -0.025, 'Dorsal:\nstronger response,\nslower RT',
            ha='center', va='top', fontsize=6.5, color=PAL['slow'])
    panel_letter(ax, 'f', x=-0.1)

    # g — cross-ROI correlation of single-trial responses ------------------
    ax = fig.add_subplot(row3[0])
    m = corr.values.copy()
    m[np.triu_indices_from(m)] = np.nan
    im = ax.imshow(m, cmap='mako_r', vmin=0, vmax=0.8)
    ax.set_xticks(range(11), ROI_ORDER, rotation=90)
    ax.set_yticks(range(11), ROI_ORDER)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    for i in range(11):
        for j in range(i):
            ax.text(j, i, f'{m[i, j]:.2f}'[1:], ha='center', va='center',
                    fontsize=4.5,
                    color='white' if m[i, j] > 0.45 else PAL['fg'])
    cax = ax.inset_axes([0.55, 0.88, 0.42, 0.04])
    cb = fig.colorbar(im, cax=cax, orientation='horizontal', ticks=[0, .4, .8])
    cb.outline.set_linewidth(0.5)
    cb.ax.tick_params(labelsize=6, length=2, width=0.5)
    cb.set_label('Single-trial response r', fontsize=6, labelpad=1)
    ax.text(0.55, 0.56, 'Shared most with\nneighbouring areas', fontsize=6.5,
            transform=ax.transAxes, color=PAL['fg'])
    panel_letter(ax, 'g', x=-0.18, y=1.0)

    # h — within-ROI separability of the components ------------------------
    ax = fig.add_subplot(row3[1])
    c = coll.set_index('roi').loc[ROI_ORDER]
    lines = [('r_TG', 'Target – global', PAL['target']),
             ('r_GD', 'Global – distractor', PAL['muted']),
             ('r_TD', 'Target – distractor', PAL['dist'])]
    for col, lab, colr in lines:
        ax.plot(range(11), c[col], 'o-', color=colr, ms=3, lw=1)
    for k, (col, lab, colr) in enumerate(lines):
        ax.text(0.02, 0.36 - 0.09 * k, lab, transform=ax.transAxes,
                color=colr, fontsize=6.5)
    ax.axhspan(0.9, 1.0, color=PAL['muted'], alpha=0.25, lw=0, zorder=0)
    ax.text(3.5, 0.95, 'Components not separable per trial', fontsize=6.5,
            color=PAL['fg'], va='center', ha='center')
    ax.set_xticks(range(11), ROI_ORDER, rotation=90)
    ax.set(ylim=(0, 1.0), xlim=(-0.4, 10.4), yticks=[0, .25, .5, .75, 1],
           ylabel='Within-ROI r between\nsingle-trial gains')
    panel_letter(ax, 'h', x=-0.13, y=1.0)

    for a in fig.axes:
        if a.axison and a.get_label() != '<colorbar>' and not a.images \
                and a.get_xlim()[1] > 1.5:
            sns.despine(ax=a, offset=4)
    fig.savefig(out.with_suffix('.pdf'))
    fig.savefig(out.with_suffix('.svg'))
    print(f'-> {out.with_suffix(".pdf")}')
    print('Target repeat (gain):\n', st_T.round(3).to_string(index=False))
    print('Distractor repeat (gain):\n', st_D.round(3).to_string(index=False))
    for k, d in beh.groupby('kind'):
        t, p = stats.ttest_1samp(d['d'], 0)
        print(f'RT {k} repeat − switch: {d["d"].mean():.1f} ms, t={t:.2f}, p={p:.2g}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('notes/data'))
    parser.add_argument('--out', type=Path,
                        default=Path('notes/figures/trialwise_af_story'))
    args = parser.parse_args()
    main(args.data, args.out)
