"""Figure 6: the bar stimulus can be decoded from V1–V3, and less of it is decoded in the quadrant facing HP.

``collect`` (cluster): concatenates the per-run readouts of ``retsupp/decode/decode_final.py`` into one table. The
other summaries (decoding quality, eccentricity bands, HP-rotated maps, display frames) come from
``retsupp/visualize/paper/fig6_collect.py``.

``plot`` (local): the pre-specified test is per subject and ROI the mean over runs of decoded intensity(HP quadrant) − mean
decoded intensity(three other quadrants); Wilcoxon signed-rank across subjects, one-sided (HP < LP), Holm across V1, V2, V3
(``notes/primary_test_2026-11.md``). The figure shows the unconstrained decoder (every pixel decoded); the decoder with the
bar outline given gives the same result and is left out for now.

``--source`` picks the pRF fits the decoder used (``canonical`` or ``jointhrf``; file names carry it), ``--rois`` the
areas of b and e; c, d and f always pool the early areas (V1–V3), g comes from ``fig_bar_pass_average.py``.

Usage::

    python -m retsupp.visualize.paper.fig6_decoding collect --source jointhrf \\
        --out notes/data/decoding_readout_jointhrf_2026-11.tsv                     # compute node
    python -m retsupp.visualize.paper.fig6_decoding plot                             # Fig. 6 (joint HRF, nine ROIs)
    python -m retsupp.visualize.paper.fig6_decoding plot --source canonical --rois V1 V2 V3 \\
        --name figS_decoding_canonical                                            # robustness: canonical HRF
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage, stats

ROIS = ['V1', 'V2', 'V3']
EARLY = ['V1', 'V2', 'V3']                   # pooled in c, d, f, g
DECODER = 'free'
BAR_PASS = Path('notes/data/bar_pass_average_2026-11')
BAR_WINDOW = (1.6, 6.4)
LOC_COLORS = {'hp': '#D62828', 'adjacent': '#52606D', 'opposite': '#9AA5B1'}
DATA = Path('notes/data')
EXAMPLES = {'canonical': ('sub-07', 'ses-1_run-3', 'V1'),     # the clearest of the top runs by r (inspected visually)
            'jointhrf': ('sub-03', 'ses-1_run-4', 'V1'),
            'dn_roihrf': ('sub-03', 'ses-1_run-4', 'V1')}       # rank 1 for DN too; the same run as Fig. 6a
FRAMES = [15, 21, 28, 72, 81, 89, 130]      # TRs shown in a (bar left → right, top → bottom, then blank)
TR = 1.6
APERTURE, ITEM_ECC, ITEM_R = 3.17, 4.0, .75
BANDS = [('0-1.5', 0, 1.5, '0–1.5°'), ('1.5-2.5', 1.5, 2.5, '1.5–2.5°'), ('2.5-ap', 2.5, APERTURE, '2.5°+')]
ITEM_ANGLES = (90, 180, 270, 0)              # HP first: after rotation HP sits at 12 o'clock
ECC_MAX = 3.45                               # f's x axis ends just past the aperture edge
BAND_SHADE = ['#E8ECF0', '#D5DCE3', '#C2CBD5']
INSET_TXT = """ Right of the map, the same contrast on single bar positions without any averaging over the visual field: the decoded bar (decoder with the bar outline given) minus the mean decoded bar at the same time point over the subject's runs, averaged within each HP location first and then over the four, so the stimulus is identical and only the HP location differs; each run scaled to a common mean intensity on the bar first, rotated so HP is at the upper right in 90° steps (horizontal bars reflected onto vertical ones across the diagonal through HP); left to right, a bar on the far side (centre {ins_far}), through the centre ({ins_mid}) and through the quadrant facing HP (centre {ins_hp}); under them, the difference on the bar's segment in the quadrant next to HP (far side) and in the quadrant facing HP ({ins_seg}); same colour scale as the map, on the bar's footprint (supplementary figure of decoded bars for all positions)."""
BAR_LEGEND = """
**Figure S. The measured response when the bar crosses a pRF near an item location, by where HP is.** Without a model, the measured response
(cleaned BOLD, % signal) of V1–V3 voxels with cross-run reliability > 0 and a pRF centre within 2° of an item location,
locked to the moment the bar centre crosses the pRF centre along the bar's direction of motion (left: sketch, and the
mean response of these voxels, which peaks about 3 s after the crossing). Each voxel's location is classed relative to
the run's HP location as in Fig. 6e: HP (red), next to HP (the two neighbouring LP locations, dark grey) or opposite (light
grey); responses are averaged over passes and voxels per subject, ROIs pooled weighted by voxel count, and shown
relative to the mean of the four locations; lines and bands, mean ± SEM over subjects. HP minus the mean of the three LP
locations in {w0:.1f}–{w1:.1f} s (grey; * one-sided Wilcoxon p < .05): {bar}. Search items are not removed and the
response before the crossing is already rising (the bar enters the pRF earlier), so no pre-crossing baseline is
subtracted; with one (−6.4 to 0 s) the difference is {bar_base}, not significant. {source}
"""
LEGEND = """
{title} The bar stimulus of every run was decoded from the BOLD activity of one ROI at a time with pRFs
({prf_model}, {hrf}, no gain terms) fitted on the other runs (cross-validation folds), so the decoded run
never entered the pRF fit; voxels with cross-run reliability > 0 on the training runs (at most 2000 per ROI); every pixel
of every frame decoded; the encoding model contains only the bar, not the search items. {areas} a, True bar (top) and decoded intensity in a single held-out run (middle;
{ex_sub} {ex_run}, V1, frame-wise r = {ex_r:.2f}: rank {ex_rank} of {n_runs} subject × run × ROI decodes, chosen as the
clearest of the five best; the median run has r = {r_med:.2f}) and averaged over all runs and subjects (V1; bottom), at
seven time points of the run (the last one without a bar); decoded rows on a magma scale from 0 to each row's maximum. Grey circle, the {ap}° aperture of the bar. b, Per subject,
the spatial correlation between decoded and true frame inside the aperture, averaged over the frames with a bar and over
runs (blue dots, n = {n} subjects), against the same with the order of the true frames shuffled (100 permutations; grey).
Points and bars: median and bootstrapped 95% CI. Decoded vs shuffled: Wilcoxon signed-rank test, one-sided (*** p <
.001); {quality}. The correlation peaks at lag 0 in V1–V3, so the decoded time course is not displaced. c, Each
run's visual field rotated so that the HP location (red dashed circle; grey, the LP locations, drawn with the item
footprint, radius {item_r}°) lies at 12 o'clock (90° steps, then 45° with bilinear interpolation); shown is the decoded
intensity on the bar's pixels minus its average over the four rotations (in % of that average), averaged over runs,
V1–V3 and subjects, smoothed (Gaussian, σ = 0.4°). The dotted diagonals bound the quadrant facing HP.{insets} d, Per subject and ROI, the mean decoded intensity
on the bar's pixels in the quadrant facing HP minus the mean over the three other quadrants, averaged over runs; dots are
subjects, points and bars the median and bootstrapped 95% CI. Wilcoxon signed-rank test, one-sided (HP < other
quadrants), Benjamini-Hochberg across the ROIs per decoder; * q < .05. {tests}.{outline} e, Decoded
intensity on the bar's pixels per quadrant of the aperture, in % of the mean of the four quadrants, in the same rotated
frame (icons: HP ringed in red at the top, the quadrant shaded): the quadrant facing HP, the two quadrants next to it
(left and right pooled; they do not differ, two-sided Wilcoxon p = {q_lr:.2f}) and the opposite quadrant;
per subject and ROI the decoded intensity per quadrant averaged over runs and then expressed in % of the mean of the four quadrants, then averaged over V1–V3; dots are subjects, points and bars the mean and within-subject SEM (Cousineau-Morey).
{quad}. f, The difference of d by eccentricity, in %
of the decoded intensity in the other quadrants at the same eccentricity, mean of V1–V3 per subject; line and band, mean ± SEM over
subjects (rings of 0.25°), in the three bands shaded. Items (radius {item_r}°) begin at {item_in:.2f}°
eccentricity, just outside the aperture, and their responses are in the BOLD data but not in the encoding model; if item
responses leaking into the aperture produced the difference, it would grow toward the aperture edge. It does not: per
band (mean ± SEM; one-sided Wilcoxon signed-rank test, uncorrected, * p < .05) {bands}; the three bands do not differ
(Friedman test, p = {friedman:.2f}; outer minus inner band, two-sided Wilcoxon p = {outer_inner:.2f}). Within 0.75° of
the fovea, where the four quadrants meet, there is no difference ({fovea}). {source}
"""


def collect(bids, source, out):
    root = Path(bids) / 'derivatives' / 'decoded_final_2026-11' / source
    files = sorted(root.glob('*/sub-*/readout_*.tsv'))
    d = pd.concat([pd.read_csv(f, sep='\t') for f in files])
    d.to_csv(out, sep='\t', index=False)
    print(f'{len(files)} readouts -> {out}')


def effects(d, measure='drive_bar'):
    """Per subject × ROI × decoder: mean over runs of HP − mean(other quadrants)."""
    g = d.groupby(['subject', 'roi', 'decoder', 'session', 'run'])
    run = g.apply(lambda x: x.loc[x.is_hp, measure].mean() - x.loc[~x.is_hp, measure].mean(), include_groups=False)
    return run.groupby(['subject', 'roi', 'decoder']).mean().rename('diff').reset_index()


def holm(p):
    p = np.asarray(p, float)
    out, run = np.empty_like(p), 0.
    for i, j in enumerate(np.argsort(p)):
        run = max(run, min(1., (len(p) - i) * p[j]))
        out[j] = run
    return out


def test(e, rois=ROIS, decoders=(DECODER,)):
    rows = []
    for dec in decoders:
        for roi in rois:
            v = e[(e.decoder == dec) & (e.roi == roi)]['diff'].dropna().to_numpy()
            if len(v) < 5:
                continue
            w = stats.wilcoxon(v, alternative='less')
            rows.append(dict(decoder=dec, roi=roi, n=len(v), median=np.median(v), pct_below=(v < 0).mean() * 100, p=w.pvalue))
    t = pd.DataFrame(rows)
    for dec in t.decoder.unique():
        k = t.decoder == dec
        t.loc[k, 'p_holm'] = holm(t.loc[k, 'p'])
    return t


def band_effects(bands, decoder=DECODER, rois=EARLY):
    """Per subject × band (mean over ``rois``): HP − other quadrants in % of the other quadrants, per band/ring."""
    b = bands[(bands.decoder == decoder) & bands.roi.isin(rois)]
    hp = b[b.is_hp].groupby(['subject', 'roi', 'band', 'session', 'run']).drive_bar.mean()
    lp = b[~b.is_hp].groupby(['subject', 'roi', 'band', 'session', 'run']).drive_bar.mean()
    r = pd.DataFrame({'diff': hp - lp, 'lp': lp}).groupby(['subject', 'roi', 'band']).mean()
    r['rel'] = 100 * r['diff'] / r['lp']
    return r.groupby(['subject', 'band']).rel.mean().unstack()


def rel_maps(maps, decoder=DECODER, rois=EARLY):
    """Per subject: HP-rotated drive minus its 4-fold rotation average, in % of that average, mean over ``rois``."""
    j = list(maps['decoders']).index(decoder)
    ri = [list(maps['rois']).index(r) for r in rois]
    x = maps['drive'][:, ri, j].astype(float)                        # sub, roi, y, x
    sym = np.mean([np.rot90(x, k, axes=(2, 3)) for k in range(4)], 0)
    ref = np.nanmean(sym, 0, keepdims=True)                          # group average, per ROI and pixel
    return rotate_hp_up(np.nanmean((x - sym) / ref, 1) * 100)


def rotate_hp_up(rm):
    """Rotate maps (HP upper right) by a further 45° counter-clockwise so HP sits at 12 o'clock (bilinear)."""
    c = np.linspace(-5, 5, 50)
    X, Y = np.meshgrid(c, c)
    k = np.sqrt(.5)
    sx, sy = k * X + k * Y, -k * X + k * Y                           # source position R(-45°)(x, y)
    coords = [(sy + 5) / (c[1] - c[0]), (sx + 5) / (c[1] - c[0])]    # (row, col)
    out = []
    for m in rm:
        ok = np.isfinite(m)
        v = ndimage.map_coordinates(np.where(ok, m, 0), coords, order=1, cval=0)
        w = ndimage.map_coordinates(ok.astype(float), coords, order=1, cval=0)
        out.append(np.where(w > .5, v / np.maximum(w, 1e-9), np.nan))
    return np.array(out)


def stars(p):
    return '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else ''


def fmt_p(p):
    return f'p = {p:.2g}' if p >= .001 else f'p = {p:.1e}'.replace('e-0', ' × 10^-').replace('e-', ' × 10^-')


# ----------------------------------------------------------------------------------------------------------- panels
def disc(ax, img, cmap, vmax, vmin=0, r=APERTURE):
    from matplotlib.patches import Circle
    c = np.linspace(-5, 5, 50)
    X, Y = np.meshgrid(c, c)
    img = np.where(np.hypot(X, Y) <= r + .3, img, np.nan)
    h = .5 * (c[1] - c[0])
    im = ax.imshow(img, origin='lower', extent=[-5 - h, 5 + h, -5 - h, 5 + h], cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation='nearest')
    clip = Circle((0, 0), r, transform=ax.transData, fc='none', ec='none')
    ax.add_patch(clip)
    im.set_clip_path(clip)
    ax.add_patch(Circle((0, 0), r, fill=False, ec='.6', lw=.5))
    ax.set_xlim(-3.45, 3.45)
    ax.set_ylim(-3.45, 3.45)
    ax.set_axis_off()
    return im


def panel_frames(fig, box, frames, W, H, style='blue', times=None, labels=None):
    """a: rows true bar, one decoded run, group mean; columns time points (``times``, default FRAMES)."""
    import matplotlib.colors as mcolors
    from retsupp.visualize.paper.paper_style import FG, STIM
    FR = list(times or FRAMES)
    x0, y0, w, h = box
    n = len(FR)
    size = min(w / n * .92, h / 3 * .92)
    gapx = (w - n * size) / (n - 1)
    gapy = (h - 3 * size) / 2
    cm = mcolors.LinearSegmentedColormap.from_list('stim', ['#FFFFFF', STIM])
    labs = labels or ('Bar', f'Best run\n{frames["ex_label"]}', f'Mean of {frames.get("n_group", 30)}\nsubjects (V1)')
    rows = [(labs[0], frames['paradigm'].astype(float), 1.),
            (labs[1], frames['example'], np.nanpercentile(frames['example'], 99.7)),
            (labs[2], frames['group'], np.nanpercentile(frames['group'], 99.7))]
    axes = []
    for i, (lab, arr, vmax) in enumerate(rows):
        for k, t in enumerate(FR):
            ax = fig.add_axes([(x0 + k * (size + gapx)) / W, (y0 + h - (i + 1) * size - i * gapy) / H, size / W, size / H])
            if style == 'magma' and i > 0:               # decoded rows as Fig. 4's model panels: magma on black
                from matplotlib.patches import Circle
                c = np.linspace(-5, 5, 50)
                Xg, Yg = np.meshgrid(c, c)
                img = np.where(np.hypot(Xg, Yg) <= APERTURE + .1, np.clip(arr[t], 0, None), 0)
                ax.imshow(img, origin='lower', extent=[-5.1, 5.1, -5.1, 5.1], cmap='magma', vmin=0, vmax=vmax,
                          interpolation='bilinear')
                ax.add_patch(Circle((0, 0), APERTURE, fill=False, ec='.75', lw=.4))
                ax.set_xlim(-3.6, 3.6)
                ax.set_ylim(-3.6, 3.6)
                ax.set_axis_off()
            else:
                disc(ax, arr[t], cm, vmax)
            if i == 0:
                ax.text(.5, 1.06, f'{t * TR:.0f} s', transform=ax.transAxes, ha='center', va='bottom', fontsize=6.5, color='.35')
            if k == 0:
                ax.text(-.12, .5, lab, transform=ax.transAxes, ha='right', va='center', fontsize=6.5, color=FG)
            axes.append(ax)
    return axes


def subject_dots(ax, x, v, color, seed, s=7, alpha=.4, med=True, open_=False, ms=5.5, width=.13):
    from retsupp.visualize.paper.fig45_common import median_ci
    rng = np.random.default_rng(seed)
    if s:
        ax.scatter(x + rng.uniform(-width, width, len(v)), v, s=s, color=color, alpha=alpha, lw=0, zorder=2)
    if med:
        m, lo, hi = median_ci(v)
        ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], fmt='o', ms=ms, color=color, mec=color if open_ else 'w',
                    mfc='w' if open_ else color, mew=1. if open_ else .7, lw=1.3, capsize=0, zorder=4)


def roi_xs(rois):
    """x positions: 0, 1, 2 for the early areas alone; the shared pathway layout (ROI_X) for more."""
    from retsupp.visualize.paper.paper_style import ROI_X
    return np.arange(len(rois), dtype=float) if len(rois) <= 3 else np.array([ROI_X[r] for r in rois])


def finish_roi_axis(ax, rois):
    from retsupp.visualize.paper.paper_style import roi_axis
    xs = roi_xs(rois)
    if len(rois) <= 3:
        ax.set_xticks(xs)
        ax.set_xticklabels(rois)
        ax.set_xlim(-.5, len(rois) - .5)
    else:
        roi_axis(ax, labels=rois, rotate=False, fontsize=6 if ax.get_position().width * ax.figure.get_figwidth() < 2.3 else 6.5)


def panel_quality(ax, qs, pq, rois=ROIS):
    from retsupp.visualize.paper.paper_style import FG, MUTED, STIM, offset_spines
    many = len(rois) > 3
    for i, (roi, x) in enumerate(zip(rois, roi_xs(rois))):
        g = qs[qs.roi == roi]
        if not len(g):
            continue
        subject_dots(ax, x - .14, g.r.to_numpy(), STIM, i, s=5 if many else 7, width=.1 if many else .13,
                     ms=4.5 if many else 5.5)
        subject_dots(ax, x + .22, g.r_shuffle.to_numpy(), MUTED, 10 + i, s=0, ms=3.5)
        ax.text(x, .7, stars(pq[roi]), ha='center', va='bottom', fontsize=6.5, color=STIM)
    ax.axhline(0, color='.6', lw=.7, ls='--', zorder=0)
    ax.set_ylim(-.06, .7)
    ax.set_yticks([0, .2, .4, .6])
    ax.set_ylabel('Decoded vs true bar (r)')
    xr = roi_xs(rois)[-1] + .45
    ax.text(xr, .2 if not many else .62, 'Held-out run', color=STIM, fontsize=6.5, ha='right', va='center')
    ax.text(xr, .08 if not many else .54, 'Frames shuffled', color='.45', fontsize=6.5, ha='right', va='center')
    offset_spines(ax)
    finish_roi_axis(ax, rois)


def items(ax, hp_color, lw=.8):
    from matplotlib.patches import Circle
    for k, a in enumerate(ITEM_ANGLES):
        p = ITEM_ECC * np.array([np.cos(np.deg2rad(a)), np.sin(np.deg2rad(a))])
        ax.add_patch(Circle(p, ITEM_R, fill=False, ec=hp_color if k == 0 else '.55', lw=lw, ls=(0, (2.5, 1.2)),
                            clip_on=False))


def smoothed_map(rm):
    """Group map of c: each subject's map smoothed inside the aperture (Gaussian, 2 pixels = 0.4°), then averaged."""
    c = np.linspace(-5, 5, 50)
    X, Y = np.meshgrid(c, c)
    inside = np.hypot(X, Y) <= APERTURE
    sm = np.array([ndimage.gaussian_filter(np.where(inside, np.nan_to_num(p), 0), 2) /
                   ndimage.gaussian_filter(inside.astype(float), 2) for p in rm])
    return np.nanmean(sm, 0), inside


def panel_map(ax, rm, vmax=3, label='HP', tag=True, cmap='RdBu_r', diag=('.55', .4, ':'), lp_label=False):
    from retsupp.visualize.paper.paper_style import HP
    im = disc(ax, smoothed_map(rm)[0], cmap, vmax, -vmax)
    for a in (45, 135):
        t = np.deg2rad(a)
        ax.plot([-APERTURE * np.cos(t), APERTURE * np.cos(t)], [-APERTURE * np.sin(t), APERTURE * np.sin(t)],
                color=diag[0], lw=diag[1], ls=diag[2])
    if lp_label:
        ax.text(-ITEM_ECC, -ITEM_R - .25, 'LP', color='.4', fontsize=6, ha='center', va='top')
    items(ax, HP)
    ax.set_xlim(-4.95, 4.95)
    ax.set_ylim(-4.95, 4.95)
    ax.set_aspect('equal')
    if label:
        ax.text(ITEM_R + .25, ITEM_ECC, label, color=HP, fontsize=6.5, ha='left', va='center')
    if tag:
        area_tag(ax, x=0., y=1.)
    return im


def area_tag(ax, text='V1–V3 pooled', x=1., y=1., ha='left'):
    ax.text(x, y, text, transform=ax.transAxes, fontsize=6, color='.4', ha=ha, va='top',
            bbox=dict(boxstyle='round,pad=.25', fc='#F1F3F5', ec='none'))


def panel_angle(ax, rm):
    """d: drive by direction from fixation, clockwise from HP (12 o'clock); the four item directions drawn as icons."""
    import matplotlib.transforms as mtrans
    from matplotlib.patches import Circle, Wedge
    from retsupp.visualize.paper.paper_style import FG, HP, MUTED, offset_spines
    c = np.linspace(-5, 5, 50)
    X, Y = np.meshgrid(c, c)
    inside = np.hypot(X, Y) <= APERTURE
    ang = (90 - np.degrees(np.arctan2(Y, X)) + 180) % 360 - 180      # clockwise from 12 o'clock
    edges = np.arange(-180, 181, 22.5)
    prof = np.array([[np.nanmean(p[inside & (ang >= a) & (ang < b)]) for a, b in zip(edges[:-1], edges[1:])] for p in rm])
    x = (edges[:-1] + edges[1:]) / 2
    x = np.r_[x[-1] - 360, x, x[0] + 360]
    prof = np.c_[prof[:, -1:], prof, prof[:, :1]]
    m, se = prof.mean(0), prof.std(0, ddof=1) / np.sqrt(len(prof))
    ax.axvspan(-45, 45, color=HP, alpha=.1, lw=0, zorder=0)
    ax.fill_between(x, m - se, m + se, color=FG, alpha=.18, lw=0)
    ax.plot(x, m, color=FG, lw=1.2)
    ax.axhline(0, color='.6', lw=.7, ls='--', zorder=0)
    ax.set_xlim(-180, 180)
    ax.set_xticks([-180, -90, 0, 90, 180])
    ax.set_xticklabels([])
    ax.set_ylim(-2.6, 2.2)
    ax.set_yticks([-2, 0, 2])
    ax.set_yticklabels(['−2', '0', '2'])
    ax.set_ylabel('Decoded intensity\nvs average (%)')
    ax.text(0, 2.15, 'Quadrant\nfacing HP', color=HP, fontsize=6.5, ha='center', va='top')
    # Item icons under the ticks: HP red at 0, the LP items grey.
    tr = mtrans.blended_transform_factory(ax.transData, ax.transAxes)
    for xx, lab, col in [(-180, 'Below', MUTED), (-90, 'Left', MUTED), (0, 'HP', HP), (90, 'Right', MUTED),
                         (180, 'Below', MUTED)]:
        ax.plot(xx, -.15, 'o', transform=tr, ms=5, mfc='w' if col == MUTED else '#F6D4D4', mec=col, mew=.9, clip_on=False)
        ax.text(xx, -.27, lab, transform=tr, ha='center', va='top', fontsize=6.5, color=HP if col == HP else '.4')
    ax.text(.5, -.44, 'Direction from fixation, relative to HP', transform=ax.transAxes, ha='center', va='top',
            fontsize=8, color=FG)
    # Mini display: HP on top, the shaded quadrant facing it.
    ins = ax.inset_axes([1.0, .55, .3, .45])
    ins.add_patch(Wedge((0, 0), APERTURE, 45, 135, facecolor=HP, alpha=.15, lw=0))
    ins.add_patch(Circle((0, 0), APERTURE, fill=False, ec='.55', lw=.5))
    for k, a in enumerate(ITEM_ANGLES):
        p = ITEM_ECC * np.array([np.cos(np.deg2rad(a)), np.sin(np.deg2rad(a))])
        ins.add_patch(Circle(p, ITEM_R, facecolor='#F6D4D4' if k == 0 else 'w', ec=HP if k == 0 else MUTED, lw=.6))
    ins.plot(0, 0, '+', color=FG, ms=3, mew=.5)
    ins.set_xlim(-5, 5)
    ins.set_ylim(-5, 5)
    ins.set_aspect('equal')
    ins.set_axis_off()
    area_tag(ax, x=1.02, y=.4)
    offset_spines(ax)


QUAD_ANGLE = {'upper_right': 45, 'upper_left': 135, 'lower_left': 225, 'lower_right': 315}
QUAD_CLASSES = [('hp', 'HP'), ('adjacent', 'Next to HP'), ('opposite', 'Opposite')]


def quadrant_effects(d, decoder=DECODER, rois=EARLY):
    """Per subject: decoded intensity per quadrant (by position relative to HP) in % of the mean of the four, mean of
    runs, then of ``rois``. Columns hp, left, right, opposite and adjacent (= mean of left and right)."""
    d = d[(d.decoder == decoder) & d.roi.isin(rois)].copy()
    rel = (d.quadrant.map(QUAD_ANGLE) - d.hp_location.map(QUAD_ANGLE)) % 360
    d['pos'] = rel.map({0: 'hp', 90: 'left', 180: 'opposite', 270: 'right'})
    r = d.groupby(['subject', 'roi', 'pos']).drive_bar.mean().unstack()
    r = r.div(r.mean(1), axis=0) * 100 - 100
    s = r.groupby('subject').mean()
    s['adjacent'] = (s.left + s.right) / 2
    return s


def quadrant_stats(s, sem='within'):
    out = dict(p_lr=stats.wilcoxon(s.left - s.right).pvalue,
               p_adj=stats.wilcoxon(s.hp - s.adjacent, alternative='less').pvalue,
               p_opp=stats.wilcoxon(s.hp - s.opposite, alternative='less').pvalue,
               p_fried=stats.friedmanchisquare(s.hp, s.left, s.right, s.opposite).pvalue)
    m, se = s.mean(), s.sem()
    out['text'] = (f'HP {m.hp:+.2f} ± {se.hp:.2f}%, next to HP {m.adjacent:+.2f} ± {se.adjacent:.2f}%, opposite '
                   f'{m.opposite:+.2f} ± {se.opposite:.2f}% (mean ± SEM); HP below next to HP, one-sided Wilcoxon '
                   f'p = {out["p_adj"]:.3f}{" (" + stars(out["p_adj"]) + ")" if out["p_adj"] < .05 else ""}; HP below opposite, p = {out["p_opp"]:.2f}; Friedman test across the four '
                   f'quadrants p = {out["p_fried"]:.2f}')
    return out


def panel_quadrants(ax, s, st, pred=None):
    """d: decoded intensity per quadrant relative to the mean of the four, in g's HP / next / opposite order and colours."""
    import matplotlib.transforms as mtrans
    from matplotlib.patches import Circle, Wedge
    from retsupp.visualize.paper.paper_style import FG, HP, offset_spines
    cls = [c for c, _ in QUAD_CLASSES]
    v = s[cls]
    norm = v.sub(v.mean(1), axis=0).add(v.values.mean(), axis=0)
    k = len(cls)
    rng = np.random.default_rng(3)
    for i, c in enumerate(cls):
        col = LOC_COLORS[c]
        ax.scatter(i + rng.uniform(-.13, .13, len(v)), v[c], s=6, color=col, alpha=.4, lw=0, zorder=2)
        se = norm[c].std(ddof=1) / np.sqrt(len(v)) * np.sqrt(k / (k - 1))
        ax.errorbar(i + .27, v[c].mean(), yerr=se, fmt='o', ms=5, color=col, mec='w', mew=.7, lw=1.4, capsize=0, zorder=4)
    if pred is not None:   # gain-field model: line + band (mean ± SEM over subjects) across each condition
        for i, c in enumerate(cls):
            pm, pse = pred[c].mean(), pred[c].sem()
            ax.fill_between([i - .2, i + .45], pm - pse, pm + pse, color=FG, alpha=.15, lw=0, zorder=1)
            ax.plot([i - .2, i + .45], [pm, pm], color=FG, lw=1.1, zorder=3)
        ax.plot([1.2, 1.5], [-5.2, -5.2], color=FG, lw=1.1)
        ax.add_patch(plt_rect((1.2, -5.55), .3, .7, ax.transData, FG))
        ax.text(1.55, -5.2, 'Gain-field model', fontsize=6, color=FG, ha='left', va='center')
    lo, hi = -6, 5
    ax.set_ylim(lo, hi)
    if st['p_adj'] < .05:
        y = 3.9
        ax.plot([.27, .27, 1.27, 1.27], [y - .3, y, y, y - .3], color=FG, lw=.7)
        ax.text(.77, y + .05, stars(st['p_adj']), ha='center', va='bottom', fontsize=7, color=FG)
    ax.axhline(0, color='.6', lw=.7, ls='--', zorder=0)
    ax.set_xlim(-.45, 2.55)
    ax.set_xticks(np.arange(3) + .1)
    ax.set_xticklabels([])
    ax.tick_params(axis='x', length=0)
    ax.set_yticks([-5, 0, 5])
    ax.set_yticklabels(['−5', '0', '5'])
    ax.set_ylabel('Decoded intensity vs\nmean of quadrants (%)')
    # Icons under each condition: HP ringed at the top, the quadrant(s) shaded.
    tr = mtrans.blended_transform_factory(ax.transData, ax.transAxes)
    wedges = {'hp': [(45, 135)], 'adjacent': [(135, 225), (-45, 45)], 'opposite': [(225, 315)]}
    bb = ax.get_window_extent() if ax.figure else None
    for i, (c, lab) in enumerate(QUAD_CLASSES):
        ins = ax.inset_axes([i + .1 - .36, -.56, .72, .5], transform=tr)
        for a0, a1 in wedges[c]:
            ins.add_patch(Wedge((0, 0), APERTURE, a0, a1, facecolor=LOC_COLORS[c], alpha=.35, lw=0))
        ins.add_patch(Circle((0, 0), APERTURE, fill=False, ec='.55', lw=.5))
        for j, a in enumerate(ITEM_ANGLES):
            q = ITEM_ECC * np.array([np.cos(np.deg2rad(a)), np.sin(np.deg2rad(a))])
            ins.add_patch(Circle(q, ITEM_R, facecolor='w', ec=HP if j == 0 else '.6', lw=.9 if j == 0 else .5))
        ins.set_xlim(-5, 5)
        ins.set_ylim(-5, 5)
        ins.set_aspect('equal')
        ins.set_axis_off()
        ax.text(i + .1, -.6, lab, transform=tr, ha='center', va='top', fontsize=6,
                color=HP if c == 'hp' else '.3')
    offset_spines(ax)


def panel_test(ax, e, t, rois=ROIS, decoder=DECODER, t_o=None):
    """d: HP − other quadrants per ROI; stars from Benjamini-Hochberg across the ROIs (per decoder). Open markers and
    the upper star row: the decoder with the bar outline given."""
    from retsupp.visualize.paper.paper_style import HP, offset_spines
    vals = e[(e.decoder == decoder) & e.roi.isin(rois)]['diff']
    lo, hi = (-.0132, .008) if len(rois) <= 3 else tuple(np.r_[-1.08, 1.08] * np.nanpercentile(np.abs(vals), 98.5))
    both = t_o is not None
    for i, (roi, x) in enumerate(zip(rois, roi_xs(rois))):
        v = e[(e.decoder == decoder) & (e.roi == roi)]['diff'].dropna().to_numpy()
        if not len(v):
            continue
        xf = x - (.13 if both else 0)
        subject_dots(ax, xf, v, HP, i, s=5 if len(rois) > 3 else 7, width=.1 if len(rois) > 3 else .13,
                     ms=4.5 if len(rois) > 3 else 5.5)
        for sgn, lim in ((-1, lo), (1, hi)):
            out = (v < lo) if sgn < 0 else (v > hi)
            if out.any():
                ax.scatter(np.full(out.sum(), xf), np.full(out.sum(), lim), s=9, marker='v' if sgn < 0 else '^',
                           color=HP, alpha=.6, lw=0, clip_on=False)
        if roi in set(t.roi):
            ax.text(x, hi, stars(t[t.roi == roi].q.iloc[0]), ha='center', va='bottom', fontsize=7, color=HP)
        if both:
            vo = e[(e.decoder == 'outline') & (e.roi == roi)]['diff'].dropna().to_numpy()
            if len(vo):
                subject_dots(ax, x + .2, vo, HP, i, s=0, open_=True, ms=4)
            if roi in set(t_o.roi):
                ax.text(x, hi + .11 * (hi - lo), stars(t_o[t_o.roi == roi].q.iloc[0]), ha='center', va='bottom',
                        fontsize=7, color=HP, clip_on=False)
    ax.axhline(0, color='.6', lw=.7, ls='--', zorder=0)
    ax.set_ylim(lo, hi)
    ax.set_ylabel('Decoded intensity,\nHP − other quadrants' if len(rois) <= 3 else 'HP − other\nquadrants')
    offset_spines(ax)
    if both:                                        # star-row key: filled = unconstrained, open = bar outline given
        x0 = roi_xs(rois)[-1] + .75
        for yy, mfc, lab in ((hi + .035 * (hi - lo), HP, 'Unconstrained'), (hi + .145 * (hi - lo), 'w', 'Outline given')):
            ax.plot(x0, yy, 'o', ms=3.5, color=HP, mfc=mfc, mec=HP, mew=.9, clip_on=False)
            ax.text(x0 + .35, yy, lab, ha='left', va='center', fontsize=6.5, color=HP, clip_on=False)
    if len(rois) > 3:
        ax.set_yticks([-.005, 0, .005])
        ax.set_yticklabels(['−.005', '0', '.005'])
    else:
        ax.set_yticks([-.01, -.005, 0, .005])
        ax.set_yticklabels(['−.010', '−.005', '0', '.005'])
    finish_roi_axis(ax, rois)


def test_bh(e, rois, decoder=DECODER):
    """One-sided Wilcoxon (HP < other quadrants) per ROI, Benjamini-Hochberg across the ROIs."""
    rows = []
    for roi in rois:
        v = e[(e.decoder == decoder) & (e.roi == roi)]['diff'].dropna().to_numpy()
        if len(v) < 5:
            continue
        rows.append(dict(roi=roi, n=len(v), median=np.median(v), pct_below=(v < 0).mean() * 100,
                         p=stats.wilcoxon(v, alternative='less').pvalue))
    t = pd.DataFrame(rows)
    t['q'] = stats.false_discovery_control(t.p)
    return t


def test_exploratory(e, rois, decoder=DECODER):
    """One-sided Wilcoxon (HP < other quadrants) per higher ROI, Benjamini-Hochberg across them."""
    rows = []
    for roi in rois:
        if roi in EARLY:
            continue
        v = e[(e.decoder == decoder) & (e.roi == roi)]['diff'].dropna().to_numpy()
        if len(v) < 5:
            continue
        rows.append(dict(roi=roi, n=len(v), median=np.median(v), pct_below=(v < 0).mean() * 100,
                         p=stats.wilcoxon(v, alternative='less').pvalue))
    t = pd.DataFrame(rows)
    if len(t):
        t['p_bh'] = stats.false_discovery_control(t.p)
    return t


def panel_bands_sketch(ax):
    from matplotlib.patches import Circle, Wedge
    from retsupp.visualize.paper.paper_style import FG, HP
    for (key, a, b, lab), col in zip(BANDS, BAND_SHADE):
        ax.add_patch(Wedge((0, 0), b, 0, 360, width=b - a, facecolor=col, ec='w', lw=.6))
    ax.add_patch(Circle((0, 0), APERTURE, fill=False, ec='.45', lw=.6))
    for k, a in enumerate(ITEM_ANGLES):
        p = ITEM_ECC * np.array([np.cos(np.deg2rad(a)), np.sin(np.deg2rad(a))])
        ax.add_patch(Circle(p, ITEM_R, facecolor='.88' if k else '#F6D4D4', ec=HP if k == 0 else '.55', lw=.6))
    ax.plot(0, 0, '+', color=FG, ms=3, mew=.5)
    ax.set_xlim(-4.9, 4.9)
    ax.set_ylim(-4.9, 4.9)
    ax.set_aspect('equal')
    ax.set_axis_off()
    ax.text(ITEM_R + .25, ITEM_ECC, 'HP item', color=HP, fontsize=6.5, ha='left', va='center')


def panel_ecc(ax, be, pb, pred=None, compact=False):
    from retsupp.visualize.paper.paper_style import FG, HP, offset_spines
    rings = sorted(c for c in be.columns if c.startswith('ring'))
    x = np.array([float(r.split('_')[1]) for r in rings]) + .125
    x[-1] = (3.0 + APERTURE) / 2
    v = be[rings].to_numpy()
    m, se = v.mean(0), v.std(0, ddof=1) / np.sqrt(len(v))
    lo, hi = -4, 2.2
    for (key, a, b, lab), col in zip(BANDS, BAND_SHADE):
        ax.axvspan(a, b, color=col, lw=0, zorder=0)
        ax.text((a + b) / 2, hi, lab, ha='center', va='top', fontsize=6.5, color='.35')
        ax.text((a + b) / 2, hi - .62, stars(pb[key]), ha='center', va='top', fontsize=7, color=HP)
    # Beyond the aperture edge there is no bar to decode; the items start just outside it.
    ax.axvspan(APERTURE, ECC_MAX, facecolor='w', edgecolor='.6', hatch='////', lw=0, zorder=0)
    ax.text((APERTURE + ECC_MAX) / 2, (lo + hi) / 2, 'Aperture edge' if compact else 'Aperture edge (items beyond, at 4°)',
            rotation=90, ha='center',
            va='center', fontsize=6, color='.35', bbox=dict(boxstyle='square,pad=.15', fc='w', ec='none'))
    if pred is None:
        ax.fill_between(x, m - se, m + se, color=HP, alpha=.2, lw=0)
        ax.plot(x, m, color=HP, lw=1.3)
    else:   # data as points + SEM, the gain-field model as line + band
        pv = pred[rings].to_numpy()
        pm, pse = pv.mean(0), pv.std(0, ddof=1) / np.sqrt(len(pv))
        ax.fill_between(x, pm - pse, pm + pse, color=FG, alpha=.15, lw=0, zorder=1)
        ax.plot(x, pm, color=FG, lw=1.1, zorder=2)
        ax.errorbar(x, m, yerr=se, fmt='o', ms=3, color=HP, mec='w', mew=.5, lw=.9, capsize=0, zorder=3)
        if compact == 2:   # glyph key in the free strip at the bottom left
            ax.add_patch(plt_rect((.08, -3.75), .3, .35, ax.transData, FG))
            ax.plot([.08, .38], [-3.575, -3.575], color=FG, lw=1.1)
            ax.text(.45, -3.575, 'Gain-field model', fontsize=6.5, color=FG, ha='left', va='center')
        else:
            ax.text(.05, -2.7, 'Gain-field model', fontsize=6.5, color=FG, ha='left', va='top')
    ax.axhline(0, color='.6', lw=.7, ls='--', zorder=0)
    ax.set_xlim(0, ECC_MAX)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_ylim(lo, hi)
    ax.set_yticks([-4, -2, 0, 2])
    ax.set_yticklabels(['−4', '−2', '0', '2'])
    ax.set_xlabel('Eccentricity (°)')
    ax.set_ylabel('HP − other\nquadrants (%)')
    if compact == 2:
        ax.annotate('Not larger near items', xy=(x[-2], m[-2] + se[-2]), xytext=(1.25, .75), fontsize=6.5,
                    color=HP, ha='left', va='center',
                    arrowprops=dict(arrowstyle='-|>', color=HP, lw=1., mutation_scale=7, shrinkA=1, shrinkB=4,
                                    relpos=(.88, 0.)))
    elif compact:   # narrow e: label in the clear space above the curve, straight connector down to the outer ring
        ax.annotate('Not larger near the items', xy=(x[-2], m[-2] + se[-2]), xytext=(1.3, .55), fontsize=6.5,
                    color=HP, ha='left', va='center',
                    arrowprops=dict(arrowstyle='-|>', color=HP, lw=1., mutation_scale=7, shrinkA=2, shrinkB=4,
                                    relpos=(1., .5)))
    else:
        ax.annotate('Not larger near the items', xy=(2.9, m[-2]), xytext=(1.0, -3.4), fontsize=6.5, color=HP,
                    ha='left', va='center',
                    arrowprops=dict(arrowstyle='-|>', color=HP, lw=1., mutation_scale=7, shrinkA=2, shrinkB=5,
                                    connectionstyle='angle3,angleA=0,angleB=70', relpos=(1., .5)))
    offset_spines(ax)


def load_bar_pass(folder=BAR_PASS, rois=EARLY):
    """Bar-pass-locked responses per subject × location class × time, pooled over ``rois`` (weighted by voxel count)."""
    d = pd.concat([pd.read_csv(f, sep='\t') for f in sorted(Path(folder).glob('sub-*_2026-11.tsv'))])
    d = d[d.roi.isin(rois)]
    p = d.groupby(['subject', 'cls', 't']).apply(lambda x: np.average(x.data, weights=x.n_voxels), include_groups=False)
    w = p.rename('y').reset_index().pivot_table(index=['subject', 't'], columns='cls', values='y').reset_index()
    w['lp'] = (2 * w.adjacent + w.opposite) / 3        # the three LP locations: two neighbours, one opposite
    return w


def bar_pass_stats(w):
    """HP vs the three LP locations in BAR_WINDOW: no baseline (cleaned BOLD is centred per run) and, as a check, with
    the pre-crossing baseline (−6.4 to 0 s, which already contains the rising response)."""
    win = w[(w.t >= BAR_WINDOW[0]) & (w.t <= BAR_WINDOW[1])].groupby('subject').mean()
    base = w[w.t <= 0].groupby('subject').mean()
    out = {}
    for name, x in (('raw', win), ('base', win - base)):
        dd = x.hp - x.lp
        out[name] = dict(diff=dd.mean(), rel=(dd / x.lp).mean() * 100, below=(dd < 0).mean() * 100,
                         p=stats.wilcoxon(dd, alternative='less').pvalue, n=len(dd))
        for o in ('adjacent', 'opposite'):
            out[name][f'p_{o}'] = stats.wilcoxon(x.hp - x[o], alternative='less').pvalue
            out[name][f'rel_{o}'] = ((x.hp - x[o]) / x[o]).mean() * 100
    return out


PRF_TEAL = '#2A9D8F'


def bar_pass_icon(fig, x0, y0, w, mean_tc):
    """The display (HP on top) with a pRF near the HP item and the bar sweeping across it, and below it the mean
    measured response (all locations) with the moment the bar crosses the pRF centre at 0."""
    from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
    from retsupp.visualize.paper.paper_style import FG, HP, MUTED, STIM
    W, H = fig.get_size_inches()
    ax = fig.add_axes([x0 / W, (y0 + .42) / H, w / W, w / H])
    ax.add_patch(Circle((0, 0), APERTURE, fill=False, ec=MUTED, lw=.5))
    for k, a in enumerate(ITEM_ANGLES):
        q = ITEM_ECC * np.array([np.cos(np.deg2rad(a)), np.sin(np.deg2rad(a))])
        ax.add_patch(Circle(q, ITEM_R, facecolor='w', ec=HP if k == 0 else '.6', lw=.9 if k == 0 else .5, zorder=3))
    prf = np.array([.2, 2.6])
    ax.add_patch(Circle(prf, 1.0, facecolor=PRF_TEAL, ec='none', alpha=.45, zorder=4))
    ax.plot(*prf, 'o', ms=1.6, color=PRF_TEAL, zorder=5)
    bx, bw = -.25, .8                                  # vertical bar, moving right, about to cross the pRF centre
    hh = np.sqrt(APERTURE ** 2 - (bx + bw / 2) ** 2)
    ax.add_patch(Rectangle((bx - bw / 2, -hh), bw, 2 * hh, facecolor=STIM, alpha=.55, lw=0, zorder=2))
    ax.add_patch(FancyArrowPatch((bx + .5, -1.2), (bx + 2.3, -1.2), arrowstyle='-|>', mutation_scale=6, color=STIM,
                                 lw=.8, zorder=6))
    ax.plot(0, 0, '+', color=FG, ms=3, mew=.5, zorder=6)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect('equal')
    ax.set_axis_off()
    ax.text(prf[0] + 1.25, prf[1] + .1, 'pRF', fontsize=5.5, color=PRF_TEAL, ha='left', va='center', zorder=7)
    # The mean response, small, to show its shape.
    a2 = fig.add_axes([(x0 + .08) / W, y0 / H, (w - .1) / W, .3 / H])
    m = mean_tc[(mean_tc.index >= -4.8) & (mean_tc.index <= 14.4)]
    a2.plot(m.index, m.values, color=PRF_TEAL, lw=1.1)
    a2.axvline(0, color=STIM, lw=.8)
    a2.text(0, 1.04, 'Bar crosses pRF', transform=a2.get_xaxis_transform(), fontsize=5.5, color=STIM, ha='center',
            va='bottom')
    a2.set_xlim(-4.8, 14.4)
    a2.set_ylim(m.min() - .2, m.max() + .3)
    a2.set_xticks([0, 10])
    a2.set_xticklabels(['0', '10 s'], fontsize=5.5)
    a2.set_yticks([])
    a2.tick_params(axis='x', length=2, pad=1)
    for sp in ('left', 'top', 'right'):
        a2.spines[sp].set_visible(False)
    a2.spines['bottom'].set_linewidth(.5)
    return ax, a2


def panel_bar_pass(ax, w, st):
    """g: response at each item location relative to the mean of the four locations, time-locked to the bar crossing the
    pRF; HP, next to HP and opposite in d's order and colours."""
    from retsupp.visualize.paper.paper_style import FG, HP, offset_spines
    w = w[(w.t >= -4.8) & (w.t <= 14.4)].copy()
    m4 = (w.hp + 2 * w.adjacent + w.opposite) / 4
    ax.axvspan(*BAR_WINDOW, color='.92', lw=0, zorder=0)
    for c, lab in QUAD_CLASSES[::-1]:
        dd = (w[c] - m4).groupby(w.t)
        m, se = dd.mean(), dd.std(ddof=1) / np.sqrt(w.subject.nunique())
        ax.fill_between(m.index, m - se, m + se, color=LOC_COLORS[c], alpha=.18, lw=0)
        ax.plot(m.index, m.values, color=LOC_COLORS[c], lw=1.4 if c == 'hp' else 1.1)
    ax.axhline(0, color='.6', lw=.7, ls='--', zorder=0)
    ax.axvline(0, color='.75', lw=.6, zorder=0)
    ax.text(np.mean(BAR_WINDOW), .062, stars(st['raw']['p']), ha='center', va='bottom', fontsize=7, color=HP)
    ax.set_xlim(-4.8, 14.4)
    ax.set_xticks([0, 5, 10])
    ax.set_ylim(-.095, .11)
    ax.set_yticks([-.05, 0, .05])
    ax.set_yticklabels(['−.05', '0', '.05'])
    ax.set_ylabel('Δ response vs mean of\n4 locations (% signal)')
    ax.set_title('V1–V3: each location minus\nthe mean of the four', fontsize=6.5, color=FG, pad=4)
    ax.set_xlabel('Time from crossing (s)')
    for k, (c, lab) in enumerate(QUAD_CLASSES):
        y = .97 - k * .1
        ax.plot([.6, .67], [y, y], transform=ax.transAxes, color=LOC_COLORS[c], lw=1.4, clip_on=False)
        ax.text(.69, y, lab, transform=ax.transAxes, color=LOC_COLORS[c] if c != 'opposite' else '.5', fontsize=6,
                va='center')
    offset_spines(ax)


def bar_insets(fig, source, xy, W, H, size=.7, gap=.1, vmax=3., bins=(2, 3, 4), fs=6):
    """Three decoded-bar difference images beside c (decoder with the bar outline given), as a sweep: a bar on the far
    side, through the centre and through the quadrant facing HP; under each its label and, where significant, the
    segment difference. Returns their centre labels and segment statistics."""
    from scipy import ndimage
    from retsupp.visualize.paper.figS_decoded_bars import (_disc, _smooth_masked, data_name, pooled_condition,
                                                           segment_stats)
    z = np.load(DATA / data_name(source, 'outline').replace('decoded_bars_', 'decoded_bars_condition_'))
    P = pooled_condition(z['mean'])
    grid = z['grid']
    centres = (z['edges'][:-1] + z['edges'][1:]) / 2
    c = np.linspace(-5, 5, 50)
    X, Y = np.meshgrid(c, c)
    inside = np.hypot(X, Y) <= APERTURE
    labs, segs = [], []
    names = ('Far side', 'Centre', 'Through HP\nquadrant')
    keyseg = ('Next to HP (upper left)', None, 'HP')
    for k, (b, lab, ks) in enumerate(zip(bins, names, keyseg)):
        bar = P[:, b, 2].mean(0) > .5
        foot = ndimage.binary_dilation(bar, iterations=1) & inside
        rel = np.mean([100 * P[s_, b, 1] / P[s_, b, 0][bar & inside].mean() for s_ in range(len(P))], 0)
        ax = fig.add_axes([(xy[0] + k * (size + gap)) / W, xy[1] / H, size / W, size / H])
        _disc(ax, _smooth_masked(rel, foot), 'RdBu_r', -vmax, vmax, outline=bar & inside)
        st = segment_stats(P, grid, b)
        txt = lab
        if ks and ks in st:
            m, se, p, n = st[ks]
            star = '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else ''
            txt += f'\n{m:+.1f}%{star}'.replace('-', '−')
            segs.append((ks, m, se, p))
        ax.text(.5, -.02, txt, transform=ax.transAxes, ha='center', va='top', fontsize=fs, color='.3', linespacing=1.05)
        labs.append(f'{centres[b]:+.1f}°'.replace('-', '−').replace('+0.0', '0'))
    return labs, segs


SWEEP_TXT = """ Right of the map, the same contrast on single bar positions without any averaging over the visual field, as the bar sweeps across the aperture: the decoded bar (decoder with the bar outline given) minus the mean decoded bar at the same time point over the subject's runs, averaged within each HP location first and then over the four, so the stimulus is identical and only the HP location differs; each run scaled to a common mean intensity on the bar first, rotated so HP is at the upper right in 90° steps, horizontal bars reflected onto vertical ones across the diagonal through HP. Top, the difference image at each of the {nb} bar positions (bar centre on the axis below; negative, the side away from HP), smoothed (Gaussian, σ = 0.2°), same colour scale as the map, on the bar's footprint. Bottom, the difference on the bar's segment in the quadrant facing HP (red; only bars that reach it) and in the two quadrants next to HP (grey; mean of the segments present), in % of the mean decoded intensity on the bar; mean ± SEM over subjects; stars, two-sided Wilcoxon signed-rank test per position, uncorrected (* p < .05, ** p < .01). {sweep_stats}."""


def plt_rect(xy, w, h, transform, color, alpha=.15):
    from matplotlib.patches import Rectangle
    return Rectangle(xy, w, h, transform=transform, facecolor=color, alpha=alpha, lw=0, clip_on=False)


def sweep_segments(P, grid, opposite=False):
    """Per subject × bar position (P from ``pooled_condition``): the decoded − mean-over-HP difference on the bar's segment
    in the quadrant facing HP and in the quadrants next to it (mean of those present), % of the bar's mean."""
    from retsupp.visualize.paper.figS_decoded_bars import QUAD
    X, Y = grid[:, 0].reshape(50, 50), grid[:, 1].reshape(50, 50)
    inside = np.hypot(X, Y) <= APERTURE
    ns, nb = P.shape[0], P.shape[1]
    hp_v, nx_v, op_v = np.full((ns, nb), np.nan), np.full((ns, nb), np.nan), np.full((ns, nb), np.nan)
    for b in range(nb):
        bar = (P[:, b, 2].mean(0) > .5) & inside
        bm = np.array([P[s, b, 0][bar].mean() for s in range(ns)])
        seg = {}
        for qn, (sx, sy) in QUAD.items():
            mk = bar & (np.sign(X) == sx) & (np.sign(Y) == sy)
            if mk.sum() >= 10:
                seg[qn] = np.array([P[s, b, 1][mk].mean() for s in range(ns)]) / bm * 100
        if 'HP' in seg:
            hp_v[:, b] = seg['HP']
        nk = [k for k in seg if k.startswith('Next')]
        if nk:
            nx_v[:, b] = np.mean([seg[k] for k in nk], 0)
        if 'Opposite' in seg:
            op_v[:, b] = seg['Opposite']
    return (hp_v, nx_v, op_v) if opposite else (hp_v, nx_v)


def load_prediction(prefix, decoder=DECODER, rois=EARLY):
    """The gain-field model's prediction through the Fig. 6 readouts (``simulate_decode.py --mode fitted``, collected by
    ``collect_simulate_decode.py --fitted``): per subject, readout(model) − readout(control), the paired run with the
    gains at 0 and the same noise. Returns quadrant effects (as ``quadrant_effects``), band effects (as
    ``band_effects``) and the bar-sweep segment differences (hp, next; subject × bar position)."""
    from retsupp.visualize.paper.figS_decoded_bars import pooled_condition
    d = pd.read_csv(DATA / f'{prefix}_readout_2026-11.tsv', sep='\t')
    q = {v: quadrant_effects(g, decoder, rois) for v, g in d.groupby('variant')}
    b = pd.read_csv(DATA / f'{prefix}_bands_2026-11.tsv.gz', sep='\t')
    be = {v: band_effects(g, decoder, rois) for v, g in b.groupby('variant')}
    sw = {}
    for v in ('model', 'control'):
        z = np.load(DATA / f'decoded_bars_condition_{prefix}_{v}_outline_2026-11.npz')
        hp_v, nx_v, op_v = sweep_segments(pooled_condition(z['mean']), z['grid'], opposite=True)
        sw[v] = dict(hp=hp_v, next=nx_v, opp=op_v, subjects=list(np.asarray(z['subjects'])))
    subs = sorted(set(sw['model']['subjects']) & set(sw['control']['subjects']))
    ix = {v: [sw[v]['subjects'].index(x) for x in subs] for v in sw}
    sweep = {k: sw['model'][k][ix['model']] - sw['control'][k][ix['control']] for k in ('hp', 'next', 'opp')}
    sweep['subjects'] = np.array(subs)
    qd = (q['model'] - q['control']).dropna()
    bd = (be['model'] - be['control']).dropna()
    r = d[(d.decoder == decoder) & d.roi.isin(rois)].drop_duplicates(['subject', 'session', 'run', 'roi', 'variant'])
    qual = pd.read_csv(DATA / f'{prefix}_quality_2026-11.tsv', sep='\t')
    qual = qual[qual.roi.isin(rois)].groupby(['variant', 'subject']).r.mean().groupby('variant').median()
    extra = dict(coverage=float((r.n_voxels / r.n_decoder_voxels).median() * 100), r_model=float(qual['model']),
                 r_control=float(qual['control']))
    mfile = {v: DATA / f'decoding_maps_{prefix}_{v}_2026-11.npz' for v in ('model', 'control')}
    if all(f.exists() for f in mfile.values()):   # c's map: rel_maps per variant, then model − control per subject
        mz = {v: np.load(f) for v, f in mfile.items()}
        assert list(mz['model']['subjects']) == list(mz['control']['subjects'])
        extra['rm'] = rel_maps(mz['model'], decoder, rois) - rel_maps(mz['control'], decoder, rois)
    return qd, bd, sweep, extra


def sweep_half_icons(ax, w=.075, h=.5, y=-.98):
    """Under the sweep's x axis: for bars left and right of the centre, an aperture icon (HP upper right, ringed red)
    with the quadrants that bar crosses shaded in the colours of d, and their names."""
    from matplotlib.patches import Circle, Wedge
    from retsupp.visualize.paper.paper_style import HP
    halves = [(0., [(90, 180, 'adjacent'), (180, 270, 'opposite')], 'Next to HP + opposite', 'left'),
              (1. - w, [(0, 90, 'hp'), (270, 360, 'adjacent')], 'HP + next to HP', 'right')]
    for x, wedges, lab, side in halves:
        ins = ax.inset_axes([x, y, w, h])
        for a0, a1, c in wedges:
            ins.add_patch(Wedge((0, 0), APERTURE, a0, a1, facecolor=LOC_COLORS[c], alpha=.35, lw=0))
        ins.add_patch(Circle((0, 0), APERTURE, fill=False, ec='.55', lw=.5))
        for j, a in enumerate((45, 135, 225, 315)):
            q = ITEM_ECC * np.array([np.cos(np.deg2rad(a)), np.sin(np.deg2rad(a))])
            ins.add_patch(Circle(q, ITEM_R, facecolor='w', ec=HP if j == 0 else '.6', lw=.9 if j == 0 else .5))
        ins.set_xlim(-5, 5)
        ins.set_ylim(-5, 5)
        ins.set_aspect('equal')
        ins.set_axis_off()
        if side == 'left':
            ax.text(x + w + .01, y + h / 2, lab, transform=ax.transAxes, fontsize=6, color='.3', ha='left', va='center')
        else:
            ax.text(x - .01, y + h / 2, lab, transform=ax.transAxes, fontsize=6, color='.3', ha='right', va='center')


def bar_sweep(fig, source, x0, x1, y_img, y_ax, h_ax, W, H, vmax=3., pred=None, size=.56, opposite=False):
    """The bar sweep beside c: the decoded-bar difference image at every bar position (outline decoder), aligned above
    a plot of the segment differences in the quadrant facing HP and next to it against the bar centre (data: points
    and SEM; with ``pred``, the gain-field model's prediction as line and SEM band)."""
    from scipy import ndimage
    from retsupp.visualize.paper.figS_decoded_bars import QUAD, _disc, _smooth_masked, pooled_condition
    from retsupp.visualize.paper.paper_style import FG as FG_, HP, MUTED, offset_spines
    z = np.load(DATA / f'decoded_bars_condition_{source}_outline_2026-11.npz')
    P = pooled_condition(z['mean'])
    grid = z['grid']
    centres = (z['edges'][:-1] + z['edges'][1:]) / 2
    X, Y = grid[:, 0].reshape(50, 50), grid[:, 1].reshape(50, 50)
    inside = np.hypot(X, Y) <= APERTURE
    ns = len(P)
    hp_v, nx_v, op_v = sweep_segments(P, grid, opposite=True)
    lo, hi = -3.15, 3.15
    ax = fig.add_axes([x0 / W, y_ax / H, (x1 - x0) / W, h_ax / H])
    for b, c in enumerate(centres):
        bar = P[:, b, 2].mean(0) > .5
        foot = ndimage.binary_dilation(bar, iterations=1) & inside
        rel = np.mean([100 * P[s, b, 1] / P[s, b, 0][bar & inside].mean() for s in range(ns)], 0)
        xc = x0 + (c - lo) / (hi - lo) * (x1 - x0)
        iax = fig.add_axes([(xc - size / 2) / W, y_img / H, size / W, size / H])
        _disc(iax, _smooth_masked(rel, foot), 'RdBu_r', -vmax, vmax, outline=bar & inside)
    stats_txt = []
    series = [(nx_v, MUTED, 'Next to HP', 1, 'next'), (hp_v, HP, 'HP quadrant', -1, 'hp')]
    if opposite:   # v9: quadrant colours of d, and the opposite segment (bars left of centre) as a dashed third line
        series = [(nx_v, LOC_COLORS['adjacent'], 'Next to HP', 1, 'next'), (op_v, LOC_COLORS['opposite'], 'Opposite', -1, 'opp'),
                  (hp_v, HP, 'HP quadrant', -1, 'hp')]
    for v, col, lab, dy, key in series:
        ls = '--' if key == 'opp' else '-'
        ok = np.isfinite(v).all(0)
        m, se = np.nanmean(v, 0), np.nanstd(v, 0, ddof=1) / np.sqrt(ns)
        if pred is None:
            ax.fill_between(centres[ok], (m - se)[ok], (m + se)[ok], color=col, alpha=.2, lw=0)
            ax.plot(centres[ok], m[ok], 'o-', color=col, ms=3, lw=1.2)
        else:   # model as line + band, data as points + SEM
            pv = pred[key]
            pm, pse = np.nanmean(pv, 0), np.nanstd(pv, 0, ddof=1) / np.sqrt(np.isfinite(pv).sum(0))
            ax.fill_between(centres[ok], (pm - pse)[ok], (pm + pse)[ok], color=col, alpha=.2, lw=0, zorder=1)
            ax.plot(centres[ok], pm[ok], ls, color=col, lw=1.1, zorder=2)
            ax.errorbar(centres[ok] + dy * .04, m[ok], yerr=se[ok], fmt='o', ms=3, color=col, mec='w', mew=.5, lw=.9,
                        capsize=0, zorder=3)
        last = np.flatnonzero(ok)[-1]
        if key == 'opp':   # its bars end at the centre: label at its left end, below the points
            first = np.flatnonzero(ok)[0]
            ax.text(centres[first] - .1, min(m[first] - se[first], pm[first] - pse[first] if pred is not None else 0) - .35,
                    lab, color=col, fontsize=6.5, va='top', ha='left')
        else:
            ax.text(centres[last] + .12, m[last] + dy * .55, lab, color=col, fontsize=6.5, va='center', ha='left')
        for b in np.flatnonzero(ok):
            p = stats.wilcoxon(v[:, b]).pvalue
            stats_txt.append(f'{lab.lower()} at {centres[b]:+.1f}°: {m[b]:+.2f} ± {se[b]:.2f}%, p = {p:.3f}')
            st = '**' if p < .01 else '*' if p < .05 else ''
            if st:
                ax.text(centres[b], m[b] + dy * (se[b] + .25), st, color=col, fontsize=7, ha='center',
                        va='bottom' if dy > 0 else 'top')
    if pred is not None:
        tf = ax.transAxes
        kx, ky = (.55, .95) if opposite else (.03, .105)
        ax.add_patch(plt_rect((kx, ky - .035), .06, .07, tf, FG_))
        ax.plot([kx, kx + .06], [ky, ky], transform=tf, color=FG_, lw=1.1, clip_on=False)
        ax.text(kx + .07, ky, 'Gain-field model', transform=tf, fontsize=6.5, color=FG_, va='center')
        ax.plot([kx + .39, kx + .39], [ky - .045, ky + .045], transform=tf, color=FG_, lw=.9, clip_on=False)
        ax.plot(kx + .39, ky, 'o', transform=tf, ms=3, color=FG_, mec='w', mew=.5, clip_on=False)
        ax.text(kx + .405, ky, 'Data', transform=tf, fontsize=6.5, color=FG_, va='center')
    if opposite:   # which quadrants a bar crosses on each side of the centre (HP upper right, ringed red)
        sweep_half_icons(ax)
    ax.axhline(0, color='.6', lw=.6, ls='--', zorder=0)
    ax.set_xlim(lo, hi)
    ax.set_xticks([-2.7, -1.8, -0.9, 0, 0.9, 1.8, 2.7])
    ax.set_xticklabels([f'{t:+.1f}'.replace('-', '−').replace('+0.0', '0') for t in ax.get_xticks()], fontsize=6.5)
    ax.set_ylim(-2.6, 2.1)
    ax.set_yticks([-2, 0, 2])
    ax.tick_params(labelsize=6.5)
    ax.set_xlabel('Bar centre (°)' if opposite else 'Bar centre (°; positive, toward HP)', fontsize=7, labelpad=1)
    ax.set_ylabel('Segment\ndifference (%)', fontsize=7, labelpad=2)
    offset_spines(ax)
    return len(centres), '; '.join(stats_txt).replace('-', '−')


SWEEP_AREA_LEGEND = """
**Figure S. The readouts of Fig. 6c for every area.** Per area (rows of the pathway columns; n, subjects with that area: an area was skipped for a subject when fewer than 20 voxels passed the criteria), left: each run's visual field rotated so that the HP location lies at 12 o'clock, the decoded intensity on the bar's pixels minus its average over the four rotations, in % of that average, averaged over runs and subjects and smoothed (σ = 0.4°), as in Fig. 6c (unconstrained decoder; same colour scale in every area). Right: as the bar sweeps across the aperture, the decoded bar (decoder with the bar outline given) minus the mean decoded bar at the same time point over the subject's runs (balanced over HP locations), on the bar's segment in the quadrant facing HP (red), in the quadrants next to HP (dark grey) and in the quadrant opposite HP (light grey, dashed line), in % of the mean decoded intensity on the bar; rotated so that HP is at the upper right, horizontal bars reflected onto vertical ones; bars left of the centre cross the quadrants next to HP and opposite it, bars right of the centre the quadrant facing HP and the other quadrant next to it. Points and bars, mean ± SEM over subjects; stars, two-sided Wilcoxon signed-rank test per position, uncorrected (* p < .05, ** p < .01). Lines and bands in V1–V3: the gain-field model's prediction (simulation from the joint fit minus the paired simulation with the gains at 0, decoded as the data, as in Fig. 6); it was simulated for V1–V3 only. {stats}
"""


def sweep_area_series(seg, roi):
    """Per subject × bar position for one area (pooled orientation): HP, next to HP (mean of the two) and opposite."""
    g = seg[(seg.roi == roi) & (seg.orientation == 'pooled')]
    t = g.pivot_table(index=['subject', 'bin'], columns='quadrant', values='diff_pct')
    nxt = t[[c for c in ('next_ul', 'next_lr') if c in t]].mean(1)
    out = {}
    for k, col in (('hp', t.get('hp')), ('next', nxt), ('opp', t.get('opposite'))):
        out[k] = col.unstack('bin') if col is not None else None
    return out


def plot_sweep_per_area(name='figS_decoding_sweep_per_area', source='jointhrf', decoder=DECODER, bins=7):
    import matplotlib.pyplot as plt
    from retsupp.visualize.paper.paper_style import DOUBLE, FG, HP, ROI_GROUPS, offset_spines, save
    seg = pd.read_csv(DATA / f'decoded_bars_segments_{source}_outline_b{bins}_2026-11.tsv', sep='\t')
    sim_f = DATA / f'decoded_bars_segments_decode_simfit_outline_b{bins}_2026-11.tsv'
    sim = pd.read_csv(sim_f, sep='\t') if sim_f.exists() else None
    maps = np.load(DATA / f'decoding_maps_{source}_2026-11.npz')
    edges = np.linspace(-APERTURE, APERTURE, bins + 1)
    centres = (edges[:-1] + edges[1:]) / 2
    cols = {'hp': (HP, '-', 'HP'), 'next': (LOC_COLORS['adjacent'], '-', 'Next to HP'),
            'opp': (LOC_COLORS['opposite'], '--', 'Opposite')}
    W, H = DOUBLE, 5.2
    fig = plt.figure(figsize=(W, H))
    colw, rowh, top = W / 3, 1.18, H - .32
    stats_txt, im = [], None
    for gi, (gname, rois) in enumerate(ROI_GROUPS):
        x0 = gi * colw
        fig.text((x0 + colw / 2) / W, (H - .12) / H, gname, ha='center', va='center', fontsize=7.5, color='.3')
        for ri, roi in enumerate(rois):
            y0 = top - (ri + 1) * rowh
            # map (free decoder), subjects with this area only
            rm = rel_maps(maps, decoder, [roi])
            rm = rm[np.isfinite(rm).any((1, 2))]
            n_map = len(rm)
            ax = fig.add_axes([(x0 + .02) / W, (y0 + .2) / H, .78 / W, .78 / H])
            im = panel_map(ax, rm, label='', tag=False)
            # sweep
            ser = sweep_area_series(seg, roi)
            n = int(seg[seg.roi == roi].subject.nunique())
            ax = fig.add_axes([(x0 + 1.07) / W, (y0 + .32) / H, colw - 1.22, .7 / H] if False else
                              [(x0 + 1.17) / W, (y0 + .32) / H, (colw - 1.3) / W, .66 / H])
            prd = None
            if sim is not None and roi in set(sim.roi):
                sm = {v: sweep_area_series(sim[sim.variant == v], roi) for v in ('model', 'control')}
                prd = {k: (sm['model'][k] - sm['control'][k]) for k in cols if sm['model'][k] is not None}
            for k, (col, ls, lab) in cols.items():
                v = ser[k]
                if v is None:
                    continue
                v = v.reindex(columns=range(bins))
                ok = v.notna().sum(0).to_numpy() >= 5
                m, se = v.mean().to_numpy(), v.sem().to_numpy()
                xs = centres + {'hp': .06, 'next': 0, 'opp': -.06}[k]
                if prd is not None and k in prd:
                    pv = prd[k].reindex(columns=range(bins))
                    pm, pse = pv.mean().to_numpy(), pv.sem().to_numpy()
                    ax.fill_between(centres[ok], (pm - pse)[ok], (pm + pse)[ok], color=col, alpha=.18, lw=0, zorder=1)
                    ax.plot(centres[ok], pm[ok], ls, color=col, lw=1., zorder=2)
                ax.errorbar(xs[ok], m[ok], yerr=se[ok], fmt='o', ms=2.6, color=col, mec='w', mew=.4, lw=.8, capsize=0,
                            zorder=3)
                for b in np.flatnonzero(ok):
                    vv = v[b].dropna()
                    p_ = stats.wilcoxon(vv).pvalue
                    stats_txt.append((roi, lab, centres[b], vv.mean(), vv.sem(), p_, len(vv)))
                    st = stars(p_)
                    if st:
                        up = m[b] > 0
                        ax.text(xs[b], m[b] + (se[b] + .3) * (1 if up else -1), st, color=col, fontsize=6.5,
                                ha='center', va='bottom' if up else 'top')
            ax.axhline(0, color='.6', lw=.6, ls='--', zorder=0)
            ax.set_xlim(-3.15, 3.15)
            ax.set_ylim(-5, 5)
            ax.set_yticks([-4, 0, 4])
            ax.set_yticklabels(['−4', '0', '4'])
            ax.set_xticks([-2.7, 0, 2.7])
            ax.set_xticklabels(['−2.7', '0', '+2.7'])
            ax.tick_params(labelsize=6)
            if ri == len(rois) - 1:
                ax.set_xlabel('Bar centre (°)', fontsize=6.5, labelpad=1)
            if gi == 0:
                ax.set_ylabel('Difference (%)', fontsize=6.5, labelpad=1)
            ax.text(.02, 1.0, f'{roi} (n = {n})', transform=ax.transAxes, fontsize=7, color=FG, va='bottom',
                    fontweight='bold')
            if n_map != n:
                print(roi, 'map n', n_map, 'sweep n', n)
            offset_spines(ax)
    # glyph key and colour bar in the empty cell under the dorsal column
    kx, ky = colw + .25, top - 3 * rowh + .55
    kax = fig.add_axes([kx / W, (ky - .45) / H, 1.9 / W, .75 / H])
    kax.set_axis_off()
    kax.set_xlim(0, 1)
    kax.set_ylim(0, 1)
    for i, (k, (col, ls, lab)) in enumerate(cols.items()):
        y = .9 - i * .2
        kax.errorbar([.06], [y], yerr=[[.06], [.06]], fmt='o', ms=2.6, color=col, mec='w', mew=.4, lw=.8)
        kax.text(.12, y, lab, fontsize=6.5, color=col, va='center')
        kax.plot([.5, .62], [y, y], ls, color=col, lw=1.)
    kax.text(.66, .9, 'Gain-field model\n(V1–V3)', fontsize=6.5, color=FG, va='top', linespacing=1.)
    cax = fig.add_axes([(kx + .1) / W, (ky - .55) / H, .9 / W, .06 / H])
    cb = fig.colorbar(im, cax=cax, orientation='horizontal', ticks=[-3, 0, 3])
    cb.ax.set_xticklabels(['−3%', '0', '3%'], fontsize=6)
    cb.ax.tick_params(length=2, width=.5, pad=1)
    cb.outline.set_linewidth(.4)
    cb.ax.text(-.06, .5, 'Less', transform=cb.ax.transAxes, ha='right', va='center', fontsize=6, color='#2166AC')
    cb.ax.text(1.06, .5, 'More', transform=cb.ax.transAxes, ha='left', va='center', fontsize=6, color='#B2182B')
    names = {'HP': 'HP-quadrant segment', 'Next to HP': 'next-to-HP segment', 'Opposite': 'opposite segment'}
    sig = [f'{r} {names[lab]} at {c:+.1f}°: {m_:+.2f} ± {se_:.2f}%, p = {p_:.3f} (n = {n_})'.replace('-', '−')
           for r, lab, c, m_, se_, p_, n_ in stats_txt if p_ < .05]
    txt = (f'Of the {len(stats_txt)} position tests (all areas, uncorrected), {len(sig)} reach p < .05 '
           f'(about {.05 * len(stats_txt):.0f} expected by chance): ' + '; '.join(sig) + '.') if sig else \
        'No position reaches p < .05.'
    save(fig, name, SWEEP_AREA_LEGEND.format(stats=txt))
    return stats_txt


def plot_all_rois(name='figS_decoding_all_rois', source='jointhrf'):
    """Supplementary figure: the group-mean decoded frames per ROI at the time points of Fig. 6a (unconstrained decoder)."""
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from retsupp.visualize.paper.paper_style import FG, MUTED, ROI_GROUPS, STIM, save
    z = np.load(DATA / f'decoding_frames_allrois_{source}_2026-11.npz')
    rois, G, par = list(z['rois']), z['group'], z['paradigm'].astype(float)
    q = pd.read_csv(DATA / f'decoding_quality_{source}_2026-11.tsv', sep='\t')
    qs = q[q.decoder == 'free'].groupby(['subject', 'roi']).r.mean().reset_index()
    med, nsub = qs.groupby('roi').r.median(), qs.groupby('roi').subject.nunique()
    W, H = 7.25, 6.9
    fig = plt.figure(figsize=(W, H))
    size, gap, x0 = .54, .1, 1.6
    c = np.linspace(-5, 5, 50)
    X, Y = np.meshgrid(c, c)
    cm = mcolors.LinearSegmentedColormap.from_list('stim', ['#FFFFFF', STIM])
    rows = [('Bar', None)] + [(r, r) for _, g in ROI_GROUPS for r in g]
    y = H - .45
    ypos = {}
    prev_group = None
    for lab, roi in rows:
        grp = next((gn for gn, g in ROI_GROUPS if roi in g), None)
        y -= size + (.06 if grp == prev_group else .2)
        prev_group = grp
        ypos[lab] = y
        arr = par if roi is None else G[rois.index(roi)]
        vmax = 1. if roi is None else np.nanpercentile(arr, 99.7)
        for k, t in enumerate(FR):
            ax = fig.add_axes([(x0 + k * (size + gap)) / W, y / H, size / W, size / H])
            if roi is None:
                disc(ax, arr[k], cm, vmax)
            else:
                img = np.where(np.hypot(X, Y) <= APERTURE + .1, np.clip(arr[k], 0, None), 0)
                ax.imshow(img, origin='lower', extent=[-5.1, 5.1, -5.1, 5.1], cmap='magma', vmin=0, vmax=vmax,
                          interpolation='bilinear')
                ax.add_patch(Circle((0, 0), APERTURE, fill=False, ec='.75', lw=.4))
                ax.set_xlim(-3.6, 3.6)
                ax.set_ylim(-3.6, 3.6)
                ax.set_axis_off()
            if roi is None:
                ax.text(.5, 1.06, f'{t * TR:.0f} s', transform=ax.transAxes, ha='center', va='bottom', fontsize=6.5,
                        color='.35')
            if k == 0:
                txt = 'Bar' if roi is None else f'{roi}\nn = {nsub[roi]}, r = {med[roi]:.2f}'.replace('0.', '.')
                ax.text(-.1, .5, txt, transform=ax.transAxes, ha='right', va='center', fontsize=6.5, color=FG)
    for gname, g in ROI_GROUPS:
        y0, y1 = ypos[g[-1]], ypos[g[0]] + size
        fig.add_artist(plt.Line2D([.3 / W] * 2, [y0 / H, y1 / H], color=MUTED, lw=.7))
        fig.text(.2 / W, (y0 + y1) / 2 / H, gname, rotation=90, ha='center', va='center', fontsize=6.5, color='.4')
    fig.text(.2 / W, (H - .18) / H, 'The bar decoded from each area: mean of the subjects\' decoded frames',
             fontsize=7.5, color=STIM, va='center')
    legend = f"""
**Figure S. The bar decoded from each of the nine areas.** As Fig. 6a (bottom row) for every ROI: the decoded frames
(unconstrained decoder, pRFs with one HRF per subject and ROI fitted on the other runs, held-out runs) averaged over each
subject's runs and then over subjects, at seven time points of the run (the last without a bar); top row, the true bar.
Each ROI on a magma scale from 0 to its own maximum (99.7th percentile). Row labels: number of subjects with that ROI
(subjects with fewer than 20 usable voxels skipped) and the median over subjects of the frame-wise correlation between
decoded and true frame (Fig. 6b). The bar is visible in every area, sharpest in V1–V3 and blurred in the areas with
larger pRFs.
"""
    save(fig, name, legend)


def plot_all_rois_examples(name='figS_decoding_all_rois_examples', source='jointhrf', show=(1, 4, 5)):
    """Supplementary figure: per ROI, three example subjects (75th, 50th and 25th percentile of decoding quality r),
    their decoded frames averaged over runs, at three time points."""
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from retsupp.visualize.paper.paper_style import FG, MUTED, ROI_GROUPS, STIM, save
    z = np.load(DATA / f'decoding_example_subjects_{source}_2026-11.npz')
    ex = pd.read_csv(DATA / f'decoding_example_subjects_{source}_2026-11.tsv', sep='\t')
    fr = np.load(DATA / f'decoding_frames_allrois_{source}_2026-11.npz')
    par = fr['paradigm'].astype(float)
    frames = list(z['frames'])
    W, H = 7.25, 7.4
    fig = plt.figure(figsize=(W, H))
    size, gap, bgap, x0 = .5, .04, .22, 1.35
    c = np.linspace(-5, 5, 50)
    X, Y = np.meshgrid(c, c)
    cm = mcolors.LinearSegmentedColormap.from_list('stim', ['#FFFFFF', STIM])
    pct_lab = {75: '75th percentile of r', 50: 'Median r', 25: '25th percentile of r'}

    def xpos(blk, k):
        return x0 + blk * (3 * (size + gap) + bgap) + k * (size + gap)

    y = H - .85
    for blk, pct in enumerate((75, 50, 25)):
        fig.text((xpos(blk, 0) + 1.5 * size + gap) / W, (H - .14) / H, pct_lab[pct], ha='center', va='center',
                 fontsize=6.5, color=FG)
        for k, j in enumerate(show):
            ax = fig.add_axes([xpos(blk, k) / W, y / H, size / W, size / H])
            disc(ax, par[j], cm, 1.)
            ax.text(.5, 1.04, f'{frames[j] * TR:.0f} s', transform=ax.transAxes, ha='center', va='bottom', fontsize=5.5,
                    color='.35')
    fig.text((x0 - .1) / W, (y + size / 2) / H, 'Bar', ha='right', va='center', fontsize=6.5, color=FG)
    ypos = {}
    prev = None
    for gname, g in ROI_GROUPS:
        for roi in g:
            y -= size + .17 + (.12 if gname != prev else 0)
            prev = gname
            ypos[roi] = y
            for blk, pct in enumerate((75, 50, 25)):
                r = ex[(ex.roi == roi) & (ex.pct == pct)].iloc[0]
                arr = z[f'sub-{int(r.subject):02d}_roi-{roi}']
                vmax = np.nanpercentile(arr[list(show)], 99.7)
                for k, j in enumerate(show):
                    ax = fig.add_axes([xpos(blk, k) / W, y / H, size / W, size / H])
                    img = np.where(np.hypot(X, Y) <= APERTURE + .1, np.clip(arr[j], 0, None), 0)
                    ax.imshow(img, origin='lower', extent=[-5.1, 5.1, -5.1, 5.1], cmap='magma', vmin=0, vmax=vmax,
                              interpolation='bilinear')
                    ax.add_patch(Circle((0, 0), APERTURE, fill=False, ec='.75', lw=.4))
                    ax.set_xlim(-3.6, 3.6)
                    ax.set_ylim(-3.6, 3.6)
                    ax.set_axis_off()
                fig.text((xpos(blk, 1) + size / 2) / W, (y - .02) / H, f'sub-{int(r.subject):02d}, r = {r.r:.2f}'.replace('0.', '.'),
                         ha='center', va='top', fontsize=5.5, color='.35')
            fig.text((x0 - .1) / W, (y + size / 2) / H, roi, ha='right', va='center', fontsize=7, color=FG)
    for gname, g in ROI_GROUPS:
        y0, y1 = ypos[g[-1]] - .12, ypos[g[0]] + size
        fig.add_artist(plt.Line2D([.45 / W] * 2, [y0 / H, y1 / H], color=MUTED, lw=.7))
        fig.text(.33 / W, (y0 + y1) / 2 / H, gname, rotation=90, ha='center', va='center', fontsize=6.5, color='.4')
    legend = f"""
**Figure S. The bar decoded from each area in example subjects.** For each ROI, the subjects at the 75th, 50th and 25th
percentile of decoding quality in that ROI (frame-wise correlation between decoded and true frame, mean over runs, as in
Fig. 6b; label: subject and r), their decoded frames (unconstrained decoder, pRFs with one HRF per subject and ROI fitted
on the other runs, held-out runs) averaged over their runs (11 or 12), at three time points (a vertical bar near the
centre, a horizontal bar through the centre and one in the lower part); top row, the true bar. Each subject × ROI on a
magma scale from 0 to its own maximum (99.7th percentile). The examples differ between ROIs because the percentiles are
taken within each ROI.
"""
    save(fig, name, legend)


def plot_bar_crossing(name='figS_bar_crossing_response', source_note=''):
    """Supplementary figure (was Fig. 6g): the measured response when the bar crosses a pRF near an item location."""
    import matplotlib.pyplot as plt
    from retsupp.visualize.paper.paper_style import save
    bp = load_bar_pass()
    bst = bar_pass_stats(bp)
    pred = load_prediction(prediction, decoder) if prediction else None
    W, H = 3.6, 1.75
    fig = plt.figure(figsize=(W, H))
    bar_pass_icon(fig, .05, .2, .95, bp.groupby('t')[['hp', 'adjacent', 'opposite']].mean().mean(1))
    ax = fig.add_axes([1.75 / W, .42 / H, 1.4 / W, 1.0 / H])
    panel_bar_pass(ax, bp, bst)
    r_, b_ = bst['raw'], bst['base']
    bar_txt = (f'{r_["diff"]:+.3f}% signal ({r_["rel"]:+.1f}% of the LP response), {r_["below"]:.0f}% of subjects below 0, '
               f'one-sided Wilcoxon p = {r_["p"]:.3f} (vs the two neighbouring LP locations {r_["rel_adjacent"]:+.1f}%, '
               f'p = {r_["p_adjacent"]:.3f}; vs the opposite one {r_["rel_opposite"]:+.1f}%, p = {r_["p_opposite"]:.3f})')
    save(fig, name, BAR_LEGEND.format(bar=bar_txt, bar_base=f'{b_["rel"]:+.1f}%, p = {b_["p"]:.2f}', w0=BAR_WINDOW[0],
                                      w1=BAR_WINDOW[1], source=source_note))


# ------------------------------------------------------------------------------------------------------------ figure
def test_figure(e, t, rois, decoder, t_out, d_text, name):
    """The per-area HP − other quadrants test (Fig. 6d before v3) as a supplementary figure."""
    import matplotlib.pyplot as plt
    from retsupp.visualize.paper.paper_style import DOUBLE, save
    W, H = DOUBLE * .75, 1.95
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([.75 / W, .5 / H, (W - 1.9) / W, 1.0 / H])
    panel_test(ax, e, t, rois, decoder, t_out)
    body = d_text.strip()
    body = body[0].upper() + body[1:] if not body.startswith('d,') else body[3:].strip()
    body = body[0].upper() + body[1:]
    legend = ('**Figure S. Decoded intensity in the quadrant facing HP, per area (Fig. 6, per area).** Decoding as in Fig. 6. '
              + body)
    save(fig, name, legend)


PRED_TXT = """{map_txt}Lines and bands in c (bottom), d and e: what the fitted gain-field model predicts for these readouts (mean ± SEM over subjects; points with bars are the data). Per subject and run, the BOLD of V1–V3 was simulated from the subject's joint fit (pRFs fitted together with the gain fields on all runs: sustained HP and LP gains and field width, transient distractor and target gains, search items, one HRF per subject and ROI) plus the real run's residuals under that fit, phase-randomised (one random phase per frequency for all voxels, so the noise keeps each voxel's spectrum and the covariance between voxels but nothing locked to the stimulus), and decoded exactly as the data (the same fold pRFs without gain terms, which never saw the run, and the same noise model and readouts), on the decoder's voxels that are also in the joint fit; a paired simulation with the five gains at 0 and the same noise was decoded alike, and the prediction is the difference (model − gains at 0) per subject, averaged like the data (n = {n}). {numbers}"""


def sweep_data(source):
    from retsupp.visualize.paper.figS_decoded_bars import pooled_condition
    z = np.load(DATA / f'decoded_bars_condition_{source}_outline_2026-11.npz')
    hp_v, nx_v = sweep_segments(pooled_condition(z['mean']), z['grid'])
    return dict(hp=hp_v, next=nx_v, subjects=np.asarray(z['subjects']),
                centres=(z['edges'][:-1] + z['edges'][1:]) / 2)


def prediction_text(pred, qe, be, sw, r_data=None, ecc=True, numbers_only=False):
    """Predicted vs observed: quadrant contrasts (HP − next, next − opposite), the sweep segments and the eccentricity
    profile; per-subject correlation of predicted and observed HP − next."""
    qp, bp_, swp, extra = pred
    mn = lambda x, sign='': format(x, sign + '.2f').replace('-', '−')   # noqa: E731
    fmt = lambda v: f'{mn(v.mean(), "+")} ± {v.sem():.2f}%'   # noqa: E731
    con = {}
    for nm, (a, b) in {'HP − next to HP': ('hp', 'adjacent'), 'next to HP − opposite': ('adjacent', 'opposite'),
                       'HP − opposite': ('hp', 'opposite')}.items():
        o, m = qe[a] - qe[b], qp[a] - qp[b]
        both = o.index.intersection(m.index)
        r, pr = stats.pearsonr(o[both], m[both])
        con[nm] = f'{nm}: observed {fmt(o)}, predicted {fmt(m)} (per-subject r = {mn(r)}, p = {pr:.3f}; observed − predicted, two-sided Wilcoxon p = {stats.wilcoxon((o - m).dropna()).pvalue:.3f})'
    quad = 'Quadrants (d): ' + '; '.join(con.values()) + '; predicted per quadrant HP ' + fmt(qp.hp) + ', next to HP ' + \
        fmt(qp.adjacent) + ', opposite ' + fmt(qp.opposite)
    # Sweep: observed and predicted segment differences per bar position.
    cen = sw['centres']
    rows = []
    obs_all, pr_all = [], []
    for k, lab in (('hp', 'HP quadrant'), ('next', 'next to HP')):
        o, m = sw[k], swp[k]
        ok = np.isfinite(o).all(0) & np.isfinite(m).all(0)
        om, mm = np.nanmean(o, 0), np.nanmean(m, 0)
        obs_all += list(om[ok])
        pr_all += list(mm[ok])
        rows.append(f'{lab} segment, mean over bar positions observed {mn(om[ok].mean(), "+")}%, predicted {mn(mm[ok].mean(), "+")}%')
    r_sw = np.corrcoef(obs_all, pr_all)[0, 1]
    sweep = 'Bar sweep (c): ' + '; '.join(rows) + f'; correlation of the observed and predicted group means over the ' \
        f'{len(obs_all)} segment × position points r = {mn(r_sw)}'
    ecc_txt = (f'Whole aperture (e): HP − other quadrants observed {fmt(be["all"])}, predicted {fmt(bp_["all"])}'
               if ecc else '')
    cov = (f' The joint fit covers a median {extra["coverage"]:.0f}% of the decoder\'s voxels per run and area, and the '
           f'simulated runs decode about as well as the real ones (frame-wise r, median over subjects of the V1–V3 mean: '
           f'{extra["r_model"]:.2f} simulated' + (f', {r_data:.2f} measured' if r_data is not None else '') + ').')
    map_txt = ''
    if 'map_r' in extra:
        map_txt = ('The right circle in c ("Gain-field model") is the same map computed from the simulation described '
                   'next: per subject the rotated map of the simulation with the fitted gains minus that of the paired '
                   'simulation with the gains at 0 (each built exactly as the data map: drive on the bar\'s pixels, '
                   'minus its average over the four rotations, in % of that average, V1–V3), then smoothed and averaged '
                   f'over subjects as the data; same colour scale. Spatial correlation of the two group maps inside the '
                   f'aperture r = {mn(extra["map_r"])}. ')
    numbers = quad + '. ' + sweep + '.' + (' ' + ecc_txt + '.' if ecc_txt else '')
    if numbers_only:
        return numbers + cov
    return PRED_TXT.format(map_txt=map_txt, n=len(qp), numbers=numbers) + cov


PATHWAYS = {'Early': ['V1', 'V2', 'V3'], 'Dorsal': ['V3AB', 'IPS'], 'Occipitotemporal': ['hV4', 'VO', 'LO', 'TO']}
PATHWAY_TXT = """ f, The same contrast as d for every area, by pathway: per subject and area, the decoded intensity on the bar's pixels per quadrant averaged over runs and expressed in % of the mean of the four quadrants (as in d), HP minus the mean of the three other quadrants; then the mean over the areas of a pathway that the subject has (an area was skipped for a subject when fewer than 20 voxels passed the criteria). Dots are subjects, points and bars the mean ± SEM; small labelled markers, the per-area means. One-sided Wilcoxon signed-rank test (HP < other quadrants), uncorrected: {pw}. Data only: the gain-field model was simulated for V1–V3 alone."""


def pathway_effects(d, decoder=DECODER):
    """Per subject × area: HP − mean of the other three quadrants, in % of the mean of the four (``quadrant_effects``
    with one area), and per subject × pathway the mean over the areas present."""
    per = {}
    for roi in sum(PATHWAYS.values(), []):
        if roi in set(d.roi):
            q = quadrant_effects(d, decoder, [roi])
            per[roi] = q.hp - (q.left + q.right + q.opposite) / 3
    per = pd.DataFrame(per)
    pw = pd.DataFrame({k: per[[r for r in rs if r in per]].mean(1, skipna=True) for k, rs in PATHWAYS.items()})
    return per, pw


def pathway_stats(per, pw):
    out = {}
    for k in pw:
        v = pw[k].dropna()
        out[k] = dict(m=v.mean(), se=v.sem(), n=len(v), p=stats.wilcoxon(v, alternative='less').pvalue,
                      below=(v < 0).mean() * 100)
    hi = per[PATHWAYS['Dorsal'] + PATHWAYS['Occipitotemporal']].mean(1, skipna=True).dropna()
    out['Higher'] = dict(m=hi.mean(), se=hi.sem(), n=len(hi), p=stats.wilcoxon(hi, alternative='less').pvalue,
                         below=(hi < 0).mean() * 100)
    return out


def panel_pathways(ax, per, pw, st):
    from retsupp.visualize.paper.paper_style import FG, HP, offset_spines
    rng = np.random.default_rng(5)
    lo, hi = -8, 8.5
    for i, k in enumerate(pw):
        v = pw[k].dropna()
        x = i * 1.25
        ax.scatter(x + rng.uniform(-.14, .14, len(v)), v, s=5, color=HP, alpha=.35, lw=0, zorder=2)
        ax.errorbar(x + .26, v.mean(), yerr=v.sem(), fmt='o', ms=4.5, color=HP, mec='w', mew=.6, lw=1.3, capsize=0,
                    zorder=4)
        # per-area means: small open markers with their names, labels pushed apart vertically
        rois = [r for r in PATHWAYS[k] if r in per]
        ms = per[rois].mean()
        ax.scatter(np.full(len(ms), x + .5), ms, s=9, facecolor='w', edgecolor='.35', lw=.6, zorder=3)
        ys = ms.sort_values()
        lab = ys.to_numpy().astype(float)
        gap = .9
        for j in range(1, len(lab)):
            lab[j] = max(lab[j], lab[j - 1] + gap)
        lab -= max(0, (lab.mean() - ys.mean()))
        for (r, y0), yl in zip(ys.items(), lab):
            ax.plot([x + .55, x + .6], [y0, yl], color='.6', lw=.4)
            ax.text(x + .62, yl, r, fontsize=6, color='.35', va='center', ha='left')
        p_ = st[k]['p']
        ax.text(x + .13, hi - .3, (stars(p_) + '\n' if p_ < .05 else '\n') + f'p = {p_:.3f}'.replace('0.', '.'),
                ha='center', va='top', fontsize=6, color=HP if p_ < .05 else '.4', linespacing=1.)
    ax.axhline(0, color='.6', lw=.7, ls='--', zorder=0)
    ax.set_xlim(-.35, 2 * 1.25 + .95)
    ax.set_ylim(lo, hi)
    ax.set_yticks([-6, -3, 0, 3])
    ax.set_yticklabels(['−6', '−3', '0', '3'])
    ax.set_xticks(np.arange(3) * 1.25 + .13)
    ax.set_xticklabels(['Early', 'Dorsal', 'Occipito-\ntemporal'], fontsize=6.5)
    ax.tick_params(axis='x', length=0)
    ax.set_ylabel('HP − other (%)', labelpad=1)
    offset_spines(ax)


AREA_TXT = """ f, The same contrast as d (per subject and area, the decoded intensity on the bar's pixels per quadrant averaged over runs, in % of the mean of the four quadrants, HP minus the mean of the three other quadrants), pooled first and then per area. Left, pooled: the mean over all nine areas and over the areas of each pathway that a subject has (an area was skipped for a subject when fewer than 20 voxels passed the criteria); dots are subjects, points and bars the mean ± SEM; one-sided Wilcoxon signed-rank test (HP < other quadrants), uncorrected. Right, per area: mean ± SEM over subjects; the same test, Benjamini-Hochberg across the nine areas; stars mark q < .05. {pw}. Data only: the gain-field model was simulated for V1–V3 alone."""


def area_stats(per, pw):
    """Pooled columns (all areas, each pathway): one-sided Wilcoxon; areas: one-sided Wilcoxon, BH across the nine."""
    pooled = {'All areas': per.mean(1, skipna=True)}
    pooled.update({k: pw[k] for k in pw})
    out = {'pooled': {}, 'area': {}}
    for k, v in pooled.items():
        v = v.dropna()
        out['pooled'][k] = dict(v=v, m=v.mean(), se=v.sem(), n=len(v), p=stats.wilcoxon(v, alternative='less').pvalue,
                                below=(v < 0).mean() * 100)
    rois = [r for r in ROI_ORDER_ALL if r in per]
    ps = np.array([stats.wilcoxon(per[r].dropna(), alternative='less').pvalue for r in rois])
    qs = stats.false_discovery_control(ps)
    for r, p_, q_ in zip(rois, ps, qs):
        v = per[r].dropna()
        out['area'][r] = dict(m=v.mean(), se=v.sem(), n=len(v), p=p_, q=q_)
    return out


def area_text(per, ast):
    f = lambda x: f'{x:+.2f}'.replace('-', '−')   # noqa: E731
    pooled = '; '.join(f'{k} {f(d["m"])} ± {d["se"]:.2f}%, p = {d["p"]:.3f}, {d["below"]:.0f}% of subjects below 0, '
                       f'n = {d["n"]}' for k, d in ast['pooled'].items())
    areas = '; '.join(f'{r} {f(d["m"])} ± {d["se"]:.2f}%, p = {d["p"]:.3f}, q = {d["q"]:.3f}, n = {d["n"]}'
                      for r, d in ast['area'].items())
    return 'Pooled: ' + pooled + '. Per area: ' + areas


ROI_ORDER_ALL = ['V1', 'V2', 'V3', 'V3AB', 'IPS', 'hV4', 'VO', 'LO', 'TO']


def panel_areas(ax, per, pw, ast):
    """f (v9): pooled columns first (all areas, Early, Dorsal, Occipitotemporal: subject dots, mean ± SEM, p), then the
    nine areas grouped by pathway (mean ± SEM, stars where BH q < .05), as in Fig. 5c."""
    from retsupp.visualize.paper.paper_style import FG, HP, ROI_X, offset_spines, roi_axis
    rng = np.random.default_rng(5)
    lo, hi = -8, 8
    pooled_x = {'All areas': -9.4, 'Early': -7.2, 'Dorsal': -5.0, 'Occipitotemporal': -2.8}
    for k, x in pooled_x.items():
        d = ast['pooled'][k]
        ax.scatter(x - .35 + rng.uniform(-.3, .3, d['n']), d['v'], s=5, color=HP, alpha=.35, lw=0, zorder=2)
        ax.errorbar(x + .4, d['m'], yerr=d['se'], fmt='o', ms=4.5, color=HP if d['p'] < .05 else '.45', mec='w', mew=.6,
                    lw=1.3, capsize=0, zorder=4)
        lab = (stars(d['p']) + ' ' if d['p'] < .05 else '') + f'p = {d["p"]:.3f}'.replace('0.', '.')
        ax.text(x, hi, lab, ha='center', va='top', fontsize=6, color=HP if d['p'] < .05 else '.45')
    ax.axvline(-1.4, color='.85', lw=.6, zorder=0)
    for r, d in ast['area'].items():
        x = ROI_X[r]
        sig = d['q'] < .05
        ax.errorbar(x, d['m'], yerr=d['se'], fmt='o', ms=3.5, color=HP if sig else '.45', mec='w', mew=.5, lw=1.,
                    capsize=0, zorder=4)
        if sig:
            ax.text(x, hi, stars(d['q']), ha='center', va='top', fontsize=7, color=HP)
    sig = [(r, d) for r, d in ast['area'].items() if d['q'] < .05]
    if sig:   # the q values once, over the significant areas (one label when they share a value)
        qs_ = sorted({f'{d["q"]:.3f}' for _, d in sig})
        xm = np.mean([ROI_X[r] for r, _ in sig])
        ax.text(xm, hi - 1.6, ('q = ' + ', '.join(qs_)).replace('0.', '.'), ha='center', va='top', fontsize=6,
                color=HP)
    ax.axhline(0, color='.6', lw=.7, ls='--', zorder=0)
    ax.set_ylim(lo, hi)
    ax.set_yticks([-6, -3, 0, 3])
    ax.set_yticklabels(['−6', '−3', '0', '3'])
    ax.set_ylabel('HP − other (%)', labelpad=1)
    roi_axis(ax, rotate=False, fontsize=6.5, group_fontsize=6.5)
    xs = list(pooled_x.values()) + [ROI_X[r] for r in ROI_ORDER_ALL]
    ax.set_xticks(xs)
    ax.set_xticklabels(['All areas', 'Early', 'Dorsal', 'Occipito-\ntemporal'] + ROI_ORDER_ALL, fontsize=6.5)
    ax.set_xlim(-10.6, ROI_X['TO'] + .6)
    ax.tick_params(axis='x', length=0, pad=2)
    ax.text(-6.1, hi + .3, 'Pooled', ha='center', va='bottom', fontsize=6.5, color='.4')
    ax.text((ROI_X['V1'] + ROI_X['TO']) / 2, hi + .3, 'Per area', ha='center', va='bottom', fontsize=6.5, color='.4')
    offset_spines(ax)


MAP_CMAP = 'BrBG'          # v10: diverging map without the stimulus blue or the HP red (brown = less, teal = more)
DIFF_LABEL = 'Decoded intensity\n(% vs mean)'
EFFECT_LABEL = 'HP − other\nquadrants (%)'
PATHWAY_ORDER = [('Early', ['V1', 'V2', 'V3']), ('Occipitotemporal', ['hV4', 'VO', 'LO', 'TO']), ('Dorsal', ['V3AB', 'IPS'])]


def glyph_key(ax, entries, x, y, dy=.0, dx=0., fs=6.5, seg=.06, transform=None):
    """Glyph key: entries (label, kind, color, ls); kind 'line' (line + band) or 'point' (point with bar)."""
    tf = transform or ax.transAxes
    for i, (lab, kind, col, ls) in enumerate(entries):
        xx, yy = x + i * dx, y - i * dy
        if kind == 'line':
            ax.add_patch(plt_rect((xx, yy - .045), seg, .09, tf, col, alpha=.18))
            ax.plot([xx, xx + seg], [yy, yy], transform=tf, color=col, lw=1.1, ls=ls, clip_on=False)
        else:
            ax.plot([xx + seg / 2] * 2, [yy - .06, yy + .06], transform=tf, color=col, lw=.9, clip_on=False)
            ax.plot(xx + seg / 2, yy, 'o', transform=tf, ms=3.2, color=col, mec='w', mew=.5, clip_on=False)
        ax.text(xx + seg + .015, yy, lab, transform=tf, fontsize=fs, color=col if kind == 'point' else '.25',
                va='center', ha='left')


def sweep_half_icons_up(ax, w=.1, h=.55, y=-1.0):
    """Under the slim sweep: what a bar left and right of the centre crosses, drawn with HP at 12 o'clock as in the
    maps (the sweep frame has HP upper right; turned 45° counter-clockwise, its bars run diagonally): the quadrants the
    bar crosses shaded in d's colours, the bar itself as a dark stroke."""
    from matplotlib.patches import Circle, Polygon, Wedge
    from retsupp.visualize.paper.paper_style import HP
    k = np.sqrt(.5)
    rot = lambda px, py: (k * px - k * py, k * px + k * py)   # noqa: E731  sweep frame -> HP at 12 o'clock
    halves = [(0., [(135, 225, 'adjacent'), (225, 315, 'opposite')], -1.4, 'Next to HP + opposite', 'left'),
              (1. - w, [(45, 135, 'hp'), (315, 405, 'adjacent')], 1.4, 'HP + next to HP', 'right')]
    for x, wedges, bx, lab, side in halves:
        ins = ax.inset_axes([x, y, w, h])
        for a0, a1, c in wedges:
            ins.add_patch(Wedge((0, 0), APERTURE, a0, a1, facecolor=LOC_COLORS[c], alpha=.35, lw=0))
        ins.add_patch(Circle((0, 0), APERTURE, fill=False, ec='.55', lw=.5))
        hw = .45
        corners = [rot(bx - hw, -2.6), rot(bx + hw, -2.6), rot(bx + hw, 2.6), rot(bx - hw, 2.6)]
        ins.add_patch(Polygon(corners, closed=True, facecolor='.25', alpha=.55, lw=0))
        for j, a in enumerate(ITEM_ANGLES):
            q = ITEM_ECC * np.array([np.cos(np.deg2rad(a)), np.sin(np.deg2rad(a))])
            ins.add_patch(Circle(q, ITEM_R, facecolor='w', ec=HP if j == 0 else '.6', lw=.9 if j == 0 else .5))
        ins.set_xlim(-5, 5)
        ins.set_ylim(-5, 5)
        ins.set_aspect('equal')
        ins.set_axis_off()
        tx = x + w + .015 if side == 'left' else x - .015
        ax.text(tx, y + h / 2, lab, transform=ax.transAxes, fontsize=6, color='.3', ha=side, va='center')


def panel_sweep_slim(ax, source, pred=None):
    """c (v10): the segment differences against the bar centre (outline decoder), no thumbnails: HP quadrant (red),
    next to HP (dark grey) and opposite (light grey, dashed); data points ± SEM, the model as line + band; stars in one
    colour (two-sided Wilcoxon per position, uncorrected). Returns the per-position statistics text."""
    from retsupp.visualize.paper.figS_decoded_bars import pooled_condition
    from retsupp.visualize.paper.paper_style import FG, HP, offset_spines
    z = np.load(DATA / f'decoded_bars_condition_{source}_outline_2026-11.npz')
    P = pooled_condition(z['mean'])
    centres = (z['edges'][:-1] + z['edges'][1:]) / 2
    hp_v, nx_v, op_v = sweep_segments(P, z['grid'], opposite=True)
    ns = len(P)
    series = [(nx_v, LOC_COLORS['adjacent'], 'Next to HP', 'next', '-', 0.),
              (op_v, '#7B8794', 'Opposite', 'opp', (0, (3, 1.5)), -.05), (hp_v, HP, 'HP quadrant', 'hp', '-', .05)]
    ax.axhline(0, color='.85', lw=.6, zorder=0)
    stats_txt = []
    for v, col, lab, key, ls, dx in series:
        ok = np.isfinite(v).all(0)
        m, se = np.nanmean(v, 0), np.nanstd(v, 0, ddof=1) / np.sqrt(ns)
        if pred is not None and key in pred:
            pv = pred[key]
            pm, pse = np.nanmean(pv, 0), np.nanstd(pv, 0, ddof=1) / np.sqrt(np.isfinite(pv).sum(0))
            ax.fill_between(centres[ok], (pm - pse)[ok], (pm + pse)[ok], color=col, alpha=.16, lw=0, zorder=1)
            ax.plot(centres[ok], pm[ok], color=col, lw=1.1, ls=ls, zorder=2)
        ax.errorbar(centres[ok] + dx, m[ok], yerr=se[ok], fmt='o', ms=3, color=col, mec='w', mew=.5, lw=.9, capsize=0,
                    zorder=3)
        for b in np.flatnonzero(ok):
            p_ = stats.wilcoxon(v[:, b]).pvalue
            stats_txt.append(f'{lab} at {centres[b]:+.1f}°: {m[b]:+.2f} ± {se[b]:.2f}%, p = {p_:.3f}')
            st = stars(p_)
            if st:
                up = m[b] > 0
                ax.text(centres[b] + dx, m[b] + (se[b] + .15) * (1 if up else -1), st, color=FG, fontsize=7,
                        ha='center', va='bottom' if up else 'top')
        # direct labels at the line ends, staggered so they never stack
        i = np.flatnonzero(ok)
        if key == 'opp':      # labels inside the plot, in free space next to their own line
            ax.text(centres[i[0]] - .15, -1.9, lab, color=col, fontsize=6.5, ha='left', va='center')
        elif key == 'next':
            ax.text(1.35, 1.75, lab, color=col, fontsize=6.5, ha='center', va='center')
        else:
            ax.text(1.95, -2.3, lab, color=col, fontsize=6.5, ha='center', va='center')
    ax.set_xlim(-3.1, 3.1)
    ax.set_xticks([-2.7, -1.8, -0.9, 0, 0.9, 1.8, 2.7])
    ax.set_xticklabels([f'{t:+.1f}'.replace('-', '−').replace('+0.0', '0') for t in ax.get_xticks()], fontsize=6.5)
    ax.set_ylim(-2.6, 2.4)
    ax.set_yticks([-2, 0, 2])
    ax.set_yticklabels(['−2', '0', '2'])
    ax.tick_params(labelsize=6.5)
    ax.set_xlabel('Bar centre (°)', fontsize=7, labelpad=1)
    ax.set_ylabel('Segment\ndifference (%)', fontsize=7, labelpad=2)
    offset_spines(ax)
    sweep_half_icons_up(ax)
    return '; '.join(stats_txt).replace('-', '−')


def panel_quadrants_v10(ax, s, st, pred=None):
    """d (v10): per quadrant (HP, next to HP, opposite) decoded intensity in % vs the mean of the four; subject dots,
    mean ± SEM (plain SEM over subjects, as row 3); the model as a line + band centred on the data mean."""
    import matplotlib.transforms as mtrans
    from matplotlib.patches import Circle, Wedge
    from retsupp.visualize.paper.paper_style import FG, HP, offset_spines
    cls = [c for c, _ in QUAD_CLASSES]
    rng = np.random.default_rng(3)
    for i, c in enumerate(cls):
        col = LOC_COLORS[c]
        v = s[c]
        ax.scatter(i - .2 + rng.uniform(-.12, .12, len(v)), v, s=6, color=col, alpha=.4, lw=0, zorder=2)
        if pred is not None:
            pm, pse = pred[c].mean(), pred[c].sem()
            ax.fill_between([i + .02, i + .38], pm - pse, pm + pse, color=FG, alpha=.15, lw=0, zorder=1)
            ax.plot([i + .02, i + .38], [pm, pm], color=FG, lw=1.1, zorder=3)
        ax.errorbar(i + .2, v.mean(), yerr=v.sem(), fmt='o', ms=4.5, color=col, mec='w', mew=.7, lw=1.3, capsize=0,
                    zorder=4)
    lo, hi = -5.5, 4.8
    if st['p_adj'] < .05:
        y = 3.8
        ax.plot([.2, .2, 1.2, 1.2], [y - .3, y, y, y - .3], color=FG, lw=.7)
        ax.text(.7, y + .05, stars(st['p_adj']), ha='center', va='bottom', fontsize=7, color=FG)
    ax.set_ylim(lo, hi)
    ax.axhline(0, color='.85', lw=.6, zorder=0)
    ax.set_xlim(-.5, 2.5)
    ax.set_xticks([])
    ax.set_yticks([-4, 0, 4])
    ax.set_yticklabels(['−4', '0', '4'])
    ax.set_ylabel(DIFF_LABEL, fontsize=7)
    tr = mtrans.blended_transform_factory(ax.transData, ax.transAxes)
    wedges = {'hp': [(45, 135)], 'adjacent': [(135, 225), (-45, 45)], 'opposite': [(225, 315)]}
    for i, (c, lab) in enumerate(QUAD_CLASSES):
        ins = ax.inset_axes([i - .3, -.5, .6, .44], transform=tr)
        for a0, a1 in wedges[c]:
            ins.add_patch(Wedge((0, 0), APERTURE, a0, a1, facecolor=LOC_COLORS[c], alpha=.35, lw=0))
        ins.add_patch(Circle((0, 0), APERTURE, fill=False, ec='.55', lw=.5))
        for j, a in enumerate(ITEM_ANGLES):
            q = ITEM_ECC * np.array([np.cos(np.deg2rad(a)), np.sin(np.deg2rad(a))])
            ins.add_patch(Circle(q, ITEM_R, facecolor='w', ec=HP if j == 0 else '.6', lw=.9 if j == 0 else .5))
        ins.set_xlim(-5, 5)
        ins.set_ylim(-5, 5)
        ins.set_aspect('equal')
        ins.set_axis_off()
        ax.text(i, -.54, lab, transform=tr, ha='center', va='top', fontsize=6.5, color=HP if c == 'hp' else '.3')
    offset_spines(ax)


def panel_pathways_v10(ax, per, ast):
    """Row 3: the HP − other quadrants contrast pooled per pathway (Early, Occipitotemporal, Dorsal): subject dots,
    mean ± SEM, stars and exact one-sided p."""
    from retsupp.visualize.paper.paper_style import FG, HP, offset_spines
    rng = np.random.default_rng(5)
    lo, hi = -8, 9
    for i, (k, _) in enumerate(PATHWAY_ORDER):
        d = ast['pooled'][k]
        ax.scatter(i - .17 + rng.uniform(-.14, .14, d['n']), d['v'], s=6, color=HP, alpha=.35, lw=0, zorder=2)
        ax.errorbar(i + .17, d['m'], yerr=d['se'], fmt='o', ms=4.5, color=HP, mec='w', mew=.7, lw=1.3, capsize=0,
                    zorder=4)
        lab = (stars(d['p']) + '\n' if d['p'] < .05 else '\n') + f'p = {d["p"]:.3f}'.replace('0.', '.')
        ax.text(i, hi, lab, ha='center', va='top', fontsize=6.5, color=FG, linespacing=1.)
    ax.axhline(0, color='.85', lw=.6, zorder=0)
    ax.set_ylim(lo, hi)
    ax.set_yticks([-6, -3, 0, 3])
    ax.set_yticklabels(['−6', '−3', '0', '3'])
    ax.set_xlim(-.55, len(PATHWAY_ORDER) - .45)
    ax.set_xticks(range(len(PATHWAY_ORDER)))
    ax.set_xticklabels([f'{k}\n({", ".join(r)})' for k, r in PATHWAY_ORDER], fontsize=6.5)
    ax.tick_params(axis='x', length=0)
    ax.set_ylabel(EFFECT_LABEL, fontsize=7)
    offset_spines(ax)


def panel_quality_v10(ax, qs, pq, rois):
    """b (v10): decoded vs true bar per area; held-out runs (stimulus blue, median and bootstrapped 95% CI over subject
    dots) and frames shuffled (grey, visible); glyph key."""
    from retsupp.visualize.paper.paper_style import FG, STIM, offset_spines, roi_axis
    for i, (roi, x) in enumerate(zip(rois, roi_xs(rois))):
        g = qs[qs.roi == roi]
        if not len(g):
            continue
        subject_dots(ax, x - .12, g.r.to_numpy(), STIM, i, s=5, width=.09, ms=4.2)
        subject_dots(ax, x + .2, g.r_shuffle.to_numpy(), '.45', 10 + i, s=0, ms=3.6)
        ax.text(x, .64, stars(pq[roi]), ha='center', va='bottom', fontsize=6.5, color=FG)
    ax.axhline(0, color='.85', lw=.6, zorder=0)
    ax.set_ylim(-.06, .76)
    ax.set_yticks([0, .2, .4, .6])
    ax.set_yticklabels(['0', '.2', '.4', '.6'])
    ax.set_ylabel('Decoded vs\ntrue bar (r)', fontsize=7)
    tf = ax.transAxes
    for j, (lab, col) in enumerate((('Held-out run', STIM), ('Frames shuffled', '.45'))):
        xx = .47 + j * .25
        ax.plot(xx, 1.08, 'o', transform=tf, ms=3.6, color=col, mec='w', mew=.6, clip_on=False)
        ax.text(xx + .02, 1.08, lab, transform=tf, fontsize=6.5, color=col, ha='left', va='center')
    offset_spines(ax)
    roi_axis(ax, rotate=False, fontsize=6, group_fontsize=6)


LEGEND_V10 = """
**Figure 6. The bar stimulus can be read out of visual cortex, and it is read out at lower intensity in the quadrant facing the HP location.** The bar of every run was decoded from the BOLD activity of one area at a time with pRFs (difference of Gaussians, one HRF per subject and area fitted jointly with the pRFs, no gain terms) fitted on the other runs (cross-validation folds), so the decoded run never entered the pRF fit; voxels with cross-run reliability > 0 on the training runs (at most 2000 per area); the encoding model contains only the bar, not the search items. Throughout, the visual field is rotated so that the HP location lies at 12 o'clock (red ring; grey rings, the LP locations). Points and bars in c (bottom), d and e are the mean ± SEM over subjects (n = {n}); lines and bands in c and d are the gain-field model's prediction (mean ± SEM; see below). a, True bar (top) and decoded intensity in one held-out run ({ex_sub} {ex_run}, V1; frame-wise r = {ex_r:.2f}, rank {ex_rank} of {n_runs} subject × run × area decodes; the median run has r = {r_med:.2f}) and averaged over runs and subjects (V1), at four time points; decoded rows on a magma scale from 0 to each row's maximum; grey circle, the {ap}° aperture of the bar. b, Per subject and area, the spatial correlation between decoded and true frame inside the aperture, averaged over frames with a bar and runs (blue dots), against the same with the order of the true frames shuffled (100 permutations; grey); points and bars, median and bootstrapped 95% CI; decoded vs shuffled, one-sided Wilcoxon signed-rank test (*** p < .001); {quality}. c, V1–V3 pooled. Left, the decoded intensity on the bar's pixels minus its average over the four rotations, in % of that average (diverging scale: brown less, teal more), averaged over runs, V1–V3 and subjects and smoothed (Gaussian, σ = 0.4°); the dotted diagonals bound the quadrant facing HP. Right of it, the same map from the gain-field model simulation (spatial correlation with the data map inside the aperture r = {map_r}). Bottom right, single bar positions without averaging over the field: the decoded bar (decoder with the bar outline given) minus the mean decoded bar at the same time point over the subject's runs (balanced over HP locations), each run first scaled to a common mean intensity on the bar, on the bar's segment in the quadrant facing HP (red), in the quadrants next to HP (dark grey; mean of the segments present) and in the quadrant opposite HP (light grey, dashed), in % of the mean decoded intensity on the bar, against the bar centre (horizontal bars reflected onto vertical ones across the diagonal through HP; positive, toward HP). The icons under the axis show which quadrants a bar left or right of the centre crosses (dark stroke, the bar), so the next-to-HP segment exists along the whole sweep and the other two along one half each. Stars, two-sided Wilcoxon signed-rank test per position, uncorrected (* p < .05, ** p < .01): {sweep_stats}. d, V1–V3 pooled: decoded intensity on the bar's pixels per quadrant of the aperture (free decoder), per subject and area averaged over runs and expressed in % vs the mean of the four quadrants, then averaged over V1–V3: the quadrant facing HP, the two next to it (pooled; they do not differ, two-sided Wilcoxon p = {q_lr:.2f}) and the opposite one; {quad}. The difference does not grow toward the aperture edge, next to the items (Supplementary Fig. S, decoded intensity by eccentricity). e, The contrast of d (HP minus the mean of the three other quadrants, same normalisation) for every area, averaged per subject over the areas of a pathway that the subject has (an area was skipped for a subject when fewer than 20 voxels passed the criteria; dorsal n = {n_dorsal}); one-sided Wilcoxon signed-rank test (HP < other quadrants), uncorrected: {pathways}. Per-area values with Benjamini-Hochberg q are in Supplementary Fig. S (decoded intensity at HP per area); individually, only V2 and V3 reach q < .05.

Gain-field model: per subject and run, the BOLD of V1–V3 was simulated from the subject's joint fit (pRFs fitted together with the gain fields on all runs: sustained HP and LP gains and field width, transient distractor and target gains, search items, one HRF per subject and area) plus the real run's residuals under that fit, phase-randomised (one random phase per frequency for all voxels), and decoded exactly as the data, on the decoder's voxels that are also in the joint fit; a paired simulation with the five gains at 0 and the same noise was decoded alike, and the prediction is the difference per subject, averaged like the data. {pred_numbers}
"""


def plot_v10(name='fig6_decoding_v10', source='jointhrf', decoder=DECODER, prediction='decode_simfit', supp_suffix='',
             ecc_name=None):
    """Condensed Fig. 6: a (4 frames) | b; c maps (data | model) + slim sweep | d quadrants; e pathway-pooled contrast.
    Eccentricity panel and per-area values go to supplementary figures (``figS_decoding_eccentricity``,
    ``figS_decoding_hp_per_area`` with BH q of the d-normalised contrast)."""
    import matplotlib.pyplot as plt
    from retsupp.visualize.paper.paper_style import DOUBLE, FG, HP, ROI_ORDER, STIM, letter, save
    rois = list(ROI_ORDER)
    d = pd.read_csv(DATA / f'decoding_readout_{source}_2026-11.tsv', sep='\t')
    e = effects(d)
    t = test_bh(e, rois, decoder)
    t_out = test_bh(e, rois, 'outline') if 'outline' in set(d.decoder) else None
    q = pd.read_csv(DATA / f'decoding_quality_{source}_2026-11.tsv', sep='\t')
    qs = q[q.decoder == decoder].groupby(['subject', 'roi'])[['r', 'r_shuffle']].mean().reset_index()
    pq = {roi: stats.wilcoxon(g.r - g.r_shuffle, alternative='greater').pvalue for roi, g in qs.groupby('roi')}
    maps = np.load(DATA / f'decoding_maps_{source}_2026-11.npz')
    rm = rel_maps(maps, decoder)
    fr = np.load(DATA / f'decoding_frames_{source}_2026-11.npz')
    ex = np.load(DATA / f'decoding_example_run_{source}_2026-11.npz')
    sub, run, roi = EXAMPLES[source]
    frames = dict(paradigm=fr['paradigm'], group=fr['group'][list(fr['rois']).index('V1'), list(fr['decoders']).index('free')].astype(float),
                  example=ex[f'{sub}_{run}_roi-{roi}'].astype(float))
    n_group = int(fr['n_group']) if 'n_group' in fr.files else 30
    qf = q[q.decoder == 'free'].sort_values('r', ascending=False).reset_index(drop=True)
    hit = (qf.subject == int(sub[4:])) & (qf.session == int(run[4])) & (qf.run == int(run[-1])) & (qf.roi == roi)
    ex_r, ex_rank = qf[hit].r.iloc[0], int(np.flatnonzero(hit)[0]) + 1
    pred = load_prediction(prediction, decoder)
    qe = quadrant_effects(d, decoder)
    qst = quadrant_stats(qe, sem='plain')
    per, pw = pathway_effects(d, decoder)
    ast = area_stats(per, pw)

    W, H = DOUBLE, 6.3
    fig = plt.figure(figsize=(W, H))
    # Row 1
    fig.text(.02 / W, 6.17 / H, 'a', fontsize=8, fontweight='bold', va='center')
    fig.text(.2 / W, 6.17 / H, 'Bar decoded from held-out runs', fontsize=7.5, color=STIM, va='center')
    panel_frames(fig, (.62, 4.38, 2.3, 1.52), frames, W, H, style='magma', times=[15, 28, 72, 89],
                 labels=('Bar', f'Best run\nr = {ex_r:.2f}'.replace('0.', '.'), f'Mean\n(n = {n_group})'))
    ax = fig.add_axes([3.75 / W, 4.72 / H, 3.25 / W, 1.15 / H])
    panel_quality_v10(ax, qs, pq, rois)
    letter(ax, 'b', x=-.14, y=1.04)
    # Row 2
    fig.text(.02 / W, 4.12 / H, 'c', fontsize=8, fontweight='bold', va='center')
    fig.text(.2 / W, 4.12 / H, 'Lower intensity facing HP (V1–V3 pooled)', fontsize=7.5, color=HP, va='center')
    ax = fig.add_axes([.05 / W, 2.62 / H, 1.05 / W, 1.05 / H])
    im = panel_map(ax, rm, tag=False, cmap=MAP_CMAP, diag=('.3', .6, ':'), lp_label=True)
    ax.text(.5, 1.0, 'Data', transform=ax.transAxes, fontsize=6.5, color='.3', ha='center', va='bottom')
    ax2 = fig.add_axes([1.17 / W, 2.62 / H, 1.05 / W, 1.05 / H])
    panel_map(ax2, pred[3]['rm'], label='', tag=False, cmap=MAP_CMAP, diag=('.3', .6, ':'))
    ax2.text(.5, 1.0, 'Gain-field model', transform=ax2.transAxes, fontsize=6.5, color=FG, ha='center', va='bottom')
    mo, ins_ = smoothed_map(rm)
    mm = smoothed_map(pred[3]['rm'])[0]
    ok = ins_ & np.isfinite(mo) & np.isfinite(mm)
    map_r = float(np.corrcoef(mo[ok], mm[ok])[0, 1])
    cax = fig.add_axes([.62 / W, 2.42 / H, 1.0 / W, .06 / H])
    cb = fig.colorbar(im, cax=cax, orientation='horizontal', ticks=[-3, 0, 3])
    cb.ax.set_xticklabels(['−3', '0', '3'], fontsize=6)
    cb.ax.tick_params(length=2, width=.5, pad=1)
    cb.outline.set_linewidth(.4)
    cb.ax.set_xlabel('Decoded intensity (% vs mean)', fontsize=6.5, labelpad=2)
    ax = fig.add_axes([2.95 / W, 2.98 / H, 2.1 / W, .72 / H])
    sweep_stats = panel_sweep_slim(ax, source, pred[2])
    glyph_key(ax, [('Gain-field model', 'line', FG, '-'), ('Data', 'point', FG, None)], x=.0, y=1.13, dx=.42)
    ax = fig.add_axes([5.62 / W, 3.0 / H, 1.42 / W, .7 / H])
    panel_quadrants_v10(ax, qe, qst, pred[0])
    letter(ax, 'd', x=-.36, y=1.08)
    # Row 3
    fig.text(.02 / W, 1.78 / H, 'e', fontsize=8, fontweight='bold', va='center')
    fig.text(.2 / W, 1.78 / H, 'Beyond V1–V3: occipitotemporal, not dorsal', fontsize=7.5, color=HP, va='center')
    ax = fig.add_axes([.72 / W, .5 / H, 2.4 / W, 1.05 / H])
    panel_pathways_v10(ax, per, ast)
    boxes = {}
    for a_ in fig.axes:
        bb = a_.get_window_extent()
        boxes[a_] = (bb.width / fig.dpi, bb.height / fig.dpi)
    print('e axes box', [f'{w:.2f} x {h:.2f}' for a_, (w, h) in boxes.items() if a_ is ax])

    # Caption numbers
    nq = qs.groupby('roi').r.median()
    quality = '; '.join(f'{r} median r = {nq[r]:.2f} vs {qs[qs.roi == r].r_shuffle.median():.3f}, {fmt_p(pq[r])}'
                        for r in rois if r in nq)
    f2 = lambda x: f'{x:+.2f}'.replace('-', '−')   # noqa: E731
    pw_txt = '; '.join(f'{k} ({", ".join(r_)}) {f2(ast["pooled"][k]["m"])} ± {ast["pooled"][k]["se"]:.2f}%, '
                       f'p = {ast["pooled"][k]["p"]:.3f}, {ast["pooled"][k]["below"]:.0f}% of subjects below 0, '
                       f'n = {ast["pooled"][k]["n"]}' for k, r_ in PATHWAY_ORDER)
    pw_txt += (f'; all nine areas {f2(ast["pooled"]["All areas"]["m"])} ± {ast["pooled"]["All areas"]["se"]:.2f}%, '
               f'p = {ast["pooled"]["All areas"]["p"]:.3f}')
    r_data = q[(q.decoder == decoder) & q.roi.isin(EARLY)].groupby('subject').r.mean().median()
    pt = prediction_text(pred[:3] + ({k: v for k, v in pred[3].items() if k != 'map_r'},), qe, None, sweep_data(source),
                         r_data, ecc=False, numbers_only=True)
    leg = LEGEND_V10.format(n=qs.subject.nunique(), ex_sub=sub, ex_run=run.replace('_', ' '), ex_r=ex_r, ex_rank=ex_rank,
                            n_runs=len(qf), r_med=qf.r.median(), ap=APERTURE, quality=quality, map_r=f2(map_r).lstrip('+'),
                            sweep_stats=sweep_stats, q_lr=qst['p_lr'], quad=qst['text'], pathways=pw_txt,
                            n_dorsal=ast['pooled']['Dorsal']['n'], pred_numbers=pt)
    save(fig, name, leg)
    print('map r', round(map_r, 3))
    print('pathways', pw_txt)

    # Supplement: per-area test (unchanged) + the d-normalised per-area values with BH q
    tests = '; '.join(f'{r.roi} median {r["median"]:+.4f}, {r.pct_below:.0f}% of subjects below 0, n = {r.n}, '
                      f'p = {r.p:.3f}, q = {r.q:.3f}' for _, r in t.iterrows())
    outline = ''
    if t_out is not None:
        outline = (' Open markers and the upper row of stars: the decoder with the bar outline given (only the pixels the '
                   'bar covers are decoded, the rest fixed at 0; median and bootstrapped 95% CI), same tests: '
                   + '; '.join(f'{r.roi} median {r["median"]:+.4f}, p = {r.p:.3f}, q = {r.q:.3f}' for _, r in t_out.iterrows()))
        outline += '. Filled markers and the lower row of stars: the unconstrained decoder.'
    i0, i1 = LEGEND.index(' d, Per subject and ROI'), LEGEND.index(' e, Decoded')
    d_text = LEGEND[i0:i1].format(tests=tests, outline=outline)
    d_text += (' With the normalisation of Fig. 6d,e (per subject and area, the decoded intensity per quadrant in % vs '
               'the mean of the four, HP minus the mean of the other three; mean ± SEM; one-sided Wilcoxon, '
               'Benjamini-Hochberg across the nine areas): ' + area_text(per, ast).split('Per area: ')[1] + '.')
    test_figure(e, t, rois, decoder, t_out, d_text, f'figS_decoding_hp_per_area{supp_suffix}')
    plot_eccentricity(ecc_name or f'figS_decoding_eccentricity{supp_suffix}', source, decoder, pred[1])
    return fig


def plot_eccentricity(name, source='jointhrf', decoder=DECODER, pred=None):
    """Supplement (was Fig. 6e): HP − other quadrants by eccentricity, with the gain-field model."""
    import matplotlib.pyplot as plt
    from retsupp.visualize.paper.paper_style import DOUBLE, save
    bands = pd.read_csv(DATA / f'decoding_bands_{source}_2026-11.tsv.gz', sep='\t')
    be = band_effects(bands, decoder)
    pb = {k: stats.wilcoxon(be[k], alternative='less').pvalue for k in ['all'] + [b[0] for b in BANDS]}
    fried = stats.friedmanchisquare(*[be[b[0]] for b in BANDS]).pvalue
    oi = stats.wilcoxon(be['2.5-ap'] - be['0-1.5']).pvalue
    fov = be[['ring_0.00', 'ring_0.25', 'ring_0.50']].mean(1)
    W, H = DOUBLE * .62, 1.9
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([.75 / W, .42 / H, (W - 1.0) / W, 1.3 / H])
    panel_ecc(ax, be, pb, pred=pred)
    area_tag(ax, x=.02, y=.3)
    f2 = lambda x: f'{x:+.1f}'.replace('-', '−')   # noqa: E731
    band_txt = '; '.join(f'{lab} {f2(be[k].mean())} ± {be[k].sem():.1f}%, p = {pb[k]:.3f}' for k, a, b, lab in BANDS)
    band_txt += f'; whole aperture {f2(be["all"].mean())} ± {be["all"].sem():.1f}%, p = {pb["all"]:.3f}'
    pm = pred['all'] if pred is not None else None
    legend = (f"**Figure S. The decoded difference at HP does not grow toward the items (Fig. 6, by eccentricity).** "
              f"Decoding as in Fig. 6 (V1–V3 pooled, unconstrained decoder). Per subject and area, the decoded intensity on "
              f"the bar's pixels in the quadrant facing HP minus that in the three other quadrants, in % of the latter, per "
              f"ring of 0.25° eccentricity, mean of V1–V3 per subject; points and bars, mean ± SEM over subjects (n = "
              f"{len(be)}); line and band, the gain-field model's prediction (as in Fig. 6; whole aperture "
              f"{f2(pm.mean()) if pm is not None else 'n/a'} ± {pm.sem():.1f}%), in the three bands shaded. Items (radius "
              f"{ITEM_R}°) begin at {ITEM_ECC - ITEM_R:.2f}° eccentricity, just outside the aperture, and their responses "
              f"are in the BOLD data but not in the encoding model; if item responses leaking into the aperture produced "
              f"the difference, it would grow toward the aperture edge. It does not: per band (one-sided Wilcoxon "
              f"signed-rank test, uncorrected, * p < .05, ** p < .01) {band_txt}; the three bands do not differ (Friedman "
              f"test, p = {fried:.2f}; outer minus inner band, two-sided Wilcoxon p = {oi:.2f}). Within 0.75° of the "
              f"fovea there is no difference ({f2(fov.mean())} ± {fov.sem():.1f}%).")
    save(fig, name, legend)


def plot(name='fig6_decoding', source='canonical', rois=ROIS, decoder=DECODER, source_note='', frames_style='magma', v3=False,
         sweep=False, prediction=None, pathways=False, sweep_opposite=False):
    """v3: 180 mm wide; the per-area test (old d) goes to its own supplementary figure, the rest re-lettered a-e."""
    import matplotlib.pyplot as plt
    from retsupp.visualize.paper.paper_style import FG, HP, STIM, letter, save
    d = pd.read_csv(DATA / f'decoding_readout_{source}_2026-11.tsv', sep='\t')
    e = effects(d)
    t = test_bh(e, rois, decoder)
    t_out = test_bh(e, rois, 'outline') if 'outline' in set(d.decoder) else None
    print(t.round(4).to_string(index=False))
    if t_out is not None:
        print('outline', t_out.round(4).to_string(index=False))
    print(t.round(4).to_string(index=False))

    q = pd.read_csv(DATA / f'decoding_quality_{source}_2026-11.tsv', sep='\t')
    qs = q[q.decoder == decoder].groupby(['subject', 'roi'])[['r', 'r_shuffle'] + [f'r_lag_{k}' for k in range(-3, 4)]].mean().reset_index()
    pq = {roi: stats.wilcoxon(g.r - g.r_shuffle, alternative='greater').pvalue for roi, g in qs.groupby('roi')}

    bands = pd.read_csv(DATA / f'decoding_bands_{source}_2026-11.tsv.gz', sep='\t')
    be = band_effects(bands, decoder)
    pb = {k: stats.wilcoxon(be[k], alternative='less').pvalue for k in ['all'] + [b[0] for b in BANDS]}
    fried = stats.friedmanchisquare(*[be[b[0]] for b in BANDS]).pvalue
    oi = stats.wilcoxon(be['2.5-ap'] - be['0-1.5']).pvalue
    fov = be[['ring_0.00', 'ring_0.25', 'ring_0.50']].mean(1)

    maps = np.load(DATA / f'decoding_maps_{source}_2026-11.npz')
    rm = rel_maps(maps, decoder)
    fr = np.load(DATA / f'decoding_frames_{source}_2026-11.npz')
    ex = np.load(DATA / f'decoding_example_run_{source}_2026-11.npz')
    sub, run, roi = EXAMPLES[source]
    frames = dict(paradigm=fr['paradigm'], group=fr['group'][list(fr['rois']).index('V1'), list(fr['decoders']).index('free')].astype(float),
                  example=ex[f'{sub}_{run}_roi-{roi}'].astype(float))
    if 'n_group' in fr.files:
        frames['n_group'] = int(fr['n_group'])
    qf = q[q.decoder == 'free'].sort_values('r', ascending=False).reset_index(drop=True)
    hit = (qf.subject == int(sub[4:])) & (qf.session == int(run[4])) & (qf.run == int(run[-1])) & (qf.roi == roi)
    ex_r, ex_rank = qf[hit].r.iloc[0], int(np.flatnonzero(hit)[0]) + 1
    frames['ex_label'] = f'{sub}, {roi}\nr = {ex_r:.2f}'.replace('0.', '.')
    print('top runs', qf.head(5)[['subject', 'session', 'run', 'roi', 'r']].to_string(index=False))

    bp = load_bar_pass()
    bst = bar_pass_stats(bp)
    pred = load_prediction(prediction, decoder) if prediction else None

    from retsupp.visualize.paper.paper_style import DOUBLE
    Y4 = 1.6 if pathways == 'areas' else 0.        # '--pathways areas': f gets a row of its own at the bottom
    W, H = (DOUBLE, 6.3 + Y4) if v3 else (7.25, 6.3 + Y4)
    DY = .55 + Y4                                 # rows 1-2 sit above the taller row 3
    fig = plt.figure(figsize=(W, H))
    # Row 1: a frames, b quality.
    many = len(rois) > 3
    panel_frames(fig, (.75, 3.75 + DY, 3.3 if many else 4.0, 1.6), frames, W, H, style=frames_style)
    fig.text(.2 / W, (5.62 + DY) / H, 'The bar stimulus can be decoded from held-out runs' +
             ('' if many else ' in ' + ', '.join(rois[:-1]) + f' and {rois[-1]}'), fontsize=7.5, color=STIM, va='center')
    fig.text(.02 / W, (5.62 + DY) / H, 'a', fontsize=8, fontweight='bold', va='center', ha='left')
    if many and v3:
        ax = fig.add_axes([4.9 / W, (4.08 + DY) / H, 2.08 / W, 1.2 / H])
    elif many:
        ax = fig.add_axes([4.95 / W, (4.08 + DY) / H, 2.2 / W, 1.2 / H])
    else:
        ax = fig.add_axes([5.55 / W, (3.85 + DY) / H, 1.6 / W, 1.42 / H])
    panel_quality(ax, qs, pq, rois)
    letter(ax, 'b', x=-.27 if many else -.42, y=1.08)

    # Row 2: c map, d angle, e the test.
    fig.text(.2 / W, (3.4 + DY) / H, 'The bar is decoded at lower intensity in the quadrant facing the HP location', fontsize=7.5,
             color=HP, va='center')
    two = bool(pred) and 'rm' in pred[3]
    if two:   # data and gain-field model circles side by side, the sweep shifted right
        ax = fig.add_axes([.05 / W, (1.86 + DY) / H, 1.2 / W, 1.2 / H])
        im = panel_map(ax, rm, tag=False)
        area_tag(ax, x=0., y=.1)
        ax.text(.5, 1.0, 'Data', transform=ax.transAxes, fontsize=6.5, color='.3', ha='center', va='bottom')
        ax2 = fig.add_axes([1.33 / W, (1.86 + DY) / H, 1.2 / W, 1.2 / H])
        panel_map(ax2, pred[3]['rm'], label='', tag=False)
        ax2.text(.5, 1.0, 'Gain-field model', transform=ax2.transAxes, fontsize=6.5, color=FG, ha='center',
                 va='bottom')
        mo, mm = smoothed_map(rm)[0], smoothed_map(pred[3]['rm'])[0]
        ins = smoothed_map(rm)[1]
        ok = ins & np.isfinite(mo) & np.isfinite(mm)
        pred[3]['map_r'] = float(np.corrcoef(mo[ok], mm[ok])[0, 1])
        print('map correlation predicted vs observed', round(pred[3]['map_r'], 3))
    else:
        ax = fig.add_axes([.1 / W, (1.78 + DY) / H, 1.36 / W, 1.36 / H])
        im = panel_map(ax, rm)
    ins_file = DATA / f'decoded_bars_condition_{source}_outline_2026-11.npz'
    sweep_out = None
    if sweep:  # the whole sweep of bar positions, images aligned above the segment differences
        sweep_out = bar_sweep(fig, source, *((2.95, 6.5) if two else (2.05, 6.55)), 2.72 + DY, 1.98 + DY, .52, W, H,
                              pred=pred[2] if pred else None, size=.48 if two else .56, opposite=sweep_opposite)
        ins_lab = None
    elif v3:     # without the per-area panel the insets get room: larger, labels at 7 pt
        ins_lab = bar_insets(fig, source, (2.0, 1.98 + DY), W, H, size=1.12, gap=.5, fs=7) if ins_file.exists() else None
    else:
        ins_lab = bar_insets(fig, source, (1.5, 2.2 + DY), W, H, gap=.06) if ins_file.exists() else None
    fig.text(.02 / W, (3.4 + DY) / H, 'c', fontsize=8, fontweight='bold', va='center', ha='left')
    cax = fig.add_axes([((.86 if two else .38) / W), (1.68 + DY) / H, .86 / W, .06 / H])
    cb = fig.colorbar(im, cax=cax, orientation='horizontal', ticks=[-3, 0, 3])
    cb.ax.set_xticklabels(['−3%', '0', '3%'], fontsize=6)
    cb.ax.tick_params(length=2, width=.5, pad=1)
    cb.outline.set_linewidth(.4)
    cb.ax.text(-.06, .5, 'Less', transform=cb.ax.transAxes, ha='right', va='center', fontsize=6, color='#2166AC')
    cb.ax.text(1.06, .5, 'More', transform=cb.ax.transAxes, ha='left', va='center', fontsize=6, color='#B2182B')
    qe = quadrant_effects(d, decoder)
    qst = quadrant_stats(qe)
    if not v3:
        if many:
            ax = fig.add_axes([4.35 / W, (2.05 + DY) / H, 2.15 / W, .95 / H])
        else:
            ax = fig.add_axes([3.4 / W, (2.0 + DY) / H, 2.2 / W, 1.1 / H])
        panel_test(ax, e, t, rois, decoder, t_out)
        letter(ax, 'd', x=-.2 if many else -.42, y=1.15)

    # Row 3: e quadrants, f eccentricity.
    fig.text(.2 / W, (1.93 + Y4) / H, 'Weaker than the quadrants next to it, and not driven by the items outside the aperture',
             fontsize=7.5, color=HP, va='center')
    if pathways == 'areas':
        fig.text(.2 / W, 1.5 / H, 'Pooled over areas also beyond V1–V3 (occipitotemporal); per area reliable in V2 and V3',
                 fontsize=7.5, color=HP, va='center')
        fig.text(.02 / W, 1.5 / H, 'f', fontsize=8, fontweight='bold', va='center', ha='left')
    elif pathways:
        fig.text(5.45 / W, 1.93 / H, 'Also in occipitotemporal areas', fontsize=7.5, color=HP, va='center')
    qe = quadrant_effects(d, decoder)
    qst = quadrant_stats(qe)
    ax = fig.add_axes([.7 / W, (.72 + Y4) / H, 1.25 / W, .9 / H])
    panel_quadrants(ax, qe, qst, pred=pred[0] if pred else None)
    area_tag(ax, x=1.0, y=1.0, ha='right')
    fig.text(.02 / W, (1.93 + Y4) / H, 'd' if v3 else 'e', fontsize=8, fontweight='bold', va='center', ha='left')
    ew = 2.25 if pathways is True else 3.9 if v3 else 3.95
    ax = fig.add_axes([2.95 / W, (.42 + Y4) / H, ew / W, 1.2 / H])
    panel_ecc(ax, be, pb, pred=pred[1] if pred else None, compact=pathways is True)
    area_tag(ax, x=.02, y=.3)
    letter(ax, 'e' if v3 else 'f', x=-.17 * (3.9 / ew), y=1.02)
    pw_txt = ''
    if pathways == 'areas':
        per, pw = pathway_effects(d, decoder)
        ast = area_stats(per, pw)
        ax = fig.add_axes([.75 / W, .5 / H, 5.6 / W, .85 / H])
        panel_areas(ax, per, pw, ast)
        bb = ax.get_window_extent()
        print(f'f axes box {bb.width / fig.dpi:.2f} x {bb.height / fig.dpi:.2f} in (w x h)')
        pw_txt = area_text(per, ast)
        print('areas', pw_txt)
    elif pathways:
        per, pw = pathway_effects(d, decoder)
        pst = pathway_stats(per, pw)
        ax = fig.add_axes([5.75 / W, .52 / H, 1.25 / W, 1.0 / H])
        panel_pathways(ax, per, pw, pst)
        bb = ax.get_window_extent()
        print(f'f axes box {bb.width / fig.dpi:.2f} x {bb.height / fig.dpi:.2f} in (w x h)')
        letter(ax, 'f', x=-.3, y=1.02)
        fmt_pw = lambda k: (f'{k} {pst[k]["m"]:+.2f} ± {pst[k]["se"]:.2f}%, p = {pst[k]["p"]:.3f}, '   # noqa: E731
                            f'{pst[k]["below"]:.0f}% of subjects below 0, n = {pst[k]["n"]}').replace('-', '−')
        pw_txt = '; '.join(fmt_pw(k) for k in ('Early', 'Dorsal', 'Occipitotemporal')) + '; all higher areas (dorsal and ' \
            'occipitotemporal) ' + fmt_pw('Higher').split(' ', 1)[1] + '; per area ' + ', '.join(
                f'{r} {per[r].mean():+.2f}% (n = {per[r].count()})' for r in per).replace('-', '−')
        print('pathways', pw_txt)

    nq = qs.groupby('roi').r.median()
    quality = '; '.join(f'{r} median r = {nq[r]:.2f} vs {qs[qs.roi == r].r_shuffle.median():.3f}, {fmt_p(pq[r])}'
                        for r in rois if r in nq)
    tests = '; '.join(f'{r.roi} median {r["median"]:+.4f}, {r.pct_below:.0f}% of subjects below 0, n = {r.n}, '
                      f'p = {r.p:.3f}, q = {r.q:.3f}' for _, r in t.iterrows())
    outline = ''
    if t_out is not None:
        outline = (' Open markers and the upper row of stars: the decoder with the bar outline given (only the pixels the '
                   'bar covers are decoded, the rest fixed at 0; median and bootstrapped 95% CI), same tests: '
                   + '; '.join(f'{r.roi} median {r["median"]:+.4f}, p = {r.p:.3f}, q = {r.q:.3f}' for _, r in t_out.iterrows()))
        outline += '. Filled markers and the lower row of stars: the unconstrained decoder.'
    band_txt = '; '.join(f'{lab} {be[k].mean():+.1f} ± {be[k].sem():.1f}%, p = {pb[k]:.3f}' for k, a, b, lab in BANDS)
    band_txt += f'; whole aperture {be["all"].mean():+.1f} ± {be["all"].sem():.1f}%, p = {pb["all"]:.3f}'
    fovea = f'{fov.mean():+.1f} ± {fov.sem():.1f}%'
    r_, b_ = bst['raw'], bst['base']
    bar_txt = (f'{r_["diff"]:+.3f}% signal ({r_["rel"]:+.1f}% of the LP response), {r_["below"]:.0f}% of subjects below 0, '
               f'one-sided Wilcoxon p = {r_["p"]:.3f} (vs the two neighbouring LP locations {r_["rel_adjacent"]:+.1f}%, '
               f'p = {r_["p_adjacent"]:.3f}; vs the opposite one {r_["rel_opposite"]:+.1f}%, p = {r_["p_opposite"]:.3f})')
    bar_base = f'{b_["rel"]:+.1f}%, p = {b_["p"]:.2f}'
    n_roi = qs.groupby('roi').subject.nunique()
    if many:
        higher = [r for r in rois if r not in EARLY]
        areas = ('The higher areas (' + ', '.join(higher) + ') were decoded the same way; an area was skipped for a '
                 'subject when fewer than 20 voxels passed the criteria (n subjects: ' +
                 ', '.join(f'{r} {n_roi.get(r, 0)}' for r in rois) + ').')
    else:
        areas = 'Only V1–V3 are shown.'
    hrf = {'canonical': 'canonical HRF', 'jointhrf': 'one HRF per subject and ROI, fitted jointly with the pRFs',
           'dn_roihrf': 'one HRF per subject and ROI, the median of the training fits\' per-voxel HRFs'}.get(source, source)
    prf_model = 'divisive normalization' if source.startswith('dn') else 'difference of Gaussians'
    if name.startswith('figS') and source.startswith('dn'):
        title = ('**Figure S. Decoding with divisive-normalization pRFs (robustness check of Fig. 6).** Everything as in '
                 'Fig. 6, but the decoder\'s pRFs are divisive-normalization pRFs with one HRF per subject and ROI (the '
                 'median of the per-voxel HRFs of the training fits) instead of difference-of-Gaussians pRFs.')
    elif name.startswith('figS'):
        title = (f'**Figure S. Decoding with pRFs that use the canonical HRF (robustness check of Fig. 6).** Everything as '
                 f'in Fig. 6, but the decoder\'s pRFs use one canonical HRF instead of an HRF per subject and ROI.')
    else:
        title = ('**Figure 6. The bar stimulus can be read out of visual cortex, and it is read out at lower intensity in '
                 'the quadrant facing the HP location.**')
    leg = LEGEND.format(title=title, areas=areas, hrf=hrf, prf_model=prf_model, n=qs.subject.nunique(), quality=quality, tests=tests, outline=outline,
                                  bands=band_txt, friedman=fried, outer_inner=oi, fovea=fovea, ex_sub=sub, ex_r=ex_r,
                                  ex_rank=ex_rank, n_runs=len(qf), r_med=qf.r.median(), ex_run=run.replace('_', ' '),
                                  ap=APERTURE, item_r=ITEM_R, insets=(SWEEP_TXT.format(nb=sweep_out[0], sweep_stats=sweep_out[1]) if sweep_out else INSET_TXT.format(ins_far=ins_lab[0][0], ins_mid=ins_lab[0][1], ins_hp=ins_lab[0][2], ins_seg='; '.join(f'{k} segment {m:+.2f} ± {se:.2f}%, two-sided Wilcoxon p = {p:.3f}' for k, m, se, p in ins_lab[1])) if ins_lab else ''), item_in=ITEM_ECC - ITEM_R, bar=bar_txt, bar_base=bar_base,
                                  w0=BAR_WINDOW[0], w1=BAR_WINDOW[1], source=source_note,
                                  q_lr=qst['p_lr'], quad=qst['text'])
    if sweep_opposite:
        old = 'in the two quadrants next to HP (grey; mean of the segments present)'
        assert old in leg
        leg = leg.replace(old, 'in the two quadrants next to HP (dark grey; mean of the segments present) and in the '
                          'quadrant opposite HP (light grey, dashed; only bars left of the centre reach it). A bar left of '
                          'the centre crosses the quadrants next to HP and opposite it, a bar right of the centre the '
                          'quadrant facing HP and the other quadrant next to it (icons under the axis, HP ringed red; the '
                          'colours are those of d), so the next-to-HP line runs across the whole sweep and the other two '
                          'only over one half each')
    if pathways == 'areas':
        leg = leg.rstrip() + AREA_TXT.format(pw=pw_txt) + '\n'
    elif pathways:
        leg = leg.rstrip() + PATHWAY_TXT.format(pw=pw_txt) + '\n'
    if pred:
        r_data = q[(q.decoder == decoder) & q.roi.isin(EARLY)].groupby('subject').r.mean().median()
        pt = prediction_text(pred, qe, be, sweep_data(source), r_data)
        leg = leg.rstrip() + ' ' + pt + '\n'
        print('prediction', pt)
    if v3:
        i0 = leg.index(' d, Per subject and ROI, the mean decoded intensity')
        i1 = leg.index(' e, Decoded')
        d_text = leg[i0:i1]
        leg = (leg[:i0] + ' Per area, the same contrast is in Supplementary Fig. S (decoded intensity at HP per area).'
               + leg[i1:]).replace(' e, Decoded', ' d, Decoded').replace(
            'f, The difference of d by eccentricity', 'e, The quadrant facing HP minus the three other quadrants, by eccentricity')
        save(fig, name, leg)
        test_figure(e, t, rois, decoder, t_out, d_text, name.replace('fig6_decoding', 'figS_decoding_hp_per_area'))
    else:
        save(fig, name, leg)
    print('quadrants', qst['text'], 'left vs right', round(qst['p_lr'], 3))
    print('quality', quality)
    print('tests', tests)
    print('outline', outline)
    print('lag medians', qs.groupby('roi')[[f'r_lag_{k}' for k in range(-3, 4)]].median().round(3).to_string())
    print('bands', band_txt, 'friedman', round(fried, 3), 'outer-inner', round(oi, 3), 'fovea', fovea)
    print('bar pass', bar_txt, '| with baseline', bar_base)
    return t


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('step', choices=['collect', 'plot', 'plot_bar_crossing', 'plot_all_rois', 'plot_sweep_per_area',
                                     'plot_all_rois_examples'])
    ap.add_argument('--bids-folder', default='/shares/zne.uzh/gdehol/ds-retsupp')
    ap.add_argument('--source', default='jointhrf', help='jointhrf (main Fig. 6) or canonical (figS_decoding_canonical)')
    ap.add_argument('--rois', nargs='+', default=['V1', 'V2', 'V3', 'V3AB', 'IPS', 'hV4', 'VO', 'LO', 'TO'],
                    help='areas of b and e (c, d, f, g pool V1–V3)')
    ap.add_argument('--decoder', default=DECODER)
    ap.add_argument('--out', default=None)
    ap.add_argument('--name', default='fig6_decoding')
    ap.add_argument('--supp-suffix', default='', help='appended to the names of the plot_all_rois / _examples / bar_crossing figures')
    ap.add_argument('--frames-style', default='magma', choices=['blue', 'magma'], help='decoded rows of a: magma on black or blue')
    ap.add_argument('--v3', action='store_true', help='180 mm wide; the per-area test to a supplementary figure; a-e')
    ap.add_argument('--sweep', action='store_true', help='c: every bar position with its segment differences, not three insets')
    ap.add_argument('--pathways', nargs='?', const=True, default=False,
                    help="f: the HP − other quadrants contrast of d per pathway; '--pathways areas': pooled columns "
                         "first, then every area (BH across the nine)")
    ap.add_argument('--v10', action='store_true', help='condensed Fig. 6 (plot_v10) and its supplements')
    ap.add_argument('--ecc-name', default=None, help='v10: name of the eccentricity supplement')
    ap.add_argument('--sweep-opposite', action='store_true',
                    help='c: the opposite-quadrant segment as a third line and icons of the quadrants each half crosses')
    ap.add_argument('--prediction', default=None,
                    help='file prefix of the gain-field model prediction (collect_simulate_decode.py --fitted), e.g. '
                         'decode_simfit: drawn over c (sweep), d and e')
    a = ap.parse_args()
    if a.step == 'collect':
        collect(a.bids_folder, a.source, a.out)
    elif a.step == 'plot_all_rois_examples':
        plot_all_rois_examples(name='figS_decoding_all_rois_examples' + a.supp_suffix)
    elif a.step == 'plot_sweep_per_area':
        plot_sweep_per_area(name=a.name, source=a.source)
    elif a.step == 'plot_all_rois':
        plot_all_rois(name='figS_decoding_all_rois' + a.supp_suffix)
    elif a.step == 'plot_bar_crossing':
        plot_bar_crossing(name='figS_bar_crossing_response' + a.supp_suffix, source_note='')
    elif a.v10:
        plot_v10(a.name, a.source, a.decoder, prediction=a.prediction or 'decode_simfit', supp_suffix=a.supp_suffix,
                 ecc_name=a.ecc_name)
    else:
        plot(a.name, a.source, a.rois, a.decoder, source_note=f'pRFs: {a.source} fold fits.', frames_style=a.frames_style, v3=a.v3, sweep=a.sweep,
             prediction=a.prediction, pathways=a.pathways, sweep_opposite=a.sweep_opposite)
