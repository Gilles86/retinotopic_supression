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

    python -m retsupp.visualize.paper.fig6_decoding collect --source canonical \\
        --out /shares/zne.uzh/gdehol/ds-retsupp/derivatives/paper_2026-11/decoding_readout_canonical.tsv
    python -m retsupp.visualize.paper.fig6_decoding plot --source canonical
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
EXAMPLE = ('sub-07', 'ses-1_run-3', 'V1')    # the clearest of the top runs by r (inspected visually)
FRAMES = [15, 21, 28, 72, 81, 89, 130]      # TRs shown in a (bar left → right, top → bottom, then blank)
TR = 1.6
APERTURE, ITEM_ECC, ITEM_R = 3.17, 4.0, .75
BANDS = [('0-1.5', 0, 1.5, '0–1.5°'), ('1.5-2.5', 1.5, 2.5, '1.5–2.5°'), ('2.5-ap', 2.5, APERTURE, '2.5°+')]
ITEM_ANGLES = (90, 180, 270, 0)              # HP first: after rotation HP sits at 12 o'clock
BAND_SHADE = ['#E8ECF0', '#D5DCE3', '#C2CBD5']
LEGEND = """
**Figure 6. The bar stimulus can be read out of early visual cortex, and it is read out at lower intensity in the quadrant
facing the HP location.** The bar stimulus of every run was decoded from the BOLD activity of V1, V2 or V3 with pRFs (difference
of Gaussians, canonical HRF, no gain terms) fitted on the other runs (cross-validation folds), so the decoded run never
entered the pRF fit; voxels with cross-run reliability > 0 on the training runs (at most 2000 per ROI); every pixel of
every frame decoded; the encoding model contains only the bar, not the search items. Only V1–V3 were decoded: reading
out a bar needs small pRFs, which is why the pre-specified test is in these three areas; the higher areas (V3AB, IPS,
hV4, VO, LO, TO) have not been decoded. a, True bar (top) and decoded intensity in a single held-out run (middle;
{ex_sub} {ex_run}, V1, frame-wise r = {ex_r:.2f}: rank {ex_rank} of {n_runs} subject × run × ROI decodes, chosen as the
clearest of the five best; the median run has r = {r_med:.2f}) and averaged over all runs and subjects (V1; bottom), at
seven time points of the run (the last one without a bar). Grey circle, the {ap}° aperture of the bar. b, Per subject,
the spatial correlation between decoded and true frame inside the aperture, averaged over the frames with a bar and over
runs (blue dots, n = {n} subjects), against the same with the order of the true frames shuffled (100 permutations; grey).
Points and bars: median and bootstrapped 95% CI. Decoded vs shuffled: Wilcoxon signed-rank test, one-sided (*** p <
.001); {quality}. The correlation peaks at lag 0 in all three ROIs, so the decoded time course is not displaced. c, Each
run's visual field rotated so that the HP location (red dashed circle; grey, the LP locations, drawn with the item
footprint, radius {item_r}°) lies at 12 o'clock (90° steps, then 45° with bilinear interpolation); shown is the decoded
intensity on the bar's pixels minus its average over the four rotations (in % of that average), averaged over runs,
V1–V3 and subjects, smoothed (Gaussian, σ = 0.4°). The dotted diagonals bound the quadrant facing HP. d, The same as a
function of direction from fixation, measured clockwise from the HP direction (bins of 22.5°), whole aperture; circles
under the axis mark the item directions (red, HP; grey, LP); shaded red, the quadrant facing HP (the inset display shows
it). Line and band: mean ± SEM over subjects. e, The pre-specified test: per subject and ROI, the mean decoded intensity
on the bar's pixels in the quadrant facing HP minus the mean over the three other quadrants, averaged over runs; dots are
subjects, points and bars the median and bootstrapped 95% CI. Wilcoxon signed-rank test, one-sided (HP < other
quadrants), Holm across the three ROIs; * p_Holm < .05. {tests}.{outline} f, The difference of e by eccentricity, in %
of the decoded intensity in the other quadrants at the same eccentricity, mean of V1–V3 per subject; line and band, mean ± SEM over
subjects (rings of 0.25°); left, the three bands drawn on the display. Items (radius {item_r}°) begin at {item_in:.2f}°
eccentricity, just outside the aperture, and their responses are in the BOLD data but not in the encoding model; if item
responses leaking into the aperture produced the difference, it would grow toward the aperture edge. It does not: per
band (mean ± SEM; one-sided Wilcoxon signed-rank test, uncorrected, * p < .05) {bands}; the three bands do not differ
(Friedman test, p = {friedman:.2f}; outer minus inner band, two-sided Wilcoxon p = {outer_inner:.2f}). Within 0.75° of
the fovea, where the four quadrants meet, there is no difference ({fovea}). g, Without a model: the measured response
(cleaned BOLD, % signal) of V1–V3 voxels with cross-run reliability > 0 and a pRF centre within 2° of an item location,
locked to the moment the bar centre crosses the pRF centre along the bar's direction of motion, by where the voxel's
location lies relative to the run's HP location (inset: red, HP; dark grey, the two neighbouring LP locations; light grey,
the opposite one); averaged over passes and voxels per subject, ROIs pooled weighted by voxel count; lines and bands,
mean and within-subject SEM (Cousineau-Morey) over subjects. Bottom, HP minus the mean of the three LP locations, mean ±
SEM. In {w0:.1f}–{w1:.1f} s (grey): {bar}. Search items are not removed and the response before the crossing is already
rising (the bar enters the pRF earlier), so no pre-crossing baseline is subtracted; with one (−6.4 to 0 s) the
difference is {bar_base}. {source}
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


def panel_frames(fig, box, frames, W, H):
    """a: rows true bar, one decoded run, group mean; columns time points."""
    import matplotlib.colors as mcolors
    from retsupp.visualize.paper.paper_style import FG, STIM
    x0, y0, w, h = box
    n = len(FRAMES)
    size = min(w / n * .92, h / 3 * .92)
    gapx = (w - n * size) / (n - 1)
    gapy = (h - 3 * size) / 2
    cm = mcolors.LinearSegmentedColormap.from_list('stim', ['#FFFFFF', STIM])
    rows = [('Bar', frames['paradigm'].astype(float), 1.),
            (f'Best run\n{frames["ex_label"]}', frames['example'], np.nanpercentile(frames['example'], 99.7)),
            ('Mean of 30\nsubjects (V1)', frames['group'], np.nanpercentile(frames['group'], 99.7))]
    axes = []
    for i, (lab, arr, vmax) in enumerate(rows):
        for k, t in enumerate(FRAMES):
            ax = fig.add_axes([(x0 + k * (size + gapx)) / W, (y0 + h - (i + 1) * size - i * gapy) / H, size / W, size / H])
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


def panel_quality(ax, qs, pq, rois=ROIS):
    from retsupp.visualize.paper.paper_style import FG, MUTED, STIM, offset_spines
    for i, roi in enumerate(rois):
        g = qs[qs.roi == roi]
        subject_dots(ax, i - .14, g.r.to_numpy(), STIM, i)
        subject_dots(ax, i + .2, g.r_shuffle.to_numpy(), MUTED, 10 + i, s=5, width=.07, ms=4)
        ax.text(i - .14, .7, stars(pq[roi]), ha='center', va='bottom', fontsize=7, color=STIM)
    ax.axhline(0, color='.6', lw=.7, ls='--', zorder=0)
    ax.set_xticks(range(len(rois)))
    ax.set_xticklabels(rois)
    ax.set_xlim(-.5, len(rois) - .5)
    ax.set_ylim(-.06, .7)
    ax.set_yticks([0, .2, .4, .6])
    ax.set_ylabel('Decoded vs true bar (r)')
    ax.text(len(rois) - .55, .2, 'Held-out run', color=STIM, fontsize=6.5, ha='right', va='center')
    ax.text(len(rois) - .55, .08, 'Frames shuffled', color='.45', fontsize=6.5, ha='right', va='center')
    offset_spines(ax)


def items(ax, hp_color, lw=.8):
    from matplotlib.patches import Circle
    for k, a in enumerate(ITEM_ANGLES):
        p = ITEM_ECC * np.array([np.cos(np.deg2rad(a)), np.sin(np.deg2rad(a))])
        ax.add_patch(Circle(p, ITEM_R, fill=False, ec=hp_color if k == 0 else '.55', lw=lw, ls=(0, (2.5, 1.2)),
                            clip_on=False))


def panel_map(ax, rm, vmax=3):
    from retsupp.visualize.paper.paper_style import HP
    c = np.linspace(-5, 5, 50)
    X, Y = np.meshgrid(c, c)
    inside = np.hypot(X, Y) <= APERTURE
    sm = np.array([ndimage.gaussian_filter(np.where(inside, np.nan_to_num(p), 0), 2) /
                   ndimage.gaussian_filter(inside.astype(float), 2) for p in rm])
    im = disc(ax, np.nanmean(sm, 0), 'RdBu_r', vmax, -vmax)
    for a in (45, 135):
        t = np.deg2rad(a)
        ax.plot([-APERTURE * np.cos(t), APERTURE * np.cos(t)], [-APERTURE * np.sin(t), APERTURE * np.sin(t)],
                color='.55', lw=.4, ls=':')
    items(ax, HP)
    ax.set_xlim(-4.95, 4.95)
    ax.set_ylim(-4.95, 4.95)
    ax.set_aspect('equal')
    ax.text(ITEM_R + .25, ITEM_ECC, 'HP', color=HP, fontsize=6.5, ha='left', va='center')
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


def panel_test(ax, e, t, rois=ROIS, decoder=DECODER):
    from retsupp.visualize.paper.paper_style import HP, offset_spines
    lo, hi = -.0132, .008
    for i, roi in enumerate(rois):
        v = e[(e.decoder == decoder) & (e.roi == roi)]['diff'].to_numpy()
        subject_dots(ax, i, v, HP, i)
        n_lo = (v < lo).sum()
        if n_lo:
            ax.scatter(np.full(n_lo, i), np.full(n_lo, lo), s=10, marker='v', color=HP, alpha=.6, lw=0, clip_on=False)
        q = t[(t.decoder == decoder) & (t.roi == roi)].p_holm.iloc[0]
        ax.text(i, hi, stars(q), ha='center', va='bottom', fontsize=7, color=HP)
    ax.axhline(0, color='.6', lw=.7, ls='--', zorder=0)
    ax.set_xticks(range(len(rois)))
    ax.set_xticklabels(rois)
    ax.set_xlim(-.5, len(rois) - .5)
    ax.set_ylim(lo, hi)
    ax.set_yticks([-.01, -.005, 0, .005])
    ax.set_yticklabels(['−.010', '−.005', '0', '.005'])
    ax.set_ylabel('Decoded intensity,\nHP − other quadrants')
    offset_spines(ax)


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


def panel_ecc(ax, be, pb):
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
    ax.axvspan(ITEM_ECC - ITEM_R, ITEM_ECC + ITEM_R, color='.88', lw=0, zorder=0)
    ax.text(ITEM_ECC, hi, 'Items', ha='center', va='top', fontsize=6.5, color='.35')
    ax.text(ITEM_ECC, lo + .25, 'Outside\nthe bar', ha='center', va='bottom', fontsize=6, color='.45')
    ax.fill_between(x, m - se, m + se, color=HP, alpha=.2, lw=0)
    ax.plot(x, m, color=HP, lw=1.3)
    ax.axhline(0, color='.6', lw=.7, ls='--', zorder=0)
    ax.set_xlim(0, ITEM_ECC + ITEM_R)
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_ylim(lo, hi)
    ax.set_yticks([-4, -2, 0, 2])
    ax.set_yticklabels(['−4', '−2', '0', '2'])
    ax.set_xlabel('Eccentricity (°)')
    ax.set_ylabel('HP − other\nquadrants (%)')
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


def panel_bar_pass(ax, ax2, w, st):
    """g: response when the bar passes the pRF, by where the voxel's item location is relative to HP."""
    from matplotlib.patches import Circle
    from retsupp.visualize.paper.paper_style import FG, HP, offset_spines
    w = w[(w.t >= -4.8) & (w.t <= 14.4)]
    cls = ['hp', 'adjacent', 'opposite']
    wide = w.set_index(['subject', 't'])[cls]
    # Cousineau-Morey within-subject SEM
    norm = wide.sub(wide.mean(1), axis=0).add(wide.values.mean(), axis=0)
    k = len(cls)
    for c in cls[::-1]:
        m = wide[c].groupby('t').mean()
        se = norm[c].groupby('t').std(ddof=1) / np.sqrt(w.subject.nunique()) * np.sqrt(k / (k - 1))
        ax.fill_between(m.index, m - se, m + se, color=LOC_COLORS[c], alpha=.18, lw=0)
        ax.plot(m.index, m, color=LOC_COLORS[c], lw=1.3 if c == 'hp' else 1.)
    ax.axhline(0, color='.6', lw=.7, ls='--', zorder=0)
    ax.axvline(0, color='.75', lw=.6, zorder=0)
    ax.set_xlim(-4.8, 14.4)
    ax.set_xticks([0, 5, 10])
    ax.set_xticklabels([])
    ax.set_ylim(-1.2, 2.6)
    ax.set_yticks([0, 2])
    ax.set_ylabel('Response\n(% signal)')
    ax.yaxis.set_label_coords(-.24, .5)
    # key: the display with each location in its colour
    ins = ax.inset_axes([1.04, .25, .38, .75 * 1.55 / .65 * .38 / .75])
    ins.add_patch(Circle((0, 0), APERTURE, fill=False, ec='.6', lw=.5))
    for a, c in zip(ITEM_ANGLES, ['hp', 'adjacent', 'opposite', 'adjacent']):
        q = ITEM_ECC * np.array([np.cos(np.deg2rad(a)), np.sin(np.deg2rad(a))])
        ins.add_patch(Circle(q, ITEM_R * 1.25, facecolor=LOC_COLORS[c], ec='none'))
    ins.plot(0, 0, '+', color=FG, ms=3, mew=.5)
    ins.set_xlim(-5.2, 5.2)
    ins.set_ylim(-5.2, 5.2)
    ins.set_aspect('equal')
    ins.set_axis_off()
    for k, (c, lab) in enumerate((('hp', 'HP'), ('adjacent', 'Next to HP'), ('opposite', 'Opposite'))):
        ax.text(1.04, .12 - k * .2, lab, transform=ax.transAxes, color=LOC_COLORS[c], fontsize=6, ha='left', va='top')
    area_tag(ax, x=1.04, y=-.55)
    offset_spines(ax)
    # HP − LP
    d = (wide.hp - (2 * wide.adjacent + wide.opposite) / 3).groupby('t')
    m, se = d.mean(), d.std(ddof=1) / np.sqrt(w.subject.nunique())
    ax2.axvspan(*BAR_WINDOW, color='.92', lw=0, zorder=0)
    ax2.fill_between(m.index, m - se, m + se, color=HP, alpha=.2, lw=0)
    ax2.plot(m.index, m, color=HP, lw=1.3)
    ax2.axhline(0, color='.6', lw=.7, ls='--', zorder=0)
    ax2.text(np.mean(BAR_WINDOW), .07, stars(st['raw']['p']), ha='center', va='top', fontsize=7, color=HP)
    ax2.set_xlim(-4.8, 14.4)
    ax2.set_xticks([0, 5, 10])
    ax2.set_ylim(-.16, .08)
    ax2.set_yticks([-.1, 0])
    ax2.set_yticklabels(['−.1', '0'])
    ax2.set_ylabel('HP − LP')
    ax2.yaxis.set_label_coords(-.24, .5)
    ax2.set_xlabel('Time from bar crossing the pRF (s)')
    offset_spines(ax2)


# ------------------------------------------------------------------------------------------------------------ figure
def plot(name='fig6_decoding', source='canonical', rois=ROIS, decoder=DECODER, source_note=''):
    import matplotlib.pyplot as plt
    from retsupp.visualize.paper.paper_style import FG, HP, STIM, letter, save
    d = pd.read_csv(DATA / f'decoding_readout_{source}_2026-11.tsv', sep='\t')
    e = effects(d)
    t = test(e, rois, (decoder,))
    t_out = test(e, rois, ('outline',)) if 'outline' in set(d.decoder) else None
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
    sub, run, roi = EXAMPLE
    frames = dict(paradigm=fr['paradigm'], group=fr['group'][list(fr['rois']).index('V1'), list(fr['decoders']).index('free')].astype(float),
                  example=ex[f'{sub}_{run}_roi-{roi}'].astype(float))
    qf = q[q.decoder == 'free'].sort_values('r', ascending=False).reset_index(drop=True)
    hit = (qf.subject == int(sub[4:])) & (qf.session == int(run[4])) & (qf.run == int(run[-1])) & (qf.roi == roi)
    ex_r, ex_rank = qf[hit].r.iloc[0], int(np.flatnonzero(hit)[0]) + 1
    frames['ex_label'] = f'{sub}, {roi}\nr = {ex_r:.2f}'.replace('0.', '.')
    print('top runs', qf.head(5)[['subject', 'session', 'run', 'roi', 'r']].to_string(index=False))

    bp = load_bar_pass()
    bst = bar_pass_stats(bp)

    W, H = 7.25, 6.3
    DY = .55                                      # rows 1-2 sit above the taller row 3
    fig = plt.figure(figsize=(W, H))
    # Row 1: a frames, b quality.
    panel_frames(fig, (.75, 3.75 + DY, 4.0, 1.6), frames, W, H)
    fig.text(.2 / W, (5.62 + DY) / H, 'The bar stimulus can be decoded from held-out runs in ' + ', '.join(rois[:-1]) + f' and {rois[-1]}',
             fontsize=7.5, color=STIM, va='center')
    fig.text(.02 / W, (5.62 + DY) / H, 'a', fontsize=8, fontweight='bold', va='center', ha='left')
    ax = fig.add_axes([5.55 / W, (3.85 + DY) / H, 1.6 / W, 1.42 / H])
    panel_quality(ax, qs, pq, rois)
    letter(ax, 'b', x=-.42, y=1.08)

    # Row 2: c map, d angle, e the test.
    fig.text(.2 / W, (3.4 + DY) / H, 'The bar is decoded at lower intensity in the quadrant facing the HP location', fontsize=7.5,
             color=HP, va='center')
    ax = fig.add_axes([.1 / W, (1.78 + DY) / H, 1.42 / W, 1.42 / H])
    im = panel_map(ax, rm)
    fig.text(.02 / W, (3.4 + DY) / H, 'c', fontsize=8, fontweight='bold', va='center', ha='left')
    cax = fig.add_axes([.38 / W, (1.68 + DY) / H, .86 / W, .06 / H])
    cb = fig.colorbar(im, cax=cax, orientation='horizontal', ticks=[-3, 0, 3])
    cb.ax.set_xticklabels(['−3%', '0', '3%'], fontsize=6)
    cb.ax.tick_params(length=2, width=.5, pad=1)
    cb.outline.set_linewidth(.4)
    cb.ax.text(-.06, .5, 'Less', transform=cb.ax.transAxes, ha='right', va='center', fontsize=6, color='#2166AC')
    cb.ax.text(1.06, .5, 'More', transform=cb.ax.transAxes, ha='left', va='center', fontsize=6, color='#B2182B')
    ax = fig.add_axes([2.2 / W, (2.1 + DY) / H, 1.75 / W, 1.0 / H])
    panel_angle(ax, rm)
    letter(ax, 'd', x=-.3, y=1.04)
    ax = fig.add_axes([5.55 / W, (1.98 + DY) / H, 1.6 / W, 1.15 / H])
    panel_test(ax, e, t, rois, decoder)
    letter(ax, 'e', x=-.42, y=1.04)

    # Row 3: f eccentricity, g the measured response.
    fig.text(.2 / W, 1.93 / H, 'Not driven by the items outside the aperture, and seen in the measured response', fontsize=7.5,
             color=HP, va='center')
    fig.text(.02 / W, 1.93 / H, 'f', fontsize=8, fontweight='bold', va='center', ha='left')
    ax = fig.add_axes([.05 / W, .35 / H, 1.05 / W, 1.05 / H])
    panel_bands_sketch(ax)
    ax = fig.add_axes([1.75 / W, .42 / H, 2.55 / W, 1.2 / H])
    panel_ecc(ax, be, pb)
    area_tag(ax, x=.02, y=.3)
    ax = fig.add_axes([5.0 / W, 1.0 / H, 1.55 / W, .65 / H])
    ax2 = fig.add_axes([5.0 / W, .42 / H, 1.55 / W, .45 / H])
    panel_bar_pass(ax, ax2, bp, bst)
    letter(ax, 'g', x=-.42, y=1.02)

    nq = qs.groupby('roi').r.median()
    quality = '; '.join(f'{r} median r = {nq[r]:.2f} vs {qs[qs.roi == r].r_shuffle.median():.3f}, {fmt_p(pq[r])}' for r in rois)
    tests = '; '.join(f'{r.roi} median {r["median"]:+.4f}, {r.pct_below:.0f}% of subjects below 0, p = {r.p:.3f}, '
                      f'p_Holm = {r.p_holm:.3f}' for _, r in t.iterrows())
    outline = ''
    if t_out is not None:
        outline = (' The decoder with the bar outline given (only the bar\'s pixels decoded) gives the same result: '
                   + ', '.join(f'{r.roi} p_Holm = {r.p_holm:.3f}' for _, r in t_out.iterrows()) + '.')
    band_txt = '; '.join(f'{lab} {be[k].mean():+.1f} ± {be[k].sem():.1f}%, p = {pb[k]:.3f}' for k, a, b, lab in BANDS)
    band_txt += f'; whole aperture {be["all"].mean():+.1f} ± {be["all"].sem():.1f}%, p = {pb["all"]:.3f}'
    fovea = f'{fov.mean():+.1f} ± {fov.sem():.1f}%'
    r_, b_ = bst['raw'], bst['base']
    bar_txt = (f'{r_["diff"]:+.3f}% signal ({r_["rel"]:+.1f}% of the LP response), {r_["below"]:.0f}% of subjects below 0, '
               f'one-sided Wilcoxon p = {r_["p"]:.3f} (vs the two neighbouring LP locations {r_["rel_adjacent"]:+.1f}%, '
               f'p = {r_["p_adjacent"]:.3f}; vs the opposite one {r_["rel_opposite"]:+.1f}%, p = {r_["p_opposite"]:.3f})')
    bar_base = f'{b_["rel"]:+.1f}%, p = {b_["p"]:.2f}'
    save(fig, name, LEGEND.format(n=qs.subject.nunique(), quality=quality, tests=tests, outline=outline,
                                  bands=band_txt, friedman=fried, outer_inner=oi, fovea=fovea, ex_sub=sub, ex_r=ex_r,
                                  ex_rank=ex_rank, n_runs=len(qf), r_med=qf.r.median(), ex_run=run.replace('_', ' '),
                                  ap=APERTURE, item_r=ITEM_R, item_in=ITEM_ECC - ITEM_R, bar=bar_txt, bar_base=bar_base,
                                  w0=BAR_WINDOW[0], w1=BAR_WINDOW[1], source=source_note))
    print('quality', quality)
    print('tests', tests)
    print('outline', outline)
    print('lag medians', qs.groupby('roi')[[f'r_lag_{k}' for k in range(-3, 4)]].median().round(3).to_string())
    print('bands', band_txt, 'friedman', round(fried, 3), 'outer-inner', round(oi, 3), 'fovea', fovea)
    print('bar pass', bar_txt, '| with baseline', bar_base)
    return t


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('step', choices=['collect', 'plot'])
    ap.add_argument('--bids-folder', default='/shares/zne.uzh/gdehol/ds-retsupp')
    ap.add_argument('--source', default='canonical')
    ap.add_argument('--rois', nargs='+', default=ROIS, help='areas of b and e (c, d, f, g pool V1–V3)')
    ap.add_argument('--decoder', default=DECODER)
    ap.add_argument('--out', default=None)
    ap.add_argument('--name', default='fig6_decoding')
    a = ap.parse_args()
    if a.step == 'collect':
        collect(a.bids_folder, a.source, a.out)
    else:
        plot(a.name, a.source, a.rois, a.decoder, source_note=f'pRFs: {a.source} fold fits.')
