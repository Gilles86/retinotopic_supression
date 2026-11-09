"""Figure 6: the bar stimulus can be decoded from V1–V3, and less of it is decoded in the quadrant facing HP.

``collect`` (cluster): concatenates the per-run readouts of ``retsupp/decode/decode_final.py`` into one table. The
other summaries (decoding quality, eccentricity bands, HP-rotated maps, display frames) come from
``retsupp/visualize/paper/fig6_collect.py``.

``plot`` (local): the pre-specified test is per subject and ROI the mean over runs of drive(HP quadrant) − mean
drive(three other quadrants); Wilcoxon signed-rank across subjects, one-sided (HP < LP), Holm across V1, V2, V3
(``notes/primary_test_2026-11.md``). The outline-fixed decoder (bar footprint given, intensities decoded) is the test;
the free decoder is shown beside it.

Usage::

    python -m retsupp.visualize.paper.fig6_decoding collect --source canonical \\
        --out /shares/zne.uzh/gdehol/ds-retsupp/derivatives/paper_2026-11/decoding_readout_canonical.tsv
    python -m retsupp.visualize.paper.fig6_decoding plot --table notes/data/decoding_readout_canonical_2026-11.tsv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage, stats

ROIS = ['V1', 'V2', 'V3']
DECODERS = [('outline', 'Bar outline given'), ('free', 'Unconstrained')]
DATA = Path('notes/data')
EXAMPLE = ('sub-23', 'ses-2_run-3', 'V1')
FRAMES = [15, 21, 28, 72, 81, 89, 130]      # TRs shown in a (bar left → right, top → bottom, then blank)
TR = 1.6
APERTURE, ITEM_ECC, ITEM_R = 3.17, 4.0, .75
BANDS = [('0-1.5', 0, 1.5, '0–1.5°'), ('1.5-2.5', 1.5, 2.5, '1.5–2.5°'), ('2.5-ap', 2.5, APERTURE, '2.5°–edge')]
BAND_SHADE = ['#E8ECF0', '#D5DCE3', '#C2CBD5']
LEGEND = """
**Figure 6. The bar stimulus can be read out of early visual cortex, and less of it is read out in the quadrant facing
the HP location.** The bar stimulus of every run was decoded from the BOLD activity of V1, V2 or V3 with pRFs (difference
of Gaussians, canonical HRF, no gain terms) fitted on the other runs (cross-validation folds), so the decoded run never
entered the pRF fit; voxels with cross-run reliability > 0 on the training runs (at most 2000 per ROI); the encoding model
contains only the bar, not the search items. a, True bar (top) and decoded intensity in a single held-out run of one
subject ({ex_sub}, V1, frame-wise r = {ex_r:.2f}; middle) and averaged over all runs and subjects (V1; bottom), at seven
time points of the run (the last one without a bar). Grey circle, the {ap}° aperture of the bar. b, Per subject, the
spatial correlation between decoded and true frame inside the aperture, averaged over the frames with a bar and over
runs (unconstrained decoder; blue dots, n = {n} subjects), against the same with the order of the true frames shuffled
(100 permutations; grey). Points and bars: median and bootstrapped 95% CI. Decoded vs shuffled: Wilcoxon signed-rank
test, one-sided (*** p < .001); {quality}. The decoded time course is not displaced in time: the correlation peaks at lag 0 in all
three ROIs. With the bar outline given (the decoder of c–f) the correlation is trivially high (median r = {r_outline}). c,
Each run's visual field rotated in 90° steps so that the HP location (red dashed circle; grey, the LP locations, drawn
with the item footprint, radius {item_r}°) lies upper right, then the decoded intensity on the bar's pixels minus its
average over the four rotations (in % of that average), averaged over runs, V1–V3 and subjects, smoothed (Gaussian, σ =
0.4°); bar-outline decoder. d, The same as a function of polar angle relative to the HP direction (bins of 22.5°), whole
aperture; shaded red, the quadrant facing HP. Line and band: mean ± SEM over subjects. e, The pre-specified test: per
subject and ROI, the mean decoded intensity on the bar's pixels in the quadrant facing HP minus the mean over the three
other quadrants, averaged over runs. Filled: bar outline given, intensities decoded (the pre-specified decoder; dots are
subjects); open: every pixel decoded. Points and bars: median and bootstrapped 95% CI. Wilcoxon signed-rank test, one-sided
(HP < other quadrants), Holm across the three ROIs; * p_Holm < .05. {tests}. f, The difference of e by eccentricity, in %
of the drive in the other quadrants at the same eccentricity, mean of V1–V3 per subject; line and band, mean ± SEM over
subjects (rings of 0.25°); left, the three bands drawn on the display. Items (radius {item_r}°) begin at
{item_in:.2f}° eccentricity, just outside the aperture, and their responses are in the BOLD data but not in the
encoding model; if item responses leaking into the aperture produced the difference, it would grow toward the aperture
edge. It does not: per band (mean ± SEM; one-sided Wilcoxon signed-rank test, uncorrected, * p < .05) {bands}; the three bands do not differ
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


def test(e):
    rows = []
    for dec, _ in DECODERS:
        for roi in ROIS:
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


def band_effects(bands, decoder='outline'):
    """Per subject × band (V1–V3 mean): HP − other quadrants in % of the other quadrants, per eccentricity band/ring."""
    b = bands[bands.decoder == decoder]
    hp = b[b.is_hp].groupby(['subject', 'roi', 'band', 'session', 'run']).drive_bar.mean()
    lp = b[~b.is_hp].groupby(['subject', 'roi', 'band', 'session', 'run']).drive_bar.mean()
    r = pd.DataFrame({'diff': hp - lp, 'lp': lp}).groupby(['subject', 'roi', 'band']).mean()
    r['rel'] = 100 * r['diff'] / r['lp']
    return r.groupby(['subject', 'band']).rel.mean().unstack()


def rel_maps(maps, decoder='outline'):
    """Per subject: HP-rotated drive minus its 4-fold rotation average, in % of that average, mean over V1–V3."""
    j = list(maps['decoders']).index(decoder)
    x = maps['drive'][:, :, j].astype(float)                         # sub, roi, y, x
    sym = np.mean([np.rot90(x, k, axes=(2, 3)) for k in range(4)], 0)
    ref = np.nanmean(sym, 0, keepdims=True)                          # group average, per ROI and pixel
    return np.nanmean((x - sym) / ref, 1) * 100


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
            (f'One run\n({frames["ex_label"]})', frames['example'], np.nanpercentile(frames['example'], 99.7)),
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


def panel_quality(ax, qs, pq):
    from retsupp.visualize.paper.paper_style import FG, MUTED, STIM, offset_spines
    for i, roi in enumerate(ROIS):
        g = qs[qs.roi == roi]
        subject_dots(ax, i - .14, g.r.to_numpy(), STIM, i)
        subject_dots(ax, i + .2, g.r_shuffle.to_numpy(), MUTED, 10 + i, s=5, width=.07, ms=4)
        ax.text(i - .14, .7, stars(pq[roi]), ha='center', va='bottom', fontsize=7, color=STIM)
    ax.axhline(0, color='.6', lw=.7, ls='--', zorder=0)
    ax.set_xticks(range(3))
    ax.set_xticklabels(ROIS)
    ax.set_xlim(-.5, 2.5)
    ax.set_ylim(-.06, .7)
    ax.set_yticks([0, .2, .4, .6])
    ax.set_ylabel('Decoded vs true bar (r)')
    ax.text(2.45, .2, 'Held-out run', color=STIM, fontsize=6.5, ha='right', va='center')
    ax.text(2.45, .08, 'Frames shuffled', color='.45', fontsize=6.5, ha='right', va='center')
    offset_spines(ax)


def items(ax, hp_color, lw=.8):
    from matplotlib.patches import Circle
    for k, a in enumerate((45, 135, 225, 315)):
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
    for a in (0, 90):
        t = np.deg2rad(a)
        ax.plot([-APERTURE * np.cos(t), APERTURE * np.cos(t)], [-APERTURE * np.sin(t), APERTURE * np.sin(t)],
                color='.55', lw=.4, ls=':')
    items(ax, HP)
    ax.set_xlim(-4.95, 4.95)
    ax.set_ylim(-4.95, 4.95)
    ax.set_aspect('equal')
    p = ITEM_ECC / np.sqrt(2)
    ax.text(p, p + ITEM_R + .25, 'HP', color=HP, fontsize=6.5, ha='center', va='bottom')
    return im


def panel_angle(ax, rm):
    from retsupp.visualize.paper.paper_style import FG, HP, offset_spines
    c = np.linspace(-5, 5, 50)
    X, Y = np.meshgrid(c, c)
    inside = np.hypot(X, Y) <= APERTURE
    ang = (np.degrees(np.arctan2(Y, X)) - 45 + 180) % 360 - 180
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
    ax.set_xticklabels(['−180', '−90', 'HP', '90', '180'])
    ax.set_ylim(-2.6, 2.2)
    ax.set_yticks([-2, -1, 0, 1, 2])
    ax.set_xlabel('Polar angle from HP (°)')
    ax.set_ylabel('Drive vs rotation\nmean (%)')
    ax.text(0, 2.15, 'Quadrant\nfacing HP', color=HP, fontsize=6.5, ha='center', va='top')
    offset_spines(ax)


def panel_test(ax, e, t):
    from retsupp.visualize.paper.paper_style import HP, offset_spines
    lo, hi = -.0132, .008
    for i, roi in enumerate(ROIS):
        v = e[(e.decoder == 'outline') & (e.roi == roi)]['diff'].to_numpy()
        subject_dots(ax, i - .12, v, HP, i)
        n_lo = (v < lo).sum()
        if n_lo:
            ax.scatter(np.full(n_lo, i - .12), np.full(n_lo, lo), s=10, marker='v', color=HP, alpha=.6, lw=0, clip_on=False)
        vf = e[(e.decoder == 'free') & (e.roi == roi)]['diff'].to_numpy()
        subject_dots(ax, i + .24, vf, HP, i, s=0, open_=True, ms=4.5)
        q = t[(t.decoder == 'outline') & (t.roi == roi)].p_holm.iloc[0]
        ax.text(i - .12, hi, stars(q), ha='center', va='bottom', fontsize=7, color=HP)
    ax.axhline(0, color='.6', lw=.7, ls='--', zorder=0)
    ax.set_xticks(range(3))
    ax.set_xticklabels(ROIS)
    ax.set_xlim(-.5, 2.6)
    ax.set_ylim(lo, hi)
    ax.set_yticks([-.01, -.005, 0, .005])
    ax.set_yticklabels(['−.010', '−.005', '0', '.005'])
    ax.set_ylabel('Decoded drive,\nHP − other quadrants')
    offset_spines(ax)


def panel_key_test(ax):
    from retsupp.visualize.paper.paper_style import HP
    tf = ax.transAxes
    for k, (lab, open_) in enumerate([('Bar outline given', False), ('Unconstrained', True)]):
        y = .15 - k * .1
        ax.plot(.42, y, 'o', transform=tf, ms=4, color=HP, mfc='w' if open_ else HP, mec=HP, mew=1. if open_ else 0,
                clip_on=False)
        ax.text(.47, y, lab, transform=tf, fontsize=6.5, color=HP, va='center')


def panel_bands_sketch(ax):
    from matplotlib.patches import Circle, Wedge
    from retsupp.visualize.paper.paper_style import FG, HP
    for (key, a, b, lab), col in zip(BANDS, BAND_SHADE):
        ax.add_patch(Wedge((0, 0), b, 0, 360, width=b - a, facecolor=col, ec='w', lw=.6))
    ax.add_patch(Circle((0, 0), APERTURE, fill=False, ec='.45', lw=.6))
    for k, a in enumerate((45, 135, 225, 315)):
        p = ITEM_ECC * np.array([np.cos(np.deg2rad(a)), np.sin(np.deg2rad(a))])
        ax.add_patch(Circle(p, ITEM_R, facecolor='.88' if k else '#F6D4D4', ec=HP if k == 0 else '.55', lw=.6))
    ax.plot(0, 0, '+', color=FG, ms=3, mew=.5)
    ax.set_xlim(-4.9, 4.9)
    ax.set_ylim(-4.9, 4.9)
    ax.set_aspect('equal')
    ax.set_axis_off()
    p = ITEM_ECC / np.sqrt(2)
    ax.text(p, p + ITEM_R + .2, 'HP item', color=HP, fontsize=6.5, ha='center', va='bottom')


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


# ------------------------------------------------------------------------------------------------------------ figure
def plot(table, name='fig6_decoding', source='canonical', source_note=''):
    import matplotlib.pyplot as plt
    from retsupp.visualize.paper.paper_style import FG, HP, STIM, letter, save
    d = pd.read_csv(table, sep='\t')
    e = effects(d)
    t = test(e)
    print(t.round(4).to_string(index=False))

    q = pd.read_csv(DATA / f'decoding_quality_{source}_2026-11.tsv', sep='\t')
    qs = q[q.decoder == 'free'].groupby(['subject', 'roi'])[['r', 'r_shuffle'] + [f'r_lag_{k}' for k in range(-3, 4)]].mean().reset_index()
    pq = {roi: stats.wilcoxon(g.r - g.r_shuffle, alternative='greater').pvalue for roi, g in qs.groupby('roi')}
    r_outline = q[q.decoder == 'outline'].groupby('subject').r.mean().median()

    bands = pd.read_csv(DATA / f'decoding_bands_{source}_2026-11.tsv.gz', sep='\t')
    be = band_effects(bands)
    pb = {k: stats.wilcoxon(be[k], alternative='less').pvalue for k in ['all'] + [b[0] for b in BANDS]}
    fried = stats.friedmanchisquare(*[be[b[0]] for b in BANDS]).pvalue
    oi = stats.wilcoxon(be['2.5-ap'] - be['0-1.5']).pvalue
    fov = be[['ring_0.00', 'ring_0.25', 'ring_0.50']].mean(1)

    maps = np.load(DATA / f'decoding_maps_{source}_2026-11.npz')
    rm = rel_maps(maps)
    fr = np.load(DATA / f'decoding_frames_{source}_2026-11.npz')
    ex = np.load(DATA / f'decoding_example_run_{source}_2026-11.npz')
    sub, run, roi = EXAMPLE
    frames = dict(paradigm=fr['paradigm'], group=fr['group'][0, list(fr['decoders']).index('free')].astype(float),
                  example=ex[f'{sub}_{run}_roi-{roi}'].astype(float), ex_label=f'{sub}, {roi}')
    ex_r = q[(q.subject == int(sub[4:])) & (q.session == int(run[4])) & (q.run == int(run[-1])) & (q.roi == roi)
             & (q.decoder == 'free')].r.iloc[0]

    W, H = 7.25, 5.75
    fig = plt.figure(figsize=(W, H))
    # Row 1: a frames, b quality.
    ax0 = panel_frames(fig, (.75, 3.75, 4.0, 1.6), frames, W, H)[0]
    fig.text(.2 / W, 5.62 / H, 'The bar stimulus can be decoded from held-out runs', fontsize=7.5, color=STIM,
             va='center', fontweight='bold')
    fig.text(.02 / W, 5.62 / H, 'a', fontsize=8, fontweight='bold', va='center', ha='left')
    ax = fig.add_axes([5.55 / W, 3.85 / H, 1.6 / W, 1.42 / H])
    panel_quality(ax, qs, pq)
    letter(ax, 'b', x=-.42, y=1.08)

    # Row 2: c map, d angle, e the test.
    fig.text(.2 / W, 3.4 / H, 'Less of the bar is decoded in the quadrant facing the HP location', fontsize=7.5,
             color=HP, va='center', fontweight='bold')
    ax = fig.add_axes([.1 / W, 1.78 / H, 1.42 / W, 1.42 / H])
    im = panel_map(ax, rm)
    fig.text(.02 / W, 3.4 / H, 'c', fontsize=8, fontweight='bold', va='center', ha='left')
    cax = fig.add_axes([.38 / W, 1.68 / H, .86 / W, .06 / H])
    cb = fig.colorbar(im, cax=cax, orientation='horizontal', ticks=[-3, 0, 3])
    cb.ax.set_xticklabels(['−3%', '0', '3%'], fontsize=6)
    cb.ax.tick_params(length=2, width=.5, pad=1)
    cb.outline.set_linewidth(.4)
    cb.ax.text(-.06, .5, 'Less', transform=cb.ax.transAxes, ha='right', va='center', fontsize=6, color='#2166AC')
    cb.ax.text(1.06, .5, 'More', transform=cb.ax.transAxes, ha='left', va='center', fontsize=6, color='#B2182B')
    ax = fig.add_axes([2.25 / W, 1.98 / H, 2.0 / W, 1.15 / H])
    panel_angle(ax, rm)
    letter(ax, 'd', x=-.3, y=1.04)
    ax = fig.add_axes([5.55 / W, 1.98 / H, 1.6 / W, 1.15 / H])
    panel_test(ax, e, t)
    panel_key_test(ax)
    letter(ax, 'e', x=-.42, y=1.04)

    # Row 3: f eccentricity.
    fig.text(.2 / W, 1.38 / H, 'Not driven by the items outside the aperture', fontsize=7.5, color=HP, va='center',
             fontweight='bold')
    ax = fig.add_axes([.25 / W, .08 / H, 1.15 / W, 1.15 / H])
    panel_bands_sketch(ax)
    fig.text(.02 / W, 1.38 / H, 'f', fontsize=8, fontweight='bold', va='center', ha='left')
    ax = fig.add_axes([2.25 / W, .38 / H, 4.6 / W, .85 / H])
    panel_ecc(ax, be, pb)

    nq = qs.groupby('roi').r.median()
    quality = '; '.join(f'{r} median r = {nq[r]:.2f} vs {qs[qs.roi == r].r_shuffle.median():.3f}, {fmt_p(pq[r])}' for r in ROIS)
    tests = '; '.join(f'{r.roi} ({dict(DECODERS)[r.decoder].lower()}) median {r["median"]:+.4f}, {r.pct_below:.0f}% of '
                      f'subjects below 0, p = {r.p:.3f}, p_Holm = {r.p_holm:.3f}' for _, r in t.iterrows())
    band_txt = '; '.join(f'{lab} {be[k].mean():+.1f} ± {be[k].sem():.1f}%, p = {pb[k]:.3f}' for k, a, b, lab in BANDS)
    band_txt += f'; whole aperture {be["all"].mean():+.1f} ± {be["all"].sem():.1f}%, p = {pb["all"]:.3f}'
    fovea = f'{fov.mean():+.1f} ± {fov.sem():.1f}%'
    save(fig, name, LEGEND.format(n=qs.subject.nunique(), quality=quality, r_outline=f'{r_outline:.2f}', tests=tests,
                                  bands=band_txt, friedman=fried, outer_inner=oi, fovea=fovea, ex_sub=sub, ex_r=ex_r,
                                  ap=APERTURE, item_r=ITEM_R, item_in=ITEM_ECC - ITEM_R, source=source_note))
    print('quality', quality)
    print('lag medians', qs.groupby('roi')[[f'r_lag_{k}' for k in range(-3, 4)]].median().round(3).to_string())
    print('bands', band_txt, 'friedman', round(fried, 3), 'outer-inner', round(oi, 3), 'fovea', fovea)
    return t


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('step', choices=['collect', 'plot'])
    ap.add_argument('--bids-folder', default='/shares/zne.uzh/gdehol/ds-retsupp')
    ap.add_argument('--source', default='canonical')
    ap.add_argument('--out', default=None)
    ap.add_argument('--table', default=None)
    ap.add_argument('--name', default='fig6_decoding')
    a = ap.parse_args()
    if a.step == 'collect':
        collect(a.bids_folder, a.source, a.out)
    else:
        plot(a.table, a.name, a.source, source_note=f'pRFs: {a.source} fold fits.')
