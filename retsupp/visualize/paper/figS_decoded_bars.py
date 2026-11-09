"""Supplementary figure: the decoded bars themselves, averaged per bar position in the HP-rotated frame.

``collect`` (cluster): every decoded run of ``retsupp/decode/decode_final.py`` (unconstrained decoder, V1–V3) is
rotated in 90° steps so the HP location lies upper right (90° steps keep the bars vertical or horizontal, so no
interpolation). Frames with a bar inside the aperture are grouped by bar orientation (vertical = moving sideways,
horizontal = moving up or down; taken from the rotated true bar) and by the position of the bar centre along its
direction of motion (``N_BINS`` bins across the aperture). Per subject and bin the decoded frames and the true bar frames
are averaged over runs and ROIs. Writes ``notes/data/decoded_bars_<source>_2026-11.npz``.

``plot`` (local): row 1 vertical bars left → right, row 2 horizontal bars bottom → top, group mean; row 3 the decoded
intensity on the bar in its half nearer HP (upper half of a vertical bar, right half of a horizontal one) minus the
half farther away, in % of the bar's mean, against the bar position (positive = the side of the aperture HP is on).

Usage::

    python -m retsupp.visualize.paper.figS_decoded_bars collect --source jointhrf --out-dir notes/data   # compute node
    python -m retsupp.visualize.paper.figS_decoded_bars plot --source jointhrf
"""
from __future__ import annotations

import argparse
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROIS = ['V1', 'V2', 'V3']
N_BINS = 7
APERTURE = 3.17
EDGES = np.linspace(-APERTURE, APERTURE, N_BINS + 1)


def one_run(f):
    from retsupp.visualize.paper.fig6_collect import rotation_index
    with np.load(f) as z:
        dec = z['decoded'].astype(np.float32)
        par = z['paradigm'].astype(np.float32)
        grid = z['grid']
        hp = str(z['hp_location'])
    T = dec.shape[0]
    ri = rotation_index(grid, hp)
    D = dec.reshape(T, -1)[:, ri]
    P = par.reshape(T, -1)[:, ri]
    x, y = grid[:, 0], grid[:, 1]
    inside = np.hypot(x, y) <= APERTURE
    s = np.zeros((2, N_BINS, 2, len(x)))            # orientation, bin, (decoded, true), pixel
    n = np.zeros((2, N_BINS))
    for t in range(T):
        on = (P[t] > 0) & inside
        if on.sum() < 5:
            continue
        sx, sy = np.ptp(x[on]), np.ptp(y[on])
        o = 0 if sx < sy else 1                    # 0: vertical bar (position = x), 1: horizontal bar (position = y)
        pos = (x[on] if o == 0 else y[on]).mean()
        b = np.clip(np.searchsorted(EDGES, pos) - 1, 0, N_BINS - 1)
        s[o, b, 0] += D[t]
        s[o, b, 1] += P[t]
        n[o, b] += 1
    sub = int(f.parent.name.split('-')[1])
    return sub, s, n


def collect(bids, source, out_dir, jobs=4):
    root = Path(bids) / 'derivatives' / 'decoded_final_2026-11' / source / 'free'
    files = [f for f in sorted(root.glob('sub-*/decoded_ses-*_run-*_roi-*.npz')) if f.stem.split('roi-')[1] in ROIS]
    print(len(files), 'decoded runs', flush=True)
    acc, cnt = {}, {}
    with Pool(jobs) as pool:
        for sub, s, n in pool.imap_unordered(one_run, files, chunksize=4):
            acc[sub] = acc.get(sub, 0) + s
            cnt[sub] = cnt.get(sub, 0) + n
    subs = sorted(acc)
    with np.errstate(invalid='ignore'):
        mean = np.array([acc[s] / cnt[s][:, :, None, None] for s in subs]).astype(np.float32)
    grid = np.load(files[0])['grid']
    out = Path(out_dir) / f'decoded_bars_{source}_2026-11.npz'
    np.savez_compressed(out, mean=mean.reshape(len(subs), 2, N_BINS, 2, 50, 50), n=np.array([cnt[s] for s in subs]),
                        subjects=np.array(subs), edges=EDGES, grid=grid)
    print('Wrote', out, 'frames per bin (all subjects):', np.array([cnt[s] for s in subs]).sum(0))


def half_contrast(mean, grid):
    """Per subject, orientation and bin: decoded intensity on the bar's pixels in the half nearer HP minus the half
    farther away, in % of the bar's mean."""
    x, y = grid[:, 0].reshape(50, 50), grid[:, 1].reshape(50, 50)
    inside = np.hypot(x, y) <= APERTURE
    S, O, B = mean.shape[:3]
    out = np.full((S, O, B), np.nan)
    for o in range(O):
        near = (y > 0) if o == 0 else (x > 0)
        for b in range(B):
            bar = (np.nanmean(mean[:, o, b, 1], 0) > .5) & inside
            for s in range(S):
                d = mean[s, o, b, 0]
                m = np.nanmean(d[bar])
                out[s, o, b] = 100 * (np.nanmean(d[bar & near]) - np.nanmean(d[bar & ~near])) / m
    return out


def plot(source, name='figS_decoded_bars_rotated'):
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from retsupp.visualize.paper.paper_style import FG, HP, MUTED, STIM, letter, offset_spines, save
    z = np.load(Path('notes/data') / f'decoded_bars_{source}_2026-11.npz')
    mean, n, grid, edges = z['mean'], z['n'], z['grid'], z['edges']
    centres = (edges[:-1] + edges[1:]) / 2
    grp = np.nanmean(mean, 0)                                   # orientation, bin, (dec, true), 50, 50
    hc = half_contrast(mean, grid)
    pooled = np.nanmean(hc, 1)                                  # subjects × bins (vertical and horizontal pooled)

    W, H = 7.25, 3.45
    fig = plt.figure(figsize=(W, H))
    cm = mcolors.LinearSegmentedColormap.from_list('stim', ['#FFFFFF', STIM])
    vmax = np.nanpercentile(grp[:, :, 0], 99.5)
    c = np.linspace(-5, 5, 50)
    X, Y = np.meshgrid(c, c)
    size, gap, x0 = .62, .1, 1.3
    for o, lab in enumerate(['Vertical bar,\nleft to right', 'Horizontal bar,\nbottom to top']):
        for b in range(len(centres)):
            ax = fig.add_axes([(x0 + b * (size + gap)) / W, (H - .55 - (o + 1) * size - o * .12) / H, size / W, size / H])
            img = np.where(np.hypot(X, Y) <= APERTURE + .3, grp[o, b, 0], np.nan)
            im = ax.imshow(img, origin='lower', extent=[-5.1, 5.1, -5.1, 5.1], cmap=cm, vmin=0, vmax=vmax,
                           interpolation='nearest')
            clip = Circle((0, 0), APERTURE, transform=ax.transData, fc='none', ec='none')
            ax.add_patch(clip)
            im.set_clip_path(clip)
            ax.add_patch(Circle((0, 0), APERTURE, fill=False, ec='.6', lw=.5))
            ax.contour(X, Y, grp[o, b, 1], levels=[.5], colors=[FG], linewidths=.4, linestyles='dotted')
            q = 4 / np.sqrt(2)
            ax.add_patch(Circle((q, q), .75, fill=False, ec=HP, lw=.8, ls=(0, (2, 1.2))))
            ax.set_xlim(-4.9, 4.9)
            ax.set_ylim(-4.9, 4.9)
            ax.set_aspect('equal')
            ax.set_axis_off()
            if o == 0:
                ax.text(.5, 1.02, f'{centres[b]:+.1f}°'.replace('-', '−'), transform=ax.transAxes, ha='center', va='bottom', fontsize=6,
                        color='.4')
            if b == 0:
                ax.text(-.08, .5, lab, transform=ax.transAxes, ha='right', va='center', fontsize=6.5, color=FG)
    fig.text(.02 / W, (H - .2) / H, 'a', fontsize=8, fontweight='bold', va='center')
    fig.text(.2 / W, (H - .2) / H, 'Decoded bars, HP location rotated to the upper right (V1–V3, mean of 30 subjects)',
             fontsize=7.5, color=STIM, va='center')
    # Row 3: near-HP half minus far half.
    ax = fig.add_axes([2.6 / W, .42 / H, 2.9 / W, 1.0 / H])
    m, se = np.nanmean(pooled, 0), np.nanstd(pooled, 0, ddof=1) / np.sqrt(np.isfinite(pooled).sum(0))
    ax.axvspan(0, APERTURE, color=HP, alpha=.07, lw=0)
    ax.fill_between(centres, m - se, m + se, color=HP, alpha=.2, lw=0)
    ax.plot(centres, m, 'o-', color=HP, ms=3.5, lw=1.2)
    ps = [stats.wilcoxon(pooled[:, b][np.isfinite(pooled[:, b])]).pvalue for b in range(len(centres))]
    lo, hi = np.nanmin(m - se) - 1, np.nanmax(m + se) + 1.5
    for b, p in enumerate(ps):
        st = '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else ''
        ax.text(centres[b], hi - .3, st, ha='center', va='top', fontsize=7, color=HP)
    ax.axhline(0, color='.6', lw=.7, ls='--', zorder=0)
    ax.set_xlim(-APERTURE, APERTURE)
    ax.set_ylim(lo, hi)
    ax.set_xlabel('Bar position (°; positive = HP side)')
    ax.set_xticks([-3, -2, -1, 0, 1, 2, 3])
    ax.set_xticklabels(['−3', '−2', '−1', '0', '1', '2', '3'])
    ax.set_ylabel('Half nearer HP −\nhalf farther (%)')
    ax.text(APERTURE - .1, lo + .2, 'Bar passes the\nHP quadrant', ha='right', va='bottom', fontsize=6, color=HP)
    offset_spines(ax)
    letter(ax, 'b', x=-.85, y=1.02)
    # Sketch: what the near and far halves are.
    sk = fig.add_axes([1.05 / W, .4 / H, 1.05 / W, 1.05 / H])
    from matplotlib.patches import Rectangle
    sk.add_patch(Circle((0, 0), APERTURE, fill=False, ec='.6', lw=.5))
    q = 4 / np.sqrt(2)
    sk.add_patch(Circle((q, q), .75, fill=False, ec=HP, lw=.8, ls=(0, (2, 1.2))))
    hh = np.sqrt(APERTURE ** 2 - 1.2 ** 2)
    sk.add_patch(Rectangle((.8, 0), .8, hh, facecolor=HP, alpha=.45, lw=0))
    sk.add_patch(Rectangle((.8, -hh), .8, hh, facecolor=MUTED, alpha=.6, lw=0))
    sk.text(2.0, hh * .45, 'Near', fontsize=6, color=HP, va='center')
    sk.text(2.0, -hh * .55, 'Far', fontsize=6, color='.4', va='center')
    sk.plot(0, 0, '+', color=FG, ms=3, mew=.5)
    sk.set_xlim(-4.9, 4.9)
    sk.set_ylim(-4.9, 4.9)
    sk.set_aspect('equal')
    sk.set_axis_off()
    nb = n.sum(0)
    legend = f"""
**Figure S. The decoded bars, averaged per bar position with the HP location rotated to the upper right.** Every run's
decoded frames (unconstrained decoder, V1, V2 and V3; pRFs: {source} fold fits) and its true bar frames were rotated in
90° steps so that the HP location (red dashed circle) lies upper right; 90° steps keep the bars vertical or horizontal.
Frames with a bar inside the aperture (grey circle, {APERTURE}°) were grouped by bar orientation and by the position of
the bar centre along its motion ({N_BINS} bins; centres above row 1), averaged over runs and ROIs per subject, then over
subjects (n = {mean.shape[0]}). Dotted outline, the true bar. Frames per bin, all subjects and ROIs: vertical
{', '.join(str(int(v)) for v in nb[0])}; horizontal {', '.join(str(int(v)) for v in nb[1])}. b, Decoded intensity on the
bar's pixels in the half of the bar nearer HP (upper half of a vertical bar, right half of a horizontal one) minus the
half farther away, in % of the bar's mean, vertical and horizontal bars pooled; positive positions are on HP's side of the
aperture (shaded), where the near half lies in the quadrant facing HP and the far half in a neighbouring one; on the
other side the near half lies in a quadrant next to HP and the far half in the opposite quadrant. Line and band: mean ± SEM over subjects;
Wilcoxon signed-rank test against 0 per position, two-sided, uncorrected (* p < .05, ** p < .01): {', '.join(f'{c:+.1f}° {v:+.2f}% p = {p:.3f}' for c, v, p in zip(centres, m, ps))}.
"""
    save(fig, name, legend)
    print('frames per bin', nb)
    print('half contrast', np.round(m, 2), 'p', np.round(ps, 3))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('step', choices=['collect', 'plot'])
    ap.add_argument('--bids-folder', default='/shares/zne.uzh/gdehol/ds-retsupp')
    ap.add_argument('--source', default='jointhrf')
    ap.add_argument('--out-dir', default='notes/data')
    ap.add_argument('-j', '--jobs', type=int, default=4)
    a = ap.parse_args()
    if a.step == 'collect':
        collect(a.bids_folder, a.source, a.out_dir, a.jobs)
    else:
        plot(a.source)
