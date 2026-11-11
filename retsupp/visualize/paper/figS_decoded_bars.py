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


def data_name(source, decoder):
    return f'decoded_bars_{source}_2026-11.npz' if decoder == 'free' else f'decoded_bars_{source}_{decoder}_2026-11.npz'


def collect(bids, source, out_dir, jobs=4, decoder='free'):
    root = Path(bids) / 'derivatives' / 'decoded_final_2026-11' / source / decoder
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
    out = Path(out_dir) / data_name(source, decoder)
    np.savez_compressed(out, mean=mean.reshape(len(subs), 2, N_BINS, 2, 50, 50), n=np.array([cnt[s] for s in subs]),
                        subjects=np.array(subs), edges=EDGES, grid=grid)
    print('Wrote', out, 'frames per bin (all subjects):', np.array([cnt[s] for s in subs]).sum(0))


def one_subject_roi(args):
    """Condition-difference frames for one subject × ROI (each run first scaled to a common mean intensity on the bar):
    each run's decoded frame minus the mean decoded frame at the
    same time point over the subject's runs (mean within each HP location first, then over the four), then rotated so
    HP lies upper right and binned by bar orientation and position. Returns sums over runs."""
    from retsupp.visualize.paper.fig6_collect import rotation_index
    files = args
    runs = []
    for f in files:
        with np.load(f) as z:
            runs.append((z['decoded'].astype(np.float32), z['paradigm'].astype(np.float32), z['grid'],
                         str(z['hp_location'])))
    T = min(r[0].shape[0] for r in runs)
    grid = runs[0][2]
    # Run gain: HP locations come in blocks of consecutive runs, so slow changes in overall decoded intensity (drift,
    # session) would not cancel in the mean over HP locations; each run is scaled to the subject's mean intensity on
    # the bar's pixels inside the aperture first.
    ins = np.hypot(grid[:, 0], grid[:, 1]) <= APERTURE
    gains = [dec[:T].reshape(T, -1)[:, ins][par[:T].reshape(T, -1)[:, ins] > 0].mean() for dec, par, _, _ in runs]
    runs = [(dec * (np.mean(gains) / g_), par, gr, hp) for (dec, par, gr, hp), g_ in zip(runs, gains)]
    by_hp = {}
    for dec, _, _, hp in runs:
        by_hp.setdefault(hp, []).append(dec[:T].reshape(T, -1))
    base = np.mean([np.mean(v, 0) for v in by_hp.values()], 0)          # balanced over HP locations
    x, y = grid[:, 0], grid[:, 1]
    inside = np.hypot(x, y) <= APERTURE
    s = np.zeros((2, N_BINS, 3, len(x)))            # orientation, bin, (decoded, decoded − mean over HP, true), pixel
    n = np.zeros((2, N_BINS))
    for dec, par, _, hp in runs:
        ri = rotation_index(grid, hp)
        D = dec[:T].reshape(T, -1)
        Dd = (D - base)[:, ri]
        D, P = D[:, ri], par[:T].reshape(T, -1)[:, ri]
        for t in range(T):
            on = (P[t] > 0) & inside
            if on.sum() < 5:
                continue
            o = 0 if np.ptp(x[on]) < np.ptp(y[on]) else 1
            pos = (x[on] if o == 0 else y[on]).mean()
            b = np.clip(np.searchsorted(EDGES, pos) - 1, 0, N_BINS - 1)
            s[o, b, 0] += D[t]
            s[o, b, 1] += Dd[t]
            s[o, b, 2] += P[t]
            n[o, b] += 1
    sub = int(files[0].parent.name.split('-')[1])
    return sub, s, n, len(by_hp)


def collect_condition(bids, source, out_dir, jobs=4, decoder='free'):
    root = Path(bids) / 'derivatives' / 'decoded_final_2026-11' / source / decoder
    tasks = []
    for d in sorted(root.glob('sub-*')):
        for roi in ROIS:
            fs = sorted(d.glob(f'decoded_ses-*_run-*_roi-{roi}.npz'))
            if fs:
                tasks.append(fs)
    print(len(tasks), 'subject x ROI', flush=True)
    acc, cnt = {}, {}
    with Pool(jobs) as pool:
        for sub, s, n, nhp in pool.imap_unordered(one_subject_roi, tasks):
            if nhp < 4:
                print(f'sub-{sub:02d}: only {nhp} HP locations', flush=True)
            acc[sub] = acc.get(sub, 0) + s
            cnt[sub] = cnt.get(sub, 0) + n
    subs = sorted(acc)
    with np.errstate(invalid='ignore'):
        mean = np.array([acc[s] / cnt[s][:, :, None, None] for s in subs]).astype(np.float32)
    out = Path(out_dir) / data_name(source, decoder).replace('decoded_bars_', 'decoded_bars_condition_')
    np.savez_compressed(out, mean=mean.reshape(len(subs), 2, N_BINS, 3, 50, 50), n=np.array([cnt[s] for s in subs]),
                        subjects=np.array(subs), edges=EDGES, grid=np.load(tasks[0][0])['grid'])
    print('Wrote', out, 'frames per bin:', np.array([cnt[s] for s in subs]).sum(0))


ALL_ROIS = ['V1', 'V2', 'V3', 'V3AB', 'IPS', 'hV4', 'VO', 'LO', 'TO']
EARLY_ROIS = ['V1', 'V2', 'V3']
SEG_QUAD = {'hp': (1, 1), 'next_ul': (-1, 1), 'next_lr': (1, -1), 'opposite': (-1, -1)}


def _segments(dec, diff, bar, X, Y):
    """Difference on the bar's pixels per quadrant, in % of the mean decoded intensity on the bar."""
    bm = dec[bar].mean()
    out = {}
    for qn, (sx, sy) in SEG_QUAD.items():
        mk = bar & (np.sign(X) == sx) & (np.sign(Y) == sy)
        if mk.sum() >= 10:
            out[qn] = 100 * diff[mk].mean() / bm
    return out


def collect_condition_per_roi(bids, source, out_dir, jobs=4, decoder='outline', exclude=()):
    """All nine ROIs, reduced on the compute node (bins: the module's ``N_BINS``):

    * ``decoded_bars_segments_<source>_<decoder>_b<N>_2026-11.tsv``: per subject × ROI × orientation (vertical,
      horizontal, pooled = horizontal reflected across the HP diagonal onto vertical) × bin × quadrant, the
      decoded − mean-over-HP difference on the bar's pixels in that quadrant, % of the mean decoded intensity on the bar.
    * ``decoded_bars_images_<source>_<decoder>_b<N>_2026-11.npz``: group means (over subjects) of, per orientation ×
      bin, the decoded image in % of the bar's mean and the difference image in % of the bar's mean, V1–V3 averaged per
      subject first (``early_dec``, ``early_diff``, ``early_bar``); and per ROI the pooled-orientation difference image
      (``roi_diff``, with ``roi_n`` subjects).
    """
    root = Path(bids) / 'derivatives' / 'decoded_final_2026-11' / source / decoder
    tasks, keys = [], []
    for d in sorted(root.glob('sub-*')):
        if int(d.name.split('-')[1]) in set(exclude):
            continue
        for roi in ALL_ROIS:
            fs = sorted(d.glob(f'decoded_ses-*_run-*_roi-{roi}.npz'))
            if fs:
                tasks.append(fs)
                keys.append((int(d.name.split('-')[1]), roi))
    print(len(tasks), 'subject x ROI', N_BINS, 'bins', flush=True)
    grid = np.load(tasks[0][0])['grid']
    X, Y = grid[:, 0].reshape(50, 50), grid[:, 1].reshape(50, 50)
    inside = np.hypot(X, Y) <= APERTURE
    centres = (EDGES[:-1] + EDGES[1:]) / 2
    shape = (2, N_BINS, 50, 50)
    early = {}                                          # subject -> list of (dec_rel, diff_rel, bar) for V1-V3
    roi_sum = {r: np.zeros((N_BINS, 50, 50)) for r in ALL_ROIS}
    roi_cnt = {r: np.zeros((N_BINS, 50, 50)) for r in ALL_ROIS}
    roi_n = {r: 0 for r in ALL_ROIS}
    frames = np.zeros((2, N_BINS))
    rows = []
    with Pool(jobs) as pool:
        for (sub, roi), (_, s_, n_, nhp) in zip(keys, pool.imap(one_subject_roi, tasks)):
            if roi in EARLY_ROIS:
                frames += n_
            with np.errstate(invalid='ignore'):
                m = (s_ / n_[:, :, None, None]).reshape(2, N_BINS, 3, 50, 50)
            pooled = (m[0] + np.swapaxes(m[1], -1, -2)) / 2
            dec_rel, diff_rel, bars = np.full(shape, np.nan), np.full(shape, np.nan), np.zeros(shape)
            for b in range(N_BINS):
                for o, (oname, mm) in enumerate((('vertical', m[0, b]), ('horizontal', m[1, b]), ('pooled', pooled[b]))):
                    bar = (mm[2] > .5) & inside
                    if bar.sum() < 10:
                        continue
                    for qn, v in _segments(mm[0], mm[1], bar, X, Y).items():
                        rows.append(dict(subject=sub, roi=roi, orientation=oname, bin=b, centre=centres[b], quadrant=qn,
                                         diff_pct=v, n_hp=nhp))
                    bm = mm[0][bar].mean()
                    if o < 2:
                        dec_rel[o, b] = np.where(inside, 100 * mm[0] / bm, np.nan)
                        diff_rel[o, b] = np.where(inside, 100 * mm[1] / bm, np.nan)
                        bars[o, b] = mm[2]
                    else:
                        v = np.where(inside, 100 * mm[1] / bm, np.nan)
                        ok = np.isfinite(v)
                        roi_sum[roi][b] += np.where(ok, v, 0)
                        roi_cnt[roi][b] += ok
            roi_n[roi] += 1
            if roi in EARLY_ROIS:
                early.setdefault(sub, []).append((dec_rel, diff_rel, bars))
    tag = f'{source}_{decoder}_b{N_BINS}'
    pd.DataFrame(rows).to_csv(Path(out_dir) / f'decoded_bars_segments_{tag}_2026-11.tsv', sep='\t', index=False)
    per_sub = [[np.nanmean([x[k] for x in v], 0) for k in range(3)] for v in early.values()]
    with np.errstate(invalid='ignore'):
        np.savez_compressed(Path(out_dir) / f'decoded_bars_images_{tag}_2026-11.npz',
                            early_dec=np.nanmean([p_[0] for p_ in per_sub], 0).astype(np.float32),
                            early_diff=np.nanmean([p_[1] for p_ in per_sub], 0).astype(np.float32),
                            early_bar=np.nanmean([p_[2] for p_ in per_sub], 0).astype(np.float32),
                            roi_diff=np.array([roi_sum[r] / roi_cnt[r] for r in ALL_ROIS]).astype(np.float32),
                            roi_n=np.array([roi_n[r] for r in ALL_ROIS]), rois=np.array(ALL_ROIS), centres=centres,
                            edges=EDGES, frames_early=frames, n_early=len(per_sub), grid=grid)
    print('Wrote', tag, 'frames per bin (V1-V3):', frames, flush=True)


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


def mirror_diff(mean, grid, sigma=2.):
    """Per subject, orientation and bin: decoded image minus its mirror image (vertical bars flipped top-bottom,
    horizontal bars left-right, so the bar maps onto itself and the HP quadrant onto the neighbouring one), in % of
    the mean decoded intensity on the bar, smoothed (Gaussian, ``sigma`` pixels) inside the aperture."""
    from scipy import ndimage
    x, y = grid[:, 0].reshape(50, 50), grid[:, 1].reshape(50, 50)
    inside = np.hypot(x, y) <= APERTURE
    S, O, B = mean.shape[:3]
    out = np.full((S, O, B, 50, 50), np.nan)
    wsm = ndimage.gaussian_filter(inside.astype(float), sigma)
    for o in range(O):
        for b in range(B):
            bar = (np.nanmean(mean[:, o, b, 1], 0) > .5) & inside
            for s_ in range(S):
                d = mean[s_, o, b, 0]
                mir = np.flip(d, axis=0 if o == 0 else 1)
                diff = 100 * (d - mir) / np.nanmean(d[bar])
                sm = ndimage.gaussian_filter(np.where(inside, np.nan_to_num(diff), 0), sigma) / wsm
                out[s_, o, b] = np.where(inside, sm, np.nan)
    return out


def plot(source, name='figS_decoded_bars_rotated', decoder='free'):
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from retsupp.visualize.paper.paper_style import FG, HP, MUTED, STIM, letter, offset_spines, save
    z = np.load(Path('notes/data') / data_name(source, decoder))
    mean, n, grid, edges = z['mean'], z['n'], z['grid'], z['edges']
    centres = (edges[:-1] + edges[1:]) / 2
    grp = np.nanmean(mean, 0)                                   # orientation, bin, (dec, true), 50, 50
    hc = half_contrast(mean, grid)
    pooled = np.nanmean(hc, 1)                                  # subjects × bins (vertical and horizontal pooled)

    md = np.nanmean(mirror_diff(mean, grid), 0)                 # orientation, bin, 50, 50
    from scipy import ndimage
    W, H = 7.25, 4.75
    fig = plt.figure(figsize=(W, H))
    cm = mcolors.LinearSegmentedColormap.from_list('stim', ['#FFFFFF', STIM])
    vmax = np.nanpercentile(grp[:, :, 0], 99.5)
    c = np.linspace(-5, 5, 50)
    X, Y = np.meshgrid(c, c)
    size, gap, x0 = .62, .1, 1.3
    dvm = 3.
    tops = {}
    for o, lab in enumerate(['Vertical bar,\nleft to right', 'Horizontal bar,\nbottom to top']):
        for k in range(2):                              # 0: decoded image, 1: minus its mirror image
            yb = H - .55 - (2 * o + k + 1) * size - k * .06 - o * .2
            for b in range(len(centres)):
                ax = fig.add_axes([(x0 + b * (size + gap)) / W, yb / H, size / W, size / H])
                if k == 0:
                    img = np.where(np.hypot(X, Y) <= APERTURE + .3, grp[o, b, 0], np.nan)
                    im = ax.imshow(img, origin='lower', extent=[-5.1, 5.1, -5.1, 5.1], cmap=cm, vmin=0, vmax=vmax,
                                   interpolation='nearest')
                else:
                    foot = ndimage.binary_dilation(grp[o, b, 1] > .5, iterations=1) & (np.hypot(X, Y) <= APERTURE)
                    foot &= (Y > 0) if o == 0 else (X > 0)          # the half nearer HP (the other is its negative)
                    img = np.where(foot, md[o, b], np.nan)
                    im = ax.imshow(img, origin='lower', extent=[-5.1, 5.1, -5.1, 5.1], cmap='RdBu_r', vmin=-dvm,
                                   vmax=dvm, interpolation='nearest')
                    dim = im
                clip = Circle((0, 0), APERTURE, transform=ax.transData, fc='none', ec='none')
                ax.add_patch(clip)
                im.set_clip_path(clip)
                ax.add_patch(Circle((0, 0), APERTURE, fill=False, ec='.6', lw=.5))
                if k == 0:
                    ax.contour(X, Y, grp[o, b, 1], levels=[.5], colors=[FG], linewidths=.4, linestyles='dotted')
                else:
                    ax.plot([-APERTURE, APERTURE] if o == 0 else [0, 0], [0, 0] if o == 0 else [-APERTURE, APERTURE],
                            color='.6', lw=.4, ls=':')
                q = 4 / np.sqrt(2)
                ax.add_patch(Circle((q, q), .75, fill=False, ec=HP, lw=.8, ls=(0, (2, 1.2))))
                ax.set_xlim(-4.9, 4.9)
                ax.set_ylim(-4.9, 4.9)
                ax.set_aspect('equal')
                ax.set_axis_off()
                if o == 0 and k == 0:
                    ax.text(.5, 1.02, f'{centres[b]:+.1f}°'.replace('-', '−'), transform=ax.transAxes, ha='center',
                            va='bottom', fontsize=6, color='.4')
                if b == 0:
                    ax.text(-.08, .5, lab if k == 0 else ('Upper half minus\nlower half' if o == 0 else
                                                          'Right half minus\nleft half'),
                            transform=ax.transAxes, ha='right', va='center', fontsize=6.5, color=FG if k == 0 else '.35')
    cax = fig.add_axes([(x0 + 7 * (size + gap) + .05) / W, (H - .55 - 4 * size - .5) / H, .07 / W, (2.5 * size) / H])
    cb = fig.colorbar(dim, cax=cax, ticks=[-dvm, 0, dvm])
    cb.ax.set_yticklabels([f'−{dvm:.0f}%', '0', f'{dvm:.0f}%'], fontsize=6)
    cb.ax.tick_params(length=2, width=.5, pad=1)
    cb.outline.set_linewidth(.4)
    cb.ax.text(.5, 1.04, 'Stronger\nthan mirror', transform=cb.ax.transAxes, ha='center', va='bottom', fontsize=6,
               color='#B2182B')
    cb.ax.text(.5, -.04, 'Weaker', transform=cb.ax.transAxes, ha='center', va='top', fontsize=6, color='#2166AC')
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
    dec_txt = {'free': 'unconstrained decoder, every pixel decoded',
               'outline': 'decoder with the bar outline given: only the pixels the bar covers are decoded, the rest is 0'}[decoder]
    legend = f"""
**Figure S. The decoded bars, averaged per bar position with the HP location rotated to the upper right.** Every run's
decoded frames ({dec_txt}, V1, V2 and V3; pRFs: {source} fold fits) and its true bar frames were rotated in
90° steps so that the HP location (red dashed circle) lies upper right; 90° steps keep the bars vertical or horizontal.
Frames with a bar inside the aperture (grey circle, {APERTURE}°) were grouped by bar orientation and by the position of
the bar centre along its motion ({N_BINS} bins; centres above row 1), averaged over runs and ROIs per subject, then over
subjects (n = {mean.shape[0]}). Dotted outline, the true bar. Frames per bin, all subjects and ROIs: vertical
{', '.join(str(int(v)) for v in nb[0])}; horizontal {', '.join(str(int(v)) for v in nb[1])}. Rows 2 and 4: each
subject's decoded image minus its mirror image (vertical bars flipped top to bottom, horizontal bars left to right, so the
bar maps onto itself and the HP quadrant onto a neighbouring one), in % of the mean decoded intensity on the bar,
smoothed (Gaussian, σ = 0.4°), averaged over subjects, and shown on the bar's footprint (one pixel of margin) in the half
nearer HP only, the other half being its negative: blue, weaker than the mirror image. Where the bar crosses the
quadrant facing HP (upper right) the near half is blue; on the other side of the aperture the near half (a quadrant next
to HP) is red against the opposite quadrant. b, Decoded intensity on the
bar's pixels in the half of the bar nearer HP (upper half of a vertical bar, right half of a horizontal one) minus the
half farther away, in % of the bar's mean, vertical and horizontal bars pooled; positive positions are on HP's side of the
aperture (shaded), where the near half lies in the quadrant facing HP and the far half in a neighbouring one; on the
other side the near half lies in a quadrant next to HP and the far half in the opposite quadrant. Line and band: mean ± SEM over subjects;
Wilcoxon signed-rank test against 0 per position, two-sided, uncorrected (* p < .05, ** p < .01): {', '.join(f'{c:+.1f}° {v:+.2f}% p = {p:.3f}' for c, v, p in zip(centres, m, ps))}.
"""
    save(fig, name, legend)
    print('frames per bin', nb)
    print('half contrast', np.round(m, 2), 'p', np.round(ps, 3))


QUAD = {'HP': (1, 1), 'Next to HP (upper left)': (-1, 1), 'Next to HP (lower right)': (1, -1), 'Opposite': (-1, -1)}


def pooled_condition(mean):
    """Vertical bars and horizontal bars reflected onto them across the diagonal through HP (x <-> y), which keeps HP
    upper right: per subject × bin, (decoded, decoded − mean over HP locations, true bar)."""
    return (mean[:, 0] + np.swapaxes(mean[:, 1], -1, -2)) / 2


def segment_stats(P, grid, b):
    """Per quadrant: the decoded − mean-over-HP difference on the bar's pixels in that quadrant, % of the bar's mean."""
    X, Y = grid[:, 0].reshape(50, 50), grid[:, 1].reshape(50, 50)
    bar = (P[:, b, 2].mean(0) > .5) & (np.hypot(X, Y) <= APERTURE)
    bm = np.array([P[s_, b, 0][bar].mean() for s_ in range(len(P))])
    out = {}
    for name, (sx, sy) in QUAD.items():
        mk = bar & (np.sign(X) == sx) & (np.sign(Y) == sy)
        if mk.sum() < 10:
            continue
        v = np.array([P[s_, b, 1][mk].mean() for s_ in range(len(P))]) / bm * 100
        out[name] = (v.mean(), v.std(ddof=1) / np.sqrt(len(v)), stats.wilcoxon(v).pvalue, int(mk.sum()))
    return out


def plot_condition(source, name='figS_decoded_bars_rotated', decoder='free'):
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from scipy import ndimage
    from retsupp.visualize.paper.paper_style import FG, HP, STIM, save
    z = np.load(Path('notes/data') / data_name(source, decoder).replace('decoded_bars_', 'decoded_bars_condition_'))
    mean, n, grid, edges = z['mean'], z['n'], z['grid'], z['edges']
    centres = (edges[:-1] + edges[1:]) / 2
    P = pooled_condition(mean)                                   # subject, bin, 3, 50, 50
    c = np.linspace(-5, 5, 50)
    X, Y = np.meshgrid(c, c)
    inside = np.hypot(X, Y) <= APERTURE
    show = [b for b in range(len(centres)) if abs(centres[b]) < 2.]
    W, H = 7.25, 2.75
    fig = plt.figure(figsize=(W, H))
    cm = mcolors.LinearSegmentedColormap.from_list('stim', ['#FFFFFF', STIM])
    size, gap, x0 = 1.0, .12, 1.0
    dvm = 3.
    grp_dec = P[:, :, 0].mean(0)
    vmax = np.nanpercentile(grp_dec[show], 99.5)
    seg = {b: segment_stats(P, grid, b) for b in range(len(centres))}
    for k, lab in enumerate(['Decoded bar', 'Minus the mean\nover HP locations']):
        yb = H - .45 - (k + 1) * size - k * .12
        for j, b in enumerate(show):
            ax = fig.add_axes([(x0 + j * (size + gap)) / W, yb / H, size / W, size / H])
            foot = ndimage.binary_dilation(P[:, b, 2].mean(0) > .5, iterations=1) & inside
            if k == 0:
                img = grp_dec[b]
                im = ax.imshow(img, origin='lower', extent=[-5.1, 5.1, -5.1, 5.1], cmap=cm, vmin=0, vmax=vmax,
                               interpolation='bilinear')
            else:
                rel = []
                for s_ in range(len(P)):
                    bm = P[s_, b, 0][foot].mean()
                    d = np.where(foot, 100 * P[s_, b, 1] / bm, 0)
                    rel.append(ndimage.gaussian_filter(d, 1) / np.maximum(ndimage.gaussian_filter(foot * 1., 1), 1e-6))
                img = np.where(foot, np.mean(rel, 0), np.nan)
                im = ax.imshow(img, origin='lower', extent=[-5.1, 5.1, -5.1, 5.1], cmap='RdBu_r', vmin=-dvm, vmax=dvm,
                               interpolation='nearest')
                dim = im
                ax.contour(X, Y, (P[:, b, 2].mean(0) > .5) & inside, levels=[.5], colors=['.5'], linewidths=.4)
                for qn, (sx, sy) in QUAD.items():
                    if qn in seg[b] and seg[b][qn][2] < .05:
                        p = seg[b][qn][2]
                        st = '***' if p < .001 else '**' if p < .01 else '*'
                        bx = centres[b]
                        ax.text(bx + (1.0 if bx >= 0 else -1.0), sy * 1.6, st, ha='center', va='center', fontsize=7,
                                color=FG)
            clip = Circle((0, 0), APERTURE, transform=ax.transData, fc='none', ec='none')
            ax.add_patch(clip)
            im.set_clip_path(clip)
            ax.add_patch(Circle((0, 0), APERTURE, fill=False, ec='.6', lw=.5))
            q = 4 / np.sqrt(2)
            ax.add_patch(Circle((q, q), .75, fill=False, ec=HP, lw=.9, ls=(0, (2, 1.2))))
            if j == 0 and k == 0:
                ax.text(q + .2, q + .95, 'HP', color=HP, fontsize=6.5, ha='center', va='bottom')
            ax.set_xlim(-4.9, 4.9)
            ax.set_ylim(-4.9, 4.9)
            ax.set_aspect('equal')
            ax.set_axis_off()
            if k == 0:
                ax.text(.5, 1.0, f'Bar at {centres[b]:+.1f}°'.replace('-', '−').replace('+0.0', '0').replace('−0.0', '0'),
                        transform=ax.transAxes, ha='center', va='bottom', fontsize=6.5, color='.35')
            if j == 0:
                ax.text(-.06, .5, lab, transform=ax.transAxes, ha='right', va='center', fontsize=7, color=FG)
    cax = fig.add_axes([(x0 + len(show) * (size + gap) + .05) / W, .3 / H, .07 / W, .8 / H])
    cb = fig.colorbar(dim, cax=cax, ticks=[-dvm, 0, dvm])
    cb.ax.set_yticklabels([f'−{dvm:.0f}%', '0', f'+{dvm:.0f}%'], fontsize=6)
    cb.ax.tick_params(length=2, width=.5, pad=1)
    cb.outline.set_linewidth(.4)
    cb.ax.text(.5, 1.06, 'Stronger', transform=cb.ax.transAxes, ha='center', va='bottom', fontsize=6, color='#B2182B')
    cb.ax.text(.5, -.06, 'Weaker', transform=cb.ax.transAxes, ha='center', va='top', fontsize=6, color='#2166AC')
    fig.text(.2 / W, (H - .18) / H, 'The same bar is decoded weaker where it passes the quadrant facing HP '
             '(V1–V3, mean of 30 subjects)', fontsize=7.5, color=HP, va='center')
    dec_txt = {'free': 'unconstrained decoder, every pixel decoded',
               'outline': 'decoder with the bar outline given: only the pixels the bar covers are decoded'}[decoder]
    segtxt = '; '.join(f'bar at {centres[b]:+.1f}°: ' + ', '.join(f'{qn} {v[0]:+.2f} ± {v[1]:.2f}% (p = {v[2]:.3f})'
                                                               for qn, v in seg[b].items()) for b in show)
    nb = n.sum(0)
    legend = f"""
**Figure S. The same bar is decoded at lower intensity where it passes the quadrant facing the HP location.** The bar
sequence is the same in every run, so each decoded frame ({dec_txt}; V1, V2 and V3; pRFs: {source} fold fits) can be
compared with the mean decoded frame at the same time point over the subject's runs, averaged within each HP location
first and then over the four (three for sub-01 and sub-02, whose runs cover three HP locations): the stimulus is the same
on both sides of the subtraction and only the HP location differs. Each run's frames were then rotated in 90° steps so
that the HP location (red dashed circle) lies upper right, grouped by bar position (bins of {edges[1] - edges[0]:.1f}°;
the positions where the whole bar lies inside the {APERTURE}° aperture are shown), and horizontal bars were reflected
across the diagonal through HP onto vertical ones (a symmetry of the rotated display that keeps HP in place). Top, the
decoded bar; bottom, the difference, in % of the subject's mean decoded intensity on the bar, smoothed (Gaussian, σ =
0.2°), on the bar's footprint (grey outline); means over runs and ROIs per subject, then over subjects (n =
{len(P)}). Frames per bin, vertical and horizontal, all subjects and ROIs: {', '.join(str(int(v)) for v in nb.sum(0))}.
Per quadrant, the difference on the part of the bar in that quadrant (mean ± SEM over subjects; two-sided Wilcoxon
signed-rank test against 0, uncorrected; stars on the image: * p < .05, ** p < .01, *** p < .001): {segtxt}.
"""
    save(fig, name, legend)
    print(segtxt)


def _seg_contrasts(seg, rois, orientation):
    """Per subject (mean over ``rois``): the HP-quadrant segment of bars on HP's side (0.4° < centre < 1.5°) and the
    next-to-HP segment of bars on the far side (−1.5° < centre < −0.4°; upper left for vertical bars, lower right for
    horizontal ones, pooled: upper left)."""
    d = seg[seg.roi.isin(rois) & (seg.orientation == orientation)]
    near = d[(d.centre > .4) & (d.centre < 1.5) & (d.quadrant == 'hp')]
    nq = 'next_lr' if orientation == 'horizontal' else 'next_ul'
    far = d[(d.centre < -.4) & (d.centre > -1.5) & (d.quadrant == nq)]
    f = lambda x: x.groupby(['subject', 'roi']).diff_pct.mean().groupby('subject').mean()
    return f(near), f(far)


def _smooth_masked(img, mask, sigma=1.):
    from scipy import ndimage
    num = ndimage.gaussian_filter(np.where(mask, np.nan_to_num(img), 0), sigma)
    den = ndimage.gaussian_filter(mask * 1., sigma)
    return np.where(mask, num / np.maximum(den, 1e-6), np.nan)


def _disc(ax, img, cmap, vmin, vmax, hp=True, outline=None):
    from matplotlib.patches import Circle
    from retsupp.visualize.paper.paper_style import HP
    im = ax.imshow(img, origin='lower', extent=[-5.1, 5.1, -5.1, 5.1], cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation='nearest')
    clip = Circle((0, 0), APERTURE, transform=ax.transData, fc='none', ec='none')
    ax.add_patch(clip)
    im.set_clip_path(clip)
    ax.add_patch(Circle((0, 0), APERTURE, fill=False, ec='.6', lw=.4))
    if outline is not None:
        c = np.linspace(-5, 5, 50)
        ax.contour(c, c, outline, levels=[.5], colors=['.55'], linewidths=.3)
    if hp:
        q = 4 / np.sqrt(2)
        ax.add_patch(Circle((q, q), .75, fill=False, ec=HP, lw=.7, ls=(0, (2, 1.2))))
    ax.set_xlim(-4.9, 4.9)
    ax.set_ylim(-4.9, 4.9)
    ax.set_aspect('equal')
    ax.set_axis_off()
    return im


def _half_test(e_o, b, oname, centre):
    """Per subject (V1–V3 mean): the half of the bar nearer HP minus the other half (upper − lower for vertical bars,
    right − left for horizontal ones), two-sided Wilcoxon; None for the central bin."""
    if abs(centre) < .3:
        return None
    g = e_o[e_o.bin == b].groupby(['subject', 'quadrant', 'roi']).diff_pct.mean().groupby(['subject', 'quadrant']).mean().unstack()
    if oname == 'vertical':
        a, c = ('hp', 'next_lr') if centre > 0 else ('next_ul', 'opposite')
    else:
        a, c = ('hp', 'next_ul') if centre > 0 else ('next_lr', 'opposite')
    if a not in g or c not in g:
        return None
    d = (g[a] - g[c]).dropna()
    return stats.wilcoxon(d).pvalue if len(d) > 10 else None


def _half_contrasts(seg, rois, orientation):
    """Per subject: bars on HP's side (0.4–1.5°), HP-quadrant half minus the other half; bars on the far side,
    next-to-HP half minus the opposite-quadrant half; and the bar's overall difference."""
    d = seg[seg.roi.isin(rois) & (seg.orientation == orientation)]
    w = d.pivot_table(index=['subject', 'roi', 'bin', 'centre'], columns='quadrant', values='diff_pct').reset_index()
    nq, oq = ('next_lr', 'next_ul') if orientation == 'horizontal' else ('next_ul', 'next_lr')
    a = w[(w.centre > .4) & (w.centre < 1.5)]
    b = w[(w.centre < -.4) & (w.centre > -1.5)]
    f = lambda x, v: v.groupby([x.subject, x.roi]).mean().groupby(level=0).mean()
    overall = d.groupby(['subject', 'roi']).diff_pct.mean().groupby('subject').mean()
    return f(a, a['hp'] - a[oq]), f(b, b[nq] - b['opposite']), overall


def plot_split(source='jointhrf', decoder='outline', n_bins=11, name='figS_decoded_bars_rotated'):
    """Main decoded-bars supplement: V1–V3, vertical and horizontal bars in separate rows (no reflection)."""
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    from scipy import ndimage
    from retsupp.visualize.paper.paper_style import FG, HP, STIM, save
    tag = f'{source}_{decoder}_b{n_bins}'
    z = np.load(Path('notes/data') / f'decoded_bars_images_{tag}_2026-11.npz')
    seg = pd.read_csv(Path('notes/data') / f'decoded_bars_segments_{tag}_2026-11.tsv', sep='\t')
    centres, frames = z['centres'], z['frames_early']
    c = np.linspace(-5, 5, 50)
    X, Y = np.meshgrid(c, c)
    inside = np.hypot(X, Y) <= APERTURE
    W, H = 7.25, 3.3
    fig = plt.figure(figsize=(W, H))
    cm = mcolors.LinearSegmentedColormap.from_list('stim', ['#FFFFFF', STIM])
    size, gap, x0 = .5, .035, 1.0
    dvm = 4.
    vmax = np.nanpercentile(z['early_dec'], 99.5)
    # per-segment tests (V1–V3 mean per subject), two-sided, uncorrected
    early = seg[seg.roi.isin(EARLY_ROIS)]
    labels = [('Vertical bars,\nleft to right', 'Minus mean over HP\nlocations, centred'),
              ('Horizontal bars,\nbottom to top', 'Minus mean over HP\nlocations, centred')]
    y = H - .5
    for o, oname in enumerate(['vertical', 'horizontal']):
        e_o = early[early.orientation == oname]
        for k in range(2):
            y -= size + (.04 if k else .1)
            for b, cb_ in enumerate(centres):
                ax = fig.add_axes([(x0 + b * (size + gap)) / W, y / H, size / W, size / H])
                bar = z['early_bar'][o, b]
                foot = ndimage.binary_dilation(bar > .5, iterations=1) & inside
                if k == 0:
                    _disc(ax, z['early_dec'][o, b], cm, 0, vmax)
                else:
                    dimg = z['early_diff'][o, b]
                    dimg = dimg - np.nanmean(dimg[bar > .5])               # centred on the bar's own mean difference
                    im = _disc(ax, _smooth_masked(dimg, foot), 'RdBu_r', -dvm, dvm, outline=bar > .5)
                    p = _half_test(e_o, b, oname, cb_)
                    if p is not None and p < .05:
                        st = '***' if p < .001 else '**' if p < .01 else '*'
                        px, py = (cb_ - 1.1 * np.sign(cb_), 1.7) if o == 0 else (1.7, cb_ - 1.0 * np.sign(cb_))
                        ax.text(px, py, st, ha='center', va='center', fontsize=5.5, color=FG)
                if o == 0 and k == 0:
                    ax.text(.5, 1.02, f'{cb_:+.1f}°'.replace('-', '−').replace('+0.0', '0').replace('−0.0', '0'), transform=ax.transAxes,
                            ha='center', va='bottom', fontsize=5.5, color='.4')
                if b == 0:
                    ax.text(-.08, .5, labels[o][k], transform=ax.transAxes, ha='right', va='center', fontsize=6.5,
                            color=FG if k == 0 else '.35')
        y -= .12
    cax = fig.add_axes([(x0 + len(centres) * (size + gap) + .06) / W, .45 / H, .06 / W, 1.0 / H])
    cbar = fig.colorbar(im, cax=cax, ticks=[-dvm, 0, dvm])
    cbar.ax.set_yticklabels([f'−{dvm:.0f}%', '0', f'+{dvm:.0f}%'], fontsize=5.5)
    cbar.ax.tick_params(length=2, width=.5, pad=1)
    cbar.outline.set_linewidth(.4)
    cbar.ax.text(.5, 1.05, 'Stronger', transform=cbar.ax.transAxes, ha='center', va='bottom', fontsize=5.5, color='#B2182B')
    cbar.ax.text(.5, -.05, 'Weaker', transform=cbar.ax.transAxes, ha='center', va='top', fontsize=5.5, color='#2166AC')
    fig.text(.2 / W, (H - .17) / H, 'The same bar is decoded weaker where it passes the quadrant facing HP, for vertical and '
             'horizontal bars (V1–V3)', fontsize=7.5, color=HP, va='center')
    summ = {}
    for oname in ('vertical', 'horizontal', 'pooled'):
        summ[oname] = [(v.mean(), v.sem(), stats.wilcoxon(v).pvalue, len(v)) for v in _half_contrasts(seg, EARLY_ROIS, oname)]
    fmt = lambda t: f'{t[0]:+.2f} ± {t[1]:.2f}% (p = {t[2]:.3f})'
    summary = '; '.join(f'{o} bars: HP-quadrant half minus the other half, bars on HP\'s side (centre 0.4–1.5°), '
                        f'{fmt(v[0])}; next-to-HP half minus the opposite-quadrant half, bars on the far side, {fmt(v[1])}'
                        for o, v in summ.items() if o != 'pooled')
    summary += (f'; pooled (horizontal bars reflected onto vertical ones) {fmt(summ["pooled"][0])} and '
                f'{fmt(summ["pooled"][1])}')
    offset = (f'vertical bars {fmt(summ["vertical"][2])}, horizontal bars {fmt(summ["horizontal"][2])}')
    dec_txt = {'free': 'unconstrained decoder', 'outline': 'decoder with the bar outline given: only the pixels the bar '
               'covers are decoded'}[decoder]
    legend = f"""
**Figure S. The same bar is decoded at lower intensity where it passes the quadrant facing the HP location, for vertical
and for horizontal bars.** The bar sequence is the same in every run, so each decoded frame ({dec_txt}; pRFs: {source}
fold fits) was compared with the mean decoded frame at the same time point over the subject's runs, averaged within each
HP location first and then over the four (three for sub-01 and sub-02); the stimulus is identical on both sides of the
subtraction. Each run's frames were rotated in 90° steps so that the HP location (red dashed circle) lies upper right;
vertical bars (rows 1, 2) and horizontal bars (rows 3, 4) are shown separately, grouped by the position of the bar centre
({n_bins} bins of {centres[1] - centres[0]:.2f}° across the {APERTURE}° aperture; bars at the edge are clipped by the
aperture). Each run's decoded frames were first scaled to a common mean intensity on the bar. Rows 1 and 3: the decoded
bar; rows 2 and 4: the difference, in % of the mean decoded intensity on the bar, on the bar's footprint (grey outline),
smoothed (Gaussian, σ = 0.2°), and centred on the bar's own mean difference, so the images show where along the bar it is
weaker or stronger. The centring removes an offset that differs between the orientations (the difference averaged over
the whole bar: {offset}): in the HP-rotated frame, bars of one orientation are decoded somewhat stronger and bars of the
other somewhat weaker; this is an interaction of bar orientation with the diagonal the HP location lies on (it cancels
when the orientations are pooled, and it does not enter the within-run quadrant comparisons of Fig. 6). Per subject the mean over runs and over V1, V2 and
V3, then over subjects (n = {int(z['n_early'])}). Frames per bin (V1–V3, all subjects): vertical
{', '.join(str(int(v)) for v in frames[0])}; horizontal {', '.join(str(int(v)) for v in frames[1])}. Stars, next to the half of the bar nearer HP: that half minus the other half, two-sided
Wilcoxon signed-rank test, uncorrected (* p < .05, ** p < .01, *** p < .001). Mean ± SEM over subjects, two-sided
Wilcoxon: {summary}. The half of the bar in the quadrant facing HP is weaker for both orientations, an internal
replication.
"""
    save(fig, name, legend)
    print(summary)


def plot_per_roi(source='jointhrf', decoder='outline', n_bins=11, name='figS_decoded_bars_per_roi'):
    import matplotlib.pyplot as plt
    from scipy import ndimage
    from retsupp.visualize.paper.paper_style import FG, HP, MUTED, ROI_GROUPS, offset_spines, save
    tag = f'{source}_{decoder}_b{n_bins}'
    z = np.load(Path('notes/data') / f'decoded_bars_images_{tag}_2026-11.npz')
    seg = pd.read_csv(Path('notes/data') / f'decoded_bars_segments_{tag}_2026-11.tsv', sep='\t')
    centres = z['centres']
    show = [b for b in range(len(centres)) if abs(centres[b]) < 2.]
    rois = list(z['rois'])
    c = np.linspace(-5, 5, 50)
    X, Y = np.meshgrid(c, c)
    inside = np.hypot(X, Y) <= APERTURE
    bars = (z['early_bar'][0] + np.swapaxes(z['early_bar'][1], -1, -2)) / 2      # pooled footprint per bin
    rows = {}
    for roi in rois:
        a_, b_, _ = _half_contrasts(seg, [roi], 'pooled')
        rows[roi] = [(v.mean(), v.std(ddof=1) / np.sqrt(len(v)) * stats.t.ppf(.975, len(v) - 1),
                      stats.wilcoxon(v).pvalue, len(v)) for v in (a_, b_)]
    q = [stats.false_discovery_control([rows[r][k][2] for r in rois]) for k in range(2)]
    W, H = 7.25, 6.0
    fig = plt.figure(figsize=(W, H))
    size, gap, x0 = .5, .03, 1.0
    dvm = 6.
    top = H - .45
    rowpos = {}
    yy = top
    for gname, groi in ROI_GROUPS:
        for r in groi:
            yy -= size + .03
            rowpos[r] = yy
        yy -= .14
    for roi in rois:
        i = rois.index(roi)
        for j, b in enumerate(show):
            ax = fig.add_axes([(x0 + j * (size + gap)) / W, rowpos[roi] / H, size / W, size / H])
            foot = ndimage.binary_dilation(bars[b] > .5, iterations=1) & inside
            im = _disc(ax, _smooth_masked(z['roi_diff'][i, b], foot), 'RdBu_r', -dvm, dvm, outline=bars[b] > .5)
            if roi == rois[0]:
                ax.text(.5, 1.03, f'{centres[b]:+.1f}°'.replace('-', '−').replace('+0.0', '0').replace('−0.0', '0'), transform=ax.transAxes,
                        ha='center', va='bottom', fontsize=6, color='.4')
            if j == 0:
                ax.text(-.08, .5, f'{roi}\nn = {int(z["roi_n"][i])}', transform=ax.transAxes, ha='right', va='center',
                        fontsize=6.5, color=FG)
    for gname, groi in ROI_GROUPS:
        y0, y1 = rowpos[groi[-1]], rowpos[groi[0]] + size
        fig.add_artist(plt.Line2D([.2 / W] * 2, [y0 / H, y1 / H], color=MUTED, lw=.7))
        fig.text(.12 / W, (y0 + y1) / 2 / H, gname, rotation=90, ha='center', va='center', fontsize=6, color='.4')
    # Right: the two segment contrasts per ROI.
    xa = x0 + len(show) * (size + gap) + .4
    for k, (title, col) in enumerate([('HP half − other half,\nbars on HP\'s side', HP),
                                      ('Next-to-HP half −\nopposite, far side', '#52606D')]):
        ax = fig.add_axes([(xa + k * 1.0) / W, rowpos[rois[-1]] / H, .82 / W, (top - rowpos[rois[-1]]) / H])
        for roi in rois:
            m, ci, p, n = rows[roi][k]
            yc = rowpos[roi] + size / 2
            ax.errorbar(m, yc, xerr=ci, fmt='o', ms=3.5, color=col, mec='w', mew=.5, lw=1., capsize=0)
            qq = q[k][rois.index(roi)]
            if qq < .05:
                ax.text(m + ci + .4, yc, '***' if qq < .001 else '**' if qq < .01 else '*', va='center', fontsize=6.5,
                        color=col)
        ax.axvline(0, color='.6', lw=.7, ls='--', zorder=0)
        ax.set_ylim(rowpos[rois[-1]], top)
        ax.set_yticks([])
        ax.spines['left'].set_visible(False)
        ax.set_xlim(-6, 6)
        ax.set_xticks([-5, 0, 5])
        ax.set_xticklabels(['−5', '0', '5'], fontsize=6)
        ax.set_xlabel('Difference (%)', fontsize=6.5)
        ax.set_title(title, fontsize=6.5, color=col, pad=3)
        offset_spines(ax)
    cax = fig.add_axes([(x0 + .85) / W, .22 / H, 2.0 / W, .07 / H])
    cbar = fig.colorbar(im, cax=cax, orientation='horizontal', ticks=[-dvm, 0, dvm])
    cbar.ax.set_xticklabels([f'−{dvm:.0f}%', '0', f'+{dvm:.0f}%'], fontsize=6)
    cbar.ax.tick_params(length=2, width=.5, pad=1)
    cbar.outline.set_linewidth(.4)
    cbar.ax.text(-.04, .5, 'Weaker', transform=cbar.ax.transAxes, ha='right', va='center', fontsize=6, color='#2166AC')
    cbar.ax.text(1.04, .5, 'Stronger', transform=cbar.ax.transAxes, ha='left', va='center', fontsize=6, color='#B2182B')
    fig.text(.2 / W, (H - .17) / H, 'Decoded bar minus the mean over HP locations, per ROI (HP upper right; vertical and '
             'horizontal bars pooled)', fontsize=7.5, color=HP, va='center')
    txt = '; '.join(f'{r} {rows[r][0][0]:+.2f}% (95% CI ± {rows[r][0][1]:.2f}, p = {rows[r][0][2]:.3f}, q = {q[0][i]:.3f}) / '
                    f'{rows[r][1][0]:+.2f}% (± {rows[r][1][1]:.2f}, p = {rows[r][1][2]:.3f}, q = {q[1][i]:.3f})'
                    for i, r in enumerate(rois))
    legend = f"""
**Figure S. Decoded bars minus their mean over HP locations, per ROI.** As the decoded-bars figure (decoder with the bar
outline given; pRFs: {source} fold fits; HP rotated to the upper right in 90° steps), per ROI, with horizontal bars
reflected across the diagonal through HP onto vertical ones (a symmetry of the rotated display that keeps HP in place),
for the bar positions with the bar inside the aperture ({n_bins} bins in total); difference in % of the mean decoded
intensity on the bar, on the bar's footprint, smoothed (σ = 0.2°), mean over subjects (n per ROI at the left; an ROI
was skipped for a subject with fewer than 20 usable voxels). Right: per ROI, for bars on HP's side (centre 0.4–1.5°) the
difference on the half of the bar in the quadrant facing HP minus that on the other half, and for bars on the far side
(centre −1.5 to −0.4°) the half in the quadrant next to HP (upper left) minus the half in the opposite quadrant; mean and 95% CI over subjects; two-sided Wilcoxon signed-rank test,
Benjamini-Hochberg across the nine ROIs per contrast (* q < .05). HP-quadrant / next-to-HP: {txt}.
"""
    save(fig, name, legend)
    print(txt)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('step', choices=['collect', 'collect_condition', 'collect_per_roi', 'plot', 'plot_condition',
                                     'plot_per_roi', 'plot_split'])
    ap.add_argument('--bids-folder', default='/shares/zne.uzh/gdehol/ds-retsupp')
    ap.add_argument('--source', default='jointhrf')
    ap.add_argument('--out-dir', default='notes/data')
    ap.add_argument('-j', '--jobs', type=int, default=4)
    ap.add_argument('--decoder', default='outline', choices=['free', 'outline'], help='outline (main figure) or free')
    ap.add_argument('--name', default='figS_decoded_bars_rotated')
    ap.add_argument('--n-bins', type=int, default=N_BINS, help='bar-position bins across the aperture')
    ap.add_argument('--exclude-subjects', nargs='*', type=int, default=[], help='collect_per_roi: subjects left out')
    a = ap.parse_args()
    if a.n_bins != N_BINS:                     # module-level so the forked workers see it
        N_BINS = a.n_bins
        EDGES = np.linspace(-APERTURE, APERTURE, N_BINS + 1)
    if a.step == 'collect_per_roi':
        collect_condition_per_roi(a.bids_folder, a.source, a.out_dir, a.jobs, a.decoder, a.exclude_subjects)
    elif a.step == 'plot_per_roi':
        plot_per_roi(a.source, a.decoder, a.n_bins)
    elif a.step == 'plot_split':
        plot_split(a.source, a.decoder, a.n_bins, a.name)
    elif a.step == 'collect_condition':
        collect_condition(a.bids_folder, a.source, a.out_dir, a.jobs, a.decoder)
    elif a.step == 'plot_condition':
        plot_condition(a.source, a.name, a.decoder)
    elif a.step == 'collect':
        collect(a.bids_folder, a.source, a.out_dir, a.jobs, a.decoder)
    else:
        plot(a.source, a.name, a.decoder)
