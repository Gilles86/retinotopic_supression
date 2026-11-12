"""Figure 6, collect step (cluster): reduce the decoded runs of ``retsupp/decode/decode_final.py`` to small summaries.

Reads every ``decoded_ses-S_run-R_roi-ROI.npz`` under ``derivatives/decoded_final_2026-11/<source>/<decoder>/sub-XX/``
and writes, to ``--out-dir`` (``notes/data`` of the repo, 2026-11 suffix):

``decoding_quality_<source>_2026-11.tsv``
    Per run × ROI × decoder: how well the decoded frames match the true bar. Per frame with bar pixels inside the
    aperture, the Pearson correlation over the aperture pixels between decoded and true frame; ``r`` is the mean over
    frames. Null: the same with the true frames shuffled (``N_SHUFFLE`` random permutations of the frame order;
    ``r_shuffle`` their mean, ``r_shuffle_p95`` the 95th percentile of the per-permutation means). ``r_lag_k``: the
    true frames shifted by k TRs (k = -3 … 3), to check the decoded time course is not displaced.

``decoding_bands_<source>_2026-11.tsv``
    Per run × ROI × decoder × quadrant × eccentricity band: the readout of ``decode_final.readout`` (mean decoded
    intensity over the pixel-frames the true bar covers, ``drive_bar``, and over all pixel-frames, ``mean_all``)
    restricted to the band. Bands: the three of the figure (0–1.5°, 1.5–2.5°, 2.5° – aperture), ``all`` (the whole
    aperture, identical to the original readout) and 0.25°-wide rings for profiles.

``decoding_maps_<source>_2026-11.npz``
    Per subject × ROI × decoder, each run's visual field rotated in 90° steps so the HP location sits upper right,
    then averaged over runs: ``drive`` (per pixel, mean decoded intensity over the frames the bar covers it) and
    ``mean`` (over all frames). Shape (n_sub, n_roi, n_dec, 50, 50), rows = y (bottom to top), columns = x.

``decoding_frames_<source>_2026-11.npz``
    Frames for display: the true bar frames (identical over runs up to a few pixels), the group mean of the decoded
    frames per ROI × decoder (mean over runs, then subjects), and for the example subjects the decoded frames of
    the run mean (free decoder). Their single runs go to ``decoding_example_runs_<source>_2026-11.npz`` (large; trimmed
    to the shown run before syncing).

Usage (compute node)::

    python -m retsupp.visualize.paper.fig6_collect --out-dir notes/data -j 4
"""
from __future__ import annotations

import argparse
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ALL_ROIS = ['V1', 'V2', 'V3', 'V3AB', 'IPS', 'hV4', 'VO', 'LO', 'TO',   # fixed order (also seeds the shuffles)
            'mid', 'high', 'dorsal', 'occtemp']                         # pooled sets (decode_final.POOLED), appended
ANGLE = {'upper_right': 45, 'upper_left': 135, 'lower_left': 225, 'lower_right': 315}
QUADRANTS = {'upper_right': (1, 1), 'upper_left': (-1, 1), 'lower_left': (-1, -1), 'lower_right': (1, -1)}
BANDS = [('0-1.5', 0., 1.5), ('1.5-2.5', 1.5, 2.5), ('2.5-ap', 2.5, np.inf)]
RINGS = np.arange(0, 3.26, .25)
LAGS = range(-3, 4)
N_SHUFFLE = 100
EXAMPLES = (23, 3)
N_TR = 258


@lru_cache(maxsize=None)
def aperture(bids, subject, session, run):
    from retsupp.utils.data import Subject
    return float(Subject(subject, bids_folder=bids).get_experimental_settings(session, run)['radius_bar_aperture'])


def rotation_index(grid, hp):
    """Flat index map: rotated[i] = original[idx[i]], rotating the field so ``hp`` lands upper right."""
    th = np.deg2rad(45 - ANGLE[hp])          # rotate by th: point p goes to R(th) p
    c, s = np.cos(th), np.sin(th)
    # value at new position q comes from R(-th) q
    src = np.stack([c * grid[:, 0] + s * grid[:, 1], -s * grid[:, 0] + c * grid[:, 1]], 1)
    key = {tuple(np.round(p, 3)): i for i, p in enumerate(grid)}
    return np.array([key[tuple(np.round(p, 3))] for p in src])


def frame_corr(dec, par):
    """Row-wise Pearson r between (T, P) arrays; NaN where either row is constant."""
    d = dec - dec.mean(1, keepdims=True)
    p = par - par.mean(1, keepdims=True)
    den = np.sqrt((d ** 2).sum(1) * (p ** 2).sum(1))
    with np.errstate(invalid='ignore', divide='ignore'):
        return (d * p).sum(1) / den


def one_run(args):
    f, bids, source = args
    dec_name, sub_dir = f.parent.parent.name, f.parent.name
    subject = int(sub_dir.split('-')[1])
    _, ses, run, roi = f.stem.split('_')
    session, run, roi = int(ses.split('-')[1]), int(run.split('-')[1]), roi.split('-')[1]
    with np.load(f) as z:
        decoded = z['decoded'].astype(np.float32)
        paradigm = z['paradigm'].astype(np.float32)
        grid = z['grid']
        hp = str(z['hp_location'])
    T = decoded.shape[0]
    D, P = decoded.reshape(T, -1), paradigm.reshape(T, -1)
    ap = aperture(bids, subject, session, run)
    ecc = np.hypot(grid[:, 0], grid[:, 1])
    inside = ecc <= ap
    meta = dict(subject=subject, session=session, run=run, roi=roi, decoder=dec_name, source=source, hp_location=hp,
                aperture=ap)

    # Quality: per-frame spatial correlation inside the aperture.
    Di, Pi = D[:, inside], P[:, inside]
    has = Pi.sum(1) > 0
    r = frame_corr(Di[has], Pi[has])
    rng = np.random.default_rng(subject * 1000 + session * 100 + run * 10 + ALL_ROIS.index(roi))
    idx = np.flatnonzero(has)
    null = []
    for _ in range(N_SHUFFLE):
        perm = rng.permutation(idx)
        null.append(np.nanmean(frame_corr(Di[idx], Pi[perm])))
    q = dict(meta, r=np.nanmean(r), r_median=np.nanmedian(r), n_frames=int(has.sum()), r_shuffle=np.mean(null),
             r_shuffle_p95=np.percentile(null, 95))
    for k in LAGS:
        # decoded frame t vs true frame t - k
        if k >= 0:
            dd, pp = Di[k:], Pi[:T - k]
        else:
            dd, pp = Di[:T + k], Pi[-k:]
        m = pp.sum(1) > 0
        q[f'r_lag_{k}'] = np.nanmean(frame_corr(dd[m], pp[m]))

    # Bands per quadrant.
    bar = P > 0
    rows = []
    bands = [('all', 0., np.inf)] + BANDS + [(f'ring_{a:.2f}', a, b) for a, b in zip(RINGS[:-1], RINGS[1:])]
    for qn, (sx, sy) in QUADRANTS.items():
        qm = inside & (np.sign(grid[:, 0]) == sx) & (np.sign(grid[:, 1]) == sy)
        for bn, lo, hi in bands:
            m = qm & (ecc >= lo) & (ecc < hi)
            if not m.any():
                continue
            on = bar[:, m]
            rows.append(dict(meta, quadrant=qn, is_hp=qn == hp, band=bn, n_pixels=int(m.sum()),
                             n_bar_pixel_frames=int(on.sum()),
                             drive_bar=float(D[:, m][on].mean()) if on.any() else np.nan,
                             mean_all=float(D[:, m].mean())))

    # Rotated maps (HP upper right).
    ri = rotation_index(grid, hp)
    cnt = bar.sum(0)
    with np.errstate(invalid='ignore', divide='ignore'):
        drive = np.where(cnt > 0, (D * bar).sum(0) / cnt, np.nan)
    maps = np.stack([drive[ri], D.mean(0)[ri]]).reshape(2, 50, 50)
    return meta, q, rows, maps, (D.reshape(T, 50, 50) if T == N_TR else None), P.reshape(T, 50, 50)


def main(bids, source, out_dir, jobs, rois=None, decoders=('free', 'outline')):
    root = Path(bids) / 'derivatives' / 'decoded_final_2026-11' / source
    files = sorted(root.glob('*/sub-*/decoded_ses-*_run-*_roi-*.npz'))
    found = {f.stem.split('roi-')[1] for f in files}
    ROIS = [r for r in ALL_ROIS if r in found and (rois is None or r in rois)]
    DECODERS = [d for d in decoders if any(f.parent.parent.name == d for f in files)]
    files = [f for f in files if f.stem.split('roi-')[1] in ROIS and f.parent.parent.name in DECODERS]
    print('ROIs', ROIS, 'decoders', DECODERS, flush=True)
    print(f'{len(files)} decoded runs', flush=True)
    out_dir = Path(out_dir)
    subjects = sorted({int(f.parent.name.split('-')[1]) for f in files})
    S = {s: i for i, s in enumerate(subjects)}
    shape = (len(subjects), len(ROIS), len(DECODERS))
    map_sum = np.zeros(shape + (2, 50, 50))
    map_n = np.zeros(shape + (2, 50, 50))
    frame_sum = {}          # (subject, roi, dec) -> sum over runs of decoded frames
    frame_n = {}
    examples = {}
    paradigm_ref = None
    quality, bands = [], []
    with Pool(jobs) as pool:
        for k, (meta, q, rows, maps, frames, par) in enumerate(pool.imap_unordered(one_run, [(f, bids, source) for f in files], chunksize=4)):
            quality.append(q)
            bands.extend(rows)
            ix = (S[meta['subject']], ROIS.index(meta['roi']), DECODERS.index(meta['decoder']))
            ok = np.isfinite(maps)
            map_sum[ix] += np.where(ok, maps, 0)
            map_n[ix] += ok
            if frames is not None:
                key = (meta['subject'], meta['roi'], meta['decoder'])
                frame_sum[key] = frame_sum.get(key, 0) + frames
                frame_n[key] = frame_n.get(key, 0) + 1
                if paradigm_ref is None and meta['subject'] >= 3:
                    paradigm_ref = par
                if meta['subject'] in EXAMPLES and meta['decoder'] == 'free':
                    examples[f"sub-{meta['subject']:02d}_ses-{meta['session']}_run-{meta['run']}_roi-{meta['roi']}"] = \
                        frames.astype(np.float16)
            if k % 200 == 0:
                print(k, flush=True)
    q = pd.DataFrame(quality).sort_values(['subject', 'roi', 'decoder', 'session', 'run'])
    q.to_csv(out_dir / f'decoding_quality_{source}_2026-11.tsv', sep='\t', index=False)
    b = pd.DataFrame(bands)
    b.to_csv(out_dir / f'decoding_bands_{source}_2026-11.tsv', sep='\t', index=False)
    with np.errstate(invalid='ignore'):
        mp = map_sum / map_n
    np.savez_compressed(out_dir / f'decoding_maps_{source}_2026-11.npz', drive=mp[:, :, :, 0].astype(np.float32),
                        mean=mp[:, :, :, 1].astype(np.float32), n_runs=map_n[:, :, :, 0].max((-1, -2)),
                        subjects=np.array(subjects), rois=np.array(ROIS), decoders=np.array(DECODERS),
                        grid=np.load(files[0])['grid'])
    group = np.full((len(ROIS), len(DECODERS), N_TR, 50, 50), np.nan, np.float32)
    for i, roi in enumerate(ROIS):
        for j, dec in enumerate(DECODERS):
            per_sub = [frame_sum[k] / frame_n[k] for k in frame_sum if k[1] == roi and k[2] == dec]
            group[i, j] = np.mean(per_sub, 0)
    ex_mean = {f'sub-{s:02d}_roi-{roi}_mean': (frame_sum[(s, roi, 'free')] / frame_n[(s, roi, 'free')]).astype(np.float16)
               for s in EXAMPLES for roi in ROIS if (s, roi, 'free') in frame_sum}
    np.savez_compressed(out_dir / f'decoding_frames_{source}_2026-11.npz', paradigm=paradigm_ref.astype(np.uint8),
                        group=group.astype(np.float16), rois=np.array(ROIS), decoders=np.array(DECODERS), **ex_mean)
    # Single runs of the example subjects (~90 MB): kept on the cluster, trimmed to the shown run before syncing.
    np.savez_compressed(out_dir / f'decoding_example_runs_{source}_2026-11.npz', **examples)
    print(q.groupby(['roi', 'decoder'])[['r', 'r_shuffle']].mean().round(3))
    print('Wrote', out_dir)


def trim_examples(source, out_dir, keep):
    """Keep only the shown runs (``keep``: 'sub-23_ses-2_run-3' ...) of the example-runs file, all ROIs."""
    src = Path(out_dir) / f'decoding_example_runs_{source}_2026-11.npz'
    with np.load(src) as z:
        sel = {k: z[k] for k in z.files if any(k.startswith(p + '_roi-') for p in keep)}
    out = Path(out_dir) / f'decoding_example_run_{source}_2026-11.npz'
    np.savez_compressed(out, **sel)
    print('Kept', sorted(sel), '->', out)


def extract_runs(bids, source, out_dir, keys, name):
    """Decoded frames (free decoder) of the given runs ('sub-18_ses-2_run-5_roi-V1' ...) into one small NPZ."""
    root = Path(bids) / 'derivatives' / 'decoded_final_2026-11' / source / 'free'
    out = {}
    for k in keys:
        sub, ses, run, roi = k.split('_')
        with np.load(root / sub / f'decoded_{ses}_{run}_{roi}.npz') as z:
            out[k] = z['decoded']
    dst = Path(out_dir) / f'decoding_{name}_{source}_2026-11.npz'
    np.savez_compressed(dst, **out)
    print('Wrote', sorted(out), '->', dst)


def extract_subject_means(bids, source, out_dir, keys, frames, decoder='free'):
    """Per 'sub-XX_roi-ROI': the decoded frames averaged over the subject's runs, at the given frame indices."""
    root = Path(bids) / 'derivatives' / 'decoded_final_2026-11' / source / decoder
    out = {}
    for k in keys:
        sub, roi = k.split('_')
        runs = [np.load(f)['decoded'].astype(np.float32)[frames] for f in sorted((root / sub).glob(f'decoded_ses-*_run-*_{roi}.npz'))]
        out[k] = np.mean(runs, 0)
        out[k + '_nruns'] = np.array(len(runs))
    dst = Path(out_dir) / f'decoding_example_subjects_{source}_2026-11.npz'
    np.savez_compressed(dst, frames=np.array(frames), **out)
    print('Wrote', dst, len(keys), 'subject x ROI')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bids-folder', default='/shares/zne.uzh/gdehol/ds-retsupp')
    ap.add_argument('--source', default='canonical')
    ap.add_argument('--out-dir', default='notes/data')
    ap.add_argument('-j', '--jobs', type=int, default=4)
    ap.add_argument('--rois', nargs='+', default=None, help='default: every ROI found under the source')
    ap.add_argument('--decoders', nargs='+', default=['free', 'outline'], help='decoder folders to read (others are ignored)')
    ap.add_argument('--trim', nargs='+', default=None, help="only trim the example runs, e.g. 'sub-23_ses-2_run-3'")
    ap.add_argument('--extract', nargs='+', default=None, help="only extract decoded runs, e.g. 'sub-18_ses-2_run-5_roi-V1'")
    ap.add_argument('--extract-name', default='example_candidates')
    ap.add_argument('--subject-means', nargs='+', default=None, help="'sub-XX_roi-ROI' keys: run-mean frames")
    ap.add_argument('--frames', nargs='+', type=int, default=[15, 21, 28, 72, 81, 89, 130])
    a = ap.parse_args()
    if a.subject_means:
        extract_subject_means(a.bids_folder, a.source, a.out_dir, a.subject_means, a.frames)
    elif a.extract:
        extract_runs(a.bids_folder, a.source, a.out_dir, a.extract, a.extract_name)
    elif a.trim:
        trim_examples(a.source, a.out_dir, a.trim)
    else:
        main(a.bids_folder, a.source, a.out_dir, a.jobs, a.rois, a.decoders)
