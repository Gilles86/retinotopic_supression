"""Group-mean decoded frames per ROI with some subjects left out (cluster).

The same reduction as the ``group`` array of ``fig6_collect`` (per subject × ROI the decoded frames averaged over that
subject's full-length runs, then averaged over subjects), for one decoder, written as
``decoding_group_frames_<source>_<decoder>_2026-11.npz`` (group: ROI × 258 × 50 × 50, float16; rois; subjects;
n_subjects per ROI).

    python -m retsupp.visualize.paper.group_decoded_frames --source jointhrf --exclude-subjects 1 2 --out-dir <dir>
"""
from __future__ import annotations

import argparse
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from retsupp.visualize.paper.fig6_collect import ALL_ROIS, N_TR


def subject_roi_mean(files):
    s, n = 0., 0
    for f in files:
        with np.load(f) as z:
            d = z['decoded'].astype(np.float32)
        if d.shape[0] == N_TR:
            s, n = s + d, n + 1
    return (s / n) if n else None


def main(bids, source, decoder, out_dir, exclude, jobs):
    root = Path(bids) / 'derivatives' / 'decoded_final_2026-11' / source / decoder
    tasks, keys = [], []
    for d in sorted(root.glob('sub-*')):
        sub = int(d.name.split('-')[1])
        if sub in exclude:
            continue
        for roi in ALL_ROIS:
            fs = sorted(d.glob(f'decoded_ses-*_run-*_roi-{roi}.npz'))
            if fs:
                tasks.append(fs)
                keys.append((sub, roi))
    print(len(tasks), 'subject x ROI', flush=True)
    sums = {r: 0. for r in ALL_ROIS}
    cnt = {r: 0 for r in ALL_ROIS}
    with Pool(jobs) as pool:
        for (sub, roi), m in zip(keys, pool.imap(subject_roi_mean, tasks)):
            if m is not None:
                sums[roi] = sums[roi] + m
                cnt[roi] += 1
    rois = [r for r in ALL_ROIS if cnt[r]]
    group = np.stack([sums[r] / cnt[r] for r in rois]).astype(np.float16)
    out = Path(out_dir) / f'decoding_group_frames_{source}_{decoder}_2026-11.npz'
    np.savez_compressed(out, group=group, rois=np.array(rois), n_subjects=np.array([cnt[r] for r in rois]),
                        subjects=np.array(sorted({k[0] for k in keys})))
    print('Wrote', out, dict(zip(rois, [cnt[r] for r in rois])), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bids-folder', default='/shares/zne.uzh/gdehol/ds-retsupp')
    ap.add_argument('--source', default='jointhrf')
    ap.add_argument('--decoder', default='free')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--exclude-subjects', nargs='*', type=int, default=[])
    ap.add_argument('-j', '--jobs', type=int, default=4)
    a = ap.parse_args()
    main(a.bids_folder, a.source, a.decoder, a.out_dir, set(a.exclude_subjects), a.jobs)
