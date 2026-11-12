"""Concatenate the per-run quadrant readouts of ``simulate_decode.py`` into one table (compute node, not login).

Reads ``<bids>/derivatives/decoded_sim_2026-11/<source>/noise-<noise>/gain-*/<decoder>/sub-XX/readout_*.tsv``
(columns include ``gain_amplitude``, ``decoder``, ``drive_bar`` and ``true_drive_bar``) and writes one TSV. Refuses
to overwrite an existing output.

Usage::

    srun --account=hare.econ.uzh -c1 --mem 4G --time 10 \\
        $HOME/data/conda/envs/retsupp_cuda/bin/python -m retsupp.decode.collect_simulate_decode \\
        --out notes/data/decode_sim_2026-11.tsv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main(bids, source, noise, out):
    out = Path(out)
    if out.exists():
        raise FileExistsError(f'{out} exists; not overwriting')
    root = Path(bids) / 'derivatives' / 'decoded_sim_2026-11' / source / f'noise-{noise}'
    files = sorted(root.glob('gain-*/*/sub-*/readout_*.tsv'))
    d = pd.concat([pd.read_csv(f, sep='\t') for f in files], ignore_index=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(out, sep='\t', index=False)
    print(f'{len(files)} readouts, {len(d)} rows -> {out}')
    print(d.groupby(['gain_amplitude', 'decoder']).subject.nunique().rename('n_subjects').to_string())


def collect_fitted(bids, source, noise, out_dir, prefix='decode_simfit', jobs=4):
    """``simulate_decode.py --mode fitted`` outputs (``derivatives/decoded_sim_fitted_2026-11``), reduced like the data of
    Fig. 6, per variant (``model``, ``control``): quadrant readouts (``<prefix>_readout_2026-11.tsv``), eccentricity
    bands and decoding quality of ``fig6_collect.one_run`` for the ``free`` decoder (``<prefix>_bands_2026-11.tsv.gz``,
    ``<prefix>_quality_2026-11.tsv``) and the bar-sweep condition differences of ``figS_decoded_bars.one_subject_roi``
    for the ``outline`` decoder (``decoded_bars_condition_<prefix>_<variant>_outline_2026-11.npz``)."""
    from multiprocessing import Pool

    import numpy as np

    from retsupp.visualize.paper import figS_decoded_bars as fb
    from retsupp.visualize.paper.fig6_collect import one_run
    out_dir = Path(out_dir)
    outs = [out_dir / f'{prefix}_{k}' for k in ('readout_2026-11.tsv', 'bands_2026-11.tsv.gz', 'quality_2026-11.tsv')]
    for o in outs:
        if o.exists():
            raise FileExistsError(f'{o} exists; not overwriting')
    root = Path(bids) / 'derivatives' / 'decoded_sim_fitted_2026-11' / source / f'noise-{noise}'
    files = sorted(root.glob('*/*/sub-*/readout_*.tsv'))
    d = pd.concat([pd.read_csv(f, sep='\t') for f in files], ignore_index=True)
    d.to_csv(outs[0], sep='\t', index=False)
    print(f'{len(files)} readouts -> {outs[0]}')
    print(d.groupby(['variant', 'decoder', 'roi']).subject.nunique().rename('n_subjects').to_string(), flush=True)

    npz = sorted(root.glob('*/free/sub-*/decoded_*.npz'))
    bands, quality = [], []
    with Pool(jobs) as pool:
        for f, (meta, q, rows, *_rest) in zip(npz, pool.imap(one_run, [(f, bids, source) for f in npz], chunksize=4)):
            v = f.parent.parent.parent.name
            quality.append(dict(q, variant=v))
            bands.extend(dict(r, variant=v) for r in rows)
    pd.DataFrame(bands).to_csv(outs[1], sep='\t', index=False)
    pd.DataFrame(quality).to_csv(outs[2], sep='\t', index=False)
    print(f'{len(npz)} decoded runs (free) -> {outs[1]}, {outs[2]}', flush=True)

    for v in ('model', 'control'):
        out = out_dir / f'decoded_bars_condition_{prefix}_{v}_outline_2026-11.npz'
        if out.exists():
            print(f'{out} exists; skipped')
            continue
        tasks = []
        for sd in sorted((root / v / 'outline').glob('sub-*')):
            for roi in fb.ROIS:
                fs = sorted(sd.glob(f'decoded_ses-*_run-*_roi-{roi}.npz'))
                if fs:
                    tasks.append(fs)
        acc, cnt = {}, {}
        with Pool(jobs) as pool:
            for sub, s_, n, nhp in pool.imap_unordered(fb.one_subject_roi, tasks):
                acc[sub] = acc.get(sub, 0) + s_
                cnt[sub] = cnt.get(sub, 0) + n
        subs = sorted(acc)
        with np.errstate(invalid='ignore'):
            mean = np.array([acc[s] / cnt[s][:, :, None, None] for s in subs]).astype(np.float32)
        np.savez_compressed(out, mean=mean.reshape(len(subs), 2, fb.N_BINS, 3, 50, 50), n=np.array([cnt[s] for s in subs]),
                            subjects=np.array(subs), edges=fb.EDGES, grid=np.load(tasks[0][0])['grid'])
        print(f'{v}: {len(tasks)} subject x ROI -> {out}', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--bids-folder', default='/shares/zne.uzh/gdehol/ds-retsupp')
    p.add_argument('--source', default='canonical')
    p.add_argument('--noise', default='phase')
    p.add_argument('--out', default='notes/data/decode_sim_2026-11.tsv')
    p.add_argument('--fitted', action='store_true', help='collect simulate_decode.py --mode fitted (see collect_fitted)')
    p.add_argument('--out-dir', default='notes/data')
    p.add_argument('--prefix', default='decode_simfit')
    p.add_argument('-j', '--jobs', type=int, default=4)
    a = p.parse_args()
    if a.fitted:
        collect_fitted(a.bids_folder, a.source, a.noise, a.out_dir, a.prefix, a.jobs)
        raise SystemExit
    main(a.bids_folder, a.source, a.noise, a.out)
