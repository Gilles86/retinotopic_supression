"""Run a figure script with some subjects left out (default: the pilots sub-01 and sub-02).

Per-subject estimates do not depend on other subjects, so leaving subjects out only changes what the figure scripts pool.
This runner patches the two readers the scripts use:

* ``pandas.read_csv``: rows whose ``subject`` column (int, ``'sub-01'``, ``'s01'``) is excluded are dropped.
* ``numpy.load``: in an ``.npz`` with a ``subjects`` array, every array whose first axis has that length is subset; keys
  named ``sub-XX_...`` of an excluded subject are dropped.

Group-level products that were computed over all subjects (LOSO δ, the Fig. 7 hierarchical model, group-mean decoded
images, the hierarchy regression) cannot be filtered; their n = 28 versions are recomputed and put in ``--redirect``
(default ``notes/data/n28``) under the same file name. Any read of ``notes/data/<path>`` is served from
``<redirect>/<path>`` when that file exists. Every file read is logged to ``--log`` (subject column or not, subjects
kept), so group-level inputs that were not recomputed can be spotted.

    python -m retsupp.visualize.paper.run_excluding --exclude 1 2 -- retsupp.visualize.paper.fig3_item_locations --name x
"""
from __future__ import annotations

import argparse
import re
import runpy
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXCL: set[int] = set()
REDIRECT = Path('notes/data/n28')
LOG: list[dict] = []
_read_csv = pd.read_csv
_np_load = np.load


def subject_number(v):
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return int(v) if np.isfinite(v) else None
    m = re.search(r'(\d+)', str(v))
    return int(m.group(1)) if m else None


def redirect(path):
    p = Path(path)
    try:
        rel = p.resolve().relative_to(Path('notes/data').resolve())
    except (ValueError, OSError):
        return path
    alt = REDIRECT / rel
    return str(alt) if alt.exists() else path


def read_csv(path, *a, **k):
    src = redirect(path) if isinstance(path, (str, Path)) else path
    d = _read_csv(src, *a, **k)
    entry = dict(file=str(path), served=str(src), kind='csv', subject_col='subject' in getattr(d, 'columns', []))
    if entry['subject_col'] and EXCL:
        keep = ~d['subject'].map(subject_number).isin(EXCL)
        d = d[keep]
        entry['subjects'] = d['subject'].nunique()
    LOG.append(entry)
    return d


class Filtered(dict):
    """An npz as a dict with the excluded subjects removed; usable as a context manager like NpzFile."""

    @property
    def files(self):
        return list(self.keys())

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def close(self):
        pass


def np_load(path, *a, **k):
    src = redirect(path) if isinstance(path, (str, Path)) else path
    z = _np_load(src, *a, **k)
    entry = dict(file=str(path), served=str(src), kind='npy')
    if not hasattr(z, 'files'):
        LOG.append(entry)
        return z
    out = Filtered()
    keep = None
    if 'subjects' in z.files and EXCL:
        subs = np.asarray(z['subjects'])
        keep = ~np.isin([subject_number(s) for s in subs], list(EXCL))
        entry['subjects'] = int(keep.sum())
    for key in z.files:
        m = re.match(r'sub-(\d+)_', key)
        if m and int(m.group(1)) in EXCL:
            continue
        v = z[key]
        if keep is not None and getattr(v, 'ndim', 0) >= 1 and v.shape[0] == len(keep):
            v = v[keep]
        out[key] = v
    entry['subject_keys_dropped'] = sum(1 for key in z.files if re.match(r'sub-(\d+)_', key)
                                        and int(re.match(r'sub-(\d+)_', key).group(1)) in EXCL)
    LOG.append(entry)
    return out


def install(exclude, redirect_dir=None):
    global EXCL, REDIRECT
    EXCL = set(int(s) for s in exclude)
    if redirect_dir:
        REDIRECT = Path(redirect_dir)
    pd.read_csv = read_csv
    np.load = np_load


def main():
    argv = sys.argv[1:]
    split = argv.index('--')
    ap = argparse.ArgumentParser()
    ap.add_argument('--exclude', nargs='+', type=int, default=[1, 2])
    ap.add_argument('--redirect', default='notes/data/n28')
    ap.add_argument('--log', default=None, help='TSV of every file read')
    a = ap.parse_args(argv[:split])
    module, *rest = argv[split + 1:]
    install(a.exclude, a.redirect)
    sys.argv = [module] + rest
    try:
        runpy.run_module(module, run_name='__main__', alter_sys=True)
    finally:
        if a.log:
            Path(a.log).parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(LOG).to_csv(a.log, sep='\t', index=False)


if __name__ == '__main__':
    main()
