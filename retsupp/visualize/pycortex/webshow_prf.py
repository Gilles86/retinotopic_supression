"""Quick PRF webshow script for IPython.

Usage in IPython (pycortex2 env):

    %run retsupp/visualize/pycortex/webshow_prf.py

Then:

    ds = build(subject=2)        # OLD m4 only (currently on disk)
    cortex.webshow(ds)

Once NEW m4 surface giis land locally (e.g. at
``/data/ds-retsupp.NEW/derivatives/prf/model4/sub-XX/...``):

    ds = build(subject=2, also_new='/data/ds-retsupp.NEW')
    cortex.webshow(ds)

The dataset will then have OLD_* and NEW_* layers visible in the
dropdown.
"""
from __future__ import annotations

from pathlib import Path

import cortex
import numpy as np

from retsupp.utils.data import Subject
from retsupp.visualize.utils import get_alpha_vertex


def _layers(subject: int, model: int, bids: Path,
             r2_thr: float, prefix: str) -> dict:
    pc_subject = f'retsupp.sub-{subject:02d}'
    sub = Subject(subject, bids_folder=bids)
    pars = sub.get_prf_parameters_surface(model=model)
    mask = (pars['r2'] > r2_thr).values

    pars['theta'] = np.arctan2(pars['y'], pars['x'])
    pars['ecc']   = np.hypot(pars['x'], pars['y'])

    return {
        f'{prefix}_R2':    get_alpha_vertex(pars['r2'].values,    mask,
            cmap='hot', vmin=0.0, vmax=0.5, subject=pc_subject),
        f'{prefix}_polar': get_alpha_vertex(pars['theta'].values, mask,
            cmap='hsv', vmin=-np.pi, vmax=np.pi, subject=pc_subject),
        f'{prefix}_ecc':   get_alpha_vertex(pars['ecc'].values,   mask,
            cmap='nipy_spectral', vmin=0.0, vmax=4.0, subject=pc_subject),
        f'{prefix}_sd':    get_alpha_vertex(pars['sd'].values,    mask,
            cmap='nipy_spectral', vmin=0.0, vmax=4.0, subject=pc_subject),
    }


def build(subject: int = 2, model: int = 4, r2_thr: float = 0.05,
          old_bids: str = '/data/ds-retsupp',
          also_new: str | None = None) -> cortex.Dataset:
    """Build a pycortex Dataset with OLD layers (and optional NEW)."""
    ds = {}
    ds.update(_layers(subject, model, Path(old_bids), r2_thr, 'OLD'))
    if also_new is not None:
        new = Path(also_new)
        gii = (new / 'derivatives' / 'prf' / f'model{model}'
               / f'sub-{subject:02d}'
               / f'sub-{subject:02d}_desc-x.optim.nilearn_space-fsnative_hemi-L.func.gii')
        if not gii.exists():
            print(f'[WARN] NEW giis not found at {gii} — skipping')
        else:
            ds.update(_layers(subject, model, new, r2_thr, 'NEW'))
    return cortex.Dataset(**ds)


# When run as a script, build and open webshow on the default subject.
if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('subject', nargs='?', type=int, default=2)
    p.add_argument('--model', type=int, default=4)
    p.add_argument('--r2-thr', type=float, default=0.05)
    p.add_argument('--also-new', default=None,
                   help='Path to a SECOND BIDS root containing NEW '
                        'm4 fsnative giis (added as NEW_* layers).')
    args = p.parse_args()
    ds = build(subject=args.subject, model=args.model,
               r2_thr=args.r2_thr, also_new=args.also_new)
    print('  Layers:', list(ds.views.keys()))
    print('  Opening webshow — Ctrl-C to stop the server.')
    cortex.webshow(ds)
