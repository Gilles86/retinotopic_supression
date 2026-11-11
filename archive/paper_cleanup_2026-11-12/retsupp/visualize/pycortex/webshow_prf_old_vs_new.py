"""Webshow PRF parameters for one subject — both OLD m4 (bar paradigm)
and NEW m4 (full paradigm) side-by-side, IF NEW giis exist locally.

Opens a pycortex webshow server in the browser. Hover/click to inspect
voxels on the 3D inflated cortex.

Run from the pycortex2 env:
    ~/mambaforge/envs/pycortex2/bin/python \
        retsupp/visualize/pycortex/webshow_prf_old_vs_new.py 2
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cortex
import numpy as np
from retsupp.utils.data import Subject
from retsupp.visualize.utils import get_alpha_vertex


def build_dataset(subject: int, model: int, bids_folder: Path,
                  r2_thr: float, label_prefix: str):
    pc_subject = f'retsupp.sub-{subject:02d}'
    sub = Subject(subject, bids_folder=bids_folder)
    pars = sub.get_prf_parameters_surface(model=model)
    mask = (pars['r2'] > r2_thr).values

    pars['theta'] = np.arctan2(pars['y'], pars['x'])
    pars['ecc']   = np.hypot(pars['x'], pars['y'])

    return {
        f'{label_prefix}_R2':    get_alpha_vertex(
            pars['r2'].values, mask, cmap='hot', vmin=0.0, vmax=0.5,
            subject=pc_subject),
        f'{label_prefix}_polar': get_alpha_vertex(
            pars['theta'].values, mask, cmap='hsv',
            vmin=-np.pi, vmax=np.pi, subject=pc_subject),
        f'{label_prefix}_ecc':   get_alpha_vertex(
            pars['ecc'].values, mask, cmap='nipy_spectral',
            vmin=0.0, vmax=4.0, subject=pc_subject),
        f'{label_prefix}_sd':    get_alpha_vertex(
            pars['sd'].values, mask, cmap='nipy_spectral',
            vmin=0.0, vmax=4.0, subject=pc_subject),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('subject', type=int)
    p.add_argument('--model', type=int, default=4)
    p.add_argument('--old-bids', default='/data/ds-retsupp')
    p.add_argument('--new-bids', default='/data/ds-retsupp.new',
                   help='Where the NEW (full-paradigm) m4 fsnative '
                        'giis live; default /data/ds-retsupp.new. If '
                        'this dir does not have the giis we just show '
                        'OLD.')
    p.add_argument('--r2-thr', type=float, default=0.05)
    p.add_argument('--port', type=int, default=8080)
    args = p.parse_args()

    ds = {}
    ds.update(build_dataset(args.subject, args.model,
                             Path(args.old_bids), args.r2_thr, 'OLD'))
    new_root = Path(args.new_bids)
    has_new = (new_root / 'derivatives' / 'prf' / f'model{args.model}'
               / f'sub-{args.subject:02d}' /
               f'sub-{args.subject:02d}_desc-x.optim.nilearn_space-fsnative_hemi-L.func.gii').exists()
    if has_new:
        print(f'Found NEW giis under {new_root} — adding NEW_* layers')
        ds.update(build_dataset(args.subject, args.model,
                                 new_root, args.r2_thr, 'NEW'))
    else:
        print(f'NEW giis not found at {new_root}; showing OLD only.')
        print('  Once cluster surf-sample finishes, mv local '
              'derivatives/prf/model4 -> model4.OLD_BAR and rsync the '
              'NEW giis from cluster (then re-run with --new-bids).')

    print(f'\n  Open http://localhost:{args.port}/  in your browser '
          f'(Ctrl-C to close)')
    cortex.webshow(cortex.Dataset(**ds), port=args.port)


if __name__ == '__main__':
    main()
