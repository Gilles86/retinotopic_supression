"""Gaussian pRF under the gain fields: the DoG gain-field classes with the surround amplitude fixed at 0.

``gaussian(cls)`` returns a subclass of a DoG gain-field class (``..._factorial``, ``..._factorial_items``,
``TargetSplitDoG``, ...) whose forward transform sets ``srf_amplitude`` to 0, so the surround term of the DoG
(``srf_amplitude · amplitude · srf_size`` times a Gaussian) vanishes and the pRF is the centre Gaussian, with the
same (legacy) normalisation as braincoder's ``GaussianPRF2D``. ``srf_amplitude`` and ``srf_size`` stay in the
parameter vector (so tables and pickles keep the DoG columns) but get no gradient. Everything else (gain fields,
item scale, HRF) is the DoG class's own code.

Natural-space parameters passed to ``predict`` must carry ``srf_amplitude = 0``; ``gauss_init`` does that.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

_CACHE: dict = {}


def gaussian(cls):
    """Subclass of the DoG gain-field class ``cls`` with the surround amplitude fixed at 0."""
    if cls in _CACHE:
        return _CACHE[cls]

    class Gauss(cls):
        @tf.function
        def _transform_parameters_forward(self, parameters):
            out = super()._transform_parameters_forward(parameters)
            return tf.concat([out[:, :5], tf.zeros_like(out[:, 5:6]), out[:, 6:]], axis=1)

        @tf.function
        def _transform_parameters_backward(self, parameters):
            # srf_amplitude is 0 in natural space; any positive value keeps the raw value finite.
            p = tf.concat([parameters[:, :5], tf.ones_like(parameters[:, 5:6]), parameters[:, 6:]], axis=1)
            return super()._transform_parameters_backward(p)

    Gauss.__name__ = Gauss.__qualname__ = f'Gauss_{cls.__name__}'
    _CACHE[cls] = Gauss
    return Gauss


def gauss_init(bids_folder, subject, prf_pars, folder='prf_cv_roi_jointhrf', stem='gauss_final2_stage-3', srf_size=2.0):
    """Per-voxel template for the Gaussian ladder: x, y, sd from the Gaussian fold fits with the joint per-ROI HRF
    (``cv/fit_prf_cv_roi_hrf_joint.py --family gauss --tag final2``, stage 3, mean over the folds; the HRF of the
    ladder's table comes from the same fits), the m4 values where a voxel has no fold fit; srf_amplitude 0.

    ``prf_pars`` is the whole-brain m4 table (rows = BOLD-mask voxels, the ladder's voxel index); returns a copy with
    x, y, sd, srf_amplitude and srf_size replaced, and the number of voxels taken from the fold fits.
    """
    d = Path(bids_folder) / 'derivatives' / folder / f'sub-{subject:02d}'
    folds = []
    for f in sorted(d.glob(f'sub-{subject:02d}_fold-*_{stem}_pars.pkl.gz')):
        ids = pd.read_csv(str(f).replace('_pars.pkl.gz', '_cvr2.tsv.gz'), sep='\t').voxel_index.to_numpy()
        p = pd.read_pickle(f).reset_index(drop=True)[['x', 'y', 'sd']]
        p.index = ids
        folds.append(p)
    if not folds:
        raise FileNotFoundError(f'no {stem} fold fits in {d}')
    mean = pd.concat(folds).groupby(level=0).mean().dropna()
    out = prf_pars.copy()
    ix = mean.index.to_numpy()
    for c in ('x', 'y', 'sd'):
        out.loc[ix, c] = mean[c].to_numpy()
    out['srf_amplitude'] = 0.0
    out['srf_size'] = srf_size
    return out, len(folds), ix
