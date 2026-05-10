"""Verify the numerical mechanism of phantom R²=1 voxels.

Hypothesis: phantom voxels have parameters at boundary (sd→0, srf_size→0,
amplitude=-46.11, baseline=-2.63). The DoG prediction collapses to a
constant. R² = 1 - SSres/SStot. If the model fit perfectly drove
predictions to equal the data, R² = 1.

But the parameters are 'pinned' constants across all phantoms -- they're
NOT being fit per-voxel. So predictions are the same constant across
all phantom voxels. For R²=1 to hold we'd need the data to also be that
constant.

ALTERNATIVE: the cleaned BOLD has some voxels with very low SStot.
If SSres is tiny too, the ratio can underflow → R² ≈ 1.

Let's directly compute R² for a phantom voxel against the DoG prediction
using its actual fitted parameters and the concatenated paradigm.
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from retsupp.utils.data import Subject  # noqa: E402

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')  # CPU only, plenty

from braincoder.hrf import SPMHRFModel
from braincoder.models import DifferenceOfGaussiansPRF2DWithHRF
from braincoder.utils.stats import get_rsq

BIDS = '/data/ds-retsupp'
sub = Subject(2, bids_folder=BIDS)
masker = sub.get_bold_mask(return_masker=True)
masker.fit()

# Load m4 params.
M4_PARS = ['x', 'y', 'sd', 'amplitude', 'baseline',
           'srf_amplitude', 'srf_size', 'hrf_delay', 'hrf_dispersion']
base = sub.bids_folder / 'derivatives' / 'prf' / 'model4' / 'sub-02'
prf = {}
for par in M4_PARS + ['r2']:
    fn = base / f'sub-02_desc-{par}.nii.gz'
    prf[par] = masker.transform(str(fn)).flatten().astype(np.float32)

phantom_idx = np.where(prf['r2'] >= 0.999)[0]
signal_idx = np.where(prf['r2'] < 0.999)[0]
print(f'phantom n = {len(phantom_idx)}, signal n = {len(signal_idx)}')

# Inspect parameters across phantoms: are they really all identical?
print('\nPer-parameter uniqueness across the 22127 phantoms:')
for par in M4_PARS + ['r2']:
    a = prf[par][phantom_idx]
    print(f'  {par:16s}: mean={a.mean():.6g}  std={a.std():.6g}  '
          f'min={a.min():.6g}  max={a.max():.6g}  '
          f'unique-count={len(np.unique(a))}')

# Pick a handful of phantom voxels (random)
rng = np.random.default_rng(7)
pick = rng.choice(phantom_idx, size=5, replace=False)

# Build cleaned BOLD + paradigm exactly as fit_prf.py does.
from retsupp.modeling.fit_prf import load_concatenated
print('\nLoading concatenated cleaned BOLD + paradigm (this may take a min)...')
data, paradigm, grid_coords = load_concatenated(sub, masker, 50, 'full')
print(f'  data shape: {data.shape}')

# Subset to phantom picks + some signal voxels.
extra_signal = rng.choice(signal_idx, size=5, replace=False)
all_idx = np.concatenate([pick, extra_signal])
data_sub = data[:, all_idx]
print(f'  using {len(all_idx)} voxels')

# Build the model.
hrf = SPMHRFModel(tr=1.6, delay=4.5, dispersion=0.75)
model = DifferenceOfGaussiansPRF2DWithHRF(
    grid_coordinates=grid_coords, paradigm=paradigm, hrf_model=hrf,
    data=data_sub, flexible_hrf_parameters=True,
)

# Pull each voxel's parameter row.
rows = []
for vi in all_idx:
    rows.append({p: float(prf[p][vi]) for p in M4_PARS})
pars_df = pd.DataFrame(rows).astype(np.float32)

# Predict.
predictions = model.predict(paradigm=paradigm, parameters=pars_df)
preds = predictions.values  # (T, V)
print(f'\nPrediction stats:')
print(f'  shape: {preds.shape}')
for k, vi in enumerate(all_idx):
    p = preds[:, k]
    d = data_sub[:, k]
    sstot = ((d - d.mean()) ** 2).sum()
    ssres = ((d - p) ** 2).sum()
    r2 = 1 - ssres / max(sstot, 1e-30)
    print(f'  vox {vi:7d}  pred[range={p.max()-p.min():.6g}]  '
          f'data[std={d.std():.6g}]  '
          f'sstot={sstot:.4g}  ssres={ssres:.4g}  R²={r2:.6f}  '
          f'(stored R²={prf["r2"][vi]:.4f})')

# Compute also via the official function.
data_df = pd.DataFrame(data_sub)
r2_official = get_rsq(data_df, predictions)
print(f'\nOfficial braincoder get_rsq:')
for k, vi in enumerate(all_idx):
    print(f'  vox {vi:7d}: stored {prf["r2"][vi]:.6f}  '
          f'recomputed {r2_official.iloc[k]:.6f}')
