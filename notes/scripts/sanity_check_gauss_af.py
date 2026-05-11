"""Sanity check: instantiate Gaussian AF sharedSigma model with mock data.

Verifies:
- Model imports and instantiates.
- Forward pass returns finite predictions of the right shape.
- sharedSigma constraint sets sigma_T_dyn := sigma_dyn (slot 12 := slot 8).
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from braincoder.hrf import SPMHRFModel
from retsupp.modeling.local_models import (
    GaussianDynamicAttentionFieldPRF2DWithHRF_v3_target_sharedSigma as M,
)

T, G_side = 5, 10
G = G_side * G_side
n_C = 4
gx, gy = np.meshgrid(np.linspace(-5, 5, G_side),
                     np.linspace(-5, 5, G_side))
grid_coordinates = np.stack(
    [gx.ravel(), gy.ravel()], axis=1).astype(np.float32)
paradigm = np.random.rand(T, G).astype(np.float32)
condition_indicator = np.zeros((T, n_C), dtype=np.float32)
condition_indicator[:, 0] = 1.0
dynamic_indicator = np.zeros((T, n_C), dtype=np.float32)
dynamic_indicator[2, 0] = 1.0
target_indicator = np.zeros((T, n_C), dtype=np.float32)
target_indicator[3, 1] = 1.0
ring_positions = np.array([[2.83, 2.83], [-2.83, 2.83],
                            [-2.83, -2.83], [2.83, -2.83]],
                           dtype=np.float32)

hrf = SPMHRFModel(tr=1.6, delay=4.5, dispersion=0.75)
model = M(
    grid_coordinates=grid_coordinates,
    paradigm=paradigm,
    hrf_model=hrf,
    condition_indicator=condition_indicator,
    dynamic_indicator=dynamic_indicator,
    target_indicator=target_indicator,
    ring_positions=ring_positions,
    mode='signed',
)
print('Model built OK')
print('Parameter labels:', model.parameter_labels)

init = pd.DataFrame([{
    'x': 1.0, 'y': 0.5, 'sd': 1.0, 'baseline': 0.0, 'amplitude': 1.0,
    'sigma_AF': 2.0, 'g_HP': 0.0, 'g_LP': 0.0,
    'sigma_dyn': 2.0, 'g_HP_dyn': 0.0, 'g_LP_dyn': 0.0,
    'g_T_dyn': 0.0, 'sigma_T_dyn': 2.0,
}])
pred = model.predict(parameters=init)
print('Prediction shape:', pred.shape)
print('Any NaN?', pred.isna().any().any())
print('Sample (zero gains):', pred.iloc[:, 0].values)

# Test sharedSigma constraint applied: forward transform should
# overwrite slot 12 (sigma_T_dyn) with slot 8 (sigma_dyn).
raw_init = init.values.astype(np.float32)
post = model._transform_parameters_forward(raw_init).numpy()
print('Post-forward sigma_dyn (slot 8):', post[0, 8])
print('Post-forward sigma_T_dyn (slot 12):', post[0, 12])
assert np.isclose(post[0, 8], post[0, 12]), (
    "sharedSigma constraint failed: sigma_T_dyn != sigma_dyn")
print('sharedSigma constraint OK: slot 12 == slot 8')

# Set sigma_T_dyn raw to crazy value to confirm it gets overridden.
init_bad = init.copy()
init_bad['sigma_T_dyn'] = 9.99
raw_bad = init_bad.values.astype(np.float32)
post_bad = model._transform_parameters_forward(raw_bad).numpy()
print('Bad init sigma_T_dyn=9.99, post-forward slot 12:',
      post_bad[0, 12], 'slot 8:', post_bad[0, 8])
assert np.isclose(post_bad[0, 8], post_bad[0, 12]), (
    "sharedSigma constraint failed when sigma_T_dyn != sigma_dyn in raw")
print('All sanity checks PASS')
