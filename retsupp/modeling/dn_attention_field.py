"""Gain fields on a divisive-normalization pRF (with a fixed HRF).

Same shared parameters and gain-field equations as
``DoGDynamicAttentionFieldPRF2DWithHRF_v3_target_sharedSigma_factorial_items``,
but the voxel's response is the DN model of Aqil et al. (2021)::

    S_eff(t, g) = S(t, g) · β(g) · max(M(t, g), 0)
    N(t) = (a <S_eff, G_c> + b) / (|a| c <S_eff, G_s> + d) − b / d
    BOLD(t) = HRF * N(t) + bold_baseline

M(t, g) is the sum of the sustained field (1 + g·A, per HP condition) and the
phasic fields of distractor and target. The attention field multiplies the
stimulus, so it enters both the activation and the normalization pool, as in
Reynolds & Heeger (2009). With a DoG pRF it multiplies centre and surround too.
Unlike a DoG, the denominator needs S_eff >= 0 to stay positive, hence the floor.

Slots 0-15 keep the parent's meaning and order, with ``amplitude`` = a,
``baseline`` = bold_baseline, ``srf_amplitude`` = c and ``srf_size`` = σ_s/σ_c
(>= 1, as in braincoder's DN model). ``neural_baseline`` = b and
``surround_baseline`` = d are appended.
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from braincoder.models import GaussianPRF2D
from retsupp.modeling.local_models import (
    DoGDynamicAttentionFieldPRF2DWithHRF_v3_target_sharedSigma_factorial_items as _Items,
)

N_PARENT = 16
SRF_SIZE, NEURAL_BASELINE, SURROUND_BASELINE = 6, 16, 17
SRF_SIZE_EPS = 1e-4


def _replace_col(x, ix, col):
    return tf.concat([x[:, :ix], col, x[:, ix + 1:]], axis=1)


class DNDynamicAttentionFieldPRF2D(_Items):
    parameter_labels = _Items.parameter_labels + ['neural_baseline', 'surround_baseline']

    def _gauss(self, parameters, sd):
        pars = tf.concat([parameters[..., 0:2], sd, tf.zeros_like(sd), tf.ones_like(sd)],
                         axis=-1)
        return GaussianPRF2D._get_rf(self, self.grid_coordinates, pars)

    @tf.function
    def _basis_predictions(self, paradigm, parameters):
        # paradigm (B, T, G), parameters (B, V, P). The gains are shared, so the
        # fields are computed once, from the first voxel.
        beta = parameters[0, 0, 15]
        pixel_scale = 1.0 + (beta - 1.0) * self._tf_item_mask              # (G,)
        sustained = self._attention_modulation(parameters[:, :1])[:, 0]    # (B, C, G)
        field = tf.einsum('tc,bcg->btg', self._tf_condition_indicator, sustained)
        phasic = (self._attention_modulation_dynamic_v3(parameters)
                  + self._attention_modulation_target(parameters))         # (T, G)
        field = tf.maximum(field + self._tf_sign * phasic[tf.newaxis], 0.0)
        s_eff = paradigm * pixel_scale[tf.newaxis, tf.newaxis, :] * field

        sd = parameters[..., 2:3]
        rf = self._gauss(parameters, sd)
        srf = self._gauss(parameters, sd * parameters[..., SRF_SIZE:SRF_SIZE + 1])
        a = parameters[:, :, 4][:, tf.newaxis, :]
        c = parameters[:, :, 5][:, tf.newaxis, :]
        b = parameters[:, :, NEURAL_BASELINE][:, tf.newaxis, :]
        d = parameters[:, :, SURROUND_BASELINE][:, tf.newaxis, :]
        conv_c = tf.einsum('btg,bvg->btv', s_eff, rf)
        conv_s = tf.einsum('btg,bvg->btv', s_eff, srf)
        return (a * conv_c + b) / (c * tf.abs(a) * conv_s + d) - b / d

    @tf.function
    def _predict(self, paradigm, parameters, weights):
        convolved = self.hrf_model.convolve(self._basis_predictions(paradigm, parameters))
        return convolved + parameters[:, :, 3][:, tf.newaxis, :]

    @tf.function
    def _transform_parameters_forward(self, parameters):
        enc = super()._transform_parameters_forward(parameters[:, :N_PARENT])
        srf_size = 1.0 + tf.math.softplus(parameters[:, SRF_SIZE:SRF_SIZE + 1])
        return tf.concat([_replace_col(enc, SRF_SIZE, srf_size),
                          tf.math.softplus(parameters[:, N_PARENT:])], axis=1)

    @tf.function
    def _transform_parameters_backward(self, parameters):
        safe = _replace_col(parameters[:, :N_PARENT], SRF_SIZE,
                            tf.ones_like(parameters[:, SRF_SIZE:SRF_SIZE + 1]))
        enc = super()._transform_parameters_backward(safe)
        srf_raw = tfp.math.softplus_inverse(
            tf.maximum(parameters[:, SRF_SIZE:SRF_SIZE + 1] - 1.0, SRF_SIZE_EPS))
        return tf.concat([_replace_col(enc, SRF_SIZE, srf_raw),
                          tfp.math.softplus_inverse(parameters[:, N_PARENT:])], axis=1)


def init_from_m5(sub, voxel_mask):
    """Whole-brain m5 fit of the masked voxels, in this class's column names."""
    m5 = sub.get_prf_parameters_volume(model=5, return_images=False)
    m5 = m5.rename(columns={'rf_amplitude': 'amplitude', 'bold_baseline': 'baseline'})
    out = m5.loc[voxel_mask].reset_index(drop=True)
    cols = ['x', 'y', 'sd', 'baseline', 'amplitude', 'srf_amplitude', 'srf_size',
            'neural_baseline', 'surround_baseline']
    bad = ~np.isfinite(out[cols].to_numpy()).all(axis=1)
    if bad.any():
        raise ValueError(f'm5 has non-finite parameters in {bad.sum()} of {len(out)} voxels')
    return out
