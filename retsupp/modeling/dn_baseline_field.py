"""Gain-field models in which the sustained field acts on the DN baselines, not (only) on the stimulus.

Parent: the DN gain-field model of ``dn_attention_field`` (m7_full, FAMILY=dn), in which the field
multiplies the stimulus and so enters numerator and denominator alike::

    S_eff(t, g) = S(t, g) · β(g) · max(M_c(g) + P(t, g), 0),   M_c(g) = max(1 + Σ_ℓ w_cℓ A_ℓ(g), 0)
    N(t) = (a <S_eff, G_c> + b) / (c|a| <S_eff, G_s> + d) − b / d

Here a second sustained field, with the same width σ_AF and its own HP/LP pattern, scales the voxel's
neural baseline b and surround (normalization) baseline d, per HP condition c::

    Φ_c(g)  = Σ_ℓ u_cℓ A_ℓ(g; σ_AF),   u_cℓ = cos θ (ℓ = HP of c), sin θ (ℓ ≠ HP of c)
    F_c(v)  = Σ_g Ĝ_v(g) Φ_c(g),         Ĝ_v = centre pRF Gaussian normalised to sum 1 on the grid
    b_c(v)  = b_v · exp(w_b F_c(v)),     d_c(v) = d_v · exp(w_d F_c(v))
    N(t)    = (a <S_eff, G_c> + b_c) / (c|a| <S_eff, G_s> + d_c) − b_c / d_c

F_c(v) is the field averaged over the voxel's pRF. The shape (σ_AF, θ) is shared by both baselines; the
two weights w_b and w_d are separate. θ sets the HP/LP pattern (0: only at HP, π/2: only at the three LP
locations, π/4: the same at all four, −π/4: HP against LP); (w, θ) and (−w, θ + π) are the same model.
exp(w F) ≈ 1 + w F for small effects and keeps both baselines positive. The condition-specific b_c/d_c is
subtracted, so a blank screen still gives N = 0 and the field only changes stimulus-evoked responses (a
per-run constant would be absorbed by the BOLD baseline anyway).

``targets`` picks the version: ('b',) neural-baseline field (w_d := 0), ('d',) surround-baseline field
(w_b := 0), ('b', 'd') both. At w_b = w_d = 0 the prediction equals the parent's exactly, whatever θ.
Slots 0-17 keep the parent's meaning; w_b, w_d, theta_bd are appended (identity transform).
"""
from __future__ import annotations

import tensorflow as tf

from retsupp.modeling.dn_attention_field import (
    NEURAL_BASELINE, SRF_SIZE, SURROUND_BASELINE, DNDynamicAttentionFieldPRF2D,
)

NEW = ['w_b', 'w_d', 'theta_bd']
_N = len(DNDynamicAttentionFieldPRF2D.parameter_labels)
W_B, W_D, THETA = _N, _N + 1, _N + 2


class DNBaselineField(DNDynamicAttentionFieldPRF2D):
    parameter_labels = DNDynamicAttentionFieldPRF2D.parameter_labels + NEW

    def __init__(self, *, targets=('b', 'd'), **kwargs):
        bad = set(targets) - {'b', 'd'}
        if bad or not targets:
            raise ValueError(f"targets must be a non-empty subset of ('b', 'd'), got {targets}")
        super().__init__(**kwargs)
        self._use = tf.constant([float('b' in targets), float('d' in targets), 1.0], dtype=tf.float32)

    def pool_field(self, parameters):
        """F_c(v) per HP condition, shape (B, V, C): Φ_c averaged over each voxel's centre pRF."""
        sigma_af, theta = parameters[0, 0, 7], parameters[0, 0, THETA]
        is_hp = self._tf_is_hp                                                 # (C, L)
        u = tf.cos(theta) * is_hp + tf.sin(theta) * (1.0 - is_hp)
        gx = self._grid_coordinates[:, 0][tf.newaxis, :]
        gy = self._grid_coordinates[:, 1][tf.newaxis, :]
        rx = self._tf_ring_positions[:, 0][:, tf.newaxis]
        ry = self._tf_ring_positions[:, 1][:, tf.newaxis]
        a = tf.exp(-((gx - rx) ** 2 + (gy - ry) ** 2) / (2.0 * sigma_af ** 2))  # (L, G)
        phi = tf.einsum('cl,lg->cg', u, a)
        rf = self._gauss(parameters, parameters[..., 2:3])                       # (B, V, G)
        rf = rf / tf.maximum(tf.reduce_sum(rf, axis=-1, keepdims=True), 1e-12)
        return tf.einsum('bvg,cg->bvc', rf, phi)

    @tf.function
    def _basis_predictions(self, paradigm, parameters):
        # Stimulus side exactly as in the parent (sustained input field + phasic fields).
        beta = parameters[0, 0, 15]
        pixel_scale = 1.0 + (beta - 1.0) * self._tf_item_mask
        sustained = self._attention_modulation(parameters[:, :1])[:, 0]
        field = tf.einsum('tc,bcg->btg', self._tf_condition_indicator, sustained)
        phasic = (self._attention_modulation_dynamic_v3(parameters)
                  + self._attention_modulation_target(parameters))
        field = tf.maximum(field + self._tf_sign * phasic[tf.newaxis], 0.0)
        s_eff = paradigm * pixel_scale[tf.newaxis, tf.newaxis, :] * field

        sd = parameters[..., 2:3]
        rf = self._gauss(parameters, sd)
        srf = self._gauss(parameters, sd * parameters[..., SRF_SIZE:SRF_SIZE + 1])
        a = parameters[:, :, 4][:, tf.newaxis, :]
        c = parameters[:, :, 5][:, tf.newaxis, :]
        b = parameters[:, :, NEURAL_BASELINE][:, tf.newaxis, :]
        d = parameters[:, :, SURROUND_BASELINE][:, tf.newaxis, :]
        # Baseline field per TR: F(t, v) = Σ_c ci[t, c] F_c(v).
        f_t = tf.einsum('tc,bvc->btv', self._tf_condition_indicator, self.pool_field(parameters))
        b = b * tf.exp(parameters[0, 0, W_B] * f_t)
        d = d * tf.exp(parameters[0, 0, W_D] * f_t)
        conv_c = tf.einsum('btg,bvg->btv', s_eff, rf)
        conv_s = tf.einsum('btg,bvg->btv', s_eff, srf)
        return (a * conv_c + b) / (c * tf.abs(a) * conv_s + d) - b / d

    @tf.function
    def _transform_parameters_forward(self, parameters):
        return tf.concat([super()._transform_parameters_forward(parameters[:, :_N]),
                          parameters[:, _N:] * tf.cast(self._use, parameters.dtype)[tf.newaxis, :]], axis=1)

    @tf.function
    def _transform_parameters_backward(self, parameters):
        return tf.concat([super()._transform_parameters_backward(parameters[:, :_N]),
                          parameters[:, _N:] * tf.cast(self._use, parameters.dtype)[tf.newaxis, :]], axis=1)
