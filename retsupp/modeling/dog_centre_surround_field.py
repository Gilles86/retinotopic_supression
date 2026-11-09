"""DoG gain-field model in which the sustained field can weigh the pRF centre and surround differently.

Parent: the DoG gain-field model of the ladder (m7_full, FAMILY=dog,
``DoGDynamicAttentionFieldPRF2DWithHRF_v3_target_sharedSigma_factorial_items``). There the field multiplies the
stimulus before pooling, so the centre and the surround Gaussian see the same modulated stimulus::

    F_c(g)    = Σ_ℓ w_cℓ A_ℓ(g; σ_AF),   w_cℓ = g_HP if ℓ is HP of condition c, else g_LP
    M_c(g)    = max(1 + F_c(g), 0)                     sustained field, per HP condition c
    P(t, g)   = transient distractor + target fields
    r_v(t)    = Σ_g S_β(t, g) · [M_c(t)(g) + P(t, g)] · [G_c,v(g) − G_s,v(g)] + baseline_v
    BOLD(t)   = HRF * r_v(t)

with S_β the stimulus with the item pixels scaled by β, G_c the centre Gaussian (peak amplitude_v) and G_s the
surround Gaussian (width srf_size·sd, peak srf_amplitude·amplitude). Here one shared tilt κ splits the sustained
field between the two poolings::

    M^cen_c(g) = max(1 + (1 + κ) F_c(g), 0),    M^sur_c(g) = max(1 + (1 − κ) F_c(g), 0)
    r_v(t)     = Σ_g S_β [M^cen_c(t) + P] G_c,v  −  Σ_g S_β [M^sur_c(t) + P] G_s,v  + baseline_v

g_HP, g_LP (in F) are the common gain, κ the tilt: κ = 0 is m7 exactly, κ = +1 the field on the centre (driving)
pooling only (weight 2g), κ = −1 on the surround (suppressive) pooling only; |κ| > 1 gives the other pooling a field
of opposite sign. The transient terms P act on both poolings, as in m7. One field per subject × ROI, shared over
voxels: σ_AF, g_HP, g_LP, κ are shared like every other gain-field term.

``kappa_fixed`` clamps κ (e.g. +1 / −1 for the centre-only / surround-only special cases). ``kappa`` is appended as
slot 16 (identity transform); slots 0-15 keep the parent's meaning.
"""
from __future__ import annotations

import tensorflow as tf

from retsupp.modeling.local_models import (
    DoGDynamicAttentionFieldPRF2DWithHRF_v3_target_sharedSigma_factorial_items as _Items,
)

NEW = 'kappa'
_N = len(_Items.parameter_labels)   # 16
KAPPA = _N


class DoGCentreSurroundField(_Items):
    parameter_labels = _Items.parameter_labels + [NEW]

    def __init__(self, *, kappa_fixed=None, **kwargs):
        super().__init__(**kwargs)
        self._kappa_fixed = kappa_fixed

    def field_sum(self, parameters):
        """F_c(g) = sign · Σ_ℓ w_cℓ A_ℓ(g), shape (C, G); shared terms read from voxel 0."""
        sigma_af = parameters[0, 0, 7]
        g_hp, g_lp = parameters[0, 0, 8], parameters[0, 0, 9]
        is_hp = self._tf_is_hp                                                 # (C, L)
        w = is_hp * g_hp + (1.0 - is_hp) * g_lp
        gx = self._grid_coordinates[:, 0][tf.newaxis, :]
        gy = self._grid_coordinates[:, 1][tf.newaxis, :]
        rx = self._tf_ring_positions[:, 0][:, tf.newaxis]
        ry = self._tf_ring_positions[:, 1][:, tf.newaxis]
        a = tf.exp(-((gx - rx) ** 2 + (gy - ry) ** 2) / (2.0 * sigma_af ** 2))  # (L, G)
        return self._tf_sign * tf.einsum('cl,lg->cg', w, a)

    def centre_and_surround(self, parameters):
        """Centre Gaussian G_c and surround G_s (B, V, G), G_c − G_s = the parent's DoG exactly."""
        p = parameters[:, :, :15]
        p_c = tf.concat([p[..., :5], tf.zeros_like(p[..., 5:6]), p[..., 6:]], axis=-1)
        rf_c = self._get_rf(self.grid_coordinates, p_c)          # srf_amplitude 0: centre only
        rf_dog = self._get_rf(self.grid_coordinates, p)
        return rf_c, rf_c - rf_dog

    @tf.function
    def _basis_predictions(self, paradigm, parameters):
        beta = parameters[0, 0, 15]
        s = paradigm * (1.0 + (beta - 1.0) * self._tf_item_mask)[tf.newaxis, tf.newaxis, :]
        kappa = parameters[0, 0, KAPPA]
        f = self.field_sum(parameters)                                         # (C, G)
        m_cen = tf.maximum(1.0 + (1.0 + kappa) * f, 0.0)
        m_sur = tf.maximum(1.0 + (1.0 - kappa) * f, 0.0)
        ci = self._tf_condition_indicator                                      # (T, C)
        phasic = self._tf_sign * (self._attention_modulation_dynamic_v3(parameters[:, :, :15])
                                  + self._attention_modulation_target(parameters[:, :, :15]))   # (T, G)
        field_cen = tf.einsum('tc,cg->tg', ci, m_cen) + phasic
        field_sur = tf.einsum('tc,cg->tg', ci, m_sur) + phasic
        rf_c, rf_s = self.centre_and_surround(parameters)
        result = (tf.einsum('btg,bvg->btv', s * field_cen[tf.newaxis], rf_c)
                  - tf.einsum('btg,bvg->btv', s * field_sur[tf.newaxis], rf_s))
        return result + parameters[:, tf.newaxis, :, 3]

    def _kappa_col(self, parameters):
        k = parameters[:, _N:_N + 1]
        if self._kappa_fixed is None:
            return k
        return tf.ones_like(k) * float(self._kappa_fixed)

    @tf.function
    def _transform_parameters_forward(self, parameters):
        return tf.concat([super()._transform_parameters_forward(parameters[:, :_N]),
                          self._kappa_col(parameters)], axis=1)

    @tf.function
    def _transform_parameters_backward(self, parameters):
        return tf.concat([super()._transform_parameters_backward(parameters[:, :_N]),
                          self._kappa_col(parameters)], axis=1)
