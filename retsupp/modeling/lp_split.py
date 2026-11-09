"""Gain-field models with the sustained LP gain split into the locations next to HP and opposite it.

In the decoded bar (and the measured bar responses) the quadrant at HP is lowest, the two quadrants
next to it are highest and the opposite one lies in between. With one sustained gain at HP and one
shared by the three LP locations, the full model can only produce HP < next <= opposite (through the
overlap of the Gaussian fields). Here the LP gain varies with the ring distance to HP.

One parameter is appended, ``g_adj_minus_opp`` (δ). ``g_LP`` keeps its meaning as the mean sustained
gain over the three LP locations, so dS = g_HP − g_LP is the same contrast as in m7_full::

    g_adj = g_LP + δ/3      (the two ring positions next to HP, tied)
    g_opp = g_LP − 2δ/3     (the ring position opposite HP)

so that (2·g_adj + g_opp)/3 = g_LP and g_adj − g_opp = δ. The terms map onto the decoding contrasts:
next − opposite = δ, HP − next = dS − δ/3, HP − opposite = dS + 2δ/3. The sustained field for the run
whose HP is ring position c is::

    M_c(g) = max(1 + Σ_ℓ w_cℓ · A_ℓ(g), 0),   w_cℓ = g_HP, g_adj, g_opp for ring distance 0, 1, 2

with A_ℓ the Gaussian of width σ_AF at ring position ℓ. The ring is in angular order (UR, UL, LL, LR,
``get_ring_positions``), so the neighbours of c are c ± 1 and the opposite position c + 2 (mod 4). At
δ = 0 the prediction equals the parent's, so a fit started from m7_full cannot end below it.
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf

from retsupp.modeling.dn_attention_field import DNDynamicAttentionFieldPRF2D
from retsupp.modeling.local_models import (
    DoGDynamicAttentionFieldPRF2DWithHRF_v3_target_sharedSigma_factorial_items as _Items,
)

NEW = 'g_adj_minus_opp'


def ring_distance(n=4):
    """(C, L) ring steps between condition c's HP (ring position c) and ring position ℓ: 0, 1 or 2."""
    d = (np.arange(n)[None, :] - np.arange(n)[:, None]) % n
    return np.minimum(d, n - d)


def _sustained_field(model, parameters, delta):
    """Per-condition sustained field max(1 + Σ_ℓ w_cℓ A_ℓ, 0), shape (C, G); gains from voxel 0."""
    sigma_af = parameters[0, 0, 7]
    g_hp, g_lp = parameters[0, 0, 8], parameters[0, 0, 9]
    dist = model._tf_ring_distance                                            # (C, L)
    w = (tf.cast(tf.equal(dist, 0.), tf.float32) * g_hp
         + tf.cast(tf.equal(dist, 1.), tf.float32) * (g_lp + delta / 3.0)
         + tf.cast(tf.equal(dist, 2.), tf.float32) * (g_lp - 2.0 * delta / 3.0))
    gx = model._grid_coordinates[:, 0][tf.newaxis, :]
    gy = model._grid_coordinates[:, 1][tf.newaxis, :]
    rx = model._tf_ring_positions[:, 0][:, tf.newaxis]
    ry = model._tf_ring_positions[:, 1][:, tf.newaxis]
    a = tf.exp(-((gx - rx) ** 2 + (gy - ry) ** 2) / (2.0 * sigma_af ** 2))   # (L, G)
    return tf.maximum(1.0 + model._tf_sign * tf.einsum('cl,lg->cg', w, a), 0.0)


def _init_ring_distance(model):
    if model.n_conditions != 4:
        raise ValueError(f'LP split needs the 4-position ring, got {model.n_conditions}')
    model._tf_ring_distance = tf.constant(ring_distance(4), dtype=tf.float32)


class LPSplitDoG(_Items):
    """DoG gain-field model (items, fixed HRF) + ``g_adj_minus_opp`` in slot 16."""
    parameter_labels = _Items.parameter_labels + [NEW]
    _N = len(_Items.parameter_labels)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        _init_ring_distance(self)

    @tf.function
    def _basis_predictions(self, paradigm, parameters):
        # Same response as the parent (sustained field per HP condition, phasic distractor and target
        # fields, all on the item-scaled stimulus), with the sustained field built from three gains.
        beta = parameters[0, 0, 15]
        paradigm_eff = paradigm * (1.0 + (beta - 1.0) * self._tf_item_mask)[tf.newaxis, tf.newaxis, :]
        sustained = _sustained_field(self, parameters, parameters[0, 0, self._N])          # (C, G)
        field = tf.einsum('tc,cg->tg', self._tf_condition_indicator, sustained)
        phasic = (self._attention_modulation_dynamic_v3(parameters[:, :, :15])
                  + self._attention_modulation_target(parameters[:, :, :15]))           # (T, G)
        rf = self._get_rf(self.grid_coordinates, parameters[:, :, :15])
        result = tf.einsum('btg,bvg->btv',
                           paradigm_eff * (field + self._tf_sign * phasic)[tf.newaxis], rf)
        return result + parameters[:, tf.newaxis, :, 3]

    @tf.function
    def _transform_parameters_forward(self, parameters):
        return tf.concat([super()._transform_parameters_forward(parameters[:, :self._N]),
                          parameters[:, self._N:]], axis=1)

    @tf.function
    def _transform_parameters_backward(self, parameters):
        return tf.concat([super()._transform_parameters_backward(parameters[:, :self._N]),
                          parameters[:, self._N:]], axis=1)


class LPSplitDN(DNDynamicAttentionFieldPRF2D):
    """DN gain-field model + ``g_adj_minus_opp`` in slot 18."""
    parameter_labels = DNDynamicAttentionFieldPRF2D.parameter_labels + [NEW]
    _N = len(DNDynamicAttentionFieldPRF2D.parameter_labels)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        _init_ring_distance(self)

    @tf.function
    def _attention_modulation(self, parameters):
        # The DN class reads the sustained field as (B, V, C, G) from the first voxel.
        field = _sustained_field(self, parameters, parameters[0, 0, self._N])
        shape = tf.shape(parameters)
        return tf.broadcast_to(field[tf.newaxis, tf.newaxis],
                               tf.stack([shape[0], shape[1], tf.shape(field)[0], tf.shape(field)[1]]))

    @tf.function
    def _transform_parameters_forward(self, parameters):
        return tf.concat([super()._transform_parameters_forward(parameters[:, :self._N]),
                          parameters[:, self._N:]], axis=1)

    @tf.function
    def _transform_parameters_backward(self, parameters):
        return tf.concat([super()._transform_parameters_backward(parameters[:, :self._N]),
                          parameters[:, self._N:]], axis=1)
