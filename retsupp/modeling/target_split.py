"""Gain-field models with the target gain split by target location: at HP vs elsewhere.

Phase D2 of the paper plan. One parameter is appended, ``g_T_HP_dyn``, the transient gain when the
target appears at the run's HP location; ``g_T_dyn`` keeps the gain for targets at the other three
locations. The target field becomes::

    g_T_dyn · Σ_ℓ tgt_ℓ(t)·(1 − hp_ℓ(t))·A_ℓ  +  g_T_HP_dyn · Σ_ℓ tgt_ℓ(t)·hp_ℓ(t)·A_ℓ

with hp_ℓ(t) the condition indicator (1 at the HP ring position) and A_ℓ the target Gaussian
(σ tied to σ_dyn). At g_T_HP_dyn = g_T_dyn the prediction equals the parent's, so a fit started from
m7 cannot end below it. dT = g_T_HP_dyn − g_T_dyn.
"""
from __future__ import annotations

import tensorflow as tf

from retsupp.modeling.dn_attention_field import DNDynamicAttentionFieldPRF2D
from retsupp.modeling.local_models import (
    DoGDynamicAttentionFieldPRF2DWithHRF_v3_target_sharedSigma_factorial_items as _Items,
)

NEW = 'g_T_HP_dyn'


def _target_at_hp_field(model, parameters):
    """Σ_ℓ tgt_ℓ(t)·hp_ℓ(t)·A_ℓ(g), shape (T, G); σ from slot 14 (tied to σ_dyn)."""
    sigma = parameters[0, 0, 14]
    gx = model._grid_coordinates[:, 0][tf.newaxis, :]
    gy = model._grid_coordinates[:, 1][tf.newaxis, :]
    rx = model._tf_ring_positions[:, 0][:, tf.newaxis]
    ry = model._tf_ring_positions[:, 1][:, tf.newaxis]
    a = tf.exp(-((gx - rx) ** 2 + (gy - ry) ** 2) / (2.0 * sigma ** 2))
    return tf.einsum('tl,lg->tg', model._tf_target_indicator * model._tf_condition_indicator, a)


class TargetSplitDoG(_Items):
    """DoG gain-field model (items, fixed HRF) + ``g_T_HP_dyn`` in slot 16."""
    parameter_labels = _Items.parameter_labels + [NEW]
    _N = len(_Items.parameter_labels)

    @tf.function
    def _basis_predictions(self, paradigm, parameters):
        base = super()._basis_predictions(paradigm, parameters[:, :, :self._N])
        # The DoG response is linear in the field, so the extra target-at-HP term adds.
        beta = parameters[0, 0, 15]
        paradigm_eff = paradigm * (1.0 + (beta - 1.0) * self._tf_item_mask)[tf.newaxis, tf.newaxis, :]
        delta = parameters[0, 0, self._N] - parameters[0, 0, 13]
        field = delta * _target_at_hp_field(self, parameters)
        rf = self._get_rf(self.grid_coordinates, parameters[:, :, :self._N])
        return base + self._tf_sign * tf.einsum('btg,bvg->btv', paradigm_eff * field[tf.newaxis], rf)

    @tf.function
    def _transform_parameters_forward(self, parameters):
        return tf.concat([super()._transform_parameters_forward(parameters[:, :self._N]),
                          parameters[:, self._N:]], axis=1)

    @tf.function
    def _transform_parameters_backward(self, parameters):
        return tf.concat([super()._transform_parameters_backward(parameters[:, :self._N]),
                          parameters[:, self._N:]], axis=1)


class TargetSplitDN(DNDynamicAttentionFieldPRF2D):
    """DN gain-field model + ``g_T_HP_dyn`` in slot 18."""
    parameter_labels = DNDynamicAttentionFieldPRF2D.parameter_labels + [NEW]
    _N = len(DNDynamicAttentionFieldPRF2D.parameter_labels)

    @tf.function
    def _attention_modulation_target(self, parameters):
        # The DN response is not linear in the field, so the split goes into the field itself.
        delta = parameters[0, 0, self._N] - parameters[0, 0, 13]
        return (super()._attention_modulation_target(parameters)
                + delta * _target_at_hp_field(self, parameters))

    @tf.function
    def _transform_parameters_forward(self, parameters):
        return tf.concat([super()._transform_parameters_forward(parameters[:, :self._N]),
                          parameters[:, self._N:]], axis=1)

    @tf.function
    def _transform_parameters_backward(self, parameters):
        return tf.concat([super()._transform_parameters_backward(parameters[:, :self._N]),
                          parameters[:, self._N:]], axis=1)
