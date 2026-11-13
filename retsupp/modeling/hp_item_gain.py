"""DoG gain-field models in which the HP location changes the transient item gains multiplicatively or additively.

Parent: the ladder's m5_TDS_dS (sustained HP and LP gains, one transient distractor gain g_D = g_LP_dyn at every
location, one transient target gain g_T; ``DoGDynamicAttentionFieldPRF2DWithHRF_v3_target_sharedSigma_factorial_items``
with g_HP_dyn tied to g_LP_dyn). There the transient field is::

    P(t, g) = g_D · Σ_ℓ d_ℓ(t) A_ℓ(g) + g_T · Σ_ℓ tgt_ℓ(t) A_ℓ(g),   A_ℓ = Gaussian at ring position ℓ, width σ_dyn

with d_ℓ, tgt_ℓ the distractor and target on-fractions. These models add one shared term that acts on both item
gains at the HP location only (hp_ℓ(t) = 1 at the run's HP position)::

    mult ('hp_scale' = h):   at HP, g_D → g_D (1 + h)   and   g_T → g_T (1 + h)
    add  ('hp_offset' = a):  at HP, g_D → g_D + a       and   g_T → g_T + a

so P gains  c_D · Σ_ℓ d_ℓ hp_ℓ A_ℓ + c_T · Σ_ℓ tgt_ℓ hp_ℓ A_ℓ  with (c_D, c_T) = (h g_D, h g_T) or (a, a). The DoG
response is linear in the field, so the term adds to the parent's prediction; h = 0 / a = 0 is the parent exactly.
Multiplicative: the HP change is proportional to the gain the item gets (a large target gain changes more than a
small distractor gain); additive: the same change for both items. Both have the parent's terms plus one.

The new term is slot 16 (identity transform); slots 0-15 keep the parent's meaning.
"""
from __future__ import annotations

import tensorflow as tf

from retsupp.modeling.local_models import (
    DoGDynamicAttentionFieldPRF2DWithHRF_v3_target_sharedSigma_factorial_items as _Items,
)

MODES = {'mult': 'hp_scale', 'add': 'hp_offset'}
_N = len(_Items.parameter_labels)   # 16


def _items_at_hp_fields(model, parameters):
    """(distractor at HP, target at HP) fields, each (T, G); width σ_dyn (slot 10; the target width is tied to it)."""
    sigma = parameters[0, 0, 10]
    gx = model._grid_coordinates[:, 0][tf.newaxis, :]
    gy = model._grid_coordinates[:, 1][tf.newaxis, :]
    rx = model._tf_ring_positions[:, 0][:, tf.newaxis]
    ry = model._tf_ring_positions[:, 1][:, tf.newaxis]
    a = tf.exp(-((gx - rx) ** 2 + (gy - ry) ** 2) / (2.0 * sigma ** 2))     # (L, G)
    hp = model._tf_condition_indicator                                       # (T, L)
    f_d = tf.einsum('tl,lg->tg', model._tf_dynamic_indicator * hp, a)
    f_t = tf.einsum('tl,lg->tg', model._tf_target_indicator * hp, a)
    return f_d, f_t


class HPItemGainDoG(_Items):
    """m5-type DoG gain-field model + one HP term on both transient item gains; use the subclasses below."""
    _mode = None

    @tf.function
    def _basis_predictions(self, paradigm, parameters):
        base = super()._basis_predictions(paradigm, parameters[:, :, :_N])
        beta = parameters[0, 0, 15]
        paradigm_eff = paradigm * (1.0 + (beta - 1.0) * self._tf_item_mask)[tf.newaxis, tf.newaxis, :]
        k = parameters[0, 0, _N]
        if self._mode == 'mult':
            c_d = k * parameters[0, 0, 11]   # g_HP_dyn (= g_LP_dyn when tied, as in m5)
            c_t = k * parameters[0, 0, 13]   # g_T_dyn
        else:
            c_d = k
            c_t = k
        f_d, f_t = _items_at_hp_fields(self, parameters)
        field = c_d * f_d + c_t * f_t
        rf = self._get_rf(self.grid_coordinates, parameters[:, :, :_N])
        return base + self._tf_sign * tf.einsum('btg,bvg->btv', paradigm_eff * field[tf.newaxis], rf)

    @tf.function
    def _transform_parameters_forward(self, parameters):
        return tf.concat([super()._transform_parameters_forward(parameters[:, :_N]), parameters[:, _N:]], axis=1)

    @tf.function
    def _transform_parameters_backward(self, parameters):
        return tf.concat([super()._transform_parameters_backward(parameters[:, :_N]), parameters[:, _N:]], axis=1)


class HPItemMultDoG(HPItemGainDoG):
    """At HP both transient item gains are scaled by (1 + hp_scale)."""
    _mode = 'mult'
    parameter_labels = _Items.parameter_labels + [MODES['mult']]


class HPItemAddDoG(HPItemGainDoG):
    """At HP both transient item gains get + hp_offset."""
    _mode = 'add'
    parameter_labels = _Items.parameter_labels + [MODES['add']]


CLASSES = {'mult': HPItemMultDoG, 'add': HPItemAddDoG}
