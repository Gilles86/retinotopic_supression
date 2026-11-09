"""Gain-field models with one width for the sustained and the transient fields.

The full model (m7_full) has two widths: σ_AF for the sustained field (HP and LP gains) and σ_dyn for
the transient fields of distractor and target (σ_T_dyn is already tied to σ_dyn). Here σ_dyn and
σ_T_dyn are tied to σ_AF in the forward transform, so the model has one width and one shared parameter
fewer than its parent::

    σ_dyn := σ_T_dyn := σ_AF

The gains and their equations are unchanged. The model is nested in m7_full (equal to it wherever the
two widths of m7 are equal), so its training SSE cannot be below m7's. Fitted from m7_full by refitting
σ_AF and all gains (``fit_cv_ladder_nested --models m7_full_sigtied``); the question is whether held-out
prediction gets worse without the second width.
"""
from __future__ import annotations

import tensorflow as tf

from retsupp.modeling.dn_attention_field import DNDynamicAttentionFieldPRF2D
from retsupp.modeling.local_models import (
    DoGDynamicAttentionFieldPRF2DWithHRF_v3_target_sharedSigma_factorial_items as _Items,
)

SIGMA_AF, SIGMA_DYN, SIGMA_T_DYN = 7, 10, 14


def _tie(out):
    """Copy column σ_AF into σ_dyn and σ_T_dyn (same softplus in both slots, so it holds raw or usable)."""
    s = out[:, SIGMA_AF:SIGMA_AF + 1]
    return tf.concat([out[:, :SIGMA_DYN], s, out[:, SIGMA_DYN + 1:SIGMA_T_DYN], s,
                      out[:, SIGMA_T_DYN + 1:]], axis=1)


class SigmaTiedDoG(_Items):
    """DoG gain-field model (items, fixed HRF) with σ_dyn = σ_T_dyn = σ_AF."""

    @tf.function
    def _transform_parameters_forward(self, parameters):
        return _tie(super()._transform_parameters_forward(parameters))

    @tf.function
    def _transform_parameters_backward(self, parameters):
        return _tie(super()._transform_parameters_backward(parameters))


class SigmaTiedDN(DNDynamicAttentionFieldPRF2D):
    """DN gain-field model with σ_dyn = σ_T_dyn = σ_AF."""

    @tf.function
    def _transform_parameters_forward(self, parameters):
        return _tie(super()._transform_parameters_forward(parameters))

    @tf.function
    def _transform_parameters_backward(self, parameters):
        return _tie(super()._transform_parameters_backward(parameters))
