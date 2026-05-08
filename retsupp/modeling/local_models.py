"""Project-local braincoder model variants for the retsupp project.

These are project-specific extensions of braincoder models. They live
here (rather than in ``libs/braincoder``) because they are tied to the
retsupp data layout and analysis questions and are not yet stable
enough to upstream.

Currently provides
------------------
``DoGDynamicAttentionFieldPRF2DWithHRF_v3_target``
    Extends the canonical
    :class:`braincoder.models.DoGDynamicAttentionFieldPRF2DWithHRF_v3`
    with a fifth, *target-onset* spatial-modulation term. The same
    visual transient that drives a (negative) gain at the distractor
    location ('history-driven suppression') is hypothesised to drive a
    POSITIVE gain when it appears at the search TARGET location
    ('attentional capture'). Fitting both gains in the same model
    provides an internal positive control for the AF framework.

``DoGDynamicAttentionFieldPRF2DWithHRF_v3_target_oversampled``
    Same model as above but built to operate the entire convolution
    chain at a *fine* timestep ``dt = TR / N``. The paradigm,
    condition_indicator, dynamic_indicator and target_indicator must be
    supplied at fine dt; the HRF must be constructed at fine dt; the
    BOLD predictions are downsampled by ``[::N]`` along the time axis
    before being returned, so they line up with the BOLD data at TR
    resolution. Use ``N=1`` for current-behaviour parity.

``DoGDynamicAttentionFieldPRF2DWithHRF_v3_target_sharedSigma``
    Same model as ``..._v3_target`` but ``sigma_T_dyn`` is forced to
    equal ``sigma_dyn`` in every forward pass. Used to test whether the
    target-onset Gaussian's spatial extent really differs from the
    distractor-onset one, or whether the larger σ_T_dyn estimates seen
    in V3AB / VO are identifiability slack. The parameter vector still
    contains a ``sigma_T_dyn`` slot (so the rest of the fit machinery
    is unchanged), but its raw value is overridden in
    ``_transform_parameters_forward`` and therefore has no effect on
    the loss.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp

from braincoder.models import (
    DoGDynamicAttentionFieldPRF2D_v3,
    DoGDynamicAttentionFieldPRF2DWithHRF_v3,
    HRFEncodingModel,
)


# ---------------------------------------------------------------------------
# Encoding-only (no HRF) version. We need this so the HRF wrapper has a
# clean encoding-side __init__ + transform pair to delegate to.
# ---------------------------------------------------------------------------
class DoGDynamicAttentionFieldPRF2D_v3_target(DoGDynamicAttentionFieldPRF2D_v3):
    """v3 + phasic-target gain. Encoding-only (no HRF) base.

    Extends :class:`DoGDynamicAttentionFieldPRF2D_v3` with two NEW
    shared parameters appended at the end of the parameter list:

    - ``g_T_dyn``  — gain on the per-TR phasic TARGET-onset modulation,
      shared across the four ring positions (targets are not split into
      HP/LP because the target appears roughly uniformly at each
      position regardless of the run-level HP).
    - ``sigma_T_dyn`` — extent of the target-onset Gaussian (peak at the
      ring position), independent of ``sigma_AF`` and ``sigma_dyn``.

    Per-voxel parameters (7) — unchanged
    ------------------------------------
    ``x``, ``y``, ``sd``, ``baseline``, ``amplitude``,
    ``srf_amplitude``, ``srf_size``.

    Shared parameters (8 = 6 from v3 + 2 new)
    -----------------------------------------
    ``sigma_AF``, ``g_HP``, ``g_LP``,
    ``sigma_dyn``, ``g_HP_dyn``, ``g_LP_dyn``,
    ``g_T_dyn``, ``sigma_T_dyn``.

    Total: 15 encoding parameters per voxel (7 per-voxel + 8 shared).

    Modulation field
    ----------------
    Identical to v3 (the per-condition sustained term plus the
    HP/LP-split phasic-distractor term) PLUS::

        + g_T_dyn · Σ_ℓ tgt_ℓ(t) · A_ℓ^{tgt}(g)

    where ``tgt_ℓ(t)`` is the per-TR per-location target-on overlap
    fraction (input ``target_indicator``, shape ``(T, n_C)``) and
    ``A_ℓ^{tgt}(g)`` is a peak-normalized Gaussian centred at ring
    position ``ℓ`` with width ``sigma_T_dyn``.

    Implementation notes
    --------------------
    The new parameters are appended at the END of the encoding-pars
    block (slots 13, 14). All v3 slot indices (sigma_AF=7, g_HP=8,
    g_LP=9, sigma_dyn=10, g_HP_dyn=11, g_LP_dyn=12) are unchanged so
    the parent v3 ``_attention_modulation`` and
    ``_attention_modulation_dynamic_v3`` continue to work without
    modification. The HRF wrapper handles the (possibly flexible) HRF
    parameters at the very end of the parameter vector.
    """

    parameter_labels = [
        'x', 'y', 'sd', 'baseline', 'amplitude',
        'srf_amplitude', 'srf_size',
        'sigma_AF', 'g_HP', 'g_LP',
        'sigma_dyn', 'g_HP_dyn', 'g_LP_dyn',
        'g_T_dyn', 'sigma_T_dyn',
    ]

    # Number of encoding parameters in the parent v3 (used to slice).
    _N_V3_ENC_PARS = 13

    def __init__(self, grid_coordinates=None, paradigm=None, data=None,
                 parameters=None, condition_indicator=None,
                 dynamic_indicator=None,
                 target_indicator=None,
                 ring_positions=None, mode='suppression',
                 weights=None, omega=None,
                 positive_image_values_only=True,
                 verbosity=logging.INFO, **kwargs):
        if target_indicator is None:
            raise ValueError(
                "DoGDynamicAttentionFieldPRF2D_v3_target requires a "
                "`target_indicator` array of shape "
                "(n_timepoints, n_ring_positions).")

        super().__init__(
            grid_coordinates=grid_coordinates, paradigm=paradigm, data=data,
            parameters=parameters,
            condition_indicator=condition_indicator,
            dynamic_indicator=dynamic_indicator,
            ring_positions=ring_positions, mode=mode,
            weights=weights, omega=omega,
            positive_image_values_only=positive_image_values_only,
            verbosity=verbosity, **kwargs)

        target_indicator = np.asarray(target_indicator, dtype=np.float32)
        if target_indicator.ndim != 2:
            raise ValueError(
                f"target_indicator must be 2-D (T, n_ring_positions); "
                f"got shape {target_indicator.shape}.")
        if target_indicator.shape[1] != self.n_conditions:
            raise ValueError(
                f"target_indicator has {target_indicator.shape[1]} "
                f"channels but ring_positions has {self.n_conditions}; "
                "channels must align with ring_positions.")
        if target_indicator.shape[0] != self.dynamic_indicator.shape[0]:
            raise ValueError(
                f"target_indicator has {target_indicator.shape[0]} "
                f"timepoints but dynamic_indicator has "
                f"{self.dynamic_indicator.shape[0]}; they must match.")
        self.target_indicator = target_indicator
        self._tf_target_indicator = tf.constant(self.target_indicator,
                                                dtype=tf.float32)

    # ----- Modulation field -------------------------------------------------

    @tf.function
    def _attention_modulation_target(self, parameters):
        """Per-TR phasic-target modulation field on the stimulus grid.

        Returns
        -------
        field : tf.Tensor, shape ``(T, G)``
            Sums over the 4 ring positions of
            ``g_T_dyn · tgt_ℓ(t) · A_ℓ^{tgt}(g)``, with
            ``A_ℓ^{tgt}(g) = exp(-||g - r_ℓ||² / (2 σ_T_dyn²))``.

        Notes
        -----
        Mirrors the structure of
        :meth:`DoGDynamicAttentionFieldPRF2D_v3._attention_modulation_dynamic_v3`
        (which is what the parent uses for the phasic distractor term).
        Parameter slot indices for the two new params:

        - ``g_T_dyn``      -> slot 13
        - ``sigma_T_dyn``  -> slot 14
        """
        # Shared parameters (read from first batch / first voxel).
        g_T_dyn = parameters[0, 0, 13]                     # scalar
        sigma_T_dyn = parameters[0, 0, 14]                 # scalar

        # Grid: (G, 2).
        gx = self._grid_coordinates[:, 0][tf.newaxis, :]   # (1, G)
        gy = self._grid_coordinates[:, 1][tf.newaxis, :]

        # Ring positions: (n_C, 1).
        rx = self._tf_ring_positions[:, 0][:, tf.newaxis]
        ry = self._tf_ring_positions[:, 1][:, tf.newaxis]

        # Per-ring TARGET Gaussian (peak-normalized to 1): (n_C, G).
        diff_sq = (gx - rx) ** 2 + (gy - ry) ** 2
        A_tgt = tf.exp(-diff_sq / (2.0 * sigma_T_dyn ** 2))  # (n_C, G)

        # tgt_ℓ(t): (T, n_C). Per-TR per-ring target on-fraction.
        tgt = self._tf_target_indicator                     # (T, n_C)

        # Σ_ℓ tgt[t, ℓ] · A_ℓ^{tgt}(g): (T, G).
        field_tgt = tf.einsum('tl,lg->tg', tgt, A_tgt)

        return g_T_dyn * field_tgt                          # (T, G)

    @tf.function
    def _basis_predictions(self, paradigm, parameters):
        # paradigm: (B, T, G)
        # parameters: (B, V, n_parameters=15)
        #
        # Same structure as the v3 parent's _basis_predictions, but the
        # phasic modulation also includes the target-onset term.

        # Per-voxel DoG receptive field: (B, V, G). Inherited.
        rf = self._get_rf(self.grid_coordinates, parameters)

        # Sustained per-condition AF modulation: (B, V, n_C, G).
        # DoGAttentionFieldPRF2D._attention_modulation reads slots 7,8,9.
        mod_sustained = self._attention_modulation(parameters)

        # Effective per-condition RF (sustained part): (B, V, n_C, G).
        eff_rf_per_cond = rf[:, :, tf.newaxis, :] * mod_sustained

        # Sustained partial: (B, T, V) via condition_indicator selection.
        partial = tf.einsum('btg,bvcg->btvc', paradigm, eff_rf_per_cond)
        ci = self._tf_condition_indicator       # (T, n_C)
        sustained = tf.einsum('tc,btvc->btv', ci, partial)

        # Phasic-distractor modulation: (T, G), HP/LP split, with σ_dyn.
        mod_dyn = self._attention_modulation_dynamic_v3(parameters)

        # NEW: phasic-target modulation: (T, G), with σ_T_dyn.
        mod_tgt = self._attention_modulation_target(parameters)

        # Combined phasic field (additive on the (T, G) plane).
        mod_phasic = mod_dyn + mod_tgt           # (T, G)

        # Phasic partial: (B, T, V).
        sign = self._tf_sign
        eff_paradigm_phasic = paradigm * mod_phasic[tf.newaxis, :, :]
        phasic = sign * tf.einsum('btg,bvg->btv', eff_paradigm_phasic, rf)

        result = sustained + phasic

        baseline = parameters[:, tf.newaxis, :, 3]
        result = result + baseline

        return result

    # ----- Parameter transforms --------------------------------------------

    @tf.function
    def _transform_parameters_forward(self, parameters):
        """Encoding-only forward transform.

        Slots 0..12 are the v3 encoding parameters: delegate to the v3
        parent's transform. Slots 13 (``g_T_dyn``) and 14
        (``sigma_T_dyn``) are NEW: g_T_dyn is sign-aware (signed mode
        passes through; otherwise softplus); sigma_T_dyn is always
        softplus-positive (matches sigma_AF / sigma_dyn).
        """
        v3_pars = DoGDynamicAttentionFieldPRF2D_v3._transform_parameters_forward(
            self, parameters[:, :self._N_V3_ENC_PARS])

        if self._signed_gains:
            g_T_dyn = parameters[:, 13][:, tf.newaxis]
        else:
            g_T_dyn = tf.math.softplus(parameters[:, 13][:, tf.newaxis])
        sigma_T_dyn = tf.math.softplus(parameters[:, 14][:, tf.newaxis])

        return tf.concat([v3_pars, g_T_dyn, sigma_T_dyn], axis=1)

    @tf.function
    def _transform_parameters_backward(self, parameters):
        v3_pars = DoGDynamicAttentionFieldPRF2D_v3._transform_parameters_backward(
            self, parameters[:, :self._N_V3_ENC_PARS])

        if self._signed_gains:
            g_T_dyn_unb = parameters[:, 13][:, tf.newaxis]
        else:
            g_T_dyn_unb = tfp.math.softplus_inverse(
                parameters[:, 13][:, tf.newaxis])
        sigma_T_dyn_unb = tfp.math.softplus_inverse(
            parameters[:, 14][:, tf.newaxis])

        return tf.concat([v3_pars, g_T_dyn_unb, sigma_T_dyn_unb], axis=1)


# ---------------------------------------------------------------------------
# HRF-convolved variant (the one the fit script uses).
# ---------------------------------------------------------------------------
class DoGDynamicAttentionFieldPRF2DWithHRF_v3_target(
    DoGDynamicAttentionFieldPRF2DWithHRF_v3,
    DoGDynamicAttentionFieldPRF2D_v3_target,
):
    """HRF-convolved version of
    :class:`DoGDynamicAttentionFieldPRF2D_v3_target`.

    Free parameters::

        ['x', 'y', 'sd', 'baseline', 'amplitude',
         'srf_amplitude', 'srf_size',
         'sigma_AF', 'g_HP', 'g_LP',
         'sigma_dyn', 'g_HP_dyn', 'g_LP_dyn',
         'g_T_dyn', 'sigma_T_dyn']
        (+ HRF parameters if flexible)

    During joint AF + DoG-PRF fitting, pass

        shared_pars=['sigma_AF', 'g_HP', 'g_LP',
                     'sigma_dyn', 'g_HP_dyn', 'g_LP_dyn',
                     'g_T_dyn', 'sigma_T_dyn']

    to the :class:`braincoder.optimize.ParameterFitter`.
    """

    def __init__(self, grid_coordinates=None, paradigm=None, data=None,
                 parameters=None, condition_indicator=None,
                 dynamic_indicator=None,
                 target_indicator=None,
                 ring_positions=None, mode='suppression',
                 positive_image_values_only=True,
                 weights=None, hrf_model=None,
                 flexible_hrf_parameters=False,
                 verbosity=logging.INFO, **kwargs):
        # Build the encoding side (with target indicator).
        DoGDynamicAttentionFieldPRF2D_v3_target.__init__(
            self, grid_coordinates=grid_coordinates, paradigm=paradigm,
            data=data, parameters=parameters,
            condition_indicator=condition_indicator,
            dynamic_indicator=dynamic_indicator,
            target_indicator=target_indicator,
            ring_positions=ring_positions, mode=mode,
            weights=weights, verbosity=verbosity,
            positive_image_values_only=positive_image_values_only, **kwargs)

        # Then attach the HRF.
        HRFEncodingModel.__init__(self, hrf_model=hrf_model,
                                  flexible_hrf_parameters=flexible_hrf_parameters,
                                  **kwargs)

    @tf.function
    def _transform_parameters_forward(self, parameters):
        if self.flexible_hrf_parameters:
            n_hrf_pars = len(self.hrf_model.parameter_labels)
            encoding_pars = (
                DoGDynamicAttentionFieldPRF2D_v3_target
                ._transform_parameters_forward(
                    self, parameters[:, :-n_hrf_pars])
            )
            hrf_pars = self.hrf_model._transform_parameters_forward(
                parameters[:, -n_hrf_pars:])
            return tf.concat([encoding_pars, hrf_pars], axis=1)
        else:
            return (
                DoGDynamicAttentionFieldPRF2D_v3_target
                ._transform_parameters_forward(self, parameters)
            )

    @tf.function
    def _transform_parameters_backward(self, parameters):
        if self.flexible_hrf_parameters:
            n_hrf_pars = len(self.hrf_model.parameter_labels)
            encoding_pars = (
                DoGDynamicAttentionFieldPRF2D_v3_target
                ._transform_parameters_backward(
                    self, parameters[:, :-n_hrf_pars])
            )
            hrf_pars = self.hrf_model._transform_parameters_backward(
                parameters[:, -n_hrf_pars:])
            return tf.concat([encoding_pars, hrf_pars], axis=1)
        else:
            return (
                DoGDynamicAttentionFieldPRF2D_v3_target
                ._transform_parameters_backward(self, parameters)
            )


# ---------------------------------------------------------------------------
# Temporally-oversampled HRF-convolved variant.
# ---------------------------------------------------------------------------
class DoGDynamicAttentionFieldPRF2DWithHRF_v3_target_oversampled(
    DoGDynamicAttentionFieldPRF2DWithHRF_v3_target,
):
    """Temporally-oversampled v3 + target model.

    Identical free parameters and forward model as
    :class:`DoGDynamicAttentionFieldPRF2DWithHRF_v3_target`, but the
    entire convolution chain is run at a fine timestep ``dt = TR /
    oversampling``. The caller is responsible for supplying every
    time-indexed input on that fine grid:

    - ``paradigm``           — shape ``(T*N, G)``
    - ``condition_indicator`` — shape ``(T*N, n_C)``
    - ``dynamic_indicator``   — shape ``(T*N, n_C)``
    - ``target_indicator``    — shape ``(T*N, n_C)``
    - ``hrf_model``           — built with ``tr=original_tr/N`` so its
      kernel is sampled at fine dt.

    The BOLD data passed to the fitter remains at TR resolution
    (``T`` rows). After the parent's ``_predict`` (which produces a
    ``(B, T*N, V)`` HRF-convolved prediction tensor), this subclass
    subsamples along the time axis with ``[:, ::N, :]`` to reduce the
    output to ``(B, T, V)``, matching the BOLD shape used in the loss.

    With ``oversampling=1`` the subsample is ``[:, ::1, :]`` i.e. a
    no-op, and the model is mathematically identical to the parent.

    Notes
    -----
    The HRF convolution always uses ``unique_hrfs=False`` here (shared
    canonical HRF). braincoder's ``_convolve_shared`` takes the time
    axis from ``timeseries.shape[1]`` so it works at any timestep.
    """

    def __init__(self, grid_coordinates=None, paradigm=None, data=None,
                 parameters=None, condition_indicator=None,
                 dynamic_indicator=None,
                 target_indicator=None,
                 ring_positions=None, mode='suppression',
                 positive_image_values_only=True,
                 weights=None, hrf_model=None,
                 flexible_hrf_parameters=False,
                 oversampling=1,
                 verbosity=logging.INFO, **kwargs):
        if int(oversampling) < 1:
            raise ValueError(
                f"oversampling must be >= 1, got {oversampling}")
        self.oversampling = int(oversampling)

        super().__init__(
            grid_coordinates=grid_coordinates, paradigm=paradigm,
            data=data, parameters=parameters,
            condition_indicator=condition_indicator,
            dynamic_indicator=dynamic_indicator,
            target_indicator=target_indicator,
            ring_positions=ring_positions, mode=mode,
            positive_image_values_only=positive_image_values_only,
            weights=weights, hrf_model=hrf_model,
            flexible_hrf_parameters=flexible_hrf_parameters,
            verbosity=verbosity, **kwargs)

    @tf.function
    def _predict(self, paradigm, parameters, weights):
        """Run the parent forward pass, then subsample by ``oversampling``.

        The parent's ``_predict`` returns ``(B, T_fine, V)``. We slice
        ``[:, ::N, :]`` along axis 1 to align with the BOLD data at TR
        resolution.
        """
        pred_fine = super()._predict(paradigm, parameters, weights)
        # tf.strided_slice via Python slice on axis 1.
        return pred_fine[:, ::self.oversampling, :]

    def predict(self, paradigm=None, parameters=None, weights=None):
        """Public wrapper that returns a TR-indexed DataFrame.

        We cannot simply call the inherited ``predict`` because it uses
        the (fine) paradigm's index for the result rows — but our
        ``_predict`` already downsamples to ``T`` rows. So we
        re-implement the wrapper with a subsampled index.
        """
        weights, weights_ = self._get_weights(weights)

        paradigm = self.get_paradigm(paradigm)
        paradigm_ = self._get_paradigm(paradigm)[np.newaxis, ...]

        parameters = self._get_parameters(parameters)
        parameters_ = (parameters.values[np.newaxis, ...]
                       if parameters is not None else None)

        # _predict subsamples to TR resolution.
        predictions = self._predict(paradigm_, parameters_, weights_)[0]

        # Subsample paradigm.index (which is at fine dt) to match.
        idx = paradigm.index[::self.oversampling]
        # Defensive: shapes must agree after subsampling.
        if len(idx) != int(predictions.shape[0]):
            # Fall back to a plain RangeIndex if the user passed a paradigm
            # whose length isn't an exact multiple of oversampling.
            idx = pd.RangeIndex(int(predictions.shape[0]))

        if weights is None:
            return pd.DataFrame(
                predictions.numpy(), index=idx, columns=parameters.index)
        else:
            return pd.DataFrame(
                predictions.numpy(), index=idx, columns=weights.columns)


# ---------------------------------------------------------------------------
# Shared-σ variant: σ_T_dyn forced to equal σ_dyn.
# ---------------------------------------------------------------------------
class DoGDynamicAttentionFieldPRF2DWithHRF_v3_target_sharedSigma(
    DoGDynamicAttentionFieldPRF2DWithHRF_v3_target,
):
    """v3 + target with ``sigma_T_dyn`` tied to ``sigma_dyn``.

    Same parameter vector and forward model as
    :class:`DoGDynamicAttentionFieldPRF2DWithHRF_v3_target`, but
    ``sigma_T_dyn`` is overwritten with ``sigma_dyn`` at the very end
    of the forward parameter transform. The two phasic Gaussians
    (distractor-onset and target-onset) therefore share a single
    spatial extent, which we call the "phasic σ".

    The parameter vector still contains a ``sigma_T_dyn`` slot — we do
    NOT renumber slot indices or rewrite ``parameter_labels`` so the
    rest of the braincoder fit machinery (ParameterFitter, shared_pars
    plumbing, output DataFrame columns) is unchanged. The σ_T_dyn raw
    variable still exists in the optimiser's free-parameter set, but
    every forward pass overwrites the post-softplus σ_T_dyn with the
    post-softplus σ_dyn before the loss is computed, so σ_T_dyn's raw
    variable has zero gradient flowing back to it through the model.

    Initialisation
    --------------
    Callers should set ``init_pars['sigma_T_dyn'] = init_pars['sigma_dyn']``
    before passing inits to the fitter, so the (effectively unused)
    σ_T_dyn raw variable starts at the right place. The fit script
    handles this when ``--shared-target-sigma`` is set.

    Composing with temporal oversampling
    ------------------------------------
    For shared-σ + temporal oversampling, write a small subclass that
    inherits from BOTH this class and
    :class:`DoGDynamicAttentionFieldPRF2DWithHRF_v3_target_oversampled`,
    or replicate this override on a sibling oversampled class.
    """

    @tf.function
    def _transform_parameters_forward(self, parameters):
        # Run the parent forward transform (handles HRF + encoding pars,
        # applies softplus to both sigma_dyn (slot 10) and sigma_T_dyn
        # (slot 14)).
        out = super()._transform_parameters_forward(parameters)

        # Force sigma_T_dyn := sigma_dyn after the parent transform.
        # `out` shape: (n_voxels, n_parameters[+ n_hrf_pars]).
        sigma_dyn_col = out[:, 10:11]                         # (V, 1)
        # Build the new tensor by concatenation around slot 14.
        # Slots: [0..14, hrf...] -> replace slot 14 with sigma_dyn_col.
        before = out[:, :14]                                  # (V, 14)
        after = out[:, 15:]                                   # (V, rest)
        out_tied = tf.concat([before, sigma_dyn_col, after], axis=1)
        return out_tied

    @tf.function
    def _transform_parameters_backward(self, parameters):
        # Run the parent backward transform (inverts softplus on both
        # sigma_dyn slot 10 and sigma_T_dyn slot 14 in raw space).
        out = super()._transform_parameters_backward(parameters)

        # Tie the raw σ_T_dyn slot to the raw σ_dyn slot so that, if
        # callers ever recover the "raw" parameter vector, both raw
        # variables agree. Both sigmas use softplus so equal raw values
        # give equal post-softplus values.
        raw_sigma_dyn_col = out[:, 10:11]                     # (V, 1)
        before = out[:, :14]                                  # (V, 14)
        after = out[:, 15:]                                   # (V, rest)
        out_tied = tf.concat([before, raw_sigma_dyn_col, after], axis=1)
        return out_tied
