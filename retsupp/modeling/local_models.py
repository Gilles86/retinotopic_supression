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

``DoGDynamicAttentionFieldPRF2DWithHRF_v3_target_sharedSigma_factorial``
    sharedSigma v3+target with per-gain sign-constraints. Used by
    ``fit_af_prf_cv_v2.py`` for the 18-class CV-v2 factorial: each of
    the five gains (``g_HP``, ``g_LP``, ``g_HP_dyn``, ``g_LP_dyn``,
    ``g_T_dyn``) is mapped through one of {``plus``, ``minus``,
    ``zero``, ``free``}.  σ_T_dyn := σ_dyn is preserved.

``DoGDynamicAttentionFieldPRF2DWithHRF_v3_target_sharedSigma_runPosition``
    sharedSigma v3+target with per-run-position sustained gains. The
    single sustained pair ``(g_HP, g_LP)`` is replaced by 6 parameters
    ``g_{HP,LP}_pos{0,1,2}`` — one pair per chronological position of
    the run within its HP block. Lets us inspect arbitrary learning
    curves over the 3-run HP blocks without forcing a linear-slope
    parameterization. Subclasses the factorial machinery for the
    dynamic + target gains; the 6 new sustained gains are always
    signed-free (no factorial sign pattern is applied to them).

``GaussianAFDriveModel`` / ``GaussianAFAnalyticalShiftModel`` /
``GaussianAFNumericalShiftModel``
    Three formulations of attention-field-modulated PRFs on a
    *Gaussian* (not DoG) backbone, used to compare three different
    ways of injecting the AF effect on the prediction:

    - DRIVE   (Model A): amplitude/drive modulation via "1 + ..." on
      the stimulus paradigm before integration. This is the
      AF+ formulation of Sumiya et al., implemented numerically.
    - ANALYTICAL  (Model B): closed-form precision-weighted shift of
      the PRF center per TR (no offset). Departure from Sumiya's
      strict-positive-gain formulation: signed gains are allowed and
      contribute *signed* precision (gain<0 pushes COM away).
    - NUMERICAL  (Model C): build the modulated effective PRF
      ``M(g, t) · PRF_v(g)`` numerically per TR per voxel, extract the
      |eff_PRF| centre-of-mass as the per-TR shifted centre, and
      refit a symmetric Gaussian with the original σ_PRF. This
      mirrors what the conditionwise PRF analysis does (a Gaussian is
      always re-fit to the shifted RF).

    All three classes share an 8-shared-parameter setup (σ_AF, σ_dyn,
    g_HP, g_LP, g_HP_dyn, g_LP_dyn, g_T_dyn, σ_T_dyn := σ_dyn) and a
    13-slot encoding parameter vector before HRF; see the class
    docstrings.
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
    DynamicAttentionFieldPRF2D_v3,
    DynamicAttentionFieldPRF2DWithHRF_v3,
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


# ---------------------------------------------------------------------------
# Factorial sign-constraint variant of the shared-σ + target model.
# Used by ``fit_af_prf_cv_v2.py`` (CV-v2 18-class factorial).
# ---------------------------------------------------------------------------

def _gain_forward_factorial(raw, sign):
    """Per-gain sign-constraint applied to the raw variable in forward pass.

    ``raw`` is the (V, 1) raw parameter column for one gain.  ``sign`` is
    one of {'plus', 'minus', 'zero', 'free'}.

    - ``plus``  : ``+softplus(raw)``           (sign ≥ 0, optimised)
    - ``minus`` : ``-softplus(raw)``           (sign ≤ 0, optimised)
    - ``zero``  : ``0`` (the raw variable is also marked fixed by the
      caller via ``fixed_pars`` so its gradient is zeroed; the forward
      transform additionally clamps to exact 0).
    - ``free``  : pass through (fully signed-unconstrained).
    """
    if sign == 'plus':
        return tf.math.softplus(raw)
    if sign == 'minus':
        return -tf.math.softplus(raw)
    if sign == 'zero':
        return tf.zeros_like(raw)
    if sign == 'free':
        return raw
    raise ValueError(f'Unknown sign {sign!r}')


def _gain_backward_factorial(usable, sign):
    """Inverse transform of :func:`_gain_forward_factorial`.

    Used to map a usable-space init value (e.g. ``-0.10``) back to the
    raw variable used by the optimiser.  ``zero`` and ``free`` pass
    through; ``plus``/``minus`` use ``softplus_inverse`` on the
    magnitude.
    """
    if sign == 'plus':
        return tfp.math.softplus_inverse(tf.maximum(usable, 1e-4))
    if sign == 'minus':
        return tfp.math.softplus_inverse(tf.maximum(-usable, 1e-4))
    if sign == 'zero':
        return tf.zeros_like(usable)
    if sign == 'free':
        return usable
    raise ValueError(f'Unknown sign {sign!r}')


class DoGDynamicAttentionFieldPRF2DWithHRF_v3_target_sharedSigma_factorial(
    DoGDynamicAttentionFieldPRF2DWithHRF_v3_target_sharedSigma,
):
    """sharedSigma v3+target with per-gain sign-constraints (CV-v2).

    Identical forward model and parameter vector as
    :class:`DoGDynamicAttentionFieldPRF2DWithHRF_v3_target_sharedSigma`
    (σ_T_dyn is still tied to σ_dyn at every forward pass), but each of
    the five gain parameters --- ``g_HP``, ``g_LP``, ``g_HP_dyn``,
    ``g_LP_dyn``, ``g_T_dyn`` --- is mapped through one of four
    constraint transforms picked at construction:

        plus    g = +softplus(raw)   — optimised, sign ≥ 0
        minus   g = -softplus(raw)   — optimised, sign ≤ 0
        zero    g = 0.0              — clamped to 0; raw also fixed by
                                        the caller via ``fixed_pars``
        free    g = raw              — signed-unconstrained

    The factorial design used by ``fit_af_prf_cv_v2.py`` crosses the
    sustained pair (g_HP, g_LP) and the dynamic pair (g_HP_dyn,
    g_LP_dyn) over three sign-patterns each, plus a binary on/off on
    g_T_dyn (free vs zero) — 3 × 3 × 2 = 18 cells in total.

    The model is always constructed with ``mode='signed'`` so that the
    parent's own ``_signed_gains`` logic stays out of the way; this
    subclass owns the full forward/backward parameter transform
    end-to-end (DoG transforms, σ_AF / σ_dyn softplus, per-gain
    constraints, σ_T_dyn := σ_dyn tying, optional HRF tail).
    """

    _GAIN_SLOT_INDEX = {
        'g_HP': 8,
        'g_LP': 9,
        'g_HP_dyn': 11,
        'g_LP_dyn': 12,
        'g_T_dyn': 13,
    }
    _GAIN_NAMES = ('g_HP', 'g_LP', 'g_HP_dyn', 'g_LP_dyn', 'g_T_dyn')

    def __init__(self, *, sign_pattern, **kwargs):
        # sign_pattern: dict with the 5 _GAIN_NAMES -> str sign tag.
        missing = [g for g in self._GAIN_NAMES if g not in sign_pattern]
        if missing:
            raise ValueError(
                f"sign_pattern is missing entries for {missing}. "
                f"Required keys: {list(self._GAIN_NAMES)}.")
        bad = {g: s for g, s in sign_pattern.items()
               if s not in ('plus', 'minus', 'zero', 'free')}
        if bad:
            raise ValueError(
                f"sign_pattern values must be in "
                f"{{'plus','minus','zero','free'}}; got {bad}.")
        # Force signed mode so the parent's _signed_gains branch is
        # taken (passes raw gains through unchanged); we re-apply our
        # own per-gain sign transforms in this subclass's forward.
        kwargs['mode'] = 'signed'
        super().__init__(**kwargs)
        self._sign_pattern = dict(sign_pattern)

    # ---- Forward / backward parameter transforms -------------------------

    @tf.function
    def _transform_parameters_forward(self, parameters):
        # Split off optional HRF tail.
        if self.flexible_hrf_parameters:
            n_hrf = len(self.hrf_model.parameter_labels)
            enc = parameters[:, :-n_hrf]
            hrf_tail = parameters[:, -n_hrf:]
        else:
            enc = parameters
            hrf_tail = None

        x = enc[:, 0:1]
        y = enc[:, 1:2]
        sd = tf.math.softplus(enc[:, 2:3])
        baseline = enc[:, 3:4]
        amplitude = enc[:, 4:5]
        srf_amp = tf.math.softplus(enc[:, 5:6])
        srf_size = tf.math.softplus(enc[:, 6:7])
        sigma_AF = tf.math.softplus(enc[:, 7:8])
        g_HP = _gain_forward_factorial(enc[:, 8:9],
                                       self._sign_pattern['g_HP'])
        g_LP = _gain_forward_factorial(enc[:, 9:10],
                                       self._sign_pattern['g_LP'])
        sigma_dyn = tf.math.softplus(enc[:, 10:11])
        g_HP_dyn = _gain_forward_factorial(enc[:, 11:12],
                                           self._sign_pattern['g_HP_dyn'])
        g_LP_dyn = _gain_forward_factorial(enc[:, 12:13],
                                           self._sign_pattern['g_LP_dyn'])
        g_T_dyn = _gain_forward_factorial(enc[:, 13:14],
                                          self._sign_pattern['g_T_dyn'])
        # σ_T_dyn := σ_dyn (the whole point of the sharedSigma class).
        sigma_T_dyn = sigma_dyn

        out_enc = tf.concat([
            x, y, sd, baseline, amplitude, srf_amp, srf_size,
            sigma_AF, g_HP, g_LP,
            sigma_dyn, g_HP_dyn, g_LP_dyn,
            g_T_dyn, sigma_T_dyn,
        ], axis=1)

        if hrf_tail is not None:
            hrf_pars = self.hrf_model._transform_parameters_forward(hrf_tail)
            return tf.concat([out_enc, hrf_pars], axis=1)
        return out_enc

    @tf.function
    def _transform_parameters_backward(self, parameters):
        if self.flexible_hrf_parameters:
            n_hrf = len(self.hrf_model.parameter_labels)
            enc = parameters[:, :-n_hrf]
            hrf_tail = parameters[:, -n_hrf:]
        else:
            enc = parameters
            hrf_tail = None

        x = enc[:, 0:1]
        y = enc[:, 1:2]
        sd = tfp.math.softplus_inverse(enc[:, 2:3])
        baseline = enc[:, 3:4]
        amplitude = enc[:, 4:5]
        srf_amp = tfp.math.softplus_inverse(enc[:, 5:6])
        srf_size = tfp.math.softplus_inverse(enc[:, 6:7])
        sigma_AF = tfp.math.softplus_inverse(enc[:, 7:8])
        g_HP = _gain_backward_factorial(enc[:, 8:9],
                                        self._sign_pattern['g_HP'])
        g_LP = _gain_backward_factorial(enc[:, 9:10],
                                        self._sign_pattern['g_LP'])
        sigma_dyn = tfp.math.softplus_inverse(enc[:, 10:11])
        g_HP_dyn = _gain_backward_factorial(enc[:, 11:12],
                                            self._sign_pattern['g_HP_dyn'])
        g_LP_dyn = _gain_backward_factorial(enc[:, 12:13],
                                            self._sign_pattern['g_LP_dyn'])
        g_T_dyn = _gain_backward_factorial(enc[:, 13:14],
                                           self._sign_pattern['g_T_dyn'])
        # Tie raw σ_T_dyn := raw σ_dyn so the recovered raw parameter
        # vector is internally consistent.  Both use softplus, so equal
        # raw values give equal post-softplus values.
        sigma_T_dyn = sigma_dyn

        out_enc = tf.concat([
            x, y, sd, baseline, amplitude, srf_amp, srf_size,
            sigma_AF, g_HP, g_LP,
            sigma_dyn, g_HP_dyn, g_LP_dyn,
            g_T_dyn, sigma_T_dyn,
        ], axis=1)

        if hrf_tail is not None:
            hrf_pars = self.hrf_model._transform_parameters_backward(hrf_tail)
            return tf.concat([out_enc, hrf_pars], axis=1)
        return out_enc


# ---------------------------------------------------------------------------
# Per-run-position sustained-gain variant of the shared-σ + target model.
# Used to test for learning over the 3-run HP blocks (VSS2026).
# ---------------------------------------------------------------------------
class DoGDynamicAttentionFieldPRF2DWithHRF_v3_target_sharedSigma_runPosition(
    DoGDynamicAttentionFieldPRF2DWithHRF_v3_target_sharedSigma_factorial,
):
    """sharedSigma v3+target with per-run-position sustained gains.

    Same forward model as
    :class:`DoGDynamicAttentionFieldPRF2DWithHRF_v3_target_sharedSigma_factorial`,
    except the SUSTAINED gain pair ``(g_HP, g_LP)`` is *replaced* by
    SIX new shared parameters, one per chronological position
    (``0/1/2``) of the run within its HP block::

        g_HP_pos0, g_HP_pos1, g_HP_pos2
        g_LP_pos0, g_LP_pos1, g_LP_pos2

    For TR ``t`` the effective sustained gains are::

        g_HP_eff[t] = Σ_r runpos[t, r] · g_HP_pos[r]
        g_LP_eff[t] = Σ_r runpos[t, r] · g_LP_pos[r]

    where ``runpos`` is a one-hot ``(T, 3)`` indicator passed in via the
    ``run_position_indicator`` constructor kwarg. The original ``g_HP``
    / ``g_LP`` slots in the parameter vector (slots 8 and 9) still
    exist (so the parent's slot indexing for σ_AF, σ_dyn, dynamic gains,
    σ_T_dyn etc. is unchanged), but their values are *unused* in this
    subclass's forward pass. Callers should pass ``g_HP=g_LP=0.0`` at
    init.

    Dynamic-distractor and target-onset gains (``g_HP_dyn``,
    ``g_LP_dyn``, ``g_T_dyn``) are unchanged from the factorial parent —
    no run-position structure on those — and the parent's per-gain
    sign-pattern factorial machinery is preserved for them. The 6 new
    sustained gains are always signed-free (no factorial sign mapping):
    we don't apply ``_gain_forward_factorial`` to them.

    Parameter vector (encoding side, 21 slots)
    ------------------------------------------
    Same first 15 slots as the parent (so HRF tail and parent slot
    indices are unchanged), then the 6 new gains appended at the end::

        ['x', 'y', 'sd', 'baseline', 'amplitude',
         'srf_amplitude', 'srf_size',
         'sigma_AF', 'g_HP', 'g_LP',
         'sigma_dyn', 'g_HP_dyn', 'g_LP_dyn',
         'g_T_dyn', 'sigma_T_dyn',
         'g_HP_pos0', 'g_HP_pos1', 'g_HP_pos2',
         'g_LP_pos0', 'g_LP_pos1', 'g_LP_pos2']
        (+ HRF parameters if flexible)

    Notes
    -----
    The 6 new gains are appended AFTER the parent's 15 encoding slots.
    This means:

    - The HRF tail (if flexible) sits AFTER all 21 encoding slots, so
      the HRF transform sees a 21-wide encoding tensor (vs 15 for the
      parent). We override the forward/backward transforms accordingly.
    - The parent's slot indices for ``g_HP_dyn`` (11), ``g_LP_dyn``
      (12), ``g_T_dyn`` (13), ``sigma_T_dyn`` (14), σ_AF (7), σ_dyn
      (10) are preserved — both ``_basis_predictions`` (which reads
      σ_dyn / dynamic gains) and the parent's σ_T_dyn := σ_dyn tying
      keep working unchanged.
    - The original (slot 8/9) ``g_HP`` / ``g_LP`` are not read in this
      subclass's ``_basis_predictions``; we set them to 0 in the
      forward transform regardless of the raw variable, both for
      cleanliness and so that any leftover dependency in inherited code
      paths sees zero gain.
    """

    # Parent's 15 encoding labels (inherited from ..._v3_target) plus
    # the 6 new run-position gains.
    parameter_labels = [
        'x', 'y', 'sd', 'baseline', 'amplitude',
        'srf_amplitude', 'srf_size',
        'sigma_AF', 'g_HP', 'g_LP',
        'sigma_dyn', 'g_HP_dyn', 'g_LP_dyn',
        'g_T_dyn', 'sigma_T_dyn',
        'g_HP_pos0', 'g_HP_pos1', 'g_HP_pos2',
        'g_LP_pos0', 'g_LP_pos1', 'g_LP_pos2',
    ]

    # Number of "parent encoding" slots (matches the factorial parent's
    # encoding-side slot count). We append 6 new gain slots after this.
    _N_PARENT_ENC_PARS = 15
    _N_RUNPOS_PARS = 6  # 3 HP + 3 LP

    def __init__(self, *, run_position_indicator, sign_pattern=None,
                 **kwargs):
        # Default: all gains signed-free. The 'unused' slots g_HP / g_LP
        # are left 'free' too — they don't enter our forward pass, but
        # the factorial parent still sees a value for them.
        if sign_pattern is None:
            sign_pattern = {
                'g_HP': 'free', 'g_LP': 'free',
                'g_HP_dyn': 'free', 'g_LP_dyn': 'free',
                'g_T_dyn': 'free',
            }
        super().__init__(sign_pattern=sign_pattern, **kwargs)

        rp = np.asarray(run_position_indicator, dtype=np.float32)
        if rp.ndim != 2:
            raise ValueError(
                f"run_position_indicator must be 2-D (T, 3); got shape "
                f"{rp.shape}.")
        if rp.shape[1] != 3:
            raise ValueError(
                f"run_position_indicator must have 3 columns (one-hot "
                f"over run-positions 0/1/2); got shape {rp.shape}.")
        # Sanity: same length as the dynamic_indicator the parent stored.
        T_expected = int(self.dynamic_indicator.shape[0])
        if rp.shape[0] != T_expected:
            raise ValueError(
                f"run_position_indicator has {rp.shape[0]} timepoints "
                f"but expected {T_expected} (matches dynamic_indicator).")
        self.run_position_indicator = rp
        self._tf_run_position_indicator = tf.constant(rp, dtype=tf.float32)

    # ----- Forward / backward parameter transforms -----------------------

    @tf.function
    def _transform_parameters_forward(self, parameters):
        """Forward transform.

        Layout::

            [parent-encoding (15)] [runpos gains (6)] [hrf tail (optional)]

        We replicate the factorial parent's encoding-side transform
        inline here (so we don't have to fight the parent's HRF-split
        logic). The 6 new gains pass through unchanged (signed-free);
        the HRF tail is passed through the HRF model's own transform.
        """
        if self.flexible_hrf_parameters:
            n_hrf = len(self.hrf_model.parameter_labels)
            enc = parameters[:, :self._N_PARENT_ENC_PARS]
            runpos = parameters[
                :, self._N_PARENT_ENC_PARS:
                self._N_PARENT_ENC_PARS + self._N_RUNPOS_PARS]
            hrf_tail = parameters[:, -n_hrf:]
        else:
            enc = parameters[:, :self._N_PARENT_ENC_PARS]
            runpos = parameters[
                :, self._N_PARENT_ENC_PARS:
                self._N_PARENT_ENC_PARS + self._N_RUNPOS_PARS]
            hrf_tail = None

        # Replicates factorial-parent _transform_parameters_forward.
        x = enc[:, 0:1]
        y = enc[:, 1:2]
        sd = tf.math.softplus(enc[:, 2:3])
        baseline = enc[:, 3:4]
        amplitude = enc[:, 4:5]
        srf_amp = tf.math.softplus(enc[:, 5:6])
        srf_size = tf.math.softplus(enc[:, 6:7])
        sigma_AF = tf.math.softplus(enc[:, 7:8])
        g_HP = _gain_forward_factorial(enc[:, 8:9],
                                       self._sign_pattern['g_HP'])
        g_LP = _gain_forward_factorial(enc[:, 9:10],
                                       self._sign_pattern['g_LP'])
        sigma_dyn = tf.math.softplus(enc[:, 10:11])
        g_HP_dyn = _gain_forward_factorial(enc[:, 11:12],
                                           self._sign_pattern['g_HP_dyn'])
        g_LP_dyn = _gain_forward_factorial(enc[:, 12:13],
                                           self._sign_pattern['g_LP_dyn'])
        g_T_dyn = _gain_forward_factorial(enc[:, 13:14],
                                          self._sign_pattern['g_T_dyn'])
        # σ_T_dyn := σ_dyn (the sharedSigma constraint).
        sigma_T_dyn = sigma_dyn

        out_enc = tf.concat([
            x, y, sd, baseline, amplitude, srf_amp, srf_size,
            sigma_AF, g_HP, g_LP,
            sigma_dyn, g_HP_dyn, g_LP_dyn,
            g_T_dyn, sigma_T_dyn,
            # 6 new gains: signed-free pass-through.
            runpos,
        ], axis=1)

        if hrf_tail is not None:
            hrf_pars = self.hrf_model._transform_parameters_forward(hrf_tail)
            return tf.concat([out_enc, hrf_pars], axis=1)
        return out_enc

    @tf.function
    def _transform_parameters_backward(self, parameters):
        if self.flexible_hrf_parameters:
            n_hrf = len(self.hrf_model.parameter_labels)
            enc = parameters[:, :self._N_PARENT_ENC_PARS]
            runpos = parameters[
                :, self._N_PARENT_ENC_PARS:
                self._N_PARENT_ENC_PARS + self._N_RUNPOS_PARS]
            hrf_tail = parameters[:, -n_hrf:]
        else:
            enc = parameters[:, :self._N_PARENT_ENC_PARS]
            runpos = parameters[
                :, self._N_PARENT_ENC_PARS:
                self._N_PARENT_ENC_PARS + self._N_RUNPOS_PARS]
            hrf_tail = None

        x = enc[:, 0:1]
        y = enc[:, 1:2]
        sd = tfp.math.softplus_inverse(enc[:, 2:3])
        baseline = enc[:, 3:4]
        amplitude = enc[:, 4:5]
        srf_amp = tfp.math.softplus_inverse(enc[:, 5:6])
        srf_size = tfp.math.softplus_inverse(enc[:, 6:7])
        sigma_AF = tfp.math.softplus_inverse(enc[:, 7:8])
        g_HP = _gain_backward_factorial(enc[:, 8:9],
                                        self._sign_pattern['g_HP'])
        g_LP = _gain_backward_factorial(enc[:, 9:10],
                                        self._sign_pattern['g_LP'])
        sigma_dyn = tfp.math.softplus_inverse(enc[:, 10:11])
        g_HP_dyn = _gain_backward_factorial(enc[:, 11:12],
                                            self._sign_pattern['g_HP_dyn'])
        g_LP_dyn = _gain_backward_factorial(enc[:, 12:13],
                                            self._sign_pattern['g_LP_dyn'])
        g_T_dyn = _gain_backward_factorial(enc[:, 13:14],
                                           self._sign_pattern['g_T_dyn'])
        sigma_T_dyn = sigma_dyn  # Tie raw -> raw (both softplus).

        out_enc = tf.concat([
            x, y, sd, baseline, amplitude, srf_amp, srf_size,
            sigma_AF, g_HP, g_LP,
            sigma_dyn, g_HP_dyn, g_LP_dyn,
            g_T_dyn, sigma_T_dyn,
            runpos,  # 6 new signed-free gains pass-through.
        ], axis=1)

        if hrf_tail is not None:
            hrf_pars = self.hrf_model._transform_parameters_backward(hrf_tail)
            return tf.concat([out_enc, hrf_pars], axis=1)
        return out_enc

    # ----- Forward model --------------------------------------------------

    @tf.function
    def _basis_predictions(self, paradigm, parameters):
        """v3+target+sharedSigma forward pass with per-run-position
        SUSTAINED gain.

        Mirrors the parent's
        :meth:`DoGDynamicAttentionFieldPRF2D_v3_target._basis_predictions`
        but builds the sustained modulation directly on a (T, G) field
        with per-TR effective gains computed from the 6 new run-position
        gain parameters.

        Parameters
        ----------
        paradigm : tf.Tensor, shape ``(B, T, G)``
        parameters : tf.Tensor, shape ``(B, V, n_parameters)`` where
            ``n_parameters = 21`` (encoding) ``[+ n_hrf]``.
        """
        # --- Per-voxel DoG receptive field: (B, V, G). ---
        rf = self._get_rf(self.grid_coordinates, parameters)

        # --- Per-TR effective sustained gains (from the 6 new pars). ---
        # parameters[..., 15:18] = g_HP_pos0/1/2; [18:21] = g_LP_pos0/1/2.
        # Shared across batches/voxels: read from [0, 0, ...].
        g_HP_pos = parameters[0, 0, 15:18]                   # (3,)
        g_LP_pos = parameters[0, 0, 18:21]                   # (3,)

        runpos = self._tf_run_position_indicator             # (T, 3)
        # Effective gains per TR: (T,).
        g_HP_eff = tf.linalg.matvec(runpos, g_HP_pos)        # (T,)
        g_LP_eff = tf.linalg.matvec(runpos, g_LP_pos)        # (T,)

        # --- Sustained AF Gaussians per ring position: (n_C, G). ---
        sigma_AF = parameters[0, 0, 7]                       # scalar
        gx = self._grid_coordinates[:, 0][tf.newaxis, :]     # (1, G)
        gy = self._grid_coordinates[:, 1][tf.newaxis, :]
        rx = self._tf_ring_positions[:, 0][:, tf.newaxis]    # (n_C, 1)
        ry = self._tf_ring_positions[:, 1][:, tf.newaxis]
        diff_sq = (gx - rx) ** 2 + (gy - ry) ** 2
        A_sus = tf.exp(-diff_sq / (2.0 * sigma_AF ** 2))     # (n_C, G)

        # condition_indicator[t, ℓ] is 1 iff ring ℓ is the HP for TR t's
        # run; (1 - ci[t, ℓ]) marks the LP rings.
        ci = self._tf_condition_indicator                    # (T, n_C)
        # HP field per TR: ci[t, ℓ] · A_ℓ(g) summed over ℓ -> (T, G).
        field_hp = tf.einsum('tl,lg->tg', ci, A_sus)
        # LP field per TR: (1 - ci[t, ℓ]) · A_ℓ(g) summed over ℓ.
        field_lp = tf.einsum('tl,lg->tg', 1.0 - ci, A_sus)

        # Sustained modulation field on (T, G):
        # m_sus[t, g] = 1 + sign · ( g_HP_eff[t] · field_hp[t, g]
        #                          + g_LP_eff[t] · field_lp[t, g] )
        sign = self._tf_sign
        mod_sustained_tg = 1.0 + sign * (
            g_HP_eff[:, tf.newaxis] * field_hp
            + g_LP_eff[:, tf.newaxis] * field_lp
        )
        mod_sustained_tg = tf.maximum(mod_sustained_tg, 0.0)

        # Sustained partial: (B, T, V) using paradigm * mod_sus then RF.
        eff_paradigm_sus = paradigm * mod_sustained_tg[tf.newaxis, :, :]
        sustained = tf.einsum('btg,bvg->btv', eff_paradigm_sus, rf)

        # --- Phasic distractor (v3, σ_dyn, slot 10/11/12). ---
        mod_dyn = self._attention_modulation_dynamic_v3(parameters)

        # --- Phasic target (slot 13 g_T_dyn, slot 14 σ_T_dyn := σ_dyn). ---
        mod_tgt = self._attention_modulation_target(parameters)

        mod_phasic = mod_dyn + mod_tgt                       # (T, G)
        eff_paradigm_phasic = paradigm * mod_phasic[tf.newaxis, :, :]
        # Note: phasic terms multiply by ``sign`` in the parent forward.
        # We replicate that here.
        phasic = sign * tf.einsum('btg,bvg->btv',
                                  eff_paradigm_phasic, rf)

        result = sustained + phasic
        baseline = parameters[:, tf.newaxis, :, 3]
        result = result + baseline
        return result


# ---------------------------------------------------------------------------
# Three Gaussian-backbone AF formulations for the model-comparison sweep.
#
# Common encoding-side parameter layout (13 slots, before optional HRF tail):
#     0  x
#     1  y
#     2  sd                 (per-voxel)
#     3  baseline           (per-voxel)
#     4  amplitude          (per-voxel)
#     5  sigma_AF           (shared)
#     6  g_HP               (shared)
#     7  g_LP               (shared)
#     8  sigma_dyn          (shared)
#     9  g_HP_dyn           (shared)
#     10 g_LP_dyn           (shared)
#     11 g_T_dyn            (shared)
#     12 sigma_T_dyn        (shared, tied to sigma_dyn — sharedSigma)
#
# The three subclasses differ ONLY in ``_basis_predictions``. They share the
# constructor (target_indicator + dynamic_indicator + condition_indicator),
# parameter labels, parameter transforms (forward/backward) and the
# ``sharedSigma`` constraint sigma_T_dyn := sigma_dyn.
# ---------------------------------------------------------------------------
class _GaussianAFTargetBase(DynamicAttentionFieldPRF2D_v3):
    """Common Gaussian-backbone v3 + target encoding-only base.

    Adds two new shared parameters at the END of the parent's encoding
    block (slots 11, 12):

    - ``g_T_dyn``     — gain on the per-TR phasic TARGET-onset modulation.
    - ``sigma_T_dyn`` — extent of the target-onset Gaussian. Tied to
      ``sigma_dyn`` in every forward pass (sharedSigma constraint —
      cf. ``DoGDynamicAttentionFieldPRF2DWithHRF_v3_target_sharedSigma``).

    This class is NOT meant to be used directly — it provides only the
    parameter-vector machinery and the target-modulation field. The
    three concrete model classes below override ``_basis_predictions``
    to implement the DRIVE / ANALYTICAL-SHIFT / NUMERICAL-SHIFT
    forward models.
    """

    parameter_labels = [
        'x', 'y', 'sd', 'baseline', 'amplitude',
        'sigma_AF', 'g_HP', 'g_LP',
        'sigma_dyn', 'g_HP_dyn', 'g_LP_dyn',
        'g_T_dyn', 'sigma_T_dyn',
    ]

    # Number of v3 (parent) encoding params.
    _N_V3_ENC_PARS = 11

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
                "Gaussian AF + target model requires a "
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

    # ----- Per-TR effective gain × Gaussian helpers ------------------------

    @tf.function
    def _gaussian_field_per_ring(self, sigma):
        """Peak-normalized Gaussian per ring on the stimulus grid.

        Returns ``A`` of shape ``(n_C, G)`` with
        ``A[ℓ, g] = exp(-||g - r_ℓ||² / (2 σ²))``.
        """
        gx = self._grid_coordinates[:, 0][tf.newaxis, :]   # (1, G)
        gy = self._grid_coordinates[:, 1][tf.newaxis, :]
        rx = self._tf_ring_positions[:, 0][:, tf.newaxis]  # (n_C, 1)
        ry = self._tf_ring_positions[:, 1][:, tf.newaxis]
        diff_sq = (gx - rx) ** 2 + (gy - ry) ** 2          # (n_C, G)
        return tf.exp(-diff_sq / (2.0 * sigma ** 2))

    @tf.function
    def _per_tr_modulation_field(self, parameters):
        """Build ``M(g, t)`` on the stimulus grid (excluding the +1).

        Returns ``mod_tg`` of shape ``(T, G)`` with
        ``mod_tg = sign · ( g_HP · ci · A_AF + g_LP · (1-ci) · A_AF
                           + g_HP_dyn · d · ci · A_dyn
                           + g_LP_dyn · d · (1-ci) · A_dyn
                           + g_T_dyn  · tgt · A_T )``.

        Note this returns the *additive* modulation; callers add 1 to
        get the AF+ multiplicative drive used by Model A.
        """
        # Shared scalars.
        sigma_AF = parameters[0, 0, 5]
        g_HP = parameters[0, 0, 6]
        g_LP = parameters[0, 0, 7]
        sigma_dyn = parameters[0, 0, 8]
        g_HP_dyn = parameters[0, 0, 9]
        g_LP_dyn = parameters[0, 0, 10]
        g_T_dyn = parameters[0, 0, 11]
        # sigma_T_dyn := sigma_dyn (sharedSigma); slot 12 is functionally
        # inert (raw variable still exists in the optimiser's set, but
        # the forward transform overwrites it).
        sigma_T_dyn = sigma_dyn

        # Per-ring Gaussians.
        A_AF = self._gaussian_field_per_ring(sigma_AF)        # (n_C, G)
        A_dyn = self._gaussian_field_per_ring(sigma_dyn)
        A_T = self._gaussian_field_per_ring(sigma_T_dyn)

        ci = self._tf_condition_indicator                     # (T, n_C)
        d = self._tf_dynamic_indicator                        # (T, n_C)
        tgt = self._tf_target_indicator                       # (T, n_C)

        # Sustained: g_HP * ci * A_AF + g_LP * (1-ci) * A_AF.
        sus_hp = tf.einsum('tl,lg->tg', ci, A_AF)              # (T, G)
        sus_lp = tf.einsum('tl,lg->tg', 1.0 - ci, A_AF)
        # Dynamic distractor.
        dyn_hp = tf.einsum('tl,lg->tg', d * ci, A_dyn)
        dyn_lp = tf.einsum('tl,lg->tg', d * (1.0 - ci), A_dyn)
        # Dynamic target.
        tgt_field = tf.einsum('tl,lg->tg', tgt, A_T)

        sign = self._tf_sign
        mod = sign * (
            g_HP * sus_hp + g_LP * sus_lp
            + g_HP_dyn * dyn_hp + g_LP_dyn * dyn_lp
            + g_T_dyn * tgt_field
        )
        return mod                                            # (T, G)

    @tf.function
    def _per_tr_signed_precision_centers(self, parameters):
        """Per-TR per-ring signed precision contribution and centres.

        For the analytical-shift formulation we need, per (t, ℓ):

        - effective signed precision c_ℓ(t) (= gain / σ²), summing the
          three signed-gain contributions:
            * sustained:  (g_HP · ci + g_LP · (1-ci)) / σ_AF²
            * dynamic-d:  d · (g_HP_dyn · ci + g_LP_dyn · (1-ci)) / σ_dyn²
            * target:     tgt · g_T_dyn / σ_T_dyn²
          (Sumiya's strict no-offset model assumes positive gains. Our
          ``mode='signed'`` extension allows c_ℓ(t) to be negative,
          which shifts the COM AWAY from ring ℓ. See the class
          docstring.)
        - the ring centres r_ℓ.

        Returns
        -------
        c : tf.Tensor, shape (T, n_C)
            Signed per-TR per-ring precision contribution.
        rx, ry : tf.Tensor, shape (n_C,)
            Ring centres (cached on the model).
        """
        sigma_AF = parameters[0, 0, 5]
        g_HP = parameters[0, 0, 6]
        g_LP = parameters[0, 0, 7]
        sigma_dyn = parameters[0, 0, 8]
        g_HP_dyn = parameters[0, 0, 9]
        g_LP_dyn = parameters[0, 0, 10]
        g_T_dyn = parameters[0, 0, 11]
        sigma_T_dyn = sigma_dyn  # sharedSigma

        ci = self._tf_condition_indicator                     # (T, n_C)
        d = self._tf_dynamic_indicator                        # (T, n_C)
        tgt = self._tf_target_indicator                       # (T, n_C)

        c_sus = (g_HP * ci + g_LP * (1.0 - ci)) / (sigma_AF ** 2)
        c_dyn = d * (g_HP_dyn * ci + g_LP_dyn * (1.0 - ci)) / (sigma_dyn ** 2)
        c_tgt = tgt * g_T_dyn / (sigma_T_dyn ** 2)

        c = c_sus + c_dyn + c_tgt                              # (T, n_C)

        rx = self._tf_ring_positions[:, 0]                     # (n_C,)
        ry = self._tf_ring_positions[:, 1]
        return c, rx, ry

    # ----- Parameter transforms -------------------------------------------

    @tf.function
    def _transform_parameters_forward(self, parameters):
        """Forward transform.

        Slots 0..10 = parent's v3 encoding pars. Slots 11/12 = NEW.
        ``sigma_T_dyn`` (slot 12) is OVERWRITTEN with ``sigma_dyn``
        (slot 8) — the sharedSigma constraint. ``g_T_dyn`` (slot 11)
        is sign-aware (signed mode passes through; otherwise softplus).
        """
        v3_pars = DynamicAttentionFieldPRF2D_v3._transform_parameters_forward(
            self, parameters[:, :self._N_V3_ENC_PARS])

        if self._signed_gains:
            g_T_dyn = parameters[:, 11][:, tf.newaxis]
        else:
            g_T_dyn = tf.math.softplus(parameters[:, 11][:, tf.newaxis])

        # sigma_T_dyn := sigma_dyn (post-softplus). v3_pars[:, 8] is
        # the post-softplus sigma_dyn.
        sigma_T_dyn = v3_pars[:, 8:9]

        return tf.concat([v3_pars, g_T_dyn, sigma_T_dyn], axis=1)

    @tf.function
    def _transform_parameters_backward(self, parameters):
        v3_pars = DynamicAttentionFieldPRF2D_v3._transform_parameters_backward(
            self, parameters[:, :self._N_V3_ENC_PARS])

        if self._signed_gains:
            g_T_dyn_unb = parameters[:, 11][:, tf.newaxis]
        else:
            g_T_dyn_unb = tfp.math.softplus_inverse(
                parameters[:, 11][:, tf.newaxis])

        # Tie raw σ_T_dyn := raw σ_dyn so the recovered raw vector is
        # internally consistent (both use softplus -> equal raw -> equal
        # post-softplus).
        sigma_T_dyn_raw = v3_pars[:, 8:9]
        return tf.concat([v3_pars, g_T_dyn_unb, sigma_T_dyn_raw], axis=1)


# ---------------------------------------------------------------------------
# Model A: DRIVE — multiplicative AF+ on the paradigm.
# ---------------------------------------------------------------------------
class GaussianAFDriveModel(_GaussianAFTargetBase):
    """Gaussian-PRF + AF+ DRIVE forward model (Sumiya AF+ numerical).

    For each TR ``t`` and voxel ``v``::

        M(g, t) = 1 + sign · ( sustained_HP/LP + dynamic_HP/LP + target )
        prediction(t, v) = ∫ PRF_v(g) · paradigm(g, t) · M(g, t) dg

    The PRF centre is held fixed; spatial modulation acts as a per-grid
    scalar on the stimulus drive. This is the "current" formulation
    used in the rest of retsupp.

    Free parameters (encoding side, 13 slots): see
    :class:`_GaussianAFTargetBase`. ``sigma_T_dyn`` is tied to
    ``sigma_dyn`` (sharedSigma).
    """

    @tf.function
    def _basis_predictions(self, paradigm, parameters):
        # paradigm:    (B, T, G)
        # parameters:  (B, V, 13[ + n_hrf])

        rf = self._get_rf(self.grid_coordinates, parameters)   # (B, V, G)

        mod_tg = self._per_tr_modulation_field(parameters)     # (T, G)
        # AF+: 1 + signed sum of attention contributions; clamp ≥ 0.
        m_full = tf.maximum(1.0 + mod_tg, 0.0)                  # (T, G)

        eff_paradigm = paradigm * m_full[tf.newaxis, :, :]     # (B, T, G)
        result = tf.einsum('btg,bvg->btv', eff_paradigm, rf)   # (B, T, V)

        baseline = parameters[:, tf.newaxis, :, 3]
        return result + baseline


# ---------------------------------------------------------------------------
# Model B: ANALYTICAL SHIFT — precision-weighted COM (Sumiya no-offset).
# ---------------------------------------------------------------------------
class GaussianAFAnalyticalShiftModel(_GaussianAFTargetBase):
    """Analytical precision-weighted-mean shift of the PRF centre.

    For each TR ``t`` and voxel ``v``, build a *per-TR* shifted
    Gaussian centre as the precision-weighted mean of the voxel's PRF
    centre and the four ring-position AF Gaussians, with per-ring
    effective signed precision::

        c_ℓ(t) = (g_HP · ci_ℓ + g_LP · (1-ci_ℓ)) / σ_AF²
               + d_ℓ(t) · (g_HP_dyn · ci_ℓ + g_LP_dyn · (1-ci_ℓ)) / σ_dyn²
               + tgt_ℓ(t) · g_T_dyn / σ_T_dyn²

        τ_v = 1 / σ_PRF²
        τ_total(t) = τ_v + Σ_ℓ c_ℓ(t)
        μ_x(t, v) = ( τ_v · μ_PRF_x_v + Σ_ℓ c_ℓ(t) · r_ℓ_x ) / τ_total(t)
        μ_y(t, v) = ( τ_v · μ_PRF_y_v + Σ_ℓ c_ℓ(t) · r_ℓ_y ) / τ_total(t)

    Then evaluate a per-TR symmetric Gaussian centred at (μ_x, μ_y)
    with the SAME σ_PRF and amplitude as the original PRF, integrate
    against the paradigm.

    For Sumiya's strict positive-gain formulation, c_ℓ ≥ 0. Here
    ``mode='signed'`` allows c_ℓ < 0 (gain<0 pushes COM AWAY from
    ring ℓ), which is the natural extension.

    Numerical safety
    ----------------
    τ_total(t) can vanish or go negative under signed-gain extreme
    cases. We clamp τ_total to a small positive floor before dividing
    so the per-TR Gaussian remains well-defined; in practice the
    optimiser stays well away from this floor at sensible parameter
    values.
    """

    # Floor on total precision to avoid divisions by ~0 in signed mode.
    _TAU_FLOOR = 1e-6

    @tf.function
    def _basis_predictions(self, paradigm, parameters):
        # paradigm:   (B, T, G)
        # parameters: (B, V, 13[+n_hrf])
        B = tf.shape(paradigm)[0]
        T = tf.shape(paradigm)[1]
        V = tf.shape(parameters)[1]

        # Per-voxel PRF: μ_x, μ_y, σ_PRF, amplitude.
        mu_x_v = parameters[:, :, 0]                           # (B, V)
        mu_y_v = parameters[:, :, 1]
        sd_v = parameters[:, :, 2]
        amplitude_v = parameters[:, :, 4]
        baseline_v = parameters[:, :, 3]

        # Per-TR per-ring signed precision contribution (T, n_C) and
        # ring centres (n_C,).
        c_tl, rx, ry = self._per_tr_signed_precision_centers(parameters)
        # c_tl: (T, n_C). Sums to (T,).
        c_sum_t = tf.reduce_sum(c_tl, axis=1)                  # (T,)
        # Σ_ℓ c_ℓ(t) · r_ℓ : (T,).
        c_x_t = tf.einsum('tl,l->t', c_tl, rx)                 # (T,)
        c_y_t = tf.einsum('tl,l->t', c_tl, ry)

        # Per-voxel precision τ_v = 1/σ_PRF².
        tau_v = 1.0 / (sd_v ** 2)                              # (B, V)

        # Broadcast to (B, V, T): tau_v[:, :, None] + c_sum_t[None, None, :].
        tau_total = (
            tau_v[:, :, tf.newaxis]
            + c_sum_t[tf.newaxis, tf.newaxis, :]
        )                                                       # (B, V, T)
        tau_total_safe = tf.maximum(tau_total, self._TAU_FLOOR)

        num_x = (
            tau_v[:, :, tf.newaxis] * mu_x_v[:, :, tf.newaxis]
            + c_x_t[tf.newaxis, tf.newaxis, :]
        )                                                       # (B, V, T)
        num_y = (
            tau_v[:, :, tf.newaxis] * mu_y_v[:, :, tf.newaxis]
            + c_y_t[tf.newaxis, tf.newaxis, :]
        )

        mu_x_tv = num_x / tau_total_safe                       # (B, V, T)
        mu_y_tv = num_y / tau_total_safe

        # Build a per-(B, V, T) Gaussian on the grid: shape (B, V, T, G).
        # We then take the einsum with paradigm to get predictions.
        # Memory: (B, V, T, G). For B=1, V=500, T=2500, G=2500: too big.
        # So compute the integral directly without materialising the
        # full (V, T, G) tensor by writing
        #
        #   pred[b, t, v] = amp_v * Σ_g paradigm[b, t, g]
        #                   · gauss(g; μ_x_tv[b, v, t], μ_y_tv[b, v, t], σ_v)
        #
        # via tf.scan over T (each step holds a (V, G) RF) — but for
        # eager-mode TF a simple Python loop is fine and respects
        # gradient flow.
        gx = self._grid_coordinates[:, 0]                      # (G,)
        gy = self._grid_coordinates[:, 1]
        pixel_area = tf.constant(self.pixel_area, dtype=tf.float32)
        norm_factor = sd_v * tf.sqrt(2.0 * np.pi) / pixel_area  # (B, V)

        def _per_t_step(t_idx):
            mu_x_t = mu_x_tv[:, :, t_idx]                      # (B, V)
            mu_y_t = mu_y_tv[:, :, t_idx]
            # (B, V, G) Gaussian centred at (mu_x_t, mu_y_t) with σ=sd_v.
            dx = gx[tf.newaxis, tf.newaxis, :] - mu_x_t[:, :, tf.newaxis]
            dy = gy[tf.newaxis, tf.newaxis, :] - mu_y_t[:, :, tf.newaxis]
            gauss = tf.exp(
                -(dx * dx + dy * dy)
                / (2.0 * sd_v[:, :, tf.newaxis] ** 2)
            ) * amplitude_v[:, :, tf.newaxis]                  # (B, V, G)
            gauss = gauss / norm_factor[:, :, tf.newaxis]      # (B, V, G)

            par_t = paradigm[:, t_idx, :]                      # (B, G)
            # Σ_g par_t · gauss : (B, V).
            return tf.einsum('bg,bvg->bv', par_t, gauss)

        # tf.map_fn over T -> (T, B, V); transpose to (B, T, V).
        pred_t = tf.map_fn(
            _per_t_step, tf.range(T),
            fn_output_signature=tf.TensorSpec(
                shape=[None, None], dtype=tf.float32),
        )                                                       # (T, B, V)
        pred = tf.transpose(pred_t, perm=[1, 0, 2])            # (B, T, V)

        return pred + baseline_v[:, tf.newaxis, :]


# ---------------------------------------------------------------------------
# Model C: NUMERICAL SHIFT — AF+ then re-fit Gaussian to COM of |eff|.
# ---------------------------------------------------------------------------
class GaussianAFNumericalShiftModel(_GaussianAFTargetBase):
    """AF+ numerical product, then refit a symmetric Gaussian.

    For each TR ``t`` and voxel ``v``::

        M(g, t)            = 1 + sign · attention_modulation
        eff_PRF(g, t, v)   = M(g, t) · PRF_v(g)
        (μ_x, μ_y)(t, v)   = COM of |eff_PRF(g, t, v)| over g
        prediction(t, v)   = amp_v · ∫ paradigm(g, t)
                              · Gauss(g; (μ_x, μ_y)(t, v), σ_PRF_v) dg

    This is the AF+ Sumiya formulation done numerically: a real "1+"
    offset enters via M(g, t), but the *prediction* uses a refitted
    SYMMETRIC Gaussian (centre = COM of |eff|, σ = original σ_PRF,
    amplitude = original PRF amplitude). This mirrors what the
    conditionwise PRF analysis does (a Gaussian is always re-fit to
    the shifted RF).

    Memory
    ------
    A naive (T, V, G) eff_PRF tensor is large. We stream per TR via
    ``tf.map_fn`` over the time axis: each step holds an intermediate
    of shape ``(B, V, G)`` (the per-TR effective PRF and the per-TR
    refitted Gaussian). Gradients still flow correctly.
    """

    # |eff| total mass below this is treated as "no signal at this TR
    # for this voxel" -> fall back to original PRF centre.
    _MASS_FLOOR = 1e-8

    @tf.function
    def _basis_predictions(self, paradigm, parameters):
        # paradigm:   (B, T, G)
        # parameters: (B, V, 13[+n_hrf])
        T = tf.shape(paradigm)[1]

        rf = self._get_rf(self.grid_coordinates, parameters)   # (B, V, G)

        mod_tg = self._per_tr_modulation_field(parameters)     # (T, G)
        # AF+: M(g, t) = 1 + signed sum, clamped ≥ 0.
        m_full = tf.maximum(1.0 + mod_tg, 0.0)                  # (T, G)

        # Per-voxel original PRF centre + size + amplitude.
        mu_x_v = parameters[:, :, 0]                            # (B, V)
        mu_y_v = parameters[:, :, 1]
        sd_v = parameters[:, :, 2]
        amplitude_v = parameters[:, :, 4]
        baseline_v = parameters[:, :, 3]

        gx = self._grid_coordinates[:, 0]                       # (G,)
        gy = self._grid_coordinates[:, 1]
        pixel_area = tf.constant(self.pixel_area, dtype=tf.float32)
        norm_factor = sd_v * tf.sqrt(2.0 * np.pi) / pixel_area  # (B, V)

        def _per_t_step(t_idx):
            m_t = m_full[t_idx, :]                              # (G,)
            # Effective per-voxel RF on grid: (B, V, G).
            eff = rf * m_t[tf.newaxis, tf.newaxis, :]
            # COM of |eff| per voxel.
            abs_eff = tf.abs(eff)                               # (B, V, G)
            mass = tf.reduce_sum(abs_eff, axis=2)               # (B, V)
            mass_safe = tf.maximum(mass, self._MASS_FLOOR)
            com_x = (
                tf.reduce_sum(abs_eff * gx[tf.newaxis, tf.newaxis, :],
                              axis=2) / mass_safe
            )                                                   # (B, V)
            com_y = (
                tf.reduce_sum(abs_eff * gy[tf.newaxis, tf.newaxis, :],
                              axis=2) / mass_safe
            )
            # Where mass is degenerate, fall back to original centre so
            # the Gaussian is still well-defined.
            mass_ok = mass > self._MASS_FLOOR                   # (B, V)
            com_x = tf.where(mass_ok, com_x, mu_x_v)
            com_y = tf.where(mass_ok, com_y, mu_y_v)

            # Refitted symmetric Gaussian at the COM with original σ_v.
            dx = gx[tf.newaxis, tf.newaxis, :] - com_x[:, :, tf.newaxis]
            dy = gy[tf.newaxis, tf.newaxis, :] - com_y[:, :, tf.newaxis]
            gauss = tf.exp(
                -(dx * dx + dy * dy)
                / (2.0 * sd_v[:, :, tf.newaxis] ** 2)
            ) * amplitude_v[:, :, tf.newaxis]                   # (B, V, G)
            gauss = gauss / norm_factor[:, :, tf.newaxis]

            par_t = paradigm[:, t_idx, :]                       # (B, G)
            # Σ_g par_t · gauss : (B, V).
            return tf.einsum('bg,bvg->bv', par_t, gauss)

        pred_t = tf.map_fn(
            _per_t_step, tf.range(T),
            fn_output_signature=tf.TensorSpec(
                shape=[None, None], dtype=tf.float32),
        )                                                        # (T, B, V)
        pred = tf.transpose(pred_t, perm=[1, 0, 2])             # (B, T, V)

        return pred + baseline_v[:, tf.newaxis, :]


# ---------------------------------------------------------------------------
# HRF-convolved wrappers for the three Gaussian-AF models. Mirrors the
# pattern used by ``DoGDynamicAttentionFieldPRF2DWithHRF_v3_target`` and
# friends above. Each wrapper just delegates the parameter transform to
# its encoding-side base via ``_GaussianAFTargetBase`` while letting the
# HRF tail (if flexible) pass through the HRF model's transform.
# ---------------------------------------------------------------------------
class _GaussianAFTargetWithHRFBase:
    """Mixin: parameter transform + HRF tail handling.

    Used by the three concrete WithHRF subclasses below. Each
    inherits from ``HRFEncodingModel`` and one of the three encoding
    classes; this mixin provides the shared
    ``_transform_parameters_forward`` / ``..._backward``.
    """

    @tf.function
    def _transform_parameters_forward(self, parameters):
        if self.flexible_hrf_parameters:
            n_hrf_pars = len(self.hrf_model.parameter_labels)
            encoding_pars = (
                _GaussianAFTargetBase._transform_parameters_forward(
                    self, parameters[:, :-n_hrf_pars])
            )
            hrf_pars = self.hrf_model._transform_parameters_forward(
                parameters[:, -n_hrf_pars:])
            return tf.concat([encoding_pars, hrf_pars], axis=1)
        return _GaussianAFTargetBase._transform_parameters_forward(
            self, parameters)

    @tf.function
    def _transform_parameters_backward(self, parameters):
        if self.flexible_hrf_parameters:
            n_hrf_pars = len(self.hrf_model.parameter_labels)
            encoding_pars = (
                _GaussianAFTargetBase._transform_parameters_backward(
                    self, parameters[:, :-n_hrf_pars])
            )
            hrf_pars = self.hrf_model._transform_parameters_backward(
                parameters[:, -n_hrf_pars:])
            return tf.concat([encoding_pars, hrf_pars], axis=1)
        return _GaussianAFTargetBase._transform_parameters_backward(
            self, parameters)


def _gaussian_af_with_hrf_init(self, *, grid_coordinates, paradigm, data,
                               parameters, condition_indicator,
                               dynamic_indicator, target_indicator,
                               ring_positions, mode,
                               positive_image_values_only,
                               weights, hrf_model,
                               flexible_hrf_parameters,
                               verbosity, **kwargs):
    """Shared __init__ for the three Gaussian-AF WithHRF subclasses."""
    # Encoding side: routes through whichever concrete encoding class
    # this subclass inherits from (DRIVE / ANALYTICAL / NUMERICAL),
    # all of which extend _GaussianAFTargetBase.
    # We intentionally call the encoding base directly (not super())
    # so the MRO with HRFEncodingModel doesn't pick up the wrong
    # constructor.
    # Find the concrete encoding base in the MRO.
    encoding_cls = None
    for cls in type(self).__mro__:
        if (cls is not type(self)
                and issubclass(cls, _GaussianAFTargetBase)
                and cls is not _GaussianAFTargetBase):
            encoding_cls = cls
            break
    if encoding_cls is None:
        raise RuntimeError(
            f"Could not locate a concrete _GaussianAFTargetBase "
            f"subclass in MRO of {type(self).__name__}.")
    encoding_cls.__init__(
        self, grid_coordinates=grid_coordinates, paradigm=paradigm,
        data=data, parameters=parameters,
        condition_indicator=condition_indicator,
        dynamic_indicator=dynamic_indicator,
        target_indicator=target_indicator,
        ring_positions=ring_positions, mode=mode,
        weights=weights, verbosity=verbosity,
        positive_image_values_only=positive_image_values_only, **kwargs)
    HRFEncodingModel.__init__(
        self, hrf_model=hrf_model,
        flexible_hrf_parameters=flexible_hrf_parameters, **kwargs)


class GaussianAFDriveModelWithHRF(
    _GaussianAFTargetWithHRFBase,
    HRFEncodingModel,
    GaussianAFDriveModel,
):
    """HRF-convolved version of :class:`GaussianAFDriveModel`."""

    def __init__(self, grid_coordinates=None, paradigm=None, data=None,
                 parameters=None, condition_indicator=None,
                 dynamic_indicator=None,
                 target_indicator=None,
                 ring_positions=None, mode='suppression',
                 positive_image_values_only=True,
                 weights=None, hrf_model=None,
                 flexible_hrf_parameters=False,
                 verbosity=logging.INFO, **kwargs):
        _gaussian_af_with_hrf_init(
            self, grid_coordinates=grid_coordinates, paradigm=paradigm,
            data=data, parameters=parameters,
            condition_indicator=condition_indicator,
            dynamic_indicator=dynamic_indicator,
            target_indicator=target_indicator,
            ring_positions=ring_positions, mode=mode,
            positive_image_values_only=positive_image_values_only,
            weights=weights, hrf_model=hrf_model,
            flexible_hrf_parameters=flexible_hrf_parameters,
            verbosity=verbosity, **kwargs)


class GaussianAFAnalyticalShiftModelWithHRF(
    _GaussianAFTargetWithHRFBase,
    HRFEncodingModel,
    GaussianAFAnalyticalShiftModel,
):
    """HRF-convolved version of :class:`GaussianAFAnalyticalShiftModel`."""

    def __init__(self, grid_coordinates=None, paradigm=None, data=None,
                 parameters=None, condition_indicator=None,
                 dynamic_indicator=None,
                 target_indicator=None,
                 ring_positions=None, mode='suppression',
                 positive_image_values_only=True,
                 weights=None, hrf_model=None,
                 flexible_hrf_parameters=False,
                 verbosity=logging.INFO, **kwargs):
        _gaussian_af_with_hrf_init(
            self, grid_coordinates=grid_coordinates, paradigm=paradigm,
            data=data, parameters=parameters,
            condition_indicator=condition_indicator,
            dynamic_indicator=dynamic_indicator,
            target_indicator=target_indicator,
            ring_positions=ring_positions, mode=mode,
            positive_image_values_only=positive_image_values_only,
            weights=weights, hrf_model=hrf_model,
            flexible_hrf_parameters=flexible_hrf_parameters,
            verbosity=verbosity, **kwargs)


class GaussianAFNumericalShiftModelWithHRF(
    _GaussianAFTargetWithHRFBase,
    HRFEncodingModel,
    GaussianAFNumericalShiftModel,
):
    """HRF-convolved version of :class:`GaussianAFNumericalShiftModel`."""

    def __init__(self, grid_coordinates=None, paradigm=None, data=None,
                 parameters=None, condition_indicator=None,
                 dynamic_indicator=None,
                 target_indicator=None,
                 ring_positions=None, mode='suppression',
                 positive_image_values_only=True,
                 weights=None, hrf_model=None,
                 flexible_hrf_parameters=False,
                 verbosity=logging.INFO, **kwargs):
        _gaussian_af_with_hrf_init(
            self, grid_coordinates=grid_coordinates, paradigm=paradigm,
            data=data, parameters=parameters,
            condition_indicator=condition_indicator,
            dynamic_indicator=dynamic_indicator,
            target_indicator=target_indicator,
            ring_positions=ring_positions, mode=mode,
            positive_image_values_only=positive_image_values_only,
            weights=weights, hrf_model=hrf_model,
            flexible_hrf_parameters=flexible_hrf_parameters,
            verbosity=verbosity, **kwargs)
