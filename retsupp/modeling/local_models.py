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
    DivisiveNormalizationGaussianPRF2D,
    DivisiveNormalizationGaussianPRF2DWithHRF,
    DoGDynamicAttentionFieldPRF2D_v3,
    DoGDynamicAttentionFieldPRF2DWithHRF_v3,
    DynamicAttentionFieldPRF2D_v3,
    DynamicAttentionFieldPRF2DWithHRF_v3,
    HRFEncodingModel,
    _sd_softplus_forward,
    _sd_softplus_inverse,
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
        passes through; otherwise softplus); sigma_T_dyn uses the
        shifted-softplus controlled by ``self.sd_min`` (matches sigma_AF
        / sigma_dyn).
        """
        v3_pars = DoGDynamicAttentionFieldPRF2D_v3._transform_parameters_forward(
            self, parameters[:, :self._N_V3_ENC_PARS])

        if self._signed_gains:
            g_T_dyn = parameters[:, 13][:, tf.newaxis]
        else:
            g_T_dyn = tf.math.softplus(parameters[:, 13][:, tf.newaxis])
        sigma_T_dyn = _sd_softplus_forward(
            parameters[:, 14][:, tf.newaxis], self.sd_min)

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
        sigma_T_dyn_unb = _sd_softplus_inverse(
            parameters[:, 14][:, tf.newaxis], self.sd_min)

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
# Stricter-tied variants of the shared-σ + target DoG model.
# ---------------------------------------------------------------------------

class DoGDynamicAttentionFieldPRF2DWithHRF_v3_target_sharedSigma_sharedDynGain(
    DoGDynamicAttentionFieldPRF2DWithHRF_v3_target_sharedSigma,
):
    """sharedSigma v3+target with a SINGLE dynamic-distractor gain.

    On top of the parent's σ_T_dyn := σ_dyn tying, this also ties
    ``g_LP_dyn := g_HP_dyn`` (slot 12 := slot 11). Tests whether the
    distractor-onset transient is meaningfully different at HP vs LP
    locations, or whether one gain suffices.

    Initialisation
    --------------
    Callers should set ``init_pars['g_LP_dyn'] = init_pars['g_HP_dyn']``
    before passing inits to the fitter. The fit script handles this
    when ``--shared-dyn-gain`` is set.
    """

    @tf.function
    def _transform_parameters_forward(self, parameters):
        out = super()._transform_parameters_forward(parameters)
        # Tie g_LP_dyn (slot 12) := g_HP_dyn (slot 11).
        g_hp_dyn_col = out[:, 11:12]
        before = out[:, :12]
        after = out[:, 13:]
        return tf.concat([before, g_hp_dyn_col, after], axis=1)

    @tf.function
    def _transform_parameters_backward(self, parameters):
        out = super()._transform_parameters_backward(parameters)
        raw_g_hp_dyn_col = out[:, 11:12]
        before = out[:, :12]
        after = out[:, 13:]
        return tf.concat([before, raw_g_hp_dyn_col, after], axis=1)


class DoGDynamicAttentionFieldPRF2DWithHRF_v3_target_allSharedSigma(
    DoGDynamicAttentionFieldPRF2DWithHRF_v3_target_sharedSigma,
):
    """v3+target with ALL three attention-Gaussian widths tied.

    Strictest σ-constraint: ties ``σ_AF`` (slot 7) := ``σ_dyn`` (slot 10)
    on top of the parent's already-existing ``σ_T_dyn := σ_dyn``. One
    shared width parameter (``σ_dyn``) controls the sustained, phasic-
    distractor, and phasic-target Gaussians.

    Initialisation
    --------------
    Callers should set
    ``init_pars['sigma_AF'] = init_pars['sigma_T_dyn'] = init_pars['sigma_dyn']``
    before passing inits to the fitter. The fit script handles this
    when ``--all-shared-sigma`` is set.
    """

    @tf.function
    def _transform_parameters_forward(self, parameters):
        # Parent already ties σ_T_dyn (slot 14) := σ_dyn (slot 10).
        out = super()._transform_parameters_forward(parameters)
        # Additionally tie σ_AF (slot 7) := σ_dyn (slot 10).
        sigma_dyn_col = out[:, 10:11]
        before = out[:, :7]
        after = out[:, 8:]
        return tf.concat([before, sigma_dyn_col, after], axis=1)

    @tf.function
    def _transform_parameters_backward(self, parameters):
        out = super()._transform_parameters_backward(parameters)
        raw_sigma_dyn_col = out[:, 10:11]
        before = out[:, :7]
        after = out[:, 8:]
        return tf.concat([before, raw_sigma_dyn_col, after], axis=1)


class DoGDynamicAttentionFieldPRF2DWithHRF_v3_target_allSharedSigma_sharedDynGain(
    DoGDynamicAttentionFieldPRF2DWithHRF_v3_target_allSharedSigma,
):
    """All-σ-tied AND single-dyn-gain variant.

    Inherits the ``σ_AF := σ_T_dyn := σ_dyn`` tying from
    :class:`...allSharedSigma`, AND additionally ties
    ``g_LP_dyn := g_HP_dyn``. The most-restricted of the four-variant
    family.
    """

    @tf.function
    def _transform_parameters_forward(self, parameters):
        out = super()._transform_parameters_forward(parameters)
        g_hp_dyn_col = out[:, 11:12]
        before = out[:, :12]
        after = out[:, 13:]
        return tf.concat([before, g_hp_dyn_col, after], axis=1)

    @tf.function
    def _transform_parameters_backward(self, parameters):
        out = super()._transform_parameters_backward(parameters)
        raw_g_hp_dyn_col = out[:, 11:12]
        before = out[:, :12]
        after = out[:, 13:]
        return tf.concat([before, raw_g_hp_dyn_col, after], axis=1)


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
        sd = _sd_softplus_forward(enc[:, 2:3], self.sd_min)
        baseline = enc[:, 3:4]
        amplitude = enc[:, 4:5]
        srf_amp = tf.math.softplus(enc[:, 5:6])
        srf_size = _sd_softplus_forward(enc[:, 6:7], self.sd_min)
        sigma_AF = _sd_softplus_forward(enc[:, 7:8], self.sd_min)
        g_HP = _gain_forward_factorial(enc[:, 8:9],
                                       self._sign_pattern['g_HP'])
        g_LP = _gain_forward_factorial(enc[:, 9:10],
                                       self._sign_pattern['g_LP'])
        sigma_dyn = _sd_softplus_forward(enc[:, 10:11], self.sd_min)
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
        sd = _sd_softplus_inverse(enc[:, 2:3], self.sd_min)
        baseline = enc[:, 3:4]
        amplitude = enc[:, 4:5]
        srf_amp = tfp.math.softplus_inverse(enc[:, 5:6])
        srf_size = _sd_softplus_inverse(enc[:, 6:7], self.sd_min)
        sigma_AF = _sd_softplus_inverse(enc[:, 7:8], self.sd_min)
        g_HP = _gain_backward_factorial(enc[:, 8:9],
                                        self._sign_pattern['g_HP'])
        g_LP = _gain_backward_factorial(enc[:, 9:10],
                                        self._sign_pattern['g_LP'])
        sigma_dyn = _sd_softplus_inverse(enc[:, 10:11], self.sd_min)
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
        sd = _sd_softplus_forward(enc[:, 2:3], self.sd_min)
        baseline = enc[:, 3:4]
        amplitude = enc[:, 4:5]
        srf_amp = tf.math.softplus(enc[:, 5:6])
        srf_size = _sd_softplus_forward(enc[:, 6:7], self.sd_min)
        sigma_AF = _sd_softplus_forward(enc[:, 7:8], self.sd_min)
        g_HP = _gain_forward_factorial(enc[:, 8:9],
                                       self._sign_pattern['g_HP'])
        g_LP = _gain_forward_factorial(enc[:, 9:10],
                                       self._sign_pattern['g_LP'])
        sigma_dyn = _sd_softplus_forward(enc[:, 10:11], self.sd_min)
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
        sd = _sd_softplus_inverse(enc[:, 2:3], self.sd_min)
        baseline = enc[:, 3:4]
        amplitude = enc[:, 4:5]
        srf_amp = tfp.math.softplus_inverse(enc[:, 5:6])
        srf_size = _sd_softplus_inverse(enc[:, 6:7], self.sd_min)
        sigma_AF = _sd_softplus_inverse(enc[:, 7:8], self.sd_min)
        g_HP = _gain_backward_factorial(enc[:, 8:9],
                                        self._sign_pattern['g_HP'])
        g_LP = _gain_backward_factorial(enc[:, 9:10],
                                        self._sign_pattern['g_LP'])
        sigma_dyn = _sd_softplus_inverse(enc[:, 10:11], self.sd_min)
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


# ---------------------------------------------------------------------------
# Repeat-distractor variant of the sharedSigma factorial model.
# Splits the dynamic gains by whether THIS trial's distractor was at the
# same ring location as the PREVIOUS trial's distractor (repeat) or not
# (switch). g_T_dyn is unchanged. The "switch" gains live in the parent's
# slots 11/12 (the original g_HP_dyn / g_LP_dyn); the new "repeat" gains
# are appended at slots 15/16. Sustained gains and signs unchanged.
# ---------------------------------------------------------------------------
class DoGDynamicAttentionFieldPRF2DWithHRF_v3_target_sharedSigma_repeat(
    DoGDynamicAttentionFieldPRF2DWithHRF_v3_target_sharedSigma_factorial,
):
    """sharedSigma + factorial with split repeat/switch dynamic gains.

    The dynamic-distractor pulse is split into two TR-level indicators:

    - ``switch_indicator(t, ℓ) = dynamic_indicator(t, ℓ)
                              − repeat_indicator(t, ℓ)``
        — the dynamic-on-fraction at ring ℓ for trials whose distractor
        was *not* at the same ring as the immediately preceding trial.
    - ``repeat_indicator(t, ℓ)``
        — the dynamic-on-fraction at ring ℓ for trials whose distractor
        WAS at the same ring as the previous trial. Provided by the
        caller via :meth:`Subject.get_repeat_distractor_indicator`.

    The forward dynamic-AF field becomes::

        field_dyn(t, g) =
            g_HP_dyn_switch · Σ_ℓ switch[t,ℓ] · is_hp[t,ℓ] · A_ℓ^dyn(g)
          + g_LP_dyn_switch · Σ_ℓ switch[t,ℓ] · (1−is_hp[t,ℓ]) · A_ℓ^dyn(g)
          + g_HP_dyn_repeat · Σ_ℓ repeat[t,ℓ] · is_hp[t,ℓ] · A_ℓ^dyn(g)
          + g_LP_dyn_repeat · Σ_ℓ repeat[t,ℓ] · (1−is_hp[t,ℓ]) · A_ℓ^dyn(g)

    where ``A_ℓ^dyn`` uses ``sigma_dyn`` (slot 10), as in the parent.
    For backward compatibility with the parent class's slot indexing
    the ORIGINAL ``g_HP_dyn`` (slot 11) and ``g_LP_dyn`` (slot 12)
    are now the **switch** gains; the two new **repeat** gains live at
    slots 15 and 16.

    Encoding parameter vector (17 slots before optional HRF tail)::

        [parent's 15 slots ...]
        15  g_HP_dyn_repeat   (signed-free)
        16  g_LP_dyn_repeat   (signed-free)

    The parent's σ_T_dyn := σ_dyn tying is preserved. The parent's
    factorial sign-pattern still applies to ``g_HP``, ``g_LP``,
    ``g_HP_dyn`` (= switch), ``g_LP_dyn`` (= switch), ``g_T_dyn``;
    the two new repeat gains are always signed-free.
    """

    parameter_labels = [
        'x', 'y', 'sd', 'baseline', 'amplitude',
        'srf_amplitude', 'srf_size',
        'sigma_AF', 'g_HP', 'g_LP',
        'sigma_dyn', 'g_HP_dyn', 'g_LP_dyn',
        'g_T_dyn', 'sigma_T_dyn',
        'g_HP_dyn_repeat', 'g_LP_dyn_repeat',
    ]

    _N_PARENT_ENC_PARS = 15
    _N_REPEAT_PARS = 2

    def __init__(self, *, repeat_indicator, sign_pattern=None, **kwargs):
        if sign_pattern is None:
            sign_pattern = {
                'g_HP': 'free', 'g_LP': 'free',
                'g_HP_dyn': 'free', 'g_LP_dyn': 'free',
                'g_T_dyn': 'free',
            }
        super().__init__(sign_pattern=sign_pattern, **kwargs)

        rp = np.asarray(repeat_indicator, dtype=np.float32)
        if rp.ndim != 2:
            raise ValueError(
                f"repeat_indicator must be 2-D (T, n_ring_positions); "
                f"got shape {rp.shape}.")
        if rp.shape[1] != int(self._tf_dynamic_indicator.shape[1]):
            raise ValueError(
                f"repeat_indicator has {rp.shape[1]} ring channels but "
                f"dynamic_indicator has "
                f"{int(self._tf_dynamic_indicator.shape[1])}.")
        T_expected = int(self._tf_dynamic_indicator.shape[0])
        if rp.shape[0] != T_expected:
            raise ValueError(
                f"repeat_indicator has {rp.shape[0]} timepoints but "
                f"dynamic_indicator has {T_expected}.")
        self.repeat_indicator = rp
        self._tf_repeat_indicator = tf.constant(rp, dtype=tf.float32)
        # Switch indicator = dynamic - repeat (per-element subtraction).
        # Both have on-fractions in [0, 1]; their per-element sum equals
        # the dynamic_indicator, so the resulting switch_indicator is in
        # [0, 1] as well.
        self._tf_switch_indicator = (self._tf_dynamic_indicator
                                     - self._tf_repeat_indicator)

    # ----- Parameter transforms -------------------------------------------

    @tf.function
    def _transform_parameters_forward(self, parameters):
        if self.flexible_hrf_parameters:
            n_hrf = len(self.hrf_model.parameter_labels)
            enc = parameters[:, :self._N_PARENT_ENC_PARS]
            tail_extra = parameters[
                :, self._N_PARENT_ENC_PARS:
                self._N_PARENT_ENC_PARS + self._N_REPEAT_PARS]
            hrf_tail = parameters[:, -n_hrf:]
        else:
            enc = parameters[:, :self._N_PARENT_ENC_PARS]
            tail_extra = parameters[
                :, self._N_PARENT_ENC_PARS:
                self._N_PARENT_ENC_PARS + self._N_REPEAT_PARS]
            hrf_tail = None

        # Replicate parent's encoding-side transform inline.
        x = enc[:, 0:1]
        y = enc[:, 1:2]
        sd = _sd_softplus_forward(enc[:, 2:3], self.sd_min)
        baseline = enc[:, 3:4]
        amplitude = enc[:, 4:5]
        srf_amp = tf.math.softplus(enc[:, 5:6])
        srf_size = _sd_softplus_forward(enc[:, 6:7], self.sd_min)
        sigma_AF = _sd_softplus_forward(enc[:, 7:8], self.sd_min)
        g_HP = _gain_forward_factorial(enc[:, 8:9],
                                       self._sign_pattern['g_HP'])
        g_LP = _gain_forward_factorial(enc[:, 9:10],
                                       self._sign_pattern['g_LP'])
        sigma_dyn = _sd_softplus_forward(enc[:, 10:11], self.sd_min)
        g_HP_dyn = _gain_forward_factorial(enc[:, 11:12],
                                           self._sign_pattern['g_HP_dyn'])
        g_LP_dyn = _gain_forward_factorial(enc[:, 12:13],
                                           self._sign_pattern['g_LP_dyn'])
        g_T_dyn = _gain_forward_factorial(enc[:, 13:14],
                                          self._sign_pattern['g_T_dyn'])
        sigma_T_dyn = sigma_dyn

        out_enc = tf.concat([
            x, y, sd, baseline, amplitude, srf_amp, srf_size,
            sigma_AF, g_HP, g_LP,
            sigma_dyn, g_HP_dyn, g_LP_dyn,
            g_T_dyn, sigma_T_dyn,
            tail_extra,  # 2 new gains: signed-free pass-through.
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
            tail_extra = parameters[
                :, self._N_PARENT_ENC_PARS:
                self._N_PARENT_ENC_PARS + self._N_REPEAT_PARS]
            hrf_tail = parameters[:, -n_hrf:]
        else:
            enc = parameters[:, :self._N_PARENT_ENC_PARS]
            tail_extra = parameters[
                :, self._N_PARENT_ENC_PARS:
                self._N_PARENT_ENC_PARS + self._N_REPEAT_PARS]
            hrf_tail = None

        x = enc[:, 0:1]
        y = enc[:, 1:2]
        sd = _sd_softplus_inverse(enc[:, 2:3], self.sd_min)
        baseline = enc[:, 3:4]
        amplitude = enc[:, 4:5]
        srf_amp = tfp.math.softplus_inverse(enc[:, 5:6])
        srf_size = _sd_softplus_inverse(enc[:, 6:7], self.sd_min)
        sigma_AF = _sd_softplus_inverse(enc[:, 7:8], self.sd_min)
        g_HP = _gain_backward_factorial(enc[:, 8:9],
                                        self._sign_pattern['g_HP'])
        g_LP = _gain_backward_factorial(enc[:, 9:10],
                                        self._sign_pattern['g_LP'])
        sigma_dyn = _sd_softplus_inverse(enc[:, 10:11], self.sd_min)
        g_HP_dyn = _gain_backward_factorial(enc[:, 11:12],
                                            self._sign_pattern['g_HP_dyn'])
        g_LP_dyn = _gain_backward_factorial(enc[:, 12:13],
                                            self._sign_pattern['g_LP_dyn'])
        g_T_dyn = _gain_backward_factorial(enc[:, 13:14],
                                           self._sign_pattern['g_T_dyn'])
        sigma_T_dyn = sigma_dyn

        out_enc = tf.concat([
            x, y, sd, baseline, amplitude, srf_amp, srf_size,
            sigma_AF, g_HP, g_LP,
            sigma_dyn, g_HP_dyn, g_LP_dyn,
            g_T_dyn, sigma_T_dyn,
            tail_extra,
        ], axis=1)

        if hrf_tail is not None:
            hrf_pars = self.hrf_model._transform_parameters_backward(hrf_tail)
            return tf.concat([out_enc, hrf_pars], axis=1)
        return out_enc

    # ----- Dynamic modulation override ------------------------------------

    @tf.function
    def _attention_modulation_dynamic_v3(self, parameters):
        """Per-TR dynamic-AF field, split by repeat vs switch trials.

        Returns
        -------
        mod_dyn : tf.Tensor, shape (T, G)

            g_HP_dyn_switch · Σ_ℓ switch[t,ℓ] · is_hp[t,ℓ] · A_ℓ^dyn(g)
          + g_LP_dyn_switch · Σ_ℓ switch[t,ℓ] · (1−is_hp[t,ℓ]) · A_ℓ^dyn(g)
          + g_HP_dyn_repeat · Σ_ℓ repeat[t,ℓ] · is_hp[t,ℓ] · A_ℓ^dyn(g)
          + g_LP_dyn_repeat · Σ_ℓ repeat[t,ℓ] · (1−is_hp[t,ℓ]) · A_ℓ^dyn(g)
        """
        # σ_dyn at slot 10. Switch gains at 11/12 (parent's g_HP_dyn /
        # g_LP_dyn); repeat gains at 15/16.
        sigma_dyn = parameters[0, 0, 10]
        g_HP_dyn_switch = parameters[0, 0, 11]
        g_LP_dyn_switch = parameters[0, 0, 12]
        g_HP_dyn_repeat = parameters[0, 0, 15]
        g_LP_dyn_repeat = parameters[0, 0, 16]

        gx = self._grid_coordinates[:, 0][tf.newaxis, :]
        gy = self._grid_coordinates[:, 1][tf.newaxis, :]
        rx = self._tf_ring_positions[:, 0][:, tf.newaxis]
        ry = self._tf_ring_positions[:, 1][:, tf.newaxis]
        diff_sq = (gx - rx) ** 2 + (gy - ry) ** 2
        A_dyn = tf.exp(-diff_sq / (2.0 * sigma_dyn ** 2))      # (n_C, G)

        is_hp = self._tf_condition_indicator                   # (T, n_C)
        d_sw = self._tf_switch_indicator                       # (T, n_C)
        d_rp = self._tf_repeat_indicator                       # (T, n_C)

        w_hp_sw = d_sw * is_hp
        w_lp_sw = d_sw * (1.0 - is_hp)
        w_hp_rp = d_rp * is_hp
        w_lp_rp = d_rp * (1.0 - is_hp)

        field_hp_sw = tf.einsum('tl,lg->tg', w_hp_sw, A_dyn)
        field_lp_sw = tf.einsum('tl,lg->tg', w_lp_sw, A_dyn)
        field_hp_rp = tf.einsum('tl,lg->tg', w_hp_rp, A_dyn)
        field_lp_rp = tf.einsum('tl,lg->tg', w_lp_rp, A_dyn)

        return (g_HP_dyn_switch * field_hp_sw
                + g_LP_dyn_switch * field_lp_sw
                + g_HP_dyn_repeat * field_hp_rp
                + g_LP_dyn_repeat * field_lp_rp)


# ---------------------------------------------------------------------------
# Per-run-position dyn-HP variant of the run-position model. Extends the
# runPosition class by ALSO splitting g_HP_dyn into 3 per-position gains.
# Sustained gains (already split: g_HP_pos0/1/2, g_LP_pos0/1/2) plus dyn LP
# (single g_LP_dyn) and dyn target (single g_T_dyn) stay as-is. 11 free
# gain params total: 6 sustained + 3 dyn HP + 1 dyn LP + 1 dyn T.
# ---------------------------------------------------------------------------
class DoGDynamicAttentionFieldPRF2DWithHRF_v3_target_sharedSigma_runPosition_dynHP(
    DoGDynamicAttentionFieldPRF2DWithHRF_v3_target_sharedSigma_runPosition,
):
    """runPosition + per-run-position dyn-HP gain.

    Same as the parent
    :class:`DoGDynamicAttentionFieldPRF2DWithHRF_v3_target_sharedSigma_runPosition`,
    but the single phasic ``g_HP_dyn`` (slot 11) is replaced by three
    new shared parameters one per chronological within-block run-position::

        g_HP_dyn_pos0, g_HP_dyn_pos1, g_HP_dyn_pos2

    For TR ``t`` the effective phasic-HP gain is::

        g_HP_dyn_eff[t] = Σ_r runpos[t, r] · g_HP_dyn_pos[r]

    where ``runpos`` is the (T, 3) one-hot indicator inherited from the
    parent. The original ``g_HP_dyn`` slot (11) is kept (for slot
    consistency with the parent) but ZEROED in the forward transform so
    that any inherited code path reads zero gain.

    Encoding parameter vector (24 slots before optional HRF tail)::

        [parent's 21 slots: 15 enc + 6 runPosition gains]
        21  g_HP_dyn_pos0   (signed-free)
        22  g_HP_dyn_pos1   (signed-free)
        23  g_HP_dyn_pos2   (signed-free)
    """

    parameter_labels = [
        'x', 'y', 'sd', 'baseline', 'amplitude',
        'srf_amplitude', 'srf_size',
        'sigma_AF', 'g_HP', 'g_LP',
        'sigma_dyn', 'g_HP_dyn', 'g_LP_dyn',
        'g_T_dyn', 'sigma_T_dyn',
        'g_HP_pos0', 'g_HP_pos1', 'g_HP_pos2',
        'g_LP_pos0', 'g_LP_pos1', 'g_LP_pos2',
        'g_HP_dyn_pos0', 'g_HP_dyn_pos1', 'g_HP_dyn_pos2',
    ]

    # Total slots inherited from parent's encoding portion (15 + 6 runpos).
    _N_PARENT_RUNPOS_ENC_PARS = 21
    _N_DYN_HP_POS_PARS = 3

    @tf.function
    def _transform_parameters_forward(self, parameters):
        if self.flexible_hrf_parameters:
            n_hrf = len(self.hrf_model.parameter_labels)
            enc = parameters[:, :self._N_PARENT_ENC_PARS]
            runpos = parameters[
                :, self._N_PARENT_ENC_PARS:
                self._N_PARENT_ENC_PARS + self._N_RUNPOS_PARS]
            dyn_hp_pos = parameters[
                :, self._N_PARENT_RUNPOS_ENC_PARS:
                self._N_PARENT_RUNPOS_ENC_PARS + self._N_DYN_HP_POS_PARS]
            hrf_tail = parameters[:, -n_hrf:]
        else:
            enc = parameters[:, :self._N_PARENT_ENC_PARS]
            runpos = parameters[
                :, self._N_PARENT_ENC_PARS:
                self._N_PARENT_ENC_PARS + self._N_RUNPOS_PARS]
            dyn_hp_pos = parameters[
                :, self._N_PARENT_RUNPOS_ENC_PARS:
                self._N_PARENT_RUNPOS_ENC_PARS + self._N_DYN_HP_POS_PARS]
            hrf_tail = None

        # Replicate factorial-parent transform for the 15 enc slots.
        x = enc[:, 0:1]
        y = enc[:, 1:2]
        sd = _sd_softplus_forward(enc[:, 2:3], self.sd_min)
        baseline = enc[:, 3:4]
        amplitude = enc[:, 4:5]
        srf_amp = tf.math.softplus(enc[:, 5:6])
        srf_size = _sd_softplus_forward(enc[:, 6:7], self.sd_min)
        sigma_AF = _sd_softplus_forward(enc[:, 7:8], self.sd_min)
        g_HP = _gain_forward_factorial(enc[:, 8:9],
                                       self._sign_pattern['g_HP'])
        g_LP = _gain_forward_factorial(enc[:, 9:10],
                                       self._sign_pattern['g_LP'])
        sigma_dyn = _sd_softplus_forward(enc[:, 10:11], self.sd_min)
        # Force the legacy g_HP_dyn slot to 0; the per-position gains
        # do all the work.
        g_HP_dyn = tf.zeros_like(enc[:, 11:12])
        g_LP_dyn = _gain_forward_factorial(enc[:, 12:13],
                                           self._sign_pattern['g_LP_dyn'])
        g_T_dyn = _gain_forward_factorial(enc[:, 13:14],
                                          self._sign_pattern['g_T_dyn'])
        sigma_T_dyn = sigma_dyn

        out_enc = tf.concat([
            x, y, sd, baseline, amplitude, srf_amp, srf_size,
            sigma_AF, g_HP, g_LP,
            sigma_dyn, g_HP_dyn, g_LP_dyn,
            g_T_dyn, sigma_T_dyn,
            runpos, dyn_hp_pos,
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
            dyn_hp_pos = parameters[
                :, self._N_PARENT_RUNPOS_ENC_PARS:
                self._N_PARENT_RUNPOS_ENC_PARS + self._N_DYN_HP_POS_PARS]
            hrf_tail = parameters[:, -n_hrf:]
        else:
            enc = parameters[:, :self._N_PARENT_ENC_PARS]
            runpos = parameters[
                :, self._N_PARENT_ENC_PARS:
                self._N_PARENT_ENC_PARS + self._N_RUNPOS_PARS]
            dyn_hp_pos = parameters[
                :, self._N_PARENT_RUNPOS_ENC_PARS:
                self._N_PARENT_RUNPOS_ENC_PARS + self._N_DYN_HP_POS_PARS]
            hrf_tail = None

        x = enc[:, 0:1]
        y = enc[:, 1:2]
        sd = _sd_softplus_inverse(enc[:, 2:3], self.sd_min)
        baseline = enc[:, 3:4]
        amplitude = enc[:, 4:5]
        srf_amp = tfp.math.softplus_inverse(enc[:, 5:6])
        srf_size = _sd_softplus_inverse(enc[:, 6:7], self.sd_min)
        sigma_AF = _sd_softplus_inverse(enc[:, 7:8], self.sd_min)
        g_HP = _gain_backward_factorial(enc[:, 8:9],
                                        self._sign_pattern['g_HP'])
        g_LP = _gain_backward_factorial(enc[:, 9:10],
                                        self._sign_pattern['g_LP'])
        sigma_dyn = _sd_softplus_inverse(enc[:, 10:11], self.sd_min)
        # Legacy slot — kept at 0 in forward; map to 0 in backward too.
        g_HP_dyn = tf.zeros_like(enc[:, 11:12])
        g_LP_dyn = _gain_backward_factorial(enc[:, 12:13],
                                            self._sign_pattern['g_LP_dyn'])
        g_T_dyn = _gain_backward_factorial(enc[:, 13:14],
                                           self._sign_pattern['g_T_dyn'])
        sigma_T_dyn = sigma_dyn

        out_enc = tf.concat([
            x, y, sd, baseline, amplitude, srf_amp, srf_size,
            sigma_AF, g_HP, g_LP,
            sigma_dyn, g_HP_dyn, g_LP_dyn,
            g_T_dyn, sigma_T_dyn,
            runpos, dyn_hp_pos,
        ], axis=1)

        if hrf_tail is not None:
            hrf_pars = self.hrf_model._transform_parameters_backward(hrf_tail)
            return tf.concat([out_enc, hrf_pars], axis=1)
        return out_enc

    # ----- Dynamic modulation override ------------------------------------

    @tf.function
    def _attention_modulation_dynamic_v3(self, parameters):
        """Per-TR dynamic-AF field with per-run-position g_HP_dyn.

        Mirrors the v3 parent's dynamic field, except ``g_HP_dyn``
        becomes a per-TR effective gain computed from the 3 new gains
        and the run-position one-hot indicator.
        """
        sigma_dyn = parameters[0, 0, 10]
        # Slot 11 is forced to 0 in the forward transform; we ignore it
        # entirely and read the per-position gains instead.
        g_HP_dyn_pos = parameters[0, 0, 21:24]                # (3,)
        g_LP_dyn = parameters[0, 0, 12]                        # scalar

        runpos = self._tf_run_position_indicator               # (T, 3)
        g_HP_dyn_eff = tf.linalg.matvec(runpos, g_HP_dyn_pos)  # (T,)

        gx = self._grid_coordinates[:, 0][tf.newaxis, :]
        gy = self._grid_coordinates[:, 1][tf.newaxis, :]
        rx = self._tf_ring_positions[:, 0][:, tf.newaxis]
        ry = self._tf_ring_positions[:, 1][:, tf.newaxis]
        diff_sq = (gx - rx) ** 2 + (gy - ry) ** 2
        A_dyn = tf.exp(-diff_sq / (2.0 * sigma_dyn ** 2))      # (n_C, G)

        is_hp = self._tf_condition_indicator                    # (T, n_C)
        d = self._tf_dynamic_indicator                          # (T, n_C)
        w_hp = d * is_hp                                        # (T, n_C)
        w_lp = d * (1.0 - is_hp)
        field_hp = tf.einsum('tl,lg->tg', w_hp, A_dyn)         # (T, G)
        field_lp = tf.einsum('tl,lg->tg', w_lp, A_dyn)         # (T, G)

        return (g_HP_dyn_eff[:, tf.newaxis] * field_hp
                + g_LP_dyn * field_lp)


# ---------------------------------------------------------------------------
# Per-trial dynamic-gain variant of the shared-σ + target model.
# Each trial gets its own scalar g_HP_dyn / g_LP_dyn / g_T_dyn — a dense
# parameterization that lets us inspect single-trial fluctuations of the
# attention-field gains. Used by ``fit_dog_dyn_v3_per_trial.py``.
# ---------------------------------------------------------------------------
class DoGDynamicAttentionFieldPRF2DWithHRF_v3_target_sharedSigma_perTrial(
    DoGDynamicAttentionFieldPRF2DWithHRF_v3_target_sharedSigma,
):
    """sharedSigma v3+target with per-trial dynamic gains.

    Same forward model as
    :class:`DoGDynamicAttentionFieldPRF2DWithHRF_v3_target_sharedSigma`,
    except the three SCALAR dynamic gains ``g_HP_dyn``, ``g_LP_dyn``,
    ``g_T_dyn`` are replaced by THREE PER-TRIAL VECTORS of length
    ``n_trials``::

        g_HP_dyn_t{k:04d}, k = 0..n_trials-1
        g_LP_dyn_t{k:04d}, k = 0..n_trials-1
        g_T_dyn_t{k:04d},  k = 0..n_trials-1

    For TR ``t``, the effective gains are formed by summing each trial's
    gain weighted by that trial's per-TR pulse contribution.  Concretely,
    we precompute three ``(T, n_trials)`` pulse tensors (each row is the
    on-fraction of the corresponding trial's distractor / target pulse at
    TR ``t``):

    - ``dyn_pulse_per_trial`` : distractor pulse (any ring; the trial's
      ring is encoded statically in ``trial_ring_idx``).
    - ``tgt_pulse_per_trial`` : target pulse.

    Plus a static per-trial vector::

    - ``trial_ring_idx`` : (n_trials,) int32 ∈ {0,1,2,3,-1}, the ring
      index of trial ``k``'s distractor (-1 if no distractor).
    - ``trial_target_ring_idx`` : (n_trials,) int32, the ring index of
      trial ``k``'s target.
    - ``trial_is_hp`` : (n_trials,) float32 in {0, 1}; 1 iff trial ``k``'s
      distractor was at the HP ring for its run (i.e. this trial's pulse
      contributes to ``g_HP_dyn`` rather than ``g_LP_dyn``).

    The forward becomes::

        # Distractor field
        # Per-TR per-ring effective gain × pulse
        contrib_hp[t, ℓ] = Σ_k (dyn_pulse_per_trial[t, k]
                                * trial_is_hp[k]
                                * one_hot(trial_ring_idx[k], n_C)[ℓ]
                                * g_HP_dyn_trial[k])
        # similar for LP using (1 - trial_is_hp)
        # And for target (no HP/LP split)
        contrib_tgt[t, ℓ] = Σ_k (tgt_pulse_per_trial[t, k]
                                 * one_hot(trial_target_ring_idx[k], n_C)[ℓ]
                                 * g_T_dyn_trial[k])

        field_hp = Σ_ℓ contrib_hp[t, ℓ] · A_dyn[ℓ, g]   (T, G)
        field_lp = Σ_ℓ contrib_lp[t, ℓ] · A_dyn[ℓ, g]   (T, G)
        field_tgt = Σ_ℓ contrib_tgt[t, ℓ] · A_dyn[ℓ, g]  (T, G)
        mod_dyn  = field_hp + field_lp                  (sign + 1 added later)
        mod_tgt  = field_tgt

    where ``A_dyn`` uses ``sigma_dyn`` (sharedSigma → also used for the
    target Gaussian).

    The static ``g_HP_dyn`` / ``g_LP_dyn`` / ``g_T_dyn`` slots in the
    parent's parameter vector are kept (for slot-index compatibility
    with the parent's σ_T_dyn := σ_dyn tying machinery), but their values
    are FORCED TO ZERO in the forward transform. The forward model reads
    only the per-trial vectors.

    Parameter vector
    ----------------
    Same first 15 slots as the parent (so HRF tail and parent slot
    indices are unchanged), then ``3 * n_trials`` per-trial gains
    appended at the end::

        ['x', 'y', 'sd', 'baseline', 'amplitude',
         'srf_amplitude', 'srf_size',
         'sigma_AF', 'g_HP', 'g_LP',
         'sigma_dyn', 'g_HP_dyn', 'g_LP_dyn',
         'g_T_dyn', 'sigma_T_dyn',
         'g_HP_dyn_t0000', ..., 'g_HP_dyn_t{N-1:04d}',
         'g_LP_dyn_t0000', ..., 'g_LP_dyn_t{N-1:04d}',
         'g_T_dyn_t0000',  ..., 'g_T_dyn_t{N-1:04d}']
        (+ HRF parameters if flexible)

    All per-trial gains are signed-free (no softplus, no factorial sign
    constraint) so the optimiser can find positive or negative gain on
    each trial. They are initialised to 0 and an optional L2 penalty
    can be applied by the caller via ``l2_penalty_per_trial``.

    Notes
    -----
    - The legacy ``g_HP_dyn`` / ``g_LP_dyn`` / ``g_T_dyn`` slots (11/12/
      13) are fixed to 0 every forward pass; their raw variables drift
      freely (no gradient), but their post-softplus values are 0 so they
      have no effect on the loss.
    - With n_trials ≈ 720, this adds ~2160 free shared scalars on top
      of the parent's 8. Coupled with a few hundred voxels, the joint
      fit can take a few minutes per ROI.
    - ``trial_is_hp`` is a 0/1 *static* float vector — no per-TR
      computation needed.
    """

    def __init__(self, *,
                 dyn_pulse_per_trial,
                 tgt_pulse_per_trial,
                 trial_ring_idx,
                 trial_target_ring_idx,
                 trial_is_hp,
                 l2_penalty_per_trial: float = 0.0,
                 **kwargs):
        super().__init__(**kwargs)

        # --- Validate and store per-trial inputs. ---
        dyn = np.asarray(dyn_pulse_per_trial, dtype=np.float32)
        tgt = np.asarray(tgt_pulse_per_trial, dtype=np.float32)
        ring = np.asarray(trial_ring_idx, dtype=np.int32)
        tgt_ring = np.asarray(trial_target_ring_idx, dtype=np.int32)
        is_hp = np.asarray(trial_is_hp, dtype=np.float32)

        if dyn.ndim != 2:
            raise ValueError(
                f"dyn_pulse_per_trial must be 2-D (T, n_trials); got {dyn.shape}")
        if tgt.ndim != 2 or tgt.shape != dyn.shape:
            raise ValueError(
                f"tgt_pulse_per_trial must match dyn shape; got {tgt.shape} vs {dyn.shape}")
        T_expected = int(self.dynamic_indicator.shape[0])
        if dyn.shape[0] != T_expected:
            raise ValueError(
                f"dyn_pulse_per_trial has {dyn.shape[0]} timepoints "
                f"but expected {T_expected} (matches dynamic_indicator).")
        n_trials = dyn.shape[1]
        if ring.shape != (n_trials,) or tgt_ring.shape != (n_trials,) \
                or is_hp.shape != (n_trials,):
            raise ValueError(
                f"trial_*_idx / trial_is_hp must each be (n_trials={n_trials},); "
                f"got shapes ring={ring.shape}, tgt_ring={tgt_ring.shape}, "
                f"is_hp={is_hp.shape}")

        self.n_trials = n_trials
        self.dyn_pulse_per_trial = dyn
        self.tgt_pulse_per_trial = tgt
        self.trial_ring_idx = ring
        self.trial_target_ring_idx = tgt_ring
        self.trial_is_hp = is_hp
        self.l2_penalty_per_trial = float(l2_penalty_per_trial)

        # Build constants for the forward pass.
        # one_hot encoding of trial_ring_idx -> (n_trials, n_C). For trials
        # without a distractor (ring = -1) the row is all-zero so they
        # don't contribute to dyn modulation.
        n_C = int(self.n_conditions)

        def one_hot_with_negative(idx, n):
            mask = (idx >= 0).astype(np.float32)
            safe = np.where(idx >= 0, idx, 0).astype(np.int32)
            oh = np.zeros((len(idx), n), dtype=np.float32)
            oh[np.arange(len(idx)), safe] = 1.0
            return oh * mask[:, None]

        ring_oh = one_hot_with_negative(ring, n_C)             # (n_trials, n_C)
        tgt_ring_oh = one_hot_with_negative(tgt_ring, n_C)     # (n_trials, n_C)

        self._tf_dyn_pulse_per_trial = tf.constant(dyn, dtype=tf.float32)
        self._tf_tgt_pulse_per_trial = tf.constant(tgt, dtype=tf.float32)
        self._tf_trial_ring_oh = tf.constant(ring_oh, dtype=tf.float32)
        self._tf_trial_tgt_ring_oh = tf.constant(tgt_ring_oh, dtype=tf.float32)
        self._tf_trial_is_hp = tf.constant(is_hp, dtype=tf.float32)

        # --- Extend parameter_labels with per-trial slots. ---
        per_trial_labels = (
            [f'g_HP_dyn_t{k:04d}' for k in range(n_trials)]
            + [f'g_LP_dyn_t{k:04d}' for k in range(n_trials)]
            + [f'g_T_dyn_t{k:04d}'  for k in range(n_trials)]
        )
        self._N_PARENT_ENC_PARS = 15  # the v3+target+sharedSigma encoding pars
        self._N_PER_TRIAL_PARS = 3 * n_trials
        # Use the inherited (parent's) labels and append. Note: setting on
        # instance overrides the class attribute.
        self.parameter_labels = list(self.parameter_labels) + per_trial_labels

    # ----- Forward / backward parameter transforms -----------------------

    @tf.function
    def _transform_parameters_forward(self, parameters):
        """Forward transform.

        Layout::

            [parent-encoding (15)] [per-trial gains (3*n_trials)] [hrf tail (optional)]

        Replicates the parent's encoding-side transform inline (so we
        don't fight the parent's HRF-split logic), forces the legacy
        scalar dyn-gain slots (11/12/13) to zero (the per-trial
        vectors carry the gain), and passes the per-trial gains
        through unchanged (signed-free).
        """
        n_pt = self._N_PER_TRIAL_PARS
        n_parent = self._N_PARENT_ENC_PARS
        if self.flexible_hrf_parameters:
            n_hrf = len(self.hrf_model.parameter_labels)
            enc = parameters[:, :n_parent]
            per_trial = parameters[:, n_parent:n_parent + n_pt]
            hrf_tail = parameters[:, -n_hrf:]
        else:
            enc = parameters[:, :n_parent]
            per_trial = parameters[:, n_parent:n_parent + n_pt]
            hrf_tail = None

        # Parent encoding-side transforms.
        x = enc[:, 0:1]
        y = enc[:, 1:2]
        sd = _sd_softplus_forward(enc[:, 2:3], self.sd_min)
        baseline = enc[:, 3:4]
        amplitude = enc[:, 4:5]
        srf_amp = tf.math.softplus(enc[:, 5:6])
        srf_size = _sd_softplus_forward(enc[:, 6:7], self.sd_min)
        sigma_AF = _sd_softplus_forward(enc[:, 7:8], self.sd_min)
        # g_HP / g_LP : the parent (v3_target / sharedSigma chain) treats
        # them through the parent v3 transform. We replicate the signed
        # branch (the fit script always uses mode='signed') for
        # simplicity; if mode != 'signed' a softplus would apply, but
        # this class is intended for signed mode only.
        if self._signed_gains:
            g_HP = enc[:, 8:9]
            g_LP = enc[:, 9:10]
        else:
            g_HP = tf.math.softplus(enc[:, 8:9])
            g_LP = tf.math.softplus(enc[:, 9:10])
        sigma_dyn = _sd_softplus_forward(enc[:, 10:11], self.sd_min)
        # Force legacy scalar dyn gains to zero — per-trial vectors carry
        # the dyn signal.
        g_HP_dyn = tf.zeros_like(enc[:, 11:12])
        g_LP_dyn = tf.zeros_like(enc[:, 12:13])
        g_T_dyn = tf.zeros_like(enc[:, 13:14])
        sigma_T_dyn = sigma_dyn  # sharedSigma constraint.

        out_enc = tf.concat([
            x, y, sd, baseline, amplitude, srf_amp, srf_size,
            sigma_AF, g_HP, g_LP,
            sigma_dyn, g_HP_dyn, g_LP_dyn,
            g_T_dyn, sigma_T_dyn,
            # Per-trial gains: pass-through (signed-free).
            per_trial,
        ], axis=1)

        if hrf_tail is not None:
            hrf_pars = self.hrf_model._transform_parameters_forward(hrf_tail)
            return tf.concat([out_enc, hrf_pars], axis=1)
        return out_enc

    @tf.function
    def _transform_parameters_backward(self, parameters):
        n_pt = self._N_PER_TRIAL_PARS
        n_parent = self._N_PARENT_ENC_PARS
        if self.flexible_hrf_parameters:
            n_hrf = len(self.hrf_model.parameter_labels)
            enc = parameters[:, :n_parent]
            per_trial = parameters[:, n_parent:n_parent + n_pt]
            hrf_tail = parameters[:, -n_hrf:]
        else:
            enc = parameters[:, :n_parent]
            per_trial = parameters[:, n_parent:n_parent + n_pt]
            hrf_tail = None

        x = enc[:, 0:1]
        y = enc[:, 1:2]
        sd = _sd_softplus_inverse(enc[:, 2:3], self.sd_min)
        baseline = enc[:, 3:4]
        amplitude = enc[:, 4:5]
        srf_amp = tfp.math.softplus_inverse(enc[:, 5:6])
        srf_size = _sd_softplus_inverse(enc[:, 6:7], self.sd_min)
        sigma_AF = _sd_softplus_inverse(enc[:, 7:8], self.sd_min)
        if self._signed_gains:
            g_HP = enc[:, 8:9]
            g_LP = enc[:, 9:10]
        else:
            g_HP = tfp.math.softplus_inverse(enc[:, 8:9])
            g_LP = tfp.math.softplus_inverse(enc[:, 9:10])
        sigma_dyn = _sd_softplus_inverse(enc[:, 10:11], self.sd_min)
        # Map zero-clamped legacy slots back to zero raw too.
        g_HP_dyn = tf.zeros_like(enc[:, 11:12])
        g_LP_dyn = tf.zeros_like(enc[:, 12:13])
        g_T_dyn = tf.zeros_like(enc[:, 13:14])
        sigma_T_dyn = sigma_dyn

        out_enc = tf.concat([
            x, y, sd, baseline, amplitude, srf_amp, srf_size,
            sigma_AF, g_HP, g_LP,
            sigma_dyn, g_HP_dyn, g_LP_dyn,
            g_T_dyn, sigma_T_dyn,
            per_trial,  # signed-free pass-through
        ], axis=1)

        if hrf_tail is not None:
            hrf_pars = self.hrf_model._transform_parameters_backward(hrf_tail)
            return tf.concat([out_enc, hrf_pars], axis=1)
        return out_enc

    # ----- Modulation field overrides ------------------------------------

    @tf.function
    def _attention_modulation_dynamic_v3(self, parameters):
        """Per-TR dynamic-AF modulation using PER-TRIAL HP/LP gains.

        The per-trial gains live at slots ``[15 : 15 + n_trials]`` (HP)
        and ``[15 + n_trials : 15 + 2*n_trials]`` (LP). They are read
        from batch 0 / voxel 0 since they are shared across voxels.
        """
        n = self.n_trials
        # Per-trial HP and LP gain vectors: (n_trials,).
        g_HP_dyn_trial = parameters[0, 0, 15:15 + n]
        g_LP_dyn_trial = parameters[0, 0, 15 + n:15 + 2 * n]

        # Distractor pulse field: sigma from sigma_dyn.
        sigma_dyn = parameters[0, 0, 10]
        gx = self._grid_coordinates[:, 0][tf.newaxis, :]   # (1, G)
        gy = self._grid_coordinates[:, 1][tf.newaxis, :]
        rx = self._tf_ring_positions[:, 0][:, tf.newaxis]  # (n_C, 1)
        ry = self._tf_ring_positions[:, 1][:, tf.newaxis]
        diff_sq = (gx - rx) ** 2 + (gy - ry) ** 2
        A_dyn = tf.exp(-diff_sq / (2.0 * sigma_dyn ** 2))   # (n_C, G)

        # Per-trial weighted contribution to the time × ring grid.
        # Each trial's pulse is at one ring (encoded by ring_oh), and is
        # routed to HP or LP based on trial_is_hp.
        # contrib_hp[t, ℓ] = Σ_k pulse[t,k] * is_hp[k] * ring_oh[k,ℓ] * g_HP[k]
        #                  = einsum('tk,kl->tl',
        #                           pulse * (is_hp * g_HP)[None,:], ring_oh)
        pulse = self._tf_dyn_pulse_per_trial          # (T, n_trials)
        ring_oh = self._tf_trial_ring_oh              # (n_trials, n_C)
        is_hp = self._tf_trial_is_hp                  # (n_trials,)

        # Effective per-trial weights for HP and LP.
        w_hp_per_trial = is_hp * g_HP_dyn_trial          # (n_trials,)
        w_lp_per_trial = (1.0 - is_hp) * g_LP_dyn_trial  # (n_trials,)

        # (T, n_C) contributions.
        contrib_hp_tl = tf.einsum(
            'tk,kl->tl',
            pulse * w_hp_per_trial[tf.newaxis, :],
            ring_oh,
        )
        contrib_lp_tl = tf.einsum(
            'tk,kl->tl',
            pulse * w_lp_per_trial[tf.newaxis, :],
            ring_oh,
        )

        # Spread to grid via Gaussian per ring -> (T, G).
        field_hp_tg = tf.einsum('tl,lg->tg', contrib_hp_tl, A_dyn)
        field_lp_tg = tf.einsum('tl,lg->tg', contrib_lp_tl, A_dyn)

        return field_hp_tg + field_lp_tg                  # (T, G)

    @tf.function
    def _attention_modulation_target(self, parameters):
        """Per-TR target field using PER-TRIAL gains (sharedSigma = sigma_dyn).
        """
        n = self.n_trials
        g_T_dyn_trial = parameters[0, 0, 15 + 2 * n:15 + 3 * n]   # (n_trials,)

        sigma_dyn = parameters[0, 0, 10]                          # scalar
        gx = self._grid_coordinates[:, 0][tf.newaxis, :]
        gy = self._grid_coordinates[:, 1][tf.newaxis, :]
        rx = self._tf_ring_positions[:, 0][:, tf.newaxis]
        ry = self._tf_ring_positions[:, 1][:, tf.newaxis]
        diff_sq = (gx - rx) ** 2 + (gy - ry) ** 2
        A_tgt = tf.exp(-diff_sq / (2.0 * sigma_dyn ** 2))         # (n_C, G)

        pulse_tgt = self._tf_tgt_pulse_per_trial                  # (T, n_trials)
        tgt_ring_oh = self._tf_trial_tgt_ring_oh                  # (n_trials, n_C)

        # (T, n_C) contribution from all trials' target pulses.
        contrib_tl = tf.einsum(
            'tk,kl->tl',
            pulse_tgt * g_T_dyn_trial[tf.newaxis, :],
            tgt_ring_oh,
        )
        # Spread to grid -> (T, G).
        return tf.einsum('tl,lg->tg', contrib_tl, A_tgt)


# ---------------------------------------------------------------------------
# Gaussian-backbone v3 + target counterpart of the DoG sharedSigma model.
#
# This is the model used by ``fit_gaussian_dynamic_af_braincoder.py``.
# It mirrors :class:`DoGDynamicAttentionFieldPRF2DWithHRF_v3_target_sharedSigma`
# but the per-voxel kernel is a plain 2-D Gaussian (no DoG surround), so
# the per-voxel free parameters are just ``x, y, sd, baseline, amplitude``
# — initialised from model 1 (Gaussian PRF, fixed HRF).
#
# Forward model
# -------------
# Identical SUSTAINED + PHASIC additive structure as the DoG version:
#
#     sustained: ∫ paradigm · rf · (1 + sign · mod_sustained) dg
#                via the inherited DynamicAttentionFieldPRF2D_v3._basis_predictions
#                logic (sustained part).
#     phasic:    sign · ∫ paradigm · rf · (mod_dyn + mod_tgt) dg
#
# Parameter slot indices
# ----------------------
# 0:x, 1:y, 2:sd, 3:baseline, 4:amplitude,
# 5:sigma_AF, 6:g_HP, 7:g_LP,
# 8:sigma_dyn, 9:g_HP_dyn, 10:g_LP_dyn,
# 11:g_T_dyn, 12:sigma_T_dyn.
#
# ``sigma_T_dyn`` is forced to equal ``sigma_dyn`` in every forward pass
# (sharedSigma constraint).
# ---------------------------------------------------------------------------
class GaussianDynamicAttentionFieldPRF2D_v3_target(DynamicAttentionFieldPRF2D_v3):
    """Gaussian v3 + phasic-target gain. Encoding-only (no HRF) base.

    Mirrors :class:`DoGDynamicAttentionFieldPRF2D_v3_target` but with a
    Gaussian voxel kernel (no DoG surround).

    Per-voxel parameters (5)
    ------------------------
    ``x``, ``y``, ``sd``, ``baseline``, ``amplitude``.

    Shared parameters (8 = 6 from v3 + 2 new)
    -----------------------------------------
    ``sigma_AF``, ``g_HP``, ``g_LP``,
    ``sigma_dyn``, ``g_HP_dyn``, ``g_LP_dyn``,
    ``g_T_dyn``, ``sigma_T_dyn``.

    Total: 13 encoding parameters per voxel.
    """

    parameter_labels = [
        'x', 'y', 'sd', 'baseline', 'amplitude',
        'sigma_AF', 'g_HP', 'g_LP',
        'sigma_dyn', 'g_HP_dyn', 'g_LP_dyn',
        'g_T_dyn', 'sigma_T_dyn',
    ]

    # Number of encoding parameters in the parent v3 (used to slice).
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
                "GaussianDynamicAttentionFieldPRF2D_v3_target requires a "
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

    @tf.function
    def _attention_modulation_target(self, parameters):
        """Per-TR phasic-target modulation field on the stimulus grid.

        Returns ``g_T_dyn · Σ_ℓ tgt_ℓ(t) · A_ℓ^{tgt}(g)`` of shape
        ``(T, G)``. Mirrors
        :meth:`DoGDynamicAttentionFieldPRF2D_v3_target._attention_modulation_target`
        but with parameter slots adjusted for the Gaussian v3 backbone
        (slot 11 = g_T_dyn, slot 12 = sigma_T_dyn).
        """
        g_T_dyn = parameters[0, 0, 11]                     # scalar
        sigma_T_dyn = parameters[0, 0, 12]                 # scalar

        gx = self._grid_coordinates[:, 0][tf.newaxis, :]   # (1, G)
        gy = self._grid_coordinates[:, 1][tf.newaxis, :]
        rx = self._tf_ring_positions[:, 0][:, tf.newaxis]
        ry = self._tf_ring_positions[:, 1][:, tf.newaxis]

        diff_sq = (gx - rx) ** 2 + (gy - ry) ** 2
        A_tgt = tf.exp(-diff_sq / (2.0 * sigma_T_dyn ** 2))  # (n_C, G)

        tgt = self._tf_target_indicator                     # (T, n_C)

        field_tgt = tf.einsum('tl,lg->tg', tgt, A_tgt)
        return g_T_dyn * field_tgt                          # (T, G)

    @tf.function
    def _basis_predictions(self, paradigm, parameters):
        # paradigm: (B, T, G)
        # parameters: (B, V, n_parameters=13)
        # Same structure as the v3 parent's _basis_predictions, but the
        # phasic modulation also includes the target-onset term.

        # Per-voxel Gaussian RF on the grid: (B, V, G).
        rf = self._get_rf(self.grid_coordinates, parameters)

        # Sustained per-condition AF modulation: (B, V, n_C, G).
        # AttentionFieldPRF2D._attention_modulation reads slots 5,6,7.
        mod_sustained = self._attention_modulation(parameters)

        # Effective per-condition RF (sustained part): (B, V, n_C, G).
        eff_rf_per_cond = rf[:, :, tf.newaxis, :] * mod_sustained

        # Sustained partial: (B, T, V) via condition_indicator selection.
        partial = tf.einsum('btg,bvcg->btvc', paradigm, eff_rf_per_cond)
        ci = self._tf_condition_indicator       # (T, n_C)
        sustained = tf.einsum('tc,btvc->btv', ci, partial)

        # Phasic-distractor modulation: (T, G), HP/LP split, with σ_dyn.
        mod_dyn = self._attention_modulation_dynamic_v3(parameters)

        # Phasic-target modulation: (T, G), with σ_T_dyn.
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

    @tf.function
    def _transform_parameters_forward(self, parameters):
        """Encoding-only forward transform.

        Slots 0..10 are the v3 encoding parameters: delegate to the v3
        parent's transform. Slots 11 (``g_T_dyn``) and 12
        (``sigma_T_dyn``) are NEW: g_T_dyn is sign-aware (signed mode
        passes through; otherwise softplus); sigma_T_dyn is always
        softplus-positive (matches sigma_AF / sigma_dyn).
        """
        v3_pars = DynamicAttentionFieldPRF2D_v3._transform_parameters_forward(
            self, parameters[:, :self._N_V3_ENC_PARS])

        if self._signed_gains:
            g_T_dyn = parameters[:, 11][:, tf.newaxis]
        else:
            g_T_dyn = tf.math.softplus(parameters[:, 11][:, tf.newaxis])
        sigma_T_dyn = _sd_softplus_forward(
            parameters[:, 12][:, tf.newaxis], self.sd_min)

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
        sigma_T_dyn_unb = _sd_softplus_inverse(
            parameters[:, 12][:, tf.newaxis], self.sd_min)

        return tf.concat([v3_pars, g_T_dyn_unb, sigma_T_dyn_unb], axis=1)


class GaussianDynamicAttentionFieldPRF2DWithHRF_v3_target(
    DynamicAttentionFieldPRF2DWithHRF_v3,
    GaussianDynamicAttentionFieldPRF2D_v3_target,
):
    """HRF-convolved version of
    :class:`GaussianDynamicAttentionFieldPRF2D_v3_target`.

    Mirrors :class:`DoGDynamicAttentionFieldPRF2DWithHRF_v3_target` but
    with the Gaussian PRF backbone.

    Free parameters::

        ['x', 'y', 'sd', 'baseline', 'amplitude',
         'sigma_AF', 'g_HP', 'g_LP',
         'sigma_dyn', 'g_HP_dyn', 'g_LP_dyn',
         'g_T_dyn', 'sigma_T_dyn']
        (+ HRF parameters if flexible)

    During joint AF + PRF fitting, pass

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
        GaussianDynamicAttentionFieldPRF2D_v3_target.__init__(
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
                GaussianDynamicAttentionFieldPRF2D_v3_target
                ._transform_parameters_forward(
                    self, parameters[:, :-n_hrf_pars])
            )
            hrf_pars = self.hrf_model._transform_parameters_forward(
                parameters[:, -n_hrf_pars:])
            return tf.concat([encoding_pars, hrf_pars], axis=1)
        else:
            return (
                GaussianDynamicAttentionFieldPRF2D_v3_target
                ._transform_parameters_forward(self, parameters)
            )

    @tf.function
    def _transform_parameters_backward(self, parameters):
        if self.flexible_hrf_parameters:
            n_hrf_pars = len(self.hrf_model.parameter_labels)
            encoding_pars = (
                GaussianDynamicAttentionFieldPRF2D_v3_target
                ._transform_parameters_backward(
                    self, parameters[:, :-n_hrf_pars])
            )
            hrf_pars = self.hrf_model._transform_parameters_backward(
                parameters[:, -n_hrf_pars:])
            return tf.concat([encoding_pars, hrf_pars], axis=1)
        else:
            return (
                GaussianDynamicAttentionFieldPRF2D_v3_target
                ._transform_parameters_backward(self, parameters)
            )


class GaussianDynamicAttentionFieldPRF2DWithHRF_v3_target_sharedSigma(
    GaussianDynamicAttentionFieldPRF2DWithHRF_v3_target,
):
    """Gaussian v3 + target with ``sigma_T_dyn`` tied to ``sigma_dyn``.

    Same parameter vector and forward model as
    :class:`GaussianDynamicAttentionFieldPRF2DWithHRF_v3_target`, but
    ``sigma_T_dyn`` is overwritten with ``sigma_dyn`` at the very end
    of the forward parameter transform. The two phasic Gaussians
    (distractor-onset and target-onset) therefore share a single
    spatial extent.

    Mirrors
    :class:`DoGDynamicAttentionFieldPRF2DWithHRF_v3_target_sharedSigma`
    but with the Gaussian PRF backbone (per-voxel: x, y, sd, baseline,
    amplitude). σ_T_dyn lives at slot 12 (vs slot 14 for DoG); σ_dyn
    lives at slot 8 (vs slot 10 for DoG).

    Initialisation
    --------------
    Callers should set ``init_pars['sigma_T_dyn'] = init_pars['sigma_dyn']``
    before passing inits to the fitter, so the (effectively unused)
    σ_T_dyn raw variable starts at the right place. The fit script
    handles this.
    """

    @tf.function
    def _transform_parameters_forward(self, parameters):
        # Run the parent forward transform (handles HRF + encoding pars,
        # applies softplus to both sigma_dyn (slot 8) and sigma_T_dyn
        # (slot 12)).
        out = super()._transform_parameters_forward(parameters)

        # Force sigma_T_dyn := sigma_dyn after the parent transform.
        # `out` shape: (n_voxels, n_parameters[+ n_hrf_pars]).
        sigma_dyn_col = out[:, 8:9]                           # (V, 1)
        before = out[:, :12]                                  # (V, 12)
        after = out[:, 13:]                                   # (V, rest)
        out_tied = tf.concat([before, sigma_dyn_col, after], axis=1)
        return out_tied

    @tf.function
    def _transform_parameters_backward(self, parameters):
        # Run the parent backward transform (inverts softplus on both
        # sigma_dyn slot 8 and sigma_T_dyn slot 12 in raw space).
        out = super()._transform_parameters_backward(parameters)

        # Tie the raw σ_T_dyn slot to the raw σ_dyn slot so that, if
        # callers ever recover the "raw" parameter vector, both raw
        # variables agree. Both sigmas use softplus so equal raw values
        # give equal post-softplus values.
        raw_sigma_dyn_col = out[:, 8:9]                       # (V, 1)
        before = out[:, :12]                                  # (V, 12)
        after = out[:, 13:]                                   # (V, rest)
        out_tied = tf.concat([before, raw_sigma_dyn_col, after], axis=1)
        return out_tied


class GaussianDynamicAttentionFieldPRF2DWithHRF_v3_target_sharedSigma_sharedDynGain(
    GaussianDynamicAttentionFieldPRF2DWithHRF_v3_target_sharedSigma,
):
    """Gaussian sharedSigma v3+target with single dynamic-distractor gain.

    On top of the parent's σ_T_dyn := σ_dyn tying, also ties
    ``g_LP_dyn`` (slot 10) := ``g_HP_dyn`` (slot 9). Tests whether
    the distractor-onset transient is meaningfully different at HP
    vs LP locations, or whether one gain suffices.

    Initialisation
    --------------
    Callers should set ``init_pars['g_LP_dyn'] = init_pars['g_HP_dyn']``
    before passing inits to the fitter.
    """

    @tf.function
    def _transform_parameters_forward(self, parameters):
        out = super()._transform_parameters_forward(parameters)
        # Tie g_LP_dyn (slot 10) := g_HP_dyn (slot 9).
        g_hp_dyn_col = out[:, 9:10]
        before = out[:, :10]
        after = out[:, 11:]
        return tf.concat([before, g_hp_dyn_col, after], axis=1)

    @tf.function
    def _transform_parameters_backward(self, parameters):
        out = super()._transform_parameters_backward(parameters)
        raw_g_hp_dyn_col = out[:, 9:10]
        before = out[:, :10]
        after = out[:, 11:]
        return tf.concat([before, raw_g_hp_dyn_col, after], axis=1)


class GaussianDynamicAttentionFieldPRF2DWithHRF_v3_target_allSharedSigma(
    GaussianDynamicAttentionFieldPRF2DWithHRF_v3_target_sharedSigma,
):
    """Gaussian v3 + target with ALL Gaussian widths tied to ``sigma_dyn``.

    Stricter constraint than
    :class:`GaussianDynamicAttentionFieldPRF2DWithHRF_v3_target_sharedSigma`,
    which only ties ``sigma_T_dyn := sigma_dyn``. This subclass
    additionally ties ``sigma_AF := sigma_dyn``, so a single shared
    width parameter (``sigma_dyn``, slot 8) controls all three Gaussian
    attention fields:

        sigma_AF    (slot 5)  := sigma_dyn  (slot 8)
        sigma_T_dyn (slot 12) := sigma_dyn  (slot 8)

    The σ_AF override is applied on top of the parent's σ_T_dyn override
    by composing the forward / backward transforms. Slot indices reflect
    the Gaussian v3 backbone (per-voxel x, y, sd, baseline, amplitude).

    Initialisation
    --------------
    Callers should set
    ``init_pars['sigma_AF'] = init_pars['sigma_T_dyn'] = init_pars['sigma_dyn']``
    before passing inits to the fitter, so the (effectively unused)
    σ_AF and σ_T_dyn raw variables start at the right place. The fit
    script handles this.
    """

    @tf.function
    def _transform_parameters_forward(self, parameters):
        # Run the parent (sharedSigma) forward transform first. After
        # this, slot 12 (σ_T_dyn) already equals slot 8 (σ_dyn); slot 5
        # (σ_AF) still has its own softplus-positive value.
        out = super()._transform_parameters_forward(parameters)

        # Force σ_AF := σ_dyn after the parent transform.
        # `out` shape: (n_voxels, n_parameters[+ n_hrf_pars]).
        sigma_dyn_col = out[:, 8:9]                           # (V, 1)
        before = out[:, :5]                                   # (V, 5)
        after = out[:, 6:]                                    # (V, rest)
        out_tied = tf.concat([before, sigma_dyn_col, after], axis=1)
        return out_tied

    @tf.function
    def _transform_parameters_backward(self, parameters):
        # Parent backward already ties raw σ_T_dyn := raw σ_dyn.
        out = super()._transform_parameters_backward(parameters)

        # Tie the raw σ_AF slot to the raw σ_dyn slot so that, if callers
        # ever recover the "raw" parameter vector, all three sigmas
        # agree. All three use softplus, so equal raw values give equal
        # post-softplus values.
        raw_sigma_dyn_col = out[:, 8:9]                       # (V, 1)
        before = out[:, :5]                                   # (V, 5)
        after = out[:, 6:]                                    # (V, rest)
        out_tied = tf.concat([before, raw_sigma_dyn_col, after], axis=1)
        return out_tied


class GaussianDynamicAttentionFieldPRF2DWithHRF_v3_target_allSharedSigma_sharedDynGain(
    GaussianDynamicAttentionFieldPRF2DWithHRF_v3_target_allSharedSigma,
):
    """Gaussian: all σ tied AND single dyn-gain.

    Inherits the σ_AF := σ_T_dyn := σ_dyn tying from the parent and
    additionally ties g_LP_dyn (slot 10) := g_HP_dyn (slot 9). Most-
    restricted Gaussian variant in the 4-class factorial.
    """

    @tf.function
    def _transform_parameters_forward(self, parameters):
        out = super()._transform_parameters_forward(parameters)
        g_hp_dyn_col = out[:, 9:10]
        before = out[:, :10]
        after = out[:, 11:]
        return tf.concat([before, g_hp_dyn_col, after], axis=1)

    @tf.function
    def _transform_parameters_backward(self, parameters):
        out = super()._transform_parameters_backward(parameters)
        raw_g_hp_dyn_col = out[:, 9:10]
        before = out[:, :10]
        after = out[:, 11:]
        return tf.concat([before, raw_g_hp_dyn_col, after], axis=1)


# ===========================================================================
# Divisive-Normalization counterpart of the canonical DoG-AF model.
# Lets us fit AF on top of m6 (DN+HRF) PRFs instead of m4 (DoG+HRF).
# ===========================================================================

class DivisiveNormalizationDynamicAttentionFieldPRF2DWithHRF_v3_target_sharedSigma(
    DivisiveNormalizationGaussianPRF2DWithHRF,
):
    """AF + dynamic distractor + target + sharedSigma, on a DN PRF base.

    Same modulation logic as the DoG counterpart
    (:class:`DoGDynamicAttentionFieldPRF2DWithHRF_v3_target_sharedSigma`):

    - Sustained per-condition AF (σ_AF, g_HP, g_LP) — modulates the
      stimulus drive within each (HP, condition) block.
    - Dynamic per-TR distractor (σ_dyn, g_HP_dyn, g_LP_dyn) — additive,
      sign-aware (suppression by default).
    - Per-TR target (g_T_dyn, σ_T_dyn) — additive at target-on TRs.
    - σ_T_dyn := σ_dyn enforced post-softplus (sharedSigma).

    Where DoG and DN differ is the **response function**. After
    computing the AF-modulated stimulus
    ``stim_mod(t, g) = stim(t, g) · M(t, g)`` (with M including all
    three modulation terms), we apply Divisive Normalization rather
    than a Difference-of-Gaussians difference::

        neural    = rf_amp · ⟨stim_mod, G_c⟩ + neural_baseline
        suppress  = |rf_amp| · srf_amp · ⟨stim_mod, G_s⟩ + surround_baseline
        response  = neural / suppress + bold_baseline

    Both center G_c and surround G_s see the SAME modulated stimulus,
    so attention shrinks/grows BOTH the drive and the normalisation
    pool — the conceptually clean "AF on top of DN" that the DoG-AF
    model can't express.

    Parameter layout (17 encoding + HRF)
    ------------------------------------
    Per-voxel (9):
        0 x, 1 y, 2 sd,
        3 rf_amplitude, 4 srf_amplitude, 5 srf_size,
        6 neural_baseline, 7 surround_baseline, 8 bold_baseline.
    Shared (8):
        9 sigma_AF, 10 g_HP, 11 g_LP,
        12 sigma_dyn, 13 g_HP_dyn, 14 g_LP_dyn,
        15 g_T_dyn, 16 sigma_T_dyn  (tied := sigma_dyn).

    Slot indices shift by +2 vs the DoG counterpart because the DN
    parameter vector inserts ``neural_baseline`` and
    ``surround_baseline`` ahead of the AF block.
    """

    parameter_labels = [
        'x', 'y', 'sd',
        'rf_amplitude', 'srf_amplitude', 'srf_size',
        'neural_baseline', 'surround_baseline', 'bold_baseline',
        'sigma_AF', 'g_HP', 'g_LP',
        'sigma_dyn', 'g_HP_dyn', 'g_LP_dyn',
        'g_T_dyn', 'sigma_T_dyn',
    ]

    def __init__(self, grid_coordinates=None, paradigm=None, data=None,
                 parameters=None,
                 condition_indicator=None, dynamic_indicator=None,
                 target_indicator=None, ring_positions=None,
                 mode='suppression',
                 weights=None, omega=None,
                 positive_image_values_only=True,
                 hrf_model=None, flexible_hrf_parameters=False,
                 verbosity=logging.INFO, **kwargs):
        if condition_indicator is None:
            raise ValueError(
                "DN-AF model requires `condition_indicator` of shape "
                "(n_timepoints, n_ring_positions).")
        if dynamic_indicator is None:
            raise ValueError(
                "DN-AF model requires `dynamic_indicator` of shape "
                "(n_timepoints, n_ring_positions).")
        if target_indicator is None:
            raise ValueError(
                "DN-AF model requires `target_indicator` of shape "
                "(n_timepoints, n_ring_positions).")
        if ring_positions is None:
            raise ValueError(
                "DN-AF model requires `ring_positions` of shape "
                "(n_conditions, 2).")
        if mode not in ('suppression', 'attraction', 'signed'):
            raise ValueError(
                f"mode must be 'suppression', 'attraction', or 'signed', "
                f"got {mode!r}")

        self.mode = mode
        self._sign = -1.0 if mode == 'suppression' else +1.0
        self._signed_gains = (mode == 'signed')

        self.condition_indicator = np.asarray(condition_indicator,
                                              dtype=np.float32)
        self.dynamic_indicator = np.asarray(dynamic_indicator,
                                            dtype=np.float32)
        self.target_indicator = np.asarray(target_indicator,
                                           dtype=np.float32)
        self.ring_positions = np.asarray(ring_positions, dtype=np.float32)
        self.n_conditions = self.ring_positions.shape[0]
        # is_hp[c, ℓ] == 1 iff ring ℓ is the HP for condition c.
        self._is_hp = np.eye(self.n_conditions, dtype=np.float32)

        super().__init__(
            grid_coordinates=grid_coordinates,
            paradigm=paradigm, data=data, parameters=parameters,
            weights=weights,
            hrf_model=hrf_model,
            flexible_hrf_parameters=flexible_hrf_parameters,
            positive_image_values_only=positive_image_values_only,
            verbosity=verbosity, **kwargs)

        # Cache TF constants for speed.
        self._tf_condition_indicator = tf.constant(self.condition_indicator,
                                                   dtype=tf.float32)
        self._tf_dynamic_indicator = tf.constant(self.dynamic_indicator,
                                                  dtype=tf.float32)
        self._tf_target_indicator = tf.constant(self.target_indicator,
                                                 dtype=tf.float32)
        self._tf_ring_positions = tf.constant(self.ring_positions,
                                              dtype=tf.float32)
        self._tf_is_hp = tf.constant(self._is_hp, dtype=tf.float32)
        self._tf_sign = tf.constant(self._sign, dtype=tf.float32)

    # ----------------------------------------------------------------
    # AF modulation methods. All slots shifted by +2 vs the DoG
    # counterpart because DN inserts neural_baseline + surround_baseline
    # before the AF block.
    # ----------------------------------------------------------------

    @tf.function
    def _attention_modulation_sustained(self, parameters):
        """Per-condition sustained AF on the grid. Slots 9, 10, 11."""
        gx = self._grid_coordinates[:, 0][tf.newaxis, tf.newaxis, tf.newaxis, :]
        gy = self._grid_coordinates[:, 1][tf.newaxis, tf.newaxis, tf.newaxis, :]

        sigma_AF = parameters[:, :, 9, tf.newaxis, tf.newaxis]    # (B,V,1,1)
        g_HP = parameters[:, :, 10, tf.newaxis, tf.newaxis]
        g_LP = parameters[:, :, 11, tf.newaxis, tf.newaxis]

        rx = self._tf_ring_positions[:, 0][tf.newaxis, tf.newaxis, :, tf.newaxis]
        ry = self._tf_ring_positions[:, 1][tf.newaxis, tf.newaxis, :, tf.newaxis]

        diff_sq = (gx - rx) ** 2 + (gy - ry) ** 2
        A = tf.exp(-diff_sq / (2.0 * sigma_AF ** 2))   # (B,V,n_C_ring,G)

        is_hp = self._tf_is_hp[tf.newaxis, tf.newaxis, :, :]
        w = is_hp * g_HP + (1.0 - is_hp) * g_LP

        modulation_sum = tf.einsum('bvcl,bvlg->bvcg', w, A)
        mod = 1.0 + self._tf_sign * modulation_sum   # (B,V,n_C,G)
        return tf.maximum(mod, 0.0)

    @tf.function
    def _attention_modulation_dynamic(self, parameters):
        """Per-TR phasic distractor field on the grid. Slots 12, 13, 14."""
        sigma_dyn = parameters[0, 0, 12]
        g_HP_dyn = parameters[0, 0, 13]
        g_LP_dyn = parameters[0, 0, 14]

        gx = self._grid_coordinates[:, 0][tf.newaxis, :]
        gy = self._grid_coordinates[:, 1][tf.newaxis, :]
        rx = self._tf_ring_positions[:, 0][:, tf.newaxis]
        ry = self._tf_ring_positions[:, 1][:, tf.newaxis]

        diff_sq = (gx - rx) ** 2 + (gy - ry) ** 2
        A_dyn = tf.exp(-diff_sq / (2.0 * sigma_dyn ** 2))   # (n_C, G)

        is_hp_per_tr = self._tf_condition_indicator   # (T, n_C)
        d = self._tf_dynamic_indicator                # (T, n_C)
        w_hp = d * is_hp_per_tr
        w_lp = d * (1.0 - is_hp_per_tr)

        field_hp = tf.einsum('tl,lg->tg', w_hp, A_dyn)
        field_lp = tf.einsum('tl,lg->tg', w_lp, A_dyn)
        return g_HP_dyn * field_hp + g_LP_dyn * field_lp   # (T, G)

    @tf.function
    def _attention_modulation_target(self, parameters):
        """Per-TR target field on the grid. Slots 15, 16."""
        g_T_dyn = parameters[0, 0, 15]
        sigma_T_dyn = parameters[0, 0, 16]

        gx = self._grid_coordinates[:, 0][tf.newaxis, :]
        gy = self._grid_coordinates[:, 1][tf.newaxis, :]
        rx = self._tf_ring_positions[:, 0][:, tf.newaxis]
        ry = self._tf_ring_positions[:, 1][:, tf.newaxis]

        diff_sq = (gx - rx) ** 2 + (gy - ry) ** 2
        A_tgt = tf.exp(-diff_sq / (2.0 * sigma_T_dyn ** 2))   # (n_C, G)

        tgt = self._tf_target_indicator   # (T, n_C)
        field_tgt = tf.einsum('tl,lg->tg', tgt, A_tgt)
        return g_T_dyn * field_tgt   # (T, G)

    # ----------------------------------------------------------------
    # Forward / backward parameter transforms.
    # DN PRF block (0..8) — delegate slots 0..7 to DN parent, slot 8
    # (bold_baseline) is identity. AF block (9..16) — direct transforms.
    # sharedSigma tying: sigma_T_dyn (slot 16) := sigma_dyn (slot 12).
    # ----------------------------------------------------------------

    @tf.function
    def _transform_parameters_forward(self, parameters):
        if self.flexible_hrf_parameters:
            n_hrf = len(self.hrf_model.parameter_labels)
            enc = parameters[:, :-n_hrf]
            hrf_raw = parameters[:, -n_hrf:]
            hrf_out = self.hrf_model._transform_parameters_forward(hrf_raw)
        else:
            enc = parameters
            hrf_out = None

        dn = DivisiveNormalizationGaussianPRF2D._transform_parameters_forward(
            self, enc[:, :8])
        bold_bl = enc[:, 8:9]   # identity (free param)
        prf_out = tf.concat([dn, bold_bl], axis=1)   # (V, 9)

        sigma_AF = _sd_softplus_forward(enc[:, 9:10], self.sd_min)
        if self._signed_gains:
            g_HP = enc[:, 10:11]
            g_LP = enc[:, 11:12]
        else:
            g_HP = tf.math.softplus(enc[:, 10:11])
            g_LP = tf.math.softplus(enc[:, 11:12])

        sigma_dyn = _sd_softplus_forward(enc[:, 12:13], self.sd_min)
        if self._signed_gains:
            g_HP_dyn = enc[:, 13:14]
            g_LP_dyn = enc[:, 14:15]
            g_T_dyn = enc[:, 15:16]
        else:
            g_HP_dyn = tf.math.softplus(enc[:, 13:14])
            g_LP_dyn = tf.math.softplus(enc[:, 14:15])
            g_T_dyn = tf.math.softplus(enc[:, 15:16])

        # sharedSigma: σ_T_dyn := σ_dyn post-softplus.
        sigma_T_dyn = sigma_dyn

        enc_out = tf.concat(
            [prf_out, sigma_AF, g_HP, g_LP,
             sigma_dyn, g_HP_dyn, g_LP_dyn, g_T_dyn, sigma_T_dyn], axis=1)

        if hrf_out is not None:
            return tf.concat([enc_out, hrf_out], axis=1)
        return enc_out

    @tf.function
    def _transform_parameters_backward(self, parameters):
        if self.flexible_hrf_parameters:
            n_hrf = len(self.hrf_model.parameter_labels)
            enc = parameters[:, :-n_hrf]
            hrf_post = parameters[:, -n_hrf:]
            hrf_raw = self.hrf_model._transform_parameters_backward(hrf_post)
        else:
            enc = parameters
            hrf_raw = None

        dn_raw = DivisiveNormalizationGaussianPRF2D._transform_parameters_backward(
            self, enc[:, :8])
        bold_bl = enc[:, 8:9]
        prf_raw = tf.concat([dn_raw, bold_bl], axis=1)

        sigma_AF_raw = _sd_softplus_inverse(enc[:, 9:10], self.sd_min)
        if self._signed_gains:
            g_HP_raw = enc[:, 10:11]
            g_LP_raw = enc[:, 11:12]
        else:
            g_HP_raw = tfp.math.softplus_inverse(enc[:, 10:11])
            g_LP_raw = tfp.math.softplus_inverse(enc[:, 11:12])

        sigma_dyn_raw = _sd_softplus_inverse(enc[:, 12:13], self.sd_min)
        if self._signed_gains:
            g_HP_dyn_raw = enc[:, 13:14]
            g_LP_dyn_raw = enc[:, 14:15]
            g_T_dyn_raw = enc[:, 15:16]
        else:
            g_HP_dyn_raw = tfp.math.softplus_inverse(enc[:, 13:14])
            g_LP_dyn_raw = tfp.math.softplus_inverse(enc[:, 14:15])
            g_T_dyn_raw = tfp.math.softplus_inverse(enc[:, 15:16])

        sigma_T_dyn_raw = sigma_dyn_raw   # raw-side tying for symmetry

        enc_raw = tf.concat(
            [prf_raw, sigma_AF_raw, g_HP_raw, g_LP_raw,
             sigma_dyn_raw, g_HP_dyn_raw, g_LP_dyn_raw,
             g_T_dyn_raw, sigma_T_dyn_raw], axis=1)

        if hrf_raw is not None:
            return tf.concat([enc_raw, hrf_raw], axis=1)
        return enc_raw

    # ----------------------------------------------------------------
    # Basis predictions: DN response on AF-modulated stimulus drive.
    # ----------------------------------------------------------------

    @tf.function
    def _basis_predictions(self, paradigm, parameters):
        # paradigm: (B, T, G). parameters: (B, V, 17).

        mod_sustained_per_cond = self._attention_modulation_sustained(parameters)
        ci = self._tf_condition_indicator                              # (T, n_C)
        mod_sustained_per_tr = tf.einsum('tc,bvcg->bvtg',
                                          ci, mod_sustained_per_cond)  # (B,V,T,G)

        mod_dyn = self._attention_modulation_dynamic(parameters)       # (T, G)
        mod_tgt = self._attention_modulation_target(parameters)        # (T, G)

        # `mod_sustained` already contains (1 + sign * sum). The
        # phasic terms add additional sign·field on top.
        sign = self._tf_sign
        full_mod = mod_sustained_per_tr + sign * (
            mod_dyn[tf.newaxis, tf.newaxis, :, :]
            + mod_tgt[tf.newaxis, tf.newaxis, :, :]
        )
        full_mod = tf.maximum(full_mod, 0.0)   # (B, V, T, G)

        eff_paradigm = paradigm[:, tf.newaxis, :, :] * full_mod

        # Build unit-amplitude center + surround Gaussians via `_get_rf`
        # (inherited from GaussianPRF2D). Pass dummy baseline=0,
        # amplitude=1 — same trick as DivisiveNormalizationGaussianPRF2D.
        mu_x = parameters[:, :, 0, tf.newaxis]
        mu_y = parameters[:, :, 1, tf.newaxis]
        sd = parameters[:, :, 2, tf.newaxis]
        srf_size = parameters[:, :, 5, tf.newaxis]

        rf_c_params = tf.concat(
            [mu_x, mu_y, sd, tf.zeros_like(mu_x), tf.ones_like(mu_x)], axis=2)
        rf_c = self._get_rf(self.grid_coordinates, rf_c_params)        # (B,V,G)

        rf_s_params = tf.concat(
            [mu_x, mu_y, sd * srf_size,
             tf.zeros_like(mu_x), tf.ones_like(mu_x)], axis=2)
        rf_s = self._get_rf(self.grid_coordinates, rf_s_params)        # (B,V,G)

        # eff_paradigm: (B,V,T,G), rf_*: (B,V,G) → (B,V,T)
        conv_c = tf.einsum('bvtg,bvg->bvt', eff_paradigm, rf_c)
        conv_s = tf.einsum('bvtg,bvg->bvt', eff_paradigm, rf_s)

        rf_amp = parameters[:, :, 3, tf.newaxis]       # (B,V,1), signed
        srf_amp = parameters[:, :, 4, tf.newaxis]      # > 0
        neural_bl = parameters[:, :, 6, tf.newaxis]    # > 0
        surround_bl = parameters[:, :, 7, tf.newaxis]  # > 0

        neural = rf_amp * conv_c + neural_bl                          # (B,V,T)
        suppress = tf.abs(rf_amp) * srf_amp * conv_s + surround_bl
        response = neural / suppress

        # NOTE: bold_baseline is NOT added here. It's added post-HRF in
        # `_predict` (see below), matching the DN parent's structure.

        return tf.transpose(response, perm=[0, 2, 1])   # (B, T, V)

    # ----------------------------------------------------------------
    # Override _predict: pass the FULL parameter vector to
    # _basis_predictions (the DN parent's _predict slices [..., :8],
    # which is wrong for our 17-encoding-param layout); handle DC
    # offset removal + HRF convolution + bold_baseline ourselves.
    # Mirrors DivisiveNormalizationGaussianPRF2DWithHRF._predict but
    # with the extra AF params left intact.
    # ----------------------------------------------------------------

    @tf.function
    def _predict(self, paradigm, parameters, weights):
        # Encoding pars only (strip HRF off the tail before forward
        # math — we re-attach via the HRF model below).
        if self.flexible_hrf_parameters:
            n_hrf = len(self.hrf_model.parameter_labels)
            enc_pars = parameters[:, :, :-n_hrf]
        else:
            enc_pars = parameters

        # AF-modulated DN response (B, T, V), without bold_baseline.
        pre_convolve = self._basis_predictions(paradigm, enc_pars)

        # Subtract DN's irreducible DC offset (neural_baseline /
        # surround_baseline) so the HRF convolution sees a zero-mean
        # signal at "no stim" timepoints. Matches the DN parent's
        # _predict.
        neural_bl = enc_pars[:, :, 6]                    # (B, V)
        surround_bl = enc_pars[:, :, 7]                  # (B, V)
        dc = (neural_bl / surround_bl)[:, tf.newaxis, :] # (B, 1, V)
        pre_convolve = pre_convolve - dc

        kwargs = {}
        if self.flexible_hrf_parameters:
            for ix, label in enumerate(self.hrf_model.parameter_labels):
                kwargs[label] = parameters[:, :, -n_hrf + ix]

        # HRF convolve.
        pred_convolved = self.hrf_model.convolve(pre_convolve, **kwargs)

        # bold_baseline is an additive constant at the BOLD level.
        bold_bl = enc_pars[:, tf.newaxis, :, 8]   # (B, 1, V)
        pred_convolved = pred_convolved + bold_bl

        return pred_convolved


# ---------------------------------------------------------------------------
# Klein-style ANALYTICAL SHIFT on a DoG backbone — 6-σ split, no gains.
# ---------------------------------------------------------------------------

class DoGKleinShift_v3_target_6sigma(
    DoGDynamicAttentionFieldPRF2DWithHRF_v3_target_sharedSigma,
):
    """DoG + analytical Klein-shift, 6 σ split (HP/LP for sus / dyn / target),
    no gain parameters.

    The forward model has NO multiplicative modulation of the stimulus
    drive. Instead, at each TR ``t`` and voxel ``v`` we shift the DoG
    center AND surround Gaussians' positions and sizes via Klein's
    Gaussian-product precision-weighted-mean formula, applied
    separately to center (σ_C = sd_v) and surround (σ_S = sd_v · srf_size_v).

    Per-TR precision contribution from each ring ℓ::

        c_ℓ(t) = ci_ℓ(t) · [ 1/σ_HP_sus²  +  d_ℓ(t)/σ_HP_dyn²
                            + tgt_ℓ(t)/σ_HP_T² ]
               + (1-ci_ℓ(t)) · [ 1/σ_LP_sus²  +  d_ℓ(t)/σ_LP_dyn²
                                + tgt_ℓ(t)/σ_LP_T² ]

    where ``ci_ℓ(t) ∈ {0, 1}`` says whether ring ℓ is the HP-of-the-
    current-condition at TR ``t``, ``d_ℓ(t) ∈ [0, 1]`` is the per-TR
    distractor-overlap fraction at ring ℓ, ``tgt_ℓ(t)`` is the per-TR
    target-overlap fraction.

    Then per voxel × TR (separately for center and surround)::

        τ_total_C(t, v) = 1/sd_v²                + Σ_ℓ c_ℓ(t)
        τ_total_S(t, v) = 1/(sd_v·srf_size_v)²    + Σ_ℓ c_ℓ(t)

        μ_C_eff(t, v)   = (μ_v/sd_v² + Σ_ℓ c_ℓ(t)·r_ℓ) / τ_total_C(t, v)
        μ_S_eff(t, v)   = (μ_v/(sd_v·srf_size_v)² + Σ_ℓ c_ℓ(t)·r_ℓ) /
                          τ_total_S(t, v)
        σ_C_eff(t, v)   = 1 / sqrt(τ_total_C(t, v))    # strict Klein
        σ_S_eff(t, v)   = 1 / sqrt(τ_total_S(t, v))

    BOLD response (area-preserving)::

        # Compensation factor preserves each unnormalized Gaussian's
        # integral (2π·σ²) at its at-rest value, so σ-shrinkage becomes a
        # *pure* shape change (narrower peak, taller peak) without
        # implicitly down-scaling the integrated PRF response. Without
        # this, σ_eff < sd_v would silently couple Klein-shifts to an
        # ~σ²-amplitude reduction at the attended location — the wrong
        # sign for retsupp's HP-attraction effect from AF+1.
        G̃_C(t, v, p)    = (sd_v² / σ_C_eff(t,v)²) · G(p; μ_C_eff, σ_C_eff)
        G̃_S(t, v, p)    = ((sd_v·srf_size_v)² / σ_S_eff(t,v)²)
                           · G(p; μ_S_eff, σ_S_eff)
        DoG_eff(t, v, p) = G̃_C(t, v, p) − srf_amp_v · G̃_S(t, v, p)
        response(t, v)   = amplitude_v · ∫ stim(t, p) · DoG_eff(t, v, p) dp
                         + baseline_v
        (then HRF convolve)

    Parameter layout (13 encoding + HRF)
    -------------------------------------
    Per-voxel (7):
        0 x, 1 y, 2 sd, 3 baseline, 4 amplitude,
        5 srf_amplitude, 6 srf_size.
    Shared (6):
        7 sigma_HP_sus, 8 sigma_LP_sus,
        9 sigma_HP_dyn, 10 sigma_LP_dyn,
        11 sigma_HP_T,  12 sigma_LP_T.

    No gain parameters anywhere; all per-ring "strength" is encoded by
    the σ values (smaller σ → larger precision contribution → bigger
    shift toward that ring).

    `mode` is ignored — pure Klein product has no sign; attraction is
    structural.
    """

    parameter_labels = [
        'x', 'y', 'sd', 'baseline', 'amplitude',
        'srf_amplitude', 'srf_size',
        'sigma_HP_sus', 'sigma_LP_sus',
        'sigma_HP_dyn', 'sigma_LP_dyn',
        'sigma_HP_T',   'sigma_LP_T',
    ]

    # Floor on total precision to avoid divisions by ~0.
    _TAU_FLOOR = 1e-6

    @tf.function
    def _per_tr_precision_centers(self, parameters):
        """Per-TR precision contribution & moment from the 6 σ's.

        Returns
        -------
        c_sum_t : (T,)   Σ_ℓ c_ℓ(t)
        c_x_t   : (T,)   Σ_ℓ c_ℓ(t) · r_ℓ_x
        c_y_t   : (T,)   Σ_ℓ c_ℓ(t) · r_ℓ_y
        """
        sigma_HP_sus = parameters[0, 0, 7]
        sigma_LP_sus = parameters[0, 0, 8]
        sigma_HP_dyn = parameters[0, 0, 9]
        sigma_LP_dyn = parameters[0, 0, 10]
        sigma_HP_T   = parameters[0, 0, 11]
        sigma_LP_T   = parameters[0, 0, 12]

        ci  = self._tf_condition_indicator   # (T, n_C) ∈ {0, 1}
        d   = self._tf_dynamic_indicator     # (T, n_C) ∈ [0, 1]
        tgt = self._tf_target_indicator      # (T, n_C) ∈ [0, 1]

        # Per-(t, ℓ) precision contribution.
        c_sus = ci         / (sigma_HP_sus ** 2) + (1.0 - ci) / (sigma_LP_sus ** 2)
        c_dyn = d * (ci    / (sigma_HP_dyn ** 2) + (1.0 - ci) / (sigma_LP_dyn ** 2))
        c_tgt = tgt * (ci  / (sigma_HP_T   ** 2) + (1.0 - ci) / (sigma_LP_T   ** 2))
        c_tl = c_sus + c_dyn + c_tgt          # (T, n_C)

        rx = self._tf_ring_positions[:, 0]   # (n_C,)
        ry = self._tf_ring_positions[:, 1]
        c_sum_t = tf.reduce_sum(c_tl, axis=1)             # (T,)
        c_x_t   = tf.einsum('tl,l->t', c_tl, rx)          # (T,)
        c_y_t   = tf.einsum('tl,l->t', c_tl, ry)          # (T,)
        return c_sum_t, c_x_t, c_y_t

    @tf.function
    def _basis_predictions(self, paradigm, parameters):
        # paradigm:   (B, T, G)
        # parameters: (B, V, 13[+n_hrf])
        T = tf.shape(paradigm)[1]

        mu_x_v = parameters[:, :, 0]         # (B, V)
        mu_y_v = parameters[:, :, 1]
        sd_v   = parameters[:, :, 2]
        baseline_v  = parameters[:, :, 3]
        amplitude_v = parameters[:, :, 4]
        srf_amp_v   = parameters[:, :, 5]
        srf_size_v  = parameters[:, :, 6]

        # Per-voxel center & surround precisions.
        tau_v_C = 1.0 / (sd_v ** 2)                                 # (B, V)
        sd_S_v  = sd_v * srf_size_v
        tau_v_S = 1.0 / (sd_S_v ** 2)                               # (B, V)

        # Per-TR shared precision contribution (depends on t, not v).
        c_sum_t, c_x_t, c_y_t = self._per_tr_precision_centers(parameters)

        # Per (B, V, T) total precision.
        tau_total_C = tau_v_C[:, :, tf.newaxis] + c_sum_t[tf.newaxis, tf.newaxis, :]
        tau_total_S = tau_v_S[:, :, tf.newaxis] + c_sum_t[tf.newaxis, tf.newaxis, :]
        tau_total_C_safe = tf.maximum(tau_total_C, self._TAU_FLOOR)
        tau_total_S_safe = tf.maximum(tau_total_S, self._TAU_FLOOR)

        # Per (B, V, T) shifted center & surround means.
        num_x_C = tau_v_C[:, :, tf.newaxis] * mu_x_v[:, :, tf.newaxis] \
                  + c_x_t[tf.newaxis, tf.newaxis, :]
        num_y_C = tau_v_C[:, :, tf.newaxis] * mu_y_v[:, :, tf.newaxis] \
                  + c_y_t[tf.newaxis, tf.newaxis, :]
        num_x_S = tau_v_S[:, :, tf.newaxis] * mu_x_v[:, :, tf.newaxis] \
                  + c_x_t[tf.newaxis, tf.newaxis, :]
        num_y_S = tau_v_S[:, :, tf.newaxis] * mu_y_v[:, :, tf.newaxis] \
                  + c_y_t[tf.newaxis, tf.newaxis, :]

        mu_C_x = num_x_C / tau_total_C_safe                         # (B, V, T)
        mu_C_y = num_y_C / tau_total_C_safe
        mu_S_x = num_x_S / tau_total_S_safe
        mu_S_y = num_y_S / tau_total_S_safe

        sigma_C_eff = 1.0 / tf.sqrt(tau_total_C_safe)               # (B, V, T)
        sigma_S_eff = 1.0 / tf.sqrt(tau_total_S_safe)

        # Vectorized inner loop: chunk over T (Python for-loop unrolled at
        # trace time) and materialize the full (B, V, Tc, G) Gaussian tensor
        # per chunk. Wrap the per-chunk forward in tf.recompute_grad so the
        # gradient tape doesn't retain the (B, V, Tc, G) intermediates —
        # they are recomputed during the backward pass. This is what makes
        # V=500 + T=3096 + G=4096 fit in ~10 GB instead of >150 GB, and
        # replaces tf.map_fn's per-TR Python dispatch (~8 s/iter at V=43)
        # with a single tensor op per chunk (target ~1 s/iter at V=500).
        gx = self._grid_coordinates[:, 0]                           # (G,)
        gy = self._grid_coordinates[:, 1]
        gx_b = gx[tf.newaxis, tf.newaxis, tf.newaxis, :]            # (1,1,1,G)
        gy_b = gy[tf.newaxis, tf.newaxis, tf.newaxis, :]
        srf_amp_b = srf_amp_v[:, :, tf.newaxis, tf.newaxis]         # (B,V,1,1)
        amp_b     = amplitude_v[:, :, tf.newaxis, tf.newaxis]
        # Area-preserving compensation: each unnormalized Gaussian's
        # integral is 2π·σ². Under Klein, σ_C_eff < sd_v (and σ_S_eff <
        # sd_v·srf_size_v), so without this scaling the σ-shrinkage would
        # *implicitly* shrink the integrated PRF response (~σ²) and act as
        # an unintended amplitude modulation. Multiplying by sd_v²/σ_eff²
        # holds each Gaussian's area fixed at its at-rest value, so Klein
        # becomes a pure position+dispersion modulation. At rest
        # (σ_eff = sd_v) the factor is 1 → behavior matches the parent
        # DoG forward exactly, preserving compatibility with m4-initialized
        # amplitude_v / srf_amplitude calibration.
        sd_v_b = sd_v[:, :, tf.newaxis, tf.newaxis]                 # (B,V,1,1)
        sd_S_v_b = sd_S_v[:, :, tf.newaxis, tf.newaxis]
        area_C0 = sd_v_b * sd_v_b                                   # (B,V,1,1)
        area_S0 = sd_S_v_b * sd_S_v_b

        @tf.recompute_grad
        def _chunk_forward(par_c, mu_Cx_c, mu_Cy_c, sC_c,
                            mu_Sx_c, mu_Sy_c, sS_c):
            # par_c:   (B, Tc, G)
            # mu_*_c:  (B, V, Tc, 1) ; sC_c, sS_c: (B, V, Tc, 1)
            dx_C = gx_b - mu_Cx_c                                   # (B,V,Tc,G)
            dy_C = gy_b - mu_Cy_c
            G_C  = (area_C0 / (sC_c * sC_c)) \
                   * tf.exp(-(dx_C * dx_C + dy_C * dy_C)
                            / (2.0 * sC_c * sC_c))

            dx_S = gx_b - mu_Sx_c
            dy_S = gy_b - mu_Sy_c
            G_S  = (area_S0 / (sS_c * sS_c)) \
                   * tf.exp(-(dx_S * dx_S + dy_S * dy_S)
                            / (2.0 * sS_c * sS_c))

            dog = (G_C - srf_amp_b * G_S) * amp_b                   # (B,V,Tc,G)
            # pred[b, v, tc] = sum_g paradigm[b, tc, g] · dog[b, v, tc, g]
            return tf.einsum('btg,bvtg->bvt', par_c, dog)           # (B,V,Tc)

        T_static = paradigm.shape[1]
        T_CHUNK = 64
        if T_static is None:
            # Fallback if T is dynamic at trace time; one chunk = whole T.
            t_starts = [0]
            chunk_sizes = [T]
        else:
            t_starts = list(range(0, T_static, T_CHUNK))
            chunk_sizes = [min(T_CHUNK, T_static - s) for s in t_starts]

        pred_chunks = []
        for t0, tc in zip(t_starts, chunk_sizes):
            par_c   = paradigm[:, t0:t0 + tc, :]                    # (B,Tc,G)
            mu_Cx_c = mu_C_x[:, :, t0:t0 + tc, tf.newaxis]          # (B,V,Tc,1)
            mu_Cy_c = mu_C_y[:, :, t0:t0 + tc, tf.newaxis]
            sC_c    = sigma_C_eff[:, :, t0:t0 + tc, tf.newaxis]
            mu_Sx_c = mu_S_x[:, :, t0:t0 + tc, tf.newaxis]
            mu_Sy_c = mu_S_y[:, :, t0:t0 + tc, tf.newaxis]
            sS_c    = sigma_S_eff[:, :, t0:t0 + tc, tf.newaxis]
            pred_chunks.append(
                _chunk_forward(par_c, mu_Cx_c, mu_Cy_c, sC_c,
                                mu_Sx_c, mu_Sy_c, sS_c)
            )

        pred = tf.concat(pred_chunks, axis=2)                       # (B,V,T)
        pred = tf.transpose(pred, perm=[0, 2, 1])                   # (B,T,V)
        return pred + baseline_v[:, tf.newaxis, :]

    # ----- Parameter transforms (6 σ's, all softplus, no gain params) ------

    @tf.function
    def _transform_parameters_forward(self, parameters):
        # Per-voxel slots 0..6 — same as DoG parent's encoding (no transform
        # except sd which needs softplus with sd_min). srf_amplitude and
        # srf_size also need positivity (softplus). amplitude is signed.
        # We use _sd_softplus_forward for σ-like values; identity for x, y;
        # softplus for srf_amplitude, srf_size; identity for amplitude.
        x = parameters[:, 0:1]
        y = parameters[:, 1:2]
        sd = _sd_softplus_forward(parameters[:, 2:3], self.sd_min)
        baseline = parameters[:, 3:4]
        amplitude = parameters[:, 4:5]
        srf_amplitude = tf.math.softplus(parameters[:, 5:6])
        srf_size = tf.math.softplus(parameters[:, 6:7]) + 1.0
        # 6 σ's, all softplus with sd_min floor.
        sigmas_raw = parameters[:, 7:13]
        sigmas = _sd_softplus_forward(sigmas_raw, self.sd_min)

        enc = tf.concat([x, y, sd, baseline, amplitude,
                          srf_amplitude, srf_size, sigmas], axis=1)

        if self.flexible_hrf_parameters:
            n_hrf = len(self.hrf_model.parameter_labels)
            hrf_pars = self.hrf_model._transform_parameters_forward(
                parameters[:, -n_hrf:])
            return tf.concat([enc, hrf_pars], axis=1)
        return enc

    @tf.function
    def _transform_parameters_backward(self, parameters):
        x = parameters[:, 0:1]
        y = parameters[:, 1:2]
        sd = _sd_softplus_inverse(parameters[:, 2:3], self.sd_min)
        baseline = parameters[:, 3:4]
        amplitude = parameters[:, 4:5]
        srf_amplitude = tfp.math.softplus_inverse(parameters[:, 5:6])
        srf_size = tfp.math.softplus_inverse(parameters[:, 6:7] - 1.0)
        sigmas_raw = _sd_softplus_inverse(parameters[:, 7:13], self.sd_min)
        enc = tf.concat([x, y, sd, baseline, amplitude,
                          srf_amplitude, srf_size, sigmas_raw], axis=1)
        if self.flexible_hrf_parameters:
            n_hrf = len(self.hrf_model.parameter_labels)
            hrf_pars = self.hrf_model._transform_parameters_backward(
                parameters[:, -n_hrf:])
            return tf.concat([enc, hrf_pars], axis=1)
        return enc
