"""Fit attention-field models (AF and AF+) to conditionwise PRF parameters.

NOTE: parameters are bounded and reparameterized to keep the optimizer in
physically sensible territory:
    - μ_AF is parameterized as (ecc_C, θ_C) with ecc_C ∈ [0, 4°] (never
      outside the distractor ring).
    - log(σ_AF) ∈ [log(0.5), log(15)] (covers narrow focal attention
      through broad spatial spreads; AF can be much wider than PRFs).
    - log(b/a) ∈ [-3, 3] for AF+ (between near-pure-AF and near-no-AF).
    - Multi-start from 4 initializations is used to escape local optima.

R² is reported as the *within-voxel* version: ss_residual / ss_within
where ss_within is the sum over voxels of the within-voxel-across-condition
variance. R² > 0 means the model explains real conditionwise variation;
R² ≈ 0 means the fit is degenerate (μ_AD ≈ mean across conditions per
voxel, the same as the "no AF" solution).

For each voxel × HP condition we observe a 2-D PRF center (x_v_C, y_v_C).
We fit a generative model that explains these observations as a base PRF
position perturbed by a condition-specific attention/suppression field:

  Simple AF (no offset, closed-form Gaussian × Gaussian):
      μ_AD_v_C = (μ_AF_C · σ_base_v² + base_v · σ_AF²) / (σ_AF² + σ_base_v²)

  AF+ (multiplicative offset; AD-pRF is a mixture of two Gaussians,
        center taken as center-of-mass):
      μ_AD_v_C = w(Δ) · μ_AD_simpleAF  +  (1 − w(Δ)) · base_v
      w(Δ) = 1 / (1 + (b/a) · (σ_AF² + σ_base²)/σ_AF²
                          · exp(Δ²/(2(σ_AF² + σ_base²))))
      Δ = ‖base_v − μ_AF_C‖

For both models we fit per subject:
    - Per voxel: (x0_v, y0_v) base PRF position. σ_base_v is fixed at the
      voxel's `sd_mean_model` (independent mean-model fit, condition-blind).
    - Shared: σ_AF (one value per ROI/subject).
    - Shared: 4 free AF centers μ_AF_C ∈ ℝ², one per condition. They are
      free parameters so the fit can verify whether they land at the
      expected geometric positions (e.g. antipodes of HPs under a
      suppression interpretation).
    - AF+ adds: log(b/a) — the offset ratio governing locality.

The trick that keeps this tractable: given the outer (shared) parameters,
the per-voxel base position has a *closed-form* MSE optimum, so we only
need to optimize the few outer parameters numerically.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import minimize


@dataclass
class AFFitResult:
    """Container for AF / AF+ fit output."""
    model: str  # "AF" or "AF+"
    sigma_AF: float
    mu_AF: np.ndarray            # (n_conditions, 2)
    base: np.ndarray             # (N, 2) per-voxel base position
    log_b_over_a: Optional[float]  # AF+ only
    predicted: np.ndarray        # (N, n_conditions, 2)
    observed: np.ndarray         # (N, n_conditions, 2)
    sigma_base: np.ndarray       # (N,)
    r2: float
    loss: float
    success: bool
    n_params: int
    n_obs: int

    @property
    def aic(self) -> float:
        # AIC under Gaussian-residual assumption: n·log(loss/n) + 2·k
        n = self.n_obs
        k = self.n_params
        return n * np.log(self.loss / n) + 2 * k


def _solve_base(
    observed: np.ndarray,
    alpha: np.ndarray,
    mu_AF: np.ndarray,
) -> np.ndarray:
    """Closed-form per-voxel base for the simple AF model.

    Given α_v and μ_AF_C, the model is
        observed_v_C = α_v · μ_AF_C + (1 − α_v) · base_v
    so the MSE-optimal base is
        base_v = ( mean_obs_v − α_v · mean_AF ) / (1 − α_v)
    where the means are taken over conditions.
    """
    mean_obs = observed.mean(axis=1)        # (N, 2)
    mean_AF = mu_AF.mean(axis=0)            # (2,)
    return (mean_obs - alpha[:, None] * mean_AF[None, :]) / (1 - alpha[:, None])


def _within_voxel_r2(observed: np.ndarray, predicted: np.ndarray) -> float:
    """R² of conditionwise variation around the per-voxel mean.

    Zero for the degenerate solution (predictions = per-voxel mean across
    conditions); positive when the model explains real conditionwise
    variation; can be negative if the model is worse than the voxel-mean
    baseline.
    """
    voxel_mean = observed.mean(axis=1, keepdims=True)
    ss_res = float(((observed - predicted) ** 2).sum())
    ss_within = float(((observed - voxel_mean) ** 2).sum())
    if ss_within == 0:
        return np.nan
    return 1 - ss_res / ss_within


def _polar_to_xy(eccs: np.ndarray, thetas: np.ndarray) -> np.ndarray:
    """(C,), (C,) -> (C, 2)  Cartesian AF centers."""
    return np.stack([eccs * np.cos(thetas), eccs * np.sin(thetas)], axis=1)


def _af_unpack(theta: np.ndarray, n_conditions: int):
    """Unpack [log_sigma_AF, ecc_0..ecc_C, theta_0..theta_C]."""
    log_sigma_AF = theta[0]
    eccs = theta[1:1 + n_conditions]
    thetas = theta[1 + n_conditions:1 + 2 * n_conditions]
    sigma_AF = float(np.exp(log_sigma_AF))
    mu_AF = _polar_to_xy(eccs, thetas)
    return sigma_AF, mu_AF


def _af_loss(theta, observed, sigma_base):
    n_conditions = observed.shape[1]
    sigma_AF, mu_AF = _af_unpack(theta, n_conditions)
    alpha = sigma_base ** 2 / (sigma_base ** 2 + sigma_AF ** 2)
    base = _solve_base(observed, alpha, mu_AF)
    pred = (alpha[:, None, None] * mu_AF[None, :, :]
            + (1 - alpha[:, None, None]) * base[:, None, :])
    return float(((observed - pred) ** 2).sum())


def _af_initial_thetas(
    n_conditions: int,
    sigma_AF_init: float,
    af_at_minus_hp: bool,
) -> np.ndarray:
    """Build a polar-parameterized init vector."""
    from retsupp.utils.data import distractor_locations, location_angles
    cond_keys = ["upper right", "upper left", "lower left", "lower right"]
    angles = np.array([location_angles[k] for k in cond_keys])
    eccs = np.array([np.hypot(*distractor_locations[k]) for k in cond_keys])
    if af_at_minus_hp:
        # Antipodes: angle + π, same eccentricity.
        angles = (angles + np.pi) % (2 * np.pi)
    return np.concatenate([
        [np.log(sigma_AF_init)],
        eccs,
        angles,
    ])


_AF_BOUNDS = lambda n_C: (
    [(np.log(0.5), np.log(15.0))]              # log_sigma_AF
    + [(0.0, 4.0)] * n_C                        # eccentricity per condition
    + [(-4 * np.pi, 4 * np.pi)] * n_C           # angle per condition
)


def fit_af(
    observed: np.ndarray,
    sigma_base: np.ndarray,
    init_mu_AF_at_minus_hp: bool = True,
    multi_start: bool = True,
) -> AFFitResult:
    """Fit the simple AF model with bounds + multi-start.

    observed: shape (N, n_conditions, 2)
    sigma_base: shape (N,)
    """
    n_conditions = observed.shape[1]
    bounds = _AF_BOUNDS(n_conditions)

    inits = []
    if multi_start:
        for sigma_init in (1.0, 3.0):
            for at_minus in (True, False):
                inits.append(_af_initial_thetas(n_conditions, sigma_init, at_minus))
    else:
        inits.append(_af_initial_thetas(n_conditions, 2.0, init_mu_AF_at_minus_hp))

    best = None
    for theta0 in inits:
        try:
            res = minimize(_af_loss, theta0, args=(observed, sigma_base),
                           method="L-BFGS-B", bounds=bounds,
                           options={"maxiter": 1000})
            if best is None or res.fun < best.fun:
                best = res
        except Exception:
            continue

    sigma_AF, mu_AF = _af_unpack(best.x, n_conditions)
    alpha = sigma_base ** 2 / (sigma_base ** 2 + sigma_AF ** 2)
    base = _solve_base(observed, alpha, mu_AF)
    pred = (alpha[:, None, None] * mu_AF[None, :, :]
            + (1 - alpha[:, None, None]) * base[:, None, :])

    ss_res = float(((observed - pred) ** 2).sum())
    r2 = _within_voxel_r2(observed, pred)
    n_obs = observed.size
    n_params = 2 * len(observed) + 1 + 2 * n_conditions
    return AFFitResult(
        model="AF", sigma_AF=float(sigma_AF), mu_AF=mu_AF, base=base,
        log_b_over_a=None, predicted=pred, observed=observed,
        sigma_base=sigma_base, r2=r2, loss=ss_res,
        success=bool(best.success), n_params=n_params, n_obs=n_obs,
    )


def _af_plus_predict_centers(
    base: np.ndarray,
    sigma_base: np.ndarray,
    mu_AF: np.ndarray,
    sigma_AF: float,
    b_over_a: float,
):
    """Closed-form AD-pRF center-of-mass for the AF+ model.

    base: (N, 2)
    sigma_base: (N,)
    mu_AF: (n_conditions, 2)
    sigma_AF: scalar
    b_over_a: scalar offset ratio (must be > 0).
    """
    diff = base[:, None, :] - mu_AF[None, :, :]            # (N, C, 2)
    delta_sq = (diff ** 2).sum(axis=-1)                     # (N, C)

    sigma_base_sq = sigma_base[:, None] ** 2
    sigma_sum_sq = sigma_AF ** 2 + sigma_base_sq            # (N, 1)

    # Simple-AF AD-pRF center.
    mu_simple = (
        mu_AF[None, :, :] * sigma_base_sq[:, :, None]
        + base[:, None, :] * sigma_AF ** 2
    ) / sigma_sum_sq[:, :, None]                            # (N, C, 2)

    # Mixing weight w. Cap exponent for numerical stability.
    log_arg = delta_sq / (2.0 * sigma_sum_sq)
    log_arg = np.clip(log_arg, -50.0, 50.0)                 # avoid overflow
    ratio_factor = sigma_sum_sq / sigma_AF ** 2             # (N, 1)
    w = 1.0 / (1.0 + b_over_a * ratio_factor * np.exp(log_arg))

    pred = w[:, :, None] * mu_simple + (1 - w[:, :, None]) * base[:, None, :]
    return pred, w


def _af_suppression_predict_centers(
    base: np.ndarray,
    sigma_base: np.ndarray,
    mu_AF: np.ndarray,
    sigma_AF: float,
    b: float,
):
    """Closed-form AD-pRF center-of-mass for the suppressive AF model.

    R(x) = (b − A(x)) · S(x) where A is a Gaussian peaked at μ_AF and
    S is the SD-pRF. Requires b > 1 so R remains positive.

    The shift is in direction (base − μ_AF) — away from μ_AF — and
    decays to zero for voxels far from μ_AF (locality property).
    """
    diff = base[:, None, :] - mu_AF[None, :, :]            # (N, C, 2)
    delta_sq = (diff ** 2).sum(axis=-1)                     # (N, C)

    sigma_base_sq = sigma_base[:, None] ** 2
    sigma_sum_sq = sigma_AF ** 2 + sigma_base_sq            # (N, 1)

    # Overlap fraction ρ = σ_AF²/(σ_AF² + σ_base²) · exp(-Δ²/(2(σ_AF² + σ_base²)))
    sigma_ratio = sigma_AF ** 2 / sigma_sum_sq              # (N, 1)
    log_arg = -delta_sq / (2.0 * sigma_sum_sq)
    log_arg = np.clip(log_arg, -50.0, 50.0)
    rho = sigma_ratio * np.exp(log_arg)                     # (N, C)

    # Shift coefficient: σ_base²/(σ_AF² + σ_base²) · ρ/(b − ρ)
    base_factor = sigma_base_sq / sigma_sum_sq              # (N, 1)
    denom = np.maximum(b - rho, 1e-9)                       # avoid /0
    coef = base_factor * rho / denom                         # (N, C)

    pred = base[:, None, :] + diff * coef[:, :, None]
    return pred, rho, coef


def _af_supp_unpack(theta, n_conditions):
    """[log_sigma_AF, log(b-1), ecc_0..ecc_C, theta_0..theta_C]."""
    log_sigma_AF = theta[0]
    log_b_minus_1 = theta[1]
    eccs = theta[2:2 + n_conditions]
    thetas = theta[2 + n_conditions:2 + 2 * n_conditions]
    sigma_AF = float(np.exp(log_sigma_AF))
    b = float(1.0 + np.exp(log_b_minus_1))
    mu_AF = _polar_to_xy(eccs, thetas)
    return sigma_AF, b, mu_AF


def _af_supp_loss(theta, observed, sigma_base):
    n_conditions = observed.shape[1]
    sigma_AF, b, mu_AF = _af_supp_unpack(theta, n_conditions)
    # For suppressive AF, the per-voxel base optimum has no clean closed
    # form. Use the AF+-style trick: solve for base under the simple-AF
    # *attractive* approximation; this provides a reasonable starting
    # estimate, refined inside the predict step.
    alpha = sigma_base ** 2 / (sigma_base ** 2 + sigma_AF ** 2)
    base = _solve_base(observed, alpha, mu_AF)
    pred, _, _ = _af_suppression_predict_centers(base, sigma_base, mu_AF, sigma_AF, b)
    return float(((observed - pred) ** 2).sum())


_AF_SUPP_BOUNDS = lambda n_C: (
    [(np.log(0.5), np.log(15.0))]              # log_sigma_AF
    + [(-3.0, 3.0)]                             # log(b-1)  → b ∈ (1.05, 21)
    + [(0.0, 4.0)] * n_C                        # ecc per condition
    + [(-4 * np.pi, 4 * np.pi)] * n_C           # angle per condition
)


def _af_supp_initial_thetas(
    n_conditions: int,
    sigma_AF_init: float,
    b_init: float,
    af_at_hp: bool = True,
) -> np.ndarray:
    """Init AF centers AT HP (not at antipodes!) under the suppressive model."""
    base = _af_initial_thetas(n_conditions, sigma_AF_init, af_at_minus_hp=not af_at_hp)
    return np.concatenate([base[:1], [np.log(max(b_init - 1.0, 0.05))], base[1:]])


def fit_af_suppression(
    observed: np.ndarray,
    sigma_base: np.ndarray,
    multi_start: bool = True,
) -> AFFitResult:
    """Fit suppressive AF model: μ_AF lives at HP (not antipode); voxels
    pushed away from μ_AF; effect localized via the offset b > 1.
    """
    n_conditions = observed.shape[1]
    bounds = _AF_SUPP_BOUNDS(n_conditions)

    inits = []
    if multi_start:
        for sigma_init in (1.0, 2.0, 4.0):
            for b_init in (1.5, 3.0, 8.0):
                inits.append(_af_supp_initial_thetas(
                    n_conditions, sigma_init, b_init, af_at_hp=True,
                ))
    else:
        inits.append(_af_supp_initial_thetas(n_conditions, 2.0, 2.0, True))

    best = None
    for theta0 in inits:
        try:
            res = minimize(_af_supp_loss, theta0, args=(observed, sigma_base),
                           method="L-BFGS-B", bounds=bounds,
                           options={"maxiter": 2000})
            if best is None or res.fun < best.fun:
                best = res
        except Exception:
            continue

    sigma_AF, b, mu_AF = _af_supp_unpack(best.x, n_conditions)
    alpha = sigma_base ** 2 / (sigma_base ** 2 + sigma_AF ** 2)
    base = _solve_base(observed, alpha, mu_AF)
    pred, _, _ = _af_suppression_predict_centers(base, sigma_base, mu_AF, sigma_AF, b)

    ss_res = float(((observed - pred) ** 2).sum())
    r2 = _within_voxel_r2(observed, pred)
    n_obs = observed.size
    n_params = 2 * len(observed) + 2 + 2 * n_conditions
    return AFFitResult(
        model="AF-supp", sigma_AF=float(sigma_AF), mu_AF=mu_AF, base=base,
        log_b_over_a=float(np.log(b - 1.0)),  # store log(b-1)
        predicted=pred, observed=observed, sigma_base=sigma_base,
        r2=r2, loss=ss_res, success=bool(best.success),
        n_params=n_params, n_obs=n_obs,
    )


def _af_plus_unpack(theta, n_conditions):
    """[log_sigma_AF, log_b_over_a, ecc_0..ecc_C, theta_0..theta_C]."""
    log_sigma_AF = theta[0]
    log_b_over_a = theta[1]
    eccs = theta[2:2 + n_conditions]
    thetas = theta[2 + n_conditions:2 + 2 * n_conditions]
    sigma_AF = float(np.exp(log_sigma_AF))
    b_over_a = float(np.exp(log_b_over_a))
    mu_AF = _polar_to_xy(eccs, thetas)
    return sigma_AF, b_over_a, mu_AF


def _af_plus_loss(theta, observed, sigma_base):
    n_conditions = observed.shape[1]
    sigma_AF, b_over_a, mu_AF = _af_plus_unpack(theta, n_conditions)
    alpha = sigma_base ** 2 / (sigma_base ** 2 + sigma_AF ** 2)
    base = _solve_base(observed, alpha, mu_AF)
    pred, _ = _af_plus_predict_centers(base, sigma_base, mu_AF, sigma_AF, b_over_a)
    return float(((observed - pred) ** 2).sum())


_AF_PLUS_BOUNDS = lambda n_C: (
    [(np.log(0.5), np.log(15.0))]              # log_sigma_AF
    + [(-3.0, 3.0)]                             # log_b_over_a
    + [(0.0, 4.0)] * n_C                        # ecc per condition
    + [(-4 * np.pi, 4 * np.pi)] * n_C           # angle per condition
)


def _af_plus_initial_thetas(
    n_conditions: int,
    sigma_AF_init: float,
    b_over_a_init: float,
    af_at_minus_hp: bool,
) -> np.ndarray:
    base = _af_initial_thetas(n_conditions, sigma_AF_init, af_at_minus_hp)
    return np.concatenate([base[:1], [np.log(b_over_a_init)], base[1:]])


def fit_af_plus(
    observed: np.ndarray,
    sigma_base: np.ndarray,
    multi_start: bool = True,
) -> AFFitResult:
    """Fit the AF+ model (multiplicative offset) with bounds + multi-start."""
    n_conditions = observed.shape[1]
    bounds = _AF_PLUS_BOUNDS(n_conditions)

    inits = []
    if multi_start:
        for sigma_init in (1.0, 3.0):
            for ba_init in (0.3, 2.0):
                for at_minus in (True, False):
                    inits.append(_af_plus_initial_thetas(
                        n_conditions, sigma_init, ba_init, at_minus,
                    ))
    else:
        inits.append(_af_plus_initial_thetas(n_conditions, 2.0, 1.0, True))

    best = None
    for theta0 in inits:
        try:
            res = minimize(_af_plus_loss, theta0, args=(observed, sigma_base),
                           method="L-BFGS-B", bounds=bounds,
                           options={"maxiter": 2000})
            if best is None or res.fun < best.fun:
                best = res
        except Exception:
            continue

    sigma_AF, b_over_a, mu_AF = _af_plus_unpack(best.x, n_conditions)
    alpha = sigma_base ** 2 / (sigma_base ** 2 + sigma_AF ** 2)
    base = _solve_base(observed, alpha, mu_AF)
    pred, _ = _af_plus_predict_centers(base, sigma_base, mu_AF, sigma_AF, b_over_a)

    ss_res = float(((observed - pred) ** 2).sum())
    r2 = _within_voxel_r2(observed, pred)
    n_obs = observed.size
    n_params = 2 * len(observed) + 2 + 2 * n_conditions
    return AFFitResult(
        model="AF+", sigma_AF=float(sigma_AF), mu_AF=mu_AF, base=base,
        log_b_over_a=float(np.log(b_over_a)), predicted=pred, observed=observed,
        sigma_base=sigma_base, r2=r2, loss=ss_res,
        success=bool(best.success), n_params=n_params, n_obs=n_obs,
    )


def fit_four_af_competing(
    observed: np.ndarray,
    sigma_base: np.ndarray,
    ring_positions: np.ndarray | None = None,
    hp_indices_per_condition: np.ndarray | None = None,
    sigma_bounds: tuple = (0.5, 8.0),
    g_bounds: tuple = (0.001, 2.0),
    mode: str = "suppression",
):
    """Fit the 4-AF competing suppression model.

    For each voxel `v` in condition `C` with HP ring index `H_C`:

        suppression: R_v_C(x) = (1 − g_HP · A_{H_C}(x) − g_LP · Σ_{ℓ ≠ H_C} A_ℓ(x)) · S_v(x)
        attraction:  R_v_C(x) = (1 + g_HP · A_{H_C}(x) + g_LP · Σ_{ℓ ≠ H_C} A_ℓ(x)) · S_v(x)

    where A_ℓ(x) is a Gaussian centered at ring position ℓ with shared
    size σ_AF, and S_v(x) is the SD-pRF Gaussian at base_v with σ_base_v.
    All four AFs are present in every condition; only the *amplitude* of
    the AF at the HP location differs from the AFs at the three LP
    locations.

    `mode='suppression'` (default): voxels pushed AWAY from each ring
    location, more strongly from HP than from LP. Predicted shifts of
    PRFs near a ring point away from that ring.

    `mode='attraction'`: voxels pulled TOWARD each ring location, more
    strongly toward HP. The Sumiya AF+ analog with positive amplitudes.

    Center of mass closed form (sum/difference of Gaussian-Gaussian
    products, all analytically tractable):

        μ_AD_v_C = ( I_S · base_v − Σ_ℓ g_C_ℓ · I_Aℓ_S · μ_AD_ℓ_v )
                  / ( I_S         − Σ_ℓ g_C_ℓ · I_Aℓ_S         )

        I_S        = ∫ S_v
        I_Aℓ_S     = ∫ A_ℓ · S_v
                   = (2π σ_AF² σ_base² / (σ_AF² + σ_base²))
                     · exp(−Δ_ℓ² / (2(σ_AF² + σ_base²)))
        μ_AD_ℓ_v   = (ring_ℓ · σ_base² + base_v · σ_AF²) / (σ_AF² + σ_base²)
        Δ_ℓ        = ‖base_v − ring_ℓ‖

    Free parameters (3 total per ROI per subject):
        σ_AF        ∈ [0.5, 8] deg
        g_HP        ∈ [0.001, 2]   suppression amplitude at HP
        g_LP        ∈ [0.001, 2]   suppression amplitude at the three LP locations

    Per-voxel base position is fixed at the mean across conditions
    (closed-form approximation; works because the 4-condition design has
    the property that the average AF effect across conditions ≈ 0 by
    ring symmetry).

    Returns dict with σ_AF, g_HP, g_LP, ratio, within-voxel R², loss.
    """
    from scipy.optimize import minimize
    import pandas as pd

    if mode not in ("suppression", "attraction"):
        raise ValueError(f"mode must be 'suppression' or 'attraction', got {mode!r}")
    sign = -1.0 if mode == "suppression" else +1.0

    if ring_positions is None:
        from retsupp.utils.data import distractor_locations
        keys = ["upper right", "upper left", "lower left", "lower right"]
        ring_positions = np.array([list(distractor_locations[k]) for k in keys])
    if hp_indices_per_condition is None:
        # CONDITIONS = ["upper_right", "upper_left", "lower_left", "lower_right"]
        hp_indices_per_condition = np.array([0, 1, 2, 3])

    n_voxels, n_conditions, _ = observed.shape
    base = observed.mean(axis=1)  # (N, 2) approximate base position
    sigma_base_sq = sigma_base ** 2

    # Precompute distance² from each voxel's base to each ring position. (N, 4)
    diff = base[:, None, :] - ring_positions[None, :, :]
    delta_sq = (diff ** 2).sum(axis=-1)

    # Mask: g_per_condition_ring[C, ℓ] = 1 if ℓ == H_C (HP), else 0.
    is_hp = np.zeros((n_conditions, len(ring_positions)))
    is_hp[np.arange(n_conditions), hp_indices_per_condition] = 1.0  # (n_C, 4)

    def predict(theta):
        log_sigma_AF, log_g_HP, log_g_LP = theta
        sigma_AF = float(np.exp(log_sigma_AF))
        g_HP = float(np.exp(log_g_HP))
        g_LP = float(np.exp(log_g_LP))

        sigma_sum_sq = sigma_AF ** 2 + sigma_base_sq                    # (N,)
        # Z_v = 2π σ_AF² σ_base² / σ_sum_sq
        Z = 2 * np.pi * sigma_AF ** 2 * sigma_base_sq / sigma_sum_sq    # (N,)
        log_arg = -delta_sq / (2 * sigma_sum_sq[:, None])
        log_arg = np.clip(log_arg, -50.0, 50.0)
        I_AS = Z[:, None] * np.exp(log_arg)                              # (N, 4)

        I_S = 2 * np.pi * sigma_base_sq                                  # (N,)

        # μ_AD_ℓ_v: weighted mean of base and ring_ℓ
        mu_AD_ring = (
            ring_positions[None, :, :] * sigma_base_sq[:, None, None]
            + base[:, None, :] * sigma_AF ** 2
        ) / sigma_sum_sq[:, None, None]                                  # (N, 4, 2)

        # g[C, ℓ] = g_HP if ℓ == H_C, else g_LP. Shape (n_C, 4)
        g_C_ℓ = is_hp * g_HP + (1 - is_hp) * g_LP                        # (n_C, 4)
        # weighted_I_AS[v, c, ℓ] = g[c, ℓ] · I_AS[v, ℓ]
        weighted_I_AS = g_C_ℓ[None, :, :] * I_AS[:, None, :]             # (N, n_C, 4)
        weighted_term = (
            weighted_I_AS[:, :, :, None] * mu_AD_ring[:, None, :, :]
        ).sum(axis=2)                                                    # (N, n_C, 2)
        # `sign`: +1 for attraction, -1 for suppression. Both produce a
        # local effect because of the SD-pRF baseline (I_S[v]·base[v]).
        numer = I_S[:, None, None] * base[:, None, :] + sign * weighted_term     # (N, n_C, 2)
        denom = I_S[:, None] + sign * weighted_I_AS.sum(axis=2)                  # (N, n_C)
        # Avoid divide-by-near-zero (b=1, sum of suppressions can approach 1).
        denom = np.where(np.abs(denom) < 1e-9, 1e-9, denom)
        return numer / denom[:, :, None]

    def loss(theta):
        pred = predict(theta)
        return float(((observed - pred) ** 2).sum())

    # Initial guess; multi-start to reduce local-minimum risk.
    inits = []
    for sigma_init in (1.0, 2.5, 5.0):
        for g_HP_init, g_LP_init in [(0.2, 0.05), (0.5, 0.1), (0.1, 0.1)]:
            inits.append(np.array([
                np.log(sigma_init),
                np.log(g_HP_init),
                np.log(g_LP_init),
            ]))
    bounds = [
        (np.log(sigma_bounds[0]), np.log(sigma_bounds[1])),
        (np.log(g_bounds[0]),     np.log(g_bounds[1])),
        (np.log(g_bounds[0]),     np.log(g_bounds[1])),
    ]

    best = None
    for theta0 in inits:
        try:
            res = minimize(
                loss, theta0, method="L-BFGS-B", bounds=bounds,
                options={"maxiter": 1000},
            )
            if best is None or res.fun < best.fun:
                best = res
        except Exception:
            continue

    sigma_AF = float(np.exp(best.x[0]))
    g_HP = float(np.exp(best.x[1]))
    g_LP = float(np.exp(best.x[2]))

    pred = predict(best.x)
    voxel_mean = observed.mean(axis=1, keepdims=True)
    ss_res = float(((observed - pred) ** 2).sum())
    ss_within = float(((observed - voxel_mean) ** 2).sum())
    r2 = 1 - ss_res / ss_within if ss_within > 0 else np.nan

    return {
        "mode": mode,
        "sigma_AF": sigma_AF,
        "g_HP": g_HP,
        "g_LP": g_LP,
        "g_HP_over_g_LP": g_HP / g_LP,
        "log_g_HP_over_g_LP": float(np.log(g_HP / g_LP)),
        "r2": r2,
        "loss": float(best.fun),
        "success": bool(best.success),
        "n_voxels": n_voxels,
    }


def fit_four_af_per_subject(
    pars, rois, modes=("suppression", "attraction"),
    conditions=("upper_right", "upper_left", "lower_left", "lower_right"),
):
    """Fit the 4-AF competing model per subject per ROI, both modes.

    Returns DataFrame with one row per (subject, roi, mode).
    """
    import pandas as pd
    work = pars.reset_index()
    work = work[work["roi_base"].isin(rois)]

    rows = []
    for sub in sorted(work["subject"].unique()):
        for roi in rois:
            sub_df = work[(work["subject"] == sub) & (work["roi_base"] == roi)]
            if len(sub_df) < 40:
                continue

            pivot = sub_df.pivot_table(
                index=["roi_base", "roi", "voxel"],
                columns="condition", values=["x", "y", "sd_mean_model"],
            )
            keep = pivot[("x", conditions[0])].notna()
            for c in conditions:
                keep &= pivot[("x", c)].notna() & pivot[("y", c)].notna()
            keep &= pivot[("sd_mean_model", conditions[0])].notna()
            pivot = pivot[keep]
            if len(pivot) < 40:
                continue

            obs_arrays = []
            for cond in conditions:
                obs_arrays.append(np.stack([
                    pivot[("x", cond)].values,
                    pivot[("y", cond)].values,
                ], axis=1))
            obs = np.stack(obs_arrays, axis=1)               # (N, 4, 2)
            sigma_base = pivot[("sd_mean_model", conditions[0])].values
            sigma_base = np.where(sigma_base > 0.05, sigma_base, 0.05)

            for mode in modes:
                try:
                    fit = fit_four_af_competing(obs, sigma_base, mode=mode)
                    rows.append({"subject": sub, "roi": roi, **fit})
                except Exception as e:
                    print(f"  fit failed for sub-{sub} {roi} mode={mode}: {e}")
                    continue
    return pd.DataFrame(rows)


def fit_af_rotated_canonical(
    pars,
    rois,
    sigma_AF_bounds=(0.5, 8.0),
    fixed_sigma_AF=None,
    conditions=("upper_right", "upper_left", "lower_left", "lower_right"),
):
    """Fit the AF model in canonical (HP-aligned) frame, per subject.

    If `fixed_sigma_AF` is given, σ_AF is fixed at that value and only
    μ_AF_canonical is computed (closed form). This eliminates the σ_AF
    × μ_AF degeneracy and makes the per-subject μ_AF positions directly
    comparable.

    Each voxel × condition observation has been rotated so the HP for
    that condition lands at (0, 4°). All four conditions share a single
    μ_AF_canonical. By the 4-condition symmetry (Σ_C R_C = 0), the
    per-voxel base position drops out of the closed-form solution.

    Free parameters per subject:
        σ_AF                ∈ [0.5, 15]
        μ_AF_canonical_x    ∈ ℝ
        μ_AF_canonical_y    ∈ ℝ        (HP is at +4 in y; suppression
                                        would push the fit to negative y)

    Returns
    -------
    DataFrame with columns: subject, sigma_AF, mu_AF_x, mu_AF_y,
    n_voxels, R2 (within-voxel).
    """
    from scipy.optimize import minimize_scalar

    work = pars.reset_index()
    work = work[work["roi_base"].isin(rois)]

    rows = []
    for sub in sorted(work["subject"].unique()):
        sub_df = work[work["subject"] == sub]

        pivot_dx = sub_df.pivot_table(
            index=["roi_base", "roi", "voxel"],
            columns="condition", values="dx_rotated",
        )
        pivot_dy = sub_df.pivot_table(
            index=["roi_base", "roi", "voxel"],
            columns="condition", values="dy_rotated",
        )
        sd_mean = sub_df.groupby(
            ["roi_base", "roi", "voxel"], observed=True
        )["sd_mean_model"].first()

        keep = (
            pivot_dx.notna().all(axis=1)
            & pivot_dy.notna().all(axis=1)
            & sd_mean.notna() & (sd_mean > 0)
        )
        pivot_dx = pivot_dx.loc[keep]
        pivot_dy = pivot_dy.loc[keep]
        sd_mean = sd_mean.loc[keep]

        if len(pivot_dx) < 10:
            continue

        dx = pivot_dx.values  # (N, 4)
        dy = pivot_dy.values  # (N, 4)
        sigma_base = sd_mean.values
        S_x = dx.sum(axis=1)
        S_y = dy.sum(axis=1)

        def loss(log_sigma_AF):
            sigma_AF = float(np.exp(log_sigma_AF))
            alpha = sigma_base ** 2 / (sigma_base ** 2 + sigma_AF ** 2)
            denom = 4.0 * np.sum(alpha ** 2)
            if denom < 1e-12:
                return np.inf
            mu_x = float(np.sum(alpha * S_x) / denom)
            mu_y = float(np.sum(alpha * S_y) / denom)
            res_x = dx - alpha[:, None] * mu_x
            res_y = dy - alpha[:, None] * mu_y
            return float((res_x ** 2 + res_y ** 2).sum())

        if fixed_sigma_AF is not None:
            sigma_AF = float(fixed_sigma_AF)
            loss_val = loss(np.log(sigma_AF))
            saturated = False
        else:
            result = minimize_scalar(
                loss,
                bounds=(np.log(sigma_AF_bounds[0]), np.log(sigma_AF_bounds[1])),
                method="bounded",
                options={"xatol": 1e-3},
            )
            sigma_AF = float(np.exp(result.x))
            loss_val = float(result.fun)
            # Saturation if σ_AF is at one of the bounds.
            saturated = (
                abs(sigma_AF - sigma_AF_bounds[0]) < 0.05
                or abs(sigma_AF - sigma_AF_bounds[1]) < 0.05
            )

        alpha = sigma_base ** 2 / (sigma_base ** 2 + sigma_AF ** 2)
        denom = 4.0 * np.sum(alpha ** 2)
        mu_x = float(np.sum(alpha * S_x) / denom)
        mu_y = float(np.sum(alpha * S_y) / denom)

        ss_within = float((dx ** 2 + dy ** 2).sum())
        r2 = 1 - loss_val / ss_within if ss_within > 0 else np.nan

        rows.append({
            "subject": sub,
            "sigma_AF": sigma_AF,
            "mu_AF_x": mu_x,
            "mu_AF_y": mu_y,
            "n_voxels": int(len(pivot_dx)),
            "R2": r2,
            "loss": loss_val,
            "saturated": saturated,
        })

    import pandas as pd
    return pd.DataFrame(rows)


def prepare_voxel_data(
    pars,
    subject,
    rois,
    conditions=("upper_right", "upper_left", "lower_left", "lower_right"),
):
    """Pack a long-format conditionwise dataframe into the (N, C, 2) tensor
    the fit functions expect, plus the matching σ_base array.

    Returns
    -------
    observed : (N, C, 2) float array of (x, y) per voxel per condition.
    sigma_base : (N,) per-voxel mean-model PRF size (σ_base proxy).
    voxel_index : pandas.MultiIndex with the rows.
    """
    import pandas as pd
    sub_df = pars.reset_index()
    sub_df = sub_df[(sub_df["subject"] == subject) & (sub_df["roi_base"].isin(rois))]
    pivot = sub_df.pivot_table(
        index=["roi_base", "roi", "voxel"],
        columns="condition", values=["x", "y", "sd_mean_model"],
    )
    if pivot.empty:
        return None, None, None
    obs_arrays = []
    for cond in conditions:
        x_col = pivot[("x", cond)]
        y_col = pivot[("y", cond)]
        obs_arrays.append(np.stack([x_col.values, y_col.values], axis=1))
    obs = np.stack(obs_arrays, axis=1)  # (N, C, 2)

    # σ_base is the same across conditions (it's the mean-model fit), so
    # take any column.
    first_cond = conditions[0]
    sigma_base = pivot[("sd_mean_model", first_cond)].values

    # Drop rows with NaNs.
    valid = (
        ~np.isnan(obs).any(axis=(1, 2))
        & ~np.isnan(sigma_base)
        & (sigma_base > 0)
    )
    return obs[valid], sigma_base[valid], pivot.index[valid]


def fit_per_subject(
    pars,
    rois,
    model: str = "AF",
    conditions=("upper_right", "upper_left", "lower_left", "lower_right"),
):
    """Fit AF or AF+ separately for every subject in `pars`.

    Returns a list of (subject_id, AFFitResult, voxel_index) tuples.
    Uses multi-start optimization with bounded parameters.
    """
    subjects = sorted(pars.reset_index()["subject"].unique())
    results = []
    for sub in subjects:
        observed, sigma_base, voxel_index = prepare_voxel_data(pars, sub, rois, conditions)
        if observed is None or len(observed) < 10:
            continue
        if model == "AF":
            fit = fit_af(observed, sigma_base)
        elif model == "AF+":
            fit = fit_af_plus(observed, sigma_base)
        else:
            raise ValueError(f"Unknown model: {model}")
        results.append((sub, fit, voxel_index))
    return results
