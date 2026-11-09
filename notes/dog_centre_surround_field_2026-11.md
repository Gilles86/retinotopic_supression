# Sustained field on the DoG centre vs surround (2026-11-09)

Question: does the sustained HP/LP gain field of the DoG gain-field model act on the pRF's driving (centre)
pooling, its suppressive (surround) pooling, or both, and does a tilt between the two fix remaining misfit?
DoG analogue of `notes/dn_gain_field_variants_2026-11.md`. Main model: common gain + tilt κ (the user's
parameterisation); centre-only and surround-only are κ = ±1.

**One field per subject × ROI**, shared over voxels, as in the ladder: σ_AF, g_HP, g_LP, κ are shared; every voxel's
pRF is frozen at the final m0 fit.

## Equations

Current model (m7_full, family dog: `DoGDynamicAttentionFieldPRF2DWithHRF_v3_target_sharedSigma_factorial_items`,
`_basis_predictions` in `retsupp/modeling/local_models.py` and braincoder's `DoGAttentionFieldPRF2D`):

    F_c(g)  = Σ_ℓ w_cℓ A_ℓ(g; σ_AF),   w_cℓ = g_HP if ℓ is HP of condition c, else g_LP     (A_ℓ: peak-1 Gaussian at ring ℓ)
    M_c(g)  = max(1 + F_c(g), 0)                                                           sustained field
    P(t, g) = transient distractor (HP/LP) + target fields
    r_v(t)  = Σ_g S_β(t, g) · [M_c(t)(g) + P(t, g)] · [G_c,v(g) − G_s,v(g)] + baseline_v
    BOLD    = HRF_ROI * r_v

S_β: stimulus with the item pixels scaled by β (item_scale). G_c: centre Gaussian (sd, peak = amplitude);
G_s: surround Gaussian (sd·srf_size, peak = srf_amplitude·amplitude; volume srf_amplitude·srf_size² of the centre's).
The field multiplies the stimulus, so centre and surround see the same modulated stimulus.

Tilted model (`retsupp/modeling/dog_centre_surround_field.py`, ladder model `m8_full_dogk`):

    M^cen_c = max(1 + (1 + κ) F_c, 0),     M^sur_c = max(1 + (1 − κ) F_c, 0)
    r_v(t)  = Σ_g S_β [M^cen_c(t) + P] G_c,v  −  Σ_g S_β [M^sur_c(t) + P] G_s,v  + baseline_v

g_HP, g_LP are the common gain, κ the tilt. κ = 0 is m7 exactly (checked: max |Δpred| = 6e-6 on a 27-unit scale in
a synthetic test, and at the start of every fit), κ = +1 is the field on the centre pooling only (weight 2g),
κ = −1 on the surround pooling only, |κ| > 1 gives the other pooling a field of the opposite sign (bound
κ ∈ [−2, 2]). Transient terms act on both poolings, as in m7. One new shared term (m7: 8 → 9).
Effective HP contrasts: dS_cen = (1 + κ)(g_HP − g_LP), dS_sur = (1 − κ)(g_HP − g_LP).

`m8_full_dogcen` / `m8_full_dogsur` fit κ fixed at +1 / −1 (same code, `kappa_fixed`).

**Identifiability, expected.** κ and the gain trade off: a stronger centre field (κ up) with a smaller g gives a
similar HP-quadrant response, and with surround volumes near 1 the centre and surround fields partly cancel (the
DoG's response to a uniform field change is small). What separates them is spatial: a surround field reaches
voxels whose centre is far from HP (prediction table below) and changes the bar-on-centre vs bar-in-surround
profile. The fit stores a Gauss-Newton Hessian; `collect_dog_cs_field` reports corr(κ, g_HP), corr(κ, g_LP),
corr(κ, dS).

## Predictions

`retsupp/modeling/sim_dog_cs_field_signatures.py` → `notes/data/dog_cs_field_signatures_2026-11.tsv`. Field at
HP only (g_HP = −0.3, g_LP = 0, σ_AF = 1.5°), synthetic voxels on the four diagonals at 1–3°, sd 0.6/1.2°,
srf_size 1.3/1.8/3.2/20, surround volume 0.6/0.9 of the centre's (the range of the final m7 fits: median sd 0.83,
srf_size 1.6 (IQR 1.3–3.2, 10 % above 20), volume 0.87 (IQR 0.66–1.15)); bars in four directions, items on.
Change of the response before the HRF, % of the voxel's own bar-at-centre response, median over voxels:

| Field at HP (g_HP = −0.3) | Bar | HP quadrant | next to HP | opposite | next − opp |
|---|---|---:|---:|---:|---:|
| m7: both poolings (κ = 0) | on centre | −13.1 | 0.0 | +0.1 | −0.1 |
| | in surround | −2.2 | +0.5 | +0.1 | +0.4 |
| | items only | −2.0 | +0.2 | +0.1 | +0.1 |
| (i) centre only (κ = +1) | on centre | −49.6 | −2.0 | −0.1 | −1.9 |
| | in surround | −22.1 | −0.9 | −0.1 | −0.8 |
| | items only | −17.5 | −0.3 | 0.0 | −0.3 |
| (ii) surround only (κ = −1) | on centre | **+15.1** | **+2.1** | +0.3 | +1.8 |
| | in surround | +11.0 | +2.2 | +0.4 | +1.9 |
| | items only | +8.6 | +0.6 | +0.2 | +0.4 |
| tilt to centre (κ = +0.5) | on centre | −31.7 | −1.3 | 0.0 | −1.3 |
| tilt to surround (κ = −0.5) | on centre | +2.2 | +1.3 | +0.2 | +1.1 |
| | in surround | +3.8 | +1.5 | +0.2 | +1.2 |

(g_HP = +0.3 flips every sign exactly to this precision.)

- **(i) centre only**: local suppression, four times the m7 effect at the same g (weight 2g and no cancelling
  surround), a small loss next to HP (the field's tail reaches their centres), nothing opposite. Bar on centre is
  hit more than bar in surround (−50 vs −22 %).
- **(ii) surround only**: a reduction at HP *raises* responses: voxels at HP subtract less (+15 %), and
  neighbours whose surround covers the HP items gain (+2 %), more than the opposite quadrant (+0.3 %). The
  HP-centred voxels' centre drive is untouched; the gain comes through the surround, so it is relatively larger
  when the bar is in the surround (11 vs 15 %, ratio 0.73 vs 0.45 under (i)). The release grows with surround
  volume (HP +10 % at volume 0.6, +23 % at 0.9) and vanishes with very wide flat surrounds (srf_size 20: +0.2 %).
  For neighbours it is ~3.5 % for srf_size 1.3–3.2.
- **m7 (both)** lies in between (without the floor the response is linear in κ, so each voxel's change at κ = 0 is
  the mean of its κ = ±1 changes): centre and surround fields partly cancel for HP-centred voxels (−13 %), and the
  neighbours get almost nothing. A tilt toward the surround (κ = −0.5) keeps the HP effect small and
  raises the next quadrant: HP +2, next +1.3, opposite +0.2 for g_HP < 0.
- So in the DoG the sign of the HP-quadrant change depends on κ: g_HP < 0 with κ < −~0.6 predicts an HP
  *increase*. dS alone is therefore not interpretable as suppression without κ; read dS_cen and dS_sur.
- Distinguishing signatures: (1) next − opposite (positive for surround, negative for centre); (2) the ratio of
  next-to-HP to HP-quadrant effects (0.14 surround vs 0.04 centre); (3) surround/centre bar-position ratio at HP.

## Canary

(pending — filled in below)

## Jobs

(pending)
