# DoG dynamic AF v3 + target capture

**Class**: `retsupp.modeling.local_models.DoGDynamicAttentionFieldPRF2DWithHRF_v3_target`

**Fit script**:
`retsupp/modeling/fit_dog_dynamic_af_braincoder.py --model-version v3 --with-target`

**Output dir**: `derivatives/af_prf_joint_dynamic_v3_dog_with_target/`

**Cluster jobs**: 2773884 (smoke) + 2773916 (sub-2 V2..VO + sub-25 expansion array=2-40) + 2774188 (remaining 25 subjects array=1-200). All 240 tasks completed.

## Forward model

Adds a phasic target-onset term to the DoG-dyn-v3 modulation:

```
M(g, t) = 1 + sign · [ g_HP · A_HP_run(g)
                       + g_LP · Σ_{ℓ ≠ HP_run} A_ℓ(g)
                       + g_HP_dyn · d_HP_run(t) · A_HP_run^dyn(g)
                       + g_LP_dyn · Σ_{ℓ ≠ HP_run} d_ℓ(t) · A_ℓ^dyn(g)
                       + g_T_dyn · Σ_ℓ tgt_ℓ(t) · A_ℓ^tgt(g) ]
```

- `tgt_ℓ(t)` — per-TR boolean target-on indicator at location ℓ.
- `A_ℓ^tgt(g)` — Gaussian at ring position ℓ with extent `sigma_T_dyn`.
- 2 NEW shared params: `g_T_dyn`, `sigma_T_dyn`. All other params unchanged.

Targets are NOT split into HP-vs-LP because targets appear at all 4
locations roughly uniformly across trials (regardless of HP). One
shared `g_T_dyn` collapses across HP/LP target locations.

## Headline results (n = 30)

### Target-onset gain (`g_T_dyn`, Wilcoxon two-sided vs 0)

| ROI | g_T_dyn (median) | p |
|---|---|---|
| V1 | -0.16 | n.s. |
| V2 | +0.14 | n.s. |
| **V3** | **+0.67** | 0.003 |
| **V3AB** | **+3.51** | <0.001 |
| **hV4** | **+1.86** | <0.001 |
| **LO** | **+1.18** | <0.001 |
| **TO** | **+2.81** | <0.001 |
| **VO** | **+5.41** | <0.001 |

**Clean hierarchy effect** — V1/V2 stay null; V3 onwards strongly
positive. 6 of 8 ROIs show significant target capture. Predicted
positive control for the AF framework.

### Sustained HP-vs-LP (g_HP < g_LP, one-sided)

| ROI | med | p |
|---|---|---|
| V2 | -0.07 | 0.002 |
| V3 | -0.12 | 0.001 |
| **hV4** | **-0.24** | **0.015** ← NEW |
| **TO** | **-0.50** | **0.005** ← NEW |
| V3AB | -0.25 | 0.085 (trend) |
| VO | -0.34 | 0.079 (trend) |

Sustained HP-specificity GROWS once target is controlled — adds
hV4 and TO, both trending V3AB and VO.

### Dynamic HP-vs-LP (g_HP_dyn < g_LP_dyn, one-sided)

All n.s. The previously significant V3AB / LO / TO (in the no-target
fits) collapsed — target term absorbed the dynamic HP-specificity.

User insight: target and distractor are spatially uncorrelated
(target is always at a non-distractor location), so they should be
dissociable in principle. The collapse may be partly identifiability
slack (σ_T_dyn drifting to large values in V3AB/VO), to be tested
with the shared-σ variant.

### Raw sustained gains (Wilcoxon two-sided vs 0)

| ROI | g_HP | g_LP |
|---|---|---|
| V1 | +0.51 *** | +0.59 *** |
| V2 | +0.13 * | +0.37 *** |
| V3 | +0.11 n.s. | +0.14 *** |
| **V3AB** | **-0.92 ***** | **-0.85 *** ** |
| hV4 | -0.01 n.s. | +0.11 n.s. |
| LO | +0.10 n.s. | +0.03 n.s. |
| TO | -0.09 n.s. | +0.38 n.s. |
| **VO** | **-0.84 ****  | **-0.49 ****  |

**V3AB and VO flipped to sustained SUPPRESSION** when target is
modelled. V1/V2 stay positive (sustained capture). Mid-tier ROIs
mostly n.s.

### Phasic average (g_dyn_avg = ½(g_HP_dyn + g_LP_dyn))

Mostly n.s. (only V2 −0.22 ** ). The strong V1/V2/TO phasic
suppression from the no-target model attenuated.

### σ comparison

| ROI | σ_dyn (distractor) | σ_T_dyn (target) |
|---|---|---|
| V1 | 2.4 | 2.5 |
| V2 | 2.3 | 2.6 |
| V3 | 3.6 | 2.1 |
| V3AB | 3.5 | **7.2** |
| hV4 | 2.0 | 3.2 |
| LO | 2.4 | 2.7 |
| TO | 3.0 | 4.6 |
| VO | 4.0 | **9.2** |

σ_T_dyn unusually large in V3AB and VO (7-9°). Suspicious — could be
soaking up baseline trends rather than localizing target onsets.
**Test with shared-σ variant (σ_dyn = σ_T_dyn enforced) is queued.**

## Reframed take-home

1. **V1/V2**: Pure sustained CAPTURE at all candidate distractor
   locations. No HP-specific priority. No top-down target capture.
   Low-level retinotopic.
2. **V3 / hV4**: Sustained HP-specific suppression starts. Some target
   capture (V3 +0.67 **, hV4 +1.86 ***).
3. **V3AB / LO / TO**: Sustained suppression broadly + strong target
   capture. Mid/dorsal areas where attention shapes the response
   strongly.
4. **VO**: Sustained suppression + huge target capture. Ventral
   object-processing pulled by attention.

**Mechanism (revised)**: priority-map at HP is sustained, not phasic.
The phasic component is dominated by target capture (what the brain
has to do during search). The "phasic distractor suppression" finding
in the no-target model was probably partially a regression artifact
when targets weren't modelled.

## Open

- **Shared-σ variant** (queued): forces `σ_T_dyn = σ_dyn` to test if
  the σ_T_dyn≈9° in V3AB/VO was identifiability slack rather than a
  real wider target spread. If sustained HP-LP and sustained gains
  stay similar, target capture story is robust.
- **Temporal-oversampling variant** (queued, V3AB only):
  computes phasic indicators at TR/4 and TR/8 to test if sub-TR onset
  aliasing hurts σ_dyn / σ_T_dyn precision.
- **HRF fitting** (planned): the AF model currently uses a fixed
  canonical HRF (delay=4.5, dispersion=0.75). Per-voxel HRFs from the
  model 4 init are discarded. Two-step protocol: phase 1 (current),
  phase 2 (open HRF for joint fit). Test whether parameter estimates
  shift.

## See also

- [af_dynamic_v3_dog.md](af_dynamic_v3_dog.md) — original (pre-target)
  fit. Notes superseded; preserved for historical reference.
- [../analyses/cross_validated_r2.md](../analyses/cross_validated_r2.md)
  — 17-class CV factorial in flight (3509/4080 done; failures being
  resubmitted).
- `notes/figures/af_dyn_v3_dog_target_results.pdf` — figures for this
  model.
