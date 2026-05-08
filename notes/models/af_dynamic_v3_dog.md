# DoG dynamic AF v3

**Class**: `DoGDynamicAttentionFieldPRF2DWithHRF_v3`
(`libs/braincoder/braincoder/models.py`, commit `702d9f0` on
`feature/attention_field_prf`).

**Fit script**:
`retsupp/modeling/fit_dog_dynamic_af_braincoder.py --model-version v3`

**Output dir**: `derivatives/af_prf_joint_dynamic_v3_dog/`

**Cluster job**: 2761299 (240 tasks, all done).

## Forward model

Voxel kernel: difference of Gaussians
(`x, y, sd, baseline, amplitude, srf_amplitude, srf_size`) so we
properly model center-surround structure in early visual cortex.

Modulation field:
```
M(g, t) = 1 + sign · [ g_HP · A_HP_run(g)
                       + g_LP · Σ_{ℓ ≠ HP_run} A_ℓ(g)
                       + g_HP_dyn · d_HP_run(t) · A_HP_run^dyn(g)
                       + g_LP_dyn · Σ_{ℓ ≠ HP_run} d_ℓ(t) · A_ℓ^dyn(g) ]
```
- `A_ℓ` uses sustained `σ_AF`.
- `A_ℓ^dyn` uses `σ_dyn` (different scale).
- 13 per-voxel + shared params: 7 DoG voxel + 6 shared
  (`σ_AF, g_HP, g_LP, σ_dyn, g_HP_dyn, g_LP_dyn`).

## Headline numbers (DoG, n = 30)

### σ comparison (paired Wilcoxon σ_dyn vs σ_AF)

| ROI | σ_AF | σ_dyn | p |
|---|---|---|---|
| V1 | 2.40 | **0.66** | 0.001 |
| V2 | 2.74 | **0.88** | 0.003 |
| V3 | 2.86 | **0.79** | <0.0001 |
| V3AB | 4.97 | **1.11** | <0.001 |
| hV4 | 3.62 | **0.76** | <0.0001 |
| LO | 3.12 | **1.00** | 0.002 |
| TO | 4.25 | **1.08** | <0.0001 |
| VO | 2.72 | 1.57 | n.s. |

σ_dyn ≪ σ_AF in essentially every ROI except VO. **Caveat**: original
inits were σ_AF=2.0, σ_dyn=0.5, which biased the optimizer toward
σ_dyn < σ_AF. Init defaults have been changed to σ_AF=σ_dyn=2.0
(neutral) and a robustness refit on 5 subjects is queued
(`derivatives/af_prf_joint_dynamic_v3_dog_neutralsigma/`). Update this
section once that test lands.

### RAW gain signs per ROI (Wilcoxon two-sided vs 0)

This is the key reframing — **the raw gains do NOT match the
"proactive HP suppression" interpretation that the differential alone
suggested**.

**Sustained** — both `g_HP` and `g_LP` are POSITIVE almost everywhere:

| ROI | g_HP | g_LP |
|---|---|---|
| V1 | +0.63 *** | +0.48 *** |
| V2 | +0.27 *** | +0.41 *** |
| V3 | +0.10 ** | +0.19 *** |
| V3AB | +0.60 *** | +0.54 *** |
| hV4 | +0.28 ** | +0.22 *** |
| LO | +0.12 n.s. | +0.08 ** |
| TO | +0.93 *** | +1.05 *** |
| VO | -0.04 n.s. | +0.82 n.s. |

The sustained AF is **capture-like / enhancement, not suppression**.
Voxels near every candidate distractor location have *enhanced*
baseline responsiveness. The sustained "HP < LP" finding (see below)
is HP being **slightly less enhanced than LP**, not HP being
suppressed.

**Dynamic** — both `g_HP_dyn` and `g_LP_dyn` are NEGATIVE in the
suppression-sensitive ROIs:

| ROI | g_HP_dyn | g_LP_dyn |
|---|---|---|
| V1 | -2.76 *** | -2.10 * |
| V2 | -1.86 *** | -1.38 *** |
| V3 | -0.83 n.s. | -0.81 n.s. |
| V3AB | -2.78 ** | -1.10 n.s. |
| hV4 | -0.13 n.s. | +0.04 n.s. |
| LO | -0.54 * | +0.91 n.s. |
| TO | -3.80 *** | -2.77 * |
| VO | +2.37 n.s. | +1.86 n.s. |

When a distractor actually appears, response goes DOWN — more strongly
at HP than at LP. V3AB and LO show clean HP-only dynamic suppression
(LP n.s.); V1/V2/TO show suppression at both with HP stronger.

### Sustained HP-vs-LP differential (`g_HP < g_LP`, one-sided)

| ROI | med diff | p |
|---|---|---|
| V2 | -0.05 | 0.014 |
| V3 | -0.10 | 0.004 |
| VO | -0.95 | 0.005 |
| (others) | n.s. | |

In V2/V3/VO the sustained HP gain is **less positive** than the LP
gain — i.e. less anticipatory enhancement at HP than at LP. NOT HP
being suppressed.

### Dynamic HP-vs-LP differential (`g_HP_dyn < g_LP_dyn`, one-sided)

| ROI | med diff | p |
|---|---|---|
| V3AB | **-1.24** | **0.0001** |
| LO | **-1.52** | **0.002** |
| TO | **-1.17** | **0.006** |
| (others) | n.s. | |

In V3AB/LO/TO the dynamic suppression is significantly stronger when
the distractor lands at HP than at LP. The phasic system is
HP-specific in mid-tier ROIs.

### Phasic effect at all (`g_dyn_avg ≠ 0`, two-sided)

| ROI | med | p |
|---|---|---|
| V1 | -1.62 | 0.002 |
| V2 | -1.50 | <0.001 |
| TO | -2.77 | 0.002 |
| (others) | n.s. or trend |

Net phasic SUPPRESSION in V1/V2/TO when ANY distractor appears.

## ⚠️ Superseded interpretation — see "Re-reading after target term"

The take-home below was based on the v3 + DoG fit WITHOUT a target-onset
term. Adding a target-capture term (see
[af_dynamic_v3_dog_with_target.md](af_dynamic_v3_dog_with_target.md))
**changes the readout substantially**:

- Sustained HP-vs-LP differential GROWS — V2/V3/hV4/TO all significant
  for HP < LP (vs only V2/V3 here).
- Dynamic HP-LP differential DISAPPEARS — V3AB/LO/TO no longer
  significant once target is in the model.
- V3AB and VO sustained gains flip from near-zero to clearly negative
  (sustained suppression). V1/V2 stay positive (sustained capture).
- Phasic suppression in V1/V2 attenuates a lot.

The cleaner story is in the with-target notes. The take-home below is
preserved for historical reference.

---

## Take-home (DoG, original — pre-target-term)

1. **Sustained AF = anticipatory CAPTURE, not suppression.** Both g_HP
   and g_LP are positive almost everywhere — the brain prioritizes the
   candidate-distractor zones. In V2/V3/VO the priority for HP is
   weaker than for LP (the differential test).
2. **Dynamic AF = phasic SUPPRESSION** when the distractor actually
   appears. Hits ALL distractor positions in V1/V2/TO; HP-specifically
   in V3AB/LO/TO.
3. **σ_dyn ≪ σ_AF** in 7/8 ROIs — the phasic mechanism is local
   (~0.7–1°, near distractor disk size); the sustained mechanism is
   broad (~2–5°). Pending init-bias check.

This maps cleanly onto the Wang & Theeuwes signal-suppression story:
priority map sets up baseline enhancement at candidate locations; an
active suppressive mechanism kicks in when a distractor actually
appears.

## Open

- [analyses/cross_validated_r2.md](../analyses/cross_validated_r2.md):
  17-class CV factorial is the rigorous test of which gain-sign
  pattern actually generalizes per ROI. Submit pending smoke-test
  pass.
- σ-init robustness check
  (`derivatives/af_prf_joint_dynamic_v3_dog_neutralsigma/`,
  5 subjects, `fit_dog_dynamic_af_neutralsigma.sh`).
- VO is consistently weird (σ saturation, dynamic gains positive). Not
  obviously fitting the rest of the hierarchy. Worth a separate look.

## See also

- `notes/figures/af_dyn_v3_dog_results.pdf` — figures generated from
  this fit.
- `analyses/sigma_comparison.md` — σ_AF vs σ_dyn cross-kernel.
- `analyses/hp_vs_lp_test.md` — HP-vs-LP differential analysis.
