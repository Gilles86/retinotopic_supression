# Gain findings — DoG dyn-v3 + target capture (n=30)

The most complex model fit so far. **5 free gain parameters per voxel
collection** (all sign-free, all init = 0):

- `g_HP`, `g_LP` — sustained AF at HP and LP locations.
- `g_HP_dyn`, `g_LP_dyn` — phasic AF at HP and LP distractor onsets.
- `g_T_dyn` — phasic AF at target onsets (any of the 4 ring locations).

All Wilcoxon vs 0, two-sided, n=30. Median values shown.

## Per-ROI summary

| ROI | g_HP (sus) | g_LP (sus) | g_HP_dyn | g_LP_dyn | g_T_dyn |
|---|---|---|---|---|---|
| V1 | **+0.51 ***** | **+0.59 ***** | −0.04 n.s. | −0.23 * | −0.16 n.s. |
| V2 | **+0.13 *** | **+0.37 ***** | −0.17 n.s. | **−0.34 *** ** | +0.14 n.s. |
| V3 | +0.11 n.s. | **+0.14 ***** | −0.07 n.s. | −0.14 n.s. | **+0.67 *** ** |
| V3AB | **−0.92 ***** | **−0.85 *** ** | **−0.46 *** | −0.14 n.s. | **+3.51 ***** |
| hV4 | −0.01 n.s. | +0.11 n.s. | +0.01 n.s. | −0.13 n.s. | **+1.86 ***** |
| LO | +0.10 n.s. | +0.03 ** | **−0.33 *** | −0.02 n.s. | **+1.18 ***** |
| TO | −0.09 n.s. | +0.38 n.s. | −0.45 n.s. | −0.85 n.s. | **+2.81 ***** |
| VO | **−0.84 ***** | **−0.49 *** | −0.32 n.s. | +0.37 n.s. | **+5.41 ***** |

## What each row says

**V1**: Sustained capture at all candidate locations (g_HP +, g_LP +).
No phasic distractor structure. No target capture. Pure low-level
retinotopic.

**V2**: Sustained capture (slightly less at HP than LP). Modest phasic
suppression at LP-distractor. No target capture.

**V3**: Faint sustained capture at LP. Target capture starts
(+0.67 *** ).

**V3AB**: **Sustained SUPPRESSION** at both HP and LP (both gains
strongly negative). HP-specific phasic distractor suppression.
Strong target capture (+3.51).

**hV4**: Sustained gains null. Strong target capture (+1.86).

**LO**: HP-only phasic distractor suppression (g_HP_dyn −0.33 * ).
Target capture (+1.18).

**TO**: Net null sustained. Phasic distractor suppression in V1/V2
direction. Strong target capture (+2.81).

**VO**: Sustained SUPPRESSION (both negative). No phasic distractor
structure. Strongest target capture in the brain (+5.41).

## Two patterns emerge

**Sustained AF**:
- V1, V2, (V3) → POSITIVE: anticipatory enhancement at the candidate
  distractor locations.
- V3AB, VO → NEGATIVE: sustained priority-map suppression.
- Mid-tier (hV4, LO, TO) → NULL.

**Target capture (g_T_dyn)**:
- Hierarchical: zero in V1/V2, growing through V3 (+0.67) → V3AB
  (+3.51) → VO (+5.41).
- Cleanest positive control of the AF framework — predicts that an
  attentionally relevant transient should produce *enhancement*, not
  suppression. Confirmed.

**Phasic distractor (g_HP_dyn, g_LP_dyn)**:
- Mostly weakened relative to the no-target model. The target term
  absorbs variance. V1/V2 lose the "distractor suppression" we
  previously claimed.
- HP-specific phasic suppression survives in V3AB (g_HP_dyn −0.46 *)
  and LO (g_HP_dyn −0.33 *) — these are the only ROIs where the
  HP-distractor specifically (not just any distractor) gets phasically
  suppressed.

## Caveats

- **σ values are unreliable** — see [issues/identifiability.md](issues/identifiability.md).
  σ_dyn is poorly identifiable; σ_T_dyn drifted to ~9° in V3AB/VO.
  The shared-σ variant (σ_T_dyn := σ_dyn) is queued. Gains are more
  robust because of stronger gradient signal.
- This is the v3-target model with σ inits at 2.0/2.0/2.0 (neutral).
  Large-σ init re-fits are queued.

## Source

`derivatives/af_prf_joint_dynamic_v3_dog_with_target/`, n=240 fits
(30 subjects × 8 ROIs).

Data: [data/af_dog_v3_target_parameters.tsv](data/af_dog_v3_target_parameters.tsv).
Figures: [figures/af_dyn_v3_dog_target_results.pdf](figures/af_dyn_v3_dog_target_results.pdf).
Full writeup: [models/af_dynamic_v3_dog_with_target.md](models/af_dynamic_v3_dog_with_target.md).
