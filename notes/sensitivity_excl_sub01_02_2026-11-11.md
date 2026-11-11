# Sensitivity: the main results without the two pilots (sub-01, sub-02), 2026-11-11

All per-subject estimates are unchanged (per-subject fits do not depend on other subjects). Everything that pools across
subjects was recomputed with n = 28: Wilcoxon tests, Benjamini-Hochberg (BH) across areas, bootstrapped CIs, the Fig. 3
ANOVAs, the hierarchy regression, the leave-one-subject-out (LOSO) δ (median of the *other* subjects, now 27), and the
Fig. 7 hierarchical model. The n = 30 column reproduces `writing_package_2026-11-10/key_numbers.md` in every row
checked (exceptions noted).

Sources:
- `retsupp/visualize/paper/sensitivity_excl_subjects.py` → `notes/data/sensitivity_excl0102_2026-11.tsv` (Figs. 1–3, 5, 6,
  Gaussian/DN robustness; both columns).
- LOSO: `retsupp/modeling/loso_fixed_group_exact.py --exclude-subjects 1 2` (cluster job 8261937) →
  `notes/data/loso_exact_final2_excl0102_2026-11.tsv`, `notes/data/loso_exact_robust_final2_excl0102_2026-11.tsv`; summary
  `notes/data/loso_per_step_excl0102_2026-11.tsv`.
- Fig. 7: `retsupp/visualize/paper/trialwise_meta_bayes.py` on the parametric/check tables without sub-01/02 →
  `notes/data/trialwise_meta_bayes_excl0102_2026-11/` (0–1 divergences per fit, R̂ ≤ 1.01, bulk ESS ≥ 880; the typical gain
  used for the % scale is recomputed from the 28).

## 1. What differs in sub-01 and sub-02

From the per-run settings (`sourcedata/behavior/logs/sub-{1,2}/ses-*/…_expsettings.yml`, identical for the two pilots)
and `Subject.get_hpd_locations()`:

- **Bar speed 0.5°/s instead of 0.168°/s** (all 24 pilot runs): each sweep of the aperture (6.34° + 0.79° bar width) takes
  ≈ 14 s instead of ≈ 42 s, about three times faster. The pilot settings also lack the `break_duration` / `rest_duration`
  entries (4.8 s) of the later version. The search task itself (items, trial timing, trial counts) is the same code.
  Minor: display shift −0.80° instead of −0.83°.
- **HP schedule:** per session 2, 3 and 1 runs at three HP locations, the same sequence in both sessions, so one diagonal
  location is never HP and the three that are get 4, 6 and 2 runs (sub-01: lower right, upper right, lower left, never
  upper left; sub-02: upper left, upper right, lower left, never lower right). Everyone else: four blocks of three runs,
  each location HP once, HP moving to the opposite location within a session.

## 2. Results, n = 30 vs n = 28

"Dir." = subjects in the direction of the effect. Bold = a change at the figure's own threshold.

### Fig. 1 Behaviour (median within-subject difference, ms [95% CI], Wilcoxon)

| Result | n = 30 | n = 28 | Changed? |
|---|---|---|---|
| Distractor HP − LP | −26.1 [−34.4, −20.9], 30/30, p = 1.9e-9 | −25.0 [−32.4, −20.0], 28/28, p = 7.5e-9 | no |
| Distractor LP − absent | +43.7, 28/30, p = 2.6e-8 | +42.6, 26/28, p = 1.0e-7 | no |
| Distractor HP − absent | +15.8 [+2.1, +23.2], 21/30, p = .0012 | +14.6 [+0.3, +21.7], 19/28, p = .0041 | no |
| Target HP − LP | +26.8, 26/30, p = 3.9e-7 | +24.3, 24/28, p = 1.5e-6 | no |
| Target repeat − switch | −38.4, 28/30, p = 9e-9 | −38.7, 26/28, p = 3.7e-8 | no |
| Distractor repeat − switch | −15.1, 21/30, p = .0035 | −15.1, 20/28, p = .0013 | no |

### Fig. 2d pRF models (held-out R² gained, median [CI])

| Result | n = 30 | n = 28 | Changed? |
|---|---|---|---|
| DoG over Gaussian | +.0035, 30/30 | +.0033, 28/28 | no |
| Joint HRF on top of DoG | +.0107, 29/30 | +.0106, 27/28 | no |
| Both | +.0189, 30/30 | +.0179, 28/28 | no |

### Fig. 3 Item locations

| Result | n = 30 | n = 28 | Changed? |
|---|---|---|---|
| Target − neutral (early / dorsal / occ.-temp., % signal) | +0.125 / +0.258 / +0.229, all p ≤ 7e-9 | +0.125 / +0.264 / +0.236, all p ≤ 3e-8 | no |
| Distractor − neutral (LP) | +0.072 / +0.147 / +0.150, p ≤ 4e-5 | +0.072 / +0.149 / +0.156, p ≤ 1e-4 | no |
| Early: location-type ANOVA | F(2,58) = 6.28, p = .003 | F(2,54) = 7.67, p = .0012 | stronger |
| Early: next − opposite | −13.0%, p = .047 | −14.9%, p = .022 | stronger |
| Early: HP − opposite | −19.0%, p = .011 | −21.6%, p = .0052 | stronger |
| Early: HP − next | −6.1%, p = .135 | −6.7%, p = .095 | n.s. both |
| Early: location × item | p = .514 | p = .454 | no |
| All areas: ANOVA | F = 4.73, p = .012 | F = 6.31, p = .0035 | stronger |
| All areas: HP − next / next − opp / HP − opp | −6.4% (.029) / −6.8% (.12) / −13.2% (.028) | −6.9% (.018) / −8.6% (.060) / −15.5% (.013) | stronger, next − opp still n.s. |
| HP vs all LP (early / all areas) | −8.6%, p = .039 / −7.3%, p = .027 | −9.8%, p = .024 / −8.4%, p = .015 | stronger |
| Dorsal, occ.-temp. ANOVA | p = .86, .49 | p = .86, .23 | no |

### Fig. 5 Gain fields (pooled = per-subject median over areas; Wilcoxon; per area BH q < .05)

| Result | n = 30 | n = 28 | Changed? |
|---|---|---|---|
| Target gain, joint | +.18, 90%, p = 8e-8; 8 areas q < .05 (V2–TO) | +.16, 89%, p = 3e-7; 6 areas (V2, LO drop out) | **fewer areas** (V2 +.076, LO +.15 were the weakest) |
| Target gain per doubling of pRF size | +.32, q = .0006 | +.33, q = .0017 | no |
| Distractor gain, joint | +.065, 77%, p = .011; V3AB, hV4, TO | +.065, 75%, p = .023; V3AB, hV4, TO | no (weaker p) |
| Distractor on top of target, alone | +.035, p = .0066; V3AB, hV4, TO | +.031, p = .019; V3AB, hV4 (TO drops) | **weaker, one area fewer** |
| Sustained S alone | −.032, 97%, p = 3.7e-9; V2–TO | −.034, 100%, p = 7.5e-9; V2–TO | no |
| **dS alone** | −.012, 80%, p = 5.6e-5; V3 (q = .020), TO (.018) | −.014, 79%, p = 1.1e-4; V3 (.046), TO (.035) | no |
| dS joint (m7) | −.031, 77%, p = .0050; V3, hV4 | −.034, 79%, p = .0027; V3, hV4, TO | stronger (+TO) |
| dD alone | −.013, p = .050; LO | −.010, p = .099; LO | **drops above .05** (was borderline) |
| dT alone | −.034, p = .016 | −.034, p = .018 | no |
| dD − dT alone | +.002, p = .37 | +.009, p = .24 | no |
| dD joint | −.010, p = .47 | +.004, p = .29 | no |
| Split: next − opposite | +.0001, p = .87 | −.006, p = .73 | no |
| Split: HP − next | −.012, p = .092 | −.016, p = .070 | no |
| Split: HP − opposite | −.031, 80%, p = .0087 | −.033, 82%, p = .0048 | stronger |
| Split: target HP − LP | −.019, p = .46 | +.047, p = .37 | no (n.s.) |
| One shared width: dS | −.019, 77%, p = .055 | −.023, 79%, p = .036 | **crosses .05** |

(The hierarchy slope reproduces key_numbers only with `notes/data/fig1_prf_sigma.tsv`, the model-4 size table the
final2 regression used; with the final-fit sizes the n = 30 target slope is +.39. Both n = 28 values use the same table
as their n = 30 counterpart.)

### Supplementary table: LOSO (DoG; ΔR² better / sign-flip direction)

| Term | n = 30 | n = 28 | Changed? |
|---|---|---|---|
| Target gain | 77%, p = .0006 / 97%, p < 1e-4 | 75%, p = .0007 / 96%, p < 1e-4 | no |
| **Distractor gain (tied)** | 77%, p = .031 / 80%, p = .0010 | 61%, p = .16 / 75%, p = .0034 | **better-out-of-subject no longer significant** |
| Distractor on top of target | 63%, p = .11 / 80%, p < 1e-4 | 61%, p = .33 / 75%, p = .0005 | no |
| Sustained S | 87% / 100%, p < 1e-4 | 93% / 100%, p < 1e-4 | no |
| dS | 67%, p = .18 / 77%, p = .0024 | 68%, p = .41 / 75%, p = .0071 | no |
| dD | 57%, p = 1.0 / 70%, p = .10 | 43%, p = .25 / 50%, p = .96 | no (direction gone) |
| dT | 63%, p = .24 / 70%, p = .019 | 61%, p = .25 / 71%, p = .014 | no |

### Fig. 6 Decoding (V1–V3 pooled, free decoder, % of the four-quadrant mean)

| Result | n = 30 | n = 28 | Changed? |
|---|---|---|---|
| d: HP / next / opposite | −0.85 / +0.46 / −0.08% | −1.02 / +0.57 / −0.12% | larger |
| d: HP < next (one-sided) | p = .015 | p = .0038 | stronger |
| d: HP < opposite | p = .13 | p = .082 | n.s. both |
| d: Friedman, four quadrants | p = .062 | **p = .013** | **becomes significant** |
| d: next vs opposite (two-sided) | p = .16 | p = .070 | n.s. both |
| e: whole aperture / 1.5–2.5° / 2.5°+ | −1.10% (.033) / −1.37% (.029) / −1.30% (.040) | −1.32% (.010) / −1.64% (.0096) / −1.52% (.020) | stronger |
| e: 0–1.5° | −0.57%, p = .10 | −0.73%, p = .069 | n.s. both |
| Suppl. per area, BH q < .05 | none (V2, V3, V3AB, VO, TO q = .076–.077) | **V2, V3 (q = .038)** | **two areas survive** |
| b: frame-wise r, V1 (median) | .51 | .52 | no |
| Decoded bars, HP half − other half (vertical / horizontal / pooled) | −1.61 (.031) / −1.83 (.001) / −1.72% (.004) | −1.71 (.034) / −1.88 (.002) / −1.80% (.006) | no |
| Decoded bars, next − opposite half, pooled | +1.04%, p = .047 | +1.21%, p = .019 | stronger |
| Gaussian pRFs: HP vs next (two-sided) | p = .036 | p = .0083 | stronger |
| DN pRFs: HP vs next (two-sided, free / outline) | p = .105 / .011 | **p = .036** / .0081 | **free decoder becomes significant** |

(The panel-c inset values, −1.16% and +1.13%, were not recomputed here; their source was not located in this pass. The
decoded-bars contrasts above are the same measurement on the half-bars.)

### Fig. 7 Gains and behaviour (hierarchical model; posterior mean [95% HDI], P of the sign)

| Result | n = 30 | n = 28 | Changed? |
|---|---|---|---|
| Target gain on target repeat, all areas (% of typical gain) | −19% [−33, −7], P = .999 | −19% [−33, −4], P = .997 | no |
| … early / dorsal / occ.-temp. | −29% (.978) / −19% (.961) / −13% (.956) | −26% (.962) / −22% (.964) / −13% (.945) | **early falls below .975** |
| … absolute units | −0.053, P = .998 | −0.052, P = .993 | no |
| … without SE weighting | P = .978 | P = .968 | slightly below .975 |
| Distractor gain on distractor repeat | −23%, P = .87 | −30%, P = .92 | no (n.s.) |
| Distractor gain per SD log RT | −0.023 [−0.040, −0.004], P = .993 | −0.024 [−0.043, −0.005], P = .990 | no |
| **Global gain per SD log RT** | −0.030 [−0.060, −0.002], P = .979 | −0.021 [−0.046, +0.006], P = .943 | **no longer credible** |
| Target gain per SD log RT | +0.009, P = .81 | +0.009, P = .80 | no |

### Robustness: Gaussian and DN pRFs (pooled, Wilcoxon)

| Result | n = 30 | n = 28 | Changed? |
|---|---|---|---|
| dS alone, exact (DoG / Gauss / DN) | p = 3e-5 / 2.3e-4 / .0032 | p = 9e-5 / 4.2e-4 / .0052 | no |
| dS joint (Gauss / DN) | −.063, p = .078 / −.074, p = .0076 | −.088, **p = .027** / −.078, p = .0024 | **Gaussian crosses .05** |
| dS LOSO direction (Gauss / DN) | 70%, p = .011 / 70%, p = .045 | 79%, p = .0044 / 71%, p = .038 | no |
| dS LOSO better (Gauss / DN) | 57%, p = .64 / 60%, p = .27 | 61%, p = .54 / 64%, p = .24 | no |
| Distractor on top of target alone (Gauss / DN) | p = .011 / .052 | p = .032 / .12 | weaker |
| DN distractor gain (tied) LOSO | 57% better, p = .56 | **18% better, p = .018 (worse out of subject)** | **flips to worse** |
| Target gain (all models, joint and alone) | all p ≤ 1e-7 | all p ≤ 4e-7 | no |

## 3. Verdict

Mixed, and on balance slightly **more robust for the paper's main line**:

- **More robust:** everything about the HP *location*. Fig. 3: the early-area location effect and its contrasts (next −
  opposite p = .047 → .022). Fig. 5: dS joint gains TO, the split model's HP − opposite gets stronger, and the one-width
  model's dS crosses .05. Fig. 6: the decoded HP reduction grows (−0.85 → −1.02%), the Friedman test becomes significant,
  V2 and V3 survive BH per area (none did before), and DN-pRF decoding of HP vs next becomes significant. Gaussian joint
  dS crosses .05.
- **Unchanged:** behaviour (the HP benefit is 28/28), the pRF-model gains, target and distractor item responses, the
  headline dS estimated alone (p = 1e-4, 79%, V3 and TO), dS LOSO (still direction yes, prediction no), the target-gain
  results, the Fig. 7 target-repeat effect over all areas and the distractor-gain RT effect.
- **Less robust:** the *distractor-gain* claims. The tied distractor gain no longer improves out-of-subject prediction
  (77%, p = .031 → 61%, p = .16); the distractor gain on top of the target, estimated alone, loses TO (p = .007 → .019);
  dD alone goes from p = .050 to .099; under DN pRFs the distractor gain predicts worse out of subject. Fig. 7: the global
  gain's RT effect is no longer credible (P = .979 → .943), and the early group's target-repeat effect drops just below
  P = .975. Target gain loses V2 and LO per area (the two weakest).

None of the claims in the README storyline reverses sign. Leaving the pilots out strengthens Figs. 3 and 6 (the
model-free and model-inverted HP effects) and costs the weakest distractor-gain and global-gain statements.

## 4. Draft Methods sentence (if the pilots are dropped)

"Two further participants were tested with a pilot version of the paradigm, in which the mapping bar moved about three
times faster (0.5° s⁻¹ instead of 0.168° s⁻¹) and the high-probability location took only three of the four positions,
in blocks of 2, 3 and 1 runs per session; they were excluded from all analyses, which leaves 28 participants. Including
them does not change the direction of any result (Supplementary Table S…)."

(If they are kept instead: "Two participants were tested with a pilot version … (bar 0.5° s⁻¹; three HP locations in
blocks of 2, 3 and 1 runs); leaving them out does not change the direction of any result and strengthens the effects at
the HP location (Supplementary Table S…).")
