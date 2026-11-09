# Claims and caveats (2026-11-09)

The guardrail for the text. A sentence in the paper should map onto a claim in section 1, be phrased no more strongly
than it is here, and not contradict section 2. Numbers: `key_numbers.md`. Gain numbers are provisional until the final
DoG rerun; the wording below is chosen so that it should survive the rerun, but check it against the new numbers.

## 1. Claims the data support

**Behaviour (Fig. 1; final).**
- Responses are faster when the distractor appears at the HP location than at another location, in all 30
  participants, and slower when the target appears there.
- A target or distractor at the same location as on the previous trial speeds responses.

**pRFs (Fig. 2; final).**
- pRFs mapped during search are retinotopic and mostly contralateral, and their size grows with eccentricity beyond V1.
- A DoG pRF and one HRF per subject × ROI each improve held-out prediction over a Gaussian pRF with the canonical HRF,
  in nearly every subject.
- V1 pRF size does not grow with eccentricity here; say that it is likely limited by the resolution of the paradigm
  (bar width, HRF, fixation), not that V1 pRFs are constant in size.

**Responses at the item locations, without a model (Fig. 3; final).**
- Voxels whose pRF lies at an item location respond more when the target or the distractor is there than when a
  neutral item is, in all three pathways.
- In early visual cortex (V1–V3 pooled), the response to an item depends on where it is relative to the HP location:
  least at HP, less next to HP than opposite it (next − opposite and HP − opposite reliable; HP vs next not).
- This location effect does not differ between target, distractor and neutral items (no location × item
  interaction), so it is a property of the location, not of what appears there.
- In dorsal and occipitotemporal areas there is no reliable location effect.

**Gain fields (Fig. 5; provisional).**
- Target gain is positive in every area beyond V1 and grows along the hierarchy (with pRF size).
- Distractor gain is positive but weaker, reliable in V3AB, IPS, hV4 and VO.
- The sustained gain field is wide (FWHM ≈ 5°), wider than the spacing of neighbouring items, so it spreads into the
  neighbouring locations.
- Sustained gain is lower at the HP location than at the LP locations (dS < 0) when the term is estimated alone, in
  most areas (seven of nine) and pooled over areas; jointly with all other terms it is negative in the same direction
  but no single area survives correction. The pooled effect is the claim, not any single area.
- "The transient HP − LP gain is less for the distractor (in V3AB, IPS, hV4 and VO, estimated alone); the target term
  is not reliable; the two do not differ."
- dS predicts new subjects (LOSO) under DoG pRFs; dS estimated alone is negative under DN pRFs too.

**Decoding (Fig. 6; final).**
- The bar can be decoded from held-out runs in all nine areas, best in V1–V3, with pRFs that have no gain terms.
- "The bar is decoded at lower intensity in the quadrant facing HP than in the neighbouring quadrants, pooled over
  V1–V3." The effect is spread across eccentricity, not concentrated near the items, which argues against item
  responses leaking into the aperture.
- With the stimulus held identical (each decoded bar minus the same bar averaged over HP locations), the half of a bar
  in the quadrant facing HP is decoded weaker than its other half, for vertical and for horizontal bars.
- The result holds with the decoder constrained to the bar outline and with canonical-HRF pRFs; it is weaker with DN
  pRFs.

**Gain and behaviour (Fig. 7; prototype, October fits).**
- Target gain is lower on trials where the target repeats its location, in V3AB, IPS, hV4 and VO.
- Within subjects, trial by trial, target, distractor and global gain predict response time: larger distractor gain
  goes with faster responses in V1/V2, larger target gain with slower responses in V3AB/hV4/LO, and global gain with
  faster responses in V2/V3 and slower responses in IPS. These are Bayesian posterior probabilities, not corrected
  across areas; say so.

## 2. Claims the data do NOT support (do not write these)

- **Neighbour enhancement.** Nothing shows more response or gain next to HP than elsewhere. In the measured item
  responses the neighbours are *lower* than the opposite location. In the decoded bar the neighbouring quadrants are
  above the four-quadrant mean (+0.46%) and the next-to-HP half of a bar is decoded stronger than the opposite half
  (+1.04%, p = .047, uncorrected), but next vs opposite quadrant is not reliable in any pRF model (p = .13–.43).
  Do not call this "enhancement"; it is a relative measure (each value against a mean that includes HP).
- **Distractor-specific suppression vs the target.** dD is reliable estimated alone in four areas, but dD and dT do not
  differ, dD is not reliable in the joint fit, and the measured item responses show no location × item interaction.
  Say "less transient gain for a distractor at HP", never "the distractor, unlike the target, is suppressed at HP".
- **Decoding effects in single areas after correction.** No area survives BH across the nine (smallest q = .076).
  Only the pooled V1–V3 contrast and the identical-stimulus bar halves are claims.
- **A per-area dS claim from the joint fit.** No area survives correction jointly; per-area dS is only claimed
  estimated alone, and then as a pattern, not area by area.
- **DN-parameter-specific effects.** The per-condition DN fits (b, d, σ, amplitudes by quadrant) are exploratory and
  mostly null; the baseline-field variants do not separate b from d (canary) and trade off with the input gain.
  Do not say the HP effect "lives in" normalization or baseline parameters.
- **DN reproduces everything.** DN gives the same sign for dS but LOSO fails (63%, p = .23) and the decoding effect is
  weaker (HP < next p = .057). Say the claim depends partly on the pRF model.
- **Items improve out-of-subject prediction in 100% of subjects** (Table 1). That number scores items turned *down*
  from bar strength, not items vs none; against no items it is 80% (DoG) and 3% (DN). Do not cite the 100%.
- **Brain–behaviour across subjects.** dS does not relate to the behavioural HP benefit (ρ = −.17, p = .37). Report
  it as a null with the power caveat; do not interpret its sign.
- **Learning over runs.** No evidence, and the test is underpowered; do not say suppression was "learned in cortex"
  over the session.
- **Proactive vs reactive** from our data alone: we have no omission trials. The sustained term and the bar-drive
  effect are measured across the block, but the bar-crossing effect depends on the baseline (see section 4).
- **Field size along the hierarchy.** Many transient fits sit at a σ bound; do not interpret slopes of field size.
- **Anything on pRF shifts.** Archived; not part of the paper.

## 3. Caveats the text must state

- Gain numbers are provisional; trial-wise analyses (Fig. 7) use earlier fits.
- Within-subject CV cannot detect a dS of our size (simulation), so it is not evidence against dS.
- HP lies at 4°, outside the 3.17° bar aperture; only pRFs that reach the ring carry the gain effect, and the decoded
  bar measures drive *between* the items, not at them.
- Decoding statistics: one-sided Wilcoxon per area, Benjamini-Hochberg across the nine areas (as in Fig. 6d). The paper
  does not frame any test as pre-specified (user decision, 2026-11-09).
- The HP-rotation null for dS (relabelling HP to the other locations) has not been run yet; it runs on the final fits.
- Fig. 3 location contrasts are uncorrected planned contrasts; the ANOVA main effect of location type is the
  protected test in early areas.
- V1 pRF sizes are at a floor.

## 4. Relation to Richter, van Moorselaar & Theeuwes (eLife 2025)

**Their design** (from a summary; **to verify against the paper** before citing any detail): eight bars on a ring
(eccentricity ≈ 5°, to verify); four alternating "locations of interest"; the distractor four times more likely at
one of them (57% vs 14% each, to verify); 3T fMRI, n = 28; location ROIs = the 20 most responsive voxels per location
from a flicker localizer in early visual cortex; GLM.

**Their result on search trials:** HP distractor location (HPDL) ≈ the neighbouring location of interest (NL-near, 90°
away) < the diagonally opposite one (NL-far). HPDL vs NL-far t(27) = 3.53, d = .59; NL-near vs NL-far t(27) = 3.06,
d = .51; HPDL vs NL-near d = .08, BF10 = 0.19. The pattern held for distractor, target and neutral items, and on
omission trials (no display), which they read as proactive. (All to verify against the paper.)

**Ours, measured item responses in early areas (Fig. 3e):** HP − next −5% (p = .16), next − opposite −14% (p = .029),
HP − opposite −19% (p = .011), no location × item interaction. That is the same pattern, a replication at 7T with
pRF-defined voxels and deconvolved responses. Phrase it as "the same spatial pattern as Richter et al. (2025)", with
the caveat that our HP vs next difference is numerically larger than theirs but not reliable either.

**Our decoded bar (Fig. 6)** is a different measure: the drive of the bar inside the aperture, between the items, not
the response to the items. There, HP is below its neighbours, and next vs opposite is not reliable (p = .13–.43 across
pRF models; numerically next > opposite). So there is no real conflict with Richter et al.; do not present the
decoding as contradicting their NL-near < NL-far result, and do not present it as replicating it either.

**Proactive.** The decoded bar effect is measured on the bar, across the block, outside the items, which is in line
with a proactive (sustained) change, as their omission-trial result suggests. The bar-crossing supplement points the
same way (lower response at HP when the bar crosses a pRF near an item, p = .017), but it is not significant with a
pre-crossing baseline (p = .16), so cite it only as consistent, not as evidence.

## 5. Open questions and the analyses that will answer them

| Question | Analysis | Status |
|---|---|---|
| Do the gain numbers hold on the final inputs (per-ROI HRF)? | Final DoG rerun, job 8196318; then Fig. 5, Table 1, parameter table, LOSO, Fig. 7 redone | running |
| Is dS below what relabelling HP gives? | HP-rotation null on the conditional dS, 100,000 draws | not yet run |
| Is the sustained loss at HP shared with the neighbours (as in Fig. 3 and Richter et al.), or HP-only? | Split model: separate sustained terms next to HP and opposite HP | running |
| Is the target gain at HP lower than at LP (dT), and does it differ from dD? | Split model: target at HP | running |
| Is the transient-field width separable from the sustained width? | m7_full_sigtied (one shared width) | running |
| Does the HP effect act on DN baselines rather than the input gain? | Baseline-field variants (b, d, both), held-out test, recovery | running (jobs 8203960, 8206646, 8206793, 8206356/8206625) |
| Does the DoG centre/surround balance tilt at HP? | DoG centre/surround tilt model | running |
| Do the results hold with Gaussian and DN pRFs with fitted per-ROI HRFs? | Robustness ladders and decodes (`notes/robustness_plan_2026-11.md`) | running |
| Can the decoder see a gain effect of our size, and would it produce the next-to-HP pattern? | Decoder simulation with an HP-only gain field | running |
| Does gain at HP predict RT depending on what HP holds? | Plan D4, trial-wise model on the final fits | not yet run |
| How reliable is per-subject dS across sessions? | Split-session dS | not yet run |
