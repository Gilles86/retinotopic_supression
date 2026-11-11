# Claims and caveats (2026-11-12, final fits, n = 28)

The guardrail for the text. A sentence in the paper should map onto a claim in section 1, be phrased no more strongly
than it is here, and not contradict section 2. Numbers: `key_numbers.md`. All gain numbers are on the final fits (DoG
pRFs, one jointly fitted HRF per subject × area, voxels reliable across runs). **n = 28**: the two pilot participants
(sub-01, sub-02; faster bar, three-location HP schedule) are left out of everything; every claim below also holds in
direction with them included (`figures/tableS_sensitivity_n30.md`).

**How the argument is built (user decision, 2026-11-10).** "I don't want to lead with a model comparison that is noisy
and is a null. Let's try to make a narrative that's easy to follow." The paper argues from what the gain fields look
like: the fitted parameters, their consistency across subjects and areas, and their agreement with the model-free item
responses (Fig. 3) and the gain-free decoding (Fig. 6). Held-out prediction (leave-one-subject-out, LOSO) is reported in
Methods and a supplementary table, described neutrally; no main-text sentence leads with it or hinges on it.

## 1. Claims the data support

**Behaviour (Fig. 1).**
- Responses are faster when the distractor appears at the HP location than at another location, in all 28
  participants, and slower when the target appears there.
- A target or distractor at the same location as on the previous trial speeds responses.

**pRFs (Fig. 2).**
- pRFs mapped during search are retinotopic and mostly contralateral, and their size grows with eccentricity beyond V1.
- A DoG pRF and one HRF per subject × area, fitted jointly with the pRFs, each improve held-out prediction over a
  Gaussian pRF with the canonical HRF, in nearly every subject.
- The search display drives voxels according to the distance of their pRF from the nearest item.
- V1 pRF size does not grow with eccentricity here; say it is likely limited by the resolution of the paradigm.

**Responses at the item locations, without a model (Fig. 3).**
- Voxels whose pRF lies at an item location respond more when the target or the distractor is there than when a
  neutral item is, in all three pathways.
- In early visual cortex (V1–V3 pooled) the response depends on where the item is relative to HP: least at HP, less next
  to HP than opposite it (HP − opposite and next − opposite reliable; HP vs next not; location type F(2, 54) = 7.67,
  p = .001).
- The location effect does not differ between target, distractor and neutral items: it is a property of the location.
- Dorsal and occipitotemporal areas show no reliable location effect.
- At the item locations the target response exceeds the distractor response in every pathway, early visual cortex
  included (early +.038% signal, q = .015): target > distractor > neutral, as in Richter et al. 2025.

**The gain fields (Fig. 5).**
- Target gain is positive and reliable from V3 on (V3–TO except LO after correction) and grows along the hierarchy
  (with pRF size).
- Distractor gain is positive but weaker, reliable in V3AB, hV4 and TO (joint fit; also on top of the target term).
- Sustained gain is lower at the HP location than at the LP locations (dS < 0): pooled over areas when estimated alone
  (79% of subjects) and in the joint fit (79%), reliable per area in V3 and TO (alone) and V3, hV4 and TO (joint),
  negative in every area. The pooled effect, consistent across subjects, is the claim, not any single area.
- The lower sustained gain at HP holds with Gaussian and DN pRFs too, estimated alone (75–82% of subjects in each) and
  in the joint fit (p ≤ .027; supplement).
- The sustained gain is non-zero at the LP locations too (joint fit, g_LP pooled −.067, negative in 24/28 subjects, p = 2.3 × 10⁻⁶; g_HP −.148): a
  field at HP alone (g_LP = 0) is rejected. All four locations are searched (the target can appear at any of them), so
  no location is a gain-free reference; the HP effect is defined relative to the other three (dS = g_HP − g_LP).
- The gain at HP is lowest at HP itself; next to HP and opposite HP do not differ (split model).
- Transient gain at HP is slightly lower than at LP for the target (pooled p = .018); for the distractor it points the
  same way but is not reliable (p = .099); the two do not differ (p = .24). Say "less transient gain at HP for whatever appears there", in line with Fig. 3 (no
  location × item interaction) and with the slower responses to a target at HP (Fig. 1).
- The sustained field is about 3.2° wide at half maximum (≈ 5° in V1–V3, ≈ 3° beyond), narrower than the 5.7° spacing
  of neighbouring items in most areas.

**Decoding without gain terms (Fig. 6).**
- The bar can be decoded from held-out runs in all nine areas, best in V1–V3, with pRFs that have no gain terms.
- The bar is decoded at lower intensity in the quadrant facing HP than in the neighbouring quadrants, pooled over V1–V3
  (and per area in V2 and V3 after correction across nine areas);
  the effect is spread across eccentricity, which argues against item responses leaking into the aperture.
- With the stimulus held identical, the half of a bar facing HP is decoded weaker than its other half, for vertical and
  horizontal bars.
- A simulated gain field at HP only, of about the fitted size, is recovered by the decoder at a similar magnitude.

**Gain and behaviour (Fig. 7).**
- Within subjects, the target gain is lower on trials where the target repeats its location (about 19% of the typical
  target gain, averaged over areas); the distractor gain shows no credible change when the distractor repeats.
- On slower trials the distractor gain is lower; the target gain does not change with RT. (The global gain over the
  whole display points the same way but is not credible at n = 28, P = .943; do not claim it.)

## 2. Claims the data do NOT support (do not write these)

- **"The HP effect improves out-of-subject prediction."** On the final fits it does not: dS added with the other
  subjects' size predicts a new subject better in 68% (p = .41). Only its direction generalises (75%, p = .0071).
  (On the earlier fits it was 77%, p = .010; do not cite that.)
- **Distractor-specific suppression at HP.** The transient HP − LP term is about as negative for the target as for the
  distractor, they do not differ, the joint fit shows neither, and the measured responses have no location × item
  interaction. Never "the distractor, unlike the target, is suppressed at HP". (The 09 package had dD reliable in four
  areas; on the final fits only LO, and pooled p = .099.)
- **Neighbour suppression or enhancement in the gain fields.** The split model finds next to HP = opposite (p = .73).
  The measured item responses (Fig. 3) show next < opposite in early areas; the decoded bar shows next ≥ opposite
  (n.s.). Do not claim either direction for the neighbours from the model; present Fig. 3 as measured.
- **The neighbour decrease is not a centre/surround effect of the field at HP** (2026-11-11, recomputed at n = 28;
  `notes/data/n28/kappa_neighbour_2026-11_n28.tsv`, `retsupp/visualize/paper/kappa_neighbour_summary.py`). The fitted
  DoG centre/surround tilt (κ < 0: slightly more of the HP effect on the suppressive surround), run through the Fig. 3
  pipeline on noise-free model predictions, changes next − opposite by +0.27% (median +0.29%, 61% of subjects > 0,
  p = .45; all areas +0.27%, p = .18) and the decoded bar drive by +0.06% (p = .33); neither the joint fit (m7_full,
  −1.3%, p = .16) nor the κ model reproduces the measured −14.9%. Discussion sentence: "A sustained HP field that
  suppresses the pRF surround slightly more than its centre (the fitted tilt) predicts, if anything, a small relative
  increase next to HP (+0.3% of the mean response in early visual cortex, n.s.), opposite in sign and about fifty times
  smaller than the measured decrease (−15%); the lower response next to HP is therefore not a centre/surround effect
  of the gain field at HP." Why small: the sustained field (σ ≈ 2.0° in these fits) has fallen to a few % at the
  neighbouring item 5.66° away, early-area surrounds are ≈ 1.8 × the centre σ of ≈ 0.72° (16% of the centre amplitude),
  and κ is ≈ −0.07 in early areas.
- **A field wider than the item spacing.** That was the earlier fits (4.8°); the final sustained field is 3.2°.
- **Single-area decoding effects beyond V2 and V3**: only V2 and V3 survive BH across nine (q = .038); do not claim
  others (hV4, VO, TO, V3AB q = .15).
- **DN-baseline or normalization accounts.** The data cannot place the HP effect on the DN baselines rather than on the
  input gain (held-out worse in 28/28; recovery reads such effects as input gain). Do not say the effect "lives in"
  normalization.
- **The DoG centre/surround tilt (κ) as a finding.** Its direction is consistent (93% of subjects at n = 28) but it is
  tiny and barely helps prediction (68%, p = .043; worse on held-out runs); supplementary figure `figS_dog_kappa_n28`
  and a sentence at most.
- **Items improve prediction in 100% of subjects.** That row scores item weight on bar-only pRFs against items at bar
  strength, not items vs none. Do not cite it.
- **Brain–behaviour across subjects.** dS does not relate to the behavioural HP benefit (dS alone ρ = −.11 [−.50,
  +.30], p = .57; final fits, n = 28). Report as a null with the power caveat only if needed.
- **A global-gain RT effect** (Fig. 7b): not credible at n = 28 (P = .943); it was P = .979 with the pilots.
- **The tied distractor gain improves out-of-subject prediction**: not at n = 28 (61%, p = .16).
- **RT regressed on per-trial gains** (the October Fig. 7): retired; on the final fits it gives isolated effects only.
- **Learning over runs, proactive vs reactive, pRF shifts**: not supported or out of scope (pRF shifts archived; do not
  mention).

## 3. Caveats (Discussion notes and Methods, not the storyline)

- **Dependence on the pRF model.** dS is negative under all three pRF models in the joint fit (DoG −.034, p = .0027; DN
  −.078, p = .0024; Gaussian −.088, p = .027). The decoding result holds with Gaussian pRFs and somewhat weaker with DN
  pRFs (HP vs next unconstrained decoder p = .036, outline decoder p = .008). Target and distractor gain are larger under Gaussian
  and DN pRFs. Estimated alone, dS is negative under all three (DoG −.014, Gaussian −.008, DN −.028; 75–82% of subjects; p = 1.1 × 10⁻⁴, 4.2 × 10⁻⁴, .0052), and its direction generalises across subjects in all three (sign-flip p = .0071, .0044, .038). The transient HP − LP terms are less consistent across models (dD: DoG p = .099, Gaussian .69, DN .24). The distractor gain predicts new subjects worse under DN pRFs (18% better, p = .018). Figs. 3 and 7 were not redone with Gaussian or DN pRFs.
- **Held-out prediction (LOSO).** Target and sustained gain predict new subjects (the tied distractor gain does not at
  n = 28: 61%, p = .16); the HP-specific terms
  (dS, dD, dT) do not, though dS and dT point the right way in most subjects (sign-flip test). The effects are small
  relative to between-subject variability.
- **Fig. 7 depends on weighting by uncertainty.** The RT effects (panel b) hold only when each subject × area estimate
  is weighted by its jackknife SE (unweighted P ≈ .5–.7); the target-repeat effect keeps its direction unweighted (P = .968, just below .975).
  The target-repeat effect is clearest as the mean over all areas (P = .997); no group alone reaches P > .975. Panel a is
  relative to each area's typical gain (median |gain|, because the signed median is near 0 in V1 and V2).
- **Fig. 7 is a different readout from the older version**: how the gains shift with RT, not how gains predict RT.
- **HP lies at 4°, outside the 3.17° bar aperture**: only pRFs reaching the ring carry the gain effect, and the decoded
  bar measures drive between the items, not at them.
- **Fig. 3 contrasts** are uncorrected planned contrasts; the location-type main effect is the protected test.
  One dorsal subject (sub-30) is left out of the relative analyses (mean response not positive); dorsal n = 26 (b, c)
  and 25 (d, e).
- **Decoding statistics**: one-sided Wilcoxon per area, BH across nine areas. No test is framed as pre-specified.
- **The pilots.** Leaving them out was decided on paradigm grounds (faster bar, different HP schedule) after the
  sensitivity analysis had been run; say so plainly if asked, and point to `tableS_sensitivity_n30.md`: no result changes
  sign, the HP-location effects are stronger without them, the weakest distractor-gain and global-gain statements weaker.
- **V1 pRF sizes** are at a floor.
- **Transient field widths** often sit at a σ bound; do not interpret them.

- **Sessions and carry-over (Richter follow-up A1–A4; decision for the authors, see README).** HP moves to the
  opposite location within a session and to the other diagonal between sessions, so in session 2 the two neighbours of
  HP were former HP locations. The effects do not differ reliably between sessions or blocks: that interaction test is
  the inference (A1 session and block differences p ≥ .26; A3 sustained dS session 2 − 1 −.022, p = .095). Per-session
  values are descriptive only: each session has half the data, so per-session tests are underpowered, and the
  difference between a significant and a non-significant test is not necessarily significant. Never write "a
  session-2 effect", "only in session 2" or "builds up across sessions". A location that was HP before does not
  respond less than a never-HP location (A2, p = .54), and behaviour shows no carry-over once practice is taken out
  (A4): no evidence of carry-over. Carry-over cannot be strictly excluded for the neighbour effect (in session 2 the
  neighbours were former HP locations), but the data give no sign of it.
- **The gain-field model reproduces about half of the measured HP − LP reduction** (early −5.4 vs −11.7%), carried by
  the sustained HP − LP term, and none of next − opposite (−1.3 vs −14.9%). Do not present the model as accounting for
  the full measured pattern.
- **Behaviour by location relative to HP** (Richter follow-up B): a distractor next to HP is the slowest (38 ms slower
  than opposite or at HP); opposite HP is about as fast as HP (−3.8 ms, n.s.); the target cost is graded HP > next >
  opposite. So "the benefit at HP" against all LP locations is partly a cost next to HP. Richter et al. report the
  benefit only at HP and no target cost.

- **Model vs measured (2026-11-12).** The joint fit's predicted HP − LP lies inside the measured CI (gap p = .095), so
  do not write that the model "explains only half"; write that it accounts for the reduced response at HP relative to
  its neighbours but not for the next < opposite gradient (gap p = .040).
- **Behaviour next to HP is not carry-over (2026-11-12).** The pattern (next slowest, opposite ≈ HP, graded target
  cost) is present in session 1 block 1, before any location had been HP; no history interaction (p ≥ .79).

- **Fixation in the scanner was not monitored (2026-11-12).** The eye tracker ran only in the behavioural pretest
  (beep when gaze left a 1° window during search); it was off in both MRI sessions. Write this in the Methods and, if
  needed, as a limitation; do not claim fixation control in the scanner.
- **Distortion correction was fieldmap-based (2026-11-12).** fMRIPrep's reports show phase-difference-map SDC in every
  run (no SyN), despite `--force syn-sdc` in the call; write "fieldmap-based (phase-difference) distortion correction".
- **2D vs 3D EPI is unresolved.** The sidecar's "3D" is from a hand-written template and conflicts with multiband 3;
  do not write "3D EPI" until the exam card or a PAR header is checked.
- **κ does not change dS (2026-11-12, n = 28):** median change −0.0001, p = .33.

## 4. Relation to Richter, van Moorselaar & Theeuwes (eLife 2025)

As in the 09 package; their design details are to verify against the paper (`⟦VERIFY⟧`). Their result: HP distractor
location ≈ the neighbouring location of interest < the diagonally opposite one, for distractor, target and neutral
items, and on omission trials. Ours (Fig. 3e, early areas, final fits, n = 28): HP − next −6.7% (p = .095),
next − opposite −14.9% (p = .022), HP − opposite −21.6% (p = .005), no location × item interaction: the same spatial pattern at 7T
with pRF-defined voxels. The fitted sustained field (split model) does not reproduce the neighbour effect (next =
opposite) and the decoded bar shows next ≥ opposite (n.s.), so the neighbour pattern rests on the measured responses
only; do not present the model or the decoding as replicating it, nor as contradicting it. Richter follow-up
(2026-11-11): the location effects do not differ reliably between sessions, and there is no evidence of carry-over
of earlier HP locations (section 3); target >
distractor > neutral in early visual cortex as in their data; per early area the HP − opposite reduction is clearest
in V1 (−29.8%, p = .002), then V2 (−20.0%, p = .010), V3 n.s., unlike their V2 > V1 (exploratory, uncorrected);
behaviourally ours differs from theirs (neighbour cost for the distractor, graded target cost).

## 5. Open items

| Item | Status |
|---|---|
| Fig. 3 with Gaussian and DN pRF centres | not run (the export uses DoG model code; would need family-specific bar predictions) |
| Fig. 7 with Gaussian and DN fits | not run |
| HP-rotation null for dS | not run; proposed to drop unless the user wants it |
| Within-subject CV power simulation (figS_cv_power) | on the earlier fits; left out of this package, since LOSO is now supplementary |
| Between-subject dS vs behaviour | final fits, n = 28: ρ = −.11, p = .57; text only if needed |
| Richter et al. follow-up | done (`richter_followup_2026-11-11.md`); the Discussion consequences are decisions for the authors (README) |
