# Claims and caveats (2026-11-10, final fits)

The guardrail for the text. A sentence in the paper should map onto a claim in section 1, be phrased no more strongly
than it is here, and not contradict section 2. Numbers: `key_numbers.md`. All gain numbers are on the final fits (DoG
pRFs, one jointly fitted HRF per subject × area, voxels reliable across runs).

**How the argument is built (user decision, 2026-11-10).** "I don't want to lead with a model comparison that is noisy
and is a null. Let's try to make a narrative that's easy to follow." The paper argues from what the gain fields look
like: the fitted parameters, their consistency across subjects and areas, and their agreement with the model-free item
responses (Fig. 3) and the gain-free decoding (Fig. 6). Held-out prediction (leave-one-subject-out, LOSO) is reported in
Methods and a supplementary table, described neutrally; no main-text sentence leads with it or hinges on it.

## 1. Claims the data support

**Behaviour (Fig. 1).**
- Responses are faster when the distractor appears at the HP location than at another location, in all 30
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
  to HP than opposite it (HP − opposite and next − opposite reliable; HP vs next not; location type F(2, 58) = 6.28,
  p = .003).
- The location effect does not differ between target, distractor and neutral items: it is a property of the location.
- Dorsal and occipitotemporal areas show no reliable location effect.

**The gain fields (Fig. 5).**
- Target gain is positive in every area beyond V1 and grows along the hierarchy (with pRF size).
- Distractor gain is positive but weaker, reliable in V3AB, hV4 and TO (joint fit; also on top of the target term).
- Sustained gain is lower at the HP location than at the LP locations (dS < 0): pooled over areas when estimated alone
  (80% of subjects) and in the joint fit (77%), reliable per area in V3 and TO (alone) and V3 and hV4 (joint), negative
  in every area. The pooled effect, consistent across subjects, is the claim, not any single area.
- The lower sustained gain at HP holds with Gaussian and DN pRFs too, estimated alone (77–83% of subjects in each;
  supplement).
- The gain at HP is lowest at HP itself; next to HP and opposite HP do not differ (split model).
- Transient gain at HP is slightly lower than at LP for the target (pooled p = .016) and for the distractor (p = .050),
  and the two do not differ. Say "less transient gain at HP for whatever appears there", in line with Fig. 3 (no
  location × item interaction) and with the slower responses to a target at HP (Fig. 1).
- The sustained field is about 3.3° wide at half maximum (≈ 5° in V1–V3, ≈ 3° beyond), narrower than the 5.7° spacing
  of neighbouring items in most areas.

**Decoding without gain terms (Fig. 6).**
- The bar can be decoded from held-out runs in all nine areas, best in V1–V3, with pRFs that have no gain terms.
- The bar is decoded at lower intensity in the quadrant facing HP than in the neighbouring quadrants, pooled over V1–V3;
  the effect is spread across eccentricity, which argues against item responses leaking into the aperture.
- With the stimulus held identical, the half of a bar facing HP is decoded weaker than its other half, for vertical and
  horizontal bars.
- A simulated gain field at HP only, of about the fitted size, is recovered by the decoder at a similar magnitude.

**Gain and behaviour (Fig. 7).**
- Within subjects, the target gain is lower on trials where the target repeats its location (about 19% of the typical
  target gain, averaged over areas); the distractor gain shows no credible change when the distractor repeats.
- On slower trials the distractor gain and the global gain over the whole display are lower; the target gain does not
  change with RT.

## 2. Claims the data do NOT support (do not write these)

- **"The HP effect improves out-of-subject prediction."** On the final fits it does not: dS added with the other
  subjects' size predicts a new subject better in 67% (p = .18). Only its direction generalises (77%, p = .0024).
  (On the earlier fits it was 77%, p = .010; do not cite that.)
- **Distractor-specific suppression at HP.** The transient HP − LP term is about as negative for the target as for the
  distractor, they do not differ, the joint fit shows neither, and the measured responses have no location × item
  interaction. Never "the distractor, unlike the target, is suppressed at HP". (The 09 package had dD reliable in four
  areas; on the final fits only LO, and pooled p = .050.)
- **Neighbour suppression or enhancement in the gain fields.** The split model finds next to HP = opposite (p = .87).
  The measured item responses (Fig. 3) show next < opposite in early areas; the decoded bar shows next ≥ opposite
  (n.s.). Do not claim either direction for the neighbours from the model; present Fig. 3 as measured.
- **A field wider than the item spacing.** That was the earlier fits (4.8°); the final sustained field is 3.3°.
- **Single-area decoding effects after correction**: no area survives BH across nine (smallest q = .076).
- **DN-baseline or normalization accounts.** The data cannot place the HP effect on the DN baselines rather than on the
  input gain (held-out worse in 30/30; recovery reads such effects as input gain). Do not say the effect "lives in"
  normalization.
- **The DoG centre/surround tilt (κ) as a finding.** Its direction is consistent (90% of subjects) but it is tiny and
  does not help prediction; supplement sentence at most.
- **Items improve prediction in 100% of subjects.** That row scores item weight on bar-only pRFs against items at bar
  strength, not items vs none. Do not cite it.
- **Brain–behaviour across subjects.** dS does not relate to the behavioural HP benefit (ρ = −.17, p = .37, earlier
  fits). Report as a null with the power caveat only if needed.
- **RT regressed on per-trial gains** (the October Fig. 7): retired; on the final fits it gives isolated effects only.
- **Learning over runs, proactive vs reactive, pRF shifts**: not supported or out of scope (pRF shifts archived; do not
  mention).

## 3. Caveats (Discussion notes and Methods, not the storyline)

- **Dependence on the pRF model.** dS is negative under all three pRF models in the joint fit (DoG −.031, p = .005; DN
  −.074, p = .0076; Gaussian −.063, p = .078, n.s.). The decoding result holds with Gaussian pRFs and is weaker with DN
  pRFs (unconstrained decoder p = .10, outline decoder p = .011). Target and distractor gain are larger under Gaussian
  and DN pRFs. Estimated alone, dS is negative under all three (DoG −.012, Gaussian −.005, DN −.019; 77–83% of subjects; p = 5.6 × 10⁻⁵, 2.3 × 10⁻⁴, .0032), and its direction generalises across subjects in all three (sign-flip p = .0024, .011, .045). The transient HP − LP terms are less consistent across models (dD: DoG p = .050, Gaussian .44, DN .13). Figs. 3 and 7 were not redone with Gaussian or DN pRFs.
- **Held-out prediction (LOSO).** Target, distractor and sustained gain predict new subjects; the HP-specific terms
  (dS, dD, dT) do not, though dS and dT point the right way in most subjects (sign-flip test). The effects are small
  relative to between-subject variability.
- **Fig. 7 depends on weighting by uncertainty.** The RT effects (panel b) hold only when each subject × area estimate
  is weighted by its jackknife SE (unweighted P ≈ .65); the target-repeat effect holds either way (unweighted
  P = .978). The target-repeat effect is clearest as the mean over all areas; no group alone excludes 0. Panel a is
  relative to each area's typical gain (median |gain|, because the signed median is near 0 in V1 and V2).
- **Fig. 7 is a different readout from the older version**: how the gains shift with RT, not how gains predict RT.
- **HP lies at 4°, outside the 3.17° bar aperture**: only pRFs reaching the ring carry the gain effect, and the decoded
  bar measures drive between the items, not at them.
- **Fig. 3 contrasts** are uncorrected planned contrasts; the location-type main effect is the protected test.
  One dorsal subject (sub-30) is left out of the relative analyses (mean response not positive).
- **Decoding statistics**: one-sided Wilcoxon per area, BH across nine areas. No test is framed as pre-specified.
- **V1 pRF sizes** are at a floor.
- **Transient field widths** often sit at a σ bound; do not interpret them.

## 4. Relation to Richter, van Moorselaar & Theeuwes (eLife 2025)

As in the 09 package; their design details are to verify against the paper (`⟦VERIFY⟧`). Their result: HP distractor
location ≈ the neighbouring location of interest < the diagonally opposite one, for distractor, target and neutral
items, and on omission trials. Ours (Fig. 3e, early areas, final fits): HP − next −6.1% (p = .135), next − opposite
−13.0% (p = .047), HP − opposite −19.0% (p = .011), no location × item interaction: the same spatial pattern at 7T
with pRF-defined voxels. The fitted sustained field (split model) does not reproduce the neighbour effect (next =
opposite) and the decoded bar shows next ≥ opposite (n.s.), so the neighbour pattern rests on the measured responses
only; do not present the model or the decoding as replicating it, nor as contradicting it.

## 5. Open items

| Item | Status |
|---|---|
| Fig. 3 with Gaussian and DN pRF centres | not run (the export uses DoG model code; would need family-specific bar predictions) |
| Fig. 7 with Gaussian and DN fits | not run |
| HP-rotation null for dS | not run; proposed to drop unless the user wants it |
| Within-subject CV power simulation (figS_cv_power) | on the earlier fits; left out of this package, since LOSO is now supplementary |
| Between-subject dS vs behaviour | earlier fits; text only if needed |
