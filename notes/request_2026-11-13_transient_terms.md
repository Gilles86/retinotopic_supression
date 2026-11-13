# Request, 13 Nov 2026: transient HP terms, the HP reduction and the item gains

Received as a request text in the coordinating session.

**Context.** In the manuscript we are deciding whether the transient HP − LP terms (dT, dD; Fig. 5d) stay in the main
text or move to the supplement. We also need to answer the question: "how does the reduction at the HP location relate
to the priority given to the target and the distractor?"

**Ground rules for every item:**
- Use the refitted fits (update 8), n = 28.
- Use the same statistics and reporting format as key_numbers.md: median, bootstrapped 95% CI, k of n, Wilcoxon, and
  BH across areas where a result is per area.
- Put a plain-language summary line under every table.
- Do not polish figures yet.

## 1. Do the transient HP terms add anything?

- **a. Held-out prediction.** Compare the joint fit with dT and dD, without them, and with dS only. Use
  leave-one-run-out cross-validation and report ΔR² per participant and area, pooled and per pathway.
- **b. Evidence for the null.** Bayes factors, or equivalence tests with a stated bound, for dD = 0, dT = 0 and
  dD − dT = 0, pooled over areas.
- **c. Sensitivity.** Simulate from each participant's fitted model with a distractor-specific reduction at HP of
  −2%, −5% and −10%, and with none. Add realistic noise (phase-randomised residuals), refit, and report how often dD and
  dD − dT are detected. We need to know which size of distractor-specific effect we could have seen.

## 2. Does the sustained field alone explain the item responses at HP?

Pass the noise-free predictions through the Fig. 3 analysis (as in discussion_checks section 2), separately for
target, distractor and neutral items, for three models:
- sustained HP − LP only (no dT, dD);
- the joint fit;
- no HP terms at all.

Report predicted vs measured HP − LP per item type, and the location × item-type interaction, in early areas and per
pathway.

## 3. How do the HP reduction and the item gains combine?

- **a. Multiplicative or additive?** Is the reduction at HP proportional to the gain an item receives, or the same for
  every item? Compare two versions of the joint model:
  - HP scales the target and distractor gains by a common factor;
  - HP adds a common offset to them.

  Report held-out ΔR² and which version each participant prefers.
- **b. Sizes side by side,** in the same units and from the same fit: target gain, distractor gain and sustained
  HP − LP, per area and pooled, plus the ratio (sustained HP − LP) / target gain.
- **c. Across participants and across areas:** does a larger target or distractor gain go with a larger reduction at
  HP? Spearman ρ with CI, within and between participants.

## 4. Bookkeeping

- **a.** One table stating, for every number in the gain-field section of the Results (target, distractor, dS alone,
  dS joint, dT, dD, widths), which fit it comes from.
- **b.** A version of Fig. 5 without panel d (panels a to c only), and panel d as a separate supplementary figure, so
  we can choose. Keep v11 as it is.

## 5. Two checks still open from yesterday (same package if possible)

- **a. Fig. 3d/e.** For the dorsal and occipitotemporal pathways, in relative and absolute units:
  - HP − opposite, next − opposite and HP − next, with bootstrapped 95% CIs;
  - the pathway × location interaction;
  - whether the early-area estimate lies inside the CI of each later pathway.
- **b. Model vs measured for the item-type contrasts of Fig. 3b, c.** For target − neutral, distractor − neutral and
  target − distractor, per pathway and for V1, V2 and V3 separately. Also, per area, the number of voxels whose pRF
  overlaps an item location. We need to know whether "target gain absent in V1 and V2" is a property of the data or a
  lack of sensitivity.

## Delivery

A writing package update with:
- key_numbers additions;
- the tables as .md;
- UPDATE_NOTES stating, for each item, what it means for the text in one or two sentences.
