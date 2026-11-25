# Answers for the writing agent: Docky's edits (25 Nov 2026)

Full numbers and sources: `reports/docky_followup_2026-11-25.md`. n = 28 throughout. Pooled = per participant the median over nine areas, then median [95% CI], k/28, Wilcoxon p. Trial-wise = posterior mean [95% HDI], P = posterior probability of the sign.

## 0. Before anything else

Both manuscript tabs still carry the gain-field numbers from before the zero-sum refit. Apply update 13 (`notes/writing_package_2026-11-16_update13/UPDATE_NOTES.md`, section 2) first. The key replacements: target gain +23% [+14, +35], p = 4.1 × 10⁻⁷; distractor gain +11% [+1.4, +17], p = .0065; dS alone −1.5% [−2.4, −0.9], 21/28, p = .0071 (no single area survives BH); dS joint −5.5% [−7.6, −2.1], p = .012 (hV4 −8.2%, q = .022); dS / target gain −0.24 [−0.43, −0.10]; distractor gain per SD log RT −2.7 [−4.7, −0.7], P = .996; target repeat −19% [−29, −9].

## 1. Abstract: "the gain was reduced about 20%"

Wrong. The gain at HP is a few percent of the drive lower (−1.5% alone, −5.5% joint), about a quarter of the target gain. The decoded bar is 1.1% weaker in the HP quadrant (median −1.10% [−1.52, −0.34], 21/28, p = .0033, one-sided). Nothing in the gain is 20%; the near-20% numbers are the old ratio to the target gain (−0.20, now −0.24) and the measured early-visual response at HP vs opposite (−21.6%), which is a response, not a gain.

> "At the high-probability location, the gain was reduced by a few percent of the drive, about a quarter of the gain at the target, throughout the block."

"The salient distractor received about half this gain" still holds (+11% vs +23%).

## 2. Discussion: suppression is "divisive rather than subtractive"

Not supported. The data favour a gain over a trial-locked subtraction but cannot separate a divisive from a sustained subtractive mechanism:
- For a gain: the bar response is shallower where the bar crosses HP (slope −2.1% [−4.6, −0.7], 20/28, p = .012), about the size of dS; an additive response at HP locked to each trial is absent (held out 16/28 better, p = .41) although one of matched size would have been found in every simulation.
- Not testable: a subtractive offset sustained over a block (each run is expressed relative to its own mean, and HP is constant within a run).
- Not separable: divisive-normalization baseline variants (S12) predict held-out runs worse in all participants (p = 7.5 × 10⁻⁹), but at our effect size simulated baseline effects are recovered as input gain 72% of the time, so this does not decide.
- Multiplicative vs additive HP change on the item gains (3a): neither supported (h −9.6%, p = .19; a +0.8%, p = .48); both predict held-out runs worse than no term.

> "The reduction at the HP location behaved as a change in gain: the response to the mapping bar was shallower where the bar crossed the HP location, and there was no additive response at that location locked to the trials. A subtractive offset sustained over a whole block cannot be measured in our design, and divisive and gain accounts of an effect this small could not be told apart, so our data favour a change in gain without deciding between divisive and subtractive mechanisms."

Keep the Richter et al. argument but drop "divisive rather than subtractive" and "strongly".

## 3. Gradient over the LP locations (HP < next < opposite)

The variant with separate next-to-HP and opposite-HP sustained gains (earlier fits; no zero-sum or held-out version) finds no gradient: next − opposite −0.6% [−3.2, +2.0], 15/28, p = .73. Only HP vs opposite is reliable (−3.3% [−5.2, −1.9], 23/28, p = .0048). It therefore cannot reproduce the measured next-vs-opposite difference (−14.9 ± 6.0%). Moreover, the measured items put next below opposite while the decoded bar puts next above opposite (+0.68 ± 0.32%, p = .070), so no single term fits both. Keep one shared LP gain and keep the Discussion paragraph that leaves the LP gradients open; it can add:

> "A variant with separate sustained gains next to and opposite the HP location found no difference between them (−0.6%, p = .73)."

## 4. Target repeat: priming or repetition suppression?

- Target gain on target-repeat trials: −19% [−29, −9], P > .999. Log RT is in the same model; with the global gain added as well, −14% [−24, −5], P = .998. So it is not an RT effect.
- A target at the previous distractor location: no change (−1.8 points [−6.5, +3.3], P = .78).
- A neutral item at the previous target location also loses gain (−6.5 points [−10.7, −2.2], P = .996, negative in every area), and so does a neutral item at the previous distractor location (−4.7 [−8.1, −1.1], P = .994).

Neither account fits cleanly. Priming as raised priority would give more gain at that location, not less; repetition suppression of the stimulus does not apply, because a neutral item is a different stimulus from the target before it. Best supported: a location that held the target or the distractor gives less gain to whatever appears there next. Docky's Discussion sentence ("demonstrating a neural adaptation effect ... shifting synaptic connectivity") should be softened to:

> "The target gain was lower when the target repeated its location, and so was the gain of a neutral item at the previous target location, so the reduction belongs to the location rather than to the repeated target. Faster responses to repeated targets therefore did not come with more gain at their location."

## 5. Distractor gain and RT

**a. Sign.** Distractor gain is lower on slower trials: −2.7 [−4.7, −0.7] per SD of log RT, P = .996. The abstract ("distractor gain was higher on fast response trials") matches. The Discussion sentence matches in direction but needs two changes: drop "reliably", and drop "after regressing out the sustained suppression" (the model holds the sustained field fixed; nothing is regressed out).

> "Distractor gain was lower on trials with slower responses."

**b. Looser voxel selections.** It weakens but stays credible: r > 0.2 −2.5 [−4.5, −0.6], P = .994; r > 0.1 −1.9 [−3.7, +0.0], P = .978. Replace the Results clause "became weaker with looser voxel selections, so we interpret it with caution" with:

> "The effect became smaller with looser voxel selections but kept its sign (r > 0.1: −1.9%, P = .978), and it depends on weighting each estimate by its precision, so we interpret it with caution."

It is also not credible in the joint latent RT model, where only the global gain predicts RT (−17 ms per SD [−31, −4], P = .995). Keep the cautious framing; the EEG Pd paragraph should not lean on it as a firm result.

**c. Early vs late opposite RT relations.** This was the October 2026 mixed-model analysis (log RT on per-trial gains, n = 30, 11 areas). There, a larger distractor gain went with faster responses in V1/V2 and slower in SPL1, but the robust reversal was in the overall trial amplitude, and distractor-specific signs in dorsal areas were not identifiable (gains r = .92 to .98 collinear). The model, voxels and areas (SPL1, FEF dropped) have changed since. In the current model the relation is negative in all nine areas (−1.8 to −3.7). Do not mention an early/late reversal.

## 6. Fig. 1

Use Fig. 1 v6 (`notes/figures/paper_2026-11/fig1_paradigm_behaviour_v6_n28_palA`). It already has the strips: LP − absent +42.6 ms [+27.0, +50.7], 26/28, p = 1.0 × 10⁻⁷; HP − absent +14.6 ms [+0.3, +21.7], 19/28, p = .0041. The manuscript still shows the older version.

## 7. Fig. 5a with HP on a diagonal

The alternative is `notes/figures/paper_2026-11/fig5_gain_fields_v11_abc_zs_n28_palA_hpdiag.pdf`, a pure display rotation (HP at the upper right). If it replaces the current one, the caption reads "The HP location is at the upper right" instead of "at the top". Fig. 4a already has HP on a diagonal; Fig. 6 still has HP at the top.

## 8. Supplementary videos

Feasible; drafts in `notes/figures/videos/` (animated Fig. 4b schematic, 12 s; true vs decoded bar in V1, held-out run, 16 s). The Fig. 4b video uses the schematic's exaggerated gains; with the fitted gains the effect would not be visible. Possible legends:
- "Supplementary Video 1. The gain-field model (schematic, Fig. 4b) over a stretch of a run: the effective drive on the screen, the stimulus in one pRF, the gain it sees, and the predicted BOLD response with and without gain, in a block with the HP location near the pRF and one with it opposite."
- "Supplementary Video 2. The true bar and the bar decoded from V1 in a held-out run (participant 3), and the decoded bar averaged over participants."

## 9. p values

Table of all 69 p and q values in the Results (exact, threshold form, Docky's current form, update-13 value) is in section 9 of the full report. Fix one typo in Docky's tab ("p ==< 0.05"), and note the early-pathway target − distractor test is q = .015, not < .001.
