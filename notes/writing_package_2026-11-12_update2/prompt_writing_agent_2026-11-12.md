# Prompt for the writing agent (2026-11-12): revision pass — n = 28 (pilots left out)

**This is a revision pass, not a redraft.** The current manuscript exists (drafted from earlier packages at n = 30).
Keep its structure and prose; update it to this package:

1. **n = 30 → n = 28 everywhere.** The two pilot participants (sub-01, sub-02) are left out of every analysis (user
   decision, 2026-11-12). Replace every number with the n = 28 value in `key_numbers.md`; every "30 participants",
   "30/30", "n = 30", degrees of freedom (e.g. F(2, 58) → F(2, 54)) and subject count changes. Do not keep any n = 30
   number in the main text; the n = 30 values live only in Supplementary Table `tableS_sensitivity_n30.md`.
2. **Methods: the exclusion.** Add the sentence in `README.md` (CHANGES 2026-11-12): the pilots were tested with a
   pilot version (bar 0.5°/s instead of 0.168°/s; HP at three locations in blocks of 2, 3 and 1 runs), excluded from all
   analyses, leaving 28; including them does not change the direction of any result (Supplementary Table S, the new
   `tableS_sensitivity_n30.md`). Participants paragraph: n = 28 demographics from `key_numbers.md` (Methods facts).
3. **Claims that changed** (see `claims_and_caveats.md`): Fig. 7b now claims only the distractor gain (the global gain is
   not credible at n = 28, P = .943); Fig. 5d's transient HP − LP reduction is "target and distractor alike", reliable
   alone only for the target (dD p = .099); the tied distractor gain no longer improves out-of-subject prediction;
   decoding per area now has V2 and V3 surviving BH; target gain per area is reliable from V3 on (V2, LO no longer).
4. **Figure references** to the `_n28` files and panel letters in the `README.md` figure table. **Fig. 1 has a new
   layout:** the target and distractor priming effects are one panel, **Fig. 1e** (within-subject repeat − switch
   differences); their raw response times are Supplementary `figS_priming_rt_v4_n28`. Replace every "Fig. 1e, f" by
   "Fig. 1e". Fig. 6: a, b, c, d (HP vs next, % of the four-quadrant mean), e (eccentricity); the per-area test is
   Supplementary `figS_decoding_hp_per_area_n28`. New supplementary figure `figS_dog_kappa_n28` (κ).
5. **Fixes:** decoder voxels "at most 2000 per area" (not 200); trial counts (20,040 trials, 17,292 correct).
6. The Discussion sentence on κ and the neighbour pattern in `claims_and_caveats.md` (now −15%, n = 28), and the
   Methods sentence on why the sustained field has a gain at all four locations (`key_numbers.md`, Fig. 5 S paragraph).

Mark every remaining gap as `⟦TODO: …⟧` (ethics body/number, payment, the sidecar and distortion-correction checks
listed as open in `key_numbers.md`). The Richter et al. follow-up is merged (`key_numbers.md` section, report
`richter_followup_2026-11-11.md`): add its new facts where they belong (target > distractor > neutral in early areas;
behaviour by location relative to HP; per-area location effect), but **do not resolve the three "Decisions for the
authors" in `README.md`** (sessions and carry-over; the model reproduces about half of the measured HP reduction;
behaviour next to HP): where the text touches them, keep the current wording and insert `⟦DECISION: …⟧` with the
relevant numbers. Never describe an effect as "a session-2 effect", "only in session 2" or "building up across
sessions": the session differences are not significant, and that interaction is the inference. Return a list of all changes you made and all tokens.

You are revising the retsupp paper (7T fMRI, visual search with a high-probability distractor location, pRF mapping
during search, gain-field models) together with Gilles de Hollander. Write in his voice: load the
**scientific-writing** skill first and follow it (declarative Results headings, prediction before result, colon with
capital letter, intervals with every effect size, no bullet lists in Results or Discussion).

## The story you are telling

One easy-to-follow storyline, seven figures, each answering one question that leads to the next:

1. People learn to suppress the HP location (faster when the distractor is there, in all 28 participants; slower when
   the target is there) and are primed by the previous trial's locations (Fig. 1).
2. pRFs mapped during search let us see the display through each voxel's receptive field (Fig. 2).
3. Without any model, voxels at the item locations respond more to the target and the distractor than to a neutral
   item, and in early visual cortex least when the item is at the HP location, for every item type (Fig. 3).
4. A gain-field model describes how attention could reshape the drive each pRF receives: transient gain at the target
   and the distractor, sustained gain at the four item locations that can differ at HP (Fig. 4).
5. The fitted gain fields: target gain rising along the hierarchy, weaker distractor gain, and lower sustained gain at
   HP, consistent across participants and areas; the transient gain at HP is slightly lower, target and distractor
   alike, so the reduction belongs to the location (Fig. 5).
6. Decoding the display from pRFs without any gain terms shows the same at the HP location: the bar is read out weaker
   in the quadrant facing HP, pooled over V1–V3 and per area in V2 and V3 (Fig. 6).
7. Trial by trial, the gain fields move with history and behaviour: the target gain drops when the target repeats its
   location, and the distractor gain is lower on slower trials (Fig. 7).

The argument rests on what the gain fields look like (parameter estimates and their consistency across subjects and
areas), together with the model-free responses (Fig. 3) and the gain-free decoding (Fig. 6). Do not lead with, or
hinge any main-text sentence on, held-out model comparison (leave-one-subject-out): that goes to Methods and a
supplementary table, described neutrally (user decision, 2026-11-10). Caveats (dependence on the pRF model,
out-of-subject prediction, the SE weighting in Fig. 7) belong in the Discussion and Methods, not in the storyline.

## Read, in this order

All in the package folder (`/Users/gdehol/git/retsupp/notes/writing_package_2026-11-12/`, zipped as
`writing_package_2026-11-12.zip`):

1. `README.md`: storyline, decisions, words, outline, figure list.
2. `claims_and_caveats.md`: **the guardrail.** Every claim you write must be in its section 1, phrased no more
   strongly than there; nothing from its section 2 may appear; the caveats of section 3 go in Discussion and Methods.
3. `key_numbers.md`: the only source of numbers.
4. `figures/*.md` and `figures/*.png`: the legends are the ground truth for what each figure shows (use the current
   versions named in the `README.md` figure table).
5. `writing_followup_2026-11-10.md`: task and acquisition facts (its numbers are n = 30 and superseded by
   `key_numbers.md`).

Background only if needed: `/Users/gdehol/git/retsupp/notes/fitting_lessons_2026-11.md` (Methods details) and the repo
`/Users/gdehol/git/retsupp/CLAUDE.md` (experimental constants). Where any other note disagrees with the package, the
package wins. Ignore the older packages (`writing_package_2026-11/`, `writing_package_2026-11-10/`).

## Order of sections

Go through v2 in this order: Results (figure order), Methods, Introduction and Discussion, Abstract.

1. Results: one subsection per figure, headings written as claims that follow the storyline. If a heading cannot be
   stated as a claim that `claims_and_caveats.md` supports, say so instead of writing around it.
2. Methods (including the leave-one-subject-out analysis and the supplementary table of what each term adds to
   out-of-subject prediction, described neutrally).
3. Introduction (ends with the "Here, we…", roadmap and preview paragraphs).
4. Discussion (the reduction at HP as a property of the location; relation to Richter, van Moorselaar & Theeuwes,
   eLife 2025, as set out in `claims_and_caveats.md` section 4, with their design details as `⟦VERIFY: …⟧` until
   checked; dependence on the pRF model; why individual differences cannot be resolved; HP outside the bar aperture).
5. Abstract.

## House rules

- Never "singleton". Both target and distractor are singletons; say "target" and "distractor".
- No "top-down", "bottom-up" or "selection history" as names for model terms or effects. Name the contrast.
- HP = high-probability distractor location, LP = the other three; "next to HP" and "opposite HP". One name per
  construct throughout.
- Name model codes with what they are ("the joint fit with all terms (m7_full)"), never a bare code.
- Numbers only from `key_numbers.md`. Never round further, interpolate or read a number off a figure. If a number is
  missing, write `⟦TODO: …⟧` and list every token at the end of your response.
- Name the model when a number comes from a model (gain estimates, decoded intensity, out-of-subject prediction); the
  model-free observables are response times and the measured BOLD responses of Fig. 3 and the bar-crossing supplement.
- Stars in figures, exact p in legends and text. Decoding per area: one-sided Wilcoxon, BH across nine areas. Never
  call a test pre-specified or pre-registered.
- Do not mention pRF-shift analyses.
- Never call the R² mixture model "FDR-corrected".
- American spelling unless the user says otherwise.

Return the revised text section by section, each with a short list of what changed and why, and end with the list of
`⟦TODO⟧`, `⟦VERIFY⟧` and `⟦DECISION⟧` tokens.
