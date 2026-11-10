# Prompt for the writing agent (2026-11-10)

You are drafting the retsupp paper (7T fMRI, visual search with a high-probability distractor location, pRF mapping
during search, gain-field models) together with Gilles de Hollander. Write in his voice: load the
**scientific-writing** skill first and follow it (declarative Results headings, prediction before result, colon with
capital letter, intervals with every effect size, no bullet lists in Results or Discussion).

## The story you are telling

One easy-to-follow storyline, seven figures, each answering one question that leads to the next:

1. People learn to suppress the HP location (faster when the distractor is there, in all 30 participants; slower when
   the target is there) and are primed by the previous trial's locations (Fig. 1).
2. pRFs mapped during search let us see the display through each voxel's receptive field (Fig. 2).
3. Without any model, voxels at the item locations respond more to the target and the distractor than to a neutral
   item, and in early visual cortex least when the item is at the HP location, for every item type (Fig. 3).
4. A gain-field model describes how attention could reshape the drive each pRF receives: transient gain at the target
   and the distractor, sustained gain at the four item locations that can differ at HP (Fig. 4).
5. The fitted gain fields: target gain rising along the hierarchy, weaker distractor gain, and lower sustained gain at
   HP, consistent across participants and areas; the transient gain at HP is slightly lower for whatever item appears
   there, so the reduction belongs to the location (Fig. 5).
6. Decoding the display from pRFs without any gain terms shows the same at the HP location: the bar is read out weaker
   in the quadrant facing HP (Fig. 6).
7. Trial by trial, the gain fields move with history and behaviour: the target gain drops when the target repeats its
   location, and the distractor and global gains are lower on slower trials (Fig. 7).

The argument rests on what the gain fields look like (parameter estimates and their consistency across subjects and
areas), together with the model-free responses (Fig. 3) and the gain-free decoding (Fig. 6). Do not lead with, or
hinge any main-text sentence on, held-out model comparison (leave-one-subject-out): that goes to Methods and a
supplementary table, described neutrally (user decision, 2026-11-10). Caveats (dependence on the pRF model,
out-of-subject prediction, the SE weighting in Fig. 7) belong in the Discussion and Methods, not in the storyline.

## Read, in this order

All in `/Users/gdehol/git/retsupp/notes/writing_package_2026-11-10/`:

1. `README.md`: storyline, decisions, words, outline, figure list.
2. `claims_and_caveats.md`: **the guardrail.** Every claim you write must be in its section 1, phrased no more
   strongly than there; nothing from its section 2 may appear; the caveats of section 3 go in Discussion and Methods.
3. `key_numbers.md`: the only source of numbers.
4. `figures/*.md` and `figures/*.png`: the legends are the ground truth for what each figure shows.

Background only if needed: `/Users/gdehol/git/retsupp/notes/fitting_lessons_2026-11.md` (Methods details) and the repo
`/Users/gdehol/git/retsupp/CLAUDE.md` (experimental constants). Where any other note disagrees with the package, the
package wins. Ignore the older package `/Users/gdehol/git/retsupp/notes/writing_package_2026-11/`.

## Order of sections

Draft the Results first, in figure order, then Methods, then Introduction and Discussion, then the Abstract.

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

Return the text section by section, each with a short note on what it relies on, and end with the list of `⟦TODO⟧` and
`⟦VERIFY⟧` tokens.
