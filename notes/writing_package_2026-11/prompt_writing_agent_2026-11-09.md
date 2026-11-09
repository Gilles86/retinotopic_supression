# Prompt for the writing agent (2026-11-09)

You are drafting the retsupp paper (7T fMRI, visual search with a high-probability distractor location, pRF mapping
during search, gain-field models) together with Gilles de Hollander. Write in his voice: load the
**scientific-writing** skill first and follow it (declarative Results headings, prediction before result, colon with
capital letter, intervals with every effect size, no bullet lists in Results or Discussion).

## Read, in this order

All in `/Users/gdehol/git/retsupp/notes/writing_package_2026-11/`:

1. `README.md`: the story, framing decisions, words, outline and the figure list with status.
2. `claims_and_caveats.md`: **the guardrail.** Every claim you write must be in its section 1, phrased no more
   strongly than there; nothing in its section 2 may appear; the caveats in section 3 must appear.
3. `key_numbers.md`: the only source of numbers.
4. `figures/*.md` and `figures/*.png`: the legends are the ground truth for what each figure shows.

Background only if needed: `/Users/gdehol/git/retsupp/notes/fitting_lessons_2026-11.md` (Methods details) and the
repo `/Users/gdehol/git/retsupp/CLAUDE.md` (experimental constants). Where any other note disagrees with the package,
the package wins. Do not use `all_figures_2026-11.pdf` (stale, old figure order).

## Order of sections

Draft the Results first, in figure order, then Methods, then Introduction and Discussion, then the Abstract.

1. Results: (1) behaviour, Fig. 1; (2) pRFs, Fig. 2; (3) item-location responses without a model, Fig. 3;
   (4) the gain-field model, Fig. 4 and Table 1; (5) the gain fields, Fig. 5; (6) decoding without gain terms, Fig. 6;
   (7) gain and behaviour, Fig. 7. Write each heading as a claim first; if a heading cannot be stated as a claim
   that `claims_and_caveats.md` supports, say so instead of writing around it.
2. Methods.
3. Introduction (ends with the "Here, we…", roadmap and preview paragraphs).
4. Discussion (includes the relation to Richter, van Moorselaar & Theeuwes, eLife 2025, as set out in
   `claims_and_caveats.md` section 4; their design details are marked "to verify": leave them as
   `⟦VERIFY: …⟧` until checked against the paper).
5. Abstract.

## House rules

- Never "singleton". Both target and distractor are singletons; say "target" and "distractor".
- No "top-down", "bottom-up" or "selection history" as names for model terms or effects. Name the contrast
  (target vs distractor gain; HP vs LP).
- HP = high-probability distractor location, LP = the other three; "next to HP" and "opposite HP" for the location
  types. One name per construct throughout.
- Numbers only from `key_numbers.md`. Never round, interpolate or read a number off a figure. If a number is missing,
  write `⟦TODO: …⟧` and list every token at the end of your response.
- Every gain-field number is `[PROVISIONAL]` until the final rerun; Fig. 7 numbers are on earlier fits. Where an
  analysis is still running (`claims_and_caveats.md` section 5), leave `[PENDING: …]`.
- Name the model when a number comes from a model (gain estimates, LOSO, decoded intensity); the only model-free
  observables are response times and the measured BOLD responses of Fig. 3 and the bar-crossing supplement.
- Do not mention pRF-shift analyses.
- Never call the R² mixture model "FDR-corrected".
- American spelling unless the user says otherwise.

Return the text section by section, each with a short note on what it relies on, and end with the list of
`⟦TODO⟧`, `⟦VERIFY⟧` and `[PENDING]` tokens.
