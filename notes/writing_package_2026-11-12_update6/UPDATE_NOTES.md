# Update 6 (2026-11-12): gain or baseline shift at the HP location

**Applies on top of update 5.** Nothing in update 5 is replaced; this adds one supplementary figure, a block of numbers
and suggested wording.

**These results are on the current final fits** (pRFs and HRF fitted jointly; gain fields fitted with the pRFs frozen).
A refit of the pRFs together with the gain fields is under way; these tests may be rerun on it, and the numbers may
change slightly.

## What's in it

| File | What |
|---|---|
| `figures/figS_baseline_vs_gain_n28.{pdf,png,md}` | New supplementary figure: a schematic; b bar-response slope at HP; c held-out ΔR² of a trial-locked additive response at HP; d recovery |
| `key_numbers_additions.md` | The new numbers, to append to key_numbers.md as a section (sources given) |
| `baseline_vs_gain_2026-11-12.md` | The full report: identifiability, methods, results table, caveats, wording |

## What to do in the manuscript

1. Methods (one sentence, where the cleaning is described): "Because each run's time series was expressed as percent
   signal change of its own mean and the HP location was constant within a run, a sustained additive offset at the HP
   location is not identifiable in these data; we therefore tested the two forms of a baseline shift that remain
   testable: a change in the slope of the response to the mapping bar, and a trial-locked additive response."
2. Results 4, after "We also tested variants...": the paragraph under "Suggested wording" in the report.
3. Discussion, the paragraph on Richter et al.'s proactive suppression: the sentence(s) under "Suggested wording".
4. Supplementary numbering: add figS_baseline_vs_gain_n28 at the end of the supplementary figures, or after the
   model-variant figures (DoG centre/surround tilt, DN gain locus) if the text cites it there.
5. Do not claim that the effect is "not a baseline shift" in general: a block-long baseline shift is not measurable here.
   The supported claims are (i) the response to the bar is reduced in slope at HP (a gain signature) and (ii) there is no
   trial-locked additive baseline at HP, which the simulations show we could have detected.
