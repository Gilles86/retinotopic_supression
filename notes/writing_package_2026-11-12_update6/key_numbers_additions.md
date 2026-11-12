# key_numbers additions: gain or baseline shift at the HP location (2026-11-12, n = 28)

Append to key_numbers.md (update 5) as a new section. All on the current final fits (pRFs and HRF fitted jointly,
gain fields fitted with the pRFs frozen); may be rerun after the refit of the pRFs together with the gain fields.
Per participant the median over areas; median over participants, bootstrapped 95% CI (10,000), Wilcoxon signed-rank.
Gains in %; trial-locked amplitudes in % signal change at full pRF overlap; held-out ΔR² on run 3, × 10⁻⁴.
Full report: `baseline_vs_gain_2026-11-12.md` (this update).

## Gain or baseline shift at HP (supplementary figure figS_baseline_vs_gain_n28)

- Sustained additive offset at HP: not identifiable (each run is % signal change of its own mean, HP constant within a
  run; the additive regressor keeps 0.0 of its sum of squares after run-mean removal in all 239 participant × area
  fits). Largest |run mean| of the cleaned BOLD 3.3 × 10⁻⁵ % vs median within-run SD 1.23%
  (`retsupp/preprocess/clean.py`; Fig. 3 location series).
- Bar-response slope at HP vs the LP locations (b): all areas −2.1% [−4.6, −0.7], 20/28 below 0, p = .012; HP vs next
  −3.5% [−5.2, −0.9], 21/28, p = .0011; next vs opposite −0.5% [−3.0, +3.0], p = .78; early −3.4% [−4.9, −0.8],
  p = .10; dorsal −2.2% [−5.8, +3.9], p = .55; occipitotemporal −2.5% [−6.0, −0.7], p = .063
  (`notes/data/baseline_vs_gain_step1_n28_2026-11-12.tsv`).
- Trial-locked additive response at HP (c), on the joint fit (`notes/data/baseline_vs_gain_terms_n28_2026-11-12.tsv`):
  - locked to search onsets: HP-specific amplitude +1.5% signal [+0.0, +2.9], 19/28 above 0, p = .010 (positive, not a
    suppression); held-out ΔR² +0.13 [−0.44, +0.70], better in 16/28, p = .41;
  - locked to fixation onsets: HP-specific amplitude −0.7% [−1.0, +0.9], p = .85; held-out ΔR² +1.33 [+0.64, +2.59],
    24/28, p = 9 × 10⁻⁵, carried by the trial response common to all four locations (+6.8% [+1.6, +17.2], 22/28,
    p = 4 × 10⁻⁵), not by the HP-specific part.
  - with the transient HP − LP terms: held-out worse (−2.2 [−3.0, −0.3], better in 4/28, p = 9 × 10⁻⁵); the transient
    distractor HP − LP gain moves by −1.8 points [−3.7, −1.3] (p = 9 × 10⁻⁴) and dS by −0.78 points [−1.05, −0.50]
    (p = 4 × 10⁻⁵) when the trial-locked term is added (`..._change_n28_2026-11-12.tsv`): the terms are partly
    collinear, so these joint estimates are not interpreted.
- Recovery (d; 10 simulated datasets × 28 participants, both terms fitted; detected = group p < .05, negative;
  `..._confusion_n28_2026-11-12.tsv`): simulated gain (−1.4%) → gain 0.6, neither 0.4, trial baseline 0.0; simulated
  trial-locked baseline (matched in signal power) → trial baseline 1.0, gain 0.0; simulated sustained offset (removed
  with the run mean) → neither 1.0.
