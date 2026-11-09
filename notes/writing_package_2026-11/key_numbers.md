# Key numbers (2026-11-06)

n = 30 throughout. Tests are two-sided Wilcoxon signed-rank tests across subjects unless stated; BH =
Benjamini-Hochberg. Gain numbers: DoG pRFs, current fits, **provisional** until the Phase C rerun.
Source files are under `/Users/gdehol/git/retsupp/notes/`.

## Behaviour (final; `data/behaviour_rt_2026-11.tsv`, Fig. 1)

| Effect | Value | Subjects | p |
|---|---|---|---|
| Distractor at LP minus at HP (HP benefit) | mean 27.8 ms (median 26.1) | 30/30 faster at HP | < .001 |
| Distractor at LP minus absent (capture cost) | mean 43.0 ms | | |
| Target at HP minus at LP (cost of a target at HP) | mean 28.4 ms slower at HP | 26/30 | < .001 |
| Target repeats its location (repeat − switch) | median −38 ms | 28/30 | < .001 |
| Distractor repeats its location (repeat − switch, within HP and LP trials) | median −15 ms | 21/30 | .003 |

## pRF models (`data/prf_cv_roi_summary.tsv`; held-out R², reliable voxels, median over subjects)

| Gaussian | DoG | DN | DoG + fitted HRF | DN + fitted HRF |
|---|---|---|---|---|
| .201 | .210 | .216 | .238 | .245 |

DN > DoG in 30/30 subjects (by .005). Per-ROI HRF test pending.

## Gain-field ladder, leave-one-subject-out (`data/loso_per_step_2026-11.tsv`, Table 1; per ROI, figS_loso_per_roi)

Share of subjects whose held-out prediction improves when the term is added with the effect size from
the other subjects; median over ROIs.

| Term | DoG | p | DN | p |
|---|---|---|---|---|
| Items | 100% | < .001 | 100% | < .001 |
| Target gain (T) | 90% | < .001 | 97% | < .001 |
| Distractor gain, tied (D) | 50% | .81 | 53% | .84 |
| Sustained gain, tied (S) | 90% | < .001 | 90% | < .001 |
| dS | 77% | .010 | 63% | .23 |
| dD | 63% | .09 | 67% | .08 |
| D given T | pending | | pending | |

## Gain parameters (pooled = each subject's median over the nine ROIs)

| Term | Estimate | Median [95% CI] | % < 0 | p |
|---|---|---|---|---|
| dS | conditional | −.009 [−.014, −.006] | 80% | < .001 |
| dS | joint (m7), DoG | −.018 [−.032, −.004] | 73% | .006 |
| dS | joint (m7), DN | −.027 [−.038, −.017] | 80% | .004 |
| dD | conditional | −.031 [−.038, −.013] | 83% | < .001 |
| dD | joint (m7), DoG | −.018 [−.059, +.024] | 57% | .44 |
| T | joint, DoG | +.25 [+.21, +.32] | 3% | < .001 |
| D (averaged over locations) | joint, DoG | +.11 [+.04, +.12] | 20% | < .001 |

Per ROI (joint, DoG medians; * BH q < .05 across ROIs; `figures/paper_2026-11/table_parameters_2026-11.md`):

- T: V1 +.01, V2 +.12*, V3 +.30*, V3AB +.30*, IPS +.39*, hV4 +.34*, VO +.35*, LO +.26*, TO +.25*
- D: V1 +.03, V2 +.05, V3 +.03, V3AB +.14*, IPS +.18*, hV4 +.21*, VO +.10*, LO +.09, TO +.09
- dS (joint): V1 −.04, V2 −.05, V3 −.05, others −.00 to −.02 (none survives BH alone; the pooled test is the claim)

## Along the hierarchy (`data/hierarchy_regression_2026-11.tsv`, supplementary figure `figS_gain_vs_prf_size`)

Per subject, OLS slope of the parameter on log₂ pRF size over the nine ROIs (pRF size: group median at
1-2.5° eccentricity, 0.57° in V1 to 1.63° in IPS); Wilcoxon on the slopes, BH over 8 parameters.

| Parameter | Change per doubling of pRF size | Subjects in that direction | q |
|---|---|---|---|
| Target gain | +.24 | 90% | < .001 |
| Distractor gain | +.06 | 77% | .010 |
| dS, conditional | flat | | .45 |
| dS, joint | +.03 (less negative higher up) | 73% | .010 |
| Field sizes | unstable: many fits at the σ bounds (up to 73%); supplement only, do not interpret slopes | | |

## Gain and behaviour (Fig. 7; trial-wise analyses on the earlier fits, Oct 2026)

- Target gain, repeat − switch: negative in V3AB, IPS, hV4, VO (BH q < .05).
- Bayesian hierarchical model of log RT per ROI (bambi; random intercepts and gain slopes per subject;
  no divergences, R̂ ≤ 1.01): distractor gain on HP trials goes with faster responses in V1 and V2
  (posterior P(< 0) > .99); target gain with slightly slower responses in V3AB, hV4, LO; global gain
  with faster responses in V2/V3 and slower in IPS (and FEF). Target repeat −0.21 SD of log RT,
  distractor repeat −0.06 SD.
- Between subjects (text only): dS vs behavioural HP benefit ρ = −.17, p = .37 (expected direction);
  no ROI survives correction. Run-level: null. Distractor gain, repeat vs switch: no effect (IPS
  p = .01, q = .11).

## Sensitivity (text)

- Within-subject CV fails dS, and so does a simulation that plants our effect size: break-even near
  dS = −.05, our estimate about −.016 (`figS_cv_power.pdf`).
- Per-subject dS noise SD about .03 against an effect of .01-.02.
