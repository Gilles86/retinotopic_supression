# Richter follow-up (2026-11-11)

> **Framing note for this package (2026-11-12, user):** "the difference between significant and non-significant is
> not necessarily significant". Where this report calls an effect "a session-2 effect", "present only in session 2",
> or says it "grows" over sessions, read instead: the session and block differences are not significant (A1 p ≥ .26;
> A3 p = .095), so the effects do not differ reliably between sessions; per-session values are descriptive; a former
> HP location is not lower than a never-HP one (A2 p = .54), so there is no evidence of carry-over (it cannot be
> strictly excluded for the neighbour effect). `claims_and_caveats.md` and `README.md` use this framing.


Follow-up on the comparison with Richter, van Moorselaar & Theeuwes (eLife 2025): items A1–A4, B, C and D1–D6. The
user approved the cluster jobs for A1–A3 and C; they have run (job list at the end). No existing fit was changed, and no
package file was edited: the ready-to-merge CHANGES text is the last section.

**Sample.** sub-01 and sub-02 are left out of the paper (user decision 2026-11-11). Every number below is n = 28
(sub-03–30), with the n = 30 value in brackets `[n30: …]` where key_numbers has one.

**Conventions:**
- Effects are the median of within-subject differences, with a percentile bootstrap 95% CI over subjects (10 000
  resamples, seed 0) and a Wilcoxon signed-rank p, unless the line says mean ± SEM.
- Code: `/Users/gdehol/git/retsupp/retsupp/visualize/paper/richter_followup.py`. With `--all-subjects` it gives n = 30.
- Data: `/Users/gdehol/git/retsupp/notes/data/richter_<item>_{n28,n30}_2026-11-11.tsv`.
- Trial table: `/Users/gdehol/git/retsupp/notes/data/behaviour_trials_2026-11-11.tsv.gz`, collected from the events
  exactly as Fig. 1 (`plot_rt_talk.collect_trials`). It reproduces Fig. 1: HP − LP median −26.1, mean −27.8 at n = 30.

## Contradictions with key_numbers.md

1. **"Correct trials, 21,480 in total"** (Fig. 1 section) is wrong. 21,480 is *all* trials. Correct trials with an RT:
   18,609 [n30], of which 17,292 are from sub-03–30.
2. **Decoder voxels.** key_numbers says "up to 200 voxels per area". The decoder's default (`decode_final.py
   --max-voxels 2000`) applies, and `decode_final.sh` does not override it, so the Fig. 6 legend ("at most 2000 per
   ROI") is right. The cap did not bind in the log checked (sub-10 ses-1 run-1: 37 voxels in IPS to 580 in V1).
   key_numbers should read "at most 2000 voxels per area". Per instruction, key_numbers.md is not edited here.
3. **Fig. 3 legend, dorsal n.** key_numbers is right (28 in b, c; 27 in d, e). The legend said "n = 27 in b–e". Fixed
   in the script template (`retsupp/visualize/paper/fig3_item_locations.py`) and in
   `notes/figures/paper_2026-11/fig3_item_locations_v3.md`, not in the package.
4. **"HP vs all three LP locations, early −8.6%"** comes from the earlier HP/LP deconvolution (two location classes)
   and its ANOVA. The location-type deconvolution of Fig. 3d, e gives a different HP − mean-of-three-LP:
   −11.7 ± 4.0% [n30: −10.4 ± 3.9%]. Both are correct for their deconvolution. C below uses the Fig. 3d, e one,
   because the model prediction goes through that pipeline.
5. **Between-subject dS vs HP benefit.** key_numbers has ρ = −.17 (p = .37, earlier fits). On the final fits:

   | dS measure | ρ (n = 28) | 95% CI | p | n = 30 |
   |---|---|---|---|---|
   | dS alone (DoG) | −.11 | [−.50, +.30] | .57 | ρ = −.07, p = .70 |
   | dS joint (m7) | −.01 | [−.39, +.38] | .97 | ρ = +.05, p = .79 |

   Still null, but the number changes.

## A1. Carry-over of earlier HP locations: Fig. 3d, e per session and per block (early areas, items pooled)

**Method.**
- The Fig. 3 relative deconvolution was rerun on the final location time series, separately for:
  - each session;
  - each block within a session (the session's first or second HP location);
  - each session × block.
- Output: `derivatives/paper_2026-11/event_deconv_relative_final2_split_2026-11-11_*` (job 8262343/8262500); local
  copies are in `notes/data/`, not committed.
- **Percentages:** each split's window means are % of the subject's mean response in the full-data deconvolution (the
  Fig. 3e denominator). That denominator is shared across splits. Normalising within each split instead gives the same
  conclusions but noisier extremes, because a split's own mean can be small.
- n = 28. The regular schedule only exists from sub-03, so the n = 30 bracket is identical.

**Contrasts per split (mean ± SEM, median, Wilcoxon p):**

| Split | HP − opposite | Next − opposite | HP − next |
|---|---|---|---|
| All runs (Fig. 3e) | −21.6 ± 6.8, p = .005 | −14.9 ± 6.0, p = .022 | −6.7 ± 3.6, p = .095 |
| Session 1 | −2.2 ± 13.8 (median −18.7), p = .26 | −3.5 ± 12.6 (−1.1), p = .66 | +1.4 ± 12.3, p = .75 |
| Session 2 | −34.6 ± 13.2 (−23.6), **p = .013** | −20.2 ± 11.8 (−14.8), p = .11 | −14.3 ± 11.5, p = .14 |
| Session 2 − 1 | −32.4 ± 23.0, p = .39 | −16.7 ± 21.0, p = .58 | −15.7 ± 22.7, p = .37 |
| Block 1 (HP location new in the session) | −34.4 ± 15.1 (−42.8), **p = .023** | −15.1 ± 10.0, p = .095 | −18.1 ± 7.7, **p = .032** |
| Block 2 (opposite = HP of block 1) | +3.1 ± 15.8, p = .81 | −2.0 ± 11.1, p = .99 | +5.2 ± 9.8, p = .88 |
| Session 1, block 1 (no location ever HP) | −26.9 ± 21.7 (median +2.9), p = .43 | −13.2 ± 15.7 (−3.5), p = .49 | −13.7 ± 14.9, p = .40 |

All areas: session 1 HP − opposite −7.8 ± 9.7, p = .16; session 2 −20.2 ± 8.5, p = .019.

**Plainly:**
- **Session 1, where the neighbours had never been HP:** the lower response next to HP does not survive (−3.5 ± 12.6%,
  p = .66). In the first block, where no location had ever been HP, it is −13.2 ± 15.7% (median −3.5), p = .49.
- **Session 2 carries the measured pattern,** HP − opposite included. There the two "next" locations were HP in
  session 1.
- **Block 2 shows no location effect.** Its opposite location was HP in block 1, which fits a reduction that outlasts
  the HP block. But block 2 is also where HP − opposite itself vanishes.
- **The session and block differences are not significant themselves** (p ≥ .26), and every split has only a quarter to
  half of the data.
- **Conclusion:** the data cannot separate a neighbour effect from carry-over of earlier HP locations. Without
  carry-over the neighbour effect is not demonstrable. The HP effect itself also grows from session 1 to session 2, so
  "learning over sessions" and "carry-over" are confounded in this design.

**Source:** `notes/data/richter_A1_carry_over_common_denominator_n28_2026-11-11.tsv`. The within-split normalisation
is in `richter_A1_A2_carry_over_n28_2026-11-11.tsv`. With that normalisation: session 1 next − opposite +7.9 ± 22.2,
p = .76; session 2 −21.1 ± 11.0, p = .070; session 2 HP − opposite −37.3 ± 13.9, p = .008.

## A2. Each location recoded by its HP history (early areas, % of the split's mean)

Session × block splits:
- **Current HP:** always the HP location.
- **HP in the previous block:** opposite, in block 2.
- **HP in the previous session:** both next locations, in session 2.
- **Never HP so far:** next in session 1; opposite in block 1.

| History | Response (mean ± SEM, % of the split's mean; n = 27) |
|---|---|
| Current HP | 83.2 ± 14.3 |
| HP in the previous block | 99.5 ± 7.9 |
| HP in the previous session | 113.7 ± 14.1 |
| Never HP | 101.0 ± 5.9 |

**Repeated-measures ANOVA on the LP locations, location (next, opposite) × was HP before (yes, no):**

| Effect | Early | All areas |
|---|---|---|
| Location | F(1, 26) = 1.47, p = .24 | F(1, 27) = 0.04, p = .84 |
| Was HP before | F(1, 26) = 0.39, p = .54 | F(1, 27) = 0.14, p = .71 |
| Interaction | F(1, 26) = 0.07, p = .79 | F(1, 27) = 0.86, p = .36 |

**Direct contrasts:**

| Contrast | Early | All areas |
|---|---|---|
| Next, was HP (session 2) − never (session 1) | +10.4 ± 16.9, p = .86 | +7.4 ± 14.2, p = .90 |
| Opposite, was HP (block 2) − never (block 1) | +2.6 ± 18.7, p = .82 | −13.7 ± 14.2, p = .49 |

- Only the current HP location is low. A former HP location does not respond less than one that was never HP, so this
  analysis finds no carry-over.
- Together with A1, the neighbour effect is not explained by a lasting reduction at former HP locations. But it isn't
  demonstrated independently of them either.

**Source:** `richter_A1_A2_carry_over_n28_2026-11-11.tsv`, `richter_A1_A2_split_values_n28_2026-11-11.tsv`.

## A3. The sustained gain per session

**Method.** The sustained terms were estimated alone per session, with the gain-field fits of all runs fixed:
- dS = g_HP − g_LP, added to m4_TDS;
- next − opposite, the LP split (new mechanism `adjopp`), added to m7_full.

Linear sufficient statistics come only from the session's runs (`fit_fixed_group_mechanisms --sessions`). Output:
`derivatives/af_fixed_group_linear_session{1,2}_2026-11-11` (jobs 8262545, 8262606, 8262607).

A full per-session refit of the split model was **not** run: it would be about 2 × 270 ladder fits, hundreds of
task-hours. The linear per-session estimate is the cheap version agreed. The exact (ΔSSE grid) version is not
implemented with `--sessions`.

Per subject, the median over areas:

| Term | Session 1 | Session 2 | Session 2 − 1 |
|---|---|---|---|
| dS, all areas | −.006 [−.021, +.021], p = .96 | **−.035** [−.062, −.005], 68% < 0, **p = .011** | −.022 [−.091, +.013], p = .095 |
| dS, early | −.009, p = .85 | −.038, p = .048 | |
| Next − opposite, all areas | +.005, p = .36 | −.002, p = .61 | −.007, p = .48 |
| Next − opposite, early | +.040 [−.001, +.101], p = .034 | +.032, p = .25 | |

For comparison, all runs: dS alone −.0138 (D1); split-model next − opposite −.006 (n.s.).

- **The sustained HP suppression is a session-2 effect.** It is absent in session 1 and clear in session 2. The
  difference between sessions is not significant.
- **The fitted sustained gain shows no neighbour effect in either session.** Next − opposite is ≈ 0 overall. In early
  areas in session 1 it even points the other way (+.040, p = .034, uncorrected).
- **Consistent with A1:** the HP effect measured in Fig. 3 and in the gain fields is carried by session 2.

**Source:** `notes/data/richter_A3_session_gains_n28_2026-11-11.tsv`. n = 30 brackets in `..._n30_...`: session 2 dS
−.035, p = .008; session 1 −.006, p = .87.

## A4. Behaviour by HP history (sub-03–30)

**Two confounds in the raw comparison:**
- A location that was HP before only occurs at a fixed relation to the current HP: opposite HP in a session's second
  block, or next to HP in session 2.
- It also only occurs later in the experiment, so practice speeds it up.

So each comparison below has a raw version and a difference in differences (DiD). The DiD subtracts the practice change
over the same step at a location whose history did not change.

| Comparison (RT, ms) | Distractor | Target |
|---|---|---|
| Opposite: block 2 (was HP in block 1) − block 1 (never) | −31.0 [−49.5, −0.4], p = .012 | −34.2 [−55.6, −4.7], p = .007 |
| DiD, previous-block HP: [opposite b2 − b1] − [next b2 − b1] | −9.3 [−31.1, +9.2], p = .35 | −4.3 [−31.2, +13.5], p = .48 |
| Next: session 2 (was HP in session 1) − session 1 (never) | −6.1 [−59.4, +13.3], p = .25 | −21.5 [−51.1, +12.5], p = .21 |
| DiD, previous-session HP: [next s2 − s1] − [opposite (block 1) s2 − s1] | +17.5 [−27.7, +38.7], p = .37 | +1.6 [−40.9, +43.3], p = .99 |

- **Raw comparison:** responses are faster at a former-HP opposite location.
- **After practice is taken out:** no carry-over is detectable in behaviour, for the distractor or the target, from
  the previous block or the previous session.
- **Power:** CIs are wide, because each cell is only 1/4 to 1/2 of the data.
- **Source:** `richter_A4_behaviour_history_n28_2026-11-11.tsv`; cells in `richter_A4_behaviour_history_cells_n28_2026-11-11.tsv`.

## B. Behaviour next to HP vs opposite HP

Correct trials. "Next" means the two adjacent diagonal locations, pooled; "opposite" means the diagonal opposite HP.
The target cells use distractor-present trials only.

**Cell means (n = 28):**

| Cell | Distractor RT | Distractor accuracy | Target RT | Target accuracy |
|---|---|---|---|---|
| HP | 874 ms | 86.6% | 905 ms | 83.5% |
| Next to HP | 912 ms | 82.6% | 881 ms | 85.4% |
| Opposite HP | 880 ms | 85.7% | 870 ms | 87.7% |
| Absent | 861 ms | 89.5% | — | — |

**Contrasts, median [95% CI], p, with n = 30 in brackets:**

| Contrast | Distractor RT (ms) | Distractor accuracy (%) | Target RT (ms) | Target accuracy (%) |
|---|---|---|---|---|
| Next − opposite | **+37.8** [+24.2, +45.5], p < .001 [n30 +34.6] | −1.1 [−6.3, +0.3], p = .018 | **+10.4** [+0.2, +18.4], p = .005 [n30 +12.5] | −1.7 [−4.3, −0.2], p = .006 |
| HP − next | **−37.8** [−41.9, −30.6], p < .001 [n30 −37.8] | +3.1 [+1.5, +5.6], p < .001 | **+20.6** [+15.7, +29.7], p < .001 [n30 +21.0] | −1.2 [−3.0, +0.2], p = .025 |
| HP − opposite | −3.8 [−15.9, +3.7], p = .34 [n30 −6.1, p = .15] | +0.6 [−0.3, +1.8], p = .21 | **+30.2** [+22.8, +48.3], p < .001 [n30 +34.0] | −4.9 [−5.6, −3.1], p < .001 |
| HP − absent (distractor) | +14.6 [+0.3, +21.7], p = .004 [n30 +15.8] | −2.9 [−4.0, −1.6], p < .001 | — | — |

- **The distractor benefit isn't confined to HP.** A distractor next to HP is the slowest location: 38 ms slower than
  at HP and 38 ms slower than opposite. A distractor opposite HP is almost as fast as at HP (−3.8 ms, n.s.). So the
  "HP benefit" against all LP locations comes from the neighbours. Opposite HP was itself HP in the session's
  previous block (A4); the raw A4 comparison shows faster responses there, the DiD does not.
- **There is a target cost.** A target at HP is slower than next to HP (+21 ms) and than opposite (+30 ms), and less
  accurate (−4.9% vs opposite). The target cost is graded: HP > next > opposite.
- **Compared with Richter et al.** They report the benefit only at HP itself and no target cost. Our data differ on
  both counts. There is a neighbour cost for the distractor (next slower than opposite) and a graded target cost.
- **Source:** `richter_B_behaviour_contrasts_n28_2026-11-11.tsv` and `richter_B_behaviour_cells_n28_2026-11-11.tsv`.

## C. How much of the measured HP reduction the model reproduces

Early and all areas, % of the subject's mean response, mean ± SEM, Wilcoxon p. The m7_full noise-free prediction went
through exactly the Fig. 3 pipeline (`notes/data/kappa_neighbour_2026-11/event_deconv_m7`, from job 8236933/34).

| Contrast | Measured, early | m7 predicted, early | Measured, all areas | m7 predicted, all areas |
|---|---|---|---|---|
| HP − LP (mean of three) | −11.7 ± 4.0, p = .009 [n30 −10.4] | −5.4 ± 3.7, p = .036 [n30 −5.1] | −9.7 ± 3.1, p = .006 [n30 −8.7] | −4.1 ± 3.0, p = .015 [n30 −3.9] |
| HP − opposite | −21.6 ± 6.8, p = .005 [n30 −19.0] | −6.3 ± 3.8, p = .023 [n30 −5.7] | −15.5 ± 5.3, p = .013 [n30 −13.2] | −4.7 ± 3.0, p = .010 [n30 −4.2] |
| HP − next | −6.7 ± 3.6, p = .095 [n30 −6.1] | −4.9 ± 3.7, p = .045 [n30 −4.9] | −6.9 ± 2.7, p = .018 [n30 −6.4] | −3.8 ± 3.0, p = .020 [n30 −3.8] |
| Next − opposite | −14.9 ± 6.0, p = .022 [n30 −13.0] | −1.3 ± 0.8, p = .16 [n30 −0.9] | −8.6 ± 4.6, p = .060 [n30 −6.8] | −0.9 ± 0.5, p = .17 [n30 −0.4] |

- **The joint fit reproduces about half of the HP − LP reduction** (early: −5.4 of −11.7; all areas: −4.1 of −9.7),
  about 30% of HP − opposite, most of HP − next, and none of next − opposite.
- **Which term carries it:** see C(i), (ii) below.
- **Source:** `richter_C_D1_location_contrasts_n28_2026-11-11.tsv` (rows source = data and m7 prediction).


### C(i), (ii): which term carries the predicted reduction (job 8262344/8262544)

**Method.** Noise-free predictions per variant went through the same Fig. 3 pipeline. Output:
`derivatives/paper_2026-11/event_deconv_terms_pred_2026-11-11_*`. To remove a term, its two gains were set to their
mean:
- sustained: (g_HP + 3·g_LP)/4;
- transient: weighted by each subject's pulses at HP.

Early areas n = 27: sub-30's predicted mean response is not positive, as in the data. Mean ± SEM, % of the mean
response.

| Variant | HP − LP, early | HP − opposite, early | Next − opposite, early | HP − LP, all areas | HP − opposite, all areas |
|---|---|---|---|---|---|
| No HP − LP difference (reference) | −2.6 ± 2.1, p = .32 | −2.0 ± 1.3, p = .11 | +0.8 ± 1.3 | −1.8 ± 2.1, p = .54 | −1.2 ± 1.3, p = .97 |
| Sustained only (S) | −7.3 ± 4.8, p = .025 | −6.8 ± 4.5, p = .025 | +0.7 ± 1.8 | −5.9 ± 4.0, p = .016 | −5.3 ± 3.7, p = .014 |
| Sustained + dD (= m7 as fitted) | −5.1 ± 4.1, p = .11 | −4.4 ± 4.1, p = .11 | +1.0 ± 1.7 | −3.7 ± 3.4, p = .043 | −3.1 ± 3.3, p = .040 |
| Sustained + dD + dT (m8_full_dT) | −4.1 ± 3.8, p = .19 | −2.8 ± 4.0, p = .41 | +1.9 ± 2.2 | −2.9 ± 3.1, p = .043 | −1.8 ± 3.2, p = .12 |

**What each term adds:**

| Term | HP − LP, early | HP − LP, all areas |
|---|---|---|
| Sustained HP − LP (S − reference) | −4.7 ± 4.6 (median −12.0), p = .065 | −4.0 ± 3.7 (median −8.4), p = .043 |
| Distractor transient dD (SD − S) | +2.2 ± 2.1, p = .073 | +2.2 ± 1.3, p = .060 |
| Target transient dT (SDT − SDT0) | +0.8 ± 1.9, p = .47 | +0.3 ± 1.5, p = .71 |

- **The sustained HP − LP term carries the model's predicted reduction at HP.**
- **The transient terms do not add to it.** dD pulls it slightly back (the fitted dD makes the HP response a little
  larger on distractor trials than the sustained field alone would), and dT adds nothing.
- **No variant predicts the neighbour effect** (next − opposite +0.7 to +1.9%; measured −14.9%).
- **The model reproduces about half of the measured HP − LP** (−5 to −7% predicted vs −11.7% measured, early).
- **Consistency check:** the m7 rerun here (SD) agrees with the earlier m7 prediction (−5.4 ± 3.7 at n = 28 with
  sub-30 included).

**Source:** `notes/data/richter_C_term_contrasts_n28_2026-11-11.tsv` (n = 30 in `..._n30_...`).

## D1. Missing intervals

| Quantity | n = 28 | n = 30 (key_numbers) | Source |
|---|---|---|---|
| Held-out R², DN − DoG (stage 3, fold mean, median over nine areas) | +.0048 [+.0023, +.0094], 25/28, p = 5e-6 | +.0063 [+.0025, +.0112] | `prf_jointhrf_summary_final2_2026-11.tsv` |
| HP − opposite, early | −21.6 ± 6.8% (CI of mean [−34.9, −8.9]) | −19.0 ± 6.6% [−32.0, −6.6] | `event_deconv_relative_final2_2026-11` |
| HP − opposite, all areas | −15.5 ± 5.3% [−25.4, −5.1] | −13.2 ± 5.2% [−23.2, −2.9] | same |
| dS alone, DoG | −.0138 [−.0179, −.0059], p = 9e-5 | −.0117 [−.0166, −.0059] | `conditional_estimates_dog_final2_2026-11.tsv` |
| dS alone, Gaussian | −.0076 [−.0264, −.0010], p = 4e-4 | −.0053 [−.0245, −.0011] | `conditional_exact_final2_allfam_2026-11.tsv` |
| dS alone, DN | −.0281 [−.0411, −.0072], p = .005 | −.0191 [−.0396, −.0072] | same |
| Split model, next − opposite | −.0059 [−.0324, +.0195], p = .73 | +.0001 [−.0291, +.0195] | `gain_field_parameters_final2_2026-11.tsv` (m8_full_adjopp) |
| Split model, HP − next | −.0163 [−.0689, +.0054], p = .070 | −.0124 [−.0462, +.0058] | same |
| Split model, HP − opposite | −.0331 [−.0518, −.0194], p = .005 | −.0310 [−.0488, −.0137] | same |
| ρ(dS, HP benefit) | see contradiction 5 | | |

- **Split-model parametrisation.** g_LP is the mean of the three LP gains and d = g_adj_minus_opp
  (`collect_lp_split.py`). So next = g_LP + d/3 and opposite = g_LP − 2d/3. With that, the key_numbers n30 values
  reproduce exactly.

## D2. Without the two pilots

- **What the old numbers were:** −26.6 ms and +27.3 ms are **means** of the within-subject differences (n = 28).
- **The medians (n = 28), which are now the primary numbers:**

  | Effect | Median [95% CI] | Mean | p |
  |---|---|---|---|
  | HP benefit (distractor at HP − LP) | −25.0 ms [−32.4, −20.0] | −26.6 | 7.5e-9 |
  | Target cost (target at HP − LP, distractor trials) | +24.3 ms [+18.5, +36.4] | +27.3 | 1.5e-6 |

- **Source:** `richter_D2_without_pilots_n28_2026-11-11.tsv`.

## D3. Decoder voxel count

See contradiction 2. The Fig. 6 legend is right: at most 2000. key_numbers is wrong.

## D4. Fig. 3 dorsal n

See contradiction 3. key_numbers is right; the legend is fixed in `notes/figures/paper_2026-11` only.

## D5. Target vs distractor at the item locations (window 3.2–6.4 s, % signal; BH over the three pathways)

**Target − distractor, both at LP locations:**

| Pathway | Median [95% CI] | p | q | n = 30 |
|---|---|---|---|---|
| Early | +.038 [+.004, +.105] | .015 | .015 | +.038, p = .015 |
| Dorsal (n = 26) | +.126 [+.049, +.184] | 1e-4 | 4e-4 | n = 28: +.114 |
| Occipitotemporal | +.097 [+.055, +.146] | 4e-4 | 6e-4 | +.095 |

- **Ordering:** target > distractor in every pathway, early included. With target and distractor each > neutral
  (key_numbers), that gives target > distractor > neutral in early visual cortex, as in Richter et al.
- **Target averaged over HP and LP, against the distractor at LP:** early +.020, p = .25 (n.s.), because the target at
  HP is lower; dorsal +.074, q = .047; occipitotemporal +.072, q = .004.
- **Source:** `richter_D5_target_vs_distractor_n28_2026-11-11.tsv`.

## D6. Location effect per early area (uncorrected, exploratory)

| Area | HP − opposite | Next − opposite | HP − next |
|---|---|---|---|
| V1 | −29.8 [−61.2, −12.9], p = .002 | −14.2 [−47.5, +10.3], p = .090 | −12.7 [−22.5, +7.7], p = .15 |
| V2 | −20.0 [−41.9, −3.9], p = .010 | −10.1 [−25.6, +3.7], p = .14 | −9.8 [−25.0, −0.6], p = .17 |
| V3 | −11.4 [−18.5, +0.2], p = .33 | −5.9 [−14.0, +5.1], p = .21 | −3.3 [−20.8, +21.6], p = .88 |

- **Unlike Richter et al.:** the HP − opposite reduction is clearest in V1, then V2, and weakest in V3. Richter et al.
  found the reduction in V2 and only weakly in V1.
- **Noise:** V1 is noisy. Its mean is −52.8 ± 16.4%, because single-area percentages have small denominators.
- **n = 30:** V1 p = .003, V2 p = .015, V3 p = .44.
- **Source:** `richter_D6_per_area_n28_2026-11-11.tsv`.

## Jobs run (account hare.econ.uzh, approved by the user)

| Job | What | Tasks | Time per task | Output |
|---|---|---|---|---|
| 8262343 (canary), 8262500 | A1, A2: split deconvolutions | sub-03–30 | ~1 min | `derivatives/paper_2026-11/event_deconv_relative_final2_split_2026-11-11_*` |
| 8262344 (canary), 8262544 | C: per-term predictions and deconvolution | sub-01–30 | ~4 min | `derivatives/paper_2026-11/location_timeseries_terms_pred_2026-11-11_*`, `event_deconv_terms_pred_2026-11-11_*` |
| 8262545 (canary), 8262606, 8262607 | A3: per-session dS and adjopp, linear | sub-01–30 per session | ~3.5 min | `derivatives/af_fixed_group_linear_session{1,2}_2026-11-11` |
| 8262315, 8262316, 8262311, 8262317 | Failed or cancelled first attempts | — | — | 8262311 was submitted before the cluster clone had pulled the new code and was cancelled; the others lacked `~/pylibs/nideconv` on the path or asked for `--exact` with `--sessions` |

**Code:**
- `retsupp/visualize/paper/richter_followup.py`: all summaries.
- `retsupp/visualize/paper/deconvolve_location_responses.py`: `--split`.
- `retsupp/visualize/paper/predict_location_timeseries_terms.py` and its slurm script.
- `retsupp/visualize/paper/slurm_jobs/deconv_history_splits.sh`.
- `retsupp/modeling/fit_fixed_group_mechanisms.py`: `adjopp`, only when named, so the default list is unchanged.

## CHANGES (ready to merge into the n = 28 writing package; dated 2026-11-11)

**Behaviour (Fig. 1 section).**
- Correct trials: 17,292 (n = 28). The earlier "21,480 correct trials" counted all trials.
- Pilots: the old −26.6 and +27.3 ms were means. As medians: HP benefit −25.0 ms [−32.4, −20.0], target cost
  +24.3 ms [+18.5, +36.4].
- New: behaviour by location relative to HP (B).
  - **Distractor:** fastest at HP (874 ms), nearly as fast opposite HP (880 ms; HP − opposite −3.8 ms
    [−15.9, +3.7], p = .34), slowest next to HP (912 ms; next − opposite +37.8 ms [+24.2, +45.5], p < .001).
  - **Target:** cost graded HP > next > opposite (HP − next +20.6 ms, HP − opposite +30.2 ms, both p < .001; accuracy
    at HP −4.9% vs opposite).
  - **Relation to Richter et al.:** they report the benefit only at HP itself and no target cost.
- Carry-over in behaviour (A4): after the practice change is subtracted, no effect of a location having been HP
  before (previous block: distractor −9.3 ms, p = .35; previous session: +17.5 ms, p = .37).

**Item locations (Fig. 3 section).**
- HP − opposite (Fig. 3e, early): −21.6 ± 6.8% (95% CI of the mean [−34.9, −8.9]); all areas −15.5 ± 5.3%
  [−25.4, −5.1].
- HP − mean of the three LP locations in the Fig. 3d, e deconvolution: early −11.7 ± 4.0%, all areas −9.7 ± 3.1%. The
  −8.6% / −7.3% come from the earlier HP/LP deconvolution; say which one is quoted.
- Dorsal n: 26 in b, c (n = 28 sample); key_numbers is right, and the legend template is fixed.
- Target − distractor (both at LP): early +.038 [+.004, +.105], q = .015; dorsal +.126, q < .001; occipitotemporal
  +.097, q < .001. So target > distractor > neutral in early visual cortex, as in Richter et al.
- Per early area (exploratory, uncorrected), HP − opposite: V1 −29.8% (p = .002), V2 −20.0% (p = .010), V3 −11.4%
  (p = .33). Next − opposite is n.s. in each area. Unlike Richter et al., the effect is not weaker in V1 than in V2.
- **Carry-over (A1, A2):**
  - The neighbour effect is not demonstrable without carry-over. Session 1 next − opposite −3.5 ± 12.6%, p = .66;
    session 2 −20.2 ± 11.8%, p = .11.
  - The measured HP effect is also mostly a session-2 effect: session 1 HP − opposite −2.2 ± 13.8%, p = .26; session 2
    −34.6 ± 13.2%, p = .013.
  - Session differences are not significant (p ≥ .37).
  - Locations recoded by HP history show no reduction at former HP locations (was HP before: F(1, 26) = 0.39, p = .54).
  - Discussion wording: "The lower response next to HP could not be separated from carry-over of earlier HP locations
    (it was present only in session 2, where the neighbouring locations had been HP in session 1), and the HP
    reduction itself was larger in the second session."

**Gain fields (Fig. 5 section).**
- dS estimated alone (DoG): −.0138 [−.0179, −.0059], p = 9 × 10⁻⁵ (n = 28). Gaussian −.0076 [−.0264, −.0010];
  DN −.0281 [−.0411, −.0072].
- Split model, as medians over areas with CIs:

  | Contrast | Median [95% CI] | p |
  |---|---|---|
  | Next − opposite | −.006 [−.032, +.020] | .73 |
  | HP − next | −.016 [−.069, +.005] | .070 |
  | HP − opposite | −.033 [−.052, −.019] | .005 |

- Per session (A3; estimated alone, all-run fits fixed): dS session 1 −.006, p = .96; session 2 −.035 [−.062, −.005],
  p = .011; difference p = .095. The sustained HP suppression appears in the second session. Next − opposite ≈ 0 in
  both sessions.
- Between subjects: ρ(dS alone, HP benefit) = −.11 [−.50, +.30], p = .57; ρ(joint dS) = −.01, p = .97. Replaces
  ρ = −.17 from the earlier fits.

**Model vs measurement (C).**
- The joint fit reproduces about half of the measured HP − LP reduction (early −5.4 vs −11.7%; all areas −4.1 vs
  −9.7%), about 30% of HP − opposite, and none of next − opposite (−1.3 vs −14.9%).
- The sustained HP − LP term carries it (sustained alone: −7.3%). The distractor transient term slightly reduces it
  (+2.2%). The target transient term adds nothing (+0.8%).

**pRF models (Fig. 2 section).** DN − DoG held-out R², n = 28: +.0048 [+.0023, +.0094], 25/28, p = 5 × 10⁻⁶.

**Methods (Fig. 6).** Decoder voxels: at most 2000 per area by cross-run reliability, not 200. The Fig. 6 legend is
right.
