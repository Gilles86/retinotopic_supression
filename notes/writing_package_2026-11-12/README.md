# retsupp paper: writing package (2026-11-12, final fits, n = 28; revision pass on the current manuscript)

This package replaces the 2026-11-11 package (`/Users/gdehol/git/retsupp/notes/writing_package_2026-11-10/`, zip
`writing_package_2026-11-11.zip`). **The main change: the two pilot participants (sub-01, sub-02) are left out of every
analysis (n = 28)**, on paradigm grounds; see CHANGES 2026-11-12 below. Hand this folder to the writing agent; it is
self-contained. Read in this order:

1. this file (storyline, decisions, words, outline, figure list with status); start with `prompt_writing_agent_2026-11-12.md`;
2. `claims_and_caveats.md`: **the guardrail** (what the data support, what they do not, caveats);
3. `key_numbers.md`: the only place to take numbers from, each with its source file (all at n = 28);
4. `figures/`: one PDF + PNG + legend `.md` per figure (current versions, rendered at n = 28, from
   `/Users/gdehol/git/retsupp/notes/figures/paper_2026-11/`); `all_figures_2026-11-12.pdf` has them all in order;
   `figures/tableS_sensitivity_n30.md` has every main result with the pilots included (n = 30);
5. `writing_followup_2026-11-10.md`: answers to the follow-up list on manuscript v2 (task and acquisition facts; its
   numbers are at n = 30 and are superseded by `key_numbers.md`).

Where older notes disagree with this package, this package wins.

**Richter et al. follow-up: done** (`richter_followup_2026-11-11.md` in this package, n = 28): numbers merged into
`key_numbers.md` (new section), caveats into `claims_and_caveats.md` (section 3); it raises the **Decisions for the
authors** below.

## Decisions for the authors (open; the storyline below is NOT yet adjusted to them)

1. **Sessions and carry-over (Fig. 3, Fig. 5c; Richter follow-up A1–A4).** HP moves to the opposite location within a
   session and to the other diagonal between sessions, so in session 2 the two neighbours of HP were former HP
   locations. The inference is the interaction test, and it is null: the location effects do not differ reliably
   between sessions or blocks (A1 session and block differences p ≥ .26; A3 sustained dS session 2 − 1 −.022,
   p = .095). Each session alone has half the data, so per-session tests are underpowered, and a difference between a
   significant and a non-significant per-session test is not itself significant; do not describe any effect as
   "a session-2 effect", "only in session 2" or "building up across sessions". A location that was HP before does not
   respond less than one that was never HP (A2, p = .54), and behaviour shows no carry-over once practice is taken out
   (A4), so there is no evidence of carry-over. It cannot be strictly excluded for the neighbour effect, because in
   session 2 the neighbours were former HP locations, but the data give no sign of it. The per-session numbers are
   descriptive (`key_numbers.md`, Richter section; possibly a supplementary table). Decide whether the paper reports
   the per-session split (supplement, with the interaction test as the inference) and whether the Discussion adds a
   sentence that carry-over cannot be strictly excluded for the neighbour effect.
2. **How much the model explains.** The joint gain-field fit reproduces about half of the measured HP − LP reduction
   (early −5.4 vs −11.7%), carried by the sustained term, and none of next − opposite. Decide how Fig. 5 / the
   Discussion phrase the link between the measured responses and the gain fields.
3. **Behaviour next to HP.** A distractor next to HP is the slowest (+37.8 ms vs opposite); opposite HP is about as fast
   as HP; the target cost is graded HP > next > opposite. Decide whether Fig. 1 / the Results report the three-way split
   and how the Discussion compares it with Richter et al. (benefit only at HP, no target cost).

## Storyline: seven figures, one question each

Each figure answers one question that leads to the next. One sentence per figure, adjusted to what the n = 28 numbers
show; caveats are in `claims_and_caveats.md`, not here.

1. **Do people learn where the distractor tends to be?** Yes: responses are faster when the distractor appears at the
   high-probability (HP) location, in all 28 participants, slower when the target appears there, and faster when the
   target or the distractor repeats the previous trial's location (Fig. 1).
2. **Can we see the search display through each voxel's receptive field?** Yes: pRFs mapped during search are
   retinotopic, and the search display drives each voxel according to how close its pRF is to the nearest item (Fig. 2).
3. **Do voxels at the item locations respond less at HP?** Yes, without any model: they respond more to the target and
   the distractor than to a neutral item, and in early visual cortex least when the item is at HP (22% below the
   opposite location), less next to HP than opposite it (15%), for every item type alike (Fig. 3).
4. **How could attention reshape what each pRF sees?** A gain-field model: each pRF reads the display through transient
   gain at the target and the distractor and a sustained gain at the four item locations, which can differ between HP
   and the other locations (Fig. 4).
5. **What do the fitted gain fields look like?** Target gain rises along the visual hierarchy, distractor gain is weaker
   (reliable in V3AB, hV4 and TO), and the sustained gain is lower at HP than at the other locations in 79% of
   participants, with every area's median in that direction; the transient gain at HP is also slightly lower, for the
   target and the distractor alike (reliable only for the target when estimated alone), so the reduction belongs to the
   location rather than to the distractor (Fig. 5).
6. **Is the HP location also visible without gain terms in the model?** Yes: decoding the bar stimulus from held-out
   runs with plain pRFs, the bar is read out at lower intensity in the quadrant facing HP than in its neighbours (V1–V3
   pooled; per area in V2 and V3 after correction), and a simulated gain field at HP of the fitted size produces an
   effect of the same order (Fig. 6).
7. **Do the gain fields change from trial to trial with history and behaviour?** Yes: the target gain drops by about a
   fifth when the target repeats its location, while the distractor gain does not change on repeats, and the distractor
   gain is lower on slower trials (Fig. 7).

## Decisions already made (follow them)

1. **Narrative built on the gain fields themselves (user, 2026-11-10).** "I don't want to lead with a model comparison
   that is noisy and is a null. Let's try to make a narrative that's easy to follow." So: the main text argues from the
   fitted parameters (target, distractor, sustained gain and dS = sustained gain at HP − at LP), their consistency
   across subjects and areas, and their agreement with the model-free responses (Fig. 3) and the gain-free decoding
   (Fig. 6). Leave-one-subject-out prediction goes to Methods and a supplementary table ("how much each term adds to
   out-of-subject prediction"), described neutrally; no main-text sentence leads with it or hinges on it. Table 1 of
   the earlier plan is no longer a main-text model-comparison table; the main-text table, if any, is the parameter
   estimates.
2. **Model.** DoG pRFs with one HRF per subject × area, fitted jointly with the pRFs, are the main model. Gaussian and DN
   pRFs (each with its own jointly fitted HRF) are robustness checks in the supplement, reported whatever they show.
3. **Measured before modelled.** Figs. 2 and 3 are measured; the model comes in Fig. 4.
4. **Estimated alone vs jointly.** "Estimated alone" = one term added to its base model, the subject's own estimate;
   the headline for dS. "Joint" = all terms fitted together (m7_full), reported alongside.
5. **Statistics.** Wilcoxon signed-rank across subjects, Benjamini-Hochberg across the nine areas; stars in figures,
   exact p in legends. Decoding per area is one-sided with BH across nine. No test is framed as pre-specified.
6. **Null results** go in the text or supplement, not as main-figure panels on their own.
7. **Scope.** Gain-field models only; pRF-shift analyses are archived and not mentioned.
8. **Pilots left out (user, 2026-11-12).** sub-01 and sub-02 were tested with a pilot version of the paradigm (bar
   0.5°/s instead of 0.168°/s; HP at three locations in blocks of 2, 3 and 1 runs) and are left out of every analysis:
   n = 28. State this in Methods as decided on paradigm grounds, and point to the n = 30 results in Supplementary Table
   `tableS_sensitivity_n30.md` (no result changes sign).

## Words

- Never "singleton": both target and distractor are singletons. Say "target" and "distractor".
- No "top-down", "bottom-up" or "selection history" as labels for terms. Name the contrast (target vs distractor gain;
  HP vs LP).
- HP = high-probability distractor location; LP = the other three; "next to HP" (the two locations 90° away) and
  "opposite HP" (diagonally opposite). Spell out once.
- Gains are fractional changes in drive at the field centre (+0.30 = 30% more drive).
- Name model codes with what they are (e.g. "the joint fit (m7_full)", "the model with the LP gain split next to vs
  opposite HP"), never a bare code.
- Voxel selection is cross-run reliability; never call the R² mixture "FDR-corrected".

## Outline

**Introduction.** Learned distractor suppression in behaviour; where and how it is implemented in visual cortex
(Richter et al. 2025 as closest precedent); gain as a mechanism; pRF mapping during search lets one model the spatial
gain of each voxel population; target, distractor and HP vs LP as contrasts. End with the "Here, we…", roadmap and
preview paragraphs.

**Results** (headings as claims, one per figure, following the storyline above).

**Discussion.** The reduction at HP belongs to the location (sustained, and transient for any item); relation to
Richter et al. 2025 (same item-response pattern; the neighbour effect rests on the measured responses); target vs
distractor gain along the hierarchy; trial-by-trial history effects; why individual differences cannot be resolved
(small effects, out-of-subject prediction of the HP term not reliable); dependence on the pRF model; HP outside the bar
aperture.

**Methods.** The "Methods facts" section at the end of `key_numbers.md` (participants, task, acquisition, preprocessing, decoder, software, global gain; with the open items), `writing_followup_2026-11-10.md`, the legends, and the repo `CLAUDE.md` (TR 1.6 s, 258 volumes, ring at 4°,
item size 1.5°, bar aperture 3.17°, 2 sessions × 6 runs; sub-20 ses-1 and sub-24 ses-2 have 5 runs).

## Figures

All figures are rendered at n = 28 (suffix `_n28`; Fig. 1 `_v4_n28`, a new layout with the two priming effects merged
into one panel e).

| Figure | File (in `figures/`) | Message | Status |
|---|---|---|---|
| 1 Paradigm and behaviour | `fig1_paradigm_behaviour_v4_n28` (raw RTs of e in `figS_priming_rt_v4_n28`) | Learned suppression of HP; target at HP slowed; repeats speed responses (e: target and distractor repeat − switch as within-subject differences) | FINAL |
| 2 pRF fits | `fig2_prf_fits_n28` | pRFs map the display; DoG and the joint per-area HRF improve held-out fits; drive falls off with distance from the nearest item | FINAL |
| 3 Item locations | `fig3_item_locations_n28` (+ `tableS_item_locations_n28.md`) | More response to target and distractor; in early areas least at HP, less next to HP than opposite, for every item | FINAL |
| 4 Model | `fig4_model_hp_blocks` | Illustration of the gain-field model; no data (unchanged) | FINAL |
| 5 Gain fields | `fig5_gain_fields_n28` (widths in `figS_field_widths_n28`) | Target gain beyond V1/V2; distractor gain in V3AB, hV4, TO; sustained gain lower at HP; transient gain at HP slightly lower, target and distractor alike | FINAL |
| 6 Decoding | `fig6_decoding_n28` (per-area test in `figS_decoding_hp_per_area_n28`) | Bar decodable everywhere; read out weaker in the quadrant facing HP | FINAL |
| 7 Gains and behaviour | `fig7_compact_n28` | Target gain drops on target repeats; distractor gain lower on slow trials | FINAL |
| Table S Parameters | `table_parameters_2026-11_n28` | All gain terms per area, DoG and DN, joint and estimated alone | FINAL |
| Table S Out-of-subject prediction | `tableS_loso_n28` | What each term adds to out-of-subject prediction (supplement only) | FINAL |
| Table S Sensitivity | `tableS_sensitivity_n30.md` | Every main result with the two pilots included (n = 30) vs n = 28 | FINAL |

**Supplementary figures** (all in `figures/`, n = 28):

| Figure | Supports | Status |
|---|---|---|
| `figS_priming_rt_v4_n28` | Raw response times behind Fig. 1e | FINAL |
| `figS_pooled_dS_dD_n28` | dS and dD per subject, pooled over areas, alone and joint | FINAL |
| `figS_field_widths_n28` | Field widths (FWHM) per area, fits at a bound counted | FINAL |
| `figS_decoding_hp_per_area_n28` | HP quadrant − other three per area | FINAL |
| `tableS_item_locations_n28.md` | Fig. 3 statistics per item type, per pathway, HP vs all LP, DCT basis | FINAL |
| `figS_gain_vs_prf_size_n28` | Target gain rises with pRF size | FINAL |
| `figS_loso_per_roi_n28` | Out-of-subject prediction per area and term | FINAL |
| `figS_dog_kappa_n28` | DoG centre/surround tilt of the sustained field (κ) | FINAL (X) |
| `figS_decoding_prf_models_n28` | Decoding quadrant contrasts with DoG, Gaussian and DN pRFs (joint HRF) | FINAL (R) |
| `figS_decoding_canonical_n28` | Fig. 6 with canonical-HRF pRFs | FINAL (R) |
| `figS_decoded_bars_rotated_n28`, `figS_decoded_bars_per_roi_n28` | Same bar, HP half weaker | FINAL |
| `figS_decoding_all_rois_n28`, `figS_decoding_all_rois_examples_n28` | The bar decoded from each area (examples re-chosen at the 75th/50th/25th percentile of the 28) | FINAL |
| `figS_decode_simulation_n28` | The decoder recovers an HP-only gain field of the fitted size | FINAL |
| `figS_bar_crossing_response_n28` | Measured response when the bar crosses a pRF near an item; baseline-dependent | FINAL (cite as consistent only) |
| `figS_gain_fields_dn_n28` | Fig. 5 with DN pRFs (final DN fits) | FINAL (R) |
| `figS_gain_fields_gauss_n28` | Fig. 5 with Gaussian pRFs | FINAL (R) |
| `figS_dn_gain_locus_n28` | The HP effect cannot be placed on DN baselines rather than input gain | FINAL (X) |
| `figS_dn_params_by_hp_n28` | DN pRF parameters per HP condition | FINAL (X) |

Dropped from the package: `fig_event_responses_2x3_deconv` (earlier HP vs LP version of Fig. 3, superseded) and
`figS_field_size` (superseded by the widths figure). The n = 30 versions of every figure stay on disk in
`/Users/gdehol/git/retsupp/notes/figures/paper_2026-11/` (`*_v3`, `*_v4`, `*_final2`).

## Freeze line (proposed, not yet confirmed by the user)

- **Must finish:** main DoG numbers at n = 28 (done), Gaussian and DN robustness (decoding and joint fits done;
  estimated-alone terms and out-of-subject tests done for both; Fig. 3 and Fig. 7 not redone with them), κ and DN-baseline LOSO (done), legend fixes (done).
- **Freeze (no more work):** DN locus, DoG centre/surround tilt, split models, field widths, decoder simulation.
- **Drop:** field-width filter, per-condition DN fits beyond the existing supplement, new model variants, the
  HP-rotation null for dS unless the user wants it.

## CHANGES 2026-11-12 (n = 28: the two pilots left out)

- **Richter et al. follow-up merged** (`richter_followup_2026-11-11.md`): behaviour by location relative to HP (B), target
  vs distractor at the item locations (D5), per-area location effect (D6), carry-over by session and block (A1–A4), and
  how much of the measured HP reduction the joint fit reproduces (C) in `key_numbers.md`; consequences listed under
  "Decisions for the authors". Fixes from it: HP − LP at item locations is −11.7% (early) in the Fig. 3d, e
  deconvolution and −9.8% in the earlier HP/LP deconvolution (n = 28; say which); decoder voxels ≤ 2000; correct trials
  17,292.

- **Pilots left out (user decision).** sub-01 and sub-02 were tested with a pilot version of the paradigm: the mapping
  bar moved at 0.5°/s instead of 0.168°/s (≈ 14 s instead of ≈ 42 s per sweep, which changes the pRF mapping), and HP
  took only three of the four locations, in blocks of 2, 3 and 1 runs, in the same order in both sessions (one diagonal
  location never HP); the search task was the same code. They are left out of every analysis: **n = 28**. Methods
  sentence (draft): "Two further participants were tested with a pilot version of the paradigm, in which the mapping
  bar moved about three times faster (0.5° s⁻¹ instead of 0.168° s⁻¹) and the high-probability location took only three
  of the four positions, in blocks of 2, 3 and 1 runs per session; they were excluded from all analyses, which leaves
  28 participants. Including them does not change the direction of any result (Supplementary Table S…)."
- **Every number and figure at n = 28.** Per-subject fits are unchanged (they do not depend on other subjects); every
  group statistic was recomputed, including what pools over subjects inside an analysis: the leave-one-subject-out δ
  (pilots out of the pool), the Fig. 7 hierarchical model (refitted), the group-mean decoded frames and decoded-bar
  images (re-collected on the cluster), the hierarchy regression, the decoding example subjects (re-chosen by
  percentile among the 28). New supplementary `tableS_sensitivity_n30.md`.
- **Stronger without the pilots:** the HP-location effects. Fig. 3 early areas: location type F(2, 54) = 7.67,
  p = .001 (was .003); next − opposite −14.9%, p = .022 (was −13.0%, p = .047); HP − opposite −21.6%, p = .005. Fig. 6:
  HP < next p = .004 (was .015); Friedman across quadrants p = .013 (was .062); per area V2 and V3 survive BH (none
  did). Fig. 5: dS joint p = .0027 and TO joins V3 and hV4; split model HP − opposite p = .0048. Gaussian joint dS
  p = .027 (was .078).
- **Unchanged:** behaviour (HP benefit 28/28), Fig. 2d, Fig. 3b, c, dS estimated alone (−.014, 79%, p = 1.1 × 10⁻⁴;
  V3 and TO), dS out of subject (direction yes, prediction no), target-gain results, the Fig. 7 target-repeat effect
  (all areas P = .997) and the distractor-gain RT effect (P = .990).
- **Weaker without the pilots:** the distractor-gain statements and the global gain. The tied distractor gain no
  longer improves out-of-subject prediction (61%, p = .16; was 77%, p = .031); under DN pRFs it predicts worse
  (18% better, p = .018). The distractor gain on top of the target, estimated alone, p = .019 (was .007), loses TO.
  dD alone p = .099 (was .050), so Fig. 5d's title now reads "Transient gain at HP: slightly less, target and
  distractor alike". **Fig. 7b: the global-gain RT effect is no longer credible (P = .943, was .979)**; the panel title
  is now "Lower distractor gain on slower trials" and the storyline sentence 7 says so. The early group's target-repeat
  effect falls just below the .975 threshold (P = .962). Target gain loses V2 and LO per area (the two weakest).
- **Fig. 1 v4 layout** (separate request): the target and distractor repeat effects are one panel e of within-subject
  differences; their raw RTs moved to `figS_priming_rt_v4_n28`. Refer to the priming panel as "Fig. 1e".
- **Fixes:** the decoder uses at most **2000** voxels per area (the 2026-11-11 key_numbers said 200; the Fig. 6 legend
  was right); "21,480 correct trials" was the total number of trials (n = 30; 18,609 correct). At n = 28: 20,040 trials,
  17,292 correct.
- **New supplementary figure** `figS_dog_kappa_n28` (the DoG centre/surround tilt κ); κ and DN-baseline out-of-subject
  tests rerun at n = 28 (`notes/data/n28/loso_kappa_2026-11_n28.tsv`, `loso_dnw_2026-11_n28.tsv`).

## CHANGES 2026-11-11 (revision pass on manuscript v2)

- **Figures v3** (follow-up part B): Fig. 1 with within-subject difference strips (median, 95% CI, counts); Fig. 3 with
  a short legend, details in `tableS_item_locations.md`; Fig. 5b only the to-scale field, widths to `figS_field_widths_v3`;
  Fig. 6 old d to `figS_decoding_hp_per_area_v3`, re-lettered (old e → d, old f → e), larger inset text; Fig. 7 narrower;
  all main figures 180 mm, text ≥ 6 pt.
- **Fig. 2 v4:** panels b, c from the final pRFs (joint-HRF DoG fold fits; held-out R² ≥ .1); the earlier versions used
  the model-4 mean fits.
- **Behaviour as medians** with bootstrapped 95% CIs (they match the Wilcoxon tests and the figure); CIs added for
  Fig. 2d and Fig. 3b, c; Fig. 3 dorsal n = 28 (b, c) vs 27 (d, e) explained.
- **Methods facts** added to `key_numbers.md`: participants (aggregate demographics), task, acquisition,
  preprocessing, decoder, software versions, the global gain. Open: ethics body/number and payment (authors), the
  sidecar "3D"/multiband and `syn-sdc` checks, braincoder commit per fit.
- **κ and the neighbour pattern:** the fitted centre/surround tilt predicts +0.2% next to HP (n.s.), opposite in sign
  and ~50× smaller than the measured −13% (`notes/data/kappa_neighbour_2026-11.tsv`); Discussion sentence in
  `claims_and_caveats.md`.
- `writing_followup_2026-11-10.md` (answers to the follow-up list) is part of the package.

## CHANGES 2026-11-10 (since the 2026-11-09 package)

- **All gain numbers are on the final fits** (joint per-area HRF, reliable voxels); no `[PROVISIONAL]` tokens remain.
- **Narrative:** LOSO moved to Methods + supplementary table (user decision). On the final fits dS no longer improves
  out-of-subject prediction (67%, p = .18; was 77%, p = .010); its estimate and direction remain (alone −.012, 80% of
  subjects, p = 5.6 × 10⁻⁵; direction 77%, p = .0024).
- **Fig. 5:** distractor gain now reliable in V3AB, hV4, TO (was V3AB, IPS, hV4, VO); the transient HP − LP reduction is
  no longer distractor-specific (dD pooled p = .050, only LO; dT p = .016; they do not differ); the sustained field is
  3.3° wide (was 4.8°), narrower than the item spacing.
- **Fig. 2d** uses the jointly fitted HRF (was the median-of-voxels HRF); **Fig. 2e and Fig. 3** use the final pRFs and
  HRF for voxel selection and bar removal; results nearly unchanged (one dorsal subject left out of the relative
  analyses).
- **Fig. 7** replaced by the compact version: history and RT as modulators of the gains estimated on the BOLD, combined
  by a hierarchical Bayesian model.
- **New supplements:** split models, Gaussian/DN decoding, decoder simulation legend, DN-by-HP legend, complete DN
  locus held-out set, κ and DN-baseline LOSO.
- **Table 1:** the "items 100%" row is explained (bar-only pRFs, not items vs none) and the table moves to the
  supplement.
