# retsupp paper: writing package (2026-11-11, final fits; revision pass on manuscript v2)

This package replaces the 2026-11-09 package (`/Users/gdehol/git/retsupp/notes/writing_package_2026-11/`), which was
never implemented. Hand this folder to the writing agent; it is self-contained. Read in this order:

1. this file (storyline, decisions, words, outline, figure list with status); start with `prompt_writing_agent_2026-11-11.md`;
2. `claims_and_caveats.md`: **the guardrail** (what the data support, what they do not, caveats);
3. `key_numbers.md`: the only place to take numbers from, each with its source file;
4. `figures/`: one PDF + PNG + legend `.md` per figure (current versions, from
   `/Users/gdehol/git/retsupp/notes/figures/paper_2026-11/`); `all_figures_2026-11-11.pdf` has them all in order;
5. `writing_followup_2026-11-10.md`: answers to the follow-up list on manuscript v2 (CIs, task and acquisition facts,
   figure changes).

Where older notes disagree with this package, this package wins.

## Storyline: seven figures, one question each

Each figure answers one question that leads to the next. One sentence per figure, adjusted to what the final numbers
show; caveats are in `claims_and_caveats.md`, not here.

1. **Do people learn where the distractor tends to be?** Yes: responses are faster when the distractor appears at the
   high-probability (HP) location, in all 30 participants, slower when the target appears there, and faster when the
   target or the distractor repeats the previous trial's location (Fig. 1).
2. **Can we see the search display through each voxel's receptive field?** Yes: pRFs mapped during search are
   retinotopic, and the search display drives each voxel according to how close its pRF is to the nearest item (Fig. 2).
3. **Do voxels at the item locations respond less at HP?** Yes, without any model: they respond more to the target and
   the distractor than to a neutral item, and in early visual cortex least when the item is at HP (19% below the
   opposite location), less next to HP than opposite it, for every item type alike (Fig. 3).
4. **How could attention reshape what each pRF sees?** A gain-field model: each pRF reads the display through transient
   gain at the target and the distractor and a sustained gain at the four item locations, which can differ between HP
   and the other locations (Fig. 4).
5. **What do the fitted gain fields look like?** Target gain rises along the visual hierarchy, distractor gain is weaker,
   and the sustained gain is lower at HP than at the other locations in 80% of participants and in every area; the
   transient gain at HP is also slightly lower, for the target as much as for the distractor, so the reduction belongs to
   the location rather than to the distractor (Fig. 5). *How the story bends:* the 09 package expected a
   distractor-specific transient suppression; on the final fits it is not distractor-specific, which fits Fig. 3 (no
   location × item interaction) and Fig. 1 (a target at HP is slowed).
6. **Is the HP location also visible without gain terms in the model?** Yes: decoding the bar stimulus from held-out
   runs with plain pRFs, the bar is read out at lower intensity in the quadrant facing HP than in its neighbours (V1–V3
   pooled), and a simulated gain field at HP of the fitted size produces an effect of the same order (Fig. 6).
7. **Do the gain fields change from trial to trial with history and behaviour?** Yes: the target gain drops by about a
   fifth when the target repeats its location, while the distractor gain does not change on repeats, and the distractor
   and global gains are lower on slower trials (Fig. 7).

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

| Figure | File (in `figures/`) | Message | Status |
|---|---|---|---|
| 1 Paradigm and behaviour | `fig1_paradigm_behaviour_v3` (with within-subject difference strips) | Learned suppression of HP; target at HP slowed; repeats speed responses | FINAL |
| 2 pRF fits | `fig2_prf_fits_v4` | pRFs map the display; DoG and the joint per-area HRF improve held-out fits; drive falls off with distance from the nearest item | FINAL |
| 3 Item locations | `fig3_item_locations_v3` (+ `tableS_item_locations.md`) | More response to target and distractor; in early areas least at HP, less next to HP than opposite, for every item | FINAL |
| 4 Model | `fig4_model_hp_blocks` | Illustration of the gain-field model; no data | FINAL |
| 5 Gain fields | `fig5_gain_fields_v3` (widths in `figS_field_widths_v3`) | Target gain beyond V1; distractor gain in V3AB, hV4, TO; sustained gain lower at HP; transient gain at HP slightly lower for both items | FINAL |
| 6 Decoding | `fig6_decoding_v3` (old d → `figS_decoding_hp_per_area_v3`; old e, f → d, e) | Bar decodable everywhere; read out weaker in the quadrant facing HP | FINAL |
| 7 Gains and behaviour | `fig7_compact_v3` | Target gain drops on target repeats; distractor and global gain lower on slow trials | FINAL |
| Table S Parameters | `table_parameters_2026-11_final2` | All gain terms per area, DoG and DN, joint and estimated alone | FINAL |
| Table S Out-of-subject prediction | `tableS_loso_final2` | What each term adds to out-of-subject prediction (supplement only) | FINAL |

**Supplementary figures** (all in `figures/`):

| Figure | Supports | Status |
|---|---|---|
| `figS_pooled_dS_dD_final2` | dS and dD per subject, pooled over areas, alone and joint | FINAL |
| `figS_field_widths_v3` | Field widths (FWHM) per area, fits at a bound counted (was Fig. 5b) | FINAL |
| `figS_decoding_hp_per_area_v3` | HP quadrant − other three per area (was Fig. 6d) | FINAL |
| `tableS_item_locations.md` | Fig. 3 statistics per item type, per pathway, HP vs all LP, DCT basis | FINAL |
| `figS_field_size_final2` | Field size per area (earlier layout) | superseded by `figS_field_widths_v3`; not in the zip |
| `figS_gain_vs_prf_size_final2` | Target gain rises with pRF size | FINAL |
| `figS_loso_per_roi_final2` | Out-of-subject prediction per area and term | FINAL |
| `figS_decoding_prf_models_final2` | Decoding quadrant contrasts with DoG, Gaussian and DN pRFs (joint HRF) | FINAL (R) |
| `figS_decoding_canonical` | Fig. 6 with canonical-HRF pRFs | FINAL (R) |
| `figS_decoded_bars_rotated`, `figS_decoded_bars_per_roi` | Same bar, HP half weaker | FINAL |
| `figS_decoding_all_rois`, `figS_decoding_all_rois_examples` | The bar decoded from each area | FINAL |
| `figS_decode_simulation` | The decoder recovers an HP-only gain field of the fitted size | FINAL |
| `figS_bar_crossing_response` | Measured response when the bar crosses a pRF near an item; baseline-dependent | FINAL (cite as consistent only) |
| `figS_gain_fields_dn_final2` | Fig. 5 with DN pRFs (final DN fits) | FINAL (R) |
| `figS_gain_fields_gauss_final2` | Fig. 5 with Gaussian pRFs | FINAL (R) |
| `figS_dn_gain_locus_complete` | The HP effect cannot be placed on DN baselines rather than input gain | FINAL (X) |
| `figS_dn_params_by_hp` | DN pRF parameters per HP condition (legend added) | FINAL (X) |
| `fig_event_responses_2x3_deconv` | Earlier HP vs LP version of Fig. 3 (earlier pRFs) | superseded by Fig. 3; optional |

Superseded versions stay on disk in `figures/` but are left out of the zip: `fig1_paradigm_behaviour`,
`fig2_prf_fits_final2`, `fig2_prf_fits_v3`, `fig3_item_locations_final2`, `fig5_gain_fields_final2`, `fig6_decoding`,
`fig7_compact_v2`, `figS_field_size_final2`.

Retired: the October-fit `fig7_behaviour` and `fig7_behaviour_final2` (RT regressed on per-trial gains); `table1_ladder`
as a main table; `figS_cv_power` (earlier fits).

## Freeze line (proposed, not yet confirmed by the user)

- **Must finish:** main DoG numbers (done), Gaussian and DN robustness (decoding and joint fits done;
  estimated-alone terms and out-of-subject tests done for both; Fig. 3 and Fig. 7 not redone with them), κ and DN-baseline LOSO (done), legend fixes (done).
- **Freeze (no more work):** DN locus, DoG centre/surround tilt, split models, field widths, decoder simulation.
- **Drop:** field-width filter, per-condition DN fits beyond the existing supplement, new model variants, the
  HP-rotation null for dS unless the user wants it.

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
