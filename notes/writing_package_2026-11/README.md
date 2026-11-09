# retsupp paper: writing package (2026-11-06)

For the writing agent. Read this file first, then `key_numbers.md`, then the figure legends in
`figures/` in this folder: one PNG per figure with its legend (`.md`) beside it, plus Table 1 and the
parameter table. These are a snapshot of 2026-11-06; the live versions are in
`/Users/gdehol/git/retsupp/notes/figures/paper_2026-11/`. All figures as one PDF:
`all_figures_2026-11.pdf` in this folder (local only, not in git). Background: `notes/paper_plan_2026-11.md` (plan and rules),
`notes/results_overview_2026-11.md` (every result with a grade), `notes/fitting_lessons_2026-11.md`
(methods details). Where those older notes disagree with this package, this package wins.

**Everything with a gain number is provisional.** The final rerun (plan Phase C) on the final HRF and
voxel selection has not happened yet. Write the text with the numbers given here, mark every gain
number `[PROVISIONAL]`, and leave `[PENDING: ...]` placeholders where a result does not exist yet.
Behavioural numbers are final.

## The story in one paragraph

Thirty participants searched eight bars for the one with a different orientation (the target) and
reported whether it contained a grey dot (two buttons: dot, no dot), while a colour-singleton distractor
appeared more often at one location (high probability, HP). Behaviour shows the learned suppression
clearly: responses are 28 ms faster when the distractor is at HP than at another location, in all 30
participants. During the same 7T fMRI runs a bar swept the visual field, so each voxel's population
receptive field (pRF) could be mapped. We model each voxel's response as its pRF reading the display
through a gain field: transient gain at the target and at the distractor during search, and sustained
gain at the four item locations across a block, allowed to differ between HP and the other locations.
The target draws strong gain that grows along the visual hierarchy; the distractor draws less. The key
test, dS = sustained gain at HP minus at the other locations, is negative: visual cortex gives slightly
less gain to the location where the distractor is likely. The effect is small, and with n = 30 and gain
fields several degrees wide it cannot be resolved per participant, so it does not relate to individual
differences in behaviour. Trial by trial, gain does relate to behaviour: target gain drops when the target
repeats its location, and the gain on a given trial predicts that trial's response time, with opposite
signs in early visual and parietal cortex.

## Framing decisions already made (follow them)

1. **Model.** DoG pRFs are the main model, chosen for interpretability (a gain scales the drive
   directly; under divisive normalisation part of it is divided back out) and fixed in the plan before
   the key test. DN pRFs are the robustness check in the supplement, reported whatever they show. Say
   openly that DN fits the pRFs slightly better (held-out R² .216 vs .210) and that dS is weaker there.
   Gaussian pRFs may be added to the same supplement.
2. **Power framing (user, 2026-11-06).** State explicitly that with n = 30 and gain fields this wide
   (FWHM of several degrees, often wider than the 4° ring), condition differences are hard to detect:
   dS is tiny relative to between-subject variance and per-subject dS is too unreliable for
   brain-behaviour correlations across subjects. That is why the behavioural analyses use trial-to-trial
   fluctuations within subjects.
3. **Null results** go in the text, not the main figures. Report them with the test and the caveat
   about power (e.g. dS vs behavioural HP benefit across subjects: ρ = −.17, p = .37).
4. **Within-subject cross-validation** cannot test dS (a simulation with our effect size fails it too);
   present it as underpowered, not as evidence against dS. Leave-one-subject-out (LOSO) is the
   model-comparison metric.
5. **Conditional vs joint estimates.** Joint = all gains fitted together (m7_full). Conditional = one
   term estimated alone on top of m4 (target, distractor and sustained gain already fitted), a
   one-step linear estimate; it is also the effect size used in LOSO. dS agrees between the two; dD does
   not (conditional negative, joint not different from 0), so dD is not a robust result.
6. **Scope.** Gain-field models only. pRF-shift analyses are archived; do not mention or cite them.

## Words

- Never "singleton" for the distractor: both target and distractor are singletons. Say "target" and
  "distractor".
- Do not label single model terms "top-down", "bottom-up" or "selection history". The three-way split
  of Awh, Belopolsky & Theeuwes (2012) can structure the introduction and discussion, mapped onto
  contrasts: target vs distractor gain (goal relevance, both being singletons) and HP vs LP
  (selection history, carried by dS). The sustained gain shared by all four locations is a no-control
  term, not history.
- Never call the R² mixture model "FDR-corrected".
- HP = high-probability distractor location; LP = the three other locations. Spell out once.
- Gains are fractional changes in drive at the field centre (+0.30 = 30% more drive).

## Proposed outline

**Introduction.** Learned distractor suppression in behaviour; open question of where and how it is
implemented in visual cortex; gain vs other accounts; pRF mapping during search lets one model the
spatial gain profile per voxel population; the three sources of gain (target, distractor, history) as
contrasts. End with what we did and found.

**Results**
1. Learned suppression in behaviour (Fig. 1). HP benefit 28 ms (30/30); target at HP slower by 28 ms.
2. pRFs and model fits (Fig. 2). Retinotopy, contralaterality, size vs eccentricity, held-out R², responses to the
   search display falling off with distance from the nearest item. DN/Gaussian and HRF choices: supplement.
3. Responses at the item locations, without a model (Fig. 3). More response to the target and the distractor than to
   a neutral item. In early areas a gradient over location types, the same for every item type: least response at
   HP, less next to HP than opposite it (next − opposite −14%, p = .029; HP − opposite −19%, p = .011; HP − next −5%,
   n.s.; DCT basis alike); HP vs all LP −8% (p = .042). Dorsal and occipitotemporal: no reliable location effect.
4. A gain-field model of pRFs during search (Fig. 4, Table 1). Nested ladder of models; items as
   stimuli; LOSO per step (Table 1).
5. The gain fields (Fig. 5). Target gain everywhere beyond V1; distractor gain weaker, mainly dorsal and hV4/VO
   (Fig. 5a; gain against pRF size in the supplement, its slope may partly reflect a trade-off with item weight and
   field size; field sizes: supplement). Less sustained gain at the likely distractor location (Fig. 5c, b): dS
   conditional and joint; per ROI; LOSO 77% (per ROI, figS_loso_per_roi); `[PENDING: HP-rotation null, the pre-specified
   primary test]`. Location as in Fig. 1c, d (Fig. 5d): less gain for a distractor at HP (dD, estimated alone) in
   V3AB, IPS, hV4, VO, not robust in the joint fit; target gain HP − LP (estimated alone) not reliable after correction
   (smallest q = .07), and null in the joint fit.
6. Converging evidence without gain terms (Fig. 6, decoding; pRFs with an HRF per subject × ROI, no gain terms,
   held-out runs). The bar can be decoded from every area (median frame-wise r .52/.47/.40 in V1/V2/V3, .12–.29 in
   the higher areas, all far above shuffled frames). Decoded intensity is lower in the quadrant facing HP: per ROI
   (one-sided Wilcoxon, Benjamini-Hochberg across the nine ROIs) V2, V3, V3AB, VO and TO p = .013–.043 but none
   survives correction (q = .076–.077; V1 q = .41); pooled over V1–V3 the HP quadrant is below its
   neighbours (p = .015) and below the other quadrants over the whole aperture (p = .033); the effect is spread from
   1° to the aperture edge, not larger near the items; with the stimulus held identical (each frame minus its mean over
   HP locations), the half of a bar in the HP quadrant is decoded weaker than its other half, for vertical (−1.6%,
   p = .031) and horizontal bars (−1.8%, p = .001) (supplement). The bar-outline decoder gives the
   same (q = .090 for V2, V3, V3AB and TO). The measured bar-pass response (supplement) points the
   same way but its test depends on the baseline (p = .017 without, n.s. with).
   Canonical-HRF pRFs (supplement): V2 .044, V3 .045.
7. Trial history in the gains, and gain and behaviour trial by trial (Fig. 7). History (Fig. 7a, as Fig. 1e, f;
   the location effects of Fig. 1c, d are Fig. 5d): repeats lower target gain in V3AB, IPS, hV4, VO; distractor gain
   repeat − switch not reliable after correction. Per-trial gain predicts response time, sign flips between early
   visual and parietal cortex; distractor gain averaged over HP and LP trials (Fig. 7b).
   `[PENDING: D4, gain at HP split by what HP holds on the trial]`.

**Discussion.** Small but consistent proactive suppression as reduced sustained gain at HP; the
target-vs-distractor contrast as goal relevance; why individual differences cannot be resolved (power
framing above); dependence on the pRF model (DoG vs DN); HP lies outside the bar-mapped aperture
(3.17°) so only pRFs reaching the ring carry the effect; limits: provisional fits, wide fields,
trial-wise analyses on earlier fits.

**Methods.** Use `notes/fitting_lessons_2026-11.md` and the figure legends; constants in the repo
`CLAUDE.md` (TR 1.6 s, 258 volumes, ring at 4°, item size 1.5°, bar aperture 3.17°, 2 sessions × 6
runs, sub-20 ses-1 and sub-24 ses-2 have 5 runs).

## Figures (PNG and legend for each in `figures/`)

Order since 2026-11-09: measured results (pRF fits, item locations) before the model.

| Figure | File | Status |
|---|---|---|
| 1 Paradigm and behaviour | `fig1_paradigm_behaviour.pdf` (response times; e, f repeat vs switch) | final |
| 2 pRF fits | `fig2_prf_fits.pdf` | provisional (d: HRF choice) |
| 3 Responses at the item locations (model-free; d, e by location type: HP, next to HP, opposite) | `fig3_item_locations.pdf` | final apart from layout |
| 4 Gain-field model | `fig4_model_hp_blocks.pdf` (chosen 2026-11-06) | final |
| 5 Gain fields (model parameters; d, distractor and target HP − LP as Fig. 1c, d) | `fig5_gain_fields.pdf` | provisional |
| 6 Decoding | `fig6_decoding.pdf` | final inputs |
| 7 Trial history in the gains; behaviour trial by trial | `fig7_behaviour.pdf` (a history, as Fig. 1e, f; b trial by trial) | prototype on earlier fits (Oct 2026) |
| Table 1 Ladder | `table1_ladder.md` | provisional |
| Table S Parameters | `table_parameters_2026-11.md` | provisional |
| Supplement | `figS_gain_vs_prf_size.pdf` (was Fig. 5a), `figS_field_size.pdf`, `figS_cv_power.pdf`, `figS_pooled_dS_dD.pdf`, `figS_gainfield_size.pdf`, `figS_loso_per_roi.pdf` (was Fig. 5b), `figS_gain_fields_dn.pdf` (Fig. 5 with DN pRFs) | provisional |
| Supplement (decoding) | `figS_decoding_canonical.pdf`, `figS_decoding_dn.pdf`, `figS_decoded_bars_rotated.pdf`, `figS_decoded_bars_per_roi.pdf`, `figS_decoding_all_rois.pdf`, `figS_decoding_all_rois_examples.pdf`, `figS_bar_crossing_response.pdf` | final |

## Open items the text must leave room for

- HP-rotation null for dS (primary test; plan B/C3).
- Final HRF (per subject × ROI test running) and voxel selection; then the Phase C rerun of all fits.
- D2: target gain at HP vs LP (fit running, DoG).
- D4: trial-wise gain at HP split by what HP holds (fit running).
- Reliability of per-subject dS across sessions (would justify the power framing with a number).
