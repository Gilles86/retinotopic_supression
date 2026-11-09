# retsupp paper: writing package (updated 2026-11-09)

For the writing agent and the user. Read in this order:

1. this file (story, outline, figure list with claims, key numbers and status);
2. `claims_and_caveats.md` (what the data support, what they do not, open questions): **the guardrail**;
3. `key_numbers.md` (every number, with source file and status; the only place to take numbers from);
4. the figure legends in `figures/` (one PNG + legend `.md` per figure, copied 2026-11-09 from
   `/Users/gdehol/git/retsupp/notes/figures/paper_2026-11/`, which holds the live versions).

The legends are the ground truth. Where older notes (`notes/results_overview_2026-11.md`,
`notes/paper_plan_2026-11.md`, the 2026-11-05 prompts) disagree with this package, this package wins.
`all_figures_2026-11.pdf` in this folder is a **stale** snapshot of 2026-11-06 in the old figure order; do not use it.

**Status labels used below.**
- **final**: inputs and analysis fixed; numbers can go in the text.
- **provisional**: gain-field numbers from the current fits. The final DoG rerun with one HRF per subject × ROI
  (cluster job 8196318) will recompute every gain number. Write them as `[PROVISIONAL]`.
- **prototype**: computed on the earlier October 2026 fits; will be redone on the final fits.
- **exploratory**: not part of the planned analyses; supplement or text at most, no claims.

## The story in one paragraph

Thirty participants searched a ring of eight bars for the one with a different orientation (the target) and
reported whether it carried a dot, while a colour-defined distractor appeared more often at one location (the
high-probability, HP, location). Responses were 28 ms faster when the distractor was at HP, in all 30
participants, and slower when the target was there. During the same 7T runs a bar swept the visual field inside a
3.17° aperture, so each voxel's population receptive field (pRF) could be mapped and the search display could be
read through it. Measured without a model, voxels whose pRF sits at an item location respond more to the target and
the distractor than to a neutral item, and in early visual cortex (V1–V3) they respond least when the item is at the
HP location, less next to HP than opposite it, for every item type: the same spatial pattern Richter, van Moorselaar
& Theeuwes (eLife 2025) reported at 3T, here with pRF-defined voxels at 7T. A gain-field model, in which each pRF reads
the display through transient gain at the target and the distractor and a sustained gain at the four item locations,
gives strong target gain that rises along the hierarchy, weaker distractor gain, and slightly less sustained gain at HP
than at the other locations (dS < 0, estimated alone) in most areas. A model with no gain terms points the same way:
the bar, decoded from held-out runs, is decoded at lower intensity in the quadrant facing HP than in its neighbouring
quadrants (pooled over V1–V3), although no single area survives correction. The effects are small relative to
between-subject variance and the gain fields are several degrees wide, so they cannot be resolved per participant;
within participants, trial by trial, target gain is lower when the target repeats its location, and the target,
distractor and global gain on a trial predict that trial's response time.

## Framing decisions already made (follow them)

1. **Model.** DoG pRFs with one HRF per subject × ROI are the main model, fixed before the key test. DN and Gaussian
   pRFs are robustness checks in the supplement, reported whatever they show. DN fits the pRFs slightly better.
2. **Measured before modelled.** Figs. 2 and 3 are measured (pRF fits, item-location responses); the model comes in
   Fig. 4. The model-free item-location result (Fig. 3) and the gain-free decoding (Fig. 6) carry the HP claim
   together with dS; none carries it alone.
3. **Power framing (user, 2026-11-06).** With n = 30 and gain fields this wide (sustained FWHM ≈ 5°, wider than the
   spacing of neighbouring items), condition differences are hard to detect; per-subject dS is too unreliable for
   brain–behaviour correlations across subjects. That is why behaviour is related to gain within subjects, trial by
   trial (Fig. 7).
4. **Null results** go in the text, with the test and the power caveat, not in main figures.
5. **Estimated alone vs jointly.** "Estimated alone" (conditional) = one term added on top of the item model with
   target, distractor and sustained gain; the headline effect size for dS and the one used in LOSO. "Joint" = all
   terms fitted together (m7_full), reported as a check.
6. **Within-subject cross-validation** cannot test dS (a planted effect of our size fails it too); say so, do not
   present it as evidence against dS. Leave-one-subject-out (LOSO) is the model-comparison metric.
7. **Scope.** Gain-field models only. pRF-shift analyses are archived: do not mention or cite them.

## Words

- Never "singleton": both target and distractor are singletons. Say "target" and "distractor".
- No "top-down", "bottom-up" or "selection history" as labels for model terms. The Awh, Belopolsky & Theeuwes
  (2012) split may structure the introduction and discussion, mapped onto contrasts: target vs distractor gain;
  HP vs LP (dS). The sustained gain shared by all four locations is a no-control term, not history.
- HP = high-probability distractor location; LP = the three other locations. "Next to HP" = the two neighbouring
  locations (90° away); "opposite" = the diagonally opposite one. Spell out once.
- Gains are fractional changes in drive at the field centre (+0.30 = 30% more drive).
- Never call the R² mixture model "FDR-corrected". Voxel selection is cross-run reliability.

## Outline (new order, 2026-11-09)

**Introduction.** Learned distractor suppression in behaviour; where and how it is implemented in visual cortex
(Richter et al. 2025 as the closest precedent: lower early-visual responses at HP and next to it); gain as a
mechanism; pRF mapping during search lets one model the spatial gain profile of each voxel population; target,
distractor and HP vs LP as contrasts. End with what we did and found.

**Results**
1. Learned suppression in behaviour (Fig. 1).
2. pRFs mapped during search (Fig. 2).
3. Responses at the item locations, without a model (Fig. 3).
4. A gain-field model of pRFs during search (Fig. 4, Table 1).
5. The gain fields: target, distractor, and less sustained gain at HP (Fig. 5).
6. Converging evidence without gain terms: decoding the bar (Fig. 6).
7. Gain and behaviour: history and trial-by-trial response times (Fig. 7).

**Discussion.** Small, consistent lower response/gain at HP in early visual cortex; relation to Richter et al. 2025
(same pattern; our decoded bar is a different measure); target vs distractor gain; why individual differences
cannot be resolved; dependence on the pRF model (DoG vs DN); HP lies outside the bar aperture (3.17°), so only pRFs
reaching the ring carry the gain effect; limits: provisional fits, wide fields, trial-wise analyses on earlier fits,
HP-rotation null pending.

**Methods.** `notes/fitting_lessons_2026-11.md`, the legends, and the repo `CLAUDE.md` (TR 1.6 s, 258 volumes, ring
at 4°, item size 1.5°, bar aperture 3.17°, 2 sessions × 6 runs; sub-20 ses-1 and sub-24 ses-2 have 5 runs; sub-01
and sub-02 have a different block order).

## Figures

| Figure | File (in `figures/`) | Claim it supports | Key numbers | Status |
|---|---|---|---|---|
| 1 Paradigm and behaviour | `fig1_paradigm_behaviour` | Learned suppression of the HP location; a target at HP is slowed; location repeats speed responses | HP benefit 27.8 ms (30/30, p < .001); target at HP +28.4 ms (26/30); target repeat −38 ms, distractor repeat −15 ms | final |
| 2 pRF fits | `fig2_prf_fits` | pRFs are well mapped during search; retinotopic, contralateral, size grows with eccentricity (flat in V1); the search display drives voxels by distance to the nearest item | Held-out R² gained over Gaussian + canonical HRF: DoG +.0035, per-ROI HRF +.011, both +.017 | final apart from d (d uses the median-of-voxels HRF until the jointly fitted HRF is in) |
| 3 Responses at the item locations | `fig3_item_locations` | More response to target and distractor than to a neutral item; in early areas HP < next to HP < opposite, same for every item type | Early: next − opposite −14% (p = .029), HP − opposite −19% (p = .011), HP − next −5% (p = .16); location × item F(4,116) = 1.01, p = .41 | final apart from layout |
| 4 Gain-field model | `fig4_model_hp_blocks` | Illustration of the model; no data | none | final |
| Table 1 Ladder | `table1_ladder` | Which terms predict new subjects (LOSO) | T 90%, S 90%, dS 77% (DoG) | provisional; the "Items 100%" row is wrong as stated (see caveats) |
| 5 Gain fields | `fig5_gain_fields` | Target gain beyond V1, rising up the hierarchy; weaker distractor gain (dorsal, hV4, VO); sustained field ≈ 5° FWHM; less sustained gain at HP (dS < 0, estimated alone, 7 of 9 ROIs); less transient gain for a distractor at HP (4 ROIs), target HP − LP not reliable, the two not different | dS pooled −.009, 80% < 0, p < .001; dD in V3AB, IPS, hV4, VO q < .05 | provisional |
| 6 Decoding | `fig6_decoding` | The bar is decodable from all nine areas; decoded at lower intensity in the quadrant facing HP than next to it (V1–V3 pooled) | r .51/.47/.40 in V1/V2/V3; HP −0.85% vs next +0.46%, p = .015; no ROI q < .05 (smallest .076) | final |
| 7 Gains and behaviour | `fig7_behaviour` | Less target gain on target repeats (V3AB, IPS, hV4, VO); per-trial target, distractor and global gain predict RT | Target repeat − switch −.34 to −.40, q ≤ .025 | prototype (October fits) |
| Table S Parameters | `table_parameters_2026-11.md` | All gain terms per ROI, DoG and DN, joint and estimated alone | | provisional |

**Supplementary figures** (all in `figures/`):

| Figure | Supports | Status |
|---|---|---|
| `figS_decoding_canonical` | Fig. 6 with canonical-HRF pRFs (V1–V3): same result, V2 and V3 q = .034 (BH over V1–V3) | final |
| `figS_decoding_dn` | Fig. 6 with DN pRFs: bar decodable everywhere; HP-quadrant effect weaker (HP < next p = .057) | final |
| `figS_decoding_prf_models` | Quadrant contrasts across pRF models; next vs opposite not reliable in any (p = .13–.43) | final |
| `figS_decoded_bars_rotated` | Same bar, HP-quadrant half weaker for vertical (−1.6%, p = .031) and horizontal bars (−1.8%, p = .001) | final |
| `figS_decoded_bars_per_roi` | The same per ROI; only V3 survives BH (−2.7%, q = .011) | final |
| `figS_decoding_all_rois`, `figS_decoding_all_rois_examples` | The bar decoded from each area, group and example subjects | final |
| `figS_bar_crossing_response` | Measured response when the bar crosses a pRF near an item: lower at HP (p = .017), but n.s. with a pre-crossing baseline (p = .16) | final; baseline-dependent |
| `figS_loso_per_roi` | LOSO per ROI and term | provisional (earlier fits) |
| `figS_gain_vs_prf_size` | Target (+.24/doubling) and distractor (+.06) gain rise with pRF size | provisional |
| `figS_field_size`, `figS_gainfield_size` | Field widths; many transient fits at a bound | provisional |
| `figS_pooled_dS_dD` | dS and dD pooled over ROIs, estimated alone and jointly | provisional |
| `figS_cv_power` | Within-subject CV cannot detect a dS of our size | provisional |
| `figS_gain_fields_dn` | Fig. 5 with DN pRFs: dS estimated alone negative in every ROI, pooled −.014, p = .001 | provisional (canonical-HRF DN fits) |
| `figS_dn_gain_locus` | Where the HP effect acts in a DN pRF (input gain vs baselines); b, c pending | exploratory |
| `figS_dn_params_by_hp.png` | DN pRF parameters fitted per HP condition (PNG rendered from the PDF; **no legend yet**; stats in `notes/data/dn_by_hp_stats_2026-11.tsv`); mostly null | exploratory |
| `fig_event_responses_2x3_deconv` | Earlier HP vs LP version of Fig. 3 per item type (place undecided; superseded in the main text by Fig. 3d, e) | final apart from layout |

Not in the package (in the live folder but not listed in its README): `fig_hp_gain_heatmap`, `fig_hp_heatmap`,
`fig_hp_items`. Treat them as working figures.

## Open items the text must leave room for (`[PENDING: ...]`)

Running on the cluster (2026-11-09):
- Final DoG rerun with one HRF per subject × ROI (job 8196318): recomputes every number in Fig. 5, Table 1, the
  parameter table, the provisional supplements and, after it, Fig. 7.
- Split models: next to HP vs opposite as separate sustained terms; target gain at HP vs LP (the dT term).
- One shared width for sustained and transient fields (m7_full_sigtied).
- DN fields on the neural and surround baselines (jobs 8203960, 8206646, 8206793, recovery 8206356/8206625).
- DoG centre/surround tilt.
- Gaussian and DN robustness with jointly fitted per-ROI HRFs (ladders and decoding).
- Decoder simulation with an HP-only gain field (can the decoder see a gain of our size?).

Not yet run:
- HP-rotation null for dS, the pre-specified primary test (`notes/primary_test_2026-11.md`).
- Trial-wise gain at HP split by what HP holds (plan D4).
- Reliability of per-subject dS across sessions (would put a number on the power framing).
