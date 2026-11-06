# Open technical issues: causes (2026-11-06)

Investigation only; nothing in the pipeline was changed. Probe scripts (session scratchpad, run on
compute nodes via `srun`): `items_probe.py`, `items_ref.py`, `ds_probe.py`. Data: the exact LOSO
curves in `derivatives/af_fixed_group_exact_{dog,dn}/` (what `notes/data/loso_per_step_2026-11.tsv`,
Table 1 and Fig. 3e are built from) and the ladder pickles in `af_prf_cv_ladder_nested_full_bounded*`.

## a) `item_scale` under DN: LOSO δ at 0 vs fitted 1.6

**Cause: two things, neither a bug in the fits.**

1. **Different base model.** The LOSO "items" step (`fit_fixed_group_mechanisms.py`, mechanism
   `items`) perturbs `item_scale` on the **m00_bar** fit, whose per-voxel pRFs were fitted to the bar
   only. The bar covers 3.17°; the items sit at 4°, so those pRFs are unconstrained exactly where the
   items are. The fitted `item_scale` in m0/m7 comes from **refitting every voxel's pRF together with
   the items**. So the two numbers answer different questions: "how much item drive suits pRFs that
   never saw items" vs "how much item drive, once the pRFs are refitted".
2. **Reference point of the score.** The step starts at `item_scale` = 1 (items at bar strength,
   δ = 0) and scores ΔR² relative to that, with δ on a grid from −1 (no items) to +.5. The best δ is
   negative: the bar-only pRFs predict worse with items at bar strength than with weaker items.
   So "Items: 100% improved, ΔR² 329e-5 (DoG) / 2191e-5 (DN)" means **"turning the items down from
   bar strength helps"**, not "adding items helps".

**Evidence (30 subjects × 11 ROIs).**

| | DoG | DN |
|---|---|---|
| Own best δ on the m00 base (median) | −.81 (item_scale .19); 39% at the grid edge (0) | −1.00 (item_scale 0); 97% at the edge |
| LOSO item_scale (median δ from other subjects) | .16 | 0 |
| Fitted item_scale, m0_prf (median [10th, 90th pct]) | 6.8 [2.0, 23.8] | 1.6 [.29, 10.4] |
| Fitted item_scale, m7_full | 6.6 [2.1, 22.4] | 1.6 [.28, 10.5] |
| LOSO items as reported (vs item_scale 1) | 100%, p < .001, ΔR² +329e-5 | 100%, p < .001, +2191e-5 |
| LOSO items **vs no items** (item_scale 0) | **80%, p = .0013, ΔR² +11.3e-5** | **3%, p < .001, −0.01e-5** |
| Items at bar strength vs no items | 0% improved, −290e-5 | 0%, −2191e-5 |

**Consequence for the draft.** "Items improve out-of-subject prediction in 100% of subjects" (Table 1,
Fig. 3e, writing package, results overview) is not supported as stated. Against no items, on bar-only
pRFs, items help in 80% of subjects under DoG (much smaller ΔR²) and not at all under DN.

**Proposed fix.** Test items where they are fitted: base = m0_prf pRFs with `item_scale` set to 0,
perturb towards the other subjects' fitted m0 `item_scale`, score against item_scale 0. Cheaper
interim: rescore the existing curves against δ = −1 (done above for the numbers; the DN grid stops at
item_scale 1.5, below many fitted DN values, so the m0-based test is the proper one). Confidence: high
for the cause, high for the rescored numbers.

## b) dS generalises in LOSO under DoG but not DN

**Cause: under DN the data constrain dS about four times less, so per-subject estimates are twice
as variable, and the out-of-subject δ often lands on the wrong side of a subject's own optimum.**
The parameter test (sign of each subject's own estimate) is unaffected by that scatter; LOSO is not.

**Evidence (exact curves, nine ROIs, signal voxels).**

| | DoG | DN |
|---|---|---|
| LOSO dS | 77%, p = .010 | 63%, p = .23 |
| Own dS < 0 (all AF voxels) | 73% of cells | 73% of cells |
| Median LOSO δ | −.008 | −.012 (1.7× larger) |
| SD of own dS across subjects (median over ROIs) | .032 | .066 (2.1×) |
| Curvature of ΔSSE in δ (ΔR² ×10⁻⁵ per δ², median) | 4,060 | 680 (DN/DoG per cell .23) |
| Best achievable ΔR² per cell (median, ×10⁻⁵) | .98 | .84 |
| Cell-wise ΔR², DoG vs DN | Spearman ρ = .56 | |

Per ROI (LOSO % improved, DoG vs DN): IPS 83 vs 28, V3AB 76 vs 68, TO 77 vs 69, VO 69 vs 54, hV4 62
vs 66, V3 63 vs 57, LO 55 vs 59, V1 53 vs 53, V2 53 vs 50. Most of the loss is IPS, where DN's curve is
flat and its LOSO δ (−.007) is small relative to the scatter.

Why the curve is flatter: in the DN model the gain multiplies the drive in the numerator and also
feeds the normalisation pool (`retsupp/modeling/dn_attention_field.py`), so part of a gain change is
divided back out. A given δ changes the prediction about 4× less (curvature .23×), the optimum is
less sharply defined, and own estimates scatter more. The best a subject could gain is similar (.84 vs
.98), so the effect is not absent under DN; it is less identifiable per subject.

**Proposed fix.** None needed in the pipeline. Report: same direction and same share of negative
subjects under both models; LOSO, which needs subjects to agree on the size, fails under DN because DN
estimates are noisier. Optional check: LOSO with δ from a robust centre (trimmed mean) or excluding
IPS. Confidence: moderate (the mechanism fits all numbers; the normalisation explanation for the
flatter curve is inferred from the model structure, not tested by simulation).

## c) Flat V1 pRF size vs eccentricity (Fig. 3c)

**Cause (most likely): V1 pRFs sit near the spatial resolution of this paradigm; the measured σ
(≈ .5–.6°) is a floor, not the true size. Not specific to a model or to the voxel selection.**

**Evidence.**

- The table (`notes/data/fig1_prf_sigma.tsv`, `plot_fig1.py collect`: model 4 = DoG, flexible HRF,
  r² ≥ .1, ≥ 50% mass inside the aperture, σ in [.3, 4]) gives V1 medians .58, .58, .56, .58, .60, .57,
  .65° from .25° to 3.25° eccentricity, while V2 rises .60 → .75 and V3 .61 → .91.
- Flat in every pRF model (five top-tier subjects, V1, r² > .1, median σ per .5° bin from 0 to 3.5°):
  Gaussian .49–.55; DoG .52–.59; Gaussian + HRF .46–.52; DoG + HRF .53–.61; DN rises slightly,
  .40 → .55. So not the DoG surround and not the flexible HRF.
- Not the selection: without the σ and aperture filters the medians are the same; only 1–5% of V1
  voxels are below .35° (DN 14%), minimum .23°, so nothing piles up at the .3° filter or a model floor.
- Paradigm limits: bar width .79°; the bar sweeps 7.1° in 42.4 s (.17°/s, .27° per TR), so the
  response to a V1 pRF is set mostly by bar width and HRF, and a 1 s error in the HRF's width moves σ by
  ~.15–.2°. Voxels are 1.7 mm; fixation jitter of a few tenths of a degree adds in quadrature
  (σ_obs² = σ² + σ_jitter²), which flattens small σ to a constant. Expected V1 σ at 0.5–3° is about
  .2–.5°; ours is at or above the top of that range everywhere.

**Proposed fix / test.** Do not change the fits. (1) Simulate: plant V1-like pRFs (σ .2–.6° rising
with eccentricity) in this exact paradigm with the canonical HRF and realistic noise, fit them with
model 4, and see whether recovered σ flattens at ~.5°. (2) If eye-tracking data exist (EyeLink, see
`experiment/trial.py` drift correction), estimate fixation dispersion per run. Report V1 σ as limited
by the paradigm in the legend/supplement. Confidence: moderate (all evidence is consistent; the
simulation would settle it).
