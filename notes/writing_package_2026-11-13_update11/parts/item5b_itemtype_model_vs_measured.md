# Item 5b: item-type contrasts of Fig. 3b, c, model vs measured, per pathway and for V1, V2, V3 (n = 28)

**Question.** The joint fit gives no target gain in V1 and V2 (Fig. 5b: V1 −2.3%, V2 +5.2%, n.s.). Is that a property
of the data, or did the data lack the sensitivity to show one?

**Short answer.** It is not a lack of sensitivity, and the data do not lack a target response there. At the item
locations the target evokes more response than a neutral item in V1 and in V2:
- V1: +0.071 % signal [+0.034, +0.098], 20/28, p < .001;
- V2: +0.146 [+0.093, +0.175], 27/28, p < .001.

The smallest effect the data could detect (minimum detectable effect, MDE) is 0.039 % signal in V1 and 0.053 in V2.
The joint fit predicts no target − neutral difference in V1 (−0.004) and almost none in V2 (+0.007), and the measured
values lie well outside the range the model allows with this noise.

So "target gain absent in V1 and V2" describes the fitted model, not the data. It should not be written as "the target
gain is absent in V1 and V2".

This is part of a general pattern. The joint fit under-predicts target − neutral and distractor − neutral at the item
locations in every pathway and early area, by 0.05–0.10 % signal (p ≤ .003 everywhere). It does predict
target − distractor correctly everywhere (residual p ≥ .11). In V1 target − distractor is 0 in both data and model
(−0.028, p = .51).

So the model captures the difference between target and distractor but misses a response, larger than to a neutral
item, that the two share. That miss is largest in V1 and V2, where the predicted item responses are small. The
predicted neutral-item response is also about half the measured one there: V1 0.14 vs 0.29 % signal, V2 0.25 vs 0.35.

## Data and method

**Model and pipeline.** Noise-free predictions of the joint gain-field fit m7_full as fitted (DoG pRF; item weight,
target, distractor and sustained gains, with dS and dD). The pRFs were refitted together with the gains
(`..._reliable_roihrf_polished`; the joint polish is **not fully converged**, and a convergence run is in progress
elsewhere). Measured and predicted data go through the same Fig. 3 refit pipeline as in item 2:
- the same voxels;
- the bar-alone prediction removed;
- nideconv with the six HP / LP event types of Fig. 3b, c (`deconvolve_location_responses.py` without `--relative`);
- the window 3.2–6.4 s, pre-onset subtracted.

**Contrasts (% signal), as in Fig. 3b, c:**
- target − neutral: both averaged over HP and LP;
- distractor − neutral: both at LP;
- target − distractor: both at LP.

**Statistics.**
- Measured: median [bootstrapped 95% CI], k of n, Wilcoxon p, and BH q across the three pathways or across V1–V3.
- **Model band:** 5000 simulated experiments. In each, a participant's value is their predicted contrast plus a residual
  (measured − predicted, centred on 0) drawn with replacement across participants. The band is the 95% interval of the
  simulated group median, i.e. where the measured median should fall if the model were right and the noise were as
  measured.
- **Power:** the share of simulated experiments with Wilcoxon p < .05 in the model's direction.
- **MDE:** 2.8 × the measured between-participant SE (80% power, two-sided α = .05).

We used between-participant residuals rather than phase-randomised residual time series. They include all the
measurement noise of the window means at no extra cost.

## Table 5b-1. Measured vs joint-fit prediction (% signal)

| Group | Contrast | Measured | Joint fit predicted | Model band (95%) | Measured inside? | Power of the model's prediction | MDE | Measured − joint fit |
|---|---|---|---|---|---|---|---|---|
| Early | target − neutral | +0.120 [+0.102, +0.151], 27/28 > 0, p < .001 | +0.043 [+0.004, +0.056], 21/28 | [−0.002, +0.078] | no | 46% | 0.036 | +0.074 [+0.039, +0.133], p < .001 |
| Early | distractor − neutral | +0.058 [+0.032, +0.115], 24/28, p < .001 | +0.008 [−0.021, +0.036], 14/28 | [−0.025, +0.038] | no | 2% | 0.047 | +0.060 [+0.039, +0.085], p < .001 |
| Early | target − distractor | +0.038 [+0.001, +0.101], 20/28, p = .012 | +0.031 [+0.012, +0.052], 22/28 | [+0.002, +0.086] | yes | 33% | 0.049 | +0.020 [−0.011, +0.060], p = .20 |
| Dorsal | target − neutral | +0.246 [+0.183, +0.314], 26/26, p < .001 | +0.173 [+0.105, +0.238], 23/26 | [+0.103, +0.233] | no | 100% | 0.110 | +0.064 [+0.036, +0.138], p = .001 |
| Dorsal | distractor − neutral | +0.132 [+0.063, +0.210], 23/26, p < .001 | +0.060 [+0.015, +0.096], 22/26 | [−0.022, +0.070] | no | 9% | 0.127 | +0.049 [+0.024, +0.095], p < .001 |
| Dorsal | target − distractor | +0.128 [+0.050, +0.184], 20/26, p < .001 | +0.105 [+0.049, +0.186], 22/26 | [+0.073, +0.196] | yes | 93% | 0.081 | +0.024 [−0.023, +0.061], p = .34 |
| Occ.-temporal | target − neutral | +0.218 [+0.176, +0.269], 28/28, p < .001 | +0.168 [+0.134, +0.209], 25/28 | [+0.106, +0.199] | no | 100% | 0.058 | +0.050 [+0.014, +0.084], p = .0029 |
| Occ.-temporal | distractor − neutral | +0.153 [+0.082, +0.196], 25/28, p < .001 | +0.069 [+0.019, +0.106], 20/28 | [−0.001, +0.102] | no | 38% | 0.077 | +0.069 [+0.031, +0.118], p < .001 |
| Occ.-temporal | target − distractor | +0.105 [+0.043, +0.163], 22/28, p < .001 | +0.091 [+0.057, +0.138], 26/28 | [+0.059, +0.168] | yes | 97% | 0.100 | +0.020 [−0.040, +0.053], p = .55 |
| **V1** | **target − neutral** | **+0.071 [+0.034, +0.098], 20/28, p < .001, q < .001** | **−0.004 [−0.013, +0.000], 11/28** | **[−0.037, +0.036]** | **no** | 1% | **0.039** | **+0.085 [+0.038, +0.110], p < .001, q < .001** |
| V1 | distractor − neutral | +0.041 [+0.018, +0.089], 22/28, p < .001, q = .0014 | −0.001 [−0.025, +0.010], 14/28 | [−0.050, +0.031] | no | 2% | 0.046 | +0.050 [+0.014, +0.088], p = .0027, q = .0027 |
| V1 | target − distractor | −0.028 [−0.050, +0.030], 12/28 > 0, p = .51, q = .51 | −0.002 [−0.020, +0.015], 13/28 | [−0.045, +0.065] | yes | 0% | 0.054 | −0.010 [−0.060, +0.054], p = .75 |
| **V2** | **target − neutral** | **+0.146 [+0.093, +0.175], 27/28, p < .001, q < .001** | **+0.007 [−0.001, +0.060], 19/28** | **[−0.027, +0.070]** | **no** | 17% | **0.053** | **+0.100 [+0.051, +0.152], p < .001, q < .001** |
| V2 | distractor − neutral | +0.068 [+0.018, +0.143], 20/28, p = .0011, q = .0014 | +0.002 [−0.020, +0.026], 15/28 | [−0.033, +0.050] | no | 3% | 0.057 | +0.061 [+0.023, +0.096], p < .001, q = .0011 |
| V2 | target − distractor | +0.056 [+0.001, +0.122], 20/28, p = .023, q = .035 | +0.028 [−0.008, +0.047], 16/28 | [−0.010, +0.088] | yes | 20% | 0.075 | +0.042 [−0.005, +0.096], p = .11 |
| V3 | target − neutral | +0.195 [+0.139, +0.226], 26/28, p < .001, q < .001 | +0.048 [+0.026, +0.155], 22/28 | [+0.040, +0.142] | no | 97% | 0.050 | +0.087 [+0.032, +0.134], p < .001 |
| V3 | distractor − neutral | +0.066 [+0.044, +0.158], 21/28, p = .0014, q = .0014 | −0.001 [−0.020, +0.048], 13/28 | [−0.039, +0.054] | no | 0% | 0.077 | +0.069 [+0.031, +0.109], p = .002 |
| V3 | target − distractor | +0.143 [+0.040, +0.195], 21/28, p = .0016, q = .0047 | +0.066 [+0.049, +0.100], 22/28 | [+0.024, +0.138] | no (above) | 81% | 0.086 | +0.046 [−0.025, +0.094], p = .14 |

Measured q: BH across the three pathways, or across V1–V3, per contrast. Every measured q is below .05 except V1
target − distractor.

**In plain words:**
- The data detect a target response above a neutral item in V1 and V2, at twice the MDE in V1 and almost three times
  in V2.
- The joint fit predicts none there, so its "no target gain in V1 and V2" is not forced by the data.
- In every area the model is too small for target − neutral and distractor − neutral, and about right for
  target − distractor.
- The detection power of the model's own predictions shows what the model would let us see. In V1 and V2 it predicts
  target − neutral too small to detect (1% and 17%), yet the data show a clear effect.

**Relative to the neutral-item response** (median over participants, items at LP):
- target is above neutral by 10% in V1 (model −3%), 34% in V2 (model 3%) and 39% in V3 (model 33%);
- early 28% (model 16%), dorsal 47% (40%), occipitotemporal 40% (41%).

So the shortfall is concentrated in V1 and V2. The predicted neutral-item response is about half the measured one in
V1 (0.14 vs 0.29 % signal) and two thirds in V2 (0.25 vs 0.35).

## Table 5b-2. Voxels on the item locations per area (median over the 28 participants)

Counts are per item location (the mean over the four locations within a participant), with the sparsest location in
brackets. All counts are voxels reliable across runs (r > 0) in the refitted fit.
- **Fig. 3, within 2°:** pRF centre within 2° of the location, each voxel assigned to its nearest location (the Fig. 3
  selection).
- **pRF overlaps the item:** the distance from the pRF centre to the item's edge is below half the FWHM of the pRF
  centre (2.355·σ/2), with item radius 0.75° (size_stimuli / 2). A voxel can overlap more than one item.
- **Both:** overlap and within 2°.

| Area | Fig. 3, within 2° | pRF overlaps the item | Both | Median FWHM of the Fig. 3 voxels | Participants with an empty location (Fig. 3) | Reliable voxels in the area |
|---|---|---|---|---|---|---|
| V1 | 25.4 (6.5) | 18.5 (7) | 17.5 | 1.9° | 2 | 395 |
| V2 | 25.3 (12.5) | 23.3 (11.5) | 21.3 | 2.4° | 2 | 379 |
| V3 | 15.1 (7) | 18.4 (8) | 14.3 | 3.1° | 1 | 364 |
| V3AB | 6.5 (1) | 10.3 (2) | 6.5 | 4.2° | 12 | 90 |
| IPS | 0.25 (0) | 1.3 (0) | 0 | 6.6° | 24 | 29 |
| hV4 | 7.1 (0) | 9.6 (1) | 7.0 | 4.0° | 16 | 115 |
| VO | 1.3 (0) | 2.3 (0) | 1.3 | 6.2° | 21 | 16 |
| LO | 3.0 (0) | 7.8 (1) | 2.9 | 4.3° | 17 | 216 |
| TO | 1.8 (0) | 10.9 (4) | 1.8 | 8.9° | 21 | 80 |

**In plain words:** V1 and V2 have the most item-location voxels of any area (about 25 per location under the Fig. 3
rule, 18–23 whose pRF overlaps the item). So the V1 and V2 item contrasts are the best measured, not the worst.

The overlap rule keeps fewer V1 voxels than the 2° rule, because V1 pRFs are small (FWHM 1.9°, so the reach is
0.75° + 0.95° ≈ 1.7° < 2°). It keeps more in the later areas, whose pRFs are large (TO 8.9°). There most overlapping
voxels have their centre more than 2° from the item.

## Sources

- Code:
  - `retsupp/visualize/paper/predict_location_timeseries_polished.py` (predictions and `sub-XX_overlap.tsv` counts);
  - `retsupp/visualize/paper/slurm_jobs/predict_location_timeseries_polished.sh`;
  - `retsupp/visualize/paper/items_model_vs_measured.py --part items` (local).
- Tables:
  - `notes/data/items_type_model_vs_measured_n28_2026-11-13.tsv` (rows measured / joint fit predicted / measured − joint
    fit, with the model band, power, MDE, q);
  - `notes/data/items_voxel_overlap_n28_2026-11-13.tsv`; per participant
    `notes/data/polished_pred_overlap_2026-11-13/sub-XX_overlap.tsv`.
- Deconvolutions:
  - measured `notes/data/event_deconv_hplp_polished_2026-11-13/`;
  - predicted `notes/data/event_deconv_polished_pred_2026-11-13_hplp_SD/` (also nohp, S, SDT, SDT0);
  - cluster `derivatives/paper_2026-11/`.
- Fitted joint-fit target gains (median, fraction): V1 −0.023, V2 +0.052, V3 +0.333
  (`polished_pred_overlap_2026-11-13/sub-XX_terms.tsv`; key_numbers update 8: V1 −2.3%, V2 +5.2%).
