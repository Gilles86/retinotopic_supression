# Side project: do hippocampal / MTL voxels have pRFs?

Branch `side/hippocampus-2026-11`, separate from the paper analyses. Derivatives only in
`derivatives/side_hpc_*`; code in `retsupp/side_hippocampus/`.

**Verdict: no pRF signal in hippocampus, entorhinal, parahippocampal cortex or amygdala.** Held-out pRF predictions
are no better than a spatially scrambled stimulus in any participant. The fraction of voxels beating the scramble
null is at chance (5 %), and those voxels show no contralateral bias. The positive control (lateral occipital
cortex) passes every test by a wide margin, so the method can find pRFs. Step 1 is clearly null, so step 2 (gain
fields, HP − LP sustained gain) was not run: without a pRF there is nothing for a gain field to modulate. Any null
HP gain there would be uninterpretable rather than exciting.

## Question

Statistical learning is often placed in the hippocampus / MTL. Do MTL voxels have pRFs in this paradigm (bar sweep +
search array), and if so, do they carry an HP-specific gain field?

## Data and regions

n = 28 participants (sub-03 … sub-30), 2 sessions × 5–6 runs, cleaned BOLD (258 volumes). Regions from FreeSurfer
`aparc+aseg`, resampled (nearest neighbour) to BOLD space, inside the BOLD mask: hippocampus (17/53), entorhinal
(1006/2006), parahippocampal (1016/2016), amygdala (18/54). Controls: lateral occipital (1011/2011, positive) and
precentral (1024/2024, negative). Median voxels per participant: hippocampus 1371, parahippocampal 710, amygdala 537,
entorhinal 461. The controls were subsampled at random to 1500 voxels.

## Methods (step 1)

- **Cross-validated pRFs** (`fit_prf_cv_region.py`). These are the paper's folds: fold k holds out the k-th run of
  every HP condition, so there are 3 folds of ~8 training and ~4 test runs. On the training runs only:
  1. a Gaussian pRF, canonical HRF (grid + GD);
  2. a DoG pRF seeded from 1;
  3. a DoG pRF with one HRF per region, fitted jointly with the pRFs (`fit_roi` / `flex_factory` of
     `modeling/cv/fit_prf_cv_roi_hrf_joint.py`, unchanged).

  The stimulus is the paper's bar + search-item rectangles, resolution 50, radius 5°, with sd_min = 0.2.
  Held-out R² (cvR²) per voxel = 1 − ΣSSE/ΣSST over the three test sets. The baseline is the test-run mean, so
  cvR² > 0 means the prediction is better than a flat line.
- **Null: spatial scramble.** The same fits, on a stimulus whose pixels are permuted by one fixed random permutation
  per participant. Frame by frame this keeps how much stimulus is on screen (bar and search-array onsets). Any
  response locked to the task or to "stimulus present" can still be fitted, but where the stimulus is gets lost. So
  cvR² above scramble requires visual-field-specific (retinotopic) structure. This addresses the concern from the
  quick check: the repeating, task-locked response in hippocampus that precentral cortex shows even more strongly.
- **Per participant × region:**
  - Δ = median cvR² (true) − median cvR² (scramble).
  - "pRF voxels": true cvR² above the 95th percentile of that participant's and region's scramble cvR². Expected
    fraction 5 % without retinotopic signal.
  - Among pRF voxels, the fraction with the centre in the contralateral hemifield.
- **Group tests:** n = 28, Wilcoxon signed-rank, exact, one-sided in the direction a pRF predicts. Benjamini-Hochberg
  across the four MTL regions. The controls are reported uncorrected.
- Numbers below are for the DoG with region HRF. The Gaussian and the DoG with canonical HRF give the same picture
  (`data/prf_cv_summary.tsv`, column `model`).

## Results

![pRF evidence](fig_prf_evidence_dog_roihrf.png)

`fig_prf_evidence_dog_roihrf.pdf`, 180 mm. **a**, Held-out R², true − scramble. **b**, Fraction of voxels above the
scramble 95th percentile; the dashed line is the 5 % expected by chance. **c**, Contralateral fraction among those
voxels. Dots: participants. Diamonds: mean ± SEM. \*: p < .05, one-sided (BH across the four MTL regions).

| Region | cvR² true | cvR² scramble | Δ (true − scramble) | Participants Δ > 0 | p (BH) | Fraction > scramble 95th pct | p (BH) | Contralateral | p (BH) |
|---|---|---|---|---|---|---|---|---|---|
| Hippocampus | −0.004 | −0.001 | −0.0026 ± 0.0001 | 0 / 28 | 1.0 | 0.039 ± 0.002 | 1.0 | 0.51 ± 0.02 | .41 |
| Entorhinal | −0.003 | −0.001 | −0.0024 ± 0.0001 | 0 / 28 | 1.0 | 0.042 ± 0.002 | 1.0 | 0.49 ± 0.03 | .65 |
| Parahippocampal | −0.003 | −0.001 | −0.0026 ± 0.0001 | 0 / 28 | 1.0 | 0.047 ± 0.003 | 1.0 | 0.50 ± 0.03 | .41 |
| Amygdala | −0.004 | −0.001 | −0.0025 ± 0.0001 | 0 / 28 | 1.0 | 0.035 ± 0.002 | 1.0 | 0.53 ± 0.03 | .41 |
| Precentral (neg.) | −0.003 | −0.000 | −0.0021 ± 0.0002 | 1 / 28 | 1.0 (unc.) | 0.108 ± 0.009 | 3.7 × 10⁻⁸ (unc.) | 0.53 ± 0.02 | .038 (unc.) |
| Lateral occipital (pos.) | 0.036 | 0.019 | +0.017 ± 0.002 | 25 / 28 | 9.3 × 10⁻⁸ (unc.) | 0.183 ± 0.008 | 3.7 × 10⁻⁹ (unc.) | 0.97 ± 0.004 | 3.7 × 10⁻⁹ (unc.) |

cvR² columns: mean over participants of the median over voxels. ± is SEM over participants. Full table with exact p:
`data/prf_cv_tests_dog_roihrf.tsv`.

Reading:

- **MTL: no pRFs.** Median held-out R² is below zero in every region and below the scramble fits in all 28
  participants. The true-stimulus pRF has more freedom to fit noise in the training runs, and it pays for that on
  the test runs. Only 4–5 % of voxels beat the scramble 95th percentile, which is chance or slightly below. Those
  voxels sit at 50 % contralateral, as they would if their centres were noise.
- **Lateral occipital**: about twice the scramble cvR². 18 % of voxels beat the null and 97 % of those are
  contralateral. Fold-to-fold pRF positions agree at r ≈ 0.99. The pipeline detects real pRFs.
- **Precentral** has 11 % of voxels above the scramble null, with a weak contralateral tendency (53 %, p = .038
  uncorrected). That fits a small retinotopic patch, plausibly the frontal eye field in the precentral sulcus. The
  MTL regions show nothing similar.
- Fold-to-fold agreement of pRF x is above the scramble in every region, including the negative control
  (`d_xagree` in the test table). The folds share half their training runs, and the bar stimulus constrains noise
  fits differently from the scrambled one, so this measure does not tell signal from noise here. It is left out of
  the figure.
- This agrees with the earlier quick check. MTL has a tiny repeating response, but precentral shows more of it, and
  whole-brain pRF centres show no contralateral bias. The response is task-locked, not tied to the visual field.

## Step 2 (gain fields): not run

The plan gated step 2 on a pRF signal in step 1. The gain-field model (pRFs + T/D/S gain fields, m4_TDS / m7_full)
only says something where a voxel's response depends on stimulus position. In the MTL it would fit noise, and a null
dS would carry no information. A pRF-free HP test (response at the HP vs the LP locations) is a different question:
it asks whether MTL responds to HP *events*, not whether it has visual-field maps. It was not run here. If wanted, it
can reuse the item-location time-series pipeline (`visualize/paper/export_location_timeseries.py`) with these
regions as masks.

## Reproduce

```bash
# cluster, clone ~/retsupp_side_hpc on this branch
sbatch --array=3-30 retsupp/side_hippocampus/slurm_jobs/extract_region_ts.sh        # -> derivatives/side_hpc_region_ts
for R in hippocampus entorhinal parahippocampal amygdala lateral_occipital precentral; do for V in true scramble; do
  REGION=$R VARIANT=$V sbatch --array=3-30 retsupp/side_hippocampus/slurm_jobs/fit_prf_cv_region.sh; done; done
srun ... python -m retsupp.side_hippocampus.collect_prf_cv --voxels-out ~/side_hpc/prf_cv_voxels.tsv.gz \
    --summary-out ~/side_hpc/prf_cv_summary.tsv                                        # compute node
# local
python -m retsupp.side_hippocampus.plot_prf_cv --summary notes/side_hippocampus/data/prf_cv_summary.tsv \
    --out-dir notes/side_hippocampus
```

Resources: extraction 2 CPUs, 24 GB, ~2 min per participant. pRF fits 8 CPUs, 16 GB, 25–90 min per participant ×
region × variant; two large-region tasks needed 24 GB / 5 h. In total about 2,500 vCPU-hours, roughly 5 CHF.
