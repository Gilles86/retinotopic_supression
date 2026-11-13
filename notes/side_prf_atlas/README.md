# Side analysis: pRF atlas website and negative pRFs in the default network (2026-11-13)

Separate from the paper analyses. Code: `retsupp/side_prf_atlas/`. Derivatives (cluster and `/data/ds-retsupp`):
`side_prf_atlas_signed/` (per subject), `side_prf_atlas_group/` (group). n = 28 (sub-03..sub-30).

## 1. Website

Static pycortex bundle at `notes/side_prf_atlas/site/` (index.html + data/ + one .ctm/.json/.svg; 23 MB). It needs
plain static hosting over http(s): pycortex fetches the mesh by XHR, so opening index.html via `file://` does not work.
Checked in Chrome from a local `http.server`: it loads with no console errors, the legend follows the selected map, and
it opens on an inflated posterior view. Group maps only, no individual maps. It uses its own pycortex subject,
`fsaverage_retsuppatlas` (a copy of fsaverage with a cleaned overlay), so the shared fsaverage cache was not touched.

Layers:

| Layer | Values | Mask |
|---|---|---|
| Polar angle | circular mean of model-4 angle (from projected x, y) | pRF mask, plus Rayleigh p < .05 across the subjects used |
| Eccentricity, pRF size sigma, HRF peak delay | medians of model 4 | pRF mask |
| Reliable response (% subjects) | model-free cross-run reliability prevalence | binomial vs 5%, BH q < .01 |
| Cross-validated pRF (% subjects) | signed Gaussian pRF, CV over sessions vs shifted-stimulus null | binomial vs 5%, q < .01 |
| Signed pRF: positive vs negative | (n_pos - n_neg) / (n_pos + n_neg) | CV mask |
| Negative pRFs (% subjects) | prevalence of CV negative pRFs | binomial vs 2.5%, q < .01 |
| Negative pRFs: polar angle | circular mean | negative mask, plus Rayleigh p < .05 |
| Model-4 R² (median, in-sample) | unmasked | none |

Outlines: V1, V2, V3, hV4, V3a, LO, TO1/2, IPS0-3 and FEF from the fsaverage overlay, plus the Yeo-7 default network
(Dice 0.96 against the annotation). The numerosity ROIs and embedded images of other projects were removed.

Masks. These use no R² mixture and no p_signal.
- Model-free, per subject and vertex: the correlation of each run with the mean of the other runs beats a null in which
  that mean is circularly shifted by 30-228 TRs (100 shifts, p < .05). This mask alone covers 182,745 of 327,684
  vertices. That is too permissive for pRF parameters: it admits task-locked responses that are not retinotopic, and
  model-4 eccentricities in those vertices are noise (seen in v1 of the maps).
- pRF mask (used for the parameter layers): the subject's vertex also has a cross-validated positive pRF. At group
  level this is the positive-pRF mask (binomial vs 2.5%, q < .01; 29,703 vertices). Parameters are summarised over the
  subjects passing both tests at that vertex, with at least 5 required.

Model 4 parameters available per subject: x, y, sd, amplitude, baseline, srf_amplitude, srf_size, hrf_delay,
hrf_dispersion, r2, theta, ecc. These are the current fits (May 2026; there is no later model-4 fit).
- The existing `*_space-fsaverage_*.func.gii` files are missing for sub-16.
- In those files, theta was interpolated as an angle.
- So all maps were re-projected from the volume: nearest voxel at mid-thickness, then the nearest vertex on
  `?h.sphere.reg`. Polar angle is computed from x and y.
- Coordinate convention check: the signed fit agrees with model 4 on position, r(x) = 0.85 and r(y) = 0.85.

## 2. Negative pRFs in the default network (Szinte & Knapen 2020)

**The existing fits cannot answer this.**
- The braincoder 2D model's amplitude transform is the identity, so it could in principle go negative.
- But `fit_prf.grid_fit` uses `fit_grid(..., use_correlation_cost=True)` with the default `positive_amplitude=True`,
  which keeps the most POSITIVELY correlated pRF.
- `refine_baseline_and_amplitude` then clips the amplitude to >= 1e-4.
- Only 1-2% of model-4 voxels drift below 0 during GD.

**New fit (`fit_signed_prf.py`)**:
- Gaussian pRF grid on the bar-only stimulus, canonical HRF, picking the largest |r|; 6,920 pRFs, sigma 0.25-6 deg.
- Fitted on the session-1 mean and tested on the session-2 mean, and the reverse. The bar sequence is identical across
  runs (mean absolute frame difference <= 0.0003).
- Null: 50 circularly shifted stimuli.
- A vertex counts as negative when it beats the null (p < .05, cv r > 0) and the amplitude is negative in the
  all-runs fit and in both session fits.
- Chance is about 2% (tested against 2.5%).

Per-subject fractions of vertices (mean over 28 subjects; t tests over subjects).
`notes/side_prf_atlas/data/negative_prf_summary.tsv` has everything.

| Region (Yeo-7 × aparc) | CV pRF | negative | positive | neg. laterality (−1 = contra) |
|---|---|---|---|---|
| DMN, all | 6.3% | 3.2% (p = .007) | 2.2% | −0.12 (p = .002) |
| DMN angular / inferior parietal | 9.0% | 4.8% (p < .001) | 3.2% | −0.32 (p < .001) |
| DMN lateral temporal | 6.7% | 3.6% (p = .001) | 2.0% | −0.07 (n.s.) |
| DMN medial parietal (PCC/isthmus/precuneus) | 6.0% | 3.3% (p = .04) | 1.8% | −0.14 (p = .008) |
| DMN medial prefrontal | 5.4% | 2.4% (n.s.) | 2.1% | −0.02 (n.s.) |
| Early visual (pericalcarine/cuneus/lingual) | 26% | 13% | 12% | −0.72 |
| Visual network (Yeo 1) | 35% | 8.7% | 24% | −0.60 |

**Answer: weakly, mostly in angular gyrus / inferior parietal, and much more weakly than Szinte & Knapen.**
- Within the DMN, sign-consistent cross-validated pRFs are more often negative than positive: 59% negative, versus
  51-78% in S&K.
- Negative pRFs in angular/inferior parietal and medial parietal cortex are biased toward the contralateral hemifield:
  laterality −0.32 and −0.14, versus S&K's ANG −0.14 to −0.54 and MED_PAR −0.02 to −0.40.
- Medial prefrontal cortex shows nothing.
- But the effect is a few percent of vertices above a ~2% chance level.
- At group level only 179 of 61,370 DMN vertices pass the negative-pRF mask, almost all in the right inferior
  parietal / banks of the STS. In the visual network, 4,480 of 44,151 vertices pass.
- The strongest negative pRFs here are in early visual cortex. They lie in the peripheral (anterior) parts of
  V1-V3, are clearly retinotopic (contralateral, smooth polar angle in the "Negative pRFs: polar angle" layer) and
  are the classic negative BOLD beyond the stimulated aperture.

Retinotopic organisation of DMN negative pRFs:
- The contralateral bias (above) is the cleanest evidence. Cross-validation does not select for it.
- They beat the non-spatial "bar on" model in 72% of vertices (cv r 0.17 vs 0.12). This comparison is partly circular,
  because vertices were selected on the pRF's cv r.
- Session-1 and session-2 polar angles agree (mean cos 0.55). This is circular too, because CV selects vertices whose
  sessions agree; positive DMN pRFs agree as well (0.62).
- Eccentricity and size correlate positively (Spearman about 0.25-0.30). This is partly built into the grid: small pRFs
  centred outside the 3.17 deg aperture predict nothing and were dropped.
- Group polar-angle maps of DMN negative pRFs are not coherent: Rayleigh p < .05 in only about 10% of the 179 vertices.

Differences from S&K:
- Our aperture is 3.17 deg against their about 8 deg.
- The bar runs concurrently with a visual search task, which itself drives DMN deactivation that is not locked to
  the bar.
- About 1.3 h of data per subject, with n = 28 on fsaverage. Their individual datasets were densely sampled, and
  their HCP average used 181 subjects.
- Their fits had no cross-validation and no null.

## Jobs and cost

- Canary 8332747: 8,000 voxels, 113 s, peak 8.5 GB. Numpy used only about 26% of 8 CPUs, so the full run used 4 CPUs.
- Array 8332798 (sub-03..30, standard partition, 4 CPUs, 16 GB): 21-45 min per subject, 58 CPU-hours in total.
- Group jobs 8332825 (v1 maps) and 8334413 (v2 maps, used by the site): under 1 min each.

## Rebuild

```bash
sbatch --array=3-30 retsupp/side_prf_atlas/slurm_jobs/fit_signed_prf.sh        # cluster
sbatch retsupp/side_prf_atlas/slurm_jobs/group_fsaverage.sh
rsync -a sciencecluster:/shares/zne.uzh/gdehol/ds-retsupp/derivatives/side_prf_atlas_group/ /data/ds-retsupp/derivatives/side_prf_atlas_group/
~/mambaforge/envs/retsupp/bin/python retsupp/side_prf_atlas/summarize_negative_prfs.py \
    --region-stats /data/ds-retsupp/derivatives/side_prf_atlas_group/region_stats.tsv
~/mambaforge/envs/pycortex2/bin/python retsupp/side_prf_atlas/build_site.py \
    --group-npz /data/ds-retsupp/derivatives/side_prf_atlas_group/group_maps_fsaverage_v2.npz \
    --annot-dir <dir with ?h.Yeo2011_7Networks_N1000.annot from fsaverage/label> --out notes/side_prf_atlas/site
```
