# VSS 2026 — figure plan

Talk: **History-Driven Suppression Reshapes Visuospatial Tuning Across the Visual Hierarchy**
Authors: Dock H. Duncan¹, Gilles de Hollander², Ningkai Wang¹, David Richter³, Tomas Knapen¹, Jan Theeuwes¹,³,⁴

## What the slide deck already commits to

Reading the current draft (`VSS2026.pdf`):

- Slide 1 — title.
- Slide 2 — "What is attention again? (in vision)" — schematic of visual pathway (retina → LGN → V1 → ventral/dorsal streams) + a PRF tiling diagram across V1–V4. *Borrowed.*
- Slide 3 — "Salience" — concept slide, photo collage. *Borrowed.*
- Slide 4 — "Statistically Learned Suppression" — Theeuwes integrated-priority-map cartoon (bottom-up + history + top-down). *Borrowed.*
- Slide 5 — Klein, Harvey & Dumoulin 2014 paper screenshot, motivating PRF mapping as a tool.
- Slide 6 — paradigm: PRF bar + concurrent additional-singleton search at 8 locations. 7T Philips, 28 participants, 2 sessions each.
- Slide 7 — methods: blocked HP-distractor design, 4 HP locations, targets balanced across all 8 positions so target processing cannot drive any HP-specific shift. **Open question:** do PRFs shift toward the HP location (bottom-up gain) or away (suppression / receptivity reduction)?
- **Slides 8–11 are placeholders:** "PRF map on 3D + flat brain to show fitting worked" → "Shifting receptive fields across conditions, could do arrow plots" → "Shift by processing area, show shift is greatest in later areas" → "Interpreting the shift / What does a receptive field shift mean?"
- Slide 12 — closing.

So the figures to design fit four placeholder slots, and one likely interpretation slide. They need to land on three claims, in order: (1) the PRF mapping worked, (2) PRFs shift as a function of HP-distractor location, (3) the shift grows up the hierarchy.

## Proposed figures

### Figure A — sanity check that PRF mapping worked (slide 8)

**Goal:** convince the audience the data are good before we ask anything subtle.

- A1. **R² flatmap of one good subject**, mean-fit (model 4 = DoG + flexible HRF), thresholded at e.g. R² > 0.05. Use `visualize/flatten_r2_images.py` + pycortex. One hemisphere is enough.
- A2. **Polar-angle and eccentricity flatmaps** for the same subject, masked by the same R². Side-by-side strengthens the "this is real retinotopy."
- A3. (Optional, smaller inset) **Mean R² per ROI across subjects**, boxplot or stripplot, V1 → V2 → V3 → V3AB → hV4 → LO → TO → VO. This sets up the hierarchy ordering used in Figure C.

Source notebook: `notebooks/check_inferred_pars.ipynb` and `notebooks/analyze_retinotopy_data.ipynb` already have most of the building blocks; `Subject.get_prf_parameters_surface(model=4)` gives the surface arrays.

### Figure B — PRFs shift across HP-distractor conditions (slides 9–10)

**Goal:** show the central effect, both per-voxel and aggregated.

- B1. **Arrow / quiver plot in visual-field coordinates.** For one ROI (V1 first, then V4 / LO / TO in supp), each arrow goes from the voxel's mean-condition PRF center `(x_mean, y_mean)` to the condition-specific PRF center `(x, y)`. Color = HP-distractor condition (4 colors, matching the slide-7 schematic). Restrict to voxels with mean-model R² > 0.2 and ecc < 3°. The quartet of HP locations on a 4° ring goes on top as colored stars.
  - Implementation: `Subject.get_conditionwise_summary_prf_pars(model=8)` already returns `x, y, x_mean, y_mean` and `condition`. Use `seaborn.FacetGrid(col='condition')` × `ax.quiver`.
- B2. **Rotated-coordinate hexbin** ("everyone's distractor at 12 o'clock"). Use `x_rotated, y_rotated` columns; color hexbins by `distance_from_distractor − distance_from_distractor_mean` (negative = pulled toward HP, positive = pushed away). One panel per ROI, averaged across subjects. This is the cleanest single-image summary of the effect.
  - The transformation is already in `data.py` (`location_angles` + `rotate_x_y`); `notebooks/sampling_and_rotating.ipynb` is presumably where this lives.
- B3. **Distance-from-HP-distractor as a function of mean-PRF distance from HP**, lineplot with 95% CI band, separate line per ROI or per condition. If suppression "pushes away" rather than "attracts toward", this curve should sit above the y=x line near the HP location and converge to it far away. Use `seaborn.lineplot(errorbar=("ci", 95))` over per-subject means.

### Figure C — effect grows up the visual hierarchy (slide 11)

**Goal:** the headline quantitative claim.

- C1. **Per-ROI dot plot** of one summary statistic (e.g. mean inward shift in mm, or mean Δ-distance in degrees) on x = ROI ordered V1 → V2 → V3 → V3AB → hV4 → LO → TO, with subject-level dots and group mean ± 95% CI. Suppression-style accounts predict a monotonically growing effect with hierarchy level.
  - Per global instructions: `sns.scatterplot` per subject + `ax.errorbar` overlay for the group, inside a single axis. Sort subjects by ROI-mean estimate.
- C2. **Attention-model parameter** (`attention_sd` and log-`ratio` from `fit_attention_model.py`) per subject per ROI. Same plot style as C1, two columns. The `ratio` parameter is the model's read-out of "is the HP location attracting (>1) or repelling (<1)?"; plotting it across ROIs operationalizes the spotlight-vs-suppression distinction quantitatively.

### Figure D — interpretation cartoon (slide 12)

**Goal:** what does a shift mean? The slide already asks the question.

- D1. **Two-panel cartoon**: (left) classical attentional-attraction story (Klein 2014), pRFs pulled toward attended locus → finer sampling there → "zoom lens"; (right) suppression story, pRFs pushed away from HP location → coarser sampling there → "blind spot for distractors". Pair with the empirical direction observed in Figure B.
- D2. (Optional) brief **schematic of the attention-field model with offsets** (Sumiya/Abdirashid 2026, ch. 2) overlaid on our finding. Useful only if there is time and the result actually matches the AF+ (offset on attention field) model — which would predict local effects only, consistent with C1's hierarchy gradient if pRF size grows.

## Practical to-dos before producing figures

1. Re-run conditionwise PRF fits with the latest models (model 4 baseline + model 8 conditionwise are the current consensus). Confirm `derivatives/prf_summaries.conditionwise/model8/` exists for all 28 included subjects.
2. Decide on inclusion criteria (current default in `fit_attention_model.py`: mean R² > 0.2, ecc < 3°, sd > 0.1). Apply uniformly across A/B/C.
3. For Figure A, pick one "showcase" subject and verify the polar-angle map is clean. `notebooks/visualize_fsaverage.py` and `import_freesurfer_subject.py` are the relevant entry points; pycortex env is `pycortex2` (per global CLAUDE).
4. The placeholder caption "could do arrow plots" on slide 9 maps directly to B1 — start there, it is the most visually compelling single panel.

## Budget / risk

- The "shift grows up the hierarchy" claim (slide 11) is the riskiest and most novel; if the gradient is weak, lead with B2 (rotated hexbin) which makes a strong qualitative point even without a clean ROI ordering.
- Don't over-commit to the "suppression vs attraction" framing in the figures themselves — let the arrows in B1 tell the audience the direction; save the labeling battle for slide 12.
