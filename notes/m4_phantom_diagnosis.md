# M4 (DoG + flexible HRF) phantom-perfect voxel diagnosis

Date: 2026-05-10

## TL;DR

Model-4 PRF fits produce ~22 000 phantom-perfect voxels per subject
(sub-02; smaller counts for sub-01/sub-05) with R² ≥ 0.999, scattered
mostly outside cortex (upper brain slices, edges, partial-volume
voxels). They are **numerically artefactual**, not real model fits:

1. The DoG forward pass collapses to `sd = 0` for these voxels,
   making the Gaussian `exp(...)/0/0 = NaN` everywhere on the grid.
2. The braincoder R² function
   (`braincoder/utils/stats.py::get_rsq`) computes
   `ssq_resid = (resid**2).sum(0)`, where `resid = data - predictions`.
   With `predictions = NaN`, `resid = NaN`, but
   **pandas' `.sum()` skips NaN by default**, returning `0.0`. The
   result: `R² = 1 - 0 / SStot = 1.0` for every phantom voxel.
3. Per-parameter uniqueness across all 22 127 sub-02 phantoms is
   **exactly 1**: every phantom voxel stores the same magic constants
   (`amplitude = -46.1136`, `baseline = -2.6318`, `x ≈ y ≈ 0`,
   `sd = srf_amplitude = srf_size = hrf_delay = hrf_dispersion = 0`).
   These are the post-`_transform_parameters_forward` values; the
   pre-transform untransformed parameters all hit the same softplus
   floor during GD. Adam pushed sd to softplus(very-negative) ≈ 0,
   predictions went NaN, gradients went NaN, and `best_parameters`
   froze at the last-pre-NaN state for the whole cohort.

## Files produced

- `notes/scripts/diagnose_m4_phantoms.py` — main diagnostic.
- `notes/scripts/verify_phantom_mechanism.py` — numerical proof that
  phantom predictions are NaN and `get_rsq` returns 1.0.
- `notes/figures/m4_phantom_diagnosis_params_sub-{01,02,05}.pdf` —
  parameter histograms, phantom vs M1-signal.
- `notes/figures/m4_phantom_diagnosis_spatial_sub-{01,02,05}.pdf` —
  axial slices showing phantom-vs-signal locations.
- `notes/figures/m4_phantom_diagnosis_bold_sub-{01,02}.pdf` — example
  BOLD timeseries (cleaned + fmriprep).
- `notes/figures/m4_phantom_diagnosis_r2_vs_sd_sub-{01,02,05}.pdf` —
  joint M4 sd vs R² scatter.
- `notes/data/m4_phantom_filter_table_sub-{01,02,05}.tsv` — candidate
  filter evaluations per subject.
- `notes/data/m4_phantom_cross_subject.tsv` — cross-subject summary.

## Findings (1–6 from the brief)

### 1. Where do phantoms live spatially?

Scattered across the whole BOLD mask, with **highest density in
upper slices (Z=45, 60)** — i.e. away from occipital cortex. Some
ride the edge of the brain mask; many sit inside parenchyma but
have very low stimulus-modulated signal (white matter,
partial-volume, sub-cortical). They are NOT concentrated in
ventricles. The signal-m1 voxels (M1 R² > 0.05 & sd ≥ 0.5)
concentrate in posterior occipital cortex (Z=15, 30), with **zero
overlap with phantoms** (sub-02: 0 / 6363, sub-01: 0 / 7570).

Phantom counts per subject scanned:

| Subject | Phantoms (m4 R² ≥ 0.999) | Signal (m1) | Overlap |
|---|---:|---:|---:|
| sub-01 | 1 111 | 7 570 | 0 |
| sub-02 | 22 127 | 6 363 | 0 |
| sub-05 |   508 | 5 890 | 1 |

Sub-02 has ~20× more phantoms than sub-01/05, suggesting subject-
specific BOLD properties (low cleaning quality?  more masked-edge
voxels?  something about that subject's NORDIC/preproc) drive the
problem.

### 2. What are the BOLD timeseries like?

Phantom voxels in cleaned BOLD have visibly **less variance** than
signal voxels (sub-02 phantom median var = 0.78, signal = 1.88).
In fmriprep BOLD (raw, no cleaning), phantoms look smooth and
nearly flat (low temporal SNR). 20 / 22127 phantom voxels have
literally zero variance in cleaned BOLD, but the vast majority just
have **low stimulus-locked signal**. Not NaN, not constant —
just noisy/weak BOLD.

### 3. What are the fitted PRF parameters?

All at degenerate constants (unique-count = 1 across the 22 127
sub-02 phantoms):

| par | value |
|---|---|
| x | -0.0039 |
| y | -0.036 |
| sd | 0 |
| amplitude | -46.1136 |
| baseline | -2.6318 |
| srf_amplitude | 0 |
| srf_size | 0 |
| hrf_delay | 0 |
| hrf_dispersion | 0 |

The exact-same constants across 22 127 voxels are the smoking gun
that this is an optimizer failure mode, not a real fit.

### 4. Reproducibility across subjects

Same parameter pattern in sub-01 and sub-05 (all phantoms at
identical degenerate values), but the **count** varies wildly
(sub-02 = 22 k, sub-01 = 1 k, sub-05 = 0.5 k). Spatial locations
are in each subject's own native space and not directly comparable;
they cluster in low-tSNR regions for each.

### 5. Is it a fitting bug or a data property?

**Both — and it's a known kind of bug in braincoder.**

The data property: voxels with very low SNR have a flat cost
landscape; GD has no signal to push parameters toward sensible
values.

The bug: when GD runs long enough on such a voxel, Adam pushes
`sd` toward 0 (softplus → 0 from very-negative untransformed
value). At sd = 0 the Gaussian
`exp(-d²/(2·0²)) / (0·√(2π)/pixel_area)` evaluates to NaN. The
prediction is NaN for all timepoints, the gradient is NaN, and:

- In `optimize.py`, `r2 = 1 - ssq/ssq_data` with NaN ssq gives
  NaN. `improved_r2s = r2 > best_r2` returns False for NaN, so
  `best_parameters` doesn't get updated past the moment things
  went bad — but `trainable_voxel_specific_parameters` themselves
  become NaN and Adam happily continues.
- The final stored parameters are the **transformed** (forward-
  mapped) values of whatever sat in `best_parameters` (or the
  trainable variable; the chain is a bit tangled but the net
  effect is "all parameters at the constant produced by
  `softplus(very-negative)` = 0 plus the constants from identity-
  transformed pars stuck wherever they last were").
- Then `pars['r2'] = f.get_rsq(pars).values` (`fit_prf.py:156`)
  recomputes R² post-fit by evaluating predictions and using
  `braincoder.utils.stats.get_rsq`. Predictions are NaN. R² is:

```python
resid = data - predictions       # all NaN
ssq_resid = (resid**2).sum(0)    # pandas .sum skips NaN → 0.0
r2 = 1 - ssq_resid / ssq_data    # = 1.0 - 0 / nonzero = 1.0
```

Confirmed numerically in `verify_phantom_mechanism.py`:

```
NaN predictions, R² = [1. 1. 1. 1.]    # pandas .sum skips NaN
```

This is technically a **braincoder bug** — `get_rsq` should not
silently return 1.0 for NaN predictions. A fix upstream
(`braincoder/utils/stats.py`) would be a one-liner:

```python
ssq_resid = (resid**2).sum(0, skipna=False)  # propagate NaN
# or, explicitly:
ssq_resid = np.where(resid.isna().any(0), np.nan, (resid**2).sum(0))
```

That would make phantom R² = NaN, which our `np.isfinite` filter
already drops.

### 6. Proposed filter

`r2 < 0.999` is the **most reliable** filter (drops 100% of phantoms,
0% of M1-signal voxels in every subject). It is also the most
**interpretable**: "the fit isn't suspiciously perfect."

But a defensive filter that doesn't rely on the magic 0.999
threshold is even better: **drop voxels with `sd == 0` or any
non-finite parameter or non-finite prediction**. This catches the
root cause (sd = 0 → NaN prediction) rather than the symptom.

Candidate filters on sub-02 (phantom n = 22 127, M1-signal n = 6 363):

| filter | phantom drop | signal drop |
|---|---:|---:|
| `r2 < 0.999` | 100.0 % | 0.0 % |
| `sd > 0.05` | 100.0 % | 0.016 % |
| `sd > 0.05 AND r2 < 0.999` | 100.0 % | 0.016 % |
| `(sd > 0.05) OR (srf_size > 0.5)` | 100.0 % | 0.0 % |
| `sd > 0.1` | 100.0 % | 10.3 % |
| `amplitude > 1e-4` | 0.0 % | 0.0 % |

The current `decoder.py` filter `sd >= 0.5` drops 100% of phantoms
**but also drops 96 % of real m4 signal voxels** (because real DoG
fits have small centre σ; the surround handles the spatial extent).
That is the proximate cause of the StimulusFitter failures.

**Recommendation: replace the m4 filter with**

```python
keep = (np.isfinite(prf['sd'])
        & (prf['sd'] > 0.05)         # drops sd=0 phantoms
        & (prf['r2'] > r2_min)       # drops bad fits
        & (prf['r2'] < 0.999)        # belt-and-braces phantom guard
        & np.all(np.stack([np.isfinite(prf[p]) for p in PRF_PARS]),
                 axis=0))
```

This:
- Drops 100 % of phantoms (any one of `sd>0.05`, `r2<0.999`,
  or finiteness would suffice; together they are robust).
- Drops < 0.02 % of M1-signal voxels.
- Is interpretable: "centre σ above 0.05° (otherwise the Gaussian
  is degenerate), R² in the plausible range (0, 0.999), all
  parameters finite."

The `sd >= 0.5` line in `decoder.py:67` should be removed —
that's a Gaussian-PRF heuristic that doesn't transfer to DoG.

## Proposed code patch

In `retsupp/decode/decoder.py::select_roi_voxels`, replace the
filter block:

```python
keep = (roi_flat & finite & (sd >= sd_min) & (r2 > r2_min)
        & (ecc <= ecc_max))
```

with:

```python
# Drop phantom-perfect voxels (sd→0 → NaN predictions → braincoder
# get_rsq returns 1.0 due to pandas skipna). The (sd > 0.05)
# clause catches the sd→0 collapse and is consistent with real
# DoG fits (signal voxels have sd in (0.05, 1.5)°). r2 < 0.999
# is a belt-and-braces guard. See notes/m4_phantom_diagnosis.md.
keep = (roi_flat & finite
        & (sd > 0.05)
        & (r2 > r2_min) & (r2 < 0.999)
        & (ecc <= ecc_max))
```

A matching helper for `Subject.get_prf_parameters_volume` (so that
notebooks using that API also get phantom-free voxels) would be:

```python
def _phantom_mask_m4(prf: dict) -> np.ndarray:
    """True for non-phantom m4 voxels.

    Phantoms have sd→0 (degenerate Gaussian, NaN predictions),
    which propagates to R²=1.0 via pandas skipna in
    braincoder.utils.stats.get_rsq. See
    notes/m4_phantom_diagnosis.md.
    """
    finite = np.all(
        np.stack([np.isfinite(prf[p]) for p in
                  ('x', 'y', 'sd', 'amplitude', 'baseline',
                   'srf_amplitude', 'srf_size',
                   'hrf_delay', 'hrf_dispersion', 'r2')]),
        axis=0)
    return finite & (prf['sd'] > 0.05) & (prf['r2'] < 0.999)
```

## Upstream braincoder fix (suggested)

In `libs/braincoder/braincoder/utils/stats.py::get_rsq`, make NaN
predictions yield NaN R² (not 1.0):

```python
def get_rsq(data, predictions, zerovartonan=True,
            allow_biased_residuals=False):
    resid = data - predictions
    ssq_data = ((data - data.mean(0))**2).sum(0)
    if allow_biased_residuals:
        ssq_resid = ((resid - resid.mean(0))**2).sum(0, skipna=False)
    else:
        ssq_resid = (resid**2).sum(0, skipna=False)
    r2 = 1 - (ssq_resid / ssq_data)
    if zerovartonan:
        r2[data.var() == 0] = np.nan
    r2.name = 'r2'
    return r2
```

Then the m4 NIfTI's r2 channel would write **NaN** for phantoms,
and the downstream `np.isfinite(prf['r2'])` filter we already have
in `select_roi_voxels` would catch them with no further changes.
This is the most principled long-term fix; the local filter above
is the immediate downstream fix that doesn't require refitting.
