# V1 decoding handoff — prompt for the decoder agent

You are picking up a V1 stimulus-decoding task on the retsupp dataset.
The PRF fits have been validated; your job is to take them and decode
the per-TR visual stimulus from the V1 BOLD timeseries using
`braincoder.stimuli.StimulusFitter` or equivalent.

## Where to find the data

A single per-subject NPZ lives at:

```
/shares/zne.uzh/gdehol/ds-retsupp/derivatives/v1_decode_handoffs/sub-{NN}_m4_V1.npz
```

Available subjects right now: **sub-15** (clean, sandbox-validated).
sub-03 and sub-04 are queued and will land shortly under the same
filename pattern.

Avoid **sub-01 and sub-02** — they have a different sourcedata layout
and a buggy counterbalancing order (CLAUDE.md §"Project-specific
gotchas"). The PRF fits exist for them too, but the BOLD's HP-condition
block pattern is `AA BBB CCC DDD A` rather than the intended
`AAA BBB CCC DDD`.

## NPZ schema

```python
import numpy as np
d = np.load("sub-15_m4_V1.npz")

# PRF parameters (m4 = DoG with flex HRF), one entry per V1 voxel
d["x"]              # (V,) PRF center x, degrees
d["y"]              # (V,) PRF center y, degrees
d["sd"]             # (V,) center Gaussian σ in degrees; > 0.2 strict
d["amplitude"]      # (V,) center response amplitude (signed)
d["baseline"]       # (V,) DC baseline (signed)
d["srf_amplitude"]  # (V,) surround amplitude as fraction of |amp|, > 0
d["srf_size"]       # (V,) surround:center σ RATIO (>1, NOT an absolute σ)
d["hrf_delay"]      # (V,) SPM HRF delay (seconds)
d["hrf_dispersion"] # (V,) SPM HRF dispersion
d["r2"]             # (V,) per-voxel R² of the fit (mark_invalid_fits → r2=0 if degenerate)

# Voxel metadata
d["hemi"]           # (V,) "L" or "R"
d["voxel_idx"]      # (V,) brain-mask flat index — useful if you want to
                    #      project back to NIfTI via NiftiMasker

# BOLD: cleaned, concatenated across all 12 runs (2 sessions × 6 runs),
# 258 TRs per run, voxel-aligned to the params above.
d["bold"]           # (T, V) float32  where T = 12*258 = 3096

# Constants
d["tr"]              # 1.6 seconds
d["resolution"]      # 50 (PRF grid pixels per dimension)
d["grid_radius"]     # 5.0 (degrees of visual field captured by the grid)
d["aperture_radius"] # 3.17 (bar aperture radius in degrees)
```

## Model contract (important)

The PRF model is **m4 = Difference-of-Gaussians + flexible HRF** as
implemented in `braincoder.models.DifferenceOfGaussiansPRF2DWithHRF`.
Per-TR prediction is:

```
center      = amplitude * exp(-r² / 2sd²)               # center Gaussian
surround    = amplitude * srf_amplitude * exp(-r² / 2(sd·srf_size)²)
prf         = center - surround                          # DoG
response_TR = <stim, prf> * HRF_kernel(delay, dispersion) + baseline
```

Note `srf_size > 1` is a **ratio**, not an absolute σ — surround σ
= `sd * srf_size`. With the fixed transforms,
`srf_size ∈ (1, ∞)` strictly, and `sd ∈ (0.2, ∞)`.

## Paradigm

The 12-run timecourse in `d["bold"]` was acquired while the subject:

1. Did a visual-search task (8-item array of oriented rectangles at
   4° eccentricity diagonals, with a colour-singleton distractor at
   one of 4 locations per run — the HP location is run-specific).
2. *Concurrently* saw a sweeping bar PRF mapping stimulus inside a
   3.17° aperture.

For decoding, you'll typically want the bar-only stimulus reconstruction
on a 50×50 grid (radius 5°). The bar paradigm is reconstructable via
`retsupp.utils.data.Subject.get_stimulus(session, run, resolution=50)`
— this returns a `(T, R, R)` binary stimulus tensor matching the
concatenated BOLD layout. To get the same 12-run concatenation:

```python
from retsupp.utils.data import Subject
sub = Subject(15, "/shares/zne.uzh/gdehol/ds-retsupp")
import numpy as np
stim_chunks = []
for ses in (1, 2):
    for run in sub.get_runs(ses):
        stim_chunks.append(
            sub.get_stimulus(session=ses, run=run, resolution=50).astype(np.float32))
stim = np.concatenate(stim_chunks, axis=0)  # (T=3096, 50, 50)
```

If you want the **full** stimulus (bar + search array), use
`sub.get_stimulus_with_distractors(...)` instead — same return shape
on the extended grid (radius 5°).

## How to decode

Standard pattern with braincoder:

```python
import numpy as np
from braincoder.models import DifferenceOfGaussiansPRF2DWithHRF
from braincoder.stimuli import StimulusFitter
from braincoder.hrf import SPMHRFModel

d = np.load("sub-15_m4_V1.npz")
pars = np.column_stack([d["x"], d["y"], d["sd"], d["baseline"], d["amplitude"],
                         d["srf_amplitude"], d["srf_size"],
                         d["hrf_delay"], d["hrf_dispersion"]])

# Build model that matches the fit
hrf = SPMHRFModel(tr=1.6, delay=4.5, dispersion=0.75)
model = DifferenceOfGaussiansPRF2DWithHRF(
    grid_coordinates=...,           # build from grid_radius + resolution
    paradigm=stim.reshape(stim.shape[0], -1),
    hrf_model=hrf,
    flexible_hrf_parameters=True,
    sd_min=0.2,                     # MUST match the fit's floor
    data=d["bold"],
    parameters=pars)

# Decode the stimulus per TR
fitter = StimulusFitter(data=d["bold"], model=model)
# IMPORTANT: always pass a non-zero l2_norm. Without it the decoded
# values explode for high-r² voxels (Gilles's stated convention).
decoded = fitter.fit(l2_norm=1e-2, ...)
```

## Things to be careful about

1. **sd_min match**. The model object you instantiate for decoding
   MUST be constructed with `sd_min=0.2` — same as the fit. If
   not, `_sd_softplus_inverse` will fail when projecting the fit
   parameters back to raw form.
2. **srf_size is a ratio**. Don't confuse with absolute σ.
3. **l2_norm in StimulusFitter**. Always pass a non-zero value (per
   global retsupp/braincoder convention) — otherwise decoded values
   blow up.
4. **R² filtering**. Voxels with r²=0 are sentinels (post-hoc
   `mark_invalid_fits` marked them; the fit didn't converge for those).
   Skip them.
5. **Voxel selection for decoding**. Typical filter:
   `r2 > FDR_threshold` (use `retsupp.utils.data.select_well_fit_voxels`
   with `n_params=9, n_timepoints=3096`). For decoding, you may want
   to be stricter than for visualisation — e.g. top-quartile R².

## When sub-03 / sub-04 land

Same NPZ filename pattern. The handoff sbatch is `build_v1_handoff.sh`
in retsupp/modeling/slurm_jobs/ (will commit shortly). Override SUB=N
and MODEL=M as needed.
