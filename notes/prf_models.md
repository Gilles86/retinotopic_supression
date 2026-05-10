# PRF model hierarchy

Configured in `retsupp/modeling/fit_prf.py` (`MODEL_CONFIG` dict).
All models fit the same paradigm (`--paradigm-kind=full` = concatenated
12 runs by default) and produce per-voxel parameter NIfTIs in
`derivatives/prf/model{N}/sub-XX/`.

| ID | Class | Encoding | flex HRF | Init from | Extras (init) |
|---|---|---|:-:|:-:|---|
| 1 | `GaussianPRF2DWithHRF` | Gaussian | ✗ | grid | – |
| 2 | `DifferenceOfGaussiansPRF2DWithHRF` | DoG | ✗ | model 1 | `srf_amplitude=0.05`, `srf_size=2.0` |
| 3 | `GaussianPRF2DWithHRF` | Gaussian | ✓ | model 1 | `hrf_delay=4.5`, `hrf_dispersion=0.75` |
| 4 | `DifferenceOfGaussiansPRF2DWithHRF` | DoG | ✓ | model 3 | `srf_amplitude=0.05`, `srf_size=2.0` |
| 5 | `DivisiveNormalizationGaussianPRF2DWithHRF` | DN | ✗ | model 4 | DN params |
| 6 | `DivisiveNormalizationGaussianPRF2DWithHRF` | DN | ✓ | model 4 | DN params |

Model 4 = "canonical mean PRF model" used by the AF analyses.

## Param semantics

- `x, y` — PRF centre (degrees of visual angle).
- `sd` — Gaussian std (degrees).
- `baseline`, `amplitude` — additive + multiplicative (applied to the
  HRF-convolved signal; cf. comment in `models.py`).
- `srf_amplitude` — surround Gaussian amplitude as a fraction of the
  centre Gaussian (DoG only; range [0, 1]).
- `srf_size` — surround Gaussian std as a multiple of the centre `sd`
  (DoG only; > 1 means surround is wider than centre).
- `hrf_delay`, `hrf_dispersion` — SPM-style canonical-HRF parameters
  (only fitted when `flex_hrf=True`).
- `r2` — coefficient of determination after fit.

## Pipeline order on the cluster

```
m1 grid+GD          (uses lots of GPU)
m2 from m1 (GD)       parallel with m3 (both depend only on m1)
m3 from m1 (GD)
m4 from m3 (GD)       canonical
m5/m6 from m4 (GD)    DN (rarely used)
```

`fit_prf.py` chunks the volume by `--voxel-chunk-size` (default 10000)
and writes per-chunk NPZs to
`derivatives/prf/model{N}/sub-XX/chunks/`. `merge_prf_chunks.py` then
concatenates these into per-parameter NIfTIs.

## GD hyperparameters

Defaults in `fit_prf.py`:

- `learning_rate = 0.005` (passed to `gd_fit(..., lr=...)`)
- `--max-n-iterations 2000`

These have been validated empirically and produce sensible R² across
models. Don't tweak without a strong reason.

## Chunk size (`--voxel-chunk-size`)

GPU benchmarks (sub-02, model 1, full paradigm, 1h walltime):

| chunk | L4 | V100 | A100 | H100 |
|---|---|---|---|---|
| 10k | TIMEOUT | TIMEOUT | – | – |
| 30k | TIMEOUT | 58:48 | 39:51 | 29:46 |
| 80k | – | 59:47 | 38:21 | 28:35 |

10k is too small (overhead dominated). 30k is the safe minimum.
**50k is a good default** for the production chain — fewer XLA
recompiles per subject (~6 chunks instead of ~29), no real cost. 80k
gives marginal additional speedup (1–2 min/subject).

## Common gotchas

- **DoG transform bug** (fixed 2026-05-10 in `braincoder@7c50f2c`):
  before the fix, `DifferenceOfGaussiansPRF2D` inherited the parent
  Gaussian's 5-column `_transform_parameters_{forward,backward}`,
  silently dropping `srf_amplitude` / `srf_size`. With flex HRF this
  also scrambled HRF cols into the surround positions in `_predict`,
  giving DoG fits R² *worse* than plain Gaussian. Always make sure
  the cluster's `libs/braincoder` HEAD is at least `7c50f2c` before
  fitting models 2 or 4.

- Concatenated paradigm. With `--paradigm-kind=full` the model fits
  to all 12 runs concatenated, not the run-mean. Total BOLD variance
  is much higher (per-run noise is no longer averaged), so absolute
  R² is mechanically smaller than the old "mean" pipeline. Don't
  worry about med R² ~0.002 — relative improvements between models
  matter.

- Init NaNs. If a prior model's NIfTI has NaN in any voxel, the
  subsequent GD propagates NaN cost → all-NaN parameters → format
  error in `format_parameters`. The chunked fit re-loads init from
  disk per chunk, so this is per-subject, not per-chunk.
