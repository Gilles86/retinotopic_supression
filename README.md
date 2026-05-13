# retsupp — Retinotopy + Suppression

A 7T fMRI experiment investigating whether **history-driven attentional suppression** reshapes population receptive fields (PRFs) in human visual cortex.

Participants perform a visual search task in which a singleton distractor is, with high probability, presented at one location ("HP location") within a block. While they search, a separate sweeping-bar PRF mapping stimulus is shown across the entire screen. The science question: do PRFs near the HP location shift, sharpen, or weaken — and how does this depend on the visual area's place in the hierarchy?

Collaborators: Dock H. Duncan (Vrije Universiteit Amsterdam, lead investigator), Christian Olivers (VU), the Theeuwes group, plus partners at UZH, Granada, and Zhejiang.

---

## Dataset

- BIDS dataset: `ds-retsupp`
- 30 subjects (`retsupp/data/subjects.yml`), 2 sessions × 6 runs (a few subjects: 5 runs — always call `Subject.get_runs(session)`).
- TR = 1.6 s, 258 volumes per run.
- Acquisition: 7T Philips Achieva, Spinoza Centre Amsterdam.
- Denoising: NORDIC (`nordic/run_nordic.m`).
- Preprocessing: fMRIprep (T1w + fsnative output spaces), `--dummy-scans 4`.

Local mirror: `/data/ds-retsupp`. Compute cluster mirror: `/shares/zne.uzh/gdehol/ds-retsupp` (sciencecluster, UZH).

---

## Quick start

```bash
git clone --recurse-submodules https://github.com/Gilles86/retinotopic_supression.git retsupp
cd retsupp

# Create the main env (TF 2.14 CPU; works for local analysis + most cluster jobs).
mamba env create -f create_env/retsupp_cpu.yml
mamba activate retsupp
pip install -e .              # install retsupp itself
pip install -e libs/braincoder  # editable submodule install
```

For GPU PRF fits on the cluster, use `retsupp_cuda` (`create_env/retsupp_cuda.yml`). For neuropythy atlas inference: `retsupp_neuropythy`.

---

## Entry point: `Subject`

Almost everything goes through this class.

```python
from retsupp.utils.data import Subject

sub = Subject(subject_id=5, bids_folder='/data/ds-retsupp')

# Cleaned BOLD for one run
bold = sub.get_bold(session=1, run=1, type='cleaned')

# Reconstructed bar-mapping stimulus paradigm (T, R, R)
stim = sub.get_stimulus(session=1, run=1, resolution=50)

# Where the HP distractor was per (session, run)
hpd = sub.get_hpd_locations()

# PRF parameters as NIfTIs (mean across runs, model 4 = DoG + flex HRF)
pars = sub.get_prf_parameters_volume(model=4, type='mean')

# V1 left-hemisphere ROI (neuropythy atlas, BOLD-space)
v1l = sub.get_retinotopic_roi('V1_L', bold_space=True, model=4)
```

Subject paths, session/run quirks (subjects 1-2 have a different sourcedata layout; sub-20 ses-1 and sub-24 ses-2 have 5 runs), and the BIDS↔YAML session-numbering offset are all encapsulated here.

---

## Pipeline overview

```
DICOM ──► BIDS ──► NORDIC ──► fMRIprep ──┬──► clean ──► PRF fits ──► surface sample ──► neuropythy atlas ──► ROIs
                                         │              (chunked,                                            │
                                         │               GPU)                                                ▼
                                         └──► glmsingle (single-trial betas)                       conditionwise PRFs
                                                                                                            │
                                                                                                            ▼
                                                                                                  attention model
                                                                                                  (per-subject, per-ROI)
```

| Step | Script | Notes |
|---|---|---|
| 1. DICOM → BIDS | `retsupp/prepare/convert_raw_mri_data.py` | |
| 2. NORDIC | `retsupp/nordic/run_nordic.m` | MATLAB |
| 3. fMRIprep | `retsupp/cluster_preproc/fmriprep.sh` | Apptainer, cluster-only |
| 4. Clean | `retsupp/preprocess/clean.py` | Interpolates motion-outlier frames, regresses confounds |
| 5. PRF fits | `retsupp/modeling/fit_prf.py` | Chunked GPU sweep via `submit_prf_sweep_persub.sh`; 6 models — see below |
| 6. Surface sample | `retsupp/surface/sample_prf_to_surface_nilearn.py` | Resamples PRF params to fsnative / fsaverage gifti |
| 7. Neuropythy atlas | `retsupp/neuropythy/register_retinotopy.py` | Wang/Benson `inferred_*.mgz` per model |
| 8. ROIs | `retsupp/eccentric_glm/get_rois.py` | Per-ROI × hemisphere × quadrant masks |
| 9. Eccentric GLM | `retsupp/eccentric_glm/fit_glm.py` | Per-distractor-location GLM on PRF-regressed residuals |
| 10. Attention model | `retsupp/modeling/fit_attention_model.py` | 2-parameter precision-weighted product-of-Gaussians fit |

### PRF model labels

| Label | Model | Flex HRF | Notes |
|---|---|---|---|
| 1 | Gaussian | no | Grid + GD, used to seed everything else |
| 2 | Difference-of-Gaussians | no | Surround inits from m1 |
| 3 | Gaussian | yes | HRF refinement from canonical |
| 4 | DoG | yes | **Canonical "mean" model** for downstream analyses |
| 5 | Divisive Normalization | no | Inits from m2 |
| 6 | Divisive Normalization | yes | Inits from m5 |

Models are fit as a warm-start chain: m1 → m2/m3 (parallel) → m4/m5 (parallel) → m6. See `retsupp/modeling/fit_prf.py` for the schedule definitions.

---

## Repo layout

```
retsupp/                  # main Python package
  utils/data.py           # Subject class (data access)
  preprocess/             # clean + mean BOLD
  modeling/               # PRF fits + warm-start chain + attention model
  neuropythy/             # retinotopic atlas inference (Benson + Wang)
  surface/                # volume → surface resampling
  eccentric_glm/          # per-distractor-location GLM
  glm/                    # single-trial GLMs (GLMsingle)
  visualize/              # pycortex visualization
  notebooks/              # exploratory + figure-building notebooks
experiment/               # PsychoPy task code
libs/braincoder/          # forked encoding-model library (submodule)
create_env/               # conda env definitions
notes/                    # working notes (rsync recipes, model implementation notes, ...)
```

---

## Computational details

- PRF fitting uses the [`braincoder`](https://github.com/Gilles86/braincoder) library (TensorFlow backend) — composable Gaussian / DoG / DN PRF models with HRF-aware forward functions, multi-stage gradient-descent fitting, and Bayesian decoding.
- Fits are **chunked** (voxel-wise) and submitted as SLURM array jobs to the UZH sciencecluster. See `submit_prf_sweep_persub.sh` for the full per-subject dependency chain.
- All derived parameter NIfTIs are written as **float32** with `scl_slope=1, scl_inter=0` — see `CLAUDE.md` §"NIfTI dtype trap" for why.

---

## Citation / contact

If you use this code or its derived analyses, please contact Gilles de Hollander (`gilles.de.hollander@gmail.com`) — paper in preparation.

Key dependencies:
- `braincoder` (de Hollander) — encoding-model fits and Bayesian decoding
- `nilearn`, `nibabel`, `nilearn.maskers`
- `tensorflow` 2.14 (with optional CUDA)
- `neuropythy` (Benson) — retinotopic atlas inference

---

## For developers / maintainers

See [`CLAUDE.md`](CLAUDE.md) for project-specific conventions, cluster operations, debugging notes, and gotchas that aren't immediately obvious from the code.
