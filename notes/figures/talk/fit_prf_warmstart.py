"""Multi-stage warm-start PRF refit, V1 only, all subjects.

Motivation
----------
The canonical model 4 (DoG + flex HRF) fits on this dataset show a
pathological coverage pattern: PRF centers cluster at the 4 search-
array positions rather than tiling the visual field. This happens
*even though the surround search-array stimuli are in the paradigm
tensor* and *even though the GD step initializes from model 1's clean
retinotopy*. The single-shot GD step (`lr=0.005`, 2000 iters) lets the
optimizer migrate the *spatial* parameters (x, y, sd) of the surround
PRF models toward distractor positions where the DoG has a deeper
loss basin.

A model-1 fit (pure Gaussian, no surround) has no such pathology —
it can only fit the bar paradigm. So model 1's retinotopy is trusted.

This script implements a **staged warm-start**: each stage fixes
the parameters that the previous stage already trusted, and only
optimizes the new degrees of freedom. The final stage with a small
learning rate is a fine-tune, not a migration.

Schedule per model
------------------
**Core rule**: spatial parameters (x, y, sd) and HRF parameters
(hrf_delay, hrf_dispersion) are NEVER free at the same time. HRF
gradients at lr=0.005 dominate the spatial gradient (script comment
in fit_prf.py concedes this), so joint optimization corrupts
retinotopy. Every stage must either freeze SPATIAL or freeze HRF.

- **m2** (DoG, fixed HRF) — no HRF, only spatial<->surround conflict:
    A. freeze x, y, sd; lr=0.005; 1500 iters  (warm surround)
    B. all free; lr=0.001; 300 iters          (refine — no HRF to clash)

- **m3** (Gaussian, flex HRF) — spatial trusted from m1:
    A. freeze spatial; lr=0.005; 800 iters     (warm HRF only)
    B. freeze HRF; lr=0.001; 300 iters         (refine spatial)

- **m4** (DoG, flex HRF) — spatial frozen until last, HRF frozen at last:
    A. freeze spatial + HRF; lr=0.005; 1500 iters  (warm surround only)
    B. freeze spatial; lr=0.005; 800 iters          (warm HRF, surround free)
    C. freeze HRF; lr=0.001; 300 iters              (refine spatial + surround)

- **m5** (DN, fixed HRF) — no HRF, only spatial<->DN conflict:
    A. freeze x, y, sd; lr=0.005; 1500 iters
    B. all free; lr=0.001; 300 iters

- **m6** (DN, flex HRF) — chains *from m5*, not from m1+m4. m5 already
  has clean DN params (no HRF), so we only need to warm the HRF:
    A. freeze spatial + DN params + amp/baseline; lr=0.005; (HRF only)
    B. freeze HRF; lr=0.001; (refine spatial + DN)

Model 1 has no surround and no flex HRF → no schedule needed; skip.

Iteration counts
----------------
The literal numbers below are **heuristic seeds** — chosen to roughly
split the existing 2000-iter single-shot budget across stages. They
should be tuned by inspecting where median R² plateaus per stage,
or replaced with an early-stopping criterion (Δ R² < 1e-4 over a
window of 50 iters). Treat as starting values, not gospel.

Init source
-----------
Stage A's init params come from the **current model-1 fit on disk**
(clean retinotopy + amp/baseline). Surround / DN / HRF params are
seeded with the same `extra_init` values used by the original
`MODEL_CFG` in `fit_prf.py`.

Output
------
One TSV per (model): `notes/data/prf_warmstart_m{N}_V1.tsv` with
per-voxel rows `(subject, hemi, x, y, sd, ..., r2_grid, r2_final)`.

CLI
---
    python fit_prf_warmstart.py <model> [--subjects 1-30]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from nilearn import image, maskers

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from retsupp.modeling.fit_prf import (  # noqa: E402
    MODEL_CFG, load_concatenated, load_prior_pars,
)
from retsupp.utils.data import Subject  # noqa: E402
from braincoder.optimize import ParameterFitter  # noqa: E402
from braincoder.hrf import SPMHRFModel  # noqa: E402

DEFAULT_BIDS = "/data/ds-retsupp"
RESOLUTION = 50
OUT_DIR = REPO / "notes" / "data"
# Set at CLI; thread through to the per-subject loader.
BIDS = DEFAULT_BIDS

# ------------------------------------------------------------------
# Schedule definitions. Each stage = (fixed_pars list, lr, n_iters).
# ------------------------------------------------------------------
SPATIAL = ["x", "y", "sd"]
HRF = ["hrf_delay", "hrf_dispersion"]

DN_PARAMS = ["rf_amplitude", "srf_amplitude", "srf_size",
             "neural_baseline", "surround_baseline", "bold_baseline"]

# Each stage = (fixed_pars, lr, max_iters, r2_atol).
# - max_iters: upper bound; braincoder plateau-stops earlier when
#   median ΔR² over the last `lag` (=100) iters is below r2_atol.
# - r2_atol: default 1e-6 is too strict (per-iter ΔR² is ~1e-4), so
#   we use 1e-4 on warm/coarse stages and 1e-5 on the refine stage
#   where smaller improvements still matter.
SCHEDULES = {
    # m1 is NOT refit by this script — the on-disk m1 fits are
    # already converged. Refining with all-free GD at lr=0.005
    # overshoots and degrades them (p90 R² 0.21 → 0.005 in smoke test).
    # Use cached m1 NIfTIs directly as the chain root.
    # m2 (DoG, fixed HRF). Init from m1.
    2: [
        (SPATIAL,          0.005, 3000, 1e-4),  # warm surround
        ([],               0.001, 1500, 1e-5),  # refine all
    ],
    # m3 (Gaussian, flex HRF). Init from m1. Spatial trusted.
    3: [
        (SPATIAL,          0.005, 3000, 1e-4),  # warm HRF
        (HRF,              0.001, 1500, 1e-5),  # refine spatial
    ],
    # m4 (DoG, flex HRF). Init from m2 — surround already clean.
    4: [
        # Stage A: only HRF moves; spatial AND surround frozen.
        (SPATIAL + ["srf_amplitude", "srf_size", "amplitude", "baseline"],
                           0.005, 3000, 1e-4),
        # Stage B: HRF frozen; refine spatial + surround.
        (HRF,              0.001, 1500, 1e-5),
    ],
    # m5 (DN, fixed HRF). Init from m2 — DoG surround params remapped
    # into DN parameterisation (see init_for_model).
    5: [
        (SPATIAL,          0.005, 3000, 1e-4),  # warm DN params
        ([],               0.001, 1500, 1e-5),  # refine all
    ],
    # m6 (DN, flex HRF). Init from m5 — DN params already clean.
    6: [
        (SPATIAL + DN_PARAMS, 0.005, 3000, 1e-4),  # warm HRF only
        (HRF,                 0.001, 1500, 1e-5),  # refine spatial + DN
    ],
}

# Which model's fit output should we use as the warm-start init?
# m1 refines from on-disk m1 (no chain).
CHAIN_INIT_FROM = {
    2: 1,
    3: 1,
    4: 2,
    5: 2,
    6: 5,
}


def _assert_no_joint_spatial_hrf(model_label):
    """Sanity guard for the core rule: spatial and HRF must not be
    free in the same stage."""
    for i, stage in enumerate(SCHEDULES[model_label], start=1):
        fixed = stage[0]
        fixed_set = set(fixed)
        has_spatial = any(p in fixed_set for p in SPATIAL)
        has_hrf = any(p in fixed_set for p in HRF)
        if model_label in (3, 4, 6):
            assert has_spatial or has_hrf, (
                f"Model {model_label} stage {i}: spatial AND HRF both "
                f"free — violates core rule.")


for _m in SCHEDULES:
    _assert_no_joint_spatial_hrf(_m)


def get_v1_voxel_idx(sub, masker):
    """Return (idx_in_masker_flat, hemi_label) for V1 voxels.

    Optimised vs. calling ``Subject.get_retinotopic_roi`` twice (once
    per hemi):
      - That path resamples varea AND aparc+aseg into BOLD space on
        EACH call → 4 expensive resamples for V1_L + V1_R combined.
      - This version resamples varea once and aparc+aseg once per
        hemi (3 total) and indexes into the cached arrays directly.
      - Skips two ``masker.transform`` round-trips (NIfTI → flat) by
        using the masker's mask array directly.
    """
    # Resample varea atlas + per-hemi cortex masks ONCE each.
    varea = sub.get_retinotopic_atlas(bold_space=True).get_fdata().astype(
        np.int32)
    lh = sub.get_hemisphere_mask("L", bold_space=True).get_fdata().astype(
        bool)
    rh = sub.get_hemisphere_mask("R", bold_space=True).get_fdata().astype(
        bool)
    v1 = (varea == 1)   # Benson V1 label.

    # Map BOLD-space boolean masks → masker flat indices. The masker's
    # `mask_img_` defines which voxels appear (in order) in its flat
    # output, so we just index the BOLD-shape boolean array by the
    # mask's ravel-True positions.
    bold_mask = masker.mask_img_.get_fdata().astype(bool).ravel()
    v1_l_flat = (v1 & lh).ravel()[bold_mask]
    v1_r_flat = (v1 & rh).ravel()[bold_mask]

    l_idx = np.where(v1_l_flat)[0]
    r_idx = np.where(v1_r_flat)[0]
    if len(l_idx) == 0 and len(r_idx) == 0:
        return np.array([], dtype=int), np.array([], dtype="<U1")
    idx = np.concatenate([l_idx, r_idx])
    hemi = np.concatenate([
        np.full(len(l_idx), "L", dtype="<U1"),
        np.full(len(r_idx), "R", dtype="<U1"),
    ])
    return idx, hemi


def run_schedule(model_label, factory, data, paradigm, init_pars,
                 schedule):
    """Run staged GD according to ``schedule``. Returns final params.

    Each schedule stage is `(fixed_pars, lr, max_iter, r2_atol)`.
    `r2_atol` is the plateau threshold passed to ParameterFitter.fit;
    braincoder stops early when median ΔR² over `lag=100` iters falls
    below this.
    """
    pars = init_pars.copy()
    for stage_i, stage in enumerate(schedule, start=1):
        # Backwards-compat: allow 3-tuple stages without r2_atol.
        if len(stage) == 4:
            fixed, lr, n_iter, r2_atol = stage
        else:
            fixed, lr, n_iter = stage
            r2_atol = 1e-6
        fixed_present = [p for p in fixed if p in pars.columns]
        m = factory(data, paradigm)
        f = ParameterFitter(m, data, paradigm)
        print(f"  stage {stage_i}: fix={fixed_present}  lr={lr}  "
              f"max_iter={n_iter}  r2_atol={r2_atol}")
        t0 = time.time()
        pars = f.fit(init_pars=pars, max_n_iterations=n_iter,
                     learning_rate=lr,
                     fixed_pars=fixed_present if fixed_present else None,
                     r2_atol=r2_atol)
        # Drop derived columns so the next stage's fitter sees a clean frame.
        for derived in ("r2", "theta", "ecc"):
            if derived in pars.columns:
                pars = pars.drop(columns=[derived])
        # Quick post-stage R² readout — uses a fresh fitter so no
        # stale optimizer state.
        post_m = factory(data, paradigm)
        post_fitter = ParameterFitter(post_m, data, paradigm)
        med_r2 = float(post_fitter.get_rsq(pars).median())
        print(f"           took {time.time()-t0:.1f}s; "
              f"median r²={med_r2:.4f}")
    # Re-attach r² to the final frame.
    final_m = factory(data, paradigm)
    final_fitter = ParameterFitter(final_m, data, paradigm)
    pars["r2"] = final_fitter.get_rsq(pars).values
    return pars


# σ floor enforced by braincoder's shifted-softplus parameterisation.
# Below this value every sigma-like parameter (sd, srf_size, etc.) is
# clamped — prevents sigma-collapse pathology where σ → 0 produces
# spuriously good R² on noise. See notes/m6_dn_diagnosis.md.
SD_MIN = 0.3


def build_model_factory(model_label, grid_coords):
    """Returns a closure (data, paradigm) -> instantiated model.
    Mirrors the factory used in fit_prf.main()."""
    cfg = MODEL_CFG[model_label]
    cls = cfg["cls"]
    hrf = SPMHRFModel(tr=1.6, delay=4.5, dispersion=0.75)

    def factory(data, paradigm):
        return cls(grid_coordinates=grid_coords, paradigm=paradigm,
                   hrf_model=hrf, data=data,
                   flexible_hrf_parameters=cfg["flex_hrf"],
                   sd_min=SD_MIN)
    return factory


# Columns that are derived statistics, never model parameters. Always
# stripped from init frames before passing to ParameterFitter.fit so
# the optimizer can't be confused by them.
DERIVED_COLS = ["r2", "theta", "ecc", "p_signal", "subject", "hemi"]


def load_init_pars(subject_id, init_from, masker, derivs, v1_idx):
    """Load prior fit for a warm-start chain, V1-subsetted.

    Strategy:
      1. Try this subject's warm-start TSV (`prf_warmstart_m{N}_V1_
         sub-{NN}.tsv`) — already V1-only, no subsetting needed.
      2. Else fall back to cached NIfTIs at `derivatives/prf/model{N}/`
         and subset to V1 via `v1_idx`.
    Either way, derived statistic columns (`r2, theta, ecc, p_signal`)
    are stripped so the optimizer can't see them.
    """
    tsv = OUT_DIR / (f"prf_warmstart_m{init_from}_V1_"
                     f"sub-{subject_id:02d}.tsv")
    if tsv.exists():
        print(f"  init source: warmstart TSV (m{init_from})")
        df = pd.read_csv(tsv, sep="\t")
    else:
        print(f"  init source: cached NIfTIs (m{init_from})")
        df = load_prior_pars(subject_id, init_from, derivs, masker)
        df = df.iloc[v1_idx].reset_index(drop=True)
    return df.drop(columns=DERIVED_COLS, errors="ignore"
                   ).reset_index(drop=True)


def adapt_init_for(target_model, init_from, prior):
    """Transform prior parameters into the form expected by
    `target_model`'s stage A.

    Per-pair adapters (only the pairs we use in CHAIN_INIT_FROM):
      m1 ← m1:  no-op (refining m1 itself)
      m2 ← m1:  add srf_amplitude=5e-2, srf_size=2.0
      m3 ← m1:  add hrf_delay=4.5, hrf_dispersion=0.75
      m4 ← m2:  add hrf_delay=4.5, hrf_dispersion=0.75
      m5 ← m2:  remap amplitude→rf_amplitude; drop baseline;
                add neural_baseline=1, surround_baseline=1, bold_baseline=0
      m6 ← m5:  add hrf_delay=4.5, hrf_dispersion=0.75
    """
    init = prior.copy()
    pair = (init_from, target_model)
    if pair == (1, 1):
        return init  # refining m1 itself
    if pair == (1, 2):
        init["srf_amplitude"] = 5e-2
        init["srf_size"] = 2.0
        return init
    if pair == (1, 3):
        init["hrf_delay"] = 4.5
        init["hrf_dispersion"] = 0.75
        return init
    if pair == (2, 4):
        init["hrf_delay"] = 4.5
        init["hrf_dispersion"] = 0.75
        return init
    if pair == (2, 5):
        # DoG surround → DN params
        if "amplitude" in init.columns:
            init["rf_amplitude"] = init["amplitude"]
        init = init.drop(columns=["amplitude", "baseline"],
                         errors="ignore")
        init["neural_baseline"] = 1.0
        init["surround_baseline"] = 1.0
        init["bold_baseline"] = 0.0
        return init
    if pair == (5, 6):
        init["hrf_delay"] = 4.5
        init["hrf_dispersion"] = 0.75
        return init
    raise ValueError(f"No adapter defined for m{init_from} → m{target_model}")


def fit_one_subject(subject_id, model_label):
    sub = Subject(subject_id, BIDS)
    derivs = Path(BIDS) / "derivatives"
    init_from = CHAIN_INIT_FROM[model_label]
    print(f"\n=== sub-{subject_id:02d} · model {model_label} "
          f"(init from m{init_from}) ===")

    # Brain mask + masker (needed for both cache loading and init).
    first_run = sub.get_runs(1)[0]
    bold_mask = sub.get_bold_mask(session=1, run=first_run)
    masker = maskers.NiftiMasker(mask_img=bold_mask)
    masker.fit()

    # Try the pre-built cleaned-BOLD ROI cache first — saves ~6 min of
    # NIfTI loads per task. Built by build_cleaned_roi_cache.py.
    cache_file = (derivs / "cleaned_roi_cache" / f"sub-{subject_id:02d}"
                  / f"sub-{subject_id:02d}_roi-V1_res-{RESOLUTION}.npz")
    if cache_file.exists():
        print(f"  using cleaned-BOLD ROI cache: {cache_file.name}")
        c = np.load(cache_file)
        data = c["bold"].astype(np.float32)
        paradigm = c["paradigm"].astype(np.float32)
        grid_coords = c["grid_coords"]
        v1_idx = c["voxel_idx"]
        v1_hemi = c["hemi"].astype("<U1")
    else:
        print(f"  no cache — building from cleaned NIfTIs (~6 min)")
        v1_idx, v1_hemi = get_v1_voxel_idx(sub, masker)
        if len(v1_idx) == 0:
            print(f"  no V1 voxels — skipping")
            return None
        data_full, paradigm, grid_coords = load_concatenated(
            sub, masker, RESOLUTION, "full")
        data = data_full[:, v1_idx].astype(np.float32)
    if len(v1_idx) == 0:
        print(f"  no V1 voxels — skipping")
        return None
    print(f"  V1 voxels: {len(v1_idx)};  "
          f"BOLD: {data.shape}; paradigm: {paradigm.shape}")

    # Load init from previous step in the chain. Handles both
    # warmstart-TSV and cached-NIfTI sources via fallback logic.
    prior = load_init_pars(subject_id, init_from, masker, derivs, v1_idx)
    if len(prior) != len(v1_idx):
        raise RuntimeError(
            f"init voxel count mismatch: prior has {len(prior)}, "
            f"V1 has {len(v1_idx)}")

    init = adapt_init_for(model_label, init_from, prior)
    print(f"  init cols: {list(init.columns)}")

    factory = build_model_factory(model_label, grid_coords)
    schedule = SCHEDULES[model_label]
    pars = run_schedule(model_label, factory, data, paradigm, init, schedule)

    pars["subject"] = subject_id
    pars["hemi"] = v1_hemi
    return pars


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", type=int, choices=list(SCHEDULES))
    ap.add_argument("--subjects", default="1-30",
                    help="Range like '1-30' or comma list '1,5,8'")
    ap.add_argument("--bids-folder", default=DEFAULT_BIDS,
                    help="BIDS root (cluster: /shares/zne.uzh/gdehol/ds-retsupp)")
    ap.add_argument("--out", default=None,
                    help="Output TSV path. Default: notes/data/"
                         "prf_warmstart_m{model}_V1[_sub{N}].tsv")
    args = ap.parse_args()

    global BIDS
    BIDS = args.bids_folder

    if "-" in args.subjects:
        lo, hi = args.subjects.split("-")
        subjects = list(range(int(lo), int(hi) + 1))
    else:
        subjects = [int(s) for s in args.subjects.split(",")]

    if args.out:
        out_path = Path(args.out)
    elif len(subjects) == 1:
        # Single subject → suffix by subject id so SLURM array tasks
        # don't overwrite each other.
        out_path = OUT_DIR / (
            f"prf_warmstart_m{args.model}_V1_sub-{subjects[0]:02d}.tsv")
    else:
        out_path = OUT_DIR / f"prf_warmstart_m{args.model}_V1.tsv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing rows to {out_path}")

    all_dfs = []
    for s in subjects:
        try:
            df = fit_one_subject(s, args.model)
        except FileNotFoundError as e:
            print(f"  sub-{s:02d}: skip ({e})")
            continue
        except Exception as e:
            print(f"  sub-{s:02d}: ERROR {type(e).__name__}: {e}")
            continue
        if df is None:
            continue
        all_dfs.append(df)
        # Incremental save so a crash doesn't lose progress.
        pd.concat(all_dfs, ignore_index=True).to_csv(
            out_path, sep="\t", index=False)

    if all_dfs:
        out = pd.concat(all_dfs, ignore_index=True)
        out.to_csv(out_path, sep="\t", index=False)
        print(f"\nDone. Wrote {len(out):,} voxel-rows across "
              f"{out.subject.nunique()} subjects to {out_path}")
    else:
        print("\nNo successful fits.")


if __name__ == "__main__":
    main()
