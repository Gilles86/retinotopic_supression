"""V1-only multi-stage warm-start PRF refit, all subjects.

Same per-model recipe as the canonical pipeline — schedules, chain,
adapters, σ-floor — imported from :mod:`retsupp.modeling.fit_prf`.
This script only adds the V1-specific bits:

  - select V1 voxels via the Benson atlas + hemisphere masks,
  - reuse the pre-built cleaned-BOLD V1 cache when present,
  - prefer the per-subject warm-start TSV as init source over the
    cached NIfTIs, with a hard ``voxel_idx`` consistency check,
  - write a per-(subject, model) TSV instead of NIfTIs.

Output: ``notes/data/prf_warmstart_m{N}_V1[_sub-{NN}].tsv`` with
per-voxel rows ``(subject, hemi, voxel_idx, x, y, sd, ..., r2)``.

CLI::

    python fit_prf_warmstart.py <model> [--subjects 1-30]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from nilearn import maskers

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from retsupp.modeling.fit_prf import (  # noqa: E402
    MODEL_CFG, build_model_factory, gd_fit_scheduled, grid_fit,
    load_concatenated, load_prior_pars,
)
from retsupp.utils.data import Subject, mark_invalid_fits  # noqa: E402


DEFAULT_BIDS = "/data/ds-retsupp"
RESOLUTION = 50
SD_MIN = 0.2
OUT_DIR = REPO / "notes" / "data"
BIDS = DEFAULT_BIDS

# Columns stripped from init frames before passing to ParameterFitter
# — derived stats + meta tagged onto warm-start TSVs.
DERIVED_COLS = ["r2", "theta", "ecc", "p_signal",
                "subject", "hemi", "voxel_idx"]


def get_v1_voxel_idx(sub, masker):
    """Return ``(masker_flat_idx, hemi_label)`` for V1 voxels.

    Faster than two ``Subject.get_retinotopic_roi`` calls because it
    resamples ``varea`` + hemi masks once and indexes the cached
    arrays directly.
    """
    varea = sub.get_retinotopic_atlas(bold_space=True).get_fdata().astype(np.int32)
    lh = sub.get_hemisphere_mask("L", bold_space=True).get_fdata().astype(bool)
    rh = sub.get_hemisphere_mask("R", bold_space=True).get_fdata().astype(bool)
    v1 = (varea == 1)
    bold_mask = masker.mask_img_.get_fdata().astype(bool).ravel()
    v1_l = (v1 & lh).ravel()[bold_mask]
    v1_r = (v1 & rh).ravel()[bold_mask]
    l_idx, r_idx = np.where(v1_l)[0], np.where(v1_r)[0]
    if len(l_idx) == 0 and len(r_idx) == 0:
        return np.array([], dtype=int), np.array([], dtype="<U1")
    idx = np.concatenate([l_idx, r_idx])
    hemi = np.concatenate([np.full(len(l_idx), "L", dtype="<U1"),
                            np.full(len(r_idx), "R", dtype="<U1")])
    return idx, hemi


def load_v1_data(sub, masker, derivs):
    """Get the V1 BOLD subset + paradigm + grid.

    Uses the cleaned-BOLD V1 cache when present (built by
    ``build_cleaned_roi_cache.py``); falls back to building from the
    cleaned NIfTIs (~6 min).
    """
    cache = (derivs / "cleaned_roi_cache" / f"sub-{sub.subject_id:02d}"
             / f"sub-{sub.subject_id:02d}_roi-V1_res-{RESOLUTION}.npz")
    if cache.exists():
        print(f"  using cleaned-BOLD ROI cache: {cache.name}")
        c = np.load(cache)
        return (c["bold"].astype(np.float32), c["paradigm"].astype(np.float32),
                c["grid_coords"], c["voxel_idx"], c["hemi"].astype("<U1"))

    print(f"  no cache — building from cleaned NIfTIs (~6 min)")
    v1_idx, v1_hemi = get_v1_voxel_idx(sub, masker)
    if len(v1_idx) == 0:
        return None, None, None, v1_idx, v1_hemi
    data_full, paradigm, grid_coords = load_concatenated(
        sub, masker, RESOLUTION, "full")
    return (data_full[:, v1_idx].astype(np.float32), paradigm,
            grid_coords, v1_idx, v1_hemi)


def load_v1_init(subject_id, init_from, masker, derivs, v1_idx):
    """Load init for a V1 chain step. Warm-start TSV first; cached
    NIfTIs as fallback.

    When the TSV is used, its ``voxel_idx`` column is verified against
    the current V1 mask — any drift surfaces as a hard error.
    """
    tsv = OUT_DIR / f"prf_warmstart_m{init_from}_V1_sub-{subject_id:02d}.tsv"
    if tsv.exists():
        print(f"  init source: warmstart TSV (m{init_from})")
        df = pd.read_csv(tsv, sep="\t")
        if "voxel_idx" in df.columns:
            prior = df["voxel_idx"].to_numpy()
            if not np.array_equal(prior, v1_idx):
                raise RuntimeError(
                    f"voxel_idx mismatch m{init_from} TSV vs current V1: "
                    f"prior {len(prior)}, current {len(v1_idx)}")
    else:
        print(f"  init source: cached NIfTIs (m{init_from})")
        df = load_prior_pars(subject_id, init_from, derivs, masker,
                              sd_min=SD_MIN)
        df = df.iloc[v1_idx].reset_index(drop=True)
    return df.drop(columns=DERIVED_COLS, errors="ignore").reset_index(drop=True)


def fit_one_subject(subject_id, model_label):
    sub = Subject(subject_id, BIDS)
    derivs = Path(BIDS) / "derivatives"
    cfg = MODEL_CFG[model_label]
    init_from = cfg["init_from"]
    print(f"\n=== sub-{subject_id:02d} · model {model_label} "
          f"(init from {f'm{init_from}' if init_from else 'grid'}) ===")

    first_run = sub.get_runs(1)[0]
    bold_mask = sub.get_bold_mask(session=1, run=first_run)
    masker = maskers.NiftiMasker(mask_img=bold_mask)
    masker.fit()

    data, paradigm, grid_coords, v1_idx, v1_hemi = load_v1_data(
        sub, masker, derivs)
    if len(v1_idx) == 0:
        print("  no V1 voxels — skipping")
        return None
    print(f"  V1 voxels: {len(v1_idx)};  BOLD: {data.shape};  "
          f"paradigm: {paradigm.shape}")

    factory = build_model_factory(cfg, grid_coords, sd_min=SD_MIN)
    chunk_size = max(len(v1_idx), 1)  # V1 fits comfortably in one chunk

    if init_from is None:
        init = grid_fit(factory(data, paradigm), data, paradigm,
                         chunk_size=chunk_size, debug=False, sd_min=SD_MIN)
        init = init.drop(columns=["r2", "theta", "ecc"], errors="ignore")
    else:
        prior = load_v1_init(subject_id, init_from, masker, derivs, v1_idx)
        init = cfg["adapt"](prior) if cfg.get("adapt") else prior
    print(f"  init cols: {list(init.columns)}")

    pars = gd_fit_scheduled(factory, data, paradigm, init,
                             chunk_size=chunk_size,
                             schedule=cfg["schedule"])
    pars["subject"] = subject_id
    pars["hemi"] = v1_hemi
    # Persist masker flat-index so chained runs can verify they're on
    # the exact same V1 voxel set (rather than just trusting row order).
    pars["voxel_idx"] = v1_idx
    mark_invalid_fits(pars, data)
    return pars


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", type=int, choices=list(MODEL_CFG))
    ap.add_argument("--subjects", default="1-30",
                    help="Range '1-30' or comma list '1,5,8'")
    ap.add_argument("--bids-folder", default=DEFAULT_BIDS)
    ap.add_argument("--out", default=None)
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
        # Per-subject TSV so SLURM array tasks don't collide.
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
