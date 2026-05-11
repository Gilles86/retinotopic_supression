"""Pre-build per-subject, per-ROI BOLD + paradigm cache.

Motivation
----------
``fit_prf.load_concatenated`` loads 12 cleaned-BOLD NIfTIs from disk
and runs each through a brain-mask masker (each transform: load ~600 MB
gzipped volume + mask). For an ROI-restricted analysis this is hugely
wasteful — we only need ~3000 V1 voxels but load 80k× more data.

This cache writes one ``.npz`` per (subject, ROI, resolution) with:
  - bold (T_total, V) float32  — concatenated cleaned BOLD, ROI voxels
  - paradigm (T_total, G) uint8 — full bar+distractor paradigm tensor
  - grid_coords (G, 2) float32
  - voxel_idx (V,) int32  — indices in the brain-mask masker's flat output
  - hemi (V,) <U1  — 'L' or 'R'

Once built, ``fit_prf_warmstart.load_cached_or_concat`` reads it in
<1s instead of re-doing the 12 NIfTI loads.

CLI:
    python build_prf_cache.py <subject> [--roi V1] [--resolution 50]
    python build_prf_cache.py 1-30  # range form, sequential

Output:
    {bids}/derivatives/prf_cache/sub-XX/sub-XX_roi-{ROI}_res-{N}.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import time

import numpy as np
from nilearn import maskers

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from retsupp.modeling.fit_prf import load_concatenated  # noqa: E402
from retsupp.utils.data import Subject  # noqa: E402


def cache_path(bids_folder: Path, subject: int, roi: str,
               resolution: int) -> Path:
    return (bids_folder / "derivatives" / "prf_cache"
            / f"sub-{subject:02d}"
            / f"sub-{subject:02d}_roi-{roi}_res-{resolution}.npz")


def build(subject: int, bids_folder: Path, roi: str = "V1",
          resolution: int = 50):
    sub = Subject(subject, bids_folder)

    # Brain-mask masker (same one fit_prf uses).
    first_run = sub.get_runs(1)[0]
    bold_mask = sub.get_bold_mask(session=1, run=first_run)
    masker = maskers.NiftiMasker(mask_img=bold_mask)
    masker.fit()

    # ROI indices in the brain-mask masker's flat output.
    varea = sub.get_retinotopic_atlas(bold_space=True).get_fdata().astype(
        np.int32)
    lh = sub.get_hemisphere_mask("L", bold_space=True).get_fdata().astype(
        bool)
    rh = sub.get_hemisphere_mask("R", bold_space=True).get_fdata().astype(
        bool)
    # Only V1 supported for now — extend by importing the labels dict
    # from `Subject.get_retinotopic_labels` when more ROIs are needed.
    if roi != "V1":
        raise NotImplementedError(
            f"Only V1 implemented; got '{roi}'. Easy extension.")
    roi_3d = (varea == 1)

    bold_mask_flat = masker.mask_img_.get_fdata().astype(bool).ravel()
    l_flat = (roi_3d & lh).ravel()[bold_mask_flat]
    r_flat = (roi_3d & rh).ravel()[bold_mask_flat]
    l_idx = np.where(l_flat)[0]
    r_idx = np.where(r_flat)[0]
    voxel_idx = np.concatenate([l_idx, r_idx]).astype(np.int32)
    hemi = np.concatenate([np.full(len(l_idx), "L", dtype="<U1"),
                           np.full(len(r_idx), "R", dtype="<U1")])
    if len(voxel_idx) == 0:
        print(f"  sub-{subject:02d}: no {roi} voxels — skip")
        return False

    # Load everything via fit_prf's helper. Then subset BOLD to ROI.
    t0 = time.time()
    print(f"  loading BOLD + paradigm (resolution={resolution})...")
    data_full, paradigm, grid_coords = load_concatenated(
        sub, masker, resolution, "full")
    print(f"  ...loaded in {time.time()-t0:.1f}s; "
          f"BOLD shape {data_full.shape}, paradigm shape {paradigm.shape}")
    bold_roi = data_full[:, voxel_idx].astype(np.float32)

    # Paradigm is binary {0, 1} — pack to uint8 to save space.
    par_u8 = paradigm.astype(np.uint8)

    out = cache_path(bids_folder, subject, roi, resolution)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        bold=bold_roi,
        paradigm=par_u8,
        grid_coords=grid_coords.astype(np.float32),
        voxel_idx=voxel_idx,
        hemi=hemi,
    )
    sz_mb = out.stat().st_size / 1024 / 1024
    print(f"  wrote {out}  ({sz_mb:.1f} MB; "
          f"{bold_roi.shape[1]} ROI voxels)")
    return True


def parse_subjects(spec: str):
    if "-" in spec:
        lo, hi = spec.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(s) for s in spec.split(",")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("subjects", help="range '1-30' or list '1,5,8'")
    ap.add_argument("--roi", default="V1")
    ap.add_argument("--resolution", type=int, default=50)
    ap.add_argument("--bids-folder", default="/data/ds-retsupp")
    args = ap.parse_args()

    bids = Path(args.bids_folder)
    for s in parse_subjects(args.subjects):
        out = cache_path(bids, s, args.roi, args.resolution)
        if out.exists():
            print(f"sub-{s:02d}: cache exists ({out.stat().st_size/1024/1024:.1f} MB) — skip")
            continue
        print(f"\n=== sub-{s:02d} === ")
        try:
            build(s, bids, roi=args.roi, resolution=args.resolution)
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
