"""Pre-build per-subject, per-ROI BOLD + paradigm cache.

Motivation
----------
``fit_prf.load_concatenated`` loads 12 cleaned-BOLD NIfTIs from disk
and runs each through a brain-mask masker (each transform: load
~600 MB gzipped volume + mask). For ROI analyses this is hugely
wasteful — we re-do all the IO every time even though we only care
about ~3000 voxels per ROI.

This script does the expensive load ONCE per subject and writes
**one small ``.npz`` per (subject, ROI)** — V1, V2, V3, V3AB, hV4,
LO, TO, VO, IPS, SPL1, FEF (all 11 ROIs used by the validity figure).
Each file ~5–30 MB; total per subject ~150 MB; ~4.5 GB for all 30
subjects across all ROIs.

File contents:
  - bold (T_total, V_roi) float32   — concatenated cleaned BOLD, ROI only
  - paradigm (T_total, G) uint8     — full bar+distractor paradigm tensor
  - grid_coords (G, 2) float32
  - voxel_idx (V_roi,) int32        — indices in brain-mask masker's
                                      flat output (so callers can map
                                      ROI voxels back into the full
                                      masker if they need to)
  - hemi (V_roi,) <U1               — 'L' or 'R' per voxel

CLI:
    python build_cleaned_roi_cache.py <subject>   [--resolution 50]
    python build_cleaned_roi_cache.py 1-30        # range, sequential

Output:
    {bids}/derivatives/cleaned_roi_cache/sub-XX/
        sub-XX_roi-{ROI}_res-{N}.npz
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


# All 11 retinotopic ROIs from the validity figure. Build one .npz per
# (subject, ROI) so loading is cheap.
ALL_ROIS = ["V1", "V2", "V3", "V3AB", "hV4", "LO", "TO", "VO",
            "IPS", "SPL1", "FEF"]

# Benson visual-area labels and Wang labels (mirrors data.py).
BENSON_LABELS = {1: "V1", 2: "V2", 3: "V3", 4: "hV4",
                 5: "VO1", 6: "VO2", 7: "LO1", 8: "LO2",
                 9: "TO1", 10: "TO2", 11: "V3A", 12: "V3B"}
WANG_LABELS = {1: "V1v", 2: "V1d", 3: "V2v", 4: "V2d",
               5: "V3v", 6: "V3d", 7: "hV4", 8: "VO1", 9: "VO2",
               10: "PHC1", 11: "PHC2", 12: "TO2", 13: "TO1",
               14: "LO2", 15: "LO1", 16: "V3B", 17: "V3A",
               18: "IPS0", 19: "IPS1", 20: "IPS2", 21: "IPS3",
               22: "IPS4", 23: "IPS5", 24: "SPL1", 25: "FEF"}
# ROI → list of label codes (per atlas). Compound ROIs union their
# components. IPS/SPL1/FEF live in Wang only; rest are Benson.
BENSON_ROI_LABELS = {
    "V1": [1], "V2": [2], "V3": [3], "hV4": [4],
    "VO": [5, 6], "LO": [7, 8], "TO": [9, 10],
    "V3AB": [11, 12],
}
WANG_ROI_LABELS = {
    "IPS": [18, 19, 20, 21, 22, 23],
    "SPL1": [24],
    "FEF": [25],
}


def cache_path(bids_folder: Path, subject: int, roi: str,
               resolution: int) -> Path:
    return (bids_folder / "derivatives" / "cleaned_roi_cache"
            / f"sub-{subject:02d}"
            / f"sub-{subject:02d}_roi-{roi}_res-{resolution}.npz")


def build(subject: int, bids_folder: Path, resolution: int = 50,
          rois=ALL_ROIS):
    """Load BOLD+paradigm once, write one .npz per ROI."""
    sub = Subject(subject, bids_folder)

    # Skip subject entirely if all output files already exist.
    if all(cache_path(bids_folder, subject, r, resolution).exists()
           for r in rois):
        print(f"  sub-{subject:02d}: all caches present — skip")
        return False

    # Brain-mask masker (matches fit_prf).
    first_run = sub.get_runs(1)[0]
    bold_mask = sub.get_bold_mask(session=1, run=first_run)
    masker = maskers.NiftiMasker(mask_img=bold_mask)
    masker.fit()
    bold_mask_flat = masker.mask_img_.get_fdata().astype(bool).ravel()

    # Per-voxel atlas labels (in masker-flat order).
    print("  resampling Benson + Wang atlases to BOLD space...")
    t0 = time.time()
    varea = sub.get_retinotopic_atlas(bold_space=True).get_fdata().astype(
        np.int8).ravel()[bold_mask_flat]
    wang = sub.get_wang_atlas(bold_space=True).get_fdata().astype(
        np.int8).ravel()[bold_mask_flat]
    lh = sub.get_hemisphere_mask("L", bold_space=True).get_fdata().astype(
        bool).ravel()[bold_mask_flat]
    rh = sub.get_hemisphere_mask("R", bold_space=True).get_fdata().astype(
        bool).ravel()[bold_mask_flat]
    print(f"  ...atlases ready in {time.time()-t0:.1f}s")

    # Load BOLD + paradigm ONCE — the expensive step.
    t0 = time.time()
    print(f"  loading BOLD + paradigm (resolution={resolution})...")
    data_full, paradigm, grid_coords = load_concatenated(
        sub, masker, resolution, "full")
    print(f"  ...loaded in {time.time()-t0:.1f}s "
          f"({data_full.shape} BOLD, {paradigm.shape} paradigm)")

    par_u8 = paradigm.astype(np.uint8)
    grid_f = grid_coords.astype(np.float32)

    # Per-ROI extraction + write.
    n_written = 0
    for roi in rois:
        out = cache_path(bids_folder, subject, roi, resolution)
        if out.exists():
            continue
        # Build a per-voxel mask for this ROI using the appropriate atlas.
        if roi in BENSON_ROI_LABELS:
            roi_mask = np.isin(varea, BENSON_ROI_LABELS[roi])
        elif roi in WANG_ROI_LABELS:
            roi_mask = np.isin(wang, WANG_ROI_LABELS[roi])
        else:
            print(f"  unknown roi '{roi}' — skip")
            continue
        l_idx = np.where(roi_mask & lh)[0]
        r_idx = np.where(roi_mask & rh)[0]
        voxel_idx = np.concatenate([l_idx, r_idx]).astype(np.int32)
        if len(voxel_idx) == 0:
            print(f"  {roi}: no voxels — skip")
            continue
        hemi = np.concatenate([
            np.full(len(l_idx), "L", dtype="<U1"),
            np.full(len(r_idx), "R", dtype="<U1"),
        ])
        bold_roi = data_full[:, voxel_idx].astype(np.float32)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, bold=bold_roi, paradigm=par_u8,
                            grid_coords=grid_f, voxel_idx=voxel_idx,
                            hemi=hemi)
        sz_mb = out.stat().st_size / 1024 / 1024
        print(f"  {roi:6s}: {len(voxel_idx):>4d} voxels  "
              f"→ {sz_mb:.1f} MB")
        n_written += 1
    print(f"  sub-{subject:02d}: wrote {n_written}/{len(rois)} ROIs")
    return n_written > 0


def parse_subjects(spec: str):
    if "-" in spec:
        lo, hi = spec.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(s) for s in spec.split(",")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("subjects", help="range '1-30' or list '1,5,8'")
    ap.add_argument("--resolution", type=int, default=50)
    ap.add_argument("--bids-folder", default="/data/ds-retsupp")
    args = ap.parse_args()

    bids = Path(args.bids_folder)
    for s in parse_subjects(args.subjects):
        print(f"\n=== sub-{s:02d} === ")
        try:
            build(s, bids, resolution=args.resolution)
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
