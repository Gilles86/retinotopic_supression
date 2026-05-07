#!/usr/bin/env python3
"""
Fit the 4-AF competing model per subject per ROI and save results to TSV.

Two fits per subject × ROI × mode (suppression / attraction):
    R(x) = (1 ± g_HP · A_HP(x) ± g_LP · Σ_{ℓ ≠ HP} A_ℓ(x)) · S_v(x)

Free params per fit:
    σ_AF                 ∈ [0.5, 8] deg  (shared)
    g_HP                 ∈ [0.001, 2]    (shared)
    g_LP                 ∈ [0.001, 2]    (shared)
    (x0_v, y0_v) per voxel — fitted via iterative residual updates +
                              optional final scipy L-BFGS-B minimization.

Inclusion: voxels with `r2_mean_model > --min-r2` (default 0.3) AND that
pass the aperture filter (≤50% of PRF mass outside the aperture).

Output: TSV with one row per (subject, roi, mode) and columns:
    sigma_AF, g_HP, g_LP, g_HP_over_g_LP, log_g_HP_over_g_LP, r2,
    loss, success, n_voxels.

Usage:
    python -m retsupp.modeling.fit_four_af \\
        --bids-folder /data/ds-retsupp \\
        --out notes/four_af_fits.tsv \\
        --min-r2 0.3 \\
        --joint-base-fit
"""
from __future__ import annotations

import argparse
from pathlib import Path

from retsupp.visualize.vss2026_arrows import load_all_conditionwise
from retsupp.visualize.meeting_report import filter_prf_inside_aperture
from retsupp.modeling.af_model import fit_four_af_per_subject


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bids-folder", default="/data/ds-retsupp")
    parser.add_argument("--model", type=int, default=8,
                        help="Conditionwise PRF model index (default 8 = DoG flexible HRF)")
    parser.add_argument("--out", type=Path,
                        default=Path("notes/four_af_fits.tsv"))
    parser.add_argument("--rois", nargs="+",
                        default=["V1", "V2", "V3", "V3AB", "hV4", "LO", "TO", "VO"])
    parser.add_argument("--min-r2", type=float, default=0.3,
                        help="Drop voxels with r2_mean_model below this (default 0.3)")
    parser.add_argument("--aperture-radius", type=float, default=3.17,
                        help="Stimulus aperture radius in degrees")
    parser.add_argument("--max-frac-outside", type=float, default=0.5,
                        help="Drop voxels with > this fraction of any conditionwise PRF outside aperture")
    parser.add_argument("--n-iter", type=int, default=3,
                        help="Iterations of residual-based base correction")
    parser.add_argument("--joint-base-fit", action="store_true",
                        help="Final per-voxel scipy L-BFGS-B fit of (x0, y0) given fixed shared params")
    parser.add_argument("--modes", nargs="+",
                        default=["suppression", "attraction"],
                        choices=["suppression", "attraction"])
    args = parser.parse_args()

    print(f"Loading conditionwise data (model={args.model}) from {args.bids_folder} ...")
    pars = load_all_conditionwise(bids_folder=args.bids_folder, model=args.model)
    print(f"  {len(pars):,} rows, {pars['subject'].nunique()} subjects")

    print(f"Aperture filter (drop voxels with any PRF >{args.max_frac_outside:.0%} "
          f"outside R={args.aperture_radius}°)...")
    n_before = len(pars)
    pars = filter_prf_inside_aperture(
        pars, aperture_radius=args.aperture_radius,
        max_fraction_outside=args.max_frac_outside, use_mean_only=False,
    )
    print(f"  {n_before:,} → {len(pars):,} rows ({100*(n_before-len(pars))/n_before:.1f}% dropped)")

    print(f"Fitting 4-AF model "
          f"({'+'.join(args.modes)}) per subject per ROI:")
    print(f"  ROIs: {args.rois}")
    print(f"  min_r2_mean_model = {args.min_r2}")
    print(f"  refit_base = True, n_iter = {args.n_iter}")
    print(f"  joint_base_fit = {args.joint_base_fit}")

    df = fit_four_af_per_subject(
        pars, args.rois,
        modes=tuple(args.modes),
        min_r2_mean_model=args.min_r2,
        refit_base=True, n_iter=args.n_iter,
        joint_base_fit=args.joint_base_fit,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, sep="\t", index=False)
    print(f"\nWrote {args.out} ({len(df)} rows)")
    print()
    # Per (mode, ROI) summary.
    for mode in args.modes:
        print(f"=== {mode} ===")
        sub = df[df["mode"] == mode]
        if sub.empty:
            continue
        grp = sub.groupby("roi", observed=True)
        summary = grp.agg(
            n_sub=("subject", "size"),
            R2_med=("r2", "median"),
            sigma_AF_med=("sigma_AF", "median"),
            g_HP_med=("g_HP", "median"),
            g_LP_med=("g_LP", "median"),
            log_ratio_med=("log_g_HP_over_g_LP", "median"),
        )
        summary = summary.reindex(args.rois)
        print(summary.to_string(float_format=lambda v: f"{v:7.4f}"))
        print()


if __name__ == "__main__":
    main()
