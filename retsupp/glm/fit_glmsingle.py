#!/usr/bin/env python3
"""
Fit GLMsingle single-trial betas for the retsupp visual-search task.

Design matrix
-------------
One event is modelled per trial: the target/search-array onset (the
`target` event in the events.tsv). The condition label encodes the
DISTRACTOR LOCATION on that trial (1.0=upper_right, 3.0=upper_left,
5.0=lower_left, 7.0=lower_right, 10.0=no distractor — see
`Subject.location_mapping`).

Each unique distractor location gets its own column in the design
matrix, shared consistently across all runs and sessions. This is
critical for GLMsingle's fracridge cross-validation: GLMsingle
identifies conditions by design-matrix column index, so assigning the
same distractor location the same column in every run ensures that
same-location trials are grouped together when selecting the ridge
regularisation parameter alpha via leave-one-run-out cross-validation.

GLMsingle then expands the condition-level design matrix internally to
a single-trial design, yielding one beta per trial presentation per
voxel.

Output
------
Two 4-D NIfTI images are saved (x × y × z × n_trials), plus per-trial
metadata:

  desc-distractor_pe.nii.gz  — single-trial betas at target/search-array onset
  desc-R2_pe.nii.gz          — cross-validated R² of the final GLMsingle model
  desc-trials.tsv            — per-trial labels: subject, session, run, trial,
                                onset, distractor_location, distractor_label
                                (e.g. upper_right), is_hp_distractor, ...

For downstream analysis (Richter 2025-style), pair the trial betas with
the per-trial metadata to compute mean BOLD per (location ROI ×
stimulus type × HP-status × subject).

Usage
-----
    python -m retsupp.glm.fit_glmsingle 5
    python -m retsupp.glm.fit_glmsingle 5 --sessions 1
    python -m retsupp.glm.fit_glmsingle 5 --bids-folder /data/ds-retsupp
    python -m retsupp.glm.fit_glmsingle 5 --debug   # write all 4 model steps + figures
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from nilearn import image
from scipy.interpolate import interp1d

from retsupp.utils.data import Subject

warnings.filterwarnings("ignore")

TR = 1.6          # retsupp acquisition TR
TR_UP = 0.4       # upsampled TR (factor 4)
STIM_DUR = 1.0    # search-array effective duration; not critical (GLMsingle is robust)


def upsample_bold(bold_4d: np.ndarray, factor: float) -> np.ndarray:
    """Linearly upsample a (x, y, z, t) array along the time axis by `factor`."""
    x, y, z, t = bold_4d.shape
    t_orig = np.arange(t, dtype=np.float64)
    n_new = int(round(t * factor))
    t_new = np.linspace(0, t - 1, n_new)
    flat = bold_4d.reshape(-1, t).astype(np.float32)
    up = interp1d(
        t_orig, flat, axis=1, kind="linear",
        fill_value="extrapolate", assume_sorted=True,
    )(t_new)
    return up.reshape(x, y, z, n_new)


def make_condition_label(row) -> str:
    """Map a target event row to its condition label (= distractor location)."""
    loc = row["distractor_location"]
    if pd.isna(loc):
        return "distractor_nan"
    return f"distractor_{loc:.1f}"


def build_condition_index(all_events) -> dict[str, int]:
    """{condition_label: column_index} — globally consistent across all runs.

    Iterates only over TARGET events (one per trial) so the condition
    set doesn't include NaN-distractor labels from unrelated event types
    (pulse, instruction, etc.) that have no distractor_location.
    """
    conditions = set()
    for ev in all_events:
        target_ev = ev[ev["event_type"] == "target"]
        if len(target_ev) == 0:
            continue
        conditions.update(target_ev.apply(make_condition_label, axis=1).unique())
    # Stable ordering: by the float code, NaN last.
    def _key(c):
        suffix = c.replace("distractor_", "")
        if suffix == "nan":
            return float("inf")
        return float(suffix)
    return {c: i for i, c in enumerate(sorted(conditions, key=_key))}


def build_design_matrix(events_run, n_vols, condition_to_idx, tr=TR_UP):
    """Return (dm, trial_meta) for one run.

    dm         : binary (n_vols × n_conditions) array, 1 at the upsampled-TR row
                 nearest the target onset.
    trial_meta : list of dicts in onset-time order, with keys:
                   condition (label), distractor_location (float),
                   distractor_label (str), onset (float, sec).
                 Order matches GLMsingle's single-trial beta ordering.
    """
    ev = events_run.reset_index(drop=True).copy()
    ev = ev[ev["event_type"] == "target"].sort_values("onset").reset_index(drop=True)
    ev["condition"] = ev.apply(make_condition_label, axis=1)

    # Map distractor codes to spatial labels (for downstream readability).
    # 1.0 → upper_right, 3.0 → upper_left, 5.0 → lower_left, 7.0 → lower_right,
    # 10.0/NaN → no distractor.  Matches Subject.location_mapping.
    code_to_label = {1.0: "upper_right", 3.0: "upper_left",
                     5.0: "lower_left", 7.0: "lower_right",
                     10.0: "no_distractor"}

    dm = np.zeros((n_vols, len(condition_to_idx)))
    trial_meta = []
    for _, row in ev.iterrows():
        onset_tr = int(np.round(row["onset"] / tr))
        col = condition_to_idx[row["condition"]]
        dm[min(onset_tr, n_vols - 1), col] = 1.0
        loc = row["distractor_location"]
        trial_meta.append({
            "condition": row["condition"],
            "distractor_location": loc if not pd.isna(loc) else np.nan,
            "distractor_label": code_to_label.get(loc, "no_distractor")
                                if not pd.isna(loc) else "no_distractor",
            "onset": float(row["onset"]),
        })
    return dm, trial_meta


def main(
    subject: int,
    sessions=None,
    bids_folder: str = "/data/ds-retsupp",
    bold_type: str = "fmriprep",
    debug: bool = False,
    smoothed: bool = False,
):
    sub = Subject(subject, bids_folder=bids_folder)

    if sessions is None:
        sessions = [1, 2]

    is_single_session = len(sessions) == 1
    ses_entity = f"_ses-{sessions[0]}" if is_single_session else ""
    print(
        f"sub-{int(subject):02d}  "
        f"{'ses-' + str(sessions[0]) if is_single_session else 'all-sessions'}  "
        f"[bold={bold_type}]"
    )

    # First pass: collect all events to build a globally consistent condition map.
    session_run_events = {}
    for session in sessions:
        runs = sub.get_runs(session)
        events_per_run = {run: sub.get_onsets(session, run) for run in runs}
        session_run_events[session] = (runs, events_per_run)

    condition_to_idx = build_condition_index(
        events_per_run[run]
        for session, (runs, events_per_run) in session_run_events.items()
        for run in runs
    )
    print(f"  {len(condition_to_idx)} conditions: {list(condition_to_idx)}")

    # HP location per (session, run) — consistent across all trials in that run.
    hpd_locations = sub.get_hpd_locations()
    print(f"  HP locations per (session, run): {hpd_locations}")

    # Second pass: load BOLD and build design matrices.
    data = []
    X = []
    all_trial_meta = []
    session_indicators = []
    ref_bold_img = None

    for session in sessions:
        runs, events_per_run = session_run_events[session]
        print(f"  ses-{session}: {len(runs)} runs: {runs}")

        for run in runs:
            bold_path = sub.get_bold(session=session, run=run, type=bold_type, return_image=False)
            if ref_bold_img is None:
                ref_bold_img = image.load_img(str(bold_path))

            img = (
                image.smooth_img(str(bold_path), fwhm=5.0) if smoothed
                else image.load_img(str(bold_path))
            )
            bold_data = img.get_fdata()
            n_vols = bold_data.shape[3]

            upsample_factor = TR / TR_UP
            bold_data = upsample_bold(bold_data, upsample_factor)
            n_vols_up = bold_data.shape[3]

            dm, trial_meta = build_design_matrix(
                events_per_run[run], n_vols_up, condition_to_idx, tr=TR_UP,
            )
            # Annotate per-trial HP status.
            hp_label = hpd_locations.get((session, run), "no distractor")
            hp_label_underscore = hp_label.replace(" ", "_") if hp_label else "no_distractor"
            for tm in trial_meta:
                tm["subject"] = int(subject)
                tm["session"] = session
                tm["run"] = run
                tm["hp_location"] = hp_label_underscore
                tm["is_hp_distractor"] = (tm["distractor_label"] == hp_label_underscore)

            data.append(bold_data)
            X.append(dm)
            all_trial_meta.extend(trial_meta)
            # GLMsingle uses the session indicator to handle session-to-session
            # scaling differences — index sessions starting at 1.
            session_indicators.append(session)
            print(
                f"    run-{run}: {n_vols} vols → {n_vols_up} upsampled, "
                f"{len(trial_meta)} trials, dm shape {dm.shape}"
            )

    opt = dict(
        wantlibrary=1,
        wantglmdenoise=1,
        wantfracridge=1,
        wantfileoutputs=[1, 1, 1, 1] if debug else [0, 0, 0, 1],
        sessionindicator=np.array(session_indicators)[np.newaxis, :],
        n_pcs=20,
    )

    glmsingle_deriv = "glmsingle.smoothed" if smoothed else "glmsingle"
    sub_dir = Path(bids_folder) / "derivatives" / glmsingle_deriv / f"sub-{int(subject):02d}"
    out_dir = sub_dir / (f"ses-{sessions[0]}" if is_single_session else "") / "func"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir.parent / "figures"

    from glmsingle.glmsingle import GLM_single
    print(f"Fitting GLMsingle ({len(X)} runs, {sum(x.shape[0] for x in X)} upsampled TRs total)...")
    results = GLM_single(opt).fit(
        X, data, STIM_DUR, TR_UP,
        outputdir=str(out_dir),
        figuredir=str(fig_dir),
    )

    betas = results["typed"]["betasmd"]  # (x, y, z, n_total_trials)
    n_trials = betas.shape[-1]
    assert n_trials == len(all_trial_meta), (
        f"Beta count {n_trials} != trial count {len(all_trial_meta)}"
    )

    fn_template = (
        f"sub-{int(subject):02d}{ses_entity}_task-search"
        f"_space-T1w_desc-{{desc}}_pe.nii.gz"
    )
    image.new_img_like(ref_bold_img, betas).to_filename(
        str(out_dir / fn_template.format(desc="distractor"))
    )
    image.new_img_like(ref_bold_img, results["typed"]["R2"]).to_filename(
        str(out_dir / fn_template.format(desc="R2"))
    )

    # Per-trial metadata as TSV. Order matches the 4th dimension of the betas.
    meta_df = pd.DataFrame(all_trial_meta)
    meta_df["trial_index"] = np.arange(n_trials)
    meta_tsv = out_dir / fn_template.format(desc="trials").replace("_pe.nii.gz", ".tsv")
    meta_df.to_csv(meta_tsv, sep="\t", index=False)

    print(f"\nSaved to {out_dir}:")
    print(f"  betas:    sub-..._desc-distractor_pe.nii.gz   ({betas.shape})")
    print(f"  R2:       sub-..._desc-R2_pe.nii.gz")
    print(f"  trials:   sub-..._desc-trials.tsv             ({len(meta_df)} rows)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("subject", type=int, help="Subject number, e.g. 5 or 24")
    parser.add_argument(
        "--sessions", type=int, nargs="+", default=None,
        help="Session number(s) to fit. Default: both sessions.",
    )
    parser.add_argument("--bids-folder", default="/data/ds-retsupp")
    parser.add_argument(
        "--bold-type", default="fmriprep",
        choices=["fmriprep", "cleaned"],
        help="Which preprocessed BOLD to use. Default fmriprep (raw, GLMsingle "
             "handles confound regression internally via glmdenoise).",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Write outputs and diagnostic figures for all 4 GLMsingle steps.",
    )
    parser.add_argument(
        "--smoothed", action="store_true",
        help="Spatially smooth BOLD with a 5mm FWHM kernel before fitting. "
             "Outputs go to derivatives/glmsingle.smoothed/.",
    )
    args = parser.parse_args()

    main(
        args.subject,
        sessions=args.sessions,
        bids_folder=args.bids_folder,
        bold_type=args.bold_type,
        debug=args.debug,
        smoothed=args.smoothed,
    )
