"""Joint AF + PRF fit on a SLICE of runs (first or last per HP block).

Same model as :mod:`retsupp.modeling.fit_af_prf_braincoder` — joint
attention-field + PRF fit at the BOLD level using
``AttentionFieldPRF2DWithHRF`` in 'signed' mode, full paradigm
(extended grid + distractor disks), model-4 mean-PRF init — but
restricted to a subset of runs identified by their position within
each HP-distractor block.

Motivation
----------
The retsupp paradigm groups runs into blocks of consecutive runs that
share the same high-probability (HP) distractor location. Subjects
LEARN that HP statistic over the course of a block. We expect HP
suppression (g_HP < g_LP in 'signed' mode) to be WEAKER on the FIRST
run of each block (haven't learned yet) and STRONGER on the LAST run
(have learned). This script fits separately on the first / last runs
to test that.

A "block" is a maximal run of consecutive (session, run) pairs
sharing the same HP location. The walk order is:
``ses-1 run-1, ses-1 run-2, ..., ses-1 run-N, ses-2 run-1, ...``,
skipping runs not present in :meth:`Subject.get_runs`. Blocks span
session boundaries iff the HP carries over (which the design avoids,
but the logic is robust either way). If a block contains a single
run, that run is BOTH first and last.

Output
------
Pickle + TSV under
``derivatives/af_prf_joint_runslice_{first,last}/sub-XX/...`` with the
suffix ``_runslice-{first,last}_af-prf-pars.tsv`` /
``_af-prf-fit.pkl``.

Usage
-----
``python -m retsupp.modeling.fit_af_prf_runslice 5 --roi V3AB --run-slice first``
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from nilearn import image, input_data
from tqdm import tqdm

from braincoder.hrf import SPMHRFModel
from braincoder.models import AttentionFieldPRF2DWithHRF
from braincoder.optimize import ParameterFitter
from retsupp.utils.data import (
    Subject,
    distractor_locations,
)


CONDITIONS = ['upper_right', 'upper_left', 'lower_left', 'lower_right']


def get_ring_positions():
    """Return the (n_C, 2) array of ring positions matching CONDITIONS order."""
    keys = ['upper right', 'upper left', 'lower left', 'lower right']
    return np.array([list(distractor_locations[k]) for k in keys], dtype=np.float32)


def identify_blocks(sub: Subject):
    """Walk all (session, run) pairs and group consecutive same-HP runs.

    Returns
    -------
    blocks : list of list of (session, run)
        Each inner list is a contiguous block of runs sharing the same
        HP location. Walk order is ses-1 then ses-2, runs in the order
        returned by :meth:`Subject.get_runs`. Runs whose HP is missing
        or not in :data:`CONDITIONS` (e.g. 'no distractor') still
        contribute to the walk for block boundary purposes — a change
        of HP starts a new block — but they are kept in the block (so
        callers must filter by the same CONDITIONS check downstream
        when actually loading data).
    hp_per_run : dict
        The full HP-per-run map, for reference.
    """
    hp_per_run = sub.get_hpd_locations()

    walk = []
    for session in [1, 2]:
        for run in sub.get_runs(session):
            walk.append((session, run))

    blocks = []
    current_block = []
    current_hp = object()  # sentinel — will mismatch on first iter
    for sr in walk:
        hp = hp_per_run.get(sr)
        if hp != current_hp:
            if current_block:
                blocks.append(current_block)
            current_block = [sr]
            current_hp = hp
        else:
            current_block.append(sr)
    if current_block:
        blocks.append(current_block)

    return blocks, hp_per_run


def select_run_slice(sub: Subject, run_slice: str):
    """Return the list of (session, run) pairs to keep for the slice.

    Parameters
    ----------
    sub : Subject
    run_slice : {'first', 'last'}
        'first' -> keep the first run of each block; 'last' -> keep
        the last. If a block has a single run, both first and last
        return that run.

    Returns
    -------
    keep : set of (session, run)
    blocks : list of list of (session, run)
        The block grouping (for logging).
    hp_per_run : dict
    """
    if run_slice not in ('first', 'last'):
        raise ValueError(
            f"run_slice must be 'first' or 'last', got {run_slice!r}"
        )
    blocks, hp_per_run = identify_blocks(sub)
    keep = set()
    for block in blocks:
        if run_slice == 'first':
            keep.add(block[0])
        else:
            keep.add(block[-1])
    return keep, blocks, hp_per_run


def build_data_and_paradigm(sub: Subject, keep: set,
                            resolution: int = 40,
                            grid_radius: float = 5.0):
    """Load cleaned BOLD + paradigm + condition_indicator for one subject,
    restricted to the runs in ``keep``.

    Mirrors the 'full' paradigm path of
    :mod:`retsupp.modeling.fit_af_prf_braincoder`: extended grid +
    distractor disks via :meth:`Subject.get_stimulus_with_distractors`.

    Returns
    -------
    bold_df : DataFrame, shape (n_T_total, n_voxels)
    paradigm : ndarray, shape (n_T_total, n_grid)
    condition_indicator : ndarray, shape (n_T_total, n_conditions)
    grid_coordinates : ndarray, shape (n_grid, 2)
    masker : NiftiMasker
    runs_used : list of (session, run, hp)
        The runs that actually contributed BOLD (i.e. ``keep`` minus
        any whose HP is not in CONDITIONS).
    """
    bold_mask = sub.get_bold_mask()
    masker = input_data.NiftiMasker(mask_img=bold_mask)

    # Extended grid coordinates (shared across runs).
    gx, gy = sub.get_extended_grid_coordinates(
        resolution=resolution, session=1, run=1,
        grid_radius=grid_radius,
    )
    grid_coordinates = np.stack(
        (gx.ravel(), gy.ravel()), axis=1,
    ).astype(np.float32)

    hp_per_run = sub.get_hpd_locations()
    print('HP per run:', hp_per_run)

    bold_chunks = []
    paradigm_chunks = []
    cond_indicator_chunks = []
    masker.fit(bold_mask)

    runs_used = []
    session_runs = [(s, r) for s in [1, 2] for r in sub.get_runs(s)]
    for session, run in tqdm(session_runs, desc='Loading runs'):
        if (session, run) not in keep:
            continue
        hp = hp_per_run[(session, run)]
        if hp not in CONDITIONS:
            print(f'  ses-{session}_run-{run}: HP={hp!r} not in CONDITIONS, '
                  f'skipping (slice run with no usable HP)')
            continue

        par_run = sub.get_stimulus_with_distractors(
            session=session, run=run, resolution=resolution,
            grid_radius=grid_radius,
        ).astype(np.float32)
        par_run_flat = par_run.reshape((par_run.shape[0], -1))
        n_T_run = par_run_flat.shape[0]

        bold_fn = (
            sub.bids_folder / 'derivatives' / 'cleaned'
            / f'sub-{sub.subject_id:02d}' / f'ses-{session}' / 'func'
            / f'sub-{sub.subject_id:02d}_ses-{session}_task-search_'
              f'desc-cleaned_run-{run}_bold.nii.gz'
        )
        data = masker.transform(bold_fn).astype(np.float32)
        if data.shape[0] < n_T_run:
            print(f'  ses-{session}_run-{run}: short by '
                  f'{n_T_run - data.shape[0]} TRs, padding with zeros')
            pad = np.zeros((n_T_run - data.shape[0], data.shape[1]),
                           dtype=np.float32)
            data = np.vstack([data, pad])
        elif data.shape[0] > n_T_run:
            data = data[:n_T_run]

        hp_idx = CONDITIONS.index(hp)
        cond_indicator = np.zeros((n_T_run, len(CONDITIONS)),
                                  dtype=np.float32)
        cond_indicator[:, hp_idx] = 1.0

        bold_chunks.append(data)
        paradigm_chunks.append(par_run_flat)
        cond_indicator_chunks.append(cond_indicator)
        runs_used.append((session, run, hp))

    if not bold_chunks:
        raise RuntimeError(
            'No usable runs in the requested slice — every selected run '
            'had HP not in CONDITIONS.'
        )

    bold = np.vstack(bold_chunks)
    paradigm_full = np.vstack(paradigm_chunks)
    condition_indicator = np.vstack(cond_indicator_chunks)

    print(f'Loaded BOLD: shape {bold.shape}, paradigm {paradigm_full.shape}, '
          f'condition_indicator {condition_indicator.shape}')
    print(f'Runs used: {runs_used}')
    return (
        pd.DataFrame(bold),
        paradigm_full,
        condition_indicator,
        grid_coordinates,
        masker,
        runs_used,
    )


def select_roi_voxels(sub: Subject, roi: str, prf_pars: pd.DataFrame,
                       r2_thr: float = 0.05):
    """Return a boolean mask over voxels selecting `roi` voxels with PRF R² > thr.

    Some "ROIs" combine multiple atlas labels (e.g. V3AB = V3A ∪ V3B,
    LO = LO1 ∪ LO2, TO = TO1 ∪ TO2, VO = VO1 ∪ VO2). We expand these
    aliases here, matching the original fit_af_prf_braincoder logic.
    """
    roi_aliases = {
        'V3AB': ['V3A', 'V3B'],
        'LO': ['LO1', 'LO2'],
        'TO': ['TO1', 'TO2'],
        'VO': ['VO1', 'VO2'],
    }
    component_rois = roi_aliases.get(roi, [roi])

    masker_full = sub.get_bold_mask(return_masker=True)
    masker_full.fit()
    roi_arr = np.zeros(prf_pars.shape[0], dtype=bool)
    for r in component_rois:
        roi_img = sub.get_retinotopic_roi(roi=r, bold_space=True)
        roi_arr |= masker_full.transform(roi_img).astype(bool).flatten()
    r2_mask = (prf_pars['r2'].values > r2_thr) & roi_arr
    return r2_mask


def main(subject: int, bids_folder: str = '/data/ds-retsupp',
         roi: str = 'V3AB',
         run_slice: str = 'first',
         resolution: int = 50,
         max_n_iterations: int = 1500,
         r2_thr: float = 0.05,
         model_label: int = 4,
         max_voxels: int | None = None,
         mode: str = 'signed',
         learning_rate: float = 0.01,
         grid_radius: float = 5.0,
         output_subdir: str | None = None):
    bids_folder = Path(bids_folder)
    sub = Subject(subject, bids_folder)
    if output_subdir is None:
        output_subdir = f'af_prf_joint_runslice_{run_slice}'
    out_dir = bids_folder / 'derivatives' / output_subdir / f'sub-{subject:02d}'
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'== sub-{subject:02d} | roi={roi} | mode={mode} | '
          f'run-slice={run_slice} ==')

    # 1) Identify runs to keep.
    keep, blocks, hp_per_run = select_run_slice(sub, run_slice)
    print('Block grouping:')
    for b in blocks:
        block_hp = hp_per_run.get(b[0])
        print(f'  block (HP={block_hp}): {b}')
    print(f'  -> keeping ({run_slice}): {sorted(keep)}')

    # 2) Load BOLD + paradigm + condition_indicator across kept runs.
    bold_df, paradigm, condition_indicator, grid_coords, masker, runs_used = (
        build_data_and_paradigm(
            sub, keep,
            resolution=resolution,
            grid_radius=grid_radius,
        )
    )

    # 3) Restrict to ROI voxels with decent mean-model PRF R².
    prf_pars = sub.get_prf_parameters_volume(model=model_label, return_images=False)
    if not isinstance(prf_pars, pd.DataFrame):
        prf_pars = pd.DataFrame(prf_pars)
    voxel_mask = select_roi_voxels(sub, roi, prf_pars, r2_thr=r2_thr)
    print(f'ROI {roi} | r2>{r2_thr}: {voxel_mask.sum()} voxels')
    if voxel_mask.sum() == 0:
        raise RuntimeError(f'No voxels survive: ROI={roi}, r2>{r2_thr}.')

    if max_voxels is not None and voxel_mask.sum() > max_voxels:
        order = np.argsort(prf_pars['r2'].values)[::-1]
        keep_v = np.zeros_like(voxel_mask)
        ranked = [v for v in order if voxel_mask[v]][:max_voxels]
        keep_v[ranked] = True
        voxel_mask = keep_v
        print(f'  -> top {max_voxels} voxels by r²')

    bold_sub = bold_df.loc[:, voxel_mask].copy()
    init_pars = prf_pars.loc[voxel_mask, ['x', 'y', 'sd', 'amplitude', 'baseline']].copy()
    init_pars = init_pars[['x', 'y', 'sd', 'baseline', 'amplitude']]
    init_pars['sigma_AF'] = 2.0
    init_pars['g_HP'] = 0.0 if mode == 'signed' else 0.30
    init_pars['g_LP'] = 0.0 if mode == 'signed' else 0.10

    # 4) Build the AF+PRF model and the fitter.
    ring_positions = get_ring_positions()
    print('Ring positions:\n', ring_positions)

    hrf_model = SPMHRFModel(tr=sub.get_tr(session=1, run=1), delay=4.5, dispersion=0.75)

    model = AttentionFieldPRF2DWithHRF(
        grid_coordinates=grid_coords,
        paradigm=paradigm,
        hrf_model=hrf_model,
        condition_indicator=condition_indicator,
        ring_positions=ring_positions,
        mode=mode,
    )

    fitter = ParameterFitter(model, bold_sub, paradigm)

    refined_pars = fitter.refine_baseline_and_amplitude(init_pars, l2_alpha=1e-3)

    fit_pars = fitter.fit(
        init_pars=refined_pars,
        max_n_iterations=max_n_iterations,
        shared_pars=['sigma_AF', 'g_HP', 'g_LP'],
        learning_rate=learning_rate,
    )

    r2 = fitter.get_rsq(fit_pars) if hasattr(fitter, 'get_rsq') else fitter.r2
    print(f'Mean R²: {np.nanmean(r2):.4f}')
    print('Shared parameters:')
    print(fit_pars[['sigma_AF', 'g_HP', 'g_LP']].iloc[0])

    fit_pars['r2'] = r2.values if hasattr(r2, 'values') else r2

    # 5) Save outputs.
    suffix = f'runslice-{run_slice}'
    out_tsv = out_dir / (
        f'sub-{subject:02d}_roi-{roi}_mode-{mode}_{suffix}_af-prf-pars.tsv'
    )
    fit_pars.to_csv(out_tsv, sep='\t')
    out_pkl = out_dir / (
        f'sub-{subject:02d}_roi-{roi}_mode-{mode}_{suffix}_af-prf-fit.pkl'
    )
    with open(out_pkl, 'wb') as f:
        pickle.dump({
            'fit_pars': fit_pars,
            'r2': r2,
            'shared_pars': fit_pars[['sigma_AF', 'g_HP', 'g_LP']].iloc[0].to_dict(),
            'voxel_mask_indices': np.where(voxel_mask)[0],
            'mode': mode,
            'roi': roi,
            'resolution': resolution,
            'subject': subject,
            'paradigm_type': 'full',
            'grid_radius': grid_radius,
            'run_slice': run_slice,
            'runs_used': runs_used,
            'blocks': blocks,
            'hp_per_run': hp_per_run,
        }, f)
    print(f'Saved: {out_tsv}')
    print(f'Saved: {out_pkl}')

    return fit_pars, r2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('subject', type=int, help='Subject ID (e.g. 5 -> sub-05)')
    parser.add_argument('--bids-folder', default='/data/ds-retsupp')
    parser.add_argument('--roi', default='V3AB',
                        help='Retinotopic ROI to fit (default V3AB).')
    parser.add_argument('--run-slice', choices=['first', 'last'], default='first',
                        help="Which run within each HP-block to keep. "
                             "'first' = first run of each block (4 blocks "
                             "= 4 runs typically), 'last' = last run of "
                             "each block.")
    parser.add_argument('--resolution', type=int, default=50)
    parser.add_argument('--max-n-iterations', type=int, default=1500)
    parser.add_argument('--r2-thr', type=float, default=0.05)
    parser.add_argument('--model-label', type=int, default=4,
                        help='Mean-model PRF used for x/y/sd/amplitude/'
                             'baseline initialization.')
    parser.add_argument('--max-voxels', type=int, default=0,
                        help='Cap on voxels (0 = no cap; default 0).')
    parser.add_argument('--mode',
                        choices=['suppression', 'attraction', 'signed'],
                        default='signed')
    parser.add_argument('--learning-rate', type=float, default=0.01)
    parser.add_argument('--grid-radius', type=float, default=5.0)
    parser.add_argument('--output-subdir', default=None,
                        help='Output derivatives subdir. Default: '
                             "'af_prf_joint_runslice_{first,last}'.")
    args = parser.parse_args()
    main(
        args.subject,
        bids_folder=args.bids_folder,
        roi=args.roi,
        run_slice=args.run_slice,
        resolution=args.resolution,
        max_n_iterations=args.max_n_iterations,
        r2_thr=args.r2_thr,
        model_label=args.model_label,
        max_voxels=None if args.max_voxels == 0 else args.max_voxels,
        mode=args.mode,
        learning_rate=args.learning_rate,
        grid_radius=args.grid_radius,
        output_subdir=args.output_subdir,
    )
