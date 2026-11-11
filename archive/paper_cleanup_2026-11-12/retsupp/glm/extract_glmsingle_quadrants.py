"""Extract per-(subject, ROI, quadrant, trial) GLMSingle betas to TSV.

Slow step (loads ~5GB pe array per subject + masks) — run once per
data update. The plotter ``plot_glmsingle_quadrants.py`` then reads
the small TSV and renders fast.

Output: ``derivatives/glmsingle_summaries/quadrant_betas.tsv``
  Columns: subject, roi, quadrant, trial_idx, beta, condition,
           hp_relation, n_voxels.

  ``condition`` ∈ {target, distractor, neutral}
  ``hp_relation`` is HP-quadrant relative to Q (HP/orth/opposite),
  meaningful for distractor + target conditions; for neutral it's
  also computed (HP relative to Q on that run).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import image
from tqdm import tqdm

from retsupp.utils.data import Subject
from retsupp.glm.plot_glmsingle_quadrants import (
    LOC_LABEL, QUADRANTS, ROI_ORDER, ROI_ALIASES,
    QUADRANT_HEMI, hp_relation, load_quadrant_mask,
)


def extract_subject(subject: int, bids_folder: Path,
                    rois=ROI_ORDER):
    sub = Subject(subject, bids_folder)
    glm_dir = (bids_folder / 'derivatives' / 'glmsingle'
               / f'sub-{subject:02d}' / 'func')
    pe_nii = (glm_dir / f'sub-{subject:02d}_task-search_space-T1w_'
                          f'desc-distractor_pe.nii.gz')
    tr_tsv = (glm_dir / f'sub-{subject:02d}_task-search_space-T1w_'
                          f'desc-trials.tsv')
    if not pe_nii.exists() or not tr_tsv.exists():
        return None

    trials = pd.read_csv(tr_tsv, sep='\t')
    if 'target_location' not in trials.columns:
        target_locs = []
        for s in [1, 2]:
            for r in sub.get_runs(s):
                ons = sub.get_onsets(s, r)
                tgt = ons[ons['event_type'] == 'target'].sort_values('onset')
                target_locs += list(tgt.get('target_location', np.nan).values)
        if len(target_locs) >= len(trials):
            trials['target_location'] = target_locs[:len(trials)]
    trials['target_label'] = trials.get('target_location',
                                          pd.Series([np.nan])).map(LOC_LABEL)
    trials['distractor_label'] = trials['distractor_location'].map(LOC_LABEL)

    pe_img = nib.load(str(pe_nii))
    target_3d = nib.Nifti1Image(np.zeros(pe_img.shape[:3], dtype=np.int8),
                                 pe_img.affine)
    pe_arr = pe_img.get_fdata(dtype=np.float32)
    n_trials = pe_arr.shape[3]

    stim_rois_dir = bids_folder / 'derivatives' / 'stimulus_rois' \
                    / f'sub-{subject:02d}'

    rows = []
    for roi in rois:
        for quad in QUADRANTS:
            mask_img = load_quadrant_mask(stim_rois_dir, subject, roi, quad)
            if mask_img is None:
                continue
            mask_img = image.resample_to_img(
                mask_img, target_3d, interpolation='nearest',
                force_resample=True, copy_header=True,
            )
            mask = mask_img.get_fdata().astype(bool)
            n_vox = int(mask.sum())
            if n_vox < 5:
                continue
            per_trial = pe_arr[mask].mean(axis=0)[:n_trials]
            for t_idx in range(min(len(trials), n_trials)):
                row = trials.iloc[t_idx]
                tgt_q = row.get('target_label')
                dst_q = row.get('distractor_label')
                hp_q = row.get('hp_location')
                if pd.isna(hp_q) or not isinstance(hp_q, str):
                    rel = ''
                else:
                    rel = hp_relation(quad, hp_q)
                if tgt_q == quad and dst_q != quad:
                    cond = 'target'
                elif dst_q == quad and tgt_q != quad:
                    cond = 'distractor'
                elif tgt_q != quad and dst_q != quad:
                    cond = 'neutral'
                else:
                    continue
                rows.append({
                    'subject': subject, 'roi': roi, 'quadrant': quad,
                    'trial_idx': t_idx, 'beta': float(per_trial[t_idx]),
                    'condition': cond, 'hp_relation': rel,
                    'n_voxels': n_vox,
                })
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bids-folder', default='/data/ds-retsupp')
    p.add_argument('--subjects', type=int, nargs='+',
                   default=list(range(1, 31)))
    p.add_argument('--rois', nargs='+', default=ROI_ORDER)
    p.add_argument('--out', type=Path,
                   default=Path('/data/ds-retsupp/derivatives/glmsingle_summaries/quadrant_betas.tsv'))
    args = p.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    print(f'Output TSV: {args.out}')

    bids = Path(args.bids_folder)
    all_rows = []
    for s in tqdm(args.subjects, desc='subjects'):
        try:
            df = extract_subject(s, bids, rois=args.rois)
        except Exception as e:
            print(f'  sub-{s:02d}: failed ({e})'); continue
        if df is None or len(df) == 0:
            continue
        all_rows.append(df)
        print(f'  sub-{s:02d}: {len(df)} rows extracted')
    if not all_rows:
        raise RuntimeError('No subjects yielded data.')
    df = pd.concat(all_rows, ignore_index=True)
    df.to_csv(args.out, sep='\t', index=False)
    print(f'\nWrote: {args.out}  ({len(df)} rows, '
          f'{df.subject.nunique()} subjects)')


if __name__ == '__main__':
    main()
