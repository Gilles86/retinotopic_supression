"""GLMSingle: per-trial beta by what landed at each quadrant.

Uses the **stimulus-ROIs** built by ``retsupp/eccentric_glm/get_rois.py``
(voxels whose Benson-PRF lies within ±1.5° of a ring position).
For each subject × ROI × quadrant Q:

  - mean per-trial beta across ROI×Q voxels  →  (n_trials,) timeseries
  - classify each trial by what landed at Q:
       'target'      target_location == Q
       'distractor'  distractor_location == Q
       'neutral'     neither at Q (target+distractor at OTHER rings,
                     or no distractor + target elsewhere)

Then COLLAPSE across the 4 quadrants (UR, UL, LL, LR) for each
(subject, ROI). Group: average across subjects.

Optional split by HP-condition relative to the quadrant Q (the run's
HP same/different quadrant) to inspect HP-specific suppression of
distractor responses.

Output: ``notes/figures/glmsingle_quadrants.pdf``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from nilearn import image
from tqdm import tqdm

from retsupp.utils.data import Subject


LOC_LABEL = {1.0: 'upper_right', 3.0: 'upper_left',
             5.0: 'lower_left', 7.0: 'lower_right'}
QUADRANTS = ('upper_right', 'upper_left', 'lower_left', 'lower_right')
# Quadrant -> hemisphere(s) producing voxels in stimulus_rois.
QUADRANT_HEMI = {
    'upper_right': 'L',  # LH represents contralateral right visual field
    'upper_left':  'R',
    'lower_right': 'L',
    'lower_left':  'R',
}
ROI_ORDER = ['V1', 'V2', 'V3', 'V3AB', 'hV4', 'LO1', 'LO2', 'TO1', 'TO2']
ROI_ALIASES = {
    'V3AB': ['V3A', 'V3B'],
    'LO': ['LO1', 'LO2'],
    'TO': ['TO1', 'TO2'],
}


def load_quadrant_mask(stimulus_rois_dir: Path, subject: int,
                        roi: str, quadrant: str):
    """Load the stimulus-ROI mask for (roi, quadrant). Returns nilearn img.

    Filename pattern from get_rois.py:
        sub-XX_desc-{ROI_NAME}_{HEMI}_{QUADRANT}_roi.nii.gz
    """
    components = ROI_ALIASES.get(roi, [roi])
    hemi = QUADRANT_HEMI[quadrant]
    masks = []
    for r in components:
        fn = (stimulus_rois_dir
              / f'sub-{subject:02d}_desc-{r}_{hemi}_{quadrant}_roi.nii.gz')
        if fn.exists():
            masks.append(image.load_img(str(fn)))
    if not masks:
        return None
    if len(masks) == 1:
        return masks[0]
    arr = np.any(np.stack([m.get_fdata().astype(bool) for m in masks]),
                 axis=0).astype(np.uint8)
    return nib.Nifti1Image(arr, masks[0].affine)


def per_subject_per_quadrant(subject: int, bids_folder: Path,
                              rois=ROI_ORDER):
    """Returns long-format DataFrame: subject, roi, quadrant,
    condition (target/distractor/neutral), n_trials, n_voxels, mean_beta."""
    sub = Subject(subject, bids_folder)
    glm_dir = (bids_folder / 'derivatives' / 'glmsingle'
               / f'sub-{subject:02d}' / 'func')
    pe_nii = (glm_dir / f'sub-{subject:02d}_task-search_space-T1w_'
                          f'desc-distractor_pe.nii.gz')
    tr_tsv = (glm_dir / f'sub-{subject:02d}_task-search_space-T1w_'
                          f'desc-trials.tsv')
    if not pe_nii.exists() or not tr_tsv.exists():
        return pd.DataFrame()

    trials = pd.read_csv(tr_tsv, sep='\t')
    if 'target_location' not in trials.columns:
        # Need to merge with original onsets — fall back to derived cols.
        # The trials tsv already has distractor_location; we need target_loc.
        # Pull from per-run events tsv.
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
            # Per-trial mean within this ROI×Q mask.
            per_trial = pe_arr[mask].mean(axis=0)[:n_trials]

            # Classify each trial: target_at_Q / distractor_at_Q / neutral.
            tgt = trials['target_label'].values == quad
            dist = trials['distractor_label'].values == quad
            target_idx = np.where(tgt & ~dist)[0]
            dist_idx = np.where(dist & ~tgt)[0]
            neutral_idx = np.where(~tgt & ~dist)[0]
            for cond, idx in [('target', target_idx),
                              ('distractor', dist_idx),
                              ('neutral', neutral_idx)]:
                if len(idx) == 0:
                    continue
                beta = float(np.nanmean(per_trial[idx]))
                rows.append(dict(
                    subject=subject, roi=roi, quadrant=quad,
                    condition=cond, n_trials=len(idx), n_voxels=n_vox,
                    mean_beta=beta,
                ))
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bids-folder', default='/data/ds-retsupp')
    p.add_argument('--subjects', type=int, nargs='+',
                   default=list(range(1, 31)))
    p.add_argument('--rois', nargs='+', default=ROI_ORDER)
    p.add_argument('--out', type=Path,
                   default=Path('notes/figures/glmsingle_quadrants.pdf'))
    args = p.parse_args()

    out = args.out
    if not out.is_absolute():
        out = Path('/Users/gdehol/git/retsupp') / out
    out.parent.mkdir(parents=True, exist_ok=True)
    bids = Path(args.bids_folder)

    print(f'Subjects: {args.subjects}')
    print(f'ROIs: {args.rois}')

    all_rows = []
    for s in tqdm(args.subjects, desc='subjects'):
        try:
            df = per_subject_per_quadrant(s, bids, rois=args.rois)
        except Exception as e:
            print(f'sub-{s:02d}: failed ({e})'); continue
        if len(df):
            all_rows.append(df)
    if not all_rows:
        raise RuntimeError('No subject yielded data.')
    df = pd.concat(all_rows, ignore_index=True)

    # Save the long-format TSV alongside the figure.
    tsv_out = out.with_suffix('.tsv')
    df.to_csv(tsv_out, sep='\t', index=False)
    print(f'\nLong-format data: {tsv_out}  ({len(df)} rows)')
    print(df.groupby(['roi', 'condition']).size().unstack(fill_value=0))

    # Collapse across quadrants per (subject, ROI, condition) by averaging.
    df_coll = (df.groupby(['subject', 'roi', 'condition'], as_index=False)
                  ['mean_beta'].mean())

    cond_order = ['target', 'distractor', 'neutral']
    cond_color = {'target': '#2ca02c', 'distractor': '#d62728',
                  'neutral': '0.6'}

    with PdfPages(out) as pdf:
        # ---- Page 1: per-ROI bars (group mean ± SEM across subjects).
        rois_pres = [r for r in args.rois if r in df_coll['roi'].unique()]
        n = len(rois_pres)
        ncol = 4 if n >= 4 else n
        nrow = int(np.ceil(n / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 3.0 * nrow),
                                  sharey=True, squeeze=False)
        axes = axes.flatten()
        for i, roi in enumerate(rois_pres):
            ax = axes[i]
            roi_df = df_coll[df_coll['roi'] == roi]
            xs, means, sems, colors = [], [], [], []
            for ci, cond in enumerate(cond_order):
                vals = roi_df.loc[roi_df['condition'] == cond,
                                    'mean_beta'].values
                if len(vals) == 0:
                    continue
                xs.append(ci)
                means.append(np.mean(vals))
                sems.append(np.std(vals, ddof=1) / np.sqrt(len(vals))
                            if len(vals) > 1 else 0.0)
                colors.append(cond_color[cond])
            ax.bar(xs, means, yerr=sems, color=colors, edgecolor='k',
                   capsize=3, alpha=0.85)
            # individual subjects as scatter
            for ci, cond in enumerate(cond_order):
                vals = roi_df.loc[roi_df['condition'] == cond,
                                    'mean_beta'].values
                ax.scatter([ci] * len(vals) + np.random.normal(0, 0.05, len(vals)),
                            vals, color=cond_color[cond], alpha=0.4,
                            s=12, edgecolors='none')
            ax.set_xticks(range(len(cond_order)))
            ax.set_xticklabels(cond_order, rotation=20, fontsize=8)
            ax.axhline(0, color='k', lw=0.5, alpha=0.3)
            ax.set_title(roi, fontsize=11, weight='bold')
            if i % ncol == 0:
                ax.set_ylabel('mean β at quadrant\n(collapsed over UR/UL/LL/LR)',
                              fontsize=8)
            ax.grid(alpha=0.12, axis='y')
        for j in range(len(rois_pres), len(axes)):
            axes[j].axis('off')
        n_subs = df_coll['subject'].nunique()
        fig.suptitle(
            f'GLMSingle: response at quadrant where target/distractor landed\n'
            f'collapsed over UR/UL/LL/LR; n={n_subs} subjects',
            fontsize=12, weight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        pdf.savefig(fig); plt.close(fig)

        # ---- Page 2: paired (target − distractor) per ROI, per subject.
        diff = (df_coll.pivot_table(index=['subject', 'roi'],
                                      columns='condition',
                                      values='mean_beta')
                    .reset_index())
        if 'target' in diff.columns and 'distractor' in diff.columns:
            diff['t_minus_d'] = diff['target'] - diff['distractor']
            fig, ax = plt.subplots(figsize=(8, 5))
            order = [r for r in args.rois if r in diff['roi'].unique()]
            sns.boxplot(data=diff, x='roi', y='t_minus_d', order=order,
                         color='lightgray', ax=ax, showfliers=False)
            sns.stripplot(data=diff, x='roi', y='t_minus_d', order=order,
                           color='k', size=3, alpha=0.5, ax=ax, jitter=0.2)
            ax.axhline(0, color='r', ls='--', lw=0.8)
            ax.set_ylabel('mean β: target − distractor   (at quadrant)',
                          fontsize=10)
            ax.set_xlabel('ROI', fontsize=10)
            ax.set_title('Target vs Distractor response at the quadrant\n'
                         '(positive = bigger response to target than '
                         'distractor at the same RF)', fontsize=11)
            fig.tight_layout()
            pdf.savefig(fig); plt.close(fig)

    print(f'\nDone: {out}')


if __name__ == '__main__':
    main()
