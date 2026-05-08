"""Per-ROI per-distractor-location BOLD analysis from GLMsingle betas.

Reads:
  derivatives/glmsingle/sub-XX/func/sub-XX_task-search_space-T1w_desc-distractor_pe.nii.gz
    — 4D (x, y, z, n_trials) single-trial betas.
  derivatives/glmsingle/sub-XX/func/sub-XX_task-search_space-T1w_desc-trials.tsv
    — per-trial metadata: distractor_location, distractor_label, is_hp_distractor.

For each subject × ROI:
  • Mean trial-level beta per distractor condition
    (5 conditions: HP, the 3 LPs, no_distractor).
  • Optional: Richter-style preferred-location analysis (group voxels
    by their PRF-preferred ring location, ask whether HPD-preferred
    voxels are suppressed even on no-distractor trials).

Pages:
  1. Per-ROI mean BOLD per distractor condition (group + per-subject).
  2. HP vs LP vs no-distractor contrast per ROI.
  3. (optional) Voxel-preferred-location analysis.

Usage
-----
    python -m retsupp.glm.analyze_glmsingle \\
        --bids-folder /data/ds-retsupp \\
        --out notes/glmsingle.pdf
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from nilearn import input_data
from scipy import stats

from retsupp.utils.data import Subject

ROI_ORDER = ['V1', 'V2', 'V3', 'V3AB', 'hV4', 'LO', 'TO', 'VO']
COND_ORDER = ['no_distractor', 'upper_right', 'upper_left',
               'lower_left', 'lower_right']
LOC_LABEL = {1.0: 'upper_right', 3.0: 'upper_left', 5.0: 'lower_left',
              7.0: 'lower_right', 10.0: 'no_distractor'}


def per_subject_per_roi(subject: int, bids_folder: Path,
                          rois: list[str] = ROI_ORDER,
                          r2_thr: float | None = None):
    """Aggregate single-trial betas to per-(ROI, distractor) means.

    Returns long-format DataFrame: subject, roi, distractor, n_trials,
    n_voxels, mean_beta.
    """
    sub = Subject(subject, bids_folder)
    glm_dir = (bids_folder / 'derivatives' / 'glmsingle'
                / f'sub-{subject:02d}' / 'func')
    pe_nii = (glm_dir / f'sub-{subject:02d}_task-search_space-T1w_'
                          f'desc-distractor_pe.nii.gz')
    r2_nii = (glm_dir / f'sub-{subject:02d}_task-search_space-T1w_'
                          f'desc-R2_pe.nii.gz')
    tr_tsv = (glm_dir / f'sub-{subject:02d}_task-search_space-T1w_'
                          f'desc-trials.tsv')
    if not pe_nii.exists() or not tr_tsv.exists():
        return pd.DataFrame()
    trials = pd.read_csv(tr_tsv, sep='\t')
    if 'distractor_label' not in trials.columns:
        trials['distractor_label'] = trials['distractor_location'].map(LOC_LABEL)
    # Single-trial betas masked to the BOLD mask of the subject
    # (GLMsingle output is in T1w space; we use the bold_mask resampled
    # to T1w space). To keep things simple use a brain mask from the
    # GLMsingle space (the file's own non-zero voxels).
    from nilearn import image
    import nibabel as nib
    pe_img = nib.load(str(pe_nii))   # don't load data yet
    n_trials_img = pe_img.shape[3]
    # Build a 3D shape-only reference image so resample_to_img doesn't
    # eagerly load the 4D pe_img into memory.
    target_3d = nib.Nifti1Image(
        np.zeros(pe_img.shape[:3], dtype=np.int8),
        pe_img.affine,
    )
    # We need ROI masks in the SAME space as pe_img. The GLMsingle is in
    # T1w (anatomical) space; our retinotopic atlas is in T1w via
    # neuropythy (mri/inferred_varea.mgz).
    atlas = sub.get_retinotopic_atlas(bold_space=False)
    atlas = image.resample_to_img(atlas, target_img=target_3d,
                                     interpolation='nearest',
                                     force_resample=True, copy_header=True)
    atlas_arr = atlas.get_fdata().astype(int)
    if r2_nii.exists() and r2_thr is not None:
        r2_img = image.resample_to_img(image.load_img(str(r2_nii)),
                                          target_img=target_3d,
                                          interpolation='linear',
                                          force_resample=True, copy_header=True)
        r2_arr = r2_img.get_fdata()
    else:
        r2_arr = None

    rows = []
    aliases = {
        'V3AB': ['V3A', 'V3B'],
        'LO':   ['LO1', 'LO2'],
        'TO':   ['TO1', 'TO2'],
        'VO':   ['VO1', 'VO2'],
    }
    label_to_idx = {v: k for k, v in sub.get_retinotopic_labels().items()}
    from nilearn.masking import apply_mask
    for roi in rois:
        components = aliases.get(roi, [roi])
        ids = [label_to_idx.get(r) for r in components if r in label_to_idx]
        if not ids:
            continue
        mask = np.isin(atlas_arr, ids)
        if r2_arr is not None and r2_thr is not None:
            mask &= (r2_arr > r2_thr)
        n_vox = int(mask.sum())
        if n_vox < 5:
            continue
        # Memory-efficient: build a Nifti mask, apply it (reads only
        # masked voxels via memory-mapped access). Returns (n_trials, n_vox).
        mask_img = nib.Nifti1Image(mask.astype(np.int8), pe_img.affine)
        flat_betas = apply_mask(pe_img, mask_img)   # (n_trials, n_vox)
        per_trial_mean = flat_betas.mean(axis=1)
        for cond, mask_t in trials.groupby('distractor_label'):
            idx = mask_t.index.values
            idx = idx[(idx >= 0) & (idx < n_trials_img)]
            if len(idx) == 0:
                continue
            beta = float(np.nanmean(per_trial_mean[idx]))
            sem = float(np.nanstd(per_trial_mean[idx], ddof=1)
                          / np.sqrt(len(idx)))
            rows.append(dict(
                subject=subject, roi=roi, distractor=cond,
                n_trials=int(len(idx)), n_voxels=n_vox,
                mean_beta=beta, sem_beta=sem,
            ))
    return pd.DataFrame(rows)


def plot_per_roi_grid(df: pd.DataFrame, pdf: PdfPages):
    """Per-ROI bar/violin: mean trial beta per distractor condition,
    one bar per condition."""
    rois = [r for r in ROI_ORDER if r in df['roi'].unique()]
    n = len(rois); ncol = 4; nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3.5 * nrow),
                              sharey=True)
    axes = np.atleast_2d(axes)
    df = df.copy()
    df['distractor'] = pd.Categorical(df['distractor'], categories=COND_ORDER,
                                        ordered=True)
    for i, roi in enumerate(rois):
        ax = axes[i // ncol, i % ncol]
        sub = df[df['roi'] == roi]
        if len(sub) == 0:
            ax.axis('off'); continue
        # Per-(condition) mean ± SEM across subjects.
        agg = sub.groupby('distractor', observed=True)['mean_beta'].agg(
            ['mean', lambda v: v.std(ddof=1) / np.sqrt(max(len(v), 1)),
             'count'])
        agg.columns = ['m', 'sem', 'n']
        agg = agg.reindex(COND_ORDER).dropna()
        x = np.arange(len(agg))
        # Bars colored: HP/LP/none.
        is_hp_color = ['C0' if cond == 'no_distractor'
                        else '0.6' for cond in agg.index]
        ax.bar(x, agg['m'], yerr=agg['sem'],
                color=is_hp_color, edgecolor='k', linewidth=0.5,
                error_kw=dict(capsize=4, lw=1.0))
        # Per-subject dots.
        for j, cond in enumerate(agg.index):
            vals = sub[sub['distractor'] == cond]['mean_beta'].values
            ax.scatter([j] * len(vals), vals, color='k', s=8,
                        alpha=0.4, zorder=5)
        ax.axhline(0, color='gray', lw=0.4)
        ax.set_xticks(x); ax.set_xticklabels(agg.index, rotation=30,
                                              fontsize=8)
        ax.set_title(f'{roi}  n_subj = {sub["subject"].nunique()}',
                      fontsize=11)
        ax.grid(alpha=0.2, axis='y')
    for j in range(n, nrow * ncol):
        axes[j // ncol, j % ncol].axis('off')
    fig.suptitle('GLMsingle: mean trial beta per ROI per distractor condition',
                  fontsize=13, weight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig); plt.close(fig)


def plot_hp_vs_lp_test(df: pd.DataFrame, pdf: PdfPages):
    """For each subject × ROI: contrast HP-distractor vs LP-distractor mean.

    Aggregate per (subject, roi) by collapsing the 3 LP conditions into
    one mean, then test HP < LP per ROI across subjects (paired t).
    """
    df = df.copy()
    df['cond_kind'] = df.apply(
        lambda r: 'no_distractor' if r['distractor'] == 'no_distractor'
                  else 'HP' if r['distractor']
                       == _hp_distractor_per_subject(r['subject'])
                  else 'LP',
        axis=1,
    )
    # We can't actually classify HP/LP per row without per-run HP info.
    # Skip that here and just compare distractor vs no-distractor.
    rois = [r for r in ROI_ORDER if r in df['roi'].unique()]
    rec = []
    for roi in rois:
        sub_roi = df[df['roi'] == roi]
        per_subj = sub_roi.groupby(['subject', 'distractor'], observed=True)[
            'mean_beta'].mean().reset_index()
        wide = per_subj.pivot_table(index='subject', columns='distractor',
                                       values='mean_beta')
        if 'no_distractor' not in wide.columns:
            continue
        # Mean across the 4 ring locations vs no_distractor.
        ring_cols = [c for c in COND_ORDER[1:] if c in wide.columns]
        wide['any_distractor'] = wide[ring_cols].mean(axis=1)
        diff = wide['any_distractor'] - wide['no_distractor']
        diff = diff.dropna()
        if len(diff) < 4:
            continue
        t, p = stats.ttest_1samp(diff, 0)
        rec.append(dict(roi=roi, n=len(diff),
                        mean=float(diff.mean()),
                        sem=float(diff.std(ddof=1) / np.sqrt(len(diff))),
                        t=float(t), p=float(p)))
    summary = pd.DataFrame(rec)
    if summary.empty:
        return

    fig, ax = plt.subplots(figsize=(11, 5.5))
    rois_ = list(summary['roi'])
    x = np.arange(len(rois_))
    cols = ['C2' if (np.isfinite(p) and p < 0.05 and m > 0)
             else 'C3' if (np.isfinite(p) and p < 0.05 and m < 0)
             else '0.55'
             for m, p in zip(summary['mean'], summary['p'])]
    ax.bar(x, summary['mean'], yerr=summary['sem'],
            color=cols, edgecolor='k', linewidth=0.7,
            error_kw=dict(capsize=5, lw=1.5))
    for i, (_, row) in enumerate(summary.iterrows()):
        sig = ('***' if row['p'] < 0.001 else
                '**' if row['p'] < 0.01 else
                '*' if row['p'] < 0.05 else 'n.s.')
        ax.text(i, row['mean'] + (row['sem'] if row['mean'] >= 0 else -row['sem']) +
                 (0.005 if row['mean'] >= 0 else -0.005),
                 sig, ha='center',
                 va='bottom' if row['mean'] >= 0 else 'top',
                 fontsize=11,
                 weight='bold' if row['p'] < 0.05 else 'normal')
    ax.axhline(0, color='gray', lw=0.5, ls='--')
    ax.set_xticks(x); ax.set_xticklabels(rois_, fontsize=12)
    ax.set_ylabel('mean β (any-distractor) − β (no-distractor)',
                   fontsize=12)
    ax.set_title(
        'Distractor presence effect on BOLD per ROI '
        '(any-distractor − no-distractor; paired t per subject)',
        fontsize=12, weight='bold',
    )
    ax.grid(alpha=0.2, axis='y')
    pdf.savefig(fig); plt.close(fig)


def _hp_distractor_per_subject(subject: int):
    # Stub — true mapping requires per-run HP location.
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bids-folder', type=Path,
                        default=Path('/data/ds-retsupp'))
    parser.add_argument('--subjects', type=int, nargs='+',
                        default=list(range(1, 31)))
    parser.add_argument('--out', type=Path,
                        default=Path('notes/glmsingle.pdf'))
    parser.add_argument('--tsv-out', type=Path,
                        default=Path('notes/glmsingle_per_roi.tsv'))
    parser.add_argument('--r2-thr', type=float, default=None,
                        help='Optional GLMsingle R² threshold for voxel '
                             'inclusion.')
    args = parser.parse_args()

    rows = []
    for s in args.subjects:
        try:
            df_s = per_subject_per_roi(s, args.bids_folder, ROI_ORDER,
                                          r2_thr=args.r2_thr)
            if not df_s.empty:
                rows.append(df_s)
                print(f'sub-{s:02d}: {len(df_s)} ROI×condition rows.')
            else:
                print(f'sub-{s:02d}: no GLMsingle output yet — skipping.')
        except Exception as e:
            print(f'sub-{s:02d}: ERROR — {e}')
    if not rows:
        raise SystemExit('No GLMsingle outputs found.')
    df = pd.concat(rows, ignore_index=True)
    args.tsv_out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.tsv_out, sep='\t', index=False)
    print(f'Wrote {args.tsv_out} ({len(df)} rows, {df.subject.nunique()} subj)')

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(args.out) as pdf:
        # Cover.
        fig, ax = plt.subplots(figsize=(11, 8.5)); ax.axis('off')
        ax.text(0.5, 0.95, 'GLMsingle distractor analysis',
                 ha='center', va='top', fontsize=18, weight='bold',
                 transform=ax.transAxes)
        ax.text(0.04, 0.85,
                 f'Subjects available: {df.subject.nunique()}\n'
                 f'ROIs: {df.roi.nunique()}\n'
                 f'r²-thr: {args.r2_thr}\n'
                 f'\n'
                 f'Single-trial betas → per-(subject, ROI, distractor) mean.\n'
                 f'5 distractor conditions: no_distractor, upper_right,\n'
                 f'upper_left, lower_left, lower_right.\n',
                 ha='left', va='top', fontsize=11,
                 family='monospace', transform=ax.transAxes)
        pdf.savefig(fig); plt.close(fig)

        plot_per_roi_grid(df, pdf)
        plot_hp_vs_lp_test(df, pdf)
    print(f'Wrote {args.out}')


if __name__ == '__main__':
    main()
