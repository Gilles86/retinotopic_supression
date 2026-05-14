"""Compute the per-(subject, ROI) R² mixture for every subject with
a fitted model and dump a per-subject diagnostic PDF.

Outputs:
  derivatives/prf/model{N}/sub-XX/sub-XX_desc-p_signal.{nii.gz, json}
  derivatives/prf_diagnostics/r2_mixture/model{N}/sub-XX_r2_mixture.pdf
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from nilearn import maskers
from tqdm import tqdm

from retsupp.utils.data import Subject
from retsupp.modeling.compute_r2_mixture import (
    fit_one_roi, ROI_DEFAULTS,
)


def make_plot(sub: Subject, masker, summary: dict, p_signal_all,
              r2: np.ndarray, rois, out_pdf: Path,
              fdr_alpha: float = 0.05):
    """Per-ROI diagnostic PDF: histogram of logit(R²) + the two GMM
    components + the tail-FDR threshold at α=``fdr_alpha``. Delegates
    each subplot to :func:`braincoder.utils.stats.plot_r2_mixture`.
    """
    from braincoder.utils.stats import plot_r2_mixture
    sns.set_style('whitegrid')
    n = len(rois)
    cols = (n + 1) // 2
    fig, axes = plt.subplots(2, cols, figsize=(3.6 * cols, 7.2),
                              squeeze=False)
    axes = axes.ravel()
    for i, roi in enumerate(rois):
        ax = axes[i]
        info = summary.get(roi, {})
        if not info or 'signal_mu' not in info:
            ax.text(0.5, 0.5, f'{roi}: skipped',
                    transform=ax.transAxes, ha='center')
            ax.set_axis_off()
            continue
        if roi == 'BRAIN':
            roi_mask = np.ones(r2.size, dtype=bool)
        elif roi == 'GM':
            try:
                roi_mask = masker.transform(sub.get_gm_mask(bold_space=True))\
                                 .flatten().astype(bool)
            except Exception:
                ax.set_axis_off(); continue
        else:
            try:
                roi_mask = masker.transform(
                    sub.get_retinotopic_roi(roi=roi, bold_space=True)
                ).flatten().astype(bool)
            except Exception:
                ax.set_axis_off(); continue
        in_roi = roi_mask & np.isfinite(r2) & (r2 > 0) & (r2 < 0.99)
        n_sig = int((p_signal_all[in_roi] > 0.5).sum())
        plot_r2_mixture(info, r2=r2[in_roi], alpha=fdr_alpha, ax=ax,
                         title=f'{roi}  ({int(in_roi.sum())} vox; '
                               f'{n_sig} p_sig>0.5)')
        # Cross-component tails (signal PDF evaluated deep in noise, etc.)
        # plummet to 1e-30; clamp so the histogram is visible.
        ax.set_ylim(bottom=1e-3, top=ax.get_ylim()[1])
        ax.tick_params(axis='x', labelsize=8)
        ax.legend(fontsize=7, loc='upper right', framealpha=0.9)
    for j in range(n, len(axes)):
        axes[j].set_axis_off()
    fig.suptitle(f'sub-{sub.subject_id:02d} model {summary.get("model", "?")}: '
                 f'per-ROI 2-comp Gaussian mixture on logit(R²), α={fdr_alpha}',
                 fontsize=11, weight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches='tight'); plt.close(fig)


def run_one(subject: int, model: int, bids_folder: Path,
             rois: list[str], plots_dir: Path) -> dict | None:
    sub = Subject(subject, bids_folder)
    masker = maskers.NiftiMasker(mask_img=sub.get_bold_mask())
    masker.fit()

    r2_path = (bids_folder / 'derivatives' / 'prf'
               / f'model{model}' / f'sub-{subject:02d}'
               / f'sub-{subject:02d}_desc-r2.nii.gz')
    if not r2_path.exists():
        return None
    r2 = masker.transform(str(r2_path)).flatten()

    p_signal_all = np.full(r2.size, np.nan, dtype=np.float32)
    summary = {'model': model}
    # 'BRAIN' and 'GM' are whole-mask pseudo-ROIs (BOLD mask, GM probseg≥0.5).
    # Fitted first so robust whole-brain mixtures exist before per-ROI fits.
    rois_with_global = ['BRAIN', 'GM'] + [r for r in rois if r not in ('BRAIN', 'GM')]
    for roi in rois_with_global:
        if roi == 'BRAIN':
            roi_mask = np.ones(r2.size, dtype=bool)
        elif roi == 'GM':
            try:
                roi_mask = masker.transform(sub.get_gm_mask(bold_space=True))\
                                 .flatten().astype(bool)
            except Exception as e:
                summary[roi] = {'reason': f'gm load failed: {e}'}; continue
        else:
            try:
                roi_mask = masker.transform(
                    sub.get_retinotopic_roi(roi=roi, bold_space=True)
                ).flatten().astype(bool)
            except Exception as e:
                summary[roi] = {'reason': f'roi load failed: {e}'}; continue
        idx = np.where(roi_mask & np.isfinite(r2)
                       & (r2 > 0) & (r2 < 0.99))[0]
        if len(idx) < 50:
            summary[roi] = {'reason': f'only {len(idx)} usable voxels'}
            continue
        out = fit_one_roi(r2[idx])
        if out['fit'] is not None:
            # Per-voxel posteriors only stored for the per-ROI fits
            # (not BRAIN / GM — those would overwrite ROI values).
            if roi not in ('BRAIN', 'GM'):
                p_signal_all[idx] = out['p_signal']
            summary[roi] = out['fit']
        else:
            summary[roi] = {'reason': out['reason']}

    out_dir = (bids_folder / 'derivatives' / 'prf'
               / f'model{model}' / f'sub-{subject:02d}')
    # Float32 wrap — see CLAUDE.md §"NIfTI dtype trap".
    nii = masker.inverse_transform(p_signal_all)
    nii.set_data_dtype(np.float32)
    nii.header.set_slope_inter(slope=1, inter=0)
    nii.to_filename(str(out_dir
                       / f'sub-{subject:02d}_desc-p_signal.nii.gz'))
    with open(out_dir / f'sub-{subject:02d}_desc-p_signal.json',
              'w') as fh:
        json.dump(summary, fh, indent=2)
    pdf = plots_dir / f'sub-{subject:02d}_r2_mixture.pdf'
    plot_rois = ['BRAIN', 'GM'] + [r for r in rois if r not in ('BRAIN', 'GM')]
    make_plot(sub, masker, summary, p_signal_all, r2, plot_rois, pdf)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bids-folder', default='/data/ds-retsupp')
    p.add_argument('--model', type=int, default=1)
    p.add_argument('--subjects', type=int, nargs='+',
                   default=list(range(1, 31)))
    p.add_argument('--rois', nargs='+', default=ROI_DEFAULTS)
    args = p.parse_args()

    bids = Path(args.bids_folder)
    plots_dir = (bids / 'derivatives' / 'prf_diagnostics'
                 / 'r2_mixture' / f'model{args.model}')
    plots_dir.mkdir(parents=True, exist_ok=True)

    all_summary = {}
    for s in tqdm(args.subjects):
        try:
            summary = run_one(s, args.model, bids, args.rois, plots_dir)
        except Exception as e:
            print(f'  sub-{s:02d}: FAILED ({e})')
            continue
        if summary is None:
            continue
        all_summary[s] = summary

    # Combined per-subject overview.
    out_summary = plots_dir / 'all_subjects_summary.json'
    with open(out_summary, 'w') as fh:
        json.dump(all_summary, fh, indent=2)
    print(f'\nWrote {out_summary}')
    print(f'Per-subject PDFs in {plots_dir}/')

    # Overview swarm: per-(sub, ROI) signal weight + R² threshold.
    import pandas as pd
    rows = []
    for s, summary in all_summary.items():
        sub = Subject(s, bids)
        for roi in (['BRAIN'] + list(args.rois)):
            info = summary.get(roi, {})
            if 'signal_mu' not in info:
                continue
            t05 = sub.get_r2_threshold(args.model, roi, posterior=0.5)
            rows.append({
                'subject': s, 'roi': roi,
                'signal_weight_pct': 100 * info['signal_weight'],
                'signal_mean': info['signal_mean_r2'],
                'noise_mean':  info['noise_mean_r2'],
                'r2_thresh_p50': t05,
                'n_voxels': info['n_voxels'],
            })
    if rows:
        df = pd.DataFrame(rows)
        tsv = plots_dir / 'all_subjects_summary.tsv'
        df.to_csv(tsv, sep='\t', index=False)

        order = ['BRAIN'] + list(args.rois)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        sns.boxplot(data=df, x='roi', y='signal_weight_pct',
                    order=order, color='0.85', width=0.55,
                    fliersize=0, ax=axes[0])
        sns.stripplot(data=df, x='roi', y='signal_weight_pct',
                      order=order, color='#2c7fb8', size=5, alpha=0.75,
                      jitter=0.18, ax=axes[0])
        axes[0].set_title('% voxels in signal component', fontsize=12, weight='bold')
        axes[0].set_xlabel(''); axes[0].set_ylabel('signal weight (%)')
        axes[0].tick_params(axis='x', rotation=20)

        sns.boxplot(data=df, x='roi', y='r2_thresh_p50',
                    order=order, color='0.85', width=0.55,
                    fliersize=0, ax=axes[1])
        sns.stripplot(data=df, x='roi', y='r2_thresh_p50',
                      order=order, color='#d62728', size=5, alpha=0.75,
                      jitter=0.18, ax=axes[1])
        axes[1].set_title('R² threshold at p_signal = 0.5', fontsize=12, weight='bold')
        axes[1].set_xlabel(''); axes[1].set_ylabel('R² threshold')
        axes[1].tick_params(axis='x', rotation=20)

        fig.suptitle(f'Logit-Gaussian mixture R²-thresholding — model '
                     f'{args.model}, n={df.subject.nunique()} subjects',
                     fontsize=13, weight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        out = plots_dir / 'overview_swarm.pdf'
        fig.savefig(out, bbox_inches='tight'); plt.close(fig)
        print(f'Wrote {out}')
        print(f'Wrote {tsv}')


if __name__ == '__main__':
    main()
