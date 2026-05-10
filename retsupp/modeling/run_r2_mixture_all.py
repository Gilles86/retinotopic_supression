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
              r2: np.ndarray, rois, out_pdf: Path):
    sns.set_style('whitegrid')
    from scipy.stats import beta as beta_dist
    n = len(rois)
    cols = (n + 1) // 2
    fig, axes = plt.subplots(2, cols, figsize=(2.4 * cols, 6.4),
                              squeeze=False)
    axes = axes.ravel()
    for i, roi in enumerate(rois):
        ax = axes[i]
        info = summary.get(roi, {})
        if 'signal_mean' not in info:
            ax.text(0.5, 0.5, f'{roi}: skipped',
                    transform=ax.transAxes, ha='center')
            ax.set_axis_off()
            continue
        if roi == 'BRAIN':
            roi_mask = np.ones(r2.size, dtype=bool)
        else:
            try:
                roi_mask = masker.transform(
                    sub.get_retinotopic_roi(roi=roi, bold_space=True)
                ).flatten().astype(bool)
            except Exception:
                ax.set_axis_off(); continue
        in_roi = roi_mask & np.isfinite(r2) & (r2 > 0) & (r2 < 0.99)
        x = np.clip(r2[in_roi], 1e-6, 1 - 1e-6)
        # Plot histogram on raw R² with log y-axis to expose the signal tail.
        ax.hist(x, bins=80, density=True, color='0.85',
                edgecolor='0.5', alpha=0.9)
        # Beta-mixture component pdfs.
        grid = np.linspace(1e-3, max(x.max(), 0.2), 500)
        n_g = (beta_dist.pdf(grid, info['noise_alpha'],  info['noise_beta'])
               * info['noise_weight'])
        s_g = (beta_dist.pdf(grid, info['signal_alpha'], info['signal_beta'])
               * info['signal_weight'])
        ax.plot(grid, n_g, color='#1f77b4', lw=2, label='noise (Beta)')
        ax.plot(grid, s_g, color='#d62728', lw=2, label='signal (Beta)')
        ax.plot(grid, n_g + s_g, color='k', lw=1, ls='--', alpha=0.6)
        # Threshold where p_sig=0.5.
        p_at_grid = s_g / (n_g + s_g + 1e-12)
        # Pick first crossing right of the noise mode.
        start = max(0, int(np.searchsorted(grid, info['noise_mean'])))
        seg = p_at_grid[start:]
        cr = np.where(seg >= 0.5)[0]
        if len(cr):
            r2_thresh = float(grid[start + cr[0]])
            ax.axvline(r2_thresh, color='k', ls=':', lw=1.3,
                        label=f'p_sig=0.5\n(R²={r2_thresh:.3f})')
        n_sig = (p_signal_all[in_roi] > 0.5).sum()
        ax.set_title(f'{roi}  ({in_roi.sum()} vox; {n_sig} signal)',
                     fontsize=10, weight='bold')
        ax.set_xlabel('R²', fontsize=8)
        ax.set_yscale('log')
        ax.set_ylim(1e-2, max(n_g.max(), s_g.max()) * 4)
        ax.legend(fontsize=6, loc='upper right')
    for j in range(n, len(axes)):
        axes[j].set_axis_off()
    fig.suptitle(f'sub-{sub.subject_id:02d} model {summary.get("model", "?")}: '
                 'per-ROI 2-component Beta mixture on R² (log y)',
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
    # 'BRAIN' is a pseudo-ROI using the whole BOLD mask. Fitted first
    # so robust mixtures are available even for ROIs with little signal.
    rois_with_brain = ['BRAIN'] + [r for r in rois if r != 'BRAIN']
    for roi in rois_with_brain:
        if roi == 'BRAIN':
            roi_mask = np.ones(r2.size, dtype=bool)
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
            # (not BRAIN — the BRAIN posterior would overwrite ROI
            # values otherwise).
            if roi != 'BRAIN':
                p_signal_all[idx] = out['p_signal']
            summary[roi] = out['fit']
        else:
            summary[roi] = {'reason': out['reason']}

    out_dir = (bids_folder / 'derivatives' / 'prf'
               / f'model{model}' / f'sub-{subject:02d}')
    nii = masker.inverse_transform(p_signal_all)
    nii.to_filename(str(out_dir
                       / f'sub-{subject:02d}_desc-p_signal.nii.gz'))
    with open(out_dir / f'sub-{subject:02d}_desc-p_signal.json',
              'w') as fh:
        json.dump(summary, fh, indent=2)
    pdf = plots_dir / f'sub-{subject:02d}_r2_mixture.pdf'
    plot_rois = ['BRAIN'] + list(rois)
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
            if 'signal_alpha' not in info:
                continue
            t05 = sub.get_r2_threshold(args.model, roi, posterior=0.5)
            rows.append({
                'subject': s, 'roi': roi,
                'signal_weight_pct': 100 * info['signal_weight'],
                'signal_mean': info['signal_mean'],
                'noise_mean':  info['noise_mean'],
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

        fig.suptitle(f'Beta-mixture R²-thresholding — model {args.model}, '
                     f'n={df.subject.nunique()} subjects',
                     fontsize=13, weight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        out = plots_dir / 'overview_swarm.pdf'
        fig.savefig(out, bbox_inches='tight'); plt.close(fig)
        print(f'Wrote {out}')
        print(f'Wrote {tsv}')


if __name__ == '__main__':
    main()
