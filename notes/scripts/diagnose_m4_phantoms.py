"""Diagnose m4 (DoG + flexible HRF) "phantom-perfect" voxels.

Phantom = voxel with R^2 >= 0.999 in the model-4 PRF fit. There are
~22,000 of them per subject and they cause problems downstream.

This script:
  1. Loads m1 + m4 PRF NIfTIs for sub-02 (and optionally sub-01, sub-05).
  2. Classifies voxels into {phantom, signal, other} per subject.
  3. Plots phantom mask slices, BOLD timeseries (cleaned + fmriprep),
     parameter histograms, R^2 vs sd, init-value clustering.
  4. Tries spatial overlap of phantoms across subjects (after each
     subject is in its own native space -- so we look at relative
     fractions, not exact voxels).
  5. Writes everything to notes/figures/m4_phantom_diagnosis_*.pdf
     and a one-pager note.

Run with:  ~/mambaforge/envs/retsupp/bin/python notes/scripts/diagnose_m4_phantoms.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import nibabel as nib

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from retsupp.utils.data import Subject  # noqa: E402

BIDS = '/data/ds-retsupp'
FIG_DIR = REPO / 'notes' / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Presentation-grade font sizes.
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 15,
})


M4_PARS = ['x', 'y', 'sd', 'amplitude', 'baseline',
           'srf_amplitude', 'srf_size', 'hrf_delay', 'hrf_dispersion']
M1_PARS = ['x', 'y', 'sd', 'amplitude', 'baseline']


def load_pars(sub: Subject, model: int, par_names: list[str]):
    """Load PRF NIfTI params as (V,) arrays through the BOLD-mask masker."""
    base = (sub.bids_folder / 'derivatives' / 'prf'
            / f'model{model}' / f'sub-{sub.subject_id:02d}')
    masker = sub.get_bold_mask(return_masker=True)
    masker.fit()
    out = {}
    for par in par_names:
        fn = base / f'sub-{sub.subject_id:02d}_desc-{par}.nii.gz'
        out[par] = masker.transform(str(fn)).flatten().astype(np.float32)
    out['r2'] = masker.transform(
        str(base / f'sub-{sub.subject_id:02d}_desc-r2.nii.gz')
    ).flatten().astype(np.float32)
    return out, masker


def classify(prf_m4, prf_m1, r2_phantom=0.999, r2_signal_m1=0.05,
             sd_signal_m1=0.5):
    """Return boolean masks (phantom, signal_m1, other)."""
    phantom = prf_m4['r2'] >= r2_phantom
    signal_m1 = (prf_m1['r2'] > r2_signal_m1) & (prf_m1['sd'] >= sd_signal_m1)
    return phantom, signal_m1


def figure_param_distributions(prf_m4, prf_m1, phantom, signal_m1, subject: int):
    """Histograms of m4 parameters for phantom vs signal voxels.

    Phantom voxels often pile at init values -- m4 init comes from
    model 3 (Gauss + flex HRF), so x/y/sd/amplitude/baseline are
    re-used; srf_amplitude is initialised at 1e-3 and srf_size at 2.0
    (see fit_prf.py MODEL_CFG[4]). hrf_delay/hrf_dispersion are
    initialised by model 3 to 4.5/0.75.
    """
    init_vals = {'srf_amplitude': 1e-3, 'srf_size': 2.0,
                 'hrf_delay': 4.5, 'hrf_dispersion': 0.75}
    pars = M4_PARS + ['r2']
    n = len(pars)
    nc = 5
    nr = int(np.ceil(n / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(nc * 3.2, nr * 2.8))
    axes = axes.flatten()
    for ax, par in zip(axes, pars):
        a = prf_m4[par][phantom]
        b = prf_m4[par][signal_m1]
        # Robust range for the histogram (1-99% across both).
        both = np.concatenate([a[np.isfinite(a)], b[np.isfinite(b)]])
        if len(both) == 0:
            continue
        lo, hi = np.percentile(both, [0.5, 99.5])
        if hi - lo < 1e-6:
            hi = lo + 1e-6
        bins = np.linspace(lo, hi, 60)
        ax.hist(a, bins=bins, alpha=0.5, label=f'Phantom (n={phantom.sum()})',
                density=True, color='C3')
        ax.hist(b, bins=bins, alpha=0.5, label=f'Signal m1 (n={signal_m1.sum()})',
                density=True, color='C0')
        if par in init_vals:
            ax.axvline(init_vals[par], color='k', linestyle='--', lw=1,
                       label=f'Init={init_vals[par]}')
        ax.set_title(par)
    # Hide unused axes.
    for ax in axes[len(pars):]:
        ax.set_visible(False)
    axes[0].legend(loc='upper right', fontsize=8)
    fig.suptitle(f'Sub-{subject:02d} | M4 parameter distributions: '
                 f'Phantom (R²≥0.999) vs M1-signal voxels')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = FIG_DIR / f'm4_phantom_diagnosis_params_sub-{subject:02d}.pdf'
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {out}')
    return out


def figure_spatial(masker, phantom, signal_m1, prf_m4, subject: int):
    """Show phantom mask + brain mask + signal mask in axial slices."""
    mask_img = masker.mask_img_
    mask_arr = mask_img.get_fdata().astype(bool)
    affine = mask_img.affine

    def to_volume(bool_arr_in_masker):
        out = np.zeros(mask_arr.shape, dtype=np.float32)
        out[mask_arr] = bool_arr_in_masker.astype(np.float32)
        return out

    phantom_vol = to_volume(phantom)
    signal_vol = to_volume(signal_m1)
    r2_vol = to_volume(prf_m4['r2'])

    # Pick 6 axial slices spanning the brain.
    z_present = np.where(mask_arr.any(axis=(0, 1)))[0]
    if len(z_present) < 6:
        z_picks = z_present
    else:
        z_picks = z_present[
            np.linspace(0, len(z_present) - 1, 6).astype(int)]

    fig, axes = plt.subplots(2, len(z_picks), figsize=(2.6 * len(z_picks), 6),
                             squeeze=False)
    for j, z in enumerate(z_picks):
        ax = axes[0, j]
        ax.imshow(mask_arr[:, :, z].T, origin='lower', cmap='Greys',
                  alpha=0.4)
        # Phantom voxels in red, signal voxels in cyan.
        ph_slice = phantom_vol[:, :, z]
        sg_slice = signal_vol[:, :, z]
        ax.imshow(np.ma.masked_where(ph_slice == 0, ph_slice).T,
                  origin='lower', cmap='Reds', alpha=0.9, vmin=0, vmax=1)
        ax.imshow(np.ma.masked_where(sg_slice == 0, sg_slice).T,
                  origin='lower', cmap='Blues', alpha=0.6, vmin=0, vmax=1)
        ax.set_title(f'Z={z}')
        ax.set_xticks([]); ax.set_yticks([])
        if j == 0:
            ax.set_ylabel('Phantom (red) /\nM1-signal (blue)')

        ax = axes[1, j]
        ax.imshow(mask_arr[:, :, z].T, origin='lower', cmap='Greys',
                  alpha=0.4)
        r2s = np.ma.masked_where(~mask_arr[:, :, z], r2_vol[:, :, z])
        ax.imshow(r2s.T, origin='lower', cmap='inferno', vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        if j == 0:
            ax.set_ylabel('M4 R²')

    fig.suptitle(f'Sub-{subject:02d} | Spatial: phantom voxels (red), '
                 f'M1-signal (blue), and M4 R² (bottom)')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = FIG_DIR / f'm4_phantom_diagnosis_spatial_sub-{subject:02d}.pdf'
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {out}')
    return out


def figure_bold_traces(sub: Subject, masker, phantom, signal_m1,
                       n_examples: int = 8):
    """Plot example BOLD timeseries from phantom + signal voxels."""
    rng = np.random.default_rng(42)
    phantom_idx = np.where(phantom)[0]
    signal_idx = np.where(signal_m1)[0]
    pick_p = rng.choice(phantom_idx, size=min(n_examples, len(phantom_idx)),
                        replace=False)
    pick_s = rng.choice(signal_idx, size=min(n_examples, len(signal_idx)),
                        replace=False)

    # Load run 1 cleaned + fmriprep for sub. Fallback to a later (ses, run)
    # pair if the first one is missing on disk locally.
    cleaned = None
    fmriprep = None
    for ses in (1, 2):
        for run in sub.get_runs(ses):
            cleaned_fn = (sub.bids_folder / 'derivatives' / 'cleaned'
                          / f'sub-{sub.subject_id:02d}' / f'ses-{ses}' / 'func'
                          / f'sub-{sub.subject_id:02d}_ses-{ses}_task-search_'
                            f'desc-cleaned_run-{run}_bold.nii.gz')
            fmriprep_fn = (sub.bids_folder / 'derivatives' / 'fmriprep'
                           / f'sub-{sub.subject_id:02d}' / f'ses-{ses}' / 'func'
                           / f'sub-{sub.subject_id:02d}_ses-{ses}_task-search_'
                             f'rec-NORDIC_run-{run}_space-T1w_desc-preproc_'
                             f'bold.nii.gz')
            if cleaned_fn.exists() and fmriprep_fn.exists():
                print(f'    Loading cleaned + fmriprep BOLD for '
                      f'ses-{ses} run-{run}...')
                cleaned = masker.transform(str(cleaned_fn))[:258].astype(np.float32)
                fmriprep = masker.transform(str(fmriprep_fn))[:258].astype(np.float32)
                break
        if cleaned is not None:
            break
    if cleaned is None:
        print(f'    No cleaned BOLD found for sub-{sub.subject_id:02d}, '
              f'skipping BOLD trace plot.')
        return None, None

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    t = np.arange(cleaned.shape[0]) * 1.6
    # Top-left: cleaned, phantom.
    ax = axes[0, 0]
    for k, vi in enumerate(pick_p):
        ax.plot(t, cleaned[:, vi], lw=0.7, alpha=0.7, color=f'C{k}')
    ax.set_title(f'Cleaned BOLD | Phantom voxels (n shown={len(pick_p)})')
    ax.set_ylabel('Signal')
    # Top-right: cleaned, signal.
    ax = axes[0, 1]
    for k, vi in enumerate(pick_s):
        ax.plot(t, cleaned[:, vi], lw=0.7, alpha=0.7, color=f'C{k}')
    ax.set_title(f'Cleaned BOLD | M1-signal voxels (n={len(pick_s)})')
    # Bottom-left: fmriprep, phantom.
    ax = axes[1, 0]
    for k, vi in enumerate(pick_p):
        ax.plot(t, fmriprep[:, vi], lw=0.7, alpha=0.7, color=f'C{k}')
    ax.set_title(f'fMRIprep BOLD | Phantom voxels')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Signal')
    # Bottom-right: fmriprep, signal.
    ax = axes[1, 1]
    for k, vi in enumerate(pick_s):
        ax.plot(t, fmriprep[:, vi], lw=0.7, alpha=0.7, color=f'C{k}')
    ax.set_title(f'fMRIprep BOLD | M1-signal voxels')
    ax.set_xlabel('Time (s)')

    fig.suptitle(f'Sub-{sub.subject_id:02d} ses-{ses} run-{run} | BOLD traces')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = FIG_DIR / f'm4_phantom_diagnosis_bold_sub-{sub.subject_id:02d}.pdf'
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {out}')

    # Compute summary stats on cleaned vs fmriprep for ALL phantom and
    # ALL signal voxels (not just the 8 we plotted) for the note.
    stats = {
        'phantom_cleaned_var_mean': float(np.var(cleaned[:, phantom_idx],
                                                 axis=0).mean()),
        'phantom_cleaned_var_median': float(np.median(np.var(
            cleaned[:, phantom_idx], axis=0))),
        'phantom_cleaned_n_zerovar': int((np.var(
            cleaned[:, phantom_idx], axis=0) < 1e-12).sum()),
        'phantom_cleaned_n_constant': int(
            (cleaned[:, phantom_idx].std(axis=0) < 1e-8).sum()),
        'phantom_fmriprep_var_mean': float(np.var(fmriprep[:, phantom_idx],
                                                  axis=0).mean()),
        'phantom_fmriprep_n_zerovar': int((np.var(
            fmriprep[:, phantom_idx], axis=0) < 1e-12).sum()),
        'signal_cleaned_var_mean': float(np.var(cleaned[:, signal_idx],
                                                axis=0).mean()),
        'signal_cleaned_var_median': float(np.median(np.var(
            cleaned[:, signal_idx], axis=0))),
        'n_phantom': int(phantom_idx.size),
        'n_signal_m1': int(signal_idx.size),
    }
    return out, stats


def figure_r2_vs_sd(prf_m4, prf_m1, phantom, signal_m1, subject: int):
    """Joint scatter of m4 sd vs m4 R²; colour by phantom/signal/other."""
    rng = np.random.default_rng(42)
    n_total = prf_m4['r2'].size
    other = (~phantom) & (~signal_m1)
    # Subsample for plotting speed.
    sub_other = rng.choice(np.where(other)[0],
                            size=min(20000, other.sum()), replace=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    ax = axes[0]
    ax.scatter(prf_m4['sd'][sub_other], prf_m4['r2'][sub_other],
               s=1, color='lightgray', alpha=0.3, label='Other')
    ax.scatter(prf_m4['sd'][signal_m1], prf_m4['r2'][signal_m1],
               s=2, color='C0', alpha=0.5, label='Signal (M1)')
    ax.scatter(prf_m4['sd'][phantom], prf_m4['r2'][phantom],
               s=1, color='C3', alpha=0.3, label='Phantom')
    ax.axhline(0.999, color='k', linestyle=':', lw=1)
    ax.axvline(0.5, color='k', linestyle=':', lw=1)
    ax.set_xlim(0, 5)
    ax.set_ylim(-0.1, 1.05)
    ax.set_xlabel('M4 sd (center σ, deg)')
    ax.set_ylabel('M4 R²')
    ax.set_title('M4 sd vs R²')
    ax.legend(markerscale=4)

    ax = axes[1]
    ax.scatter(prf_m4['srf_size'][sub_other], prf_m4['r2'][sub_other],
               s=1, color='lightgray', alpha=0.3, label='Other')
    ax.scatter(prf_m4['srf_size'][signal_m1], prf_m4['r2'][signal_m1],
               s=2, color='C0', alpha=0.5, label='Signal (M1)')
    ax.scatter(prf_m4['srf_size'][phantom], prf_m4['r2'][phantom],
               s=1, color='C3', alpha=0.3, label='Phantom')
    ax.axvline(2.0, color='k', linestyle='--', lw=1, label='Init srf_size=2.0')
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.1, 1.05)
    ax.set_xlabel('M4 srf_size (surround σ, deg)')
    ax.set_ylabel('M4 R²')
    ax.set_title('M4 srf_size vs R²')
    ax.legend(markerscale=4)

    fig.suptitle(f'Sub-{subject:02d} | M4 R² vs (sd, srf_size); '
                 f'phantom voxels cluster at sd→0')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = FIG_DIR / f'm4_phantom_diagnosis_r2_vs_sd_sub-{subject:02d}.pdf'
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {out}')
    return out


def figure_proposed_filter(prf_m4, phantom, signal_m1, subject: int):
    """Evaluate candidate filters on phantom and signal sets."""
    candidates = {
        'r2 < 0.999':         (prf_m4['r2'] < 0.999),
        'sd > 0.05':          (prf_m4['sd'] > 0.05),
        'sd > 0.05 AND r2 < 0.999':
            (prf_m4['sd'] > 0.05) & (prf_m4['r2'] < 0.999),
        'amplitude > 1e-4':   (np.abs(prf_m4['amplitude']) > 1e-4),
        'sd > 0.1 (broader)': (prf_m4['sd'] > 0.1),
        '(sd > 0.05) OR (srf_size > 0.5)':
            (prf_m4['sd'] > 0.05) | (prf_m4['srf_size'] > 0.5),
        '(sd > 0.1) AND (amplitude != 0)':
            (prf_m4['sd'] > 0.1) & (np.abs(prf_m4['amplitude']) > 1e-6),
    }
    rows = []
    for name, mask in candidates.items():
        # We want: keep almost all signal voxels, drop all phantoms.
        kept_phantom = int((mask & phantom).sum())
        kept_signal = int((mask & signal_m1).sum())
        dropped_phantom = int((~mask & phantom).sum())
        dropped_signal = int((~mask & signal_m1).sum())
        rows.append({
            'filter': name,
            'kept_phantom': kept_phantom,
            'phantom_drop_pct': 100 * dropped_phantom / max(1, phantom.sum()),
            'kept_signal': kept_signal,
            'signal_drop_pct': 100 * dropped_signal / max(1, signal_m1.sum()),
            'kept_total': int(mask.sum()),
        })
    df = pd.DataFrame(rows)
    df.to_csv(FIG_DIR.parent / 'data' /
              f'm4_phantom_filter_table_sub-{subject:02d}.tsv',
              sep='\t', index=False)
    print(f'  Filter table sub-{subject:02d}:')
    print(df.to_string(index=False))
    return df


def main(subjects=(2, 1, 5)):
    summary = {}
    for subj in subjects:
        try:
            sub = Subject(subj, bids_folder=BIDS)
            print(f'\n=== sub-{subj:02d} ===')
            prf_m4, masker = load_pars(sub, 4, M4_PARS)
            prf_m1, _ = load_pars(sub, 1, M1_PARS)
            phantom, signal_m1 = classify(prf_m4, prf_m1)
            print(f'  whole-brain phantom (m4 R²≥0.999): {phantom.sum()}')
            print(f'  whole-brain M1-signal: {signal_m1.sum()}')
            print(f'  overlap phantom & m1-signal: '
                  f'{(phantom & signal_m1).sum()}')

            figure_param_distributions(prf_m4, prf_m1, phantom, signal_m1, subj)
            figure_spatial(masker, phantom, signal_m1, prf_m4, subj)
            _, stats = figure_bold_traces(sub, masker, phantom, signal_m1)
            figure_r2_vs_sd(prf_m4, prf_m1, phantom, signal_m1, subj)
            df_filter = figure_proposed_filter(prf_m4, phantom, signal_m1, subj)

            summary[subj] = {
                'n_phantom': int(phantom.sum()),
                'n_signal_m1': int(signal_m1.sum()),
                'phantom_overlap_signal_m1': int(
                    (phantom & signal_m1).sum()),
                'filter_table': df_filter,
            }
            if stats is not None:
                summary[subj].update(stats)
            # Save the masks themselves for cross-subject comparison
            # (subjects are in their own native space, so we just track
            # how many phantoms per subject, not exact overlap).
        except FileNotFoundError as e:
            print(f'  SKIP sub-{subj:02d}: {e}')

    # Pretty-print summary across subjects
    print('\n=== Cross-subject summary ===')
    rows = []
    for subj, s in summary.items():
        rows.append({
            'subject': subj,
            'n_phantom': s['n_phantom'],
            'n_signal_m1': s['n_signal_m1'],
            'phantom_overlap_signal': s['phantom_overlap_signal_m1'],
            'phantom_cleaned_var_median': s.get('phantom_cleaned_var_median'),
            'phantom_cleaned_n_zerovar': s.get('phantom_cleaned_n_zerovar'),
            'phantom_fmriprep_var_mean': s.get('phantom_fmriprep_var_mean'),
            'phantom_fmriprep_n_zerovar': s.get('phantom_fmriprep_n_zerovar'),
            'signal_cleaned_var_median': s.get('signal_cleaned_var_median'),
        })
    df_cross = pd.DataFrame(rows)
    df_cross.to_csv(FIG_DIR.parent / 'data' /
                    'm4_phantom_cross_subject.tsv', sep='\t', index=False)
    print(df_cross.to_string(index=False))
    print('\nDone.')


if __name__ == '__main__':
    main()
