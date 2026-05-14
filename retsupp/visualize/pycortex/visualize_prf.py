import argparse
import cortex
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from neural_priors.utils.data import Subject, get_all_subject_ids
from retsupp.visualize.utils import get_alpha_vertex
from tqdm.contrib.itertools import product
from itertools import product as product_
import pandas as pd
from pathlib import Path
from nilearn import surface
from retsupp.utils.data import Subject


def _theta_color_value(theta, hemi):
    """Replicate the visual-field-angle → HSV-index mapping used to build
    the `theta_` overlay. `theta` is the pRF angle (radians, atan2(y, x)).
    Returns a value in [0, 1] to be passed through the 'hsv' colormap.
    """
    if hemi == 'R':
        v = np.mod(theta, 2 * np.pi)
        return np.clip((v - 0.25 * np.pi) / (1.5 * np.pi), 0, 1)
    if hemi == 'L':
        v = np.mod(theta + np.pi, 2 * np.pi)
        return 1.0 - np.clip((v - 0.25 * np.pi) / (1.5 * np.pi), 0, 1)
    raise ValueError(f"hemi must be 'L' or 'R', got {hemi!r}")


def make_theta_legend(save_path: Path | None = None, n: int = 720):
    """Minimalistic two-panel polar legend for the `theta_` overlay.
    Only the 270° wedge that maps to real (non-clipped) HSV colors is
    drawn — the 90° ipsilateral wedge (where the mapping saturates to a
    single color) is left blank. No text/ticks; meant to be inset.
    """
    theta = np.linspace(-np.pi, np.pi, n, endpoint=False)
    cmap = plt.get_cmap('hsv')
    width = 2 * np.pi / n

    # Drop the 90° ipsilateral wedge for each hemisphere:
    #   R hem: clip wedge is around 0°   (right meridian) → keep |θ| > π/4
    #   L hem: clip wedge is around 180° (left meridian)  → keep |θ| < 3π/4
    keep_by_hemi = {
        'L': np.abs(theta) <= 0.75 * np.pi,
        'R': np.abs(theta) >= 0.25 * np.pi,
    }

    fig, axes = plt.subplots(1, 2, figsize=(4, 2),
                             subplot_kw={'projection': 'polar'})
    fig.patch.set_facecolor('white')
    for ax, hemi in zip(axes, ['L', 'R']):
        keep = keep_by_hemi[hemi]
        colors = cmap(_theta_color_value(theta[keep], hemi))
        ax.bar(theta[keep], np.ones(keep.sum()), width=width * 1.02,
               color=colors, bottom=0.7, edgecolor='none', linewidth=0)
        ax.set_ylim(0, 1.7)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['polar'].set_visible(False)
        ax.set_facecolor('white')

    fig.tight_layout(pad=0.1)
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        print(f"Saved theta_ legend → {save_path}")
    return fig


def make_theta_full_legend(save_path: Path | None = None, n: int = 720):
    """Full 360° polar legend for the raw `theta` overlay (atan2(y,x) mapped
    through 'hsv' with vmin=-π, vmax=π). Single ring, no hemisphere split,
    no clipped wedge — angle maps directly to color.
    """
    theta = np.linspace(-np.pi, np.pi, n, endpoint=False)
    cmap = plt.get_cmap('hsv')
    width = 2 * np.pi / n
    colors = cmap((theta + np.pi) / (2 * np.pi))

    fig, ax = plt.subplots(figsize=(2, 2),
                           subplot_kw={'projection': 'polar'})
    fig.patch.set_facecolor('white')
    ax.bar(theta, np.ones(n), width=width * 1.02,
           color=colors, bottom=0.7, edgecolor='none', linewidth=0)
    ax.set_ylim(0, 1.7)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['polar'].set_visible(False)
    ax.set_facecolor('white')

    fig.tight_layout(pad=0.1)
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        print(f"Saved theta legend → {save_path}")
    return fig


def make_ecc_legend(save_path: Path | None = None, vmax: float = 4.0,
                    n_r: int = 200, n_theta: int = 360):
    """Minimalist polar eccentricity legend: a `nipy_spectral` disk from
    0° to vmax with light radial tick labels (since the scale is the
    legend's whole point). Designed to inset next to a flatmap.
    """
    r = np.linspace(0, vmax, n_r)
    theta = np.linspace(0, 2 * np.pi, n_theta)
    T, R = np.meshgrid(theta, r, indexing='ij')

    fig, ax = plt.subplots(figsize=(2, 2),
                           subplot_kw={'projection': 'polar'})
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    ax.pcolormesh(T, R, R, cmap='nipy_spectral', vmin=0.0, vmax=vmax,
                  shading='auto', rasterized=True)
    ax.set_xticks([])
    ticks = np.arange(1.0, vmax + 0.001, 1.0)
    ax.set_yticks(ticks)
    ax.set_yticklabels([f'{t:.0f}°' for t in ticks], fontsize=22,
                       fontweight='bold', color='white')
    for label in ax.get_yticklabels():
        label.set_path_effects([
            path_effects.Stroke(linewidth=3.5, foreground='black'),
            path_effects.Normal(),
        ])
    ax.set_rlabel_position(135)
    ax.set_ylim(0, vmax)
    ax.spines['polar'].set_visible(False)
    ax.grid(False)

    fig.tight_layout(pad=0.1)
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight', transparent=True)
        print(f"Saved eccentricity legend → {save_path}")
    return fig


def main(subject: int = 24, model: int = 4,
         fdr_alpha: float = 0.05, r2_thr: float | None = None,
         r2_max: float = 0.999):

    pc_subject = f'retsupp.sub-{subject:02d}'

    bids_folder = Path('/data/ds-retsupp')
    sub = Subject(subject, bids_folder=bids_folder)

    pars = sub.get_prf_parameters_surface(model=model)

    # Phantom-voxel guard: drop vertices with r² ≥ r2_max BEFORE computing
    # the FDR threshold. m4 (DoG + flex HRF) and m6 (DN) sometimes collapse
    # σ → 0 via softplus, producing NaN model predictions; the legacy
    # get_rsq pandas-sum-with-skipna=True reduces those to ss_resid=0 and
    # gives r²=1.0. Phantoms otherwise dominate the colormap.
    r2_vals = pars['r2'].values.copy()
    phantom = (r2_vals >= r2_max) | ~np.isfinite(r2_vals)
    print(f"  phantom-vertex count (r²≥{r2_max} or NaN): {int(phantom.sum())} "
          f"({phantom.mean():.2%} of vertices)")

    if r2_thr is None:
        # Whole-brain logit-Gaussian mixture on R² → tail-FDR threshold.
        # First call per (subject, model) fits + caches JSON + writes a
        # diagnostic PDF under derivatives/prf_diagnostics/r2_mixture/.
        r2_thr = sub.get_r2_fdr_threshold(model=model, alpha=fdr_alpha)
        print(f"Logit-Gaussian tail-FDR R² threshold (α={fdr_alpha}): "
              f"{r2_thr:.4f}")
    else:
        print(f"Using manual R² threshold: {r2_thr:.4f}")

    mask = ((r2_vals > r2_thr) & (r2_vals < r2_max)
            & np.isfinite(r2_vals))

    r2_vertex_thr = get_alpha_vertex(pars['r2'].values, mask, cmap='hot', vmin=0.0, vmax=0.5, subject=f'retsupp.sub-{subject:02d}')

    pars['theta'] = np.arctan2(pars['y'], pars['x'])

    theta_r = np.mod(pars['theta'].loc['R'], 2 * np.pi)
    theta_r = np.clip((theta_r - (.25 * np.pi)) / (1.5*np.pi), 0, 1)

    theta_l = np.mod(pars['theta'].loc['L'] + np.pi, 2 * np.pi)
    theta_l = -np.clip((theta_l - (.25 * np.pi)) / (1.5*np.pi), 0, 1) + 1


    theta_ = get_alpha_vertex(np.concatenate([theta_l, theta_r]), mask, cmap='hsv', vmin=0.0, vmax=1.0, subject=pc_subject)


    pars['ecc'] = np.sqrt(pars['x']**2 + pars['y']**2)
    theta = get_alpha_vertex(pars['theta'].values, mask, cmap='hsv', vmin=-np.pi, vmax=np.pi, subject=pc_subject)
    ecc = get_alpha_vertex(pars['ecc'].values, mask, cmap='nipy_spectral', vmin=0.0, vmax=4.0, subject=pc_subject)
    sd_vmax = float(np.quantile(pars['sd'].values[mask], 0.95))
    print(f"sd colormap: vmin=0, vmax={sd_vmax:.2f}° (95th pctile of sig vertices)")
    sd = get_alpha_vertex(pars['sd'].values, mask, cmap='viridis',
                          vmin=0.0, vmax=sd_vmax, subject=pc_subject)

    x = get_alpha_vertex(pars['x'].values, mask, cmap='coolwarm', vmin=-4.0, vmax=4.0, subject=pc_subject)
    y = get_alpha_vertex(pars['y'].values, mask, cmap='coolwarm', vmin=-4.0, vmax=4.0, subject=pc_subject)

    ds = {'theta': theta, 'r2_thr': r2_vertex_thr,
        'theta_': theta_, 'ecc': ecc,
        'x': x, 'y': y, 'sd': sd}

    # DN-specific overlays: models 5 (fixed HRF) and 6 (flex HRF) both
    # use DivisiveNormalizationGaussianPRF2DWithHRF and carry these
    # columns. Gate on column presence so future DN variants pick them
    # up automatically.
    for par in ('neural_baseline', 'surround_baseline'):
        if par in pars.columns:
            vmax = np.quantile(pars[par].values[mask], 0.95)
            ds[par] = get_alpha_vertex(pars[par].values, mask,
                                        cmap='viridis', vmin=0.0, vmax=vmax,
                                        subject=pc_subject)

    extra_pars = {
        'srf_size':       ('viridis',  'q'),
        'srf_amplitude':  ('viridis',  'q'),
        'rf_amplitude':   ('coolwarm', 'sym'),
        'amplitude':      ('coolwarm', 'sym'),
        'baseline':       ('coolwarm', 'sym'),
        'hrf_delay':      ('viridis',  'q'),
        'hrf_dispersion': ('viridis',  'q'),
        'bold_baseline':  ('coolwarm', 'sym'),
    }
    for par, (cmap, scaling) in extra_pars.items():
        if par not in pars.columns or par in ds:
            continue
        vals = pars[par].values
        finite_in_mask = mask & np.isfinite(vals)
        if not finite_in_mask.any():
            continue
        if scaling == 'q':
            vmin = float(np.quantile(vals[finite_in_mask], 0.05))
            vmax = float(np.quantile(vals[finite_in_mask], 0.95))
            if vmin == vmax:
                vmin, vmax = vmin - 1e-6, vmax + 1e-6
        else:
            v = float(np.quantile(np.abs(vals[finite_in_mask]), 0.95))
            vmin, vmax = -v, v
        print(f"{par} colormap: vmin={vmin:.3f}, vmax={vmax:.3f}")
        ds[par] = get_alpha_vertex(vals, mask, cmap=cmap,
                                   vmin=vmin, vmax=vmax, subject=pc_subject)



    # Prefix all model-derived overlays with subject + model so webshow can
    # disambiguate when multiple subjects/models are loaded side-by-side.
    model_prefix = f's{subject:02d}_m{model}_'
    ds = {f'{model_prefix}{k}': v for k, v in ds.items()}

    # Inferred (neuropythy) pars don't depend on the PRF model — prefix
    # only with subject so they group separately in the field list.
    subj_prefix = f's{subject:02d}_'
    try:
        inferred_pars = sub.get_inferred_prf_pars_surf()
        ecc_inferred = get_alpha_vertex(inferred_pars['eccen'].values, mask, cmap='nipy_spectral', vmin=0.0, vmax=4.0, subject=pc_subject)
        theta_inferred = get_alpha_vertex(inferred_pars['angle'].values, mask, cmap='hsv', vmin=0, vmax=180, subject=pc_subject)
        roi = get_alpha_vertex(inferred_pars['varea'].values, alpha=~inferred_pars['varea'].isnull().values, subject=pc_subject, cmap='tab20')

        ds.update({f'{subj_prefix}ecc_inferred': ecc_inferred,
                   f'{subj_prefix}roi': roi,
                   f'{subj_prefix}theta_inferred': theta_inferred})

    except Exception as e:
        print(f"Could not load inferred pars for subject {subject}: {e}")
        ecc_inferred = None
        theta_inferred = None
        roi = None



    ds = cortex.Dataset(**ds)

    cortex.webshow(ds)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('subject', type=int, default=24, help='Subject ID to visualize')
    argparser.add_argument('--model', type=int, default=4, help='Model')
    argparser.add_argument('--fdr-alpha', type=float, default=0.05,
                           help='Tail-FDR α for the whole-brain logit-Gaussian '
                                'mixture on R² (default: 0.05). Lower → stricter. '
                                'Ignored if --r2-thr is set. First call per '
                                '(subject, model) writes a cache JSON + '
                                'diagnostic PDF under derivatives/.')
    argparser.add_argument('--r2-thr', type=float, default=None,
                           help='Manual R² threshold; if given, overrides '
                                'FDR-based thresholding (revert path).')
    argparser.add_argument('--r2-max', type=float, default=0.999,
                           help='Upper R² bound — vertices ≥ this are treated '
                                'as phantom (σ→0 collapse) and masked out '
                                '(default 0.999).')
    args = argparser.parse_args()
    main(subject=args.subject, model=args.model,
         fdr_alpha=args.fdr_alpha, r2_thr=args.r2_thr,
         r2_max=args.r2_max)
