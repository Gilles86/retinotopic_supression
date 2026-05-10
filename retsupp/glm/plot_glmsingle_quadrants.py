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
ROI_ORDER = ['V1', 'V2', 'V3', 'V3AB', 'hV4', 'VO', 'LO', 'TO']
ROI_ALIASES = {
    'V3AB': ['V3A', 'V3B'],
    'VO': ['VO1', 'VO2'],
    'LO': ['LO1', 'LO2'],
    'TO': ['TO1', 'TO2'],
}


def hp_relation(quadrant: str, hp_quadrant: str) -> str:
    """Where is HP relative to this quadrant Q? HP/orth/opposite."""
    if quadrant == hp_quadrant:
        return 'HP'
    pairs = [('upper_right', 'lower_left'), ('upper_left', 'lower_right')]
    for a, b in pairs:
        if {quadrant, hp_quadrant} == {a, b}:
            return 'opposite'
    return 'orth'


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

            tgt = trials['target_label'].values == quad
            dist = trials['distractor_label'].values == quad
            hp_per_trial = trials['hp_location'].values
            # Per-trial classification:
            #   target            : target landed at Q
            #   distractor_HP     : distractor at Q AND run's HP == Q
            #   distractor_orth   : distractor at Q AND HP perpendicular
            #   distractor_opposite : distractor at Q AND HP at antipode
            #   neutral           : nothing at Q
            for trial_idx in range(min(len(trials), len(per_trial))):
                if tgt[trial_idx] and not dist[trial_idx]:
                    cond = 'target'
                elif dist[trial_idx] and not tgt[trial_idx]:
                    hp = hp_per_trial[trial_idx]
                    if pd.isna(hp) or not isinstance(hp, str):
                        cond = 'distractor_unknown'
                    else:
                        rel = hp_relation(quad, hp)
                        cond = f'distractor_{rel}'   # HP/orth/opposite
                elif not tgt[trial_idx] and not dist[trial_idx]:
                    cond = 'neutral'
                else:
                    continue
                rows.append(dict(
                    subject=subject, roi=roi, quadrant=quad,
                    condition=cond, n_voxels=n_vox,
                    beta=float(per_trial[trial_idx]),
                ))
    return pd.DataFrame(rows)


def _draw_bracket(ax, x1, x2, y, h, sig_text, fontsize=12):
    """Bracket + significance stars above two x positions."""
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y],
            lw=1.2, color='k', clip_on=False)
    ax.text((x1 + x2) / 2, y + h, sig_text,
            ha='center', va='bottom', fontsize=fontsize, color='k',
            clip_on=False)


def _draw_group_bracket(ax, x_single, x_group, y, h, sig_text,
                         fontsize=12):
    """Bracket with one foot on x_single and one foot per x in x_group
    (so it visually 'covers' the whole group). Stars centered above."""
    x_max_g = max(x_group)
    x_min_g = min(x_group)
    # Left single foot.
    ax.plot([x_single, x_single], [y, y + h], lw=1.2, color='k',
            clip_on=False)
    # Top horizontal connecting single side to far edge of group.
    far = x_max_g if x_single < x_min_g else x_min_g
    near_g = x_min_g if x_single < x_min_g else x_max_g
    ax.plot([x_single, far], [y + h, y + h], lw=1.2, color='k',
            clip_on=False)
    # Feet at each group x (and a continuous line across the group).
    for xg in x_group:
        ax.plot([xg, xg], [y + h, y], lw=1.2, color='k', clip_on=False)
    cx = (x_single + (x_min_g + x_max_g) / 2) / 2
    ax.text(cx, y + h, sig_text, ha='center', va='bottom',
             fontsize=fontsize, color='k', clip_on=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bids-folder', default='/data/ds-retsupp')
    p.add_argument('--subjects', type=int, nargs='+',
                   default=list(range(1, 31)))
    p.add_argument('--rois', nargs='+', default=ROI_ORDER)
    p.add_argument('--out', type=Path,
                   default=Path('notes/figures/glmsingle_quadrants.pdf'))
    p.add_argument('--from-tsv', type=Path, default=None,
                   help='Skip extraction; read long-format TSV directly. '
                        'Defaults to <out>.tsv if it exists.')
    args = p.parse_args()

    out = args.out
    if not out.is_absolute():
        out = Path('/Users/gdehol/git/retsupp') / out
    out.parent.mkdir(parents=True, exist_ok=True)
    bids = Path(args.bids_folder)

    print(f'Subjects: {args.subjects}')
    print(f'ROIs: {args.rois}')

    tsv_path = args.from_tsv if args.from_tsv else out.with_suffix('.tsv')
    if tsv_path.exists() and not args.from_tsv:
        print(f'Reading cached TSV: {tsv_path}  '
              f'(delete to force re-extract)')
    if tsv_path.exists():
        df = pd.read_csv(tsv_path, sep='\t')
        print(f'Loaded {len(df)} rows from {tsv_path}')
    else:
        all_rows = []
        for s in tqdm(args.subjects, desc='subjects'):
            try:
                d = per_subject_per_quadrant(s, bids, rois=args.rois)
            except Exception as e:
                print(f'sub-{s:02d}: failed ({e})'); continue
            if len(d):
                all_rows.append(d)
        if not all_rows:
            raise RuntimeError('No subject yielded data.')
        df = pd.concat(all_rows, ignore_index=True)
        df.to_csv(tsv_path, sep='\t', index=False)
        print(f'\nWrote: {tsv_path}  ({len(df)} rows)')

    # Collapse: mean beta per (subject, ROI, condition) — averages over
    # quadrants AND trials within each (subject, ROI, condition) cell.
    df_coll = (df.groupby(['subject', 'roi', 'condition'], as_index=False)
                  ['beta'].mean()
                  .rename(columns={'beta': 'mean_beta'}))

    cond_order = ['neutral', 'target',
                  'distractor_opposite', 'distractor_orth', 'distractor_HP']
    # Cluster the 3 distractor conditions tightly; gap between groups.
    cond_x = {
        'neutral':              0.0,
        'target':               1.6,
        'distractor_opposite':  3.2,
        'distractor_orth':      3.7,
        'distractor_HP':        4.2,
    }
    cond_color = {
        'neutral': '0.55',
        'target': '#2ca02c',
        'distractor_opposite': '#1f77b4',
        'distractor_orth': '#9467bd',
        'distractor_HP': '#d62728',
    }
    cond_short = {
        'neutral': 'Neutral',
        'target': 'Target',
        'distractor_opposite': 'D-opp',
        'distractor_orth': 'D-orth',
        'distractor_HP': 'D-HP',
    }

    with PdfPages(out) as pdf:
        # Page 1: per-ROI swarm plots with group mean ± SEM + brackets.
        from scipy import stats as scistats
        rois_pres = [r for r in args.rois if r in df_coll['roi'].unique()]
        n = len(rois_pres)
        ncol = 3
        nrow = int(np.ceil((n + 1) / ncol))   # +1 for the legend cell
        fig, axes = plt.subplots(nrow, ncol, figsize=(5.6 * ncol, 5.0 * nrow),
                                  sharey=False, squeeze=False)
        axes = axes.flatten()
        rng = np.random.default_rng(0)
        for i, roi in enumerate(rois_pres):
            ax = axes[i]
            roi_df = df_coll[df_coll['roi'] == roi]

            means_sems = []
            for ci, cond in enumerate(cond_order):
                vals = roi_df.loc[roi_df['condition'] == cond,
                                    'mean_beta'].values
                if len(vals) == 0:
                    continue
                xc = cond_x[cond]
                jitter = rng.normal(0, 0.07, len(vals))
                ax.scatter([xc] * len(vals) + jitter, vals,
                           color=cond_color[cond], alpha=0.45, s=30,
                           edgecolors='white', linewidths=0.4, zorder=2)
                m = float(np.mean(vals))
                sem = float(np.std(vals, ddof=1) / np.sqrt(len(vals))
                            if len(vals) > 1 else 0.0)
                means_sems.append((m, sem))
                ax.errorbar([xc], [m], yerr=[sem], fmt='o',
                            color='k', mfc='k', mec='k',
                            markersize=10, capsize=6, capthick=2.2,
                            elinewidth=2.2, zorder=3)

            # Y-lim: zoom on means ± SEM bars so the small differences
            # are readable. Headroom only for the bracket stack at top.
            ms = [x[0] for x in means_sems]
            sems = [x[1] for x in means_sems]
            m_lo = min(m - 2.5 * s for m, s in zip(ms, sems))
            m_hi = max(m + 2.5 * s for m, s in zip(ms, sems))
            span = max(m_hi - m_lo, 1e-3)
            lo = m_lo - 0.05 * span
            hi = m_hi
            ax.set_ylim(lo, hi + 0.85 * span)
            xs = [cond_x[c] for c in cond_order]
            ax.set_xlim(min(xs) - 0.6, max(xs) + 0.6)
            ax.set_xticks(xs)
            ax.set_xticklabels([cond_short[c] for c in cond_order],
                                rotation=30, fontsize=12, ha='right')
            ax.tick_params(axis='y', labelsize=11)
            ax.axhline(0, color='k', lw=0.6, alpha=0.4)
            ax.set_title(roi, fontsize=16, weight='bold')
            if i % ncol == 0:
                ax.set_ylabel('mean β  (per-quadrant, collapsed)',
                              fontsize=13)
            ax.grid(alpha=0.12, axis='y')

            # Brackets + stars for the contrasts of interest.
            pivot = roi_df.pivot_table(index='subject', columns='condition',
                                         values='mean_beta')
            d_cols = [c for c in
                      ('distractor_opposite', 'distractor_orth',
                       'distractor_HP') if c in pivot.columns]
            if len(d_cols) >= 2:
                pivot['all_distractors'] = pivot[d_cols].mean(axis=1)
            d_centroid = float(np.mean([cond_x[c] for c in d_cols]))
            xpos = dict(cond_x)
            xpos['all_distractors'] = d_centroid
            contrasts = [
                ('distractor_HP', 'distractor_opposite'),
                ('distractor_HP', 'distractor_orth'),
                ('target', 'neutral'),
                ('target', 'all_distractors'),
                ('neutral', 'all_distractors'),
            ]
            ymax = ax.get_ylim()[1]
            ymin = ax.get_ylim()[0]
            level_step = 0.065 * (ymax - ymin)
            level_y = m_hi + 0.10 * span
            for a, b in contrasts:
                if a not in pivot.columns or b not in pivot.columns:
                    continue
                paired = pivot[[a, b]].dropna()
                if len(paired) < 5:
                    continue
                t, p = scistats.ttest_rel(paired[a], paired[b])
                if p < 0.001:
                    sig = '***'
                elif p < 0.01:
                    sig = '**'
                elif p < 0.05:
                    sig = '*'
                else:
                    sig = 'n.s.'
                if sig == 'n.s.':
                    continue   # skip n.s. brackets to keep plot clean
                xa = xpos[a]
                xb = xpos[b]
                bh = 0.018 * (ymax - ymin)
                if 'all_distractors' in (a, b):
                    x_single = xa if a != 'all_distractors' else xb
                    x_group = [cond_x[c] for c in d_cols]
                    _draw_group_bracket(ax, x_single, x_group,
                                          level_y, bh, sig, fontsize=14)
                else:
                    _draw_bracket(ax, min(xa, xb), max(xa, xb),
                                  level_y, bh, sig, fontsize=14)
                level_y += level_step

        # Graphical inset in last cell (#8): per-condition mini-diamonds.
        # Each row shows the 4-position search array (HPD always on top)
        # with the focal Q highlighted in the condition color, and a
        # T/D/- letter denoting what landed there.
        legend_ax = axes[len(rois_pres)]
        legend_ax.set_xlim(0, 1); legend_ax.set_ylim(0, 1)
        legend_ax.set_xticks([]); legend_ax.set_yticks([])
        for spine in legend_ax.spines.values():
            spine.set_visible(False)

        legend_ax.text(0.50, 0.96, 'What landed at quadrant Q',
                        transform=legend_ax.transAxes,
                        fontsize=15, weight='bold', va='top', ha='center')
        legend_ax.text(0.50, 0.90,
                        '(HPD = high-prob distractor location, fixed per run)',
                        transform=legend_ax.transAxes,
                        fontsize=10, color='0.40', va='top', ha='center',
                        style='italic')

        # Position scheme inside each mini-diamond (axes-fraction offsets
        # from the icon's centre, with the icon centred at icon_cx).
        offsets = {
            'top':    ( 0.000,  0.045),   # HPD always here
            'right':  ( 0.045,  0.000),
            'bottom': ( 0.000, -0.045),
            'left':   (-0.045,  0.000),
        }
        # For each condition: (focal positions, letter-on-focal).
        # Distractor-orth pools BOTH perpendicular slots, so both are
        # highlighted (left + right). Other conditions: one focal slot.
        cond_focal = {
            'neutral':              (['right'],         '–'),
            'target':               (['right'],         'T'),
            'distractor_opposite':  (['bottom'],        'D'),
            'distractor_orth':      (['right', 'left'], 'D'),
            'distractor_HP':        (['top'],           'D'),
        }

        from matplotlib.patches import Circle
        row_top = 0.80
        row_h = 0.15
        icon_cx = 0.16
        for k, cond in enumerate(cond_order):
            cy = row_top - k * row_h
            focal_list, letter = cond_focal[cond]

            # Outer dotted ring suggests the search array.
            ring = Circle((icon_cx, cy), 0.052, ec='0.75', fc='none',
                          ls=':', lw=0.8, transform=legend_ax.transAxes,
                          clip_on=False)
            legend_ax.add_patch(ring)

            for pos_name, (dx, dy) in offsets.items():
                px, py = icon_cx + dx, cy + dy
                is_focal = (pos_name in focal_list)
                is_hpd = (pos_name == 'top')
                if is_hpd:
                    hpd_marker = Circle((px, py), 0.020,
                                         ec='#d62728', fc='none',
                                         ls='--', lw=1.0,
                                         transform=legend_ax.transAxes,
                                         clip_on=False, zorder=2)
                    legend_ax.add_patch(hpd_marker)
                if is_focal:
                    legend_ax.scatter([px], [py], s=240,
                                       color=cond_color[cond], alpha=0.85,
                                       edgecolors='black', linewidths=1.0,
                                       transform=legend_ax.transAxes,
                                       zorder=4, clip_on=False)
                    legend_ax.text(px, py, letter, ha='center',
                                    va='center', fontsize=11,
                                    fontweight='bold', color='white',
                                    transform=legend_ax.transAxes,
                                    zorder=5)
                else:
                    legend_ax.scatter([px], [py], s=70,
                                       color='0.85',
                                       edgecolors='0.55', linewidths=0.6,
                                       transform=legend_ax.transAxes,
                                       zorder=3, clip_on=False)

            # Bold short label + description right of the icon.
            legend_ax.text(0.32, cy + 0.022, cond_short[cond],
                            transform=legend_ax.transAxes,
                            fontsize=14, va='center', weight='bold',
                            color=cond_color[cond])
            descs = {
                'neutral': 'nothing at Q',
                'target': 'target at Q',
                'distractor_opposite': 'distractor at Q  (Q opposite HPD)',
                'distractor_orth':     'distractor at Q  (Q perpendicular)',
                'distractor_HP':       'distractor at Q  (Q = HPD)',
            }
            legend_ax.text(0.32, cy - 0.025, descs[cond],
                            transform=legend_ax.transAxes,
                            fontsize=10, va='center', color='0.30')

        # HPD legend: tiny dashed-ring sample.
        from matplotlib.patches import Circle as _C
        legend_ax.add_patch(_C((0.05, 0.04), 0.015, ec='#d62728',
                                fc='none', ls='--', lw=1.2,
                                transform=legend_ax.transAxes,
                                clip_on=False))
        legend_ax.text(0.09, 0.04, '= HPD (top of each icon)',
                        transform=legend_ax.transAxes, fontsize=9,
                        va='center', color='#d62728')
        legend_ax.text(0.55, 0.04,
                        '*** p<0.001   ** p<0.01   * p<0.05',
                        transform=legend_ax.transAxes, fontsize=9,
                        color='0.30', va='center')

        # Hide any remaining unused axes.
        for j in range(len(rois_pres) + 1, len(axes)):
            axes[j].axis('off')

        n_subs = df_coll['subject'].nunique()
        fig.suptitle(
            f'GLMSingle β at retinotopic quadrant: '
            f'distractor split by HP-relation '
            f'(n={n_subs}, collapsed UR/UL/LL/LR)',
            fontsize=18, weight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig); plt.close(fig)

    print(f'\nDone: {out}')


if __name__ == '__main__':
    main()
