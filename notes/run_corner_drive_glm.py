"""Group-level GLM of decoded drive at the 4 corner locations.

For each (sub, ROI):
  - For each run, sample decoded drive inside a small disk at each of
    the 4 corner positions (4° eccentricity, diagonals).
  - For each (TR, corner) tuple, build 6 indicator regressors capturing
    the corner's cognitive meaning at that TR:
        HPD       = corner is the HP location AND distractor on it
        LPD-orth  = corner is an orthogonal LP AND distractor on it
        LPD-opp   = corner is the opposite LP AND distractor on it
        HPT       = corner is HP AND target on it
        LPT-orth  = corner is orth LP AND target on it
        LPT-opp   = corner is opp LP AND target on it
    Indicators are fractional in [0, 1] from
    Subject.get_dynamic_indicator / get_target_indicator (sub-TR
    overlap weighted).
  - Pool all (TR, corner) tuples and fit:
        drive = X · β + intercept
  - 6 β values per subject.

Group: per-condition mean β ± SEM across subjects + paired stats
(HP vs ortho, HP vs opp, etc.).

Excludes sub-01/02 (counterbalancing bug).

Usage::

    python notes/run_corner_drive_glm.py V3AB vox2000_psig0.5
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from retsupp.utils.data import Subject, distractor_locations


ROI = sys.argv[1] if len(sys.argv) > 1 else 'V3AB'
VOX_TAG = sys.argv[2] if len(sys.argv) > 2 else 'vox2000_psig0.5'
DISK_RADIUS = 0.6   # deg
EXCLUDE = {1, 2}
CHANNEL_LOCS = ['upper_right', 'upper_left', 'lower_left', 'lower_right']
RELATIONS = {
    0: {'HP': 0, 'orth': (1, 3), 'opp': 2},
    1: {'HP': 1, 'orth': (0, 2), 'opp': 3},
    2: {'HP': 2, 'orth': (1, 3), 'opp': 0},
    3: {'HP': 3, 'orth': (0, 2), 'opp': 1},
}
CONDITIONS = ['HPD', 'LPD-orth', 'LPD-opp',
              'HPT', 'LPT-orth', 'LPT-opp']
# Corner positions in degrees
CORNER_XY = {
    name.replace(' ', '_'): pos
    for name, pos in distractor_locations.items()
}


def _disk_means_per_tr(decoded: np.ndarray, grid: np.ndarray,
                       corners: list, radius: float) -> np.ndarray:
    """(T, V_corners) — mean decoded value inside the disk at each corner."""
    R = decoded.shape[-1]
    out = np.zeros((decoded.shape[0], len(corners)), dtype=np.float32)
    for i, (cx, cy) in enumerate(corners):
        d2 = (grid[:, 0] - cx)**2 + (grid[:, 1] - cy)**2
        mask = d2 <= radius**2
        if not mask.any():
            mask = np.zeros(grid.shape[0], dtype=bool)
            mask[d2.argmin()] = True
        out[:, i] = decoded.reshape(decoded.shape[0], -1)[:, mask].mean(axis=1)
    return out


def _corner_role(corner_ch: int, hp_ch: int) -> str:
    """Return 'HP' / 'orth' / 'opp' for a corner relative to the run's HP."""
    rel = RELATIONS[hp_ch]
    if corner_ch == rel['HP']:
        return 'HP'
    if corner_ch in rel['orth']:
        return 'orth'
    return 'opp'


def fit_subject(subject: int):
    base = Path(f'/data/ds-retsupp/derivatives/decoded/model4/sub-{subject:02d}')
    try:
        sub = Subject(subject, bids_folder='/data/ds-retsupp')
        hp_by_run = sub.get_hpd_locations()
    except Exception as e:
        print(f'  sub-{subject:02d}: subject init failed ({e})'); return None

    X_rows = []   # design matrix observations (one row per TR-corner)
    y_rows = []   # drive values
    for (ses, run), hp in hp_by_run.items():
        p = base / f'decoded_ses-{ses}_run-{run}_roi-{ROI}_{VOX_TAG}.npz'
        if not p.exists():
            continue
        with np.load(p) as d:
            dec = d['decoded'].astype(np.float32)
            grid = d['grid'].astype(np.float32)
        try:
            dist = sub.get_dynamic_indicator(session=ses, run=run)
            targ = sub.get_target_indicator(session=ses, run=run)
        except Exception:
            continue
        T = min(dec.shape[0], dist.shape[0], targ.shape[0])
        dec = dec[:T]
        dist = dist[:T]
        targ = targ[:T]

        # (T, 4) corner drive
        drive = _disk_means_per_tr(
            dec, grid,
            [CORNER_XY[c] for c in CHANNEL_LOCS],
            DISK_RADIUS)

        hp_ch = CHANNEL_LOCS.index(hp)
        # For each corner channel (=position), build 6 regressors per TR
        # PLUS a per-corner intercept (one-hot for the corner). Each
        # corner has a different baseline drive (bar-trajectory geometry,
        # PRF coverage); the per-corner intercept absorbs that so the
        # condition betas reflect event-related modulation only.
        for c in range(4):
            role = _corner_role(c, hp_ch)
            cols = {k: np.zeros(T, dtype=np.float32) for k in CONDITIONS}
            if role == 'HP':
                cols['HPD'] = dist[:, c]
                cols['HPT'] = targ[:, c]
            elif role == 'orth':
                cols['LPD-orth'] = dist[:, c]
                cols['LPT-orth'] = targ[:, c]
            else:
                cols['LPD-opp'] = dist[:, c]
                cols['LPT-opp'] = targ[:, c]
            X_run = np.stack([cols[k] for k in CONDITIONS], axis=1)
            # 4 per-corner intercepts (one-hot)
            corner_intercept = np.zeros((T, 4), dtype=np.float32)
            corner_intercept[:, c] = 1.0
            X_run = np.concatenate([X_run, corner_intercept], axis=1)
            X_rows.append(X_run)
            y_rows.append(drive[:, c])
    if not X_rows:
        return None
    X = np.concatenate(X_rows, axis=0)
    y = np.concatenate(y_rows, axis=0)
    B, *_ = np.linalg.lstsq(X, y, rcond=None)   # (10,)
    return {c: float(B[i]) for i, c in enumerate(CONDITIONS)}


def main():
    avail_subs = []
    base_all = Path('/data/ds-retsupp/derivatives/decoded/model4')
    for d in sorted(base_all.glob('sub-*')):
        try:
            n = int(d.name.split('-')[1])
        except (IndexError, ValueError):
            continue
        if n in EXCLUDE:
            continue
        if list(d.glob(f'decoded_*_roi-{ROI}_{VOX_TAG}.npz')):
            avail_subs.append(n)
    print(f'Subjects with {ROI} {VOX_TAG}: {len(avail_subs)}')

    rows = []
    t0 = time.time()
    for s in avail_subs:
        r = fit_subject(s)
        if r is None:
            continue
        for c, v in r.items():
            rows.append({'subject': s, 'condition': c, 'beta': v})
        print(f'  sub-{s:02d}: {[(c, round(r[c], 4)) for c in CONDITIONS]} '
              f'({time.time()-t0:.0f}s)')
    df = pd.DataFrame(rows)
    print(f'\nFitted {df["subject"].nunique()} subjects')

    # Summary
    summary = df.groupby('condition')['beta'].agg(['mean', 'std', 'sem',
                                                     'count']).loc[CONDITIONS]
    print('\nGroup summary:')
    print(summary.to_string(float_format=lambda v: f'{v:+.4f}'))

    # Paired stats: HPD vs each LP-distractor; HPT vs each LP-target.
    print('\nPaired tests (Wilcoxon signed-rank, two-sided):')
    pivot = df.pivot(index='subject', columns='condition', values='beta')
    for a, b in [('HPD', 'LPD-orth'), ('HPD', 'LPD-opp'),
                 ('HPT', 'LPT-orth'), ('HPT', 'LPT-opp'),
                 ('LPD-orth', 'LPD-opp'), ('LPT-orth', 'LPT-opp')]:
        d = pivot[a] - pivot[b]
        d = d.dropna()
        if len(d) < 5:
            print(f'  {a} vs {b}: n={len(d)} (skip)'); continue
        w = stats.wilcoxon(d, alternative='two-sided')
        print(f'  {a} − {b}: median diff = {d.median():+.4f}, '
              f'W={w.statistic:.1f}, p={w.pvalue:.4f}, n={len(d)}')

    # Plot — vision-science house style, split into Distractor / Target.
    import matplotlib as mpl
    mpl.rcParams.update({
        'font.family': 'Helvetica',
        'font.sans-serif': ['Helvetica', 'Helvetica Neue', 'TeX Gyre Heros', 'Arial'],
        'font.size': 9, 'axes.labelsize': 10, 'axes.titlesize': 10,
        'xtick.labelsize': 8, 'ytick.labelsize': 8,
        'axes.linewidth': 0.8,
        'axes.spines.top': False, 'axes.spines.right': False,
        'xtick.direction': 'out', 'ytick.direction': 'out',
        'xtick.major.size': 3, 'ytick.major.size': 3,
        'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
        'lines.linewidth': 1.2, 'lines.markersize': 4,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
        'savefig.dpi': 300, 'savefig.bbox': 'tight',
    })
    sns.set_context('paper')

    d_panel = ['HPD', 'LPD-orth', 'LPD-opp']
    t_panel = ['HPT', 'LPT-orth', 'LPT-opp']
    x_labels = ['HP', 'Ortho', 'Opp']

    fig, axes = plt.subplots(1, 2, figsize=(5.2, 3.0), sharey=True)
    palette_D = '#C44E52'   # red for distractor
    palette_T = '#3B5BA5'   # blue for target

    for ax, conds, color, panel_name in zip(
        axes, [d_panel, t_panel], [palette_D, palette_T],
        ['Distractor', 'Target']):
        sub_df = df[df['condition'].isin(conds)].copy()
        sub_df['cond_simple'] = sub_df['condition'].map(
            dict(zip(conds, x_labels)))
        # Subject-level lines (faint)
        for s, g in sub_df.groupby('subject'):
            g_sorted = g.set_index('cond_simple').reindex(x_labels)
            ax.plot(range(3), g_sorted['beta'], '-', color='0.7',
                     lw=0.6, alpha=0.7, zorder=1)
            ax.plot(range(3), g_sorted['beta'], 'o', color=color,
                     ms=3, alpha=0.5, mew=0, zorder=2)
        # Mean ± SEM marker
        means = sub_df.groupby('cond_simple', sort=False)['beta'].agg(['mean', 'sem'])
        means = means.reindex(x_labels)
        ax.errorbar(range(3), means['mean'], yerr=means['sem'],
                     fmt='D', color=color, mec='black', mew=1.5,
                     ms=10, capsize=0, elinewidth=1.0, zorder=10)
        ax.axhline(0, color='0.7', lw=0.6, ls='--', zorder=0)
        ax.set_xticks(range(3))
        ax.set_xticklabels(x_labels)
        ax.set_xlim(-0.4, 2.4)
        # Annotate panel role in-panel
        ax.text(0.5, 0.97, panel_name, transform=ax.transAxes,
                 ha='center', va='top', fontsize=10, weight='bold',
                 color=color)
    axes[0].set_ylabel('β  (decoded drive)')
    axes[0].set_xlabel('Event location vs HP')
    axes[1].set_xlabel('Event location vs HP')
    sns.despine(fig=fig, offset=5, trim=True)
    fig.suptitle(f'{ROI}  (n={df["subject"].nunique()} subjects, '
                  f'{VOX_TAG})', fontsize=10, y=1.02)
    out = Path('/Users/gdehol/git/retsupp/notes/figures/decode_sweep/m4/'
               f'group_{ROI}_{VOX_TAG}_corner_drive_glm.pdf')
    fig.savefig(out)
    plt.close(fig)
    print(f'\nWrote {out}')

    # Save TSV
    out_tsv = Path('/Users/gdehol/git/retsupp/notes/data/decode_sweep') / \
              f'corner_drive_glm_{ROI}_{VOX_TAG}.tsv'
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_tsv, sep='\t', index=False)
    print(f'Wrote {out_tsv}')


if __name__ == '__main__':
    main()
