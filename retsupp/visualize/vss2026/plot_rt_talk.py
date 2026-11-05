"""RT bar plots for VSS2026 talk.

Two panels in Theeuwes style:

A. Distractor location (HP / LP / Absent) — tests learned suppression
   of the high-probability distractor location. Prediction: RT(HP) <
   RT(LP) < RT(Absent baseline).

B. Target location (HP / LP), with-distractor trials only — tests
   whether the suppressed location also costs target processing.
   Prediction: RT(target@HP) > RT(target@LP).

Per-subject means; group bars with ±1 SEM; individual subjects as
overlaid thin gray lines (paired). All 30 subjects, correct trials only.
The HP location of a run comes from Subject.get_hpd_locations, since the
events of sub-01 and sub-02 have no HPL_distractor column.

Outputs:
    notes/figures/rt_talk_distractor.{pdf,svg}
    notes/figures/rt_talk_target.{pdf,svg}
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from retsupp.utils.data import Subject

OUT_DIR = Path('/Users/gdehol/git/retsupp/notes/figures')
EXCLUDE_SUBJECTS = set()
LOCATION_CODE = {'upper_right': 1., 'upper_left': 3., 'lower_left': 5., 'lower_right': 7.}

# Talk palette — slightly muted versus the brighter AF-figure palette.
# Each condition gets paired (light fill, saturated edge/marker) so the
# bar reads soft and the mean+SEM marker stays crisp.
TALK_HP_EDGE  = '#B83C3C'   # warm muted red
TALK_HP_FILL  = '#EFB4B4'
TALK_LP_EDGE  = '#2F6FA8'   # cooler ocean blue
TALK_LP_FILL  = '#A8C5DC'
TALK_ABS_EDGE = '#6E7480'   # cool slate
TALK_ABS_FILL = '#D6DADF'

TALK_FG    = '#1F2933'
TALK_MUTED = '#9AA5B1'


def _apply_rcparams_talk():
    mpl.rcParams.update({
        'font.family': 'Helvetica',
        'font.sans-serif': ['Helvetica', 'Helvetica Neue',
                            'TeX Gyre Heros', 'Arial'],
        'font.size': 14,
        'axes.labelsize': 17,
        'axes.titlesize': 19,
        'xtick.labelsize': 15,
        'ytick.labelsize': 13,
        'legend.fontsize': 13,
        'mathtext.fontset': 'stixsans',
        'axes.linewidth': 1.1,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.labelpad': 8,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.major.size': 4,
        'ytick.major.size': 4,
        'xtick.major.width': 1.1,
        'ytick.major.width': 1.1,
        'patch.linewidth': 0.6,
        'legend.frameon': False,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'svg.fonttype': 'none',
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
    })


def collect_trials(bids_folder: str) -> pd.DataFrame:
    """Concat per-trial feedback rows across all subjects/sessions/runs."""
    import yaml
    with open('/Users/gdehol/git/retsupp/retsupp/data/subjects.yml') as f:
        all_subs = yaml.safe_load(f)
    rows = []
    for sub_id in all_subs:
        if sub_id in EXCLUDE_SUBJECTS:
            continue
        try:
            sub = Subject(sub_id, bids_folder=bids_folder)
        except Exception as e:
            print(f'  sub-{sub_id:02d}: skip ({e})')
            continue
        hp_per_run = sub.get_hpd_locations()
        for sess in (1, 2):
            try:
                runs = sub.get_runs(sess)
            except Exception:
                continue
            for run in runs:
                try:
                    ev = sub.get_onsets(sess, run)
                except FileNotFoundError:
                    continue
                T = ev[ev['event_type'] == 'feedback'].copy()
                T['subject'] = sub_id
                T['session'] = sess
                T['run'] = run
                T['HPL_distractor'] = LOCATION_CODE[hp_per_run[(sess, run)]]
                rows.append(T[['subject', 'session', 'run', 'trial_nr',
                                'rt', 'correct', 'distractor_location',
                                'target_location', 'HPL_distractor']])
    df = pd.concat(rows, ignore_index=True)
    print(f'collected {len(df)} trials across {df.subject.nunique()} subjects')
    return df


def add_condition_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['dist_at_hp']   = df['distractor_location'] == df['HPL_distractor']
    df['target_at_hp'] = df['target_location']     == df['HPL_distractor']
    df['has_distractor'] = df['distractor_location'] != 10
    # Distractor location condition
    def dist_cond(r):
        if not r['has_distractor']:
            return 'Absent'
        return 'HP' if r['dist_at_hp'] else 'LP'
    df['dist_cond'] = df.apply(dist_cond, axis=1)
    # Target location condition (with-distractor trials only)
    df['target_cond'] = np.where(df['has_distractor'],
                                  np.where(df['target_at_hp'], 'HP', 'LP'),
                                  None)
    return df


def per_subject_means(df: pd.DataFrame, group_col: str, order: list[str]
                      ) -> pd.DataFrame:
    """Filter to correct trials, exclude NaN RTs, compute per-(sub, cond) mean."""
    valid = df[(df['correct'] == True) & df['rt'].notna()].copy()
    g = (valid.groupby(['subject', group_col])['rt']
              .mean().mul(1000).reset_index())   # → ms
    # ensure all subjects have all conditions
    keep = (g.groupby('subject')[group_col].nunique()
              == len(order))
    keep_subs = keep[keep].index
    g = g[g['subject'].isin(keep_subs)]
    print(f'  {group_col}: n_subjects = {g.subject.nunique()} '
          f'(of {df.subject.nunique()} total)')
    g[group_col] = pd.Categorical(g[group_col], categories=order, ordered=True)
    return g


def _format_p(p):
    """APA-style p-value label."""
    if np.isnan(p):     return 'n.s.'
    if p < 0.001:       return 'p < .001'
    if p < 0.01:        return f'p = {p:.3f}'
    return f'p = {p:.3f}'.replace('0.', '.')


def _annotate_significance(ax, x1, x2, y, p, delta_ms, color=TALK_FG):
    """Thin bracket + literal p-value + delta annotation."""
    p_txt = _format_p(p)
    h = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.012
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y],
            lw=1.0, color=color, solid_capstyle='round')
    sign = '+' if delta_ms > 0 else '−'
    ax.text((x1 + x2) / 2, y + h * 1.6,
            f'{p_txt}  ({sign}{abs(delta_ms):.0f} ms)',
            ha='center', va='bottom', fontsize=12, color=color)


def plot_bars(g: pd.DataFrame, group_col: str, order: list[str],
              fills: list[str], edges: list[str],
              title: str, ylabel: str,
              out_stem: str, pair_tests: list[tuple[str, str]],
              figsize=(7.8, 4.8),
              baseline_label: str | None = None):
    _apply_rcparams_talk()
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    summary = (g.groupby(group_col, observed=True)['rt']
                 .agg(['mean', 'sem', 'count'])
                 .reindex(order))
    x = np.arange(len(order))

    # Wide-form (subjects × conditions)
    wide = g.pivot(index='subject', columns=group_col, values='rt')
    wide = wide[order]
    n_subj = len(wide)

    y_lo = wide.values.min()
    y_hi = wide.values.max()
    pad  = (y_hi - y_lo) * 0.08

    # Optional baseline reference (e.g., the 'Absent' mean as a dashed line)
    if baseline_label is not None and baseline_label in order:
        baseline_y = float(summary.loc[baseline_label, 'mean'])
        ax.axhline(baseline_y, color='#9AA5B1', lw=1.0, ls=(0, (4, 3)),
                   zorder=1)
        ax.text(len(order) - 0.45, baseline_y, '  baseline',
                ha='left', va='center', fontsize=11, color='#5A6473',
                style='italic', zorder=1)

    # 1. Bars — soft pastel fill, saturated edge in the condition color.
    ax.bar(x, summary['mean'], color=fills,
           edgecolor=edges, linewidth=1.4, width=0.66, zorder=2)

    # 2. Per-subject dots — small, jittered, edge in the condition color
    rng = np.random.RandomState(0)
    jitter = (rng.rand(n_subj, len(order)) - 0.5) * 0.26
    for ci, (col, edge) in enumerate(zip(order, edges)):
        vals = wide[col].dropna().values
        xs = ci + jitter[:len(vals), ci]
        ax.scatter(xs, vals, s=18, facecolor='white', edgecolor=edge,
                   linewidth=0.9, alpha=0.85, zorder=4)

    # 3. Mean ± SEM markers on top of the bar — diamond with dark edge,
    #    T-style SEM whiskers above and below. Reads as a clear "this
    #    is the central tendency" sitting on top of the bar's silhouette.
    for xi, m, s, edge in zip(x, summary['mean'], summary['sem'], edges):
        ax.errorbar(xi, m, yerr=s,
                    fmt='D', markersize=8.5,
                    markerfacecolor=edge, markeredgecolor=TALK_FG,
                    markeredgewidth=1.4,
                    ecolor=TALK_FG, elinewidth=1.8, capsize=5,
                    capthick=1.6, zorder=6)

    # Y-range — generous headroom so brackets sit above the data
    n_pairs = len(pair_tests)
    head = pad * (1.8 + 1.4 * n_pairs)
    ax.set_ylim(y_lo - pad * 0.5, y_hi + head)

    # Brackets — narrowest pair lowest
    pair_widths = [(abs(order.index(b) - order.index(a)), a, b)
                   for a, b in pair_tests]
    pair_widths.sort()
    bracket_y0 = y_hi + pad * 0.4
    for i, (_, a, b) in enumerate(pair_widths):
        a_vals = wide[a].dropna().values
        b_vals = wide[b].dropna().values
        try:
            _, p = stats.wilcoxon(a_vals, b_vals)
        except ValueError:
            p = np.nan
        delta = float(np.mean(a_vals - b_vals))
        ia, ib = order.index(a), order.index(b)
        _annotate_significance(ax, ia, ib, bracket_y0 + i * pad * 1.35,
                               p, delta_ms=delta)
        print(f'  {a} vs {b}: Wilcoxon p = {p:.4f}  '
              f'(mean Δ = {delta:+.1f} ms)')

    # Y-axis cleanup — major tick every 50 ms
    from matplotlib.ticker import MultipleLocator
    ax.yaxis.set_major_locator(MultipleLocator(50))
    ax.yaxis.set_minor_locator(MultipleLocator(25))

    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=12, fontweight='bold')
    ax.set_xlim(-0.55, len(order) - 0.45)
    ax.text(0.02, 0.97, f'n = {n_subj}', transform=ax.transAxes,
            ha='left', va='top', fontsize=12, color='#5A6473')
    sns.despine(ax=ax, offset=6, trim=False)

    pdf = OUT_DIR / f'{out_stem}.pdf'
    svg = OUT_DIR / f'{out_stem}.svg'
    fig.savefig(pdf)
    fig.savefig(svg)
    plt.close(fig)
    print(f'wrote {pdf}')
    print(f'wrote {svg}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--bids-folder', default='/data/ds-retsupp')
    args = p.parse_args()

    df = collect_trials(args.bids_folder)
    df = add_condition_labels(df)

    # Panel A: distractor location
    print('\n=== distractor location ===')
    g_d = per_subject_means(df, 'dist_cond', ['HP', 'LP', 'Absent'])
    plot_bars(
        g_d, 'dist_cond',
        order=['HP', 'LP', 'Absent'],
        fills=[TALK_HP_FILL, TALK_LP_FILL, TALK_ABS_FILL],
        edges=[TALK_HP_EDGE, TALK_LP_EDGE, TALK_ABS_EDGE],
        title='Distractor location',
        ylabel='Response time (ms)',
        out_stem='rt_talk_distractor',
        pair_tests=[('HP', 'LP'), ('LP', 'Absent'), ('HP', 'Absent')],
        baseline_label='Absent',
    )

    # Panel B: target location (with-distractor trials only)
    print('\n=== target location ===')
    df_wd = df[df['has_distractor']].copy()
    g_t = per_subject_means(df_wd, 'target_cond', ['HP', 'LP'])
    plot_bars(
        g_t, 'target_cond',
        order=['HP', 'LP'],
        fills=[TALK_HP_FILL, TALK_LP_FILL],
        edges=[TALK_HP_EDGE, TALK_LP_EDGE],
        title='Target location (with-distractor trials)',
        ylabel='Response time (ms)',
        out_stem='rt_talk_target',
        pair_tests=[('HP', 'LP')],
        figsize=(6.0, 4.8),
    )


if __name__ == '__main__':
    main()
