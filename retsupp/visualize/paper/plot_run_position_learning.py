"""HP vs LP sustained gain across the 3 within-block run positions.

Renders a 3-row × 8-col figure:
  row 1: g_HP_pos0/1/2 per subject + median
  row 2: g_LP_pos0/1/2 per subject + median
  row 3: (g_HP - g_LP) per position — the "selectivity" trajectory

Wilcoxon p-values for "Δ(pos2-pos0) < 0" (one-sided) on each row.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

TSV = '/Users/gdehol/git/retsupp/notes/data/af_dog_v3_runPosition_parameters.tsv'
OUT = '/Users/gdehol/git/retsupp/notes/figures/run_position_learning.pdf'
ROI_ORDER = ['V1', 'V2', 'V3', 'V3AB', 'hV4', 'LO', 'TO', 'VO']


def wilcoxon_one_sided(x):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if len(x) < 5 or np.allclose(x, 0):
        return np.nan
    try:
        return wilcoxon(x, alternative='less').pvalue
    except ValueError:
        return np.nan


def panel(ax, df, hp_cols, label, color):
    pos = np.arange(3)
    for _, row in df.iterrows():
        ax.plot(pos, row[hp_cols].values, color=color, alpha=0.25, lw=0.6)
    median = df[hp_cols].median().values
    ax.plot(pos, median, 'o-', color='black', lw=2.5, ms=9, label='median')
    delta = df[hp_cols[2]] - df[hp_cols[0]]
    p = wilcoxon_one_sided(delta)
    ax.axhline(0, color='gray', ls=':', lw=0.7)
    ax.set_xticks(pos)
    ax.set_xticklabels(['1st', '2nd', '3rd'])
    title = f'Δ(pos2-pos0)={delta.median():+.2f}  p={p:.3f}'
    if not (np.isnan(p) or p < 0.05):
        title += ' n.s.'
    ax.set_title(title, fontsize=9)
    ax.set_ylabel(label, fontsize=9)
    ax.legend(loc='upper right', fontsize=7, framealpha=0.6)


df = pd.read_csv(TSV, sep='\t')
hp = ['g_HP_pos0', 'g_HP_pos1', 'g_HP_pos2']
lp = ['g_LP_pos0', 'g_LP_pos1', 'g_LP_pos2']

fig, axes = plt.subplots(3, 8, figsize=(22, 11), sharey='row')
for j, roi in enumerate(ROI_ORDER):
    sub = df[df.roi == roi]
    panel(axes[0, j], sub, hp, 'g_HP (sustained)', 'C0')
    panel(axes[1, j], sub, lp, 'g_LP (sustained)', 'C1')

    # selectivity row: difference per subject
    diff = sub[hp].values - sub[lp].values
    diff_df = pd.DataFrame(diff, columns=['d0', 'd1', 'd2'])
    panel(axes[2, j], diff_df, ['d0', 'd1', 'd2'], 'g_HP - g_LP', 'C2')

    axes[0, j].set_title(f'{roi}\n' + axes[0, j].get_title(), fontsize=10, fontweight='bold')
    axes[2, j].set_xlabel('within-block run')

for ax in axes.flat:
    ax.tick_params(labelsize=8)

fig.suptitle(
    'HP vs LP sustained gain across the 3 within-block run positions  (n≈28 per ROI)\n'
    'Row 3 = selectivity (HP − LP). Wilcoxon p one-sided "Δ(pos2−pos0) < 0".',
    fontsize=12, fontweight='bold', y=0.995,
)
fig.tight_layout(rect=(0, 0, 1, 0.98))
fig.savefig(OUT, bbox_inches='tight')
print('wrote', OUT)
