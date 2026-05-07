"""Aggregate the per-subject CV TSVs from cv_compare_prf_extended.py and
produce the headline plot:  delta_r2_test (extended - bar) as a function
of bar-only PRF eccentricity, faceted by subject.

The hypothesis (cluster plan #2): voxels whose bar-only PRF center is
near the distractor ring (ecc ~ 3-4°) should have positive delta_r2_test
on average; foveal voxels (ecc < 1°) should have ~0 or slightly
negative delta.

Usage:
    python plot_cv_compare_prf_extended.py \
        --bids_folder /shares/zne.uzh/gdehol/ds-retsupp \
        --output notes/cv_compare_prf_extended.pdf
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def main(bids_folder='/data/ds-retsupp',
         output='notes/cv_compare_prf_extended.pdf',
         min_train_r2=0.05):
    base = Path(bids_folder) / 'derivatives' / 'prf_extended_cv'
    files = sorted(base.glob('sub-*/sub-*_cv_compare.tsv'))
    if not files:
        raise FileNotFoundError(f'No CV TSVs found under {base}')
    print(f"Loading {len(files)} CV files ...")
    dfs = [pd.read_csv(f, sep='\t') for f in files]
    df = pd.concat(dfs, ignore_index=True)
    print(df.head())
    print(df.describe())

    # Quality filter: keep voxels that fit reasonably well on training.
    df = df.query(
        f'r2_bar_train > {min_train_r2} or r2_ext_train > {min_train_r2}'
    ).copy()

    # Eccentricity bins from the BAR-ONLY model — that is the "honest"
    # x-axis (we use it to predict where the distractor model should help).
    df['ecc_bin'] = pd.cut(
        df['ecc_bar'], bins=[0, 1, 2, 3, 4, 5, 8],
        labels=['<1°', '1-2°', '2-3°', '3-4°', '4-5°', '>5°'],
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.violinplot(
        data=df, x='ecc_bin', y='delta_r2_test',
        ax=axes[0], inner='quartile', cut=0,
    )
    axes[0].axhline(0, color='k', linestyle='--', linewidth=0.8)
    axes[0].set_xlabel('PRF eccentricity (bar-only model)')
    axes[0].set_ylabel('Δ R²_test  (extended − bar-only)')
    axes[0].set_title(f'Pooled across {df["subject"].nunique()} subjects')

    summary = (
        df.groupby(['subject', 'ecc_bin'], observed=True)
          ['delta_r2_test'].mean().reset_index()
    )
    sns.pointplot(
        data=summary, x='ecc_bin', y='delta_r2_test',
        ax=axes[1], errorbar=('ci', 95),
    )
    axes[1].axhline(0, color='k', linestyle='--', linewidth=0.8)
    axes[1].set_xlabel('PRF eccentricity (bar-only model)')
    axes[1].set_ylabel('mean Δ R²_test  (per-subject mean ± 95% CI)')
    axes[1].set_title('Subject-level summary')

    fig.suptitle('Cross-validated PRF model comparison: extended vs bar-only')
    fig.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=120)
    print(f"Saved -> {output}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--bids_folder', default='/data/ds-retsupp')
    p.add_argument('--output', default='notes/cv_compare_prf_extended.pdf')
    p.add_argument('--min_train_r2', type=float, default=0.05)
    a = p.parse_args()
    main(bids_folder=a.bids_folder, output=a.output,
         min_train_r2=a.min_train_r2)
