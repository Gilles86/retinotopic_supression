"""Figure 6 and the pre-specified decoding test: decoded bar drive in the quadrant facing HP vs the other quadrants.

``collect`` (cluster): concatenates the per-run readouts of ``retsupp/decode/decode_final.py`` into one table.
``plot`` (local): per subject and ROI, the mean over runs of drive(HP quadrant) − mean drive(three other quadrants);
Wilcoxon signed-rank across subjects, one-sided (HP < LP), Holm across V1, V2, V3 (``notes/primary_test_2026-11.md``).
The outline-fixed decoder (bar footprint given, intensities decoded) is the test; the free decoder is shown beside it.

Usage::

    python -m retsupp.visualize.paper.fig6_decoding collect --source canonical \\
        --out /shares/zne.uzh/gdehol/ds-retsupp/derivatives/paper_2026-11/decoding_readout_canonical.tsv
    python -m retsupp.visualize.paper.fig6_decoding plot --table notes/data/decoding_readout_canonical_2026-11.tsv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROIS = ['V1', 'V2', 'V3']
DECODERS = [('outline', 'Bar outline given'), ('free', 'Unconstrained')]
LEGEND = """
**Figure 6. Decoding without gain fields: less bar drive in the quadrant facing the HP location.** The bar stimulus of each
run was decoded from V1, V2 and V3 with pRFs fitted on other runs (cross-validation folds) and no gain terms, voxels with
cross-run reliability > 0 on the training runs. a, Bar outline given, intensities decoded (the pre-specified test). b,
Every pixel decoded. Per subject, the mean decoded intensity on the bar's pixels inside the aperture in the quadrant facing
HP minus the mean over the three other quadrants, averaged over runs: dots are subjects (n = {n}), points and bars the
median and bootstrapped 95% CI. Wilcoxon signed-rank test, one-sided (HP < other quadrants), Holm across the three ROIs:
{tests}. {source}
"""


def collect(bids, source, out):
    root = Path(bids) / 'derivatives' / 'decoded_final_2026-11' / source
    files = sorted(root.glob('*/sub-*/readout_*.tsv'))
    d = pd.concat([pd.read_csv(f, sep='\t') for f in files])
    d.to_csv(out, sep='\t', index=False)
    print(f'{len(files)} readouts -> {out}')


def effects(d, measure='drive_bar'):
    """Per subject × ROI × decoder: mean over runs of HP − mean(other quadrants)."""
    g = d.groupby(['subject', 'roi', 'decoder', 'session', 'run'])
    run = g.apply(lambda x: x.loc[x.is_hp, measure].mean() - x.loc[~x.is_hp, measure].mean(), include_groups=False)
    return run.groupby(['subject', 'roi', 'decoder']).mean().rename('diff').reset_index()


def test(e):
    rows = []
    for dec, _ in DECODERS:
        ps = []
        for roi in ROIS:
            v = e[(e.decoder == dec) & (e.roi == roi)]['diff'].dropna().to_numpy()
            if len(v) < 5:
                continue
            w = stats.wilcoxon(v, alternative='less')
            rows.append(dict(decoder=dec, roi=roi, n=len(v), median=np.median(v), pct_below=(v < 0).mean() * 100, p=w.pvalue))
            ps.append(w.pvalue)
    t = pd.DataFrame(rows)
    for dec in t.decoder.unique():
        k = t.decoder == dec
        p = t.loc[k, 'p'].to_numpy()
        order = np.argsort(p)
        holm = np.empty_like(p)
        run = 0.
        for i, j in enumerate(order):
            run = max(run, min(1., (len(p) - i) * p[j]))
            holm[j] = run
        t.loc[k, 'p_holm'] = holm
    return t


def plot(table, name='fig6_decoding', source_note=''):
    import matplotlib.pyplot as plt
    from retsupp.visualize.paper.fig45_common import median_ci
    from retsupp.visualize.paper.paper_style import FG, HP, SINGLE, letter, offset_spines, save
    d = pd.read_csv(table, sep='\t')
    e = effects(d)
    t = test(e)
    print(t.round(4).to_string(index=False))
    fig, axes = plt.subplots(1, 2, figsize=(SINGLE * 1.5, 2.1), sharey=True,
                             gridspec_kw=dict(left=.14, right=.98, top=.8, bottom=.16, wspace=.15))
    for ax, (dec, title), let in zip(axes, DECODERS, 'ab'):
        for i, roi in enumerate(ROIS):
            v = e[(e.decoder == dec) & (e.roi == roi)]['diff'].dropna().to_numpy()
            if not len(v):
                continue
            rng = np.random.default_rng(i)
            ax.scatter(i + rng.uniform(-.15, .15, len(v)), v, s=8, color=HP, alpha=.35, lw=0)
            m, lo, hi = median_ci(v)
            ax.errorbar(i + .3, m, yerr=[[m - lo], [hi - m]], fmt='o', ms=5.5, color=HP, mec='w', mew=.7, lw=1.4, capsize=0)
            r = t[(t.decoder == dec) & (t.roi == roi)]
            if len(r) and r.p_holm.iloc[0] < .05:
                q = r.p_holm.iloc[0]
                ax.text(i + .1, 1.0, '***' if q < .001 else '**' if q < .01 else '*', transform=ax.get_xaxis_transform(),
                        ha='center', va='bottom', fontsize=7, color=HP)
        ax.axhline(0, color='.6', lw=.7, ls='--', zorder=0)
        ax.set_xticks(np.arange(len(ROIS)) + .1)
        ax.set_xticklabels(ROIS)
        ax.set_xlim(-.5, len(ROIS) - .3)
        ax.set_title(title, fontsize=7, color=FG, loc='left', pad=10)
        offset_spines(ax)
        letter(ax, let, x=-.12 if let == 'b' else -.3, y=1.12)
    axes[0].set_ylabel('Decoded drive,\nHP − other quadrants')
    tests = '; '.join(f'{r.roi} ({r.decoder}) median {r["median"]:+.3f}, {r.pct_below:.0f}% below 0, Holm p = {r.p_holm:.2g}'
                      for _, r in t.iterrows())
    save(fig, name, LEGEND.format(n=e.subject.nunique(), tests=tests, source=source_note))
    return t


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('step', choices=['collect', 'plot'])
    ap.add_argument('--bids-folder', default='/shares/zne.uzh/gdehol/ds-retsupp')
    ap.add_argument('--source', default='canonical')
    ap.add_argument('--out', default=None)
    ap.add_argument('--table', default=None)
    ap.add_argument('--name', default='fig6_decoding')
    a = ap.parse_args()
    if a.step == 'collect':
        collect(a.bids_folder, a.source, a.out)
    else:
        plot(a.table, a.name, source_note=f'pRFs: {a.source} fold fits.')
