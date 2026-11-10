"""Decoding of the bar stimulus with different pRF models: DN vs DoG (same per-ROI HRF), and the joint-HRF DoG of Fig. 6.

Reads the summaries written by ``fig6_decoding collect`` (readout) and ``fig6_collect`` (quality, bands) for each
``--sources`` entry (``notes/data/decoding_{readout,quality,bands}_<source>_2026-11.tsv[.gz]``) and prints, per source
and decoder (free = every pixel, outline = bar outline given):

* decoding quality per ROI: median over subjects of the frame-wise r, and decoded vs shuffled frames (one-sided Wilcoxon);
* HP < other quadrants per ROI (one-sided Wilcoxon, Benjamini-Hochberg across the ROIs) and pooled over V1–V3;
* quadrant contrasts pooled over V1–V3 (HP / next to HP / opposite, % of the four-quadrant mean; two-sided Wilcoxon);
* the HP − other difference per eccentricity band (% of the other quadrants; one-sided Wilcoxon, uncorrected).

Tables go to ``notes/data/decoding_prf_models_<table>_2026-11.tsv``; ``--figure`` also draws the V1–V3 quadrant
contrasts per pRF model (``notes/figures/paper_2026-11/figS_decoding_prf_models.pdf``).

Usage (local)::

    python -m retsupp.visualize.paper.compare_decoding_prf_models --figure
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy import stats

from retsupp.visualize.paper.fig6_decoding import DATA, EARLY, band_effects, effects, quadrant_effects

ALL_ROIS = ['V1', 'V2', 'V3', 'V3AB', 'IPS', 'hV4', 'VO', 'LO', 'TO']
LABELS = {'dn_roihrf': 'DN, per-ROI HRF', 'dog_roihrf': 'DoG, per-ROI HRF', 'jointhrf': 'DoG, joint HRF (Fig. 6)',
          'gauss_jointhrf': 'Gaussian, joint HRF', 'dn_jointhrf': 'DN, joint HRF'}
DECODERS = ['free', 'outline']
BAND_KEYS = ['all', '0-1.5', '1.5-2.5', '2.5-ap']


def read(kind, source):
    for ext in ('.tsv', '.tsv.gz'):
        f = DATA / f'decoding_{kind}_{source}_2026-11{ext}'
        if f.exists():
            return pd.read_csv(f, sep='\t')
    raise FileNotFoundError(f'decoding_{kind}_{source}_2026-11')


def quality_table(q, source):
    rows = []
    for dec in DECODERS:
        s = q[q.decoder == dec].groupby(['subject', 'roi'])[['r', 'r_shuffle']].mean().reset_index()
        for roi in ALL_ROIS:
            g = s[s.roi == roi]
            if len(g) < 5:
                continue
            rows.append(dict(source=source, decoder=dec, roi=roi, n=len(g), r_median=g.r.median(),
                             r_shuffle_median=g.r_shuffle.median(),
                             p_vs_shuffle=stats.wilcoxon(g.r - g.r_shuffle, alternative='greater').pvalue))
    return pd.DataFrame(rows)


def hp_table(d, source):
    e = effects(d)
    # Relative effect: mean over runs of HP − other, in % of the mean over runs of the other quadrants.
    lp = d[~d.is_hp].groupby(['subject', 'roi', 'decoder', 'session', 'run']).drive_bar.mean()
    lp = lp.groupby(['subject', 'roi', 'decoder']).mean().rename('lp').reset_index()
    rel = e.merge(lp, on=['subject', 'roi', 'decoder'])
    rel = rel.assign(rel=100 * rel['diff'] / rel.lp)[['subject', 'roi', 'decoder', 'rel']]
    e = e.merge(rel, on=['subject', 'roi', 'decoder'])
    rows = []
    for dec in DECODERS:
        ed = e[e.decoder == dec]
        sub = []
        for roi in ALL_ROIS:
            g = ed[ed.roi == roi].dropna(subset=['diff'])
            if len(g) < 5:
                continue
            sub.append(dict(source=source, decoder=dec, roi=roi, n=len(g), rel_median=g.rel.median(), pct_below=(g['diff'] < 0).mean() * 100,
                            p=stats.wilcoxon(g['diff'], alternative='less').pvalue))
        t = pd.DataFrame(sub)
        t['q_bh'] = stats.false_discovery_control(t.p)
        pooled = ed[ed.roi.isin(EARLY)].groupby('subject')[['diff', 'rel']].mean()
        t = pd.concat([t, pd.DataFrame([dict(source=source, decoder=dec, roi='V1–V3', n=len(pooled),
                                             rel_median=pooled.rel.median(),
                                             pct_below=(pooled['diff'] < 0).mean() * 100,
                                             p=stats.wilcoxon(pooled['diff'], alternative='less').pvalue,
                                             q_bh=np.nan)])])
        rows.append(t)
    return pd.concat(rows)


def quadrant_table(d, source):
    rows, per_sub = [], []
    for dec in DECODERS:
        s = quadrant_effects(d, dec)
        per_sub.append(s.assign(source=source, decoder=dec).reset_index())
        m, se = s.mean(), s.sem()
        rows.append(dict(source=source, decoder=dec, n=len(s),
                         hp=m.hp, hp_sem=se.hp, adjacent=m.adjacent, adjacent_sem=se.adjacent,
                         opposite=m.opposite, opposite_sem=se.opposite,
                         p_hp_vs_adj=stats.wilcoxon(s.hp - s.adjacent).pvalue,
                         p_adj_vs_opp=stats.wilcoxon(s.adjacent - s.opposite).pvalue,
                         p_hp_vs_opp=stats.wilcoxon(s.hp - s.opposite).pvalue,
                         p_left_vs_right=stats.wilcoxon(s.left - s.right).pvalue,
                         pct_adj_gt_opp=(s.adjacent > s.opposite).mean() * 100))
    return pd.DataFrame(rows), pd.concat(per_sub)


def band_table(b, source):
    rows = []
    for dec in DECODERS:
        be = band_effects(b, dec)
        for k in BAND_KEYS:
            v = be[k].dropna()
            rows.append(dict(source=source, decoder=dec, band=k, n=len(v), mean=v.mean(), sem=v.sem(),
                             p=stats.wilcoxon(v, alternative='less').pvalue))
    return pd.DataFrame(rows)


def figure(per_sub, sources, name='figS_decoding_prf_models'):
    import matplotlib.pyplot as plt
    from retsupp.visualize.paper.fig6_decoding import LOC_COLORS
    from retsupp.visualize.paper.paper_style import DOUBLE, FG, letter, offset_spines, save

    classes = [('hp', 'HP'), ('adjacent', 'Next to HP'), ('opposite', 'Opposite')]
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.3), sharey=True, constrained_layout=True)
    rng = np.random.default_rng(0)
    for ax, dec, lab in zip(axes, DECODERS, 'ab'):
        for i, src in enumerate(sources):
            s = per_sub[(per_sub.source == src) & (per_sub.decoder == dec)].set_index('subject')
            v = s[[c for c, _ in classes]]
            # within-subject SEM (Cousineau-Morey)
            w = v.sub(v.mean(1), axis=0) + v.to_numpy().mean()
            k = len(classes)
            wsem = w.std(ddof=1) / np.sqrt(len(w)) * np.sqrt(k / (k - 1))
            for j, (c, _) in enumerate(classes):
                x = i * 4 + j
                ax.scatter(x + rng.uniform(-.18, .18, len(v)), v[c], s=5, color=LOC_COLORS[c], alpha=.4, lw=0)
                ax.errorbar(x, v[c].mean(), wsem[c], fmt='o', color=LOC_COLORS[c], mec=FG, mew=.8, ms=5,
                            elinewidth=1.2, capsize=0, zorder=5)
            p = stats.wilcoxon(v.adjacent - v.opposite).pvalue
            top = 6.2
            ax.plot([i * 4 + 1, i * 4 + 2], [top, top], color=FG, lw=.7)
            ax.text(i * 4 + 1.5, top + .2, f'p = {p:.3f}' if p >= .001 else 'p < .001', ha='center', va='bottom',
                    fontsize=6)
        ax.axhline(0, color='.6', lw=.6, ls='--', zorder=0)
        ax.set_xticks([i * 4 + 1 for i in range(len(sources))])
        ax.set_xticklabels([LABELS[s].replace(' (Fig. 6)', '').replace(', ', ',\n') for s in sources])
        ax.set_title('Unconstrained decoder' if dec == 'free' else 'Bar outline given', fontsize=7)
        ax.set_ylim(-8, 7.5)
        offset_spines(ax)
        letter(ax, lab)
    axes[0].set_ylabel('Decoded on bar, V1–V3\n(% of quadrant mean)')
    for j, (c, t) in enumerate(classes):
        axes[1].text(.99, .2 - j * .085, t, color=LOC_COLORS[c] if c != 'opposite' else '.45',
                     transform=axes[1].transAxes, ha='right', fontsize=6.5)
    legend = ("**Figure S. Quadrant contrasts of the decoded bar with different pRF models.** Decoded intensity on the "
              "bar's pixels per quadrant relative to the run's HP location (HP, the two quadrants next to it pooled, "
              "opposite), in % of the mean of the four, per subject the mean over runs and V1–V3. pRFs: "
              + ('; '.join(LABELS[s2] for s2 in sources) if all('jointhrf' in s2 for s2 in sources) else
                 "divisive normalization (DN) or difference of Gaussians (DoG) with the HRF fixed per subject and ROI "
                 "(median of the training-fold fits), and DoG with the HRF fitted jointly (Fig. 6)")
              + (" (each with one HRF per subject × ROI fitted jointly with its pRFs)" if all('jointhrf' in s2 for s2 in sources) else "")
              + ". a, Unconstrained decoder; "
              "b, bar outline given. Dots, subjects; points and bars, mean and within-subject SEM (Cousineau-Morey). "
              "p: next to HP vs opposite, two-sided Wilcoxon signed-rank test.")
    save(fig, name, legend)


def main(sources, do_figure, suffix=''):
    out = {k: [] for k in ('quality', 'hp', 'quadrants', 'bands')}
    per_sub = []
    for src in sources:
        d = read('readout', src)
        out['quality'].append(quality_table(read('quality', src), src))
        out['hp'].append(hp_table(d, src))
        qt, ps = quadrant_table(d, src)
        out['quadrants'].append(qt)
        per_sub.append(ps)
        out['bands'].append(band_table(read('bands', src), src))
    pd.set_option('display.width', 220)
    for k, v in out.items():
        t = pd.concat(v)
        t.to_csv(DATA / f'decoding_prf_models_{k}_2026-11{suffix}.tsv', sep='\t', index=False)
        print(f'\n== {k}')
        print(t.round(4).to_string(index=False))
    per_sub = pd.concat(per_sub)
    if do_figure:
        figure(per_sub, sources, name='figS_decoding_prf_models' + suffix)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sources', nargs='+', default=['dn_roihrf', 'dog_roihrf', 'jointhrf'])
    ap.add_argument('--figure', action='store_true')
    ap.add_argument('--suffix', default='', help='appended to the output table and figure names')
    a = ap.parse_args()
    main(a.sources, a.figure, a.suffix)
