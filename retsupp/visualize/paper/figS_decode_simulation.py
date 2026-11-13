"""Supplementary figure + stats: can the decoder turn a gain field at the HP item alone into the observed
"next to HP higher than opposite" quadrant pattern?

Input: ``notes/data/decode_sim_2026-11.tsv`` (``retsupp/decode/collect_simulate_decode.py``; readouts of
``retsupp/decode/simulate_decode.py``, canonical-HRF DoG fold pRFs, phase-randomised real residuals, sustained gain
field at HP only, FWHM 4.8°, A ∈ {0, −0.025, −0.075}) and the real readouts ``notes/data/decoding_readout_*_2026-11.tsv``.

Per subject × ROI × amplitude × decoder, quadrants are rotated to HP / left / opposite / right of HP, ``drive_bar``
(decoded) and ``true_drive_bar`` (true simulated stimulus) are averaged over runs and expressed in % of the mean of
the four quadrants (as Fig. 6e/d). The effect of the field is the paired difference gain − null (same noise draw).
Pooled = mean over V1–V3 per subject.

Usage (local)::

    ~/mambaforge/envs/retsupp/bin/python -m retsupp.visualize.paper.figS_decode_simulation
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

DATA = Path('notes/data')
FIGS = Path('notes/figures/paper_2026-11')
SUFFIX = ''
QUAD_ANGLE = {'upper_right': 45, 'upper_left': 135, 'lower_left': 225, 'lower_right': 315}
POS = {0: 'hp', 90: 'left', 180: 'opposite', 270: 'right'}
ROIS = ['V1', 'V2', 'V3']
from retsupp.visualize.paper.paper_style import LOC_COLORS as COLS  # noqa: E402  (palette A: HP and its LP tints)
LABELS = {'hp': 'HP', 'adjacent': 'Next to HP', 'opposite': 'Opposite'}


def rel_quadrants(d, measure):
    """Per (amplitude, decoder, subject, roi): measure per position, % of the mean of the four."""
    d = d.copy()
    d['pos'] = ((d.quadrant.map(QUAD_ANGLE) - d.hp_location.map(QUAD_ANGLE)) % 360).map(POS)
    keys = [k for k in ('gain_amplitude', 'decoder') if k in d] + ['subject', 'roi']
    r = d.groupby(keys + ['pos'])[measure].mean().unstack()
    r = r.div(r.mean(1), axis=0) * 100 - 100
    r['adjacent'] = (r.left + r.right) / 2
    return r


def pooled(r):
    lv = [n for n in r.index.names if n != 'roi']
    p = r.groupby(lv).mean()
    p['roi'] = 'V1–V3'
    return pd.concat([r.reset_index(), p.reset_index()], ignore_index=True)


def summarise(eff, label):
    rows = []
    for (amp, dec, roi), g in eff.groupby(['gain_amplitude', 'decoder', 'roi']):
        row = dict(kind=label, gain_amplitude=amp, decoder=dec, roi=roi, n=len(g))
        for c in ('hp', 'adjacent', 'opposite', 'left', 'right'):
            row[f'{c}_mean'], row[f'{c}_sem'] = g[c].mean(), g[c].sem()
        for name, x in (('hp_minus_next', g.hp - g.adjacent), ('next_minus_opp', g.adjacent - g.opposite),
                        ('hp_minus_opp', g.hp - g.opposite)):
            row[f'{name}_mean'] = x.mean()
            row[f'{name}_p'] = stats.wilcoxon(x).pvalue if np.any(x != 0) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    d = pd.read_csv(DATA / 'decode_sim_2026-11.tsv', sep='\t')
    dec = pooled(rel_quadrants(d, 'drive_bar'))
    tru = pooled(rel_quadrants(d, 'true_drive_bar'))
    key = ['decoder', 'subject', 'roi']
    cols = ['hp', 'left', 'right', 'opposite', 'adjacent']

    def paired(x):
        null = x[x.gain_amplitude == 0].set_index(key)[cols]
        out = []
        for a in sorted(x.gain_amplitude.unique()):
            if a == 0:
                continue
            g = x[x.gain_amplitude == a].set_index(key)[cols] - null
            out.append(g.assign(gain_amplitude=a).reset_index())
        return pd.concat(out)

    eff_dec, eff_tru = paired(dec), paired(tru)
    # Decoded minus true effect per subject: what the decoder adds.
    m = eff_dec.merge(eff_tru, on=key + ['gain_amplitude'], suffixes=('', '_true'))
    diff = m[key + ['gain_amplitude']].copy()
    for c in cols:
        diff[c] = m[c] - m[f'{c}_true']
    null_dec = dec[dec.gain_amplitude == 0]

    tab = pd.concat([summarise(eff_dec, 'decoded_effect'), summarise(eff_tru, 'true_effect'),
                     summarise(diff, 'decoded_minus_true'), summarise(null_dec, 'null_decoded')])
    out = DATA / f'decode_sim_summary_2026-11{SUFFIX}.tsv'
    tab.to_csv(out, sep='\t', index=False, float_format='%.4g')
    pd.set_option('display.width', 250)
    show = ['kind', 'gain_amplitude', 'decoder', 'roi', 'n', 'hp_mean', 'hp_sem', 'adjacent_mean', 'adjacent_sem',
            'opposite_mean', 'opposite_sem', 'hp_minus_next_mean', 'hp_minus_next_p', 'next_minus_opp_mean',
            'next_minus_opp_p', 'hp_minus_opp_p']
    print(tab[show].round(3).to_string(index=False))
    print(f'-> {out}')

    # Real data (Fig. 6), same normalisation, free and outline decoders.
    real = {}
    for src in ('canonical', 'jointhrf'):
        f = DATA / f'decoding_readout_{src}_2026-11.tsv'
        if f.exists():
            r = pooled(rel_quadrants(pd.read_csv(f, sep='\t'), 'drive_bar'))
            real[src] = r
            print(f'\nReal {src}:')
            print(r.groupby(['decoder', 'roi'])[['hp', 'adjacent', 'opposite']].agg(['mean', 'sem']).round(2).to_string())
    figure(eff_dec, eff_tru, real.get('canonical'))


def figure(eff_dec, eff_tru, real):
    mpl.rcParams.update({
        'font.family': 'Helvetica', 'font.sans-serif': ['Helvetica', 'Helvetica Neue', 'Arial'], 'font.size': 7,
        'axes.labelsize': 8, 'axes.titlesize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
        'axes.linewidth': .8, 'axes.spines.top': False, 'axes.spines.right': False, 'xtick.direction': 'out',
        'ytick.direction': 'out', 'xtick.major.size': 3, 'ytick.major.size': 3, 'xtick.major.width': .8,
        'ytick.major.width': .8, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'})
    order = ['hp', 'adjacent', 'opposite']
    dec_name = 'free'
    panels = [('Real data', real, None)]
    for a in sorted(eff_dec.gain_amplitude.unique(), reverse=True):
        panels.append((f'Simulated field at HP, A = {a:g}', eff_dec, a))
    fig, axes = plt.subplots(1, len(panels), figsize=(7.25, 2.3), constrained_layout=True)
    for ax in axes[2:]:
        ax.sharey(axes[1])
    rm = real[(real.decoder == dec_name) & (real.roi == 'V1–V3')][order].mean()
    rng = np.random.default_rng(0)
    for i, (ax, (title, src, a)) in enumerate(zip(axes, panels)):
        if src is None:
            ax.set_visible(False)
            continue
        s = src[(src.decoder == dec_name) & (src.roi == 'V1–V3')]
        if a is not None:
            s = s[s.gain_amplitude == a]
            t = eff_tru[(eff_tru.decoder == dec_name) & (eff_tru.roi == 'V1–V3') & (eff_tru.gain_amplitude == a)]
        for j, c in enumerate(order):
            v = s[c].to_numpy()
            ax.scatter(j + rng.uniform(-.12, .12, len(v)), v, s=5, color=COLS[c], alpha=.35, lw=0)
            ax.errorbar(j, v.mean(), stats.sem(v), fmt='o', ms=5, color=COLS[c], mec='.1', mew=1, elinewidth=1.2,
                        capsize=0, zorder=5)
            if a is not None:
                tv = t[c].mean()
                ax.plot([j - .3, j + .3], [tv, tv], color='.1', lw=1.2, ls=(0, (2, 1.5)), zorder=6)
                if i == 1:
                    ax.plot(j + .38, rm[c], 'D', ms=4.5, mfc='white', mec=COLS[c], mew=1.2, zorder=6)
        ax.axhline(0, color='.7', lw=.6, ls='--', zorder=0)
        ax.set_xticks(range(3), [LABELS[c] for c in order], rotation=30, ha='right')
        ax.set_title(title, fontsize=7)
        ax.text(-.18, 1.04, 'abcd'[i], transform=ax.transAxes, fontsize=8, fontweight='bold', ha='right', va='bottom')
        if i == 1:
            ax.plot(.06, .1, 'D', transform=ax.transAxes, ms=4.5, mfc='white', mec='.3', mew=1.2)
            ax.text(.11, .1, 'Real data (a)', transform=ax.transAxes, va='center', fontsize=6.5)
        if a is not None and i == len(panels) - 1:
            ax.plot([.55, .7], [.12, .12], transform=ax.transAxes, color='.1', lw=1.2, ls=(0, (2, 1.5)))
            ax.text(.73, .12, 'True', transform=ax.transAxes, va='center', fontsize=6.5)
            ax.plot(.625, .05, 'o', transform=ax.transAxes, ms=4, color='.4', mec='.1', mew=1)
            ax.text(.73, .05, 'Decoded', transform=ax.transAxes, va='center', fontsize=6.5)
    axes[0].set_ylabel('Decoded intensity vs\nmean of quadrants (%)')
    axes[1].set_ylabel('Change from null (%)')
    sns.despine(fig=fig, offset=4)
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / f'figS_decode_simulation{SUFFIX}.pdf')
    fig.savefig(FIGS / f'figS_decode_simulation{SUFFIX}.png', dpi=200)
    print(f'-> {FIGS / f"figS_decode_simulation{SUFFIX}.pdf"}')


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--suffix', default='', help='appended to the summary table and figure names')
    SUFFIX = ap.parse_args().suffix
    main()
