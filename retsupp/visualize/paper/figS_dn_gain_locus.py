"""Supplementary figure: where in the DN pRF model does the sustained HP effect act?

a, The DN response with the three places a sustained field can act: the stimulus drive (input gain, m7_full), the
neural baseline b (m8_full_dnb) and the surround baseline d (m8_full_dnd); both b and d: m8_full_dnbd.
b, Real data, per ROI: held-out R² of each variant minus m7_full (run3 folds, pRFs frozen at the all-runs fit,
shared terms refitted per fold; ``fit_cv_ladder_nested --frozen-prf``), and the HP contrast of the input gain
dS = g_HP − g_LP in m7_full and in each variant (all-runs fits).
c, Recovery: BOLD simulated from an input, a b or a d effect of equal size (RMS BOLD change of the fitted dS, and 4×)
plus phase-randomised real residuals, refitted with all models (``recover_dn_baseline_field``). Which model is
selected (likelihood-ratio rule calibrated on the input truth) and the input-gain dS each truth produces.
d, Predicted pattern (HP quadrant, next to HP, opposite; ``sim_dn_field_signatures``) per locus, bar on the pRF
centre and bar in its surround, beside the measured patterns (decoded bar, DN pRFs; item responses, early areas).
e, Model-free counterpart: per-HP-condition DN fits, b/d and the two baselines in HP-quadrant voxels minus
opposite-quadrant voxels (``figS_dn_params_by_hp``).

Usage::

    python -m retsupp.visualize.paper.figS_dn_gain_locus
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, FancyBboxPatch

from retsupp.visualize.paper.paper_style import (DOUBLE, FG, HP, MUTED, ROI_ORDER, ROI_X, letter, pending,
                                                 roi_axis, save)

D = Path('notes/data')
INPUT, BFIELD, DFIELD, BOTH = '#3D5A80', '#2A9D8F', '#7209B7', '#52606D'
MODELS = [('m8_full_dnb', 'b-field', BFIELD), ('m8_full_dnd', 'd-field', DFIELD), ('m8_full_dnbd', 'b and d', BOTH)]
LOC = [('HP', HP), ('next', '#52606D'), ('opposite', '#9AA5B1')]
TRUTHS = [('input', 'Input gain', INPUT), ('b', 'b-field', BFIELD), ('d', 'd-field', DFIELD)]


# ---------------------------------------------------------------- a: schematic
def field_icon(ax, kind, color):
    """Display outline with the HP field and, for b and d, one voxel's pRF centre and surround."""
    ax.set_aspect('equal')
    ax.set_axis_off()
    ax.set_xlim(-5.2, 5.2)
    ax.set_ylim(-5.2, 5.2)
    ring = 4. / np.sqrt(2) * np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]])
    ax.add_patch(Circle((0, 0), 3.17, fill=False, ec=MUTED, lw=.5))
    for p in ring:
        ax.add_patch(Circle(p, .5, facecolor='w', ec=MUTED, lw=.5))
    hp = ring[0]
    if kind == 'input':
        for r, a in ((2.2, .12), (1.4, .22), (.7, .35)):
            ax.add_patch(Circle(hp, r, facecolor=color, alpha=a, lw=0))
        ax.add_patch(Circle(ring[1] * .55, .7, fill=False, ec=FG, lw=.7))
        ax.add_patch(Circle(ring[1] * .55, 3.0, fill=False, ec=FG, lw=.5, ls=(0, (2, 1.5))))
    else:
        for r, a in ((2.2, .12), (1.4, .22), (.7, .35)):
            ax.add_patch(Circle(hp, r, facecolor=HP, alpha=a * .6, lw=0))
        c = hp * .62
        ax.add_patch(Circle(c, .75, facecolor=color if kind == 'b' else 'none', alpha=.55 if kind == 'b' else 1,
                            ec=FG, lw=.7))
        ax.add_patch(Circle(c, 2.1, facecolor='none', ec=color if kind == 'd' else FG,
                            lw=1.6 if kind == 'd' else .5, ls='-' if kind == 'd' else (0, (2, 1.5))))


def panel_schematic(fig, ax):
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    y_num, y_den, y_bar = .84, .58, .71
    kw = dict(fontsize=8.5, va='center', ha='left', color=FG)

    def box(x0, y, w, color):
        ax.add_patch(FancyBboxPatch((x0, y - .085), w, .17, boxstyle='round,pad=0.005,rounding_size=0.02',
                                    facecolor=color, alpha=.18, edgecolor=color, lw=.8))
    ax.text(.02, y_bar, r'$N(t)=$', **kw)
    # numerator: a<S·M, G_c> + b
    x = .135
    ax.text(x, y_num, r'$a\,\langle$', **kw)
    box(x + .045, y_num, .085, INPUT)
    ax.text(x + .048, y_num, r'$S\cdot M_c$', **kw)
    ax.text(x + .135, y_num, r'$,\,G_c\rangle\;+$', **kw)
    box(x + .245, y_num, .045, BFIELD)
    ax.text(x + .252, y_num, r'$b_c$', **kw)
    # denominator: c|a|<S·M, G_s> + d
    ax.text(x - .01, y_den, r'$c|a|\,\langle$', **kw)
    box(x + .07, y_den, .085, INPUT)
    ax.text(x + .073, y_den, r'$S\cdot M_c$', **kw)
    ax.text(x + .16, y_den, r'$,\,G_s\rangle\;+$', **kw)
    box(x + .27, y_den, .045, DFIELD)
    ax.text(x + .277, y_den, r'$d_c$', **kw)
    ax.plot([x - .01, x + .33], [y_bar] * 2, color=FG, lw=.8)
    ax.text(x + .34, y_bar, r'$-\;b_c/d_c$', **kw)
    eqs = [(INPUT, r'Input gain (m7): $M_c(g)=1+\sum_\ell w_{c\ell}A_\ell(g)$, $w=g_{HP}$ at HP, $g_{LP}$ elsewhere'),
           (BFIELD, r'b-field: $b_c=b\,\exp(w_b F_c)$'),
           (DFIELD, r'd-field: $d_c=d\,\exp(w_d F_c)$,   $F_c$ = HP/LP field ($\sigma_{AF}$, $\theta$) over the pRF')]
    for i, (col, s) in enumerate(eqs):
        ax.text(.02, .34 - i * .14, s, fontsize=6.3, color=col, va='center', ha='left')
    for i, (kind, col, lab) in enumerate((('input', INPUT, 'Input: field on the image,\nreaches the surround'),
                                          ('b', BFIELD, 'b: baseline of\nthe pRF centre'),
                                          ('d', DFIELD, 'd: normalization\nconstant'))):
        ins = ax.inset_axes([.57 + i * .145, .3, .13, .62])
        field_icon(ins, kind, col)
        ax.text(.57 + i * .145 + .065, .2, lab, fontsize=5.8, color=col, ha='center', va='top')


# ---------------------------------------------------------------- b: real data
def mean_sem(d, col):
    g = d.groupby(['roi', 'subject'])[col].mean().groupby('roi')
    return g.mean(), g.sem()


def panel_cv(ax, cv):
    for j, (m, lab, col) in enumerate(MODELS):
        d = cv[cv.model == m]
        mu, se = mean_sem(d, 'dcv_r2')
        x = np.array([ROI_X[r] for r in ROI_ORDER]) + (j - 1) * .22
        ax.errorbar(x, 1e3 * mu.reindex(ROI_ORDER), 1e3 * se.reindex(ROI_ORDER), fmt='o', ms=2.6, color=col,
                    lw=.8, capsize=0, label=lab)
    ax.axhline(0, color=FG, lw=.6)
    ax.set_ylabel('Held-out R² − input\ngain model (×10⁻³)', fontsize=6.5)
    ax.set_xticks([])


def panel_ds(ax, full):
    sets = [('m7_full', 'Input gain', INPUT)] + MODELS
    for j, (m, lab, col) in enumerate(sets):
        d = full[full.model == m]
        mu, se = mean_sem(d, 'dS')
        x = np.array([ROI_X[r] for r in ROI_ORDER]) + (j - 1.5) * .18
        ax.errorbar(x, mu.reindex(ROI_ORDER), se.reindex(ROI_ORDER), fmt='o', ms=2.4, color=col, lw=.8,
                    capsize=0, label=lab)
    ax.axhline(0, color=FG, lw=.6)
    ax.set_ylabel(r'dS = $g_{HP}-g_{LP}$', fontsize=6.5)
    roi_axis(ax, fontsize=6)


# ---------------------------------------------------------------- c: recovery
def selection(rec, q=.95):
    """Per truth × size: share of simulations in which each model is selected.

    Statistic: training-SSE gain of the b- and d-field over m7, relative to m7's SSE. Threshold: the q-quantile of
    the larger of the two gains under the input truth (pooled over units and sizes), so 'Input gain' is selected
    in about q of input-truth simulations by construction; the other rows are the test."""
    w = rec.pivot_table(index=['subject', 'roi', 'truth', 'size', 'noise'], columns='model', values='sse')
    g = pd.DataFrame({'b': (w.m7_full - w.m8_full_dnb) / w.m7_full,
                      'd': (w.m7_full - w.m8_full_dnd) / w.m7_full}, index=w.index)
    thr = g[g.index.get_level_values('truth') == 'input'].max(1).quantile(q)
    pick = np.where(g.max(1) <= thr, 'input', g.idxmax(1))
    g['pick'] = pick
    return g.reset_index(), thr


def panel_confusion(ax, rec, size):
    sel, _ = selection(rec)
    s = sel[sel['size'] == size]
    lab = [t for t, _, _ in TRUTHS]
    m = np.array([[np.mean(s[s.truth == t].pick == p) if (s.truth == t).any() else np.nan for p in lab]
                  for t in lab])
    ax.imshow(m, cmap='Greys', vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            if np.isfinite(m[i, j]):
                ax.text(j, i, f'{100 * m[i, j]:.0f}', ha='center', va='center', fontsize=6,
                        color='w' if m[i, j] > .55 else FG)
    short = ['Input', 'b', 'd']
    ax.set_xticks(range(3))
    ax.set_xticklabels(short, fontsize=6)
    ax.set_yticks(range(3))
    ax.set_yticklabels(short, fontsize=6)
    ax.set_xlabel('Selected', fontsize=6.5, labelpad=1)
    ax.set_title(f'{size:g}× fitted dS', fontsize=6.5, pad=2)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)


def panel_spurious(ax, rec):
    """dS fitted by m7 under each truth (size 1), relative to the input truth's true dS."""
    d = rec[(rec.model == 'm7_full') & ((rec['size'] == 1) | (rec.truth == 'none'))].copy()
    d['dS'] = d.g_HP - d.g_LP
    true_in = rec[(rec.truth == 'input') & (rec['size'] == 1) & (rec.model == 'm7_full')]
    ref = (true_in.true_g_HP - true_in.true_g_LP).groupby([true_in.subject, true_in.roi]).first()
    d = d.join(ref.rename('ref'), on=['subject', 'roi'])
    d['rel'] = d.dS / d.ref
    order = [('none', 'None', MUTED)] + TRUTHS
    for i, (t, lab, col) in enumerate(order):
        v = d[d.truth == t].groupby(['subject', 'roi']).rel.mean()
        ax.scatter(np.full(len(v), i) + np.random.default_rng(i).uniform(-.15, .15, len(v)), v, s=4, color=col,
                   lw=0, alpha=.7)
        ax.plot([i - .25, i + .25], [v.median()] * 2, color=col, lw=1.4)
    ax.axhline(0, color=FG, lw=.6)
    ax.axhline(1, color=INPUT, lw=.5, ls=':')
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(['None', 'Input', 'b', 'd'], fontsize=6)
    ax.set_ylabel('dS fitted by the input model\n(÷ true dS of input truth)', fontsize=6)
    ax.set_xlabel('Truth (1× size)', fontsize=6.5, labelpad=1)
    ax.set_ylim(-2.5, 4.5)


# ---------------------------------------------------------------- d: signatures
SIG = [('c_input_gain', 'Input (DN)', INPUT), ('c_input_gain_dog', 'Input (DoG)', MUTED),
       ('a_neural_baseline', 'b-field', BFIELD), ('b_surround_baseline', 'd-field', DFIELD)]
LOC_LAB = ['HP', 'Next', 'Opp.']


def shape(m):
    """Each location type minus the mean of the four quadrants (next counts twice), scaled so HP = −1."""
    m = m.sub((m.HP + 2 * m.next + m.opposite) / 4, axis=0)
    return m.div(-m.HP, axis=0)


def predicted(sim, cls):
    d = sim[sim.bar_class == cls]
    return shape(d.groupby(['variant', 'quadrant']).rel_change.median().unstack()[['HP', 'next', 'opposite']])


def panel_pred(ax, sim, cls, title, ylabel=True):
    m = predicted(sim, cls)
    w = .8 / len(SIG)
    for k, (v, lab, col) in enumerate(SIG):
        ax.bar(np.arange(3) - .4 + w * (k + .5), m.loc[v].values, w * .9, color=col, lw=0, label=lab)
    ax.axhline(0, color=FG, lw=.6)
    ax.set_xticks(range(3))
    ax.set_xticklabels(LOC_LAB, fontsize=6)
    ax.set_title(title, fontsize=6.5, pad=2)
    if ylabel:
        ax.set_ylabel('Change (HP = −1)', fontsize=6.5)


def observed(dec):
    from retsupp.visualize.paper.fig3_item_locations import REL_FOLDER, rel_stats
    _, cells, _ = rel_stats(REL_FOLDER)
    e = cells[cells.grp == 'Early'].pivot_table(index='subject', columns='loc', values='r')
    e = pd.DataFrame({'HP': e.hp, 'next': (e.nextleft + e.nextright) / 2, 'opposite': e.opposite})
    e = e.sub((e.HP + 2 * e.next + e.opposite) / 4, axis=0)
    items = pd.DataFrame({'m': e.mean(), 'se': e.sem()})
    q = dec[(dec.source == 'dn_roihrf') & (dec.decoder == 'free')].iloc[0]
    deco = pd.DataFrame({'m': [q.hp, q.adjacent, q.opposite], 'se': [q.hp_sem, q.adjacent_sem, q.opposite_sem]},
                        index=['HP', 'next', 'opposite'])
    return {'Items, early areas': items / abs(items.m.HP), 'Decoded bar (DN pRFs)': deco / abs(deco.m.HP)}


def panel_obs(ax, obs):
    for k, ((lab, t), col) in enumerate(zip(obs.items(), (FG, '#E07A5F'))):
        ax.errorbar(np.arange(3) + (k - .5) * .22, t.m.values, yerr=t.se.values, fmt='o', color=col, ms=3, lw=.9,
                    capsize=0, label=lab)
    ax.axhline(0, color=FG, lw=.6)
    ax.set_xticks(range(3))
    ax.set_xticklabels(LOC_LAB, fontsize=6)
    ax.set_title('Measured', fontsize=6.5, pad=2)


# ---------------------------------------------------------------- e: per-condition fits
def panel_by_hp(ax, st):
    sets = [('b_over_d', 'b/d', FG), ('neural_baseline', 'b', BFIELD), ('surround_baseline', 'd', DFIELD)]
    for j, (p, lab, col) in enumerate(sets):
        d = st[st.param == p].set_index('roi').reindex(ROI_ORDER)
        x = np.array([ROI_X[r] for r in ROI_ORDER]) + (j - 1) * .22
        ax.plot(x, d.mean_hp_minus_opp, 'o', ms=2.6, color=col, label=lab)
        sig = d.p_hp_vs_opp_bh < .05
        ax.plot(x[sig.values], d.mean_hp_minus_opp[sig], 'o', ms=5, mfc='none', mec=col, mew=.7)
    ax.axhline(0, color=FG, lw=.6)
    ax.set_ylabel('HP − opposite quadrant\n(log ratio)')
    ax.legend(frameon=False, fontsize=6, ncol=3, loc='upper left', bbox_to_anchor=(0, 1.12), handletextpad=.1,
              columnspacing=.6)
    roi_axis(ax, fontsize=6)


def load(name):
    p = D / name
    return pd.read_csv(p, sep='\t') if p.exists() else None


LEGEND = """
**Figure S?. Where in the divisive-normalization (DN) pRF model does the sustained HP effect act?** a, DN response
of a voxel (Aqil et al., 2021) with the three places a sustained field can act, each one field per subject × ROI
(shared parameters, fitted on all voxels of the ROI; each voxel sees the field over its own pRF): the stimulus drive,
in numerator and denominator (input gain, the model of Fig. 5, blue); the neural baseline b (teal); the surround
baseline d (purple). The b- and d-fields share one shape (width σ_AF, HP/LP pattern θ) with separate weights w_b, w_d;
at w = 0 each equals the input-gain model exactly. Scaling b and d together equals scaling the amplitude a, so only
w_b − w_d (b/d) is specific to DN. b, Top: held-out R² of each variant minus the input-gain model, mean ± SEM over
subjects (run3 folds; per-voxel pRFs frozen at the all-runs fit in every model, shared terms refitted per fold).
Bottom: dS = g_HP − g_LP of the input gain in each model (all-runs fits). c, Recovery on {rec_units}: BOLD simulated
from an input, b or d effect of equal BOLD size plus phase-randomised real residuals, refitted with every model.
Confusion: % of simulations in which each model is selected (training-SSE gain of the b- or d-field over the input
model, threshold = 95th percentile under the input truth, so the input row is ~95 % by construction). Right: dS
fitted by the input-gain model under each truth, relative to the input truth's dS. d, Predicted change of the
neural response in voxels of the HP quadrant, the two next to it and the opposite one (median over synthetic voxels
on the diagonals at 1–3°, DN parameters in the range of the fits; ``sim_dn_field_signatures``), for an HP field on
each locus and, for comparison, the DoG with the same input field; bar over the pRF centre or in its surround. Each
location type minus the mean of the four quadrants, scaled and signed so that HP = −1 (only the shape is compared).
Right: measured, same scaling: item responses in early areas (Fig. 3e) and the decoded bar (DN pRFs, per-ROI HRF;
Fig. 6 with DN), mean ± SEM, n = 30. e,
Model-free: per-HP-condition DN fits (n = 30, no CV), HP-quadrant minus opposite-quadrant voxels; rings,
p < .05 BH over ROIs.
"""


def main(name='figS_dn_gain_locus', cv_table='dn_basefield_cv_2026-11.tsv'):
    full, cv = load('dn_basefield_full_2026-11.tsv'), load(cv_table)
    rec, sim = load('dn_basefield_recovery_2026-11.tsv'), load('dn_field_signatures_2026-11.tsv')
    dec = load('decoding_prf_models_quadrants_2026-11.tsv')
    by_hp = load('dn_by_hp_stats_2026-11.tsv')

    fig = plt.figure(figsize=(DOUBLE, 6.0))
    top = fig.add_gridspec(2, 1, height_ratios=[.78, 2.8], hspace=.28, left=.075, right=.985, top=.975,
                           bottom=.075)
    gs = [top[0]] + list(top[1].subgridspec(2, 1, height_ratios=[1.6, 1.2], hspace=.78))
    ax_a = fig.add_subplot(gs[0])
    panel_schematic(fig, ax_a)
    letter(ax_a, 'a', x=0, y=.95)

    g2 = gs[1].subgridspec(1, 2, width_ratios=[1.45, 1], wspace=.28)
    g2b = g2[0].subgridspec(2, 1, height_ratios=[1, 1], hspace=.18)
    ax_b1, ax_b2 = fig.add_subplot(g2b[0]), fig.add_subplot(g2b[1])
    if cv is not None:
        panel_cv(ax_b1, cv)
    else:
        pending(ax_b1, 'held-out comparison (run3, frozen pRFs)')
    if full is not None:
        panel_ds(ax_b2, full)
        h, l = ax_b2.get_legend_handles_labels()
        ax_b1.legend(h, l, frameon=False, fontsize=6, ncol=4, loc='lower left', bbox_to_anchor=(0, 1.0),
                     handletextpad=.2, columnspacing=.9)
    else:
        pending(ax_b2, 'dS per model')
    letter(ax_b1, 'b', x=-.12, y=1.08)
    g2c = g2[1].subgridspec(2, 2, height_ratios=[1, 1.25], wspace=.25, hspace=.75)
    axc = [fig.add_subplot(g2c[0, 0]), fig.add_subplot(g2c[0, 1])]
    ax_c3 = fig.add_subplot(g2c[1, :])
    if rec is not None:
        for ax, k in zip(axc, (1.0, 4.0)):
            panel_confusion(ax, rec, k)
        axc[0].set_ylabel('Truth', fontsize=6.5, labelpad=1)
        axc[1].set_yticklabels([])
        panel_spurious(ax_c3, rec)
    else:
        for ax in axc + [ax_c3]:
            pending(ax, 'recovery')
    letter(axc[0], 'c', x=-.45, y=1.12)

    g3 = gs[2].subgridspec(1, 2, width_ratios=[1.55, 1], wspace=.3)
    g3d = g3[0].subgridspec(1, 3, wspace=.35)
    axd = [fig.add_subplot(g3d[i]) for i in range(3)]
    panel_pred(axd[0], sim, 'centre', 'Predicted, bar on pRF centre')
    panel_pred(axd[1], sim, 'surround', 'Predicted, bar in surround', ylabel=False)
    axd[1].sharey(axd[0])
    try:
        panel_obs(axd[2], observed(dec))
    except Exception as e:   # deconvolution folder not available
        pending(axd[2], f'measured ({type(e).__name__})')
    h, l = axd[0].get_legend_handles_labels()
    axd[0].legend(h, l, frameon=False, fontsize=5.8, ncol=4, loc='upper left', bbox_to_anchor=(0, 1.32),
                  handlelength=.9, handletextpad=.3, columnspacing=.7)
    axd[2].legend(frameon=False, fontsize=5.8, loc='lower left', bbox_to_anchor=(0, 1.1), handletextpad=.2)
    letter(axd[0], 'd', x=-.3, y=1.2)
    ax_e = fig.add_subplot(g3[1])
    panel_by_hp(ax_e, by_hp)
    letter(ax_e, 'e', x=-.2, y=1.05)

    rec_units = ('pending' if rec is None else
                 f"{rec.groupby(['subject', 'roi']).ngroups} subject × ROI units "
                 f"(subjects {', '.join(str(s) for s in sorted(rec.subject.unique()))}; "
                 f"{', '.join(sorted(rec.roi.unique()))}), {rec.noise.nunique()} noise draws")
    save(fig, name, LEGEND.format(rec_units=rec_units))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--name', default='figS_dn_gain_locus')
    ap.add_argument('--cv-table', default='dn_basefield_cv_2026-11.tsv', help='held-out table in notes/data')
    a = ap.parse_args()
    main(a.name, a.cv_table)
