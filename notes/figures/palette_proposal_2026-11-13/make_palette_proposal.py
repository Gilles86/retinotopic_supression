"""Palette proposal 2026-11-13 (option A): swatch sheet with CVD simulations, and mocks of key figure elements.

Run: ~/mambaforge/bin/python notes/figures/palette_proposal_2026-11-13/make_palette_proposal.py
Writes swatches.{png,pdf}, mocks.{png,pdf}, mocks_cvd.png and cvd_pairs.tsv next to this script.
Nothing here touches the paper figures.
"""
import itertools
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.patches import Circle, Rectangle

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from cvd_check import VIEWS, de, sim, sim_rgb  # noqa: E402

mpl.rcParams.update({'font.family': 'Helvetica', 'font.size': 7, 'pdf.fonttype': 42, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.linewidth': .8, 'xtick.direction': 'out',
                     'ytick.direction': 'out', 'xtick.major.size': 3, 'ytick.major.size': 3})

# ---------------------------------------------------------------- proposed palette (option A)
FG, MUTED = '#1F2933', '#9AA5B1'
HP, NEXT, OPP = '#7B3294', '#9970AB', '#D9C5E2'     # sustained field / HP, and its two LP tints (LP = NEXT)
HP_DEEP = '#40004B'                                 # edges of tint markers, darkest end of field maps
TARGET, DISTRACTOR = '#F77F00', '#4AA3DF'
NEUTRAL = '#9AA5B1'                                 # neutral item, distractor absent, shuffled null
STIM = '#3E4C59'                                    # bar stimulus (achromatic, like the checkerboard)
PRF = '#00796B'                                     # pRF outline (darker teal than #2A9D8F)
DIV_MORE = '#4D5A66'                                # "more" end of the decoding map (purple = less)
LP_GREY_NEXT, LP_GREY_OPP = '#52606D', '#9AA5B1'    # alternative: LP in greys
AREA = {'V1': '#2F4B5C', 'V2': '#5A7A8C', 'V3': '#93AEBB',                    # Early: steel greys
        'V3AB': '#7A6038', 'IPS': '#B8995E',                                  # Dorsal: umber, khaki
        'hV4': '#4F6B36', 'VO': '#76924F', 'LO': '#A3B97A', 'TO': '#CCD9AE'}  # Occipitotemporal: olives
PATH_LS = {'Early': '-', 'Dorsal': (0, (4, 1.5)), 'Occipitotemporal': (0, (1.2, 1.2))}

ROLES = [  # (group, label, hex, old hex or '')
    ('Location / sustained field', 'HP location; sustained field', HP, '#D62828'),
    ('Location / sustained field', 'LP, next to HP (generic LP)', NEXT, '#52606D'),
    ('Location / sustained field', 'LP, opposite HP', OPP, '#9AA5B1'),
    ('Location / sustained field', 'Edge / text on tints', HP_DEEP, ''),
    ('Transient fields, items', 'Target', TARGET, '#F77F00'),
    ('Transient fields, items', 'Distractor', DISTRACTOR, '#0077B6'),
    ('Transient fields, items', 'Neutral item; absent; null', NEUTRAL, '#7B8794'),
    ('Stimulus, pRF', 'Bar stimulus', STIM, '#3D5A80'),
    ('Stimulus, pRF', 'pRF outline', PRF, '#2A9D8F'),
    ('Stimulus, pRF', 'Text, axes, generic model line', FG, '#1F2933'),
    ('Real display (Fig. 1a, 4a only)', 'Screen', '#808080', '#808080'),
    ('Real display (Fig. 1a, 4a only)', 'Items (as shown)', '#E39100', '#E39100'),
    ('Real display (Fig. 1a, 4a only)', 'Distractor (as shown)', '#00A900', '#00A900'),
]
CRIT = [('HP', 'Target'), ('HP', 'Distractor'), ('Target', 'Distractor'), ('LP next', 'Distractor'),
        ('LP opposite', 'Distractor'), ('HP', 'LP next'), ('LP next', 'LP opposite'), ('LP opposite', 'Neutral'),
        ('LP next', 'Neutral'), ('pRF', 'Distractor'), ('pRF', 'Neutral'), ('pRF', 'LP next'), ('Stim', 'Distractor'),
        ('HP', 'Stim'), ('pRF', 'Target')]
NAMED = {'HP': HP, 'LP next': NEXT, 'LP opposite': OPP, 'Target': TARGET, 'Distractor': DISTRACTOR, 'Neutral': NEUTRAL,
         'pRF': PRF, 'Stim': STIM}
OLD = {'HP': '#D62828', 'LP next': '#52606D', 'LP opposite': '#9AA5B1', 'Target': '#F77F00', 'Distractor': '#0077B6',
       'Neutral': '#7B8794', 'pRF': '#2A9D8F', 'Stim': '#3D5A80'}


def div_cmap():
    return LinearSegmentedColormap.from_list('pu_gy', [HP_DEEP, HP, '#C9A9D6', '#FFFFFF', '#C4CBD2', DIV_MORE, '#2B333B'])


def field_cmap(col, deep=None):
    return LinearSegmentedColormap.from_list('f', ['#FFFFFF', col] + ([deep] if deep else []))


# ---------------------------------------------------------------- swatch sheet
def swatches():
    groups = list(dict.fromkeys(g for g, *_ in ROLES))
    n = len(ROLES) + len(groups)
    fig = plt.figure(figsize=(7.1, 8.4))
    top = fig.add_axes([0, .27, 1, .72])
    top.set_axis_off()
    top.set_xlim(0, 10)
    top.set_ylim(n + 1.4, 0)
    xs = [3.35, 5.0, 6.65, 8.3]
    for x, v in zip(xs, VIEWS):
        top.text(x + .6, .3, v, ha='center', fontsize=7.5, color=FG)
    top.text(.1, .3, 'Option A, proposed (old colour at left of each row)', fontsize=7.5, color=FG, fontweight='bold')
    y = 1.2
    for g in groups:
        top.text(.1, y, g, fontsize=6.5, color='.45', va='center')
        y += 1
        for gg, lab, h, old in ROLES:
            if gg != g:
                continue
            if old and old.lower() != h.lower():
                top.add_patch(Rectangle((.1, y - .32), .28, .64, color=old, lw=0))
                top.annotate('', xy=(.62, y), xytext=(.42, y), arrowprops=dict(arrowstyle='-|>', color='.5', lw=.6,
                                                                                mutation_scale=5))
            top.text(.72, y, lab, fontsize=6.5, va='center', color=FG)
            top.text(2.55, y, h.upper(), fontsize=6, va='center', color='.4', family='Menlo')
            for x, v in zip(xs, VIEWS):
                top.add_patch(Rectangle((x, y - .36), 1.2, .72, color=sim(h, v), lw=0))
            y += 1
    # gradients: field maps and decoding map
    yb = y + .2
    for x, v in zip(xs, VIEWS):
        for k, cm in enumerate([field_cmap(HP, HP_DEEP), field_cmap(TARGET), field_cmap(DISTRACTOR), div_cmap()]):
            g = np.linspace(0, 1, 64)
            rgb = sim_rgb(cm(g)[:, :3], v)
            top.imshow(rgb[None], extent=(x, x + 1.2, yb + k * .5 + .35, yb + k * .5), aspect='auto')
    for k, lab in enumerate(['Sustained-field map', 'Target-field map', 'Distractor-field map',
                             'Decoding map: less (purple) to more (grey)']):
        top.text(.72, yb + k * .5 + .17, lab, fontsize=6.5, va='center', color=FG)
    top.set_xlim(0, 10)
    top.set_ylim(yb + 2.2, 0)

    # CIEDE2000 table of critical pairs, old vs new
    ax = fig.add_axes([.06, .02, .9, .23])
    ax.set_axis_off()
    ax.set_xlim(0, 10)
    rows = len(CRIT)
    ax.set_ylim(rows + 2.4, -.5)
    hdr = ['Pair', 'Old: N / D / P / T', 'New: N / D / P / T']
    for x, t in zip([0, 3.6, 6.8], hdr):
        ax.text(x, 0, t, fontsize=6.5, fontweight='bold', color=FG, va='center')
    out = ['pair\tview\tdE_old\tdE_new']
    for i, (a, b) in enumerate(CRIT):
        yy = i + 1
        ax.text(0, yy, f'{a} – {b}', fontsize=6.3, color=FG, va='center')
        for x, pal in ((3.6, OLD), (6.8, NAMED)):
            vals = [de(pal[a], pal[b], v) for v in VIEWS]
            for j, val in enumerate(vals):
                col = '#B00020' if val < 10 else ('#8A6D00' if val < 15 else FG)
                ax.text(x + j * .62, yy, f'{val:4.0f}', fontsize=6.3, color=col, va='center', family='Menlo')
        for v in VIEWS:
            out.append(f'{a} – {b}\t{v}\t{de(OLD[a], OLD[b], v):.1f}\t{de(NAMED[a], NAMED[b], v):.1f}')
    ax.text(0, rows + 1.7, 'CIEDE2000 ΔE in normal vision (N) and simulated deuteranopia (D), protanopia (P), tritanopia (T), '
            'Machado et al. 2009, severity 1.\nRed < 10 (confusable), amber 10–15 (needs a second cue: lightness, edge, shape or line style).',
            fontsize=5.8, color='.4', va='center')
    (HERE / 'cvd_pairs.tsv').write_text('\n'.join(out) + '\n')
    fig.savefig(HERE / 'swatches.pdf')
    fig.savefig(HERE / 'swatches.png', dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- mock helpers
RING = 4 / np.sqrt(2) * np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]])
HP_IX, T_IX = 3, 1
X, Y = np.meshgrid(np.linspace(-6, 6, 241), np.linspace(-6, 6, 241))


def gauss(p, s):
    return np.exp(-((X - p[0]) ** 2 + (Y - p[1]) ** 2) / (2 * s ** 2))


def disc_axes(ax, lim=6):
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')
    ax.set_axis_off()


def mini_display(ax, hp=True, items=(), cx=0, cy=0, s=1.):
    """Icon: aperture, four item circles, HP ring, optional items [(ix, 'target'|'distractor')]."""
    ax.add_patch(Circle((cx, cy), 3.17 * s, fill=False, ec=MUTED, lw=.5))
    for i, p in enumerate(RING):
        ax.add_patch(Circle((cx + p[0] * s, cy + p[1] * s), .7 * s, facecolor='w', ec=MUTED, lw=.5, zorder=2))
    if hp:
        p = RING[HP_IX]
        ax.add_patch(Circle((cx + p[0] * s, cy + p[1] * s), 1.25 * s, fill=False, ec=HP, lw=.9, ls=(0, (2, 1.2)), zorder=3))
    for ix, kind in items:
        p = RING[ix]
        if kind == 'target':
            ax.plot(cx + p[0] * s, cy + p[1] * s, 'D', ms=3.2, color=TARGET, mew=0, zorder=4)
        else:
            ax.plot(cx + p[0] * s, cy + p[1] * s, 'o', ms=3.4, color=DISTRACTOR, mew=0, zorder=4)
    ax.plot(cx, cy, '+', color=FG, ms=2.5, mew=.5)


def location_icon(ax, x, y, which, s=.12, ramp=True):
    """Icon at (x, y) in axes coords marking HP, next-to-HP or opposite location fill."""
    ins = ax.inset_axes([x - s / 2, y - s / 2, s, s])
    disc_axes(ins)
    ins.add_patch(Circle((0, 0), 3.17, fill=False, ec=MUTED, lw=.5))
    nxt = [0, 2]
    opp = 1
    for i, p in enumerate(RING):
        fc = 'w'
        if which == 'hp' and i == HP_IX:
            fc = HP
        elif which == 'next' and i in nxt:
            fc = NEXT if ramp else LP_GREY_NEXT
        elif which == 'opposite' and i == opp:
            fc = OPP if ramp else LP_GREY_OPP
        ins.add_patch(Circle(p, 1.1, facecolor=fc, ec=HP if i == HP_IX else MUTED, lw=.6))
    ins.plot(0, 0, '+', color=FG, ms=2, mew=.4)


# ---------------------------------------------------------------- mocks
def mock_fields(fig, gs_row):
    """Fig. 4a-like row: on screen (real colours) -> model input (grey) x (1 + sustained + target + distractor)."""
    axs = [fig.add_subplot(gs_row[0, j]) for j in range(6)]
    bar = ((np.abs(X + 1.0) < .45) & (np.hypot(X, Y) < 3.17)).astype(float)
    items = np.zeros_like(X)
    for p in RING:
        items += ((np.abs(X - p[0]) < .75) & (np.abs(Y - p[1]) < .2)).astype(float)
    stim = np.clip(bar + items, 0, 1)
    sus = -.5 * gauss(RING[HP_IX], 1.3)
    tgt = .8 * gauss(RING[T_IX], 1.6)
    dis = .5 * gauss(RING[HP_IX], 1.6)
    # a: on screen, real colours
    ax = axs[0]
    disc_axes(ax)
    ax.add_patch(Rectangle((-6, -6), 12, 12, color='#808080', lw=0))
    ax.add_patch(Circle((0, 0), 3.17, fill=False, ec='#6E6E6E', lw=.4))
    for i, p in enumerate(RING):
        col = '#00A900' if i == HP_IX else '#E39100'
        w, h = (.4, 1.5) if i == T_IX else (1.5, .4)
        ax.add_patch(Rectangle((p[0] - w / 2, p[1] - h / 2), w, h, color=col, lw=0))
    for k in range(10):
        ax.add_patch(Rectangle((-1.45, -3.1 + k * .62), .45, .31, color='k', lw=0))
        ax.add_patch(Rectangle((-1.0, -3.1 + k * .62 + .31), .45, .31, color='k', lw=0))
    ax.add_patch(Rectangle((-1.45, -3.1), .9, 6.2, fill=False, ec='w', lw=.4))
    ax.add_patch(Circle(RING[HP_IX], 1.3, fill=False, ec=HP, lw=.9, ls=(0, (2, 1.2))))
    ax.set_title('On screen (real colours)', fontsize=6.5, color=FG, pad=2)
    # b: model input, achromatic
    ax = axs[1]
    disc_axes(ax)
    ax.add_patch(Rectangle((-6, -6), 12, 12, color='k', lw=0))
    ax.add_patch(Rectangle((-1.45, -3.0), .9, 6.0, color='.7', lw=0))
    for p in RING:
        ax.add_patch(Rectangle((p[0] - .75, p[1] - .2), 1.5, .4, color='.7', lw=0))
    ax.add_patch(Circle((2.2, -1.6), 1.0, fill=False, ec=PRF, lw=1.1))
    ax.set_title('Model input', fontsize=6.5, color=FG, pad=2)
    ax.text(0, -6.6, 'pRF', color=PRF, fontsize=6.5, ha='center', va='top')
    for ax, m, cm, title, sub, col in ((axs[2], -sus, field_cmap(HP, HP_DEEP), 'Sustained', 'Whole block, less at HP', HP),
                                       (axs[3], tgt, field_cmap(TARGET), 'Target', 'During search', TARGET),
                                       (axs[4], dis, field_cmap(DISTRACTOR), 'Distractor', 'During search', DISTRACTOR)):
        disc_axes(ax)
        ax.imshow(m, extent=(-6, 6, -6, 6), origin='lower', cmap=cm, vmin=0, vmax=m.max() * 1.05)
        ax.add_patch(Circle((0, 0), 3.17, fill=False, ec='.6', lw=.5))
        if col == HP:
            ax.add_patch(Circle(RING[HP_IX], 1.3, fill=False, ec=HP_DEEP, lw=.8, ls=(0, (2, 1.2))))
        ax.set_title(title, fontsize=6.5, color=col, pad=2)
        ax.text(0, -6.6, sub, color=col, fontsize=6, ha='center', va='top')
    ax = axs[5]
    disc_axes(ax)
    ax.imshow(stim * (1 + sus + tgt + dis), extent=(-6, 6, -6, 6), origin='lower', cmap='gray', vmin=0, vmax=1.6)
    ax.add_patch(Circle(RING[T_IX], 1.3, fill=False, ec=TARGET, lw=.9))
    ax.add_patch(Circle(RING[HP_IX], 1.3, fill=False, ec=DISTRACTOR, lw=.9))
    ax.set_title('Effective drive', fontsize=6.5, color=FG, pad=2)
    axs[0].text(-.08, 1.12, 'a', transform=axs[0].transAxes, fontsize=8, fontweight='bold', va='bottom')
    axs[0].text(.0, 1.32, 'Fig. 4a-like: model terms; stimulus images achromatic (no magma, which shares the purple)',
                transform=axs[0].transAxes, fontsize=6.5, color='.4', va='bottom')


def mock_locations(fig, gs, ramp=True, letter_='b'):
    """Fig. 3d/e-like: time course HP / next / opposite, and per-pathway strip with icons."""
    rng = np.random.default_rng(1)
    cols = {'hp': HP, 'next': NEXT if ramp else LP_GREY_NEXT, 'opposite': OPP if ramp else LP_GREY_OPP}
    txt = {'hp': HP, 'next': NEXT if ramp else LP_GREY_NEXT, 'opposite': '#8C6B9E' if ramp else '#7B8794'}
    edge = {'hp': HP_DEEP, 'next': HP_DEEP, 'opposite': HP_DEEP} if ramp else {'hp': HP_DEEP, 'next': FG, 'opposite': '#52606D'}
    ls = {'hp': '-', 'next': '-', 'opposite': (0, (3, 1.3))}
    # time course
    ax = fig.add_subplot(gs[0])
    t = np.linspace(-4, 20, 200)
    h = np.exp(-(t - 5) ** 2 / 6) - .25 * np.exp(-(t - 11) ** 2 / 12)
    for c, a in (('opposite', .05), ('next', -.005), ('hp', -.045)):
        y = a * h
        ax.fill_between(t, y - .012, y + .012, color=cols[c], alpha=.25, lw=0)
        ax.plot(t, y, color=cols[c], lw=1.4, ls=ls[c])
    ax.axhline(0, color='.6', lw=.6, ls='--', zorder=0)
    ax.axvline(0, color='.75', lw=.6, zorder=0)
    ax.set_xticks([0, 10, 20])
    ax.set_yticks([-.05, 0, .05])
    ax.set_yticklabels(['−.05', '0', '.05'])
    ax.set_xlabel('Time from search onset (s)', fontsize=6.5)
    ax.set_ylabel('Δ vs mean (% signal)', fontsize=6.5)
    for k, (c, lab) in enumerate((('hp', 'HP'), ('next', 'Next to HP'), ('opposite', 'Opposite'))):
        ax.plot([.62, .72], [.95 - k * .1] * 2, transform=ax.transAxes, color=cols[c], lw=1.4, ls=ls[c])
        ax.text(.75, .95 - k * .1, lab, transform=ax.transAxes, color=txt[c], fontsize=6, va='center')
    ax.text(-.25, 1.12, letter_, transform=ax.transAxes, fontsize=8, fontweight='bold', va='bottom')
    ax.text(0, 1.12, 'Fig. 3d/e-like: ' + ('LP as purple tints (recommended)' if ramp else 'alternative: LP in greys'),
            transform=ax.transAxes, fontsize=6.5, color='.4', va='bottom')
    # strip per pathway
    ax = fig.add_subplot(gs[1])
    groups = ['Early', 'Dorsal', 'Occ.-temporal']
    means = {'hp': [-8, -3, -5], 'next': [-2, 0, 0], 'opposite': [12, 3, 4]}
    for gi, g in enumerate(groups):
        for ci, c in enumerate(('hp', 'next', 'opposite')):
            x = gi * 1.2 + (ci - 1) * .3
            v = means[c][gi] + rng.normal(0, 5, 28)
            ax.scatter(x - .1 + rng.uniform(-.04, .04, 28), v, s=5, color=cols[c], alpha=.6, lw=.3,
                       edgecolors=edge[c] if c == 'opposite' else 'none', zorder=2)
            m, se = v.mean(), v.std() / np.sqrt(28)
            ax.plot([x + .06] * 2, [m - 2 * se, m + 2 * se], color=cols[c] if c != 'opposite' else edge[c], lw=1.2)
            ax.plot(x + .06, m, 'o', ms=4, color=cols[c], mec=edge[c], mew=.6, zorder=4)
    ax.axhline(0, color='.6', lw=.6, ls='--', zorder=0)
    ax.set_xticks([0, 1.2, 2.4])
    ax.set_xticklabels(groups, fontsize=6.5)
    ax.set_ylabel('Response vs mean (%)', fontsize=6.5)
    ax.set_ylim(-22, 30)
    for k, (c, lab) in enumerate((('hp', 'HP'), ('next', 'Next to HP'), ('opposite', 'Opposite'))):
        location_icon(ax, .05 + k * .36, 1.08, c, s=.12, ramp=ramp)
        ax.text(.12 + k * .36, 1.08, lab.replace('Next to HP', 'Next'), transform=ax.transAxes, color=txt[c],
                fontsize=6, va='center')


def mock_behaviour_decoding(fig, gs):
    """Fig. 1c-like behaviour row (distractor HP / LP / absent) and Fig. 6-like decoding map + quadrant strip."""
    rng = np.random.default_rng(3)
    # behaviour
    ax = fig.add_subplot(gs[0])
    conds = [('HP', HP, HP_DEEP, 870), ('LP', NEXT, HP_DEEP, 900), ('Absent', NEUTRAL, '#52606D', 860)]
    for i, (lab, col, ec, m) in enumerate(conds):
        v = m + rng.normal(0, 60, 28)
        ax.scatter(i - .12 + rng.uniform(-.06, .06, 28), v, s=5, color=col, alpha=.55, lw=0)
        ax.plot(i + .1, v.mean(), 'o', ms=5, color=col, mec=ec, mew=.7, zorder=4)
    ax.plot([.1, 1.1, 2.1], [870, 900, 860], color=FG, lw=.8, zorder=3)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['HP', 'LP', 'Absent'], fontsize=6.5)
    for lab, tl in zip(['HP', 'LP', 'Absent'], ax.get_xticklabels()):
        tl.set_color({'HP': HP, 'LP': NEXT, 'Absent': '.4'}[lab])
    ax.set_ylabel('Response time (ms)', fontsize=6.5)
    ax.set_ylim(700, 1050)
    for i, items in enumerate([[(HP_IX, 'distractor')], [(T_IX, 'distractor')], []]):
        ins = ax.inset_axes([i / 3 + .04, -.42, .26, .26])
        disc_axes(ins)
        mini_display(ins, items=items)
    ax.text(-.3, 1.12, 'c', transform=ax.transAxes, fontsize=8, fontweight='bold', va='bottom')
    ax.text(0, 1.12, 'Fig. 1c-like: distractor condition', transform=ax.transAxes, fontsize=6.5, color='.4', va='bottom')
    # decoding map
    ax = fig.add_subplot(gs[1])
    disc_axes(ax)
    th = np.arctan2(Y, X)
    bump = lambda p, w: np.exp((np.cos(th - np.arctan2(p[1], p[0])) - 1) / w)  # noqa: E731
    m = -1.6 * bump(RING[HP_IX], .25) + .7 * bump(RING[1], .3) + .35 * np.sin(1.1 * X + .5) * np.cos(.9 * Y)
    m = np.where(np.hypot(X, Y) < 3.17, m, np.nan)
    im = ax.imshow(m, extent=(-6, 6, -6, 6), origin='lower', cmap=div_cmap(), vmin=-2, vmax=2)
    ax.add_patch(Circle((0, 0), 3.17, fill=False, ec='.55', lw=.5))
    for i, p in enumerate(RING):
        ax.add_patch(Circle(p, .9, fill=False, ec=HP if i == HP_IX else '.55', lw=.9 if i == HP_IX else .5,
                            ls=(0, (2.5, 1.2))))
    ax.text(*(RING[HP_IX] + [1.3, -.6]), 'HP', color=HP, fontsize=6.5)
    cax = ax.inset_axes([.12, -.1, .76, .05])
    cb = fig.colorbar(im, cax=cax, orientation='horizontal', ticks=[-2, 0, 2])
    cb.ax.tick_params(labelsize=6, length=2)
    cb.outline.set_linewidth(.4)
    cb.ax.text(-.04, .5, 'Less', transform=cb.ax.transAxes, color=HP, fontsize=6, ha='right', va='center')
    cb.ax.text(1.04, .5, 'More', transform=cb.ax.transAxes, color=DIV_MORE, fontsize=6, ha='left', va='center')
    ax.set_title('Fig. 6e-like decoded intensity', fontsize=6.5, color='.4', pad=2)
    # quadrant strip
    ax = fig.add_subplot(gs[2])
    for i, (c, col, ec, mu) in enumerate((('HP', HP, HP_DEEP, -1.1), ('Next to HP', NEXT, HP_DEEP, .4),
                                          ('Opposite', OPP, HP_DEEP, .3))):
        v = mu + rng.normal(0, 1.2, 28)
        ax.scatter(i - .15 + rng.uniform(-.06, .06, 28), v, s=5, color=col, alpha=.65, lw=.3,
                   edgecolors=ec if col == OPP else 'none')
        ax.plot(i + .12, v.mean(), 'o', ms=5, color=col, mec=ec, mew=.7, zorder=4)
        ax.plot([i + .12] * 2, [v.mean() - .45, v.mean() + .45], color=ec if col == OPP else col, lw=1.2, zorder=3)
    ax.plot([0, 1], [3.6] * 2, color=HP, lw=.6)
    ax.text(.5, 3.65, '**', color=HP, ha='center', va='bottom', fontsize=7)
    ax.axhline(0, color='.6', lw=.6, ls='--', zorder=0)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['HP', 'Next to HP', 'Opposite'], fontsize=6.5)
    for tl, col in zip(ax.get_xticklabels(), (HP, NEXT, '#8C6B9E')):
        tl.set_color(col)
    ax.set_ylabel('Decoded intensity (%)', fontsize=6.5)
    ax.set_ylim(-4.5, 4.5)


def mock_pathways(fig, gs):
    """Fig. 3b-like: pathways as line styles in the quantity's own colour (here the target)."""
    ax = fig.add_subplot(gs[0])
    t = np.linspace(-4, 20, 200)
    h = np.exp(-(t - 4.8) ** 2 / 5) - .2 * np.exp(-(t - 11) ** 2 / 12)
    for amp, (name, ls) in zip((.14, .31, .27), PATH_LS.items()):
        ax.fill_between(t, amp * h - .015, amp * h + .015, color=TARGET, alpha=.15, lw=0)
        ax.plot(t, amp * h, color=TARGET, lw=1.3, ls=ls)
    for k, (name, ls) in enumerate(PATH_LS.items()):
        ax.plot([.55, .67], [.92 - k * .11] * 2, transform=ax.transAxes, color=TARGET, lw=1.3, ls=ls)
        ax.text(.7, .92 - k * .11, name.replace('Occipitotemporal', 'Occ.-temporal'), transform=ax.transAxes,
                color=FG, fontsize=6, va='center')
    ax.axhline(0, color='.6', lw=.6, ls='--', zorder=0)
    ax.set_xticks([0, 10, 20])
    ax.set_xlabel('Time from search onset (s)', fontsize=6.5)
    ax.set_ylabel('Extra response (% signal)', fontsize=6.5)
    ax.set_title('Target − neutral item', fontsize=6.5, color=TARGET, pad=2)
    ax.text(-.3, 1.12, 'd', transform=ax.transAxes, fontsize=8, fontweight='bold', va='bottom')
    ax.text(0, 1.2, 'Fig. 3b-like: pathways by line style; Fig. 2a-like area families (map only)',
            transform=ax.transAxes, fontsize=6.5, color='.4', va='bottom')
    ax = fig.add_subplot(gs[1])
    ax.set_axis_off()
    ax.set_xlim(0, 9.5)
    ax.set_ylim(0, 3)
    for gi, (g, rois) in enumerate([('Early', ('V1', 'V2', 'V3')), ('Dorsal', ('V3AB', 'IPS')),
                                    ('Occipitotemporal', ('hV4', 'VO', 'LO', 'TO'))]):
        ax.text(.0, 2.55 - gi * .95, g, fontsize=6.3, color=FG, va='center')
        for k, r in enumerate(rois):
            ax.add_patch(Rectangle((3.2 + k * 1.55, 2.25 - gi * .95), 1.45, .6, color=AREA[r], lw=0))
            ax.text(3.2 + k * 1.55 + .72, 2.55 - gi * .95, r, fontsize=6, ha='center', va='center',
                    color='w' if r in ('V1', 'V2', 'V3AB', 'hV4', 'VO') else FG)


def mocks():
    fig = plt.figure(figsize=(7.1, 7.6))
    outer = fig.add_gridspec(4, 1, height_ratios=[1.05, 1.25, 1.25, 1.15], hspace=.75, left=.08, right=.98, top=.95,
                             bottom=.04)
    mock_fields(fig, outer[0].subgridspec(1, 6, wspace=.15))
    row = outer[1].subgridspec(1, 4, wspace=.5, width_ratios=[1, 1.2, 1, 1.2])
    mock_locations(fig, [row[0], row[1]], ramp=True, letter_='b')
    mock_locations(fig, [row[2], row[3]], ramp=False, letter_='b′')
    mock_behaviour_decoding(fig, outer[2].subgridspec(1, 3, wspace=.45))
    mock_pathways(fig, outer[3].subgridspec(1, 2, wspace=.3, width_ratios=[1, 1.6]))
    fig.savefig(HERE / 'mocks.pdf')
    fig.savefig(HERE / 'mocks.png', dpi=200)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3] / 255.
    plt.close(fig)
    # 2 x 2 contact sheet: normal and three CVD simulations of the same raster
    sims = [sim_rgb(img, v) for v in VIEWS]
    fig2, axs = plt.subplots(2, 2, figsize=(14.2, 15.4))
    for ax, s, v in zip(axs.ravel(), sims, VIEWS):
        ax.imshow(np.clip(s, 0, 1))
        ax.set_title(v, fontsize=14)
        ax.set_axis_off()
    fig2.subplots_adjust(left=.01, right=.99, top=.97, bottom=.01, wspace=.03, hspace=.06)
    fig2.savefig(HERE / 'mocks_cvd.png', dpi=110)
    plt.close(fig2)


if __name__ == '__main__':
    swatches()
    mocks()
    for a, b in itertools.combinations(NAMED, 2):
        vals = [de(NAMED[a], NAMED[b], v) for v in VIEWS]
        if min(vals) < 12:
            print(f'{a:12s} {b:12s}', ' '.join(f'{x:5.1f}' for x in vals))
