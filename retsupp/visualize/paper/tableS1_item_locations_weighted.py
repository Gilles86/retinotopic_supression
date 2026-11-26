"""Supplementary Table S1 in full: the Fig. 3 statistics with the precision-weighted own-data columns.

Prepends to ``tableS_item_locations_n28.md`` (Wilcoxon, ANOVAs; unchanged) a section with each unit fitted on its own
data, every participant weighted by the precision of their estimate (``fig3_meta``: leave-one-run-out jackknife SE,
estimate ~ Student-t(ν, μ, √(SE² + τ²)), no pooling over areas): the pathways and all areas with voxels pooled as in
the manuscript (``own_data_units.tsv``), and each area (``own_data.tsv``); and adds the weighted columns to the per-area
table. SE calibration from ``calibration.tsv``.

    python -m retsupp.visualize.paper.tableS1_item_locations_weighted
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from retsupp.visualize.paper.paper_style import ROI_ORDER

ITEM_TERMS = ['target − neutral', 'distractor − neutral', 'target − distractor']
LOC_TERMS = ['HP − opposite', 'next − opposite', 'HP − next']
UNITS = ['Early', 'Dorsal', 'Occipitotemporal', 'All areas']


def f(r, nd=3, unit=''):
    p = max(r.p_gt0, r.p_lt0)
    ps = 'P > .999' if p >= .9995 else f'P = {p:.3f}'.replace('0.', '.')
    star = ' **' if p > .995 else ' *' if p > .975 else ''
    return (f"{r['mean']:+.{nd}f}{unit} [{r.hdi_lo:+.{nd}f}, {r.hdi_hi:+.{nd}f}], {ps}{star}").replace('-', '−')


def section(units, area):
    u = units.set_index(['unit', 'term'])
    a = area.set_index(['unit', 'term'])
    get = lambda unit, term: u.loc[(unit, term)] if unit in UNITS else a.loc[(unit, term)]   # noqa: E731
    rows = UNITS + ROI_ORDER
    out = ['**1a. Item types (Fig. 3b, c), % signal.** Posterior mean [95% HDI], P = posterior probability of the sign; '
           '* P > .975, ** P > .995.', '',
           '| Unit | n | Target − neutral | Distractor − neutral | Target − distractor |', '|---|---|---|---|---|']
    for r in rows:
        n = int(get(r, ITEM_TERMS[0])['n'])
        out.append(f"| {r} | {n} | " + ' | '.join(f(get(r, t)) for t in ITEM_TERMS) + ' |')
    out += ['', '**1b. Location type (Fig. 3d, e), items pooled, % signal (absolute units).**', '',
            '| Unit | n | HP − opposite | Next − opposite | HP − next |', '|---|---|---|---|---|']
    for r in rows:
        n = int(get(r, 'HP − opposite (abs)')['n'])
        out.append(f"| {r} | {n} | " + ' | '.join(f(get(r, f'{t} (abs)')) for t in LOC_TERMS) + ' |')
    out += ['', '**1c. Location type, % of the participant\'s mean response (relative units, as Fig. 3e).** Pathways and all '
            'areas only (per area the mean response is often near 0, so the relative values are unstable).', '',
            '| Unit | n | HP − opposite | Next − opposite | HP − next |', '|---|---|---|---|---|']
    for r in UNITS:
        if (r, 'HP − opposite (rel)') not in u.index:
            continue
        n = int(get(r, 'HP − opposite (rel)')['n'])
        out.append(f"| {r} | {n} | " + ' | '.join(f(get(r, f'{t} (rel)'), 1, '%') for t in LOC_TERMS) + ' |')
    return '\n'.join(out)


def main(meta, units_dir, old, out):
    area = pd.read_csv(Path(meta) / 'own_data.tsv', sep='\t')
    units = pd.read_csv(Path(units_dir) / 'own_data_units.tsv', sep='\t')
    cal = pd.read_csv(Path(meta) / 'calibration.tsv', sep='\t').set_index('term')
    div = int(area.divergences.sum() + units.divergences.sum())
    head = (
        "**Supplementary Table S1. Responses at the item locations (Fig. 3).** Same data, voxels and window (3.2 to 6.4 s "
        "after search onset) as Fig. 3. Section 1 gives the primary statistics: each unit fitted on its own data, every "
        "participant weighted by the precision of their estimate. Per participant and unit, the contrast in the window and "
        "its leave-one-run-out jackknife SE (half the odd − even-run difference over the SE: median "
        f"{cal.loc['HP − opposite (abs)', 'all']:.2f} for HP − opposite and {cal.loc['target − neutral', 'all']:.2f} for "
        "target − neutral over all areas, 0.67 expected); then estimate ~ Student-t(ν, μ, √(SE² + τ²)), ν ~ Gamma(2, 0.1), "
        "no pooling over areas (PyMC with nutpie, 4 × 2000 draws, target acceptance .99; "
        f"{div} divergences over all fits). Pathways and all areas: voxels pooled over the pathway's areas weighted by "
        "voxel count, as in Fig. 3; single areas: each area alone. Sections 2 to 5 give the earlier statistics (Wilcoxon "
        "signed-rank tests and repeated-measures ANOVAs on the unweighted values) as checks. n = 28 (fewer where "
        "participants lack voxels at some item location).")
    old_txt = Path(old).read_text()
    body = old_txt.split('\n', 1)[1]          # drop the old title paragraph
    body = (body.replace('**1. Location type × item type', '**2. Location type × item type')
                .replace('**2. Per item type:', '**3. Per item type:')
                .replace('**3. HP against all three LP', '**4. HP against all three LP')
                .replace('**4. DCT basis', '**5. DCT basis')
                .replace('## Per area: location type (Fig. 3d, e), n = 28', '## 6. Per area: location type (Fig. 3d, e), n = 28, unweighted')
                .replace('| All areas | — |', '| All areas | not computed |'))
    old_head = old_txt.split('\n', 1)[0].split('** ', 1)[1]
    txt = (head + '\n\n## 1. Precision-weighted, each unit on its own data (primary)\n\n' + section(units, area) +
           '\n\n## Sections 2 to 6: unweighted statistics (checks)\n\n' + old_head + '\n' + body)
    Path(out).write_text(txt.rstrip() + '\n')
    print('Wrote', out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--meta', default='notes/data/fig3_meta_2026-11-25')
    ap.add_argument('--units', default='notes/data/fig3_meta_units_2026-11-25')
    ap.add_argument('--old', default='notes/figures/paper_2026-11/tableS_item_locations_n28.md')
    ap.add_argument('--out', default='notes/figures/paper_2026-11/tableS1_item_locations_weighted_n28.md')
    a = ap.parse_args()
    main(a.meta, a.units, a.old, a.out)
