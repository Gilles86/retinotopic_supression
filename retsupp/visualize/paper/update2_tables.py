"""Supplementary table of the Fig. 3e location contrasts and the sustained dS per session and per block (n = 28), from
the Richter follow-up tables (A1, A3). The session and block difference tests are the inference.

    python -m retsupp.visualize.paper.update2_tables --out notes/figures/paper_2026-11/tableS_sessions_blocks_n28.md
"""
from __future__ import annotations

import argparse

import pandas as pd

A1 = 'notes/data/richter_A1_carry_over_common_denominator_n28_2026-11-11.tsv'
A3 = 'notes/data/richter_A3_session_gains_n28_2026-11-11.tsv'
LAB = {'hp_opp': 'HP − opposite', 'next_opp': 'Next − opposite', 'hp_next': 'HP − next'}
SPLITS = [('ses1', 'Session 1'), ('ses2', 'Session 2'), ('ses2 − ses1', '**Session 2 − 1**'),
          ('block1', 'Block 1'), ('block2', 'Block 2'), ('block2 − block1', '**Block 2 − 1**')]


def fmt_p(p):
    return f'{p:.3f}'.lstrip('0') if p >= .001 else f'{p:.1e}'


def cell(r, d=1):
    return (f'{r["mean"]:+.{d}f} ± {r["sem"]:.{d}f} [{r.mean_ci_lo:+.{d}f}, {r.mean_ci_hi:+.{d}f}], '
            f'p = {fmt_p(r.p)} (n = {int(r.n)})').replace('-', '−')


def main(out):
    a = pd.read_csv(A1, sep='\t')
    lines = ['**Supplementary Table. Location effects at the item locations and the sustained HP − LP gain, per session and '
             'per block (n = 28).**', '',
             'a, Item responses (Fig. 3e measure: each location type relative to the mean of the four locations, in % of '
             "the subject's mean response; items pooled), mean ± SEM [bootstrapped 95% CI of the mean], Wilcoxon signed-rank "
             'p. Rows in bold are the inference: the difference between sessions (session 2 − 1) and between the first and '
             'second block of a session (block 2 − 1), paired across participants. Source: `' + A1 + '`.', '']
    for grp in ('Early', 'All areas'):
        lines += [f'*{grp}*', '', '| Split | ' + ' | '.join(LAB.values()) + ' |', '|---|---|---|---|']
        for key, name in SPLITS:
            cells = []
            for lab in LAB:
                r = a[(a.group == grp) & (a.split == key) & (a.label == lab)]
                cells.append(cell(r.iloc[0]) if len(r) else '—')
            lines.append(f'| {name} | ' + ' | '.join(cells) + ' |')
        lines.append('')
    b = pd.read_csv(A3, sep='\t')
    lines += ['b, Sustained gain at HP minus at LP (dS) and the split of the LP gain (next to HP − opposite HP), each '
              'estimated alone per session with all other parameters fixed at the all-run fit; per-participant median over '
              'areas (or over V1–V3 for "Early"); mean ± SEM [bootstrapped 95% CI of the mean], Wilcoxon p. Source: `'
              + A3 + '`.', '', '| Quantity | Areas | Value |', '|---|---|---|']
    for r in b.itertuples():
        bold = '**' if 'session 2 − session 1' in r.label else ''
        lines.append(f'| {bold}{r.label}{bold} | {r.group} | '
                     + f'{r.mean:+.3f} ± {r.sem:.3f} [{r.mean_ci_lo:+.3f}, {r.mean_ci_hi:+.3f}], p = {fmt_p(r.p)} (n = {r.n})'
                     .replace('-', '−') + ' |')
    lines += ['', 'No session or block difference is significant (bold rows; dS session 2 − 1 p = .095), so the location '
              'effects do not differ reliably between sessions or blocks. Each session or block alone holds a half or a '
              'quarter of the data, and the per-split tests are descriptive: a difference between a significant and a '
              'non-significant split is not itself significant. In session 2 the two neighbours of HP were HP locations in '
              'session 1, so carry-over cannot be strictly excluded for the neighbour effect; a location that was HP '
              'before does not respond less than one that was never HP (`richter_followup_2026-11-11.md`, A2, p = .54).']
    open(out, 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', default='notes/figures/paper_2026-11/tableS_sessions_blocks_n28.md')
    main(ap.parse_args().out)


# k of n for every pooled share in key_numbers.md (n = 28), from update2_numbers.py
COUNTS = [
    ('Fig. 5a', 'Target gain, joint (> 0)', 'dog: Target gain, joint', '89%'),
    ('Fig. 5a', 'Distractor gain, joint (> 0)', 'dog: Distractor gain, joint', '75%'),
    ('Fig. 5a', 'Distractor gain on top of target, alone (> 0)', 'dog: Distractor on top of target, alone', '68%'),
    ('Fig. 5a', 'Target gain alone (> 0)', 'dog: Target gain alone', '—'),
    ('Fig. 5a', 'Target gain per doubling of pRF size (slope > 0)', 'Target gain per doubling of pRF size', '82%'),
    ('Fig. 5a', 'Distractor gain per doubling of pRF size (slope > 0)', 'Distractor gain per doubling of pRF size', '—'),
    ('Fig. 5', 'Sustained gain S alone (< 0)', 'dog: S alone', '100%'),
    ('Fig. 5', 'g_LP, joint (< 0)', 'dog: g_LP, joint', '24/28'),
    ('Fig. 5', 'g_HP, joint (< 0)', 'dog: g_HP, joint', '25/28'),
    ('Fig. 5c', 'dS alone (< 0)', 'dog: dS alone', '79%'),
    ('Fig. 5c', 'dS joint (< 0)', 'dog: dS, joint', '79%'),
    ('Fig. 5d', 'dD alone (< 0)', 'dog: dD alone', '61%'),
    ('Fig. 5d', 'dT alone (< 0)', 'dog: dT alone', '68%'),
    ('Fig. 5d', 'dD − dT alone (> 0)', 'dog: dD − dT alone', '—'),
    ('Fig. 5d', 'dD joint (< 0)', 'dog: dD, joint', '—'),
    ('Split', 'Sustained HP − opposite (< 0)', 'dog split model: sustained HP − opposite', '82%'),
    ('Split', 'Sustained HP − next (< 0)', 'dog split model: sustained HP − next', '—'),
    ('Split', 'Sustained next − opposite (< 0)', 'dog split model: sustained next − opposite', '—'),
    ('Split', 'One-width model dS (< 0)', 'dog one-width model: dS', '79%'),
    ('Split', 'Target split: target HP − LP (> 0)', 'dog target split: target HP − LP', '—'),
    ('Robustness', 'Gaussian dS alone (< 0)', 'gauss: dS alone (exact)', '82%'),
    ('Robustness', 'Gaussian dS joint (< 0)', 'gauss: dS, joint', '67%'),
    ('Robustness', 'DN dS alone (< 0)', 'dn: dS alone (exact)', '75%'),
    ('Robustness', 'DN dS joint (< 0)', 'dn: dS, joint', '71%'),
    ('Robustness', 'Gaussian S alone (< 0)', 'gauss: S alone (exact)', '100%'),
    ('Robustness', 'DN S alone (< 0)', 'dn: S alone (exact)', '96%'),
    ('Robustness', 'Gaussian target gain joint (> 0)', 'gauss: Target gain, joint', '—'),
    ('Robustness', 'DN target gain joint (> 0)', 'dn: Target gain, joint', '—'),
    ('κ', 'κ, full fits, per participant mean over areas (< 0)', 'kappa, full fits (per participant mean over areas)', '82%'),
]
LOSO = ['T', 'D', 'DonT', 'S', 'dS', 'dD', 'dT']
LOSO_NAMES = {'T': 'Target gain', 'D': 'Distractor gain, tied', 'DonT': 'Distractor on top of target', 'S': 'Sustained gain, tied',
              'dS': 'dS', 'dD': 'dD', 'dT': 'dT'}


def counts_table(numbers, out):
    t = pd.read_csv(numbers, sep='\t').set_index('label')
    L = ['**k of n for every pooled share in `key_numbers.md` (n = 28).** k = participants whose per-participant value '
         '(median over the nine areas unless stated) lies in the direction given; computed from the per-participant tables '
         f'by `retsupp/visualize/paper/update2_numbers.py` (`{numbers}`, column k, n; source table per row there). '
         '"key_numbers" is the share as written in the package; a mismatch is flagged.', '',
         '| Section | Effect (direction) | k / n | key_numbers says | Median [95% CI] | p |', '|---|---|---|---|---|---|']
    for sec, name, lab, kn in COUNTS:
        r = t.loc[lab]
        share = f'{100 * r.k / r.n:.0f}%'
        flag = '' if kn in ('—', share, f'{int(r.k)}/{int(r.n)}') else f' **(should be {share})**'
        num = f'{r.value:+.3f} [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}] | {fmt_p(r.p)}'.replace('-', '−')
        L.append(f'| {sec} | {name} | {int(r.k)}/{int(r.n)} | {kn}{flag} | {num} |')
    L += ['', '*Leave-one-subject-out ("better": held-out ΔR² > 0; "direction": sign-flip test > 0), median over areas per '
          'participant, n = 28.*', '', '| Term | DoG better | DoG direction | Gaussian better | Gaussian direction | DN better | '
          'DN direction |', '|---|---|---|---|---|---|---|']
    for m in LOSO:
        cells = []
        for fam in ('dog', 'gauss', 'dn'):
            for kind in ('better', 'direction'):
                lab = f'{fam} LOSO {m}: {kind}'
                cells.append(f'{int(t.loc[lab].k)}/{int(t.loc[lab].n)}' if lab in t.index else '—')
        L.append(f'| {LOSO_NAMES[m]} | ' + ' | '.join(cells) + ' |')
    for lab, name in (('LOSO kappa', 'κ (DoG centre/surround tilt)'), ('LOSO dnw', 'DN baseline field, common b,d weight'),
                      ('LOSO dnratio', 'DN baseline field, b/d ratio')):
        b, d = t.loc[f'{lab}: better'], t.loc[f'{lab}: direction']
        L.append(f'| {name} | {int(b.k)}/{int(b.n)} | {int(d.k)}/{int(d.n)} | — | — | — | — |')
    L += ['', 'κ held-out runs (run 3, pRFs frozen): ΔR² < 0 in 26/27 participants (96%; sub-08 has no run-3 fit; '
          '`notes/data/figS_dog_kappa_n28_2026-11-11.tsv`).']
    open(out, 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', default='notes/figures/paper_2026-11/tableS_sessions_blocks_n28.md')
    ap.add_argument('--counts-out', default='notes/data/update2_counts_k_of_n_n28_2026-11-12.md')
    ap.add_argument('--numbers', default='notes/data/update2_numbers_n28_2026-11-12.tsv')
    a = ap.parse_args()
    main(a.out)
    counts_table(a.numbers, a.counts_out)
