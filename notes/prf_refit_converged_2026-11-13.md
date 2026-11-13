# pRF + gain-field polish to convergence, 2026-11-13: superseded

**Superseded by the zero-sum LM refit; partial results kept for reference.** With free pRFs the shared sustained level
g_LP is not identifiable (it acts as a pRF-shape term; `notes/joint_optimisation_2026-11-13.md`), and the alternating
polish was far from the optimum. The paper's model becomes a zero-sum sustained field fitted with Levenberg–Marquardt
(another agent). Do not use the numbers below for the paper.

## What was run

- Continuation of the 2-round alternating polish (`..._reliable_roihrf_polished`) by up to 6 more rounds (8 in total),
  stopping when a round lowers training SSE by < 1e-5 (`--polish-tol`, commit 0a070de), into
  `derivatives/af_prf_cv_ladder_nested_full_bounded_maxsd20_reliable_roihrf_polished_conv` (warm start from the
  2-round fit; `--warm-polish-root`). Jobs 8332149 (m4_TDS), 8332150 (m7_full), canary 8332074 (sub-03 V1/TO/IPS m7_full).
- Complete: m4_TDS 241/241 cells, m7_full 237/241 (sub-21 V2, sub-21 V3, sub-28 V2, sub-29 V1 cancelled at round 4–5;
  their per-round checkpoints are in the root).
- Estimated alone on the 8-round m4_TDS: `derivatives/af_fixed_group_mechanisms_polished_conv_2026-11-13` (job 8340844,
  all mechanisms, same settings as 8314102), table `notes/data/conditional_estimates_polished_conv_2026-11-13.tsv`.
  Not analysed.
- Alternative start, sub-03 IPS, current selection, from the r > 0.2 polished fit, 8 rounds:
  `..._reliable_roihrf_polished_conv_warmr02` (jobs 8332077, 8332078).
- `regenerate_reference.sh` was not run.

## What it showed

- **No convergence.** Only 35/241 (m4_TDS) and 51/237 (m7_full) cells met the 1e-5 criterion within 8 rounds; median
  last-round relative SSE change 5.0e-5 / 4.1e-5 (largest in TO 1.2e-4, V3AB, VO, IPS; smallest in V1–V3 ≈ 2e-5).
  SSE vs 2 rounds: median −0.04 %. pRF centres moved little more (median 0.007–0.009°, near items 0.04°); amplitudes
  +3–4 % (+6 % near items).
- **g_LP keeps drifting; dS does not.** sub-03 IPS:

| Model | Fit | Training SSE | g_LP | dS (pp) | Target | Item weight |
|---|---|---|---|---|---|---|
| m4_TDS | clamped | 859 563 | −35 % | — | +9.3 % | 24.9 |
| | 2 rounds | 858 820 | −62 % | — | +10.0 % | 30.4 |
| | 8 rounds, current start | 857 914 | −83 % | — | +5.0 % | 38.4 |
| | 8 rounds, r > 0.2 start | 857 671 | −87 % | — | +4.0 % | 40.5 |
| m7_full | clamped | 859 147 | −33 % | +3.3 | +10.5 % | 24.3 |
| | 2 rounds | 858 243 | −59 % | +2.4 | +10.5 % | 29.0 |
| | 8 rounds, current start | 857 543 | −81 % | +1.8 | +8.0 % | 34.0 |
| | 8 rounds, r > 0.2 start | 857 300 | −86 % | +1.7 | +7.7 % | 34.0 |

  Neither start converged (last round 1.2e-4 / 8.6e-5). The SSE keeps falling as g_LP heads towards −100 % and the item
  weight rises, while dS settles (+1.7 / +1.8 pp from the two starts): the sustained level trades off against pRF
  shape and item weight, consistent with the identifiability analysis.

## Files

- `notes/data/prf_refit_conv_vs_2rounds_2026-11-13_shared.tsv` (per cell: shared terms, SSE, rounds, converged flag,
  SSE per round; `fit` = clamped means the 2-round fit, polished the 8-round fit),
  `notes/data/prf_refit_conv_vs_2rounds_2026-11-13_prf_shift.tsv`.
- `notes/data/conditional_estimates_polished_conv_2026-11-13.tsv`.
