# Update 14, 25 Nov 2026: answers to Docky's edits and hierarchical-model statistics for the gain fields

Apply on top of update 13. Two changes:
1. answers to Docky's review points, with suggested wording;
2. the gain-field terms (Fig. 5) and the trial-wise effects (Fig. 7) are now tested with one hierarchical Bayesian model (Student-t likelihood, each estimate weighted by its precision).

Where parts A and B disagree, part B wins.

## Contents

| File | What |
|---|---|
| `parts/A_answers_to_docky.md` | Short answers to Docky's points 1 to 9, with suggested wording |
| `parts/B_hierarchical_model_numbers.md` | The new statistics: Methods text, every Fig. 5 and Fig. 7 number, and the sentences in the abstract, Results and Discussion that change |
| `parts/C_full_report_with_p_table.md` | Full report with sources, plus the table of every p and q value in the Results (exact and threshold form) |
| `figures/main/fig5_gain_fields_v11_abc_zs_n28_palA_bayes` | **Fig. 5, replaces update 13's Fig. 5.** Panels b and c from the hierarchical model, with all areas and the three pathways left of the divider; panel a draws the same estimates |
| `figures/main/fig7_compact_zs_lm_2026-11-13_n28_studentt` | **Fig. 7, replaces update 13's Fig. 7.** Student-t likelihood; pathway values are the mean of their areas |
| `figures/main/fig1_paradigm_behaviour_v6_n28_palA` | Fig. 1 v6 (unchanged since update 13; Docky's tab still has the older Fig. 1). Panel c has the LP − absent and HP − absent strips he asked for |
| `figures/supplement/fig5_gain_fields_v11_abc_zs_n28_palA_hpdiag_bayes` | Alternative Fig. 5 with HP at the upper right (a real search position), for Docky's rotation question. Same numbers as the main Fig. 5 |
| `figures/supplement/figS_dS_overview_n28` | dS across all 19 fit versions and variants (pooled, nine areas, three pathways). The pooled reduction is significant in every version; only the per-area stars vary |
| `figures/videos/` | Draft supplementary videos: animated Fig. 4b (schematic gains) and true vs decoded bar in V1 (held-out run). Drafts, no captions yet |

## Bottom line

- **dS:** −3.6% of the drive [−5.2, −1.9], P > .999. Largest in early visual cortex (−6.3%), credible in all three pathways and in V1, V2, V3, V3AB, hV4 and TO.
- **Target gain:** +22.9% [+15.0, +30.7], credible in seven areas from V3 on.
- **Distractor gain:** +4.5% [+0.5, +8.6], P = .988; dorsal and occipitotemporal, not early.
- **The abstract changes:** the HP reduction is about a sixth of the target gain, and the distractor gain about a fifth (not half).
- **Fig. 7 conclusions stand:** target repeat −17%, distractor gain −2.7 per SD of log RT. Several pathway statements now become credible; part B has the wording.
- **Statistics convention for gain-field and trial-wise terms:** posterior mean, 95% HDI, P of the sign (credible: P > .975). Median/Wilcoxon results go to the supplement.

## Bookkeeping

- Estimates and posteriors: `notes/data/dS_meta_zs_lm_2026-11-25/`, script `retsupp/visualize/paper/dS_meta.py`.
- Fig. 7 posteriors: `notes/data/trialwise_meta_bayes_signal_se_zs_lm_2026-11-25_pathavg/`.
- Figures: `fig5_gain_fields_v6 --bayes-dir`, `fig7_compact --primary-t`, `figS_dS_overview`.
