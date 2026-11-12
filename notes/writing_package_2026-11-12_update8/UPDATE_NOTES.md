# Update 8 — pRFs refitted together with the gain fields (applies on top of update 7)

> **Provisional until the voxel-selection check is decided.** A second voxel selection (reliability against a
> circular-shift null, more voxels in the higher areas) is being fitted. If it clearly shrinks the standard errors it
> may replace the current selection; every file in this update is then regenerated with one command (below), and the
> numbers will change again. The direction of the results is not expected to change, but do not polish wording around
> exact values yet.

The gain-field estimates now come from fits in which each voxel's pRF was refitted together with the gain fields
(previously: pRFs held at the fit without gain fields). This is the new reference for every gain-field number. As
decided: items, target, distractor and the sustained gains are reported from the joint fit (all terms together);
dS, dD and dT estimated alone come from the refitted model with item weight, target, distractor and sustained gain.
The models without these terms were not refitted.

## Files

| File | Status | What |
|---|---|---|
| `figures/fig5_gain_fields_v11_n28.{pdf,png,md}` | **replaces** `fig5_gain_fields_v10_n28` as Fig. 5 | Same layout (option B: all-areas pooled estimate first in c); refitted fits; map widths = median among fits with a reliable sustained gain (update 7); every sentence naming significant areas computed from the data |
| `figures/figS_field_widths_v3_n28.{pdf,png,md}` | **replaces** `figS_field_widths_v2_n28` (Fig. S4) | Update 7's analysis rerun on the refitted joint fits |
| `figures/figS_gain_field_map_search_v3_n28.{pdf,png,md}` | **replaces** `figS_gain_field_map_search_v2_n28` (Fig. S5) | Refitted fits, reliable-gain widths |
| `figures/figS_gain_vs_prf_size_polished_2026-11-12_n28.{pdf,png,md}` | **replaces** `figS_gain_vs_prf_size_n28` (Fig. S6) | Refitted joint fits; now in % |
| `figures/figS_pooled_dS_dD_polished_2026-11-12_n28.{pdf,png,md}` | **replaces** `figS_pooled_dS_dD_n28` (Fig. S7) | Refitted fits; in %; "estimated alone" / "joint fit" instead of model codes; numbers in the legend |
| `figures/table_parameters_2026-11_polished_2026-11-12_n28.md` | **replaces** `table_parameters_2026-11_n28` (Table S3) | Refitted DoG fits, in %; the DN columns are the earlier DN fits (not refitted), as the title says |
| `key_numbers.md` | **append / replace** the matching lines | Every changed number, earlier → new, with its source table |
| `key_numbers_reference_full.md` | new | All gain-field numbers of the new reference, generated from the tables |
| `methods_refit.md` | new | Methods paragraph for the refit |

Captions are the `.md` next to each figure. Supplementary numbering is unchanged from update 7 (only the files
behind S4–S7 and Table S3 change).

## What to change in the manuscript

1. **Methods**: add the refit paragraph (`methods_refit.md`): pRFs refitted together with the gain fields, the
   alternating polish (two rounds: per-voxel pRFs with Adam, shared terms with bounded L-BFGS, a step kept only if it
   lowers the training error; HRF fixed), convergence: training error lower in every fit (median −0.03%), pRF centres
   move a median 0.009°.
2. **Results, the lead statistic for the sustained HP effect** (Fig. 5c): all areas pooled, dS estimated alone
   **−1.0% [−1.9, −0.4], negative in 22 of 28 participants, p = 2.7 × 10⁻⁴** (was −1.4%, p = 1.1 × 10⁻⁴). Per area,
   significant in **V3 (−2.7%), hV4 (−2.4%) and TO (−1.9%), each q = .036** (was V3 and TO). Joint fit **−3.5%
   [−6.7, −1.3], p = .0044** (was −3.4%, p = .0027), significant in V3 (−9.5%) and hV4 (−5.6%); TO (−3.1%, q = .084) no
   longer. Drop the per-pathway clause ("present in early and occipitotemporal areas, not reliable in dorsal areas"):
   after the refit no pathway survives correction (early q = .057, occipitotemporal q = .089).
3. **Results, the sustained gain shared by all four locations**: the LP gain is larger in higher areas after the refit
   (pooled −11%, was −6.7%; TO −67%, was −35%; IPS −58%, was −33%) because it trades off against the item weight
   (ρ = −.73 between the changes, across fits) and the pRF amplitude (ρ = −.17). Report its sign and where it is
   significant (unchanged: V3AB, IPS, VO, LO, TO), not its size; e.g. "a suppression shared by all four locations in
   most higher areas, whose size is poorly separated from the item weight and the pRF amplitude".
4. **Target and distractor gains: unchanged** in direction and significance (target pooled +17%, p = 3.2 × 10⁻⁷, six
   areas; distractor +8.0%, p = .023, V3AB, hV4, TO). Update the values (key_numbers.md); VO target 50 → 38%.
   Transient HP − LP (Fig. 5d): dD −0.8% (p = .14), dT −3.6% (p = .032), dD − dT n.s., as before.
5. **Field widths (update 7 paragraph)**: claim (a) holds and is stronger: sustained FWHM 2.8° [2.7, 2.9] among fits
   with a reliable gain, narrower than 5.7° in 28 of 28 participants (p = 7.5 × 10⁻⁹). Claim (b), wider in early areas:
   now p = .013 at |gain| / SE > 2 (3.4° vs 2.7°, 15 of 24) but p = .58 at > 3; keep update 7's "no evidence" wording or
   state both thresholds. Do not reinstate the Richter et al. "broad early, refined later" link on this basis.
6. **Fig. 5a text**: "the sustained field is lower at HP than at the other locations (by 2.7% in V3, 2.4% in hV4, 1.9%
   in TO)".

## Not regenerated (and why)

- **Gaussian and DN robustness fits** (Figs. S10, S11, the robustness table, the DN columns of Table S3): not refitted
  with the gain fields; they stay as they are (pRFs held at the fit without gain fields). Say so in the Methods:
  "The robustness fits with Gaussian and DN pRFs keep the pRFs at their fit without gain fields."
- **Leave-one-subject-out tests** (Fig. S8, Table S4) and the κ (Fig. S9), split-model (next vs opposite HP; target
  split) and one-width numbers: these use models (m00–m3b, m8 variants, LOSO δ) that were not refitted.
- **Per-session and per-block dS** (Table S2, Richter follow-up A3), **dS vs behaviour** (ρ = −.11), **baseline vs
  gain** (Fig. S13, update 6), **trial-wise gains** (Fig. 7) and **"how much of the measured HP reduction the joint fit
  reproduces"**: built on the earlier joint fits; not rerun in this update.
- **The distractor gain on top of the target term and the shared sustained gain estimated alone (S)**: their base
  models were not refitted, so their values are unchanged.
- **Fig. 6 decoding**: its fitted-model prediction already uses the refitted joint fit (Fig. 6 v5).

## Note on a discrepancy with the refit report

`notes/prf_refit_with_gains_2026-11-12.md` quotes dS estimated alone as −1.3% (p = 2 × 10⁻⁴) with TO at q = .065 and
hV4 −2.7%. Those numbers include estimated-alone fits whose linearity check failed (`valid` = False; 13 of 241 dS fits on
the refitted base, against 8 before, some with absurd values such as ±1000%). Every earlier paper number (and Fig. 5)
excludes them; with the same rule the refitted values are −1.0% (p = 2.7 × 10⁻⁴), hV4 −2.4% and TO −1.9% (both
q = .036). This update uses the paper's rule throughout.

## Regenerating for another reference fit (one command per stage)

Everything in this update is produced by `retsupp/visualize/paper/regenerate_reference.sh`. The reference fit is one
variable set: `REF_ROOT` (ladder root with m4_TDS and m7_full), `ALONE_FOLDER` (estimated-alone root), `TAG` (table
names) and `SUFFIX` (figure names). This update used the defaults:

```bash
# cluster login node, ~/retsupp_plan after git pull
bash retsupp/visualize/paper/regenerate_reference.sh cluster-submit    # width profiles (28 tasks, ~7 min each)
bash retsupp/visualize/paper/regenerate_reference.sh cluster-collect   # srun: joint table, alone table, widths
# laptop, repo root
REFIT_PREFIX=notes/data/prf_refit_2026-11-12 bash retsupp/visualize/paper/regenerate_reference.sh local
```

For the voxel-selection alternative, e.g.:

```bash
export REF_ROOT=af_prf_cv_ladder_nested_full_bounded_maxsd20_relnull_r0.2_roihrf_polished \
       ALONE_FOLDER=af_fixed_group_mechanisms_relnull_r0.2_polished TAG=relnull_r0.2_polished SUFFIX=_relnull02_n28
SUBMIT_ALONE=1 bash retsupp/visualize/paper/regenerate_reference.sh cluster-submit   # + the estimated-alone fits
bash retsupp/visualize/paper/regenerate_reference.sh cluster-collect
bash retsupp/visualize/paper/regenerate_reference.sh local
```

The `local` stage writes Fig. 5 v11, the search, field-width, gain-vs-pRF-size and pooled supplements, the parameter
table and `notes/data/reference_numbers_<TAG>_n28.{md,tsv}` (the full key numbers); no earlier file is overwritten.
