# Voxel selection robustness: cross-run reliability against a shift null (2026-11-12/13)

**Question.** Do the gain-field estimates (dS = HP − LP sustained gain, and the LP gain) change, and do they become
more precise, when voxels are selected by model-free cross-run reliability instead of the current selection?

**Rule fixed in advance.** A looser selection is preferred only if it clearly shrinks the SEs (across participants)
of dS **and** the LP gain. It is never chosen by p value.

**Verdict: keep the current selection.** The looser selections shrink the dS SEs somewhat. For dS estimated alone,
pooled, the SE ratio to the current selection is 0.83 (r > 0.2) and 0.77 (r > 0.1); for dS in the joint fit it is
1.15 and 0.85. But they **increase** the LP-gain SEs: 1.21 and 1.10 in m7_full, 1.19 and 1.06 in m4_TDS. So the rule
is not met. All estimates keep their sign:
- dS pooled is negative under every selection: alone −1.2 / −1.0 / −0.8 %, joint −3.6 / −4.1 / −2.5 % (means).
- The LP gain is negative under every selection, and somewhat larger with the looser ones (m7_full pooled mean
  −13.6 / −19.1 / −17.4 %).

## Selections

| | Current (paper) | r > 0.2 | r > 0.1 |
|---|---|---|---|
| Rule | p_signal > 0.5 → top 500 by R² → LOO reliability > 0 | r > 0.2, p ≤ .01 vs circular-shift null | r > 0.1, p ≤ .01 |
| Also applied | BOLD SD ≤ 20 % | BOLD SD ≤ 20 %, finite m4 pRF, no cap | same |
| Fit root | `af_prf_cv_ladder_nested_full_bounded_maxsd20_reliable_roihrf_polished` | `..._maxsd20_relnull_r0.2_roihrf_polished` | `..._maxsd20_relnull_r0.1_roihrf_polished` |
| Subject × area cells with voxels | 241 | 249 | 252 |

- **What r is.** Per voxel, the mean over runs of corr(run, mean of the other runs), on cleaned BOLD. The bar
  sequence is the same in every run.
- **Null.** The other-runs mean circularly shifted by a random 30–228 TRs per run, 100 shifts; p = (#null ≥ r + 1)/101.
- **Code and tables.** `retsupp/modeling/cv/voxel_reliability_null.py`, written to
  `derivatives/prf_cv_roi_reliability_null/`. Sub-03 was recomputed by the repo code and is bit-identical to the
  prototype; the other 27 were copied from it.
- **Ladder options.** `fit_cv_ladder_nested.py --selection-table <dir> --min-r R`.

## Fits: warm-started alternating polish (pRFs and gain fields free together)

Only m4_TDS and m7_full were fitted, each with `--warm-polish-root` (the current polished root):
- **Shared terms** start at the current polished fit of the same participant × area × model.
- **pRFs** start from that fit where the voxel was in it. New voxels start from the whole-brain DoG pRF (model 4),
  with baseline and amplitude by least squares, then Adam on their pRFs with the shared terms fixed.
- **Polish.** Then 2 alternating rounds (Adam on all pRFs, L-BFGS on the shared terms), as in the current refit.
- **Without a warm start.** Cells where the current selection had no voxels (11) ran the full chain from m00.
- **Not fitted.** Cells with no voxels under r > 0.2: sub-06 IPS, sub-25 VO, sub-26 VO.
- **Settings.** CPU, 3 cores, 34 GB, one task per participant × area × model.
- **Jobs.**
  - r > 0.2: 8312819–8312832.
  - r > 0.1: 8312833–8312846.
  - Rerun of sub-17 V3 m7_full at r > 0.1: 8331464. It failed earlier on a transient NFS permission error; this
    rerun was still running at the time of writing.

**Does the result depend on the start?** Checked on sub-03 IPS: warm canaries 8312051 / 8312053 against a cold full
chain (m00 → m7 + the same polish; 8312052 / 8312054, roots `..._coldtest_polished`).

| Selection | Model | Training SSE, warm vs cold | g_LP warm / cold | dS warm / cold |
|---|---|---|---|---|
| r > 0.2 (329 vox) | m4_TDS | −0.20 % | −64 % / −18 % | — |
| r > 0.2 | m7_full | −0.18 % | −61 % / −17 % | +2.5 / +4.3 pp |
| r > 0.1 (722 vox) | m4_TDS | −0.16 % | −62 % / −15 % | — |
| r > 0.1 | m7_full | −0.16 % | −59 % / −15 % | +3.1 / +3.9 pp |

- **The warm start ends lower.** It reaches a lower training SSE than the cold chain in every case. Neither run
  converges within 2 rounds: the SSE still drops in round 2.
- **dS barely depends on the start; the LP gain does.** The LP gain drifts as the pRFs are released. The same drift
  is in the current paper fit: sub-03 IPS m4_TDS goes from −35 % clamped to −62 % polished.
- **Consequence.** The level of the sustained LP gain (a near-uniform field that trades off against pRF amplitude)
  is set partly by how far the polish has run, not only by the data. This holds for the current paper fit as much as
  for the new selections. More polish rounds, or a convergence criterion, would be needed before this level is
  quoted with precision. dS is far less affected.

## Estimates

- **Terms.**
  - dS estimated alone: `fit_fixed_group_mechanisms --mechanisms dS` on each selection's polished m4_TDS
    (jobs 8331645, 8331646). For the current selection: the paper's `conditional_estimates_polished_2026-11-12.tsv`.
    Fits whose linearity check failed (`valid` False) are left out, as for every paper number.
  - dS in the joint fit: m7_full, g_HP − g_LP.
  - LP gain: m7_full g_LP, and the tied sustained gain of m4_TDS.
- **Units.** % gain; differences in percentage points.
- **Cells.** n = 28 (sub-01/02 excluded). Only participant × area cells present in all three selections are used
  (paired), so per-area n can be below 28. Because of the pairing, the current selection's dS-alone pooled median
  here is −0.8 %, not the paper's −1.0 % on all its cells. The joint-fit pooled values (dS −3.5 % [−6.7, −1.3];
  g_LP −11 % [−21, −7.4]) match the paper's key numbers.
- **Pooling.** Per participant the median over the nine areas, as in `reference_numbers`.
- **Statistics shown.** Mean ± SE across participants (the decision statistic) and the paper's median [bootstrapped
  95 % CI].
- **Excluded cells.** Three r > 0.2 dS-alone cells that silently fell back to an old base root (no r > 0.2 fit
  there) were moved to `af_fixed_group_mechanisms_relnull_r0.2_polished_dS_fallbackbase_bak` and are not used.
- **Pending.** Joint-fit V3 has n = 27 until the sub-17 V3 r > 0.1 rerun lands.
- **Outliers.** Per-area means in TO, IPS and hV4 carry a few extreme participants (e.g. TO dS alone +14 to +31 %
  means against medians of −1 to −1.6 %). There, compare the medians and CIs rather than the SEs.

**dS (alone)** (%, mean ± SE across participants; median [95% CI]; median voxels)

| Area | n | Current | r > 0.2 | r > 0.1 | SE ratio r0.2 / r0.1 vs current |
|---|---|---|---|---|---|
| V1 | 28 | -1.11 ± 1.89; -0.69 [-1.51, -0.00]; 394 vox | -0.99 ± 1.37; -0.40 [-1.77, -0.02]; 478 vox | -0.90 ± 1.02; -0.24 [-1.52, +0.02]; 670 vox | 0.72 / 0.54 |
| V2 | 28 | +0.06 ± 1.25; -0.15 [-1.69, +0.14]; 379 vox | +0.07 ± 0.80; -0.22 [-0.60, +0.01]; 475 vox | -0.03 ± 0.66; -0.19 [-0.69, +0.09]; 687 vox | 0.64 / 0.53 |
| V3 | 28 | -4.43 ± 1.44; -2.71 [-5.91, +0.27]; 364 vox | -3.75 ± 1.23; -2.05 [-4.26, +0.15]; 462 vox | -3.02 ± 1.02; -1.36 [-3.54, +0.29]; 636 vox | 0.85 / 0.70 |
| V3AB | 25 | -0.45 ± 0.80; -0.57 [-2.31, +0.55]; 94 vox | -0.28 ± 0.75; -1.45 [-2.06, +0.37]; 167 vox | -1.05 ± 0.81; -1.11 [-1.71, -0.24]; 389 vox | 0.94 / 1.01 |
| hV4 | 26 | -4.66 ± 2.22; -2.42 [-3.81, -0.02]; 120 vox | -2.97 ± 2.41; -2.12 [-4.51, -1.13]; 186 vox | -3.24 ± 1.64; -1.81 [-3.66, -0.72]; 339 vox | 1.09 / 0.74 |
| LO | 21 | -0.50 ± 0.74; -0.76 [-1.67, +0.14]; 265 vox | -0.35 ± 0.61; -0.42 [-1.50, +0.36]; 370 vox | -0.72 ± 0.54; -0.19 [-1.50, +0.24]; 496 vox | 0.83 / 0.73 |
| TO | 21 | +14.23 ± 20.99; -1.62 [-2.97, -0.19]; 96 vox | +26.05 ± 31.37; -1.46 [-3.88, -0.33]; 178 vox | +31.34 ± 35.00; -1.08 [-2.63, -0.01]; 332 vox | 1.49 / 1.67 |
| VO | 16 | -0.28 ± 2.45; -1.06 [-1.94, +2.36]; 19 vox | +2.44 ± 3.81; -0.72 [-1.37, +0.14]; 49 vox | -1.63 ± 1.61; -1.29 [-2.01, +0.23]; 136 vox | 1.55 / 0.65 |
| IPS | 18 | +2.51 ± 4.00; -0.23 [-0.52, +0.34]; 33 vox | -1.09 ± 2.29; -0.31 [-0.57, +0.00]; 128 vox | +0.42 ± 1.18; -0.32 [-0.57, +0.30]; 422 vox | 0.57 / 0.30 |
| pooled | 28 | -1.17 ± 0.31; -0.80 [-1.68, -0.40]; 1628 vox | -1.02 ± 0.26; -0.59 [-1.18, -0.22]; 2232 vox | -0.82 ± 0.24; -0.86 [-1.31, -0.05]; 3584 vox | 0.83 / 0.77 |

**dS (m7_full)** (%, mean ± SE across participants; median [95% CI]; median voxels)

| Area | n | Current | r > 0.2 | r > 0.1 | SE ratio r0.2 / r0.1 vs current |
|---|---|---|---|---|---|
| V1 | 28 | -9.49 ± 6.01; -7.86 [-12.75, -0.32]; 394 vox | -8.40 ± 4.34; -7.93 [-12.53, +0.86]; 478 vox | -7.46 ± 3.49; -6.71 [-11.77, +0.07]; 670 vox | 0.72 / 0.58 |
| V2 | 28 | -5.88 ± 3.01; -7.71 [-11.21, -2.02]; 379 vox | -7.01 ± 2.82; -7.63 [-9.59, -2.15]; 475 vox | -6.24 ± 2.57; -7.07 [-8.84, -2.40]; 687 vox | 0.94 / 0.85 |
| V3 | 27 | -10.21 ± 3.59; -7.41 [-19.76, +3.33]; 350 vox | -9.64 ± 3.51; -7.52 [-15.55, +1.43]; 455 vox | -7.79 ± 3.07; -6.52 [-13.02, +0.69]; 622 vox | 0.98 / 0.86 |
| V3AB | 27 | +11.49 ± 10.14; -0.61 [-4.24, +1.39]; 90 vox | +4.85 ± 4.48; -0.91 [-3.62, +1.91]; 166 vox | -2.20 ± 2.27; -1.24 [-2.74, +0.76]; 364 vox | 0.44 / 0.22 |
| hV4 | 28 | -90.36 ± 54.46; -5.62 [-22.45, -3.34]; 115 vox | -53.25 ± 39.12; -5.79 [-18.44, -2.30]; 184 vox | -26.23 ± 24.23; -4.55 [-11.81, -1.06]; 331 vox | 0.72 / 0.45 |
| LO | 28 | +21.92 ± 20.67; -0.52 [-3.08, +3.21]; 216 vox | +35.62 ± 27.00; +0.42 [-1.19, +2.98]; 276 vox | +29.45 ± 25.70; -0.27 [-2.03, +1.03]; 422 vox | 1.31 / 1.24 |
| TO | 28 | -0.63 ± 25.05; -3.11 [-8.98, -0.32]; 80 vox | +2.59 ± 35.24; -1.71 [-7.51, -0.52]; 134 vox | +0.46 ± 40.56; -1.24 [-5.82, -0.01]; 272 vox | 1.41 / 1.62 |
| VO | 21 | +9.74 ± 23.32; -0.55 [-3.04, +1.72]; 16 vox | -1.45 ± 14.61; -1.29 [-3.72, -0.07]; 48 vox | -11.46 ± 7.26; -2.36 [-4.36, +0.08]; 125 vox | 0.63 / 0.31 |
| IPS | 25 | +20.70 ± 43.71; -0.16 [-39.61, +1.13]; 29 vox | +34.95 ± 40.23; -1.14 [-17.27, +7.32]; 106 vox | -4.71 ± 12.88; -0.66 [-5.04, +3.07]; 406 vox | 0.92 / 0.29 |
| pooled | 28 | -3.57 ± 1.37; -3.47 [-6.72, -1.26]; 1696 vox | -4.07 ± 1.57; -2.38 [-3.94, -0.94]; 2354 vox | -2.46 ± 1.17; -2.77 [-4.13, -0.90]; 3838 vox | 1.15 / 0.85 |

**g_LP (m7_full)** (%, mean ± SE across participants; median [95% CI]; median voxels)

| Area | n | Current | r > 0.2 | r > 0.1 | SE ratio r0.2 / r0.1 vs current |
|---|---|---|---|---|---|
| V1 | 28 | +4.15 ± 3.34; +3.06 [+0.31, +5.06]; 394 vox | +7.94 ± 5.08; +3.44 [+1.04, +6.68]; 478 vox | +6.80 ± 3.86; +3.22 [+1.50, +5.63]; 670 vox | 1.52 / 1.16 |
| V2 | 28 | -0.95 ± 1.58; +0.58 [-4.19, +3.44]; 379 vox | -1.02 ± 1.83; +1.32 [-3.46, +3.08]; 475 vox | -0.94 ± 1.73; +1.52 [-3.52, +2.91]; 687 vox | 1.16 / 1.10 |
| V3 | 27 | -6.95 ± 2.66; -0.62 [-14.33, +1.98]; 350 vox | -10.03 ± 3.24; -1.82 [-20.76, +0.37]; 455 vox | -9.17 ± 2.95; -1.93 [-17.80, +0.83]; 622 vox | 1.22 / 1.11 |
| V3AB | 27 | -58.45 ± 36.40; -21.25 [-32.67, -9.73]; 90 vox | -63.03 ± 36.31; -26.13 [-39.86, -14.21]; 166 vox | -59.61 ± 34.62; -24.83 [-36.58, -11.90]; 364 vox | 1.00 / 0.95 |
| hV4 | 28 | -7.37 ± 8.08; -10.98 [-22.41, +1.84]; 115 vox | -1.16 ± 13.57; -13.09 [-26.76, +3.75]; 184 vox | -12.30 ± 5.90; -16.03 [-25.41, +1.61]; 331 vox | 1.68 / 0.73 |
| LO | 28 | -55.44 ± 28.52; -18.78 [-30.08, -10.14]; 216 vox | -101.02 ± 48.85; -22.30 [-38.74, -13.77]; 276 vox | -65.69 ± 35.43; -19.95 [-38.68, -10.50]; 422 vox | 1.71 / 1.24 |
| TO | 28 | -126.52 ± 46.08; -66.50 [-77.81, -52.15]; 80 vox | -175.55 ± 53.21; -75.91 [-84.37, -54.30]; 134 vox | -143.86 ± 43.58; -70.36 [-79.79, -53.32]; 272 vox | 1.15 / 0.95 |
| VO | 21 | -57.76 ± 25.34; -33.40 [-81.57, -11.96]; 16 vox | -79.43 ± 37.59; -47.62 [-72.61, -14.97]; 48 vox | -42.31 ± 12.74; -33.77 [-68.57, -16.83]; 125 vox | 1.48 / 0.50 |
| IPS | 25 | -172.47 ± 64.38; -58.33 [-71.13, -23.29]; 29 vox | -239.50 ± 75.02; -59.10 [-76.64, -11.18]; 106 vox | -187.91 ± 67.43; -58.64 [-86.25, -7.71]; 406 vox | 1.17 / 1.05 |
| pooled | 28 | -13.64 ± 2.42; -11.08 [-21.27, -7.43]; 1696 vox | -19.07 ± 2.94; -16.94 [-26.03, -11.42]; 2354 vox | -17.37 ± 2.67; -16.51 [-23.72, -10.70]; 3838 vox | 1.21 / 1.10 |

**g_LP (m4_TDS)** (%, mean ± SE across participants; median [95% CI]; median voxels)

| Area | n | Current | r > 0.2 | r > 0.1 | SE ratio r0.2 / r0.1 vs current |
|---|---|---|---|---|---|
| V1 | 28 | -1.45 ± 4.52; +0.79 [-0.91, +2.17]; 394 vox | -0.44 ± 5.70; +0.87 [-0.45, +2.50]; 478 vox | -0.71 ± 4.42; +0.82 [+0.23, +2.52]; 670 vox | 1.26 / 0.98 |
| V2 | 28 | -6.34 ± 1.80; -5.00 [-7.73, +0.73]; 379 vox | -7.39 ± 2.08; -2.09 [-10.06, +0.57]; 475 vox | -6.36 ± 1.89; -1.11 [-8.83, +0.90]; 687 vox | 1.15 / 1.05 |
| V3 | 28 | -16.79 ± 2.97; -16.11 [-22.37, -7.04]; 364 vox | -20.66 ± 3.59; -21.24 [-25.89, -8.54]; 462 vox | -18.91 ± 3.34; -18.63 [-25.54, -7.26]; 636 vox | 1.21 / 1.13 |
| V3AB | 27 | -55.46 ± 36.93; -21.86 [-33.13, -15.90]; 90 vox | -60.65 ± 36.72; -27.22 [-39.70, -17.22]; 166 vox | -43.31 ± 25.23; -24.44 [-36.36, -16.67]; 364 vox | 0.99 / 0.68 |
| hV4 | 28 | -2.27 ± 20.65; -21.37 [-29.25, -16.65]; 115 vox | +4.52 ± 19.96; -26.77 [-34.63, -16.69]; 184 vox | -5.87 ± 17.91; -25.92 [-32.50, -18.19]; 331 vox | 0.97 / 0.87 |
| LO | 28 | -86.47 ± 31.38; -24.04 [-43.37, -12.98]; 216 vox | -140.36 ± 52.85; -27.92 [-52.87, -17.51]; 276 vox | -84.69 ± 36.06; -26.32 [-50.89, -14.13]; 422 vox | 1.68 / 1.15 |
| TO | 28 | -126.96 ± 40.41; -71.88 [-84.39, -61.87]; 80 vox | -213.94 ± 63.57; -81.67 [-92.99, -63.25]; 134 vox | -147.70 ± 47.57; -75.23 [-85.33, -62.44]; 272 vox | 1.57 / 1.18 |
| VO | 21 | -82.53 ± 36.08; -56.25 [-80.27, -27.53]; 16 vox | -137.32 ± 50.62; -60.57 [-83.22, -26.90]; 48 vox | -133.33 ± 57.23; -55.29 [-80.43, -26.38]; 125 vox | 1.40 / 1.59 |
| IPS | 25 | -143.72 ± 58.39; -55.94 [-70.53, -23.09]; 29 vox | -244.41 ± 78.08; -64.06 [-79.70, -34.28]; 106 vox | -197.48 ± 72.03; -58.20 [-71.51, -7.07]; 406 vox | 1.34 / 1.23 |
| pooled | 28 | -22.48 ± 2.50; -22.12 [-30.11, -17.34]; 1696 vox | -27.06 ± 2.97; -27.79 [-36.75, -19.45]; 2354 vox | -23.43 ± 2.64; -22.62 [-31.52, -18.21]; 3934 vox | 1.19 / 1.06 |

## Sources

- **Tables.**
  - `notes/data/voxel_selection_robustness_2026-11-13.tsv`: the tables above, from
    `retsupp/visualize/paper/voxel_selection_robustness.py`.
  - `notes/data/gain_field_parameters_voxsel_relnull_2026-11-13.tsv`: `collect_gain_parameters.py`, three roots.
  - `notes/data/conditional_estimates_relnull_r0.{2,1}_polished_dS.tsv`.
- **Reproduce the summary:**

  ```
  python -m retsupp.visualize.paper.voxel_selection_robustness \
    --tsv notes/data/gain_field_parameters_voxsel_relnull_2026-11-13.tsv \
    --alone current=notes/data/conditional_estimates_polished_2026-11-12.tsv \
    --alone r0.2=notes/data/conditional_estimates_relnull_r0.2_polished_dS.tsv \
    --alone r0.1=notes/data/conditional_estimates_relnull_r0.1_polished_dS.tsv \
    --out notes/data/voxel_selection_robustness_2026-11-13.tsv
  ```
