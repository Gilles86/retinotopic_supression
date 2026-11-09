# Gain fields that act on DN parameters (2026-11-09)

Question (Tomas Knapen, via the user): does the HP effect live in specific divisive-normalization (DN)
parameters rather than in the stimulus drive? Per-HP-condition DN fits (`retsupp/modeling/fit_dn_by_hp.py`,
2 of 30 subjects) hint that b (neural baseline) is higher for pRFs in the quadrant facing HP and d
(surround baseline) higher opposite HP, with amplitudes unchanged. Here: a gain-field version of that idea.

**One field per subject × ROI in every variant.** As in the current ladder, the sustained field has only
shared parameters (σ_AF, the HP/LP pattern, the weights), fitted on all voxels of the ROI at once. Each voxel's
modulation is that field evaluated over its own (frozen) pRF. No per-voxel field parameters.

## (c) Reference: the current DN gain field (m7_full, `dn_attention_field.py`)

The field multiplies the stimulus before both centre and surround pooling (Reynolds & Heeger style):

    M_c(g)    = max(1 + Σ_ℓ w_cℓ A_ℓ(g; σ_AF), 0),   w_cℓ = g_HP if ℓ is HP of condition c, else g_LP
    S_eff(t,g)= S(t,g) · β(g) · max(M_c(t)(g) + P(t,g), 0)          P: transient distractor + target fields
    N(t)      = (a⟨S_eff, G_c⟩ + b) / (c|a|⟨S_eff, G_s⟩ + d) − b/d
    BOLD(t)   = HRF * N(t) + bold_baseline

A_ℓ: Gaussian of width σ_AF at ring position ℓ (peak 1); β: item drive scale; G_c, G_s: centre and surround
Gaussians (σ_s = srf_size·σ_c). Per voxel: x, y, σ_c, a, c, srf_size, b, d, bold_baseline (frozen at the m0 fit).
Shared: σ_AF, g_HP, g_LP, σ_dyn, g_HP_dyn, g_LP_dyn, g_T_dyn, β (8).

## Baseline-field variants (`dn_baseline_field.py`)

A second sustained field, same width σ_AF, with its own HP/LP pattern, scales b and/or d per HP condition:

    Φ_c(g) = Σ_ℓ u_cℓ A_ℓ(g; σ_AF),    u_cℓ = cos θ (ℓ = HP of c), sin θ (ℓ ≠ HP)
    F_c(v) = Σ_g Ĝ_v(g) Φ_c(g)          (field averaged over the voxel's centre pRF, Ĝ_v sums to 1)
    b_c(v) = b_v · exp(w_b F_c(v)),     d_c(v) = d_v · exp(w_d F_c(v))
    N(t)   = (a⟨S_eff, G_c⟩ + b_c) / (c|a|⟨S_eff, G_s⟩ + d_c) − b_c/d_c

Everything else (input field, transient terms) is m7's. θ sets the pattern: 0 = only at HP, π/2 = only at the
three LP locations, π/4 = equal at all four, −π/4 = HP against LP. HP-specific effect on b: w_b(cos θ − sin θ);
same for d. exp(wF) ≈ 1 + wF for small effects and keeps both positive. The condition-specific b_c/d_c is
subtracted, so a blank screen still gives N = 0 (see predictions: with per-run PSC data a change in the blank
level cannot be measured anyway).

| Model | Field acts on | New shared terms | Shared total |
|---|---|---|---|
| (c) m7_full | stimulus, numerator and denominator | — | 8 |
| (a) m8_full_dnb | neural baseline b | w_b, θ | 10 |
| (b) m8_full_dnd | surround baseline d | w_d, θ | 10 |
| (d) m8_full_dnbd | b and d, one shape, two weights | w_b, w_d, θ | 11 |

(a) and (b) are (d) with w_d = 0 and w_b = 0, so all three are nested in (d) and contain m7 (w = 0).

**How (d) differs from (c).** Input gain multiplies the stimulus pixel by pixel, before pooling: a pRF whose
*surround* covers HP is less normalized when HP is suppressed, although its centre is far from HP. The baseline
field is a per-voxel scalar taken over the pRF *centre*: it changes how strongly the voxel's own drive is
normalized, not what reaches the pool. Algebra: scaling b and d by the same k is identical to dividing a by k
(a, b, d share one free scale), so the w_b = w_d direction of (d) is an amplitude field, close to an input gain
constant over the pRF. The new, DN-specific direction is w_b − w_d: the b/d ratio. Expect w_b + w_d to trade
off against g_HP/g_LP, and w_b − w_d to be the separable part.

**Identifiability.** Within a shared fit the per-voxel b, d, a, bold_baseline are frozen, so the only possible
trade-offs are among the shared terms: w_b vs w_d (both mainly act through b/d when b/d is large, which it is:
median b ≈ 0.5–3, d ≈ 0.01–0.03 in sub-03), and w_b + w_d vs the input gains. Each fit stores a Gauss-Newton
Hessian JᵀJ of the SSE over the shared terms (`optimizer_info[k]['hessian_corr']`); correlations of its inverse
near ±1 mean the terms are not separable. θ has no curvature where both weights are 0. Canary result below: w_b and
w_d move together, along the amplitude direction, and trade off against the input gain.

**Fitting.** From the finished m7_full fold fit: per-voxel pRFs frozen, all shared terms refitted (L-BFGS-B,
bounded; w ∈ [−5, 5]) with the new weights at 0. Starts: θ = 0 with the five σ starts of the ladder, plus
θ = π/4, −π/4, π/2 at the parent σ's. (d) starts from (a)'s optimum and from (b)'s, so its training SSE is ≤ both.
Before fitting, the start is checked to reproduce the parent's prediction (max |Δpred| must be ~0; it is 0.0
exactly in a unit test and in the canary). Code: `retsupp/modeling/dn_baseline_field.py`,
`fit_cv_ladder_nested.py --family dn --models m8_full_dnb m8_full_dnd m8_full_dnbd`.

Not implemented (cheap to add if (d) is null): a field on the normalization *drive* only (the stimulus in the
denominator, not the numerator), and a d-field pooled over the surround Gaussian instead of the centre.

## Predictions, and how to read them in the data

Simulation of the DN model with each field at HP only, synthetic voxels on the four diagonals at 1–3°, DN
parameters in the range of the m7 fits, bars in four directions, items always on
(`retsupp/modeling/sim_dn_field_signatures.py`, table `notes/data/dn_field_signatures_2026-11.tsv`).
Change in the neural response, in % of the voxel's own bar-at-centre response, median over voxels:

| Field at HP | Bar class | HP quadrant | next to HP | opposite | next − opposite |
|---|---|---:|---:|---:|---:|
| (c) input gain, g_HP = −0.3 | bar on centre | −0.6 | +0.5 | +0.1 | **+0.4** |
| | bar in surround | −0.6 | +0.2 | +0.1 | +0.2 |
| | items only | −1.2 | +0.1 | 0.0 | +0.1 |
| DoG, same input field | bar on centre | −12.1 | −0.4 | 0.0 | −0.5 |
| (a) b-field, w_b = +1 | bar on centre | −5.2 | −0.3 | 0.0 | −0.3 |
| | bar in surround | −5.2 | −0.3 | 0.0 | −0.3 |
| (b) d-field, w_d = +1 | bar on centre | +2.0 | +0.2 | 0.0 | +0.2 |
| | bar in surround | +3.6 | +0.3 | 0.0 | +0.3 |

- **(c) input gain** — suppression at HP lowers responses in the HP quadrant and, through DN's suppressive
  surround, *raises* them next to HP (their surround covers the HP items, which now drive the pool less); the
  opposite quadrant is too far away to gain much. So DN with an input field at HP alone gives
  HP < opposite < next, the order seen in the decoded bars (next − opposite = +0.4 % vs HP −0.6 %). A DoG with
  the same field (subtractive surround of half the centre's volume) gives HP < next < opposite instead: the
  field's tail reaches into the neighbouring quadrants' centres and its surround release is too weak. The release grows with eccentricity of the HP-side voxels
  and with surround size up to ~6°; with very large surrounds (srf_size 12) next and opposite both gain ~3.5 %).
  Signature: an effect in voxels whose centre is far from HP, scaling with how much of their surround covers HP.
  Note the size: the same g_HP lowers HP-quadrant centre responses by 12 % in the DoG but 0.6 % in DN, because
  under DN an input gain scales numerator and denominator together and is largely normalized away. This could
  make dS weakly identified under DN, in line with the LOSO result (DoG p = .010, DN p = .23).
- **(a) b-field** — local: only voxels whose centre pRF overlaps the field change, and nearly by the same
  fraction for every bar position (raising b subtracts a term ∝ the normalization drive). No neighbour effect.
  Signature: a uniform downscaling of the HP-quadrant voxels' bar-position tuning, no change in its shape.
- **(b) d-field** — also local, but stronger when the bar is in the surround than on the centre (+3.6 vs +2.0 %):
  it changes compression. Signature: the ratio surround-response / centre-response changes at HP.
- **Blank (no bar, no items)**: zero by construction in all variants. Runs are PSC-demeaned per run
  (`clean.py`), so a per-run baseline shift — the classic "baseline" signature of (a) — is not measurable;
  only changes in stimulus-evoked responses are. Items-only TRs are not blank (the items drive the pool).

The per-condition fits (b higher facing HP, d higher opposite) would correspond to w_b > 0 and w_d < 0 with
θ ≈ 0: both raise b/d at HP, i.e. both suppress the HP quadrant, in the same direction as (a) above.

What would distinguish the variants:
1. Training SSE gain over m7 (all three nest m7) and, once a CV DN root exists, held-out R² (run3 scheme).
2. Whether the HP effect moves out of the input gains: dS = g_HP − g_LP in (a)/(b)/(d) vs m7. If the baseline
   field takes over and dS → 0, the HP effect is better described as a DN-parameter change.
3. Whether the fitted signs reproduce the per-condition fits (w_b > 0, w_d < 0, θ near 0), and whether w_b and
   w_d are separable (Hessian correlations).
4. Group test as for dS: HP-specific baseline effect w(cos θ − sin θ) against the HP-rotation null.

## Canary and job

Parent root (prototype): `derivatives/af_prf_cv_ladder_nested_full_bounded_maxsd20_dn` (all runs, no CV, fixed
canonical HRF, bounded, BOLD SD ≤ 20 %; fitted 2026-11-03, m7_full for 30 × 11 ROIs). **The final run has to wait for
the DN ladder with a per-ROI HRF** (DN's own HRF fit is pending); these fits are a prototype.

**Canary** (job 8203617, sub-03 × nine ROIs, all three variants, lowprio, 8 CPUs): 69 min, peak RSS 22.5 GB,
7.6 CPU-h. Equality check: max |Δpred| = 0 exactly in every ROI. Root:
`derivatives/af_prf_cv_ladder_nested_full_bounded_maxsd20_dn_basefield_canary`.

| ROI | ΔSSE b-only | ΔSSE d-only | ΔSSE both (rel.) | w_b, w_d (both) | dS m7 → both |
|---|---:|---:|---:|---|---|
| V1 | 0.2 | 0.3 | 25 (5e-6) | 0.060, 0.059 | −0.20 → −0.01 |
| V2 | 45 | 53 | 156 (3e-5) | −0.161, −0.163 | 0.15 → 0.11 |
| V3 | 50 | 66 | 2300 (4e-4) | 0.655, 0.656 | 0.04 → 0.68 |
| V3AB | 30 | 26 | 179 (1e-4) | −0.311, −0.308 | 0.03 → 0.00 |
| IPS | 9 | 6 | 801 (2e-4) | 0.783, 0.770 | −0.09 → 0.41 |
| hV4 | 20 | 25 | 267 (1e-4) | −0.329, −0.336 | −0.01 → 0.25 |
| VO | 24 | 18 | 409 (3e-4) | −3.41, −3.40 | −0.04 → −0.01 |
| LO | 2 | 8 | 344 (2e-4) | −0.871, −0.869 | 0.17 → −0.01 |
| TO | 9 | 6 | 322 (2e-4) | −0.701, −0.701 | 0.09 → 0.06 |

(ΔSSE = training SSE of m7 minus the variant; SSE ~1–6·10⁶.) Reading:
- **b alone or d alone does nothing**: |w| < 0.01, SSE gain ≤ 1e-5 relative, dS unchanged. The DN-specific b/d
  direction is not used by the data in this subject.
- **b and d together always move along w_b = w_d** (equal to 2–3 digits in all nine ROIs), i.e. the amplitude
  direction (scaling b and d together = dividing a). That is a gain change taken over the pRF centre, and it
  **trades off against the input gain**: dS moves by up to 0.6 (V3, IPS, hV4) while w_b − w_d stays ≈ 0. So w_b and
  w_d are not separable as such (corr ≈ +1 in the finite-difference Hessian of the canary, which was too noisy to
  quote; the full job stores a Gauss-Newton Hessian), and the joint variant's HP effect is not identified apart
  from the input gain. Expect the same in the full set; the recovery simulation tests it.

## Jobs (submitted 2026-11-09, account hare.econ.uzh, lowprio, outputs in new roots, nothing overwritten)

| Job | What | Tasks | Resources | Est. core-h |
|---|---|---|---|---|
| 8203960 | all-runs fits of dnb, dnd, dnbd on the DN root, 30 subjects × 9 ROIs, per subject (`..._dn_basefield_2026-11`) | 30 | 6 CPU, 32 GB, 3 h | ~500 |
| 8206646 | resume of 8203960 (afterany; tasks hit the 3 h limit at ~2/3, finished models are loaded) | 30 | 6 CPU, 32 GB, 4 h | ~150 |
| 8206793 | held-out test: run3 folds, pRFs frozen at the all-runs fit, m7 + variants refitted per fold (`af_prf_cv_ladder_nested_run3_frozenprf_bounded_maxsd20_dn_basefield_2026-11`) | 30 | 6 CPU, 32 GB, 6 h | ~500 |
| 8206356, 8206625 | recovery: sub-03, 08, 13, 23 × V1, V3, V3AB, hV4; truths none/input/b/d at 1× and 4× the fitted dS; 2 noise draws (`dn_baseline_field_recovery_2026-11`) | 16 | 6 CPU, 32 GB, 5 h | ~150 |

Submit lines (from `~/retsupp_plan`):

    CV_SCHEME=full BOUNDED=1 MAX_BOLD_SD=20 FAMILY=dn PER_SUBJECT=1 ROI_LIST="V1 V2 V3 V3AB IPS hV4 VO LO TO" \
      MODELS="m8_full_dnb m8_full_dnd m8_full_dnbd" BASE_ROOT=$D/af_prf_cv_ladder_nested_full_bounded_maxsd20_dn \
      OUTPUT_ROOT=$D/af_prf_cv_ladder_nested_full_bounded_maxsd20_dn_basefield_2026-11 \
      sbatch --account=hare.econ.uzh --partition=lowprio --cpus-per-task=6 --mem=32G --time=03:00:00 --array=1-30 \
      retsupp/modeling/slurm_jobs/fit_cv_ladder_nested.sh
    # held-out: same + CV_SCHEME=run3 FROZEN_PRF=1 SIGMA_STARTS=parent, OUTPUT_ROOT=..._run3_frozenprf_..._2026-11, --time=06:00:00
    sbatch --array=1-16 retsupp/modeling/slurm_jobs/recover_dn_baseline_field.sh

($D = /shares/zne.uzh/gdehol/ds-retsupp/derivatives.) Held-out choice: no DN CV root exists (only the all-runs
DN ladder), so the cheapest valid test keeps every voxel's pRF at the all-runs fit (the same in all models, so its
leak into the held-out runs does not favour any model) and refits only the shared field terms per run3 fold (one run
of each HP condition held out). Collect: `collect_dn_baseline_field --root/--cv-root`; figure:
`retsupp/visualize/paper/figS_dn_gain_locus.py`.

## Update 14:50: all-runs fits done (30 subjects × 9 ROIs)

Job 8203960 finished within its limit (≤ 2 h 28, peak RSS 29 GB, ~11 CPU-h per task), so the resume job 8206646 was
cancelled. Table: `notes/data/dn_basefield_full_2026-11.tsv` (`collect_dn_baseline_field`, also m7_full). The canary
generalises: b-only and d-only barely move (median |w| 0.004, training-SSE gain 4e-6 relative); the joint model moves
along w_b = w_d (median |w_b − w_d| 0.003 vs median |w_amp| 0.21), Gauss-Newton corr(w_b, w_d) median 0.999, while
the rotated pair is separable (corr(w_amp, w_ratio) 0.13). The amplitude direction trades off with the input gain
(corr(w_amp, g_HP) 0.57; dS of the joint model vs m7 across subject × ROI: r = 0.65, median |change| 0.05). The b/d
(DN-specific) direction is not used. Held-out comparison (8206793) and recovery (8206356/8206625) still running.

## Update 19:00: held-out comparison and recovery

**Held-out** (run3, pRFs frozen, `notes/data/dn_basefield_cv_2026-11.tsv`; 1067 of 1080 fits, subjects 10, 15 and 19
partly, resubmitted with 64 GB as 8216951 after OOM at 32 GB): every variant predicts held-out runs worse than the
input-gain model in all 30 subjects (Wilcoxon p = 2e-9 each): mean ΔR² b-field −3.4e-5, d-field −3.4e-5, both
−1.0e-4, negative in every ROI.

**Recovery** (`notes/data/dn_basefield_recovery_2026-11.tsv`, 16 units × 2 noise draws): at the fitted effect size,
b and d truths are selected as "input" in 72 % of simulations (b correct 19 %, d correct 12 %); at 4× they are
detected (input 16-19 %) but b and d are confused (b truth: 44 % b, 38 % d; d truth: 53 % d, 31 % b). A b or d
effect leaks into the input-gain dS (median 0.3-0.4× at 1×, ~0.9× of the 1× input dS at 4×). Input and none truths:
94 % and 97 % stay input.
