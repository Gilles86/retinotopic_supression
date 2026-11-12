# pRFs refitted together with the gain fields (polish), 2026-11-12

Final DoG fits: `af_prf_cv_ladder_nested_full_bounded_maxsd20_reliable_roihrf` (CV_SCHEME=full, BOUNDED, MAX_BOLD_SD=20,
RELIABLE_ONLY, per-ROI HRF from `paper_2026-11/roi_hrf_dog_final2.tsv`), nine areas, subjects 3–30 (n = 28).
Polished fits: `af_prf_cv_ladder_nested_full_bounded_maxsd20_reliable_roihrf_polished/{m4_TDS,m7_full}`, 241 of 252
cells per model (the 11 missing cells have no reliable voxels and have no clamped fit either).

## What was fitted

- The clamped fits keep the per-voxel pRFs (x, y, sd, baseline, amplitude, surround amplitude/size) at the m0 fit and
  fit only the shared gain-field terms. The polish frees every per-voxel pRF parameter together with the shared terms,
  starting from the model's own clamped optimum. m4_TDS and m7_full were polished separately (independent tasks), each
  from its own clamped optimum. m00/m0 are unchanged; the base models m0–m3b were not refitted (user decision).
- **HRF**: fixed at the per-ROI table HRF; the polish does not free it.
- **Method** (`--polish-method alt`, `fit_cv_ladder_nested.py`): two rounds of (Adam on the per-voxel pRFs with the
  shared terms fixed, then bounded L-BFGS on the shared terms with the pRFs fixed); each step kept only if training SSE
  drops. The older one-shot L-BFGS over all parameters (`polish_fit`) stalled on float32 gradient noise in some cells
  (sub-03 TO: "ABNORMAL", no step accepted) and took > 30 min in large areas.
- Cost: CPU, 3–4 CPUs, peak 22 GB; 6 min data loading + 10–45 min polish per subject × area × model; 504 tasks plus
  131 resubmitted after a 50-min limit (3 h limit). Not converged after two rounds (SSE still falling slightly).

## Fit and pRF change

- Training SSE: lower in every cell (median −0.03%, smallest −0.002%); never above clamped.
- pRF centres move little: median shift 0.009° (90th pct 0.07°); for voxels with a centre within 2° of an item
  location 0.04° (90th pct 0.11°). sd +0.4%, amplitude +1.2–1.9% (median relative change).
- Data: `notes/data/prf_refit_2026-11-12_shared.tsv`, `notes/data/prf_refit_2026-11-12_prf_shift.tsv`.

## Estimates: clamped vs polished (median over n = 28, %; * BH q < .05 over nine areas; pooled = per-subject median over areas)

Which fit: "joint" = m7_full (all terms fitted together); "alone" = the term added alone to m4_TDS (dS, dD, dT only),
linear sufficient statistics (`fit_fixed_group_mechanisms --overlay-per-model`, polished m4_TDS base). The clamped
"alone" run (`af_fixed_group_mechanisms_clampedcheck_2026-11-12`) reproduces the paper's numbers exactly.

| Term (fit) | | V1 | V2 | V3 | V3AB | hV4 | LO | TO | VO | IPS | Pooled |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dS (alone) | clamped | −0.7 | −0.4 | −3.0* | −0.6 | −1.7 | −0.6 | −1.9* | −1.4 | −0.4 | −1.4 (p = 1e-4) |
| | polished | −0.7 | −0.1 | −2.7* | −0.6 | −2.7* | −0.7 | −1.9 (q = .065) | −1.2 | −0.5 | −1.3 (p = 2e-4) |
| dS (joint) | clamped | −7.1 | −7.3 | −9.0* | −1.0 | −5.7* | −0.6 | −2.4* | −0.6 | −1.1 | −3.4 (p = .0027) |
| | polished | −7.9 | −7.7 | −9.5* | −0.6 | −5.6* | −0.5 | −3.1 (q = .084) | −0.6 | −0.2 | −3.5 (p = .0044) |
| g_LP (joint) | clamped | +2.8 | +1.3 | −0.4 | −11* | −5.8 | −8.7* | −35* | −15* | −33* | −6.7 |
| | polished | +3.1 | +0.6 | −0.4 | −21* | −11 | −19* | −67* | −33* | −58* | −11 |
| Target (joint) | clamped | −1.8 | +4.8 | +30* | +33* | +56* | +18 | +22* | +50* | +21* | +16 |
| | polished | −2.3 | +5.2 | +33* | +30* | +59* | +20 | +18* | +38* | +21* | +17 |
| Distractor (joint, avg over locations) | clamped | −0.7 | +4.3 | +3.4 | +12* | +22* | +1.6 | +8.4* | +7.2 | +2.6 | +6.5 (p = .023) |
| | polished | −0.7 | +4.3 | +6.4 | +11* | +21* | +2.2 | +8.2* | +8.0 | +1.9 | +8.0 (p = .023) |
| dD (alone) | clamped | +4.5 | −0.5 | −1.4 | −0.7 | −2.4 | −3.0 | −0.6 | −0.8 | −0.0 | −1.0 (p = .12) |
| | polished | +4.5 | −0.5 | −1.7 | −0.5 | −2.5 | −3.1 | −0.7 | −0.6 | −0.2 | −0.8 (p = .16) |
| dT (alone) | clamped | −4.7 | −2.7 | −8.0 | −5.8 | −3.0 | −5.0 | −5.7 | −1.1 | −0.4 | −3.4 (p = .017) |
| | polished | −4.6 | −2.6 | −7.8 | −5.6 | −2.6 | −4.0 | −4.6 | −1.5 | −0.4 | −3.6 (p = .040) |

Items (joint item weight × 100) rise in the higher areas (TO 459 → 600, IPS 1837 → 2179; pooled 308 → 339).

Paired polished − clamped (pooled, Wilcoxon): dS alone −0.004 pts (p = .65); dS joint −0.18 pts (p = .025); g_LP
joint −5.7 pts (p < 1e-4); g_HP −6.6 pts; target −0.4 pts (p = .005); distractor +0.03 pts (p = .57).

Full table: `notes/data/prf_refit_group_2026-11-12.tsv` (script `retsupp/visualize/paper/compare_polished_fits.py`).

## What changes for the paper

- **dS (the key claim) holds.** Pooled dS estimated alone −1.3% (79% < 0, p = 2e-4) vs −1.4%; joint −3.5% vs −3.4%.
  Per area: V3 stays significant; hV4 becomes significant estimated alone (−2.7%, q = .034); TO drops just above
  threshold in both (alone q = .065, joint q = .084) with the same median (−1.9%).
- **Target and distractor unchanged** in direction and significance (VO target 50 → 38%, still *).
- **The sustained gain shared by all four locations changes in size**: g_LP (and g_HP) become about twice as negative
  in the higher areas (TO −35 → −67%, IPS −33 → −58%), traded against higher item weight and pRF amplitude. The shared
  sustained gain is poorly separated from the pRF amplitude and item weight once the pRFs are free; quote its size with
  care (sign and per-area significance unchanged except hV4, n.s. in both).
- Figures that would change if redrawn from the polished fits: Fig. 5a maps (sustained LP/HP levels in hV4, TO and other
  higher areas), Fig. 5c per-area stars (hV4 gains one, TO loses one), the parameter table. Not regenerated yet.
