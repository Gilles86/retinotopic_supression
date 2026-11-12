# Fig. 3d, e robustness: absolute units, medians, voxel floor, leave-one-out (2026-11-12, n = 28)

Why: the Fig. 3 measure divides each participant's location responses by their mean response in the area (%). Where that
mean is small (TO, VO, IPS) the values blow up (TO HP − opposite: mean −112%, median −25%) and pairs with a mean ≤ 0 are
dropped.

Code `retsupp/visualize/paper/fig3_robustness.py`; data `notes/data/fig3_robustness_n28_2026-11-12.tsv` (all variants ×
poolings × units), `notes/data/fig3_robustness_values_n28_2026-11-12.tsv` (per participant × area, relative and absolute).
Same deconvolutions (`notes/data/event_deconv_relative_final2_2026-11`) and window (3.2–6.4 s) as Fig. 3; items pooled;
sub-03–30. Statistics: median [bootstrapped 95% CI, 10 000], k of n below 0, Wilcoxon p; Page's L for HP < next < opposite;
BH q across nine areas (per area) or three pathways. Absolute units are % signal change.

## Comparison (HP − opposite; Page = ordered test)

| Variant | Unit | n | HP − opposite median [95% CI] | k/n < 0 | p (q) | Page p (q) | LOO max p (Wilcoxon / Page) |
|---|---|---|---|---|---|---|---|
| Relative (Fig. 3) | V1 | 28 | −29.8% [−61.2, −12.9] | 21/28 | .0017 (.015) | .0038 (.017) | .003 / .007 |
| | V2 | 28 | −20.0% [−41.9, −3.9] | 20/28 | .010 (.031) | .0016 (.015) | .019 / .003 |
| | TO | 21 | −25.0% [−65.3, −11.7] | 16/21 | .010 (.031) | .0068 (.020) | .019 / .016 |
| | Early (voxel-weighted) | 28 | −21.2% [−32.3, −10.0] | 20/28 | .0052 (.016) | .0016 (.005) | .010 / .003 |
| | All areas (voxel-weighted) | 28 | −19.2% [−26.9, −11.1] | 23/28 | .013 | .0001 | .023 / .0001 |
| **Absolute** | V1 | 28 | −0.081% [−0.128, −0.053] | 21/28 | .0065 (.030) | .0038 (.017) | .012 / .007 |
| | V2 | 28 | −0.064% [−0.162, −0.025] | 20/28 | .0037 (.030) | .0016 (.015) | .007 / .003 |
| | TO | 23 | −0.131% [−0.157, −0.061] | 17/23 | .014 (.041) | .014 (.041) | .025 / .025 |
| | Early (voxel-weighted) | 28 | −0.106% [−0.133, −0.057] | 20/28 | .0048 (.014) | .0016 (.005) | .009 / .003 |
| | All areas (voxel-weighted) | 28 | −0.098% [−0.127, −0.071] | 23/28 | .0076 | .0001 | .014 / .0001 |
| Absolute, ≥ 5 voxels per location | V1 | 20 | −0.078% [−0.118, −0.000] | 14/20 | .070 (.21) | .048 (.097) | |
| | V2 | 25 | −0.070% [−0.167, −0.044] | 19/25 | .0007 (.004) | .0006 (.003) | .001 / .001 |
| | Early (voxel-weighted) | 26 | −0.113% [−0.136, −0.048] | 20/26 | .0013 (.004) | .0011 (.003) | .003 / .002 |
| | All areas (voxel-weighted) | 28 | −0.113% [−0.144, −0.069] | 22/28 | .0024 | .0007 | .005 / .001 |
| Absolute, ≥ 10 voxels per location | V2 | 18 | −0.055% [−0.161, +0.010] | 13/18 | .024 (.071) | .012 (.035) | .045 / .024 |
| | Early (voxel-weighted) | 23 | −0.078% [−0.130, −0.007] | 17/23 | .012 (.012) | .006 (.006) | .023 / .012 |

**Next − opposite** (the neighbour effect): relative, early voxel-weighted −12.1%, p = .022 (q = .065 over three
pathways); absolute, early −0.044%, p = .038 (q = .11); no single area has q < .05 in any variant. HP − next: no area
q < .05 in any variant (absolute TO p = .016, q = .15).

**Pooling.** Voxel-weighted (as Fig. 3) and the per-participant mean over areas give the same picture. The
per-participant **median over areas** is weaker for all areas (absolute HP − opposite p = .082, Page p = .008; LOO max p
= .13): with three to nine areas per participant, the median throws away most of each participant's data.

**Voxel counts.** Selected voxels at a participant's sparsest item location, median per area: V1 8, V2 12, V3 7, V3AB 3,
IPS 1, hV4 2, VO 2, LO 2, TO 2. With every location ≥ 5 voxels: V1 20, V2 25, V3 16, V3AB 7, hV4 7, LO 6, TO 4, VO 2,
IPS 0 participants; ≥ 10: V1 14, V2 18, V3 6, V3AB 3, LO 1, others 0. So the higher-area single-area values (incl. TO)
rest on one or two voxels per location; a voxel floor makes them untestable. Early visual cortex survives both floors.

**Leave-one-participant-out** (V1, V2, TO, early, all areas; Wilcoxon on HP − opposite and Page's L): no single
participant decides significance in either unit; the largest p after any one drop is ≤ .025 throughout (absolute TO
.025; relative all areas .023).

## Verdict

- **The story holds in absolute units.** HP < opposite in V1, V2 and TO (q ≤ .041) and in early visual cortex
  (−0.11% signal change, 20/28, p = .005; Page p = .0016) — the same areas as the relative measure, with no participant
  excluded and no single participant driving it. Pooled over all areas: −0.10%, 23/28, Page p = .0001.
- **The neighbour effect (next < opposite) is weaker in absolute units** (early p = .038, q = .11; relative p = .022) and is
  not reliable in any single area in either unit. Present it as a trend that the ordered test supports, not as a separate
  finding.
- **TO** is significant in both units, but its values rest on a median of two voxels per location; with a voxel floor it
  cannot be tested. Mention it only with that caveat; the inference should stay at the pathway level.
- **Measure for the paper:** absolute units (% signal change, relative to each participant's mean over the four
  locations) for the per-area supplement and the robustness statement — no blow-up, no exclusions, same conclusions.
  The relative measure can stay in main Fig. 3e for comparability with Richter et al., with one sentence that absolute
  units give the same result (early HP − opposite −0.11% signal change, p = .005; Page p = .0016). Report medians with
  bootstrapped CIs rather than means ± SEM for single areas.

Figure: `notes/figures/paper_2026-11/figS_item_locations_per_area_abs_n28.{pdf,png,md}` (one wide panel, nine areas, shared
y-axis, participant dots clipped at ±0.45% as triangles, median and bootstrapped CI, stars from Page q).
