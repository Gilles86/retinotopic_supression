# Fig. 3e under looser voxel selections, and the resolution check (2026-11-12, n = 28)

**Question:** in early areas Fig. 3e shows HP < next < opposite, and the effect is significant. In the dorsal and
occipitotemporal pathways the order is the same but not significant. Is that because too few voxels sit on an item
location, or because the pRFs are too coarse to separate the locations?

**Answer: neither change helps, so Fig. 3 keeps the current selection.**
- Blurring by the pRFs does not explain the weak higher-area effect. A modulation at the HP item keeps 97–99% of
  HP − opposite and 90–92% of HP − next after pooling within a pathway.
- More voxels lower the SE (r > 0.2 by about 20%, r > 0.1 by about half), but the effect gets smaller by about the
  same amount. Effect/SE therefore does not improve, apart from a small gain at r > 0.2 that is not stable.
- The added voxels are less reliable and carry a weaker location effect. They dilute the estimate rather than
  sharpen it.
- By the rule fixed beforehand (change only for a clear improvement in precision), the main figure keeps the current
  selection. This comparison goes to the supplement.

## 1. Resolution (pRF blur), current refitted fits

Code: `retsupp/visualize/paper/item_location_blur.py`. Data: `notes/data/fig3_item_location_blur_n28_2026-11-12.tsv`.

**Method:**
- Voxels: m7_full pRFs of `..._reliable_roihrf_polished` whose centre lies within 2° of an item location.
- Each pRF is integrated over a 1.5° disk at each of the four item locations, keeping the amplitude/sd weighting.
- Results are pooled per subject and pathway as in Fig. 3, then the median over subjects is taken.
- **Leak** is the share a voxel picks up from a location other than its own.
- For a modulation confined to the HP item, HP − opposite keeps 1 − leak(opposite) and HP − next keeps
  1 − leak(next).
- Blurring alone also creates a next − opposite difference equal to leak(next) − leak(opposite).

Results with the pRF centre only:

| Pathway | Median FWHM | HP − opp kept | HP − next kept | Next − opp from blur |
|---|---|---|---|---|
| Early | 2.35° | 1.00 | 0.995 | 0.005 |
| Dorsal | 4.2° | 0.99 | 0.92 | 0.06 |
| Occipitotemporal | 4.6° | 0.97 | 0.90 | 0.07 |

- **Per area, leak to next / to opposite:**
  - V1–V3 ≤ 1.5% / ≤ 0.1%, hV4 3 / 0%, LO 6 / 1%, V3AB 8 / 1%.
  - VO 21 / 6%, IPS 33 / 14%, TO 40 / 15%. TO's FWHM is 8.7°.
- **With the full DoG pRF:** the fitted surround is nearly flat (σ of tens of degrees, peak about 6% of the centre).
  It makes every leak negative, so the contrasts become slightly sharper (1.02–1.15) rather than weaker.
- **Early next − opposite:** the measured −14.9% cannot come from blurring, which would produce at most 0.5% of the
  HP effect.

## 2. Looser selections, same pipeline

**Code:**
- `export_location_timeseries.py --root <ladder root> [--no-reliability-filter]` takes the voxels, pRF centres and the
  bar-only prediction (gains set to 0) from that root's m7_full.
- Then `deconvolve_location_responses.py --relative`.
- Then `fig3_voxel_selection.py`, run with `run_excluding --exclude 1 2`.

**Selections compared:**
- **Current:** Fig. 3 itself, `event_deconv_relative_final2_2026-11`.
- **Refit:** the current voxels with the refitted pRFs, `..._reliable_roihrf_polished`. It gives the same numbers as
  current to within 1–2% (e.g. early HP − opposite −21.5% vs −21.6%), so it is not discussed further.
- **r > 0.2 and r > 0.1:** the `..._relnull_r0.{2,1}_roihrf_polished` fits. These keep every area voxel whose
  cross-run reliability r is above 0.2 or 0.1 and p ≤ .01 against the circular-shift null, with SD ≤ 20%.
- Fits that were missing and left out: r > 0.2 sub-06 IPS, sub-25 VO and sub-26 VO; r > 0.1 sub-17 V3.

**Tables:** `notes/data/fig3_voxel_selection_n28_2026-11-12_{stats,voxels,cells,abs_values}.tsv`. Deconvolutions are in
`notes/data/event_deconv_relative_{polished,relnull_r0.2,relnull_r0.1}_2026-11-12/`. Exports are in
`derivatives/paper_2026-11/location_timeseries_{polished,relnull_r0.2,relnull_r0.1}_2026-11-12/`.

### Voxels per item location

Median over subjects of each subject's median over the four locations. The sparsest location is in brackets.

| Area | Current | r > 0.2 | r > 0.1 |
|---|---|---|---|
| V1 | 26 (7) | 36 (18.5) | 60 (39.5) |
| V2 | 25 (12.5) | 40 (22) | 67 (41.5) |
| V3 | 14 (7) | 26 (15.5) | 46 (33) |
| V3AB | 7 (1) | 13 (5) | 39 (19) |
| IPS | 3 (0) | 6 (0) | 10 (2) |
| hV4 | 6 (0) | 13.5 (2) | 24.5 (10.5) |
| VO | 3 (0) | 9 (0) | 9 (1) |
| LO | 4 (0) | 8 (1) | 20 (7) |
| TO | 3.5 (0) | 6 (1) | 13 (4) |

With r > 0.2 or r > 0.1, dorsal reaches 27–28 subjects instead of 25–26.

### Fig. 3e statistics (% of each subject's mean response; Wilcoxon signed-rank, uncorrected, as in the figure)

| Pathway | Selection | n | Location F, p | HP − opp | HP − next | Next − opp |
|---|---|---|---|---|---|---|
| Early | current | 28 | 7.67, .001 | −21.6 ± 6.8%, p = .005 | −6.7 ± 3.7%, .095 | −14.9 ± 6.0%, .022 |
| | r > 0.2 | 28 | 8.80, < .001 | −17.2 ± 4.8%, .002 | −5.7 ± 3.4%, .14 | −11.5 ± 4.2%, .012 |
| | r > 0.1 | 28 | 6.74, .002 | −14.1 ± 4.3%, .004 | −3.5 ± 3.6%, .45 | −10.6 ± 4.0%, .018 |
| Dorsal | current | 25 | 0.16, .86 | −6.7 ± 12.1%, .24 | −2.8 ± 8.5%, .41 | −3.9 ± 14.7%, .63 |
| | r > 0.2 | 27 | 2.45, .096 | −7.1 ± 6.2%, .081 | −13.9 ± 5.3%, .034 | +6.8 ± 7.1%, .49 |
| | r > 0.1 | 28 | 0.50, .61 | −1.6 ± 3.6%, .23 | −3.1 ± 2.5%, .27 | +1.5 ± 3.3%, .47 |
| Occipitotemporal | current | 28 | 1.49, .24 | −9.8 ± 6.6%, .090 | −5.9 ± 6.3%, .28 | −3.9 ± 3.8%, .21 |
| | r > 0.2 | 28 | 1.08, .35 | −5.1 ± 4.2%, .045 | −0.6 ± 4.2%, .27 | −4.5 ± 2.9%, .070 |
| | r > 0.1 | 28 | 1.04, .36 | −4.1 ± 3.5%, .13 | −1.7 ± 2.8%, .27 | −2.5 ± 2.3%, .17 |
| All areas | current | 28 | 6.31, .003 | −15.5 ± 5.3%, .013 | −6.9 ± 2.7%, .018 | −8.6 ± 4.6%, .060 |
| | r > 0.2 | 28 | 7.58, .001 | −11.9 ± 3.7%, .007 | −5.5 ± 2.2%, .015 | −6.4 ± 3.0%, .038 |
| | r > 0.1 | 28 | 5.06, .010 | −7.2 ± 2.8%, .015 | −2.6 ± 1.7%, .12 | −4.6 ± 2.3%, .040 |

The location type × item type interaction is p ≥ .12 in every cell, so item types are pooled in every selection.

### Precision: absolute units (% signal; no division by the subject's mean, so selections are comparable)

HP − opposite, mean ± SEM, with effect/SE in brackets:

| Pathway | Current | r > 0.2 | r > 0.1 |
|---|---|---|---|
| Early | −0.092 ± 0.032 (2.8) | −0.079 ± 0.026 (3.1) | −0.050 ± 0.017 (3.0) |
| Dorsal | −0.086 ± 0.062 (1.4) | −0.088 ± 0.047 (1.9) | −0.015 ± 0.027 (0.6) |
| Occipitotemporal | −0.045 ± 0.038 (1.2) | −0.029 ± 0.035 (0.8) | −0.018 ± 0.021 (0.8) |
| All areas | −0.083 ± 0.032 (2.6) | −0.065 ± 0.024 (2.7) | −0.034 ± 0.014 (2.4) |

HP − next, effect/SE for current / r > 0.2 / r > 0.1:
- early 2.2 / 2.1 / 1.1;
- dorsal 1.7 / 2.4 / 1.4;
- occipitotemporal 0.8 / −0.1 / 0.3.

The mean response over the four locations barely changes at r > 0.2 (early 0.46 → 0.46% signal). At r > 0.1 it falls
(0.39; occipitotemporal 0.81 → 0.66). So the added voxels respond less to the items, and their location effect is
weaker still.

## Verdict

1. **Resolution is not the limit for HP − opposite.** With pooling within a pathway, the pRFs keep at least 97% of it
   in every pathway. Only HP − next in single higher areas (TO, IPS, VO) is noticeably blurred.
2. **More voxels do not help.**
   - A looser selection lowers the SE.
   - But it shrinks the effect in proportion: early HP − opposite falls from −0.092 to −0.050% signal (−21.6 to
     −14.1%) at r > 0.1.
   - Effect/SE stays flat in early areas and all areas (2.4–3.1 throughout).
   - In the higher pathways it does not rise consistently: dorsal 1.4 → 1.9 → 0.6; occipitotemporal 1.2 → 0.8 → 0.8.
   - The only nominal crossings are at r > 0.2: dorsal HP − next p = .034 and occipitotemporal HP − opposite
     p = .045. Neither survives at r > 0.1, and the dorsal location ANOVA stays non-significant (p = .096).
   - Low-reliability voxels carry a smaller location effect, plausibly because their pRF centres, and so their
     assignment to an item, are less accurate. This is a localisation limit in the data, not a blurring limit.
3. **The direction is robust.** HP − opposite is negative in every pathway under every selection (12 of 12). The
   early pattern HP < next < opposite and its significance hold under both looser selections (HP − opposite
   p ≤ .005; next − opposite p ≤ .022).
4. **Decision under the fixed rule:** Fig. 3 keeps the current selection. The higher-pathway trend should be described
   as a trend in the same direction that a looser selection does not make reliable. The supplement gets
   `notes/figures/paper_2026-11/figS_item_locations_voxel_selection_n28.{pdf,png,md}`:
   - a, voxel counts;
   - b, blur between locations;
   - c, d, HP − opposite and HP − next per pathway for the three selections.

Suggested sentence for the paper: "With looser, model-free voxel selections (cross-run reliability r > 0.2 or > 0.1,
up to twice to six times as many voxels per item location), early visual cortex showed the same ordering (HP − opposite
p ≤ .005), and HP lay below opposite in every pathway. The dorsal and occipitotemporal effects did not become reliable,
because the added voxels carried a proportionally weaker effect. pRF blur between item locations, estimated from the
fitted pRFs, retained ≥ 97% of the HP − opposite contrast in every pathway."
