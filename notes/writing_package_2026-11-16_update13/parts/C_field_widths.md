# C. Field widths on the zero-sum fits (replaces the width statements of updates 7 and 8)

**Method.**
- For each participant × area, everything is held at the zero-sum main-model fit except the width of one field.
- That width is set to each of 17 values (FWHM 1.2–18.8°, log-spaced).
- At each width the field's gains are re-solved by least squares in the model's own parametrisation:
  - sustained field: along the zero-sum direction, HP = ¾ dS, LP = −¼ dS;
  - transient field: the target gain plus one distractor gain, equal at HP and LP.
- The fit statistics are kept per run, which gives two things:
  - **Per participant:** a leave-one-run-out SE of the profile. A width is excluded when the fit there is worse by
    more than 2 SE.
  - **Group:** the profile summed over participants, with the best width recomputed leaving out one participant at a
    time.

**Results.**

1. **The optimiser found the minimum.** At the fitted width the re-solved gains reproduce the LM fit exactly (dS:
   Spearman ρ = 1.0). The best grid width is the fitted width in almost every cell: the median difference is 0 octaves
   for the transient field and 0.05 for the sustained field, and only 4% and 10% of cells differ by more than an
   octave. Where a fitted width sits on its bound, the profile minimum is at that edge as well. A width grid before
   the LM would not change the fits.
2. **Per participant, the widths are not identified.**
   - Sustained field: 215 of 241 profiles exclude no width at 2 SE.
   - Transient field: 144 of 241 profiles exclude no width. It is better constrained in V3AB, VO, hV4 and V3 than in
     V1 and V2.
   - **Split-half check.** The best width was found separately in odd and even runs. Across participant × area
     cells the two agree poorly for the sustained field: Spearman ρ = 0.10, within one octave in 44% of cells. They
     agree moderately for the transient field: ρ = 0.58, and 0.74–0.79 in V3, V3AB and VO.
   - **Size of the effect.** Moving the sustained width from its best value to the worst on the grid costs a median
     0.01% of the training error (transient 0.09%). The sustained HP − LP modulation is a few percent of the drive, so
     the data carry little information about its shape in any one participant.
3. **At the group level, both widths are well defined** (Supplementary Fig. `figS_width_profiles_zs_n28_palA`).

| Area | Sustained FWHM [leave-one-participant-out range] | Transient FWHM [range] |
|---|---|---|
| V1 | 4.0° [4.0–4.7] | 18.8° [18.8–18.8], upper end |
| V2 | 4.7° [4.7–4.7] | 18.8° [6.7–18.8], upper end |
| V3 | 4.7° [4.0–4.7] | 18.8° [18.8–18.8], upper end |
| V3AB | 5.6° [4.0–11.2] | 1.2° [1.2–1.2], lower end |
| IPS | 2.0° [1.2–5.6] | 13.3° [11.2–15.8] |
| hV4 | 6.7° [4.7–13.3] | 1.2° [1.2–1.2], lower end |
| VO | 4.7° [2.4–5.6] | 1.2° [1.2–7.9], lower end |
| LO | 5.6° [4.7–5.6] | 6.7° [6.7–9.4] |
| TO | 4.7° [2.8–5.6] | 1.2° [1.2–5.6], lower end |

4. **dS does not depend on the sustained width.** Pooled over the nine areas, dS stays negative at every grid width:
   −1.6 to −3.8 points, 17–22 of 28 participants below zero. p < .05 from FWHM 2.0 to 11.2°, and p ≤ .066 at the
   remaining widths. At the fitted widths dS is −5.3 points, 23/28, p = .003.

**What the transient width means.** The transient field sits at opposite ends of the width range in different parts
of the hierarchy:
- **V1–V3:** maximally broad. At that width the field is nearly uniform over the display, so it acts as a gain on all
  drive while a search display is on. That is a location-independent response to the display.
- **hV4, V3AB and VO (and TO):** maximally narrow. The field then acts only at the item locations, as an item-specific
  weighting.
- **LO and IPS:** in between.

**Suggested text (Results or Supplement):**

> "The sustained field had a well-defined width at the group level: summed over participants, the fit was best at a
> FWHM of 4.0–4.7° in V1–V3 and 4.7–6.7° in ventral and lateral areas (stable when leaving out any one participant),
> narrower in early visual cortex than the 5.7° between neighbouring possible target and distractor locations. Per participant the width was not
> identified (no width excluded in 215 of 241 fits), and the HP − LP difference was negative at every width tested.
> The transient field took opposite forms along the hierarchy: in V1–V3 it was as broad as the fit allowed, acting as
> a gain on all drive while the search display was on, whereas in hV4, V3AB and VO it was as narrow as allowed,
> confined to the item locations."

**Replaces:**
- **Update 8:** "sustained FWHM 2.8° [2.7, 2.9] among fits with a reliable gain". That came from the earlier fits and
  from profiles that let g_HP and g_LP vary freely.
- **The agent-rendered `figS_field_widths_v3_zs_n28`:** not in this package, for the same reason (g_HP and g_LP
  free).

**Sources:**
- **Code:**
  - `retsupp/modeling/fit_width_profiles.py`, run on the view root `..._zs_lm_ref` into
    `derivatives/field_widths_zs_lm_2026-11-14`;
  - `retsupp/visualize/paper/width_profiles_zero_sum.py`;
  - `retsupp/visualize/paper/figS_width_profiles_zero_sum.py`.
- **Tables:** `notes/data/width_profiles_zero_sum_2026-11-14.tsv` (per cell) and `..._profiles.tsv` (per width).
