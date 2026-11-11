# Update 3 to writing package 2026-11-12 (n = 28)

One new supplementary figure (item D11 of the update-2 request); nothing else changes.

| File | Status | What it is |
|---|---|---|
| `figures/figS_decoded_map_model_vs_measured_n28.{pdf,png,md}` | new | The decoded bar map of Fig. 6c (a), the same decoder on the voxels the joint gain-field fit has (b, on average 95% of the decoder's voxels), and the same decoder applied to the joint fit's noise-free prediction (c). Same colour scale; legend in the .md. |

**Where it goes.** Supplementary; cite it from the Fig. 6 Results and from the Discussion paragraph on how much of the
measured HP effect the model reproduces.

**Key numbers** (V1–V3 pooled, n = 28; quadrant means in % of the four-rotation average, mean ± SEM; Wilcoxon):
- Measured (Fig. 6c voxels): HP quadrant −0.87 ± 0.37, next to HP +0.59 ± 0.21, opposite −0.19 ± 0.25; HP − next
  −1.46 (p = .032), next − opposite +0.78 (p = .048).
- Measured, joint-fit voxels: HP −0.85 ± 0.37, next +0.60 ± 0.21, opposite −0.22 ± 0.25; HP − next −1.44 (p = .030),
  next − opposite +0.82 (p = .043).
- Joint-fit prediction: HP −0.97 ± 0.39, next +0.27 ± 0.13, opposite +0.49 ± 0.23; HP − next −1.24 (p = .015),
  next − opposite −0.22 (p = .26).
- Measured − predicted (paired): HP − next −0.21 (p = .73); next − opposite +1.04 (p = .012).

**Reading:** the fit reproduces the lower decoded intensity in the quadrant facing HP (same size as measured), but not
the higher intensity next to HP than opposite it. This matches the item-response check (`discussion_checks_2026-11-12.md`
§2: the model accounts for the dip at HP relative to its neighbours, not for the gradient over the LP locations).
Suggested sentence: "Decoding the joint gain-field fit's noise-free prediction with the same gain-free decoder
reproduced the lower intensity in the quadrant facing HP (−0.97% vs −0.85% measured) but not the higher intensity next
to HP than opposite it (Supplementary Fig. Sx)."

Sources: `retsupp/decode/decode_model_prediction.py` (cluster job 8269323, canary 8269117), maps in
`notes/data/decoded_model_pred_2026-11/`, figure `retsupp/visualize/paper/figS_decoded_map_model_vs_measured.py`.
