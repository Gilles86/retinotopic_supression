**Table S. What each gain-field term adds to out-of-subject prediction.**

| Model | Parent | Adds | Contrast | DoG: better; direction | Gaussian: better; direction | DN: better; direction |
|---|---|---|---|---|---|---|
| m00 |  | pRF on the bar |  |  |  |  |
| m0 | m00 | Search items, weight fitted | Items | 100% (p = 7.5e-09) | 93% (p = 3.7e-08); 89% (p = 4.1e-07) | 100% (p = 7.5e-09) |
| m1 | m0 | Target gain | T | 75% (p = 0.00072); 96% (p = 2.2e-08) | 89% (p = 5.7e-06); 93% (p = 3.7e-08) | 86% (p = 9.3e-05); 100% (p = 7.5e-09) |
| m2 | m0 | Distractor gain, same at all locations | D | 61% (p = 0.16); 75% (p = 0.0034) | 64% (p = 0.03); 86% (p = 9.4e-06) | 18% (p = 0.018); 32% (p = 0.12) |
| m3 | m0 | Sustained gain, same at all locations | S | 93% (p = 1.5e-06); 100% (p = 7.5e-09) | 89% (p = 4.1e-07); 100% (p = 7.5e-09) | 75% (p = 0.0012); 93% (p = 1e-06) |
| m3b | m1 | Distractor gain | D given T | 61% (p = 0.33); 75% (p = 0.00047) | 54% (p = 0.88); 68% (p = 0.015) | 61% (p = 0.78); 82% (p = 0.0082) |
| m4 | m3b | Sustained gain |  |  |  |  |
| m5 | m4 | Sustained gain differs at HP | dS | 68% (p = 0.41); 75% (p = 0.0071) | 61% (p = 0.54); 79% (p = 0.0044) | 64% (p = 0.24); 71% (p = 0.038) |
| m6 | m4 | Distractor gain differs at HP | dD | 43% (p = 0.25); 50% (p = 0.96) | 50% (p = 0.87); 57% (p = 0.26) | 54% (p = 0.6); 71% (p = 0.0052) |
| m7 | m5 | Both HP-specific terms |  |  |  |  |
| m8 | m4 | Target gain differs at HP | dT | 61% (p = 0.25); 71% (p = 0.014) | 57% (p = 0.73); 68% (p = 0.22) | 39% (p = 0.19); 57% (p = 0.75) |

How much each term adds to out-of-subject prediction (leave-one-subject-out, final fits; each pRF model with its own jointly fitted HRF per subject × ROI). Better: share of 28 subjects whose ΔR² on held-out data is positive when the term is added at the median of the other 27 subjects' own estimates (signal voxels; per subject the median over ROIs), Wilcoxon signed-rank p. Direction: sign-flip test, ΔR² at +δ minus at −δ, which removes the cost of adding a term (a true 50% null). Items: the item weight of pRFs fitted to the bar alone, scored against items at bar strength; not a test of items against no items.
