**Table S. What each gain-field term adds to out-of-subject prediction.**

| Model | Parent | Adds | Contrast | DoG: better; direction | Gaussian: better; direction | DN: better; direction |
|---|---|---|---|---|---|---|
| m00 |  | pRF on the bar |  |  |  |  |
| m0 | m00 | Search items, weight fitted | Items | 100% (p = 1.9e-09) | 93% (p = 9.3e-09); 90% (p = 5.2e-08) | 100% (p = 1.9e-09) |
| m1 | m0 | Target gain | T | 77% (p = 0.00061); 97% (p = 9.3e-09) | 90% (p = 5.7e-07); 93% (p = 1.3e-08) | 87% (p = 1.4e-05); 100% (p = 1.9e-09) |
| m2 | m0 | Distractor gain, same at all locations | D | 77% (p = 0.031); 80% (p = 0.001) | 67% (p = 0.026); 80% (p = 1.6e-05) | 57% (p = 0.56); 57% (p = 0.86) |
| m3 | m0 | Sustained gain, same at all locations | S | 87% (p = 5.1e-06); 100% (p = 1.9e-09) | 83% (p = 4.4e-05); 100% (p = 1.9e-09) | 77% (p = 0.00026); 93% (p = 2e-07) |
| m3b | m1 | Distractor gain | D given T | 63% (p = 0.11); 80% (p = 3.5e-05) | 60% (p = 0.46); 80% (p = 0.00073) | 60% (p = 0.32); 83% (p = 0.0011) |
| m4 | m3b | Sustained gain |  |  |  |  |
| m5 | m4 | Sustained gain differs at HP | dS | 67% (p = 0.18); 77% (p = 0.0024) | 57% (p = 0.64); 70% (p = 0.011) | 60% (p = 0.27); 70% (p = 0.045) |
| m6 | m4 | Distractor gain differs at HP | dD | 57% (p = 1); 70% (p = 0.1) | 47% (p = 0.75); 57% (p = 0.19) | 53% (p = 0.36); 73% (p = 0.0017) |
| m7 | m5 | Both HP-specific terms |  |  |  |  |
| m8 | m4 | Target gain differs at HP | dT | 63% (p = 0.24); 70% (p = 0.019) | 47% (p = 0.32); 53% (p = 0.72) | 60% (p = 0.92); 73% (p = 0.2) |

How much each term adds to out-of-subject prediction (leave-one-subject-out, final fits; each pRF model with its own jointly fitted HRF per subject × ROI). Better: share of 30 subjects whose ΔR² on held-out data is positive when the term is added at the median of the other 29 subjects' own estimates (signal voxels; per subject the median over ROIs), Wilcoxon signed-rank p. Direction: sign-flip test, ΔR² at +δ minus at −δ, which removes the cost of adding a term (a true 50% null). Items: the item weight of pRFs fitted to the bar alone, scored against items at bar strength; not a test of items against no items.
