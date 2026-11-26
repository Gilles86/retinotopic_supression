**Supplementary Table S4. Main results with and without the fits that ended on a bound.**

Per participant and area, the gains come from the joint zero-sum fit with DoG pRFs (241 fits, n = 28). They go through the Student-t hierarchical model of Fig. 5: posterior mean [95% HDI], P = posterior probability of the sign. A pathway is the mean of its areas.

The columns:
- **All fits.** A refit of the main analysis. It agrees with the reported values to within MCMC noise.
- **Without this term on a bound.** Leaves out the fits in which the term itself ended on a bound: 22 for dS, 27 for the target gain, 27 for the distractor gain.
- **Without any gain or the item weight on a bound.** Leaves out the same 58 fits for every term. These are the fits in which any of the three gains was on a bound (33), a single gain parameter was at its limit (24), or the item weight was at 0.05 (26) or 50 (16); a fit can be counted under more than one of these.

The excluded fits are mostly in TO (16), IPS (14), LO (7) and VO (6).

| Term | Unit | All fits (241) | Without this term on a bound | Without any gain or the item weight on a bound (183) |
|---|---|---|---|---|
| dS (HP − LP), % | All areas | −3.2 [−4.7, −1.6], P > .999 | −3.2 [−4.9, −1.6], P > .999 (219) | −3.5 [−4.9, −1.8], P > .999 |
| | Early | −5.5 [−7.9, −3.1], P > .999 | −5.4 [−7.9, −2.9], P > .999 | −6.0 [−8.5, −3.6], P > .999 |
| | Dorsal | −1.8 [−3.6, +0.0], P = .977 | −1.8 [−3.8, +0.0], P = .970 | −1.9 [−4.0, +0.3], P = .965 |
| | Occipitotemporal | −2.2 [−3.8, −0.6], P = .997 | −2.2 [−3.8, −0.5], P = .994 | −2.3 [−4.0, −0.6], P = .996 |
| Target gain, % | All areas | +23.4 [+15.2, +32.0], P > .999 | +24.7 [+15.4, +33.4], P > .999 (214) | +28.9 [+19.4, +39.7], P > .999 |
| | Early | +9.4 [−0.3, +19.6], P = .971 | +10.2 [+0.0, +20.3], P = .981 | +13.2 [+2.3, +24.2], P = .993 |
| | Dorsal | +30.5 [+18.6, +42.5], P > .999 | +32.1 [+19.5, +45.0], P > .999 | +31.3 [+16.9, +46.0], P > .999 |
| | Occipitotemporal | +30.4 [+20.4, +40.9], P > .999 | +31.8 [+21.0, +43.6], P > .999 | +39.4 [+26.0, +53.2], P > .999 |
| Distractor gain, % | All areas | +4.6 [+0.6, +8.7], P = .988 | +4.8 [+0.7, +8.9], P = .990 (214) | +7.2 [+2.6, +12.3], P = .999 |
| | Early | +1.2 [−4.1, +6.5], P = .663 | +1.4 [−3.8, +6.7], P = .692 | +3.1 [−3.0, +8.8], P = .846 |
| | Dorsal | +5.9 [+0.2, +11.6], P = .979 | +5.9 [+0.2, +11.5], P = .981 | +6.9 [+0.8, +13.8], P = .979 |
| | Occipitotemporal | +6.6 [+2.3, +11.1], P = .998 | +6.8 [+2.4, +11.6], P = .998 | +10.3 [+5.1, +16.1], P > .999 |

Sampler: PyMC with nutpie, 4 × 2000 draws, no divergences in any fit.
Outputs: `notes/data/bound_fits_zs_dog_2026-11-26/` (`bayes.tsv`, `estimates_flags.tsv`).
