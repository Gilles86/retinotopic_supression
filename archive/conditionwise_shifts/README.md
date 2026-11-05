# Archived: PRF fits per condition and their shifts

Archived 2026-11-05 at Gilles's decision. This analysis is no longer used anywhere in the project, which now looks only at gain-field models. The same goes for the shift-model CV in `derivatives/af_prf_cv_shiftvsgain`, whose fitting code was retired on 2026-10-28. Do not build on it or cite it.

It fitted PRFs separately per HP condition and asked whether their centres moved, or fitted attention models to those conditionwise parameters (Klein-style shifts, AF and AF+, the precision-weighted model, GAM and projection curves, predicted vs observed shifts, the May meeting reports).

Paths below this folder mirror where the files used to be, so imports like `retsupp.modeling.fit_condition` no longer resolve. To run something again, move it back with `git mv`.

Left in place:

- **Data.** `derivatives/prf_conditionfit*` and `derivatives/prf_summaries.conditionwise` on the cluster and locally are untouched.
- **Accessors the archived code needs.** `Subject.get_conditionwise_summary_prf_pars`, `type='conditionwise'` in `Subject.get_prf_parameters_volume`, `retsupp.visualize.utils.load_all_conditionwise`, and the `--conditionwise` option of `retsupp/modeling/summarize_prf.py`. No active analysis calls them.
- **Older archived figures.** `notes/archive/` and `notes/figures/_archive/`.
