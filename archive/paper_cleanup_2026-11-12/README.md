# Archive: paper cleanup 2026-11-12

Everything here was moved out of the main tree on 2026-11-12 because it is not needed to reproduce the paper
(n = 28 writing package `notes/writing_package_2026-11-12/`; the result → code map is `PAPER_REPRODUCTION.md` at the
repo top). Files keep their original paths below this folder and their git history (`git log --follow`). Nothing was
deleted. To restore a file: `git mv archive/paper_cleanup_2026-11-12/<path> <path>`.

How the split was made: the figure and table scripts of the package, the pipeline stages from DICOM to figures, and
everything they import or name (modules, SLURM scripts, data tables, notes), closed transitively, stay in the main
tree; tracked files outside that set moved here. Conservative where unsure (a file named by kept code stays).
`notes/data/` tables dated 2026-11 stay; `archive/conditionwise_shifts/` stays where it was.

| Folder | What it holds | Why archived |
|---|---|---|
| `notes/figures/paper_2026-11/` | Earlier versions of the paper figures (n = 30, `_final2`, `_v2`/`_v3`, `_signal`, older Fig. 7 and Fig. 3 variants, `_archive/`) | Superseded by the `_n28` versions in the package; reproducible with the scripts and flags in `PAPER_REPRODUCTION.md` |
| `notes/figures/talk/`, `notes/figures/` (top-level PDFs), `dn_maps/`, `fig1_inflated/`, `fig1_occipital/`, `_archive/` | VSS2026 and talk illustrations, older Fig. 1 renders, DN maps, May meeting material | Not in the paper (`fig1_r2low/`, used by Fig. 2, stays) |
| `notes/` (top-level notes, `archive/`, `analyses/`, `models/`, `scripts/`, `snakemake_demo/`, `bench_*.py`, `render_*.py`, `run_*_glm.py`) | Planning notes, earlier status and handover notes, earlier writing prompts, benchmarks, exploratory scripts | History; the current state is in the package and the notes that stay |
| `notes/data/` (undated tables) | Earlier attention-field parameters, smoke-test outputs, phantom-filter tables, GLMsingle summary | Not read by any kept script |
| `retsupp/visualize/paper/` | Exploratory or superseded figure scripts (earlier Fig. 3 / Fig. 7 variants, AF parameter plots, PCA/behaviour correlations, sanity plots) | Not used by any package figure |
| `retsupp/visualize/vss2026/`, `pycortex/`, `decoding/`, `plot_cv_ladder*.py` | VSS2026 movies and traces, pycortex viewers and older renders, decoded-movie scripts, early ladder plots | Talk / exploration only (Fig. 2's pycortex renderer stays) |
| `retsupp/modeling/` | Earlier model variants and checks (three-model fits, learning gains, run slices, extended stimulus, fixed-group gain, HP-rotation null), old submit scripts | Superseded by the gain-field ladder and fixed-group mechanisms |
| `retsupp/decode/` | Early decoding aggregation, smoke tests and sweeps | Superseded by `decode_final.py` |
| `retsupp/glm/` | GLMsingle fits and quadrant extraction | Not in the paper |
| `retsupp/snakemake/` | Snakemake pRF pipeline driver | Not used for the final fits |
| `retsupp/notebooks/` | Exploration and QC notebooks | Exploration (the sub-24 T1w fix stays) |
| `retsupp/behavior/`, `retsupp/test/` | Early RT summary, empty test package | Not used |

Untracked files that sat in folders moved here were moved along (they stay untracked): `notes/archive/predict_shifts*`,
`notes/figures/_archive/meeting_dock_20260509/`, `notes/figures/fig1_inflated/sub-19_*`, `sub-23_*`,
`retsupp/behavior/check_rts.ipynb`, `retsupp/glm/plot_richter_style.py`.
