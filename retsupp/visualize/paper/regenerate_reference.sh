#!/bin/bash
# Regenerate every gain-field figure, table and key number of the paper for ONE reference fit.
#
# The reference fit is one derivatives root of the gain-field ladder (with m4_TDS and m7_full; e.g. the polished fits,
# pRFs refitted together with the gain fields) plus the matching estimated-alone folder (fit_fixed_group_mechanisms
# on that root's m4_TDS). Everything downstream is named by TAG, so another reference (another voxel selection) is a
# re-run with other variables; no earlier file is overwritten.
#
# Variables (defaults: the reference of writing package update 8):
#   REF_ROOT      ladder root           af_prf_cv_ladder_nested_full_bounded_maxsd20_reliable_roihrf_polished
#   ALONE_FOLDER  estimated-alone root  af_fixed_group_mechanisms_polished_2026-11-12
#   BASE_ROOT     ladder root the other models come from (estimated-alone run only)
#                                       af_prf_cv_ladder_nested_full_bounded_maxsd20_reliable_roihrf
#   TAG           name of every table   polished_2026-11-12
#   SUFFIX        figure-name suffix    _n28   (Fig. 5 -> fig5_gain_fields_v11${SUFFIX}; supplements and the parameter
#                                               table -> <name>_${TAG}${SUFFIX})
#   FIT_LABEL     how legends name the fits
#   REFIT_PREFIX  optional collect_polished_fits prefix (SSE / pRF change vs the clamped fits), e.g. notes/data/prf_refit_2026-11-12
#
# Stages, in order:
#   1. On the cluster login node, in ~/retsupp_plan after `git pull`:
#        bash retsupp/visualize/paper/regenerate_reference.sh cluster-submit
#      submits the width profiles (28 tasks, ~10 min each) and, with SUBMIT_ALONE=1 and no ALONE_FOLDER yet, the
#      estimated-alone sufficient statistics (28 tasks, ~15 min). Wait for both.
#   2. On the cluster login node:
#        bash retsupp/visualize/paper/regenerate_reference.sh cluster-collect
#      collects the joint-fit table, the estimated-alone table and the field widths on a compute node (srun) into
#      notes/data/ of the cluster checkout.
#   3. On the laptop, in the repo:
#        bash retsupp/visualize/paper/regenerate_reference.sh local
#      rsyncs those tables, then renders Fig. 5 v11, figS_field_widths_v3, figS_gain_field_map_search_v3,
#      figS_gain_vs_prf_size, figS_pooled_dS_dD, the parameter table and the key numbers (n = 28: sub-01, sub-02 left out).
#
# Example, another voxel selection:
#   REF_ROOT=af_prf_cv_ladder_nested_full_bounded_maxsd20_relnull_r0.2_roihrf_polished \
#   ALONE_FOLDER=af_fixed_group_mechanisms_relnull_r0.2_polished TAG=relnull_r0.2_polished SUFFIX=_relnull02_n28 \
#   SUBMIT_ALONE=1 bash retsupp/visualize/paper/regenerate_reference.sh cluster-submit
#   (the same variables for cluster-collect and local)
set -eo pipefail

REF_ROOT=${REF_ROOT:-af_prf_cv_ladder_nested_full_bounded_maxsd20_reliable_roihrf_polished}
ALONE_FOLDER=${ALONE_FOLDER:-af_fixed_group_mechanisms_polished_2026-11-12}
BASE_ROOT=${BASE_ROOT:-af_prf_cv_ladder_nested_full_bounded_maxsd20_reliable_roihrf}
TAG=${TAG:-polished_2026-11-12}
SUFFIX=${SUFFIX:-_n28}
FIT_LABEL=${FIT_LABEL:-"gain-field model fits with DoG pRFs, the pRFs refitted together with the gain fields"}
BIDS=/shares/zne.uzh/gdehol/ds-retsupp
HRF_TABLE=$BIDS/derivatives/paper_2026-11/roi_hrf_dog_final2.tsv
D=notes/data
GAIN=$D/gain_field_parameters_${TAG}.tsv
COND=$D/conditional_estimates_${TAG}.tsv
WIDTHS=$D/field_widths_reliable_${TAG}_n28
SLOPES=$D/hierarchy_regression_${TAG}_n28.tsv
NUMBERS=$D/reference_numbers_${TAG}_n28
SRUN="srun --account=hare.econ.uzh --exclude=u24-chaihm0-633 -c1 --mem 8G --time 30"
PYC='$HOME/data/conda/envs/retsupp_cuda/bin/python'

case "$1" in
cluster-submit)
    OVERLAY_ROOT=$REF_ROOT OUTPUT_SUBDIR=field_widths_reliable_${TAG} \
        sbatch --exclude=u24-chaihm0-633 --job-name=width_prof_${TAG} --array=3-30 \
        retsupp/modeling/slurm_jobs/fit_width_profiles.sh
    if [ -n "$SUBMIT_ALONE" ] && [ ! -d "$BIDS/derivatives/$ALONE_FOLDER" ]; then
        OVERLAY_ROOT=$REF_ROOT OVERLAY_PER_MODEL=1 OUTPUT_SUBDIR=$ALONE_FOLDER BASE_ROOT=$BASE_ROOT HRF_TABLE=$HRF_TABLE \
            sbatch --exclude=u24-chaihm0-633 --job-name=fixed_mech_${TAG} --time=00:30:00 --array=3-30 \
            retsupp/modeling/slurm_jobs/fit_fixed_group_mechanisms.sh
    fi
    ;;
cluster-collect)
    eval $SRUN $PYC -m retsupp.visualize.paper.collect_gain_parameters --root dog=$REF_ROOT \
        --base-table $D/gain_field_parameters_final2_2026-11.tsv --out $GAIN
    eval $SRUN $PYC -m retsupp.visualize.paper.collect_conditional_estimates --bids-folder $BIDS \
        --folder $ALONE_FOLDER --out $COND
    eval $SRUN $PYC -m retsupp.visualize.paper.field_widths_reliable \
        --root $BIDS/derivatives/field_widths_reliable_${TAG} --out-prefix $WIDTHS
    ;;
local)
    PY=${PY:-$HOME/mambaforge/envs/retsupp/bin/python}
    EX="$PY -m retsupp.visualize.paper.run_excluding --exclude 1 2 --"
    rsync -a "sciencecluster:retsupp_plan/$GAIN" "sciencecluster:retsupp_plan/$COND" $D/
    rsync -a "sciencecluster:retsupp_plan/${WIDTHS}_*" $D/
    $EX retsupp.visualize.paper.hierarchy_regression --gain-table $GAIN --conditional-table $COND --out $SLOPES
    # Fig. 5 v11 and the search supplement (v3)
    $EX retsupp.visualize.paper.fig5_gain_fields_v6 --layout v11 --suffix "$SUFFIX" --gain-table $GAIN \
        --conditional-table $COND --slope-table $SLOPES --widths-table ${WIDTHS}_widths_thr2.tsv
    # parameter table, gain vs pRF size, pooled dS and dD (in %)
    $EX retsupp.visualize.paper.fig5_gain_fields --final --percent --suffix "_${TAG}${SUFFIX}" --gain-table $GAIN \
        --conditional-table $COND --slope-table $SLOPES --dt-table cond --outputs hierarchy pooled \
        --fit-label "$FIT_LABEL"
    # field widths among fits with a reliable gain (v3)
    $PY -m retsupp.visualize.paper.figS_field_widths_v2 --fits ${WIDTHS}_fits.tsv --name "figS_field_widths_v3${SUFFIX}" \
        --legend auto --fit-label "joint $FIT_LABEL"
    # key numbers
    $PY -m retsupp.visualize.paper.reference_numbers --gain-table $GAIN --conditional-table $COND \
        --widths-prefix $WIDTHS --slope-table $SLOPES ${REFIT_PREFIX:+--refit-prefix $REFIT_PREFIX} \
        --label "$TAG ($REF_ROOT; estimated alone: $ALONE_FOLDER)" --out $NUMBERS
    ;;
*)
    sed -n '2,40p' "$0"
    exit 1
    ;;
esac
