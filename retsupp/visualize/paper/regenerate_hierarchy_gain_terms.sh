#!/bin/bash
# The gain terms along the visual hierarchy (hierarchy_gain_terms + figS_hierarchy_gain_terms) for ONE reference fit.
#
# Same variables as regenerate_reference.sh (run that first: it writes the joint-fit table, the estimated-alone table
# and the field widths this script reads):
#   REF_ROOT  ladder root with m7_full   af_prf_cv_ladder_nested_full_bounded_maxsd20_reliable_roihrf_polished
#   TAG       table names                polished_2026-11-12
#   SUFFIX    figure-name suffix         _n28  (-> figS_hierarchy_gain_terms_${TAG}${SUFFIX})
#   FIT_LABEL how the legend names the fits
#
# Stages:
#   1. cluster login node, ~/retsupp_plan after `git pull`:
#        bash retsupp/visualize/paper/regenerate_hierarchy_gain_terms.sh cluster-submit
#      per-voxel sufficient statistics of the five gains on REF_ROOT's m7_full (fit_gain_bins; 28 tasks, ~5–10 min)
#   2. laptop, repo root, once those tasks are done:
#        bash retsupp/visualize/paper/regenerate_hierarchy_gain_terms.sh local
#      rsyncs the statistics to /data/ds-retsupp/derivatives/gain_bins_${TAG}, writes
#      notes/data/hierarchy_gain_terms_${TAG}_n28_*.tsv and the supplementary figure.
#
# Example, the converged refit:
#   REF_ROOT=af_prf_cv_ladder_nested_full_bounded_maxsd20_reliable_roihrf_polished_conv TAG=polished_conv_2026-11-13 \
#   bash retsupp/visualize/paper/regenerate_hierarchy_gain_terms.sh cluster-submit     (then `local`, same variables)
set -eo pipefail

REF_ROOT=${REF_ROOT:-af_prf_cv_ladder_nested_full_bounded_maxsd20_reliable_roihrf_polished}
TAG=${TAG:-polished_2026-11-12}
SUFFIX=${SUFFIX:-_n28}
FIT_LABEL=${FIT_LABEL:-"joint gain-field fits with DoG pRFs, the pRFs refitted together with the gain fields"}
BIDS=/shares/zne.uzh/gdehol/ds-retsupp
D=notes/data
BINS=gain_bins_${TAG}
OUT=$D/hierarchy_gain_terms_${TAG}_n28

case "$1" in
cluster-submit)
    OVERLAY_ROOT=$REF_ROOT OUTPUT_SUBDIR=$BINS \
        sbatch --exclude=u24-chaihm0-633 --job-name=gain_bins_${TAG} --array=3-30 \
        retsupp/modeling/slurm_jobs/fit_gain_bins.sh
    ;;
local)
    PY=${PY:-$HOME/mambaforge/envs/retsupp/bin/python}
    mkdir -p /data/ds-retsupp/derivatives/$BINS
    rsync -a sciencecluster:$BIDS/derivatives/$BINS/ /data/ds-retsupp/derivatives/$BINS/
    $PY -m retsupp.visualize.paper.hierarchy_gain_terms --gain-table $D/gain_field_parameters_${TAG}.tsv \
        --cond-lin $D/conditional_estimates_${TAG}.tsv --widths-fits $D/field_widths_reliable_${TAG}_n28_fits.tsv \
        --bins-dir /data/ds-retsupp/derivatives/$BINS --out $OUT
    $PY -m retsupp.visualize.paper.figS_hierarchy_gain_terms --prefix $OUT \
        --name figS_hierarchy_gain_terms_${TAG}${SUFFIX} --fits-note "$FIT_LABEL"
    ;;
*)
    sed -n '2,24p' "$0"
    exit 1
    ;;
esac
