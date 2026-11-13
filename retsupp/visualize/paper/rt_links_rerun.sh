#!/bin/bash
# Rerun the paper's links between gain-field estimates and response times on ONE reference fit:
#   A  between subjects: dS (estimated alone; joint m7) vs the behavioural HP benefit (LP − HP RT), Spearman
#   B  Fig. 7: trial-wise gains on top of the root's m7_full fits, hierarchical Bayesian meta-analysis, figure
# Nothing earlier is overwritten: every output is named by TAG.
#
# Variables (defaults: the polished reference of update 8):
#   REF_ROOT      ladder root with m7_full   af_prf_cv_ladder_nested_full_bounded_maxsd20_reliable_roihrf_polished
#   ALONE_FOLDER  estimated-alone root       af_fixed_group_mechanisms_polished_2026-11-12
#   TAG           name of the reference      polished_2026-11-12
#   VOXELS        signal (each area's signal voxels, as Fig. 7) or ladder (the fit's own voxels, MIN_VOXELS=10)
#   FIT_LABEL     how the Fig. 7 legend names the base fits
#
# Stages:
#   1. cluster login node, ~/retsupp_plan after `git pull`:   bash retsupp/visualize/paper/rt_links_rerun.sh cluster-submit
#      28 tasks, ~15 min each (2 CPUs, ~27 GB).
#   2. cluster login node, after the tasks finished:            bash retsupp/visualize/paper/rt_links_rerun.sh cluster-collect
#      gain-parameter and estimated-alone tables (srun; skipped when regenerate_reference.sh already wrote them).
#   3. laptop, repo root:                                       bash retsupp/visualize/paper/rt_links_rerun.sh local
#      rsync, collect, meta-analysis (bauer env, ~5 min), Fig. 7 (fig7_compact_${TAG}_n28) and the A table.
#
# Example, the converged polish:
#   REF_ROOT=af_prf_cv_ladder_nested_full_bounded_maxsd20_reliable_roihrf_polished_conv \
#   ALONE_FOLDER=<its estimated-alone folder> TAG=polished_conv_<date> bash retsupp/visualize/paper/rt_links_rerun.sh cluster-submit
set -eo pipefail

REF_ROOT=${REF_ROOT:-af_prf_cv_ladder_nested_full_bounded_maxsd20_reliable_roihrf_polished}
ALONE_FOLDER=${ALONE_FOLDER:-af_fixed_group_mechanisms_polished_2026-11-12}
TAG=${TAG:-polished_2026-11-12}
VOXELS=${VOXELS:-signal}
FIT_LABEL=${FIT_LABEL:-"each subject's joint gain-field fit with the pRFs refitted together with the gain fields (m7: DoG pRFs, one HRF per subject × area, each area's signal voxels)"}
BIDS=/shares/zne.uzh/gdehol/ds-retsupp
HRF_TABLE=$BIDS/derivatives/paper_2026-11/roi_hrf_dog_final2.tsv
EXCLUDE=u24-chaihm0-633,u24-chaiam0-613   # 613: intermittent PermissionError on /shares (2026-11-12)
if [ "$VOXELS" = signal ]; then SUB=af_trialwise_gains_signal_se_${TAG}; else SUB=af_trialwise_gains_ownvox10_${TAG}; fi
D=notes/data
GAIN=$D/gain_field_parameters_${TAG}.tsv
COND=$D/conditional_estimates_${TAG}.tsv
TW=$D/trialwise_${SUB#af_trialwise_gains_}/trialwise_af_${SUB#af_trialwise_gains_}
META=$D/trialwise_meta_bayes_${SUB#af_trialwise_gains_}
PYC='$HOME/data/conda/envs/retsupp_cuda/bin/python'
SRUN="srun --account=hare.econ.uzh --exclude=$EXCLUDE -c1 --mem 8G --time 30"

case "$1" in
cluster-submit)
    if [ "$VOXELS" = signal ]; then V="VOXELS=signal"; else V="MIN_VOXELS=10"; fi
    env REPO="$PWD" LADDER_ROOT=$BIDS/derivatives/$REF_ROOT HRF_TABLE=$HRF_TABLE POOL_GROUPS=1 OUTPUT_SUBDIR=$SUB $V \
        sbatch --account=hare.econ.uzh --partition=standard --exclude=$EXCLUDE --cpus-per-task=2 --mem=36G \
        --time=00:40:00 --job-name=trialwise_${TAG} --array=3-30 retsupp/modeling/slurm_jobs/fit_trialwise_af_gains.sh
    ;;
cluster-collect)
    [ -f "$GAIN" ] || eval $SRUN $PYC -m retsupp.visualize.paper.collect_gain_parameters --root dog=$REF_ROOT --out $GAIN
    [ -f "$COND" ] || eval $SRUN $PYC -m retsupp.visualize.paper.collect_conditional_estimates --bids-folder $BIDS \
        --folder $ALONE_FOLDER --out $COND
    ;;
local)
    PY=${PY:-$HOME/mambaforge/envs/retsupp/bin/python}
    PYB=${PYB:-$HOME/mambaforge/envs/bauer/bin/python}
    mkdir -p /data/ds-retsupp/derivatives/$SUB
    rsync -a --include='sub-*/' --include='*_parametric.tsv' --include='*_check.tsv' --include='*_bins.tsv' --exclude='*' \
        "sciencecluster:$BIDS/derivatives/$SUB/" /data/ds-retsupp/derivatives/$SUB/
    [ -f "$GAIN" ] || rsync -a "sciencecluster:retsupp_plan/$GAIN" $D/
    [ -f "$COND" ] || rsync -a "sciencecluster:retsupp_plan/$COND" $D/
    $PY -m retsupp.visualize.paper.collect_trialwise_tables /data/ds-retsupp/derivatives/$SUB $TW
    $PYB -m retsupp.visualize.paper.trialwise_meta_bayes --param ${TW}_parametric.tsv --check ${TW}_check.tsv --out $META
    $PY -m retsupp.visualize.paper.fig7_compact --compact --meta $META --param ${TW}_parametric.tsv \
        --name fig7_compact_${TAG}_n28 --fit-label "$FIT_LABEL"
    $PY -m retsupp.visualize.paper.rt_links_between_subjects \
        --fit earlier $D/conditional_estimates_dog_final2_2026-11.tsv $D/gain_field_parameters_final2_2026-11.tsv \
        --fit "$TAG" $COND $GAIN --out $D/rt_links_between_subjects_${TAG}.tsv
    $PY -m retsupp.visualize.paper.rt_links_compare --meta notes/data/n28/trialwise_meta_bayes_2026-11 earlier \
        --meta $META "$TAG" --out $D/rt_links_trialwise_${TAG}.tsv
    ;;
*)
    sed -n '2,25p' "$0"
    exit 1
    ;;
esac
