#!/bin/bash
# Update 13 inputs: every gain-field figure, table and key number on the zero-sum LM reference
# (notes/zero_sum_refit_2026-11-13.md). Same stages as regenerate_reference.sh + regenerate_hierarchy_gain_terms.sh +
# rt_links_rerun.sh, with the zero-sum estimated-alone dS (ZERO_SUM=1) and the node exclusions of 2026-11-14.
#
# REF_ROOT is a view root of symlinks: m7_full -> m5_TDS_dS_zs (the main model: item weight, target, distractor,
# zero-sum sustained field), m4_TDS -> m4_TDS_zs; m7_full_zs (supplementary, + transient HP - LP) under its own name.
#
#   bash retsupp/visualize/paper/zero_sum_reference.sh cluster-submit    # cluster login node, ~/retsupp_plan
#   bash retsupp/visualize/paper/zero_sum_reference.sh cluster-collect   # after the jobs, login node (srun)
#   bash retsupp/visualize/paper/zero_sum_reference.sh local             # laptop, repo root
set -eo pipefail
export REF_ROOT=af_prf_cv_ladder_nested_full_bounded_maxsd20_reliable_roihrf_zs_lm_ref
export ZS_ROOT=af_prf_cv_ladder_nested_full_bounded_maxsd20_reliable_roihrf_zs_lm
export ALONE_FOLDER=af_fixed_group_mechanisms_zs_lm_2026-11-13
export TAG=zs_lm_2026-11-13
export SUFFIX=_zs_n28
export FIT_LABEL="gain-field model fits with DoG pRFs, zero-sum sustained field, pRFs and gain fields fitted jointly (LM)"
BIDS=/shares/zne.uzh/gdehol/ds-retsupp
HRF_TABLE=$BIDS/derivatives/paper_2026-11/roi_hrf_dog_final2.tsv
BASE_ROOT=af_prf_cv_ladder_nested_full_bounded_maxsd20_reliable_roihrf
EX=u24-chaihm0-633,u24-chaiam0-613,u24-chaihm0-634
D=notes/data
SRUN="srun --account=hare.econ.uzh --partition=standard --exclude=$EX -c1 --mem 8G --time 30"
PYC='$HOME/data/conda/envs/retsupp_cuda/bin/python'

case "$1" in
cluster-submit)
    # field widths (regenerate_reference.sh), estimated-alone (zero-sum dS), gain bins (hierarchy), trial-wise gains
    OVERLAY_ROOT=$REF_ROOT OUTPUT_SUBDIR=field_widths_reliable_${TAG} \
        sbatch --exclude=$EX --job-name=width_prof_${TAG} --array=3-30 retsupp/modeling/slurm_jobs/fit_width_profiles.sh
    ZERO_SUM=1 OVERLAY_ROOT=$REF_ROOT OVERLAY_PER_MODEL=1 OUTPUT_SUBDIR=$ALONE_FOLDER BASE_ROOT=$BASE_ROOT \
        HRF_TABLE=$HRF_TABLE sbatch --exclude=$EX --job-name=fixed_mech_${TAG} --time=00:30:00 --array=3-30 \
        retsupp/modeling/slurm_jobs/fit_fixed_group_mechanisms.sh
    OVERLAY_ROOT=$REF_ROOT OUTPUT_SUBDIR=gain_bins_${TAG} \
        sbatch --exclude=$EX --job-name=gain_bins_${TAG} --array=3-30 retsupp/modeling/slurm_jobs/fit_gain_bins.sh
    env REPO="$PWD" LADDER_ROOT=$BIDS/derivatives/$REF_ROOT HRF_TABLE=$HRF_TABLE POOL_GROUPS=1 \
        OUTPUT_SUBDIR=af_trialwise_gains_signal_se_${TAG} VOXELS=signal \
        sbatch --account=hare.econ.uzh --partition=standard --exclude=$EX --cpus-per-task=2 --mem=36G \
        --time=00:40:00 --job-name=trialwise_${TAG} --array=3-30 retsupp/modeling/slurm_jobs/fit_trialwise_af_gains.sh
    ;;
cluster-collect)
    # regenerate_reference.sh cluster-collect (its SRUN excludes one node only), plus the per-start table and the
    # zero-sum model rows (m7_full_zs, m5_TDS_dS_zs) for the supplementary numbers
    sed -e "s/--exclude=u24-chaihm0-633/--exclude=$EX/" retsupp/visualize/paper/regenerate_reference.sh > /tmp/rr_$$.sh
    bash /tmp/rr_$$.sh cluster-collect
    rm /tmp/rr_$$.sh
    eval $SRUN $PYC -m retsupp.modeling.summarize_zero_sum --zs-root $ZS_ROOT --out $D/zero_sum_fits_2026-11-13.tsv
    ;;
local)
    sed -e "s/--exclude=u24-chaihm0-633/--exclude=$EX/" retsupp/visualize/paper/regenerate_reference.sh > /tmp/rr_$$.sh
    bash /tmp/rr_$$.sh local
    bash retsupp/visualize/paper/regenerate_hierarchy_gain_terms.sh local
    bash retsupp/visualize/paper/rt_links_rerun.sh local
    rm /tmp/rr_$$.sh
    rsync -a "sciencecluster:retsupp_plan/$D/zero_sum_fits_2026-11-13.tsv" $D/
    PY=${PY:-$HOME/mambaforge/envs/retsupp/bin/python}
    $PY -m retsupp.visualize.paper.compare_zero_sum --ref $D/reference_numbers_polished_2026-11-12_n28.tsv \
        --zs $D/reference_numbers_${TAG}_n28.tsv --zs-gain-table $D/gain_field_parameters_${TAG}.tsv \
        --zs-summary $D/zero_sum_fits_2026-11-13.tsv --out $D/compare_zero_sum_2026-11-13
    ;;
*)
    sed -n '2,12p' "$0"
    exit 1
    ;;
esac
