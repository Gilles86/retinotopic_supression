#!/bin/bash
# Aggregate the CV ladder: one array task per subject (loads that subject's
# BOLD once for the model-free null, ~5 min), then a dependent merge job
# writes cv_ladder_{per_voxel,subject,stats}.tsv.
#
# Usage (from the repo root on the cluster login node):
#   bash retsupp/modeling/slurm_jobs/aggregate_cv_ladder.sh          # loco
#   CV_SCHEME=run3 bash retsupp/modeling/slurm_jobs/aggregate_cv_ladder.sh

set -euo pipefail
CODE_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
CV_SCHEME="${CV_SCHEME:-loco}"
PYTHON='$HOME/data/conda/envs/retsupp_cuda/bin/python'
AGG="PYTHONPATH=${CODE_DIR} ${PYTHON} -u -m retsupp.modeling.aggregate_cv_ladder --cv-scheme ${CV_SCHEME}"
mkdir -p "$HOME/logs"

ARRAY_ID=$(sbatch --parsable --account=zne.uzh --partition=standard \
    --job-name="cvl_agg_${CV_SCHEME}" --array=1-30 \
    --time=00:20:00 --mem=32G --cpus-per-task=2 \
    --output="$HOME/logs/cvl_agg_${CV_SCHEME}_%A_%a.log" \
    --export=ALL \
    --wrap="${AGG} --subject \${SLURM_ARRAY_TASK_ID}")

MERGE_ID=$(sbatch --parsable --account=zne.uzh --partition=standard \
    --job-name="cvl_merge_${CV_SCHEME}" \
    --dependency="afterany:${ARRAY_ID}" \
    --time=00:15:00 --mem=16G --cpus-per-task=1 \
    --output="$HOME/logs/cvl_merge_${CV_SCHEME}_%j.log" \
    --export=ALL \
    --wrap="${AGG} --merge")

echo "per-subject array: ${ARRAY_ID}   merge: ${MERGE_ID}"
