#!/bin/bash -l
#SBATCH --job-name=dn_by_hp
#SBATCH --account=hare.econ.uzh
#SBATCH --partition=standard
#SBATCH --output=/dev/null
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#
# DN pRF fits per HP condition (fit_dn_by_hp.py). One array task per subject:
#   sbatch --array=1-30 retsupp/modeling/slurm_jobs/fit_dn_by_hp.sh
# Canary on a voxel subset: MAX_VOXELS=500 sbatch --array=3 retsupp/modeling/slurm_jobs/fit_dn_by_hp.sh
# Optional: OUT_SUBDIR=name (default prf_dn_by_hp_2026-11).

set -eo pipefail
subject=${SLURM_ARRAY_TASK_ID}
sub_pad=$(printf "%02d" "$subject")
LOGFILE="$HOME/logs/dn_by_hp_sub-${sub_pad}_${SLURM_JOB_ID}.txt"
mkdir -p "$HOME/logs"
exec >"$LOGFILE" 2>&1
export TMPDIR="/scratch/$USER/tmp/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "$TMPDIR"

PYTHON="$HOME/data/conda/envs/retsupp_cuda/bin/python"
CODE_DIR="${CODE_DIR:-$(cd "${SLURM_SUBMIT_DIR}" && git rev-parse --show-toplevel)}"
export PYTHONPATH="${CODE_DIR}${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=""
export TF_NUM_INTRAOP_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export TF_NUM_INTEROP_THREADS=2
echo "Host: $(hostname) | sub-${sub_pad} | job ${SLURM_JOB_ID} | $(date)"
echo "Code: ${CODE_DIR} @ $(git -C "$CODE_DIR" rev-parse --short HEAD)"

"$PYTHON" -u -m retsupp.modeling.fit_dn_by_hp "$subject" \
    --bids-folder /shares/zne.uzh/gdehol/ds-retsupp \
    ${OUT_SUBDIR:+--output-subdir $OUT_SUBDIR} ${MAX_VOXELS:+--max-voxels $MAX_VOXELS}

echo "Finished: $(date)"
